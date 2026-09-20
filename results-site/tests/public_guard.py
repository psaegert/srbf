"""The public results site must carry no trace of local-only methods (README.md, "Local-only methods").

Fatal checks, run before the Playwright suite in CI and locally:
  1. index.html references neither a private/ path nor index.local.html;
  2. every method key in data/*/results.js, data/*/hist/*.js, data/*/paired.js and data/*/ranks.js is in the public
     allowlist below
     (the list names PUBLIC methods only; a private method's key must never appear here);
  3. every release payload carries the complete metric registry (at least the 2026-07 site's metrics, under their
     schema-2 keys) so a regenerated release cannot silently lose metrics;
  4. in CI, results-site/private/ and index.local.html do not exist in the checkout (they are git-ignored; a forced
     add would surface here before anything deploys);
  5. no published payload carries an as-run wall-clock metric. Seconds measured where a unit happened to run are
     not comparable between methods; the only timing this benchmark publishes is the reference-machine ladder in
     timing.json (owner 2026-09-17: "We will only publish times that are calibrated. Full stop.");
  6. a sealed payload, if one is present, is sealed: the envelope carries only its own fields, the KDF is strong
     enough to be worth having, and the ciphertext reads as ciphertext (high entropy, no plaintext left in it).
     The checker is run against a deliberately bad envelope on every invocation, so it cannot pass vacuously.
"""
import base64
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

SITE = Path(__file__).resolve().parents[1]
PUBLIC_METHODS = {"e2e", "nesymres-100M", "PySR", "T8-3M", "T8-20M", "T8-120M", "prior"}
# the 2026-07 site's 21 metrics under the schema-2 keys (symbolic_recovery there = skeleton_match_raw here,
# prediction_success_rate = success), plus the release's own headline metrics
REQUIRED_METRICS = {
    "numeric_recovery_val", "expr_length_ratio", "log10_fvu_val", "log10_fvu_fit", "numeric_recovery_fit", "success",
    "skeleton_match_raw", "f1_score", "precision_score", "recall_score", "edit_distance_norm", "zss_edit_distance",
    "expr_length_ratio_abserr", "predicted_skeleton_prefix_length", "skeleton_length", "n_constants_ratio",
    "n_constants_delta", "total_nestedness_delta", "predicted_log_prob", "predicted_score",
    "symbolic_recovery", "mdl_ratio", "r2_val"}
# Wall-clock measured wherever a unit ran. Never published: not in the registry, not in a cell, not in a
# histogram, not in a paired contrast. The reference-machine ladder (timing.json) is the only timing that ships.
UNCALIBRATED_TIME_KEYS = ("fit_time", "generation_time")


def payload_of(text: str, var: str) -> Any:
    m = re.match(r"window\." + var + r" = (.*);\s*$", text, re.S)
    return json.loads(m.group(1)) if m else None


def keys_in_wrapped(text: str, pattern: str) -> set[str] | None:
    m = re.search(pattern, text, re.S)
    return set(json.loads(m.group(1))) if m else None


def _object_after(text: str, marker: str) -> dict | None:
    """The JSON object that follows `marker` (a ranks.js is a chain of Object.assign(target, {...}) calls)."""
    i = text.find(marker)
    if i < 0:
        return None
    obj, _ = json.JSONDecoder().raw_decode(text, i + len(marker))
    return obj if isinstance(obj, dict) else None


def rank_methods(text: str) -> set[str] | None:
    """Every method key a ranks.js names: the methods with a time-budget rung, both sides of every pair, and both
    sides of every record of which rungs a time-limit outcome compared."""
    at, pairs = _object_after(text, ".at,"), _object_after(text, ".pairs,")
    if at is None or pairs is None:
        return None
    rungs = _object_after(text, ".rungs,") or {}
    return set(at) | {k for pair in list(pairs) + list(rungs) for k in pair.split("|")}


SEALED_FIELDS = {"v", "kdf", "iter", "salt", "iv", "ct"}
MIN_ITERATIONS = 200_000
# A sealed blob must not carry recognisable plaintext. Every tell is long enough that random bytes do not
# produce it: a 2-byte needle turns up ~7 times by chance in 400 kB, a 7-byte one once in 10^12 payloads.
PLAINTEXT_TELLS = (b"window.", b"RESULTS_V2", b"function", b'"methods"', b'"cells"', b'"catalogs"')
assert all(len(t) >= 7 for t in PLAINTEXT_TELLS), "a short tell fires on ciphertext by chance"


def entropy(data: bytes) -> float:
    """Shannon entropy in bits per byte. Ciphertext sits at ~8.0; anything structured sits well below."""
    if not data:
        return 0.0
    counts = [0] * 256
    for b in data:
        counts[b] += 1
    n = len(data)
    return -sum((c / n) * math.log2(c / n) for c in counts if c)


def check_sealed(text: str, name: str) -> list[str]:
    """Everything that must hold for a sealed payload, whatever it holds."""
    bad = []
    m = re.search(r"window\.RESULTS_V2_SEALED\[[^\]]*\]=(\{.*\});\s*$", text, re.S)
    if not m:
        return [f"{name}: not a sealed envelope"]
    try:
        env = json.loads(m.group(1))
    except ValueError:
        return [f"{name}: envelope is not JSON"]
    if set(env) != SEALED_FIELDS:
        bad.append(f"{name}: envelope fields {sorted(set(env) ^ SEALED_FIELDS)} (only {sorted(SEALED_FIELDS)} belong here)")
    if env.get("kdf") != "PBKDF2-SHA256" or int(env.get("iter", 0)) < MIN_ITERATIONS:
        bad.append(f"{name}: kdf {env.get('kdf')!r} at {env.get('iter')} iterations is below the bar")
    try:
        ct = base64.b64decode(env.get("ct", ""), validate=True)
    except Exception:
        return bad + [f"{name}: ciphertext is not base64"]
    if len(ct) < 1024:
        bad.append(f"{name}: ciphertext is {len(ct)} bytes")
    if entropy(ct) < 7.5:
        bad.append(f"{name}: ciphertext entropy {entropy(ct):.2f} bits/byte reads as plaintext")
    for tell in PLAINTEXT_TELLS:
        if tell in ct:
            bad.append(f"{name}: ciphertext contains {tell!r}")
    return bad


def check_no_as_run_time(path: Path) -> list[str]:
    """No published file may mention an as-run wall-clock metric, wherever it is nested."""
    text = path.read_text(encoding="utf-8")
    return [f"{path}: publishes {k!r}" for k in UNCALIBRATED_TIME_KEYS if k in text]


def selftest() -> list[str]:
    """The guard checks itself: a blob that is plainly not sealed must be rejected by check_sealed."""
    plain = json.dumps({"methods": [{"key": "x", "label": "X"}], "cells": {}}).encode() * 64
    envelope = {"v": 1, "kdf": "PBKDF2-SHA256", "iter": 600000, "salt": "AA==", "iv": "AA==",
                "ct": base64.b64encode(plain).decode()}
    text = "window.RESULTS_V2_SEALED[\"t\"]=" + json.dumps(envelope) + ";\n"
    bad = [] if check_sealed(text, "selftest") else ["selftest: check_sealed accepted a plaintext payload"]
    probe = SITE / "data" / ".guard_selftest.js"
    try:
        probe.write_text('window.X={"fit_time":[1,2]};\n', encoding="utf-8")
        if not check_no_as_run_time(probe):
            bad.append("selftest: check_no_as_run_time accepted an as-run time metric")
    finally:
        probe.unlink(missing_ok=True)
    ranks = ('window.RESULTS_V2_RANKS=window.RESULTS_V2_RANKS||{};(function(){var R=window.RESULTS_V2_RANKS;R["t"]=R["t"]||{};'
             'Object.assign(R["t"].at,{"e2e":{"t1":4}});Object.assign(R["t"].pairs,{"hidden-method|e2e":{"nguyen":{"4":[12,3,4]}}});})();\n')
    if "hidden-method" not in (rank_methods(ranks) or set()):
        bad.append("selftest: rank_methods missed a method named only in a pair")
    compared = ranks.replace("})();", 'Object.assign(R["t"].rungs,{"other-hidden|e2e":{"t1":[4,4]}});})();')
    if not {"hidden-method", "other-hidden"} <= (rank_methods(compared) or set()):
        bad.append("selftest: rank_methods missed a method named only in the compared-rungs record")
    return bad


def main() -> int:
    failures = selftest()
    index = (SITE / "index.html").read_text(encoding="utf-8")
    for needle in ("private/", "index.local"):
        if needle in index:
            failures.append(f"index.html mentions {needle!r}")
    for js in sorted((SITE / "data").glob("*/results.js")):
        payload = payload_of(js.read_text(encoding="utf-8"), "RESULTS_V2")
        if not payload:
            failures.append(f"{js}: not a RESULTS_V2 payload")
            continue
        keys = {mm["key"] for mm in payload.get("methods", [])} | set(payload.get("cells", payload.get("data", {}))) | set(payload.get("status", {})) | set(payload.get("timing", {}))
        extra = sorted(keys - PUBLIC_METHODS)
        if extra:
            failures.append(f"{js}: non-public method keys {extra}")
        have = {m["key"] for m in payload.get("metrics", [])}
        missing = sorted(REQUIRED_METRICS - have)
        if missing:
            failures.append(f"{js}: metric registry lacks {missing}")
        for hj in sorted(js.parent.glob("hist/*.js")):
            ks = keys_in_wrapped(hj.read_text(encoding="utf-8"), r"\.cells,(\{.*\})\);\}\)\(\);\s*$")
            if ks is None:
                failures.append(f"{hj}: not a histogram file")
                continue
            extra = sorted(ks - PUBLIC_METHODS)
            if extra:
                failures.append(f"{hj}: non-public method keys {extra}")
        pj = js.parent / "paired.js"
        if pj.exists():
            ks = keys_in_wrapped(pj.read_text(encoding="utf-8"), r"Object\.assign\(R\[[^\]]*\],(\{.*\})\);\}\)\(\);\s*$")
            if ks is None:
                failures.append(f"{pj}: not a paired file")
            else:
                extra = sorted({k for pair in ks for k in pair.split("|")} - PUBLIC_METHODS)
                if extra:
                    failures.append(f"{pj}: non-public method keys {extra}")
        rj = js.parent / "ranks.js"
        if rj.exists():
            named = rank_methods(rj.read_text(encoding="utf-8"))
            if named is None:
                failures.append(f"{rj}: not a ranks file")
            elif sorted(named - PUBLIC_METHODS):
                failures.append(f"{rj}: non-public method keys {sorted(named - PUBLIC_METHODS)}")
    for pub in sorted((SITE / "data").rglob("*.js")):
        if pub.name != "sealed.js":     # the sealed payload is encrypted and is not a published number
            failures.extend(check_no_as_run_time(pub))
    for sj in sorted((SITE / "data").glob("*/sealed.js")):
        failures.extend(check_sealed(sj.read_text(encoding="utf-8"), str(sj)))
    if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
        for p in ("private", "index.local.html"):
            if (SITE / p).exists():
                failures.append(f"{p} exists in the CI checkout")
    for f in failures:
        print("PUBLIC GUARD:", f)
    print("public guard: OK" if not failures else f"public guard: {len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
