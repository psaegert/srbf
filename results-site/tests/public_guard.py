"""The public results site must carry no trace of local-only methods (README.md, "Local-only methods").

Fatal checks, run before the Playwright suite in CI and locally:
  1. no page (index.html and the pages around it) references a private/ path or a local page (*.local.html);
  2. every method key in data/*/results.js (its slots and indexes included), data/*/summary.js, data/*/hist/*.js,
     data/*/ranks/*.js, data/*/paired/*.js, data/*/pred/ and data/*/pv/ is
     in the public allowlist below
     (the list names PUBLIC methods only; a private method's key must never appear here);
  3. every release payload carries the complete metric registry (at least the metric floor: the site's first
     release's metrics under their schema-2 keys, and the headline ones) so a regenerated release cannot silently
     lose metrics;
  4. in CI, results-site/private/ and the local pages do not exist in the checkout (they are git-ignored; a forced
     add would surface here before anything deploys);
  5. no published payload carries an as-run wall-clock metric. Seconds measured where a unit happened to run are
     not comparable between methods; the only timing this benchmark publishes is the reference-machine ladder in
     timing.json: only times measured on the reference machine are published;
  6. a sealed payload, if one is present, is sealed: the envelope carries only its own fields, the KDF is strong
     enough to be worth having, and the ciphertext reads as ciphertext (high entropy, no plaintext left in it).
     The checker is run against a deliberately bad envelope on every invocation, so it cannot pass vacuously;
  7. no text a reader can see in a release payload (the protocol texts, the timing note, the labels and
     descriptions of metrics, methods and catalogs) matches a banned pattern. The page lint reads index.html and the
     explorer's strings; a payload is written by the exporter from files outside this repository, so it is read here,
     where a release is checked before it is published. The patterns are copy_lint's, the maintainer's local ones
     included (results-site/private/banned_patterns.json, git-ignored and absent in CI).
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
# T8-120M-pysr replaced T8-20M-pysr on the page (2026-09-29); the 20M hybrid stays admissible, it is public-safe.
# The oracle is public (owner 2026-09-25: "as all other methods, in black with a dashed line ... it is important that we
# show it"; its results reached the board 2026-10-05).
PUBLIC_METHODS = {"e2e", "nesymres-100M", "PySR", "operon", "gpgomea", "dsr", "udsr", "rilsrols", "qlattice", "T8-3M", "T8-20M",
                  "T8-120M", "T8-120M-pysr", "T8-20M-pysr", "oracle"}
# Methods with results that are withheld from the public page: the key checks below catch their keys, these their names
# in the texts. The Flash-ANSR prior is private since 2026-09-30 (owner: "hidden from the site ... private with a key");
# it lives in the sealed overlay. (T8-20M-pysr was withheld 2026-09-28 until its re-run under the two-part code.)
WITHHELD_NAMES: dict[str, str] = {r"Flash-ANSR prior": "the prior is private: sealed overlay only (owner 2026-09-30)"}
# the metric floor: the site's first release's 21 metrics under the schema-2 keys (symbolic_recovery there = skeleton_match_raw here,
# prediction_success_rate = success), plus the release's own headline metrics
REQUIRED_METRICS = {
    "numeric_recovery_val", "expr_length_ratio", "log10_fvu_val", "log10_fvu_fit", "numeric_recovery_fit", "success",
    "skeleton_match_raw", "f1_score", "precision_score", "recall_score", "edit_distance_norm", "zss_edit_distance",
    "expr_length_ratio_abserr", "predicted_skeleton_prefix_length", "skeleton_length", "n_constants_ratio",
    "n_constants_delta", "total_nestedness_delta", "predicted_log_prob", "predicted_score",
    "symbolic_recovery", "symbolic_recovery_mask_fittable", "symbolic_recovery_mask_none", "mdl_ratio", "r2_val"}
# Wall-clock measured wherever a unit ran. Never published: not in the registry, not in a cell, not in a
# histogram, not in a paired contrast. The reference-machine ladder (timing.json) is the only timing that ships.
UNCALIBRATED_TIME_KEYS = ("fit_time", "generation_time")


def payload_of(text: str, var: str) -> Any:
    m = re.match(r"window\." + var + r" = (.*);\s*$", text, re.S)
    return json.loads(m.group(1)) if m else None


def keys_in_wrapped(text: str, pattern: str) -> set[str] | None:
    m = re.search(pattern, text, re.S)
    return set(json.loads(m.group(1))) if m else None


def _object_after(text: str, *markers: str) -> dict | None:
    """The first JSON object that directly follows one of `markers` (a per-key cell file hands its cells over as
    `var P={...}`)."""
    for marker in markers:
        i = text.find(marker)
        while i >= 0:
            try:
                obj, _ = json.JSONDecoder().raw_decode(text, i + len(marker))
            except json.JSONDecodeError:
                obj = None
            if isinstance(obj, dict):
                return obj
            i = text.find(marker, i + 1)
    return None


def cell_methods(text: str) -> set[str] | None:
    """Every method key a ranks/<key>.js or paired/<key>.js names: both sides of every pair, for every metric the file
    hands over (as `var P={metric: {pair: ...}}`)."""
    cells = _object_after(text, "var P=")
    if cells is None or not all(isinstance(per, dict) for per in cells.values()):
        return None
    return {k for per in cells.values() for pair in per for k in pair.split("|")}


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
    """Everything that must hold for a sealed file, whatever it holds: one line with the envelope a page has always read
    (RESULTS_V2_SEALED), at most one more with the envelopes sealed under further keys (RESULTS_V2_SEALED_MORE, written
    first by tools/seal.mjs --source ...), nothing else; and every envelope passes the same checks."""
    first, more, envs = None, None, []
    for line in text.strip().split("\n"):
        m = re.search(r"window\.RESULTS_V2_SEALED\[[^\]]*\]=(\{.*\});$", line)
        n = re.fullmatch(r"window\.RESULTS_V2_SEALED_MORE=window\.RESULTS_V2_SEALED_MORE\|\|\{\};"
                         r"window\.RESULTS_V2_SEALED_MORE\[[^\]]*\]=(\[.*\]);", line)
        if m and first is None:
            first = m.group(1)
        elif n and more is None:
            more = n.group(1)
        else:
            return [f"{name}: a line that is neither a sealed envelope nor the further ones, or one of them twice"]
    if first is None:
        return [f"{name}: not a sealed envelope"]
    try:
        envs = [json.loads(first)] + (json.loads(more) if more is not None else [])
    except ValueError:
        return [f"{name}: envelope is not JSON"]
    if not all(isinstance(e, dict) for e in envs):
        return [f"{name}: the further envelopes are not a list of envelopes"]
    bad: list[str] = []
    for i, env in enumerate(envs):
        bad.extend(check_envelope(env, name if i == 0 else f"{name} (envelope {i + 1})"))
    return bad


def check_envelope(env: dict[str, Any], name: str) -> list[str]:
    """One envelope: its own fields only, a strong KDF, and ciphertext that does not read as plaintext."""
    bad = []
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


def payload_texts(node: Any, path: str = "") -> list[tuple[str, str]]:
    """Every string value of a payload with its path, the numeric bulk aside (cells, status, progress, timing hold no prose)."""
    if isinstance(node, str):
        return [(path, node)]
    if isinstance(node, dict):
        return [t for k, v in node.items() if not (path == "" and k in ("cells", "data", "status", "progress", "timing")) for t in payload_texts(v, f"{path}/{k}")]
    if isinstance(node, list):
        return [t for i, v in enumerate(node) for t in payload_texts(v, f"{path}[{i}]")]
    return []


def method_keys(payload: Any) -> set[str]:
    """Every method key a release payload or its summary names: its methods, their cells, status, progress and times,
    the finished and in-progress lists of the progress summary (a scheduled method is named by its label only), and
    the methods its slots place."""
    summary = payload.get("summary") or {}
    slots = payload.get("slots") or {}   # where every method sits at every slot; an overlay's record of its partners
    basis = slots.get("basis") or {}
    return ({mm["key"] for mm in payload.get("methods", [])} | set(payload.get("cells", payload.get("data", {})))
            | set(payload.get("status", {})) | set(payload.get("progress", {})) | set(payload.get("timing", {}))
            | set(summary.get("finished", [])) | set(summary.get("in_progress", []))
            | set(slots.get("at") or {}) | set(slots.get("stamp") or {}) | set(basis) | {q for per in basis.values() for q in per})


def check_payload_texts(payload: Any, name: str, banned: dict[str, str]) -> list[str]:
    out = []
    for path, text in payload_texts(payload):
        for pattern, why in banned.items():
            m = re.search(pattern, text)
            if m:
                out.append(f"{name}: {path} reads {m.group(0)!r} ({why})")
    return out


def banned_patterns() -> dict[str, str]:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import copy_lint   # the one list of what a public page must not say, local patterns merged in
    return dict(copy_lint.BANNED)


def selftest() -> list[str]:
    """The guard checks itself: a blob that is plainly not sealed must be rejected by check_sealed."""
    plain = json.dumps({"methods": [{"key": "x", "label": "X"}], "cells": {}}).encode() * 64
    envelope = {"v": 1, "kdf": "PBKDF2-SHA256", "iter": 600000, "salt": "AA==", "iv": "AA==",
                "ct": base64.b64encode(plain).decode()}
    text = "window.RESULTS_V2_SEALED[\"t\"]=" + json.dumps(envelope) + ";\n"
    bad = [] if check_sealed(text, "selftest") else ["selftest: check_sealed accepted a plaintext payload"]
    # a plaintext payload behind a sealed one is as plain: every envelope is checked
    sealed_first = {"v": 1, "kdf": "PBKDF2-SHA256", "iter": 600000, "salt": "AA==", "iv": "AA==",
                    "ct": base64.b64encode(bytes(range(256)) * 8).decode()}
    two = ("window.RESULTS_V2_SEALED_MORE=window.RESULTS_V2_SEALED_MORE||{};window.RESULTS_V2_SEALED_MORE[\"t\"]="
           + json.dumps([envelope]) + ";\nwindow.RESULTS_V2_SEALED=window.RESULTS_V2_SEALED||{};window.RESULTS_V2_SEALED[\"t\"]="
           + json.dumps(sealed_first) + ";\n")
    if check_sealed(two.replace(json.dumps([envelope]), json.dumps([sealed_first])), "selftest"):
        bad.append("selftest: check_sealed refused two sealed envelopes")
    if not check_sealed(two, "selftest"):
        bad.append("selftest: check_sealed accepted a plaintext envelope behind a sealed one")
    probe = SITE / "data" / ".guard_selftest.js"
    try:
        probe.write_text('window.X={"fit_time":[1,2]};\n', encoding="utf-8")
        if not check_no_as_run_time(probe):
            bad.append("selftest: check_no_as_run_time accepted an as-run time metric")
    finally:
        probe.unlink(missing_ok=True)
    cells = ('window.RESULTS_V2_RANKCELLS=window.RESULTS_V2_RANKCELLS||{};(function(){var R=window.RESULTS_V2_RANKCELLS,rel="t";'
             'var X=R[rel]=R[rel]||{};var P={"log10_fvu_val":{"e2e|hidden-method":{"nguyen":{"64":[12,3,4],"t10":[12,1,2]}}}};'
             'Object.keys(P).forEach(function(k){var Y=X[k]=X[k]||{};Object.keys(P[k]).forEach(function(p){Y[p]=P[k][p];});});})();\n')
    if "hidden-method" not in (cell_methods(cells) or set()):
        bad.append("selftest: cell_methods missed a method named only in a pair of a ranks/ file")
    two = cells.replace('{"log10_fvu_val":', '{"f1_score":{},"f1_score@answered":{"other-hidden|e2e":{}},"log10_fvu_val":')
    if not {"hidden-method", "other-hidden"} <= (cell_methods(two) or set()):
        bad.append("selftest: cell_methods missed a method named in the answered-only reading of a paired/ file")
    placed = {"slots": {"at": {"hidden-at": {}}, "stamp": {"hidden-stamp": {}}, "basis": {"e2e": {"hidden-partner": {}}}}}
    if not {"hidden-at", "hidden-stamp", "hidden-partner"} <= method_keys(placed):
        bad.append("selftest: method_keys missed a method named only in the slots")
    if not {"hidden-a", "hidden-b"} <= method_keys({"summary": {"finished": ["hidden-a"], "in_progress": ["hidden-b"], "scheduled": []}}):
        bad.append("selftest: method_keys missed a method named only in the progress summary")
    probe_payload = {"release": {"scoring": "fine"}, "timing_note": "measured on the forbidden-host", "cells": {"m": "the forbidden-host is numeric bulk"}}
    hits = check_payload_texts(probe_payload, "selftest", {r"forbidden-host": "selftest"})
    if len(hits) != 1 or "/timing_note" not in hits[0]:
        bad.append("selftest: check_payload_texts did not flag a banned word in a reader-facing text (or read the numeric bulk)")
    return bad


def main() -> int:
    failures = selftest()
    banned = {**banned_patterns(), **WITHHELD_NAMES}
    for page in sorted(p for p in SITE.glob("*.html") if not p.name.endswith(".local.html")):
        html = page.read_text(encoding="utf-8")
        for needle in ("private/", "index.local", "results.local", "explorer.local"):
            if needle in html:
                failures.append(f"{page.name} mentions {needle!r}")
    for js in sorted((SITE / "data").glob("*/results.js")):
        payload = payload_of(js.read_text(encoding="utf-8"), "RESULTS_V2")
        if not payload:
            failures.append(f"{js}: not a RESULTS_V2 payload")
            continue
        extra = sorted(method_keys(payload) - PUBLIC_METHODS)
        if extra:
            failures.append(f"{js}: non-public method keys {extra}")
        sj = js.parent / "summary.js"   # the Progress page's and the guide's few kB
        if sj.exists():
            summary = payload_of(sj.read_text(encoding="utf-8"), "RESULTS_V2_SUMMARY")
            if not summary:
                failures.append(f"{sj}: not a RESULTS_V2_SUMMARY payload")
            else:
                extra = sorted(method_keys(summary) - PUBLIC_METHODS)
                if extra:
                    failures.append(f"{sj}: non-public method keys {extra}")
                failures.extend(check_payload_texts(summary, str(sj.relative_to(SITE)), banned))
        failures.extend(check_payload_texts(payload, str(js.relative_to(SITE)), banned))
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
        for sub in ("ranks", "paired"):   # the per-key cells along the slots: both sides of every pair
            for cj in sorted((js.parent / sub).glob("*.js")):
                named = cell_methods(cj.read_text(encoding="utf-8"))
                if named is None:
                    failures.append(f"{cj}: not a per-key cell file")
                elif named - PUBLIC_METHODS:
                    failures.append(f"{cj}: non-public method keys {sorted(named - PUBLIC_METHODS)}")
        pd = js.parent / "pred"   # the Predictions view's files: one directory per method, and the ground truth
        if pd.is_dir():
            extra = sorted(d.name for d in pd.iterdir() if d.is_dir() and d.name != "truth" and d.name not in PUBLIC_METHODS)
            if extra:
                failures.append(f"{pd}: non-public method keys {extra}")
        extra = sorted(set(payload.get("pred") or {}) - PUBLIC_METHODS)
        if extra:
            failures.append(f"{js}: non-public method keys in the predictions index {extra}")
        pv = js.parent / "pv"   # the inputs of the paired and rank statistics: one directory per method
        if pv.is_dir():
            extra = sorted(d.name for d in pv.iterdir() if d.is_dir() and d.name not in PUBLIC_METHODS)
            if extra:
                failures.append(f"{pv}: non-public method keys {extra}")
        extra = sorted(set(payload.get("pv") or {}) - PUBLIC_METHODS)
        if extra:
            failures.append(f"{js}: non-public method keys in the per-problem pair index {extra}")
    for pub in sorted((SITE / "data").rglob("*.js")):
        if pub.name != "sealed.js":     # the sealed payload is encrypted and is not a published number
            failures.extend(check_no_as_run_time(pub))
    for sj in sorted((SITE / "data").glob("*/sealed.js")):
        failures.extend(check_sealed(sj.read_text(encoding="utf-8"), str(sj)))
    if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
        for p in ("private", "index.local.html", "results.local.html", "explorer.local.html"):
            if (SITE / p).exists():
                failures.append(f"{p} exists in the CI checkout")
    for f in failures:
        print("PUBLIC GUARD:", f)
    print("public guard: OK" if not failures else f"public guard: {len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
