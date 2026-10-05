"""The public results site must carry no trace of local-only methods (README.md, "Local-only methods").

Fatal checks, run before the Playwright suite in CI and locally:
  1. no page (index.html and the pages around it) references a private/ path or a local page (*.local.html);
  2. every method key in data/*/results.js, data/*/summary.js, data/*/hist/*.js, data/*/paired.js, data/*/ranks.js,
     data/*/pred/ and data/*/pv/ is
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
     the same holds for every file under data/*/sealed/ (the files tools/seal.mjs seals one by one): one
     envelope line, named by its own file name, an IV of 12 bytes, ciphertext that reads as ciphertext, nothing else;
  7. no text a reader can see in a release payload (the protocol texts, the timing note, the labels and
     descriptions of metrics, methods and catalogs) matches a banned pattern. The page lint reads index.html and the
     explorer's strings; a payload is written by the exporter from files outside this repository, so it is read here,
     where a release is checked before it is published. The patterns are copy_lint's, the maintainer's local ones
     included (results-site/private/banned_patterns.json, git-ignored and absent in CI).
"""
import base64
import gzip
import json
import math
import os
import re
import sys
import tempfile
import zlib
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
    """The first JSON object that directly follows one of `markers` (a ranks.js hands its objects over as
    Object.assign(target, {...}) or, for the pairs it may first map onto another key list, as `var P={...}`)."""
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


def rank_methods(text: str) -> set[str] | None:
    """Every method key a ranks.js names: the methods with a time-budget rung, both sides of every pair, and both
    sides of every record of which rungs a time-limit outcome compared."""
    at, pairs = _object_after(text, ".at,"), _object_after(text, ".pairs,", "var P=")
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


# A file sealed on its own (tools/seal.mjs): exactly this line, named by its own file name. The line's text
# is fixed but for the name (32 hex digits) and two base64 strings, so no method key or path can be in it.
SEALED_FILE = re.compile(r'\(window\.RESULTS_V2_SEALED_FILES=window\.RESULTS_V2_SEALED_FILES\|\|\{\}\)\["([0-9a-f]{32})"\]='
                         r'\{"iv":"([A-Za-z0-9+/]+={0,2})","ct":"([A-Za-z0-9+/]+={0,2})"\};\n?')
SEALED_FILE_NAME = re.compile(r"[0-9a-f]{32}\.js")
GCM_TAG, MIN_GZIP = 16, 20   # bytes: AES-GCM's tag, and the shortest gzip stream (header, an empty block, trailer)
# a byte of text: printable ASCII, tab, newline, carriage return. Ciphertext has 98/256 of them; text has nothing else.
TEXT_BYTES = frozenset(range(0x20, 0x7f)) | {0x09, 0x0a, 0x0d}


def check_sealed_file(text: str, name: str, stem: str) -> list[str]:
    """One file sealed on its own: the envelope line and nothing else, its own name inside, and ciphertext that is neither
    plaintext nor a gzip stream left unencrypted. A file can be a few hundred bytes, too short for the entropy test of a
    whole payload; the share of text bytes tells text from ciphertext at any length the format allows (a text file sits
    at 1, ciphertext at 98/256, and the bar is six standard deviations above that)."""
    m = SEALED_FILE.fullmatch(text)
    if not m:
        return [f"{name}: not one sealed-file envelope line"]
    bad = [] if m.group(1) == stem else [f"{name}: the envelope is named {m.group(1)}, the file {stem}"]
    try:
        iv, ct = base64.b64decode(m.group(2), validate=True), base64.b64decode(m.group(3), validate=True)
    except Exception:
        return bad + [f"{name}: IV or ciphertext is not base64"]
    if len(iv) != 12:
        bad.append(f"{name}: an IV of {len(iv)} bytes (AES-GCM's is 12)")
    if len(ct) < GCM_TAG + MIN_GZIP:
        return bad + [f"{name}: ciphertext is {len(ct)} bytes, shorter than any sealed file"]
    for tell in PLAINTEXT_TELLS:
        if tell in ct:
            bad.append(f"{name}: ciphertext contains {tell!r}")
    try:
        zlib.decompress(ct, 47)   # a gzip or zlib stream: compressed, never encrypted
        bad.append(f"{name}: the ciphertext is a compressed stream, not ciphertext")
    except zlib.error:
        pass
    n, p0 = len(ct), len(TEXT_BYTES) / 256
    if sum(b in TEXT_BYTES for b in ct) > n * p0 + 6 * math.sqrt(n * p0 * (1 - p0)):
        bad.append(f"{name}: ciphertext reads as text")
    if n >= 1024 and entropy(ct) < 7.5:
        bad.append(f"{name}: ciphertext entropy {entropy(ct):.2f} bits/byte reads as plaintext")
    return bad


def check_sealed_dir(sd: Path) -> list[str]:
    """data/<release>/sealed/ holds files sealed one by one and nothing else."""
    if not sd.is_dir():
        return [f"{sd}: not a directory of sealed files"]
    bad = []
    for f in sorted(sd.rglob("*")):
        if f.is_dir() or f.parent != sd or not SEALED_FILE_NAME.fullmatch(f.name):
            bad.append(f"{f}: not a file sealed one by one (only <32 hex digits>.js belong in {sd.name}/)")
            continue
        try:
            text = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            bad.append(f"{f}: not text")
            continue
        bad.extend(check_sealed_file(text, str(f), f.name[:-3]))
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
    and the finished and in-progress lists of the progress summary (a scheduled method is named by its label only)."""
    summary = payload.get("summary") or {}
    return ({mm["key"] for mm in payload.get("methods", [])} | set(payload.get("cells", payload.get("data", {})))
            | set(payload.get("status", {})) | set(payload.get("progress", {})) | set(payload.get("timing", {}))
            | set(summary.get("finished", [])) | set(summary.get("in_progress", [])))


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
    bad.extend(selftest_sealed_files())
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
    mapped = ('window.RESULTS_V2_RANKS=window.RESULTS_V2_RANKS||{};(function(){var R=window.RESULTS_V2_RANKS,rel="t",K=["a"],MAIN=true;'
              'var X=R[rel]=R[rel]||{keys:K,at:{},pairs:{}};var P={"mapped-hidden|e2e":{"nguyen":{"4":[12,3,4]}}};'
              'Object.assign(X.at,{"e2e":{"t1":4}});Object.assign(X.pairs,P);Object.assign(X.rungs,{});})();\n')
    if "mapped-hidden" not in (rank_methods(mapped) or set()):
        bad.append("selftest: rank_methods missed a method named in pairs handed over as var P")
    if not {"hidden-method", "other-hidden"} <= (rank_methods(compared) or set()):
        bad.append("selftest: rank_methods missed a method named only in the compared-rungs record")
    if not {"hidden-a", "hidden-b"} <= method_keys({"summary": {"finished": ["hidden-a"], "in_progress": ["hidden-b"], "scheduled": []}}):
        bad.append("selftest: method_keys missed a method named only in the progress summary")
    probe_payload = {"release": {"scoring": "fine"}, "timing_note": "measured on the forbidden-host", "cells": {"m": "the forbidden-host is numeric bulk"}}
    hits = check_payload_texts(probe_payload, "selftest", {r"forbidden-host": "selftest"})
    if len(hits) != 1 or "/timing_note" not in hits[0]:
        bad.append("selftest: check_payload_texts did not flag a banned word in a reader-facing text (or read the numeric bulk)")
    return bad


def selftest_sealed_files() -> list[str]:
    """The sealed-file check rejects a plaintext file, an unencrypted gzip stream, a misnamed envelope and a stray
    file, and accepts what tools/seal.mjs writes (the committed fixture, tests/fixtures/sealed_files/sealed/)."""
    bad = []
    stem = "0123456789abcdef" * 2

    def line(name: str, iv: bytes, ct: bytes) -> str:
        return ('(window.RESULTS_V2_SEALED_FILES=window.RESULTS_V2_SEALED_FILES||{})["%s"]={"iv":"%s","ct":"%s"};\n'
                % (name, base64.b64encode(iv).decode(), base64.b64encode(ct).decode()))
    plain = b'window.RESULTS_V2_PP=window.RESULTS_V2_PP||{};R["2026-09"]["hidden|feynman|16|1"]={"n":100,"s":"AwEBAQEB"};'
    ciphertext = bytes((i * 167 + 13) % 256 for i in range(600))
    if check_sealed_file(line(stem, bytes(12), ciphertext), "selftest", stem):
        bad.append("selftest: check_sealed_file refused a well-formed sealed file")
    for what, text in [("a plaintext file", plain.decode() + "\n"),
                       ("plaintext in an envelope", line(stem, bytes(12), plain)),
                       ("an unencrypted gzip stream", line(stem, bytes(12), gzip.compress(plain * 8))),
                       ("an envelope named for another file", line("f" * 32, bytes(12), ciphertext)),
                       ("an IV of the wrong length", line(stem, bytes(16), ciphertext)),
                       ("a second line", line(stem, bytes(12), ciphertext) + plain.decode() + "\n")]:
        if not check_sealed_file(text, "selftest", stem):
            bad.append(f"selftest: check_sealed_file accepted {what}")
    with tempfile.TemporaryDirectory() as tmp:
        sd = Path(tmp) / "sealed"
        sd.mkdir()
        (sd / f"{stem}.js").write_bytes(plain + b"\n")
        (sd / "notes.txt").write_text("a stray file")
        found = check_sealed_dir(sd)
        if not any(f"{stem}.js" in f for f in found) or not any("notes.txt" in f for f in found):
            bad.append("selftest: check_sealed_dir accepted a plaintext file or a stray file under sealed/")
    written = sorted((Path(__file__).resolve().parent / "fixtures" / "sealed_files" / "sealed").glob("*.js"))
    if not written:
        bad.append("selftest: no sealed files in tests/fixtures/sealed_files/sealed/ to check against")
    for f in written:
        if check_sealed_file(f.read_text(encoding="utf-8"), str(f), f.name[:-3]):
            bad.append(f"selftest: check_sealed_file refused {f.name}, which tools/seal.mjs wrote")
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
        pj = js.parent / "paired.js"
        if pj.exists():
            ks = keys_in_wrapped(pj.read_text(encoding="utf-8"), r"Object\.assign\(R\[[^\]]*\],(\{.*\})\);\}\)\(\);\s*$")
            if ks is None:
                failures.append(f"{pj}: not a paired file")
            else:
                extra = sorted({k for pair in ks for k in pair.split("|")} - PUBLIC_METHODS)
                if extra:
                    failures.append(f"{pj}: non-public method keys {extra}")
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
    for sd in sorted((SITE / "data").glob("*/sealed")):   # the files sealed one by one (per problem, per metric)
        failures.extend(check_sealed_dir(sd))
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
