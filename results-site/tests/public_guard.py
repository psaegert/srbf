"""The public results site must carry no trace of local-only methods (README.md, "Local-only methods").

Fatal checks, run before the Playwright suite in CI and locally:
  1. index.html references neither a private/ path nor index.local.html;
  2. every method key in data/*/results.js, data/*/hist/*.js and data/*/paired.js is in the public allowlist below
     (the list names PUBLIC methods only; a private method's key must never appear here);
  3. every release payload carries the complete metric registry (at least the 2026-07 site's metrics, under their
     schema-2 keys) so a regenerated release cannot silently lose metrics;
  4. in CI, results-site/private/ and index.local.html do not exist in the checkout (they are git-ignored; a forced
     add would surface here before anything deploys).
"""
import json
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
    "n_constants_delta", "total_nestedness_delta", "predicted_log_prob", "predicted_score", "fit_time",
    "symbolic_recovery", "mdl_ratio", "r2_val"}


def payload_of(text: str, var: str) -> Any:
    m = re.match(r"window\." + var + r" = (.*);\s*$", text, re.S)
    return json.loads(m.group(1)) if m else None


def keys_in_wrapped(text: str, pattern: str) -> set[str] | None:
    m = re.search(pattern, text, re.S)
    return set(json.loads(m.group(1))) if m else None


def main() -> int:
    failures = []
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
