"""The public results site must carry no trace of local-only methods (README.md, "Local-only methods").

Fatal checks, run before the Playwright suite in CI and locally:
  1. index.html references neither a private/ path nor index.local.html;
  2. every method key in data/*/results.js is in the public allowlist below (the list names PUBLIC methods only);
  3. in CI, results-site/private/ and index.local.html do not exist in the checkout (they are git-ignored; a forced
     add would surface here before anything deploys).
"""
import json
import os
import re
import sys
from pathlib import Path

SITE = Path(__file__).resolve().parents[1]
PUBLIC_METHODS = {"e2e", "nesymres-100M", "PySR", "T8-3M", "T8-20M", "T8-120M", "prior"}


def main() -> int:
    failures = []
    index = (SITE / "index.html").read_text(encoding="utf-8")
    for needle in ("private/", "index.local"):
        if needle in index:
            failures.append(f"index.html mentions {needle!r}")
    for js in sorted((SITE / "data").glob("*/results.js")):
        text = js.read_text(encoding="utf-8")
        m = re.match(r"window\.RESULTS_V2 = (.*);\s*$", text, re.S)
        if not m:
            failures.append(f"{js}: not a RESULTS_V2 payload"); continue
        payload = json.loads(m.group(1))
        keys = {mm["key"] for mm in payload.get("methods", [])} | set(payload.get("data", {})) | set(payload.get("status", {})) | set(payload.get("timing", {}))
        extra = sorted(keys - PUBLIC_METHODS)
        if extra:
            failures.append(f"{js}: non-public method keys {extra}")
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
