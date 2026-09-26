#!/bin/bash
# Local-only view of the results site with additional, unpublished methods (results-site/README.md, "Local-only
# methods"). Writes results.local.html next to results.html (the explorer's page): the same page plus ONE extra script tag that loads
# private/2026-09/results_v2_private.js (both paths are git-ignored). The public results.html is not touched and contains
# no reference to either file. Serve locally:  cd results-site && python3 -m http.server 8765  -> /results.local.html
set -eu
cd "$(dirname "$0")"
[ -f private/2026-09/results_v2_private.js ] || { echo "no private/2026-09/results_v2_private.js (write it with scripts/site_export_v2.py --private ... --private-out results-site/private/2026-09/results_v2_private.js)"; exit 1; }
for f in private/2026-09/results_v2_private.js results.local.html; do git check-ignore -q "$f" || { echo "REFUSED: $f is not git-ignored"; exit 1; }; done
python3 - <<'PY'
s = open("results.html").read()
tag = '<script src="private/2026-09/results_v2_private.js" charset="utf-8"></script>\n  <script src="explorer_v2.js"'
assert s.count('<script src="explorer_v2.js"') == 1, "results.html: explorer_v2.js script tag not found exactly once"
open("results.local.html", "w").write(s.replace('<script src="explorer_v2.js"', tag, 1).replace("<title>", "<title>[local] ", 1))
print("wrote results.local.html (local only)")
PY
