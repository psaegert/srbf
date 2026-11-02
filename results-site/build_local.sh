#!/bin/bash
# Local-only view of the results site with additional, unpublished methods (results-site/README.md, "Local-only
# methods"). Writes explorer.local.html next to explorer.html (the explorer's page): the same page plus ONE extra script tag that loads
# private/2026-09/units.local.js, every private unit's unit.js one after another (all three paths are git-ignored). The
# public explorer.html is not touched and contains no reference to any of them. Serve locally:
#   cd results-site && python3 -m http.server 8765  -> /explorer.local.html
set -eu
cd "$(dirname "$0")"
P=private/2026-09
units=$(find "$P/m" "$P/p" -name unit.js 2>/dev/null | sort)
[ -n "$units" ] || { echo "no units under $P (write them with scripts/site_export_v2.py --private ... --private-dir results-site/$P)"; exit 1; }
for f in "$P/units.local.js" explorer.local.html; do git check-ignore -q "$f" || { echo "REFUSED: $f is not git-ignored"; exit 1; }; done
cat $units > "$P/units.local.js"
python3 - <<'PY'
s = open("explorer.html").read()
tag = '<script src="private/2026-09/units.local.js" charset="utf-8"></script>\n  <script src="explorer_v2.js"'
assert s.count('<script src="explorer_v2.js"') == 1, "explorer.html: explorer_v2.js script tag not found exactly once"
open("explorer.local.html", "w").write(s.replace('<script src="explorer_v2.js"', tag, 1).replace("<title>", "<title>[local] ", 1))
print("wrote explorer.local.html (local only)")
PY
echo "$(echo "$units" | wc -l) unit(s) in $P/units.local.js"
