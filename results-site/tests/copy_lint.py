#!/usr/bin/env python3
"""Copy lint for the results site: banned patterns in VIEWER-FACING text.

Two surfaces: the full prose of index.html, and the string literals of explorer.js
(comments are internal and exempt). Every entry here is a fixed bug that must not return —
extend the list whenever a new wording bug is fixed.
"""
import re
import sys
from pathlib import Path

SITE = Path(__file__).resolve().parent.parent

# pattern -> why it is banned
BANNED = {
    r"s per expression": "time is denominated per problem; 'per expression' reads per-candidate",
    r"re-run build_": "internal instruction leaked to viewers",
    r"paired payload": "internal jargon; say what the reader sees (numbers, releases)",
    r"live in Python": "architecture justification leaked to viewers",
    r"block design": "stats jargon in a guard message",
    r"(?<!tap or )hover any cell": "hover-only affordance; must be 'tap or hover'",
    r"(?<!tap or )hover for the": "hover-only affordance; must be 'tap or hover'",
    r"precomputed in Python": "architecture note; use reproducibility framing",
    r"your previous pick": "assumes a deliberate choice; defaults are not picks",
    r"n = \d+ problems": "counts are denominated in expressions site-wide",
    r"league": "dropped vocabulary — 'primary'/'exploratory' and 'ranking' suffice",
}

def js_strings(source: str) -> list[str]:
    """Double/single-quoted string literals of explorer.js, comments stripped (crudely but
    sufficiently: full-line comments and block comments; inline '//' inside strings is safe
    because we extract strings first from non-comment lines)."""
    lines = []
    in_block = False
    for line in source.splitlines():
        stripped = line.strip()
        if in_block:
            if "*/" in stripped:
                in_block = False
            continue
        if stripped.startswith("/*"):
            in_block = "*/" not in stripped
            continue
        if stripped.startswith("//") or stripped.startswith("*"):
            continue
        lines.append(line)
    text = "\n".join(lines)
    # double-quoted literals only: all viewer copy uses them, and apostrophes inside them
    # ("method's") would make a single-quote scan pair across strings and swallow code
    return re.findall(r'"((?:[^"\\]|\\.)*)"', text)

def main() -> int:
    failures = []
    surfaces = {
        "index.html": (SITE / "index.html").read_text(encoding="utf-8"),
        "explorer.js (strings)": "\n".join(js_strings((SITE / "explorer.js").read_text(encoding="utf-8"))),
    }
    for pattern, why in BANNED.items():
        for name, text in surfaces.items():
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                snippet = text[max(0, match.start() - 40):match.end() + 40].replace("\n", " ")
                failures.append(f"{name}: /{pattern}/ ({why})\n    …{snippet}…")
    # em-dash budget: AI prose overuses them; colons, semicolons and structure read better.
    # index.html allows 0; explorer.js strings allow 2 (the matrix-diagonal placeholders).
    for name, text, budget in [("index.html", surfaces["index.html"], 0),
                               ("explorer.js (strings)", surfaces["explorer.js (strings)"], 2)]:
        count = text.count("—") + text.count("&mdash;")
        if count > budget:
            failures.append(f"{name}: {count} em-dashes (budget {budget}) — rewrite with "
                            "colons, semicolons, parentheses, or structure")
    if failures:
        print(f"COPY LINT: {len(failures)} banned pattern(s) found:\n")
        print("\n".join(failures))
        return 1
    print(f"copy lint clean ({len(BANNED)} banned patterns checked on 2 surfaces)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
