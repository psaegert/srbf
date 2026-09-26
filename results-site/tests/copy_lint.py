#!/usr/bin/env python3
"""Copy lint for the results site: banned patterns in VIEWER-FACING text.

Two surfaces: the full prose of index.html and the string literals of explorer_v2.js (comments are internal and
exempt); the reader's vocabulary (BANNED_V2) is also checked on the page's text and on the reader-facing texts the
exporter writes into the data. Every entry here is a fixed bug that must not return: extend the list whenever a new
wording bug is fixed.
"""
import ast
import json
import re
import sys
from html.parser import HTMLParser
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
    r"not a leaderboard": "self-undermining framing: say how to read the table instead",
    r"never quote": "scolding tone: name the quotable alternative instead",
    r"curve read": "retired vocabulary: the split is declared-vs-free, say 'descriptive' (same interpolation)",
}


def _local_patterns() -> dict[str, str]:
    """Patterns kept out of the repository (results-site/private/ is git-ignored): names that must not reach a
    public page and must not be spelled out in a public lint either. Absent in CI, present on a maintainer's machine."""
    path = Path(__file__).resolve().parents[1] / "private" / "banned_patterns.json"
    return json.loads(path.read_text()) if path.is_file() else {}


BANNED.update(_local_patterns())

# The reader's vocabulary: the words of our own pipeline, which a first-time reader cannot know, are banned from the
# page's prose and from the reader-facing texts the exporter and the timing script write into the data. (The
# explorer's own text is checked where the reader sees it, rendered, by the site suite: its string literals mix with
# code here.)
BANNED_V2 = {
    r"\bcatalogs?\b": "say 'problem set'",
    r"\brungs?\b": "say 'budget'",
    r"\bdraws? (\d|per\b|of\b)|\b(\d+|two|its|their|one|per|of|over|the) draws?\b": "say 'run' (every method is run twice); the verb is fine",
    r"\bpooled\b|\bpooling\b": "say what is averaged over which problems",
    r"reference[- ]machine": "say 'our timing workstation' or 'one workstation'",
    r"\bladder\b": "say 'budgets 1, 2, 4, ...'",
    r"\bcanon\b|canonical form": "say 'standard form'",
    r"\bstrat(um|a)\b|stratified": "internal statistics vocabulary",
    r"\bmu\b": "SimpliPy's internal name for its length measure",
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


def svg_geometry(text: str) -> str:
    """The drawable content of an SVG: root tag, <style>, and comments stripped, whitespace
    normalised. Used to keep the inline visual abstract identical to the brand file."""
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"<style>.*?</style>", "", text, flags=re.S)
    svg_match = re.search(r"<svg[^>]*>(.*)</svg>", text, flags=re.S)
    assert svg_match is not None, "no <svg> body found"
    body = svg_match.group(1)
    body = re.sub(r"\s+", " ", body).strip()
    return re.sub(r"> <", "><", body)


def va_drift(index_html: str) -> str | None:
    """The visual abstract is inlined in index.html (so it can follow the site theme) AND
    kept as assets/brand/visual-abstract.svg (for the repo README). Geometry must match."""
    brand = SITE.parent / "assets" / "brand" / "visual-abstract.svg"
    inline = re.search(r'<svg id="va".*?</svg>', index_html, flags=re.S)
    if not inline:
        return "index.html: inline visual abstract (<svg id=\"va\") not found"
    if svg_geometry(inline.group(0)) != svg_geometry(brand.read_text(encoding="utf-8")):
        return ("index.html vs assets/brand/visual-abstract.svg: the inline visual abstract "
                "drifted from the brand file (edit the brand SVG, then regenerate the inline "
                "copy: same geometry, styles stay in styles.css under #va)")
    return None


class _Text(HTMLParser):
    """The text of a page, without its scripts and styles."""

    def __init__(self) -> None:
        super().__init__()
        self.out: list[str] = []

    def handle_data(self, data: str) -> None:
        self.out.append(data)


def page_text(html: str) -> str:
    parser = _Text()
    parser.feed(re.sub(r"<(script|style)\b.*?</\1>", "", html, flags=re.S))
    return " ".join(parser.out)


def data_texts() -> str:
    """The reader-facing strings the exporter and the timing script put into the release data: metric definitions,
    method notes and budget units, the release's protocol texts and the time axis's note."""
    scripts = SITE.parent / "scripts"
    out: list[str] = []
    for name, targets in (("site_export_v2.py", {"METRICS", "METHODS", "PARAM_LABEL", "RELEASE_VERSIONS", "FLASH_ANSR_SELECTION"}),
                          ("site_timing.py", {"NOTE"})):
        tree = ast.parse((scripts / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in targets for t in node.targets):
                out += [c.value for c in ast.walk(node.value) if isinstance(c, ast.Constant) and isinstance(c.value, str) and " " in c.value]
            if isinstance(node, ast.Dict):
                for k, v in zip(node.keys, node.values):
                    if isinstance(k, ast.Constant) and k.value in ("scoring", "judge") and isinstance(v, ast.Constant):
                        out.append(v.value)
            if isinstance(node, ast.FunctionDef) and node.name == "note":
                out += [c.value for c in ast.walk(node) if isinstance(c, ast.Constant) and isinstance(c.value, str) and " " in c.value]
    return "\n".join(out)


def main() -> int:
    failures = []
    surfaces = {
        "index.html": (SITE / "index.html").read_text(encoding="utf-8"),
        "explorer_v2.js (strings)": "\n".join(js_strings((SITE / "explorer_v2.js").read_text(encoding="utf-8"))),
    }
    current = {"index.html (text)": page_text(surfaces["index.html"]), "release data texts": data_texts()}
    for pattern, why in BANNED_V2.items():
        for name, text in current.items():
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                snippet = text[max(0, match.start() - 40):match.end() + 40].replace("\n", " ")
                failures.append(f"{name}: /{pattern}/ ({why})\n    …{snippet}…")
    drift = va_drift(surfaces["index.html"])
    if drift:
        failures.append(drift)
    for pattern, why in BANNED.items():
        for name, text in surfaces.items():
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                snippet = text[max(0, match.start() - 40):match.end() + 40].replace("\n", " ")
                failures.append(f"{name}: /{pattern}/ ({why})\n    …{snippet}…")
    # em-dash budget: AI prose overuses them; colons, semicolons and structure read better.
    for name, text, budget in [("index.html", surfaces["index.html"], 0),
                               ("explorer_v2.js (strings)", surfaces["explorer_v2.js (strings)"], 0)]:
        count = text.count("—") + text.count("&mdash;")
        if count > budget:
            failures.append(f"{name}: {count} em-dashes (budget {budget}) — rewrite with "
                            "colons, semicolons, parentheses, or structure")
    if failures:
        print(f"COPY LINT: {len(failures)} banned pattern(s) found:\n")
        print("\n".join(failures))
        return 1
    print(f"copy lint clean ({len(BANNED)} banned patterns checked on {len(surfaces)} surfaces)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
