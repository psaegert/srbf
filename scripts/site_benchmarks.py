"""Build the results site's Benchmarks page from results-site/data/benchmarks.yaml and data/benchmarks/problem_sets.json.

The page lists every problem set srbf runs (a benchmark, or one part of it), with where its problems come from, the
tags that say what it tests, its number of problems and groups, its source and its license, and explains how srbf
runs a benchmark, what srbf decided where a benchmark left something open, and what it could not include. The list
is made from symbolic-data's release count (tools/catalog_build/count_release.py writes build/release_counts.json);
the texts are in the YAML. This script writes the page between <!-- benchmarks:begin --> and <!-- benchmarks:end -->.

    python scripts/site_benchmarks.py --counts <symbolic-data>/build/release_counts.json   # refresh the list
    python scripts/site_benchmarks.py            # rewrite the page
    python scripts/site_benchmarks.py --check    # exit 1 if the page is not what the data gives
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

SITE = Path(__file__).resolve().parents[1] / "results-site"
TEXTS = SITE / "data" / "benchmarks.yaml"
LIST = SITE / "data" / "benchmarks" / "problem_sets.json"
LEFT_OUT = SITE / "data" / "benchmarks" / "left_out.json"
PAGE = SITE / "benchmarks.html"
BEGIN, END = "<!-- benchmarks:begin -->", "<!-- benchmarks:end -->"
# what must never reach the list: a local path (a web address may hold /home/, as in space.mit.edu/home/...)
LOCAL = re.compile(r"(?<![\w.-])(/home/|/Users/|~/)")


def esc(s: Any) -> str:
    return html.escape(str(s), quote=True)


def display(row: dict[str, Any], names: dict[str, str]) -> str:
    """A problem set's name on the page: the YAML's name, else its own display name, else its benchmark's name."""
    if row["name"] in names:
        return names[row["name"]]
    if row.get("display_name") and row["display_name"] != row["name"]:
        name = row["display_name"]
    else:
        name = row.get("set_name") or str(row.get("description") or row["name"]).rstrip(".")
    return name.replace("(pinned points)", "(fixed points)")


def problem_sets(counts: dict[str, Any], texts: dict[str, Any]) -> list[dict[str, Any]]:
    """What the page shows of each problem set in the release count, sorted by name."""
    rows = []
    for r in counts["rows"]:
        tags = r.get("tags") or {}
        rows.append({
            "name": r["name"], "display": display(r, texts["names"]), "kind": r["kind"],
            "origin": tags.get("origin"), "tags": list(tags.get("tags") or []),
            "problems": int(r["problems"]), "groups": {str(k): int(v) for k, v in r["subsets"].items()},
            "sources": list(texts["sources"].get(r["name"]) or r.get("sources") or []),
            "citation": r.get("citation") or {},
            "license": texts["licenses_as"].get(r["name"]) or r.get("license"),
        })
    bad = [r["name"] for r in rows if LOCAL.search(json.dumps(r))]
    if bad:
        raise SystemExit(f"local paths in {bad}")
    return sorted(rows, key=lambda r: (r["display"].lower(), r["name"]))


def left_out(counts: dict[str, Any], texts: dict[str, Any]) -> list[dict[str, Any]]:
    """The data srbf leaves out for its license, from the release count: a whole benchmark, or some of its datasets."""
    rows = []
    for r in counts.get("left_out_for_license") or []:
        why: list[dict[str, Any]]
        if r["members"]:
            by_reason: dict[str, list[str]] = {}
            for member, reason in r["members"].items():
                by_reason.setdefault(reason, []).append(member)
            why = [{"reason": k, "members": v} for k, v in by_reason.items()]
        else:
            why = [{"reason": r["reason"], "members": None}]
        rows.append({"name": r["catalog"], "display": display({**r, "display_name": r.get("name")}, texts["names"]),
                     "problems": int(r["problems"] or 0), "whole": not r["members"], "why": why,
                     "sources": list(texts["sources"].get(r["catalog"]) or r.get("sources") or []),
                     "citation": r.get("citation") or {}})
    bad = [r["name"] for r in rows if LOCAL.search(json.dumps(r))]
    if bad:
        raise SystemExit(f"local paths in {bad}")
    return sorted(rows, key=lambda r: (r["display"].lower(), r["name"]))


def left_out_table(rows: list[dict[str, Any]], kept: dict[str, int]) -> str:
    body = []
    for r in rows:
        if r["whole"]:
            what = "all"
            why = esc(r["why"][0]["reason"])
        else:
            what = f'{fmt(r["problems"])} of {fmt(r["problems"] + kept.get(r["name"], 0) + kept.get(r["name"] + "-share-alike", 0))}'
            why = "; ".join(f'{esc(", ".join(w["members"]))}: {esc(w["reason"])}' for w in r["why"])
        paper, data = cited(r)
        body.append(f'<tr data-name="{esc(r["name"])}"><th scope="row">{esc(r["display"])}</th>'
                    f'<td class="num">{fmt(r["problems"])}</td><td>{what}</td><td>{why}</td>'
                    f'<td class="bm-src">{paper}</td><td class="bm-src">{data}</td></tr>')
    return ('<div class="bm-scroll"><table class="bm-table bm-left">\n<thead><tr><th scope="col">Benchmark</th>'
            '<th scope="col" class="num">Problems</th><th scope="col">Left out</th><th scope="col">Why</th>'
            '<th scope="col">Paper</th><th scope="col">Data</th></tr></thead>\n<tbody>\n' + "\n".join(body)
            + "\n</tbody></table></div>")


def cite_html(c: dict[str, Any] | None) -> str:
    """A paper or dataset: its short name, linked, with its title as the link's hint."""
    if not c:
        return ""
    title = " ".join(x for x in (c.get("title"), c.get("venue")) if x).replace("\u2014", ": ")   # the copy lint: no em-dash
    return (f'<a href="{esc(c["url"])}" target="_blank" rel="noopener" title="{esc(title)}">{esc(c["short"])}</a>'
            if c.get("url") else esc(c["short"]))


def cited(r: dict[str, Any]) -> tuple[str, str]:
    """The Paper and Data cells: the citations, else the first source in the paper's place."""
    c = r.get("citation") or {}
    if c:
        return cite_html(c.get("paper")), cite_html(c.get("data"))
    return (source_html(r["sources"][0]) if r["sources"] else ""), ""


def source_html(source: str) -> str:
    """A source: a link named by its site (arXiv, DOI, GitHub, ...), or the citation as written."""
    if not source.startswith(("http://", "https://")):          # a citation: its authors, year and venue
        cite = re.split(r"(?<=\)),| \u2014 |; ", source, maxsplit=1)[0].strip()
        return f'<span class="bm-cite">{esc(cite)}</span>'
    u = urlparse(source)
    host, path = u.netloc.removeprefix("www."), u.path.strip("/")
    if host == "arxiv.org":
        label = "arXiv " + re.sub(r"^(abs|pdf)/|v\d+$|\.pdf$", "", path)
    elif host == "doi.org":
        label = "doi:" + path
    elif host == "github.com":
        label = "GitHub " + "/".join(path.split("/")[:2])
    elif host == "huggingface.co":
        label = "Hugging Face " + "/".join(path.split("/")[1:3] if path.startswith("datasets/") else path.split("/")[:2])
    else:
        label = host
    return f'<a href="{esc(source)}" target="_blank" rel="noopener">{esc(label)}</a>'


def license_label(text: str | None) -> str:
    """The license's name without its notes: 'MIT (viktmar/FastSRB; ...)' -> 'MIT'; formulas taken from a source
    without a license of its own (facts) -> 'formulas only'."""
    if not text:
        return ""
    if str(text).startswith("facts only"):
        return "formulas only"
    name = re.split(r"\s*[(;]", str(text), maxsplit=1)[0].strip()
    return re.sub(r"^CC-(BY(?:-[A-Z]{2})*)-(\d\.\d)$", lambda m: f"CC {m.group(1)} {m.group(2)}", name)


def fmt(n: int) -> str:
    return f"{n:,}"


def topics(items: list[dict[str, str]]) -> str:
    return "\n".join(f'<section class="topic"><h3>{esc(t["title"])}</h3><div><p>{esc(t["text"])}</p></div></section>'
                     for t in items)


def definitions(items: list[dict[str, str]], used: set[str]) -> str:
    return "<dl class=\"bm-defs\">" + "".join(
        f'<dt><span class="bm-tag">{esc(t["name"])}</span></dt><dd>{esc(t["text"])}</dd>'
        for t in items if t["name"] in used) + "</dl>"


def table(rows: list[dict[str, Any]]) -> str:
    body = []
    for r in rows:
        groups = r["groups"]
        group_names = ", ".join(f"{k} ({v})" for k, v in groups.items())
        tags = "".join(f'<span class="bm-tag">{esc(t)}</span>' for t in r["tags"])
        paper, data = cited(r)
        body.append(
            f'<tr data-name="{esc(r["name"])}" data-origin="{esc(r["origin"])}" data-tags="{esc("|".join(r["tags"]))}" '
            f'data-problems="{r["problems"]}">'
            f'<th scope="row">{esc(r["display"])}</th><td>{esc(r["origin"])}</td><td class="bm-tags">{tags}</td>'
            f'<td class="num">{fmt(r["problems"])}</td>'
            f'<td class="num" title="{esc(group_names)}">{len(groups)}</td>'
            f'<td class="bm-src">{paper}</td><td class="bm-src">{data}</td><td>{esc(license_label(r["license"]))}</td></tr>')
    return ('<div class="bm-scroll"><table class="bm-table bm-list">\n<thead><tr><th scope="col">Problem set</th>'
            '<th scope="col">Origin</th><th scope="col">Tags</th><th scope="col" class="num">Problems</th>'
            '<th scope="col" class="num">Groups</th><th scope="col">Paper</th><th scope="col">Data</th>'
            '<th scope="col">License</th></tr></thead>'
            '\n<tbody>\n' + "\n".join(body) + "\n</tbody></table></div>")


def filters(rows: list[dict[str, Any]], texts: dict[str, Any]) -> str:
    origins = [o["name"] for o in texts["origins"] if any(r["origin"] == o["name"] for r in rows)]
    tags = [t["name"] for t in texts["tags"] if any(t["name"] in r["tags"] for r in rows)]
    seg = "".join(f'<button type="button" data-value="{esc(o)}" aria-pressed="false">{esc(o)}</button>' for o in origins)
    chips = "".join(f'<button type="button" data-value="{esc(t)}" aria-pressed="false">{esc(t)}</button>' for t in tags)
    return ('<div class="bm-filter" hidden>'
            '<div class="bm-filter-row"><span class="bm-label">Origin</span><div class="bm-seg" data-filter="origin">'
            '<button type="button" data-value="" aria-pressed="true">all</button>' + seg + '</div></div>'
            '<div class="bm-filter-row"><span class="bm-label">Tags</span><div class="bm-chips" data-filter="tags">'
            + chips + '</div></div>'
            '<p class="bm-count" aria-live="polite"></p></div>')


def build(texts: dict[str, Any], rows: list[dict[str, Any]], gone: list[dict[str, Any]]) -> str:
    n_formula = sum(r["problems"] for r in rows if r["kind"] == "formula")
    n_data = sum(r["problems"] for r in rows if r["kind"] == "data")
    n_groups = sum(len(r["groups"]) for r in rows)
    used_tags = {t for r in rows for t in r["tags"]}
    used_origins = {r["origin"] for r in rows}
    return "\n".join([
        '<section class="hero page-hero">',
        "<h1>Benchmarks</h1>",
        f'<p class="lede">{esc(texts["intro"])}</p>',
        "</section>",
        "",
        '<section id="list" class="prose">',
        "<h2>The problem sets</h2>",
        f'<p class="section-lede">{fmt(len(rows))} problem sets with {fmt(n_formula + n_data)} problems: '
        f'{fmt(n_formula)} given as formulas and {fmt(n_data)} as data, in {fmt(n_groups)} groups. '
        'The tags are explained <a href="#tags">below the list</a>.</p>',
        filters(rows, texts),
        table(rows),
        "</section>",
        "",
        '<section id="tags" class="prose">',
        "<h2>What the tags mean</h2>",
        f'<p class="section-lede">{esc(texts["tag_rule"])}</p>',
        '<section class="topic"><h3>Where the problems come from</h3><div>'
        + definitions(texts["origins"], used_origins) + "</div></section>",
        '<section class="topic"><h3>What a problem set tests</h3><div>'
        + definitions(texts["tags"], used_tags) + "</div></section>",
        f'<section class="topic"><h3>Licenses</h3><div><p>{esc(texts["licenses"])}</p></div></section>',
        "</section>",
        "",
        '<section id="how" class="prose">',
        "<h2>How srbf runs a benchmark</h2>",
        topics(texts["rules"]),
        "</section>",
        "",
        '<section id="changes" class="prose">',
        "<h2>What srbf decided</h2>",
        '<p class="section-lede">Where a benchmark left something open, or exists in copies that differ, srbf had to '
        "choose. These are the choices.</p>",
        topics(texts["changes"]),
        "</section>",
        "",
        '<section id="left-out" class="prose">',
        "<h2>What srbf could not include</h2>",
        topics(texts["left_out"]),
        "</section>",
        "",
        '<section id="license" class="prose">',
        f'<h2>{esc(texts["license_section"]["title"])}</h2>',
        f'<p class="section-lede">{esc(texts["license_section"]["text"])} In all, '
        f'{fmt(sum(r["problems"] for r in gone))} problems from {fmt(len(gone))} benchmarks.</p>',
        left_out_table(gone, {r["name"]: r["problems"] for r in rows}),
        "</section>",
    ])


def render(page: str, body: str) -> str:
    head, rest = page.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    return f"{head}{BEGIN}\n{body}\n{END}{tail}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", help="symbolic-data's build/release_counts.json: refresh the list from it")
    ap.add_argument("--check", action="store_true", help="exit 1 if the page differs from what the data gives")
    a = ap.parse_args()
    texts = yaml.safe_load(TEXTS.read_text())
    if a.counts:
        counts = json.loads(Path(a.counts).read_text())
        LIST.parent.mkdir(parents=True, exist_ok=True)
        LIST.write_text(json.dumps(problem_sets(counts, texts), indent=1, ensure_ascii=False) + "\n")
        LEFT_OUT.write_text(json.dumps(left_out(counts, texts), indent=1, ensure_ascii=False) + "\n")
    rows = json.loads(LIST.read_text())
    gone = json.loads(LEFT_OUT.read_text())
    page = PAGE.read_text()
    new = render(page, build(texts, rows, gone))
    if a.check:
        if new != page:
            print(f"{PAGE} is stale: run python scripts/site_benchmarks.py", file=sys.stderr)
            return 1
        return 0
    PAGE.write_text(new)
    print(f"{PAGE}: {len(rows)} problem sets")
    return 0


if __name__ == "__main__":
    sys.exit(main())
