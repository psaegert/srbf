"""Build the results site's Reproductions page from results-site/data/reproductions.yaml.

The page sets srbf's runs of a method next to the results its own publications report (the original works only:
the authors' papers and the benchmarks they submitted the method to), with a two-sided exact test of the
difference per row. This script computes every interval and p-value and writes the comparison sections into
results-site/reproductions.html, between the markers <!-- reproductions:begin --> and <!-- reproductions:end -->.

    python scripts/site_reproductions.py            # rewrite the page
    python scripts/site_reproductions.py --check    # exit 1 if the page is not what the data gives

The tests, one per kind of published result (`test.kind` in the YAML):

* ``per_problem``: the published results list every problem and its share of successful runs ``p_i``. If srbf's
  run solved each problem as often as the published runs did, its number of solved problems would follow the
  Poisson-binomial distribution of the ``p_i``. p is the probability of a count no more likely than srbf's.
* ``rate``: only a pooled rate ``p`` is published: the same with a binomial distribution.
* ``count``: the published number is one run's count: Fisher's exact test of the two runs.
* ``ranks``: srbf's runs are ranked among the published runs of the same dataset (its rank r = the number of
  published runs with a lower score, of R). If srbf's runs were just more of the published runs, each r would be
  uniform on 0..R; p is the probability of a mean rank no more likely than srbf's.

Intervals are Wilson 95 % intervals of srbf's rate (srbf.metrics.paired.wilson_interval).
"""
from __future__ import annotations

import argparse
import csv
import html
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.stats import binomtest, fisher_exact

from srbf.metrics.paired import wilson_interval

SITE = Path(__file__).resolve().parents[1] / "results-site"
REPO = "https://github.com/psaegert/srbf/blob/main/"
DATA = SITE / "data" / "reproductions.yaml"
PAGE = SITE / "reproductions.html"
BEGIN, END = "<!-- reproductions:begin -->", "<!-- reproductions:end -->"
ALPHA = 0.05
# a probability counts as "no more likely" within this relative tolerance (floating-point sums of the same terms)
REL = 1e-7


def poisson_binomial(p: list[float]) -> np.ndarray:
    """The distribution of the number of successes of independent trials with success probabilities p."""
    pmf = np.zeros(len(p) + 1)
    pmf[0] = 1.0
    for i, q in enumerate(p, start=1):
        pmf[1:i + 1] = pmf[1:i + 1] * (1 - q) + pmf[:i] * q
        pmf[0] *= 1 - q
    return pmf


def two_sided(pmf: np.ndarray, k: int) -> float:
    """The probability of an outcome no more likely than k: the exact two-sided p-value."""
    return float(min(1.0, pmf[pmf <= pmf[k] * (1 + REL)].sum()))


def uniform_sum(m: int, top: int) -> np.ndarray:
    """The distribution of the sum of m independent ranks, each uniform on 0..top."""
    one = np.full(top + 1, 1.0 / (top + 1))
    pmf = np.array([1.0])
    for _ in range(m):
        pmf = np.convolve(pmf, one)
    return pmf


def evaluate(row: dict[str, Any], base: Path) -> dict[str, Any]:
    """srbf's runs in one comparison row: each run's rate, interval and p-value, or the rank statistic."""
    test, srbf = row["test"], row["srbf"]
    kind = test["kind"]
    if kind == "ranks":
        ranks, top = srbf["ranks"]["of"], int(srbf["ranks"]["runs"])
        pmf = uniform_sum(len(ranks), top)
        s = int(sum(ranks))
        return {"kind": kind, "mean_rank": s / (len(ranks) * top), "runs": len(ranks), "p": two_sided(pmf, s)}
    out = []
    expected: float | None = None
    reference = float(test["rate"]) if kind == "rate" else int(test.get("k", 0)) / int(test.get("n", 1))
    if kind == "per_problem":
        with open(base / test["file"], newline="") as fh:
            rows = list(csv.DictReader(fh))
        probs = [float(r["published_rate"]) for r in rows]
        hits = sum(int(r["srbf_hit"]) for r in rows)
        pmf = poisson_binomial(probs)
        expected = float(sum(probs))
        reference = expected / len(probs)
        (k, n), = srbf["draws"]
        if (k, n) != (hits, len(rows)):
            raise ValueError(f"{row['what']}: the YAML says {k}/{n}, {test['file']} says {hits}/{len(rows)}")
    for k, n in srbf["draws"]:
        lo, hi = wilson_interval(int(k), int(n))
        if kind == "per_problem":
            p = two_sided(pmf, int(k))
        elif kind == "rate":
            p = float(binomtest(int(k), int(n), float(test["rate"])).pvalue)
        elif kind == "count":
            p = float(fisher_exact([[int(k), int(n) - int(k)], [int(test["k"]), int(test["n"]) - int(test["k"])]]).pvalue)
        else:
            raise ValueError(f"unknown test kind {kind!r}")
        out.append({"k": int(k), "n": int(n), "rate": k / n, "lo": lo, "hi": hi, "p": p})
    return {"kind": kind, "runs": out, "expected": expected, "reference": reference}


def fmt_p(p: float) -> str:
    if p < 0.001:
        return "< 0.001"
    return f"{p:.2f}" if p >= 0.1 else f"{p:.2g}"


def pct(x: float) -> str:
    return f"{100 * x:.1f} %"


def esc(s: Any) -> str:
    return html.escape(str(s), quote=True)


def reading(res: dict[str, Any]) -> str:
    """One sentence: whether srbf's runs differ from the published result, and in which direction."""
    if res["kind"] == "ranks":
        if res["p"] >= ALPHA:
            return "No significant difference."
        return "srbf's fits score " + ("lower" if res["mean_rank"] < 0.5 else "higher") + " than the published fits."
    low = [r for r in res["runs"] if r["p"] < ALPHA and r["rate"] < res["reference"]]
    high = [r for r in res["runs"] if r["p"] < ALPHA and r["rate"] > res["reference"]]
    if not low and not high:
        return "No significant difference."
    return "srbf's runs succeed " + ("less" if low else "more") + " often than the published ones."


def link(url: str, text: str) -> str:
    return f'<a href="{esc(url)}" target="_blank" rel="noopener">{esc(text)}</a>'


def citation_html(m: dict[str, Any]) -> str:
    """A method's card header: its name, its own paper as the headline citation (title linked, then the authors and
    the venue), and, in a smaller line, where the compared numbers come from when that is not the method's paper."""
    paper = m["paper"]
    parts = [f'<h3>{esc(m["method"])}</h3>',
             f'<p class="repro-paper">{link(paper["url"], paper["title"])}</p>',
             f'<p class="repro-byline">{esc(paper["authors"])} · {esc(paper["venue"])}</p>']
    res = m.get("results")
    if res:
        parts.append(f'<p class="repro-results-src">Published results from {esc(res["name"])}: {esc(res["authors"])}, '
                     f'{link(res["url"], res["title"])}, {esc(res["venue"])}</p>')
    return '<header class="repro-head">' + "".join(parts) + "</header>"


def p_values(res: dict[str, Any]) -> list[float]:
    return [res["p"]] if res["kind"] == "ranks" else [r["p"] for r in res["runs"]]


def verdict_html(res: dict[str, Any]) -> str:
    """The comparison's headline: p (one per srbf run) and the reading of it in words."""
    ps = [fmt_p(p) for p in p_values(res)]
    shown = ps[0] if len(ps) == 1 else ", ".join(ps[:-1]) + " and " + ps[-1]
    differs = "1" if any(p < ALPHA for p in p_values(res)) else "0"
    return (f'<p class="repro-verdict" data-differs="{differs}"><span class="repro-p">p = {esc(shown)}</span> '
            f'<span class="repro-verdict-text">{esc(reading(res))}</span></p>')


def row_html(row: dict[str, Any], res: dict[str, Any], base: Path) -> str:
    """One comparison: the measure, the headline (p and its reading), the table, then the published side (its
    protocol and where the number comes from, with the figure or results file) and after it srbf's side."""
    pub, srbf = row["published"], row["srbf"]
    lines = [f'<h4 class="repro-what">{esc(row["what"])}</h4>',
             verdict_html(res),
             '<div class="repro-scroll"><table class="repro-table">',
             '<thead><tr><th scope="col"></th><th scope="col">Result</th><th scope="col">p</th></tr></thead><tbody>',
             f'<tr><th scope="row">Published</th><td>{esc(pub["value"])}</td><td></td></tr>']
    if res["kind"] == "ranks":
        lines.append(f'<tr><th scope="row">srbf, {res["runs"]} fits</th><td>mean rank {res["mean_rank"]:.2f}</td>'
                     f'<td>{fmt_p(res["p"])}</td></tr>')
    else:
        many = len(res["runs"]) > 1
        for i, r in enumerate(res["runs"], start=1):
            name = f"srbf, run {i}" if many else "srbf"
            lines.append(f'<tr><th scope="row">{name}</th><td>{r["k"]} of {r["n"]} ({pct(r["rate"])}, '
                         f'95 % interval {100 * r["lo"]:.1f} to {100 * r["hi"]:.1f} %)</td><td>{fmt_p(r["p"])}</td></tr>')
    lines.append("</tbody></table></div>")
    if res.get("expected") is not None:
        n = res["runs"][0]["n"]
        lines.append(f'<p class="repro-expect">If srbf solved each problem with its published success rate, it would '
                     f'solve {res["expected"]:.1f} of {n} on average.</p>')
    if row.get("note"):
        lines.append(f'<p class="repro-note">{esc(row["note"])}</p>')
    lines.append('<div class="repro-side repro-published"><h5>Published</h5>'
                 f'<p>{esc(pub["where"])}. {esc(pub["protocol"])}.</p>{sources_html(row, base)}</div>')
    config = (f' Settings: <a href="{esc(REPO + srbf["config"])}" target="_blank" rel="noopener">'
              f'<code>{esc(srbf["config"])}</code></a>.' if srbf.get("config") else "")
    lines.append('<div class="repro-side repro-srbf"><h5>srbf</h5>'
                 f'<p>{esc(srbf["problems"])}. {esc(srbf["protocol"])}.{config}</p></div>')
    return "\n".join(line for line in lines if line)


def png_size(path: Path) -> tuple[int, int]:
    """A PNG's width and height, from its header."""
    with open(path, "rb") as fh:
        head = fh.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path} is not a PNG")
    return int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")


def sources_html(row: dict[str, Any], base: Path) -> str:
    """Where the published number comes from: a crop of the publication's figure or table, and for a number computed
    from a published results file, the pinned file and the computation."""
    parts = []
    for src in row.get("sources", []):
        if "image" in src:
            w, h = png_size(base / src["image"])
            href = "data/" + src["image"]
            parts.append(f'<figure class="repro-source"><a href="{esc(href)}" target="_blank" rel="noopener">'
                         f'<img src="{esc(href)}" width="{w}" height="{h}" loading="lazy" alt="{esc(src["caption"])}" />'
                         f'</a><figcaption>{esc(src["caption"])}</figcaption></figure>')
        # a lead-in line and what it introduces stay together: the sources' spacing falls between the groups
        if "reading" in src:
            rd = src["reading"]
            parts.append(f'<div class="repro-step"><p class="repro-codecap">How the number was read from the figure:</p>'
                         f'<p class="repro-reading-text">{esc(rd["text"])}</p></div>'
                         f'<div class="repro-step"><p class="repro-codecap">To read it again from the paper\'s own PDF, '
                         f'which the script downloads and checks:</p><pre class="repro-code"><code>pip install pymupdf\n'
                         f'python scripts/read_published_figures.py {esc(rd["script"])}</code></pre></div>')
        if "code" in src:
            parts.append(f'<div class="repro-step"><p class="repro-codecap">The published results file '
                         f'(<a href="{esc(src["file"])}" target="_blank" rel="noopener">pinned on GitHub</a>) and the '
                         f'computation:</p><pre class="repro-code"><code>{esc(src["code"])}</code></pre></div>')
    return f'<div class="repro-sources">{"".join(parts)}</div>' if parts else ""


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def build(data: dict[str, Any], base: Path) -> str:
    """Every method as one self-contained card: the compared ones with their comparisons, the rest with the reason."""
    parts = ['<section id="compared" class="prose">', "<h2>Methods with a matching published result</h2>",
             '<div class="repro-cards">']
    for m in data["compared"]:
        rows = "\n".join(f'<section class="repro-comparison">\n{row_html(r, evaluate(r, base), base)}\n</section>'
                         for r in m["rows"])
        parts.append(f'<article class="repro-card" id="repro-{slug(m["method"])}">{citation_html(m)}\n{rows}\n</article>')
    parts += ["</div>", "</section>", '<section id="not-compared" class="prose">', "<h2>Not compared yet</h2>",
              '<div class="repro-cards">']
    for m in data["not_compared"]:
        parts.append(f'<article class="repro-card" id="repro-{slug(m["method"])}">{citation_html(m)}'
                     f'<p class="repro-reason">{esc(m["reason"])}</p></article>')
    parts += ["</div>", "</section>"]
    return "\n".join(parts)


def render(page: str, body: str) -> str:
    i, j = page.find(BEGIN), page.find(END)
    if i < 0 or j < i:
        raise ValueError(f"{PAGE.name} has no {BEGIN} ... {END} block")
    return page[:i + len(BEGIN)] + "\n" + body + "\n" + page[j:]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="exit 1 if the page differs from what the data gives")
    a = ap.parse_args()
    data = yaml.safe_load(DATA.read_text(encoding="utf-8"))
    page = PAGE.read_text(encoding="utf-8")
    new = render(page, build(data, DATA.parent))
    if a.check:
        if new != page:
            print(f"{PAGE.name} is out of date: run scripts/site_reproductions.py", file=sys.stderr)
            return 1
        return 0
    PAGE.write_text(new, encoding="utf-8")
    print(f"wrote {PAGE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
