"""The Benchmarks page: every problem set srbf runs, with its origin, tags, size, source and license, and the texts that
explain them. The tests keep the page in step with its data and the list free of anything a reader cannot use."""
import importlib.util
import json
import os
import re

import yaml

_SPEC = importlib.util.spec_from_file_location(
    "site_benchmarks", os.path.join(os.path.dirname(__file__), "..", "scripts", "site_benchmarks.py"))
bench = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bench)

TEXTS = yaml.safe_load(bench.TEXTS.read_text())
ROWS = json.loads(bench.LIST.read_text())
GONE = json.loads(bench.LEFT_OUT.read_text())
PAGE = bench.PAGE.read_text()


def _text(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", re.sub(r"<(script|style)\b.*?</\1>", "", html, flags=re.S)))


def test_the_page_is_what_its_data_gives() -> None:
    assert bench.render(PAGE, bench.build(TEXTS, ROWS, GONE)) == PAGE, "run python scripts/site_benchmarks.py"


def test_the_totals_are_the_sums_of_the_list() -> None:
    lede = re.search(r"([\d,]+) problem sets with ([\d,]+) problems: ([\d,]+) given as formulas and ([\d,]+) as data, "
                     r"in ([\d,]+) groups", _text(PAGE))
    sets, problems, formulas, data, groups = (int(x.replace(",", "")) for x in lede.groups())
    assert sets == len(ROWS) == PAGE.count('" data-origin="')                 # the rows of the list
    assert problems == sum(r["problems"] for r in ROWS) == formulas + data
    assert formulas == sum(r["problems"] for r in ROWS if r["kind"] == "formula")
    assert groups == sum(len(r["groups"]) for r in ROWS)
    assert all(sum(r["groups"].values()) == r["problems"] for r in ROWS)


def test_every_problem_set_has_a_name_an_origin_and_a_source() -> None:
    names = [r["display"] for r in ROWS]
    assert len(set(names)) == len(names), "two problem sets with one name"
    for r in ROWS:
        assert r["display"] != r["name"], r["name"]                       # a name, not the file's key
        assert r["sources"], r["name"]
        assert r["origin"] in {o["name"] for o in TEXTS["origins"]}, r["name"]


def test_every_tag_shown_is_explained() -> None:
    explained = {t["name"] for t in TEXTS["tags"]}
    shown = {t for r in ROWS for t in r["tags"]}
    assert shown <= explained, shown - explained
    for tag in shown:                          # a filter button and a definition for each tag a problem set carries
        assert f'data-value="{tag}"' in PAGE and f'<dt><span class="bm-tag">{tag}</span></dt>' in PAGE


def test_the_list_holds_no_local_path_and_the_page_no_internal_word() -> None:
    assert not bench.LOCAL.search(json.dumps(ROWS))
    assert not bench.LOCAL.search("see /home/someone/x") is None          # the check finds a local path
    assert bench.LOCAL.search("https://space.mit.edu/home/tegmark/aifeynman.html") is None   # but not a web address
    assert not re.search(r"(?i)private|sealed|decrypt|password|\bowner\b|\bcatalogs?\b", _text(PAGE))


def test_a_source_is_named_by_its_site_and_a_citation_by_its_authors() -> None:
    assert ">arXiv 2508.14481<" in bench.source_html("https://arxiv.org/abs/2508.14481")
    assert ">arXiv 2206.10540<" in bench.source_html("https://arxiv.org/pdf/2206.10540v3")
    assert ">doi:10.1126/sciadv.aay2631<" in bench.source_html("https://doi.org/10.1126/sciadv.aay2631")
    assert ">GitHub EpistasisLab/pmlb<" in bench.source_html("https://github.com/EpistasisLab/pmlb/tree/master/datasets")
    assert bench.source_html("Korns 2011 (GPTP IX), Accuracy in Symbolic Regression") == \
        '<span class="bm-cite">Korns 2011 (GPTP IX)</span>'
    assert bench.license_label("MIT (viktmar/FastSRB; see THIRD_PARTY_LICENSES)") == "MIT"
    assert bench.license_label(None) == ""
    assert bench.license_label("CC-BY-SA-4.0 (Sun et al. 2023)") == "CC BY-SA 4.0"
    assert bench.license_label("facts only: geometric formulas cited from Xu et al. 2024; no upstream license") == "formulas only"


def test_the_data_left_out_for_its_license_has_its_own_section() -> None:
    names = {r["name"] for r in ROWS}
    assert GONE and not names & {g["name"] for g in GONE if g["whole"]}       # a benchmark left out is not run
    for g in GONE:
        assert f'<tr data-name="{g["name"]}">' in PAGE and all(w["reason"] for w in g["why"])
        assert g["whole"] or g["name"] in names                               # a split benchmark keeps the rest
    total = sum(g["problems"] for g in GONE)
    assert f"In all, {total:,} problems from {len(GONE)} benchmarks." in _text(PAGE)


def test_a_problem_set_cites_its_paper_and_its_data_where_both_exist() -> None:
    assert sum(1 for r in ROWS if (r["citation"] or {}).get("paper")) > 0.9 * len(ROWS)
    for r in ROWS:
        for side in ("paper", "data"):
            c = (r["citation"] or {}).get(side)
            if c:
                assert c["short"] and c["url"].startswith(("https://", "http://")), (r["name"], side)
                assert f'>{bench.esc(c["short"])}</a>' in PAGE, (r["name"], side)
    assert bench.cite_html({"short": "Keijzer 2003", "title": "Improving SR", "url": "https://doi.org/10.1007/x"}) == \
        '<a href="https://doi.org/10.1007/x" target="_blank" rel="noopener" title="Improving SR">Keijzer 2003</a>'
