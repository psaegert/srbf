"""The Reproductions page: srbf's runs of a method next to its own publications' results, with an exact two-sided test
per row. The tests here pin the statistics against independent computations and keep the page in step with its data."""
import copy
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.stats import binom, binomtest

_SPEC = importlib.util.spec_from_file_location(
    "site_reproductions", os.path.join(os.path.dirname(__file__), "..", "scripts", "site_reproductions.py"))
repro = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(repro)


def test_with_one_rate_for_every_problem_the_per_problem_distribution_is_the_binomial() -> None:
    pmf = repro.poisson_binomial([0.3] * 40)
    np.testing.assert_allclose(pmf, binom.pmf(np.arange(41), 40, 0.3), atol=1e-14)
    assert pmf.sum() == pytest.approx(1.0)


def test_per_problem_rates_narrow_the_distribution() -> None:
    # problems that are always or never solved add no spread: 60 sure ones and 40 hopeless ones fix the count at 60
    pmf = repro.poisson_binomial([1.0] * 60 + [0.0] * 40)
    assert pmf[60] == pytest.approx(1.0)
    assert repro.two_sided(pmf, 60) == pytest.approx(1.0)
    assert repro.two_sided(pmf, 59) == 0.0


def test_the_two_sided_p_value_is_the_probability_of_an_outcome_no_more_likely() -> None:
    pmf = binom.pmf(np.arange(11), 10, 0.5)
    assert repro.two_sided(pmf, 5) == pytest.approx(1.0)
    assert repro.two_sided(pmf, 0) == pytest.approx(2 * 0.5 ** 10)
    for k in range(0, 21):               # the same rule as scipy's exact binomial test
        assert repro.two_sided(binom.pmf(np.arange(21), 20, 0.3), k) == pytest.approx(binomtest(k, 20, 0.3).pvalue)


def test_ranks_among_published_runs_are_uniform_when_the_runs_are_alike() -> None:
    one = repro.uniform_sum(1, 30)
    np.testing.assert_allclose(one, np.full(31, 1 / 31))
    twelve = repro.uniform_sum(12, 30)
    assert twelve.sum() == pytest.approx(1.0)
    assert (np.arange(twelve.size) * twelve).sum() == pytest.approx(12 * 15)     # mean rank 15 of 30 per run
    res = repro.evaluate({"test": {"kind": "ranks"}, "srbf": {"ranks": {"runs": 30, "of": [15] * 12}}}, Path("."))
    assert res["mean_rank"] == pytest.approx(0.5) and res["p"] == pytest.approx(1.0)
    low = repro.evaluate({"test": {"kind": "ranks"}, "srbf": {"ranks": {"runs": 30, "of": [0] * 12}}}, Path("."))
    assert low["p"] < 1e-12 and repro.reading(low) == "srbf's fits score lower than the published fits."


def _rows() -> dict:
    return yaml.safe_load(repro.DATA.read_text(encoding="utf-8"))


def test_every_row_reads_its_published_side_and_its_counts() -> None:
    data = _rows()
    kinds = set()
    for m in data["compared"]:
        for row in m["rows"]:
            res = repro.evaluate(row, repro.DATA.parent)
            kinds.add(res["kind"])
            if res["kind"] == "ranks":
                assert 0 <= res["mean_rank"] <= 1 and 0 < res["p"] <= 1
                continue
            for run in res["runs"]:
                assert 0 < run["n"] and 0 <= run["k"] <= run["n"] and run["lo"] <= run["rate"] <= run["hi"]
                assert 0 < run["p"] <= 1
    assert kinds == {"per_problem", "rate", "count", "ranks"}
    assert {m["method"] for m in data["compared"]}.isdisjoint({m["method"] for m in data["not_compared"]})


def test_a_count_that_disagrees_with_its_per_problem_file_is_refused() -> None:
    row = next(r for m in _rows()["compared"] for r in m["rows"] if r["test"]["kind"] == "per_problem")
    bad = copy.deepcopy(row)
    bad["srbf"]["draws"] = [[bad["srbf"]["draws"][0][0] + 1, bad["srbf"]["draws"][0][1]]]
    with pytest.raises(ValueError, match="says"):
        repro.evaluate(bad, repro.DATA.parent)


def test_a_rate_row_matches_scipy_and_a_count_row_compares_two_runs() -> None:
    rate = {"what": "w", "test": {"kind": "rate", "rate": 0.866}, "srbf": {"draws": [[104, 120]]}}
    assert repro.evaluate(rate, Path("."))["runs"][0]["p"] == pytest.approx(binomtest(104, 120, 0.866).pvalue)
    same = {"what": "w", "test": {"kind": "count", "k": 40, "n": 52}, "srbf": {"draws": [[40, 52]]}}
    assert repro.evaluate(same, Path("."))["runs"][0]["p"] == pytest.approx(1.0)


def test_the_page_shows_what_its_data_gives() -> None:
    page = repro.PAGE.read_text(encoding="utf-8")
    assert repro.render(page, repro.build(_rows(), repro.DATA.parent)) == page, "run scripts/site_reproductions.py"


def test_a_change_to_the_data_shows_as_a_stale_page() -> None:
    data = _rows()
    data["compared"][0]["rows"][0]["srbf"]["draws"][0][0] -= 1
    page = repro.PAGE.read_text(encoding="utf-8")
    assert repro.render(page, repro.build(data, repro.DATA.parent)) != page


def test_every_published_number_shows_where_it_comes_from() -> None:
    import re
    for m in _rows()["compared"]:
        for row in m["rows"]:
            sources = row.get("sources") or []
            assert sources, f"{m['method']}: {row['what']} has no source"
            for src in sources:
                if "image" in src:
                    w, h = repro.png_size(repro.DATA.parent / src["image"])
                    assert w > 300 and h > 100 and src["caption"]
                if "code" in src:
                    # the file is pinned to a commit, and the code reads that same commit
                    sha = re.search(r"/([0-9a-f]{40})/", src["file"]).group(1)
                    assert sha in src["code"] and "read_feather" in src["code"]


_FIG_SPEC = importlib.util.spec_from_file_location(
    "read_published_figures", os.path.join(os.path.dirname(__file__), "..", "scripts", "read_published_figures.py"))
figures = importlib.util.module_from_spec(_FIG_SPEC)
_FIG_SPEC.loader.exec_module(figures)       # PyMuPDF is imported only when a figure is read


def test_every_number_read_from_a_figure_shows_arithmetic_that_holds_and_a_reading_to_rerun() -> None:
    import re
    seen = 0
    for m in _rows()["compared"]:
        for row in m["rows"]:
            for src in row.get("sources") or []:
                rd = src.get("reading")
                if not rd:
                    continue
                seen += 1
                assert rd["script"] in figures.READINGS
                # "(a − b) / (c − b) = v": the shown arithmetic gives the shown value
                a, b, c, b2, v = map(float, re.search(
                    r"\(([\d.]+) − ([\d.]+)\) /\s+\(([\d.]+) −\s+([\d.]+)\) = ([\d.]+)", rd["text"]).groups())
                assert b == b2 and abs((a - b) / (c - b) - v) < 5e-5
                if row["test"]["kind"] == "rate":
                    assert row["test"]["rate"] == v                       # the test uses the value read
                if row["test"]["kind"] == "count":
                    assert abs(v * row["test"]["n"] - row["test"]["k"]) < 0.01   # a whole number of laws
    assert seen == 3


def test_the_figure_reading_is_linear_between_the_two_labelled_positions() -> None:
    assert figures.linear(0.0, 159.225, 1.0, 261.869, 248.068) == pytest.approx(0.8655, abs=5e-5)
    assert figures.linear(0.0, 145.636, 1.0, 90.065, 94.340) * 52 == pytest.approx(48.0, abs=0.01)
