"""The Ranks view of the results site ships PAIRWISE outcomes (scripts/site_export_v2.py); the page turns them into
mean ranks for whatever methods and catalogs are selected. That only works if the pairwise identity holds exactly."""
import importlib.util
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent.parent


def _exporter() -> Any:
    spec = importlib.util.spec_from_file_location("site_export_v2", ROOT / "scripts" / "site_export_v2.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rows(values: list[float | None], key: str) -> dict[int, dict[str, float | None]]:
    sx = _exporter()
    return {i: {k: (v if k == key else None) for k in sx.RANK_KEYS} for i, v in enumerate(values)}


def test_a_law_without_an_answer_ranks_last_and_an_exact_recovery_first() -> None:
    sx = _exporter()
    assert sx.rank_score(None, False) == -math.inf and sx.rank_score(float("nan"), True) == -math.inf
    assert sx.rank_score(-math.inf, False) == math.inf                      # log10 FVU of an exact recovery
    assert sx.rank_score(0.5, None) == sx.rank_score(2.0, None) < sx.rank_score(1.0, None)   # a ratio: closer to 1 is better


def test_mean_ranks_follow_from_the_pairwise_outcomes_exactly() -> None:
    sx = _exporter()
    key, rng = "log10_fvu_val", np.random.default_rng(0)
    raw = {m: [None if rng.random() < 0.2 else (-math.inf if rng.random() < 0.1 else float(np.round(rng.normal(-3, 3), 0))) for _ in range(400)]
           for m in "abcd"}                                                 # rounded: plenty of ties, and laws nobody answers
    rows = {m: _rows(v, key) for m, v in raw.items()}
    # places as comparisons read the values: two recovered predictions (log10 FVU at or below log10 2^-23) tie
    direct = np.mean([rankdata([-sx.rank_score(sx.comparison_value(key, raw[m][i]), False) for m in "abcd"], method="average")
                      for i in range(400)], axis=0)
    ki = sx.RANK_KEYS.index(key)
    for j, a in enumerate("abcd"):
        lost = 0.0
        for b in "abcd":
            if a != b:
                n, *w = sx.rank_pair_cell(rows[a], rows[b])
                lost += (1 - w[2 * ki] / n) / 2          # the chance b beats a: (1 - a's mean superiority) / 2
        assert abs(1 + lost - direct[j]) < 1e-12


def test_a_slot_brackets_a_method_over_its_finished_budgets() -> None:
    """A slot reads every method where the page's position would: at a finished budget, between two (interpolated), or
    nowhere outside its timed or run range."""
    sx = _exporter()
    pts = [(0.2, 1), (0.4, 2), (0.9, 4), (2.5, 8)]
    assert sx.bracket_in(pts, 0.9) == (4, 4, 0.0)
    r1, r2, w = sx.bracket_in(pts, 1.0)
    assert (r1, r2) == (4, 8) and abs(w - math.log(1 / 0.9) / math.log(2.5 / 0.9)) < 1e-15
    assert sx.bracket_in(pts, 0.1) is None and sx.bracket_in(pts, 3.0) is None
    data = {"m": {("c", 1): {(1, i): {"success": 1.0} for i in range(3)}, ("c", 4): {(1, i): {"success": 1.0} for i in range(3)},
                  ("c", 8): {(1, i): {"success": 1.0} for i in range(2)}}}             # 8 has not finished every problem
    at = sx.brackets(data, ["m"], {"c": 3}, {"m": {"1": 0.2, "4": 0.9, "8": 2.5}}, ["2", "8", "t0.5", "t3", "t0.1"])
    assert at["m"]["2"]["c"] == [1, 4, 0.5] and "8" not in at["m"] and "t3" not in at["m"] and "t0.1" not in at["m"]
    assert at["m"]["t0.5"]["c"][:2] == [1, 4]
    assert sx.brackets(data, ["m"], {"c": 3}, {}, ["t0.5"]) == {"m": {}}               # untimed: no place in time


def test_a_metric_that_copies_another_in_every_cell_is_not_listed() -> None:
    """Recovery relative to the reference law IS numeric recovery wherever the targets are computed from the law;
    the menu lists it only once some cell tells the two apart (a catalog of measured data)."""
    sx = _exporter()
    same = {"m": {"numeric_recovery_val": [4, 10], "numeric_recovery_relative_val": [4, 10],
                  "numeric_recovery_fit": [5, 10], "numeric_recovery_relative_fit": [5, 10]}}
    apart = {"m": {"numeric_recovery_val": [0, 10], "numeric_recovery_relative_val": [7, 10],
                   "numeric_recovery_fit": [5, 10], "numeric_recovery_relative_fit": [5, 10]}}
    keys = lambda cells: {m["key"] for m in sx.listed_metrics(cells)}   # noqa: E731
    everything = {m["key"] for m in sx.registry_json()}
    internals = {m["key"] for m in sx.registry_json() if m["group"] == "Method Internals"}   # these cells hold none
    assert keys({"a": {"nguyen": {"1": same, "2": same}}}) == everything - set(sx.COPY_OF) - internals
    # one cell that tells validation apart brings that metric back, and only that one
    assert keys({"a": {"nguyen": {"1": same}, "measured": {"1": apart}}}) == everything - {"numeric_recovery_relative_fit"} - internals
    # nothing published yet: nothing is known to be a copy, so nothing is dropped
    assert keys({}) == everything


def test_a_difference_ranks_by_its_distance_from_zero() -> None:
    """Predicted minus true counts are signed: five constants too few must not beat an exact count."""
    sx = _exporter()
    for key in ("n_constants_delta", "total_nestedness_delta"):
        higher, ideal = sx.METRIC_HIGHER[key], sx.IDEAL[key]
        assert higher is None and ideal == 0.0
        assert sx.rank_score(-5.0, higher, ideal) == sx.rank_score(5.0, higher, ideal) < sx.rank_score(0.0, higher, ideal)


def test_every_ranked_metric_says_what_is_better() -> None:
    sx = _exporter()
    kinds = {m[0]: m for m in sx.METRICS}
    assert sx.RANK_KEYS[0] == "log10_fvu_val"                              # the primary league
    assert len(set(sx.RANK_KEYS)) == len(sx.RANK_KEYS)
    for key in sx.RANK_KEYS:
        assert key in kinds, key
        assert sx.METRIC_HIGHER[key] is not None or key in sx.IDEAL, f"{key} has neither a direction nor an ideal"
        assert key not in sx.EVERY_PROBLEM, f"{key} is a property of the ground truth: every method would tie"
    assert "r2_val" not in sx.RANK_KEYS and "predicted_log_prob" not in sx.RANK_KEYS


def test_the_public_guard_reads_every_method_a_written_cell_file_names(tmp_path: Path) -> None:
    """The guard refuses to publish a ranks/ or paired/ file that names a private method; it must parse what the
    exporter writes, both sides of every pair."""
    sx = _exporter()
    spec = importlib.util.spec_from_file_location("public_guard", ROOT / "results-site" / "tests" / "public_guard.py")
    assert spec is not None and spec.loader is not None
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    ranks = {"log10_fvu_val": {"hidden-a|e2e": {"cat": {"t1": [3, 1, 1]}}}}
    paired = {"f1_score": {"e2e|hidden-b": {"cat": {"64": [1, 0, 0, 1, 0, 0, 0, 0]}}}, "f1_score@answered": {"hidden-c|e2e": {}}}
    sx.write_slot_cells(str(tmp_path), "r", ranks, paired)
    assert guard.cell_methods((tmp_path / "ranks" / "log10_fvu_val.js").read_text()) == {"e2e", "hidden-a"}
    assert guard.cell_methods((tmp_path / "paired" / "f1_score.js").read_text()) == {"e2e", "hidden-b", "hidden-c"}
