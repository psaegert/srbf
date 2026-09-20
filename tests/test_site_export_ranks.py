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
    direct = np.mean([rankdata([-sx.rank_score(raw[m][i], False) for m in "abcd"], method="average") for i in range(400)], axis=0)
    ki = sx.RANK_KEYS.index(key)
    for j, a in enumerate("abcd"):
        lost = 0.0
        for b in "abcd":
            if a != b:
                n, *w = sx.rank_pair_cell(rows[a], rows[b])
                wa, wb = w[2 * ki], w[2 * ki + 1]
                lost += (wb + 0.5 * (n - wa - wb)) / n
        assert abs(1 + lost - direct[j]) < 1e-12


def test_a_time_budget_buys_the_largest_rung_timed_within_it() -> None:
    sx = _exporter()
    timing = {"m": {"1": 0.2, "2": 0.4, "4": 0.9, "8": 2.5}}
    assert sx.rung_within(timing, "m", 1.0, {1, 2, 4, 8}) == 4
    assert sx.rung_within(timing, "m", 1.0, {1, 2}) == 2                     # a rung without results cannot be read
    assert sx.rung_within(timing, "m", 0.1, {1, 2, 4, 8}) is None            # cannot finish within the budget: sits out
    assert sx.rung_within(timing, "untimed", 10.0, {1}) is None


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
    assert keys({"a": {"nguyen": {"1": same, "2": same}}}) == everything - set(sx.COPY_OF)
    # one cell that tells validation apart brings that metric back, and only that one
    assert keys({"a": {"nguyen": {"1": same}, "measured": {"1": apart}}}) == everything - {"numeric_recovery_relative_fit"}
    # nothing published yet: nothing is known to be a copy, so nothing is dropped
    assert keys({}) == everything
