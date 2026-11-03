"""Improvability: a prediction's validation error against the best a fixed pool reaches, (e - e*) / e."""
import importlib.util
import math
import os

import pytest

from srbf.metrics import ERROR_FLOOR, improvability, pool_best, validation_error

_SPEC = importlib.util.spec_from_file_location(
    "site_export_v2", os.path.join(os.path.dirname(__file__), "..", "scripts", "site_export_v2.py"))
export = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(export)


def test_the_error_is_the_rmse_in_units_of_the_targets_spread_and_recovered_predictions_tie():
    assert validation_error(-4.0, True) == pytest.approx(1e-2)            # sqrt(FVU)
    assert validation_error(-15.65, True) == ERROR_FLOOR                  # an exact fit
    assert validation_error(math.log10(2.0 ** -23), True) == pytest.approx(ERROR_FLOOR)
    assert validation_error(-8.0, True) == ERROR_FLOOR                    # below the Numeric Recovery bar: a tie
    assert validation_error(None, False) == math.inf                      # a failed prediction
    assert validation_error(None, True) is None                           # no held-out points
    assert validation_error(float("nan"), True) is None


def test_improvability_is_zero_at_the_pools_best_one_for_a_failure_and_down_to_minus_one_below_the_best():
    best = pool_best([1e-2, None, math.inf, 1e-3])
    assert best == 1e-3
    assert improvability(1e-3, best) == 0.0
    assert improvability(1e-2, best) == pytest.approx(0.9)
    assert improvability(math.inf, best) == 1.0
    assert improvability(1e-4, best) == pytest.approx(-0.9)               # outside the pool and better: the share it saves
    assert improvability(None, best) is None and improvability(1e-2, None) is None
    assert pool_best([None, math.inf]) is None
    # two recovered predictions tie, whatever digits lie below the bar
    assert improvability(validation_error(-14.0, True), validation_error(-15.65, True)) == 0.0


def _row(success, log10_fvu_val):
    values = {k: 0.0 for k in export.RATE_KEYS}
    values.update({k: None for k in export.CONT_KEYS})
    values.update(success=success, log10_fvu_val=log10_fvu_val)
    return export.Row(values)


def test_the_export_measures_every_run_against_the_public_pools_best_on_the_same_points():
    data = {
        "a": {("cat", 1): {(1, 0): _row(1.0, -4.0), (2, 0): _row(1.0, -2.0), (1, 1): _row(0.0, None)},
              ("cat", 2): {(1, 0): _row(1.0, -6.0), (2, 0): _row(1.0, -2.0), (1, 1): _row(1.0, -2.0)}},
        "b": {("cat", 1): {(1, 0): _row(1.0, -2.0), (2, 0): _row(1.0, -4.0), (1, 1): _row(1.0, None)},
              ("cat", 64): {(1, 0): _row(1.0, -16.0)}},                        # an unpublished budget: not in the pool
        "private": {("cat", 1): {(1, 0): _row(1.0, -8.0), (2, 0): _row(1.0, -1.0), (1, 1): _row(1.0, -1.0)}},
    }
    export.derive_improvability(data, ["a", "b"], lambda key, rung: rung != 64)
    imp = lambda m, rung, k: data[m][("cat", rung)][k][export.IMPROVABILITY]  # noqa: E731
    # problem 0, draw 1: the pool's best is a at budget 2 (log10 FVU -6 -> error 1e-3)
    assert imp("a", 2, (1, 0)) == 0.0
    assert imp("a", 1, (1, 0)) == pytest.approx(0.9)
    assert imp("b", 1, (1, 0)) == pytest.approx(0.99)
    # draw 2 of the same problem has its own points and its own best (b at budget 1)
    assert imp("b", 1, (2, 0)) == 0.0 and imp("a", 1, (2, 0)) == pytest.approx(0.9)
    # the unpublished budget does not set the best, but is measured against it
    assert imp("b", 64, (1, 0)) == pytest.approx((ERROR_FLOOR - 1e-3) / 1e-3)   # better than the best: below 0
    # a failed run is 1; a private method is measured against the public pool, and can beat it
    assert imp("a", 1, (1, 1)) == 1.0
    assert imp("private", 1, (1, 0)) == pytest.approx((ERROR_FLOOR - 1e-3) / 1e-3)
    assert imp("private", 1, (2, 0)) == pytest.approx(1.0 - 1e-2 / 10 ** -0.5)
    # no held-out points: no value (b answered problem 1 without a validation FVU)
    assert imp("b", 1, (1, 1)) is None


def test_the_cell_averages_improvability_over_problems_and_the_metric_is_listed_but_not_ranked():
    registry = {m["key"]: m for m in export.registry_json()}
    entry = registry[export.IMPROVABILITY]
    assert entry["kind"] == "cont" and entry["higher"] is False and entry["fmt"] == "pct"
    assert export.IMPROVABILITY not in export.RANK_KEYS            # it orders a problem's answers as the FVU does
    rows = {(1, 0): _row(1.0, -4.0), (1, 1): _row(1.0, -4.0)}
    rows[(1, 0)].derive(export.IMPROVABILITY, 0.5)
    rows[(1, 1)].derive(export.IMPROVABILITY, 1.0)
    cell = export.summarize_cell(rows, None)
    assert cell["m"][export.IMPROVABILITY][:3] == [2, 2, 1.5]


def test_the_pool_is_the_public_methods_without_the_references():
    pool = export.improvability_pool(["e2e", "PySR", "oracle", "T8-3M"])
    assert pool == ["e2e", "PySR", "T8-3M"]
