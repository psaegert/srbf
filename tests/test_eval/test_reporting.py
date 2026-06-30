"""Multi-draw reporting: per-expression grouping, placeholder exclusion, bootstrap CI shape."""
import numpy as np

from srbf.reporting import bootstrap_report, draw_distribution


def _snapshot():
    # 2 expressions x 3 draws each (6 rows) + 1 placeholder row that must be excluded.
    return {
        "benchmark_eq_id": ["E1", "E1", "E1", "E2", "E2", "E2", "E3"],
        "recovery": [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, None],
        "placeholder": [False, False, False, False, False, False, True],
    }


def test_draw_distribution_collapses_draws_per_expression():
    dist = draw_distribution(_snapshot(), "recovery")
    # E1 mean over [1,1,0] = 2/3; E2 mean over [0,0,0] = 0; E3 is placeholder -> excluded
    assert set(dist.keys()) == {"E1", "E2"}
    assert dist["E1"] == 2 / 3 and dist["E2"] == 0.0


def test_draw_distribution_without_group_key_is_per_row():
    snap = {"recovery": [1.0, 0.0, 1.0], "placeholder": [False, False, False]}
    dist = draw_distribution(snap, "recovery")
    assert sorted(dist.values()) == [0.0, 1.0, 1.0]  # each row its own group (integer keys)
    assert len(dist) == 3


def test_bootstrap_report_shape_and_bounds():
    rep = bootstrap_report(_snapshot(), "recovery", n=2000)
    assert rep["metric"] == "recovery"
    assert rep["n_groups"] == 2          # E1, E2 (E3 placeholder excluded)
    assert rep["n_rows"] == 6            # 6 non-placeholder rows
    # the per-expression values are {2/3, 0}; mean ~ 1/3, CI within [0, 2/3]
    assert 0.0 <= rep["ci_lower"] <= rep["median"] <= rep["ci_upper"] <= 2 / 3 + 1e-9
    assert rep["interval"] == 0.95


def test_bootstrap_report_empty_is_nan_not_crash():
    snap = {"benchmark_eq_id": ["E1"], "recovery": [None], "placeholder": [False]}
    rep = bootstrap_report(snap, "recovery")
    assert rep["n_groups"] == 0 and np.isnan(rep["median"])


def test_draw_distribution_unknown_metric_raises():
    import pytest
    with pytest.raises(KeyError, match="nope"):
        draw_distribution(_snapshot(), "nope")
