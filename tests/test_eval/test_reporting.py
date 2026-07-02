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


def test_bootstrap_report_is_bit_reproducible_by_default():
    # Default rng=0: two identical calls must agree to the bit, and an explicit seed must
    # reproduce the default. rng=None opts back into fresh entropy per call.
    a = bootstrap_report(_snapshot(), "recovery", n=500)
    b = bootstrap_report(_snapshot(), "recovery", n=500)
    assert a == b
    c = bootstrap_report(_snapshot(), "recovery", n=500, rng=0)
    assert a == c
    d = bootstrap_report(_snapshot(), "recovery", n=500, rng=np.random.default_rng(0))
    assert a == d


def test_bootstrap_report_empty_is_nan_not_crash():
    snap = {"benchmark_eq_id": ["E1"], "recovery": [None], "placeholder": [False]}
    rep = bootstrap_report(snap, "recovery")
    assert rep["n_groups"] == 0 and np.isnan(rep["median"])


def test_draw_distribution_unknown_metric_raises():
    import pytest
    with pytest.raises(KeyError, match="nope"):
        draw_distribution(_snapshot(), "nope")


# --- WP1 primitives: draw_values + paired_expression_deltas ---

def test_draw_values_groups_draws_uncollapsed():
    from srbf.reporting import draw_values
    dist = draw_values(_snapshot(), "recovery")
    assert set(dist.keys()) == {"E1", "E2"}
    np.testing.assert_array_equal(sorted(dist["E1"]), [0.0, 1.0, 1.0])
    np.testing.assert_array_equal(dist["E2"], [0.0, 0.0, 0.0])


def test_draw_values_refuses_row_identity_fallback():
    import pytest
    from srbf.reporting import draw_values
    snap = {"recovery": [1.0, 0.0], "placeholder": [False, False]}  # no benchmark_eq_id column
    with pytest.raises(KeyError, match="row order"):
        draw_values(snap, "recovery")


def test_paired_deltas_join_by_id_permutation_invariant():
    from srbf.reporting import draw_values, paired_expression_deltas
    a = draw_values(_snapshot(), "recovery")
    # b = row-permuted copy of the same snapshot: deltas must be exactly zero
    snap = _snapshot()
    order = [3, 0, 5, 2, 6, 1, 4]
    permuted = {k: [v[i] for i in order] for k, v in snap.items()}
    b = draw_values(permuted, "recovery")
    result = paired_expression_deltas(a, b)
    assert result["n_pairs"] == 2 and result["n_only_a"] == result["n_only_b"] == 0
    np.testing.assert_allclose(result["deltas"], 0.0)


def test_paired_deltas_reports_one_sided_ids():
    from srbf.reporting import paired_expression_deltas
    a = {"E1": np.array([1.0]), "E2": np.array([0.5]), "E3": np.array([0.0])}
    b = {"E2": np.array([0.25]), "E4": np.array([1.0])}
    result = paired_expression_deltas(a, b)
    assert result["keys"] == ["E2"]
    np.testing.assert_allclose(result["deltas"], [0.25])
    assert result["only_a"] == ["E1", "E3"] and result["only_b"] == ["E4"]
    assert result["n_only_a"] == 2 and result["n_only_b"] == 1


def test_paired_deltas_profile_values_stack():
    from srbf.reporting import paired_expression_deltas
    # aggregate returning a 1-D profile per expression -> (n_common, k) delta matrix
    a = {"E1": np.array([[1.0, 0.5], [1.0, 0.7]]), "E2": np.array([[0.0, 0.0]])}
    b = {"E1": np.array([[0.5, 0.5]]), "E2": np.array([[0.0, 0.5]])}
    result = paired_expression_deltas(a, b, aggregate=lambda v: np.nanmean(v, axis=0))
    assert result["deltas"].shape == (2, 2)
    np.testing.assert_allclose(result["deltas"][0], [0.5, 0.1])   # E1
    np.testing.assert_allclose(result["deltas"][1], [0.0, -0.5])  # E2
