"""Unit tests for evaluation metric helpers."""
from __future__ import annotations

import numpy as np
import pytest

from srbf.metrics.bootstrap import bootstrapped_metric_ci
from srbf.metrics.numeric import fvu, is_perfect_fit
from srbf.metrics.zss import build_tree, zss_tree_edit_distance


def test_fvu_perfect_fit_is_zero() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert fvu(y, y) == 0.0
    assert bool(is_perfect_fit(y, y))


def test_fvu_moderate_bad_fit_is_finite_and_not_perfect() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    yp = np.array([1.0, 2.0, 3.0, 4.0, 500.0])
    val = fvu(y, yp)
    assert np.isfinite(val) and val > 0.0
    assert not is_perfect_fit(y, yp)


def test_fvu_non_finite_prediction_is_inf() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert fvu(y, np.array([1.0, 2.0, 3.0, 4.0, np.inf])) == np.inf
    assert not is_perfect_fit(y, np.array([1.0, 2.0, 3.0, 4.0, np.inf]))


def test_fvu_divergent_finite_prediction_does_not_spuriously_perfect_fit() -> None:
    # Regression: a finite-but-divergent prediction whose squared residual overflows
    # float64 used to collapse to fvu == 0.0 (spurious is_perfect_fit) via the 1/ss_res
    # rescale. It must report a terrible fit, not a perfect one.
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    yp = np.array([1.0, 2.0, 3.0, 4.0, 1e167])
    assert fvu(y, yp) > np.finfo(np.float32).eps
    assert not is_perfect_fit(y, yp)


def test_fvu_large_magnitude_good_fit_still_recovers() -> None:
    # The overflow fallback must NOT punish a genuinely good fit on large-magnitude data
    # (where the squared residual can overflow even though the relative error is tiny).
    y = np.array([1e160, 2e160, 3e160, 4e160, 5e160])
    yp = y * (1.0 + 1e-9)
    assert bool(is_perfect_fit(y, yp))


class _FixedIndices:
    """Generator-like stub: returns a fixed index matrix from ``integers`` (duck-typed rng)."""

    def __init__(self, indices: np.ndarray, expected_size: tuple[int, int] | None = None) -> None:
        self.indices = indices
        self.expected_size = expected_size

    def integers(self, low: int, high: int, size: tuple[int, int]) -> np.ndarray:
        if self.expected_size is not None:
            assert size == self.expected_size
        return self.indices


def test_bootstrap_returns_constant_interval_when_samples_identical() -> None:
    data = np.array([1.0, 2.0, 3.0], dtype=float)
    repeats = 5
    indices = np.tile(np.arange(len(data)), (repeats, 1))

    median, lower, upper = bootstrapped_metric_ci(
        data, np.mean, n=repeats, interval=0.9,
        rng=_FixedIndices(indices, expected_size=(repeats, len(data))))
    expected_mean = float(np.mean(data))

    assert median == pytest.approx(expected_mean)
    assert lower == pytest.approx(expected_mean)
    assert upper == pytest.approx(expected_mean)


def test_bootstrap_handles_percentage_interval_and_nd_samples() -> None:
    data = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=float)
    n_samples = 4
    indices = np.array([
        [0, 0],
        [1, 1],
        [0, 1],
        [1, 0],
    ])

    def test_metric(sample) -> np.ndarray:
        return float(sample[0, 0])

    median, lower, upper = bootstrapped_metric_ci(
        data, test_metric, n=n_samples, interval=80,
        rng=_FixedIndices(indices, expected_size=(n_samples, len(data))))

    assert median == pytest.approx(1.0)
    assert lower == pytest.approx(0.0)
    assert upper == pytest.approx(2.0)


def test_bootstrap_is_bit_reproducible_with_int_seed() -> None:
    data = np.random.default_rng(3).normal(size=50)
    a = bootstrapped_metric_ci(data, np.mean, n=200, rng=7)
    b = bootstrapped_metric_ci(data, np.mean, n=200, rng=7)
    assert a == b
    c = bootstrapped_metric_ci(data, np.mean, n=200, rng=np.random.default_rng(7))
    assert a == c


def test_bootstrap_unseeded_default_still_estimates() -> None:
    data = np.random.default_rng(4).normal(loc=2.0, size=200)
    median, lower, upper = bootstrapped_metric_ci(data, np.mean, n=500)
    assert lower <= median <= upper
    assert median == pytest.approx(2.0, abs=0.5)


def test_build_tree_converts_prefix_to_expected_structure() -> None:
    operators = {"+": 2, "*": 2}
    tree = build_tree(["+", "x1", "*", "x2", "x3"], operators)

    assert tree.label == "+"
    left, right = tree.children
    assert left.label == "x1"
    assert right.label == "*"

    mul_left, mul_right = right.children
    assert mul_left.label == "x2"
    assert mul_right.label == "x3"


def test_zss_tree_edit_distance_is_zero_for_identical_trees() -> None:
    operators = {"+": 2, "*": 2}
    expression = ["+", "x1", "*", "x2", "x3"]

    distance = zss_tree_edit_distance(expression, list(expression), operators)

    assert distance == pytest.approx(0.0)


def test_zss_tree_edit_distance_detects_label_changes() -> None:
    operators = {"+": 2, "*": 2}

    distance = zss_tree_edit_distance(["+", "x", "y"], ["*", "x", "y"], operators)

    assert distance == pytest.approx(1.0)


def test_zss_tree_edit_distance_detects_structural_changes() -> None:
    operators = {"+": 2, "*": 2}

    distance = zss_tree_edit_distance(["+", "x", "y"], ["+", "x", "*", "y", "z"], operators)

    assert distance > 1.0


# --- WP1: bootstrap_band (shape-agnostic sibling of bootstrapped_metric_ci) ---

def test_bootstrap_band_scalar_matches_bootstrapped_metric_ci():
    from srbf.metrics.bootstrap import bootstrap_band
    data = np.random.default_rng(5).normal(size=40)
    rng_a = np.random.default_rng(11)
    rng_b = np.random.default_rng(11)
    est, lo, hi = bootstrap_band(data, np.nanmean, n=300, rng=rng_a)
    m, l, u = bootstrapped_metric_ci(data, np.nanmean, n=300, rng=rng_b)
    assert (float(est), float(lo), float(hi)) == (m, l, u)


def test_bootstrap_band_profile_rows_give_pointwise_bands():
    from srbf.metrics.bootstrap import bootstrap_band
    rng = np.random.default_rng(6)
    # 60 rows x 4-point profiles with column means ~ [0, 1, 2, 3]
    data = rng.normal(loc=[0.0, 1.0, 2.0, 3.0], scale=0.5, size=(60, 4))
    est, lo, hi = bootstrap_band(data, np.nanmean, n=500, rng=0)
    assert est.shape == lo.shape == hi.shape == (4,)
    assert np.all(lo <= est) and np.all(est <= hi)
    np.testing.assert_allclose(est, [0.0, 1.0, 2.0, 3.0], atol=0.3)
