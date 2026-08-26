"""The float64 FVU is certified against exact rational arithmetic.

Motivating defect (2026-08-26): a benchmark harness divided by a float32 `np.var(y)`. On FastSRB
targets reaching |y| ~ 5e36 the float32 variance overflowed to +inf, so `fvu = finite/inf = 0.0`
sailed through the float32-eps bar as a free "perfect symbolic recovery" on 12 of 110 problems.
These tests pin the magnitudes where that happens and prove the shipped float64 path is right there.
"""
import numpy as np
import pytest

from srbf.metrics.numeric import fvu, fvu_exact, is_perfect_fit


class TestExactAgreesWithFloat64:
    @pytest.mark.parametrize("scale", [1e-30, 1e-12, 1.0, 1e12, 1e24, 5e36])
    def test_agreement_across_the_fastsrb_magnitude_span(self, scale: float) -> None:
        rng = np.random.default_rng(0)
        y = (rng.normal(size=64) * scale).astype(np.float64)
        yhat = y + rng.normal(size=64) * scale * 0.01
        fast, exact = fvu(y, yhat), fvu_exact(y, yhat)
        assert np.isfinite(exact)
        assert fast == pytest.approx(exact, rel=1e-9)

    def test_perfect_fit_is_exactly_zero_at_every_scale(self) -> None:
        for scale in (1e-30, 1.0, 5e36):
            y = np.array([1.0, 2.0, 3.0, 4.0]) * scale
            assert fvu_exact(y, y.copy()) == 0.0
            assert fvu(y, y.copy()) == 0.0


class TestFloat32WouldLie:
    def test_float32_variance_overflows_where_exact_arithmetic_does_not(self) -> None:
        # The exact shape of the harness defect, pinned as a test.
        y = (np.linspace(1.0, 2.0, 64) * 5.0e36).astype(np.float64)
        assert not np.isfinite(np.var(y.astype(np.float32)))   # the bug's precondition
        assert np.isfinite(np.var(y))                          # float64 is fine

        yhat = y * 1.5                                         # a badly wrong prediction
        naive_f32 = float(np.mean((yhat - y) ** 2) / np.var(y.astype(np.float32)))
        assert naive_f32 == 0.0                                # ... scored as a PERFECT fit
        assert fvu_exact(y, yhat) > 1.0                        # truth: worse than the mean
        assert fvu(y, yhat) > 1.0
        assert not is_perfect_fit(y, yhat)

    def test_tiny_magnitudes_do_not_underflow_to_a_free_win(self) -> None:
        y = (np.linspace(1.0, 2.0, 64) * 1e-30).astype(np.float64)
        yhat = y * 1.5
        assert fvu_exact(y, yhat) > 1.0
        assert fvu(y, yhat) > 1.0
        assert not is_perfect_fit(y, yhat)


class TestExactContract:
    def test_non_finite_prediction_is_worst(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        assert fvu_exact(y, np.array([1.0, np.nan, 3.0])) == np.inf
        assert fvu_exact(y, np.array([1.0, np.inf, 3.0])) == np.inf

    def test_constant_target_needs_an_exact_reproduction(self) -> None:
        y = np.array([2.5, 2.5, 2.5])
        assert fvu_exact(y, y.copy()) == 0.0
        assert fvu_exact(y, np.array([2.5, 2.5, 2.6])) == np.inf

    def test_length_mismatch_and_empty_are_invalid(self) -> None:
        assert fvu_exact(np.array([1.0, 2.0]), np.array([1.0])) == np.inf
        assert fvu_exact(np.array([]), np.array([])) == np.inf
        assert fvu_exact(None, np.array([1.0])) == np.inf

    def test_no_rounding_at_all(self) -> None:
        # A residual 1 ULP below the representable gap of the target: float64 sums lose it,
        # exact rational arithmetic keeps it. Documents WHY the exact path exists.
        y = np.array([1.0, 1.0 + 2**-52, 1.0 + 2**-51])
        yhat = y.copy()
        yhat[0] = np.nextafter(y[0], np.inf)
        assert fvu_exact(y, yhat) > 0.0
