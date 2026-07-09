"""Exhaustive correctness + numerical-stability tests for BOTH FVU paths.

Covers the spec enumerated in experimental/eval/generator_refiner_decomposition/fvu_correctness_spec.md.

Two FVU implementations must be correct, scale-invariant, and mutually consistent under every
circumstance:
  - EVAL metric   : flash_ansr.eval.metrics.numeric.fvu(y_true, y_pred)  (recovery / is_perfect_fit)
  - SELECTION fvu : flash_ansr.scoring.compute_fvu(loss, n, variance)    (candidate scoring)

The load-bearing property is SCALE-INVARIANCE: fvu(alpha*y, alpha*yhat) == fvu(y, yhat) for any
alpha > 0. The historical bug was an ABSOLUTE variance floor (FLOAT64_EPS) in compute_fvu, which broke
scale-invariance for tiny-magnitude targets (a trivial constant scored ~perfect and was mis-selected).
"""
import inspect
import math

import numpy as np
import pytest

from srbf.metrics.numeric import fvu, is_perfect_fit, log10_fvu, safe_divide
from flash_ansr import scoring
from flash_ansr.scoring import compute_fvu, score_from_fvu

F32_EPS = float(np.finfo(np.float32).eps)


# --------------------------------------------------------------------------------------------------
# SCALE INVARIANCE  (the property the bug violated)
# --------------------------------------------------------------------------------------------------
class TestScaleInvariance:
    @pytest.mark.parametrize("alpha", [1e-100, 1e-50, 1e-15, 1e-8, 1.0, 1e8, 1e15, 1e50, 1e150])
    def test_eval_fvu_scale_invariant_in_safe_band(self, alpha):
        # eval fvu must be constant under a common rescale of y_true and y_pred (O(1) base, safe band)
        yt = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        yp = np.array([1.1, 2.0, 2.9, 4.2, 5.1])
        ref = fvu(yt, yp)
        assert fvu(alpha * yt, alpha * yp) == pytest.approx(ref, rel=1e-9)

    @pytest.mark.parametrize("alpha", [1e-30, 1e-16, 1e-8, 1.0, 1e8, 1e16, 1e30])
    def test_eval_fvu_predict_mean_is_one_at_every_scale(self, alpha):
        yt = alpha * np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert fvu(yt, np.full_like(yt, yt.mean())) == pytest.approx(1.0, rel=1e-9)

    @pytest.mark.parametrize("alpha", [1e-30, 1e-15, 1.0, 1e15, 1e30, 1e60])
    def test_scoring_fvu_scale_invariant(self, alpha):
        # compute_fvu(loss, n, var); a common data rescale multiplies BOTH loss and var by alpha**2
        loss0, var0, n = 0.0056, 2.5, 5
        ref = compute_fvu(loss0, n, var0)
        got = compute_fvu(alpha**2 * loss0, n, alpha**2 * var0)
        assert got == pytest.approx(ref, rel=1e-12)


# --------------------------------------------------------------------------------------------------
# THE HEADLINE BUG: tiny-magnitude targets must NOT read as spuriously perfect
# --------------------------------------------------------------------------------------------------
class TestTinyMagnitudeBugRegression:
    # magnitudes straddle the old absolute floor (FLOAT64_EPS=2.2e-16): every value <= ~2e-16 (i.e.
    # |y| <~ 1.5e-8) was corrupted by the old code -- not exotic. All genuinely fail on the old floor.
    @pytest.mark.parametrize("v", [2.0, 1e-8, 1e-15, 2.2e-16, 1e-16, 1e-17, 1e-32, 1e-64, 1e-100])
    def test_scoring_predict_mean_is_one_regardless_of_magnitude(self, v):
        # predict-the-mean => loss == variance => fvu == 1.0, at EVERY magnitude. The bug returned
        # loss/FLOAT64_EPS ~ 1e-49 for v=1e-64 (spuriously perfect).
        assert compute_fvu(v, 10, v) == pytest.approx(1.0, rel=1e-12)

    @pytest.mark.parametrize("var", [1e-30, 1e-64])
    def test_constant_candidate_loses_selection_to_correct(self, var):
        # The ACTUAL bug was mis-SELECTION: on a tiny-magnitude target both a trivial constant
        # (loss==var) and the correct fit (loss~0) had their fvu floored below score_from_fvu's floor,
        # collapsing their SCORES to a tie -> parsimony picked the SHORTER (constant) -> wrong output.
        # After the fix the constant's fvu==1.0 (un-floored) so it scores far worse than the correct fit.
        const_score = score_from_fvu(compute_fvu(var, 100, var), 1, 1, None, 0.05, 0.0, 0.0)
        correct_score = score_from_fvu(compute_fvu(var * 1e-16, 100, var), 12, 3, None, 0.05, 0.0, 0.0)
        assert correct_score < const_score   # lower score is selected; the correct fit must win
        # and the constant is no longer a (spurious) perfect fit
        assert compute_fvu(var, 100, var) == pytest.approx(1.0, rel=1e-9)

    def test_scoring_tiny_constant_is_not_perfect(self):
        # var(y) ~ 1e-64; a constant candidate has loss ~ var -> fvu ~ 1.0, NOT < float32 eps
        assert compute_fvu(2.0e-64, 5, 2.0e-64) > F32_EPS
        assert compute_fvu(2.0e-64, 5, 2.0e-64) == pytest.approx(1.0, rel=1e-9)

    def test_eval_agrees_tiny_constant_is_not_perfect(self):
        y = 1e-32 * np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert fvu(y, np.full_like(y, y.mean())) == pytest.approx(1.0, rel=1e-9)
        assert not is_perfect_fit(y, np.full_like(y, y.mean()))


# --------------------------------------------------------------------------------------------------
# CROSS-PATH CONSISTENCY: compute_fvu(loss, n, var(ddof=0)) == eval fvu, for the same data
# --------------------------------------------------------------------------------------------------
class TestCrossPathConsistency:
    @pytest.mark.parametrize("seed", range(8))
    @pytest.mark.parametrize("alpha", [1e-20, 1.0, 1e20])
    def test_scoring_matches_eval_ddof0(self, seed, alpha):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(5, 31))
        yt = alpha * rng.normal(0, 1, n)
        yp = yt + alpha * rng.normal(0, 0.3, n)
        loss = float(np.mean((yt - yp) ** 2))
        var = float(np.var(yt))  # ddof=0 -- MUST match numeric.fvu's plain np.mean ss_tot
        assert compute_fvu(loss, n, var) == pytest.approx(fvu(yt, yp), rel=1e-9)


# --------------------------------------------------------------------------------------------------
# DEFINITIONAL CASES (both paths)
# --------------------------------------------------------------------------------------------------
class TestDefinitional:
    def test_perfect_fit_zero(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert fvu(y, y) == 0.0
        assert is_perfect_fit(y, y)
        assert compute_fvu(0.0, 5, 2.0) == 0.0

    def test_predict_mean_is_one(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert fvu(y, np.full_like(y, 3.0)) == pytest.approx(1.0, rel=1e-12)
        assert compute_fvu(2.0, 5, 2.0) == pytest.approx(1.0)

    def test_worse_than_mean_above_one_not_clamped(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        yp = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        assert fvu(y, yp) == pytest.approx(4.0)
        assert compute_fvu(8.0, 5, 2.0) == pytest.approx(4.0)

    def test_fvu_never_negative(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            n = int(rng.integers(2, 40))
            yt = rng.normal(0, 1, n)
            yp = rng.normal(0, 1, n)
            assert fvu(yt, yp) >= 0.0


# --------------------------------------------------------------------------------------------------
# CONSTANT / DEGENERATE TARGET (variance == 0)
# --------------------------------------------------------------------------------------------------
class TestConstantTarget:
    def test_eval_constant_target(self):
        c = np.full(5, 3.0)
        assert fvu(c, c) == 0.0                 # matching const -> perfect
        assert fvu(c, c + 1.0) == np.inf        # wrong const -> inf
        assert not is_perfect_fit(c, c + 1.0)

    def test_scoring_constant_target(self):
        assert compute_fvu(0.0, 5, 0.0) == 0.0          # zero residual -> perfect
        assert compute_fvu(0.25, 5, 0.0) == np.inf      # nonzero residual -> inf


# --------------------------------------------------------------------------------------------------
# NON-FINITE INPUTS -> WORST (never spuriously best)
# --------------------------------------------------------------------------------------------------
class TestNonFiniteIsWorst:
    @pytest.mark.parametrize("loss", [np.nan, np.inf])
    def test_scoring_non_finite_loss_is_inf(self, loss):
        assert compute_fvu(loss, 5, 2.0) == np.inf

    @pytest.mark.parametrize("var", [np.nan, np.inf])
    def test_scoring_non_finite_variance_is_inf(self, var):
        assert compute_fvu(0.5, 5, var) == np.inf

    @pytest.mark.parametrize("bad_fvu", [-1.0, np.inf, -np.inf, np.nan])
    def test_score_from_fvu_non_finite_is_worst(self, bad_fvu):
        # a diverged/invalid candidate must score WORST (+inf), not best (ranking-inversion guard)
        assert score_from_fvu(bad_fvu, 0, 0, None, 0.0, 0.0, 0.0) == np.inf

    def test_score_from_fvu_perfect_is_best_finite(self):
        assert score_from_fvu(0.0, 0, 0, None, 0.0, 0.0, 0.0) == pytest.approx(float(np.log10(scoring.FLOAT64_EPS)))


# --------------------------------------------------------------------------------------------------
# EVAL NEVER-NAN + EXTREME-MAGNITUDE STABILITY
# --------------------------------------------------------------------------------------------------
class TestEvalNeverNanAndExtremes:
    def test_never_nan_over_many_inputs(self):
        rng = np.random.default_rng(1)
        cases = [
            (None, None), (np.array([1.0, 2.0]), None), (np.array([1.0, 2.0]), np.nan),
            (np.array([1.0, 2.0, 3.0]), np.array([1.0, np.inf, 3.0])),
            (np.array([1.0, np.inf]), np.array([1.0, 2.0])),
        ]
        for a in [1e-300, 1e-180, 1e-154, 1e-100, 1.0, 1e100, 1e154, 1e200]:
            base = a * rng.normal(0, 1, 6)
            cases += [(base, np.full_like(base, base.mean())), (base, base * (1 + 1e-9)), (base, np.zeros_like(base))]
        for yt, yp in cases:
            val = fvu(yt, yp)
            assert not (isinstance(val, float) and math.isnan(val)), f"fvu returned NaN for {yt!r},{yp!r}"

    @pytest.mark.parametrize("a", [1e-154, 1e-180, 1e-200])
    def test_extreme_tiny_predict_mean_is_one_not_zero_not_nan(self, a):
        # mirror of the scoring bug INSIDE eval: extreme-tiny used to underflow ss_res -> 0 (spurious
        # perfect) or produce nan. Must be ~1.0 (predict-mean) and NOT perfect.
        yt = a * np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        val = fvu(yt, np.full_like(yt, yt.mean()))
        assert not math.isnan(val)
        assert val == pytest.approx(1.0, rel=1e-6)
        assert not is_perfect_fit(yt, np.full_like(yt, yt.mean()))

    def test_extreme_tiny_exact_perfect_still_zero(self):
        yt = 1e-200 * np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert fvu(yt, yt.copy()) == 0.0

    def test_large_magnitude_good_fit_recovers(self):
        y = np.array([1e160, 2e160, 3e160, 4e160, 5e160])
        assert is_perfect_fit(y, y * (1.0 + 1e-9))

    def test_divergent_finite_prediction_not_spuriously_perfect(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        yp = np.array([1.0, 2.0, 3.0, 4.0, 1e167])
        assert fvu(y, yp) > F32_EPS
        assert not is_perfect_fit(y, yp)


# --------------------------------------------------------------------------------------------------
# INPUT CONTRACT (#7 list input, scalar nan, log10_fvu, safe_divide, is_perfect_fit threshold)
# --------------------------------------------------------------------------------------------------
class TestInputContract:
    def test_list_input_matches_ndarray(self):
        a = fvu(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.array([5.0, 4.0, 3.0, 2.0, 1.0]))
        assert fvu([1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0]) == pytest.approx(a)

    def test_scalar_nan_prediction_is_inf(self):
        assert fvu(np.array([1.0, 2.0, 3.0]), np.nan) == np.inf

    def test_none_inputs_are_inf(self):
        assert fvu(None, np.array([1.0])) == np.inf
        assert fvu(np.array([1.0]), None) == np.inf

    def test_object_array_and_zero_d_none_are_inf(self):
        # placeholder rows reach the metric as np.asarray(None) (a 0-d OBJECT array) or an object
        # array with embedded None; both are invalid -> inf (must NOT crash np.isfinite).
        assert fvu(np.array([1.0, 2.0, 3.0]), np.asarray(None)) == np.inf
        assert fvu(np.array([1.0, 2.0, 3.0]), np.array([1.0, None, 3.0], dtype=object)) == np.inf
        assert not bool(is_perfect_fit(np.array([1.0, 2.0]), np.asarray(None)))

    @pytest.mark.parametrize("k,expect", [(0.25, True), (4.0, False)])
    def test_is_perfect_fit_float32_threshold(self, k, expect):
        # craft data with fvu = k * F32_EPS: y=[0,1] -> ss_tot=var=0.25; a constant residual e gives
        # ss_res=e^2 -> fvu = e^2/0.25. (numpy bool -> compare with ==, not `is`; avoid the exact
        # boundary k=1 which is float-brittle.)
        y = np.array([0.0, 1.0])
        e = math.sqrt(k * F32_EPS * 0.25)
        assert bool(is_perfect_fit(y, y + e)) == expect

    def test_log10_fvu_specials(self):
        y = np.array([1.0, 2.0, 3.0])
        assert log10_fvu(y, y) == -np.inf            # perfect -> -inf
        assert log10_fvu(y, np.full_like(y, 2.0)) == pytest.approx(0.0)  # fvu=1 -> 0
        assert log10_fvu(None, y) == np.inf          # invalid -> +inf (log10(inf))

    @pytest.mark.parametrize("a,b,expect", [(0.0, 0.0, 0.0), (1.0, 0.0, np.inf), (0.0, 5.0, 0.0), (6.0, 2.0, 3.0)])
    def test_safe_divide_contract(self, a, b, expect):
        assert safe_divide(a, b) == expect

    def test_safe_divide_nan_propagation(self):
        assert math.isnan(safe_divide(np.nan, 1.0))
        assert math.isnan(safe_divide(1.0, np.nan))
        assert safe_divide(np.nan, 0.0) == np.inf   # b==0 branch precedes the nan check


# --------------------------------------------------------------------------------------------------
# NO REGRESSION + n=1 divergence + RECOVERY PROOF
# --------------------------------------------------------------------------------------------------
class TestRegressionAndScope:
    def test_in_distribution_scoring_unchanged(self):
        # normal-magnitude scoring (var >> FLOAT64_EPS) is byte-identical to the pre-fix behaviour
        assert compute_fvu(0.5, 10, 2.0) == 0.25

    def test_single_sample_returns_raw_loss(self):
        # n<=1: variance undefined -> raw loss (deployed convention; consistency is scoped to n>=2)
        assert compute_fvu(0.5, 1, 2.0) == 0.5
        assert compute_fvu(0.5, 0, 2.0) == 0.5

    def test_function_consistency_uses_ddof0(self):
        # compute_fvu == numeric.fvu when fed a ddof=0 variance (the function-level contract)
        rng = np.random.default_rng(3)
        yt = rng.normal(0, 1, 9)
        yp = yt + rng.normal(0, 0.4, 9)
        assert compute_fvu(float(np.mean((yt - yp) ** 2)), 9, float(np.var(yt))) == pytest.approx(fvu(yt, yp), rel=1e-9)

    def test_ddof_ratio_is_constant_factor(self):
        # The deployed feeders (flash_ansr.py, both baselines) are now standardized to ddof=0 so the
        # selection FVU equals numeric.fvu exactly. This pins the math fact that the OLD ddof=1
        # convention was a constant (n-1)/n reparametrization -- ranking-preserving (var(y) is the
        # same across a problem's candidates) and recovery-neutral -- which is why it never mattered.
        rng = np.random.default_rng(4)
        yt = rng.normal(0, 1, 7)
        yp = yt + rng.normal(0, 0.4, 7)
        loss = float(np.mean((yt - yp) ** 2))
        fvu_ddof0 = compute_fvu(loss, 7, float(np.var(yt)))            # function / eval convention
        fvu_ddof1 = compute_fvu(loss, 7, float(np.var(yt, ddof=1)))    # deployed inference convention
        assert fvu_ddof1 == pytest.approx(fvu_ddof0 * (7 - 1) / 7, rel=1e-12)

    def test_recovery_metric_independent_of_scoring_module(self):
        # the scoring fix cannot move is_perfect_fit: numeric.py does not import scoring.py
        import srbf.metrics.numeric as numeric_mod
        src = inspect.getsource(numeric_mod)
        assert "scoring" not in src
        # and is_perfect_fit on the tiny-y bug data is correctly False (eval is already right)
        y = 1e-32 * np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert not is_perfect_fit(y, np.full_like(y, y.mean()))
