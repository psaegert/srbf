"""Numeric evaluation metrics for comparing predictions to ground truth."""
from __future__ import annotations

import numpy as np


def safe_divide(a: float, b: float) -> float:
    """Divide ``a`` by ``b``, returning 0 for 0/0 and inf for x/0."""
    if b == 0:
        if a == 0:
            return 0.0
        return np.inf
    if np.isnan(a) or np.isnan(b):
        return np.nan
    return a / b


@np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore")
def fvu(y_true: np.ndarray | None, y_pred: np.ndarray | None) -> float:
    """Compute Fraction of Variance Unexplained between two arrays.

    Uses numerical scaling to avoid floating-point precision issues.

    Parameters
    ----------
    y_true : np.ndarray or None
        Ground truth values.
    y_pred : np.ndarray or None
        Predicted values.

    Returns
    -------
    float
        FVU value.  Returns ``np.inf`` when inputs are invalid.
    """
    if y_pred is None or y_true is None:
        return np.inf

    # Coerce to float64 arrays up front. Non-numeric input (a bare scalar, a python list, or a
    # None embedded in an object array from a placeholder row) is invalid -> worst (inf). Doing the
    # conversion here -- rather than a pre-conversion np.isnan() guard -- fixes both a list-input
    # crash and an object-dtype crash; a scalar/array nan is then caught by the finite-check below.
    try:
        y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
        y_true = np.asarray(y_true, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        return np.inf

    # Degenerate empty inputs -> inf per the documented contract (an empty array would crash the
    # np.max reduction in the non-finite branch below).
    if y_pred.size == 0 or y_true.size == 0:
        return np.inf

    # Ground truth finite but prediction not (incl. a scalar/array nan) → infinite error
    if np.isfinite(y_true).all() and not np.isfinite(y_pred).all():
        return np.inf

    def _normalized_by_gt() -> float:
        # Scale-invariant fallback for when the 1/ss_res rescale would mis-behave: normalize the
        # residual and total spread by the GT magnitude BEFORE squaring (an O(1) normalizer that
        # never collapses a genuine divergence to 0 nor produces nan). Used by both the OVERFLOW
        # guard (huge divergent residual) and the UNDERFLOW guard (tiny-magnitude data).
        denom = np.max(np.abs(y_true))
        if not np.isfinite(denom) or denom == 0.0:
            return np.inf
        ss_res_n = np.mean(((y_true - y_pred) / denom) ** 2)
        ss_tot_n = np.mean(((y_true - np.mean(y_true)) / denom) ** 2)
        if not np.isfinite(ss_res_n) or not np.isfinite(ss_tot_n):
            return np.inf
        return safe_divide(ss_res_n, ss_tot_n)

    # Scale by inverse MSE to avoid numerical issues. The squarings below intentionally probe for
    # overflow/underflow on extreme-magnitude inputs and branch on the finite-checks; the @np.errstate
    # decorator silences the expected numpy warnings so callers never see spam.
    ss_res = np.mean((y_true - y_pred) ** 2)
    if ss_res == 0:
        # Either a genuine perfect fit, OR squared-residual UNDERFLOW on tiny-magnitude data
        # (e.g. |y| ~ 1e-180: (diff)^2 underflows to 0 -> would spuriously read as perfect).
        # Distinguish by the un-squared max residual, which does not underflow.
        if np.max(np.abs(y_true - y_pred)) == 0.0:
            return 0.0                 # genuine perfect fit (byte-identical to before)
        return _normalized_by_gt()     # underflow -> recover a scale-invariant result

    # Overflow guard. A finite-but-DIVERGENT prediction can make the squared residual overflow to
    # +inf; the 1/ss_res rescale below would then drive scale -> 0, collapsing every term so
    # safe_divide(0, 0) returns 0.0 and is_perfect_fit() spuriously fires
    # (e.g. fvu([1,2,3,4,5], [1,2,3,4,1e167]) used to return 0.0).
    if not np.isfinite(ss_res):
        return _normalized_by_gt()

    scale = 1.0 / ss_res

    ss_res_s = np.mean((y_true * scale - y_pred * scale) ** 2)
    ss_tot_s = np.mean((y_true * scale - np.mean(y_true * scale, keepdims=True)) ** 2)

    # Underflow guard. A subnormal-but-finite ss_res makes `scale` huge enough that the rescaled
    # sums overflow to +inf -> safe_divide(inf, inf) = nan (e.g. |y| ~ 1e-154). FVU must never be
    # nan; fall back to the GT-magnitude normalization. (Never fires for normal-magnitude data, so
    # the finite path stays byte-identical.)
    if not np.isfinite(ss_res_s) or not np.isfinite(ss_tot_s):
        return _normalized_by_gt()

    return safe_divide(ss_res_s, ss_tot_s)


def log10_fvu(y_true: np.ndarray | None, y_pred: np.ndarray | None) -> float:
    """Compute log10 of the Fraction of Variance Unexplained.

    Returns ``-np.inf`` when FVU is exactly zero (perfect fit).
    """
    fvu_value = fvu(y_true, y_pred)
    if fvu_value == 0:
        return -np.inf
    return np.log10(fvu_value)


def is_perfect_fit(y_true: np.ndarray | None, y_pred: np.ndarray | None) -> bool:
    """Check if ``y_pred`` perfectly fits ``y_true`` within float32 epsilon."""
    return bool(fvu(y_true, y_pred) <= np.finfo(np.float32).eps)


def naninfmean(a: np.ndarray) -> float:
    """Compute the mean of an array, ignoring NaN and Inf values."""
    a = np.asarray(a, dtype=np.float64)
    return float(np.nanmean(a[np.isfinite(a)]))
