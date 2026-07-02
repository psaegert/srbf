"""Metric utilities, including bootstrap confidence intervals."""
from typing import Callable

import numpy as np


def bootstrapped_metric_ci(
    data: np.ndarray,
    metric: Callable[[np.ndarray], float],
    n: int = 10_000,
    interval: float = 0.95,
    rng: np.random.Generator | int | None = None,
) -> tuple[float, float, float]:
    """Estimate ``metric`` on ``data`` with a bootstrap confidence interval.

    The data is resampled with replacement ``n`` times; ``metric`` is applied to
    each resample and the point estimate plus a two-sided percentile confidence
    interval are returned.

    Parameters
    ----------
    data : np.ndarray
        The sample to bootstrap over. Resampling is performed along the first axis.
    metric : Callable[[np.ndarray], float]
        Reducer applied to each bootstrap resample to produce a scalar statistic.
    n : int, optional
        Number of bootstrap resamples, by default 10_000.
    interval : float, optional
        Confidence level as a fraction in ``(0, 1]``, by default 0.95. As a
        convenience, a value in ``(1, 100]`` is treated as a percentage and
        divided by 100 (e.g. ``95`` becomes ``0.95``); values ``> 100`` are used
        as-is.
    rng : np.random.Generator or int or None, optional
        Resampling randomness: a ``Generator`` is used as-is, an int seeds a fresh
        ``np.random.default_rng(rng)`` (bit-reproducible results), and ``None``
        (default) draws fresh entropy so repeated calls differ.

    Returns
    -------
    tuple[float, float, float]
        ``(median, ci_lower, ci_upper)``: the bootstrap median (point estimate)
        and the lower/upper percentile bounds of the confidence interval. For
        the default ``interval=0.95`` these are the 2.5th and 97.5th percentiles.
    """
    if interval > 1 and interval <= 100:
        interval /= 100

    n = int(n)

    if not hasattr(rng, "integers"):  # anything Generator-like is used as-is
        rng = np.random.default_rng(rng)
    indices = rng.integers(0, len(data), size=(n, len(data)))
    samples = data[indices]

    bootstrapped_metrics: np.ndarray | None = None
    if samples.ndim == 2:
        # Fast path: the reducers used here (np.nanmean/np.mean/np.median/...) accept an `axis`
        # kwarg, so all n resamples reduce in one vectorized call instead of an n-iteration Python
        # loop (np.apply_along_axis is itself a per-row loop). Fall back below for any metric that
        # does not support `axis` or does not return one scalar per resample.
        try:
            # metric is typed (ndarray)->float, but numpy reducers given axis= return an ndarray;
            # the ignores cover the extra kwarg and that runtime-vs-annotation return mismatch.
            vectorized = np.asarray(metric(samples, axis=1), dtype=float)  # type: ignore[call-arg, type-var]
        except TypeError:
            vectorized = None
        if vectorized is not None and vectorized.shape == (samples.shape[0],):
            bootstrapped_metrics = vectorized

    if bootstrapped_metrics is None:
        bootstrapped_metrics = np.array([metric(sample) for sample in samples])

    median = np.nanmedian(bootstrapped_metrics)
    lower = np.nanpercentile(bootstrapped_metrics, (1 - interval) / 2 * 100)
    upper = np.nanpercentile(bootstrapped_metrics, (1 + interval) / 2 * 100)

    return float(median), float(lower), float(upper)


def bootstrap_band(
    data: np.ndarray,
    reduce: Callable[..., np.ndarray | float] = np.nanmean,
    *,
    n: int = 10_000,
    interval: float = 0.95,
    rng: np.random.Generator | int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap ``reduce`` over the ROWS of ``data``; shape-agnostic sibling of
    :func:`bootstrapped_metric_ci`.

    ``data`` has shape ``(n_rows,)`` (scalar per row, e.g. one paired delta per expression) or
    ``(n_rows, k)`` (a 1-D profile per row, e.g. a recovery profile over k thresholds). Rows are
    resampled with replacement ``n`` times and ``reduce`` is applied along axis 0, so the scalar
    case reproduces :func:`bootstrapped_metric_ci` and the profile case yields pointwise bands
    (label them as such: the band is pointwise, not simultaneous).

    Parameters
    ----------
    data : np.ndarray
        ``(n_rows,)`` or ``(n_rows, k)`` array; rows are the exchangeable units.
    reduce : Callable, optional
        Statistic applied along ``axis=0`` of each resample (default ``np.nanmean``).
    n, interval, rng
        As in :func:`bootstrapped_metric_ci`.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(estimate, lower, upper)``, each of shape ``()`` for scalar rows or ``(k,)`` for
        profile rows. ``estimate`` is the median of the bootstrap distribution (house
        convention, matching :func:`bootstrapped_metric_ci`).
    """
    if interval > 1 and interval <= 100:
        interval /= 100

    n = int(n)
    data = np.asarray(data, dtype=float)
    n_rows = data.shape[0]

    if not hasattr(rng, "integers"):  # anything Generator-like is used as-is
        rng = np.random.default_rng(rng)
    indices = rng.integers(0, n_rows, size=(n, n_rows))

    # (n, n_rows) or (n, n_rows, k) resample stack; reduce along the rows axis.
    samples = data[indices]
    reduced = np.asarray(reduce(samples, axis=1), dtype=float)  # (n,) or (n, k)

    estimate = np.nanmedian(reduced, axis=0)
    lower = np.nanpercentile(reduced, (1 - interval) / 2 * 100, axis=0)
    upper = np.nanpercentile(reduced, (1 + interval) / 2 * 100, axis=0)
    return estimate, lower, upper
