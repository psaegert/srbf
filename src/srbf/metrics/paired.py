"""Paired significance tests for benchmark arm contrasts.

A symbolic-regression arm is scored as a per-problem BINARY outcome (recovered / not) over a
problem set that every arm sees. That makes arm-vs-arm an exactly-paired comparison, and the
right question is not "which percentage is bigger" but "how many problems actually MOVED, and
could that many have moved by chance". Two arms differing by 5 points on 110 problems can rest
on a net of two problems.

These are deliberately dependency-free (no scipy): an exact binomial tail is a few lines with
``math.comb`` and stays importable anywhere the metrics package is.
"""
from __future__ import annotations

from math import comb, sqrt
from typing import NamedTuple, Sequence

import numpy as np


class McNemarResult(NamedTuple):
    """Outcome of an exact McNemar test on paired binary results."""

    p_value: float
    n_a_only: int      #: problems arm A solved and arm B did not
    n_b_only: int      #: problems arm B solved and arm A did not
    n_both: int
    n_neither: int

    @property
    def n_discordant(self) -> int:
        """Problems on which the two arms disagree -- the only ones the test can see."""
        return self.n_a_only + self.n_b_only


def mcnemar_exact(a: Sequence[bool] | np.ndarray, b: Sequence[bool] | np.ndarray) -> McNemarResult:
    """Exact two-sided McNemar test on paired binary outcomes.

    Conditional on the number of discordant pairs, the count favouring one arm is Binomial(n, 0.5)
    under the null that the arms are equally likely to win a problem the other loses. The exact
    tail is used rather than the chi-square approximation because benchmark discordance counts are
    routinely below 20, where the approximation is not trustworthy.

    Parameters
    ----------
    a, b : sequence of bool
        Per-problem success indicators for the two arms, ALIGNED BY PROBLEM. Same length, same
        order; index i must be the same problem in both.

    Returns
    -------
    McNemarResult
        ``p_value`` plus the full 2x2 table, so a caller can report "13 vs 4 problems moved"
        instead of only a percentage difference.

    Raises
    ------
    ValueError
        If the two arms have different lengths (they are then not paired).
    """
    arm_a = np.asarray(a, dtype=bool).ravel()
    arm_b = np.asarray(b, dtype=bool).ravel()
    if arm_a.shape != arm_b.shape:
        raise ValueError(
            f"Paired tests need one outcome per problem in both arms, got {arm_a.size} and "
            f"{arm_b.size}. Align the arms on the problem index before testing.")

    n_a_only = int(np.sum(arm_a & ~arm_b))
    n_b_only = int(np.sum(~arm_a & arm_b))
    n_both = int(np.sum(arm_a & arm_b))
    n_neither = int(np.sum(~arm_a & ~arm_b))

    n = n_a_only + n_b_only
    if n == 0:
        # No problem moved in either direction: the arms are indistinguishable on this set.
        return McNemarResult(1.0, n_a_only, n_b_only, n_both, n_neither)

    k = min(n_a_only, n_b_only)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    p_value = min(1.0, 2.0 * tail)
    return McNemarResult(float(p_value), n_a_only, n_b_only, n_both, n_neither)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (default 95%).

    Preferred over the normal approximation at the rates benchmarks actually report: at 15/110 the
    Wald interval is visibly wrong and can extend below zero.

    Parameters
    ----------
    successes : int
        Number of successes.
    total : int
        Number of trials. ``0`` yields ``(0.0, 1.0)`` -- no information.
    z : float, optional
        Normal quantile; the default is the two-sided 95% value.

    Returns
    -------
    tuple[float, float]
        ``(lower, upper)`` as fractions in ``[0, 1]``.
    """
    if total <= 0:
        return (0.0, 1.0)
    p_hat = successes / total
    denom = 1.0 + z * z / total
    centre = (p_hat + z * z / (2 * total)) / denom
    half = (z / denom) * sqrt(p_hat * (1 - p_hat) / total + z * z / (4 * total * total))
    return (max(0.0, centre - half), min(1.0, centre + half))


def paired_difference_ci(
    a: Sequence[bool] | np.ndarray,
    b: Sequence[bool] | np.ndarray,
    n_resamples: int = 10_000,
    interval: float = 0.95,
    rng: np.random.Generator | int | None = 0,
) -> tuple[float, float, float]:
    """Bootstrap CI for the difference in success rate ``mean(b) - mean(a)``, resampled BY PROBLEM.

    Resampling problems (not outcomes independently) preserves the pairing, so the interval
    reflects the same correlation the McNemar test conditions on. A paired difference of +9 points
    whose interval crosses zero is not a result.

    Parameters
    ----------
    a, b : sequence of bool
        Per-problem success indicators, aligned by problem.
    n_resamples : int, optional
        Bootstrap resamples, by default 10_000.
    interval : float, optional
        Two-sided confidence level, by default 0.95.
    rng : np.random.Generator or int or None, optional
        Seeded by default (0) so a published interval is reproducible.

    Returns
    -------
    tuple[float, float, float]
        ``(point_estimate, ci_lower, ci_upper)`` as fractions.
    """
    arm_a = np.asarray(a, dtype=bool).ravel()
    arm_b = np.asarray(b, dtype=bool).ravel()
    if arm_a.shape != arm_b.shape:
        raise ValueError(
            f"Paired tests need one outcome per problem in both arms, got {arm_a.size} and {arm_b.size}.")
    if arm_a.size == 0:
        return (0.0, 0.0, 0.0)

    if not hasattr(rng, "integers"):
        rng = np.random.default_rng(rng)

    n_problems = arm_a.size
    idx = rng.integers(0, n_problems, size=(int(n_resamples), n_problems))
    diffs = arm_b[idx].mean(axis=1) - arm_a[idx].mean(axis=1)

    alpha = (1.0 - interval) / 2.0
    lo, hi = np.percentile(diffs, [100 * alpha, 100 * (1 - alpha)])
    point = float(arm_b.mean() - arm_a.mean())
    return (point, float(lo), float(hi))
