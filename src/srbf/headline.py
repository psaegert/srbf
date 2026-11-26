"""The headline metrics: the front area of a run and the time index of a method.

A run is a method at one budget on one problem. It returns candidates, each with a price in bits (its description
length, :func:`srbf.pricing.price_bits`) and an FVU on the validation points (:func:`srbf.metrics.numeric.fvu`).

* :func:`front_area`: the share of the problem's box that the run's candidates dominate. The box spans lengths from
  the price of the support targets' mean written as a number (:func:`lower_edge_bits`) to 26 bits per support point
  (:func:`upper_edge_bits`), on a log2 axis, and FVU from the float64 epsilon to 1, on a log10 axis.
* :func:`time_index`: the front area averaged over a time budget drawn log-uniformly from 1 ms to 1000 s, from the
  areas of a method's budgets on one problem and the budgets' times (:func:`budget_time`). Only budgets that
  every larger budget of the method outlasts count (:func:`kept_budgets`).
* :func:`average_over_problems`: the problems' values (one run per problem) averaged over problem sets by the
  random-effects average of :mod:`srbf.aggregate`.

The method's index is the average of its time indices; the front area of a budget, averaged the same way, is plotted
at the budget's time. docs/headline.md has the definitions, the reasons and the sources.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from srbf.aggregate import Average, average
from srbf.metrics.numeric import LOG10_FVU_FLOOR
from srbf.pricing import price_bits

#: The float64 machine epsilon, 2^-52: the floor of an FVU, and the FVU at which the box's right edge is priced.
EPS = float(np.finfo(np.float64).eps)
#: Bits per support target at FVU = eps: (1/2) log2(1/eps) = 26, the Gaussian rate at a mean squared error of eps
#: times the targets' variance.
BITS_PER_TARGET = 0.5 * math.log2(1.0 / EPS)
#: The time range of the index in reference-machine seconds, a protocol constant: 1 ms to 1000 s.
T_MIN = 1e-3
T_MAX = 1e3


# ---- the box -----------------------------------------------------------------------------------------------------
def _finite_targets(y_support: Any) -> np.ndarray:
    y = np.asarray(y_support, dtype=np.float64).ravel()
    return y[np.isfinite(y)]


def mean_prefix(y_support: Any) -> list[str]:
    """The constant formula "predict the mean": the mean of the finite support targets as a one-token prefix, written
    as the shortest decimal that reads back as the same float64 (Python's ``repr``). Raises ValueError when no target is
    finite or the mean overflows."""
    y = _finite_targets(y_support)
    if y.size == 0:
        raise ValueError("the support targets hold no finite value")
    with np.errstate(over="ignore"):
        mean = float(np.mean(y))
    if not math.isfinite(mean):
        raise ValueError(f"the mean of the support targets is {mean}")
    return [repr(mean)]


def lower_edge_bits(y_support: Any, engine: Any) -> float:
    """The box's left edge, L_lo: the price in bits of the support targets' mean written as a number
    (:func:`mean_prefix`), priced by :func:`srbf.pricing.price_bits` like every prediction."""
    return price_bits(engine, mean_prefix(y_support))


def upper_edge_bits(n_support: int) -> float:
    """The box's right edge, L_hi = (n/2) log2(1/eps) = 26 n bits for n support points: what the n targets cost,
    given their mean and variance, to a mean squared error of eps times their variance (FVU = eps)."""
    if int(n_support) != n_support or n_support < 1:
        raise ValueError(f"the number of support points is a positive integer, got {n_support!r}")
    return int(n_support) * BITS_PER_TARGET


def box_edges(y_support: Any, engine: Any) -> tuple[float, float]:
    """``(L_lo, L_hi)`` of a problem from its support targets: :func:`lower_edge_bits` and
    :func:`upper_edge_bits` over the finite targets."""
    y = _finite_targets(y_support)
    return lower_edge_bits(y, engine), upper_edge_bits(y.size)


def _check_edges(l_lo: float, l_hi: float) -> None:
    for name, value in (("l_lo", l_lo), ("l_hi", l_hi)):
        if not (math.isfinite(value) and value > 0):
            raise ValueError(f"{name} is a positive number of bits, got {value!r}")


# ---- the front area ----------------------------------------------------------------------------------------------
def front_coordinates(bits: Any, fvu: Any, *, l_lo: float, l_hi: float) -> tuple[np.ndarray, np.ndarray]:
    """The candidates' coordinates in the unit box, ``(u, v)``, both clipped to [0, 1]:

    * u = (log2 L - log2 L_lo) / (log2 L_hi - log2 L_lo), 0 for a candidate at most L_lo long;
    * v = (log10 F - log10 eps) / (0 - log10 eps), the FVU F floored at eps.

    A candidate whose price or FVU is not finite (an expression the engine cannot price, a fit that diverged) is left
    out. Raises ValueError for arrays of different lengths, an edge that is not a positive number, or ``l_hi <= l_lo``.
    """
    _check_edges(l_lo, l_hi)
    if l_hi <= l_lo:
        raise ValueError(f"the box has no width: l_hi = {l_hi} <= l_lo = {l_lo}")
    length = np.asarray(bits, dtype=np.float64).ravel()
    error = np.asarray(fvu, dtype=np.float64).ravel()
    if length.shape != error.shape:
        raise ValueError(f"{length.size} prices for {error.size} FVUs")
    keep = np.isfinite(length) & np.isfinite(error)
    length, error = length[keep], error[keep]
    u = np.where(length <= l_lo, 0.0,
                 (np.log2(np.maximum(length, l_lo)) - math.log2(l_lo)) / (math.log2(l_hi) - math.log2(l_lo)))
    v = (np.log10(np.maximum(error, EPS)) - LOG10_FVU_FLOOR) / (0.0 - LOG10_FVU_FLOOR)
    return np.clip(u, 0.0, 1.0), np.clip(v, 0.0, 1.0)


def dominated_area(u: Any, v: Any) -> float:
    """The area of the union of the rectangles [u_i, 1] x [v_i, 1]: the share of the unit box the points dominate (the
    2-D hypervolume with reference point (1, 1)). Computed from the staircase of the points, without a grid; 0 for no
    points. Coordinates outside [0, 1] are clipped."""
    u = np.clip(np.asarray(u, dtype=np.float64).ravel(), 0.0, 1.0)
    v = np.clip(np.asarray(v, dtype=np.float64).ravel(), 0.0, 1.0)
    if u.shape != v.shape:
        raise ValueError(f"{u.size} u for {v.size} v")
    keep = (u < 1.0) & (v < 1.0)          # a point on the far edges dominates nothing
    u, v = u[keep], v[keep]
    if u.size == 0:
        return 0.0
    order = np.lexsort((v, u))            # by u, ties by v
    u, v = u[order], v[order]
    # the front: the points below every point to their left (a dominated point or a duplicate never enters the sum)
    previous_lowest = np.append(np.inf, np.minimum.accumulate(v)[:-1])
    front = v < previous_lowest
    u, v = u[front], v[front]             # u strictly increasing, v strictly decreasing
    widths = np.diff(np.append(u, 1.0))
    return math.fsum((widths * (1.0 - v)).tolist())


def front_area(bits: Any, fvu: Any, *, l_lo: float, l_hi: float) -> float:
    """The front area of a run: the share of the box that its candidates dominate, in [0, 1].

    ``bits`` and ``fvu`` hold each candidate's price in bits and its FVU on the validation points; ``l_lo`` and
    ``l_hi`` are the box's edges (:func:`box_edges`). A run without candidates (a failed run) has area 0, as does a
    box without width (``l_hi <= l_lo``, possible only with one or two support points). Candidates as long as L_hi or
    longer, or no better than the mean (FVU >= 1), dominate nothing; see :func:`front_coordinates` for the rest."""
    _check_edges(l_lo, l_hi)
    if l_hi <= l_lo:
        return 0.0
    return dominated_area(*front_coordinates(bits, fvu, l_lo=l_lo, l_hi=l_hi))


# ---- over time ---------------------------------------------------------------------------------------------------
def budget_time(times: Any) -> float:
    """A budget's time: the geometric mean of its calibrated per-problem times in seconds (failures included).
    Raises ValueError for no times, or a time that is not a positive finite number."""
    t = np.asarray(times, dtype=np.float64).ravel()
    if t.size == 0:
        raise ValueError("a budget's time needs at least one per-problem time")
    if not (np.isfinite(t).all() and (t > 0).all()):
        raise ValueError("per-problem times are positive finite seconds")
    return float(np.exp(np.mean(np.log(t))))


def _budgets_and_times(budgets: Any, times: Any) -> tuple[np.ndarray, np.ndarray]:
    b = np.asarray(budgets, dtype=np.float64).ravel()
    t = np.asarray(times, dtype=np.float64).ravel()
    if b.shape != t.shape:
        raise ValueError(f"{b.size} budgets for {t.size} times")
    if not np.isfinite(b).all() or np.unique(b).size != b.size:
        raise ValueError("budgets are distinct finite numbers")
    if not (np.isfinite(t).all() and (t > 0).all()):
        raise ValueError("budget times are positive finite seconds")
    return b, t


def kept_budgets(budgets: Any, times: Any) -> np.ndarray:
    """Which of a method's budgets count, in the input order: a budget counts only if every larger budget of the method
    is slower (a strictly longer time). Decided by the budgets and their times alone, never by results. Budgets whose
    time is overhead-bound and out of order drop out."""
    b, t = _budgets_and_times(budgets, times)
    if b.size == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(b)
    t_sorted = t[order]
    # the shortest time among the larger budgets (inf for the largest)
    faster_later = np.append(np.minimum.accumulate(t_sorted[::-1])[::-1][1:], np.inf)
    keep = np.empty(b.size, dtype=bool)
    keep[order] = t_sorted < faster_later
    return keep


def time_index(budgets: Any, times: Any, areas: Any, *, t_min: float = T_MIN, t_max: float = T_MAX) -> float:
    """The time index of a method on one problem, in [0, 1] for areas in [0, 1].

    ``budgets``, ``times`` and ``areas`` give, per budget, the budget's value, its time in seconds (:func:`budget_time`)
    and the run's front area. The budgets that count (:func:`kept_budgets`), ordered by time tau = log10 t, define the
    front area over time q(tau): 0 before the first budget's time, linear in tau between budgets, and held at the last
    budget's area after it (the largest budget can always be run again). The index is the mean of q over
    [log10 t_min, log10 t_max]; budgets outside that range are cut at its edges. The integral of the piecewise-linear q
    is summed in closed form, trapezoid by trapezoid."""
    if not (math.isfinite(t_min) and math.isfinite(t_max) and 0 < t_min < t_max):
        raise ValueError(f"the time range is 0 < t_min < t_max, got ({t_min}, {t_max})")
    b, t = _budgets_and_times(budgets, times)
    a = np.asarray(areas, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"{b.size} budgets for {a.size} areas")
    if not np.isfinite(a).all():
        raise ValueError("areas are finite numbers")
    keep = kept_budgets(b, t)
    tau, a = np.log10(t[keep]), a[keep]
    order = np.argsort(tau)
    tau, a = tau[order], a[order]
    lo, hi = math.log10(t_min), math.log10(t_max)
    if tau.size == 0 or tau[0] >= hi:
        return 0.0
    start = max(lo, float(tau[0]))
    knots = np.concatenate(([start], tau[(tau > start) & (tau < hi)], [hi]))
    values = np.interp(knots, tau, a)     # interpolates between budgets, holds A_K after the last one
    integral = math.fsum((np.diff(knots) * (values[:-1] + values[1:]) / 2.0).tolist())
    return integral / (hi - lo)


# ---- over problems -----------------------------------------------------------------------------------------------
def average_over_problems(problems: Mapping[str, Sequence[float | Sequence[float]]]) -> Average | None:
    """The average over problem sets of a per-problem value (a time index, or the front area at one budget):
    ``problems[set]`` lists the value of each problem of the set, the value of its run. The sets are combined by the
    random-effects average of :func:`srbf.aggregate.average` (kind ``"mean"``), with its 95 % interval over these
    problem sets. A problem given as a sequence of several runs' values counts with their mean. None when no set has a
    value."""
    runs = {name: [np.ravel(np.asarray(x, dtype=np.float64)).tolist() for x in values] for name, values in problems.items()}
    return average(runs, "mean")


__all__ = [
    "BITS_PER_TARGET", "EPS", "T_MAX", "T_MIN", "average_over_problems", "box_edges", "budget_time", "dominated_area",
    "front_area", "front_coordinates", "kept_budgets", "lower_edge_bits", "mean_prefix", "time_index",
    "upper_edge_bits",
]
