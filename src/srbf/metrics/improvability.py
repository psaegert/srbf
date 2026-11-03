"""Improvability: how far a prediction's test error is from the best error a fixed pool of methods reaches.

For a problem with the pool's best error ``e*``, a prediction with error ``e`` has the improvability

    (e - e*) / e  =  1 - e* / e

(Erickson et al. 2025, TabArena; its mean over problems is the Mean Improvability).
0 is the pool's best, 1 is a prediction with an unbounded error or none at all. The error is the test RMSE in units
of the test targets' standard deviation, sqrt(FVU): on one problem's test points every method's RMSE is that same
multiple of it, so the ratio is the ratio of the RMSEs, and the problems share one scale.

Two predictions that both reproduce the test points within float32 precision (FVU <= 2^-23, Numeric Recovery) tie:
the error is read no lower than sqrt(2^-23), so the digits a recovered prediction holds below it, which are round-off,
cannot make it worse than another recovered one.

A method outside the pool can beat the pool's best: its improvability is then negative.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

#: The error floor: two predictions at or below it tie (sqrt of float32 epsilon, the Numeric Recovery bar on FVU).
ERROR_FLOOR = math.sqrt(float(np.finfo(np.float32).eps))


def validation_error(log10_fvu: float | None, success: bool) -> float | None:
    """A prediction's test error, sqrt(FVU) floored at :data:`ERROR_FLOOR`, from its log10 FVU on the test points.

    A failed prediction has an unbounded error (``inf``); a successful one without a test FVU (a problem without test
    points) has none (``None``)."""
    if not success:
        return math.inf
    if log10_fvu is None or math.isnan(log10_fvu):
        return None
    return max(10.0 ** (log10_fvu / 2.0), ERROR_FLOOR)


def pool_best(errors: Iterable[float | None]) -> float | None:
    """The best (lowest) finite error among the pool's predictions for one problem, or None if none has one."""
    finite = [e for e in errors if e is not None and math.isfinite(e)]
    return min(finite) if finite else None


def improvability(error: float | None, best: float | None) -> float | None:
    """``1 - best / error``: 0 at the pool's best, 1 for an unbounded error, negative below the pool's best.
    None where either error is undefined."""
    if error is None or best is None:
        return None
    if math.isinf(error):
        return 1.0
    return 1.0 - best / error
