"""The price of an expression: its description length in bits, with its constants.

Every price srbf computes goes through :func:`price_bits`: the judge's MDL columns (:mod:`srbf.table`,
:func:`srbf.derive_metrics`), the ``selection: {mode: mdl}`` rule for a worker's candidates
(:mod:`srbf.subprocess_adapter`), and the box of the headline metrics (:mod:`srbf.headline`). A change to how
expressions are priced is a change to this one function.
"""
from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from typing import Any


def price_bits(engine: Any, prefix: Sequence[str]) -> float:
    """The description length of an expression in prefix notation, constants included, in bits, as the SimpliPy
    ``engine`` prices it (certified, f64 parse, the default canon). NaN when the engine cannot price the expression."""
    from simplipy.engine import Mode

    try:
        return float(engine.complexity(list(prefix), certified=True, mode=Mode.f64, canon="default")) / 1000.0
    except Exception:  # noqa: BLE001 - an expression the engine cannot price has no description length
        return float("nan")


def pricer(engine: Any) -> Callable[[Sequence[str]], float]:
    """:func:`price_bits` with the engine bound: a function from a prefix to its price in bits."""
    return functools.partial(price_bits, engine)


__all__ = ["price_bits", "pricer"]
