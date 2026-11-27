"""The price of an expression: its description length in bits, with its constants, exactly as it was written.

Every price srbf computes goes through :func:`price_bits`: the judge's MDL columns (:mod:`srbf.table`,
:func:`srbf.derive_metrics`), the price of every stored candidate (:mod:`srbf.candidates`), the
``selection: {mode: mdl}`` rule for a worker's candidates (:mod:`srbf.subprocess_adapter`), and the box of the
headline metrics (:mod:`srbf.headline`). A change to how expressions are priced is a change to this one function.

The price is SimpliPy's written price, ``engine.written_complexity``: the engine's code book applied to the expression
as written, nothing simplified. Its callers hand it the written form: a method's answer as the engine's reader reads
the string the method returned, and a ground truth as its catalog writes it. Symbolic recovery is the part of the
judge that reads canonical forms (:mod:`srbf.result_processing`).
"""
from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from typing import Any


def price_bits(engine: Any, prefix: Sequence[str], *, refusals: list[str] | None = None) -> float:
    """The description length of an expression as written, constants included, in bits: the SimpliPy ``engine``'s
    written price (``written_complexity``, in milli-bits) over 1000. ``prefix`` holds the expression's tokens as the
    engine's reader returns them (``engine.read_infix``).

    NaN when the engine refuses to price the expression: a head it neither declares nor defines, or tokens that are
    not one expression. The refusal's first line is appended to ``refusals`` when a list is given, so that a caller
    can say why (:func:`srbf.candidates.judge_candidates`)."""
    try:
        return float(engine.written_complexity(list(prefix))) / 1000.0
    except Exception as exc:  # noqa: BLE001 - an expression the engine cannot price has no description length
        if refusals is not None:
            text = str(exc).strip()
            refusals.append(f"{type(exc).__name__}: {text.splitlines()[0]}" if text else type(exc).__name__)
        return float("nan")


def pricer(engine: Any) -> Callable[..., float]:
    """:func:`price_bits` with the engine bound: a function from a prefix to its price in bits."""
    return functools.partial(price_bits, engine)


__all__ = ["price_bits", "pricer"]
