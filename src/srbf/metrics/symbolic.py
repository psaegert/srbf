"""Symbolic / structural metrics for comparing predicted expressions."""
from __future__ import annotations

from collections.abc import Mapping, Sequence


#: Unary tokens that spell arithmetic, not a function: SimpliPy writes x - y as ``+ x neg y`` and x / y as
#: ``* x inv y``, so counting them would make the nesting depend on how a formula is spelled.
NOT_FUNCTIONS = frozenset({"neg", "inv"})


def total_nestedness(prefix_skeleton: Sequence[str], operator_arity: Mapping[str, int]) -> int:
    """The function nesting of a prefix skeleton: the most functions on any path from the root to a leaf.

    A function is a unary operator other than ``neg`` and ``inv`` (:data:`NOT_FUNCTIONS`). A binary operator does not
    break the nesting, and powers and roots (``pow``, ``rootn``) are binary. So ``x`` counts 0, ``sin(x)`` 1,
    ``sin(x) + cos(x)`` 1, ``sin(cos(x))`` 2, ``sin(x + cos(y))`` 2, ``exp(sin(x) + cos(x))`` 2 and ``-sin(x - y)`` 1.

    Parameters
    ----------
    prefix_skeleton : Sequence[str]
        Expression in prefix notation. A token that is not an operator is a leaf.
    operator_arity : Mapping[str, int]
        Map from operator name to its arity.

    Returns
    -------
    int
        The nesting depth.

    Raises
    ------
    ValueError
        When the tokens do not form one expression tree.
    """
    depths: list[int] = []
    for token in reversed(prefix_skeleton):
        arity = operator_arity.get(token, 0)
        if arity == 0:
            depths.append(0)
            continue
        if len(depths) < arity:
            raise ValueError(f"{token!r} lacks operands in {' '.join(map(str, prefix_skeleton))!r}")
        deepest = max(depths[-arity:])
        del depths[-arity:]
        depths.append(deepest + (1 if arity == 1 and token not in NOT_FUNCTIONS else 0))
    if len(depths) != 1:
        raise ValueError(f"not one expression tree: {' '.join(map(str, prefix_skeleton))!r}")
    return depths[0]
