"""The engine's spelling of what a method printed.

SymPy, and the baselines that print through it, write two functions in names that not every SimpliPy engine reads:
the absolute value as ``Abs`` and the square root as ``sqrt``. An engine whose config declares them reads both as
written: ``Abs`` as an alias of ``abs``, and ``sqrt`` as a defined function (the config's ``defined_functions:``), which
the engine prices at its own head and evaluates as ``rootn(u, 2)``. An engine without these entries reads such a name
as a bare token, which every later walk (pricing, simplifying, judging) rejects: the prediction then has no description
length and can never match its ground truth. For such an engine srbf writes the absolute value ``abs`` and the square
root ``rootn(u, 2)``, the engine's own operators for them.

The candidate judge and the subprocess adapter (:func:`srbf.candidates.read_spelled`), the in-process E2E and NeSymReS
adapters, and the table for the files written before they did, go through this module.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from symbolic_data.token_ops import desugar_sqrt

# names SymPy prints that an engine may spell otherwise (the square root is written by spell_square_roots)
TOKEN_SPELLINGS = {"Abs": "abs"}
SYMPY_ONLY = frozenset(TOKEN_SPELLINGS) | {"sqrt"}


def written_arity(engine: Any) -> dict[str, int]:
    """The heads ``engine`` reads in an expression as written, with their arities: its operators and the functions its
    config defines (``defined_functions:``, such as ``sqrt``), which it prices and evaluates as written."""
    defined = getattr(engine, "defined_functions", None) or {}
    return {**engine.operator_arity, **{str(name): int(spec["arity"]) for name, spec in defined.items()}}


def spell_square_roots(prefix: Sequence[str], operator_arity: Mapping[str, int]) -> list[str]:
    """``prefix`` with every ``sqrt u`` written ``rootn u 2``, unless the engine defines ``sqrt``.

    ``operator_arity`` is the engine's :func:`written_arity`. An engine that defines ``sqrt`` reads, prices and
    evaluates the head as written, and the prefix comes back as it is. An engine without the definition has no such
    head, and the root is written with the operator it has. Raises ``ValueError`` when the walk cannot place an
    argument (a function outside ``operator_arity``)."""
    tokens = [str(t) for t in prefix]
    if "sqrt" in operator_arity or "sqrt" not in tokens:
        return tokens
    return desugar_sqrt(tokens, dict(operator_arity))


def engine_spelling(prefix: Sequence[str], operator_arity: Mapping[str, int]) -> list[str]:
    """``prefix`` with SymPy's spellings replaced by the engine's: ``Abs u`` -> ``abs u``, and ``sqrt u`` ->
    ``rootn u 2`` when the engine does not define ``sqrt`` (:func:`spell_square_roots`). ``operator_arity`` is the
    engine's :func:`written_arity`.

    A prediction that also carries a function outside the vocabulary (SymPy's ``conjugate``, say) cannot be walked, so
    its square roots stay as written: it stays unreadable, as it was, and never fails the file it is judged in."""
    out = [TOKEN_SPELLINGS.get(str(t), str(t)) for t in prefix]
    try:
        return spell_square_roots(out, operator_arity)
    except ValueError:
        return out
