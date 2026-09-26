"""The engine's spelling of what a method printed.

SymPy, and the baselines that print through it, spell two functions differently from the SimpliPy engine: the
absolute value as ``Abs`` and the square root as ``sqrt``. The engine's reader is lenient, so such a name stays a bare
token, which every later walk (pricing, simplifying, judging) rejects: the prediction then has no description length
and can never match its ground truth. The subprocess adapter spells its workers' answers the engine's way; the
in-process E2E and NeSymReS adapters, and the table for the files written before they did, go through this function.
"""
from __future__ import annotations

from typing import Mapping, Sequence

from symbolic_data.token_ops import desugar_sqrt

# names SymPy prints that the engine spells otherwise (the square root is rewritten by desugar_sqrt)
TOKEN_SPELLINGS = {"Abs": "abs"}
SYMPY_ONLY = frozenset(TOKEN_SPELLINGS) | {"sqrt"}


def engine_spelling(prefix: Sequence[str], operator_arity: Mapping[str, int]) -> list[str]:
    """``prefix`` with SymPy's spellings replaced by the engine's: ``Abs u`` -> ``abs u``, ``sqrt u`` -> ``rootn u 2``."""
    out = [TOKEN_SPELLINGS.get(str(t), str(t)) for t in prefix]
    return desugar_sqrt(out, dict(operator_arity)) if "sqrt" in out else out
