"""Map a baseline's own variable spelling onto the one the ground truth uses.

E2E names the columns it is handed ``x_0, x_1, ...`` and NeSymReS names them ``x_1, x_2, ...``; the
ground-truth skeleton names the same columns ``x1, x2, ...`` in column order (a problem's
``variables`` are often catalog names such as ``v1``, so they cannot be matched by name). Without
this map a prediction that IS the law shares no variable with it, and every symbolic comparison is
structurally impossible: symbolic recovery, the raw skeleton match, the token F1 and the
variable-set precision / recall / F1 all read zero. Both adapters hand the model a known block of
columns in a known order, so the map is positional and exact.

An index the model made up (a padding column, say) has no column to map onto and is left as it is:
a prediction that reaches for a variable it was never given is a miss, not a renaming problem.
"""
from __future__ import annotations

import re
from collections.abc import Sequence

X_NAME = re.compile(r"x\d+")
MODEL_TOKEN = re.compile(r"x_(\d+)")
# The first column's index in each baseline's own spelling.
E2E_FIRST_INDEX = 0
NESYMRES_FIRST_INDEX = 1


def skeleton_variable_names(columns: Sequence[str] | None) -> list[str]:
    """The ground truth's name for every column handed to the model, in column order.

    A column that already carries an ``x<n>`` name keeps it (that is the skeleton's own spelling,
    e.g. after the unused columns were dropped); anything else is named by its position, because
    the skeleton spells the i-th variable of a problem ``x<i+1>`` whatever the catalog calls it.
    """
    return [str(c) if X_NAME.fullmatch(str(c)) else f"x{i + 1}" for i, c in enumerate(columns or [])]


def rename_variable_tokens(tokens: Sequence[str] | None, names: Sequence[str], *, first_index: int) -> list[str] | None:
    """Rename the ``x_<i>`` tokens of a prefix expression to the ground truth's names."""
    if tokens is None:
        return None
    out: list[str] = []
    for token in tokens:
        m = MODEL_TOKEN.fullmatch(str(token))
        column = int(m.group(1)) - first_index if m else -1
        out.append(names[column] if 0 <= column < len(names) else str(token))
    return out


def rename_variables_in_infix(expression: str, names: Sequence[str], *, first_index: int) -> str:
    """The same map on an infix string, so the stored expression and its prefix agree.

    The word boundary keeps an identifier that merely ends in one of these names (``mulx_0``) intact.
    """
    def sub(m: re.Match[str]) -> str:
        column = int(m.group(1)) - first_index
        return names[column] if 0 <= column < len(names) else m.group(0)

    return re.sub(r"\bx_(\d+)\b", sub, expression)
