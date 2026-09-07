"""Helpers a worker script may import inside ITS OWN interpreter (the runner puts this directory on
``sys.path``): standard library only, no srbf, no numpy.

* :func:`respell_legacy_prefix` -- the pre-0.12 simplipy hyper-operator vocabulary
  (``mult2..5``, ``div2..5``, ``pow2..5``, ``pow1_2..pow1_5``) respelled into the current one, a
  pure token rewrite with no simplification (the table of simplipy's
  ``benchmarks/corpus/convert_nv.py``).
* :func:`prefix_to_infix` -- a prefix token list rendered as the plain infix srbf's engine reads.
* :func:`substitute_placeholders` -- ``<constant>`` placeholders replaced by fitted values.
"""

BINARY = {"+", "-", "*", "/", "pow", "rootn"}
UNARY = {"abs", "inv", "neg", "sqrt", "cbrt", "sin", "cos", "tan", "asin", "acos", "atan",
         "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "exp", "log"}
LEGACY_MULT = {"mult%d" % k: str(k) for k in (2, 3, 4, 5)}
LEGACY_DIV = {"div%d" % k: str(k) for k in (2, 3, 4, 5)}
LEGACY_POW = {"pow%d" % k: str(k) for k in (2, 3, 4, 5)}
LEGACY_ROOT = {"pow1_%d" % k: str(k) for k in (2, 3, 4, 5)}
LEGACY_TOKENS = set(LEGACY_MULT) | set(LEGACY_DIV) | set(LEGACY_POW) | set(LEGACY_ROOT)


def _arity(token):
    if token in BINARY:
        return 2
    if token in UNARY or token in LEGACY_TOKENS:
        return 1
    return 0


def _respell(tokens, i):
    if i >= len(tokens):
        raise ValueError("prefix expression ends inside an operator's arguments")
    t = tokens[i]
    if t in LEGACY_MULT:
        sub, j = _respell(tokens, i + 1)
        return ["*", LEGACY_MULT[t]] + sub, j
    if t in LEGACY_DIV:
        sub, j = _respell(tokens, i + 1)
        return ["/"] + sub + [LEGACY_DIV[t]], j
    if t in LEGACY_POW:
        sub, j = _respell(tokens, i + 1)
        return ["pow"] + sub + [LEGACY_POW[t]], j
    if t in LEGACY_ROOT:
        sub, j = _respell(tokens, i + 1)
        return ["rootn"] + sub + [LEGACY_ROOT[t]], j
    arity = _arity(t)
    out = [t]
    j = i + 1
    for _ in range(arity):
        sub, j = _respell(tokens, j)
        out.extend(sub)
    return out, j


def respell_legacy_prefix(tokens):
    """Rewrite a prefix token list from the retired hyper-operator vocabulary into the current one.

    ``mult_k t -> * k t``, ``div_k t -> / t k``, ``pow_k t -> pow t k``, ``pow1_k t -> rootn t k``.
    Every other token passes through; unknown tokens are leaves (variables, numbers, ``<constant>``).
    """
    out, j = _respell(list(tokens), 0)
    if j != len(tokens):
        raise ValueError("trailing tokens after a complete prefix expression: %r" % (list(tokens)[j:],))
    return out


def _to_infix(tokens, i):
    if i >= len(tokens):
        raise ValueError("prefix expression ends inside an operator's arguments")
    t = tokens[i]
    if t in LEGACY_TOKENS:
        raise ValueError("legacy token %r: respell with respell_legacy_prefix() first" % t)
    if t in BINARY:
        a, j = _to_infix(tokens, i + 1)
        b, j = _to_infix(tokens, j)
        if t == "pow":
            return "(%s)**(%s)" % (a, b), j
        if t == "rootn":
            return "rootn(%s, %s)" % (a, b), j
        return "(%s %s %s)" % (a, t, b), j
    if t in UNARY:
        a, j = _to_infix(tokens, i + 1)
        return "%s(%s)" % (t, a), j
    return str(t), i + 1


def prefix_to_infix(tokens):
    """Render a prefix token list (current vocabulary) as plain infix: ``+ * 2 x1 pow x2 3`` ->
    ``((2 * x1) + (x2)**(3))``."""
    text, j = _to_infix(list(tokens), 0)
    if j != len(tokens):
        raise ValueError("trailing tokens after a complete prefix expression: %r" % (list(tokens)[j:],))
    return text


def substitute_placeholders(tokens, values, placeholder="<constant>"):
    """Replace each ``placeholder`` token, left to right, by the corresponding value spelled as a
    number (``repr`` of a float, so nothing is lost)."""
    values = list(values)
    out = []
    k = 0
    for t in tokens:
        if t == placeholder:
            if k >= len(values):
                raise ValueError("more placeholders than values (%d)" % len(values))
            out.append(repr(float(values[k])))
            k += 1
        else:
            out.append(t)
    if k != len(values):
        raise ValueError("%d values for %d placeholders" % (len(values), k))
    return out
