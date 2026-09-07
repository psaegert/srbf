"""The reference worker: ordinary least squares over the given variables, standard library only.

Fits ``y = a1*x1 + ... + an*xn + b`` by the normal equations and returns that line as the
expression. It exists to document the contract end to end (``load``, ``info``, ``fit``) and to
exercise the protocol in any interpreter, including a bare venv without numpy.
"""
import sys


def load(options):
    """Called once with the config's ``options``; the returned state reaches every ``fit``."""
    return {"ridge": float(options.get("ridge", 1e-9))}


def info(state):
    return {"worker": "example-ols", "python": sys.version.split()[0]}


def _solve(a, b):
    """Gaussian elimination with partial pivoting on the augmented system ``a x = b``."""
    n = len(a)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        m[col], m[pivot] = m[pivot], m[col]
        if abs(m[col][col]) < 1e-300:
            raise ValueError("singular normal equations")
        for r in range(col + 1, n):
            f = m[r][col] / m[col][col]
            for c in range(col, n + 1):
                m[r][c] -= f * m[col][c]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        s = m[r][n] - sum(m[r][c] * x[c] for c in range(r + 1, n))
        x[r] = s / m[r][r]
    return x


def fit(x, y, *, x_val, variables, meta, options, state):
    n_vars = len(x[0]) if x else 0
    if n_vars != len(variables):
        raise ValueError("%d columns but %d variable names" % (n_vars, len(variables)))
    rows = [list(map(float, r)) + [1.0] for r in x]
    y = [float(v) for v in y]
    k = n_vars + 1
    ata = [[sum(r[i] * r[j] for r in rows) for j in range(k)] for i in range(k)]
    for i in range(k):
        ata[i][i] += state["ridge"]
    aty = [sum(r[i] * yi for r, yi in zip(rows, y)) for i in range(k)]
    coef = _solve(ata, aty)
    terms = ["%r*%s" % (c, v) for c, v in zip(coef[:-1], variables)]
    expression = " + ".join(terms + [repr(coef[-1])])

    def predict(matrix):
        return [sum(c * float(v) for c, v in zip(coef[:-1], r)) + coef[-1] for r in matrix]

    return {
        "expression": expression,
        "y_pred": predict(x),
        "y_pred_val": predict(x_val),
        "constants": coef,
        "extra": {"n_support": len(rows)},
    }
