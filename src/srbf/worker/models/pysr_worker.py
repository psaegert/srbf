"""The PySR worker: PySRRegressor at its upstream defaults over flash-ansr's 23-operator vocabulary.

Runs in whichever interpreter has ``pysr`` (and its Julia backend); imports pysr and numpy only.
``options`` (from the config's ``model_adapter`` block):

* ``timeout_in_seconds`` (int, 60), ``niterations`` (int, 100): the search budget, the compute
  axis of the scaling sweeps.
* ``maxsize``, ``parsimony`` (None = PySR's own defaults; setting them is a panel knob, see
  docs/fairness.md), ``model_selection`` ('best' = PySR's default).
* ``warmup`` (True): pay the one-off Julia precompile on a throwaway model in ``load`` so the
  first timed fit starts warm.

The result carries PySR's own predictions (``y_pred``/``y_pred_val`` from ``predict``), the best
equation as ``expression`` (in the variable names srbf handed over) and the full Pareto hall of
fame under ``extra["equations"]``: plain columns only, so every selection rule stays an offline
re-score against the stored raw arrays.
"""
import numpy as np


def _require_pysr():
    try:
        from pysr import PySRRegressor
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("pysr is required in the worker's environment: pip install pysr") from exc
    return PySRRegressor


UNARY_OPERATORS = [
    "neg", "abs", "inv", "sin", "cos", "tan", "asin", "acos", "atan",
    "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "exp", "log",
]
# IEEE-754 rootn, matching simplipy.operators.rootn's table: odd integer index = signed root
# (total on R), even = principal (NaN on negatives), negative index = reciprocal (via the
# negative exponent), index 0 or non-integer = NaN.
ROOTN_JULIA = (
    r"rootn(x::T, n::T) where {T} = (isfinite(n) && n == round(n) && n != 0) ? "
    r"((x >= 0) ? abs(x)^(one(T)/n) : (isodd(Int(abs(n))) ? -abs(x)^(one(T)/n) : T(NaN))) : T(NaN)"
)
BINARY_OPERATORS = ["+", "-", "*", "/", "^", ROOTN_JULIA]


def create_model(*, timeout_in_seconds, niterations, maxsize=None, model_selection="best", parsimony=None):
    """A PySRRegressor over flash-ansr v24.0's 23-operator vocabulary, exactly (owner ruling
    2026-08-17): 17 unaries + {+, -, *, /, pow, rootn}. maxsize/parsimony are forwarded only when
    set; None = PySR's own (version-dependent) defaults, never hardcoded here."""
    PySRRegressor = _require_pysr()
    optional = {}
    if maxsize is not None:
        optional["maxsize"] = int(maxsize)
    if parsimony is not None:
        optional["parsimony"] = float(parsimony)
    return PySRRegressor(
        temp_equation_file=True,
        delete_tempfiles=True,
        timeout_in_seconds=int(timeout_in_seconds),
        niterations=int(niterations),
        model_selection=model_selection,
        unary_operators=list(UNARY_OPERATORS),
        binary_operators=list(BINARY_OPERATORS),
        # principal branch on export, the convention for fractional powers
        extra_sympy_mappings={"rootn": lambda x, n: x ** (1 / n)},
        **optional,
    )


def _model_kwargs(options):
    return dict(
        timeout_in_seconds=options.get("timeout_in_seconds", 60),
        niterations=options.get("niterations", 100),
        maxsize=options.get("maxsize"),
        model_selection=options.get("model_selection", "best"),
        parsimony=options.get("parsimony"),
    )


def warmup_fit(options):
    """Pay the Julia startup + SymbolicRegression.jl compile cost OUTSIDE timing: a throwaway
    minimal fit, best-effort (a failing warmup forfeits only the warmup)."""
    kwargs = _model_kwargs(options)
    kwargs["niterations"] = 1
    model = create_model(**kwargs)
    x = np.linspace(-1.0, 1.0, 32).reshape(-1, 1)
    y = 2.0 * x[:, 0] + 1.0
    try:
        model.fit(x, y, variable_names=["x0"])
    except Exception:  # noqa: BLE001 - never blocks evaluation
        pass


def load(options):
    model = create_model(**_model_kwargs(options))
    if bool(options.get("warmup", True)):
        warmup_fit(options)
    return {"model": model}


def info(state):
    try:
        import pysr
        version = getattr(pysr, "__version__", "?")
    except Exception:  # noqa: BLE001
        version = "?"
    return {"worker": "pysr", "pysr": version}


def fit(x, y, *, x_val, variables, meta, options, state):
    model = state["model"]
    X = np.asarray(x, dtype=float)
    X_val = np.asarray(x_val, dtype=float).reshape(-1, X.shape[1]) if x_val else np.empty((0, X.shape[1]))
    target = np.asarray(y, dtype=float).ravel()
    model.fit(X, target, variable_names=list(variables))
    y_pred = np.asarray(model.predict(X), dtype=float).ravel()
    y_pred_val = np.asarray(model.predict(X_val), dtype=float).ravel() if X_val.shape[0] else []
    extra = {}
    try:
        hof = model.equations_
        extra["equations"] = hof[["complexity", "loss", "score", "equation"]].to_dict("records")
    except Exception:  # noqa: BLE001 - persistence is best-effort
        pass
    best = model.get_best()
    return {"expression": str(best["equation"]), "y_pred": y_pred, "y_pred_val": y_pred_val, "extra": extra}
