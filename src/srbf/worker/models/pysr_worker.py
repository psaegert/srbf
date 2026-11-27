"""The PySR worker: PySRRegressor at its upstream defaults over flash-ansr's 23-operator vocabulary.

Runs in whichever interpreter has ``pysr`` (and its Julia backend); imports pysr and numpy only.
``options`` (from the config's ``model_adapter`` block):

* ``timeout_in_seconds`` (int, 60), ``niterations`` (int, 100): the search budget, the compute
  axis of the scaling sweeps.
* ``maxsize``, ``parsimony`` (None = PySR's own defaults; see docs/fairness.md), ``model_selection`` ('best' = PySR's default).
* ``warmup`` (True): pay the one-off Julia compilation on a throwaway fit in ``setup``, before the
  clock, so the first timed fit starts warm. It is skipped when srbf's own warm-up problem follows
  (``runner.warmup: true``): that is a fit at the run's own options, which compiles the same code.

The result carries PySR's own predictions (``y_pred``/``y_pred_val`` from ``predict``), the best
equation as ``expression`` (in the variable names srbf handed over), PySR's hall of fame (its
Pareto front of loss against complexity, ``equations_``) as ``candidates``, which srbf judges and
stores, and the hall of fame with PySR's own measures under ``extra["equations"]``.
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
# `abs(n) < 2^53`: every float beyond that is "an integer", and `Int(abs(n))` overflows Int64 for
# |n| >= 2^63 (InexactError) -- a GP mutation or a seed can put 1e33 into the index slot.
ROOTN_JULIA = (
    r"rootn(x::T, n::T) where {T} = (isfinite(n) && n == round(n) && n != 0 && abs(n) < 9.007199254740992e15) ? "
    r"((x >= 0) ? abs(x)^(one(T)/n) : (isodd(Int(abs(n))) ? -abs(x)^(one(T)/n) : T(NaN))) : T(NaN)"
)
BINARY_OPERATORS = ["+", "-", "*", "/", "^", ROOTN_JULIA]


def create_model(*, timeout_in_seconds, niterations, maxsize=None, model_selection="best", parsimony=None,
                 guesses=None):
    """A PySRRegressor over the 23 operators the benchmark's expressions are written in: 17 unaries +
    {+, -, *, /, pow, rootn}. maxsize/parsimony are forwarded only when
    set; None = PySR's own (version-dependent) defaults, never hardcoded here."""
    PySRRegressor = _require_pysr()
    optional = {}
    if maxsize is not None:
        optional["maxsize"] = int(maxsize)
    if parsimony is not None:
        optional["parsimony"] = float(parsimony)
    if guesses:
        # initial guesses (PySR >= 2.0): the flash-ansr -> PySR seeding bridge hands Julia-syntax infix
        # strings in the fit's variable names; PySR parses them into its initial populations
        optional["guesses"] = [str(g) for g in guesses]
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
    """Pay SymbolicRegression.jl's compile cost OUTSIDE timing: a throwaway minimal fit, best-effort (a failing warmup
    forfeits only the warmup)."""
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
    return {"model": create_model(**_model_kwargs(options))}


def setup(state, options, *, warmup):
    """Before the clock: the throwaway fit (``options.warmup``), unless srbf's warm-up problem follows (``warmup``)."""
    if bool(options.get("warmup", True)) and not warmup:
        warmup_fit(options)


def info(state):
    try:
        import pysr
        version = getattr(pysr, "__version__", "?")
    except Exception:  # noqa: BLE001
        version = "?"
    return {"worker": "pysr", "pysr": version}


def fit(x, y, *, x_val, variables, meta, options, state):
    model = state["model"]
    # per-problem overrides ride in `meta` (a caller of the adapter may set them): seeds and the iteration
    # budget of THIS fit. A fresh regressor is built for them; the warm Julia session is shared.
    meta = meta or {}
    overrides = {k: meta[k] for k in ("guesses", "niterations", "timeout_in_seconds") if meta.get(k) is not None}
    if overrides:
        kwargs = _model_kwargs(options)
        kwargs.update(overrides)
        model = create_model(**kwargs)
    X = np.asarray(x, dtype=float)
    X_val = np.asarray(x_val, dtype=float).reshape(-1, X.shape[1]) if x_val else np.empty((0, X.shape[1]))
    target = np.asarray(y, dtype=float).ravel()
    model.fit(X, target, variable_names=list(variables))
    y_pred = np.asarray(model.predict(X), dtype=float).ravel()
    y_pred_val = np.asarray(model.predict(X_val), dtype=float).ravel() if X_val.shape[0] else []
    extra = {}
    candidates = None
    try:
        hof = model.equations_
        candidates = [str(equation) for equation in hof["equation"]]
        extra["equations"] = hof[["complexity", "loss", "score", "equation"]].to_dict("records")
    except Exception:  # noqa: BLE001 - persistence is best-effort
        pass
    if overrides:
        extra["n_guesses"] = len(overrides.get("guesses") or [])
        extra["niterations_used"] = int(overrides.get("niterations", options.get("niterations", 100)))
    best = model.get_best()
    return {"expression": str(best["equation"]), "candidates": candidates, "y_pred": y_pred, "y_pred_val": y_pred_val,
            "extra": extra}
