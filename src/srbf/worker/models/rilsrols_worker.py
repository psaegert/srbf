"""The RILS-ROLS worker: rils-rols 1.6.7 in its first author's benchmark configuration.

Runs in the environment ``scripts/envs/build_rilsrols_env.sh`` builds: RILS-ROLS (Kartelj and Djukanovic, Journal of
Big Data 10:71, 2023; github.com/kartelj/rils-rols, MIT), iterated local search over expression trees with the
linear coefficients fitted by ordinary least squares, release 1.6.7 (the one its authors submitted to SRBench),
Python 3.12. Imports ``rils_rols``, sympy and numpy only.

The configuration is the one RILS-ROLS's first author committed for running it as a benchmark baseline (his
SRBench submission, cavalab/srbench srbench_2025 commit bf17345, experiment/methods/rils-rols/regressor.py):
``max_complexity=50, sample_size=0`` (the sample size chosen by the method; on up to 10,000 points it takes them
all), the size penalty at its default 0.001, without the hyperparameter grid that the benchmark's maintainers
later searched around it. His ``verbose=True`` only prints progress and is off. srbf sets only what it sets for
every method: the budget, a count of fitness evaluations, and the seed; the wall-clock limit is his paper's hour, a
guard that the ladder never reaches. RILS-ROLS's operators are fixed in its C++ core and cannot be set: ``+ - * /``,
``sin cos exp`` and the natural ``log``, ``sqrt`` and the square, all among the benchmark's operators. See
docs/models.md.

``options`` (from the config's ``model_adapter`` block):

* ``max_fit_calls`` (int, 1,000,000, the author's value): the budget, the compute axis of the scaling sweeps.
  RILS-ROLS counts one call per fitness evaluation of a whole expression, those of its local search included, and
  stops at the first check after the budget is spent (one call over it).
* ``seed`` (int, 0): mixed with a hash of the problem's data into the run's ``random_state``, so a problem's fit is
  reproducible and two draws of a law (fresh data) get different seeds.
* ``config`` (dict, none): settings that replace the author configuration's. For side experiments only; a config
  that sets it is ``harness_tuned``, and published results never set it.

One patch, applied when the environment is built (``scripts/envs/patch_rilsrols.py``): RILS-ROLS prints the
constants of its model with six decimals (``std::to_string``), so that 6.674e-11 reads as 0. The patch prints the
final model's constants with every digit (``%.17g``). The search itself keys candidates by the same printed
strings, so the patch changes only the string of the final model, not what the search compares.

The answer is the method's own: the expression its search ends with, as its ``model_string()`` returns it (the
printed model after sympy's ``simplify`` with ratio 1, which the method gives two seconds and otherwise skips),
written in the benchmark's syntax with every constant at full precision (``repr`` of the double) and in the
variable names srbf handed over. When that simplified form holds a function the benchmark's operators cannot
state, the unsimplified form (what the method returns when it skips simplification) is written instead
(``answer_form``). RILS-ROLS evaluates without protected operators, so the string computes what the method
computed; ``extra`` records the largest deviation of the string from the method's own predictions on the support
points (``string_deviation``), the fitness evaluations used, the method's own model string, and whether the time
guard stopped the run.

Every fit runs in a child process forked for it: the method's two-second timeout for sympy leaves the thread that
simplifies running when it passes, the C++ core exits the process on some errors, and its caches grow with the
search, all of which stay in the child. The child dies with its parent. The returned ``fit_time`` is the method's
fit (search and simplification), measured in the child.
"""
import ctypes
import hashlib
import multiprocessing
import os
import signal
import time
import traceback

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")   # one thread, and a single-threaded process to fork each fit from

import numpy as np  # noqa: E402 - after the thread settings, which numpy reads when it loads

VERSION = "1.6.7"

# The wall-clock limit, in seconds: the hour of the author's paper (SRBench 2.0 ran the same). A guard; the
# ladder's largest budget takes a small fraction of it.
TIME_GUARD = 3600

# The first author's benchmark configuration (SRBench submission, cavalab/srbench bf17345) on top of the
# package's defaults, written out in full so that no default can change the method silently. The budget, the time
# guard and the seed are set per fit.
AUTHOR_CONFIG = {"max_complexity": 50, "sample_size": 0, "complexity_penalty": 0.001, "verbose": False}
SET_PER_FIT = ("max_fit_calls", "max_time", "random_state")

# Seconds the worker waits beyond the guard for a fit's child before it stops it.
GRACE_SECONDS = 600

# The functions a sympy expression may hold for the benchmark to read it (docs/adapters.md): RILS-ROLS's own are
# exp, log, sin and cos (sqrt and the square are powers); simplification can rewrite them into the others.
STATED_FUNCTIONS = {"exp", "log", "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "asinh",
                    "acosh", "atanh", "Abs"}

# The check fit of ``load``: a constant that six decimals print as 0.
_CHECK_CONSTANT = 6.674e-11


def _require_rilsrols():
    try:
        from rils_rols.rils_rols import RILSROLSRegressor
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("rils_rols is required in the worker's environment: build it with "
                          "scripts/envs/build_rilsrols_env.sh") from exc
    return RILSROLSRegressor


def run_seed(x, y, seed=0):
    """A problem's seed: its data's hash mixed with the configured seed, in [0, 2**31)."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(x, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(y, dtype=np.float64).tobytes())
    h.update(str(int(seed)).encode())
    return int.from_bytes(h.digest()[:4], "little") & 0x7FFFFFFF


def build_params(max_fit_calls, seed, config=None):
    config = dict(config or {})
    unknown = sorted(set(config) - set(AUTHOR_CONFIG))
    fixed = sorted(set(config) & set(SET_PER_FIT))
    if unknown or fixed:
        raise ValueError("config may replace %s; not %s" % (", ".join(sorted(AUTHOR_CONFIG)), ", ".join(unknown + fixed)))
    if not 0 < int(max_fit_calls) < 2 ** 31:
        raise ValueError("max_fit_calls must be in [1, 2**31), got %r" % max_fit_calls)
    return {**AUTHOR_CONFIG, **config, "max_fit_calls": int(max_fit_calls), "max_time": TIME_GUARD,
            "random_state": int(seed)}


# -- the answer in the benchmark's syntax -------------------------------------------------------------------------

def benchmark_string(expression, names):
    """A sympy expression of RILS-ROLS's model (variables ``x0, x1, ...``) in the benchmark's syntax: variables
    renamed to ``names``, every Float written as ``repr`` of its double. Raises ValueError for what the benchmark's
    operators cannot state."""
    import sympy
    from sympy.printing.str import StrPrinter

    class _Printer(StrPrinter):
        def _print_Float(self, expr):
            return repr(float(expr))

    symbols = {sympy.Symbol("x%d" % i): sympy.Symbol(name) for i, name in enumerate(names)}
    stray = sorted(str(s) for s in expression.free_symbols if s not in symbols)
    if stray:
        raise ValueError("RILS-ROLS's model uses %s, but the problem has %d variables" % (", ".join(stray), len(names)))
    unstated = sorted({type(f).__name__ for f in expression.atoms(sympy.Function)} - STATED_FUNCTIONS)
    if unstated:
        raise ValueError("the benchmark's operators cannot state %s" % ", ".join(unstated))
    if expression.has(sympy.I, sympy.oo, -sympy.oo, sympy.zoo, sympy.nan):
        raise ValueError("RILS-ROLS's model is not a finite real expression: %s" % expression)
    for number in expression.atoms(sympy.Float):
        if not np.isfinite(float(number)):
            raise ValueError("RILS-ROLS's model holds a non-finite constant %s" % number)
    return _Printer().doprint(expression.xreplace(symbols))


_NAMESPACE = {"sqrt": np.sqrt, "exp": np.exp, "log": np.log, "sin": np.sin, "cos": np.cos, "tan": np.tan,
              "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan, "sinh": np.sinh, "cosh": np.cosh,
              "tanh": np.tanh, "asinh": np.arcsinh, "acosh": np.arccosh, "atanh": np.arctanh, "Abs": np.abs,
              "E": np.e, "pi": np.pi}


def string_deviation(expression, names, X, reference):
    """The largest deviation of the written expression from RILS-ROLS's own values on X, relative to the values'
    scale (at least 1), or None when it cannot be evaluated here (a variable name that is no Python identifier)."""
    try:
        namespace = dict(_NAMESPACE)
        namespace.update({name: X[:, i] for i, name in enumerate(names)})
        with np.errstate(all="ignore"):
            values = np.broadcast_to(np.asarray(eval(expression, {"__builtins__": {}}, namespace), dtype=np.float64),
                                     (X.shape[0],))
        reference = np.asarray(reference, dtype=np.float64).ravel()
        both = np.isfinite(values) & np.isfinite(reference)
        if not np.array_equal(both, np.isfinite(reference)) or not both.any():
            return None if not both.any() else float("inf")
        scale = max(1.0, float(np.max(np.abs(reference[both]))))
        return float(np.max(np.abs(values[both] - reference[both])) / scale)
    except Exception:  # noqa: BLE001 - a diagnostic, never a failure
        return None


def answer(model, names):
    """The expression srbf judges from a fitted ``RILSROLSRegressor``, as ``(string, form)``: its simplified model
    (``model_string()``) or, when that cannot be stated, the model as printed (``model``)."""
    import sympy

    try:
        return benchmark_string(model.model_string(), names), "simplified"
    except ValueError as simplified_error:
        try:
            return benchmark_string(sympy.sympify(model.model), names), "unsimplified"
        except ValueError:
            raise simplified_error from None


# -- one fit, in a child process ---------------------------------------------------------------------------------

def _die_with_parent(parent):
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl(1, signal.SIGKILL)            # PR_SET_PDEATHSIG
    except Exception:  # noqa: BLE001 - best effort off Linux
        pass
    if os.getppid() != parent:
        os._exit(1)


def _fit_in_child(connection, parent, X, Y, names, params):
    try:
        _die_with_parent(parent)
        RILSROLSRegressor = _require_rilsrols()
        model = RILSROLSRegressor(**params)
        started = time.perf_counter()
        model.fit(X, Y)
        seconds = time.perf_counter() - started
        report = {"seconds": seconds, "model": str(model.model), "fit_calls": int(model.fit_calls),
                  "total_time": float(model.total_time), "best_time": float(model.best_time),
                  "predict": np.asarray(model.predict(X), dtype=np.float64).ravel()}
        try:
            report["expression"], report["answer_form"] = answer(model, names)
        except ValueError as exc:
            report["error"] = str(exc)
        connection.send(report)
    except BaseException:  # noqa: BLE001 - the parent reports it as this problem's error
        connection.send({"error": traceback.format_exc(limit=12)})
    finally:
        connection.close()


def run_method(X, Y, names, params, wait):
    """RILS-ROLS's fit in a child process forked for it; the child's report, or an ``error`` entry."""
    context = multiprocessing.get_context("fork")
    receiver, sender = context.Pipe(duplex=False)
    child = context.Process(target=_fit_in_child, args=(sender, os.getpid(), X, Y, list(names), params), daemon=True)
    child.start()
    sender.close()
    try:
        if not receiver.poll(wait):
            return {"error": "RILS-ROLS did not return within %d s" % wait}
        try:
            return receiver.recv()
        except EOFError:
            child.join(5)
            return {"error": "RILS-ROLS's process ended without a result (exit code %s)" % child.exitcode}
    finally:
        receiver.close()
        if child.is_alive():
            child.kill()
        child.join(5)


def check_full_precision():
    """Fit ``y = 6.674e-11 * x`` and require the constant back in full: the environment carries the patch."""
    X = np.linspace(1.0, 2.0, 32).reshape(-1, 1)
    report = run_method(X, _CHECK_CONSTANT * X[:, 0], ["v1"], build_params(200, 1), wait=120)
    if "error" in report:
        raise RuntimeError("the check fit failed: %s" % report["error"])
    import sympy

    constants = [float(c) for c in sympy.sympify(report["model"]).atoms(sympy.Float)]
    if not any(abs(c / _CHECK_CONSTANT - 1) < 1e-9 for c in constants):
        raise RuntimeError("rils-rols prints its constants rounded (%r): build the environment with "
                           "scripts/envs/build_rilsrols_env.sh, which applies the full-precision patch" % report["model"])


def load(options):
    state = {"max_fit_calls": int(options.get("max_fit_calls", 1_000_000)), "seed": int(options.get("seed", 0)),
             "config": dict(options.get("config") or {})}
    build_params(state["max_fit_calls"], 0, state["config"])      # a bad option fails at start, not per problem
    _require_rilsrols()   # imported once here, so every fit's child inherits it
    import sympy  # noqa: F401 - likewise
    check_full_precision()
    return state


def info(state):
    try:
        from importlib.metadata import version as _version
        version = _version("rils-rols")
        sympy_version = _version("sympy")
    except Exception:  # noqa: BLE001
        version = sympy_version = "?"
    return {"worker": "rilsrols", "rils-rols": version, "sympy": sympy_version, "full_precision_patch": True}


def fit(x, y, *, x_val, variables, meta, options, state):
    X = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    Y = np.ascontiguousarray(np.asarray(y, dtype=np.float64).ravel())
    names = list(variables)
    seed = run_seed(X, Y, state["seed"])
    params = build_params(state["max_fit_calls"], seed, state.get("config"))
    extra = {"seed": seed, "max_fit_calls": state["max_fit_calls"]}
    report = run_method(X, Y, names, params, wait=params["max_time"] + GRACE_SECONDS)
    for key in ("model", "fit_calls", "total_time", "best_time", "answer_form"):
        if key in report:
            extra["model_string" if key == "model" else key] = report[key]
    if "fit_calls" in report:
        extra["hit_time_guard"] = bool(report["fit_calls"] < params["max_fit_calls"])
    if "error" in report:
        return {"error": report["error"], "extra": extra}
    extra["string_deviation"] = string_deviation(report["expression"], names, X, report["predict"])
    return {"expression": report["expression"], "fit_time": report["seconds"], "extra": extra}
