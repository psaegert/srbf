"""The GP-GOMEA worker: the original GP-GOMEA in its first author's benchmark configuration.

Runs in the environment ``scripts/envs/build_gpgomea_env.sh`` builds: GP-GOMEA (Virgolin, Alderliesten, Witteveen
and Bosman, Evolutionary Computation 29(2), 2021; github.com/marcovirgolin/GP-GOMEA, Apache-2.0) at commit 6a92cb6,
the commit SRBench 2021 ran, through its Python bindings (``pyGPGOMEA``). Imports those and numpy only, and runs
on Python 3.8.

The C++ source is compiled unchanged. The build script adjusts only the build system: it names the Boost.Python
and Boost.NumPy libraries for the environment's Python and adds the environment's include and library directories
(both as SRBench 2021's install script does), writes the library directory into the module's run path, and pins
the compiler's sysroot to glibc 2.17, whose libraries the linker of gcc 11.2 can read.

The configuration is the one GP-GOMEA's first author committed for running it as a benchmark baseline (SRBench
2021, cavalab/srbench commit 71ae717, experiment/methods/GPGOMEARegressor.py) without the harness's hyperparameter
grid: GP-GOMEA with the linkage-tree FOS, linear scaling, ephemeral random constants, the interleaved multistart
scheme off, one thread; everything else at the wrapper's defaults (population 500, initial tree height 4, elitism
1). srbf sets only what it sets for every method: the operators (the benchmark's operators as far as GP-GOMEA has
them), the budget (a count of evaluations) and the seed. The wall-clock limit is SRBench's 7,200 s, a guard that
the ladder never reaches. See docs/models.md.

``options`` (from the config's ``model_adapter`` block):

* ``max_evaluations`` (int, 500,000, the author's value): the budget, the compute axis of the scaling sweeps.
  GP-GOMEA counts one evaluation per fitness evaluation of a whole tree, the initial population's included, and
  checks the budget between generations, so a run overshoots it by up to one generation.
* ``seed`` (int, 0): mixed with a hash of the problem's data into the run's seed, so a problem's fit is
  reproducible and two draws of a law (fresh data) get different seeds.
* ``config`` (dict, none): settings that replace the author configuration's, ``functions`` included. For side
  experiments only, such as reproducing a published configuration; a config that sets it is ``harness_tuned``,
  and published results never set it.

What the returned expression states, and why it is not GP-GOMEA's own string:

* GP-GOMEA's operators ``p/``, ``plog`` and ``sqrt`` are protected, and their printed names hide it. Each is
  written as the function the method computes, in the benchmark's operators, so srbf evaluates what GP-GOMEA
  evaluated:

  - ``sqrt(u)`` is ``sqrt(|u|)``: written ``sqrt(abs(u))``.
  - ``plog(u)`` is ``log(|u|)``, with 0 where that is not finite: written ``log(abs(u))``, or ``0`` when ``u`` is
    zero (or ``log(|u|)`` not finite) at every support point.
  - ``p/(u, v)`` is ``sign(v) * u / (|v| + 1e-6)``, with ``sign(0) = +1``: written ``u/(v + 1e-06)`` when
    ``v >= 0`` at every support point (``v`` identically zero included), ``u/(v - 1e-06)`` when ``v < 0`` at
    every support point, and ``u/(v*(1 + 1e-06/abs(v)))`` otherwise.

  Each spelling is the method's function at every support point. The last two are also its function wherever
  ``v`` is not zero; the first wherever ``v`` is not negative.
* ``(u)^2`` is written ``(u)**2``; ``aq`` (not in the default operators) ``u/sqrt(1 + v**2)``.
* GP-GOMEA prints the linear-scaling intercept and slope with six decimals (``std::to_string``). The worker
  computes them again in float64 with the method's own least-squares step (``Utils::ComputeLinearScalingTerms``,
  the step behind its fitness and its ``predict``), on the support points and the model's output, which restores
  what the method computed. The random constants inside the tree are rounded to 1e-3 by the method itself when
  they are drawn; their printed values are exact and are kept.
* Variables are renamed from GP-GOMEA's ``x0, x1, ...`` to the names srbf hands over.

Every fit runs in a child process forked for it, as it would in a fresh interpreter: the C library's ``rand()``,
which GP-GOMEA uses to shuffle and which its ``--seed`` leaves alone, starts from its initial state in every fit
(``srand(1)``), so a fit does not depend on the fits before it; the pinned commit's memory leak across repeated fits
stays in the child; and a crash of the C++ code costs one problem, not the worker. The child dies with its parent.
The returned ``fit_time`` is GP-GOMEA's own fit, measured in the child; forking and rewriting its string are not
part of it. GP-GOMEA's own predictions are not returned (they come from the same tree): srbf evaluates the string.
"""
import ctypes
import hashlib
import math
import multiprocessing
import os
import signal
import time
import traceback

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")   # one thread, and a single-threaded process to fork each fit from

import numpy as np  # noqa: E402 - after the thread settings, which numpy reads when it loads

COMMIT = "6a92cb671c2772002b60df621a513d8b4df57887"

# The benchmark's operators GP-GOMEA has, as its own function names: + - * and exp sin cos natively; division,
# log and square root only as the protected p/, plog and sqrt (see the module docstring); pow for the exponent 2
# as ^2. GP-GOMEA has no node for general pow or rootn, abs, neg, inv, tan, or the inverse and hyperbolic
# functions; neg and inv it expresses through - and p/.
FUNCTIONS = "+_-_*_p/_plog_sqrt_exp_sin_cos_^2"

# SRBench's wall-clock limit for this method, in seconds. GP-GOMEA checks it between generations. A guard: the
# ladder's largest budget takes a small fraction of it.
TIME_GUARD = 7200

# The first author's benchmark configuration (SRBench 2021, cavalab/srbench commit 71ae717) on top of the
# wrapper's defaults at 6a92cb6 (pyGPGOMEA/GPGOMEARegressor.py), written out in full so that no default can
# change the method silently. The budget and the seed are set per fit.
AUTHOR_CONFIG = {
    "time": TIME_GUARD, "generations": -1,
    "prob": "symbreg", "linearscaling": True, "functions": FUNCTIONS, "erc": True, "classweights": False,
    "gomea": True, "gomfos": "LT",
    "subcross": 0.5, "submut": 0.5, "reproduction": 0.0, "sblibtype": False, "sbrdo": 0.0, "sbagx": 0.0,
    "unifdepthvar": True, "tournament": 4, "elitism": 1,
    "ims": False, "syntuniqinit": 1000, "popsize": 500,
    "initmaxtreeheight": 4, "maxtreeheight": 17, "maxsize": 1000,
    "parallel": False,   # one thread: with more, the evaluation count races and a run is irreproducible
    "caching": False, "silent": True,
}
SET_PER_FIT = ("evaluations", "seed")

# Seconds the worker waits beyond the guard for a fit's child before it stops it (the budget is checked only
# between generations, and a generation of a large configuration takes a while).
GRACE_SECONDS = 1800

GUARD = 1e-6                                  # p/'s guard (OpNewProtectedDivision.h)
BINARY = ("p/", "aq", "+", "-", "*")
UNARY = ("plog", "sqrt", "exp", "sin", "cos")
SQUARE = "^2"


def _require_gpgomea():
    try:
        from pyGPGOMEA import GPGOMEARegressor
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("pyGPGOMEA is required in the worker's environment: build it with "
                          "scripts/envs/build_gpgomea_env.sh") from exc
    return GPGOMEARegressor


def run_seed(x, y, seed=0):
    """A problem's seed: its data's hash mixed with the configured seed, in [0, 2**31)."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(x, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(y, dtype=np.float64).tobytes())
    h.update(str(int(seed)).encode())
    return int.from_bytes(h.digest()[:4], "little") & 0x7FFFFFFF


# -- reading GP-GOMEA's model string ---------------------------------------------------------------------------

class _Reader:
    """GP-GOMEA's printed tree (Node::GetSubtreeHumanExpression): every binary node is ``(A op B)``, every
    function ``name(A)``, a square ``(A)^2``, a variable ``xN`` and a constant ``std::to_string(value)``."""

    def __init__(self, text):
        self.text = text
        self.i = 0

    def fail(self, what):
        raise ValueError("cannot read GP-GOMEA's model at %d (%s): %r" % (self.i, what, self.text))

    def peek(self, token):
        return self.text.startswith(token, self.i)

    def expect(self, token):
        if not self.peek(token):
            self.fail("expected %r" % token)
        self.i += len(token)

    def number(self):
        start = self.i
        if self.peek("-"):
            self.i += 1
        for word in ("inf", "nan"):
            if self.peek(word):
                self.i += len(word)
                return float(self.text[start:self.i])
        while self.i < len(self.text) and (self.text[self.i].isdigit() or self.text[self.i] == "."):
            self.i += 1
        if self.i == start or self.text[start:self.i] == "-":
            self.fail("expected a number")
        return float(self.text[start:self.i])

    def operand(self):
        if self.peek("("):
            self.i += 1
            left = self.operand()
            if self.peek(")"):                     # (A)^2
                self.i += 1
                self.expect(SQUARE)
                return (SQUARE, left)
            for op in BINARY:
                if self.peek(op):
                    self.i += len(op)
                    right = self.operand()
                    self.expect(")")
                    return (op, left, right)
            self.fail("expected an operator")
        for name in UNARY:
            if self.peek(name + "("):
                self.i += len(name) + 1
                argument = self.operand()
                self.expect(")")
                return (name, argument)
        if self.peek("x") and self.i + 1 < len(self.text) and self.text[self.i + 1].isdigit():
            self.i += 1
            start = self.i
            while self.i < len(self.text) and self.text[self.i].isdigit():
                self.i += 1
            return ("var", int(self.text[start:self.i]))
        return ("const", self.number())

    def done(self):
        if self.i != len(self.text):
            self.fail("trailing text")


def read_model(text, linear_scaling=True):
    """GP-GOMEA's ``get_model()`` string as ``(a, b, tree)``: the printed intercept and slope (``None`` without
    linear scaling) and the tree of the model's inner expression."""
    reader = _Reader(text.strip())
    a = b = None
    if linear_scaling:
        a = reader.number()
        reader.expect("+")
        b = reader.number()
        reader.expect("*(")
        tree = reader.operand()
        reader.expect(")")
    else:
        tree = reader.operand()
    reader.done()
    return a, b, tree


# -- the method's function, and the expression that states it --------------------------------------------------

def _protected_division(u, v):
    sign = np.where(v < 0, -1.0, 1.0)            # OpNewProtectedDivision: sign(0) = +1
    return sign * (u / (GUARD + np.abs(v)))


def _protected_log(u):
    out = np.log(np.abs(u))                      # OpLog: non-finite results become 0
    out[~np.isfinite(out)] = 0.0
    return out


def _number(value):
    text = repr(float(value))
    return "(%s)" % text if text.startswith("-") else text


def render(tree, X, names):
    """The tree's values on ``X`` with GP-GOMEA's semantics and the expression that states them (module
    docstring), as ``(string, values)``."""
    kind = tree[0]
    if kind == "var":
        index = tree[1]
        if index >= len(names):
            raise ValueError("GP-GOMEA's model uses x%d, but the problem has %d variables" % (index, len(names)))
        return names[index], np.array(X[:, index], dtype=float)
    if kind == "const":
        if not math.isfinite(tree[1]):
            raise ValueError("GP-GOMEA's model holds a non-finite constant %r" % tree[1])
        return _number(tree[1]), np.full(X.shape[0], tree[1], dtype=float)
    parts = [render(child, X, names) for child in tree[1:]]
    with np.errstate(all="ignore"):
        if len(parts) == 2:
            (a, u), (b, v) = parts
            if kind == "+":
                return "(%s + %s)" % (a, b), u + v
            if kind == "-":
                return "(%s - %s)" % (a, b), u - v
            if kind == "*":
                return "(%s*%s)" % (a, b), u * v
            if kind == "p/":
                if np.all(v >= 0):
                    text = "(%s/(%s + 1e-06))" % (a, b)
                elif np.all(v < 0):
                    text = "(%s/(%s - 1e-06))" % (a, b)
                else:
                    text = "(%s/(%s*(1 + 1e-06/abs(%s))))" % (a, b, b)
                return text, _protected_division(u, v)
            if kind == "aq":
                return "(%s/sqrt(1 + %s**2))" % (a, b), u / np.sqrt(1.0 + v * v)
        else:
            ((a, u),) = parts
            if kind == "plog":
                values = _protected_log(u)
                raw = np.log(np.abs(u))
                return ("0" if not np.any(np.isfinite(raw)) else "log(abs(%s))" % a), values
            if kind == "sqrt":
                return "sqrt(abs(%s))" % a, np.sqrt(np.abs(u))
            if kind == "^2":
                return "(%s)**2" % a, u * u
            if kind in ("exp", "sin", "cos"):
                return "%s(%s)" % (kind, a), getattr(np, kind)(u)
    raise ValueError("unknown GP-GOMEA operator %r" % kind)


def linear_scaling_terms(P, Y):
    """Intercept and slope of the least-squares fit of ``Y`` on ``P`` (Utils::ComputeLinearScalingTerms)."""
    mean_y = float(np.mean(Y))
    mean_p = float(np.mean(P))
    var_p = P - mean_p
    denom = float(np.sum(var_p * var_p))
    if denom == 0:
        return mean_y, 0.0
    b = float(np.sum((Y - mean_y) * var_p)) / denom
    return mean_y - b * mean_p, b


def expression_from_model(text, X, Y, names, linear_scaling=True):
    """The expression srbf judges, from GP-GOMEA's ``get_model()`` string, as ``(expression, record, values)``:
    what the worker recorded on the way, and the model's values on ``X`` (the scaled model with the recomputed
    intercept and slope)."""
    printed_a, printed_b, tree = read_model(text, linear_scaling)
    inner, P = render(tree, X, names)
    record = {"printed_intercept": printed_a, "printed_slope": printed_b}
    if not np.all(np.isfinite(P)):
        raise ValueError("GP-GOMEA's model is not finite on the support points")
    if not linear_scaling:
        return inner, record, P
    if np.all(P == P[0]):
        # a constant inner expression: the method's scaled model is the mean of y (its slope multiplies 0)
        mean = float(np.mean(Y))
        record.update(intercept=mean, slope=0.0)
        return _number(mean), record, np.full_like(P, mean)
    a, b = linear_scaling_terms(P, Y)
    record.update(intercept=a, slope=b)
    return "%s + %s*%s" % (_number(a), _number(b), inner), record, a + b * P


# -- one fit, in a child process ---------------------------------------------------------------------------------

def _die_with_parent(parent):
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl(1, signal.SIGKILL)            # PR_SET_PDEATHSIG
    except Exception:  # noqa: BLE001 - best effort off Linux
        pass
    if os.getppid() != parent:
        os._exit(1)


def _reset_c_rand():
    """The C library's ``rand()`` to its initial state, as in a fresh process (GP-GOMEA shuffles with it)."""
    ctypes.CDLL(None).srand(ctypes.c_uint(1))


def _fit_in_child(connection, parent, X, Y, params):
    try:
        _die_with_parent(parent)
        _reset_c_rand()
        GPGOMEARegressor = _require_gpgomea()
        model = GPGOMEARegressor(**params)
        started = time.perf_counter()
        model.fit(X, Y)
        text = model.get_model()
        seconds = time.perf_counter() - started
        connection.send({"model": text, "seconds": seconds, "evaluations": int(model.get_evaluations()),
                         "nodes": int(model.get_n_nodes()), "predict": np.asarray(model.predict(X), dtype=float).ravel()})
    except BaseException:  # noqa: BLE001 - the parent reports it as this problem's error
        connection.send({"error": traceback.format_exc(limit=12)})
    finally:
        connection.close()


def run_method(X, Y, params, wait):
    """GP-GOMEA's fit in a child process forked for it; the child's report, or an ``error`` entry."""
    context = multiprocessing.get_context("fork")
    receiver, sender = context.Pipe(duplex=False)
    child = context.Process(target=_fit_in_child, args=(sender, os.getpid(), X, Y, params), daemon=True)
    child.start()
    sender.close()
    try:
        if not receiver.poll(wait):
            return {"error": "GP-GOMEA did not return within %d s" % wait}
        try:
            return receiver.recv()
        except EOFError:
            child.join(5)
            return {"error": "GP-GOMEA's process ended without a result (exit code %s)" % child.exitcode}
    finally:
        receiver.close()
        if child.is_alive():
            child.kill()
        child.join(5)


def build_params(max_evaluations, seed, config=None):
    config = dict(config or {})
    unknown = sorted(set(config) - set(AUTHOR_CONFIG))
    fixed = sorted(set(config) & set(SET_PER_FIT))
    if unknown or fixed:
        raise ValueError("config may replace %s; not %s" % (", ".join(sorted(AUTHOR_CONFIG)), ", ".join(unknown + fixed)))
    functions = str(config.get("functions", FUNCTIONS)).split("_")
    unstated = [f for f in functions if f not in BINARY + UNARY + (SQUARE,)]
    if unstated:
        raise ValueError("the worker cannot state GP-GOMEA's %s in the benchmark's operators" % ", ".join(unstated))
    if not 0 < int(max_evaluations) < 2 ** 31:
        raise ValueError("max_evaluations must be in [1, 2**31), got %r" % max_evaluations)
    return {**AUTHOR_CONFIG, **config, "evaluations": int(max_evaluations), "seed": int(seed)}


def load(options):
    state = {"max_evaluations": int(options.get("max_evaluations", 500_000)), "seed": int(options.get("seed", 0)),
             "config": dict(options.get("config") or {})}
    build_params(state["max_evaluations"], 0, state["config"])      # a bad option fails at start, not per problem
    _require_gpgomea()   # imported once here, so every fit's child inherits it (the import runs no C++ code)
    return state


def info(state):
    try:
        from importlib.metadata import version as _version
        version = _version("pyGPGOMEA")
    except Exception:  # noqa: BLE001
        version = "?"
    return {"worker": "gpgomea", "pyGPGOMEA": version, "gp_gomea_commit": COMMIT}


def fit(x, y, *, x_val, variables, meta, options, state):
    X = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    Y = np.ascontiguousarray(np.asarray(y, dtype=np.float64).ravel())
    seed = run_seed(X, Y, state["seed"])
    params = build_params(state["max_evaluations"], seed, state.get("config"))
    extra = {"seed": seed, "max_evaluations": state["max_evaluations"]}
    limit = int(params["time"])                  # seconds; not positive = no limit
    report = run_method(X, Y, params, wait=limit + GRACE_SECONDS if limit > 0 else None)
    if "error" in report:
        return {"error": report["error"], "extra": extra}
    # the wall-clock limit stopped the run when the budget was not reached
    extra.update(evaluations=report["evaluations"], nodes=report["nodes"], model_string=report["model"],
                 hit_time_guard=bool(limit > 0 and report["evaluations"] < params["evaluations"]))
    expression, record, values = expression_from_model(report["model"], X, Y, list(variables), bool(params["linearscaling"]))
    extra.update(record)
    # the recomputed model against GP-GOMEA's own predictions on the support points: an audit of the restoration
    scale = float(np.max(np.abs(report["predict"]))) or 1.0
    extra["predict_max_relative_deviation"] = float(np.max(np.abs(values - report["predict"]))) / scale
    return {"expression": expression, "fit_time": report["seconds"], "extra": extra}
