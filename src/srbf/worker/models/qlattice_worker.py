"""The QLattice worker: Abzu's QLattice (the ``feyn`` package, 3.5.0) in its authors' benchmark configuration.

Runs in the environment ``scripts/envs/build_qlattice_env.sh`` builds (Python 3.12, feyn 3.5.0 from PyPI); imports
``feyn``, pandas and numpy only. feyn's core is closed (a compiled module, ``_qepler``); it runs locally, without a
licence key or network access. feyn is licensed CC BY-NC-ND 4.0 (research and other non-commercial use).

The configuration is the one QLattice's authors submitted to SRBench (the 2022 competition, unchanged in SRBench's
srbench_2025 branch, experiment/methods/qlattice/regressor.py): their own epoch loop (``auto_run_time``, feyn's
``auto_run`` with a wall-clock stop and without the stype inference that later feyn versions added to ``auto_run``),
200 epochs, at most 10 edges per model, models ranked by feyn's ``wide_parsimony`` criterion, squared error, every
input numerical. srbf sets only what it sets for every method: the operators (the benchmark's operators as far as
QLattice has them), the budget (a count of epochs), the seed and the threads. See docs/models.md.

One change to the authors' loop, a crash fix: feyn 3.5.0's ``validate_data`` indexes ``stypes`` and fails on the
``None`` the loop passes, which feyn 3.0.1 (the version the authors submitted with) accepted. The worker passes an
empty dict there, which states the same thing (no semantic types: every column numerical); every other call gets
``stypes`` as the authors pass it.

``options`` (from the config's ``model_adapter`` block):

* ``n_epochs`` (int, 200, the authors' value): the budget, the compute axis of the scaling sweeps. One epoch samples
  new models from the QLattice (about a thousand), fits every model in the pool on 20,000 resampled rows with
  feyn's own optimiser, prunes the pool and updates the QLattice with it.
* ``seed`` (int, 0): mixed with a hash of the problem's data into the run's seed (``feyn.QLattice(random_seed=...)``
  seeds Python's, numpy's and the core's generators), so a problem's fit is reproducible and two draws of a law
  (fresh data) get different seeds.
* ``threads`` (int or ``"all"``, 1): the threads feyn fits its models on (``fit_models(threads=...)``, the core's
  own thread pool). A resource, not a search setting. The authors' "auto" is the machine's cores minus one; ``"all"``
  is every CPU this process may run on (the benchmark's runs, owner 2026-09-30: every method gets the whole
  reference machine).
* ``max_time`` (int, 3600): the authors' wall-clock stop in seconds (SIGALRM, as in their loop; SRBench's limit).
  When it fires, the best models so far are returned and ``hit_time_guard`` is set. A guard: the ladder never
  reaches it.
* ``config`` (dict, none): settings that replace the configuration's, ``function_names`` included. For side
  experiments only, such as reproducing a published configuration; published results never set it.

The prediction is the model the method returns first, i.e. its best by its own ranking (the first of
``get_diverse_models``, which SRBench's wrapper predicts with), written from the model's parameters at full
precision (``repr``) in the variable names srbf hands over:

* every input enters through QLattice's affine input map, ``(x - offset) * scale * w + bias``, written ``a*x + c``;
  the output through its affine output map, ``(u * w + bias) * scale``, written ``A*u + B``; ``linear`` nodes as
  ``w*u + bias``; ``squared`` as ``u**2``, ``inverse`` as ``1/u``, ``gaussian`` (not in the default operators) as
  ``exp(-2*u**2)`` or ``exp(-2*u**2 - 2*v**2)``.
* QLattice's ``exp``, ``log``, ``sqrt``, ``squared`` and ``inverse`` are protected: the model clips their argument
  (``exp`` above 6, ``log`` below 1e-4, ``sqrt`` below 0.01, ``squared`` above 20, ``inverse`` to at least 0.05 in
  magnitude, with 0 mapped to -0.05). Where a clip is active at some support or validation point, the string
  writes the clip out with ``abs`` (``min(u, k) = k + (u - k - abs(u - k))/2``, ``max`` likewise), so that the
  string computes the model's own prediction at every point srbf evaluates; elsewhere the plain function is the
  model's function on those points and is written as it is. ``extra["protections"]`` lists the clips written out.
  An ``inverse`` whose argument takes both signs on those points is written ``abs(u)/(u*max(abs(u), 0.05))``,
  the model's function wherever ``u`` is not exactly 0.

``extra`` records the seed, the epochs run, the models tried, the answer's loss, criterion and edge count, the largest
deviation of the string from the model's own predictions on the support and validation points
(``string_deviation``, ``string_deviation_val``; relative to ``max(1, max |prediction|)``) and the other models the
method returned (``diverse``). The model's own predictions are not returned: srbf evaluates the string.
"""
import hashlib
import logging
import math
import os
import signal
import time

import numpy as np

# The benchmark's operators as QLattice has them (feyn's function names): + and * natively, pow as squared, rootn
# as sqrt, inv as inverse, and exp, log, tanh; linear is the affine node through which QLattice places constants
# inside a model. QLattice has no sin, cos or other trigonometric function, no general power or root, no abs and
# no neg or subtraction node (signed weights express them). gaussian, exp(-2 u^2) and exp(-2 u^2 - 2 v^2), is left
# out: a compound of the benchmark's operators, not one of them.
FUNCTIONS = ["add", "multiply", "squared", "sqrt", "inverse", "exp", "log", "tanh", "linear"]

# SRBench's wall-clock limit for this method, in seconds. The authors' loop stops at it (SIGALRM).
TIME_GUARD = 3600

# The authors' benchmark configuration (their SRBench submission: QLatticeRegressor's defaults and the settings of
# its ``est``; SRBench sets random_state and max_time), written out in full so that no default can change the
# method silently. n_epochs is the budget; the seed and max_time are set per fit.
AUTHOR_CONFIG = {
    "kind": "regression",
    "stypes": None,              # every input numerical
    "max_complexity": 10,        # edges of the model's graph
    "query_string": None,
    "loss_function": None,       # squared error for regression
    "criterion": "wide_parsimony",
    "sample_weights": None,
    "function_names": None,      # every function feyn has; srbf's operators replace it (FUNCTIONS)
    "starting_models": None,
    "threads": 1,                # the authors' "auto" is the cores minus one; the `threads` option (a resource) sets it
}
DEFAULT_EPOCHS = 200

# feyn's protected functions (feyn/_functions.py with the core's limits); read from the core in load().
LIMITS = {"EXP_MAX": 6.0, "DIVISOR_ABSMIN": 0.05, "LOG_MIN": 1e-4, "SQRT_MIN": 0.01, "SQUARED_MAX": 20.0}

OUTPUT = "y"


class TimeGuard(Exception):
    """The authors' wall-clock stop (their ``InternalTimeOutException``)."""


def _alarm(signum, frame):
    raise TimeGuard


def _require_feyn():
    try:
        logging.getLogger("feyn").setLevel(logging.WARNING)   # feyn logs every epoch at INFO
        import feyn
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("feyn is required in the worker's environment: build it with "
                          "scripts/envs/build_qlattice_env.sh") from exc
    return feyn


def run_seed(x, y, seed=0):
    """A problem's seed: its data's hash mixed with the configured seed, in [0, 2**31)."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(x, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(y, dtype=np.float64).tobytes())
    h.update(str(int(seed)).encode())
    return int.from_bytes(h.digest()[:4], "little") & 0x7FFFFFFF


def resolve_threads(value):
    """The ``threads`` option as a count: an int >= 1, or ``"all"`` = the CPUs this process may run on."""
    if value == "all":
        try:
            return len(os.sched_getaffinity(0))
        except AttributeError:   # no affinity call on this platform
            return os.cpu_count() or 1
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("threads must be an int >= 1 or 'all', got %r" % (value,))
    return value


def build_params(n_epochs, config=None, threads=1):
    config = dict(config or {})
    unknown = sorted(set(config) - set(AUTHOR_CONFIG))
    if unknown:
        raise ValueError("config may replace %s; not %s" % (", ".join(sorted(AUTHOR_CONFIG)), ", ".join(unknown)))
    if int(n_epochs) < 1:
        raise ValueError("n_epochs must be at least 1, got %r" % n_epochs)
    return {**AUTHOR_CONFIG, "function_names": list(FUNCTIONS), "threads": int(threads), **config,
            "n_epochs": int(n_epochs)}


# -- the authors' loop -------------------------------------------------------------------------------------------

def authors_loop(feyn, ql, data, output_name, *, kind, stypes, n_epochs, threads, max_complexity, query_string,
                 loss_function, criterion, sample_weights, function_names, starting_models, max_time):
    """SRBench's ``auto_run_time`` for QLattice, statement for statement, with the stypes fix (module docstring).
    Returns the models it returns and what the run did."""
    previous = None
    if max_time:
        previous = signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(int(max_time))
    try:
        feyn.validate_data(data, kind, output_name, stypes if stypes is not None else {})   # the fix

        if n_epochs <= 0:
            raise ValueError("n_epochs must be 1 or higher.")

        if threads == "auto":
            threads = feyn.tools.infer_available_threads()
        elif isinstance(threads, str):
            raise ValueError("threads must be a number, or string 'auto'.")

        models = []
        if starting_models is not None:
            models = [m.copy() for m in starting_models]
        m_count = len(models)

        priors = feyn.tools.estimate_priors(data, output_name)
        ql.update_priors(priors)

        epochs, hit_guard = 0, False
        try:
            for epoch in range(1, n_epochs + 1):
                new_sample = ql.sample_models(data, output_name, kind, stypes, max_complexity, query_string,
                                              function_names)
                models += new_sample
                m_count += len(new_sample)

                models = feyn.fit_models(models, data=data, loss_function=loss_function, criterion=criterion,
                                         n_samples=None, sample_weights=sample_weights, threads=threads)
                models = feyn.prune_models(models)
                ql.update(models)
                epochs = epoch

            best = feyn.get_diverse_models(models)

        except TimeGuard:
            hit_guard = True
            best = feyn.get_diverse_models(models)
    finally:
        if max_time:
            signal.alarm(0)       # srbf's process runs the next fit: the alarm must not outlive this one
            signal.signal(signal.SIGALRM, previous)
    return best, {"epochs": epochs, "models_tried": m_count, "hit_time_guard": hit_guard}


# -- the model as an expression ----------------------------------------------------------------------------------

def _number(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("the model holds a non-finite parameter %r" % value)
    return repr(value)


def _affine(a, text, c):
    """``a*text + c`` at full precision, parenthesized."""
    c = float(c)
    if c == 0.0:
        return "(%s*%s)" % (_number(a), text)
    return "(%s*%s %s %s)" % (_number(a), text, "-" if c < 0 else "+", _number(abs(c)))


def _minus(text, k):
    """``text - k`` for a number ``k``."""
    return "%s %s %s" % (text, "+" if k < 0 else "-", _number(abs(k)))


def _clip_below(text, k):
    """``max(text, k)`` in the benchmark's operators: ``k + (d + |d|)/2`` with ``d = text - k``, which is ``k``
    exactly wherever the clip is active."""
    d = _minus(text, k)
    return "(%s + (%s + abs(%s))/2)" % (_number(k), d, d)


def _clip_above(text, k):
    """``min(text, k)`` in the benchmark's operators: ``k + (d - |d|)/2`` with ``d = text - k``."""
    d = _minus(text, k)
    return "(%s + (%s - abs(%s))/2)" % (_number(k), d, d)


def _children(fnames, ix):
    arity = int(fnames[ix].split(":")[1])
    if arity == 0:
        return []
    first = ix + 1
    if arity == 1:
        return [first]
    end, open_slots = first, 1
    while open_slots:
        open_slots += int(fnames[end].split(":")[1]) - 1
        end += 1
    return [first, end]


def render(model, ix, X, columns, names, limits, protections):
    """The subtree at ``ix`` as ``(text, values)``: the expression in ``names`` and the model's values on ``X`` (rows
    of the columns ``columns``), computed as feyn's protected functions compute them."""
    fname = model.fnames[ix]
    params = model.params[ix]
    kids = [render(model, child, X, columns, names, limits, protections) for child in _children(model.fnames, ix)]
    with np.errstate(all="ignore"):
        if fname == "in-linear:0":
            j = columns.index(model.names[ix])
            a = float(params["scale"]) * float(params["w"])
            c = float(params["bias"]) - float(params["scale_offset"]) * a
            values = (X[:, j] - params["scale_offset"]) * params["scale"] * params["w"] + params["bias"]
            return _affine(a, names[j], c), values
        if fname == "out-linear:1":
            (u, U), = kids
            text = _affine(float(params["w"]) * float(params["scale"]), u, float(params["bias"]) * float(params["scale"]))
            return text[1:-1], (U * params["w"] + params["bias"]) * params["scale"]
        if fname == "linear:1":
            (u, U), = kids
            return _affine(params["w"], u, params["bias"]), U * params["w"] + params["bias"]
        if fname == "add:2":
            (u, U), (v, V) = kids
            return "(%s + %s)" % (u, v), U + V
        if fname == "multiply:2":
            (u, U), (v, V) = kids
            return "(%s*%s)" % (u, v), U * V
        if fname == "tanh:1":
            (u, U), = kids
            return "tanh(%s)" % u, np.tanh(U)
        if fname == "gaussian:1":
            (u, U), = kids
            return "exp(-2.0*%s**2)" % u, np.exp(-(U * U / 0.5))
        if fname == "gaussian:2":
            (u, U), (v, V) = kids
            return "exp(-2.0*%s**2 - 2.0*%s**2)" % (u, v), np.exp(-(U * U / 0.5 + V * V / 0.5))
        if fname == "exp:1":
            (u, U), = kids
            k = limits["EXP_MAX"]
            if np.any(U > k):
                protections.append("exp")
                return "exp(%s)" % _clip_above(u, k), np.exp(np.minimum(U, k))
            return "exp(%s)" % u, np.exp(U)
        if fname == "log:1":
            (u, U), = kids
            k = limits["LOG_MIN"]
            if np.any(U < k):
                protections.append("log")
                return "log(%s)" % _clip_below(u, k), np.log(np.maximum(U, k))
            return "log(%s)" % u, np.log(U)
        if fname == "sqrt:1":
            (u, U), = kids
            k = limits["SQRT_MIN"]
            if np.any(U < k):
                protections.append("sqrt")
                return "sqrt(%s)" % _clip_below(u, k), np.sqrt(np.maximum(U, k))
            return "sqrt(%s)" % u, np.sqrt(U)
        if fname == "squared:1":
            (u, U), = kids
            k = limits["SQUARED_MAX"]
            if np.any(U > k):
                protections.append("squared")
                clipped = np.minimum(U, k)
                return "(%s**2)" % _clip_above(u, k), clipped * clipped
            return "(%s**2)" % u, U * U
        if fname == "inverse:1":
            (u, U), = kids
            k = limits["DIVISOR_ABSMIN"]
            clipped = np.where(np.abs(U) < k, np.where(U > 0, k, -k), U)
            if not np.any(np.abs(U) < k):
                return "(1/%s)" % u, 1 / U
            protections.append("inverse")
            if np.all(U > 0):
                return "(1/%s)" % _clip_below(u, k), 1 / clipped
            if np.all(U <= 0):
                return "(1/%s)" % _clip_above(u, -k), 1 / clipped
            # both signs: sign(u) * max(|u|, k), which is the model's value wherever u is not exactly 0
            return "(abs(%s)/(%s*%s))" % (u, u, _clip_below("abs(%s)" % u, k)), 1 / clipped
    raise ValueError("the worker cannot state QLattice's %r" % fname)


_NAMESPACE = {"exp": np.exp, "log": np.log, "sqrt": np.sqrt, "tanh": np.tanh, "abs": np.abs}


def string_deviation(expression, n_columns, X, reference):
    """The largest deviation of an expression written in ``x0, x1, ...`` from the model's own values on X, relative
    to ``max(1, max |reference|)``; None when there is nothing to compare, inf when their finite points differ."""
    reference = np.asarray(reference, dtype=np.float64).ravel()
    if reference.size == 0:
        return None
    try:
        namespace = dict(_NAMESPACE)
        namespace.update({"x%d" % j: X[:, j] for j in range(n_columns)})
        with np.errstate(all="ignore"):
            values = np.broadcast_to(np.asarray(eval(expression, {"__builtins__": {}}, namespace), dtype=np.float64),
                                     reference.shape)
    except Exception:  # noqa: BLE001 - a diagnostic, never a failure
        return None
    both = np.isfinite(values) & np.isfinite(reference)
    if not np.array_equal(both, np.isfinite(reference)):
        return float("inf")
    if not both.any():
        return None
    scale = max(1.0, float(np.max(np.abs(reference[both]))))
    return float(np.max(np.abs(values[both] - reference[both])) / scale)


def model_expression(model, X, columns, names, limits):
    """The model as ``(expression in names, expression in x0.., values on X, protections written out)``."""
    protections = []
    text, values = render(model, 0, X, columns, names, limits, protections)
    internal, _ = render(model, 0, X, columns, ["x%d" % j for j in range(len(columns))], limits, [])
    return text, internal, values, sorted(set(protections))


# -- the worker --------------------------------------------------------------------------------------------------

def core_limits():
    """The limits of the protected functions, as feyn's core (``_qepler``) holds them."""
    import _qepler

    return {key: float(getattr(_qepler, key)) for key in LIMITS}


def load(options):
    state = {"n_epochs": int(options.get("n_epochs", DEFAULT_EPOCHS)), "seed": int(options.get("seed", 0)),
             "max_time": int(options.get("max_time", TIME_GUARD)), "config": dict(options.get("config") or {}),
             "threads": resolve_threads(options.get("threads", 1))}
    build_params(state["n_epochs"], state["config"], state["threads"])   # a bad option fails at start
    _require_feyn()
    state["limits"] = core_limits()
    return state


def info(state):
    try:
        feyn = _require_feyn()
        return {"worker": "qlattice", "feyn": feyn.__version__, "feyn_git_sha": getattr(feyn, "__git_sha__", "?")}
    except Exception:  # noqa: BLE001
        return {"worker": "qlattice", "feyn": "?"}


def fit(x, y, *, x_val, variables, meta, options, state):
    import pandas as pd

    feyn = _require_feyn()
    X = np.asarray(x, dtype=np.float64)
    Y = np.asarray(y, dtype=np.float64).ravel()
    XV = np.asarray(x_val, dtype=np.float64).reshape(-1, X.shape[1]) if len(x_val) else np.empty((0, X.shape[1]))
    names = list(variables)
    columns = ["x%d" % j for j in range(X.shape[1])]   # feyn's column names: no ':' and never the output's name
    seed = run_seed(X, Y, state["seed"])
    params = build_params(state["n_epochs"], state.get("config"), state.get("threads", 1))
    extra = {"seed": seed, "n_epochs": params["n_epochs"], "threads": params["threads"]}

    data = pd.DataFrame({**{column: X[:, j] for j, column in enumerate(columns)}, OUTPUT: Y})
    ql = feyn.QLattice(random_seed=seed)
    started = time.perf_counter()
    best, run = authors_loop(feyn, ql, data, OUTPUT, max_time=state["max_time"], **params)
    seconds = time.perf_counter() - started
    extra.update(run)
    if not best:
        return {"error": "QLattice returned no model (every fitted model had a non-finite loss)", "extra": extra,
                "fit_time": seconds}

    model = best[0]
    points = np.vstack([X, XV])
    expression, internal, _, protections = model_expression(model, points, columns, names, state["limits"])
    own = model.predict(pd.DataFrame({column: points[:, j] for j, column in enumerate(columns)}))
    n = X.shape[0]
    extra.update(loss=float(model.loss_value), wide_parsimony=float(model.wide_parsimony), bic=float(model.bic),
                 edge_count=int(model.edge_count), protections=protections,
                 string_deviation=string_deviation(internal, len(columns), X, own[:n]),
                 string_deviation_val=string_deviation(internal, len(columns), XV, own[n:]))
    diverse = []
    for other in best[1:]:
        try:
            text = model_expression(other, points, columns, names, state["limits"])[0]
        except Exception as exc:  # noqa: BLE001 - persistence is best-effort
            text = "unstated: %s" % exc
        diverse.append({"expression": text, "loss": float(other.loss_value),
                        "wide_parsimony": float(other.wide_parsimony), "edge_count": int(other.edge_count)})
    extra["diverse"] = diverse
    return {"expression": expression, "fit_time": seconds, "extra": extra}
