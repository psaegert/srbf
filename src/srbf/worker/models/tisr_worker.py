"""The TiSR worker: TiSR (thermodynamics-informed symbolic regression) at its own defaults, returning its hall of fame.

Runs in the environment ``scripts/envs/build_tisr_env.sh`` builds: TiSR (Martinek, Frotscher, Richter and Herzog;
github.com/scoop-group/TiSR, Apache-2.0) at commit 9e628e6 of its main branch (2026-01-08), on Julia 1.12.7, with the
Julia packages of the environment's lock file, resolved from TiSR's own compatibility bounds; called through
juliacall. Imports juliacall and numpy only.

TiSR is an NSGA-II genetic programming search on islands that fits the constants of new expressions by
Levenberg-Marquardt (sometimes Nelder-Mead) and keeps a hall of fame: the Pareto front of fit error against
complexity. Every setting is TiSR's own default at the pinned commit: 20 islands of 50 expressions, at most 30 nodes,
residuals weighted by 1/|y|, constants fitted on half of the islands, recently seen expressions rejected with
probability 0.9, the hall of fame on (weighted squared error, weighted node count), one thread. srbf sets only what it
sets for every method:

* the operators: the benchmark's 23 (``+ - * / ^ rootn`` and ``neg abs inv sin cos tan asin acos atan sinh cosh
  tanh asinh acosh atanh exp log``). TiSR takes any Julia function as an operator; its evaluator guards log, sqrt,
  division and powers itself and stops the whole search when any other function throws, so the worker gives it
  ``asin acos acosh atanh`` as total functions (NaN outside their domain, as the benchmark evaluates them),
  ``neg`` as ``-x`` and ``rootn`` with the benchmark's semantics (an integer index, NaN otherwise);
* the budget, a count of generations (``n_gens``);
* the seed.

Output only: TiSR's progress printing and hall-of-fame display are off.

Every run spends its whole budget. TiSR's generational loop (src/main_loop.jl, "termination criteria") ends at the
first of four conditions, and the worker leaves only the count:

* ``n_gens`` generations done: the budget.
* ``t_lim`` seconds passed (TiSR's default 300 s): raised to ``time_guard``, a guard far above every rung of the
  ladders; ``hit_time_guard`` records a run it ended.
* ``callback`` returned true: TiSR's default callback never does, and a config cannot set another.
* the user typed ``q`` on standard input: srbf gives the worker none.

Within a generation each constant fit has its own limits, which end that fit and never the search, all at TiSR's
defaults: at most 10 Levenberg-Marquardt iterations (50 for Nelder-Mead), converged when 5 iterations improve the fit
by less than 1e-4 relative (``rel_f_tol_5_iter``), no time limit per fit (``fitting.t_lim`` infinite) and no early
stopping on held-out data (``early_stop_iter`` 0; a config cannot set it).

One answer per problem: OPEN. TiSR's search returns its hall of fame, its population, its progress and why it stopped
(src/main_loop.jl: ``return (hall_of_fame, population, prog_dict, stop_msg)``) and defines no single model. Its README
and example sort the hall of fame for the user to inspect (README: ``TiSR.convert_to_dataframe(hall_of_fame, ops,
sort_by=:max_are)``; example/example_main.jl: ``sort_by=:mare``), its export functions order their tables by the fit
objective (src/save_results.jl: ``sort_by=:ms_processed_e``), and its author's benchmark counts a run as a success when
any hall-of-fame member matches an acceptable form (FastSRB-paper-repo src/tisr.jl and FastSRB example/TiSR.jl: the
callback's loop over the hall of fame). Until a rule is chosen the worker returns no expression: every problem is
recorded with the error ``NO_ANSWER`` and carries the whole hall of fame in ``hall_of_fame``, in TiSR's export order
(lowest ``ms_processed_e`` first), each member with its expression, TiSR's measures and its ``string_deviation``.

TiSR's runs are not reproducible from the seed: two fits of the same data with the same seed can return different
halls of fame. TiSR picks parents by rank and crowding (src/selection.jl, ``parent_selection``), and an expression
that has not been through a population selection yet, as all of an island's first ones, carries rank and crowding
values that were never set (src/individual.jl, ``Individual(node::Node) = new(node)``), so the search reads whatever
that memory held. TiSR is used as it is; the worker does not patch it.

``options`` (from the config's ``model_adapter`` block):

* ``generations`` (int, 512): the budget, TiSR's ``n_gens``, the compute axis of the scaling sweeps. A generation
  breeds, fits and evaluates about 50 new expressions on each of the 20 islands.
* ``seed`` (int, 0): mixed with a hash of the problem's data into the seed TiSR's random number generator starts
  from (it does not make a run reproducible, see above).
* ``time_guard`` (float, 3,600): TiSR's ``t_lim`` in seconds, the guard.
* ``warmup`` (bool, true): a small throwaway fit in :func:`load`, so that compiling TiSR is no part of the first
  problem's time.
* ``config`` (dict, none): TiSR settings that replace its defaults, for side experiments only, such as reproducing
  a published protocol; a config that sets it is ``harness_tuned``, and published results never set it. Keys:
  ``binops`` and ``unaops`` (lists of operator names, see ``OPERATORS``), and the sections ``data_split``,
  ``general``, ``measures``, ``selection``, ``fitting``, ``mutation`` and ``grammar``, each a map from a keyword of
  TiSR's ``<section>_params`` to a value. A string value is Julia source (``'[:ms_processed_e, :compl]'``); numbers
  and booleans keep their YAML type (write ``1800.0`` where TiSR expects a float). No stop can be set this way.
* ``config_by_problem`` (dict, none): per-problem values of settings that ``config`` sets (its value holds for the
  problems not listed), keyed by the problem's ``benchmark_eq_id``, numbers or booleans only; for the same side
  experiments. They are arguments of the one compiled constructor, so a problem's own value costs no compilation.

Each member is written from TiSR's expression tree in the benchmark's syntax and the names srbf handed over, every
constant at full precision (TiSR's own printer rounds to 15 digits), and its string is evaluated against TiSR's own
values of the member on the data (``string_deviation``, relative to the values' scale). ``extra`` also records the
seed, the budget and the generations run, why TiSR stopped, the generation TiSR had reached at its progress
checkpoints (about every 5 s: ``progress``) and the worker's CPU time for the search (``cpu_seconds``; on a busy
machine it is closer than the wall clock to what an idle one takes); ``fit_time`` is TiSR's own time for the search.
"""
import hashlib
import math
import os
import sys
import time

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):  # before numpy: one thread
    os.environ.setdefault(_variable, "1")

import numpy as np  # noqa: E402

COMMIT = "9e628e68ee0b0e05b3736b3f25321e502ea03d6f"   # TiSR main, 2026-01-08
TREE = "d7f805458ee067ff63405c655490de0870ac17fe"     # the git tree of COMMIT, as the environment's Manifest pins it
TISR_UUID = "e1088136-a916-4db6-8e31-3a049800401f"
JULIA_VERSION = "1.12.7"

# What every problem records until a rule for picking one hall-of-fame member is chosen (see the module docstring).
NO_ANSWER = ("TiSR returns a hall of fame and defines no single answer; no selection rule is set, so the whole hall of "
             "fame is in the hall_of_fame column")

BINARY_OPERATORS = ("+", "-", "*", "/", "^", "rootn")
UNARY_OPERATORS = ("neg", "abs", "inv", "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh",
                   "asinh", "acosh", "atanh", "exp", "log")

# Every operator the worker can hand TiSR: (arity, how the benchmark writes it). The Julia functions behind the names
# are Base's, except for those JULIA_OPERATORS defines. sqrt, pow2 and pow3 are not among the benchmark's operators;
# they are here for published protocols that use them (TiSR's paper), and are written in the benchmark's syntax.
OPERATORS = {
    "+": (2, "({} + {})"), "-": (2, "({} - {})"), "*": (2, "({}*{})"), "/": (2, "({}/{})"),
    "^": (2, "(({})**({}))"), "rootn": (2, "rootn({}, {})"),
    "pow2": (1, "(({})**2)"), "pow3": (1, "(({})**3)"),
}
OPERATORS.update({name: (1, name + "({})") for name in UNARY_OPERATORS + ("sqrt",)})

# TiSR's evaluator refuses log and sqrt below zero, division by zero and powers of negative bases before it calls
# them (eval_equation); a function that throws on any other argument would stop the whole search. The benchmark
# evaluates these in NumPy, which returns NaN outside the domain; so do these (TiSR then drops the expression, as it
# drops every expression that is not finite on the data). They take ForwardDiff's dual numbers, which TiSR's
# Levenberg-Marquardt step passes, and mixed argument types (a variable's column next to a fitted constant).
JULIA_OPERATORS = r"""
neg(x) = -x
pow2(x) = x^2
pow3(x) = x^3
asin(x) = abs(x) <= one(x) ? Base.asin(x) : oftype(x, NaN)
acos(x) = abs(x) <= one(x) ? Base.acos(x) : oftype(x, NaN)
acosh(x) = x >= one(x) ? Base.acosh(x) : oftype(x, NaN)
atanh(x) = abs(x) <= one(x) ? Base.atanh(x) : oftype(x, NaN)
rootn(x::Real, n::Real) = _rootn(promote(x, n)...)
function _rootn(x::T, n::T) where {T}
    k = TiSR.ForwardDiff.value(n)
    (isfinite(k) && k == round(k) && k != 0 && abs(k) < 9.007199254740992e15) || return oftype(x, NaN)
    x >= zero(x) && return abs(x)^(one(T) / n)
    return isodd(round(Int64, abs(k))) ? -(abs(x)^(one(T) / n)) : oftype(x, NaN)
end
"""

JULIA_MODULE = r"""
module SrbfTiSR
using TiSR
using Random
using LinearAlgebra
using PythonCall
import Pkg

BLAS.set_num_threads(1)
%(operators)s

tisr_tree() = string(Pkg.dependencies()[Base.UUID("%(uuid)s")].tree_hash)

function _tokens!(out, node)
    if node.ari == 2
        push!(out, "b" * string(node.ind)); _tokens!(out, node.lef); _tokens!(out, node.rig)
    elseif node.ari == 1
        push!(out, "u" * string(node.ind)); _tokens!(out, node.lef)
    elseif node.ari == 0
        push!(out, "v" * string(node.ind))
    else
        push!(out, "p" * repr(Float64(node.val)))
    end
    return out
end

# One search: the data matrix with y as its last column, TiSR's Options from `make` (with the problem's own values of
# any per-problem settings), the seed, the loop. The hall of fame comes back in TiSR's export order (convert_to_dict's
# default: sortperm by :ms_processed_e, which is stable), each member with TiSR's own values of it on the data.
function run(X, y, make, n_gens, t_lim, seed, values...)
    data_matr = hcat(Matrix{Float64}(X), Vector{Float64}(y))
    Random.seed!(seed)
    started = time()
    ops, data = make(data_matr, n_gens, Float64(t_lim), values...)
    hall_of_fame, population, prog_dict, stop_msg = generational_loop(data, ops)
    seconds = time() - started
    order = sortperm([indiv.measures[:ms_processed_e] for indiv in hall_of_fame])
    members = Py[]
    for indiv in hall_of_fame[order]
        measures = Dict{String, Float64}(string(name) => Float64(value) for (name, value) in pairs(indiv.measures))
        values = Float64.(TiSR.eval_equation(indiv.node, data, ops)[1])
        push!(members, pydict(Dict("tokens" => pylist(_tokens!(String[], indiv.node)), "measures" => pydict(measures),
                                   "values" => pylist(values))))
    end
    return pydict(Dict{String, Any}("members" => pylist(members), "seconds" => seconds,
                                    "stop" => stop_msg, "generations" => Int(prog_dict["generation"][end]),
                                    "progress" => pylist([pylist(Int.(prog_dict["generation"])),
                                                          pylist(Float64.(prog_dict["time"]))])))
end
end
"""

# TiSR's keyword sections, as ``config`` names them, and the function that builds each.
SECTIONS = {"data_split": "data_split_params", "general": "general_params", "measures": "measure_params",
            "selection": "selection_params", "fitting": "fitting_params", "mutation": "mutation_params",
            "grammar": "grammar_params"}
# What srbf sets in every run and a config cannot replace: the budget, the guard, one thread, no printing.
SET_BY_WORKER = {"n_gens": "n_gens", "t_lim": "t_lim", "multithreading": "false", "print_progress": "false",
                 "show_hall_of_fame": "false"}
# Settings that could end a search, or a fit, before its budget: left at TiSR's defaults, which never do.
NO_STOPS = {"general": ("callback",), "fitting": ("early_stop_iter",)}

# The budget if a config gives none (the top of the FastSRB ladder), and the fit a started worker runs once to compile
# TiSR outside any timing.
DEFAULT_GENERATIONS = 512
WARMUP_GENERATIONS = 3


def _julia_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return {"inf": "Inf", "-inf": "-Inf"}.get(repr(value), "NaN")
        return repr(value)
    if isinstance(value, str):
        return value
    raise ValueError("a TiSR setting is a number, a boolean or a string of Julia source, not %r" % (value,))


def check_config(config):
    """The operator lists a config chooses (the benchmark's by default), after checking every key it sets."""
    unknown = sorted(set(config) - set(SECTIONS) - {"binops", "unaops"})
    if unknown:
        raise ValueError("config sets %s; it may set binops, unaops and %s" % (", ".join(unknown), ", ".join(SECTIONS)))
    for section, fn in SECTIONS.items():
        settings = config.get(section) or {}
        if not isinstance(settings, dict):
            raise ValueError("config.%s maps keywords of TiSR's %s to values" % (section, fn))
        fixed = sorted(set(settings) & set(SET_BY_WORKER)) if section == "general" else []
        if fixed:
            raise ValueError("config.general cannot set %s: the worker sets them (generations, time_guard)" % ", ".join(fixed))
        stops = sorted(set(settings) & set(NO_STOPS.get(section, ())))
        if stops:
            raise ValueError("config.%s cannot set %s: every run spends its whole budget" % (section, ", ".join(stops)))
        for value in settings.values():
            _julia_value(value)
    binops = tuple(config.get("binops", BINARY_OPERATORS))
    unaops = tuple(config.get("unaops", UNARY_OPERATORS))
    for names, arity in ((binops, 2), (unaops, 1)):
        wrong = [name for name in names if OPERATORS.get(name, (0,))[0] != arity]
        if wrong:
            raise ValueError("not %s operators the worker knows: %s" % ("binary" if arity == 2 else "unary", ", ".join(wrong)))
    for required in ("+", "*"):
        if required not in binops:
            raise ValueError("TiSR needs %s among the binary operators" % required)
    return binops, unaops


def per_problem_keys(config, config_by_problem):
    """The (section, keyword) pairs ``config_by_problem`` sets, each a number or a boolean that ``config`` also sets
    (its value for the problems not listed). They become arguments of the one compiled constructor, so a problem's
    own value costs no compilation."""
    keys = set()
    for problem, override in config_by_problem.items():
        for section, settings in override.items():
            if section not in SECTIONS or not isinstance(settings, dict):
                raise ValueError("config_by_problem.%s sets %r; it may set keywords of %s" % (problem, section, ", ".join(SECTIONS)))
            for key, value in settings.items():
                if isinstance(value, str) or not isinstance(value, (bool, int, float)):
                    raise ValueError("config_by_problem.%s.%s.%s is a number or a boolean, not %r" % (problem, section, key, value))
                if key not in (config.get(section) or {}):
                    raise ValueError("config_by_problem sets %s.%s, which config does not set: give config the value for the "
                                     "problems not listed" % (section, key))
                keys.add((section, key))
    return sorted(keys)


def problem_values(config, config_by_problem, keys, problem):
    """A problem's values of the per-problem keys, in the order of ``keys``."""
    override = config_by_problem.get(problem) or {}
    return [(override.get(section) or {}).get(key, config[section][key]) for section, key in keys]


def options_source(config, keys=()):
    """Julia source of ``(data_matr, n_gens, t_lim, p1, p2, ...) -> Options(...)`` for a config, evaluated in
    SrbfTiSR; ``p1, p2, ...`` stand for the per-problem ``keys``."""
    binops, unaops = check_config(config)
    arguments = {key: "p%d" % (i + 1) for i, key in enumerate(keys)}
    sections = []
    for section, fn in SECTIONS.items():
        settings = dict(config.get(section) or {})
        words = ["%s = %s" % (key, arguments.get((section, key)) or _julia_value(value)) for key, value in settings.items()]
        if section == "general":
            words = ["%s = %s" % item for item in SET_BY_WORKER.items()] + words
        sections.append("%s = %s(; %s)" % (section, fn, ", ".join(words)))
    return ("(data_matr, n_gens, t_lim%s) -> Options(data_matr; binops = (%s,), unaops = (%s,), %s)"
            % ("".join(", " + arguments[key] for key in keys), ", ".join(binops), ", ".join(unaops), ", ".join(sections)))


def run_seed(x, y, seed=0):
    """A problem's seed: its data's hash mixed with the configured seed, in [0, 2**31)."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(x, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(y, dtype=np.float64).tobytes())
    h.update(str(int(seed)).encode())
    return int.from_bytes(h.digest()[:4], "little") & 0x7FFFFFFF


# -- writing TiSR's tree in the benchmark's syntax ----------------------------------------------------------------

def _number(text):
    value = float(text)   # Julia's repr of a Float64 is its shortest exact decimal, which Python reads to the same double
    if not math.isfinite(value):
        raise ValueError("TiSR's expression holds a non-finite constant %s" % text)
    out = repr(value)
    return "(%s)" % out if out.startswith("-") else out


def infix(tokens, names, binops, unaops):
    """TiSR's tree (the prefix tokens SrbfTiSR._tokens! writes) as an infix string in the benchmark's syntax."""
    position = [0]

    def walk():
        if position[0] >= len(tokens):
            raise ValueError("TiSR's tree ends early: %r" % (tokens,))
        token = tokens[position[0]]
        position[0] += 1
        kind, body = token[0], token[1:]
        if kind == "p":
            return _number(body)
        if kind == "v":
            index = int(body) - 1
            if not 0 <= index < len(names):
                raise ValueError("TiSR's expression uses v%s, but the problem has %d variables" % (body, len(names)))
            return names[index]
        if kind == "u":
            argument = walk()
            return OPERATORS[unaops[int(body) - 1]][1].format(argument)
        if kind == "b":
            left = walk()
            right = walk()
            return OPERATORS[binops[int(body) - 1]][1].format(left, right)
        raise ValueError("unknown token %r in TiSR's tree" % token)

    out = walk()
    if position[0] != len(tokens):
        raise ValueError("trailing tokens in TiSR's tree: %r" % (tokens,))
    return out


def _rootn(x, n):
    """The benchmark's rootn (simplipy.operators.rootn): odd integer index signed, even principal, else NaN."""
    x = np.asarray(x, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    k = np.where((n == np.floor(n)) & np.isfinite(n) & (n != 0), np.abs(n), np.nan)
    magnitude = np.abs(x) ** (1.0 / k)
    root = np.where(np.mod(k, 2) == 1, np.where(x < 0, -magnitude, magnitude), np.where(x < 0, np.nan, magnitude))
    root = np.where(k == 1, x, root)
    return np.where(np.isnan(k), np.nan, np.where(n < 0, 1.0 / root, root))


_NAMESPACE = {"neg": np.negative, "abs": np.abs, "inv": lambda a: 1.0 / a, "sin": np.sin, "cos": np.cos, "tan": np.tan,
              "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan, "sinh": np.sinh, "cosh": np.cosh,
              "tanh": np.tanh, "asinh": np.arcsinh, "acosh": np.arccosh, "atanh": np.arctanh, "exp": np.exp,
              "log": np.log, "sqrt": np.sqrt, "rootn": _rootn}


def string_deviation(expression, names, X, reference):
    """The largest deviation of the written expression from TiSR's own values on X, relative to the values' scale,
    or None when it cannot be evaluated here (a variable name that is no Python identifier, say)."""
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


# -- Julia ----------------------------------------------------------------------------------------------------------

def julia_environment(prefix=None):
    """Point juliacall at the Julia, project and depot inside this interpreter's environment, when it has them
    (the layout scripts/envs/build_tisr_env.sh builds): offline, one thread, no conda. Returns the project."""
    prefix = prefix or sys.prefix
    project = os.path.join(prefix, "julia_env")
    julia = os.path.join(prefix, "julia-" + JULIA_VERSION, "bin", "julia")
    if not (os.path.isfile(os.path.join(project, "Manifest.toml")) and os.path.isfile(julia)):
        return None
    os.environ.update({
        "JULIA_DEPOT_PATH": os.path.join(prefix, "julia_depot"),
        "PYTHON_JULIAPKG_PROJECT": project,
        "PYTHON_JULIAPKG_EXE": julia,
        "PYTHON_JULIAPKG_OFFLINE": "yes",
        "PYTHON_JULIACALL_THREADS": "1",
        "JULIA_NUM_THREADS": "1",
        "JULIA_CONDAPKG_BACKEND": "Null",
    })
    return project


def _require_julia():
    julia_environment()
    try:
        from juliacall import Main as jl
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("juliacall and TiSR are required in the worker's environment: build it with "
                          "scripts/envs/build_tisr_env.sh") from exc
    jl.seval("import TiSR")
    jl.seval(JULIA_MODULE % {"operators": JULIA_OPERATORS, "uuid": TISR_UUID})
    return jl


def load(options):
    state = {"generations": int(options.get("generations", DEFAULT_GENERATIONS)), "seed": int(options.get("seed", 0)),
             "time_guard": float(options.get("time_guard", 3600.0)), "config": dict(options.get("config") or {}),
             "config_by_problem": {str(k): dict(v) for k, v in (options.get("config_by_problem") or {}).items()}}
    if state["generations"] < 1:
        raise ValueError("generations must be at least 1, got %r" % state["generations"])
    state["binops"], state["unaops"] = check_config(state["config"])      # a bad option fails at start, not per problem
    state["keys"] = per_problem_keys(state["config"], state["config_by_problem"])
    source = options_source(state["config"], state["keys"])
    jl = _require_julia()
    tree = str(jl.SrbfTiSR.tisr_tree())
    if tree != TREE:
        raise RuntimeError("the environment's TiSR is tree %s, not %s (commit %s); rebuild it with "
                           "scripts/envs/build_tisr_env.sh" % (tree, TREE, COMMIT))
    state.update(jl=jl, run=jl.SrbfTiSR.run, julia=str(jl.seval("string(VERSION)")),
                 make=jl.seval("source -> SrbfTiSR.eval(Meta.parse(source))")(source))
    if bool(options.get("warmup", True)):
        warmup_fit(state)
    return state


def warmup_fit(state):
    """A throwaway fit with the configured TiSR settings, outside any timed fit."""
    rng = np.random.RandomState(0)
    X = rng.uniform(1.0, 2.0, size=(64, 2))
    small = dict(state, generations=WARMUP_GENERATIONS)
    fit(X, X[:, 0] * X[:, 1] + 1.0, x_val=[], variables=["a", "b"], meta={}, options={}, state=small)


def info(state):
    out = {"worker": "tisr", "tisr_commit": COMMIT, "julia": state.get("julia")}
    try:
        from importlib.metadata import version as _version
        out["juliacall"] = _version("juliacall")
    except Exception:  # noqa: BLE001
        pass
    return out


def fit(x, y, *, x_val, variables, meta, options, state):
    X = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    Y = np.ascontiguousarray(np.asarray(y, dtype=np.float64).ravel())
    names = list(variables)
    seed = run_seed(X, Y, state["seed"])
    values = problem_values(state["config"], state["config_by_problem"], state["keys"], str((meta or {}).get("benchmark_eq_id")))
    extra = {"seed": seed, "generations": state["generations"]}
    if state["keys"]:
        extra["problem_settings"] = {"%s.%s" % key: value for key, value in zip(state["keys"], values)}

    cpu = time.process_time()
    report = state["run"](X, Y, state["make"], state["generations"], state["time_guard"], seed, *values)
    extra["cpu_seconds"] = time.process_time() - cpu
    members = list(report["members"])
    extra.update(generations_run=int(report["generations"]), stop=str(report["stop"]),
                 hit_time_guard=str(report["stop"]) == "reached time limit")
    generations, seconds = (list(part) for part in report["progress"])
    extra["progress"] = {"generation": [int(g) for g in generations], "seconds": [float(t) for t in seconds]}
    if not members:
        return {"error": "TiSR's hall of fame is empty", "fit_time": float(report["seconds"]), "extra": extra}

    hall_of_fame = []
    for member in members:
        entry = {key: float(value) for key, value in member["measures"].items()}
        try:
            entry["expression"] = infix([str(t) for t in member["tokens"]], names, state["binops"], state["unaops"])
            entry["string_deviation"] = string_deviation(entry["expression"], names, X,
                                                         np.asarray(list(member["values"]), dtype=np.float64))
        except ValueError as exc:
            entry["expression"] = None
            entry["error"] = str(exc)
        hall_of_fame.append(entry)
    extra["hall_of_fame"] = hall_of_fame
    # the search itself: TiSR's Options and its loop, not the conversion to Python or the checks above
    return {"error": NO_ANSWER, "fit_time": float(report["seconds"]), "extra": extra}
