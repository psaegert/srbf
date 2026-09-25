"""The Bingo worker: NASA's Bingo (bingo-nasa 0.5.7) in its authors' benchmark configuration.

Runs in the environment ``scripts/envs/build_bingo_env.sh`` builds (bingo-nasa 0.5.7 with its C++ extension,
MPICH and mpi4py); imports ``bingo`` and numpy only.

Bingo (Randall, Townsend, Hochhalter and Bomarito, GECCO 2022 Companion; github.com/nasa/bingo, Apache-2.0) is
genetic programming over acyclic graphs (a stack of commands that may reuse earlier results) with age-fitness Pareto
selection and Levenberg-Marquardt fitting of every individual's constants. The configuration is the one Bingo's
authors ship for benchmarking and submitted to SRBench (``bingo.symbolic_regression.srbench_interface.est``, the
same settings as their SRBench 2024/25 submission): population 500, stack size 24, simplification on, crossover
0.3, mutation 0.45, mean squared error, Levenberg-Marquardt with tolerance 1e-5, the age-fitness EA; everything
else at the library's defaults (fitness threshold 1e-16, no generation limit). srbf sets only what it sets for
every method: the operators (the benchmark's operators as far as Bingo has them), the budget (a count of fitness
evaluations) and the seed. The authors' own wall-clock limit (350 s) is replaced by a guard of 3,500 s, the limit
SRBench 2.0 ran Bingo with (Bingo stops at 97 % of it), which the ladder does not reach: the budget is a count. See
docs/models.md.

``options`` (from the config's ``model_adapter`` block):

* ``max_evals`` (int, 1,000,000): the budget, the compute axis of the scaling sweeps. Bingo counts every evaluation
  of an individual's residuals and of their Jacobian, those of the constant fitting included. It checks the count
  after its initial population and then every 10 generations, so a run overshoots it by up to 10 generations.
* ``seed`` (int, 0): mixed with a hash of the problem's data into the run's seed, so a problem's fit is
  reproducible and two draws of a law (fresh data) get different seeds.
* ``config`` (dict, none): settings that replace the author configuration's, ``operators`` included. For side
  experiments only, such as reproducing a published configuration; a config that sets it is ``harness_tuned``, and
  published results never set it.

The answer is Bingo's own: after the search, Bingo refits the constants of its Pareto front (fitness against
complexity) and of randomly chosen members of its final population, up to 100 equations (five restarts each), and
returns the one with the lowest mean squared error on the training data (``get_best_individual``, which Bingo's
SRBench interface reports). That rule favours accurate over simple equations. The worker writes that equation from
Bingo's own command stack (the simplified one Bingo evaluates), every constant at full precision, in the benchmark's
operators and srbf's variable names. Bingo's logarithm and square root are protected, and their printed names hide
it: ``log(u)`` computes ``log(|u|)`` and ``sqrt(u)`` computes ``sqrt(|u|)``, so they are written ``log(abs(u))`` and
``sqrt(abs(u))``, and srbf evaluates what Bingo evaluated. A graph reuses subexpressions; the string writes each use
out. ``extra`` keeps the largest deviation of the string from Bingo's own values on the data (``string_deviation``),
Bingo's own printed equation, the evaluations and generations used, and the Pareto front.

Nothing in Bingo is patched. Three settings of its surroundings are fixed, none of which the search sees:

* Bingo reads ``OMP_NUM_THREADS`` as the number of processes it evaluates a generation with (``0``, or unset,
  evaluates in its own process), so the worker hands it ``0`` for the length of a fit, after the numerical libraries
  have read their one-thread setting.
* Importing Bingo initializes MPI (its logging module asks mpi4py for the process count), which the worker never
  uses. By default MPICH then loads its UCX transport, which starts two threads and loads the CUDA driver on a
  machine with a GPU. The worker points MPICH at libfabric's shared-memory provider instead
  (``MPIR_CVAR_CH4_NETMOD=ofi``, ``FI_PROVIDER=shm``), with which the initialization starts no thread.
* Python's warnings from inside Bingo (overflows while evaluating random equations) are silenced, so that the
  worker's log stays readable.

One fit runs in one process on one thread.
"""
import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):  # before numpy: one thread
    os.environ.setdefault(_variable, "1")
os.environ.setdefault("MPIR_CVAR_CH4_NETMOD", "ofi")   # before mpi4py initializes MPICH: no UCX, no CUDA, no threads
os.environ.setdefault("FI_PROVIDER", "shm")

import ast  # noqa: E402
import contextlib  # noqa: E402
import hashlib  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402

# The benchmark's operators Bingo evaluates, in Bingo's names ("|" is its absolute value, "pow" its power). Bingo's
# C++ backend, which it uses whenever its extension loads, has no tan, asin, acos, atan or tanh (they exist only in
# its Python backend); neg and inv it expresses through - and /, and it has no other roots than sqrt.
OPERATORS = ("+", "-", "*", "/", "sin", "cos", "sinh", "cosh", "exp", "log", "pow", "|", "sqrt")

# The authors' benchmark configuration (bingo-nasa 0.5.7, bingo/symbolic_regression/srbench_interface.py, the same
# settings as their SRBench 2024/25 submission in cavalab/srbench), with the library defaults it relies on written
# out, so that no changed default can change the method silently. Their operator list was
# + - * / sin cos exp log sqrt; srbf's replaces it (OPERATORS). The budget and the seed are set per fit.
AUTHOR_CONFIG = {
    "population_size": 500,
    "stack_size": 24,
    "operators": list(OPERATORS),
    "use_simplification": True,
    "crossover_prob": 0.3,
    "mutation_prob": 0.45,
    "metric": "mse",
    "clo_alg": "lm",
    "generations": int(1e19),
    "fitness_threshold": 1.0e-16,
    "max_time": 3500,        # a guard (SRBench 2.0's limit for Bingo; the authors' own is 350 s): the budget is a count
    "evolutionary_algorithm": "AgeFitnessEA",
    "clo_threshold": 1.0e-5,
    "scale_max_evals": False,
}
SET_PER_FIT = ("max_evals", "random_state")

# Bingo's operator codes (bingo/symbolic_regression/agraph/operator_definitions.py) and what each computes, written
# in the benchmark's notation. The codes are checked against the installed package by srbf's tests.
INTEGER, VARIABLE, CONSTANT = -1, 0, 1
SPELLING = {
    2: "({} + {})",           # ADDITION
    3: "({} - {})",           # SUBTRACTION
    4: "({}*{})",             # MULTIPLICATION
    5: "({}/{})",             # DIVISION
    6: "sin({})",             # SIN
    7: "cos({})",             # COS
    8: "exp({})",             # EXPONENTIAL
    9: "log(abs({}))",        # LOGARITHM: log(|u|)
    10: "({})**({})",         # POWER
    11: "abs({})",            # ABS
    12: "sqrt(abs({}))",      # SQRT: sqrt(|u|)
    13: "(abs({}))**({})",    # SAFE_POWER
    14: "sinh({})",           # SINH
    15: "cosh({})",           # COSH
    16: "tan({})",            # TAN (Python backend only)
    17: "asin({})",           # ARCSIN (Python backend only)
    18: "acos({})",           # ARCCOS (Python backend only)
    19: "atan({})",           # ARCTAN (Python backend only)
    20: "tanh({})",           # TANH (Python backend only)
    21: "({})**2",            # SQUARE (Python backend only)
    22: "({})**3",            # CUBE (Python backend only)
}
BINARY = {2, 3, 4, 5, 10, 13}


def _require_bingo():
    try:
        from bingo.symbolic_regression.symbolic_regressor import SymbolicRegressor
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("bingo-nasa 0.5.7 is required in the worker's environment: "
                          "scripts/envs/build_bingo_env.sh") from exc
    return SymbolicRegressor


def run_seed(x, y, seed=0):
    """A problem's seed: its data's hash mixed with the configured seed, in [0, 2**31)."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(x, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(y, dtype=np.float64).tobytes())
    h.update(str(int(seed)).encode())
    return int.from_bytes(h.digest()[:4], "little") & 0x7FFFFFFF


def build_params(max_evals, seed, config=None):
    """The SymbolicRegressor arguments of one fit: the author configuration, a side experiment's replacements, the
    budget and the seed."""
    config = dict(config or {})
    unknown = sorted(set(config) - set(AUTHOR_CONFIG))
    fixed = sorted(set(config) & set(SET_PER_FIT))
    if unknown or fixed:
        raise ValueError("config may replace %s; not %s" % (", ".join(sorted(AUTHOR_CONFIG)), ", ".join(unknown + fixed)))
    if not 0 < int(max_evals):
        raise ValueError("max_evals must be positive, got %r" % max_evals)
    params = {**AUTHOR_CONFIG, **config, "max_evals": int(max_evals), "random_state": int(seed)}
    params["operators"] = list(params["operators"])
    return params


# -- writing Bingo's equation --------------------------------------------------------------------------------------

def _number(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Bingo's equation holds a non-finite constant %r" % value)
    text = repr(value)
    return "(%s)" % text if text.startswith("-") else text


def _integer(value):
    value = int(value)
    return "(%d)" % value if value < 0 else "%d" % value


def infix(commands, constants, names):
    """Bingo's command stack (rows ``(node, param1, param2)``, each operator referring to earlier rows, the last row
    the result) and its constants, as an expression in the benchmark's notation and ``names``. Rows the result does
    not use are skipped."""
    commands = [tuple(int(v) for v in row) for row in np.asarray(commands).reshape(-1, 3)]
    if not commands:
        raise ValueError("Bingo's equation is empty")
    used = [False] * len(commands)
    used[-1] = True
    for i in range(len(commands) - 1, -1, -1):
        node, p1, p2 = commands[i]
        if used[i] and node in SPELLING:
            used[p1] = True
            if node in BINARY:
                used[p2] = True
    text = {}
    for i, (node, p1, p2) in enumerate(commands):
        if not used[i]:
            continue
        if node == VARIABLE:
            if not 0 <= p1 < len(names):
                raise ValueError("Bingo's equation uses X_%d, but the problem has %d variables" % (p1, len(names)))
            text[i] = names[p1]
        elif node == CONSTANT:
            if not 0 <= p1 < len(constants):
                raise ValueError("Bingo's equation refers to constant %d of %d" % (p1, len(constants)))
            text[i] = _number(constants[p1])
        elif node == INTEGER:
            text[i] = _integer(p1)
        elif node in BINARY:
            text[i] = SPELLING[node].format(text[p1], text[p2])
        elif node in SPELLING:
            text[i] = SPELLING[node].format(text[p1])
        else:
            raise ValueError("Bingo operator %d has no spelling in the benchmark's operators" % node)
    return text[len(commands) - 1]


def equation_stack(equation):
    """The command stack and constants Bingo evaluates for ``equation``: the simplified stack when simplification is
    on (``_simplified_command_array``, what ``evaluate_equation_at`` and Bingo's printing read)."""
    equation.get_complexity()                  # brings the simplified stack up to date, as printing does
    return np.asarray(equation._simplified_command_array), tuple(equation.constants)


class _Floats(ast.NodeTransformer):
    """Numbers in the expression as numpy floats, so that a power of two literals follows numpy, not Python."""

    def visit_Constant(self, node):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            call = ast.Call(func=ast.Name(id="_f", ctx=ast.Load()), args=[node], keywords=[])
            return ast.copy_location(call, node)
        return node


_NAMESPACE = {"_f": np.float64, "abs": np.abs, "sin": np.sin, "cos": np.cos, "tan": np.tan, "sinh": np.sinh,
              "cosh": np.cosh, "tanh": np.tanh, "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "asin": np.arcsin,
              "acos": np.arccos, "atan": np.arctan}


def evaluate(expression, names, X):
    """The expression's values on the rows of ``X`` (numpy semantics)."""
    tree = ast.fix_missing_locations(_Floats().visit(ast.parse(expression, mode="eval")))
    namespace = dict(_NAMESPACE)
    namespace.update({name: X[:, i] for i, name in enumerate(names)})
    with np.errstate(all="ignore"):
        values = eval(compile(tree, "<bingo>", "eval"), {"__builtins__": {}}, namespace)
    return np.broadcast_to(np.asarray(values, dtype=np.float64), (X.shape[0],))


def string_deviation(expression, names, X, reference):
    """The largest deviation of the written expression from Bingo's own values on X, relative to the values' scale:
    ``inf`` when the two are not finite at the same points, ``None`` when the string cannot be evaluated here."""
    try:
        values = evaluate(expression, names, X)
    except Exception:  # noqa: BLE001 - a diagnostic, never a failure
        return None
    reference = np.asarray(reference, dtype=np.float64).ravel()
    finite = np.isfinite(reference)
    if not np.array_equal(finite, np.isfinite(values)):
        return float("inf")
    if not finite.any():
        return 0.0
    scale = max(1.0, float(np.max(np.abs(reference[finite]))))
    return float(np.max(np.abs(values[finite] - reference[finite])) / scale)


# -- the protocol ------------------------------------------------------------------------------------------------

@contextlib.contextmanager
def serial_evaluation():
    """Bingo evaluates in its own process while this is active: it reads OMP_NUM_THREADS as its process count."""
    previous = os.environ.get("OMP_NUM_THREADS")
    os.environ["OMP_NUM_THREADS"] = "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = previous


def load(options):
    state = {"max_evals": int(options.get("max_evals", 1_000_000)), "seed": int(options.get("seed", 0)),
             "config": dict(options.get("config") or {})}
    build_params(state["max_evals"], 0, state["config"])      # a bad option fails at start, not per problem
    state["SymbolicRegressor"] = _require_bingo()
    return state


def info(state):
    out = {"worker": "bingo", "operators": build_params(1, 0, state["config"])["operators"]}
    try:
        from importlib.metadata import version as _version
        out["bingo-nasa"] = _version("bingo-nasa")
    except Exception:  # noqa: BLE001
        out["bingo-nasa"] = "?"
    try:
        from bingo.symbolic_regression import ISCPP
        out["cpp_backend"] = bool(ISCPP)
    except Exception:  # noqa: BLE001
        pass
    return out


def fit(x, y, *, x_val, variables, meta, options, state):
    X = np.asarray(x, dtype=np.float64)
    Y = np.asarray(y, dtype=np.float64).ravel()
    names = list(variables)
    seed = run_seed(X, Y, state["seed"])
    params = build_params(state["max_evals"], seed, state["config"])
    model = state["SymbolicRegressor"](**params)

    started = time.perf_counter()
    with warnings.catch_warnings(), serial_evaluation():
        warnings.simplefilter("ignore")
        model.fit(X, Y)
        best = model.get_best_individual()
        commands, constants = equation_stack(best.equation)
        expression = infix(commands, constants, names)
    fit_time = time.perf_counter() - started   # the search and its answer; the bookkeeping below is not timed

    extra = {"seed": seed, "max_evals": state["max_evals"], "bingo_string": str(best),
             "fitness": float(best.fitness), "model_complexity": int(best.complexity)}
    archipelago = getattr(model, "archipelago", None)
    if archipelago is not None:
        extra["evaluations"] = int(archipelago.get_fitness_evaluation_count())
        extra["generations"] = int(archipelago.generational_age)
        extra["fitness_predictor_island"] = type(archipelago).__name__ == "FitnessPredictorIsland"
    extra["converged"] = bool(extra["fitness"] <= params["fitness_threshold"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        extra["string_deviation"] = string_deviation(expression, names, X, best.equation.evaluate_equation_at(X))
        try:
            extra["front"] = [{"expression": infix(*equation_stack(member.equation), names),
                               "complexity": int(member.complexity), "fitness": float(member.fitness)}
                              for member in model.get_pareto_front()]
        except Exception:  # noqa: BLE001 - persistence is best-effort
            pass
    return {"expression": expression, "fit_time": fit_time, "extra": extra}
