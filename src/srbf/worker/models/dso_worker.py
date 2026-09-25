"""The DSO worker: deep symbolic optimization (DSO v3.0.0) as two methods, DSR and uDSR*.

Runs in DSO's own environment (Python 3.7, TensorFlow 1.14; ``scripts/envs/build_dso_env.sh``); imports ``dso``
and numpy only, and runs on Python 3.7. Both arms call ``dso.DeepSymbolicOptimizer`` directly: DSO's scikit-learn
wrapper switches GP-meld off.

Arms (``options.arm``):

* ``dsr``: deep symbolic regression (Petersen et al., ICLR 2021), an RNN trained with the risk-seeking policy
  gradient, in the configuration DSO v3.0.0 ships as its regression default (``config_regression.json``: batch
  1,000, learning rate 0.0005, entropy weight 0.03 with gamma 0.7, epsilon 0.05, soft length and uniform arity
  priors, length 4 to 64), plus the ``const`` token, which DSO's authors advise for data with constants.
* ``udsr``: uDSR* (Landajuela et al., NeurIPS 2022) as far as it was publicly released: DSR, GP-meld (Mundhenk et
  al., NeurIPS 2021) and the polynomial token (LINEAR, ``poly``), with ``const`` and ``1.0``, in the configuration of
  the uDSR paper's Table 3: priority queue training (queue 10, learning rate 0.0025), batch and GP population 500,
  epsilon 0.02, entropy weight 0.03 with gamma 0.7 (Table 3 prints 0.3; the paper it cites and every released
  config use 0.03), 25 GP generations per iteration, LINEAR of degree 3 with at most 10 terms. The paper's AI
  Feynman step and pre-training were never released, hence the asterisk.

Both arms search over the benchmark's operators as far as DSO has them: ``+ - * /``, ``neg abs inv sin cos tan tanh
exp log``, ``pow`` as the squares, cubes and fourth powers (``n2 n3 n4``) and ``rootn`` as ``sqrt``. DSO has no
``asin acos atan sinh cosh asinh acosh atanh`` and no general power or root.

Three patches to DSO v3.0.0 are applied in :func:`load`, each the smallest change that restores the released method:

1. **GP-meld evaluates the expressions it breeds.** v3.0.0's ``dso.gp.utils.Individual.tokenized_repr`` returns
   the token array an individual was created from, not its current tree. GP-meld therefore scores every offspring
   as its unvaried parent (all cache hits), and a clone resets a varied tree, so the elite it hands back to the RNN
   are copies of RNN samples: GP-meld is inert. The patch derives the tokens from the tree, which is what DSO
   v2.1.0 (the GP-meld paper's release) and the authors' 2022 competition code evaluate.
2. **The inverse table covers ``neg`` and ``n4``.** To fit LINEAR, DSO inverts the unary operators above it through
   a table (``polyfit.inverse_function_map``) that has no entry for ``neg`` or ``n4``, so an expression with either
   above LINEAR stops the run with a ``KeyError``. The patch adds both: ``neg`` is its own inverse, and ``n4`` gets
   the principal fourth root, as the table's ``n2`` gets the principal square root. The RNN never puts ``n4`` above
   LINEAR (a prior forbids it, as for ``sin cos tan n2 abs``), but GP-meld's check of that prior reads only the
   first forbidden operator, in v3.0.0 as in the authors' 2022 code, so GP-meld breeds such expressions; the
   patch leaves that check as it is.
3. **A GP-meld clone copies the tree, not the primitive set.** v3.0.0's ``Individual.__deepcopy__`` deep-copies the
   primitive set every individual shares, on each of the tens of thousands of clones per iteration; this took most
   of an iteration's time. The patch lets the copy share it. The search is unchanged, expression for expression;
   DSO v2.1.0 and the 2022 code cloned plain trees.

``options`` (from the config's ``model_adapter`` block):

* ``arm`` (``dsr`` or ``udsr``, required).
* ``n_samples`` (int, 2,000,000): the budget, DSO's ``training.n_samples``, the compute axis of the scaling sweeps.
  It counts the expressions the RNN samples and GP-meld breeds, repeats included. DSO checks it after each
  iteration (1,000 expressions for DSR, 500 + 25 x 500 = 13,000 for uDSR*), so it is rounded up to whole
  iterations. A run stops early once an expression fits the training data to a normalized MSE below 1e-12.
* ``seed`` (int, 0): mixed with a hash of the problem's data into the run's seed.
* ``max_seconds`` (float, 3600): a wall-clock guard checked after each iteration, set far above any budget of the
  ladders. When it passes, the best expression so far is returned and ``guard_hit`` is set.
* ``warmup`` (bool, true): run a small throwaway fit in :func:`load`, so that the one-time compilation of DSO's
  numba functions is not part of the first problem's time.
* ``config`` (dict, none): settings merged over the arm's configuration (lists replace). For side experiments
  only, such as reproducing a published configuration; published results never set it.

The prediction is DSO's own answer, the expression with the highest reward (inverse normalized RMSE on the
training data) of the whole run, printed from its token sequence at full precision (DSO prints LINEAR's
coefficients to six digits) in the variable names srbf handed over. DSO evaluates in float64 without protected
operators, so the string computes what DSO scored, up to one detail: DSO stores the literal ``1.0`` in float32 and
computes a subexpression made of it alone, such as ``tanh(1.0)``, in float32 (about 1e-8 apart from the string's
value). srbf evaluates the string. ``extra`` records the samples and iterations used, the reward, the token
sequence, DSO's complexity and length of the answer, how many GP offspring were new expressions, the largest
deviation of the string from DSO's own values on the data (``string_deviation``), and the Pareto front of
complexity and reward over every expression evaluated (``front``).
"""
import os

for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):  # before numpy: one thread
    os.environ.setdefault(_variable, "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import copy  # noqa: E402
import gc  # noqa: E402
import hashlib  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

ARMS = ("dsr", "udsr")

# The benchmark's operators as DSO has them (dso/functions.py): pow only as n2/n3/n4 and rootn only as sqrt.
OPERATORS = ["add", "sub", "mul", "div", "neg", "abs", "inv", "sin", "cos", "tan", "tanh", "exp", "log",
             "sqrt", "n2", "n3", "n4"]
TERMINALS = {"dsr": ["const"], "udsr": ["1.0", "const", "poly"]}

_PRIORS = {
    "length": {"min_": 4, "max_": 64, "on": True},
    "repeat": {"tokens": "const", "min_": None, "max_": 3, "on": True},
    "inverse": {"on": True},
    "trig": {"on": True},
    "const": {"on": True},
    "no_inputs": {"on": True},
    "uniform_arity": {"on": True},
    "soft_length": {"loc": 10, "scale": 5, "on": True},
    "domain_range": {"on": False},
}
_TASK = {"task_type": "regression", "metric": "inv_nrmse", "metric_params": [1.0], "threshold": 1e-12,
         "protected": False, "reward_noise": 0.0, "decision_tree_threshold_set": []}
_TRAINING = {"baseline": "R_e", "n_cores_batch": 1, "complexity": "token", "const_optimizer": "scipy",
             "const_params": {"method": "L-BFGS-B", "options": {"gtol": 1e-3}}, "early_stopping": True,
             "use_memory": False, "verbose": False}
_POLICY = {"policy_type": "rnn", "max_length": 64, "cell": "lstm", "num_layers": 1, "num_units": 32,
           "initializer": "zeros"}
_STATE_MANAGER = {"type": "hierarchical", "observe_action": False, "observe_parent": True, "observe_sibling": True,
                  "observe_dangling": False, "embedding": False, "embedding_size": 8}
# DSO's end-of-run hall of fame and Pareto front are written only to its log directory, which a worker has none
# of; the authors' 2022 competition config switches them off too. Logging only: the search does not see them.
_LOGGING = {"hof": None, "save_pareto_front": False}

# Written out in full so that the configuration is visible here, not only in the pinned library's JSON files.
ARM_CONFIGS = {
    # DSO v3.0.0 dso/config/config_regression.json over config_common.json, unchanged.
    "dsr": {
        "task": dict(_TASK),
        "training": dict(_TRAINING, batch_size=1000, epsilon=0.05),
        "policy": dict(_POLICY),
        "state_manager": dict(_STATE_MANAGER),
        "policy_optimizer": {"policy_optimizer_type": "pg", "learning_rate": 0.0005, "optimizer": "adam",
                             "entropy_weight": 0.03, "entropy_gamma": 0.7},
        "gp_meld": {"run_gp_meld": False},
        "prior": copy.deepcopy(_PRIORS),
        "logging": dict(_LOGGING),
    },
    # The uDSR paper's Table 3 (Landajuela et al. 2022, Appendix F), in DSO v3.0.0's keys.
    "udsr": {
        "task": dict(_TASK, poly_optimizer_params={
            "degree": 3, "coef_tol": 1e-6, "regressor": "dso_least_squares",
            "regressor_params": {"cutoff_p_value": 1.0, "n_max_terms": 10, "coef_tol": 1e-6}}),
        "training": dict(_TRAINING, batch_size=500, epsilon=0.02),
        "policy": dict(_POLICY),
        "state_manager": dict(_STATE_MANAGER),
        "policy_optimizer": {"policy_optimizer_type": "pqt", "learning_rate": 0.0025, "optimizer": "adam",
                             "entropy_weight": 0.03, "entropy_gamma": 0.7, "pqt_k": 10, "pqt_batch_size": 1,
                             "pqt_weight": 200.0, "pqt_use_pg": False},
        "gp_meld": {"run_gp_meld": True, "population_size": 500, "generations": 25,
                    "crossover_operator": "cxOnePoint", "p_crossover": 0.5, "mutation_operator": "multi_mutate",
                    "p_mutate": 0.5, "tournament_size": 5, "train_n": 50, "mutate_tree_max": 3,
                    "parallel_eval": False, "verbose": False},
        "prior": copy.deepcopy(_PRIORS),
        "logging": dict(_LOGGING),
    },
}

PATCHES = ("gp_meld_evaluates_bred_trees", "linear_inverts_neg_and_n4", "gp_meld_clone_shares_primitive_set")


def _require_dso():
    """The parts of DSO the worker uses, in one namespace (a stand-in replaces it in srbf's tests)."""
    try:
        import dso
        from dso import DeepSymbolicOptimizer
        from dso.program import Program
        import dso.gp.utils as gp_utils
        from dso.task.regression import polyfit
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("DSO v3.0.0 is required in the worker's environment: scripts/envs/build_dso_env.sh") from exc
    return {"dso": dso, "DeepSymbolicOptimizer": DeepSymbolicOptimizer, "Program": Program, "gp_utils": gp_utils,
            "polyfit": polyfit}


def _tree_tokens(individual):
    """An individual's tokens read from its current tree (patch 1)."""
    return np.array([node.name for node in individual], dtype=np.int32)


def _fourth_root(y):
    """The inverse of n4 for fitting LINEAR below it: the principal root, as DSO inverts n2 by the square root."""
    return np.sqrt(np.sqrt(y))


def _sharing_primitive_set(deepcopy):
    """Individual.__deepcopy__ with the primitive set shared by the copy (patch 3)."""
    if getattr(deepcopy, "shares_primitive_set", False):
        return deepcopy

    def wrapper(self, memo):
        memo[id(self.pset)] = self.pset
        return deepcopy(self, memo)

    wrapper.shares_primitive_set = True
    return wrapper


def apply_patches(modules):
    """Apply the three patches to the loaded DSO modules; returns their names."""
    individual = modules["gp_utils"].Individual
    individual.tokenized_repr = property(_tree_tokens)
    inverses = modules["polyfit"].inverse_function_map
    inverses.setdefault("neg", np.negative)
    inverses.setdefault("n4", _fourth_root)
    individual.__deepcopy__ = _sharing_primitive_set(individual.__deepcopy__)
    return list(PATCHES)


def merge(base, override):
    """``override`` merged over ``base`` (nested dicts merge, anything else replaces); neither is modified."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def build_config(arm, X, y, *, n_samples, seed, overrides=None):
    """The DeepSymbolicOptimizer config of one fit: the arm's configuration, the operators, the data (a tuple, so
    DSO names the task ``regression`` and its seed shift is fixed), the budget and the seed."""
    if arm not in ARMS:
        raise ValueError("options.arm must be one of %s, got %r" % (", ".join(ARMS), arm))
    config = merge(ARM_CONFIGS[arm], {
        "experiment": {"logdir": None, "seed": int(seed)},
        "task": {"function_set": OPERATORS + TERMINALS[arm]},
        "training": {"n_samples": int(n_samples)},
    })
    config = merge(config, overrides)
    config["task"]["dataset"] = (X, y)
    return config


def run_seed(x, y, seed=0):
    """A problem's seed: its data's hash mixed with the configured seed, in [0, 2**31)."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(x, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(y, dtype=np.float64).tobytes())
    h.update(str(int(seed)).encode())
    return int.from_bytes(h.digest()[:4], "little") & 0x7FFFFFFF


# ---------------------------------------------------------------------------------------------------------------
# printing a token sequence

_BINARY = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
_UNARY = ("neg", "abs", "inv", "sin", "cos", "tan", "tanh", "exp", "log", "sqrt")
_POWERS = {"n2": 2, "n3": 3, "n4": 4}


def _number(value):
    text = repr(float(value))
    return "(%s)" % text if text.startswith("-") else text


def _polynomial(token, names):
    """A fitted LINEAR token, every coefficient at full precision (DSO's own printing rounds to six digits)."""
    coefs = np.asarray(token.coef, dtype=np.float64).ravel()
    if coefs.size == 0:
        return "0.0"
    terms = []
    for coef, exponents in zip(coefs, token.exponents):
        factors = [_number(coef)]
        for index, power in enumerate(exponents):
            power = int(power)
            if power == 1:
                factors.append(names[index])
            elif power > 1:
                factors.append("%s**%d" % (names[index], power))
        terms.append(" * ".join(factors))
    return "(%s)" % " + ".join(terms)


def _leaf(token, names):
    if token.input_var is not None:
        return names[token.input_var]
    if token.name == "poly":
        return _polynomial(token, names)
    return _number(np.asarray(token.value, dtype=np.float64).ravel()[0])   # an optimized const, or a literal


def infix(traversal, names):
    """The expression of a DSO token sequence (prefix order) in the benchmark's notation and ``names``."""
    stack = []
    for token in reversed(list(traversal)):
        if token.arity == 0:
            stack.append(_leaf(token, names))
            continue
        args = [stack.pop() for _ in range(token.arity)]
        name = token.name
        if name in _BINARY:
            stack.append("(%s %s %s)" % (args[0], _BINARY[name], args[1]))
        elif name in _POWERS:
            stack.append("(%s)**%d" % (args[0], _POWERS[name]))
        elif name in _UNARY:
            stack.append("%s(%s)" % (name, args[0]))
        else:
            raise ValueError("token %r has no spelling in the benchmark's operators" % name)
    if len(stack) != 1:
        raise ValueError("not a complete expression: %d subtrees left" % len(stack))
    return stack[0]


def token_sequence(traversal, names):
    """The token sequence with full-precision values, comma-separated (DSO's own repr rounds LINEAR)."""
    return ",".join(_leaf(token, names) if token.arity == 0 else token.name for token in traversal)


_NAMESPACE = {"neg": np.negative, "abs": np.abs, "inv": lambda a: 1.0 / a, "sin": np.sin, "cos": np.cos,
              "tan": np.tan, "tanh": np.tanh, "exp": np.exp, "log": np.log, "sqrt": np.sqrt}


def string_deviation(expression, names, X, reference):
    """The largest deviation of the printed expression from DSO's own values on X, relative to the values' scale,
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


# ---------------------------------------------------------------------------------------------------------------
# the Pareto front

def pareto_mask(costs):
    """Rows of ``costs`` (lower is better in every column) that no other row dominates."""
    costs = np.asarray(costs, dtype=np.float64)
    keep = np.ones(costs.shape[0], dtype=bool)
    for i in range(costs.shape[0]):
        if keep[i]:
            dominated = np.all(costs[i] <= costs, axis=1) & np.any(costs[i] < costs, axis=1)
            keep[dominated] = False
    return keep


def pareto_front(programs, names):
    """DSO's own front (complexity against reward) over every valid expression it evaluated, by complexity."""
    programs = [p for p in programs if not getattr(p, "invalid", False) and np.isfinite(p.r)]
    if not programs:
        return []
    costs = np.array([(float(p.complexity), -float(p.r)) for p in programs])
    order = np.lexsort((costs[:, 1], costs[:, 0]))
    costs, programs = costs[order], [programs[i] for i in order]
    front = []
    for keep, p in zip(pareto_mask(costs), programs):
        if keep:
            front.append({"expression": infix(p.traversal, names), "complexity": float(p.complexity),
                          "length": len(p.traversal), "reward": float(p.r)})
    return front


# ---------------------------------------------------------------------------------------------------------------
# the protocol

def load(options):
    arm = options.get("arm")
    if arm not in ARMS:
        raise ValueError("options.arm must be one of %s, got %r" % (", ".join(ARMS), arm))
    modules = _require_dso()
    patches = apply_patches(modules)
    state = {"arm": arm, "n_samples": int(options.get("n_samples", 2_000_000)), "seed": int(options.get("seed", 0)),
             "max_seconds": float(options.get("max_seconds", 3600.0)), "config": dict(options.get("config") or {}),
             "modules": modules, "patches": patches}
    if options.get("warmup", True):
        warmup_fit(state)
    return state


# A run small enough to take a second or two that still calls every compiled function of the real runs.
WARMUP = {"training": {"n_samples": 1, "batch_size": 20}, "gp_meld": {"population_size": 20, "generations": 2}}


def warmup_fit(state):
    """One small throwaway fit with the arm's configuration, outside any timed fit."""
    rng = np.random.RandomState(0)
    X = rng.uniform(1.0, 2.0, size=(32, 2))
    small = dict(state, config=merge(state["config"], WARMUP), max_seconds=float("inf"))
    fit(X, X[:, 0] * X[:, 1] + 1.0, x_val=[], variables=["a", "b"], meta={}, options={}, state=small)


def _source_commit(dso_module):
    """The commit of an editable DSO checkout (<repo>/dso/dso/__init__.py), or None."""
    try:
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(dso_module.__file__))))
        with open(os.path.join(repo, ".git", "HEAD")) as f:
            head = f.read().strip()
        if head.startswith("ref: "):
            with open(os.path.join(repo, ".git", head[5:])) as f:
                head = f.read().strip()
        return head
    except Exception:  # noqa: BLE001
        return None


def info(state):
    out = {"worker": "dso", "arm": state["arm"], "patches": list(state["patches"])}
    try:
        import tensorflow as tf
        out["tensorflow"] = tf.__version__
    except Exception:  # noqa: BLE001
        pass
    commit = _source_commit(state["modules"]["dso"])
    if commit:
        out["dso_commit"] = commit
    return out


def fit(x, y, *, x_val, variables, meta, options, state):
    X = np.asarray(x, dtype=np.float64)
    Y = np.asarray(y, dtype=np.float64).ravel()
    names = list(variables)
    seed = run_seed(X, Y, state["seed"])
    config = build_config(state["arm"], X, Y, n_samples=state["n_samples"], seed=seed, overrides=state["config"])
    modules = state["modules"]
    Program = modules["Program"]

    started = time.perf_counter()
    model = modules["DeepSymbolicOptimizer"](config)
    model.setup()
    trainer = model.trainer
    guard_hit = False
    while not trainer.done:
        model.train_one_step()
        if not trainer.done and time.perf_counter() - started > state["max_seconds"]:
            guard_hit = True
            break
    best = trainer.p_r_best
    expression = infix(best.traversal, names)
    fit_time = time.perf_counter() - started   # the search and its answer; the bookkeeping below is not timed

    extra = {"arm": state["arm"], "seed": seed, "n_samples": state["n_samples"], "nevals": int(trainer.nevals),
             "iterations": int(trainer.iteration), "reward": float(best.r), "traversal": token_sequence(best.traversal, names),
             "model_complexity": float(best.complexity), "model_length": len(best.traversal),
             "early_stop": bool((best.evaluate or {}).get("success")), "guard_hit": guard_hit,
             "unique_expressions": len(Program.cache)}
    nmse = (best.evaluate or {}).get("nmse_test")
    extra["nmse_train"] = None if nmse is None else float(nmse)
    extra["string_deviation"] = string_deviation(expression, names, X, best.execute(X))
    gp = getattr(model, "gp_controller", None)
    if gp is not None:
        log = list(gp.algorithm.logbook)
        extra["gp_offspring"] = int(sum(record["nevals"] for record in log))
        extra["gp_new_expressions"] = int(sum(record["uncached_size"] for record in log))
    if "poly" in config["task"]["function_set"]:
        degree = int(config["task"]["poly_optimizer_params"]["degree"])
        # DSO's least squares needs a row per monomial; with fewer rows LINEAR is the constant 1
        extra["linear_underdetermined"] = _comb(X.shape[1] + degree, degree) > X.shape[0]
    try:
        extra["front"] = pareto_front(list(Program.cache.values()), names)
    except Exception:  # noqa: BLE001 - persistence is best-effort
        pass

    Program.clear_cache()
    sess = getattr(model, "sess", None)
    if sess is not None:
        sess.close()
    del model, trainer, best
    gc.collect()
    return {"expression": expression, "fit_time": fit_time, "extra": extra}


def _comb(n, k):
    """math.comb, which Python 3.7 lacks."""
    out = 1
    for i in range(k):
        out = out * (n - i) // (i + 1)
    return out
