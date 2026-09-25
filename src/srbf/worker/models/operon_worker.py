"""The Operon worker: pyoperon's SymbolicRegressor in its first author's benchmark configuration.

Runs in whichever interpreter has ``pyoperon`` (0.6.1) and scikit-learn; imports those and numpy only.

The configuration is the one Operon's first author committed for running it as a benchmark baseline (his
SRBench 2024/25 submission) without its Optuna layer: NSGA-II on (R², length), one Levenberg-Marquardt step per
individual, population 1000, maximum length 50 and depth 10, and the final model chosen from the Pareto front by
minimum description length, with the noise level taken from a random forest's in-sample error as in his wrapper.
srbf sets only what it sets for every method: the operator vocabulary (the benchmark's operators as far as
Operon has them) and the budget, a count of evaluations. See docs/models.md.

``options`` (from the config's ``model_adapter`` block):

* ``max_evaluations`` (int, 1,000,000): the budget, the compute axis of the scaling sweeps. Operon counts every
  fitness, residual and Jacobian evaluation, including those of the local search.
* ``seed`` (int, 0): mixed with a hash of the problem's data into the run's seed, so a problem's fit is
  reproducible and two draws of a law (fresh data) get different seeds.
* ``config`` (dict, none): settings that replace the author configuration's, ``allowed_symbols`` included. For
  side experiments only, such as reproducing a published configuration; a config that sets it is
  ``harness_tuned``, and published results never set it.

Operon searches in float32 and always wraps its model in a fitted scale and offset (``a * f(x) + b``) with a weight
on every variable; the returned string carries every digit of those float32 constants. Operon's own predictions
are float32, so they are not returned: srbf evaluates the string.
"""
import hashlib

import numpy as np

# The benchmark's operators Operon has: 17 natively, and rootn for n = 2 and 3 as sqrt and cbrt (both with
# rootn's semantics: principal square root, signed cube root). asinh, acosh, atanh and other roots have no
# Operon node and cannot be registered from Python.
ALLOWED_SYMBOLS = ("add,sub,mul,div,pow,abs,sin,cos,tan,asin,acos,atan,sinh,cosh,tanh,exp,log,sqrt,cbrt,"
                   "constant,variable")

# The first author's benchmark configuration (SRBench 2024/25 submission, cavalab/srbench srbench_2025,
# experiment/methods/operon/regressor.py, minus Optuna; max_length at 50, its library default and the top of
# his search range). Written out in full so a changed library default cannot change the method silently.
AUTHOR_CONFIG = {
    "objectives": ["r2", "length"],
    "optimizer": "lm",
    "optimizer_iterations": 1,
    "local_search_probability": 1.0,
    "lamarckian_probability": 1.0,
    "offspring_generator": "basic",
    "reinserter": "keep-best",
    "female_selector": "tournament",
    "male_selector": "tournament",
    "tournament_size": 5,
    "epsilon": 1e-5,
    "population_size": 1000,
    "pool_size": None,
    "brood_size": 5,
    "max_length": 50,
    "max_depth": 10,
    "initialization_method": "btc",
    "generations": 1000,
    "model_selection_criterion": "minimum_description_length",
    "add_model_scale_term": True,
    "add_model_intercept_term": True,
    "n_threads": 1,      # more threads make a run irreproducible (the evaluation budget is checked in parallel)
    "max_time": None,    # no wall-clock stop: the budget is a count; srbf's worker timeout is the guard
}
GENERATION_CAP = AUTHOR_CONFIG["generations"]


def _require_operon():
    try:
        from pyoperon.sklearn import SymbolicRegressor
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("pyoperon is required in the worker's environment: pip install pyoperon==0.6.1 scikit-learn") from exc
    return SymbolicRegressor


def run_seed(x, y, seed=0):
    """A problem's seed: its data's hash mixed with the configured seed, in [0, 2**31)."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(x, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(y, dtype=np.float64).tobytes())
    h.update(str(int(seed)).encode())
    return int.from_bytes(h.digest()[:4], "little") & 0x7FFFFFFF


def noise_level(x, y, seed):
    """The author's noise estimate for MDL: the in-sample RMSE of a random forest on the training data
    (``compute_sigma`` in his wrapper; seeded here, where his is not, so the run is reproducible)."""
    from sklearn.ensemble import RandomForestRegressor

    forest = RandomForestRegressor(random_state=seed).fit(x, y)
    return float(np.sqrt(np.mean((y - forest.predict(x)) ** 2)))


def rootn_spelling(expression):
    """Rewrite Operon's ``cbrt(A)`` as the benchmark's ``rootn(A, 3)`` (signed cube root in both)."""
    while True:
        start = expression.rfind("cbrt(")   # the rightmost call holds no other cbrt in its argument
        if start < 0:
            return expression
        depth, i = 0, start + len("cbrt")
        for i in range(start + len("cbrt"), len(expression)):
            depth += {"(": 1, ")": -1}.get(expression[i], 0)
            if depth == 0:
                break
        if depth != 0:
            raise ValueError(f"unbalanced parentheses in {expression!r}")
        argument = expression[start + len("cbrt("):i]
        expression = f"{expression[:start]}rootn({argument}, 3){expression[i + 1:]}"


def create_model(*, max_evaluations, uncertainty, random_state, config=None):
    SymbolicRegressor = _require_operon()
    params = {**AUTHOR_CONFIG, "allowed_symbols": ALLOWED_SYMBOLS, **(config or {})}
    return SymbolicRegressor(**params, max_evaluations=int(max_evaluations), uncertainty=[float(uncertainty)],
                             random_state=int(random_state))


def load(options):
    return {"max_evaluations": int(options.get("max_evaluations", 1_000_000)), "seed": int(options.get("seed", 0)),
            "config": dict(options.get("config") or {})}


def info(state):
    try:
        from importlib.metadata import version as _version
        version = _version("pyoperon")   # pyoperon has no __version__
    except Exception:  # noqa: BLE001
        version = "?"
    return {"worker": "operon", "pyoperon": version}


def fit(x, y, *, x_val, variables, meta, options, state):
    X = np.asarray(x, dtype=float)
    Y = np.asarray(y, dtype=float).ravel()
    seed = run_seed(X, Y, state["seed"])
    sigma = noise_level(X, Y, seed)
    model = create_model(max_evaluations=state["max_evaluations"], uncertainty=sigma, random_state=seed,
                         config=state.get("config"))
    model.fit(X, Y)
    names = list(variables)
    expression = rootn_spelling(model.get_model_string(model.model_, 40, names))
    extra = {"sigma": sigma, "seed": seed, "max_evaluations": state["max_evaluations"]}
    stats = dict(getattr(model, "stats_", {}) or {})
    for key in ("generations", "evaluation_count", "residual_evaluations", "jacobian_evaluations", "model_length",
                "model_complexity"):
        if key in stats:
            extra[key] = int(stats[key])
    extra["hit_generation_cap"] = bool(extra.get("generations", 0) >= GENERATION_CAP)
    try:
        extra["front"] = [
            {"expression": rootn_spelling(model.get_model_string(m["tree"], 40, names)), "length": int(m["length"]),
             "complexity": int(m["complexity"]), "mean_squared_error": float(m["mean_squared_error"]),
             "minimum_description_length": float(m["minimum_description_length"]),
             "bayesian_information_criterion": float(m["bayesian_information_criterion"]),
             "akaike_information_criterion": float(m["akaike_information_criterion"])}
            for m in model.pareto_front_]
    except Exception:  # noqa: BLE001 - persistence is best-effort
        pass
    return {"expression": expression, "extra": extra}
