"""The Operon worker (srbf/worker/models/operon_worker.py) without pyoperon: a stand-in regressor with the
attributes pyoperon 0.6.1's SymbolicRegressor has after fit (model_, pareto_front_ entries, stats_,
get_model_string(model, precision, names))."""
import numpy as np
import pytest

from srbf.worker.models import operon_worker as w


class _Regressor:
    """pyoperon 0.6.1's fitted surface (pyoperon/sklearn.py: fit -> model_, pareto_front_, stats_)."""
    instances: list = []

    def __init__(self, **params):
        self.params = params
        _Regressor.instances.append(self)

    def fit(self, X, y):
        tree = "T0"
        self.model_ = tree
        self.pareto_front_ = [{"tree": tree, "length": 7, "complexity": 9, "mean_squared_error": 1e-6,
                               "minimum_description_length": -120.0, "bayesian_information_criterion": -90.0,
                               "akaike_information_criterion": -95.0}]
        self.stats_ = {"model_length": 7, "model_complexity": 9, "generations": 3, "evaluation_count": 4096,
                       "residual_evaluations": 3000, "jacobian_evaluations": 1000}
        return self

    def get_model_string(self, model, precision=3, names=None):
        assert precision == 40 and names is not None
        return f"(0.5 + (2.0 * cbrt(({names[0]} * {names[1]}))))"


@pytest.fixture
def fake_operon(monkeypatch):
    _Regressor.instances = []
    monkeypatch.setattr(w, "_require_operon", lambda: _Regressor)
    return _Regressor


def _data():
    rng = np.random.default_rng(0)
    X = rng.uniform(1, 2, size=(64, 2))
    return X, X[:, 0] * X[:, 1]


def test_the_author_configuration_the_vocabulary_and_the_budget_reach_the_regressor(fake_operon):
    X, y = _data()
    state = w.load({"max_evaluations": 4096})
    out = w.fit(X, y, x_val=[], variables=["v1", "v2"], meta={}, options={}, state=state)
    params = fake_operon.instances[-1].params
    assert params["objectives"] == ["r2", "length"] and params["optimizer_iterations"] == 1
    assert params["population_size"] == 1000 and params["max_length"] == 50 and params["max_depth"] == 10
    assert params["model_selection_criterion"] == "minimum_description_length"
    assert params["n_threads"] == 1 and params["max_time"] is None
    assert params["max_evaluations"] == 4096
    assert params["uncertainty"] == [out["extra"]["sigma"]] and out["extra"]["sigma"] >= 0
    symbols = params["allowed_symbols"].split(",")
    assert {"cbrt", "sqrt", "exp", "log", "asin", "tanh"} <= set(symbols)
    assert not {"aq", "logabs", "sqrtabs", "square", "fmin", "fmax"} & set(symbols)


def test_the_answer_is_spelled_in_the_benchmark_vocabulary_and_carries_the_front(fake_operon):
    X, y = _data()
    out = w.fit(X, y, x_val=[], variables=["v1", "v2"], meta={}, options={}, state=w.load({}))
    assert out["expression"] == "(0.5 + (2.0 * rootn((v1 * v2), 3)))"
    assert "y_pred" not in out          # Operon predicts in float32: srbf evaluates the string
    extra = out["extra"]
    assert extra["generations"] == 3 and extra["evaluation_count"] == 4096 and not extra["hit_generation_cap"]
    assert extra["front"][0]["expression"].startswith("(0.5") and extra["front"][0]["length"] == 7


def test_a_problem_gets_one_seed_from_its_data_and_the_configured_seed(fake_operon):
    X, y = _data()
    assert w.run_seed(X, y, 0) == w.run_seed(X.copy(), y.copy(), 0)
    assert w.run_seed(X, y, 0) != w.run_seed(X, y + 1e-9, 0)       # a fresh draw of the law: a different run
    assert w.run_seed(X, y, 0) != w.run_seed(X, y, 1)
    w.fit(X, y, x_val=[], variables=["v1", "v2"], meta={}, options={}, state=w.load({"seed": 1}))
    assert fake_operon.instances[-1].params["random_state"] == w.run_seed(X, y, 1)


def test_a_side_experiment_replaces_settings_and_the_default_leaves_them(fake_operon):
    X, y = _data()
    w.fit(X, y, x_val=[], variables=["v1", "v2"], meta={}, options={}, state=w.load({}))
    assert fake_operon.instances[-1].params["optimizer_iterations"] == 1
    srbench_2021 = {"objectives": ["r2"], "optimizer_iterations": 5, "population_size": 500, "pool_size": 500,
                    "allowed_symbols": "add,sub,mul,div,exp,log,sin,cos,sqrt,square,constant,variable"}
    w.fit(X, y, x_val=[], variables=["v1", "v2"], meta={}, options={}, state=w.load({"config": srbench_2021}))
    params = fake_operon.instances[-1].params
    assert params["objectives"] == ["r2"] and params["optimizer_iterations"] == 5 and params["pool_size"] == 500
    assert "square" in params["allowed_symbols"] and params["max_length"] == 50


def test_nested_cube_roots_are_rewritten_inside_out():
    assert w.rootn_spelling("cbrt((x1 + cbrt(x2)))") == "rootn((x1 + rootn(x2, 3)), 3)"
    assert w.rootn_spelling("sqrt(x1)") == "sqrt(x1)"
    with pytest.raises(ValueError, match="unbalanced"):
        w.rootn_spelling("cbrt((x1)")


def test_the_configuration_names_only_arguments_the_real_regressor_takes():
    pyoperon_sklearn = pytest.importorskip("pyoperon.sklearn")
    import inspect

    accepted = set(inspect.signature(pyoperon_sklearn.SymbolicRegressor.__init__).parameters)
    assert set(w.AUTHOR_CONFIG) | {"allowed_symbols", "max_evaluations", "uncertainty", "random_state"} <= accepted
