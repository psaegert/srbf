"""The GP-GOMEA worker (srbf/worker/models/gpgomea_worker.py) without pyGPGOMEA: a stand-in regressor with the
constructor and the fitted surface of pyGPGOMEA's GPGOMEARegressor at GP-GOMEA 6a92cb6 (fit, get_model,
get_evaluations, get_n_nodes, predict). Every fit runs in a forked child, so the stand-in reports what it was
given through a file."""
import json
import os

import numpy as np
import pytest

from srbf.worker.models import gpgomea_worker as w

# numpy's BLAS threads are already running in the test process; the worker's own process forks single-threaded
pytestmark = pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")


def _method_value(X):
    """What GP-GOMEA computes for the stand-in's inner model ``((x0p/x1)+plog(x0))``."""
    u, v = X[:, 0], X[:, 1]
    division = np.where(v < 0, -1.0, 1.0) * (u / (1e-6 + np.abs(v)))
    return division + np.log(np.abs(u))


class _Regressor:
    """pyGPGOMEA's GPGOMEARegressor at 6a92cb6: its constructor's arguments and defaults, and what a fit leaves."""
    report_to: str = ""
    crash: bool = False
    stopped_by_time: bool = False
    intercept, slope = 0.123456789012345, -2.34567890123456

    def __init__(self, time=60, generations=-1, evaluations=-1, prob='symbreg', linearscaling=True,
                 functions='+_*_-_aq', erc=True, classweights=False, gomea=True, gomfos='LT', subcross=0.5, submut=0.5,
                 reproduction=0.0, sblibtype=False, sbrdo=0.0, sbagx=0.0, unifdepthvar=True, tournament=4, elitism=1,
                 ims='5_1', syntuniqinit=1000, popsize=500, initmaxtreeheight=4, maxtreeheight=17, maxsize=1000,
                 seed=-1, parallel=1, caching=False, silent=True):
        params = dict(locals())
        params.pop("self")
        self.params = params
        with open(_Regressor.report_to, "w") as f:
            json.dump(params, f)

    def fit(self, X, y):
        if _Regressor.crash:
            os._exit(3)                     # the C++ code taking the process down
        assert X.dtype == np.float64 and X.flags["C_CONTIGUOUS"] and y.ndim == 1
        self.X = X
        return self

    def get_model(self):                    # a and b with six decimals, as std::to_string prints them
        return "%f+%f*(((x0p/x1)+plog(x0)))" % (_Regressor.intercept, _Regressor.slope)

    def get_evaluations(self):              # the budget is checked between generations: a run overshoots it
        budget = self.params["evaluations"]
        return budget // 3 if _Regressor.stopped_by_time else budget + 1500

    def get_n_nodes(self):
        return 6

    def predict(self, X):
        return _Regressor.intercept + _Regressor.slope * _method_value(X)


@pytest.fixture
def fake_gpgomea(monkeypatch, tmp_path):
    monkeypatch.setattr(w, "_require_gpgomea", lambda: _Regressor)
    monkeypatch.setattr(_Regressor, "report_to", str(tmp_path / "params.json"))
    monkeypatch.setattr(_Regressor, "crash", False)
    monkeypatch.setattr(_Regressor, "stopped_by_time", False)
    return _Regressor


def _data():
    rng = np.random.default_rng(0)
    X = rng.uniform(1, 2, size=(64, 2))
    y = _Regressor.intercept + _Regressor.slope * _method_value(X)
    return X, y


def _params():
    with open(_Regressor.report_to) as f:
        return json.load(f)


def _fit(X, y, state):
    return w.fit(X.tolist(), y.tolist(), x_val=[], variables=["v1", "v2"], meta={}, options={}, state=state)


def test_the_author_configuration_the_operators_and_the_budget_reach_the_regressor(fake_gpgomea):
    X, y = _data()
    out = _fit(X, y, w.load({"max_evaluations": 65536}))
    params = _params()
    assert params["gomea"] is True and params["gomfos"] == "LT" and params["ims"] is False
    assert params["popsize"] == 500 and params["initmaxtreeheight"] == 4 and params["elitism"] == 1
    assert params["linearscaling"] is True and params["erc"] is True and params["parallel"] is False
    assert params["generations"] == -1 and params["time"] == w.TIME_GUARD
    assert params["evaluations"] == 65536 and params["seed"] == w.run_seed(X, y, 0) == out["extra"]["seed"]
    assert params["functions"].split("_") == ["+", "-", "*", "p/", "plog", "sqrt", "exp", "sin", "cos", "^2"]
    assert out["extra"]["evaluations"] == 65536 + 1500 and out["extra"]["max_evaluations"] == 65536
    assert out["extra"]["hit_time_guard"] is False


def test_a_run_the_wall_clock_limit_stopped_is_marked(fake_gpgomea, monkeypatch):
    X, y = _data()
    monkeypatch.setattr(_Regressor, "stopped_by_time", True)
    out = _fit(X, y, w.load({"max_evaluations": 65536}))
    assert out["extra"]["hit_time_guard"] is True and out["extra"]["evaluations"] < 65536
    assert "expression" in out            # the elitist it had is still the answer


def test_the_answer_states_the_method_s_function_with_the_intercept_and_slope_at_full_precision(fake_gpgomea):
    X, y = _data()
    out = _fit(X, y, w.load({}))
    extra = out["extra"]
    assert extra["printed_intercept"] == 0.123457 and extra["printed_slope"] == -2.345679
    assert extra["intercept"] == pytest.approx(_Regressor.intercept, rel=1e-12)
    assert extra["slope"] == pytest.approx(_Regressor.slope, rel=1e-12)
    assert extra["predict_max_relative_deviation"] < 1e-12
    assert out["expression"] == "%r + (%r)*((v1/(v2 + 1e-06)) + log(abs(v1)))" % (extra["intercept"], extra["slope"])
    assert "y_pred" not in out and out["fit_time"] >= 0
    values = eval(out["expression"].replace("log", "np.log").replace("abs", "np.abs"),
                  {"np": np, "v1": X[:, 0], "v2": X[:, 1]})
    assert np.allclose(values, y, rtol=1e-12, atol=0)


def test_protected_operators_are_written_as_what_the_method_computes():
    X = np.array([[2.0, 0.5], [-3.0, 1.5], [0.7, 4.0]])
    zero = ("-", ("var", 0), ("var", 0))
    cases = {
        ("p/", ("var", 0), ("var", 1)): "(v1/(v2 + 1e-06))",                              # divisor >= 0
        ("p/", ("var", 0), ("const", -2.0)): "(v1/((-2.0) - 1e-06))",                     # divisor < 0
        ("p/", ("var", 1), ("var", 0)): "(v2/(v1*(1 + 1e-06/abs(v1))))",                  # both signs
        ("p/", ("var", 1), zero): "(v2/((v1 - v1) + 1e-06))",                             # divisor identically 0
        ("plog", ("var", 0)): "log(abs(v1))",
        ("plog", zero): "0",                                                              # log|0| is 0 here
        ("sqrt", ("var", 0)): "sqrt(abs(v1))",
        ("^2", ("var", 0)): "(v1)**2",
        ("aq", ("var", 0), ("var", 1)): "(v1/sqrt(1 + v2**2))",
    }
    namespace = {"np": np, "v1": X[:, 0], "v2": X[:, 1], "abs": np.abs, "log": np.log, "sqrt": np.sqrt}
    for tree, spelling in cases.items():
        text, values = w.render(tree, X, ["v1", "v2"])
        assert text == spelling
        written = eval(text, dict(namespace)) * np.ones(len(X))
        assert np.allclose(written, values, rtol=1e-12, atol=0), (text, written, values)
    # the method's own values, independent of the rendering
    assert w.render(("p/", ("var", 1), zero), X, ["v1", "v2"])[1] == pytest.approx(X[:, 1] / 1e-6)
    assert list(w.render(("plog", zero), X, ["v1", "v2"])[1]) == [0.0, 0.0, 0.0]


def test_the_reader_takes_gp_gomea_s_printing_apart():
    a, b, tree = w.read_model("-0.000000+1.000000*((((x10--0.731000)*sqrt(x2))p/(x0)^2))")
    assert (a, b) == (-0.0, 1.0)
    assert tree == ("p/", ("*", ("-", ("var", 10), ("const", -0.731)), ("sqrt", ("var", 2))), ("^2", ("var", 0)))
    assert w.read_model("(x0aqx1)", linear_scaling=False)[2] == ("aq", ("var", 0), ("var", 1))
    assert w.read_model("1.5", linear_scaling=False)[2] == ("const", 1.5)
    for broken in ("1.0+2.0*((x0+x1)", "1.0+2.0*((x0?x1))", "1.0+2.0*(x0) trailing"):
        with pytest.raises(ValueError, match="cannot read"):
            w.read_model(broken)
    with pytest.raises(ValueError, match="x2"):
        w.render(("var", 2), np.ones((3, 2)), ["v1", "v2"])


def test_a_constant_model_is_the_mean_of_the_targets():
    X = np.ones((4, 1))
    y = np.array([1.0, 2.0, 3.0, 6.0])
    expression, record, values = w.expression_from_model("2.999999+0.000000*(3.123000)", X, y, ["v1"])
    assert expression == "3.0" and record["slope"] == 0.0 and list(values) == [3.0] * 4


def test_a_problem_gets_one_seed_from_its_data_and_the_configured_seed(fake_gpgomea):
    X, y = _data()
    assert w.run_seed(X, y, 0) == w.run_seed(X.copy(), y.copy(), 0)
    assert w.run_seed(X, y, 0) != w.run_seed(X, y + 1e-9, 0)       # a fresh draw of the law: a different run
    assert w.run_seed(X, y, 0) != w.run_seed(X, y, 1)
    _fit(X, y, w.load({"seed": 1}))
    assert _params()["seed"] == w.run_seed(X, y, 1)


def test_a_side_experiment_replaces_settings_and_the_default_leaves_them(fake_gpgomea):
    X, y = _data()
    _fit(X, y, w.load({}))
    assert _params()["popsize"] == 500 and _params()["evaluations"] == 500_000
    srbench_2021 = {"popsize": 1000, "initmaxtreeheight": 6, "functions": "+_-_*_p/_plog_sqrt_sin_cos", "time": 28800}
    _fit(X, y, w.load({"config": srbench_2021, "max_evaluations": 1_000_000}))
    params = _params()
    assert params["popsize"] == 1000 and params["initmaxtreeheight"] == 6 and params["time"] == 28800
    assert params["functions"] == "+_-_*_p/_plog_sqrt_sin_cos" and params["evaluations"] == 1_000_000
    assert params["ims"] is False and params["gomea"] is True
    for bad in ({"popsiz": 10}, {"seed": 3}, {"evaluations": 10}):
        with pytest.raises(ValueError, match="config may replace"):
            w.load({"config": bad})
    with pytest.raises(ValueError, match="cannot state GP-GOMEA's aq0.1, and"):
        w.load({"config": {"functions": "+_-_aq0.1_and"}})


def test_a_crash_of_the_method_costs_one_problem_and_the_next_fit_runs(fake_gpgomea, monkeypatch):
    X, y = _data()
    monkeypatch.setattr(_Regressor, "crash", True)
    out = _fit(X, y, w.load({}))
    assert "without a result" in out["error"] and "exit code 3" in out["error"]
    monkeypatch.setattr(_Regressor, "crash", False)
    assert "expression" in _fit(X, y, w.load({}))


def test_the_configuration_names_only_arguments_the_real_regressor_takes():
    pygpgomea = pytest.importorskip("pyGPGOMEA")
    import inspect

    accepted = set(inspect.signature(pygpgomea.GPGOMEARegressor.__init__).parameters)
    assert set(w.AUTHOR_CONFIG) | set(w.SET_PER_FIT) <= accepted
    assert set(inspect.signature(_Regressor.__init__).parameters) == accepted
