"""The RILS-ROLS worker (srbf/worker/models/rilsrols_worker.py) without rils-rols: a stand-in regressor with the
constructor and the fitted surface of rils-rols 1.6.7's RILSROLSRegressor (fit; model, the C++ core's printed
model; model_string(), its sympy simplification; fit_calls, total_time, best_time; predict). Every fit runs in a
forked child, so the stand-in reports what it was given through a file."""
import json
import os
import re

import numpy as np
import pytest

from srbf.worker.models import rilsrols_worker as w

sympy = pytest.importorskip("sympy")

# numpy's BLAS threads are already running in the test process; the worker's own process forks single-threaded
pytestmark = pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")


class _Regressor:
    """rils-rols 1.6.7's RILSROLSRegressor: its constructor's arguments and defaults, and what a fit leaves. The
    model it "finds" is ``a*(x0*sin(x1)) + b`` (``a*x0`` on one variable) with a and b from least squares, printed
    as the patched C++ core prints it (``%.17g``) or, with ``rounded``, as the unpatched one does (``%f``)."""
    report_to: str = ""
    crash: bool = False
    stopped_by_time: bool = False
    rounded: bool = False
    simplified: str = ""                    # what model_string() returns instead of the parsed model, if set

    def __init__(self, max_fit_calls=100000, max_time=100, complexity_penalty=0.001, max_complexity=50, sample_size=1,
                 verbose=False, random_state=0):
        params = dict(locals())
        params.pop("self")
        self.params = params
        self.max_fit_calls = max_fit_calls
        with open(_Regressor.report_to, "w") as f:
            json.dump(params, f)

    @staticmethod
    def _terms(X):
        return X[:, [0]] if X.shape[1] == 1 else np.column_stack([X[:, 0] * np.sin(X[:, 1]), np.ones(len(X))])

    def fit(self, X, y):
        if _Regressor.crash:
            os._exit(3)                     # the C++ core's exit(1) on bad input, say
        self.coefficients = np.linalg.lstsq(self._terms(X), y, rcond=None)[0]
        number = (lambda v: "%f" % v) if _Regressor.rounded else (lambda v: "%.17g" % v)
        if X.shape[1] == 1:
            self.model = "(%s*x0)" % number(self.coefficients[0])
        else:
            self.model = "((%s*(x0*sin(x1)))+%s)" % tuple(number(c) for c in self.coefficients)
        self.model_simp = sympy.sympify(_Regressor.simplified or self.model)
        self.fit_calls = self.max_fit_calls // 3 if _Regressor.stopped_by_time else self.max_fit_calls + 1
        self.total_time, self.best_time = 1.5, 0.5
        return self

    def predict(self, X):
        return self._terms(np.asarray(X)) @ self.coefficients

    def model_string(self):
        return self.model_simp


@pytest.fixture
def fake_rilsrols(monkeypatch, tmp_path):
    monkeypatch.setattr(w, "_require_rilsrols", lambda: _Regressor)
    monkeypatch.setattr(_Regressor, "report_to", str(tmp_path / "params.json"))
    for flag, value in (("crash", False), ("stopped_by_time", False), ("rounded", False), ("simplified", "")):
        monkeypatch.setattr(_Regressor, flag, value)
    return _Regressor


A, B = 0.123456789012345678, -6.674e-11


def _data():
    rng = np.random.default_rng(0)
    X = rng.uniform(1, 2, size=(64, 2))
    return X, A * X[:, 0] * np.sin(X[:, 1]) + B


def _params():
    with open(_Regressor.report_to) as f:
        return json.load(f)


def _fit(X, y, state):
    return w.fit(X.tolist(), y.tolist(), x_val=[], variables=["v1", "v2"], meta={}, options={}, state=state)


def test_the_author_configuration_the_budget_and_the_seed_reach_the_regressor(fake_rilsrols):
    X, y = _data()
    out = _fit(X, y, w.load({"max_fit_calls": 65536}))
    params = _params()
    assert params["max_complexity"] == 50 and params["sample_size"] == 0 and params["complexity_penalty"] == 0.001
    assert params["verbose"] is False and params["max_time"] == w.TIME_GUARD == 3600
    assert params["max_fit_calls"] == 65536 and params["random_state"] == w.run_seed(X, y, 0) == out["extra"]["seed"]
    extra = out["extra"]
    assert extra["fit_calls"] == 65537 and extra["max_fit_calls"] == 65536 and extra["hit_time_guard"] is False
    assert extra["total_time"] == 1.5 and extra["best_time"] == 0.5 and out["fit_time"] >= 0
    assert w.load({})["max_fit_calls"] == 1_000_000


def test_a_run_the_time_guard_stopped_is_marked(fake_rilsrols, monkeypatch):
    X, y = _data()
    monkeypatch.setattr(_Regressor, "stopped_by_time", True)
    out = _fit(X, y, w.load({"max_fit_calls": 65536}))
    assert out["extra"]["hit_time_guard"] is True and out["extra"]["fit_calls"] < 65536
    assert "expression" in out            # the model it had is still the answer


def test_the_answer_carries_every_digit_and_the_benchmark_s_variable_names(fake_rilsrols):
    X, y = _data()
    out = _fit(X, y, w.load({}))
    a, b = sorted((float(c) for c in sympy.sympify(out["extra"]["model_string"]).atoms(sympy.Float)), key=abs, reverse=True)
    assert a == pytest.approx(A, rel=1e-13) and b == pytest.approx(B, rel=1e-6)
    assert out["expression"] == "%r*v1*sin(v2) - %r" % (a, -b)
    assert out["extra"]["answer_form"] == "simplified" and out["extra"]["string_deviation"] < 1e-15
    assert "y_pred" not in out
    values = eval(out["expression"], {"sin": np.sin, "v1": X[:, 0], "v2": X[:, 1]})
    assert np.allclose(values, y, rtol=1e-12, atol=0)


def test_an_environment_without_the_precision_patch_is_refused(fake_rilsrols, monkeypatch):
    monkeypatch.setattr(_Regressor, "rounded", True)
    with pytest.raises(RuntimeError, match="prints its constants rounded"):
        w.load({})


def test_the_benchmark_string_renames_every_variable_and_prints_doubles_exactly():
    x = sympy.symbols("x0:12")
    names = ["v%d" % (i + 1) for i in range(12)]
    expression = sympy.Float("0.10000000000000001", 17) * x[10] + x[1] ** 2 / sympy.sqrt(x[0]) - sympy.exp(x[11]) \
        + sympy.log(x[2]) * sympy.cos(x[3]) + sympy.Float("1e+20") * sympy.Abs(x[4]) + sympy.E
    text = w.benchmark_string(expression, names)
    assert "0.1*v11" in text and "1e+20*Abs(v5)" in text and "v12" in text and " E " in text
    assert not re.search(r"x\d", text)
    X = np.random.default_rng(1).uniform(1, 2, size=(8, 12))
    reference = sympy.lambdify(x, expression, "numpy")(*X.T)
    assert w.string_deviation(text, names, X, reference) < 1e-15
    assert w.benchmark_string(sympy.Float(-0.5) * x[0] ** sympy.Float(-0.5), ["v1"]) == "-0.5/v1**0.5"


def test_what_the_benchmark_cannot_state_is_refused():
    x0, x1 = sympy.symbols("x0 x1")
    with pytest.raises(ValueError, match="cannot state erf"):
        w.benchmark_string(sympy.erf(x0), ["v1"])
    with pytest.raises(ValueError, match="x1"):
        w.benchmark_string(x0 + x1, ["v1"])
    with pytest.raises(ValueError, match="finite real"):
        w.benchmark_string(sympy.sqrt(sympy.Integer(-2)) * x0, ["v1"])
    with pytest.raises(ValueError, match="finite real"):
        w.benchmark_string(sympy.zoo * x0, ["v1"])


def test_an_unstatable_simplification_falls_back_to_the_printed_model(fake_rilsrols, monkeypatch):
    X, y = _data()
    monkeypatch.setattr(_Regressor, "simplified", "sinc(x0)")
    out = _fit(X, y, w.load({}))
    assert out["extra"]["answer_form"] == "unsimplified" and "sinc" not in out["expression"]
    assert out["extra"]["string_deviation"] < 1e-15


def test_a_string_the_method_did_not_compute_shows_in_the_deviation():
    X = np.linspace(1, 2, 5).reshape(-1, 1)
    assert w.string_deviation("v1", ["v1"], X, X[:, 0]) == 0.0
    assert w.string_deviation("v1 + 0.5", ["v1"], X, X[:, 0]) == pytest.approx(0.25)
    assert w.string_deviation("log(v1 - 1.5)", ["v1"], X, X[:, 0]) == float("inf")   # NaN where the method has values
    assert w.string_deviation("not python", ["v1"], X, X[:, 0]) is None


def test_a_problem_gets_one_seed_from_its_data_and_the_configured_seed(fake_rilsrols):
    X, y = _data()
    assert w.run_seed(X, y, 0) == w.run_seed(X.copy(), y.copy(), 0)
    assert w.run_seed(X, y, 0) != w.run_seed(X, y + 1e-9, 0)       # a fresh draw of the law: a different run
    assert w.run_seed(X, y, 0) != w.run_seed(X, y, 1)
    _fit(X, y, w.load({"seed": 1}))
    assert _params()["random_state"] == w.run_seed(X, y, 1)


def test_a_side_experiment_replaces_settings_and_the_default_leaves_them(fake_rilsrols):
    X, y = _data()
    _fit(X, y, w.load({"config": {"complexity_penalty": 0.01, "max_complexity": 25}}))
    params = _params()
    assert params["complexity_penalty"] == 0.01 and params["max_complexity"] == 25 and params["sample_size"] == 0
    for bad in ({"max_complexiti": 10}, {"random_state": 3}, {"max_fit_calls": 10}, {"max_time": 10}):
        with pytest.raises(ValueError, match="config may replace"):
            w.load({"config": bad})
    with pytest.raises(ValueError, match="max_fit_calls"):
        w.load({"max_fit_calls": 0})


def test_a_crash_of_the_method_costs_one_problem_and_the_next_fit_runs(fake_rilsrols, monkeypatch):
    X, y = _data()
    state = w.load({})
    monkeypatch.setattr(_Regressor, "crash", True)
    out = _fit(X, y, state)
    assert "without a result" in out["error"] and "exit code 3" in out["error"]
    monkeypatch.setattr(_Regressor, "crash", False)
    assert "expression" in _fit(X, y, state)


def test_the_stand_in_matches_the_real_regressor():
    rils_rols = pytest.importorskip("rils_rols.rils_rols")
    import inspect

    accepted = set(inspect.signature(rils_rols.RILSROLSRegressor.__init__).parameters)
    assert set(w.AUTHOR_CONFIG) | set(w.SET_PER_FIT) <= accepted
    assert inspect.signature(_Regressor.__init__) == inspect.signature(rils_rols.RILSROLSRegressor.__init__)
