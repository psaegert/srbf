"""The QLattice worker (srbf/worker/models/qlattice_worker.py) without feyn: a stand-in with the surface feyn 3.5.0
has (QLattice.update_priors/sample_models/update, validate_data, tools.estimate_priors, fit_models, prune_models,
get_diverse_models; a Model's fnames, names, params, loss_value, wide_parsimony, bic, edge_count and protected
predict). The tests at the end run against the real package where it is installed (the worker's own environment)
and check the stand-in against it."""
import importlib.util
import inspect
import signal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PATH = Path(__file__).resolve().parents[2] / "src" / "srbf" / "worker" / "models" / "qlattice_worker.py"
_spec = importlib.util.spec_from_file_location("qlattice_worker", _PATH)
w = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(w)     # by path: the worker imports neither srbf nor its dependencies

LIMITS = dict(w.LIMITS)


# -- the stand-in ------------------------------------------------------------------------------------------------

def _arity(fname):
    return int(fname.split(":")[1])


def _protected(fname, params, *args):
    """feyn 3.5.0's FNAME_MAP (feyn/_functions.py), the protected variants predict uses."""
    v = np.array(args[0], dtype=float)
    if fname == "exp:1":
        return np.exp(np.minimum(v, LIMITS["EXP_MAX"]))
    if fname == "log:1":
        return np.log(np.maximum(v, LIMITS["LOG_MIN"]))
    if fname == "sqrt:1":
        return np.sqrt(np.maximum(v, LIMITS["SQRT_MIN"]))
    if fname == "squared:1":
        v = np.minimum(v, LIMITS["SQUARED_MAX"])
        return v * v
    if fname == "inverse:1":
        k = LIMITS["DIVISOR_ABSMIN"]
        return 1 / np.where(np.abs(v) < k, np.where(v > 0, k, -k), v)
    if fname == "tanh:1":
        return np.tanh(v)
    if fname == "linear:1":
        return v * params["w"] + params["bias"]
    if fname == "gaussian:1":
        return np.exp(-(v * v / 0.5))
    if fname == "gaussian:2":
        return np.exp(-(v * v / 0.5 + args[1] * args[1] / 0.5))
    if fname == "add:2":
        return v + args[1]
    if fname == "multiply:2":
        return v * args[1]
    if fname == "out-linear:1":
        return (v * params["w"] + params["bias"]) * params["scale"]
    raise ValueError(fname)


class _Model:
    """A fitted feyn.Model: prefix arrays ``fnames``/``names``/``params`` (the output first) and its scores."""

    def __init__(self, codes, params, loss=1e-6):
        self.names = [c if ":" not in c else "" for c in codes]
        self.fnames = ["out-linear:1"] + [c if ":" in c else "in-linear:0" for c in codes[1:]]
        self.params = params
        self.loss_value, self.wide_parsimony, self.bic = loss, -100.0 + loss, -90.0
        self.edge_count = len(codes) - 1

    def _value(self, ix, data):
        fname = self.fnames[ix]
        if fname == "in-linear:0":
            p = self.params[ix]
            return (data[self.names[ix]].values - p["scale_offset"]) * p["scale"] * p["w"] + p["bias"]
        kids = w._children(self.fnames, ix)
        return _protected(fname, self.params[ix], *[self._value(k, data) for k in kids])

    def predict(self, X):
        return self._value(0, X)


def _inp(offset=0.0, scale=1.0, weight=1.0, bias=0.0):
    return {"scale": scale, "scale_offset": offset, "w": weight, "bias": bias, "detect_scale": 0}


OUT = {"scale": 1.7, "scale_offset": 0.0, "w": 0.9, "bias": -0.3, "detect_scale": 0}


def _product_model(loss=1e-6):
    """y = 1.7*(0.9*((0.5*x0 + 0.1)*(2*x1 - 0.3)) - 0.3)."""
    return _Model(["y", "multiply:2", "x0", "x1"], [dict(OUT), {}, _inp(0.0, 1.0, 0.5, 0.1), _inp(0.0, 2.0, 1.0, -0.3)],
                  loss=loss)


class _Feyn:
    """The feyn module's surface the worker touches."""
    __version__ = "3.5.0"
    __git_sha__ = "stand-in"

    def __init__(self, models=None, raise_guard_at=None):
        self.calls = {"sample_models": [], "fit_models": [], "validate_data": [], "seeds": []}
        self.returned = models if models is not None else [_product_model(), _product_model(2e-6)]
        self.raise_guard_at = raise_guard_at
        feyn = self

        class QLattice:
            def __init__(self, random_seed=-1):
                feyn.calls["seeds"].append(random_seed)

            def update_priors(self, priors, reset=True):
                self.priors = priors

            def sample_models(self, input_names, output_name, kind="auto", stypes=None, max_complexity=10,
                              query_string=None, function_names=None):
                feyn.calls["sample_models"].append(dict(kind=kind, stypes=stypes, max_complexity=max_complexity,
                                                        query_string=query_string, function_names=function_names))
                return list(feyn.returned)

            def update(self, models):
                pass

        class tools:
            @staticmethod
            def estimate_priors(df, output_name, floor=0.1):
                return {c: 1.0 for c in df.columns if c != output_name}

            @staticmethod
            def infer_available_threads():
                return 11

        self.QLattice, self.tools = QLattice, tools

    def validate_data(self, data, kind, output_name, stypes={}):
        self.calls["validate_data"].append(stypes)
        for column in data.columns:     # feyn 3.5.0 indexes stypes per column: None fails with a TypeError
            try:
                stypes[column]
            except KeyError:
                pass

    def fit_models(self, models, data, loss_function=None, criterion=None, n_samples=None, sample_weights=None,
                   threads=4, immutable=False):
        self.calls["fit_models"].append(dict(loss_function=loss_function, criterion=criterion, n_samples=n_samples,
                                             threads=threads, n_models=len(models)))
        if self.raise_guard_at is not None and len(self.calls["fit_models"]) == self.raise_guard_at:
            raise w.TimeGuard
        return sorted(models, key=lambda m: m.wide_parsimony)

    def prune_models(self, models, keep_n=None):
        return models[:5]

    def get_diverse_models(self, models, n=10):
        return models[:n]


@pytest.fixture
def fake(monkeypatch):
    feyn = _Feyn()
    monkeypatch.setattr(w, "_require_feyn", lambda: feyn)
    monkeypatch.setattr(w, "core_limits", lambda: dict(LIMITS))
    return feyn


def _data(n=64):
    rng = np.random.default_rng(0)
    X = rng.uniform(1, 2, size=(n, 2))
    return X, X[:, 0] * X[:, 1]


def _fit(state, n=64, x_val=None):
    X, y = _data(n)
    XV = X[:8] + 0.5 if x_val is None else x_val
    return w.fit(X.tolist(), y.tolist(), x_val=XV.tolist(), variables=["v1", "v2"], meta={}, options={}, state=state)


# -- the configuration and the loop ------------------------------------------------------------------------------

def test_the_author_configuration_the_vocabulary_and_the_budget_reach_feyn(fake):
    out = _fit(w.load({"n_epochs": 3}))
    assert len(fake.calls["sample_models"]) == len(fake.calls["fit_models"]) == 3 and out["extra"]["epochs"] == 3
    sample = fake.calls["sample_models"][0]
    assert sample == {"kind": "regression", "stypes": None, "max_complexity": 10, "query_string": None,
                      "function_names": w.FUNCTIONS}
    assert "gaussian" not in w.FUNCTIONS and {"add", "multiply", "squared", "sqrt", "inverse", "exp", "log", "tanh",
                                              "linear"} == set(w.FUNCTIONS)
    assert all(c == {"loss_function": None, "criterion": "wide_parsimony", "n_samples": None, "threads": 1,
                     "n_models": c["n_models"]} for c in fake.calls["fit_models"])
    assert fake.calls["validate_data"] == [{}]      # the crash fix: feyn 3.5.0 cannot validate with stypes=None
    assert out["extra"]["models_tried"] == 3 * 2 and out["extra"]["hit_time_guard"] is False
    assert w.load({})["n_epochs"] == w.DEFAULT_EPOCHS == 200 and w.load({})["max_time"] == w.TIME_GUARD == 3600


def test_a_side_experiment_replaces_settings_and_bad_options_fail_at_load(fake):
    _fit(w.load({"n_epochs": 1, "config": {"function_names": None}}))
    assert fake.calls["sample_models"][-1]["function_names"] is None     # SRBench's run: every feyn function
    with pytest.raises(ValueError, match="config may replace"):
        w.load({"config": {"n_epochs": 5}})
    with pytest.raises(ValueError, match="at least 1"):
        w.load({"n_epochs": 0})


def test_a_problem_gets_one_seed_from_its_data_and_the_configured_seed(fake):
    X, y = _data()
    assert w.run_seed(X, y, 0) == w.run_seed(X.copy(), y.copy(), 0)
    assert w.run_seed(X, y, 0) != w.run_seed(X, y + 1e-9, 0)       # a fresh draw of the law: a different run
    assert w.run_seed(X, y, 0) != w.run_seed(X, y, 1)
    out = _fit(w.load({"n_epochs": 1, "seed": 1}))
    assert fake.calls["seeds"] == [w.run_seed(X, y, 1)] == [out["extra"]["seed"]]


def test_the_time_guard_returns_the_best_so_far_and_leaves_no_alarm_behind(fake, monkeypatch):
    fake.raise_guard_at = 3
    before = signal.getsignal(signal.SIGALRM)
    out = _fit(w.load({"n_epochs": 10, "max_time": 50}))
    assert out["extra"]["hit_time_guard"] is True and out["extra"]["epochs"] == 2 and "expression" in out
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0) and signal.getsignal(signal.SIGALRM) == before


def test_no_model_is_an_error_not_a_crash(fake):
    fake.returned = []
    out = _fit(w.load({"n_epochs": 1}))
    assert "error" in out and "no model" in out["error"] and "expression" not in out


# -- the answer --------------------------------------------------------------------------------------------------

def _evaluate(expression, X, names=("v1", "v2")):
    namespace = {"exp": np.exp, "log": np.log, "sqrt": np.sqrt, "tanh": np.tanh, "abs": np.abs}
    namespace.update({n: X[:, j] for j, n in enumerate(names)})
    with np.errstate(all="ignore"):
        return np.broadcast_to(np.asarray(eval(expression, {"__builtins__": {}}, namespace), dtype=float), (X.shape[0],))


def test_the_answer_is_the_first_model_at_full_precision_in_the_given_names(fake):
    X, _ = _data()
    out = _fit(w.load({"n_epochs": 1}))
    assert out["expression"] == "%r*((0.5*v1 + 0.1)*(2.0*v2 - 0.3)) - %r" % (0.9 * 1.7, 0.3 * 1.7)
    reference = _product_model().predict(pd.DataFrame({"x0": X[:, 0], "x1": X[:, 1]}))
    assert np.allclose(_evaluate(out["expression"], X), reference, rtol=1e-15, atol=1e-15)
    extra = out["extra"]
    assert extra["string_deviation"] < 1e-15 and extra["string_deviation_val"] < 1e-15 and extra["protections"] == []
    assert extra["loss"] == 1e-6 and extra["edge_count"] == 3 and len(extra["diverse"]) == 1
    assert extra["diverse"][0]["loss"] == 2e-6 and extra["diverse"][0]["expression"] == out["expression"]
    assert "y_pred" not in out and out["fit_time"] >= 0


def test_the_input_map_keeps_every_digit():
    m = _Model(["y", "x0"], [dict(OUT), _inp(0.1234567890123456, 1.9876543210987654, 0.3333333333333333, 1 / 7)])
    X = np.linspace(-2, 2, 11).reshape(-1, 1)
    text, internal, values, _ = w.model_expression(m, X, ["x0"], ["v1"], LIMITS)
    a = 1.9876543210987654 * 0.3333333333333333
    assert repr(a) in text and repr(1 / 7 - 0.1234567890123456 * a) in text
    assert w.string_deviation(internal, 1, X, m.predict(pd.DataFrame({"x0": X[:, 0]}))) < 1e-15


PROTECTED = {   # (function, input map): each clip active somewhere on [-3, 3]
    "log": ("log:1", _inp(0.1, 1.3, 0.7, 0.2)),
    "sqrt": ("sqrt:1", _inp(0.1, 1.3, 0.7, 0.2)),
    "exp": ("exp:1", _inp(0.1, 1.3, 1.7, 2.2)),
    "squared": ("squared:1", _inp(0.1, 4.3, 1.7, 2.2)),
    "inverse": ("inverse:1", _inp(0.1, 1.3, 0.7, 0.2)),            # both signs
    "inverse_positive": ("inverse:1", _inp(-3.05, 0.5, 0.9, 0.0)),  # [0.0225, 2.72]
    "inverse_negative": ("inverse:1", _inp(3.0, 0.5, 0.9, 0.0)),    # [-2.7, 0]: 0 maps to -0.05
}


@pytest.mark.parametrize("case", sorted(PROTECTED))
def test_an_active_clip_is_written_out_and_the_string_computes_the_model(case):
    fname, inp = PROTECTED[case]
    m = _Model(["y", fname, "x0"], [dict(OUT), {}, inp])
    X = np.linspace(-3, 3, 601).reshape(-1, 1)
    text, internal, _, protections = w.model_expression(m, X, ["x0"], ["v1"], LIMITS)
    assert protections == [fname.split(":")[0]] and "abs(" in text
    assert w.string_deviation(internal, 1, X, m.predict(pd.DataFrame({"x0": X[:, 0]}))) < 1e-14
    assert np.allclose(_evaluate(text, X, ["v1"]), m.predict(pd.DataFrame({"x0": X[:, 0]})), rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("case", sorted(PROTECTED))
def test_an_inactive_clip_is_written_as_the_plain_function(case):
    fname, _ = PROTECTED[case]
    m = _Model(["y", fname, "x0"], [dict(OUT), {}, _inp(0.0, 1.0, 1.0, 0.0)])
    X = np.linspace(1, 2, 11).reshape(-1, 1)            # inside every function's unclipped range
    text, internal, _, protections = w.model_expression(m, X, ["x0"], ["v1"], LIMITS)
    assert protections == [] and "abs(" not in text
    assert w.string_deviation(internal, 1, X, m.predict(pd.DataFrame({"x0": X[:, 0]}))) < 1e-15


def test_a_clip_active_only_at_validation_points_is_written_out(fake):
    fake.returned = [_Model(["y", "log:1", "x0"], [dict(OUT), {}, _inp(0.0, 1.0, 1.0, -0.5)])]   # log(x - 0.5)
    X, _ = _data()
    out = _fit(w.load({"n_epochs": 1}), x_val=np.full((4, 2), 0.2))
    assert out["extra"]["protections"] == ["log"] and out["extra"]["string_deviation_val"] < 1e-15


def test_the_other_functions_and_nesting_compute_the_model():
    codes = ["y", "add:2", "gaussian:2", "x0", "x1", "tanh:1", "linear:1", "multiply:2", "x0", "gaussian:1", "x1"]
    params = [dict(OUT), {}, {}, _inp(0.2, 0.5, 1.1, 0.1), _inp(-0.3, 0.8, 0.6, -0.2), {}, {"w": -1.3, "bias": 0.4},
              {}, _inp(0.0, 1.0, 0.5, 0.3), {}, _inp(1.0, 0.4, -0.7, 0.05)]
    m = _Model(codes, params)
    X = np.column_stack([np.linspace(-3, 3, 101), np.linspace(2, -1, 101)])
    text, internal, _, _ = w.model_expression(m, X, ["x0", "x1"], ["v1", "v2"], LIMITS)
    assert text.count("exp(-2.0*") == 2 and "tanh((-1.3*" in text
    assert w.string_deviation(internal, 2, X, m.predict(pd.DataFrame({"x0": X[:, 0], "x1": X[:, 1]}))) < 1e-15


def test_a_function_the_worker_cannot_state_is_an_error():
    m = _Model(["y", "x0"], [dict(OUT), _inp()])
    m.fnames[1] = "in-cat:0"
    with pytest.raises(ValueError, match="cannot state"):
        w.model_expression(m, np.ones((3, 1)), ["x0"], ["v1"], LIMITS)


def test_the_deviation_flags_a_string_that_is_not_the_model():
    X = np.linspace(1, 2, 5).reshape(-1, 1)
    assert w.string_deviation("x0*2", 1, X, 2 * X[:, 0]) == 0.0
    assert w.string_deviation("x0*2", 1, X, 2 * X[:, 0] + 1e-3) == pytest.approx(1e-3 / 4.001)
    assert w.string_deviation("log(x0 - 1.5)", 1, X, np.log(X[:, 0])) == float("inf")
    assert w.string_deviation("x0", 1, X[:0], np.array([])) is None


# -- against the real package (the worker's environment) ---------------------------------------------------------

def test_the_calls_name_only_arguments_the_real_functions_take():
    feyn = pytest.importorskip("feyn")
    sample = set(inspect.signature(feyn.QLattice.sample_models).parameters)
    assert {"kind", "stypes", "max_complexity", "query_string", "function_names"} <= sample
    fit_models = set(inspect.signature(feyn.fit_models).parameters)
    assert {"data", "loss_function", "criterion", "n_samples", "sample_weights", "threads"} <= fit_models
    assert set(inspect.signature(feyn.validate_data).parameters) == {"data", "kind", "output_name", "stypes"}
    assert "random_seed" in inspect.signature(feyn.QLattice).parameters
    from feyn._qlattice import _get_fnames
    assert len(_get_fnames(list(w.FUNCTIONS))) == len(w.FUNCTIONS)
    assert set(feyn.FNAME_MAP) - {"in-cat:0", "out-lr:1"} == {
        "in-linear:0", "out-linear:1", "exp:1", "gaussian:1", "inverse:1", "linear:1", "log:1", "sqrt:1",
        "squared:1", "tanh:1", "add:2", "gaussian:2", "multiply:2"}      # every one the worker states
    assert w.core_limits() == LIMITS
    with pytest.raises(TypeError):                                      # the crash the worker fixes
        feyn.validate_data(pd.DataFrame({"x0": [1.0, 2.0], "y": [1.0, 2.0]}), "regression", "y", None)


def test_the_stand_in_model_predicts_as_the_real_one():
    feyn = pytest.importorskip("feyn")
    from feyn._program import Program

    X = np.column_stack([np.linspace(-3, 3, 301), np.linspace(2, -1, 301)])
    frame = pd.DataFrame({"x0": X[:, 0], "x1": X[:, 1]})
    cases = [(["y", fname, "x0"], [dict(OUT), {}, dict(inp)]) for fname, inp in PROTECTED.values()]
    cases.append((["y", "add:2", "gaussian:2", "x0", "x1", "tanh:1", "linear:1", "multiply:2", "x0", "x1"],
                  [dict(OUT), {}, {}, _inp(0.2, 0.5, 1.1, 0.1), _inp(-0.3, 0.8, 0.6, -0.2), {},
                   {"w": -1.3, "bias": 0.4}, {}, _inp(0.0, 1.0, 0.5, 0.3), _inp(1.0, 0.4, -0.7, 0.05)]))
    for codes, params in cases:
        stand_in = _Model(codes, [dict(p) for p in params])
        real = feyn.Model(Program(codes), stand_in.fnames, [dict(p) for p in params])
        assert real.fnames == stand_in.fnames and real.names == stand_in.names and real.edge_count == stand_in.edge_count
        np.testing.assert_array_equal(real.predict(frame), stand_in.predict(frame))
        _, internal, _, _ = w.model_expression(real, X, ["x0", "x1"], ["v1", "v2"], w.core_limits())
        deviation = w.string_deviation(internal, 2, X, real.predict(frame))
        assert deviation < 1e-14


def test_a_real_fit_returns_a_string_that_computes_the_model():
    pytest.importorskip("feyn")
    X, y = _data(48)
    out = w.fit(X.tolist(), y.tolist(), x_val=(X[:6] + 1).tolist(), variables=["v1", "v2"], meta={}, options={},
                state=w.load({"n_epochs": 1}))
    assert out["extra"]["epochs"] == 1 and out["extra"]["string_deviation"] < 1e-12
    assert out["extra"]["string_deviation_val"] < 1e-12 and "v1" in out["expression"] and "x0" not in out["expression"]
