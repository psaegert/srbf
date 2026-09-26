"""The TiSR worker (srbf/worker/models/tisr_worker.py) without Julia: a stand-in for the Julia side with the surface
the worker uses (SrbfTiSR.tisr_tree(), SrbfTiSR.run(...) and its report, seval of the Options constructor), and one
test that runs the real environment, when it exists, to hold the stand-in to the real report."""
import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from srbf.worker.models import tisr_worker as w

MEASURES = ["ms_processed_e", "compl", "mse", "mae", "one_minus_abs_spearman", "mare", "max_are", "weighted_compl"]


def _report(members, stop="reached maximum number of generations", generations=8):
    """SrbfTiSR.run's report: members in TiSR's export order, each its tree's tokens, TiSR's measures and TiSR's own
    values of the member on the data (``members`` pairs tokens with those values)."""
    return {"members": [{"tokens": tokens, "measures": {m: float(i + 1) for i, m in enumerate(MEASURES)},
                         "values": list(values)} for tokens, values in members],
            "seconds": 1.25, "stop": stop, "generations": generations, "progress": [[1, generations], [0.01, 1.2]]}


class _SrbfTiSR:
    def __init__(self, fake):
        self.fake = fake

    def tisr_tree(self):
        return self.fake.tree

    def run(self, X, y, make, n_gens, t_lim, seed, *values):
        self.fake.calls.append({"X": np.asarray(X), "y": np.asarray(y), "make": make, "n_gens": n_gens,
                                "t_lim": t_lim, "seed": seed, "values": values})
        X = np.asarray(X)
        return self.fake.report(X)


class _Julia:
    """juliacall's Main with the worker's module loaded."""

    def __init__(self):
        self.tree = w.TREE
        self.calls = []
        self.sources = []
        self.SrbfTiSR = _SrbfTiSR(self)
        # 2.5*v1*v2 + 1.0 as TiSR's tree, + (1.0) (* (2.5) (* v1 v2)), and v1*v2; binops are + - * / ^ rootn
        self.report = lambda X: _report([(["b1", "p1.0", "b3", "p2.5", "b3", "v1", "v2"], 2.5 * X[:, 0] * X[:, 1] + 1.0),
                                         (["b3", "v1", "v2"], X[:, 0] * X[:, 1])])

    def seval(self, code):
        if code.startswith("source ->"):
            def evaluate(source):
                self.sources.append(source)
                return ("make", source)
            return evaluate
        if code == "string(VERSION)":
            return w.JULIA_VERSION
        raise AssertionError("unexpected seval: %s" % code)


@pytest.fixture
def julia(monkeypatch):
    fake = _Julia()
    monkeypatch.setattr(w, "_require_julia", lambda: fake)
    return fake


def _data():
    rng = np.random.default_rng(0)
    X = rng.uniform(1, 2, size=(64, 2))
    return X, 2.5 * X[:, 0] * X[:, 1] + 1.0


def test_tisr_runs_at_its_defaults_with_the_benchmark_operators_the_budget_and_one_thread(julia):
    X, y = _data()
    state = w.load({"generations": 64, "warmup": False})
    out = w.fit(X, y, x_val=[], variables=["a", "b"], meta={}, options={}, state=state)
    source = julia.sources[-1]
    assert source.startswith("(data_matr, n_gens, t_lim) -> Options(data_matr; ")
    assert "binops = (+, -, *, /, ^, rootn,)" in source
    assert "unaops = (neg, abs, inv, sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh, exp, log,)" in source
    assert ("general = general_params(; n_gens = n_gens, t_lim = t_lim, multithreading = false, print_progress = false, "
            "show_hall_of_fame = false)") in source
    assert "callback" not in source and "early_stop_iter" not in source          # TiSR's defaults: no early stop
    for section in ("data_split", "measures", "selection", "fitting", "mutation", "grammar"):
        assert "%s = %s(; )" % (section, w.SECTIONS[section]) in source      # TiSR's own defaults
    assert "fit_weights" not in source and "max_compl" not in source
    call = julia.calls[-1]
    assert call["n_gens"] == 64 and call["t_lim"] == 3600.0 and call["values"] == ()
    assert call["seed"] == w.run_seed(X, y, 0) == out["extra"]["seed"]
    assert out["fit_time"] == 1.25 and out["extra"]["generations"] == 64 and out["extra"]["generations_run"] == 8
    assert not out["extra"]["hit_time_guard"]
    assert out["extra"]["progress"] == {"generation": [1, 8], "seconds": [0.01, 1.2]}


def test_no_member_is_picked_and_the_whole_hall_of_fame_is_written_in_the_benchmark_syntax(julia):
    X, y = _data()
    out = w.fit(X, y, x_val=[], variables=["a", "b"], meta={}, options={}, state=w.load({"warmup": False}))
    assert "expression" not in out and out["error"] == w.NO_ANSWER     # the selection is open: no answer
    hof = out["extra"]["hall_of_fame"]
    assert [m["expression"] for m in hof] == ["(1.0 + (2.5*(a*b)))", "(a*b)"]
    assert set(hof[0]) == set(MEASURES) | {"expression", "string_deviation"}
    assert all(m["string_deviation"] < 1e-12 for m in hof)
    assert out["fit_time"] == 1.25


def test_a_string_that_does_not_compute_tisrs_values_shows_in_the_deviation(julia):
    X, y = _data()
    julia.report = lambda X: _report([(["b3", "v1", "v2"], X[:, 0] * X[:, 1] + 1.0)])
    out = w.fit(X, y, x_val=[], variables=["a", "b"], meta={}, options={}, state=w.load({"warmup": False}))
    assert out["extra"]["hall_of_fame"][0]["string_deviation"] > 0.1


def test_constants_are_written_at_full_precision_and_negative_ones_in_parentheses():
    names, binops, unaops = ["x", "y"], w.BINARY_OPERATORS, w.UNARY_OPERATORS
    tokens = ["b2", "p-6.6743e-11", "b3", "p0.30000000000000004", "v2"]
    assert w.infix(tokens, names, binops, unaops) == "((-6.6743e-11) - (0.30000000000000004*y))"
    assert w.infix(["p1.0e20"], names, binops, unaops) == "1e+20"
    with pytest.raises(ValueError, match="non-finite"):
        w.infix(["pInf"], names, binops, unaops)
    with pytest.raises(ValueError, match="2 variables"):
        w.infix(["v3"], names, binops, unaops)
    with pytest.raises(ValueError, match="trailing"):
        w.infix(["v1", "v2"], names, binops, unaops)


def test_every_operator_is_written_as_the_benchmark_reads_it():
    names = ["x"]
    binops = ("+", "-", "*", "/", "^", "rootn")
    unaops = ("neg", "sqrt", "pow2", "pow3", "asin", "log")
    cases = {
        ("b5", "v1", "p0.5"): "((x)**(0.5))",
        ("b6", "v1", "p3.0"): "rootn(x, 3.0)",
        ("u1", "v1"): "neg(x)",
        ("u2", "v1"): "sqrt(x)",
        ("u3", "v1"): "((x)**2)",
        ("u4", "b1", "v1", "p1.0"): "(((x + 1.0))**3)",
        ("u5", "v1"): "asin(x)",
        ("u6", "b4", "v1", "p2.0"): "log((x/2.0))",
    }
    X = np.linspace(0.1, 0.9, 9).reshape(-1, 1)
    for tokens, expected in cases.items():
        text = w.infix(list(tokens), names, binops, unaops)
        assert text == expected
        with np.errstate(all="ignore"):
            assert np.all(np.isfinite(eval(text, {"__builtins__": {}}, {**w._NAMESPACE, "x": X[:, 0]})))


def test_rootn_follows_the_benchmark():
    x = np.array([-8.0, 8.0, -4.0, 4.0, 2.0])
    np.testing.assert_allclose(w._rootn(x, 3.0)[:2], [-2.0, 2.0])
    assert np.isnan(w._rootn(-4.0, 2.0)) and w._rootn(4.0, 2.0) == 2.0
    assert np.isnan(w._rootn(2.0, 2.5)) and np.isnan(w._rootn(2.0, 0.0))
    assert math.isclose(float(w._rootn(4.0, -2.0)), 0.5)


def test_a_problem_gets_one_seed_from_its_data_and_the_configured_seed(julia):
    X, y = _data()
    assert w.run_seed(X, y, 0) == w.run_seed(X.copy(), y.copy(), 0)
    assert w.run_seed(X, y, 0) != w.run_seed(X, y + 1e-9, 0)
    assert w.run_seed(X, y, 0) != w.run_seed(X, y, 1)
    w.fit(X, y, x_val=[], variables=["a", "b"], meta={}, options={}, state=w.load({"seed": 1, "warmup": False}))
    assert julia.calls[-1]["seed"] == w.run_seed(X, y, 1)


def test_the_time_guard_is_recorded_when_it_stops_the_search(julia):
    X, y = _data()
    julia.report = lambda X: _report([(["b3", "v1", "v2"], X[:, 0] * X[:, 1])], stop="reached time limit", generations=3)
    out = w.fit(X, y, x_val=[], variables=["a", "b"], meta={}, options={},
                state=w.load({"generations": 100, "time_guard": 5.0, "warmup": False}))
    assert julia.calls[-1]["t_lim"] == 5.0
    assert out["extra"]["hit_time_guard"] and out["extra"]["generations_run"] == 3


def test_a_member_that_cannot_be_written_is_kept_with_its_reason(julia):
    X, y = _data()
    julia.report = lambda X: _report([(["pNaN"], np.full(X.shape[0], np.nan)), (["b3", "v1", "v2"], X[:, 0] * X[:, 1])])
    out = w.fit(X, y, x_val=[], variables=["a", "b"], meta={}, options={}, state=w.load({"warmup": False}))
    hof = out["extra"]["hall_of_fame"]
    assert hof[0]["expression"] is None and "non-finite" in hof[0]["error"] and hof[1]["expression"] == "(a*b)"
    julia.report = lambda X: _report([])
    assert "empty" in w.fit(X, y, x_val=[], variables=["a", "b"], meta={}, options={}, state=w.load({"warmup": False}))["error"]


def test_an_environment_without_the_pinned_tisr_is_refused(julia):
    julia.tree = "0" * 40
    with pytest.raises(RuntimeError, match="rebuild"):
        w.load({"warmup": False})


def test_a_side_experiment_replaces_settings_and_sets_per_problem_values_at_run_time(julia):
    config = {"binops": ["+", "-", "*", "/", "^"], "unaops": ["neg", "sqrt", "pow2"],
              "data_split": {"parts": "[0.5, 0.0, 0.5]"}, "general": {"island_extinction_interval": 500},
              "grammar": {"max_compl": 30, "init_tree_depth": 4}, "fitting": {"NM_prob": 0.0}}
    state = w.load({"config": config, "config_by_problem": {"I.12.1": {"grammar": {"max_compl": 7}}}, "warmup": False})
    source = julia.sources[-1]
    assert source.startswith("(data_matr, n_gens, t_lim, p1) -> ")
    assert "binops = (+, -, *, /, ^,)" in source and "unaops = (neg, sqrt, pow2,)" in source
    assert "grammar = grammar_params(; max_compl = p1, init_tree_depth = 4)" in source
    assert "data_split = data_split_params(; parts = [0.5, 0.0, 0.5])" in source and "NM_prob = 0.0" in source
    assert "island_extinction_interval = 500" in source
    X, y = _data()
    julia.report = lambda X: _report([(["u3", "v1"], X[:, 0] ** 2)])
    out = w.fit(X, y, x_val=[], variables=["a", "b"], meta={"benchmark_eq_id": "I.12.1"}, options={}, state=state)
    assert julia.calls[-1]["values"] == (7,) and out["extra"]["problem_settings"] == {"grammar.max_compl": 7}
    assert out["extra"]["hall_of_fame"][0]["expression"] == "((a)**2)"
    w.fit(X, y, x_val=[], variables=["a", "b"], meta={"benchmark_eq_id": "II.6.15a"}, options={}, state=state)
    assert julia.calls[-1]["values"] == (30,)
    assert len(julia.sources) == 1                                    # one constructor, compiled once


@pytest.mark.parametrize("options, message", [
    ({"config": {"general": {"n_gens": 5}}}, "cannot set n_gens"),
    ({"config": {"general": {"t_lim": 5.0}}}, "cannot set t_lim"),
    ({"config": {"general": {"callback": "(h, p, g, t, d, o) -> true"}}}, "cannot set callback"),
    ({"config": {"fitting": {"early_stop_iter": 5}}}, "cannot set early_stop_iter"),
    ({"config": {"fit_weights": "ones(3)"}}, "config sets fit_weights"),
    ({"config": {"unaops": ["erf"]}}, "unary operators the worker knows: erf"),
    ({"config": {"binops": ["-", "*"]}}, r"needs \+"),
    ({"config": {"grammar": {"illegal_dict": ["^"]}}}, "Julia source"),
    ({"config_by_problem": {"B1": {"grammar": {"max_compl": 9}}}}, "which config does not set"),
    ({"config": {"grammar": {"max_compl": 30}}, "config_by_problem": {"B1": {"grammar": {"max_compl": "9"}}}}, "a number or a boolean"),
    ({"generations": 0}, "at least 1"),
])
def test_a_bad_option_fails_when_the_worker_starts(julia, options, message):
    with pytest.raises(ValueError, match=message):
        w.load({**options, "warmup": False})


def test_the_worker_points_juliacall_at_its_own_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "environ", dict(os.environ))
    assert w.julia_environment(str(tmp_path)) is None
    (tmp_path / "julia_env").mkdir()
    (tmp_path / "julia_env" / "Manifest.toml").write_text("")
    (tmp_path / ("julia-" + w.JULIA_VERSION) / "bin").mkdir(parents=True)
    (tmp_path / ("julia-" + w.JULIA_VERSION) / "bin" / "julia").write_text("")
    os.environ["JULIA_DEPOT_PATH"] = "/shared/depot"                 # a shared depot is never used
    assert w.julia_environment(str(tmp_path)) == str(tmp_path / "julia_env")
    assert os.environ["JULIA_DEPOT_PATH"] == str(tmp_path / "julia_depot")
    assert os.environ["PYTHON_JULIAPKG_OFFLINE"] == "yes" and os.environ["PYTHON_JULIACALL_THREADS"] == "1"


def _tisr_python():
    explicit = os.environ.get("SRBF_TISR_PYTHON")
    if explicit:
        return explicit
    root = os.environ.get("FLASH_ANSR_ROOT")
    return os.path.join(root, "envs", "tisr", "bin", "python") if root else None


REAL_RUN = r"""
import importlib.util, json, sys
import numpy as np
spec = importlib.util.spec_from_file_location("tisr_worker", sys.argv[1])
w = importlib.util.module_from_spec(spec); spec.loader.exec_module(w)
state = w.load({"generations": 3, "warmup": False})
rng = np.random.default_rng(0)
X = rng.uniform(1.0, 2.0, size=(64, 2)); y = 2.5 * X[:, 0] * X[:, 1] + 1.0
report = state["run"](X, y, state["make"], 3, 600.0, 1)
member = report["members"][0]
print("REPORT " + json.dumps({"keys": sorted(str(k) for k in report.keys()), "member": sorted(str(k) for k in member.keys()),
      "measures": sorted(str(k) for k in member["measures"].keys()), "tokens": [str(t) for t in member["tokens"]],
      "types": [type(report["seconds"]).__name__, type(report["stop"]).__name__, type(report["generations"]).__name__],
      "progress": [[type(v).__name__ for v in part] for part in report["progress"]],
      "n_values": len(list(member["values"]))}))
out = w.fit(X, y, x_val=[], variables=["a", "b"], meta={}, options={}, state=state)
print("FIT " + json.dumps({"error": out["error"], "deviations": [m["string_deviation"] for m in out["extra"]["hall_of_fame"]],
      "generations_run": out["extra"]["generations_run"]}))
"""


def test_the_stand_in_reports_what_the_real_environment_reports(tmp_path):
    python = _tisr_python()
    if not python or not os.path.isfile(python):
        pytest.skip("no TiSR environment (scripts/envs/build_tisr_env.sh; SRBF_TISR_PYTHON or FLASH_ANSR_ROOT)")
    script = tmp_path / "real.py"
    script.write_text(REAL_RUN)
    done = subprocess.run([python, str(script), str(Path(w.__file__))], capture_output=True, text=True, timeout=1800,
                          cwd=str(tmp_path))
    assert done.returncode == 0, done.stderr[-4000:]
    lines = {line.split(" ", 1)[0]: json.loads(line.split(" ", 1)[1]) for line in done.stdout.splitlines()
             if line.startswith(("REPORT ", "FIT "))}
    real, fake = lines["REPORT"], _report([(["v1"], [0.0])])
    assert real["keys"] == sorted(fake) and real["member"] == sorted(fake["members"][0])
    assert real["measures"] == sorted(MEASURES)
    assert real["types"] == ["float", "str", "int"] and real["n_values"] == 64
    assert len(real["progress"]) == 2 and set(real["progress"][0]) == {"int"} and set(real["progress"][1]) == {"float"}
    assert all(t[0] in "buvp" for t in real["tokens"])
    assert lines["FIT"]["error"] == w.NO_ANSWER and lines["FIT"]["generations_run"] == 3
    assert all(d is not None and d < 1e-9 for d in lines["FIT"]["deviations"])
