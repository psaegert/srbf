"""Setup before the clock (docs/fairness.md): srbf's warm-up problem runs through the method before the first timed
problem, in every worker process and again after every restart; its seconds go into __meta__['setup'] and never into
a fit time. Workers without a setup hook work; a failing warm-up is reported, not ignored."""
import importlib.util
import json
import pickle
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pytest
from simplipy import SimpliPyEngine

from srbf.benchmark import Benchmark, check_warmup_setting
from srbf.core import EvaluationResult, EvaluationSample
from srbf.store import ResultStore
from srbf.subprocess_adapter import SubprocessAdapter
from srbf.warmup import WARMUP_EXPRESSION, WARMUP_ID, run_warmup, warmup_sample
from srbf.worker import RUNNER_PATH

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def engine() -> SimpliPyEngine:
    return SimpliPyEngine.load("acj-4-3", install=True)


def _sample(eq_id: str = "toy", seed: int = 0) -> EvaluationSample:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, size=(32, 2))
    y = (2.0 * x[:, 0] - 0.5 * x[:, 1] + 1.0).reshape(-1, 1)
    return EvaluationSample(x_support=x, y_support=y, x_validation=x[:4], y_validation=y[:4],
                            metadata={"eq_id": eq_id, "variables": ["x1", "x2"]})


class _ListSource:
    def __init__(self, problems):
        self._problems = problems

    def __iter__(self):
        yield from self._problems

    def size_hint(self):
        return len(self._problems)


class _ColdStartAdapter:
    """An in-process method whose first fit pays a one-time cost (a compile), like a JIT-compiled method."""

    def __init__(self, cold_s: float = 0.4, fail_warmup: bool = False):
        self.cold_s = cold_s
        self.fail_warmup = fail_warmup
        self.seen: list = []
        self.warm = False

    def prepare(self, *, data_source=None):
        self.prepared = True

    def evaluate_sample(self, sample):
        self.seen.append(dict(sample.metadata))
        if self.fail_warmup and sample.metadata.get("warmup"):
            raise RuntimeError("cannot fit the warm-up")
        started = time.perf_counter()
        if not self.warm:
            time.sleep(self.cold_s)
            self.warm = True
        record = sample.clone_metadata()
        record["fit_time"] = time.perf_counter() - started
        record["prediction_success"] = True
        return EvaluationResult(record)


# -- the problem ---------------------------------------------------------------------------------------------------
def test_the_warmup_problem_is_fixed_and_carries_its_law():
    a, b = warmup_sample(), warmup_sample()
    np.testing.assert_array_equal(a.x_support, b.x_support)
    np.testing.assert_array_equal(a.y_validation, b.y_validation)
    assert a.x_support.shape == (512, 2) and a.x_validation.shape == (512, 2)
    np.testing.assert_allclose(a.y_support.ravel(), 1.5 * a.x_support[:, 0] * np.sin(a.x_support[:, 1]) + 0.5)
    assert a.metadata["warmup"] is True and a.metadata["benchmark_eq_id"] == WARMUP_ID
    assert a.metadata["expression"] == list(WARMUP_EXPRESSION)
    assert "eval_row_index" not in a.metadata          # no row of any run


def test_the_warmup_law_is_read_by_the_engine(engine):
    sample = warmup_sample()
    from srbf.candidates import evaluate_prefix
    (y,) = evaluate_prefix(engine, list(WARMUP_EXPRESSION), ["x1", "x2"], sample.x_support)
    np.testing.assert_allclose(y, sample.y_support)


def test_run_warmup_reports_a_method_that_raises_or_does_not_answer():
    out = run_warmup(lambda s: (_ for _ in ()).throw(ValueError("no")))
    assert out["warmup_error"] == "ValueError: no" and out["warmup_fit_time"] is None
    out = run_warmup(lambda s: {"prediction_success": False, "error": "nothing fitted", "fit_time": 0.1})
    assert out["warmup_error"] == "nothing fitted" and out["warmup_fit_time"] == 0.1
    out = run_warmup(lambda s: EvaluationResult({"prediction_success": True, "fit_time": 0.2}))
    assert out["warmup_error"] is None and out["warmup_time"] >= 0.0


# -- the driver and an in-process adapter --------------------------------------------------------------------------
def test_the_warmup_never_enters_a_fit_time_and_is_recorded_in_the_meta(tmp_path):
    adapter = _ColdStartAdapter(cold_s=0.4)
    out = tmp_path / "run.pkl"
    bench = Benchmark(_ListSource([_sample("a"), _sample("b")]), adapter, result_store=ResultStore(), warmup=True,
                      candidates=False)
    snapshot = bench.run(progress=False, verbose=False, output_path=str(out), meta={"config_provenance": "harness_tuned"})
    assert [m.get("warmup") for m in adapter.seen] == [True, None, None]           # once, before the first problem
    assert snapshot["eq_id"] == ["a", "b"]                                           # the warm-up is no row
    assert max(snapshot["fit_time"]) < 0.2                                           # the cold start was paid before
    setup = pickle.loads(out.read_bytes())["__meta__"]["setup"]
    assert setup["warmup"] is True and setup["prepare_time"] >= 0.0
    (event,) = setup["events"]
    assert event["reason"] == "start" and event["warmup_error"] is None
    assert event["warmup_time"] >= 0.4 and event["warmup_fit_time"] >= 0.4


def test_without_the_warmup_the_first_fit_pays_the_cold_start(tmp_path):
    adapter = _ColdStartAdapter(cold_s=0.3)
    out = tmp_path / "run.pkl"
    snapshot = Benchmark(_ListSource([_sample("a"), _sample("b")]), adapter, result_store=ResultStore(), candidates=False).run(
        progress=False, verbose=False, output_path=str(out), meta={"config_provenance": "harness_tuned"})
    assert not any(m.get("warmup") for m in adapter.seen)
    assert snapshot["fit_time"][0] >= 0.3
    setup = pickle.loads(out.read_bytes())["__meta__"]["setup"]
    assert setup["warmup"] is False and setup["events"] == []


def test_a_failing_warmup_is_reported_and_the_run_goes_on(tmp_path):
    adapter = _ColdStartAdapter(cold_s=0.0, fail_warmup=True)
    out = tmp_path / "run.pkl"
    bench = Benchmark(_ListSource([_sample("a")]), adapter, result_store=ResultStore(), warmup=True, candidates=False)
    with pytest.warns(RuntimeWarning, match="the warm-up problem failed: RuntimeError: cannot fit the warm-up"):
        snapshot = bench.run(progress=False, verbose=False, output_path=str(out), meta={"config_provenance": "harness_tuned"})
    assert snapshot["prediction_success"] == [True]
    (event,) = pickle.loads(out.read_bytes())["__meta__"]["setup"]["events"]
    assert event["warmup_error"] == "RuntimeError: cannot fit the warm-up"


def test_runner_warmup_is_a_boolean_read_from_the_config(monkeypatch):
    assert check_warmup_setting(True) is True and check_warmup_setting(False) is False
    with pytest.raises(ValueError, match="runner.warmup is true or false"):
        check_warmup_setting("yes")
    import srbf.config as run_config
    monkeypatch.setattr(run_config, "build_catalog_source", lambda cfg, *, target_size, skip: _ListSource([_sample()]))
    monkeypatch.setattr(run_config, "build_model_adapter", lambda cfg: _ColdStartAdapter())
    base = {"data_source": {"catalog": "x"}, "model_adapter": {"type": "flash_ansr", "config_provenance": "author_blessed"}}
    assert Benchmark.from_config({"run": {**base, "runner": {}}}).warmup is False
    bench = Benchmark.from_config({"run": {**base, "runner": {"warmup": True}}})
    assert bench.warmup is True and bench.build_time is not None and bench.build_time >= 0.0
    with pytest.raises(ValueError, match="runner.warmup"):
        Benchmark.from_config({"run": {**base, "runner": {"warmup": 1}}})


def test_the_flash_ansr_adapter_keeps_the_warmup_out_of_its_ledger_store(tmp_path):
    pytest.importorskip("flash_ansr")
    from srbf.model_adapters import FlashANSRAdapter
    adapter = FlashANSRAdapter.__new__(FlashANSRAdapter)
    adapter.candidate_store_dir = str(tmp_path)
    captured = []
    adapter._capture_ledger = lambda record, result, sample=None: captured.append(record)

    class _Result:
        best = None
        generation_time = refinement_time = 0.0

    class _Model:
        generation_config = None
        numpy_errors = None
        ranking = None

        def fit(self, *args, **kwargs):
            return _Result()

    adapter.model, adapter.complexity, adapter.seed = _Model(), "none", None
    with pytest.warns(UserWarning, match="no results"):
        adapter.evaluate_sample(warmup_sample())
    assert captured == []
    with pytest.warns(UserWarning, match="no results"):
        adapter.evaluate_sample(_sample())
    assert len(captured) == 1


def test_the_timing_ladder_turns_the_warmup_on():
    spec = importlib.util.spec_from_file_location("run_timing_ladder", ROOT / "scripts" / "run_timing_ladder.py")
    rtl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rtl)
    run = {"data_source": {"catalog": "nguyen"}, "model_adapter": {"type": "subprocess"},
           "runner": {"output": "{{ROOT}}/x/nguyen.pkl"}}
    cfg = {"experiments": {"nguyen": run, "koza": {**run, "runner": {**run["runner"], "warmup": False}}}}
    derived = rtl.timing_config(cfg, ["nguyen", "koza"], None, "m", None, None)
    assert derived["experiments"]["nguyen"]["runner"]["warmup"] is True
    assert derived["experiments"]["koza"]["runner"]["warmup"] is False      # a config that says otherwise is kept


# -- the worker protocol -------------------------------------------------------------------------------------------
COLD_WORKER = '''
import os
import time

COLD = {"paid": False}


def load(options):
    return {"cold_s": float(options.get("cold_s", 0.6))}


def fit(x, y, *, x_val, variables, meta, options, state):
    if meta.get("eq_id") == "crash":
        os._exit(3)
    if meta.get("warmup") and options.get("warmup_fails") == "error":
        return {"error": "refused the warm-up"}
    if meta.get("warmup") and options.get("warmup_fails") == "crash":
        os._exit(4)
    if meta.get("eq_id") == "hang":
        time.sleep(30)
    if not COLD["paid"]:                     # the one-time cost a fresh process pays on its first fit
        time.sleep(state["cold_s"])
        COLD["paid"] = True
    return {"expression": "%s + %s" % (variables[0], variables[1])}
'''

SETUP_WORKER = '''
import time


def load(options):
    return {"loaded": True}


def setup(state, options):
    time.sleep(0.2)
    return {**state, "compiled": True}


def info(state):
    return {"compiled": bool(state.get("compiled"))}


def fit(x, y, *, x_val, variables, meta, options, state):
    assert state["compiled"]
    return {"expression": variables[0]}
'''


TOLD_WORKER = '''
import os

TOLD = {}


def setup(state, options, *, warmup):
    TOLD["warmup"] = warmup


def info(state):
    return {"told": TOLD.get("warmup")}


def fit(x, y, *, x_val, variables, meta, options, state):
    if meta.get("eq_id") == "crash":
        os._exit(3)
    return {"expression": variables[0], "extra": {"told": TOLD.get("warmup")}}
'''


def _write(tmp_path: Path, body: str, name: str = "worker.py") -> Path:
    path = tmp_path / name
    path.write_text(textwrap.dedent(body))
    return path


def test_the_runner_reports_load_and_setup_times(tmp_path):
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    proc = subprocess.Popen([sys.executable, str(RUNNER_PATH), "--connect", f"127.0.0.1:{port}",
                             str(_write(tmp_path, SETUP_WORKER))], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    listener.settimeout(30)
    conn, _ = listener.accept()
    rfile, wfile = conn.makefile("r", encoding="utf-8"), conn.makefile("w", encoding="utf-8")
    wfile.write(json.dumps({"type": "init", "options": {}}) + "\n")
    wfile.flush()
    ready = json.loads(rfile.readline())
    assert ready["info"] == {"compiled": True}                     # setup's state replaced load's
    assert ready["load_time"] >= 0.0 and ready["setup_time"] >= 0.2
    wfile.write(json.dumps({"type": "close"}) + "\n")
    wfile.flush()
    assert json.loads(rfile.readline())["type"] == "closed"
    assert proc.wait(timeout=10) == 0


def _init(tmp_path, body, message):
    """Start the runner on a worker, send ``message`` as its init and return the ready reply."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    proc = subprocess.Popen([sys.executable, str(RUNNER_PATH), "--connect", f"127.0.0.1:{listener.getsockname()[1]}",
                             str(_write(tmp_path, body))], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    listener.settimeout(30)
    conn, _ = listener.accept()
    rfile, wfile = conn.makefile("r", encoding="utf-8"), conn.makefile("w", encoding="utf-8")
    try:
        wfile.write(json.dumps(message) + "\n")
        wfile.flush()
        return json.loads(rfile.readline())
    finally:
        wfile.write(json.dumps({"type": "close"}) + "\n")
        wfile.flush()
        proc.wait(timeout=10)


def test_the_runner_tells_a_setup_that_asks_whether_srbfs_warmup_follows(tmp_path):
    assert _init(tmp_path, TOLD_WORKER, {"type": "init", "options": {}, "warmup": True})["info"] == {"told": True}
    assert _init(tmp_path, TOLD_WORKER, {"type": "init", "options": {}, "warmup": False})["info"] == {"told": False}
    assert _init(tmp_path, TOLD_WORKER, {"type": "init", "options": {}})["info"] == {"told": False}
    keywords = TOLD_WORKER.replace("def setup(state, options, *, warmup):", "def setup(state, options, **context):") \
                          .replace('TOLD["warmup"] = warmup', 'TOLD["warmup"] = context["warmup"]')
    assert _init(tmp_path, keywords, {"type": "init", "options": {}, "warmup": True})["info"] == {"told": True}
    # a setup(state, options) is called as it is written (SETUP_WORKER, test_the_runner_reports_load_and_setup_times)


def test_the_run_arms_the_warmup_before_the_first_worker_starts(engine, tmp_path):
    """Every worker the run starts, the first and a restarted one alike, is told that the warm-up problem follows."""
    adapter = SubprocessAdapter(worker=_write(tmp_path, TOLD_WORKER), simplipy_engine=engine, max_restarts=1)
    out = tmp_path / "run.pkl"
    bench = Benchmark(_ListSource([_sample("a"), _sample("crash"), _sample("b")]), adapter, result_store=ResultStore(),
                      warmup=True, candidates=False)
    snapshot = bench.run(progress=False, verbose=False, output_path=str(out), meta={"config_provenance": "harness_tuned"})
    assert snapshot["prediction_success"] == [True, False, True] and snapshot["told"] == [True, None, True]
    meta = pickle.loads(out.read_bytes())["__meta__"]
    assert meta["worker"]["told"] is True
    assert [e["reason"] for e in meta["setup"]["events"]] == ["start", "restart"]
    assert all(e["warmup_error"] is None and "warmup_time" in e for e in meta["setup"]["events"])
    adapter = SubprocessAdapter(worker=_write(tmp_path, TOLD_WORKER), simplipy_engine=engine)
    snapshot = Benchmark(_ListSource([_sample("a")]), adapter, result_store=ResultStore(), candidates=False).run(
        progress=False, verbose=False, meta={"config_provenance": "harness_tuned"})
    assert snapshot["told"] == [False]                                        # no warm-up problem: the worker's own


def test_a_worker_without_the_setup_hook_reports_none_and_warms_up(engine):
    adapter = SubprocessAdapter(worker="example", simplipy_engine=engine)
    adapter.prepare()
    try:
        event = adapter.warm_up(warmup_sample())
        assert event["setup_time"] is None and event["load_time"] >= 0.0
        assert event["warmup_error"] is None and event["warmup_time"] > 0.0 and event["warmup_fit_time"] >= 0.0
        assert adapter.evaluate_sample(_sample()).to_mapping()["prediction_success"] is True
    finally:
        adapter.close()


def test_a_worker_with_the_setup_hook_records_its_time(engine, tmp_path):
    adapter = SubprocessAdapter(worker=_write(tmp_path, SETUP_WORKER), simplipy_engine=engine)
    adapter.prepare()
    try:
        assert adapter.setup_events[0]["setup_time"] >= 0.2
        assert adapter.evaluate_sample(_sample()).to_mapping()["prediction_success"] is True
    finally:
        adapter.close()


def test_the_worker_warmup_never_enters_fit_time(engine, tmp_path):
    adapter = SubprocessAdapter(worker=_write(tmp_path, COLD_WORKER), simplipy_engine=engine, options={"cold_s": 0.6})
    adapter.prepare()
    try:
        event = adapter.warm_up(warmup_sample())
        assert event["warmup_fit_time"] >= 0.6 and event["warmup_time"] >= 0.6     # the cold start, paid here
        record = adapter.evaluate_sample(_sample()).to_mapping()
        assert record["prediction_success"] is True and record["fit_time"] < 0.3
    finally:
        adapter.close()


def test_without_the_warmup_the_workers_first_fit_pays(engine, tmp_path):
    adapter = SubprocessAdapter(worker=_write(tmp_path, COLD_WORKER), simplipy_engine=engine, options={"cold_s": 0.4})
    adapter.prepare()
    try:
        assert adapter.evaluate_sample(_sample()).to_mapping()["fit_time"] >= 0.4
        assert "warmup_time" not in adapter.setup_events[0]
    finally:
        adapter.close()


def test_a_restart_after_a_crash_warms_the_new_worker_before_its_next_fit(engine, tmp_path):
    adapter = SubprocessAdapter(worker=_write(tmp_path, COLD_WORKER), simplipy_engine=engine, options={"cold_s": 0.5},
                                max_restarts=1)
    adapter.prepare()
    try:
        adapter.warm_up(warmup_sample())
        crashed = adapter.evaluate_sample(_sample("crash")).to_mapping()
        assert crashed["prediction_success"] is False and crashed["error"].startswith("WorkerCrashed")
        record = adapter.evaluate_sample(_sample("after")).to_mapping()
        assert record["prediction_success"] is True and record["fit_time"] < 0.3   # the new process was warmed
        assert [e["reason"] for e in adapter.setup_events] == ["start", "restart"]
        assert all(e["warmup_fit_time"] >= 0.5 and e["warmup_error"] is None for e in adapter.setup_events)
    finally:
        adapter.close()


def test_a_restart_after_a_timeout_and_after_a_hang_rewarms(engine, tmp_path):
    worker = _write(tmp_path, COLD_WORKER)
    adapter = SubprocessAdapter(worker=worker, simplipy_engine=engine, options={"cold_s": 0.3}, timeout=2.0,
                                max_restarts=1)
    adapter.prepare()
    try:
        adapter.warm_up(warmup_sample())
        assert adapter.evaluate_sample(_sample("hang")).to_mapping()["error"].startswith("WorkerTimeout")
        assert adapter.evaluate_sample(_sample("after")).to_mapping()["fit_time"] < 0.2
        assert [e["reason"] for e in adapter.setup_events] == ["start", "restart"]
        assert "warmup_time" in adapter.setup_events[-1]
    finally:
        adapter.close()
    hung = SubprocessAdapter(worker=worker, simplipy_engine=engine, options={"cold_s": 0.3}, timeout=2.0,
                             hang_overdue_factor=30.0)
    hung.prepare()
    try:
        hung.warm_up(warmup_sample())
        assert hung.evaluate_sample(_sample("hang")).to_mapping()["worker_hangs"] == 2
        assert [e["reason"] for e in hung.setup_events] == ["start", "restart after a hang", "restart after a hang"]
        assert all("warmup_time" in e for e in hung.setup_events)
    finally:
        hung.close()


def test_a_warmup_the_worker_refuses_is_reported_and_the_worker_stays(engine, tmp_path):
    adapter = SubprocessAdapter(worker=_write(tmp_path, COLD_WORKER), simplipy_engine=engine,
                                options={"cold_s": 0.0, "warmup_fails": "error"})
    adapter.prepare()
    try:
        with pytest.warns(RuntimeWarning, match="the warm-up problem failed in the worker worker.py: refused the warm-up"):
            event = adapter.warm_up(warmup_sample())
        assert event["warmup_error"] == "refused the warm-up"
        assert adapter.evaluate_sample(_sample()).to_mapping()["prediction_success"] is True
    finally:
        adapter.close()


def test_a_warmup_that_crashes_the_worker_spends_the_crash_budget_and_is_reported(engine, tmp_path):
    adapter = SubprocessAdapter(worker=_write(tmp_path, COLD_WORKER), simplipy_engine=engine,
                                options={"cold_s": 0.0, "warmup_fails": "crash"}, max_restarts=1)
    adapter.prepare()
    try:
        with pytest.warns(RuntimeWarning, match="the warm-up problem failed"):
            adapter.warm_up(warmup_sample())
        assert [e["reason"] for e in adapter.setup_events] == ["start", "restart"]
        assert all(e["warmup_error"].startswith("WorkerCrashed") for e in adapter.setup_events)
        record = adapter.evaluate_sample(_sample()).to_mapping()      # the worker is gone: every problem says why
        assert record["prediction_success"] is False
        assert record["error"].startswith("worker unavailable: the warm-up problem took the worker down")
    finally:
        adapter.close()


def test_the_run_records_every_worker_start_in_its_meta(engine, tmp_path):
    adapter = SubprocessAdapter(worker=_write(tmp_path, COLD_WORKER), simplipy_engine=engine, options={"cold_s": 0.3},
                                max_restarts=1)
    out = tmp_path / "run.pkl"
    bench = Benchmark(_ListSource([_sample("a"), _sample("crash"), _sample("b")]), adapter, result_store=ResultStore(),
                      warmup=True, candidates=False)
    snapshot = bench.run(progress=False, verbose=False, output_path=str(out), meta={"config_provenance": "harness_tuned"})
    assert snapshot["eq_id"] == ["a", "crash", "b"] and snapshot["prediction_success"] == [True, False, True]
    assert snapshot["fit_time"][0] < 0.2 and snapshot["fit_time"][2] < 0.2
    setup = pickle.loads(out.read_bytes())["__meta__"]["setup"]
    assert setup["warmup"] is True
    assert [e["reason"] for e in setup["events"]] == ["start", "restart"]
    assert all(e["warmup_fit_time"] >= 0.3 for e in setup["events"])
