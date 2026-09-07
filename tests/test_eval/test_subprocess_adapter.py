"""The out-of-process adapter: a worker script in its own interpreter, the JSON-lines protocol,
the srbf-side record, and the failure handling (per-sample errors, timeouts, crashes, restarts)."""
import json
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from simplipy import SimpliPyEngine

from srbf.core import EvaluationSample
from srbf.subprocess_adapter import BUILTIN_WORKERS, SubprocessAdapter, evaluate_prefix, resolve_worker
from srbf.worker import RUNNER_PATH, srbf_worker_helpers as helpers


@pytest.fixture(scope="module")
def engine() -> SimpliPyEngine:
    return SimpliPyEngine.load("acj-4-3", install=True)


def _target(rows: np.ndarray) -> np.ndarray:
    return (2.0 * rows[:, 0] - 0.5 * rows[:, 1] + 1.0).reshape(-1, 1)


def _sample(seed: int = 0, eq_id: str = "toy", skeleton=("+", "*", "2", "x1", "x2")) -> EvaluationSample:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, size=(48, 2))
    x_val = rng.uniform(-2.0, 2.0, size=(12, 2))
    return EvaluationSample(x_support=x, y_support=_target(x), x_validation=x_val, y_validation=_target(x_val),
                            metadata={"eq_id": eq_id, "variables": ["x1", "x2"], "skeleton": list(skeleton)})


def _write_worker(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "worker.py"
    path.write_text(textwrap.dedent(body))
    return path


def _spawn_runner(worker: str):
    """Listen on a local port, start the runner against it, return (process, rfile, wfile)."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    proc = subprocess.Popen([sys.executable, str(RUNNER_PATH), "--connect", f"127.0.0.1:{port}", worker],
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    listener.settimeout(30)
    conn, _ = listener.accept()
    listener.close()
    return proc, conn.makefile("r", encoding="utf-8"), conn.makefile("w", encoding="utf-8")


def _send(wfile, obj) -> None:
    wfile.write(json.dumps(obj) + "\n")
    wfile.flush()


class TestRunnerProtocol:
    def test_init_fit_close_with_the_example_worker(self) -> None:
        proc, rfile, wfile = _spawn_runner(str(BUILTIN_WORKERS["example"]))
        _send(wfile, {"type": "init", "options": {}})
        ready = json.loads(rfile.readline())
        assert ready["type"] == "ready" and ready["info"]["worker"] == "example-ols"
        x = [[float(i), float(i % 3)] for i in range(10)]
        y = [2.0 * r[0] - 0.5 * r[1] + 1.0 for r in x]
        _send(wfile, {"type": "fit", "id": 1, "x": x, "y": y, "x_val": x[:2], "variables": ["x1", "x2"], "meta": {}})
        result = json.loads(rfile.readline())
        assert result["type"] == "result" and result["id"] == 1 and result["error"] is None
        assert "x1" in result["expression"] and isinstance(result["fit_time"], float)
        assert len(result["y_pred"]) == 10 and len(result["y_pred_val"]) == 2
        _send(wfile, {"type": "close"})
        assert json.loads(rfile.readline())["type"] == "closed"
        assert proc.wait(timeout=10) == 0

    def test_missing_worker_reports_an_error_line(self) -> None:
        proc, rfile, wfile = _spawn_runner("/nonexistent/worker.py")
        assert json.loads(rfile.readline())["type"] == "error"
        assert proc.wait(timeout=10) == 3

    def test_worker_stdout_never_reaches_the_protocol(self, tmp_path) -> None:
        worker = _write_worker(tmp_path, '''
            import sys
            def load(options):
                print("noise on stdout at load time")
                sys.stdout.write("more noise\\n")
            def fit(x, y, *, x_val, variables, meta, options, state):
                print("noise during fit")
                return {"expression": "x1"}
        ''')
        proc, rfile, wfile = _spawn_runner(str(worker))
        _send(wfile, {"type": "init", "options": {}})
        assert json.loads(rfile.readline())["type"] == "ready"
        _send(wfile, {"type": "fit", "id": 1, "x": [[1.0]], "y": [1.0], "x_val": [], "variables": ["x1"], "meta": {}})
        assert json.loads(rfile.readline())["expression"] == "x1"
        _send(wfile, {"type": "close"})
        assert json.loads(rfile.readline())["type"] == "closed"
        assert proc.wait(timeout=10) == 0
        assert "noise during fit" in (proc.stdout.read() if proc.stdout else "")


class TestAdapterEndToEnd:
    def test_example_worker_recovers_a_line(self, engine) -> None:
        adapter = SubprocessAdapter(worker="example", simplipy_engine=engine, options={"ridge": 0.0})
        adapter.prepare()
        try:
            info = adapter.worker_info()
            assert info["worker"] == "example-ols" and info["python"] == sys.executable
            sample = _sample()
            record = adapter.evaluate_sample(sample).to_mapping()
            assert record["prediction_success"] is True, record.get("error")
            assert record["fit_time"] > 0.0
            assert isinstance(record["predicted_expression_prefix"], list)
            assert record["predicted_skeleton_prefix"]
            np.testing.assert_allclose(record["y_pred_val"], sample.y_validation, atol=1e-6)
            assert record["support_fvu"] < 1e-12 and record["validation_fvu"] < 1e-12
            assert record["n_support"] == 48                     # the worker's extra, lifted into the record
            assert len(record["predicted_constants"]) == 3
        finally:
            adapter.close()
        assert adapter._process is None

    def test_masking_hands_the_worker_only_the_used_columns(self, engine) -> None:
        adapter = SubprocessAdapter(worker="example", simplipy_engine=engine)
        adapter.prepare()
        try:
            record = adapter.evaluate_sample(_sample(skeleton=("*", "2", "x1"))).to_mapping()
            assert record["prediction_success"] is True, record.get("error")
            assert record["variable_names"] == ["x1"]
            assert "x2" not in record["predicted_expression"]
            assert len(record["predicted_constants"]) == 2      # slope and intercept only
        finally:
            adapter.close()

    def test_srbf_evaluates_the_expression_when_the_worker_gives_no_predictions(self, engine, tmp_path) -> None:
        worker = _write_worker(tmp_path, '''
            def fit(x, y, *, x_val, variables, meta, options, state):
                print("this goes to stderr, never to the protocol")
                return {"expression": "2*%s - 0.5*%s + 1" % (variables[0], variables[1])}
        ''')
        adapter = SubprocessAdapter(worker=worker, simplipy_engine=engine, drop_unused_variables=False)
        adapter.prepare()
        try:
            sample = _sample()
            record = adapter.evaluate_sample(sample).to_mapping()
            assert record["prediction_success"] is True, record.get("error")
            np.testing.assert_allclose(record["y_pred"], sample.y_support, atol=1e-9)
            np.testing.assert_allclose(record["y_pred_val"], sample.y_validation, atol=1e-9)
        finally:
            adapter.close()

    def test_runs_in_a_foreign_interpreter(self, engine, tmp_path) -> None:
        venv = tmp_path / "venv"
        made = subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], capture_output=True, text=True)
        if made.returncode != 0:  # pragma: no cover - platform dependent
            pytest.skip(f"cannot create a venv here: {made.stderr.strip()[:200]}")
        python = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        adapter = SubprocessAdapter(worker="example", simplipy_engine=engine, python=str(python))
        adapter.prepare()
        try:
            # The worker reports its own sys.executable: the venv's launcher, not the base interpreter.
            assert Path(adapter.worker_info()["python"]) == python
            record = adapter.evaluate_sample(_sample()).to_mapping()
            assert record["prediction_success"] is True, record.get("error")
        finally:
            adapter.close()


class TestFailureHandling:
    WORKER = '''
        import os, time
        def fit(x, y, *, x_val, variables, meta, options, state):
            if meta.get("eq_id") == "bad":
                raise ValueError("this problem is poison")
            if meta.get("eq_id") == "slow":
                time.sleep(5)
            if meta.get("eq_id") == "crash":
                os._exit(3)
            return {"expression": "2*x1 - 0.5*x2 + 1"}
    '''

    def test_a_failing_fit_is_recorded_and_the_worker_stays_up(self, engine, tmp_path) -> None:
        adapter = SubprocessAdapter(worker=_write_worker(tmp_path, self.WORKER), simplipy_engine=engine, drop_unused_variables=False)
        adapter.prepare()
        try:
            bad = adapter.evaluate_sample(_sample(eq_id="bad")).to_mapping()
            assert bad["prediction_success"] is False and "poison" in bad["error"]
            assert "Traceback" in bad["error_detail"]
            good = adapter.evaluate_sample(_sample(eq_id="fine")).to_mapping()
            assert good["prediction_success"] is True
            assert adapter._restarts == 0
        finally:
            adapter.close()

    def test_timeout_kills_restarts_then_gives_up(self, engine, tmp_path) -> None:
        adapter = SubprocessAdapter(worker=_write_worker(tmp_path, self.WORKER), simplipy_engine=engine,
                                    timeout=0.5, max_restarts=1, drop_unused_variables=False)
        adapter.prepare()
        try:
            first = adapter.evaluate_sample(_sample(eq_id="slow")).to_mapping()
            assert first["prediction_success"] is False and first["error"].startswith("WorkerTimeout")
            assert adapter._restarts == 1 and adapter._process is not None        # restarted once
            ok = adapter.evaluate_sample(_sample(eq_id="fine")).to_mapping()
            assert ok["prediction_success"] is True
            second = adapter.evaluate_sample(_sample(eq_id="slow")).to_mapping()
            assert second["prediction_success"] is False
            assert adapter._process is None                                        # budget exhausted
            dead = adapter.evaluate_sample(_sample(eq_id="fine")).to_mapping()
            assert dead["prediction_success"] is False and dead["error"].startswith("worker unavailable")
        finally:
            adapter.close()

    def test_a_crash_is_recorded_with_the_exit_code(self, engine, tmp_path) -> None:
        adapter = SubprocessAdapter(worker=_write_worker(tmp_path, self.WORKER), simplipy_engine=engine,
                                    max_restarts=1, drop_unused_variables=False)
        adapter.prepare()
        try:
            crashed = adapter.evaluate_sample(_sample(eq_id="crash")).to_mapping()
            assert crashed["prediction_success"] is False
            assert crashed["error"].startswith("WorkerCrashed") and "returncode 3" in crashed["error"]
            assert adapter.evaluate_sample(_sample(eq_id="fine")).to_mapping()["prediction_success"] is True
        finally:
            adapter.close()

    def test_unparseable_expression_is_an_error_record(self, engine, tmp_path) -> None:
        worker = _write_worker(tmp_path, '''
            def fit(x, y, *, x_val, variables, meta, options, state):
                return {"expression": "2*x1 +"}
        ''')
        adapter = SubprocessAdapter(worker=worker, simplipy_engine=engine, drop_unused_variables=False)
        adapter.prepare()
        try:
            record = adapter.evaluate_sample(_sample()).to_mapping()
            assert record["prediction_success"] is False
            assert record["error"].startswith("Failed to")
        finally:
            adapter.close()


class TestHelpers:
    def test_legacy_vocabulary_is_respelled(self) -> None:
        legacy = ["+", "mult2", "x1", "pow1_3", "div3", "pow2", "x2"]
        assert helpers.respell_legacy_prefix(legacy) == ["+", "*", "2", "x1", "rootn", "/", "pow", "x2", "2", "3", "3"]

    def test_prefix_to_infix_round_trips_through_the_engine(self, engine) -> None:
        prefix = ["+", "*", "2", "x1", "rootn", "x2", "3"]
        infix = helpers.prefix_to_infix(prefix)
        assert list(engine.infix_to_prefix(infix)) == prefix

    def test_substitute_placeholders(self) -> None:
        out = helpers.substitute_placeholders(["*", "<constant>", "x1"], [2.5])
        assert out == ["*", "2.5", "x1"]
        with pytest.raises(ValueError):
            helpers.substitute_placeholders(["*", "<constant>", "x1"], [1.0, 2.0])

    def test_evaluate_prefix_uses_the_engine_realizations(self, engine) -> None:
        rows = np.array([[1.0, 8.0], [4.0, -8.0]])
        (values,) = evaluate_prefix(engine, ["+", "*", "2", "x1", "rootn", "x2", "3"], ["x1", "x2"], rows)
        np.testing.assert_allclose(values.ravel(), [4.0, 6.0])

    def test_resolve_worker(self, tmp_path) -> None:
        assert resolve_worker("example") == BUILTIN_WORKERS["example"]
        with pytest.raises(ValueError, match="not found"):
            resolve_worker(tmp_path / "missing.py")


class TestConfig:
    def test_subprocess_block_builds_the_adapter(self, monkeypatch) -> None:
        from srbf import config as run_config
        monkeypatch.setattr(run_config, "resolve_simplipy_engine", lambda cfg, adapter_name: object())
        adapter = run_config.build_model_adapter({
            "type": "subprocess", "worker": "example", "simplipy_engine": "unused",
            "options": {"ridge": 0.1}, "timeout": 30, "env": {"OMP_NUM_THREADS": 4}, "max_restarts": 2})
        assert isinstance(adapter, SubprocessAdapter)
        assert adapter.options == {"ridge": 0.1} and adapter.timeout == 30.0
        assert adapter.env == {"OMP_NUM_THREADS": "4"} and adapter.max_restarts == 2
        assert adapter.python == sys.executable

    def test_subprocess_block_requires_a_worker(self, monkeypatch) -> None:
        from srbf import config as run_config
        monkeypatch.setattr(run_config, "resolve_simplipy_engine", lambda cfg, adapter_name: object())
        with pytest.raises(ValueError, match="worker"):
            run_config.build_model_adapter({"type": "subprocess", "simplipy_engine": "unused"})
