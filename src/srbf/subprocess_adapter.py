"""Out-of-process model adapters: any symbolic-regression method in ITS OWN Python environment.

srbf launches ``srbf/worker/runner.py`` inside the method's interpreter and talks to it over a
local socket (the worker's own stdin/stdout/stderr are captured into the log and never carry
the protocol: a runtime that seizes them, as Julia does under PySR, cannot break the exchange);
the protocol and the worker contract are documented in that file. This module is the srbf side:
it hands each problem to the worker, reads the expression back, parses it with the run's
SimpliPy engine, evaluates it where the worker did not, and fills the result record with the
same fields the in-process adapters produce.
"""
from __future__ import annotations

import collections
import importlib
import json
import math
import os
import queue
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import simplipy
import simplipy.operators  # noqa: F401 - operator realizations reference it by dotted name
from simplipy.utils import codify, safe_f
from symbolic_data.token_ops import normalize_expression, normalize_skeleton

from srbf.core import EvaluationModelAdapter, EvaluationResult, EvaluationSample
from srbf.model_adapters import _compute_fvu_from_predictions, _compute_variable_mask
from srbf.worker import MODELS_DIR, RUNNER_PATH

BUILTIN_WORKERS: dict[str, Path] = {
    "example": MODELS_DIR / "example_worker.py",
    "pysr": MODELS_DIR / "pysr_worker.py",
}


class WorkerError(RuntimeError):
    """The worker process failed outside a single fit (crash, timeout, protocol breach)."""


class WorkerTimeout(WorkerError):
    """No reply within the configured time; the worker has been killed."""


class WorkerCrashed(WorkerError):
    """The worker exited or reported a fatal error."""


def resolve_worker(worker: str | Path) -> Path:
    """A built-in worker name (``example``, ``pysr``) or a script path, as an existing file."""
    if isinstance(worker, str) and worker in BUILTIN_WORKERS:
        return BUILTIN_WORKERS[worker]
    path = Path(worker).expanduser()
    if not path.is_file():
        raise ValueError(
            f"worker script not found: {path} (built-in workers: {', '.join(sorted(BUILTIN_WORKERS))})")
    return path.resolve()


def evaluate_prefix(engine: Any, prefix: list[str], variables: list[str], *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """Evaluate a prefix expression (numeric constants inlined) on each array of rows with the
    engine's own operator realizations; one ``(n, 1)`` column per array, empty for empty input."""
    realized = engine.operators_to_realizations(list(prefix))
    code_string = engine.prefix_to_infix(realized, realization=True)
    code = codify(code_string, list(variables))
    namespace: dict[str, Any] = {"np": np, "numpy": np, "math": math, "simplipy": simplipy}
    for root in getattr(engine, "_realization_roots", {}) or {}:
        try:
            namespace.setdefault(root, importlib.import_module(root))
        except Exception:  # noqa: BLE001 - an unimportable root surfaces as NameError below
            pass
    function = eval(code, namespace)  # noqa: S307 - the engine's own realization code
    outputs = []
    for array in arrays:
        rows = np.asarray(array, dtype=float)
        if rows.ndim != 2 or rows.shape[0] == 0:
            outputs.append(np.empty((0, 1), dtype=float))
            continue
        with np.errstate(all="ignore"):
            values = safe_f(function, rows)
        outputs.append(np.asarray(values, dtype=float).reshape(-1, 1))
    return tuple(outputs)


def _json_meta(record: Mapping[str, Any]) -> dict[str, Any]:
    """The scalar and token-list metadata of a sample, the part a worker may want to see."""
    meta: dict[str, Any] = {}
    for key, value in record.items():
        if value is None or isinstance(value, (str, int, float, bool)):
            meta[key] = value
        elif isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
            meta[key] = list(value)
    return meta


class WorkerProcess:
    """One worker interpreter: spawn, handshake, request/reply with timeouts, shutdown."""

    def __init__(
        self,
        *,
        python: str,
        worker: Path,
        options: Mapping[str, Any],
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
        startup_timeout: float = 600.0,
        log_path: str | None = None,
        stderr_tail: int = 200,
    ) -> None:
        self.python = python
        self.worker = Path(worker)
        self.options = dict(options)
        self.env = {str(k): str(v) for k, v in (env or {}).items()}
        self.cwd = cwd
        self.startup_timeout = float(startup_timeout)
        self.log_path = log_path
        self.info: dict[str, Any] = {}
        self._proc: subprocess.Popen[str] | None = None
        self._listener: socket.socket | None = None
        self._sock: socket.socket | None = None
        self._rfile: Any = None
        self._wfile: Any = None
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._output: collections.deque[str] = collections.deque(maxlen=stderr_tail)
        self._log: Any = None
        self._next_id = 0

    # -- lifecycle ---------------------------------------------------------------------------
    def start(self) -> None:
        env = dict(os.environ)
        env.update(self.env)
        env.setdefault("PYTHONUNBUFFERED", "1")
        if self.log_path:
            self._log = open(self.log_path, "a", encoding="utf-8")
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(1)
        port = self._listener.getsockname()[1]
        self._proc = subprocess.Popen(
            [self.python, str(RUNNER_PATH), "--connect", f"127.0.0.1:{port}", str(self.worker)],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env, cwd=self.cwd,
        )
        threading.Thread(target=self._pump_output, daemon=True).start()
        self._accept(self.startup_timeout)
        threading.Thread(target=self._pump_socket, daemon=True).start()
        self._send({"type": "init", "options": self.options})
        reply = self._receive(self.startup_timeout, expect="ready")
        self.info = dict(reply.get("info") or {})
        self.info["python"] = reply.get("python")
        self.info["python_version"] = reply.get("version")

    def _accept(self, timeout: float) -> None:
        """Wait for the runner to connect, noticing a worker that dies before it does."""
        assert self._listener is not None and self._proc is not None
        deadline = time.monotonic() + timeout
        self._listener.settimeout(0.5)
        while True:
            try:
                self._sock, _ = self._listener.accept()
                break
            except socket.timeout:
                if self._proc.poll() is not None:
                    time.sleep(0.2)  # let the output pump drain the last lines
                    raise WorkerCrashed(self._describe(f"worker exited before connecting (returncode {self._proc.returncode})")) from None
                if time.monotonic() > deadline:
                    self.kill()
                    raise WorkerTimeout(self._describe(f"worker did not connect within {timeout:.0f} s")) from None
        self._listener.close()
        self._listener = None
        self._sock.settimeout(None)
        self._rfile = self._sock.makefile("r", encoding="utf-8")
        self._wfile = self._sock.makefile("w", encoding="utf-8")

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def close(self, grace: float = 5.0) -> None:
        if self._proc is None:
            return
        if self.alive:
            try:
                self._send({"type": "close"})
                self._proc.wait(timeout=grace)
            except Exception:  # noqa: BLE001 - a worker that will not close gets killed
                self.kill()
        self._cleanup()

    def kill(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.kill()
            try:
                self._proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                pass
        self._cleanup()

    def _cleanup(self) -> None:
        for stream in (self._wfile, self._rfile, self._sock, self._listener):
            try:
                if stream is not None:
                    stream.close()
            except Exception:  # noqa: BLE001
                pass
        self._wfile = self._rfile = self._sock = self._listener = None
        if self._proc is not None and self._proc.stdout is not None:
            try:
                self._proc.stdout.close()
            except Exception:  # noqa: BLE001
                pass
        if self._log is not None:
            try:
                self._log.close()
            except Exception:  # noqa: BLE001
                pass
            self._log = None

    # -- protocol ----------------------------------------------------------------------------
    def _pump_socket(self) -> None:
        rfile = self._rfile
        try:
            for line in rfile:
                self._lines.put(line)
        except (ValueError, OSError):  # closed socket
            pass
        self._lines.put(None)

    def _pump_output(self) -> None:
        """The worker's stdout+stderr: kept as a tail for error reports, appended to the log."""
        assert self._proc is not None and self._proc.stdout is not None
        try:
            for line in self._proc.stdout:
                self._output.append(line.rstrip("\n"))
                if self._log is not None:
                    self._log.write(line)
                    self._log.flush()
        except ValueError:  # closed pipe
            pass

    def output_tail(self) -> str:
        return "\n".join(self._output)

    def _describe(self, what: str) -> str:
        tail = self.output_tail()
        return f"{what}\n--- worker output (tail) ---\n{tail}" if tail else what

    def _send(self, payload: Mapping[str, Any]) -> None:
        if self._wfile is None:
            raise WorkerCrashed(self._describe("worker not connected"))
        try:
            self._wfile.write(json.dumps(payload, allow_nan=True) + "\n")
            self._wfile.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            raise WorkerCrashed(self._describe(f"write to worker failed: {exc}")) from exc

    def _receive(self, timeout: float | None, *, expect: str) -> dict[str, Any]:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            try:
                line = self._lines.get(timeout=remaining)
            except queue.Empty:
                self.kill()
                raise WorkerTimeout(self._describe(f"no reply within {timeout:.0f} s")) from None
            if line is None:
                code = self._proc.poll() if self._proc is not None else None
                raise WorkerCrashed(self._describe(f"worker exited (returncode {code})"))
            try:
                message = json.loads(line)
            except ValueError:
                self._output.append("[protocol] unparseable line on stdout: " + line.rstrip())
                continue
            kind = message.get("type")
            if kind == "error":
                raise WorkerCrashed(self._describe(str(message.get("error"))))
            if kind == expect:
                return message
            self._output.append(f"[protocol] unexpected message {kind!r} while waiting for {expect!r}")

    def request(self, payload: Mapping[str, Any], timeout: float | None) -> dict[str, Any]:
        """Send one ``fit`` request and return its ``result`` (a worker-side error is inside it)."""
        self._next_id += 1
        self._send({**payload, "type": "fit", "id": self._next_id})
        while True:
            reply = self._receive(timeout, expect="result")
            if reply.get("id") == self._next_id:
                return reply
            self._output.append(f"[protocol] stale result id {reply.get('id')!r}")


class SubprocessAdapter(EvaluationModelAdapter):
    """Evaluate through a worker running in its own interpreter (see ``srbf/worker/runner.py``).

    Parameters
    ----------
    worker : str | Path
        A built-in worker name (``example``, ``pysr``) or the path of a worker script.
    simplipy_engine : SimpliPyEngine
        The engine the returned expressions are parsed, evaluated and judged with.
    python : str, optional
        The interpreter to run the worker in (default: this one). The method's own venv.
    options : mapping, optional
        Forwarded verbatim to the worker's ``load``/``fit``.
    timeout : float, optional
        Seconds per problem; on expiry the worker is killed, the problem recorded as an error and
        the worker restarted (``max_restarts`` times over the run). None = unlimited.
    drop_unused_variables : bool
        Hand the worker only the columns the ground-truth skeleton uses (the baselines'
        ``padding: false``); the worker sees the kept names in ``variables``.
    """

    def __init__(
        self,
        *,
        worker: str | Path,
        simplipy_engine: Any,
        python: str | None = None,
        options: Mapping[str, Any] | None = None,
        timeout: float | None = None,
        startup_timeout: float = 600.0,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
        drop_unused_variables: bool = True,
        max_restarts: int = 1,
        worker_log: str | None = None,
    ) -> None:
        self.worker = resolve_worker(worker)
        self.simplipy_engine = simplipy_engine
        self.python = python or sys.executable
        self.options = dict(options or {})
        self.timeout = None if timeout is None else float(timeout)
        self.startup_timeout = float(startup_timeout)
        self.env = dict(env or {})
        self.cwd = cwd
        self.drop_unused_variables = bool(drop_unused_variables)
        self.max_restarts = int(max_restarts)
        self.worker_log = worker_log
        self._process: WorkerProcess | None = None
        self._restarts = 0
        self._dead_reason: str | None = None

    def get_simplipy_engine(self) -> Any:  # pragma: no cover - trivial accessor
        """Return this adapter's SimpliPy engine."""
        return self.simplipy_engine

    def worker_info(self) -> dict[str, Any]:
        """What the worker reported at start-up (interpreter, versions), for provenance."""
        return dict(self._process.info) if self._process is not None else {}

    # -- lifecycle ---------------------------------------------------------------------------
    def prepare(self, *, data_source: Any | None = None) -> None:  # type: ignore[override]
        self._spawn()

    def close(self) -> None:
        if self._process is not None:
            self._process.close()
            self._process = None

    def __del__(self) -> None:  # pragma: no cover - best effort
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    def _spawn(self) -> None:
        process = WorkerProcess(
            python=self.python, worker=self.worker, options=self.options, env=self.env, cwd=self.cwd,
            startup_timeout=self.startup_timeout, log_path=self.worker_log,
        )
        process.start()
        self._process = process

    def _recover(self, reason: str) -> None:
        """After a crash or timeout: restart within the budget, else mark the worker dead so the
        remaining problems fail fast instead of hanging."""
        if self._process is not None:
            self._process.kill()
            self._process = None
        if self._restarts >= self.max_restarts:
            self._dead_reason = reason
            return
        self._restarts += 1
        try:
            self._spawn()
        except WorkerError as exc:
            self._dead_reason = str(exc)

    # -- evaluation --------------------------------------------------------------------------
    def evaluate_sample(self, sample: EvaluationSample) -> EvaluationResult:
        record = sample.clone_metadata()
        if self._process is None:
            if self._dead_reason is not None:
                record["error"] = "worker unavailable: " + self._dead_reason.splitlines()[0]
                record["prediction_success"] = False
                return EvaluationResult(record)
            raise RuntimeError("SubprocessAdapter.prepare must be called before evaluation")

        X_support = np.asarray(sample.x_support, dtype=float)
        X_val = np.asarray(sample.x_validation, dtype=float)
        if X_val.ndim != 2:
            X_val = X_val.reshape(0, X_support.shape[1])
        y_support = np.asarray(
            sample.y_support_noisy if sample.y_support_noisy is not None else sample.y_support, dtype=float).reshape(-1)
        variables = list(record.get("variables") or record.get("variable_names") or [])
        if len(variables) != X_support.shape[1]:
            variables = [f"x{i + 1}" for i in range(X_support.shape[1])]
        if self.drop_unused_variables:
            mask, used = _compute_variable_mask(variables, record.get("skeleton"))
            if mask is not None and used:
                X_support = X_support[:, mask]
                X_val = X_val[:, mask] if X_val.shape[0] else X_val.reshape(0, len(used))
                variables = list(used)
                record["variable_names"] = list(used)

        payload = {"x": X_support.tolist(), "y": y_support.tolist(), "x_val": X_val.tolist(),
                   "variables": variables, "meta": _json_meta(record)}
        try:
            reply = self._process.request(payload, self.timeout)
        except WorkerError as exc:
            record["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
            record["error_detail"] = str(exc)
            record["prediction_success"] = False
            self._recover(str(exc))
            return EvaluationResult(record)

        record["fit_time"] = reply.get("fit_time")
        for key, value in (reply.get("extra") or {}).items():
            if key in record:
                record.setdefault("worker_extra", {})[key] = value
            else:
                record[key] = value
        if reply.get("error"):
            detail = str(reply["error"]).rstrip()
            record["error"] = detail.splitlines()[-1] if detail else "worker error"
            record["error_detail"] = detail
            record["prediction_success"] = False
            return EvaluationResult(record)

        expression = reply.get("expression")
        if not expression:
            record["error"] = "worker returned no expression"
            record["prediction_success"] = False
            return EvaluationResult(record)
        record["predicted_expression"] = str(expression)
        try:
            prefix = list(self.simplipy_engine.infix_to_prefix(str(expression)))
            record["predicted_expression_prefix"] = normalize_expression(prefix)
            record["predicted_skeleton_prefix"] = normalize_skeleton(prefix)
        except Exception as exc:  # noqa: BLE001 - parse errors vary by engine
            record["error"] = f"Failed to parse worker expression: {exc}"
            record["prediction_success"] = False
            return EvaluationResult(record)
        if reply.get("constants") is not None:
            record["predicted_constants"] = [float(c) for c in reply["constants"]]

        try:
            y_pred_raw, y_pred_val_raw = reply.get("y_pred"), reply.get("y_pred_val")
            if y_pred_raw is None or (X_val.shape[0] > 0 and y_pred_val_raw is None):
                y_pred, y_pred_val = evaluate_prefix(self.simplipy_engine, prefix, variables, X_support, X_val)
            else:
                y_pred = np.asarray(y_pred_raw, dtype=float).reshape(-1, 1)
                y_pred_val = np.asarray(y_pred_val_raw if y_pred_val_raw is not None else [], dtype=float).reshape(-1, 1)
            if y_pred.shape[0] != X_support.shape[0]:
                raise ValueError(f"{y_pred.shape[0]} predictions for {X_support.shape[0]} support points")
            if X_val.shape[0] and y_pred_val.shape[0] != X_val.shape[0]:
                raise ValueError(f"{y_pred_val.shape[0]} predictions for {X_val.shape[0]} validation points")
        except Exception as exc:  # noqa: BLE001 - evaluation errors vary
            record["error"] = f"Failed to evaluate worker expression: {exc}"
            record["prediction_success"] = False
            return EvaluationResult(record)

        record["y_pred"] = y_pred
        record["y_pred_val"] = y_pred_val
        record["support_fvu"] = _compute_fvu_from_predictions(y_support.reshape(-1, 1), y_pred)
        validation_targets = sample.y_validation_noisy if sample.y_validation_noisy is not None else sample.y_validation
        validation_targets = np.asarray(validation_targets, dtype=float).reshape(-1, 1)
        if validation_targets.size and y_pred_val.size:
            record["validation_fvu"] = _compute_fvu_from_predictions(validation_targets, y_pred_val)
        record["prediction_success"] = True
        return EvaluationResult(record)


__all__ = [
    "BUILTIN_WORKERS", "SubprocessAdapter", "WorkerCrashed", "WorkerError", "WorkerProcess",
    "WorkerTimeout", "evaluate_prefix", "resolve_worker",
]
