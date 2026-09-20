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
import re
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
from symbolic_data.token_ops import desugar_sqrt, normalize_expression, normalize_skeleton

from srbf.core import EvaluationModelAdapter, EvaluationResult, EvaluationSample
from srbf.model_adapters import _compute_fvu_from_predictions
from srbf.variable_renaming import rename_named_variables, rename_named_variables_in_infix, skeleton_variable_names
from srbf.worker import MODELS_DIR, RUNNER_PATH

BUILTIN_WORKERS: dict[str, Path] = {
    path.stem[: -len("_worker")]: path for path in sorted(MODELS_DIR.glob("*_worker.py"))
}
"""The shipped workers by name: every ``<name>_worker.py`` under ``srbf/worker/models`` (``example``,
``pysr``, and whatever a PR adds there)."""


class WorkerError(RuntimeError):
    """The worker process failed outside a single fit (crash, timeout, protocol breach)."""


class WorkerTimeout(WorkerError):
    """No reply within the configured time; the worker has been killed."""


class WorkerCrashed(WorkerError):
    """The worker exited or reported a fatal error."""


class WorkerHang(WorkerTimeout):
    """A worker judged hung while a problem was in flight, and killed. ``kind`` says by which sign: ``idle`` (its
    process tree stopped using the CPU) or ``overdue`` (the problem ran many times longer than the problems before
    it -- a worker that is stuck while busy looks healthy to every CPU measure)."""

    def __init__(self, message: str, kind: str = "idle") -> None:
        super().__init__(message)
        self.kind = kind


def tree_cpu_seconds(pid: int) -> float | None:
    """CPU seconds (user + system, all threads) used so far by ``pid`` and every live descendant, read from
    /proc; None where /proc is unavailable (the idle watchdog then stays off and the hard timeout remains)."""
    tick = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
    children: dict[int, list[int]] = collections.defaultdict(list)
    used: dict[int, float] = {}
    try:
        entries = os.listdir("/proc")
    except OSError:
        return None
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat", encoding="utf-8") as fh:
                fields = fh.read().rsplit(")", 1)[1].split()
        except (OSError, IndexError):
            continue                                # the process ended while we looked
        children[int(fields[1])].append(int(entry))  # fields[1] = ppid (field 4 of stat)
        used[int(entry)] = (int(fields[11]) + int(fields[12])) / tick   # utime + stime (fields 14, 15)
    if pid not in used:
        return None
    total, stack = 0.0, [pid]
    while stack:
        current = stack.pop()
        total += used.get(current, 0.0)
        stack.extend(children.get(current, []))
    return total


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


# What a worker is told about a problem besides its data: which problem it is and how it was sampled. The law
# itself (its skeleton, its expression, its constants, its complexity) stays on srbf's side of the socket.
WORKER_META_KEYS = ("benchmark_eq_id", "eq_id", "catalog", "eval_row_index", "n_support", "noise_level", "variables", "variable_names")


def _json_meta(record: Mapping[str, Any]) -> dict[str, Any]:
    """The identifiers and sampling parameters of a sample: what a worker may see of a problem's metadata."""
    meta: dict[str, Any] = {}
    for key in WORKER_META_KEYS:
        value = record.get(key)
        if value is None or isinstance(value, (str, int, float, bool)):
            meta[key] = value
        elif isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
            meta[key] = list(value)
    return meta


# SymPy's printer, and with it most methods, spells two things differently from the engine's reader.
_PRINTER_SPELLINGS = ((re.compile(r"\bAbs\("), "abs("), (re.compile(r"(?<![\w.])E(?![\w.(])"), "e"))


def _unknown_symbols(prefix: list[str], operator_arity: Mapping[str, int]) -> list[str]:
    """Tokens the judge cannot read: not an operator, a variable, a number or a named constant."""
    def known(token: str) -> bool:
        if token in operator_arity or token in ("sqrt", "np.pi", "np.e", "<constant>") or re.fullmatch(r"x\d+", token):
            return True
        try:
            float(token)
            return True
        except ValueError:
            return False
    return sorted({token for token in prefix if not known(token)})


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
        hang_after_idle_s: float | None = None,
        hang_idle_cpu_s: float = 1.0,
    ) -> None:
        self.python = python
        self.hang_after_idle_s = None if hang_after_idle_s is None else float(hang_after_idle_s)
        self.hang_idle_cpu_s = float(hang_idle_cpu_s)
        self.last_window_cpu_s: float | None = None   # CPU seconds of the worker tree over the last idle window
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
            os.makedirs(os.path.dirname(os.path.abspath(self.log_path)) or ".", exist_ok=True)
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

    def _receive(self, timeout: float | None, *, expect: str, watch: bool = False,
                 overdue_s: float | None = None) -> dict[str, Any]:
        """The next ``expect`` message. With ``watch`` and an idle window set, the worker's process tree is
        sampled while it works: under ``hang_idle_cpu_s`` CPU seconds over ``hang_after_idle_s`` seconds of
        wall is a hang, whatever the hard timeout says.
        ``overdue_s`` is the caller's second sign of a hang: a reply that has not come after that many seconds."""
        began = time.monotonic()
        deadline = None if timeout is None else began + timeout
        window = self.hang_after_idle_s if watch else None
        samples: collections.deque[tuple[float, float]] = collections.deque()
        pid = self._proc.pid if self._proc is not None else None
        while True:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            polled = window is not None or overdue_s is not None
            wait = remaining if not polled else (2.0 if remaining is None else min(remaining, 2.0))
            try:
                line = self._lines.get(timeout=wait)
            except queue.Empty:
                if deadline is not None and time.monotonic() >= deadline:
                    self.kill()
                    raise WorkerTimeout(self._describe(f"no reply within {timeout:.0f} s")) from None
                if overdue_s is not None and time.monotonic() - began >= overdue_s:
                    self.kill()
                    raise WorkerHang(self._describe(
                        f"overdue: no reply after {time.monotonic() - began:.0f} s (limit {overdue_s:.0f} s)"), "overdue") from None
                cpu = tree_cpu_seconds(pid) if (window is not None and pid is not None) else None
                if cpu is not None and window is not None:
                    now = time.monotonic()
                    samples.append((now, cpu))
                    while len(samples) > 1 and now - samples[1][0] >= window:
                        samples.popleft()                   # keep exactly one sample at or beyond the window edge
                    if now - samples[0][0] >= window:
                        self.last_window_cpu_s = cpu - samples[0][1]
                        if self.last_window_cpu_s < self.hang_idle_cpu_s:
                            self.kill()
                            raise WorkerHang(self._describe(
                                f"no CPU activity: {self.last_window_cpu_s:.2f} CPU s in the last {window:.0f} s")) from None
                continue
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

    def request(self, payload: Mapping[str, Any], timeout: float | None, overdue_s: float | None = None) -> dict[str, Any]:
        """Send one ``fit`` request and return its ``result`` (a worker-side error is inside it)."""
        self._next_id += 1
        self._send({**payload, "type": "fit", "id": self._next_id})
        while True:
            reply = self._receive(timeout, expect="result", watch=True, overdue_s=overdue_s)
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
        Accepted so that existing configs keep loading, and ignored: a method sees every column of a
        problem, and choosing the relevant ones is part of what is evaluated.
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
        drop_unused_variables: bool = False,
        max_restarts: int = 1,
        worker_log: str | None = None,
        hang_after_idle_s: float | None = None,
        hang_idle_cpu_s: float = 1.0,
        hang_overdue_factor: float | None = None,
        hang_overdue_floor_s: float = 60.0,
        hang_overdue_min_history: int = 10,
        hang_log: str | None = None,
    ) -> None:
        self.worker = resolve_worker(worker)
        # The hang policy (off unless hang_after_idle_s is set): a worker that goes idle for that long while a
        # problem is in flight, or misses the hard timeout, is killed and the problem retried ONCE in a fresh
        # worker; a second hang fails the row. Every hang is logged to hang_log (JSON lines) and counted in the
        # row's `worker_hangs`; hang restarts do not spend max_restarts, which stays the budget for crashes.
        self.hang_after_idle_s = None if hang_after_idle_s is None else float(hang_after_idle_s)
        self.hang_idle_cpu_s = float(hang_idle_cpu_s)
        # The second sign of a hang: OVERDUE. A stall need not be idle -- a search can have ended while one thread
        # is busy for minutes to hours exporting a pathological result, which is exactly what a healthy short
        # fit looks like to a CPU counter. What gives it away is time: the
        # problems before it took 2 s and this one has taken 60. So a problem is overdue after
        # max(hang_overdue_floor_s, hang_overdue_factor x the median successful request of THIS run), once
        # hang_overdue_min_history requests have answered; before that only the idle window and the hard timeout
        # apply. The limit scales with the rung by construction. Off unless hang_overdue_factor is set.
        self.hang_overdue_factor = None if hang_overdue_factor is None else float(hang_overdue_factor)
        self.hang_overdue_floor_s = float(hang_overdue_floor_s)
        self.hang_overdue_min_history = int(hang_overdue_min_history)
        self._answered_s: list[float] = []            # wall seconds of the requests that answered, this run
        self.hang_log = hang_log
        self.simplipy_engine = simplipy_engine
        self.python = python or sys.executable
        self.options = dict(options or {})
        self.timeout = None if timeout is None else float(timeout)
        self.startup_timeout = float(startup_timeout)
        self.env = dict(env or {})
        self.cwd = cwd
        self.drop_unused_variables = False   # see the parameter's note: accepted, ignored
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
            hang_after_idle_s=self.hang_after_idle_s, hang_idle_cpu_s=self.hang_idle_cpu_s,
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

    def _restart_after_hang(self, reason: str) -> None:
        """A hang kills the worker; its restart does not spend the crash budget (max_restarts)."""
        if self._process is not None:
            self._process.kill()
            self._process = None
        try:
            self._spawn()
        except WorkerError as exc:
            self._dead_reason = str(exc)

    def _overdue_limit(self) -> float | None:
        """Seconds after which the problem in flight is overdue, or None while the rule is off or has too little
        to go on. The median, not the mean or the maximum: one slow answer must not move the bar."""
        if self.hang_overdue_factor is None or len(self._answered_s) < self.hang_overdue_min_history:
            return None
        ordered = sorted(self._answered_s)
        median = ordered[len(ordered) // 2] if len(ordered) % 2 else 0.5 * (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2])
        return max(self.hang_overdue_floor_s, self.hang_overdue_factor * median)

    def _log_hang(self, record: Mapping[str, Any], **event: Any) -> None:
        """One JSON line per hang event (and per recovery), for the statistics reported later."""
        if not self.hang_log:
            return
        keys = ("benchmark_eq_id", "eq_id", "catalog", "ground_truth_infix", "skeleton_hash")
        line = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "worker": self.worker.stem,
                **{k: record.get(k) for k in keys if record.get(k) is not None}, **event}
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.hang_log)) or ".", exist_ok=True)
            with open(self.hang_log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(line, default=str) + "\n")
        except OSError:
            pass                                          # a lost log line never fails the problem

    # -- evaluation --------------------------------------------------------------------------
    def evaluate_sample(self, sample: EvaluationSample, *, extra_meta: Mapping[str, Any] | None = None) -> EvaluationResult:
        """``extra_meta`` rides along in the fit payload's ``meta`` (per-problem worker inputs such as
        a caller's per-sample overrides such as initial guesses or an iteration budget); the worker decides what to do with them."""
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

        payload = {"x": X_support.tolist(), "y": y_support.tolist(), "x_val": X_val.tolist(),
                   "variables": variables, "meta": {**_json_meta(record), **dict(extra_meta or {})}}
        hangs = 0
        policy = self.hang_after_idle_s is not None or self.hang_overdue_factor is not None
        while True:
            started = time.monotonic()
            limit = self._overdue_limit()
            try:
                reply = self._process.request(payload, self.timeout, overdue_s=limit)
                self._answered_s.append(time.monotonic() - started)
                break
            except WorkerTimeout as exc:
                if not policy:                           # no hang policy: the historical timeout handling
                    record["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
                    record["error_detail"] = str(exc)
                    record["prediction_success"] = False
                    self._recover(str(exc))
                    return EvaluationResult(record)
                hangs += 1
                window_cpu = self._process.last_window_cpu_s if self._process is not None else None
                self._log_hang(record, attempt=hangs, kind=exc.kind if isinstance(exc, WorkerHang) else "timeout",
                               seconds=time.monotonic() - started, window_cpu_s=window_cpu,
                               outcome="retry" if hangs == 1 else "failed", limit_s=limit)
                self._restart_after_hang(str(exc))
                if hangs >= 2 or self._process is None:
                    record["error"] = f"hung twice ({type(exc).__name__}: {str(exc).splitlines()[0]})"
                    record["error_detail"] = str(exc)
                    record["prediction_success"] = False
                    record["worker_hangs"] = hangs
                    return EvaluationResult(record)
            except WorkerError as exc:
                record["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
                record["error_detail"] = str(exc)
                record["prediction_success"] = False
                record["worker_hangs"] = hangs
                self._recover(str(exc))
                return EvaluationResult(record)
        if policy:
            record["worker_hangs"] = hangs
            if hangs:
                self._log_hang(record, attempt=hangs + 1, kind="answer", seconds=None, window_cpu_s=None,
                               outcome="recovered")

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
        # A worker answers in the column names it was handed (PySR spells them v1, v2 on a catalog
        # that calls its columns that); the ground truth spells the same columns x1, x2, ... , so
        # both the stored expression and its prefix are mapped back before anything is judged.
        names = skeleton_variable_names(variables)
        expression = rename_named_variables_in_infix(str(expression), variables)
        for pattern, spelling in _PRINTER_SPELLINGS:      # after the renaming: a variable named E stays a variable
            expression = pattern.sub(spelling, expression)
        record["predicted_expression"] = expression
        try:
            # read_infix, not the raw infix_to_prefix: the reader's own tokens ('**', 'neg' on a
            # literal) are not the engine grammar, and simplify/complexity refuse them. The reader
            # takes any identifier as a variable, so the map runs FIRST: everything downstream --
            # the canonical form, the price, the judge -- then sees one spelling.
            prefix = rename_named_variables(list(self.simplipy_engine.read_infix(expression)), variables) or []
            # SymPy and most methods print the square root as sqrt(...); the engine's vocabulary spells it
            # rootn(u, 2), and its lenient reader would pass the bare token on to a walk that rejects it.
            prefix = [str(token) for token in prefix]
            unknown = _unknown_symbols(prefix, self.simplipy_engine.operator_arity)
            if unknown:
                raise ValueError(f"the judge does not read {', '.join(unknown)}; it reads + - * / ** and the functions "
                                 f"{' '.join(sorted(k for k, n in self.simplipy_engine.operator_arity.items() if k.isalpha()))} and sqrt")
            prefix = desugar_sqrt(prefix, dict(self.simplipy_engine.operator_arity))
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
                y_pred, y_pred_val = evaluate_prefix(self.simplipy_engine, prefix, names, X_support, X_val)
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
