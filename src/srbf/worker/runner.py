"""The srbf worker runner: runs a model-side worker script inside ITS OWN interpreter and speaks
the srbf worker protocol over a local socket.

Standard library only, executed by path (``<python> runner.py --connect 127.0.0.1:PORT <worker.py>``
or ``... --module some.module``): the model's environment needs neither srbf nor any of its
dependencies. The protocol travels over the socket srbf listens on; the worker's stdin, stdout
and stderr stay plain streams (srbf captures them into its log), so a runtime that seizes or
closes them, as Julia does under PySR, cannot touch the protocol.

Protocol, one JSON object per line (``NaN``/``Infinity`` allowed: both ends are Python)::

    -> {"type": "init", "options": {...}, "warmup": false}
    <- {"type": "ready", "python": "...", "version": "3.x.y", "worker": "...", "info": {...},
        "load_time": 0.5, "setup_time": 2.1 | null}
    -> {"type": "fit", "id": 7, "x": [[...], ...], "y": [...], "x_val": [[...], ...],
        "variables": ["x1", ...], "meta": {...}}
    <- {"type": "result", "id": 7, "expression": "...", "fit_time": 1.23,
        "y_pred": [...] | null, "y_pred_val": [...] | null, "constants": [...] | null,
        "candidates": ["...", ...] | null, "candidates_dropped": 0, "extra": {...}, "error": null | "..."}
    -> {"type": "close"}
    <- {"type": "closed"}

Worker contract -- a Python file (or module) defining:

``fit(x, y, *, x_val, variables, meta, options, state) -> dict``
    Required. ``x`` is a list of rows (one list of floats per support point), ``y`` the targets,
    ``x_val`` the validation rows (possibly empty), ``variables`` the column names of ``x`` in
    order: the returned expression must use exactly these names. Returns a dict with at least
    ``"expression"``, an infix string in plain arithmetic notation (``2*x1 + x2**3``, function
    calls for unary operators) with numeric constants, in the operator vocabulary of the SimpliPy
    engine the run is judged with (``srbf_worker_helpers.respell_legacy_prefix`` converts the
    pre-0.12 vocabulary). Optional keys: ``y_pred`` and ``y_pred_val`` (the model's own
    predictions; srbf evaluates the expression itself when they are absent), ``constants``,
    ``fit_time`` (seconds; measured around the call when absent), ``extra`` (a JSON-serializable
    dict merged into the result record), ``error`` (a message; the sample counts as failed).
    ``"candidates"``: every expression the method ends with, a list of infix strings like
    ``expression`` (each with its fitted constants written in): a method that keeps a diverse front
    of its own (a hall of fame, a Pareto front) returns that front, not its whole population; one
    that fits several expressions (samples, beams) returns those; a method with one model returns
    none. srbf judges and stores every candidate and the expression with them; ``expression``
    stays the answer. A method that ends with candidates and no rule of its own for picking one
    returns ``"candidates"`` without ``"expression"``; srbf then picks one only if the config sets
    ``selection``, and records the problem as failed otherwise. At most ``MAX_CANDIDATES``
    candidates of at most ``MAX_CANDIDATE_CHARS`` characters each are kept (the first ones); the
    rest are dropped and counted in ``candidates_dropped``.
``load(options) -> state``
    Optional. Called once after ``init`` with the config's ``options``; the returned object is
    passed to every ``fit`` as ``state``.
``setup(state, options) -> state | None``
    Optional. Data-independent preparation after ``load``: compiling, warming up on a problem of
    the worker's own. It runs before the clock starts, in every worker process srbf starts (after
    every restart too). A returned object other than None replaces ``state``. Written as
    ``setup(state, options, *, warmup)``, it is told whether srbf's warm-up problem follows
    (``init``'s ``warmup``), so that it can leave a throwaway fit of its own to that problem.
``info(state) -> dict``
    Optional. Versions and provenance for the ``ready`` message.

``ready`` reports the seconds ``load`` and ``setup`` took (``setup_time`` is null without a
``setup``); srbf records them in the run's metadata, never in a fit time. srbf's own warm-up
problem, when the run asks for one, arrives as an ordinary ``fit`` whose ``meta`` says
``"warmup": true``; its answer is discarded.
"""
import importlib
import importlib.util
import inspect
import json
import os
import socket
import sys
import time
import traceback

USAGE = ("usage: python runner.py --connect HOST:PORT <worker.py>\n"
         "       python runner.py --connect HOST:PORT --module <importable.module>")

# The protocol's limits on a fit's candidates (srbf.candidates holds the same numbers).
MAX_CANDIDATES = 100000
MAX_CANDIDATE_CHARS = 10000


def _jsonable(value):
    """Convert numpy scalars/arrays and nested containers into plain JSON-serializable values."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item") and not isinstance(value, (list, tuple, dict)):
        try:
            return _jsonable(value.item())
        except Exception:  # noqa: BLE001 - fall through to the generic cases
            pass
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return str(value)


def _load_worker(target, as_module):
    if as_module:
        return importlib.import_module(target)
    path = os.path.abspath(target)
    if not os.path.isfile(path):
        raise FileNotFoundError("worker script not found: %s" % path)
    sys.path.append(os.path.dirname(path))  # the worker may import its siblings; installed packages win
    spec = importlib.util.spec_from_file_location("srbf_worker_script", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["srbf_worker_script"] = module
    spec.loader.exec_module(module)
    return module


def _result(msg_id):
    return {"type": "result", "id": msg_id, "expression": None, "fit_time": None, "y_pred": None,
            "y_pred_val": None, "constants": None, "candidates": None, "candidates_dropped": 0, "extra": {},
            "error": None}


def _limited(candidates):
    """The candidates within the protocol's limits, and how many were dropped (a missing entry counts)."""
    kept, dropped = [], 0
    for candidate in candidates:
        text = None if candidate is None else str(candidate)
        if text is None or len(text) > MAX_CANDIDATE_CHARS or len(kept) >= MAX_CANDIDATES:
            dropped += 1
            continue
        kept.append(text)
    return kept, dropped


def _takes_warmup(setup):
    """Whether a worker's ``setup`` takes the ``warmup`` keyword (``setup(state, options, *, warmup)``)."""
    try:
        parameters = inspect.signature(setup).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.kind == p.VAR_KEYWORD or (p.name == "warmup" and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY))
               for p in parameters)


def serve(worker, target, rfile, wfile):
    """Serve the protocol for ``worker`` on the line-oriented ``rfile``/``wfile``; returns the exit code."""
    def send(obj):
        wfile.write(json.dumps(obj, allow_nan=True) + "\n")
        wfile.flush()

    fit = getattr(worker, "fit", None)
    if not callable(fit):
        send({"type": "error", "error": "worker %s defines no fit()" % target})
        return 3

    state = None
    options = {}
    for line in rfile:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError as exc:
            send({"type": "error", "error": "malformed request: %s" % exc})
            continue
        kind = msg.get("type")
        if kind == "init":
            options = msg.get("options") or {}
            try:
                started = time.perf_counter()
                load = getattr(worker, "load", None)
                state = load(options) if callable(load) else None
                load_time = time.perf_counter() - started
                setup_time = None
                setup = getattr(worker, "setup", None)
                if callable(setup):
                    started = time.perf_counter()
                    if _takes_warmup(setup):
                        prepared = setup(state, options, warmup=bool(msg.get("warmup")))
                    else:
                        prepared = setup(state, options)
                    setup_time = time.perf_counter() - started
                    if prepared is not None:
                        state = prepared
                info_fn = getattr(worker, "info", None)
                info = info_fn(state) if callable(info_fn) else {}
            except Exception:  # noqa: BLE001 - the whole traceback goes back to srbf
                send({"type": "error", "error": traceback.format_exc()})
                return 1
            send({"type": "ready", "python": sys.executable, "version": sys.version.split()[0],
                  "worker": target, "info": _jsonable(info or {}), "load_time": load_time,
                  "setup_time": setup_time})
        elif kind == "fit":
            reply = _result(msg.get("id"))
            started = time.perf_counter()
            try:
                out = fit(msg["x"], msg["y"], x_val=msg.get("x_val") or [], variables=msg.get("variables") or [],
                          meta=msg.get("meta") or {}, options=options, state=state)
                elapsed = time.perf_counter() - started
                if not isinstance(out, dict) or not {"expression", "candidates", "error"} & set(out):
                    raise TypeError("fit() must return a dict with an 'expression' key (or 'candidates', or an 'error' "
                                    "key to fail the problem), got %r" % type(out).__name__)
                for key in ("expression", "y_pred", "y_pred_val", "constants"):
                    reply[key] = _jsonable(out.get(key))
                if reply["expression"] is not None:
                    reply["expression"] = str(reply["expression"])
                if out.get("candidates") is not None:
                    reply["candidates"], reply["candidates_dropped"] = _limited(out["candidates"])
                reply["fit_time"] = float(out.get("fit_time", elapsed))
                reply["extra"] = _jsonable(out.get("extra") or {})
                if out.get("error") is not None:
                    reply["error"] = str(out["error"])
            except Exception:  # noqa: BLE001 - reported per sample, the worker stays up
                reply["fit_time"] = time.perf_counter() - started
                reply["error"] = traceback.format_exc(limit=12)
            send(reply)
        elif kind == "close":
            send({"type": "closed"})
            return 0
        else:
            send({"type": "error", "error": "unknown request type %r" % (kind,)})
    return 0


def _parse_args(argv):
    args = list(argv[1:])
    connect = None
    as_module = False
    target = None
    while args:
        a = args.pop(0)
        if a == "--connect" and args:
            connect = args.pop(0)
        elif a == "--module":
            as_module = True
        elif target is None:
            target = a
        else:
            return None
    if connect is None or target is None or ":" not in connect:
        return None
    host, port = connect.rsplit(":", 1)
    return host, int(port), target, as_module


def main(argv):
    parsed = _parse_args(argv)
    if parsed is None:
        sys.stderr.write(USAGE + "\n")
        return 2
    host, port, target, as_module = parsed
    # srbf_worker_helpers.py sits next to this file; make it importable in the foreign interpreter.
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    sock = socket.create_connection((host, port), timeout=60)
    sock.settimeout(None)
    rfile = sock.makefile("r", encoding="utf-8")
    wfile = sock.makefile("w", encoding="utf-8")
    try:
        try:
            worker = _load_worker(target, as_module)
        except Exception:  # noqa: BLE001 - srbf reads this from the error line
            wfile.write(json.dumps({"type": "error", "error": traceback.format_exc()}) + "\n")
            wfile.flush()
            return 3
        try:
            return serve(worker, target, rfile, wfile)
        except KeyboardInterrupt:
            return 130
    finally:
        for stream in (wfile, rfile):
            try:
                stream.close()
            except Exception:  # noqa: BLE001
                pass
        sock.close()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
