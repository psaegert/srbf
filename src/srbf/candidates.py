"""Every candidate formula a method ends with, stored next to the run's results and judged by srbf itself.

A method answers each problem with one expression, the record's prediction, and often ends with more: a hall of fame,
a Pareto front, the samples or beams it fitted. A run keeps all of them, problem by problem, and judges every one the
way it judges a prediction: its price in bits (:func:`srbf.pricing.price_bits`, the one price srbf computes), its FVU on the
fitting and on the held-out points (:func:`srbf.metrics.numeric.fvu`, against the targets the table judges with) and
whether it could be read, evaluated and priced at all (``status``). What a method reports about its own candidates is
never used.

Who hands what over:

* a worker returns ``candidates`` beside (or instead of) its ``expression`` (the protocol in :mod:`srbf.worker.runner`);
  the subprocess adapter passes them on with the expression as the prediction;
* an adapter inside srbf's environment puts them into its record (:data:`CANDIDATE_FIELDS`); the Flash-ANSR adapter
  hands over every candidate it fitted;
* an adapter that hands over nothing has its prediction as its only candidate.

The run takes the candidates out of the record after the method's call, so judging them is never part of ``fit_time``,
and appends them, one record per problem, to the log of its result file in the run's store: ``choices_000064.pkl``
writes ``choices_000064.candidates/choices_000064.log`` (:func:`candidate_store_path`; each shard of the run writes a
log of its own into the same directory). :func:`iter_candidates` and :func:`load_candidates` read a run's judged
candidates back as arrays, problem by problem; :func:`rejudge_candidates` judges a stored run again when the judge
changes (another price, say), from the stored strings and the points the result file keeps.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import math
import os
import pickle
import re
import signal
import struct
import threading
import time
import warnings
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from srbf.metrics.numeric import fvu
from srbf.pricing import pricer
from srbf.table import judge_dependencies
from srbf.variable_renaming import (
    rename_named_variables, rename_named_variables_in_infix, rename_prediction, skeleton_variable_names)

#: Bumped whenever a stored problem's columns change meaning.
STORE_FORMAT = 1
#: The candidate store of ``<dir>/choices_000064.pkl`` is ``<dir>/choices_000064.candidates``.
STORE_SUFFIX = ".candidates"
#: Bumped whenever a judgement changes without a change to srbf's source (it is part of the judge's fingerprint).
JUDGE_VERSION = 1

#: A worker's candidates beyond this many are dropped (the first ones are kept) and counted.
MAX_CANDIDATES = 100_000
#: A worker's candidate written with more characters than this is dropped and counted.
MAX_CANDIDATE_CHARS = 10_000
#: Seconds the judge spends on one candidate (reading, pricing and evaluating it) before it gives up on it: the
#: candidate is unevaluable, with the reason, and the judge goes on with the next one (:func:`judge_candidates`).
JUDGE_TIMEOUT_S = 10.0

STATUS_JUDGED = 0        # read, evaluated and priced
STATUS_UNREADABLE = 1    # the judge's engine cannot read it (a function outside the vocabulary, a placeholder, ...)
STATUS_UNEVALUABLE = 2   # read, but evaluating it on the points raised; or judging it took over JUDGE_TIMEOUT_S
STATUS_UNPRICED = 3      # read and evaluated, but the engine gives it no finite price
STATUS_NAMES = ("judged", "unreadable", "unevaluable", "unpriced")

#: How a candidate is written: ``infix`` in the column names the method was handed (what a worker returns), or
#: ``prefix`` tokens joined by spaces in ``x1, x2, ...`` by column position (what srbf's in-process adapters keep).
NOTATIONS = ("infix", "prefix")

#: The record keys through which an adapter hands its candidates to the run, which takes them out of the record:
#: ``candidate_expressions`` (list of strings), ``candidate_notation`` (one of :data:`NOTATIONS`, default ``infix``),
#: ``candidate_variables`` (the column names the method was handed; default the record's ``variables``),
#: ``candidate_prediction`` (the index of the method's own pick among them, or None) and ``candidates_dropped`` (how
#: many the adapter could not keep).
CANDIDATE_FIELDS = ("candidate_expressions", "candidate_notation", "candidate_variables", "candidate_prediction",
                    "candidates_dropped")


# ---- reading an expression ----------------------------------------------------------------------------------------
# SymPy's printer, and with it most methods, spells two things differently from the engine's reader.
_PRINTER_SPELLINGS = ((re.compile(r"\bAbs\("), "abs("), (re.compile(r"(?<![\w.])E(?![\w.(])"), "e"))


def spell_expression(expression: str, variables: Sequence[str]) -> str:
    """A method predicts in the column names it was handed (PySR spells them ``v1, v2`` on a catalog that calls its
    columns that); the ground truth spells the same columns ``x1, x2, ...``, so the expression is mapped back before
    anything is judged, and the printers' constant names are written as the engine reads them."""
    expression = rename_named_variables_in_infix(expression, variables)
    for pattern, spelling in _PRINTER_SPELLINGS:      # after the renaming: a variable named E stays a variable
        expression = pattern.sub(spelling, expression)
    return expression


def unknown_symbols(prefix: Sequence[str], operator_arity: Mapping[str, int]) -> list[str]:
    """Tokens the judge cannot read: not an operator, a variable, a number or a named constant."""
    def known(token: str) -> bool:
        if token in operator_arity or token in ("sqrt", "np.pi", "np.e", "<constant>") or re.fullmatch(r"x\d+", token):
            return True
        try:
            float(token)
            return True
        except ValueError:
            return False
    return sorted({str(token) for token in prefix if not known(str(token))})


def read_spelled(engine: Any, expression: str, variables: Sequence[str]) -> list[str]:
    """The engine's prefix of an infix expression already passed through :func:`spell_expression`; raises when the
    judge cannot read it."""
    from symbolic_data.token_ops import desugar_sqrt

    # read_infix, not the raw infix_to_prefix: the reader's own tokens ('**', 'neg' on a
    # literal) are not the engine grammar, and simplify/complexity refuse them. The reader
    # takes any identifier as a variable, so the map runs FIRST: everything downstream --
    # the canonical form, the price, the judge -- then sees one spelling.
    prefix = rename_named_variables(list(engine.read_infix(expression)), variables) or []
    # SymPy and most methods print the square root as sqrt(...); the engine's vocabulary spells it
    # rootn(u, 2), and its lenient reader would pass the bare token on to a walk that rejects it.
    prefix = [str(token) for token in prefix]
    unknown = unknown_symbols(prefix, engine.operator_arity)
    if unknown:
        raise ValueError(f"the judge does not read {', '.join(unknown)}; it reads + - * / ** and the functions "
                         f"{' '.join(sorted(k for k, n in engine.operator_arity.items() if k.isalpha()))} and sqrt")
    return list(desugar_sqrt(prefix, dict(engine.operator_arity)))


def read_expression(engine: Any, expression: str, variables: Sequence[str]) -> list[str]:
    """The prefix the judge reads from a method's infix string in the column names ``variables``."""
    return read_spelled(engine, spell_expression(expression, variables), variables)


def evaluate_prefix(engine: Any, prefix: Sequence[str], variables: Sequence[str], *arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """Evaluate a prefix expression (numeric constants inlined) on each array of rows with the
    engine's own operator realizations; one ``(n, 1)`` column per array, empty for empty input."""
    import simplipy
    import simplipy.operators  # noqa: F401 - operator realizations reference it by dotted name
    from simplipy.utils import codify, safe_f

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


def limit_candidates(candidates: Sequence[Any] | None) -> tuple[list[str], int]:
    """A worker's candidates within the protocol's limits: the first :data:`MAX_CANDIDATES` strings of at most
    :data:`MAX_CANDIDATE_CHARS` characters, and how many were dropped (a missing entry counts as dropped)."""
    kept: list[str] = []
    dropped = 0
    for candidate in candidates or []:
        text = None if candidate is None else str(candidate)
        if text is None or len(text) > MAX_CANDIDATE_CHARS or len(kept) >= MAX_CANDIDATES:
            dropped += 1
            continue
        kept.append(text)
    return kept, dropped


# ---- judging --------------------------------------------------------------------------------------------------------
@dataclass
class Judgement:
    """srbf's judgement of a list of candidates, one entry per candidate in the order given."""

    prefixes: list[str]          # the prefix the judge read, tokens joined by spaces ('' when unreadable)
    reasons: list[str]           # why a candidate is not judged ('' when it is)
    bits: np.ndarray             # float64, the price in bits; NaN when unreadable or unpriced
    fvu_fit: np.ndarray          # float64, FVU on the fitting points; NaN when not evaluated
    fvu_val: np.ndarray          # float64, FVU on the held-out points; NaN when not evaluated or there are none
    status: np.ndarray           # uint8, STATUS_*

    def __len__(self) -> int:
        return len(self.prefixes)


def _first_line(exc: BaseException) -> str:
    text = str(exc).strip()
    return (f"{type(exc).__name__}: {text.splitlines()[0]}" if text else type(exc).__name__)[:300]


class _CandidateTimeout(BaseException):
    """Raised into a candidate's judgement when its time is up. A BaseException, so that the ``except Exception`` of the
    code it interrupts (the price's, the engine's, the evaluation's) cannot swallow it."""


def _alarm(signum: int, frame: Any) -> None:
    raise _CandidateTimeout


@contextlib.contextmanager
def _time_limit(seconds: float | None) -> Iterator[None]:
    """Raise :class:`_CandidateTimeout` into the body once ``seconds`` have passed.

    The interval timer of this process (``setitimer(ITIMER_REAL)``) and its ``SIGALRM``: nothing is started, so nothing
    can be left running. A thread cannot be stopped from outside, so a judgement that hangs would keep its thread (and
    the GIL) after a thread-based timeout; a process per candidate would cost more than the judgement and would have to
    be reaped. The handler runs between two bytecodes of the main thread: a call into native code (numpy, a compiled
    part of the engine) finishes before it takes effect. Off the main thread, or without ``setitimer``, the body runs
    without a limit. The handler and an alarm someone else had set (a test runner's timeout) are put back, that alarm
    less the time spent here."""
    if (seconds is None or not hasattr(signal, "setitimer")
            or threading.current_thread() is not threading.main_thread()):
        yield
        return
    previous = signal.signal(signal.SIGALRM, _alarm)
    started = time.monotonic()
    outer = (0.0, 0.0)
    try:
        outer = signal.setitimer(signal.ITIMER_REAL, float(seconds))
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)    # the alarm fires once: if it came first, nothing is left
    finally:
        signal.signal(signal.SIGALRM, signal.SIG_DFL if previous is None else previous)
        if outer[0] > 0.0:
            signal.setitimer(signal.ITIMER_REAL, max(outer[0] - (time.monotonic() - started), 1e-6), outer[1])


def _read_candidate(engine: Any, text: str, notation: str, variables: Sequence[str]) -> list[str]:
    """The prefix the judge prices and evaluates: read, in the ground truth's variable names, in the form the table
    stores and judges a prediction in (``normalize_expression``, then the engine's ``convert_expression``)."""
    from symbolic_data.token_ops import normalize_expression

    from srbf.spelling import engine_spelling

    if notation == "infix":
        prefix = read_expression(engine, text, variables)
    elif notation == "prefix":
        prefix = engine_spelling(rename_prediction(text.split(), variables) or [], engine.operator_arity)
        unknown = unknown_symbols(prefix, engine.operator_arity)
        if unknown:
            raise ValueError(f"the judge does not read {', '.join(unknown)}")
    else:
        raise ValueError(f"unknown notation {notation!r}; one of {', '.join(NOTATIONS)}")
    if not prefix:
        raise ValueError("an empty expression")
    if "<constant>" in prefix:
        raise ValueError("a placeholder where a fitted constant belongs: candidates carry their constants")
    prefix = list(normalize_expression(list(prefix)))
    try:
        prefix = list(engine.convert_expression(prefix))
    except Exception:  # noqa: BLE001 - as the table does: a prefix the conversion refuses is judged as read
        pass
    return [str(token) for token in prefix]


def judge_candidates(expressions: Sequence[str], *, engine: Any, variables: Sequence[str], x_fit: Any, y_fit: Any,
                     x_val: Any = None, y_val: Any = None, notation: str = "infix",
                     timeout: float | None = JUDGE_TIMEOUT_S) -> Judgement:
    """Judge every candidate as srbf judges a prediction.

    ``variables`` are the column names the method was handed; ``x_fit``/``y_fit`` and ``x_val``/``y_val`` the fitting
    and held-out points and the targets the table judges with (the noiseless ones). A candidate is read by the engine,
    priced by :func:`srbf.pricing.price_bits` and evaluated with the engine's realizations; its FVU is
    :func:`srbf.metrics.numeric.fvu` on every point, so a non-finite value anywhere gives ``inf``. The held-out FVU is
    NaN for a problem without held-out points.

    One candidate gets ``timeout`` seconds (None: no limit) for all of it: a candidate the judge has not finished by
    then is unevaluable, with the reason, its FVUs NaN, its price kept if it was priced, and the judge goes on with the
    next one. The limit is the main thread's alarm (:func:`_time_limit`); a judgement in another thread has none."""
    n = len(expressions)
    X_fit = np.asarray(x_fit, dtype=float)
    if X_fit.ndim == 1:
        X_fit = X_fit.reshape(-1, 1)
    X_val = np.asarray(x_val if x_val is not None else np.empty((0, X_fit.shape[1])), dtype=float)
    if X_val.ndim != 2:
        X_val = X_val.reshape(-1, X_fit.shape[1]) if X_val.size else np.empty((0, X_fit.shape[1]))
    Y_fit = np.asarray(y_fit, dtype=float).reshape(-1)
    Y_val = np.asarray(y_val if y_val is not None else [], dtype=float).reshape(-1)
    has_val = X_val.shape[0] > 0 and Y_val.size == X_val.shape[0]
    variables = [str(v) for v in variables]
    names = skeleton_variable_names(variables)
    price = pricer(engine)

    prefixes = [""] * n
    reasons = [""] * n
    bits = np.full(n, np.nan)
    fvu_fit = np.full(n, np.nan)
    fvu_val = np.full(n, np.nan)
    status = np.zeros(n, dtype=np.uint8)

    def judge(i: int, text: str) -> None:
        try:
            prefix = _read_candidate(engine, text, notation, variables)
        except Exception as exc:  # noqa: BLE001 - a candidate the engine cannot read is a judged outcome
            status[i], reasons[i] = STATUS_UNREADABLE, _first_line(exc)
            return
        prefixes[i] = " ".join(prefix)
        bits[i] = price(prefix)
        try:
            y_hat_fit, y_hat_val = evaluate_prefix(engine, prefix, names, X_fit, X_val)
            if y_hat_fit.shape[0] != X_fit.shape[0] or (has_val and y_hat_val.shape[0] != X_val.shape[0]):
                raise ValueError("the expression does not give one value per point")
        except Exception as exc:  # noqa: BLE001 - evaluation errors vary
            status[i], reasons[i] = STATUS_UNEVALUABLE, _first_line(exc)
            return
        fvu_fit[i] = fvu(Y_fit, y_hat_fit)
        if has_val:
            fvu_val[i] = fvu(Y_val, y_hat_val)
        if not np.isfinite(bits[i]):
            status[i], reasons[i] = STATUS_UNPRICED, "the engine gives no finite price"

    for i, text in enumerate(expressions):
        try:
            with _time_limit(timeout):
                judge(i, str(text))
        except _CandidateTimeout:
            fvu_fit[i] = fvu_val[i] = np.nan
            status[i], reasons[i] = STATUS_UNEVALUABLE, f"judging it took longer than {timeout:g} s"
    return Judgement(prefixes, reasons, bits, fvu_fit, fvu_val, status)


def candidate_judge_fingerprint() -> str:
    """Names the candidate judge: the source of every srbf module :mod:`srbf.candidates` reaches, the versions of the
    libraries they import, and :data:`JUDGE_VERSION`. Stored with every judged problem, so a reader can tell judgements
    of another judge apart (:attr:`CandidateSet.current`)."""
    import srbf
    from importlib.metadata import PackageNotFoundError, packages_distributions, version as dist_version

    root = Path(srbf.__file__).resolve().parent
    sources, libraries = judge_dependencies(root, entry=__name__)
    h = hashlib.sha256()
    for p in sources:
        h.update(str(p.relative_to(root)).encode())
        h.update(p.read_bytes())
    distributions = packages_distributions()
    for name in libraries:
        try:
            v = dist_version((distributions.get(name) or [name])[0])
        except PackageNotFoundError:
            v = "?"
        h.update(f"{name}={v}".encode())
    h.update(f"candidates={JUDGE_VERSION}".encode())
    return h.hexdigest()[:16]


_FINGERPRINT: str | None = None


def _fingerprint() -> str:
    global _FINGERPRINT
    if _FINGERPRINT is None:
        _FINGERPRINT = candidate_judge_fingerprint()
    return _FINGERPRINT


def judge_identity(engine_name: str | None = None) -> dict[str, Any]:
    """What a store records about the judge that judged it."""
    identity: dict[str, Any] = {"fingerprint": _fingerprint(), "engine": engine_name, "price": "srbf.pricing.price_bits",
                                "fvu": "srbf.metrics.numeric.fvu"}
    for package in ("srbf", "simplipy"):
        try:
            identity[package] = importlib.import_module(package).__version__
        except Exception:  # noqa: BLE001 - a version is advisory
            identity[package] = None
    return identity


# ---- storage -------------------------------------------------------------------------------------------------------
# One append-only log per result file (``choices_000064.log`` in ``choices_000064.candidates/``; a shard writes its own,
# ``choices_000064.shard-0-of-4.log``), one record per problem: a fixed head (magic, the lengths of the header and the
# body, a CRC-32 of both), a JSON header (the row, the time, the method's pick, the notation, the variables, the judge's
# fingerprint, the columns' dtypes and shapes) and the columns' bytes, zlib-compressed. A problem fitted again later
# (a resumed run) or judged again appends or rewrites a newer record, and readers take the newest record of every row.
# A record that a killed run left half-written fails its check and ends the log; the next writer of that log cuts it off.
LOG_MAGIC = b"SRBFCND1"
_HEAD = struct.Struct("<8sIQI")      # magic, header length, body length, CRC-32 of header and body
LOG_SUFFIX = ".log"


def _pack(strings: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Strings as one UTF-8 byte array and their offsets: string i is ``data[offsets[i]:offsets[i + 1]]``."""
    blobs = [s.encode("utf-8") for s in strings]
    offsets = np.zeros(len(blobs) + 1, dtype=np.int64)
    if blobs:
        np.cumsum([len(b) for b in blobs], out=offsets[1:])
    return np.frombuffer(b"".join(blobs), dtype=np.uint8).copy(), offsets


def _unpack(data: np.ndarray, offsets: np.ndarray) -> list[str]:
    raw = np.asarray(data, dtype=np.uint8).tobytes()
    return [raw[offsets[i]:offsets[i + 1]].decode("utf-8") for i in range(len(offsets) - 1)]


def candidate_store_path(path: str | os.PathLike[str]) -> Path:
    """The candidate store of a result file: ``choices_000064.pkl`` and each of its shards
    (``choices_000064.shard-0-of-4.pkl``) share ``choices_000064.candidates``. A directory is returned as it is."""
    p = Path(path)
    if p.name.endswith(STORE_SUFFIX) or p.is_dir():
        return p
    stem = re.sub(r"\.shard-\d+-of-\d+$", "", p.stem if p.suffix else p.name)
    return p.with_name(stem + STORE_SUFFIX)


def _encode(columns: Mapping[str, np.ndarray], header: Mapping[str, Any]) -> bytes:
    arrays = {name: np.ascontiguousarray(array) for name, array in columns.items()}
    head = dict(header, columns=[[name, a.dtype.str, list(a.shape)] for name, a in arrays.items()])
    head_bytes = json.dumps(head, default=str, allow_nan=True).encode("utf-8")
    body = zlib.compress(b"".join(a.tobytes() for a in arrays.values()), 6)
    crc = zlib.crc32(body, zlib.crc32(head_bytes))
    return _HEAD.pack(LOG_MAGIC, len(head_bytes), len(body), crc) + head_bytes + body


def _decode(head: Mapping[str, Any], body: bytes) -> dict[str, np.ndarray]:
    raw = zlib.decompress(body)
    columns: dict[str, np.ndarray] = {}
    offset = 0
    for name, dtype, shape in head["columns"]:
        dt = np.dtype(dtype)
        count = int(np.prod(shape)) if shape else 1
        columns[name] = np.frombuffer(raw, dtype=dt, count=count, offset=offset).reshape(shape).copy()
        offset += count * dt.itemsize
    return columns


def _scan(path: Path) -> Iterator[tuple[int, dict[str, Any], bytes]]:
    """``(end offset, header, body)`` of every intact record of a log, in order; a record that fails its check (a
    write a killed run left unfinished) ends the log."""
    with open(path, "rb") as fh:
        while True:
            head = fh.read(_HEAD.size)
            if len(head) < _HEAD.size:
                return
            magic, head_len, body_len, crc = _HEAD.unpack(head)
            if magic != LOG_MAGIC:
                return
            head_bytes = fh.read(head_len)
            body = fh.read(body_len)
            if len(head_bytes) < head_len or len(body) < body_len or zlib.crc32(body, zlib.crc32(head_bytes)) != crc:
                return
            yield fh.tell(), json.loads(head_bytes), body


def _write_json(target: Path, payload: Mapping[str, Any]) -> None:
    """A small JSON file through a temp name of this process's own; a failure is a warning (it is advisory)."""
    tmp = target.with_name(f"{target.name}.{os.getpid()}-{time.monotonic_ns()}.tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=1, default=str))
        tmp.replace(target)
    except OSError as exc:
        warnings.warn(f"candidate store: {target.name} not written ({exc}); readers scan the logs", RuntimeWarning)
        tmp.unlink(missing_ok=True)


class CandidateLog:
    """The append-only log of one writer (a result file, or one shard of it) in a run's candidate store, with an
    advisory ``<tag>.json`` beside it: the judge, the run's meta and what the log holds."""

    def __init__(self, store: str | Path, tag: str, *, judge: Mapping[str, Any] | None = None,
                 run_meta: Mapping[str, Any] | None = None) -> None:
        self.store = Path(store)
        self.store.mkdir(parents=True, exist_ok=True)
        self.path = self.store / f"{tag}{LOG_SUFFIX}"
        self.judge = dict(judge) if judge else None
        self.run_meta = dict(run_meta) if run_meta else None
        self.rows: set[int] = set()
        end = 0
        if self.path.exists():
            for end, head, _body in _scan(self.path):
                self.rows.add(int(head["row"]))
            size = self.path.stat().st_size
            if size > end:      # the tail a killed writer left half-written
                warnings.warn(f"{self.path}: cutting off {size - end} bytes of an unfinished record", RuntimeWarning)
                with open(self.path, "r+b") as fh:
                    fh.truncate(end)
        self._fh: Any = open(self.path, "ab")
        self._write_sidecar()

    def append(self, row: int, *, expressions: Sequence[str], judgement: Judgement, notation: str,
               variables: Sequence[str], prediction: int | None = None, meta: Mapping[str, Any] | None = None) -> int:
        """Append one problem's candidates and their judgement; returns the bytes written."""
        record = _record(row, expressions=expressions, judgement=judgement, notation=notation, variables=variables,
                         prediction=prediction, meta=meta, judge=(self.judge or {}).get("fingerprint"))
        self._fh.write(record)
        self._fh.flush()
        self.rows.add(int(row))
        return len(record)

    def update_run_meta(self, run_meta: Mapping[str, Any]) -> None:
        """Add to the run's meta in ``<tag>.json`` what is known only once the method is up (its ranking, say)."""
        self.run_meta = {**(self.run_meta or {}), **run_meta}
        self._write_sidecar()

    def _write_sidecar(self) -> None:
        _write_json(self.path.with_suffix(".json"), {
            "kind": "candidates", "format": STORE_FORMAT, "judge": self.judge, "run_meta": self.run_meta,
            "log": self.path.name, "problems": len(self.rows),
            "bytes": self.path.stat().st_size if self.path.exists() else 0})

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            self._write_sidecar()


def _record(row: int, *, expressions: Sequence[str], judgement: Judgement, notation: str, variables: Sequence[str],
            prediction: int | None, meta: Mapping[str, Any] | None, judge: str | None) -> bytes:
    n = len(expressions)
    if len(judgement) != n:
        raise ValueError(f"{len(judgement)} judgements for {n} candidates")
    if prediction is not None and not 0 <= int(prediction) < n:
        raise ValueError(f"prediction index {prediction} outside the {n} candidates")
    if notation not in NOTATIONS:
        raise ValueError(f"unknown notation {notation!r}; one of {', '.join(NOTATIONS)}")
    text, text_offsets = _pack([str(e) for e in expressions])
    prefix, prefix_offsets = _pack(judgement.prefixes)
    reason, reason_offsets = _pack(judgement.reasons)
    columns = {"text": text, "text_offsets": text_offsets, "prefix": prefix, "prefix_offsets": prefix_offsets,
               "reason": reason, "reason_offsets": reason_offsets,
               # float64: a float32 column flushes an FVU of ~1e-50 to 0.0, which reads as a perfect fit
               "bits": np.asarray(judgement.bits, dtype=np.float64),
               "fvu_fit": np.asarray(judgement.fvu_fit, dtype=np.float64),
               "fvu_val": np.asarray(judgement.fvu_val, dtype=np.float64),
               "status": np.asarray(judgement.status, dtype=np.uint8)}
    header = {"row": int(row), "time": time.time(), "n": n, "prediction": None if prediction is None else int(prediction),
              "notation": notation, "variables": [str(v) for v in variables], "judge": judge, "meta": dict(meta or {})}
    return _encode(columns, header)


# ---- the run's side ------------------------------------------------------------------------------------------------
class CandidateRecorder:
    """Takes each record's candidates out of it, judges them on the problem's points and appends them to the run's
    candidate log.

    Called by the benchmark driver after the method has answered, so nothing here is part of ``fit_time``. The record
    keeps ``n_candidates`` and ``candidates_judge_time`` (seconds), and ``candidates_dropped`` when the adapter could
    not keep every candidate. ``tag`` names the log: the result file's stem (a shard's own)."""

    def __init__(self, store: str | Path, *, engine: Any, tag: str = "run", engine_name: str | None = None,
                 run_meta: Mapping[str, Any] | None = None) -> None:
        self.engine = engine
        self.engine_name = engine_name
        self.log = CandidateLog(store, tag, judge=judge_identity(engine_name), run_meta=run_meta)

    def record(self, sample: Any, record: dict[str, Any]) -> int:
        """Judge and store the candidates of one problem's ``record``; returns how many were stored."""
        fields = {key: record.pop(key) for key in CANDIDATE_FIELDS if key in record}
        variables = [str(v) for v in (record.get("variables") or record.get("variable_names") or [])]
        expressions = fields.get("candidate_expressions")
        prediction: int | None
        if expressions is None:
            # an adapter that hands no candidates over: its prediction is its only one
            prefix = record.get("predicted_expression_prefix")
            if not record.get("prediction_success") or prefix is None:
                return 0
            expressions, notation, prediction = [" ".join(map(str, prefix))], "prefix", 0
        else:
            expressions = [str(e) for e in expressions]
            notation = str(fields.get("candidate_notation") or "infix")
            pick = fields.get("candidate_prediction")
            prediction = None if pick is None else int(pick)
            variables = [str(v) for v in (fields.get("candidate_variables") or variables)]
        if fields.get("candidates_dropped"):
            record["candidates_dropped"] = int(fields["candidates_dropped"])
        if not expressions:
            return 0
        row = record.get("eval_row_index")
        if row is None:
            warnings.warn("a record without an eval_row_index: its candidates are not stored", RuntimeWarning)
            return 0
        x_fit = np.asarray(sample.x_support, dtype=float)
        x_val = np.asarray(sample.x_validation, dtype=float)
        if len(variables) != x_fit.shape[1]:
            variables = [f"x{i + 1}" for i in range(x_fit.shape[1])]
        started = time.perf_counter()
        judgement = judge_candidates(expressions, engine=self.engine, variables=variables, x_fit=x_fit,
                                     y_fit=sample.y_support, x_val=x_val, y_val=sample.y_validation, notation=notation)
        seconds = time.perf_counter() - started
        meta = {"benchmark_eq_id": record.get("benchmark_eq_id"), "n_fit": int(x_fit.shape[0]),
                "n_val": int(x_val.shape[0]) if x_val.ndim == 2 else 0, "judge_time": seconds,
                # whether the record has an answer: a method can end with candidates and none (no selection)
                "prediction_success": bool(record.get("prediction_success")),
                # the frozen draws the problem's points were read from, when they were (pairs runs on the same data)
                **{key: record[key] for key in ("draws_sha256", "draws_repeat") if record.get(key) is not None}}
        self.log.append(int(row), expressions=expressions, judgement=judgement, notation=notation, variables=variables,
                        prediction=prediction, meta=meta)
        record["n_candidates"] = len(expressions)
        record["candidates_judge_time"] = seconds
        return len(expressions)

    def update_run_meta(self, run_meta: Mapping[str, Any]) -> None:
        """See :meth:`CandidateLog.update_run_meta`."""
        self.log.update_run_meta(run_meta)

    def close(self) -> None:
        self.log.close()


# ---- reading -------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class CandidateSet:
    """The judged candidates of one problem: arrays aligned by candidate, the strings on demand."""

    row: int
    bits: np.ndarray             # float64, the price in bits
    fvu_fit: np.ndarray          # float64, FVU on the fitting points
    fvu_val: np.ndarray          # float64, FVU on the held-out points (NaN without them)
    status: np.ndarray           # uint8, STATUS_*
    prediction: np.ndarray       # bool, True for the method's own pick (at most one)
    header: dict[str, Any]
    columns: Mapping[str, np.ndarray] = field(repr=False)

    def __len__(self) -> int:
        return int(self.status.shape[0])

    @property
    def valid(self) -> np.ndarray:
        """Candidates the judge read, evaluated and priced (``status == STATUS_JUDGED``)."""
        return self.status == STATUS_JUDGED

    @property
    def predicted(self) -> int | None:
        """The index of the method's own pick, or None (a method that picked none, or failed)."""
        return self.header.get("prediction")

    @property
    def notation(self) -> str:
        return str(self.header.get("notation", "infix"))

    @property
    def variables(self) -> list[str]:
        """The column names the method was handed."""
        return list(self.header.get("variables") or [])

    @property
    def meta(self) -> dict[str, Any]:
        """What the run recorded with the problem: ``benchmark_eq_id``, ``n_fit``, ``n_val``, ``judge_time``,
        ``prediction_success`` (whether the record has an answer), and ``draws_sha256``/``draws_repeat`` for a problem
        read from frozen draws."""
        return dict(self.header.get("meta") or {})

    @property
    def expressions(self) -> list[str]:
        """Each candidate as the method wrote it."""
        return _unpack(self.columns["text"], self.columns["text_offsets"])

    @property
    def prefixes(self) -> list[list[str]]:
        """Each candidate as the judge read it (prefix tokens, ``x1, x2, ...``); empty when it could not be read."""
        return [p.split() for p in _unpack(self.columns["prefix"], self.columns["prefix_offsets"])]

    @property
    def reasons(self) -> list[str]:
        """Why a candidate is not judged; empty for the judged ones."""
        return _unpack(self.columns["reason"], self.columns["reason_offsets"])

    @property
    def current(self) -> bool:
        """Whether this srbf's candidate judge judged these candidates (see :func:`rejudge_candidates`)."""
        return self.header.get("judge") == _fingerprint()


def _candidate_set(head: dict[str, Any], body: bytes) -> CandidateSet:
    cols = _decode(head, body)
    flags = np.zeros(int(head["n"]), dtype=bool)
    if head.get("prediction") is not None:
        flags[int(head["prediction"])] = True
    return CandidateSet(row=int(head["row"]), bits=cols["bits"], fvu_fit=cols["fvu_fit"], fvu_val=cols["fvu_val"],
                        status=cols["status"], prediction=flags, header=head, columns=cols)


def _logs(path: str | os.PathLike[str]) -> list[Path]:
    return sorted(candidate_store_path(path).glob(f"*{LOG_SUFFIX}"))


def _newest(path: str | os.PathLike[str]) -> dict[int, tuple[float, Path, int]]:
    """For every row in the store: (time, log, end offset) of its newest record."""
    newest: dict[int, tuple[float, Path, int]] = {}
    for log in _logs(path):
        for end, head, _body in _scan(log):
            row, when = int(head["row"]), float(head.get("time") or 0.0)
            if row not in newest or when >= newest[row][0]:
                newest[row] = (when, log, end)
    return newest


def iter_candidates(path: str | os.PathLike[str], rows: Sequence[int] | None = None) -> Iterator[CandidateSet]:
    """The judged candidates of a run, one problem at a time in the order they were stored: ``path`` is a result file
    (whole or a shard: the run's store either way) or its store directory; ``rows`` keeps only those problems. Of a
    problem stored more than once (a resumed run) the newest record counts."""
    wanted = None if rows is None else {int(r) for r in rows}
    newest = _newest(path)
    for log in _logs(path):
        for end, head, body in _scan(log):
            row = int(head["row"])
            if (wanted is None or row in wanted) and newest[row][1:] == (log, end):
                yield _candidate_set(head, body)


def load_candidates(path: str | os.PathLike[str], rows: Sequence[int] | None = None) -> dict[int, CandidateSet]:
    """:func:`iter_candidates` as a mapping from the problem's row (``eval_row_index``) to its candidates, in row
    order."""
    return dict(sorted((c.row, c) for c in iter_candidates(path, rows)))


def read_manifests(path: str | os.PathLike[str]) -> dict[str, dict[str, Any]]:
    """The advisory ``<tag>.json`` of every log in the store (the judge, the run's meta, what the log holds), by tag."""
    out: dict[str, dict[str, Any]] = {}
    for sidecar in sorted(candidate_store_path(path).glob("*.json")):
        try:
            out[sidecar.stem] = json.loads(sidecar.read_text())
        except (OSError, ValueError):
            continue
    return out


# ---- judging a stored run again ------------------------------------------------------------------------------------
def rejudge_candidates(result_file: str | os.PathLike[str], *, engine: Any, engine_name: str | None = None) -> int:
    """Judge the stored candidates of the problems in ``result_file`` again with this srbf's judge and ``engine``, from
    the stored strings and the points the result file keeps; returns how many problems were judged. Every log of the
    store that holds one of these problems is rewritten (through a temp file) with the newest record of each row,
    those of the result file's problems judged again. A shard's file judges that shard's problems."""
    with open(result_file, "rb") as fh:
        snapshot = pickle.load(fh)
    points: dict[int, int] = {}
    for i, row in enumerate(snapshot.get("eval_row_index") or []):
        if row is not None:
            points[int(row)] = i

    def column(key: str, i: int) -> Any:
        values = snapshot.get(key)
        return values[i] if values is not None and i < len(values) else None

    identity = judge_identity(engine_name)
    newest = _newest(result_file)
    done = 0
    for log in _logs(result_file):
        keep = {end: row for row, (_when, where, end) in newest.items() if where == log}
        if not any(row in points for row in keep.values()):
            continue
        tmp = log.with_name(f"{log.name}.{os.getpid()}.tmp")
        with open(tmp, "wb") as out:
            for end, head, body in _scan(log):
                row = keep.get(end)
                if row is None:
                    continue
                if row not in points:
                    out.write(_encode(_decode(head, body), {k: v for k, v in head.items() if k != "columns"}))
                    continue
                stored = _candidate_set(head, body)
                i = points[row]
                started = time.perf_counter()
                judgement = judge_candidates(stored.expressions, engine=engine, variables=stored.variables,
                                             x_fit=np.asarray(column("x", i), dtype=float), y_fit=column("y", i),
                                             x_val=column("x_val", i), y_val=column("y_val", i), notation=stored.notation)
                meta = dict(stored.meta, judge_time=time.perf_counter() - started)
                out.write(_record(row, expressions=stored.expressions, judgement=judgement, notation=stored.notation,
                                  variables=stored.variables, prediction=stored.predicted, meta=meta,
                                  judge=identity["fingerprint"]))
                done += 1
        tmp.replace(log)
        sidecar = log.with_suffix(".json")
        info = json.loads(sidecar.read_text()) if sidecar.exists() else {"kind": "candidates", "format": STORE_FORMAT}
        _write_json(sidecar, dict(info, judge=identity, log=log.name, problems=len(keep), bytes=log.stat().st_size))
    return done


__all__ = [
    "CANDIDATE_FIELDS", "JUDGE_TIMEOUT_S", "MAX_CANDIDATES", "MAX_CANDIDATE_CHARS", "NOTATIONS", "STATUS_JUDGED", "STATUS_NAMES",
    "STATUS_UNEVALUABLE", "STATUS_UNPRICED", "STATUS_UNREADABLE", "CandidateLog", "CandidateRecorder", "CandidateSet",
    "Judgement", "candidate_judge_fingerprint", "candidate_store_path", "evaluate_prefix", "iter_candidates",
    "judge_candidates", "judge_identity", "limit_candidates", "load_candidates", "read_expression", "read_manifests",
    "read_spelled", "rejudge_candidates", "spell_expression", "unknown_symbols",
]
