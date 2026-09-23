"""One table for a whole evaluation: a row per problem and rung, every per-problem metric.

A result tree is the directory an evaluation writes for one method and one draw: ``<tree>/<catalog>/choices_<rung>.pkl``
(``niter_<rung>.pkl`` for a method whose ladder counts iterations), each file possibly split into shards
``<stem>.shard-K-of-N.pkl``. :func:`judge_result_file` turns one file into rows with :func:`srbf.derive_metrics`;
:func:`build_table` judges any number of trees in parallel and writes the rows as CSV (``srbf table``). The results
site is built from this table.

Row semantics:

* rate columns (``success``, the numeric and symbolic recoveries, ``skeleton_match_raw``) are defined for every
  problem: a failed prediction is a miss (0), not a missing value;
* analysis columns (FVU, R², distances, lengths, MDL, ...) describe the predictions that were made and are empty for a
  failed one, except the metrics whose range has a worst value (:data:`srbf.result_processing.WORST_VALUE`), which a
  failed prediction takes;
* ground-truth columns (``skeleton_length``, ``ground_truth_mdl``, ...) are defined for every problem.

Description lengths are in bits (certified, f64 parse, default canon). A per-file cache (``cache_dir``) keeps the rows
of every file judged before; its key names the file (path, size, modification time) and the judge (a hash of the
installed srbf source, the simplipy, sympy and numpy versions and the engine), so a file is judged again exactly when it
or the judge changed.
"""
from __future__ import annotations

import csv
import glob
import hashlib
import math
import os
import pickle
import re
import time
import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
import multiprocessing
from multiprocessing import TimeoutError as PoolTimeout
from pathlib import Path
from typing import Any

#: A result file's name: the rung and, for a shard, its index and count.
RESULT_FILE = re.compile(r"(?:choices|niter)_(\d+)(?:\.shard-(\d+)-of-(\d+))?\.pkl$")

ID_COLUMNS = ["model", "draw", "catalog", "rung", "shard", "row", "sha"]
RATE_COLUMNS = ["success", "numeric_recovery_val", "numeric_recovery_fit", "numeric_recovery_relative_val",
                "numeric_recovery_relative_fit", "symbolic_recovery", "symbolic_recovery_mask_fittable",
                "symbolic_recovery_mask_none", "skeleton_match_raw"]
ANALYSIS_COLUMNS = ["log10_fvu_val", "log10_fvu_fit", "r2_val", "r2_fit", "mdl_ratio", "predicted_mdl", "f1_score",
                    "precision_score", "recall_score", "edit_distance", "edit_distance_norm", "zss_edit_distance",
                    "expr_length_ratio", "expr_length_ratio_abserr", "predicted_skeleton_prefix_length",
                    "predicted_n_constants", "n_constants_ratio", "n_constants_delta", "predicted_total_nestedness",
                    "total_nestedness_delta", "f1_score_unique_variables", "precision_unique_variables",
                    "recall_unique_variables", "predicted_log_prob", "predicted_score", "predicted_pareto_rank",
                    "fit_time", "generation_time"]
GROUND_TRUTH_COLUMNS = ["skeleton_length", "ground_truth_mdl", "n_constants", "total_nestedness", "n_variables", "n_support"]
#: The table's columns, in order.
COLUMNS = ID_COLUMNS + RATE_COLUMNS + ANALYSIS_COLUMNS + GROUND_TRUTH_COLUMNS
#: Bumped whenever a row's content changes without a change to srbf's source (it is part of the cache key).
TABLE_VERSION = 1

_RECOVERIES = ["numeric_recovery_val", "numeric_recovery_fit", "numeric_recovery_relative_val", "numeric_recovery_relative_fit",
               "symbolic_recovery", "symbolic_recovery_mask_fittable", "symbolic_recovery_mask_none"]
_DERIVED_WHEN_ANSWERED = ["log10_fvu_val", "log10_fvu_fit", "r2_val", "r2_fit", "mdl_ratio", "predicted_mdl", "f1_score",
                          "edit_distance", "zss_edit_distance", "predicted_skeleton_prefix_length", "predicted_n_constants",
                          "n_constants_delta", "predicted_total_nestedness", "f1_score_unique_variables",
                          "precision_unique_variables", "recall_unique_variables"]
_RAW_WHEN_ANSWERED = ["predicted_log_prob", "predicted_score", "predicted_pareto_rank"]


@dataclass(frozen=True)
class ResultTree:
    """The result files of one method and one draw: ``<path>/<catalog>/choices_<rung>.pkl``.

    ``first_index`` is for result files that spell the variables by column index, ``x_<i>`` counted from
    ``first_index`` (E2E counts from 0, NeSymReS from 1). srbf renames them at the source, so only files written by
    older versions need it; ``None`` leaves such tokens alone.
    """

    method: str
    draw: int
    path: str
    first_index: int | None = None

    @classmethod
    def parse(cls, spec: str, first_index: int | None = None) -> "ResultTree":
        """``METHOD:DRAW:PATH``, as ``srbf table --tree`` takes it."""
        try:
            method, draw, path = spec.split(":", 2)
            return cls(method, int(draw), path, first_index)
        except ValueError:
            raise ValueError(f"a result tree is METHOD:DRAW:PATH, got {spec!r}") from None


def result_files(tree: ResultTree, *, max_rung: int | None = None) -> list[str]:
    """The result files of a tree (every catalog, every rung, whole files and shards)."""
    files = glob.glob(os.path.join(tree.path, "*", "choices_*.pkl")) + glob.glob(os.path.join(tree.path, "*", "niter_*.pkl"))
    files = [f for f in files if RESULT_FILE.search(os.path.basename(f))]
    if max_rung:
        files = [f for f in files if int(RESULT_FILE.search(os.path.basename(f)).group(1)) <= max_rung]  # type: ignore[union-attr]
    return sorted(files)


def mdl_bits(engine: Any) -> Callable[[list[str]], float]:
    """The description length the table reports, in bits: certified, f64 parse, the default canon (NaN on failure)."""
    from simplipy.engine import Mode

    def price(prefix: list[str]) -> float:
        try:
            return float(engine.complexity(list(prefix), certified=True, mode=Mode.f64, canon="default")) / 1000.0
        except Exception:  # noqa: BLE001 - an expression the engine cannot price has no description length
            return float("nan")

    return price


def _number(x: Any) -> float | str:
    """A float, or '' for an undefined value; non-finite floats are kept (the table's readers decide)."""
    if x is None:
        return ""
    try:
        return float(x)
    except (TypeError, ValueError):
        return ""


def _mean_time(t: Any) -> float | str:
    if t is None:
        return ""
    try:
        import numpy as np
        v = float(np.atleast_1d(t).mean())
    except Exception:  # noqa: BLE001 - a malformed timing entry is no time
        return ""
    return v if math.isfinite(v) else ""


def _model_sha(snapshot: dict[str, Any]) -> str:
    """The first 8 hex digits of the model weights' sha256, when the run recorded one."""
    try:
        for name, entry in (snapshot.get("__meta__", {}).get("inputs") or {}).items():
            if name.endswith("model.safetensors"):
                return str(entry.get("sha256", ""))[:8]
    except Exception:  # noqa: BLE001 - provenance is optional
        pass
    return ""


def _rename_predictions(snapshot: dict[str, Any], first_index: int | None) -> None:
    """Bring the stored predictions into the ground truth's variable spelling, in place."""
    from srbf.variable_renaming import rename_prediction

    columns = snapshot.get("variables")
    if columns is None:
        columns = snapshot.get("variable_names") or []
    for key in ("predicted_expression_prefix", "predicted_skeleton_prefix"):
        seq = snapshot.get(key)
        if seq is None:
            continue
        out = list(seq)
        for i, prefix in enumerate(out):
            if prefix is not None:
                handed = [str(c) for c in (list(columns[i]) if i < len(columns) else [])]
                out[i] = rename_prediction(prefix, handed, first_index=first_index)
        snapshot[key] = out


def judge_result_file(path: str, *, method: str, draw: int = 1, engine: Any, first_index: int | None = None) -> list[list[Any]]:
    """The table rows (in :data:`COLUMNS` order) of one result file; an empty list for a file without problems.

    ``engine`` is a loaded ``SimpliPyEngine``. Raises when the file cannot be read or judged.
    """
    import numpy as np  # noqa: F401 - the snapshots hold numpy arrays

    from srbf.metrics.symbolic import total_nestedness
    from srbf.result_processing import WORST_VALUE, derive_metrics

    with open(path, "rb") as fh:
        snap = pickle.load(fh)
    n = len(snap.get("eval_row_index", []))
    if n == 0:
        return []
    m = RESULT_FILE.search(os.path.basename(path))
    if m is None:
        raise ValueError(f"{path}: not a result file name")
    rung = int(m.group(1))
    shard = f"{m.group(2)}/{m.group(3)}" if m.group(2) else ""
    catalog = os.path.basename(os.path.dirname(path))
    sha = _model_sha(snap)
    _rename_predictions(snap, first_index)
    arity = engine.operator_arity
    s = derive_metrics(snap, engine=engine, mdl_fn=mdl_bits(engine))

    def got(k: str, i: int) -> Any:
        return s[k][i] if k in s and i < len(s[k]) else None

    def raw(k: str, i: int) -> Any:
        return snap[k][i] if k in snap and i < len(snap[k]) else None

    rows: list[list[Any]] = []
    for i in range(n):
        ok = bool(raw("prediction_success", i))
        r: dict[str, Any] = {"model": method, "draw": draw, "catalog": catalog, "rung": rung, "shard": shard,
                             "row": int(snap["eval_row_index"][i]), "sha": sha}
        sk = raw("skeleton", i)
        sk = list(sk) if sk is not None else None
        sks = got("skeleton_simplified", i)
        sks = list(sks) if sks is not None else sk
        ps = raw("predicted_skeleton_prefix", i)
        ps = list(ps) if (ok and ps is not None) else None
        # rates: a miss unless the prediction succeeded and the judge said yes
        r["success"] = int(ok)
        for k in _RECOVERIES:
            v = got(k, i)
            r[k] = int(bool(v)) if (ok and v is not None) else 0
        r["skeleton_match_raw"] = int(ok and ps is not None and sk is not None and ps == sk)
        # ground truth
        r["skeleton_length"] = _number(got("skeleton_length", i)) if got("skeleton_length", i) is not None else (len(sks) if sks else "")
        r["ground_truth_mdl"] = _number(got("ground_truth_mdl", i))
        r["n_constants"] = _number(got("n_constants", i))
        r["total_nestedness"] = _number(got("total_nestedness", i))
        r["n_variables"] = _number(got("n_variables", i))
        r["n_support"] = _number(raw("n_support", i))
        r["fit_time"] = _mean_time(raw("fit_time", i))
        r["generation_time"] = _mean_time(raw("generation_time", i))
        for k in ANALYSIS_COLUMNS:
            r.setdefault(k, "")
        if ok:
            for k in _DERIVED_WHEN_ANSWERED:
                r[k] = _number(got(k, i))
            r["expr_length_ratio"] = _number(got("skeleton_length_ratio", i))
            for k in _RAW_WHEN_ANSWERED:
                r[k] = _number(raw(k, i))
            for k in ["precision_score", "recall_score", "edit_distance_norm"]:
                r[k] = _number(got(k, i))
            if ps is not None and sks:
                ratio = r["expr_length_ratio"]
                r["expr_length_ratio_abserr"] = abs(math.log2(ratio)) if isinstance(ratio, float) and ratio > 0 and math.isfinite(ratio) else ""
                nc_gt, nc_pred = sks.count("<constant>"), ps.count("<constant>")
                r["n_constants_ratio"] = nc_pred / nc_gt if nc_gt else ""
                pn, gn = got("predicted_total_nestedness", i), got("total_nestedness", i)
                if pn is None or gn is None:
                    try:
                        pn, gn = total_nestedness(ps, arity), total_nestedness(sks, arity)
                    except Exception:  # noqa: BLE001 - an expression without a tree has no nestedness
                        pn = gn = None
                r["total_nestedness_delta"] = _number(float(pn) - float(gn)) if pn is not None and gn is not None else ""
        else:
            # a failed problem sits at the bad end of every metric whose range has one
            for k in WORST_VALUE:
                r[k] = _number(got(k, i))
        rows.append([r.get(c, "") for c in COLUMNS])
    return rows


# ---- the judge's fingerprint and the per-file cache ---------------------------------------------------------------
def judge_fingerprint(engine_name: str) -> str:
    """Names the judge: the installed srbf source, the simplipy / sympy / numpy versions, the engine, the table version."""
    import srbf

    h = hashlib.sha256()
    root = Path(srbf.__file__).resolve().parent
    for p in sorted(root.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        h.update(str(p.relative_to(root)).encode())
        h.update(p.read_bytes())
    for name in ("simplipy", "sympy", "numpy"):
        try:
            version = __import__(name).__version__
        except Exception:  # noqa: BLE001 - a missing library is part of the judge too
            version = "?"
        h.update(f"{name}={version}".encode())
    h.update(f"engine={engine_name}|table={TABLE_VERSION}".encode())
    return h.hexdigest()[:16]


def _cache_key(tree: ResultTree, path: str, fingerprint: str) -> str:
    st = os.stat(path)
    ident = f"{tree.method}|{tree.draw}|{tree.first_index}|{os.path.abspath(path)}|{st.st_size}|{st.st_mtime_ns}|{fingerprint}"
    return hashlib.sha1(ident.encode()).hexdigest()


def _cache_load(cache_dir: str, key: str) -> list[list[Any]] | None:
    p = os.path.join(cache_dir, key[:2], key + ".pkl")
    if not os.path.exists(p):
        return None
    try:
        with open(p, "rb") as fh:
            rows = pickle.load(fh)
        return rows if isinstance(rows, list) else None
    except Exception:  # noqa: BLE001 - a damaged entry is a miss
        return None


def _cache_store(cache_dir: str, key: str, rows: list[list[Any]]) -> None:
    d = os.path.join(cache_dir, key[:2])
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, f"{key}.{os.getpid()}.tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(rows, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, os.path.join(d, key + ".pkl"))


# ---- parallel judging -------------------------------------------------------------------------------------------
_ENGINE: Any = None
_ENGINE_NAME: str | None = None


def _worker_engine(name: str) -> Any:
    global _ENGINE, _ENGINE_NAME
    if _ENGINE is None or _ENGINE_NAME != name:
        from simplipy import SimpliPyEngine
        _ENGINE, _ENGINE_NAME = SimpliPyEngine.load(name, install=True), name
    return _ENGINE


def _mark(progress_path: str | None, tag: str, path: str) -> None:
    if not progress_path:
        return
    try:
        with open(progress_path, "a") as fh:
            fh.write(f"{tag} {path}\n")
    except OSError:
        pass


def _judge_job(job: tuple[ResultTree, str, str, str | None]) -> tuple[str, list[list[Any]], str | None]:
    """(path, rows, error). The file is marked in flight while it is judged, so a stall can name its suspects."""
    tree, path, engine_name, progress_path = job
    _mark(progress_path, "S", path)
    try:
        warnings.simplefilter("ignore")
        try:
            rows = judge_result_file(path, method=tree.method, draw=tree.draw, engine=_worker_engine(engine_name),
                                     first_index=tree.first_index)
        except Exception as e:  # noqa: BLE001 - a file that cannot be judged is reported, not fatal
            return path, [], f"{path}: {type(e).__name__}: {e}"
        return path, rows, None
    finally:
        _mark(progress_path, "D", path)


def _in_flight(progress_path: str | None) -> list[str] | str:
    if not progress_path or not os.path.exists(progress_path):
        return "unknown (no progress file)"
    started: list[str] = []
    done: set[str] = set()
    for line in open(progress_path):
        tag, _, p = line.partition(" ")
        if tag == "S":
            started.append(p.strip())
        elif tag == "D":
            done.add(p.strip())
    return [p for p in started if p not in done] or "none (the worker pool itself is wedged)"


@dataclass
class TableReport:
    """What :func:`build_table` did."""

    files: int
    judged: int
    from_cache: int
    rows: int
    errors: list[str]
    stalled: bool
    seconds: float


def build_table(trees: Iterable[ResultTree], out: str, *, engine: str = "acj-5-4-llm", workers: int = 4,
                cache_dir: str | None = None, max_rung: int | None = None, stall_timeout: float = 1800.0,
                progress_path: str | None = None, log: Callable[[str], None] | None = print) -> TableReport:
    """Judge every result file of ``trees`` and write the table to ``out`` (CSV, written atomically).

    Files are judged by ``workers`` processes, largest first; the rows are written in that order, whether they were
    judged now or read from ``cache_dir``. A file that cannot be read or judged is reported in ``errors`` and left out
    (a result file that is still being written is the usual cause). When no file finishes for ``stall_timeout``
    seconds the pool is stopped and the table is written with what was judged (``stalled``); ``progress_path``
    then names the files that were in flight.
    """
    say = log or (lambda _msg: None)
    t0 = time.time()
    jobs: list[tuple[ResultTree, str]] = []
    for tree in trees:
        jobs += [(tree, f) for f in result_files(tree, max_rung=max_rung)]
    jobs.sort(key=lambda j: (-os.path.getsize(j[1]), j[1], j[0].method, j[0].draw))
    cached: dict[int, list[list[Any]]] = {}
    keys: dict[int, str] = {}
    if cache_dir:
        fingerprint = judge_fingerprint(engine)
        for i, (tree, path) in enumerate(jobs):
            keys[i] = _cache_key(tree, path, fingerprint)
            hit = _cache_load(cache_dir, keys[i])
            if hit is not None:
                cached[i] = hit
    pending = [i for i in range(len(jobs)) if i not in cached]
    say(f"{len(jobs)} files: {len(pending)} to judge, {len(cached)} from the cache")
    if progress_path:
        open(progress_path, "w").close()
    errors: list[str] = []
    n_rows, judged, stalled = 0, 0, False
    tmp = out + ".tmp"
    with open(tmp, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(COLUMNS)
        # forkserver: a fork of a process that already runs threads (the engine's, numpy's) can deadlock the child
        methods = multiprocessing.get_all_start_methods()
        context = multiprocessing.get_context("forkserver" if "forkserver" in methods else "spawn")
        pool = context.Pool(max(1, workers), maxtasksperchild=25) if pending else None
        try:
            results = pool.imap(_judge_job, [(jobs[i][0], jobs[i][1], engine, progress_path) for i in pending], chunksize=1) if pool else iter(())
            for i in range(len(jobs)):
                if i in cached:
                    rows = cached[i]
                elif stalled:
                    continue
                else:
                    try:
                        _path, rows, err = results.next(timeout=stall_timeout)  # type: ignore[attr-defined]
                    except PoolTimeout:
                        say(f"STALLED after {judged}/{len(pending)} files: none finished for {stall_timeout:.0f}s; in flight: {_in_flight(progress_path)}")
                        stalled = True
                        continue
                    judged += 1
                    if err:
                        errors.append(err)
                    elif cache_dir:
                        _cache_store(cache_dir, keys[i], rows)
                    if judged % 50 == 0 or judged == len(pending):
                        say(f"{judged}/{len(pending)} files judged, {time.time() - t0:.0f}s")
                writer.writerows(rows)
                n_rows += len(rows)
        finally:
            if pool is not None:
                pool.terminate() if stalled else pool.close()
                pool.join()
    os.replace(tmp, out)
    report = TableReport(files=len(jobs), judged=judged, from_cache=len(cached), rows=n_rows, errors=errors,
                         stalled=stalled, seconds=time.time() - t0)
    for e in errors:
        say(f"ERR {e}")
    say(f"done: {n_rows} rows, {len(errors)} errors, {report.seconds:.0f}s -> {out}")
    return report


def parse_index_bases(specs: Sequence[str] | None) -> dict[str, int]:
    """``METHOD=FIRST`` pairs, as ``srbf table --index-variables`` takes them."""
    out: dict[str, int] = {}
    for spec in specs or []:
        method, sep, first = spec.partition("=")
        if not sep:
            raise ValueError(f"--index-variables takes METHOD=FIRST, got {spec!r}")
        out[method] = int(first)
    return out
