"""One table for a whole evaluation: a row per problem and rung, every per-problem metric.

A result tree is the directory an evaluation writes for one method and one draw: ``<tree>/<catalog>/choices_<rung>.pkl``
(``niter_``, ``evals_`` or ``samples_<rung>.pkl`` for a ladder that counts iterations, evaluations or sampled
expressions), each file possibly split into shards
``<stem>.shard-K-of-N.pkl``. :func:`judge_result_file` turns one file into rows with :func:`srbf.derive_metrics`;
:func:`build_table` judges any number of trees in parallel and writes the rows as CSV (``srbf table``). The results
site is built from this table.

Row semantics:

* rate columns (``success``, the numeric and symbolic recoveries, ``skeleton_match_raw``) are defined for every
  problem: a failed prediction is a miss (0), not a missing value. The exceptions are rates that are not defined: a
  problem without validation points (a data catalog whose authors give no test split, ``n_validation`` 0) has no
  validation-split rates (``numeric_recovery_val``, ``numeric_recovery_relative_val``, ``symbolic_recovery_mask_none``),
  and a split whose target is the same value on every point has no numeric rates for that split (its FVU is 0/0 or x/0
  whatever the prediction): they are empty, as that split's analysis columns are. Empty means undefined, never a miss;
* analysis columns (FVU, R², distances, lengths, MDL, ...) describe the predictions that were made and are empty for a
  failed one, except the metrics whose range has a worst value (:data:`srbf.result_processing.WORST_VALUE`), which a
  failed prediction takes;
* ground-truth columns (``skeleton_length``, ``ground_truth_mdl``, ...) are defined for every problem.

Description lengths are in bits (certified, f64 parse, default canon). A per-file cache (``cache_dir``) keeps the rows
of every file judged before; its key names the file (path, size, modification time) and the judge
(:func:`judge_fingerprint`: the source of the srbf modules the judge runs, the versions of the libraries they import,
the engine), so a file is judged again exactly when it or the judge changed.
"""
from __future__ import annotations

import csv
import glob
import hashlib
import math
import os
import pickle
import sys
import tempfile
import time
import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
import multiprocessing
from multiprocessing import TimeoutError as PoolTimeout
from pathlib import Path
from typing import Any, Mapping

from srbf.rungs import RESULT_FILE, RUNG_PREFIXES  # a result file's name: the rung and, for a shard, its index and count


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
GROUND_TRUTH_COLUMNS = ["skeleton_length", "ground_truth_mdl", "n_constants", "total_nestedness", "n_variables", "n_support",
                        "n_validation", "constant_target_fit", "constant_target_val"]
#: The expressions themselves, as the judge read them: prefix tokens joined by spaces, numbers at full precision. The
#: prediction is in the ground truth's variable names and the engine's spelling; empty when the method gave none.
EXPRESSION_COLUMNS = ["predicted_expression", "ground_truth_expression"]
#: The table's columns, in order.
COLUMNS = ID_COLUMNS + RATE_COLUMNS + ANALYSIS_COLUMNS + GROUND_TRUTH_COLUMNS + EXPRESSION_COLUMNS
#: Bumped whenever a row's content changes without a change to srbf's source (it is part of the cache key).
TABLE_VERSION = 2   # 2: n_validation, constant targets, and empty rates where they are not defined

#: The rates read on the validation points: empty for a problem without them.
_VALIDATION_RATES = {"numeric_recovery_val", "numeric_recovery_relative_val", "symbolic_recovery_mask_none"}
_FIT_RATES = {"numeric_recovery_fit", "numeric_recovery_relative_fit"}
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
    files = [f for prefix in RUNG_PREFIXES for f in glob.glob(os.path.join(tree.path, "*", f"{prefix}_*.pkl"))]
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


def _spell_predictions(snapshot: dict[str, Any], operator_arity: Mapping[str, int]) -> None:
    """Bring the stored predictions into the engine's spelling (:mod:`srbf.spelling`), in place. A file written before
    the in-process adapters did so carries SymPy's ``sqrt`` and ``Abs``, which the engine can neither price nor judge.
    Only such predictions are rewritten, into the forms the adapters now store."""
    from symbolic_data.token_ops import normalize_expression, normalize_skeleton

    from srbf.spelling import SYMPY_ONLY, engine_spelling

    for key, normalize in (("predicted_expression_prefix", normalize_expression), ("predicted_skeleton_prefix", normalize_skeleton)):
        seq = snapshot.get(key)
        if seq is None:
            continue
        snapshot[key] = [normalize(engine_spelling(prefix, operator_arity))
                         if prefix is not None and SYMPY_ONLY.intersection(map(str, prefix)) else prefix for prefix in seq]


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
    _spell_predictions(snap, engine.operator_arity)
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
        n_val = got("n_validation", i)
        undefined_val = (n_val is not None and int(n_val) == 0) or bool(got("fit_quality_undefined_val", i))
        undefined_fit = bool(got("fit_quality_undefined_fit", i))
        for k in _RECOVERIES:
            v = got(k, i)
            if (undefined_val and k in _VALIDATION_RATES) or (undefined_fit and k in _FIT_RATES):
                r[k] = ""            # no points, or a target without variance: not a miss, not defined
                continue
            r[k] = int(bool(v)) if (ok and v is not None) else 0
        r["skeleton_match_raw"] = int(ok and ps is not None and sk is not None and ps == sk)
        # ground truth
        r["skeleton_length"] = _number(got("skeleton_length", i)) if got("skeleton_length", i) is not None else (len(sks) if sks else "")
        r["ground_truth_mdl"] = _number(got("ground_truth_mdl", i))
        r["n_constants"] = _number(got("n_constants", i))
        r["total_nestedness"] = _number(got("total_nestedness", i))
        r["n_variables"] = _number(got("n_variables", i))
        r["n_support"] = _number(raw("n_support", i))
        r["n_validation"] = _number(n_val)
        r["constant_target_fit"] = int(undefined_fit)
        r["constant_target_val"] = int(bool(got("fit_quality_undefined_val", i)) and not (n_val is not None and int(n_val) == 0))
        pe, ge = raw("predicted_expression_prefix", i), raw("ground_truth_prefix", i)
        r["predicted_expression"] = " ".join(map(str, pe)) if (ok and pe is not None) else ""
        r["ground_truth_expression"] = " ".join(map(str, ge)) if ge is not None else ""
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
JUDGE_ENTRY = "srbf.table"
# Read by the judge through other libraries (simplipy's simplification), so no import of the judge's modules names it.
JUDGE_LIBRARIES = ("sympy",)


def _imported_modules(name: str, path: Path, modules: Mapping[str, Path]) -> tuple[set[str], set[str]]:
    """(srbf modules, top-level third-party packages) one srbf module imports, at module level or inside functions.

    An import names the module that defines what it takes: ``from srbf.x import y`` is ``srbf.x`` (or ``srbf.x.y`` when
    that is a module), never the packages above it, whose ``__init__`` only re-exports. A bare ``import srbf`` (to
    locate the sources) is not a dependency."""
    import ast

    package = name if path.name == "__init__.py" else name.rsplit(".", 1)[0]
    ours: set[str] = set()
    theirs: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("srbf."):
                    ours.add(alias.name)
                elif alias.name != "srbf":
                    theirs.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".")
                module = ".".join(parts[:len(parts) - (node.level - 1)] + ([node.module] if node.module else []))
            else:
                module = node.module or ""
            if module == "srbf" or module.startswith("srbf."):
                ours |= {f"{module}.{a.name}" if f"{module}.{a.name}" in modules else module for a in node.names}
            elif module:
                theirs.add(module.split(".")[0])
    stdlib = set(getattr(sys, "stdlib_module_names", ())) | {"__future__"}
    return {m for m in ours if m in modules}, theirs - stdlib


def judge_dependencies(root: Path, entry: str = JUDGE_ENTRY) -> tuple[list[Path], list[str]]:
    """The judge's sources and libraries: every srbf module ``entry`` reaches through its imports (read from the
    source, so a new dependency joins by itself), and the third-party packages those modules import. The out-of-process
    workers (``srbf/worker/``) and the list of rung file names (``srbf/rungs.py``, which decides which files are read,
    not how a row is judged) are left out."""
    modules: dict[str, Path] = {}
    for p in root.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        rel = p.relative_to(root.parent).with_suffix("")
        modules[".".join(rel.parts[:-1] if rel.name == "__init__" else rel.parts)] = p
    seen: set[str] = set()
    libraries: set[str] = set()
    todo = [entry]
    while todo:
        name = todo.pop()
        if name in seen or name not in modules:
            continue
        seen.add(name)
        ours, theirs = _imported_modules(name, modules[name], modules)
        libraries |= theirs
        todo += sorted(ours - seen)
    skipped = {f"{root.name}.rungs"}
    sources = [modules[m] for m in sorted(seen) if m not in skipped and not m.startswith(f"{root.name}.worker")]
    return sources, sorted(libraries | set(JUDGE_LIBRARIES))


def judge_fingerprint(engine_name: str) -> str:
    """Names the judge: the source of the srbf modules it runs and the versions of the libraries they import
    (:func:`judge_dependencies`), the engine and the table version. A change anywhere else in srbf (the site export,
    the reports, a baseline worker) does not re-judge every result file."""
    import srbf
    from importlib.metadata import PackageNotFoundError, packages_distributions, version as dist_version

    root = Path(srbf.__file__).resolve().parent
    sources, libraries = judge_dependencies(root)
    h = hashlib.sha256()
    for p in sources:
        h.update(str(p.relative_to(root)).encode())
        h.update(p.read_bytes())
    distributions = packages_distributions()
    for name in libraries:
        try:
            v = dist_version((distributions.get(name) or [name])[0])
        except PackageNotFoundError:   # a missing library is part of the judge too
            v = "?"
        h.update(f"{name}={v}".encode())
    h.update(f"engine={engine_name}|table={TABLE_VERSION}".encode())
    return h.hexdigest()[:16]


def _cache_key(tree: ResultTree, path: str, fingerprint: str) -> str:
    st = os.stat(path)
    ident = f"{tree.method}|{tree.draw}|{tree.first_index}|{os.path.abspath(path)}|{st.st_size}|{st.st_mtime_ns}|{fingerprint}"
    return hashlib.sha1(ident.encode()).hexdigest()


def _cache_path(cache_dir: str, key: str) -> str:
    return os.path.join(cache_dir, key[:2], key + ".pkl")


def _cache_load(cache_dir: str, key: str) -> list[list[Any]] | None:
    p = _cache_path(cache_dir, key)
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
    os.replace(tmp, _cache_path(cache_dir, key))


# ---- parallel judging -------------------------------------------------------------------------------------------
_ENGINE: Any = None
_ENGINE_NAME: str | None = None


def _worker_engine(name: str) -> Any:
    global _ENGINE, _ENGINE_NAME
    if _ENGINE is None or _ENGINE_NAME != name:
        from simplipy import SimpliPyEngine
        _ENGINE, _ENGINE_NAME = SimpliPyEngine.load(name, install=True), name
    return _ENGINE


def _mark(progress_path: str | None, tag: str, index: int, path: str) -> None:
    if not progress_path:
        return
    try:
        with open(progress_path, "a") as fh:
            fh.write(f"{tag} {index} {time.time():.3f} {path}\n")
    except OSError:
        pass


def _judge_job(job: tuple[int, ResultTree, str, str, str | None]) -> tuple[int, list[list[Any]], str | None]:
    """(job index, rows, error). The file is marked in flight while it is judged, so a file that does not finish can be
    found and stopped."""
    index, tree, path, engine_name, progress_path = job
    _mark(progress_path, "S", index, path)
    try:
        warnings.simplefilter("ignore")
        try:
            rows = judge_result_file(path, method=tree.method, draw=tree.draw, engine=_worker_engine(engine_name),
                                     first_index=tree.first_index)
        except Exception as e:  # noqa: BLE001 - a file that cannot be judged is reported, not fatal
            return index, [], f"{path}: {type(e).__name__}: {e}"
        return index, rows, None
    finally:
        _mark(progress_path, "D", index, path)


def _in_flight(progress_path: str | None) -> dict[int, tuple[float, str]]:
    """The jobs started and not done: {job index: (start time, path)}."""
    if not progress_path or not os.path.exists(progress_path):
        return {}
    running: dict[int, tuple[float, str]] = {}
    for line in open(progress_path):
        parts = line.rstrip("\n").split(" ", 3)
        if len(parts) < 4:
            continue
        tag, index, started, path = parts
        if tag == "S":
            running[int(index)] = (float(started), path)
        elif tag == "D":
            running.pop(int(index), None)
    return running


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
                progress_path: str | None = None, log: Callable[[str], None] | None = print,
                _judge: Callable[..., tuple[int, list[list[Any]], str | None]] = _judge_job) -> TableReport:
    """Judge every result file of ``trees`` and write the table to ``out`` (CSV, written atomically).

    Files are judged by ``workers`` processes, largest first, and taken as they finish. A file still running
    ``stall_timeout`` seconds after it started is stopped with its pool and judged once more in a fresh one; a file
    that does not finish twice is left out and reported (``STALLED``, ``stalled``), and every other file is judged. A
    file that cannot be read or judged is reported in ``errors`` and left out (a result file that is still being
    written is the usual cause). ``progress_path`` (a temporary file when not given) records which files are in flight.

    The rows are written in the files' order, judged now or read from ``cache_dir``, one file at a time: holding every
    file's rows until the pool is through took 5 GB for the board's 23,000 files (2026-10-08). A cache entry that cannot
    be read then is a miss like any other: the file is judged in this process and stored again. Without a cache the
    rows judged now are held until they are written.

    ``_judge`` is the job function (a test stands in a slow one).
    """
    say = log or (lambda _msg: None)
    t0 = time.time()
    jobs: list[tuple[ResultTree, str]] = []
    for tree in trees:
        jobs += [(tree, f) for f in result_files(tree, max_rung=max_rung)]
    jobs.sort(key=lambda j: (-os.path.getsize(j[1]), j[1], j[0].method, j[0].draw))
    cached: set[int] = set()
    keys: dict[int, str] = {}
    if cache_dir:
        fingerprint = judge_fingerprint(engine)
        for i, (tree, path) in enumerate(jobs):
            keys[i] = _cache_key(tree, path, fingerprint)
            if os.path.exists(_cache_path(cache_dir, keys[i])):
                cached.add(i)
    pending = [i for i in range(len(jobs)) if i not in cached]
    say(f"{len(jobs)} files: {len(pending)} to judge, {len(cached)} from the cache")
    own_progress = progress_path is None
    if progress_path is None:
        fd, progress = tempfile.mkstemp(prefix=".srbf-table-", suffix=".progress", dir=os.path.dirname(os.path.abspath(out)))
        os.close(fd)
    else:
        progress = progress_path
    open(progress, "w").close()
    errors: list[str] = []
    held: dict[int, list[list[Any]]] = {}      # rows judged now, without a cache
    failed: set[int] = set()
    judged, stalled = 0, False

    def keep(i: int, rows: list[list[Any]], err: str | None) -> None:
        if err:
            errors.append(err)
            failed.add(i)
        elif cache_dir:
            _cache_store(cache_dir, keys[i], rows)
        else:
            held[i] = rows

    # forkserver: a fork of a process that already runs threads (the engine's, numpy's) can deadlock the child
    methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context("forkserver" if "forkserver" in methods else "spawn")
    poll = max(0.5, min(30.0, stall_timeout / 10))
    timeouts: dict[int, int] = {}
    queue = list(pending)
    try:
        while queue:
            remaining = set(queue)
            pool = context.Pool(max(1, workers), maxtasksperchild=25)
            restart, last = False, time.time()
            try:
                results = pool.imap_unordered(_judge, [(i, jobs[i][0], jobs[i][1], engine, progress) for i in queue],
                                              chunksize=1)
                while remaining:
                    try:
                        i, rows, err = results.next(timeout=poll)  # type: ignore[attr-defined]
                    except PoolTimeout:
                        now = time.time()
                        running = {i: v for i, v in _in_flight(progress).items() if i in remaining}
                        late = [i for i, (started, _p) in running.items() if now - started > stall_timeout]
                        if late:
                            for i in late:
                                timeouts[i] = timeouts.get(i, 0) + 1
                                if timeouts[i] >= 2:
                                    say(f"STALLED: {jobs[i][1]} did not finish in {stall_timeout:.0f}s twice; left out")
                                    errors.append(f"{jobs[i][1]}: did not finish in {stall_timeout:.0f}s, twice")
                                    failed.add(i)
                                    remaining.discard(i)
                                    stalled = True
                                else:
                                    say(f"RETRY {jobs[i][1]}: not finished after {stall_timeout:.0f}s; judging it once more in a fresh worker")
                            restart = True
                            break
                        if not running and now - last > stall_timeout:   # nothing in flight and nothing finishing
                            say(f"STALLED after {judged}/{len(pending)} files: none finished for {stall_timeout:.0f}s and none in flight")
                            failed |= remaining
                            remaining.clear()
                            stalled = True
                            restart = True
                            break
                        continue
                    except StopIteration:
                        break
                    last = time.time()
                    remaining.discard(i)
                    judged += 1
                    keep(i, rows, err)
                    if judged % 50 == 0 or judged == len(pending):
                        say(f"{judged}/{len(pending)} files judged, {time.time() - t0:.0f}s")
            finally:
                pool.terminate() if restart else pool.close()
                pool.join()
            if not restart:
                break
            queue = sorted(remaining, key=lambda i: (-timeouts.get(i, 0), i))   # a retried file first, alone if need be
            open(progress, "w").close()

        n_rows, from_cache = 0, 0
        tmp = out + ".tmp"
        with open(tmp, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(COLUMNS)
            for i in range(len(jobs)):
                if i in failed:
                    continue
                stored: list[list[Any]] | None = held.pop(i) if i in held else None
                if stored is None and cache_dir:
                    stored = _cache_load(cache_dir, keys[i])
                    if stored is not None and i in cached:
                        from_cache += 1
                if stored is None:                             # a cache entry that cannot be read is a miss
                    _i, stored, err = _judge_job((i, jobs[i][0], jobs[i][1], engine, None))
                    keep(i, stored, err)
                    held.pop(i, None)
                    judged += 1
                    if err:
                        continue
                writer.writerows(stored)
                n_rows += len(stored)
        os.replace(tmp, out)
    finally:
        if own_progress:
            try:
                os.remove(progress)
            except OSError:
                pass
    report = TableReport(files=len(jobs), judged=judged, from_cache=from_cache, rows=n_rows, errors=errors,
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
