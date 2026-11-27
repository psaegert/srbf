"""Accepting frozen draws: each problem keeps the first seeded attempt on which its own formula can be recognized.

With one draw per problem, a draw on which the true formula cannot be recovered wastes the problem for every method.
symbolic-data draws attempt ``k = 1, 2, ...`` of a problem from its own seed (``symbolic_data.draws.CatalogDraws``):
only what the benchmark leaves to chance changes between attempts (input points, random constants, per-run values),
never the formula or its fixed constants. This module judges each attempt in that order and writes the attempts
document symbolic-data renders with (``symbolic-data render-draws --attempts``). An attempt is accepted when

(a) every target is finite and the FVU is defined on the fitting and on the held-out points (the targets vary);
(b) at some restart, srbf's oracle (:mod:`srbf.oracle`: the ground truth with its constants refitted, no checkpoint
    involved) reaches numeric recovery on the held-out points (``is_perfect_fit``), and
(c) srbf's judge recognizes that restart's refitted formula (``symbolic_recovery``).

The oracle runs restart by restart, restart ``r`` from the seed ``(seed, r)``, and stops at the first restart that
passes both (b) and (c) (:func:`srbf.oracle.first_recovery`): the check only asks whether the formula can be recovered
from the draw, and the first success answers that. An attempt fails only when none of its ``n_restarts`` restarts
passes both, so that its verdict records the draw and not the luck of one restart: where the constants of a law are
not identifiable from the data (only their product or ratio is), restarts land on different members of the family of
equivalent constants, and the judge recognizes some of them and not others. An attempt gets at most
:data:`srbf.headline.T_MAX` (1,000) CPU seconds, measured on the machine that runs the check and not calibrated to the
reference machine; at the cap the restart in flight is abandoned and the attempt fails. A failed attempt's kind is the
first check it fails: ``placeholder``, ``not_finite`` or ``fvu_undefined`` (a), ``no_prediction`` (no restart
converged), ``numeric`` (no restart reproduced the held-out points), ``symbolic`` (some did, none was recognized) or
``time`` (the cap came first).

The first accepted attempt within the render's ``max_trials`` is kept. When none is accepted, attempt 1 is kept and
the problem is flagged ``formula_inherent``. The attempts document records, for every checked problem, the kind of
each failed attempt in order (``failures``). Formulas that are constant by construction and data problems are not
checked. symbolic-data never imports srbf: the two meet in ``CatalogDraws.draw`` and the attempts document.

Every problem is an independent task: a run can be split into shards (``shard=(k, n)``), each with a journal of its
own, and :func:`merge_journals` writes the attempts document from the shards' journals. The journal holds one line per
attempt, so a shard stopped in the middle of a problem resumes at that problem's next attempt.
"""
from __future__ import annotations

import json
import os
import signal
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np

from srbf.headline import T_MAX

#: The rule, as the attempts document and the draws' manifest record it.
RULE = ("the first seeded attempt (1..max_trials) whose targets are finite with a defined FVU on both sides, on which "
        "srbf's oracle (the ground truth with its constants refitted, restart by restart from seeded starting values, "
        "at most {n_restarts} restarts) reaches, at some restart, numeric recovery on the held-out points with a refitted "
        "formula srbf's judge recognizes, stopping at the first such restart, within {time_cap:g} CPU seconds of the "
        "machine running the check; otherwise attempt 1, flagged formula_inherent. Constant formulas and data problems "
        "are not checked.")

ACCEPTED = "accepted"
#: Why an attempt is not accepted, in the order the rule checks: (a), (b), (c). ``symbolic``: some restart recovered
#: the held-out points numerically, and the judge recognized none of those restarts' formulas.
REASONS = ("placeholder", "not_finite", "fvu_undefined", "no_prediction", "numeric", "symbolic", "time")
#: The failure kind of an attempt that reached its CPU-time cap.
TIME = "time"


class AttemptTimeout(BaseException):
    """Raised inside an attempt when it reaches its CPU-time cap. A ``BaseException``, so that no ``except Exception``
    on the way (the oracle's candidate path has several) can swallow it."""


@contextmanager
def cpu_time_cap(seconds: float | None) -> Iterator[None]:
    """Raise :class:`AttemptTimeout` in the block once the process has spent ``seconds`` of CPU time in it (the
    profiling timer: user and system time of the whole process, on this machine). None: no cap. The timer and the
    previous signal handler are restored on every way out; no thread or process is started. The main thread only (the
    signal is delivered there)."""
    if seconds is None:
        yield
        return
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("cpu_time_cap needs the main thread, where the profiling signal is delivered")
    armed = [True]

    def expire(signum: int, frame: Any) -> None:
        if armed[0]:
            armed[0] = False
            raise AttemptTimeout()

    previous = signal.signal(signal.SIGPROF, expire)
    signal.setitimer(signal.ITIMER_PROF, max(float(seconds), 1e-6))
    try:
        yield
    finally:
        armed[0] = False
        signal.setitimer(signal.ITIMER_PROF, 0)
        signal.signal(signal.SIGPROF, previous)


class _One:
    """A problem source of one problem, so that the problem reaches the adapter as a run's would."""

    def __init__(self, problem: Any) -> None:
        self.problem = problem

    def __iter__(self) -> Any:
        yield self.problem

    def size_hint(self) -> int:
        return 1


class OracleCheck:
    """The acceptance test of one attempt: ``check(problem) -> "accepted"`` or the first reason (:data:`REASONS`) it fails.

    ``adapter`` is an ``oracle`` adapter (:class:`srbf.model_adapters.OracleAdapter`): its engine, its refinement
    settings (whose ``n_restarts`` is the most restarts an attempt gets), its ranking and its seed. ``judge_engine`` is the
    engine srbf's judge runs with (``srbf table``'s default, ``acj-5-4-llm``). Every restart whose answer recovers the
    held-out points numerically is judged; the first one the judge recognizes accepts the attempt. ``time_cap`` is the
    most CPU seconds an attempt gets (:data:`srbf.headline.T_MAX`; None: no cap): at the cap the restart in flight is
    abandoned and the attempt fails with :data:`TIME`. ``spent`` holds the restarts the last attempt took.
    """

    def __init__(self, adapter: Any, judge_engine: Any, engine_name: str | None = None,
                 time_cap: float | None = T_MAX) -> None:
        self.adapter = adapter
        self.judge_engine = judge_engine
        #: the judge engine's name, so that a worker drawing the attempts with the same engine can share it
        self.engine_name = engine_name
        self.time_cap = time_cap
        self.spent = 0

    def _judge(self, record: Mapping[str, Any]) -> Callable[[str], Any]:
        from srbf.table import judge_snapshot

        judged = judge_snapshot({key: [value] for key, value in record.items()}, engine=self.judge_engine)

        def first(key: str) -> Any:
            values = judged.get(key)
            return None if values is None else values[0]

        return first

    def __call__(self, problem: Any) -> str:
        self.spent = 0
        try:
            with cpu_time_cap(self.time_cap):
                return self._check(problem)
        except AttemptTimeout:
            return TIME

    def _count(self, restarts: int) -> None:
        self.spent = restarts

    def _check(self, problem: Any) -> str:
        from srbf.data_sources import CatalogSource
        from srbf.model_adapters import OracleAdapter
        from srbf.oracle import first_recovery

        if getattr(problem, "is_placeholder", False):
            return "placeholder"
        if not (np.isfinite(problem.y_support).all() and np.isfinite(problem.y_validation).all()):
            return "not_finite"
        source = CatalogSource(_One(problem))
        source.prepare(adapter=self.adapter)
        sample = next(iter(source))
        # whether the FVU is defined is a property of the targets: the judge says so before the oracle runs
        targets = self._judge({**sample.clone_metadata(), "prediction_success": False})
        if targets("fit_quality_undefined_fit") or targets("fit_quality_undefined_val"):
            return "fvu_undefined"
        y_fit = sample.y_support_noisy if sample.y_support_noisy is not None else sample.y_support

        def recognized(fit: Any) -> bool:
            judged = self._judge(OracleAdapter.answer(sample.clone_metadata(), sample, fit))
            return bool(judged("numeric_recovery_val")) and bool(judged("symbolic_recovery"))

        recovery = first_recovery(
            list(sample.metadata.get("expression") or []), sample.x_support, y_fit, sample.x_validation,
            sample.y_validation, engine=self.adapter.get_simplipy_engine(), refine=self.adapter.refine,
            ranking=self.adapter.ranking, seed=self.adapter.seed or 0,
            variable_names=sample.metadata.get("variable_names"), accept=recognized, on_restart=self._count)
        self.spent = recovery.restarts
        if recovery.recovered:
            return ACCEPTED
        if recovery.numeric:
            return "symbolic"
        return "numeric" if recovery.answered else "no_prediction"


def oracle_adapter_config(config: str | os.PathLike[str], *, n_restarts: int, seed: int,
                          experiment: str | None = None) -> dict[str, Any]:
    """The oracle's ``model_adapter`` block from a reference config (an experiment of the oracle's scaling config), at
    ``n_restarts`` restarts and seeded."""
    from srbf import config as run_config
    from srbf.sweep import resolve_sweeps

    raw = run_config.load_run_config(str(config))
    experiments = raw.get("experiments") or {}
    chosen = run_config.select_experiment(raw, experiment or (next(iter(experiments)) if experiments else None))
    resolved, _ = resolve_sweeps(chosen)[0]
    adapter = dict(run_config.extract_run_section(resolved)["model_adapter"])
    if adapter.get("type") != "oracle":
        raise ValueError(f"{config}: not an oracle config (model_adapter.type: oracle)")
    adapter.update(n_restarts=int(n_restarts), seed=int(seed))
    return adapter


def oracle_check(config: str, n_restarts: int = 1024, seed: int = 0, judge_engine: str = "acj-5-4-llm",
                 experiment: str | None = None, time_cap: float | None = T_MAX) -> OracleCheck:
    """An :class:`OracleCheck` built from the oracle's reference config (picklable through ``functools.partial``)."""
    from simplipy import SimpliPyEngine

    from srbf import config as run_config

    block = oracle_adapter_config(config, n_restarts=n_restarts, seed=seed, experiment=experiment)
    adapter = run_config.build_model_adapter(block)
    # one engine per process where the oracle's engine is the judge's (a loaded engine holds its rule set in memory)
    engine = (adapter.get_simplipy_engine() if str(block.get("simplipy_engine")) == judge_engine
              else SimpliPyEngine.load(judge_engine, install=True))
    return OracleCheck(adapter, engine, engine_name=judge_engine, time_cap=time_cap)


# ------------------------------------------------------------------------------------------------ the attempt loop
_WORKER: dict[str, Any] = {}


#: The thread pools of the numerical libraries; a worker gets one core, the pool of workers is the parallelism.
_THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def _init_worker(check_factory: Callable[[], Callable[[Any], str]], settings: Any, one_thread: bool = True) -> None:
    if one_thread:
        try:
            import torch
            torch.set_num_threads(1)
        except ImportError:  # pragma: no cover - flash-ansr's refiner imports torch; a fake check does not
            pass
    _WORKER.clear()
    _WORKER.update(check=check_factory(), settings=settings, draws={})


def _catalog(ref: str) -> Any:
    from symbolic_data.draws import CatalogDraws

    cache: dict[str, Any] = _WORKER["draws"]
    if ref not in cache:
        if len(cache) >= 8:                      # tasks come catalog by catalog: keep the few in use
            cache.pop(next(iter(cache)))
        settings, check = _WORKER["settings"], _WORKER["check"]
        shared = check.judge_engine if getattr(check, "engine_name", None) == settings.simplipy_engine else None
        cache[ref] = CatalogDraws(ref, settings, simplipy_engine=shared)
    return cache[ref]


#: One judged attempt as the journal holds it: ``{catalog, eq_id, repeat, attempt, verdict, timing}``. ``verdict`` is
#: :data:`ACCEPTED`, a :data:`REASONS` entry, or None for a problem the rule does not apply to (attempt 1, then done).
Attempt = dict[str, Any]


def judge_attempt(ref: str, row: int, repeat: int, attempt: int) -> Attempt:
    """Attempt ``attempt`` of problem ``row`` of catalog ``ref`` in ``repeat``, judged by the worker's check."""
    from symbolic_data.draws import needs_check

    start, cpu = time.perf_counter(), time.process_time()
    draws = _catalog(ref)
    check = _WORKER["check"]
    draw = draws.draw(row, repeat, attempt)
    verdict = check(draw.problem) if attempt > 1 or needs_check(draw.kind, draw.problem) else None
    # the oracle's restarts, when the check counts them (OracleCheck.spent)
    restarts = int(getattr(check, "spent", 0) or 0) if verdict is not None else 0
    return {"catalog": draws.name, "eq_id": draws.ids[row], "repeat": int(repeat), "attempt": int(attempt), "verdict": verdict,
            "timing": {"seconds": time.perf_counter() - start, "cpu_seconds": time.process_time() - cpu, "restarts": restarts}}


def _attempt_task(task: tuple[str, int, int, int]) -> Attempt:
    return judge_attempt(*task)


def problem_choice(attempts: Mapping[int, Attempt], max_trials: int) -> tuple[bool, dict[str, Any] | None, int]:
    """A problem's state from its judged attempts (by attempt number): ``(done, choice, next attempt)``. ``choice`` is
    what the attempts document lists, None for a problem the rule does not apply to."""
    failures: list[str] = []
    for k in range(1, int(max_trials) + 1):
        entry = attempts.get(k)
        if entry is None:
            return False, None, k
        verdict = entry["verdict"]
        if verdict is None:
            return True, None, k
        if verdict == ACCEPTED:
            return True, {"attempt": k, "formula_inherent": False, "failures": failures}, k
        failures.append(str(verdict))
    return True, {"attempt": 1, "formula_inherent": True, "failures": failures}, int(max_trials)


def _timing(attempts: Mapping[int, Attempt]) -> dict[str, float]:
    return {key: sum(a["timing"].get(key, 0) for a in attempts.values()) for key in ("seconds", "cpu_seconds", "restarts")}


#: The journal's layout, in its header: one line per attempt (a journal of another layout is refused).
JOURNAL = "draw-acceptance-attempts/1"


def _read_journal(path: Path) -> tuple[dict[str, Any], list[Attempt]]:
    """A journal's header and the attempts it holds. A last line cut short when the run stopped is dropped from the
    file, so that the next attempt starts a line of its own."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines:
        raise ValueError(f"{path} is empty: not a journal")
    if not text.endswith("\n"):
        lines = lines[:-1]
        path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    done: list[Attempt] = []
    for line in lines[1:]:
        try:
            done.append(json.loads(line))
        except json.JSONDecodeError:           # a line cut short by an earlier stop, before this repair existed
            continue
    return json.loads(lines[0]), done


def _journal(path: Path, header: dict[str, Any]) -> list[Attempt]:
    """The attempts a journal already holds (after checking that it was written for this run), or none."""
    if not path.exists():
        path.write_text(json.dumps(header) + "\n", encoding="utf-8")
        return []
    found, done = _read_journal(path)
    if found != header:
        raise ValueError(f"{path} was written for other catalogs, settings, another rule, another shard or another "
                         f"journal layout; use another journal")
    return done


def _plan(refs: Sequence[str], settings: Any, ids: Mapping[str, Sequence[str]]) -> tuple[
        dict[str, Any], dict[str, list[str]], list[tuple[str, int, int]], dict[tuple[str, int, int], tuple[str, str, int]]]:
    """Every catalog of ``refs`` (its source and, per repeat, the attempts to fill), its problems in order, and the
    tasks: one per problem and repeat, keyed by ``(catalog name, eq_id, repeat)``."""
    from symbolic_data.draws import CatalogDraws

    from simplipy import SimpliPyEngine

    catalogs: dict[str, Any] = {}
    order: dict[str, list[str]] = {}
    tasks: list[tuple[str, int, int]] = []
    keys: dict[tuple[str, int, int], tuple[str, str, int]] = {}
    engine = SimpliPyEngine.load(settings.simplipy_engine, install=True)     # the problem lists need it loaded once
    for ref in refs:
        draws = CatalogDraws(ref, settings, simplipy_engine=engine)
        if draws.name in catalogs:
            raise ValueError(f"two catalogs are named {draws.name!r}")
        rows = draws.rows(ids.get(ref, ids.get(draws.name, ids.get(Path(ref).stem))))
        catalogs[draws.name] = {"source_sha256": draws.identity["source_sha256"], "attempts": {str(r): {} for r in settings.repeats}}
        order[draws.name] = [draws.ids[row] for row in rows]
        for repeat in settings.repeats:
            for row in rows:
                tasks.append((ref, row, repeat))
                keys[(ref, row, repeat)] = (draws.name, draws.ids[row], repeat)
    return catalogs, order, tasks, keys


def _header(settings: Any, rule: str, catalogs: Mapping[str, Any], order: Mapping[str, list[str]],
            shard: tuple[int, int] | None) -> dict[str, Any]:
    header = {"journal": JOURNAL, "settings": settings.record(), "repeats": list(settings.repeats), "rule": rule,
              "catalogs": {name: entry["source_sha256"] for name, entry in catalogs.items()},
              "problems": {name: len(problems) for name, problems in order.items()}}
    if shard is not None:
        header["shard"] = list(shard)
    return header


def _by_problem(attempts: Sequence[Attempt]) -> dict[tuple[str, str, int], dict[int, Attempt]]:
    """Attempts grouped by problem and numbered; an attempt journaled twice keeps its first result."""
    grouped: dict[tuple[str, str, int], dict[int, Attempt]] = {}
    for entry in attempts:
        grouped.setdefault((entry["catalog"], entry["eq_id"], int(entry["repeat"])), {}).setdefault(int(entry["attempt"]), entry)
    return grouped


def _document(out: str | os.PathLike[str], catalogs: dict[str, Any], order: Mapping[str, list[str]],
              problems: Mapping[tuple[str, str, int], Mapping[int, Attempt]], settings: Any, rule: str) -> dict[str, Any]:
    """Write the attempts document of the finished ``problems`` to ``out`` and return the summary (without the wall
    time)."""
    from symbolic_data.draws import ATTEMPTS_FORMAT

    seconds: dict[str, dict[str, dict[str, float]]] = {}
    for (name, eq_id, repeat), attempts in problems.items():
        done, choice, _next = problem_choice(attempts, settings.max_trials)
        if not done:
            continue
        seconds.setdefault(name, {})[eq_id] = _timing(attempts)
        if choice is not None:
            catalogs[name]["attempts"][str(repeat)][eq_id] = choice
    for name, entry in catalogs.items():                  # the catalog's order, whatever order the workers finished in
        rank = {eq_id: i for i, eq_id in enumerate(order[name])}
        entry["attempts"] = {r: {eq_id: per[eq_id] for eq_id in sorted(per, key=rank.__getitem__)} for r, per in entry["attempts"].items()}
    doc = {"format": ATTEMPTS_FORMAT, "format_version": 1, "settings": settings.record(), "repeats": list(settings.repeats),
           "rule": rule, "catalogs": catalogs}
    Path(out).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    chosen = [c for entry in catalogs.values() for per in entry["attempts"].values() for c in per.values()]
    return {"problems": sum(len(v) for v in seconds.values()), "checked": len(chosen),
            "later_attempt": sum(c["attempt"] > 1 for c in chosen), "formula_inherent": sum(c["formula_inherent"] for c in chosen),
            "time": sum(c["failures"].count(TIME) for c in chosen), "seconds": seconds}


def accept_draws(refs: Sequence[str], out: str | os.PathLike[str], *, check_factory: Callable[[], Callable[[Any], str]],
                 rule: str, settings: Any = None, ids: Mapping[str, Sequence[str]] | None = None, jobs: int = 1,
                 journal: str | os.PathLike[str] | None = None, shard: tuple[int, int] | None = None,
                 log: Callable[[str], None] = print) -> dict[str, Any]:
    """Judge every checked problem of ``refs`` and write the attempts document to ``out``; returns a summary.

    ``check_factory`` builds the check once per worker process (a picklable callable, such as
    ``functools.partial(oracle_check, ...)``). ``ids`` restricts a catalog (by ref or name) to a frozen subset, as the
    render will. The document lists every catalog, and in it every checked problem with the attempt it keeps, its flag
    and the kind of each failed attempt in order (``failures``, :data:`REASONS`). ``journal`` appends each attempt as it
    is judged, and a run started again on the same journal goes on where it stopped: a problem whose attempts are all
    journaled is not judged again, and one stopped in the middle resumes at its next attempt. ``shard=(k, n)`` judges
    only the problems ``k, k + n, k + 2n, ...`` of the plan (every problem of every catalog, repeat by repeat): shards
    are independent runs, each with a journal of its own, and :func:`merge_journals` writes the document of all of
    them. The summary adds each problem's seconds, CPU seconds and the oracle's restarts, summed over its attempts (not
    in the document, which a rerun with the same software reproduces), and the number of attempts that hit the cap.
    """
    from symbolic_data.draws import DrawsSettings

    settings = settings or DrawsSettings()
    catalogs, order, tasks, keys = _plan(refs, settings, dict(ids or {}))
    if shard is not None:
        k, n = (int(v) for v in shard)
        if not 0 <= k < n:
            raise ValueError(f"shard {k}/{n}: the shard index must be in 0..{n - 1}")
        tasks = tasks[k::n]
    problems: dict[tuple[str, str, int], dict[int, Attempt]] = {keys[task]: {} for task in tasks}
    handle = None
    if journal is not None:
        mine = set(problems)
        for key, attempts in _by_problem(_journal(Path(journal), _header(settings, rule, catalogs, order, shard))).items():
            if key in mine:
                problems[key].update(attempts)
        handle = open(journal, "a", encoding="utf-8")
    task_of = {keys[task]: task for task in tasks}
    pending: list[tuple[str, int, int, int]] = []
    for key, attempts in problems.items():
        done, _choice, nxt = problem_choice(attempts, settings.max_trials)
        if not done:
            pending.append((*task_of[key], nxt))
    if journal is not None:
        log(f"{len(problems) - len(pending)} problems from the journal, {len(pending)} to judge "
            f"({sum(1 for t in pending if t[3] > 1)} of them resume at a later attempt)")

    def record(entry: Attempt) -> tuple[str, int, int, int] | None:
        """Journal one attempt; the problem's next attempt, or None when it is done."""
        key = (entry["catalog"], entry["eq_id"], int(entry["repeat"]))
        problems[key][int(entry["attempt"])] = entry
        if handle is not None:
            handle.write(json.dumps(entry) + "\n")
            handle.flush()
        done, _choice, nxt = problem_choice(problems[key], settings.max_trials)
        return None if done else (*task_of[key], nxt)

    started = time.perf_counter()
    try:
        if jobs > 1 and pending:
            import multiprocessing
            from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

            # the workers read these when they import numpy and torch, before any initializer runs
            saved = {variable: os.environ.get(variable) for variable in _THREAD_VARIABLES}
            os.environ.update({variable: "1" for variable in _THREAD_VARIABLES})
            with ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context("spawn"),
                                     initializer=_init_worker, initargs=(check_factory, settings)) as pool:
                running = {pool.submit(_attempt_task, task) for task in pending}
                for variable, value in saved.items():
                    if value is None:
                        os.environ.pop(variable, None)
                    else:
                        os.environ[variable] = value
                finished = 0
                while running:
                    done_now, running = wait(running, return_when=FIRST_COMPLETED)
                    for future in done_now:
                        follow = record(future.result())
                        if follow is not None:          # the problem's next attempt, behind the queued ones
                            running.add(pool.submit(_attempt_task, follow))
                        else:
                            finished += 1
                            if finished % 500 == 0:
                                log(f"{finished}/{len(pending)} problems judged ({time.perf_counter() - started:.0f} s)")
        elif pending:
            _init_worker(check_factory, settings, one_thread=False)
            for task in pending:
                step: tuple[str, int, int, int] | None = task
                while step is not None:
                    step = record(_attempt_task(step))
    finally:
        if handle is not None:
            handle.close()
    return {**_document(out, catalogs, order, problems, settings, rule), "wall_seconds": time.perf_counter() - started}


def merge_journals(refs: Sequence[str], out: str | os.PathLike[str], journals: Sequence[str | os.PathLike[str]], *,
                   rule: str, settings: Any = None, ids: Mapping[str, Sequence[str]] | None = None) -> dict[str, Any]:
    """The attempts document of a run split into shards, from the shards' journals; returns the summary.

    ``refs``, ``rule``, ``settings`` and ``ids`` are the run's. Every journal must have been written for them (any shard
    of them), and together they must finish every problem of the plan; an attempt journaled twice keeps its first
    result. Raises ``ValueError`` naming what is missing or does not belong.
    """
    from symbolic_data.draws import DrawsSettings

    settings = settings or DrawsSettings()
    catalogs, order, _tasks, keys = _plan(refs, settings, dict(ids or {}))
    expected = _header(settings, rule, catalogs, order, None)
    attempts: list[Attempt] = []
    for path in journals:
        header, done = _read_journal(Path(path))
        if {key: value for key, value in header.items() if key != "shard"} != expected:
            raise ValueError(f"{path} was written for other catalogs, settings, another rule or another journal layout")
        attempts.extend(done)
    grouped = _by_problem(attempts)
    problems = {key: grouped.get(key, {}) for key in keys.values()}
    missing = [key for key, per in problems.items() if not problem_choice(per, settings.max_trials)[0]]
    if missing:
        raise ValueError(f"{len(missing)} of {len(keys)} problems are not finished in any journal, e.g. {missing[:3]}")
    return _document(out, catalogs, order, problems, settings, rule)


__all__ = ["ACCEPTED", "JOURNAL", "REASONS", "RULE", "TIME", "AttemptTimeout", "OracleCheck", "accept_draws",
           "cpu_time_cap", "judge_attempt", "merge_journals", "oracle_adapter_config", "oracle_check", "problem_choice"]
