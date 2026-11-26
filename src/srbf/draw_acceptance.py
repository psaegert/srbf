"""Accepting frozen draws: each problem keeps the first seeded attempt on which its own formula can be recognized.

With one draw per problem, a draw on which the true formula cannot be recovered wastes the problem for every method.
symbolic-data draws attempt ``k = 1, 2, ...`` of a problem from its own seed (``symbolic_data.draws.CatalogDraws``):
only what the benchmark leaves to chance changes between attempts (input points, random constants, per-run values),
never the formula or its fixed constants. This module judges each attempt in that order and writes the attempts
document symbolic-data renders with (``symbolic-data render-draws --attempts``). An attempt is accepted when

(a) every target is finite and the FVU is defined on the fitting and on the held-out points (the targets vary);
(b) srbf's oracle (the ``flash_ansr`` adapter in oracle mode: the ground truth is the only candidate, its constants
    refitted with many seeded restarts) reaches numeric recovery on the held-out points (``is_perfect_fit``);
(c) srbf's judge recognizes the refitted formula (``symbolic_recovery``).

The first accepted attempt within the render's ``max_trials`` is kept. When none is accepted, attempt 1 is kept and
the problem is flagged ``formula_inherent``. Formulas that are constant by construction and data problems are not
checked. symbolic-data never imports srbf: the two meet in ``CatalogDraws.draw`` and the attempts document.
"""
from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

#: The rule, as the attempts document and the draws' manifest record it.
RULE = ("the first seeded attempt (1..max_trials) whose targets are finite with a defined FVU on both sides, on which "
        "srbf's oracle (the ground truth with its constants refitted, {n_restarts} seeded restarts) reaches numeric "
        "recovery on the held-out points and srbf's judge recognizes the refitted formula; otherwise attempt 1, flagged "
        "formula_inherent. Constant formulas and data problems are not checked.")

ACCEPTED = "accepted"
#: Why an attempt is not accepted, in the order the rule checks: (a), (b), (c).
REASONS = ("placeholder", "not_finite", "fvu_undefined", "no_prediction", "numeric", "symbolic")


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

    ``adapter`` is a ``flash_ansr`` adapter in oracle mode (any adapter whose ``evaluate_sample`` returns a run's
    record); ``judge_engine`` the engine srbf's judge runs with (``srbf table``'s default, ``acj-5-4-llm``).
    """

    def __init__(self, adapter: Any, judge_engine: Any, engine_name: str | None = None) -> None:
        self.adapter = adapter
        self.judge_engine = judge_engine
        #: the judge engine's name, so that a worker drawing the attempts with the same engine can share it
        self.engine_name = engine_name
        prepare = getattr(adapter, "prepare", None)
        if callable(prepare):
            prepare()

    def __call__(self, problem: Any) -> str:
        from srbf.data_sources import CatalogSource
        from srbf.table import judge_snapshot

        if getattr(problem, "is_placeholder", False):
            return "placeholder"
        if not (np.isfinite(problem.y_support).all() and np.isfinite(problem.y_validation).all()):
            return "not_finite"
        source = CatalogSource(_One(problem))
        source.prepare(adapter=self.adapter)
        sample = next(iter(source))
        record = dict(self.adapter.evaluate_sample(sample).to_mapping())
        judged = judge_snapshot({key: [value] for key, value in record.items()}, engine=self.judge_engine)

        def first(key: str) -> Any:
            values = judged.get(key)
            return None if values is None else values[0]

        if first("fit_quality_undefined_fit") or first("fit_quality_undefined_val"):
            return "fvu_undefined"
        if not record.get("prediction_success"):
            return "no_prediction"
        if not bool(first("numeric_recovery_val")):
            return "numeric"
        if not bool(first("symbolic_recovery")):
            return "symbolic"
        return ACCEPTED


def oracle_adapter_config(config: str | os.PathLike[str], *, model_path: str, n_restarts: int, seed: int,
                          experiment: str | None = None) -> dict[str, Any]:
    """The oracle's ``model_adapter`` block from a reference config (an experiment of the oracle's scaling config), at
    ``n_restarts`` restarts, seeded, on one CPU core, with the checkpoint at ``model_path``."""
    from srbf import config as run_config
    from srbf.sweep import resolve_sweeps

    raw = run_config.load_run_config(str(config))
    experiments = raw.get("experiments") or {}
    chosen = run_config.select_experiment(raw, experiment or (next(iter(experiments)) if experiments else None))
    resolved, _ = resolve_sweeps(chosen)[0]
    adapter = dict(run_config.extract_run_section(resolved)["model_adapter"])
    if (adapter.get("evaluation_config") or {}).get("generation_config", {}).get("method") != "oracle":
        raise ValueError(f"{config}: not an oracle config (evaluation_config.generation_config.method: oracle)")
    adapter.pop("evaluation_overrides", None)
    adapter["evaluation_config"] = {**adapter["evaluation_config"], "n_restarts": int(n_restarts), "device": "cpu"}
    adapter.update(model_path=str(model_path), device="cpu", seed=int(seed), refiner_workers=1, persistent_refine_pool=False)
    return adapter


def oracle_check(config: str, model_path: str, n_restarts: int = 1024, seed: int = 0, judge_engine: str = "acj-5-4-llm",
                 experiment: str | None = None) -> OracleCheck:
    """An :class:`OracleCheck` built from the oracle's reference config (picklable through ``functools.partial``)."""
    import yaml
    from simplipy import SimpliPyEngine

    from srbf import config as run_config

    block = oracle_adapter_config(config, model_path=model_path, n_restarts=n_restarts, seed=seed, experiment=experiment)
    adapter = run_config.build_model_adapter(block)
    # one engine per process where the checkpoint's engine is the judge's (a loaded engine holds its rule set in memory)
    model_config = Path(model_path) / "model.yaml"
    model_engine = (yaml.safe_load(model_config.read_text(encoding="utf-8")) or {}).get("simplipy_engine") if model_config.is_file() else None
    engine = adapter.get_simplipy_engine() if model_engine == judge_engine else SimpliPyEngine.load(judge_engine, install=True)
    return OracleCheck(adapter, engine, engine_name=judge_engine)


# ------------------------------------------------------------------------------------------------ the attempt loop
_WORKER: dict[str, Any] = {}


#: The thread pools of the numerical libraries; a worker gets one core, the pool of workers is the parallelism.
_THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def _init_worker(check_factory: Callable[[], Callable[[Any], str]], settings: Any, one_thread: bool = True) -> None:
    if one_thread:
        try:
            import torch
            torch.set_num_threads(1)
        except ImportError:  # pragma: no cover - the oracle needs torch; a fake check does not
            pass
    _WORKER.clear()
    _WORKER.update(check=check_factory(), settings=settings, draws={})


def _catalog(ref: str) -> Any:
    from symbolic_data.draws import CatalogDraws

    cache: dict[str, Any] = _WORKER["draws"]
    if ref not in cache:
        if len(cache) >= 4:                      # tasks come catalog by catalog: keep the few in use
            cache.pop(next(iter(cache)))
        settings, check = _WORKER["settings"], _WORKER["check"]
        shared = check.judge_engine if getattr(check, "engine_name", None) == settings.simplipy_engine else None
        cache[ref] = CatalogDraws(ref, settings, simplipy_engine=shared)
    return cache[ref]


def judge_attempts(ref: str, row: int, repeat: int) -> tuple[str, str, int, dict[str, Any] | None, dict[str, float]]:
    """Problem ``row`` of catalog ``ref`` in ``repeat``: its accepted attempt, or None when the rule does not apply."""
    from symbolic_data.draws import needs_check

    start, cpu = time.perf_counter(), time.process_time()

    def timing() -> dict[str, float]:
        return {"seconds": time.perf_counter() - start, "cpu_seconds": time.process_time() - cpu}

    draws = _catalog(ref)
    check = _WORKER["check"]
    first = draws.draw(row, repeat, 1)
    if not needs_check(first.kind, first.problem):
        return draws.name, draws.ids[row], repeat, None, timing()
    reasons: Counter[str] = Counter()
    for attempt in range(1, int(draws.settings.max_trials) + 1):
        draw = first if attempt == 1 else draws.draw(row, repeat, attempt)
        verdict = check(draw.problem)
        if verdict == ACCEPTED:
            choice = {"attempt": attempt, "formula_inherent": False, "failed": dict(sorted(reasons.items()))}
            return draws.name, draws.ids[row], repeat, choice, timing()
        reasons[verdict] += 1
    choice = {"attempt": 1, "formula_inherent": True, "failed": dict(sorted(reasons.items()))}
    return draws.name, draws.ids[row], repeat, choice, timing()


def _task(task: tuple[str, int, int]) -> tuple[str, str, int, dict[str, Any] | None, dict[str, float]]:
    return judge_attempts(*task)


def _journal(path: Path, header: dict[str, Any]) -> list[tuple[str, str, int, dict[str, Any] | None, dict[str, float]]]:
    """The results a journal already holds (after checking that it was written for this run), or none."""
    if not path.exists():
        path.write_text(json.dumps(header) + "\n", encoding="utf-8")
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or json.loads(lines[0]) != header:
        raise ValueError(f"{path} was written for other catalogs, settings or another rule; use another journal")
    done = []
    for line in lines[1:]:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:           # a line cut short when the run stopped
            continue
        done.append((entry["catalog"], entry["eq_id"], int(entry["repeat"]), entry["choice"], dict(entry["timing"])))
    return done


def accept_draws(refs: Sequence[str], out: str | os.PathLike[str], *, check_factory: Callable[[], Callable[[Any], str]],
                 rule: str, settings: Any = None, ids: Mapping[str, Sequence[str]] | None = None, jobs: int = 1,
                 journal: str | os.PathLike[str] | None = None, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Judge every checked problem of ``refs`` and write the attempts document to ``out``; returns a summary.

    ``check_factory`` builds the check once per worker process (a picklable callable, such as
    ``functools.partial(oracle_check, ...)``). ``ids`` restricts a catalog (by ref or name) to a frozen subset, as the
    render will. The document lists every catalog, and in it every checked problem with the attempt it keeps, its flag
    and how many attempts failed for which reason. ``journal`` appends each problem's result as it is judged, and a run
    started again on the same journal judges only the problems it does not hold. The summary adds each problem's
    seconds and CPU seconds (not in the document, which a rerun with the same software reproduces).
    """
    from symbolic_data.draws import ATTEMPTS_FORMAT, CatalogDraws, DrawsSettings

    from simplipy import SimpliPyEngine

    settings = settings or DrawsSettings()
    ids = dict(ids or {})
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
    results: list[tuple[str, str, int, dict[str, Any] | None, dict[str, float]]] = []
    handle = None
    if journal is not None:
        header = {"settings": settings.record(), "repeats": list(settings.repeats), "rule": rule,
                  "catalogs": {name: entry["source_sha256"] for name, entry in catalogs.items()},
                  "problems": {name: len(problems) for name, problems in order.items()}}
        results = _journal(Path(journal), header)
        done = {(name, eq_id, repeat) for name, eq_id, repeat, _, _ in results}
        tasks = [task for task in tasks if keys[task] not in done]
        handle = open(journal, "a", encoding="utf-8")
        log(f"{len(done)} problems from the journal, {len(tasks)} to judge")

    def record(result: tuple[str, str, int, dict[str, Any] | None, dict[str, float]]) -> None:
        results.append(result)
        if handle is not None:
            name, eq_id, repeat, choice, spent = result
            handle.write(json.dumps({"catalog": name, "eq_id": eq_id, "repeat": repeat, "choice": choice, "timing": spent}) + "\n")
            handle.flush()

    started = time.perf_counter()
    try:
        if jobs > 1 and tasks:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor, as_completed

            # the workers read these when they import numpy and torch, before any initializer runs
            saved = {variable: os.environ.get(variable) for variable in _THREAD_VARIABLES}
            os.environ.update({variable: "1" for variable in _THREAD_VARIABLES})
            with ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context("spawn"),
                                     initializer=_init_worker, initargs=(check_factory, settings)) as pool:
                futures = [pool.submit(_task, task) for task in tasks]
                for variable, value in saved.items():
                    if value is None:
                        os.environ.pop(variable, None)
                    else:
                        os.environ[variable] = value
                for count, future in enumerate(as_completed(futures), 1):
                    record(future.result())
                    if count % 500 == 0:
                        log(f"{count}/{len(tasks)} problems judged ({time.perf_counter() - started:.0f} s)")
        elif tasks:
            _init_worker(check_factory, settings, one_thread=False)
            for task in tasks:
                record(_task(task))
    finally:
        if handle is not None:
            handle.close()
    seconds: dict[str, dict[str, dict[str, float]]] = {}
    for name, eq_id, repeat, choice, spent in results:
        seconds.setdefault(name, {})[eq_id] = spent
        if choice is not None:
            catalogs[name]["attempts"][str(repeat)][eq_id] = choice
    for name, entry in catalogs.items():                  # the catalog's order, whatever order the workers finished in
        rank = {eq_id: i for i, eq_id in enumerate(order[name])}
        entry["attempts"] = {r: {eq_id: per[eq_id] for eq_id in sorted(per, key=rank.__getitem__)} for r, per in entry["attempts"].items()}
    doc = {"format": ATTEMPTS_FORMAT, "format_version": 1, "settings": settings.record(), "repeats": list(settings.repeats),
           "rule": rule, "catalogs": catalogs}
    Path(out).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    chosen = [c for entry in catalogs.values() for per in entry["attempts"].values() for c in per.values()]
    return {"problems": len(results), "checked": len(chosen), "later_attempt": sum(c["attempt"] > 1 for c in chosen),
            "formula_inherent": sum(c["formula_inherent"] for c in chosen), "seconds": seconds,
            "wall_seconds": time.perf_counter() - started}


__all__ = ["ACCEPTED", "REASONS", "RULE", "OracleCheck", "accept_draws", "judge_attempts", "oracle_adapter_config", "oracle_check"]
