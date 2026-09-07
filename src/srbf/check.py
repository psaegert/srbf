"""``srbf check``: a config, end to end, on a few real problems, before the long run.

Walks the path a run takes -- config, sweep rung, provenance label, output directory, worker and
interpreter, engine, catalog, adapter, fit, derived metrics -- one step at a time, so the failure a
run would hit after hours shows in a minute with the fix beside it. Each step prints ``ok`` or
``FAIL`` with its detail; each fit prints the expression, its time and its validation FVU.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from flash_ansr.utils.paths import substitute_root_path

PROVENANCE_HINT = "set model_adapter.config_provenance to upstream_default, author_blessed or harness_tuned (docs/fairness.md)"


@dataclass
class Step:
    name: str
    ok: bool
    detail: str = ""
    hint: str = ""


@dataclass
class CheckReport:
    steps: list[Step] = field(default_factory=list)
    records: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.steps) and all(step.ok for step in self.steps)

    @property
    def failed(self) -> list[Step]:
        return [step for step in self.steps if not step.ok]


def _short(text: Any, width: int = 200) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 3] + "..."


def _metrics_text(record: Mapping[str, Any], engine: Any) -> str:
    """The derived metrics of one record, through the standard second stage (not hand-rolled)."""
    from srbf.result_processing import derive_metrics

    derived = derive_metrics({key: [value] for key, value in record.items()}, engine=engine)
    parts = []
    fvu = derived.get("fvu_val", [None])[0]
    if fvu is not None:
        parts.append(f"FVU val {float(fvu):.3g}")
    recovered = derived.get("numeric_recovery_val", [None])[0]
    if recovered is not None:
        parts.append("recovered" if recovered else "not recovered")
    symbolic = derived.get("symbolic_recovery", [None])[0]
    if symbolic:
        parts.append("symbolic match")
    return " | ".join(parts)


def check_config(
    config: str,
    *,
    experiment: str | None = None,
    all_experiments: bool = False,
    n_problems: int = 2,
    sweep_filter: Mapping[str, Any] | None = None,
    log: Callable[[str], None] = print,
) -> CheckReport:
    """Check ``config`` end to end; the report says which step failed and how to fix it."""
    report = CheckReport()

    def step(name: str, ok: bool, detail: str = "", hint: str = "") -> None:
        report.steps.append(Step(name, ok, detail, hint))
        log(f"  {'ok  ' if ok else 'FAIL'} {name}: {detail}")
        if hint and not ok:
            log(f"       -> {hint}")

    log(f"srbf check: {config}")
    try:
        from srbf.config import load_run_config

        raw = load_run_config(config)
    except Exception as exc:  # noqa: BLE001 - every failure is a finding here
        step("config", False, f"{type(exc).__name__}: {_short(exc)}",
             "fix the YAML; a suite: config needs one run: template, an experiments: config one block per experiment")
        return report
    experiments = raw.get("experiments")
    names: list[str | None]
    if experiments:
        if experiment is not None:
            names = [experiment]
        elif all_experiments:
            names = list(experiments)
        else:
            names = [next(iter(experiments))]
        which = ", ".join(str(n) for n in names) if len(names) <= 4 else f"all {len(names)}"
        step("config", True, f"{len(experiments)} experiment(s); checking {which}")
    else:
        names = [None]
        step("config", True, "one run")
    for name in names:
        if name is not None:
            log(f"[{name}]")
        _check_experiment(raw, name, n_problems=n_problems, sweep_filter=sweep_filter, step=step, report=report, log=log)
    failed = report.failed
    if failed:
        log(f"FAIL: {len(failed)} of {len(report.steps)} steps failed ({', '.join(s.name for s in failed)})")
    else:
        log(f"PASS: {len(report.steps)} steps ok. Run it with: srbf run -c {config} -v")
    return report


def _check_experiment(raw: Mapping[str, Any], name: str | None, *, n_problems: int, sweep_filter: Mapping[str, Any] | None,
                      step: Callable[..., None], report: CheckReport, log: Callable[[str], None]) -> None:
    from srbf.config import (
        _resolve_worker_ref,
        build_catalog_source,
        build_model_adapter,
        coerce_config_provenance,
        extract_run_section,
        resolve_simplipy_engine,
        select_experiment,
    )
    from srbf.subprocess_adapter import resolve_worker
    from srbf.sweep import resolve_sweeps

    try:
        section = select_experiment(raw, name) if name is not None else dict(raw)
        rungs = resolve_sweeps(section)
    except Exception as exc:  # noqa: BLE001
        step("experiment", False, f"{type(exc).__name__}: {_short(exc)}", "--experiment must name a key under experiments:")
        return
    if sweep_filter:
        rungs = [(cfg, labels) for cfg, labels in rungs
                 if all(str(labels.get(key)) == str(value) for key, value in sweep_filter.items())]
    if not rungs:
        step("sweep", False, "no sweep rung matches the filter", "use --sweep-filter AXIS=VALUE with a value from the config's ladder")
        return
    resolved, labels = rungs[0]
    rung = ", ".join(f"{k}={v}" for k, v in labels.items()) or "no sweep"
    step("sweep", True, f"{len(rungs)} rung(s); checking {rung}")

    run_cfg = extract_run_section(resolved)
    data_cfg = run_cfg.get("data_source")
    model_cfg = run_cfg.get("model_adapter")
    runner_cfg = run_cfg.get("runner") or {}
    if not isinstance(data_cfg, Mapping) or not isinstance(model_cfg, Mapping):
        step("sections", False, "run.data_source and run.model_adapter are required", "see docs/running.md, Config anatomy")
        return
    try:
        step("config_provenance", True, coerce_config_provenance(model_cfg.get("config_provenance")))
    except ValueError as exc:
        step("config_provenance", False, _short(exc), PROVENANCE_HINT)

    output = runner_cfg.get("output")
    if output:
        path = substitute_root_path(str(output))
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
            step("output", True, path)
        except OSError as exc:
            step("output", False, f"cannot create {os.path.dirname(path)}: {exc}", "point runner.output somewhere writable")
    else:
        step("output", True, "none (results stay in memory)")

    kind = str(model_cfg.get("type", "flash_ansr")).lower()
    if kind == "subprocess":
        worker = model_cfg.get("worker")
        if worker is None:
            step("worker", False, "model_adapter.worker is missing", "a worker script path, or a built-in name (example, pysr)")
            return
        try:
            step("worker", True, str(resolve_worker(_resolve_worker_ref(worker))))
        except ValueError as exc:
            step("worker", False, _short(exc), "point model_adapter.worker at the worker file (srbf new <name> writes one)")
            return
        python = model_cfg.get("python")
        interpreter = substitute_root_path(str(python)) if python else sys.executable
        if os.path.isfile(interpreter) and os.access(interpreter, os.X_OK):
            step("python", True, interpreter + ("" if python else " (srbf's own interpreter; set model_adapter.python for the method's environment)"))
        else:
            step("python", False, f"not found: {interpreter}",
                 "create the method's environment there (python -m venv ... && pip install -r requirements.txt) or set model_adapter.python to an interpreter you have")
            return
        try:
            json.dumps(model_cfg.get("options") or {})
            step("options", True, _short(json.dumps(model_cfg.get("options") or {}), 120))
        except (TypeError, ValueError) as exc:
            step("options", False, f"not JSON-serializable: {exc}", "model_adapter.options travels to the worker as JSON: plain numbers, strings, lists, mappings")
            return

    engine = None
    if kind != "flash_ansr":
        started = time.time()
        try:
            engine = resolve_simplipy_engine(model_cfg, adapter_name=kind)
            step("engine", True, f"{model_cfg.get('simplipy_engine')} in {time.time() - started:.1f}s")
        except Exception as exc:  # noqa: BLE001
            step("engine", False, f"{type(exc).__name__}: {_short(exc)}",
                 "use the engine the catalogs are judged with: model_adapter.simplipy_engine: acj-5-4-llm")
            return

    started = time.time()
    samples = []
    try:
        source = build_catalog_source(data_cfg, target_size=n_problems, skip=0)
        for sample in source:
            samples.append(sample)
            if len(samples) >= n_problems:
                break
    except Exception as exc:  # noqa: BLE001
        step("catalog", False, f"{type(exc).__name__}: {_short(exc)}",
             "data_source.catalog is a catalog name (fastsrb, feynman, ...), an HF ref or a path; the first use needs network")
        return
    step("catalog", True, f"{data_cfg.get('catalog')}: {len(samples)} problem(s) in {time.time() - started:.1f}s")
    if not samples:
        step("catalog", False, "the catalog yielded no problem", "check data_source.sampling")
        return

    started = time.time()
    try:
        adapter = build_model_adapter(model_cfg)
        adapter.prepare(data_source=source)
    except Exception as exc:  # noqa: BLE001
        hint = ("the worker did not start: the traceback above is the worker's (it must define fit(); its imports must resolve in model_adapter.python)"
                if kind == "subprocess" else "the adapter could not load the model: check model_path / weights / device")
        step("adapter", False, f"{type(exc).__name__}: {_short(exc, 600)}", hint)
        return
    info = adapter.worker_info() if hasattr(adapter, "worker_info") else {}
    step("adapter", True, f"{kind} ready in {time.time() - started:.1f}s" + (f"; worker info {_short(json.dumps(info), 160)}" if info else ""))
    if engine is None:
        getter = getattr(adapter, "get_simplipy_engine", None)
        engine = getter() if callable(getter) else None

    try:
        for sample in samples:
            eq_id = next((sample.metadata[key] for key in ("benchmark_eq_id", "eq_id", "eval_row_index") if key in sample.metadata), "?")
            started = time.time()
            try:
                record = dict(adapter.evaluate_sample(sample).to_mapping())
            except Exception as exc:  # noqa: BLE001
                step("fit", False, f"{eq_id}: {type(exc).__name__}: {_short(exc, 400)}", "the adapter raised instead of recording the failure; a worker returns {'error': ...} for a problem it cannot fit")
                continue
            wall = time.time() - started
            report.records.append(record)
            if not record.get("prediction_success"):
                step("fit", False, f"{eq_id}: {_short(record.get('error', 'prediction_success is False'), 400)}",
                     "an unparseable expression means the operator vocabulary differs from the engine's (respell_legacy_prefix for the old mult2/pow1_3 tokens); see the worker log for tracebacks")
                continue
            fit_time = record.get("fit_time")
            timing = f"fit {float(fit_time):.2f}s" if fit_time is not None else f"wall {wall:.2f}s"
            try:
                metrics = _metrics_text(record, engine)
            except Exception as exc:  # noqa: BLE001
                step("metrics", False, f"{eq_id}: {type(exc).__name__}: {_short(exc)}", "the derived-metrics stage rejected this record; the expression must parse in the run's engine")
                continue
            step("fit", True, f"{eq_id}: {_short(record.get('predicted_expression'), 70)} | {timing}" + (f" | {metrics}" if metrics else ""))
    finally:
        close = getattr(adapter, "close", None)
        if callable(close):
            close()


__all__ = ["CheckReport", "Step", "check_config"]
