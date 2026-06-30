"""The benchmark driver: run a model adapter over a problem source, one problem after another.

``Benchmark.run`` is a PLAIN SERIAL loop -- iterate the source, evaluate each problem, append the
record, checkpoint every ``save_every``, honour ``limit``/resume. There is deliberately NO
cross-problem overlap: a benchmark runs the model on one problem at a time so per-problem wall-clock
timing is meaningful and uncontended (the inference-speed pipeline, if any, lives inside the model's
own per-problem inference, not here).

The source and adapter are duck-typed (the source yields problems and may expose ``size_hint`` /
``prepare``; the adapter exposes ``prepare`` and ``evaluate_sample(problem) -> record``), so this
driver is model- and data-layer-agnostic.
"""
from __future__ import annotations

import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

from tqdm import tqdm

from srbf.store import ResultStore
from flash_ansr.utils.paths import substitute_root_path


class Benchmark:
    """Drive a serial evaluation loop over a problem source with a model adapter."""

    def __init__(
        self,
        source: Any,
        model_adapter: Any,
        *,
        result_store: ResultStore | None = None,
        output_path: Optional[str] = None,
        save_every: Optional[int] = None,
        limit: Optional[int] = None,
        completed: bool = False,
        total_limit: Optional[int] = None,
        existing_results: int = 0,
        label: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.source = source
        self.model_adapter = model_adapter
        self.result_store = result_store or ResultStore()
        # Run identity for a sweep/experiment (e.g. {"experiment": ..., "ladder": 256}); display-only.
        self.label = dict(label) if label else {}
        # Resolved run parameters (set by `from_config`; `run()` falls back to these when its own
        # arguments are left None). `completed` => the configured target is already reached and `run()`
        # is a no-op. `total_limit` / `existing_results` are reporting-only (CLI summary).
        self.output_path = output_path
        self.save_every = save_every
        self.limit = limit
        self.completed = completed
        self.total_limit = total_limit
        self.existing_results = existing_results

    @classmethod
    def from_config(
        cls,
        config: "str | Mapping[str, Any]",
        *,
        limit_override: int | None = None,
        output_override: str | None = None,
        save_every_override: int | None = None,
        resume: bool | None = None,
        experiment: str | None = None,
    ) -> "Benchmark":
        """Build a ready-to-run `Benchmark` from a unified run config.

        Mirrors the old ``build_evaluation_run`` flow, with one load-bearing ordering invariant: the
        model adapter (which loads the model, possibly onto a GPU) is built LAST -- after resume-load,
        the resolved-total/completed check, and the cheap model-free source build -- so resuming a
        mostly-finished sweep never reloads the model for an already-complete experiment.
        """
        from srbf import config as run_config

        raw_config = run_config.load_config(config) if isinstance(config, str) else dict(config)
        config_dict = run_config.select_experiment(raw_config, experiment)
        run_cfg = run_config.extract_run_section(config_dict)

        data_cfg = run_cfg.get("data_source")
        if not isinstance(data_cfg, Mapping):
            raise KeyError("run.data_source section is required")
        model_cfg = run_cfg.get("model_adapter")
        if not isinstance(model_cfg, Mapping):
            raise KeyError("run.model_adapter section is required")
        runner_cfg = run_cfg.get("runner", {})

        save_every = save_every_override if save_every_override is not None else runner_cfg.get("save_every")
        save_every = run_config.coerce_optional_int(save_every, "runner.save_every")

        output_path = output_override or runner_cfg.get("output")
        if save_every is not None and output_path is None:
            raise ValueError("runner.output must be provided when save_every is set")

        resume_flag = runner_cfg.get("resume", True)
        if resume is not None:
            resume_flag = resume

        initial_results = None
        if resume_flag and output_path:
            initial_results = run_config.load_existing_results(output_path)
        store = ResultStore(initial_results)
        existing = store.size

        # Explicit total: CLI override, runner.limit, or data_source.target_size. When none is set, the
        # source's own size_hint bounds the run (finite for a frozen catalog; unbounded otherwise).
        limit_value = limit_override if limit_override is not None else runner_cfg.get("limit")
        limit_value = run_config.coerce_optional_int(limit_value, "runner.limit")
        if limit_value is None:
            limit_value = run_config.coerce_optional_int(data_cfg.get("target_size"), "data_source.target_size")

        total_limit: int | None
        remaining: int | None
        target_override: int | None
        if limit_value is not None:
            total_limit = limit_value
            remaining = max(0, limit_value - existing)
            if remaining == 0:
                return cls(None, None, result_store=store, output_path=output_path,
                           save_every=save_every, completed=True, total_limit=total_limit,
                           existing_results=existing)
            target_override = remaining
        else:
            total_limit = None
            remaining = None
            target_override = None

        source = run_config.build_catalog_source(data_cfg, target_size=target_override, skip=existing)

        size_hint = getattr(source, "size_hint", None)
        pending = size_hint() if callable(size_hint) else None
        if pending is not None and pending <= 0:
            return cls(None, None, result_store=store, output_path=output_path, save_every=save_every,
                       completed=True, total_limit=total_limit if total_limit is not None else existing,
                       existing_results=existing)
        if total_limit is None and pending is not None:
            # Frozen catalog with no explicit total: the source's own count is the total. Set an explicit
            # remaining cap too (belt-and-braces with the source's bound). An OPEN generative source has
            # pending=None -> remaining stays None -> the run is unbounded (capped only by --limit).
            total_limit = existing + pending
            remaining = pending

        adapter = run_config.build_model_adapter(model_cfg)  # LAST: loads the model

        return cls(source, adapter, result_store=store, output_path=output_path, save_every=save_every,
                   limit=remaining, completed=False, total_limit=total_limit, existing_results=existing)

    @classmethod
    def runs_from_config(
        cls,
        config: "str | Mapping[str, Any]",
        *,
        limit_override: int | None = None,
        output_override: str | None = None,
        save_every_override: int | None = None,
        resume: bool | None = None,
        experiment: str | None = None,
        sweep_filter: Mapping[str, Any] | None = None,
    ) -> "list[Benchmark]":
        """Expand a config (``experiments:`` map and/or inline ``!sweep``) into a list of Benchmarks.

        One Benchmark per resolved run: each ``experiments:`` entry (or the top-level run) is expanded
        by :func:`srbf.sweep.resolve_sweeps`, and each resolved single-run config is built via
        :meth:`from_config` (so resume/limit/completed math + model-LAST ordering apply per run). The
        adapter is still built lazily inside ``from_config``, so a completed run never loads its model.
        ``sweep_filter`` keeps only runs whose axis labels match (e.g. ``{"ladder": 256}``).
        """
        from srbf import config as run_config
        from srbf.sweep import register_sweep_yaml, resolve_sweeps

        register_sweep_yaml()
        raw = run_config.load_config(config) if isinstance(config, str) else dict(config)

        experiments = raw.get("experiments") if isinstance(raw, Mapping) else None
        base: list[tuple[str | None, Mapping[str, Any]]]
        if experiments:
            names = [experiment] if experiment is not None else list(experiments.keys())
            base = [(name, run_config.select_experiment(raw, name)) for name in names]
        else:
            base = [(None, dict(raw))]

        benchmarks: list[Benchmark] = []
        for exp_name, exp_cfg in base:
            for resolved, labels in resolve_sweeps(exp_cfg):
                if sweep_filter and not all(str(labels.get(k)) == str(v) for k, v in sweep_filter.items()):
                    continue
                bench = cls.from_config(
                    resolved,
                    limit_override=limit_override,
                    output_override=output_override,
                    save_every_override=save_every_override,
                    resume=resume,
                    experiment=None,
                )
                run_label: dict[str, Any] = {}
                if exp_name is not None:
                    run_label["experiment"] = exp_name
                run_label.update(labels)
                bench.label = run_label
                benchmarks.append(bench)
        return benchmarks

    def run(
        self,
        *,
        limit: Optional[int] = None,
        save_every: Optional[int] = None,
        output_path: Optional[str] = None,
        verbose: bool = True,
        progress: bool = True,
        log_placeholders: bool = True,
        summary_interval: Optional[int] = 50,
        meta: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, list[Any]]:
        """Execute the serial evaluation loop and return accumulated results."""

        # A `from_config` Benchmark whose target is already reached: nothing to do.
        if self.completed:
            if verbose:
                target = self.total_limit if self.total_limit is not None else "configured"
                print(f"Evaluation already completed ({self.existing_results}/{target}). Nothing to do.")
            return self.result_store.snapshot()

        # Fall back to the parameters `from_config` resolved when the caller leaves them unset.
        limit = limit if limit is not None else self.limit
        save_every = save_every if save_every is not None else self.save_every
        output_path = output_path if output_path is not None else self.output_path

        if save_every is not None and output_path is None:
            raise ValueError("output_path must be provided when save_every is configured")

        resolved_output: Optional[Path] = None
        if output_path is not None:
            resolved_output = Path(substitute_root_path(output_path))

        prepare_adapter = getattr(self.model_adapter, "prepare", None)
        if callable(prepare_adapter):
            prepare_adapter(data_source=self.source)

        prepare_source = getattr(self.source, "prepare", None)
        if callable(prepare_source):
            prepare_source(adapter=self.model_adapter)

        existing_results = self.result_store.size
        pending_target = limit if limit is not None else getattr(self.source, "size_hint", lambda: None)()
        if pending_target is not None:
            pending_target = max(0, int(pending_target))
        overall_target = None if pending_target is None else existing_results + pending_target

        iterator: Iterable[Any] = self.source
        processed = 0

        progress_bar = None
        if progress and verbose:
            progress_bar = tqdm(total=pending_target, desc="Evaluating", smoothing=0.0)

        tracker: _ProgressTracker | None = None
        if log_placeholders or (summary_interval is not None and summary_interval > 0):
            tracker = _ProgressTracker(
                result_store=self.result_store,
                total_target=overall_target,
                logger=lambda message: self._log_message(message, progress_bar),
                log_placeholders=log_placeholders,
            )
            tracker.print_summary("Starting state")

        try:
            for problem in iterator:
                if limit is not None and processed >= limit:
                    break

                try:
                    if getattr(problem, "is_placeholder", False):
                        result = self._build_placeholder_record(problem)
                    else:
                        result = self._evaluate(problem)
                except Exception as exc:  # pragma: no cover - defensive guard
                    result = self._handle_exception(problem, exc)

                self.result_store.append(result)
                if tracker is not None:
                    tracker.record(result)

                processed += 1

                if progress_bar is not None:
                    progress_bar.update(1)

                if save_every is not None and processed % save_every == 0 and resolved_output is not None:
                    self.result_store.save(resolved_output, meta=meta)

                if (
                    tracker is not None
                    and summary_interval is not None
                    and summary_interval > 0
                    and processed % summary_interval == 0
                ):
                    tracker.print_summary(
                        f"Progress after {processed} new problems (overall {self.result_store.size})"
                    )
        finally:
            if progress_bar is not None:
                progress_bar.close()

        final_snapshot = self.result_store.snapshot()

        if resolved_output is not None:
            self.result_store.save(resolved_output, meta=meta)

        if tracker is not None:
            tracker.print_summary("Final summary")

        return final_snapshot

    def _evaluate(self, problem: Any) -> dict[str, Any]:
        result = self.model_adapter.evaluate_sample(problem)
        mapping = result.to_mapping() if hasattr(result, "to_mapping") else result
        if not isinstance(mapping, dict):
            raise TypeError("Model adapters must return dict-like results")
        return mapping

    def _build_placeholder_record(self, problem: Any) -> dict[str, Any]:
        record = problem.clone_metadata()
        record["placeholder"] = True
        if problem.placeholder_reason is not None:
            record["placeholder_reason"] = problem.placeholder_reason
            record.setdefault("error", problem.placeholder_reason)
        else:
            record.setdefault("placeholder_reason", None)
        record.setdefault("prediction_success", False)
        return record

    def _handle_exception(self, problem: Any, exc: Exception) -> dict[str, Any]:
        warnings.warn(
            f"Problem evaluation failed with an unexpected error: {exc}. Recording placeholder result.",
            RuntimeWarning,
        )
        record = problem.clone_metadata()
        record["placeholder"] = True
        record.setdefault("placeholder_reason", "adapter_exception")
        record["error"] = str(exc)
        record["prediction_success"] = False
        return record

    @staticmethod
    def _log_message(message: str, progress_bar: tqdm | None) -> None:
        if progress_bar is not None:
            progress_bar.write(message)
        else:
            print(message, flush=True)


class _ProgressTracker:
    """Track placeholder statistics and emit human-readable summaries."""

    def __init__(
        self,
        *,
        result_store: ResultStore,
        total_target: int | None,
        logger: Callable[[str], None],
        log_placeholders: bool,
    ) -> None:
        stats = result_store.statistics()
        self.result_store = result_store
        self.total_target = total_target
        self.logger = logger
        self.log_placeholders = log_placeholders
        self.placeholder_count = stats["placeholders"]
        self.valid_count = stats["valid"]
        self.placeholder_reasons: Counter[str] = Counter(stats.get("placeholder_reasons", {}))

    def record(self, record: Mapping[str, Any]) -> None:
        if record.get("placeholder"):
            reason = str(record.get("placeholder_reason") or "unspecified")
            self.placeholder_count += 1
            self.placeholder_reasons[reason] += 1
            if self.log_placeholders:
                self._log_placeholder_event(reason, record)
        else:
            self.valid_count += 1

    def pending(self) -> int | None:
        if self.total_target is None:
            return None
        return max(0, self.total_target - self.result_store.size)

    def print_summary(self, label: str) -> None:
        self.logger(f"[Benchmark] {label}: {self._format_summary()}")

    def _format_summary(self) -> str:
        total = self.result_store.size
        pending = self.pending()
        parts = [
            f"total={total}",
            f"valid={self.valid_count}",
            f"placeholders={self.placeholder_count}",
        ]
        if pending is not None:
            parts.append(f"remaining={pending}")
        if self.placeholder_reasons:
            breakdown = ", ".join(
                f"{reason}:{count}" for reason, count in sorted(self.placeholder_reasons.items())
            )
            parts.append(f"reasons={breakdown}")
        return "; ".join(parts)

    def _log_placeholder_event(self, reason: str, record: Mapping[str, Any]) -> None:
        location = (
            record.get("benchmark_eq_id")
            or record.get("skeleton_hash")
            or record.get("skeleton")
            or record.get("sample_id")
        )
        location_text = f", location={location}" if location is not None else ""
        pending = self.pending()
        pending_text = f", remaining={pending}" if pending is not None else ""
        self.logger(
            "[Benchmark] Placeholder #{count} recorded (reason={reason}{location}). "
            "Valid={valid}, placeholders={count}{pending}.".format(
                count=self.placeholder_count,
                reason=reason,
                location=location_text,
                valid=self.valid_count,
                pending=pending_text,
            )
        )


__all__ = ["Benchmark"]
