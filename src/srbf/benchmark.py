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

from srbf.eval.result_store import ResultStore
from flash_ansr.utils.paths import substitute_root_path


class Benchmark:
    """Drive a serial evaluation loop over a problem source with a model adapter."""

    def __init__(
        self,
        source: Any,
        model_adapter: Any,
        *,
        result_store: ResultStore | None = None,
    ) -> None:
        self.source = source
        self.model_adapter = model_adapter
        self.result_store = result_store or ResultStore()

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
