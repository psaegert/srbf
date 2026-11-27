"""The benchmark driver: run a model adapter over a problem source, one problem after another.

``Benchmark.run`` is a PLAIN SERIAL loop -- iterate the source, evaluate each problem, append the
record, checkpoint every ``save_every``, honour ``limit``/resume. There is deliberately NO
cross-problem overlap: a benchmark runs the model on one problem at a time so per-problem wall-clock
timing is meaningful and uncontended (the inference-speed pipeline, if any, lives inside the model's
own per-problem inference, not here).

The source and adapter are duck-typed (the source yields problems and may expose ``size_hint`` /
``prepare``; the adapter exposes ``prepare`` and ``evaluate_sample(problem) -> record``), so this
driver is model- and data-layer-agnostic.

After each problem the driver takes the method's candidates out of the record, judges them and
stores them next to the result file (:class:`srbf.candidates.CandidateRecorder`), outside the time
the method is measured for.

Setup before the clock (docs/fairness.md): building the adapter and its ``prepare`` happen before
the first problem, and with ``runner.warmup`` srbf's warm-up problem (:mod:`srbf.warmup`) goes
through the method's fit before it, its answer discarded. A worker-backed adapter repeats the
warm-up in every worker it starts. The seconds of all of it go into ``__meta__['setup']``.
"""
from __future__ import annotations

import os
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

from tqdm import tqdm

from srbf.candidates import CANDIDATE_FIELDS, CandidateRecorder, candidate_store_path
from srbf.store import ResultStore
from srbf.shards import shard_output_path, shard_share
from srbf.paths import substitute_root_path


def check_candidates_setting(value: Any) -> bool | str:
    """``runner.candidates``: true (the default: next to the output file), false (not stored) or a directory."""
    if isinstance(value, bool) or (isinstance(value, str) and value):
        return value
    raise ValueError(f"runner.candidates is true, false or a directory, got {value!r}")


def check_warmup_setting(value: Any) -> bool:
    """``runner.warmup``: whether srbf's warm-up problem runs through the method before the first timed problem (and
    again after every worker restart). False by default; the timing ladder turns it on."""
    if isinstance(value, bool):
        return value
    raise ValueError(f"runner.warmup is true or false, got {value!r}")


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
        config_provenance: Optional[str] = None,
        shard: Optional[tuple[int, int]] = None,
        candidates: bool | str = True,
        engine_name: Optional[str] = None,
        warmup: bool = False,
    ) -> None:
        self.source = source
        # (index, count) when this run is one shard of a unit (`srbf run --shard`); stamped into __meta__.
        self.shard = (int(shard[0]), int(shard[1])) if shard is not None else None
        self.model_adapter = model_adapter
        self.result_store = result_store or ResultStore()
        # Run identity for a sweep/experiment (e.g. {"experiment": ..., "ladder": 256}); display-only.
        self.label = dict(label) if label else {}
        # Who chose the model_adapter configuration (docs/fairness.md); persisted into __meta__.
        # None (direct construction) => run() falls back to a caller-supplied meta label, then to
        # the conservative default; `from_config` always passes the validated config value.
        self.config_provenance = config_provenance
        # The config file this run was built from (set by `from_config` / `runs_from_config`): `run()` records
        # the same provenance for it as `srbf run` does, so a run driven from Python is documented alike.
        self.config_path: Optional[str] = None
        # Resolved run parameters (set by `from_config`; `run()` falls back to these when its own
        # arguments are left None). `completed` => the configured target is already reached and `run()`
        # is a no-op. `total_limit` / `existing_results` are reporting-only (CLI summary).
        self.output_path = output_path
        self.save_every = save_every
        self.limit = limit
        self.completed = completed
        self.total_limit = total_limit
        self.existing_results = existing_results
        # Where every problem's candidates are judged and stored (srbf.candidates): True = next to the output file
        # (<output stem>.candidates), a path = there, False = nowhere. `engine_name` names the judge's engine in the store.
        self.candidates = check_candidates_setting(candidates)
        self.engine_name = engine_name
        # Setup before the clock: srbf's warm-up problem before the first timed problem (srbf.warmup), and the seconds
        # spent building the adapter (set by `from_config`), preparing it and warming it up, for __meta__['setup'].
        self.warmup = check_warmup_setting(warmup)
        self.build_time: Optional[float] = None
        self._setup: dict[str, Any] = {}

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
        shard: tuple[int, int] | None = None,
        build_adapter: bool = True,
    ) -> "Benchmark":
        """Build a ready-to-run `Benchmark` from a unified run config.

        One ordering invariant carries weight: the
        model adapter (which loads the model, possibly onto a GPU) is built LAST -- after resume-load,
        the resolved-total/completed check, and the cheap model-free source build -- so resuming a
        mostly-finished sweep never reloads the model for an already-complete experiment.
        """
        from srbf import config as run_config

        def built(benchmark: "Benchmark") -> "Benchmark":
            benchmark.config_path = str(config) if isinstance(config, (str, os.PathLike)) else None
            return benchmark

        raw_config = run_config.load_run_config(config)
        config_dict = run_config.select_experiment(raw_config, experiment)
        run_cfg = run_config.extract_run_section(config_dict)

        data_cfg = run_cfg.get("data_source")
        if not isinstance(data_cfg, Mapping):
            raise ValueError("run.data_source section is required")
        model_cfg = run_cfg.get("model_adapter")
        if not isinstance(model_cfg, Mapping):
            raise ValueError("run.model_adapter section is required")
        # Validate + capture here, BEFORE the completed short-circuits below: a bad label must
        # surface even for runs that need no work, and the adapter builder is skipped for them.
        config_provenance = run_config.coerce_config_provenance(model_cfg.get("config_provenance"))
        runner_cfg = run_cfg.get("runner", {})

        save_every = save_every_override if save_every_override is not None else runner_cfg.get("save_every")
        save_every = run_config.coerce_optional_int(save_every, "runner.save_every")

        output_path = output_override or runner_cfg.get("output")
        if shard is not None and output_path:
            output_path = shard_output_path(output_path, *shard)  # each shard owns its file (and its resume)
        if save_every is not None and output_path is None:
            raise ValueError("runner.output must be provided when save_every is set")

        resume_flag = runner_cfg.get("resume", True)
        if resume is not None:
            resume_flag = resume
        candidates = check_candidates_setting(runner_cfg.get("candidates", True))
        if isinstance(candidates, str):
            candidates = substitute_root_path(candidates)
        warmup = check_warmup_setting(runner_cfg.get("warmup", False))
        engine_name = model_cfg.get("simplipy_engine")

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
        if shard is not None and limit_value is not None:
            limit_value = shard_share(limit_value, *shard)  # the shard's share of an explicit total

        total_limit: int | None
        remaining: int | None
        target_override: int | None
        if limit_value is not None:
            total_limit = limit_value
            remaining = max(0, limit_value - existing)
            if remaining == 0:
                return built(cls(None, None, result_store=store, output_path=output_path,
                                 save_every=save_every, completed=True, total_limit=total_limit,
                                 existing_results=existing, config_provenance=config_provenance, shard=shard))
            target_override = remaining
        else:
            total_limit = None
            remaining = None
            target_override = None

        source_kwargs = {"shard": shard} if shard is not None else {}  # unsharded calls keep the old signature
        source = run_config.build_catalog_source(data_cfg, target_size=target_override, skip=existing, **source_kwargs)
        draws_identity = getattr(source, "draws_identity", None)
        if callable(draws_identity):
            # rows read from other draws (or drawn afresh) are never continued with these
            from srbf.frozen_draws import check_resume
            check_resume(store.snapshot(), draws_identity())

        size_hint = getattr(source, "size_hint", None)
        pending = size_hint() if callable(size_hint) else None
        if pending is not None and pending <= 0:
            return built(cls(None, None, result_store=store, output_path=output_path, save_every=save_every,
                             completed=True, total_limit=total_limit if total_limit is not None else existing,
                             existing_results=existing, config_provenance=config_provenance, shard=shard))
        if total_limit is None and pending is not None:
            # Frozen catalog with no explicit total: the source's own count is the total. Set an explicit
            # remaining cap too (belt-and-braces with the source's bound). An OPEN generative source has
            # pending=None -> remaining stays None -> the run is unbounded (capped only by --limit).
            total_limit = existing + pending
            remaining = pending

        # LAST: loads the model. `build_adapter=False` resolves a run's progress (existing_results, total_limit,
        # completed) and nothing else: such a Benchmark is read, not run.
        build_started = time.perf_counter()
        adapter = run_config.build_model_adapter(model_cfg) if build_adapter else None
        build_time = time.perf_counter() - build_started if build_adapter else None

        benchmark = cls(source, adapter, result_store=store, output_path=output_path, save_every=save_every,
                        limit=remaining, completed=False, total_limit=total_limit, existing_results=existing,
                        config_provenance=config_provenance, shard=shard, candidates=candidates,
                        engine_name=None if engine_name is None else str(engine_name), warmup=warmup)
        benchmark.build_time = build_time
        return built(benchmark)

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
        shard: tuple[int, int] | None = None,
        build_adapter: bool = True,
    ) -> "list[Benchmark]":
        """Expand a config (``experiments:`` map and/or inline ``!sweep``) into a list of Benchmarks.

        One Benchmark per resolved run: each ``experiments:`` entry (or the top-level run) is expanded
        by :func:`srbf.sweep.resolve_sweeps`, and each resolved single-run config is built via
        :meth:`from_config` (so resume/limit/completed math + model-LAST ordering apply per run). The
        adapter is still built lazily inside ``from_config``, so a completed run never loads its model.
        ``sweep_filter`` keeps only runs whose axis labels match (e.g. ``{"ladder": 256}``).
        """
        from srbf import config as run_config
        from srbf.sweep import resolve_sweeps

        raw = run_config.load_run_config(config)

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
                    shard=shard,
                    build_adapter=build_adapter,
                )
                run_label: dict[str, Any] = {}
                if exp_name is not None:
                    run_label["experiment"] = exp_name
                run_label.update(labels)
                bench.label = run_label
                bench.config_path = str(config) if isinstance(config, (str, os.PathLike)) else None
                benchmarks.append(bench)
        return benchmarks

    def _leave_empty_shard_file(self, meta: Optional[Mapping[str, Any]]) -> None:
        """A shard can hold no problem at all (a one-expression catalog split eight ways). It is complete, not missing:
        it writes its empty file, so that `srbf merge` finds every index of the run."""
        if self.shard is None or self.output_path is None or self.existing_results:
            return
        path = Path(substitute_root_path(self.output_path))
        if path.exists():
            return
        from srbf import config as run_config
        provenance = self.config_provenance or run_config.coerce_config_provenance(dict(meta).get("config_provenance") if meta else None)
        self.result_store.save(path, meta={**(meta or {}), "config_provenance": provenance,
                                           "shard": {"index": self.shard[0], "count": self.shard[1]}})

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
            self._leave_empty_shard_file(meta)
            return self.result_store.snapshot()
        if self.model_adapter is None:
            raise RuntimeError("this Benchmark was built with build_adapter=False: it reports progress and cannot run")
        if meta is None and self.config_path is not None:
            from srbf.provenance import collect_provenance
            meta = {**collect_provenance(self.config_path, None), **self.label}

        # The label declared in the config wins (from_config sets it); a directly-constructed
        # Benchmark falls back to a caller-supplied meta label (validated), then to the
        # conservative default, so every saved __meta__ carries a resolved label.
        provenance = self.config_provenance
        if provenance is None:
            from srbf import config as run_config
            provenance = run_config.coerce_config_provenance(
                dict(meta).get("config_provenance") if meta else None)
        meta = {**(meta or {}), "config_provenance": provenance}
        if self.shard is not None:
            meta["shard"] = {"index": self.shard[0], "count": self.shard[1]}
        # The frozen draws the problems were read from; as a data input, a paired comparison checks both sides read them.
        draws_fn = getattr(self.source, "draws_identity", None)
        draws = draws_fn() if callable(draws_fn) else None
        if draws is not None:
            meta["draws"] = draws
            meta["inputs"] = {**(meta.get("inputs") or {}),
                              "catalog_draws": {"path": draws["path"], "size": draws["size"], "sha256": draws["sha256"]}}

        # Fall back to the parameters `from_config` resolved when the caller leaves them unset.
        limit = limit if limit is not None else self.limit
        save_every = save_every if save_every is not None else self.save_every
        output_path = output_path if output_path is not None else self.output_path

        if save_every is not None and output_path is None:
            raise ValueError("output_path must be provided when save_every is configured")

        resolved_output: Optional[Path] = None
        if output_path is not None:
            resolved_output = Path(substitute_root_path(output_path))

        prepare_started = time.perf_counter()
        prepare_adapter = getattr(self.model_adapter, "prepare", None)
        if callable(prepare_adapter):
            prepare_adapter(data_source=self.source)
        self._setup = {"warmup": self.warmup, "build_time": self.build_time,
                       "prepare_time": time.perf_counter() - prepare_started, "events": []}
        if self.warmup:
            self._warm_up()

        # The RESOLVED ranking (values, not a hash) beside the label: the check whose absence let
        # every earlier flash_ansr run rank at 0.0 while its config said 0.05. Read once the adapter is prepared: a
        # worker reports its own (srbf.subprocess_adapter.SubprocessAdapter.ranking_config) when it is up.
        ranking_fn = getattr(self.model_adapter, "ranking_config", None)
        meta["ranking"] = ranking_fn() if callable(ranking_fn) else None
        # The transformer's inference precision (bf16 / fp32): pools drawn in different precisions never mix.
        precision_fn = getattr(self.model_adapter, "inference_precision", None)
        if callable(precision_fn):
            meta["inference_precision"] = precision_fn()

        # What an out-of-process method reports about itself (its interpreter, package versions, checkpoint)
        # is part of what ran: it is known once the worker is up, and goes beside the rest of the provenance.
        worker_info = getattr(self.model_adapter, "worker_info", None)
        if callable(worker_info):
            meta["worker"] = worker_info() or None
        meta["setup"] = self._setup_meta()

        prepare_source = getattr(self.source, "prepare", None)
        if callable(prepare_source):
            prepare_source(adapter=self.model_adapter)

        recorder = self._candidate_recorder(resolved_output, meta)

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
                self._record_candidates(recorder, problem, result)

                self.result_store.append(result)
                if tracker is not None:
                    tracker.record(result)

                processed += 1

                if progress_bar is not None:
                    progress_bar.update(1)

                if save_every is not None and processed % save_every == 0 and resolved_output is not None:
                    meta["setup"] = self._setup_meta()
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
            # Out-of-process adapters hold a worker interpreter; release it however the loop ended.
            close_adapter = getattr(self.model_adapter, "close", None)
            if callable(close_adapter):
                close_adapter()
            if recorder is not None:
                recorder.close()

        final_snapshot = self.result_store.snapshot()

        if resolved_output is not None:
            meta["setup"] = self._setup_meta()
            self.result_store.save(resolved_output, meta=meta)

        if tracker is not None:
            tracker.print_summary("Final summary")

        return final_snapshot

    def _warm_up(self) -> None:
        """Run srbf's warm-up problem through the method, untimed, its answer discarded. A worker-backed adapter warms
        its worker itself and again after every restart (``warm_up``); an in-process adapter is warmed here, through
        its ``evaluate_sample``. A warm-up the method fails is recorded in ``__meta__['setup']`` and warned about; the
        run goes on, since the problems it then fails are recorded as failures too."""
        from srbf.warmup import run_warmup, warmup_sample

        sample = warmup_sample()
        warm = getattr(self.model_adapter, "warm_up", None)
        if callable(warm):
            warm(sample)
            return
        outcome = run_warmup(self.model_adapter.evaluate_sample, sample)
        self._setup["events"] = [{"reason": "start", **outcome}]
        if outcome["warmup_error"] is not None:
            warnings.warn(f"the warm-up problem failed: {outcome['warmup_error']}", RuntimeWarning)

    def _setup_meta(self) -> dict[str, Any]:
        """``__meta__['setup']``: whether the warm-up ran, the seconds of building and preparing the adapter, and one
        record per start of the method (a worker's every start, with its load, setup and warm-up seconds)."""
        events = getattr(self.model_adapter, "setup_events", None)
        if isinstance(events, list):
            return {**self._setup, "events": [dict(event) for event in events]}
        return {**self._setup, "events": [dict(event) for event in self._setup.get("events", [])]}

    def _candidate_recorder(self, output: Optional[Path], meta: Optional[Mapping[str, Any]]) -> Any:
        """The run's candidate store (``self.candidates``), or None: switched off, no output to put it next to, or an
        adapter without an engine to judge with."""
        if self.candidates is False:
            return None
        if isinstance(self.candidates, str):
            directory: Optional[Path] = Path(substitute_root_path(self.candidates))
        else:
            directory = None if output is None else candidate_store_path(output)
        if directory is None:
            return None
        get_engine = getattr(self.model_adapter, "get_simplipy_engine", None)
        engine = get_engine() if callable(get_engine) else getattr(self.model_adapter, "simplipy_engine", None)
        if engine is None:
            warnings.warn("the adapter has no SimpliPy engine (get_simplipy_engine() or simplipy_engine) to judge "
                          "candidates with: they are not stored", RuntimeWarning)
            return None
        run_meta = {key: (meta or {}).get(key) for key in ("config_provenance", "ranking", "inference_precision", "shard",
                                                           "draws")}
        try:
            return CandidateRecorder(directory, engine=engine, tag=output.stem if output is not None else "run",
                                     engine_name=self.engine_name, run_meta={**run_meta, **self.label})
        except OSError as exc:       # the run goes on without its candidates rather than not at all
            warnings.warn(f"the candidate store {directory} cannot be opened ({exc}): candidates are not stored", RuntimeWarning)
            return None

    @staticmethod
    def _record_candidates(recorder: Any, problem: Any, record: dict[str, Any]) -> None:
        """Judge and store the problem's candidates, or drop them from the record when there is no store. A store
        that fails warns and leaves the reason in the record's ``candidates_error``; the problem's row stays."""
        if recorder is None or getattr(problem, "is_placeholder", False):
            for key in CANDIDATE_FIELDS:
                record.pop(key, None)
            return
        try:
            recorder.record(problem, record)
        except Exception as exc:  # noqa: BLE001 - the method's answer is recorded whatever happens to its candidates
            for key in CANDIDATE_FIELDS:
                record.pop(key, None)
            record["candidates_error"] = f"{type(exc).__name__}: {exc}"
            warnings.warn(f"the candidates of this problem were not stored ({type(exc).__name__}: {exc})", RuntimeWarning)

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
        """A method that raises has failed the problem: the row is a failed prediction, a miss on every rate, like
        a failure the adapter reports itself. It is NOT a placeholder: a placeholder marks a problem that was
        never posed (the catalog could not draw valid points), and summaries leave those out."""
        warnings.warn(
            f"The method raised on this problem ({type(exc).__name__}: {exc}); recorded as a failed prediction.",
            RuntimeWarning,
        )
        record = problem.clone_metadata()
        record["placeholder"] = False
        record["placeholder_reason"] = None
        record["error"] = f"{type(exc).__name__}: {exc}"
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
