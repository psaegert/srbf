# Changelog

All notable changes to srbf are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.2] - 2026-07-02

### Changed
- **PySR runs at its upstream default `maxsize` again** (the 0.6.1 override is removed). Policy
  decision: baselines run at their library defaults — a method's default hyperparameters are part
  of the method, and a default that limits it is a property of that method, not srbf's to correct.
  The 0.6.1 audit numbers stand as documentation (at PySR's `maxsize=20`, 23/120 FastSRB and
  743/1000 v23-val ground truths are not representable; `scripts/audit_pysr_maxsize.py`,
  docs/models.md). The optional `maxsize` key in the `model_adapter` block remains for side
  experiments only; it is no longer set by default. The warmup fit and the seedable bootstrap
  from 0.6.1 are unchanged.

## [0.6.1] - 2026-07-02

Fairness + reproducibility quick wins (improvement-plan WP0).

### Fixed
- **PySR gets an explicit complexity budget (`maxsize=45`).** PySR's own default (`maxsize=20`)
  makes 23/120 FastSRB and 743/1000 v23-val ground truths structurally inexpressible under the
  adapter vocabulary (largest ground truth = 40 nodes; v23-val median 25), so runs at the library
  default measured a representation handicap rather than search quality. The audit lives in
  `scripts/audit_pysr_maxsize.py`; PySR results produced before 0.6.1 should be treated as lower
  bounds. `maxsize` is overridable per `model_adapter` block.
- **PySR timing no longer carries the Julia precompile outlier.** `PySRAdapter.prepare()` runs a
  warmup fit on a throwaway model (`warmup: true` by default), so problem 0's `fit_time` starts
  warm. The other built-in adapters already load their models in `prepare()`.

### Added
- **Seedable bootstrap.** `bootstrapped_metric_ci` accepts `rng` (`np.random.Generator` | int
  seed | `None` for fresh entropy); `bootstrap_report` defaults to `rng=0`, making reports
  bit-reproducible by default (pass `rng=None` for the previous unseeded behaviour). Interpret
  results by the confidence interval either way.

## [0.6.0] - 2026-07-01

Post-release audit round (deferred tiers C + D): baseline de-duplication, fail-fast adapter
validation, and reporting/metric performance. Re-pinned to the coordinated `flash-ansr>=0.10` /
`symbolic-data>=0.10` release.

### Added
- **`srbf.derive_metrics(snapshot, *, engine=None, operator_arity=None, simplify_fn=None)`** -- the
  clean standardized second stage. A `Benchmark.run()` emits RAW results only; `derive_metrics`
  turns one raw snapshot into a NEW snapshot with the derived metric columns (FVU, numeric/symbolic
  recovery, F1, edit distances, ...) added, without mutating the input, and composes directly with
  `bootstrap_report` / `draw_distribution`. It hides the nested-dict wrapper that the lower-level
  `compute_derived_metrics` requires. Metrics are never computed inside the run (two-stage by design).
- `compute_derived_metrics` is also exported now (the in-place primitive `derive_metrics` wraps).
- **`srbf.analysis` -- the standardized results-page layer.** Turns raw run snapshots (tagged with
  model / benchmark / scaling coordinate) into the four standardized views: a model/baseline
  `leaderboard`, `scaling` curves, a `per_benchmark` breakdown, and per-expression `distribution`
  plots (all bootstrap-CI'd, honoring the unseeded-sources policy). `build_report(runs, out_dir, ...)`
  renders them to a Markdown page + PNG figures for the docs / github.io site. Figures live behind a
  new optional `[analysis]` extra (`pip install 'srbf[analysis]'`, matplotlib); tables/leaderboards
  need no extra.
- **`srbf.analysis.export_data(runs, path)`** + `RunResult.axis` / `RunResult.version`: export tidy
  aggregated records (per `series x benchmark x axis x x-value`, each with a metric's bootstrap
  median + CI and a provenance `version`) as JSON for the **interactive results page** -- a
  client-side Plotly explorer (pick x-axis / metric / benchmark / series, with CI bands). One dataset
  can carry several sweeps (compute / noise / n_support, ...) via the `axis` field.

### Changed
- **Baselines de-duplicated onto a shared `_RefiningBaselineModel` base.** `BruteForceModel` and
  `LampleChartonModel` now share the catalog handling, X/y coercion, per-candidate refine/score/build,
  and the fit loop (~250 duplicated lines removed); each subclass keeps only its skeleton source
  (exhaustive generator vs catalog sample) and extra `__init__` knobs. Public classes, constructors,
  and results are unchanged (golden-verified identical `_results` records + ordering).
- **`FlashANSRAdapter` validates the `complexity` mode at construction.** An unknown complexity string
  now raises `ValueError` immediately instead of failing lazily on the first problem (after the slow
  model load). Valid: `"none"`, `"ground_truth"`, or an int / float / list.
- **`NeSymReSAdapter` gained a `debug` flag (default `False`); its per-sample support/validation FVU
  print is now gated by it** (was printed unconditionally for every evaluated problem), matching the
  `E2EAdapter` convention.
- Re-pinned `flash-ansr>=0.10,<1.0` and `symbolic-data>=0.10` (coordinated family release).

### Performance
- **`bootstrapped_metric_ci` vectorizes** reducers that accept an `axis` kwarg (the default `np.nanmean`
  and friends), reducing all `n` resamples in one call instead of an `n`-iteration Python loop
  (`np.apply_along_axis`); falls back to the per-resample loop for metrics without `axis` support.
- **`bootstrap_report` shares a single valid-row scan** for both the distribution and its `n_rows`
  count (was scanned twice).
- **Variable-level F1 is derived from one precision + one recall computation** per row (was recomputing
  both a second time via `f1_score`); bit-identical to the previous values (same torch formula + NaN->0).

## [0.5.5] - 2026-07-01

### Changed
- Baselines (`LampleChartonModel`, `BruteForceModel`) read the fit attempts via the new public
  `Refiner.all_constants_values` property instead of the private `_all_constants_values`. Re-pinned
  `flash-ansr>=0.9.5` (which adds the property).

## [0.5.4] - 2026-07-01

### Fixed
- **Run provenance hashes the data catalog.** `provenance._resolve_inputs` now hashes a local catalog
  artifact (a saved catalog file or directory's `catalog.yaml`/`catalog.npz`), dropping the dead
  pre-0.5 `benchmark_path`/`dataset`/`skeleton_list` branches; a bare `name[@version]` / HF ref is
  captured verbatim by `config_sha`. Previously dataset provenance silently recorded nothing about the
  data source.

## [0.5.3] - 2026-07-01

Post-release audit cleanup + a robustness fix.

### Fixed
- **`LampleChartonModel._sample_skeletons` bounds its sampling loop** (`max(1000, samples*100)`
  attempts, then returns what it has with a warning) so it can no longer hang when the pool
  persistently fails to sample or its unique skeletons are exhausted.
- Baselines call the public `flash_ansr.scoring.compute_fvu` instead of reaching into a private
  method; `bootstrapped_metric_ci` return annotation + docstring corrected to the `(median, lower,
  upper)` tuple; clearer errors for missing/mistyped benchmark-config sections; dropped a dead
  `axis_of` parameter in `_substitute`; banned-term / stale-docstring cleanups.

## [0.5.2] - 2026-07-01

Post-release audit fixes (no API change).

### Fixed
- `BruteForceModel.fit` and `LampleChartonModel.fit` wrap their fit loop in `np.errstate(...)`, so the
  global numpy error state is restored even if the loop raises a non-`ConvergenceError` exception
  (previously a raise leaked `ignore` process-wide, silently suppressing later overflow/divide/invalid
  warnings).
- `fvu()` returns `inf` for empty/degenerate inputs (per its documented "invalid -> inf" contract)
  instead of raising `ValueError` on a zero-size reduction, and now accepts list / scalar `y_pred`.
- The `!sweep` YAML tag is registered on `import srbf`, so loading a sweep config (e.g. via the shared
  flash-ansr config loader) no longer raises `ConstructorError` without a prior `register_sweep_yaml()`.

## [0.5.1] - 2026-06-30

Adds the config-sweep + reporting layer (folded in from the planned 0.5.x scope) and finishes the
config/docs migration. Re-pinned `symbolic-data>=0.9`.

### Added
- **Inline `!sweep` config cross-products** (`srbf.Sweep` / `register_sweep_yaml` / `resolve_sweeps`):
  `!sweep [..]` = an anonymous grid axis; `!sweep {name: L, values: [..]}` = a named axis (co-named
  sweeps zip element-wise). `Benchmark.runs_from_config` expands an `experiments:` map and/or `!sweep`
  into per-run Benchmarks; `srbf run --sweep-filter AXIS=VALUE` selects runs.
- **Multi-draw reporting** `bootstrap_report` / `draw_distribution`: group per-problem metrics by
  `benchmark_eq_id` and bootstrap a CI (the no-seeding reproducibility story for sampling sources).

### Changed
- Migrated all `configs/evaluation/` configs to the catalog schema + `!sweep` (the 19-rung "choices
  ladder" collapses to one zipped axis); baselines resolve their catalog by name via
  `symbolic_data.build_catalog`; docs/README rewritten to the catalog/Benchmark/`!sweep` surface.
- `test_scaling_configs.py` upgraded from a key-presence check to a real catalog-schema gate.

## [0.5.0] - 2026-06-30

The data-layer redesign: srbf consumes `symbolic_data`'s catalog/`ProblemSource` API and
flash-ansr 0.9's public inference API, and the `eval/` engine layer collapses into a single
`Benchmark` driver. Breaking. Re-pinned `flash-ansr~=0.9`, `symbolic-data>=0.8`.

### Changed
- **Data source** is now always a `symbolic_data` catalog. The old `SkeletonDatasetSource` +
  `FastSRBSource` (and the `type: skeleton_dataset` / `type: fastsrb` config split) are replaced by
  one `CatalogSource` wrapping a `symbolic_data.ProblemSource`. The `data_source` config is
  `{catalog: <name/ref>, sampling: {n_support, n_validation, noise, problems_per_expression},
  target_size}`; the frozen sha-pinned `v23-val` catalog is the drift-safe validation set (the old
  val100 skeleton-pin machinery is gone). Non-flash_ansr adapters now require an explicit
  `model_adapter.simplipy_engine` (no dataset to borrow one from).
- **`FlashANSRAdapter`** is a thin mapper over `FlashANSR.infer()` -> `InferenceResult` (best
  candidate + the full classified `CandidateLedger`); no more `model._results` / `nth_best_beam` /
  generate-refine-phase scraping. The candidate-ledger JOIN lives in flash-ansr now; srbf only
  persists it (`CandidateStoreWriter`).
- **`Benchmark`** (`srbf.Benchmark`) replaces the `Evaluation*` driver surface. `Benchmark.run` is a
  plain serial loop (no cross-problem overlap; per-problem timing stays uncontended).
  `Benchmark.from_config` absorbs `build_evaluation_run` (resume/limit/completed math), building the
  model adapter last so a resumed sweep never reloads the model for a finished experiment.
- Baselines moved onto `symbolic_data.LampleChartonCatalog`: `SkeletonPoolModel` ->
  `LampleChartonModel` (adapter type `skeleton_pool` -> `lample_charton`; param `skeleton_pool` ->
  `catalog`).
- Relocated the package: dropped the `srbf/eval/` subpackage (modules moved to top-level `srbf/`;
  `result_store` -> `store`). `import srbf.eval.X` -> `import srbf.X`.

### Removed
- `EvaluationEngine` / `OverlappedEvaluationEngine` (the cross-problem overlap), `Evaluation`,
  `EvaluationRunPlan` / `build_evaluation_run` (-> `Benchmark.from_config`), `run_config.py`,
  `data_sources.SkeletonDatasetSource` / `FastSRBSource`, and srbf's duplicate
  `build_candidate_ledger` + `FIT_*` (imported from `flash_ansr.inference` now). The
  `srbf.benchmarks` re-export shim (its `symbolic_data.FastSRBBenchmark` source was removed upstream;
  FastSRB is the `fastsrb` catalog).

### Deferred (to 0.5.1)
- Inline `!sweep` config cross-products + multi-draw bootstrap reporting (the `experiments:` map
  still works), the columnar (Parquet) result store + typed `Result`/`ResultCollection` projection
  (the pickle `ResultStore` is unchanged behind its seam), and migrating the `configs/evaluation/`
  scaling configs to the new catalog schema (cheaper bundled with the `!sweep` collapse). Configs are
  not shipped in the wheel.

## [0.4.0] - 2026-06-29

### Changed
- Dropped srbf's local `FastSRBBenchmark` fork; `srbf.benchmarks` now re-exports
  `symbolic_data.FastSRBBenchmark` (single source of the FastSRB sampler), and the eval consumers
  import it directly from `symbolic_data`. Re-pinned `flash-ansr~=0.8`.

### Removed
- The `srbf.benchmarks.fastsrb` module (minor breaking for deep imports; the class remains importable
  as `srbf.benchmarks.FastSRBBenchmark` / `symbolic_data.FastSRBBenchmark`).

## [0.3.0] - 2026-06-28

### Changed
- `SkeletonDatasetSource` delegates per-skeleton dataset sampling to
  `symbolic_data.sample_from_skeleton` (behavior-identical, byte-verified), and a flaky
  skeleton-pin test was made deterministic.

## [0.2.0] - 2026-06-28

### Changed
- Imports updated to the carved `symbolic_data` / `simplipy` packages (flash-ansr 0.7 carve).

## [0.1.0] - 2026-06-26

Initial release: the Symbolic Regression Benchmark Framework carved out of
[flash-ansr](https://github.com/psaegert/flash-ansr) (flash-ansr 0.6 split).

### Added
- Evaluation engine (`srbf.eval.engine`) and model-agnostic protocols (`srbf.eval.core`:
  `EvaluationModelAdapter`, `EvaluationDataSource`, `EvaluationSample`, `EvaluationResult`).
- `srbf run` CLI: run an evaluation from a unified config.
- Built-in model adapters: `flash_ansr`, `pysr`, `nesymres`, `e2e`, `skeleton_pool`, `brute_force`.
- Benchmarks (FastSRB) and metrics (FVU, symbolic recovery, token/edit distance).
- Baseline provisioning recipes: `scripts/patch_{nesymres,symbolicregression,typing_io}.py`.
- Documentation under `docs/`: running, benchmarks, models, and the adapter contribution guide.

### Notes
- Depends one-way on `flash-ansr` (>=0.6) and `simplipy`; `flash-ansr` never imports `srbf`.
- Status: cleanly-carved eval, not yet a general framework. A plugin `register_adapter()` entry-point
  and raw `(X, y)` dataset ingestion are planned follow-ons.
