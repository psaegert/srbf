# Changelog

All notable changes to srbf are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

Initial release: the symbolic-regression evaluation framework carved out of
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
