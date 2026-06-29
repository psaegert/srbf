# Changelog

All notable changes to srbf are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
