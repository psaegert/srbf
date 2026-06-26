# Changelog

All notable changes to srbf are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
