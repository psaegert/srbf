# `srbf.eval`

The evaluation engine, model adapters, data sources, and metrics.

User documentation lives at the repository root:

- [Running evaluations](../../../docs/running.md) — the `srbf run` CLI and config anatomy
- [Benchmarks & datasets](../../../docs/benchmarks.md) — FastSRB and building skeleton pools
- [Models & provisioning](../../../docs/models.md) — installing/patching the built-in models
- [Adding your model](../../../docs/adapters.md) — the adapter protocol + the PR contribution flow

Key modules: `core.py` (the `EvaluationModelAdapter` / `EvaluationDataSource` protocols +
`EvaluationSample` / `EvaluationResult`), `engine.py` (the runner), `run_config.py` (config parsing +
the `_ADAPTER_REGISTRY`), `model_adapters.py` (the built-in adapters), `data_sources.py`,
`metrics/` (FVU, symbolic recovery, edit distance, ...), and `result_processing.py`.
