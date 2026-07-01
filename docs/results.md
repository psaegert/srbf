# Results & analysis

The **interactive results explorer** lives on the project's results site, separate from these docs:

<p style="margin:1rem 0;">
  <a href="https://psaegert.github.io/srbf/" style="display:inline-block;padding:.6rem 1.1rem;border-radius:8px;background:#4f46e5;color:#fff;font-weight:600;text-decoration:none;">
    Open the results explorer &nbsp;&rarr;
  </a>
</p>

There you can pick an x-axis (inference compute as wall-clock time, measurement noise, or number of
samples), a metric, and a benchmark, then toggle series on and off, with bootstrap medians and 95%
confidence bands. This page documents **how those numbers are produced** and **how to reproduce them**.

## How the numbers are computed

`srbf` separates *running* a benchmark from *analysing* it. A `Benchmark.run()` emits **raw results
only** (per-problem predictions + targets); metrics are a **separate, standardized stage**:

```
Benchmark.run()  ->  raw snapshot     (predictions + targets, per problem)
derive_metrics   ->  + derived metrics (FVU, numeric recovery, F1, ...)
srbf.analysis    ->  aggregate + CI    (bootstrap median over expressions)
export_data      ->  the JSON the results explorer reads
```

`srbf.analysis` (leaderboard / scaling / per-benchmark / distribution helpers) turns a set of runs --
each a raw snapshot tagged with `(model, benchmark, axis, x)` -- into bootstrap-CI'd aggregates.
`export_data(runs, "results_data.json")` writes the tidy records the explorer loads client-side.
Metrics are the strict `is_perfect_fit` numeric recovery, token-level skeleton F1, and the median
`log10` FVU; sources are unseeded, so we report the distribution (bootstrap CI), not a single point.

For a static (matplotlib) rendering instead of the interactive explorer, `build_report(runs, out_dir)`
writes a Markdown page + PNG figures (needs the `srbf[analysis]` extra).

## Reproduce

The **canonical results** come from the sweep configs under `configs/evaluation/` (the model x
inference-compute / noise / support ladders over the shared benchmarks). Run them, then aggregate:

```python
from srbf import Benchmark
from srbf.analysis import RunResult, export_data

runs = []
for (model, benchmark, axis, x, config) in canonical_grid:      # your model x benchmark x sweep grid
    (bench,) = Benchmark.runs_from_config(config)
    runs.append(RunResult(model=model, benchmark=benchmark, axis=axis, scaling=x,
                          version="v24", snapshot=bench.run()))

export_data(runs, "results_data.json",
            engine=bench.model_adapter.get_simplipy_engine())   # -> the results explorer's data
```

The run is a compute job (real models over the benchmarks); the analysis + export stage is cheap and
deterministic given the raw pickles, so the explorer's data regenerates whenever the raw results
change.
