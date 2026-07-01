# Results

An **interactive explorer** of the symbolic-regression benchmark results. Pick an **x-axis**
(inference compute as wall-clock time, measurement noise, or number of samples), a **metric**, and a
**benchmark**, then toggle the **series** on and off. Lines are bootstrap medians over expressions;
the shaded band is a **95% confidence interval** (sources are unseeded, so we report the distribution,
not a single point). Hover a point for the exact value, its CI, and the number of expressions `n`.

> **Preliminary -- archived provenance.** These series are **re-rendered through `srbf.analysis` from
> the archived flash-ansr v23.0 evaluation** (the first paper's results), not a fresh canonical run.
> Provenance predates the current package family: flash-ansr models carry version `v23.0`;
> symbolic-data / srbf / the catalog artifacts are `-` (they did not exist at eval time). The version
> travels with each series (shown in the legend). Metrics are the strict `is_perfect_fit` (vNRR)
> numeric recovery on the validation points, token-level skeleton F1, and the median `log10` FVU;
> **symbolic recovery is omitted** (archived predictions are not normalized to the current `simplify`,
> so exact-match reads artifactually near zero). The **compute** x-axis is median per-problem
> **wall-clock fit time** -- the one budget that is comparable across methods (flash-ansr sweeps
> `choices`, PySR sweeps iterations, NeSymReS the beam width, E2E the candidate count; time unifies
> them). A fresh [canonical run](#reproduce) restores symbolic recovery and supersedes these numbers.

<div id="results-explorer">Loading the results explorer... (requires JavaScript)</div>

## What the axes show

- **Inference compute (time)** -- recovery / fit-quality improve as each method is given more
  wall-clock time. On the harder **v23-val** set the flash-ansr model-size gap is large and the
  neural/GP baselines (PySR, NeSymReS, E2E) sit on their own time--accuracy frontiers; on **FastSRB**
  the sizes converge.
- **Noise** -- robustness of recovery as relative measurement noise on the targets rises from 0 to 0.1.
- **N samples** -- how recovery depends on the number of support points the model is given.

Each series is one model or baseline; the shaded band is the bootstrap CI over the benchmark's
expressions (grouped by ground-truth skeleton).

## How the numbers are computed

`srbf` separates *running* a benchmark from *analysing* it. A `Benchmark.run()` emits **raw results
only** (per-problem predictions + targets); metrics are a **separate, standardized stage**:

```
Benchmark.run()  ->  raw snapshot     (predictions + targets, per problem)
derive_metrics   ->  + derived metrics (FVU, numeric recovery, F1, ...)
srbf.analysis    ->  aggregate + CI    (bootstrap median over expressions)
export_data      ->  the JSON this page reads
```

`srbf.analysis` (leaderboard / scaling / per-benchmark / distribution helpers) turns a set of runs --
each a raw snapshot tagged with `(model, benchmark, axis, x)` -- into bootstrap-CI'd aggregates.
`export_data(runs, "docs/results_data.json")` writes the tidy records the explorer above loads
client-side.

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

export_data(runs, "docs/results_data.json",
            engine=bench.model_adapter.get_simplipy_engine())   # -> the interactive page's data
```

The run is a compute job (real models over the benchmarks); the analysis + export stage is cheap and
deterministic given the raw pickles, so the page regenerates whenever the raw results change. For a
static (matplotlib) rendering instead of the interactive page, `build_report(runs, out_dir)` writes a
Markdown page + PNG figures (needs the `srbf[analysis]` extra).
