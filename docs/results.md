# Results & analysis

`srbf` separates *running* a benchmark from *analysing* it (see [Running evaluations](running.md) for
the raw stage). A run emits **raw results only**; metrics and the results page are a **separate,
standardized stage** so the numbers are reproducible from the raw records and never baked into a run.

```
Benchmark.run()  ->  raw snapshot        (per-problem predictions + targets)
derive_metrics   ->  + derived metrics   (FVU, numeric/symbolic recovery, F1, ...)
srbf.analysis    ->  the four views      (leaderboard, scaling, per-benchmark, distributions)
```

Because the data sources are **unseeded** (reproducibility comes from fixed catalogs, not seeds),
every headline is reported as a **distribution over expressions with a bootstrap confidence
interval**, not a single seeded point. Per-expression draws are grouped by `benchmark_eq_id`.

## The four standardized views

`srbf.analysis` turns a set of runs -- each a raw snapshot tagged with `(model, benchmark, scaling)` --
into four views (`Metric` = a snapshot column + a display label + its polarity):

- **Leaderboard** (`leaderboard`): one row per model/baseline; each metric a bootstrap median + 95% CI
  pooled over the benchmarks at a chosen scaling coordinate.
- **Scaling** (`scaling_figure`): a metric vs the scaling coordinate (e.g. inference compute), one line
  + CI band per model.
- **Per-benchmark breakdown** (`per_benchmark_figure`): the metric split by benchmark
  (FastSRB / Feynman / Nguyen / ...), so per-suite strengths and weaknesses are visible.
- **Distribution** (`distribution_figure`): the per-expression distribution of a metric per model
  (violin), honouring the "report the distribution, not a point" policy.

## Producing the page

`build_report` renders all four views to a Markdown page (`results.md`) plus PNG figures. Figures
need the optional `analysis` extra:

```sh
pip install 'srbf[analysis]'   # adds matplotlib; tables/leaderboards need no extra
```

From Python, feed it the runs directly:

```python
from srbf import Benchmark
from srbf.analysis import RunResult, build_report

runs = []
for (model, benchmark, scaling, config) in canonical_grid:      # your model x benchmark x scaling grid
    (bench,) = Benchmark.runs_from_config(config)
    runs.append(RunResult(model=model, benchmark=benchmark, scaling=scaling, snapshot=bench.run()))

build_report(runs, out_dir="docs", engine=bench.model_adapter.get_simplipy_engine())
```

Or, from the command line, describe the runs in a small manifest and let `srbf analyze` load them:

```yaml
# results_manifest.yaml -- each `path` is a pickled raw snapshot (a Benchmark.run() output)
runs:
  - {model: flash-ansr-120M, benchmark: fastsrb, scaling: 4096, path: 120M_fastsrb_4096.pkl}
  - {model: flash-ansr-3M,   benchmark: fastsrb, scaling: 4096, path: 3M_fastsrb_4096.pkl}
  - {model: brute-force,     benchmark: fastsrb,                path: bf_fastsrb.pkl}
```

```sh
srbf analyze results_manifest.yaml -o docs --title "Symbolic Regression Benchmark Results"
```

This writes `docs/results.md` (a leaderboard table) plus `docs/figures/{scaling,per_benchmark,distribution}.png`.

## The canonical run

The **canonical results** the published page reports come from the sweep configs under
`configs/evaluation/scaling/` (the model x inference-compute ladder over the shared `fastsrb`
validation benchmark). Reproduce them end to end:

```sh
srbf run -c configs/evaluation/scaling/<config>.yaml       # -> one result pickle per resolved !sweep run
# collect the pickles into results_manifest.yaml (model / benchmark / scaling / path)
srbf analyze results_manifest.yaml -o docs
```

The canonical run is a compute job (real models over the benchmark); the analysis stage above is
cheap and deterministic given the raw pickles, so the page regenerates in seconds whenever the raw
results change.
