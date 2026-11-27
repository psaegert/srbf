# Results

A run stores what the method predicted; everything else is derived from that afterwards. This page
covers the result file, the metric step, summaries with intervals, and the standard report. The
published results of the methods entered so far are on the
[results explorer](https://psaegert.github.io/srbf/); its Table view exports every number as TSV or
CSV, for a comparison with your own.

```text
srbf run        ->  result file      the data and the prediction, one row per problem
derive_metrics  ->  + metric columns FVU, recovery, lengths, ...
bootstrap_report, srbf analyze  ->  summaries with 95 % intervals, tables, figures
srbf table      ->  one CSV          a row per problem and rung, every metric
```

## The result file

`runner.output` is a pickled dict of columns: every key maps to a list with one entry per problem,
and the key `__meta__` holds the provenance.

```python
import pandas as pd

snapshot = pd.read_pickle("results/evaluation/scaling/flash-ansr-v25.0-T8-3M/nguyen/choices_000032.pkl")
meta = snapshot["__meta__"]                     # what ran; leave it in, comparisons check it
frame = pd.DataFrame({key: snapshot[key] for key in ("benchmark_eq_id", "predicted_expression", "fit_time")})
```

| columns | meaning |
|---|---|
| `benchmark_eq_id`, `eval_row_index` | the ground truth's id in its catalog and the problem's position in the run |
| `skeleton`, `expression`, `ground_truth_prefix`, `ground_truth_infix` | the ground truth: its skeleton with constants masked, and the expression with its constants, in prefix and infix notation |
| `variables` | the names of the columns as the catalog writes them, for example `v1, v2`; skeletons and predictions name the same columns `x1, x2, ...` by position |
| `x`, `y`, `x_val`, `y_val` | the support and validation points, without noise |
| `y_noisy`, `y_noisy_val` | the targets with the run's noise. A method is given `y_noisy` when the run adds noise, and never any validation target |
| `n_support`, `noise_level` | the sampling parameters of the problem |
| `draws_sha256`, `draws_repeat`, `draws_attempt`, `draws_formula_inherent`, `draws_failures` | with [frozen draws](benchmarks.md#draws-the-same-data-for-every-method): the SHA-256 of the draws file the problem was read from, its repeat, the attempt the problem keeps, whether no attempt let its formula be recognized, and the kind of each failed attempt in order (`numeric`, `symbolic`, ...) |
| `complexity` | the number of tokens of the ground truth's skeleton |
| `predicted_expression` | the prediction as an infix string, in the names `x1, x2, ...` |
| `predicted_expression_prefix`, `predicted_skeleton_prefix` | the prediction in prefix notation, with its constants and with constants masked |
| `y_pred`, `y_pred_val` | the prediction's values on the support and validation points |
| `prediction_success`, `error` | whether a prediction could be parsed and evaluated, and why not |
| `fit_time` | seconds for the problem |
| `predicted_constants`, `predicted_score`, `predicted_log_prob` | what the method reports about its prediction, where it does |
| `predicted_mdl` | the prediction's description length in bits, as the method priced it (Flash-ANSR); `derive_metrics` and `srbf table` price it again with the judge's engine, in bits |
| `placeholder`, `placeholder_reason` | see below |
| `n_candidates`, `candidates_judge_time`, `candidates_dropped`, `candidates_error` | how many [candidates](adapters.md#candidates) of the problem were stored, the seconds srbf took to judge them (not part of `fit_time`), how many the method returned beyond the limits, and why none were stored if that failed |

An adapter may add columns of its own; the Flash-ANSR adapter records `generation_time` and
`refinement_time`, and a worker's `extra` dict is merged into the row.

Every expression a method ends with, its answer among them, is stored next to the result file in
`<name>.candidates/`, one record per problem, with srbf's judgement of each: its price in bits and its
FVU on the support and validation points. `srbf.candidates.iter_candidates(path)` reads them back
([Candidates](adapters.md#candidates)).

### Failures and placeholders

Whenever a method errors or returns no result, it has failed the problem. The row is a normal row
with `prediction_success: False` and the reason in `error`, whether the method reported the failure
itself, returned something that cannot be read, ran into its time limit, or raised an exception.
It counts as a miss in every success metric, takes the worst value of the analysis metrics that
have one, and has no value in the others
([Metrics](metrics.md#success-metrics-and-analysis-metrics)).

A **placeholder** row marks a problem that was never posed: the catalog could not draw valid
points for the ground truth within `max_trials`. Placeholders keep the rows of different runs aligned and
are left out of every summary, for every method alike.

### What a result file records

`__meta__` records what ran:

| key | content |
|---|---|
| `config`, `config_sha` | the config's path and SHA-256 |
| `experiment`, and one key per sweep axis | which run of the config this is, for example `ladder: 32` |
| `config_provenance` | who chose the configuration ([Fairness](fairness.md#configuration-provenance-labels)) |
| `env` | versions of Python, torch, numpy, scipy, srbf, flash-ansr, simplipy and symbolic-data |
| `git`, `git_flash_ansr` | commit, branch and dirty state, when installed from a checkout |
| `inputs` | size and SHA-256 of the model weights, the tokenizer, a local catalog file and the frozen draws file (`catalog_draws`) |
| `draws` | with frozen draws: the catalog, its file's SHA-256, the repeat, the draws file's path and SHA-256, the manifest's SHA-256, the render's settings, the noise level and the acceptance the attempts were chosen by |
| `system` | hostname, platform, CPU count and GPU |
| `worker` | what a worker's `info()` returned: its interpreter and versions |
| `ranking`, `inference_precision` | the resolved ranking of a Flash-ANSR run and the arithmetic its transformer ran in |
| `setup` | the setup before the clock ([Fairness](fairness.md#setup-before-the-clock)): `warmup` (whether the warm-up problem ran), `build_time` and `prepare_time` (seconds spent building and preparing the adapter), and `events`, one record per start of the method: `reason` (`start`, `restart`, `restart after a hang`), a worker's `load_time` and `setup_time`, and the warm-up's `warmup_time`, `warmup_fit_time` and `warmup_error` |
| `shard`, `shards` | for a shard file, its index and count; for a merged file, what was merged |
| `timestamp` | when the run was started |

## Deriving metrics

```python
from simplipy import SimpliPyEngine
from srbf import derive_metrics

engine = SimpliPyEngine.load("acj-5-4-llm", install=True)
scored = derive_metrics(snapshot, engine=engine)      # a new dict: the raw columns plus the metrics

print(scored["fvu_val"][:3], scored["numeric_recovery_val"][:3], scored["symbolic_recovery"][:3])
```

`derive_metrics` does not change its input. The engine is the judge: it simplifies both skeletons,
supplies the operator arities the tree distance needs and prices description lengths. Pass the
engine the run was configured with (`simplipy_engine`). Without an engine, give
`operator_arity={...}` instead; the skeletons are then compared as they were written and the
description-length columns are not added. Every column is defined in [Metrics](metrics.md).

## Summaries with intervals

Sampling is not seeded, so a number is reported with its uncertainty. The unit of resampling is the
expression: the problems drawn for one expression are averaged first, and the expressions are bootstrapped.

```python
from srbf import bootstrap_report, draw_distribution

report = bootstrap_report(scored, "numeric_recovery_val")
# {'metric': 'numeric_recovery_val', 'n_groups': 12, 'n_rows': 12,
#  'median': 0.083, 'ci_lower': 0.0, 'ci_upper': 0.25, 'interval': 0.95}

per_expression = draw_distribution(scored, "log10_fvu_val")    # {expression id: mean over its problems}
```

`bootstrap_report` resamples the per-expression values 10,000 times and returns the median of the
resampled means with the percentile interval. It is seeded (`rng=0`), so a report is reproducible;
pass `rng=None` for fresh randomness, and `n`, `interval`, `aggregate` or `reduce` to change the
resampling. Placeholder rows are dropped, and so are values that are `None`; an expression whose value is
not finite is left out, which for `log10_fvu_val` means the expressions that were fitted exactly
(\(-\infty\)) and the failed predictions (\(+\infty\)). Read it next to the recovery rate.

## Averages over problem sets

`bootstrap_report` takes every expression it is given alike. `srbf analyze` below, like the
[results explorer](https://psaegert.github.io/srbf/), averages over the benchmarks instead, and within each benchmark
over its subsets, so that one large catalog does not decide the number (`srbf.aggregate`):
[How results are averaged](averaging.md) describes the method, why it was chosen, and the literature it rests on.

## One table for every problem

`srbf table` judges the result files of any number of methods, draws and rungs at once and writes one
CSV row per problem and rung ([command line](cli.md#srbf-table)):

```bash
srbf table --tree mymethod:1:results/evaluation/scaling/mymethod -o table.csv
```

| columns | meaning |
|---|---|
| `model`, `draw`, `catalog`, `rung`, `shard`, `row` | where the row comes from: the method and draw named in `--tree`, the catalog directory, the rung and shard of the file, and the problem's `eval_row_index` |
| `benchmark` | the benchmark the catalog belongs to, from the `benchmark` field of the catalog's metadata, else the catalog itself: the unit results are averaged over ([How results are averaged](averaging.md#benchmarks-their-subsets-and-their-problems)) |
| `sha` | the first 8 hex digits of the model weights' sha256, when the run recorded one |
| `success`, the recoveries, `skeleton_match_raw` | rates: 1 or 0 for every problem, a failed prediction is 0 |
| the analysis columns (`log10_fvu_val`, `r2_val`, `mdl_ratio`, `f1_score`, the edit distances, ...) | about the prediction; empty for a failed one, except the metrics with a worst value (token and variable F1, precision, recall), which a failed prediction takes |
| `skeleton_length`, `ground_truth_mdl`, `n_constants`, `total_nestedness`, `n_variables`, `n_support` | about the ground truth, for every problem |
| `fit_time`, `generation_time` | seconds as the run recorded them |

Description lengths are in bits, everywhere srbf states one: the result file, the derived metrics,
the table and the candidate stores. From Python, `srbf.table.build_table` does the same (its `benchmarks`
argument maps a catalog to its benchmark; `srbf.suites.catalog_benchmarks` reads them from the catalogs) and
`srbf.table.judge_result_file` judges a single file, without the `benchmark` column. The per-file cache holds
the judged rows only, so a catalog whose benchmark changes needs no file judged again.

## The standard report

```bash
srbf analyze -c mymethod.yaml -o report
srbf analyze -c mymethod.yaml --model mymethod \
             -c configs/evaluation/scaling/flash-ansr-v25.0-T8-3M_srbf.yaml --model flash-ansr-T8-3M \
             -o report
```

Every experiment and rung of a config whose result file exists becomes a run; the rest is skipped
with a note, so the report can be built while an evaluation is still going. The experiment's name
is the catalog, its benchmark comes from the `benchmark` field of the catalog's metadata (else the
catalog is a benchmark of its own), a numeric sweep label is the budget, and the method's name comes
from `--model` or from the adapter block. `report/results.md` holds one row per method at its largest budget,
each cell the average over the benchmarks with its 95 % interval, as the results explorer averages
([How results are averaged](averaging.md)): every benchmark counts about the same whatever its size,
for numeric recovery, symbolic recovery, skeleton F1, the MDL ratio, log10 FVU and R². `report/figures/` holds the
metric along the budget, per benchmark, and as a distribution over expressions.

### From Python

```python
from srbf.analysis import runs_from_config, leaderboard, scaling_table, per_benchmark_table, build_report

runs = runs_from_config("mymethod.yaml", model="mymethod")
table = leaderboard(runs, engine=engine)                       # one row per method
along = scaling_table(runs, "numeric_recovery_val", engine=engine)
by_benchmark = per_benchmark_table(runs, "numeric_recovery_val", engine=engine)
by_catalog = per_benchmark_table(runs, "numeric_recovery_val", engine=engine, by="catalog")
build_report(runs, "report", engine=engine)                    # results.md and figures/
```

A metric is named by its column. A run is a `RunResult(model, benchmark, snapshot, scaling=None, catalog="")`,
so results from anywhere can be put into the same tables: the catalogs of one benchmark (`catalog` names each, by
default the benchmark itself) are its subsets, and a snapshot's `subset` column, where it has one, places each problem
in a subset of its benchmark instead. `scaling_figure`, `per_benchmark_figure` and `distribution_figure` return
matplotlib figures, and `export_data(runs, "data.json", engine=engine)` writes the summaries as
JSON. Figures need `pip install "srbf[analysis]"`.
