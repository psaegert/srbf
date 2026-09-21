# Results

A run stores what the method answered; everything else is derived from that afterwards. This page
covers the result file, the metric step, summaries with intervals, and the standard report. The
published results of the methods entered so far are on the
[results explorer](https://psaegert.github.io/srbf/); its Table view exports every number as TSV or
CSV, for a comparison with your own.

```text
srbf run        ->  result file      the data and the prediction, one row per problem
derive_metrics  ->  + metric columns FVU, recovery, lengths, ...
bootstrap_report, srbf analyze  ->  summaries with 95 % intervals, tables, figures
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
| `benchmark_eq_id`, `eval_row_index` | the law's id in its catalog and the problem's position in the run |
| `skeleton`, `expression`, `ground_truth_prefix`, `ground_truth_infix` | the law: its skeleton with constants masked, and the expression with its constants, in prefix and infix notation |
| `variables` | the names of the columns as the catalog writes them, for example `v1, v2`; skeletons and answers name the same columns `x1, x2, ...` by position |
| `x`, `y`, `x_val`, `y_val` | the support and validation points, without noise |
| `y_noisy`, `y_noisy_val` | the targets with the run's noise. A method is given `y_noisy` when the run adds noise, and never any validation target |
| `n_support`, `noise_level` | the sampling parameters of the problem |
| `complexity` | the number of tokens of the law's skeleton |
| `predicted_expression` | the answer as an infix string, in the names `x1, x2, ...` |
| `predicted_expression_prefix`, `predicted_skeleton_prefix` | the answer in prefix notation, with its constants and with constants masked |
| `y_pred`, `y_pred_val` | the answer's values on the support and validation points |
| `prediction_success`, `error` | whether an answer could be parsed and evaluated, and why not |
| `fit_time` | seconds for the problem |
| `predicted_constants`, `predicted_score`, `predicted_log_prob` | what the method reports about its answer, where it does |
| `placeholder`, `placeholder_reason` | see below |

An adapter may add columns of its own; the Flash-ANSR adapter records `generation_time` and
`refinement_time`, and a worker's `extra` dict is merged into the row.

### Failures and placeholders

Whenever a method errors or returns no result, it has failed the problem. The row is a normal row
with `prediction_success: False` and the reason in `error`, whether the method reported the failure
itself, returned something that cannot be read, ran into its time limit, or raised an exception.
It counts as a miss in every success metric and has no value in the analysis metrics
([Metrics](metrics.md#success-metrics-and-analysis-metrics)).

A **placeholder** row marks a problem that was never posed: the catalog could not draw valid
points for the law within `max_trials`. Placeholders keep the rows of different runs aligned and
are left out of every summary, for every method alike.

### What a result file records

`__meta__` answers what ran:

| key | content |
|---|---|
| `config`, `config_sha` | the config's path and SHA-256 |
| `experiment`, and one key per sweep axis | which run of the config this is, for example `ladder: 32` |
| `config_provenance` | who chose the configuration ([Fairness](fairness.md#configuration-provenance-labels)) |
| `env` | versions of Python, torch, numpy, scipy, srbf, flash-ansr, simplipy and symbolic-data |
| `git`, `git_flash_ansr` | commit, branch and dirty state, when installed from a checkout |
| `inputs` | size and SHA-256 of the model weights, the tokenizer and a local catalog file |
| `system` | hostname, platform, CPU count and GPU |
| `worker` | what a worker's `info()` returned: its interpreter and versions |
| `ranking` | the resolved ranking of a Flash-ANSR run |
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
law: the problems drawn for one law are averaged first, and the laws are bootstrapped.

```python
from srbf import bootstrap_report, draw_distribution

report = bootstrap_report(scored, "numeric_recovery_val")
# {'metric': 'numeric_recovery_val', 'n_groups': 12, 'n_rows': 12,
#  'median': 0.083, 'ci_lower': 0.0, 'ci_upper': 0.25, 'interval': 0.95}

per_law = draw_distribution(scored, "log10_fvu_val")    # {law id: mean over its problems}
```

`bootstrap_report` resamples the per-law values 10,000 times and returns the median of the
resampled means with the percentile interval. It is seeded (`rng=0`), so a report is reproducible;
pass `rng=None` for fresh randomness, and `n`, `interval`, `aggregate` or `reduce` to change the
resampling. Placeholder rows are dropped, and so are values that are `None`; a law whose value is
not finite is left out, which for `log10_fvu_val` means the laws that were fitted exactly
(\(-\infty\)) and the failed predictions (\(+\infty\)). Read it next to the recovery rate.

## The standard report

```bash
srbf analyze -c mymethod.yaml -o report
srbf analyze -c mymethod.yaml --model mymethod \
             -c configs/evaluation/scaling/flash-ansr-v25.0-T8-3M_srbf.yaml --model flash-ansr-T8-3M \
             -o report
```

Every experiment and rung of a config whose result file exists becomes a run; the rest is skipped
with a note, so the report can be built while an evaluation is still going. The experiment's name
is the benchmark, a numeric sweep label is the budget, and the method's name comes from `--model`
or from the adapter block. `report/results.md` holds one row per method at its largest budget,
pooled over the benchmarks, each cell a bootstrap median with its 95 % interval, for numeric
recovery, symbolic recovery, skeleton F1, the MDL ratio, log10 FVU and R². `report/figures/` holds the
metric along the budget, per benchmark, and as a distribution over laws.

### From Python

```python
from srbf.analysis import runs_from_config, leaderboard, scaling_table, per_benchmark_table, build_report

runs = runs_from_config("mymethod.yaml", model="mymethod")
table = leaderboard(runs, engine=engine)                       # one row per method
along = scaling_table(runs, "numeric_recovery_val", engine=engine)
by_catalog = per_benchmark_table(runs, "numeric_recovery_val", engine=engine)
build_report(runs, "report", engine=engine)                    # results.md and figures/
```

A metric is named by its column. A run is a `RunResult(model, benchmark, snapshot, scaling=None)`,
so results from anywhere can be put into the same tables. `scaling_figure`, `per_benchmark_figure` and `distribution_figure` return
matplotlib figures, and `export_data(runs, "data.json", engine=engine)` writes the summaries as
JSON. Figures need `pip install "srbf[analysis]"`.
