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
| `complexity` | the number of tokens of the ground truth's skeleton |
| `predicted_expression` | the prediction as an infix string, in the names `x1, x2, ...` |
| `predicted_expression_prefix`, `predicted_skeleton_prefix` | the prediction in prefix notation, with its constants and with constants masked |
| `y_pred`, `y_pred_val` | the prediction's values on the support and validation points |
| `prediction_success`, `error` | whether a prediction could be parsed and evaluated, and why not |
| `fit_time` | seconds for the problem |
| `predicted_constants`, `predicted_score`, `predicted_log_prob` | what the method reports about its prediction, where it does |
| `placeholder`, `placeholder_reason` | see below |

An adapter may add columns of its own; the Flash-ANSR adapter records `generation_time` and
`refinement_time`, and a worker's `extra` dict is merged into the row.

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

## Averages over problem sets (the results site)

`bootstrap_report` and `srbf analyze` (below) treat every expression alike, across all benchmarks of a run. The
[results explorer](https://psaegert.github.io/srbf/) averages differently, and its numbers therefore differ from
theirs whenever the benchmarks differ in size. In srbf's suite one catalog, `erbench-syneq`, holds 5,301 of the
6,660 problems, so a number taken over all expressions is mostly a number on that catalog. The explorer averages
in three steps instead:

1. A problem's value is the mean of its runs, so every problem counts once.
2. A catalog (a problem set) is summarised by the mean \(\bar{x}_s\) and variance \(\sigma_s^2\) of its
   \(n_s\) problems' values.
3. The catalogs are combined by a random-effects average, the standard way to combine studies that measure the
   same quantity in different populations ([DerSimonian and Laird 1986](https://doi.org/10.1016/0197-2456(86)90046-2)):

$$
\hat{\theta} = \frac{\sum_s w_s\, \bar{x}_s}{\sum_s w_s},
\qquad
w_s = \frac{1}{\tau^2 + \sigma_s^2 / n_s} .
$$

The between-catalog variance \(\tau^2\) is estimated by Paule and Mandel's estimator
([Paule and Mandel 1982](https://doi.org/10.6028/jres.087.022)), which a review of the estimators recommends
([Veroniki et al. 2016](https://doi.org/10.1002/jrsm.1164)). Catalogs differ far more than their own sampling
noise, so \(\tau^2\) dominates the weights: on the 2026-09 release, `erbench-syneq` (79 % of the problems) carries
about 5 % of the weight of an average over the 29 catalogs, against 3.4 % for an equal share, and a catalog of one
or two problems about 2 %. Ratios such as the MDL ratio are averaged on a log scale (geometric means).

**Rates** (numeric and symbolic recovery and every other share of problems) are averaged by the beta-binomial
model ([Skellam 1948](https://doi.org/10.1111/j.2517-6161.1948.tb00014.x)): a catalog's sum \(k_s\) of problem
values is binomial with the catalog's own rate, and the catalogs' rates follow a beta distribution with mean
\(m\),

$$
k_s \sim \operatorname{Binomial}(n_s,\, p_s),
\qquad
p_s \sim \operatorname{Beta}\big(m\,\nu,\ (1 - m)\,\nu\big),
$$

with \(m\), the mean rate of a catalog, and the concentration \(\nu\) estimated by maximum likelihood
([Griffiths 1973](https://doi.org/10.2307/2529131)). The likelihood is closed-form, so catalogs at 0 % or 100 %
enter as they are ([Kuss 2015](https://doi.org/10.1002/sim.6383)). Averaging the log-odds of the catalogs' rates
would need a continuity correction for those, such as \((k + 0.5)/(n + 1)\), and such corrections bias small
catalogs towards 50 % ([Sweeting et al. 2004](https://doi.org/10.1002/sim.1761)). A catalog of one or two
problems carries under 1 % of a rate average's weight, since one problem says little about its catalog's rate.
A single catalog is shown as its plain rate \(k/n\) with the Wilson interval
([Wilson 1927](https://doi.org/10.1080/01621459.1927.10502953)), whose coverage stays close to 95 % even for a few
problems ([Brown, Cai and DasGupta 2001](https://doi.org/10.1214/ss/1009213286)).

**Intervals.** The 95 % interval is about the selected catalogs themselves: the weights (for rates, \(\nu\)) are
held at their estimates and only the sampling of problems within each catalog counts, the reading of a weighted
average of these catalogs' own values ([Rice, Higgins and Lumley 2018](https://doi.org/10.1111/rssa.12275)):

$$
\hat{\theta} \pm t_{0.975}\, \frac{\sqrt{\sum_s w_s^2\, \sigma_s^2 / n_s}}{\sum_s w_s},
$$

with \(t\) on \(\sum_s (n_s - 1)\) degrees of freedom. For rates the same rule goes through the model's estimating
equation, as a sandwich variance (Huber 1967, *Proceedings of the Fifth Berkeley Symposium on Mathematical
Statistics and Probability* 1, 221–233). Where a single further catalog would fall is a separate range,
\(\hat{\theta} \pm t_{0.975,\,S-2} \sqrt{\tau^2 + 1/\sum_s w_s}\) for \(S\) catalogs
([Higgins, Thompson and Spiegelhalter 2009](https://doi.org/10.1111/j.1467-985X.2008.00552.x)), and for rates the
2.5 % to 97.5 % range of the fitted beta distribution.

In a simulation on srbf's own catalog sizes (the 29 catalogs of the 2026-09 release and the 192 problem sets of
the next one; true rates fixed, problems drawn anew 100 times in each of 96 settings), the beta-binomial average
missed by 0.7 to 1.1 points on average and its interval held the true value 93 to 96 % of the time; the log-odds
average with the continuity correction missed by 5 to 6 points and held it 6 to 32 % of the time.

The reference implementation is `scripts/site_random_effects.py`; the explorer runs the same arithmetic, and the
site's test suite checks that the two agree. The [site's guide](https://psaegert.github.io/srbf/guide.html#averaging)
explains the same steps for a reader of the results.

## One table for every problem

`srbf table` judges the result files of any number of methods, draws and rungs at once and writes one
CSV row per problem and rung ([command line](cli.md#srbf-table)):

```bash
srbf table --tree mymethod:1:results/evaluation/scaling/mymethod -o table.csv
```

| columns | meaning |
|---|---|
| `model`, `draw`, `catalog`, `rung`, `shard`, `row` | where the row comes from: the method and draw named in `--tree`, the catalog directory, the rung and shard of the file, and the problem's `eval_row_index` |
| `sha` | the first 8 hex digits of the model weights' sha256, when the run recorded one |
| `success`, the recoveries, `skeleton_match_raw` | rates: 1 or 0 for every problem, a failed prediction is 0 |
| the analysis columns (`log10_fvu_val`, `r2_val`, `mdl_ratio`, `f1_score`, the edit distances, ...) | about the prediction; empty for a failed one, except the metrics with a worst value (token and variable F1, precision, recall), which a failed prediction takes |
| `skeleton_length`, `ground_truth_mdl`, `n_constants`, `total_nestedness`, `n_variables`, `n_support` | about the ground truth, for every problem |
| `fit_time`, `generation_time` | seconds as the run recorded them |

Description lengths are in bits. From Python, `srbf.table.build_table` does the same and
`srbf.table.judge_result_file` judges a single file.

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
metric along the budget, per benchmark, and as a distribution over expressions.

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
