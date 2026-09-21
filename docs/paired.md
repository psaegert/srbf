# Paired comparisons

Every method is evaluated on the same laws. A comparison of two methods should use that: a hard
law is hard for both, so the uncertainty of each method's average is dominated by the spread
between laws, and that spread cancels in the difference. `srbf.reporting` compares methods law by
law,

$$
\Delta_k = m_A^{(k)} - m_B^{(k)}, \qquad
\operatorname{Var}(\bar\Delta) = \tfrac{1}{K}\big(\operatorname{Var} A + \operatorname{Var} B - 2 \operatorname{Cov}(A, B)\big),
$$

and the covariance is large exactly because difficulty is shared. Two confidence intervals can
overlap while one method is better on law after law, so compare methods with a paired report and
do not subtract two intervals.

## A paired report

```python
from srbf.reporting import paired_report

report = paired_report(scored_a, scored_b, "numeric_recovery_val", higher_is_better=True)
print(report["delta_mean"], (report["ci_lower"], report["ci_upper"]), report["p_value"])
print(report["win_rate"], report["n_pairs"])     # {'a_better': 7, 'b_better': 4, 'tied': 1} 12
```

Both arguments are snapshots with derived metrics ([Results](results.md#deriving-metrics)) of the
same benchmark, one per method. The report pairs the laws by `benchmark_eq_id`, averages the problems of a law, and
bootstraps the laws. It contains:

| key | meaning |
|---|---|
| `delta_mean`, `ci_lower`, `ci_upper` | the mean of \(\Delta\), A minus B, with its percentile interval over laws |
| `p_value` | two-sided, from the same bootstrap |
| `delta_median`, `median_ci_lower`, `median_ci_upper` | the same for the median |
| `win_rate` | the numbers of laws where A is better, where B is better, and where they tie: `a_better`, `b_better`, `tied` |
| `prob_superiority` | the probability that A beats B on a random law, a tie counting half |
| `wilcoxon` | a signed-rank test that keeps the zeros (`zero_method="pratt"`): on a rate most laws tie, and dropping the ties would change the test |
| `mde_80` | the smallest true difference this comparison would detect 80 % of the time |
| `n_pairs`, `n_only_a`, `n_only_b` | the laws both methods have a value for, and those only one has |
| `variance_decomposition` | how much of the variance of \(\Delta\) is between laws and how much comes from the problems drawn within a law |

An exact fit has `log10_fvu_val` \(-\infty\), which makes a mean of differences infinite, and one
diverging answer decides a mean of `r2_val`. For these columns read `delta_median` and the ranks.

### What is paired

A success metric is defined for every law, so it is paired over all of them and a method that
fails on a law has missed it. An analysis metric exists only where there is an answer, so it is
paired over the laws where both methods have one, and `n_only_a` and `n_only_b` say how many were
left out. `worst_rank=True` adds rank statistics over all laws in which a missing value is placed
last; it is off by default, because a rank over all laws mixes how often a method answers into how
well it answers.

### Pairing is checked

Laws are joined by id and never by row order. The provenance of the two snapshots (`__meta__`,
which `derive_metrics` carries along) is checked: where both record a local data file, such as a
frozen catalog, it has to be the same file, and a mismatch raises `PairingContractError`. A
snapshot without `__meta__` needs `allow_unverified=True`. Comparisons are made per benchmark.

Two runs on a named catalog share the laws and not the points
([Benchmarks](benchmarks.md#the-same-points-for-several-methods)). The report then measures the
difference between the methods plus the difference between two draws of the data, and its interval
covers both.

## Verdicts against the benchmark's own noise

With several problems per law (`problems_per_expression` above 1), the noise of the benchmark
itself can be measured, and a difference can be judged against it:

```python
from srbf.reporting import draw_values, self_noise, pair_margin

values_a = draw_values(scored_a, "numeric_recovery_val")     # {law id: the values of its problems}
values_b = draw_values(scored_b, "numeric_recovery_val")
margin = pair_margin(self_noise(values_a), self_noise(values_b))

report = paired_report(scored_a, scored_b, "numeric_recovery_val", margin=margin)
print(report["verdict"], report["equivalence_attainable"])
```

`self_noise` compares a method with itself across random halves of each law's problems; `pair_margin`
combines two such nulls into the largest difference that two equally good methods would plausibly
show. The margin is derived from the data and belongs to the pair. The verdict reads the interval
against it:

| verdict | meaning |
|---|---|
| `better`, `worse` | the interval lies entirely beyond the margin: a difference larger than the benchmark's noise explains |
| `equivalent` | the interval lies entirely inside the margin: any remaining difference is smaller than the benchmark can measure |
| `undecided` | neither; read `mde_80` for what could have been detected |

`equivalence_attainable` says whether `equivalent` could be reached at this sample size at all.
When it is `False`, an `undecided` comes from the resolution of the benchmark and says nothing
about parity.

## Comparing along the budget

A **series** is one method's ladder: a mapping from rung to scored snapshot. Two series can be
compared rung by rung, for variants of one method, or at equal time, for different methods:

```python
import pandas as pd
from srbf import derive_metrics
from srbf.reporting import paired_delta_curve, paired_report_at_time

def series(method):
    return {rung: derive_metrics(pd.read_pickle(f"results/{method}/nguyen/budget_{rung:04d}.pkl"), engine=engine)
            for rung in (1, 4, 16)}

series_a, series_b = series("mymethod"), series("baseline")
curve = paired_delta_curve(series_a, series_b, "numeric_recovery_val", x_policy="time")
at_10s = paired_report_at_time(series_a, series_b, "numeric_recovery_val", t=10.0)
```

At equal time each method is brought to \(t\) per law, by linear interpolation in log time between
its two neighboring rungs; a law counts only where it has values at both neighbors on both sides.
Nothing is extrapolated. Below a method's cheapest rung there is no value. Beyond its most
expensive rung the last value is carried forward and marked `plateau`, a lower bound under the
assumption that more compute does not hurt, and a verdict that such a side could overturn by
improving is downgraded to `undecided` with `verdict_note="ladder-limited"`. The position of a rung
on the time axis is the median `fit_time` of its problems. `series_report_at_time` gives one
series' value at \(t\) under the same rules.

## Ranking many methods

```python
from srbf.reporting import draw_values, rank_league

scored = {"mymethod": scored_a, "baseline": scored_b, "another": scored_c}
values = {name: {law: v.mean() for law, v in draw_values(s, "log10_fvu_val").items()} for name, s in scored.items()}
ranks = rank_league(values, higher_is_better=False)
print(ranks["mean_ranks"], ranks["friedman_p"], ranks["cd"], ranks["cliques"])
```

Within each law the methods are ranked, 1 being best and ties sharing the mean of their places; a
method without a value for a law ranks last on it. Every law hands out the same places, so easy and
hard laws count alike. The Friedman test, corrected for ties, asks whether the mean ranks differ at
all. The critical difference after Nemenyi,

$$
\mathrm{CD} = \frac{q_{\alpha}(k, \infty)}{\sqrt 2} \sqrt{\frac{k (k + 1)}{6 K}},
$$

for \(k\) methods over \(K\) laws, is the smallest gap between two mean ranks that counts as real
when every pair is compared at once; `cliques` lists the groups of methods that lie within one CD
of each other (Demšar 2006). Ranks measure consistency: winning by a hair counts like winning by a
mile. How large a difference is belongs to the paired report.

## Exact tests for rates

For a quick contrast of two rates on the same laws, `srbf.metrics` has the exact tools:

```python
from srbf.metrics import mcnemar_exact, wilson_interval

test = mcnemar_exact(scored_a["numeric_recovery_val"], scored_b["numeric_recovery_val"])
print(test.n_a_only, test.n_b_only, test.p_value)       # the problems they disagree on, and the exact two-sided p
low, high = wilson_interval(successes=41, total=120)     # a 95 % interval for one rate
```

`mcnemar_exact` takes two columns and pairs them by position, so both runs must hold the same
problems in the same order: two complete runs of one catalog with one problem per law do.

## Why laws are the unit

The laws are the exchangeable unit of a benchmark; the problems drawn for one law are repeated
measurements of it. Averaging them per law and bootstrapping the laws is the cluster bootstrap for
the mean, and no further variance correction applies, because no data is shared between the
resampled units. `hierarchical=True` additionally resamples the problems within each law for the
rank statistics.
