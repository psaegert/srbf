# How results are averaged

The [results explorer](https://psaegert.github.io/srbf/) reports every value as an average over problem sets
(benchmarks), with a 95 % interval. This page describes how, why, and what the method rests on.
`srbf analyze` ([Results](results.md)) averages the same way; [In the package](#in-the-package) says where the code is.

## Benchmarks, their subsets and their problems

Results are organised in three levels: benchmarks, the subsets of each benchmark, and the problems of each subset.

- A **benchmark** is a problem set as its authors published it, the unit every average is over.
- A **subset** is a part of a benchmark: a part its authors published it in (SRSD's easy, medium and hard formulas,
  for instance), or the part of it that one catalog holds. srbf stores some benchmarks in several catalogs, for
  reasons that have nothing to do with the problems: data that may only be shared under other terms lives in its own
  repository (`-restricted`, `-share-alike`), problems that come with fixed data points are kept apart from those
  whose points srbf draws (`-points`), or a benchmark has a formula and a data half.
- A **problem** is one formula with its data; its value is the mean of its runs.

Each catalog names its benchmark in the `benchmark` field of its metadata
([Benchmarks](benchmarks.md#benchmarks-and-catalogs)), and a problem's subset is the `subset` field of its rows where
they carry one; otherwise its catalog is its subset. A catalog without the `benchmark` field is a benchmark of its own.
srbf guesses neither from names. `srbf table` writes the benchmark of every row next to its catalog, and `srbf analyze`
reads it from the catalogs' metadata, so the site and the package average over the same structure.

The structure is fixed before any result is looked at: it comes from how the authors split their benchmark, or from
how srbf stores it, and never from how different the parts turn out to be. Choosing it by measuring would let the
results decide which problems count how much, and a split that depends on the data being averaged makes the
average's interval wrong. How much the subsets differ does enter, but only through the weights, below: subsets that
really differ count more on their own, and subsets that are alike collapse into one benchmark. Nothing is switched by a
threshold.

The two other ways of handling a benchmark in parts both let something other than the problems decide its weight.
Treating each part as a benchmark of its own makes a benchmark split in two count about twice as much as it does whole,
since every benchmark counts about the same whatever its size (below). Merging the parts into one set ignores that
the authors' parts can differ as much as two benchmarks do. The three-level model sits between the two, where the
results put it.

A selection of problem sets in the explorer is a view: it narrows what is averaged, and the average is still over
benchmarks and their subsets. The explorer still shows each catalog on its own where it lists problem sets one by one,
and a catalog split into subsets as one row, the average over its subsets.

## Why average over problem sets

The problem sets differ a lot, in difficulty and in size. In srbf's suite one benchmark, `erbench-syneq`, holds 5,301
of the 6,660 problems, so an average over all problems is mostly an average over that one benchmark. The explorer
therefore averages in steps:

1. A problem's value is the mean of its runs, so every problem counts once.
2. A subset is summarised by the mean \(\bar{x}_{bs}\) and the variance \(\sigma_{bs}^2\) of its \(n_{bs}\)
   problems' values (benchmark \(b\), subset \(s\)).
3. The subsets of each benchmark are combined, each weighted by how precisely its own mean is known and by how much
   the subsets of a benchmark differ from each other.
4. The benchmarks are combined, each weighted by how precisely its own value is known and by how much benchmarks
   differ from each other.

Steps 3 and 4 treat each subset and each benchmark as its own population with its own true value, as a meta-analysis
treats studies of the same quantity in different populations. That is the random-effects average
([DerSimonian and Laird 1986](https://doi.org/10.1016/0197-2456(86)90046-2)) with three levels, the model meta-analysis
uses for several effect sizes from one study ([Konstantopoulos 2011](https://doi.org/10.1002/jrsm.35);
[Van den Noortgate et al. 2013](https://doi.org/10.3758/s13428-012-0261-6);
[Cheung 2014](https://doi.org/10.1037/a0032968)): here a benchmark plays the study and its subsets the effect sizes.

## Values that are not rates

A subset's mean scatters around its benchmark's true value with variance \(\tau_S^2\), and a benchmark's true value
around the average \(\theta\) with variance \(\tau_B^2\):

$$
\bar{x}_{bs} = \theta + u_b + u_{bs} + e_{bs},
\qquad
u_b \sim N(0, \tau_B^2),\quad u_{bs} \sim N(0, \tau_S^2),\quad e_{bs} \sim N\big(0,\ v_{bs}\big),\quad
v_{bs} = \sigma_{bs}^2 / n_{bs}.
$$

Given the two variances, the best (generalised least squares) estimate of \(\theta\) is computed in two steps. A
benchmark's subsets combine into its estimate, and the benchmarks combine like sets:

$$
\bar{x}_b = \frac{\sum_s w_{bs}\, \bar{x}_{bs}}{W_b},
\quad
w_{bs} = \frac{1}{\tau_S^2 + v_{bs}},\quad W_b = \sum_s w_{bs};
\qquad
\hat{\theta} = \frac{\sum_b w_b\, \bar{x}_b}{\sum_b w_b},
\quad
w_b = \frac{1}{\tau_B^2 + 1 / W_b}.
$$

A benchmark of one subset has \(1/W_b = \tau_S^2 + v_{b1}\). Both variances are estimated by Paule and Mandel's rule
([Paule and Mandel 1982](https://doi.org/10.6028/jres.087.022)), which a review of the estimators recommends
([Veroniki et al. 2016](https://doi.org/10.1002/jrsm.1164)): a variance is the value at which the weighted squared
deviations equal their expectation, and 0 where they fall short of it.

- \(\tau_S^2\) from the subsets about their benchmark's weighted mean, within benchmarks:
  \(\sum_b \sum_s w_{bs} (\bar{x}_{bs} - \bar{x}_b)^2 = \sum_b (S_b - 1)\) for benchmarks of \(S_b\) subsets.
  The benchmarks' own values cancel from these deviations, so they measure the spread of subsets alone; without a
  benchmark of two subsets, \(\tau_S^2 = 0\).
- \(\tau_B^2\) from the benchmarks' estimates with their variances \(1/W_b\), as for sets:
  \(\sum_b w_b (\bar{x}_b - \hat{\theta})^2 = B - 1\).

A subset whose problems all have the same value, or that has one problem, has no variance of its own and takes the
variance pooled over the other subsets. When no benchmark has two subsets the average is the two-level one, every
benchmark a set weighted by \(1/(\tau^2 + \sigma_s^2/n_s)\), to the last bit.

The two variances act only through the weights:

- Subsets that are alike (\(\tau_S^2 = 0\)) combine by their precision: a benchmark counts as one set holding all
  its problems, about once.
- Subsets that differ much more than benchmarks do (\(\tau_S^2 \gg \tau_B^2\)) each count about as much as a
  benchmark of their own: a benchmark of four such subsets weighs about four times as much as a benchmark of one.
- In between, a benchmark of \(S_b\) subsets weighs about \(1/(\tau_B^2 + \tau_S^2/S_b)\): more subsets tell
  more about the benchmark's true value, up to the limit that \(\tau_B^2\) sets.

Problem sets differ far more than their own sampling noise, so the variances dominate the weights: every benchmark
counts about the same whatever its size, and a set of a few problems counts less, because its own mean is uncertain.
On the 2026-09 release (29 catalogs, each a benchmark of its own), `erbench-syneq` (79 % of the problems) carries about
5 % of the weight of an average over all of them, against 3.4 % for an equal share, and a catalog of one or two
problems about 2 %.

The mean leaves out values that are not finite; a problem's value is the mean of its runs' finite values. Ratios such
as the MDL ratio are averaged in \(\log_2\) and shown back on their own scale, as geometric means, so a ratio of 2 and
one of 0.5 average to 1.

## Rates

A rate is the share of problems a method solves (numeric and symbolic recovery, success). Rates are averaged by the
beta-binomial model ([Skellam 1948](https://doi.org/10.1111/j.2517-6161.1948.tb00014.x)): a set's sum \(k_s\) of
problem values is binomial with the set's own rate \(p_s\), and the sets' rates follow a beta distribution with
mean \(m\),

$$
k_s \sim \operatorname{Binomial}(n_s,\, p_s),
\qquad
p_s \sim \operatorname{Beta}\big(m\,\nu,\ (1 - m)\,\nu\big).
$$

The average is \(m\), the mean rate of a problem set. It is estimated together with the concentration \(\nu\) by
maximum likelihood ([Griffiths 1973](https://doi.org/10.2307/2529131)), by Newton's method with the exact digamma and
trigamma derivatives; where the sets spread no more than the binomial explains, \(\nu\) is infinite and \(m\) is the
rate over all their problems.

The likelihood is closed-form, so a set where a method solves every problem, or none, counts as it is
([Kuss 2015](https://doi.org/10.1002/sim.6383)). Averaging the log-odds of the sets' rates, the obvious alternative,
needs a continuity correction for such sets, such as \((k + 0.5)/(n + 1)\), and corrections like it pull small sets
towards 50 % ([Sweeting, Sutton and Lambert 2004](https://doi.org/10.1002/sim.1761)). A logit-normal
(binomial-normal) model also avoids the correction, but its likelihood is an integral that needed about 100 quadrature
points to converge on srbf's data, which has many sets at 0 % or 100 %.

For rates a set of one or two problems counts less than for other values, under 1 % of the weight on the 2026-09
release, because a single problem says little about its set's rate.

### Rates over benchmarks and their subsets

With subsets, a subset's rate scatters around its benchmark's rate \(p_b\), and the benchmarks' rates around
\(m\):

$$
k_{bs} \sim \operatorname{Binomial}(n_{bs},\, p_{bs}),
\qquad
p_{bs} \sim \operatorname{Beta}\big(p_b\,\nu_S,\ (1 - p_b)\,\nu_S\big),
\qquad
p_b \sim \operatorname{Beta}\big(m\,\nu_B,\ (1 - m)\,\nu_B\big).
$$

Given its benchmark's rate, a subset's count is beta-binomial: its mean is \(n_{bs}\, p_b\) and its variance that of
a binomial count times the design effect \(1 + (n_{bs} - 1)\rho_S\), where \(\rho_S = 1/(1 + \nu_S)\) is the
correlation between two problems of one subset that the benchmark's rate does not explain
([Kish 1965](#sources)). The benchmarks' likelihood in this model has an integral over every \(p_b\) with no closed
form, so srbf fits it in two stages:

1. **Within benchmarks**, \(\rho_S\) by Williams' rule
   ([Williams 1982](https://doi.org/10.2307/2347977)), the counterpart for counts of Paule and Mandel's: a subset of
   \(n\) problems counts as \(c = 1/(1 + (n - 1)\rho_S)\) of them, a benchmark's rate is
   \(\hat{p}_b = \sum_s c_{bs} k_{bs} / \sum_s c_{bs} n_{bs}\), and Pearson's chi-square of the subsets about it equals
   its degrees of freedom:

    $$
    \sum_b \sum_s \frac{c_{bs}\, n_{bs}\, (k_{bs}/n_{bs} - \hat{p}_b)^2}{\hat{p}_b (1 - \hat{p}_b)} = \sum_b (S_b - 1).
    $$

    \(\rho_S = 0\) where the chi-square falls short of that, and 1 (every subset counts as one problem) where it
    exceeds it even there. A benchmark whose subsets are all at 0 % or all at 100 % says nothing about \(\rho_S\) and
    is left out of both sides. Like the variances of means, \(\rho_S\) is estimated from the deviations from each
    benchmark's own rate, with one degree of freedom spent on each; a maximum-likelihood fit of the benchmarks' rates
    alongside \(\rho_S\) would understate it with two or three subsets per benchmark
    ([Neyman and Scott 1948](https://doi.org/10.2307/1914288)).
2. **Each benchmark becomes one set of effective counts**, its subsets' counts scaled by their \(c\):
   \(n^*_b = \sum_s c_{bs}\, n_{bs}\) and \(k^*_b = \sum_s c_{bs}\, k_{bs}\) (Kish's effective sample size). Given
   \(p_b\), \(k^*_b\) then has exactly the mean \(n^*_b\, p_b\) and the variance \(n^*_b\, p_b (1 - p_b)\) of a binomial
   count of \(n^*_b\) problems, so the benchmarks enter the beta-binomial model above as sets do. \(\rho_S = 0\)
   merges a benchmark's subsets, every problem once; \(\rho_S = 1\) makes each subset count as one problem.
3. **Across benchmarks**, the beta-binomial average of the effective counts gives \(m\) and \(\nu_B\), with the
   score-test interval (below).

The approximation is in the second stage: the beta-binomial likelihood is applied to counts that match a binomial's
first two moments, not its whole distribution, with \(\rho_S\) held at its estimate. [How well it works](#how-well-it-works)
compares it with the full model fitted by maximum likelihood. When no benchmark has two subsets, \(\rho_S = 0\) and
the average is the two-level beta-binomial one, to the last bit.

## One benchmark

A single benchmark of one subset is shown as its own mean with the \(t\) interval over its problems, and a rate as its
plain rate \(k/n\) with the Wilson interval ([Wilson 1927](https://doi.org/10.1080/01621459.1927.10502953)), whose
coverage stays close to 95 % even for a few problems
([Brown, Cai and DasGupta 2001](https://doi.org/10.1214/ss/1009213286)). A single benchmark of several subsets is the
average over its subsets, their weights within it as above, and a rate its effective rate \(k^*/n^*\) with the Wilson interval on its
effective count.

## Medians and distributions

Medians and distributions weigh each subset the way the average does. With \(n_{0S} = \sigma^2 / \tau_S^2\) and
\(n_{0B} = \sigma^2 / \tau_B^2\) from the average's fit (\(\sigma^2\) the variance within subsets, pooled), a
subset of \(n_{bs}\) problems counts as \(n_{bs} / (1 + n_{bs}/n_{0S})\) problems of its benchmark, a benchmark as
their sum \(N^*_b\), and the benchmark weighs \(N^*_b / (N^*_b + n_{0B})\), shared among its subsets in proportion
to what each counts. Within a subset every problem counts once. A benchmark of one subset weighs
\(n / (n + n_{0B})\). The interval of a median follows Woodruff's method
([Woodruff 1952](https://doi.org/10.1080/01621459.1952.10483443)): the share of each subset below the median is
averaged over the benchmarks like any value, and the interval of that share is mapped back through the distribution.

## Intervals

The 95 % interval is about the selected problem sets themselves: the weights (for rates, \(\rho_S\) and \(\nu_B\))
are held at their estimates and only the sampling of problems within each subset counts. A value is then a weighted
average of these subsets' own values, the reading of such averages by
[Rice, Higgins and Lumley (2018)](https://doi.org/10.1111/rssa.12275); problem sets in general would add the spread
between benchmarks and between subsets to the interval. With \(\omega_{bs} = w_b\, w_{bs} / (W_b \sum_b w_b)\) the
share of the weight a subset carries,

$$
\hat{\theta} \pm t_{0.975}\, \sqrt{\textstyle\sum_b \sum_s \omega_{bs}^2\, \sigma_{bs}^2 / n_{bs}},
$$

with \(t\) on \(\sum_b \sum_s (n_{bs} - 1)\) degrees of freedom. For rates the interval inverts a score test on the
benchmarks' effective counts, as Wilson's
interval does for a single rate: it holds every mean rate \(m\) that Rao's score test
([Rao 1948](https://doi.org/10.1017/S0305004100023987)) does not reject at that level. At each tested \(m\) the
concentration \(\nu\) is fitted anew, and the score for \(m\) is taken with \(\nu\)'s share projected out. Its
variance comes from the sampling of problems within each benchmark: every benchmark's effective count is binomial at
its posterior rate under the tested \(m\), and the variance is summed exactly over the possible counts, not
linearised.

Both choices matter where many sets sit at 0 % or 100 %. A linearised (delta-method) interval, the sandwich variance
of Huber (1967) on the log-odds, spanned 0 % to 88 % for one success in 6,660 problems on the 2026-09 release; this
interval gives 0.006 % to 1.5 % there. Testing at each \(m\) keeps the interval between 0 and 100 %. The test is run
on a grid of step 0.5 on the log-odds scale, outward from the estimate, and the outermost kept point is refined by
bisection. Where the kept values come in more than one piece, the interval spans them all. This happens when \(\nu\),
fitted anew far from the estimate, is small enough that the test loses power: 3 of the release's 1,147 rate averages
over two or more sets, all within 0.2 points of 0 % or 100 %.

## One more benchmark

Where the value of one more benchmark would fall is a separate range, shown once at least three benchmarks are
selected: \(\hat{\theta} \pm t_{0.975,\,B-2} \sqrt{\tau_B^2 + \tau_S^2 + 1/\sum_b w_b}\) for \(B\) benchmarks
([Higgins, Thompson and Spiegelhalter 2009](https://doi.org/10.1111/j.1467-985X.2008.00552.x)), and for rates the
2.5 % to 97.5 % range of a beta distribution with mean \(m\) and \(1/(1 + \nu) = \rho_B + (1 - \rho_B)\,\rho_S\),
\(\rho_B = 1/(1 + \nu_B)\): the same spread for counts. It is the range of a benchmark of one subset, the commonest
kind; a benchmark of several subsets falls closer to the average. It is much wider than the interval: the interval says
how precisely the average is known, the range how far single benchmarks spread around it.

## How well it works

A simulation on srbf's own set sizes (the 29 catalogs of the 2026-09 release and the 192 problem sets of the next
release) fixed the sets' true rates in 96 settings (mean rates of 5, 30, 70 and 95 %, three spreads, four draws of
the sets) and drew the problems anew 100 times in each. The beta-binomial average missed by 0.7 to 1.1 points on
average; the log-odds average with the continuity correction missed by 5 to 6 points, and its interval held the true
value only 6 to 32 % of the time; the logit-normal model missed by 0.6 to 1.3 points and held it 88 to 89 % of the
time.

A second simulation tested the rate interval near the edges: mean rates of 0.2, 1, 5, 30, 95, 99 and 99.8 %, three
spreads and three draws of the sets on both set sizes, 60 redraws of the problems each. There, the beta fitted to the
sets' true rates depends on how small the smallest rates are, which no number of problems settles, so the interval was
scored against what the average estimates for these sets: the value of \(m\) where the expected score is zero, given
the sets' true rates and sizes ([White 1982](https://doi.org/10.2307/1912526)). The score-test interval held it 94 to
98 % of the time at every mean rate on both set sizes, and in at least 82 % of the redraws in every one of the 126
draws of the sets. The delta-method interval held it 91 to 99 % of the time, but at mean rates of 1 % or less and
99 % or more it was 4 to 11 times wider: a median of 19 points at a mean rate of 0.2 % on 29 sets. A version that kept
\(\nu\) fixed at its estimate held it 91 % of the time on average on 29 sets and as rarely as 86 % near the edges,
and a bootstrap that redrew the problems from the sets' posterior rates held it 81 to 98 % of the time, its intervals
too narrow.

### Three levels

The three-level average was simulated on two structures of subsets: 145 benchmarks of 255 subsets (95 benchmarks of
one subset, 25 of two, 12 of three, 8 of four, 3 of six and 2 of twelve; subsets of 1 to 1,000 problems, evenly spread
on a log scale, and one of 5,301), the size of the next release, and 29 benchmarks of 38 subsets. Each setting fixed the
subsets' true values in several draws and redrew the problems many times in each, and each interval was scored
against what it covers: the weighted average of these subsets' true values, with the weights the interval used, for
means, and for rates what the average estimates for these subsets, the beta-binomial fit to the effective counts of
every redraw together (the maximum of the expected likelihood, [White 1982](https://doi.org/10.2307/1912526)).

**Means.** With a spread within subsets of 1, five settings of the two spreads (\(\tau_B, \tau_S\) of 1 and 0, 1 and
0.3, 1 and 1, 0.3 and 1, 0 and 1), four draws each and 200 redraws: the interval held its value 90 to 95 % of the time
per setting on 145 benchmarks and 92 to 94 % on 29, and in at least 86 % of the redraws in every draw. The two-level
average on the same subset sizes, every subset a benchmark of its own, held it 92 to 93 % of the time, so the shortfall
is the two-level average's and not the third level's; leaving out the subsets of fewer than five problems, whose own
variance rests on one to three degrees of freedom, raised it to 93.5 and 95 %. The average matched three-level
restricted maximum likelihood ([Konstantopoulos 2011](https://doi.org/10.1002/jrsm.35)) to within 0.4 % to 26 % of
the interval's half-width on average per setting; the largest differences came where the spread between benchmarks
is small next to the spread within them, where Paule and Mandel's \(\tau_B^2\) comes out larger than restricted
maximum likelihood's (0.19 against 0.14 for a true 0.09 on 145 benchmarks). \(\tau_S^2\) averaged 0.91 to 1.09 for
a true 1 and 0.10 to 0.11 for 0.09, as restricted maximum likelihood's did (0.88 to 1.06, and 0.10 to 0.11).

**Rates.** Mean rates of 0.2, 1, 5, 30, 95, 99 and 99.8 %, three spreads (\(\nu_B = 3\) with subsets alike;
\(\nu_B = \nu_S = 3\); \(\nu_B = 20,\ \nu_S = 1\), subsets that differ more than benchmarks), three draws each and 60
redraws: the score-test interval held its value 94 % of the time over all 126 draws; per mean rate, 97 to 98.5 % from
1 to 99.8 % and 86 % at 0.2 % on 145 benchmarks, and 82 to 98 % on 29. The two-level average of the same redraws, every subset a
benchmark of its own, held its own value 92 % of the time. Five draws fell to 10 to 28 % for both, identically: draws
in which nearly every subset sits at 0 % or 100 % in most redraws, so that a rare miss moves the value the interval is
scored against out of the narrow interval of the redraws without one. One more fell below 80 % for each (60 % with
three levels against 93 % with two, at a mean rate of 0.2 %, where the few subsets with any success made \(\rho_S\)
swing between redraws; 55 % with two levels against 97 % with three, at 99 %). The two-stage estimate of
\(m\) differed from the full model's, the nested beta-binomial fitted by maximum likelihood (each benchmark's
likelihood integrated over its rate), by 0.001 to 0.48 points on average per mean rate, at most 2.3 points: 0 to 19 %
of the interval's half-width on average, and more than the half-width in one of 504 fits. \(\rho_S\) by Williams'
rule came out at 0.23 and 0.51 for true values of 0.25 and 0.5 at a mean rate of 30 % (15 redraws on 145 benchmarks),
where the maximum-likelihood fit with every benchmark's rate fitted alongside gave 0.17 and 0.40; at a mean rate of
5 % both gave about 0.17 for 0.25, since most subsets then hold no success at all.

## In the package

The averaging is `srbf.aggregate`. `average(problems, kind, benchmarks=..., subsets=...)` takes, for each catalog, the
values of each problem's runs, each catalog's benchmark (`{catalog: benchmark}`; a catalog it does not name is a
benchmark of its own) and each problem's subset (`{catalog: [subset of each problem]}`; by default its catalog), and a
kind (`"mean"`, `"rate"`, `"ratio"` or `"median"`). It returns the average with its interval, the range for one more
benchmark, each benchmark's weight and each subset's. `nest` builds the benchmarks > subsets > problems structure,
`combine_nested` works from per-subset sums grouped by benchmark (`nest_sets` groups per-catalog sums), and `combine`
from per-set sums, each set a benchmark of one subset. `within_tau2`, `within_rho`, `effective_sets` and
`nested_set_weights` are the steps above. `srbf.suites.catalog_benchmarks` reads each catalog's benchmark from its
metadata. `srbf analyze` and its views (`leaderboard`, `scaling_table`, `per_benchmark_table`, `export_data`,
[Results](results.md)) average over the benchmarks of a run with it, a problem's subset from the `subset` column of its
rows where they have one, so their numbers are the explorer's; `per_benchmark_table(..., by="catalog")` splits by
catalog instead. The explorer computes the same arithmetic in the browser (`results-site/explorer_v2.js`), and the
site's test suite checks that the two agree on random structures. `scripts/averaging_simulation.py` runs the
simulations of [How well it works](#how-well-it-works). `bootstrap_report` remains for a bootstrap over the
expressions of a single benchmark.

## Sources

- Brown, L. D., Cai, T. T. and DasGupta, A. (2001). Interval estimation for a binomial proportion. *Statistical
  Science* 16(2), 101–133. [doi:10.1214/ss/1009213286](https://doi.org/10.1214/ss/1009213286)
- DerSimonian, R. and Laird, N. (1986). Meta-analysis in clinical trials. *Controlled Clinical Trials* 7(3),
  177–188. [doi:10.1016/0197-2456(86)90046-2](https://doi.org/10.1016/0197-2456(86)90046-2)
- Cheung, M. W.-L. (2014). Modeling dependent effect sizes with three-level meta-analyses: a structural equation
  modeling approach. *Psychological Methods* 19(2), 211–229. [doi:10.1037/a0032968](https://doi.org/10.1037/a0032968)
- Griffiths, D. A. (1973). Maximum likelihood estimation for the beta-binomial distribution and an application to
  the household distribution of the total number of cases of a disease. *Biometrics* 29(4), 637–648.
  [doi:10.2307/2529131](https://doi.org/10.2307/2529131)
- Higgins, J. P. T., Thompson, S. G. and Spiegelhalter, D. J. (2009). A re-evaluation of random-effects
  meta-analysis. *Journal of the Royal Statistical Society, Series A* 172(1), 137–159.
  [doi:10.1111/j.1467-985X.2008.00552.x](https://doi.org/10.1111/j.1467-985X.2008.00552.x)
- Huber, P. J. (1967). The behavior of maximum likelihood estimates under nonstandard conditions. *Proceedings of
  the Fifth Berkeley Symposium on Mathematical Statistics and Probability* 1, 221–233.
- Kish, L. (1965). *Survey Sampling*. Wiley, New York.
- Konstantopoulos, S. (2011). Fixed effects and variance components estimation in three-level meta-analysis.
  *Research Synthesis Methods* 2(1), 61–76. [doi:10.1002/jrsm.35](https://doi.org/10.1002/jrsm.35)
- Kuss, O. (2015). Statistical methods for meta-analyses including information from studies without any events: add
  nothing to nothing and succeed nevertheless. *Statistics in Medicine* 34(7), 1097–1116.
  [doi:10.1002/sim.6383](https://doi.org/10.1002/sim.6383)
- Neyman, J. and Scott, E. L. (1948). Consistent estimates based on partially consistent observations.
  *Econometrica* 16(1), 1–32. [doi:10.2307/1914288](https://doi.org/10.2307/1914288)
- Paule, R. C. and Mandel, J. (1982). Consensus values and weighting factors. *Journal of Research of the National
  Bureau of Standards* 87(5), 377–385. [doi:10.6028/jres.087.022](https://doi.org/10.6028/jres.087.022)
- Rao, C. R. (1948). Large sample tests of statistical hypotheses concerning several parameters with applications to
  problems of estimation. *Mathematical Proceedings of the Cambridge Philosophical Society* 44(1), 50–57.
  [doi:10.1017/S0305004100023987](https://doi.org/10.1017/S0305004100023987)
- Rice, K., Higgins, J. P. T. and Lumley, T. (2018). A re-evaluation of fixed effect(s) meta-analysis. *Journal of
  the Royal Statistical Society, Series A* 181(1), 205–227.
  [doi:10.1111/rssa.12275](https://doi.org/10.1111/rssa.12275)
- Skellam, J. G. (1948). A probability distribution derived from the binomial distribution by regarding the
  probability of success as variable between the sets of trials. *Journal of the Royal Statistical Society,
  Series B* 10(2), 257–261. [doi:10.1111/j.2517-6161.1948.tb00014.x](https://doi.org/10.1111/j.2517-6161.1948.tb00014.x)
- Sweeting, M. J., Sutton, A. J. and Lambert, P. C. (2004). What to add to nothing? Use and avoidance of continuity
  corrections in meta-analysis of sparse data. *Statistics in Medicine* 23(9), 1351–1375.
  [doi:10.1002/sim.1761](https://doi.org/10.1002/sim.1761)
- Van den Noortgate, W., López-López, J. A., Marín-Martínez, F. and Sánchez-Meca, J. (2013). Three-level
  meta-analysis of dependent effect sizes. *Behavior Research Methods* 45(2), 576–594.
  [doi:10.3758/s13428-012-0261-6](https://doi.org/10.3758/s13428-012-0261-6)
- Veroniki, A. A., Jackson, D., Viechtbauer, W. et al. (2016). Methods to estimate the between-study variance and its
  uncertainty in meta-analysis. *Research Synthesis Methods* 7(1), 55–79.
  [doi:10.1002/jrsm.1164](https://doi.org/10.1002/jrsm.1164)
- White, H. (1982). Maximum likelihood estimation of misspecified models. *Econometrica* 50(1), 1–25.
  [doi:10.2307/1912526](https://doi.org/10.2307/1912526)
- Williams, D. A. (1982). Extra-binomial variation in logistic linear models. *Applied Statistics* 31(2), 144–148.
  [doi:10.2307/2347977](https://doi.org/10.2307/2347977)
- Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *Journal of the
  American Statistical Association* 22(158), 209–212.
  [doi:10.1080/01621459.1927.10502953](https://doi.org/10.1080/01621459.1927.10502953)
- Woodruff, R. S. (1952). Confidence intervals for medians and other position measures. *Journal of the American
  Statistical Association* 47(260), 635–646.
  [doi:10.1080/01621459.1952.10483443](https://doi.org/10.1080/01621459.1952.10483443)
