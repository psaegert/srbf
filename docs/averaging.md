# How results are averaged

The [results explorer](https://psaegert.github.io/srbf/) reports every value as an average over problem sets
(catalogs), with a 95 % interval. This page describes how, why, and what the method rests on.
`srbf analyze` ([Results](results.md)) averages the same way; [In the package](#in-the-package) says where the code is.

## Why average over problem sets

The problem sets differ a lot, in difficulty and in size. In srbf's suite one catalog, `erbench-syneq`, holds 5,301
of the 6,660 problems, so an average over all problems is mostly an average over that one catalog. The explorer
therefore averages in three steps:

1. A problem's value is the mean of its runs, so every problem counts once.
2. A problem set is summarised by the mean \(\bar{x}_s\) and the variance \(\sigma_s^2\) of its \(n_s\) problems'
   values.
3. The problem sets are combined, each weighted by how precisely its own mean is known and by how much problem
   sets differ from each other.

Step 3 treats each problem set as its own population with its own true value, as a meta-analysis treats studies of
the same quantity in different populations. That is the random-effects average
([DerSimonian and Laird 1986](https://doi.org/10.1016/0197-2456(86)90046-2)).

## Values that are not rates

$$
\hat{\theta} = \frac{\sum_s w_s\, \bar{x}_s}{\sum_s w_s},
\qquad
w_s = \frac{1}{\tau^2 + \sigma_s^2 / n_s} .
$$

The between-set variance \(\tau^2\) is estimated with Paule and Mandel's estimator
([Paule and Mandel 1982](https://doi.org/10.6028/jres.087.022)), which a review of the estimators recommends
([Veroniki et al. 2016](https://doi.org/10.1002/jrsm.1164)). A set whose problems all have the same value, or that
has one problem, has no variance of its own and takes the variance pooled over the other sets.

Problem sets differ far more than their own sampling noise, so \(\tau^2\) dominates the weights: every set counts
about the same whatever its size, and a set of a few problems counts less, because its own mean is uncertain. On the
2026-09 release, `erbench-syneq` (79 % of the problems) carries about 5 % of the weight of an average over the 29
catalogs, against 3.4 % for an equal share, and a catalog of one or two problems about 2 %.

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

## One problem set

A single problem set is shown as its own mean with the \(t\) interval over its problems, and a rate as its plain
rate \(k/n\) with the Wilson interval ([Wilson 1927](https://doi.org/10.1080/01621459.1927.10502953)), whose coverage
stays close to 95 % even for a few problems ([Brown, Cai and DasGupta 2001](https://doi.org/10.1214/ss/1009213286)).

## Medians and distributions

Medians and distributions weigh each problem set the way the average does: a set of \(n_s\) problems counts
\(n_s / (n_s + n_0)\), with \(n_0 = \sigma^2 / \tau^2\) from the average's fit, and within a set every problem counts
once. The interval of a median follows Woodruff's method
([Woodruff 1952](https://doi.org/10.1080/01621459.1952.10483443)): the share of each set below the median is averaged
over the sets like any value, and the interval of that share is mapped back through the distribution.

## Intervals

The 95 % interval is about the selected problem sets themselves: the weights (for rates, \(\nu\)) are held at their
estimates and only the sampling of problems within each set counts. A value is then a weighted average of these sets'
own values, the reading of such averages by [Rice, Higgins and Lumley (2018)](https://doi.org/10.1111/rssa.12275);
problem sets in general would add the spread between sets to the interval.

$$
\hat{\theta} \pm t_{0.975}\, \frac{\sqrt{\sum_s w_s^2\, \sigma_s^2 / n_s}}{\sum_s w_s},
$$

with \(t\) on \(\sum_s (n_s - 1)\) degrees of freedom. For rates the same rule goes through the model's estimating
equation, as a sandwich variance (Huber 1967): the estimate of \(m\) solves \(\sum_s U_s = 0\) for the sets' scores
\(U_s\), so its variance is \(\sum_s (\partial U_s / \partial k_s)^2 \operatorname{Var}(k_s) / (\sum_s \partial U_s /
\partial m)^2\), with \(\operatorname{Var}(k_s)\) the binomial variance at the set's posterior rate. The interval is
formed on the log-odds scale, so it stays between 0 and 100 %.

## One more problem set

Where the value of one more problem set would fall is a separate range, shown once at least three sets are
selected: \(\hat{\theta} \pm t_{0.975,\,S-2} \sqrt{\tau^2 + 1/\sum_s w_s}\) for \(S\) sets
([Higgins, Thompson and Spiegelhalter 2009](https://doi.org/10.1111/j.1467-985X.2008.00552.x)), and for rates the
2.5 % to 97.5 % range of the fitted beta distribution. It is much wider than the interval: the interval says how
precisely the average is known, the range how far single problem sets spread around it.

## How well it works

A simulation on srbf's own set sizes (the 29 catalogs of the 2026-09 release and the 192 problem sets of the next
release) fixed the sets' true rates in 96 settings (mean rates of 5, 30, 70 and 95 %, three spreads, four draws of
the sets) and drew the problems anew 100 times in each. The beta-binomial average missed by 0.7 to 1.1 points on
average and its interval held the true value 93 to 96 % of the time; the log-odds average with the continuity
correction missed by 5 to 6 points and held it 6 to 32 % of the time; the logit-normal model missed by 0.6 to
1.3 points and held it 88 to 89 % of the time.

## In the package

The averaging is `srbf.aggregate`: `average(problems, kind)` takes, for each problem set, the values of each problem's
runs and a kind (`"mean"`, `"rate"`, `"ratio"` or `"median"`), and returns the average with its interval, the range
for one more set and each set's weight; `combine` works from per-set sums. `srbf analyze` and its views
(`leaderboard`, `scaling_table`, `per_benchmark_table`, `export_data`, [Results](results.md)) average over the
benchmarks of a run with it, each benchmark a problem set, so their numbers are the explorer's. The explorer computes
the same arithmetic in the browser (`results-site/explorer_v2.js`), and the site's test suite checks that the two agree.
`bootstrap_report` remains for a bootstrap over the expressions of a single benchmark.

## Sources

- Brown, L. D., Cai, T. T. and DasGupta, A. (2001). Interval estimation for a binomial proportion. *Statistical
  Science* 16(2), 101–133. [doi:10.1214/ss/1009213286](https://doi.org/10.1214/ss/1009213286)
- DerSimonian, R. and Laird, N. (1986). Meta-analysis in clinical trials. *Controlled Clinical Trials* 7(3),
  177–188. [doi:10.1016/0197-2456(86)90046-2](https://doi.org/10.1016/0197-2456(86)90046-2)
- Griffiths, D. A. (1973). Maximum likelihood estimation for the beta-binomial distribution and an application to
  the household distribution of the total number of cases of a disease. *Biometrics* 29(4), 637–648.
  [doi:10.2307/2529131](https://doi.org/10.2307/2529131)
- Higgins, J. P. T., Thompson, S. G. and Spiegelhalter, D. J. (2009). A re-evaluation of random-effects
  meta-analysis. *Journal of the Royal Statistical Society, Series A* 172(1), 137–159.
  [doi:10.1111/j.1467-985X.2008.00552.x](https://doi.org/10.1111/j.1467-985X.2008.00552.x)
- Huber, P. J. (1967). The behavior of maximum likelihood estimates under nonstandard conditions. *Proceedings of
  the Fifth Berkeley Symposium on Mathematical Statistics and Probability* 1, 221–233.
- Kuss, O. (2015). Statistical methods for meta-analyses including information from studies without any events: add
  nothing to nothing and succeed nevertheless. *Statistics in Medicine* 34(7), 1097–1116.
  [doi:10.1002/sim.6383](https://doi.org/10.1002/sim.6383)
- Paule, R. C. and Mandel, J. (1982). Consensus values and weighting factors. *Journal of Research of the National
  Bureau of Standards* 87(5), 377–385. [doi:10.6028/jres.087.022](https://doi.org/10.6028/jres.087.022)
- Rice, K., Higgins, J. P. T. and Lumley, T. (2018). A re-evaluation of fixed effect(s) meta-analysis. *Journal of
  the Royal Statistical Society, Series A* 181(1), 205–227.
  [doi:10.1111/rssa.12275](https://doi.org/10.1111/rssa.12275)
- Skellam, J. G. (1948). A probability distribution derived from the binomial distribution by regarding the
  probability of success as variable between the sets of trials. *Journal of the Royal Statistical Society,
  Series B* 10(2), 257–261. [doi:10.1111/j.2517-6161.1948.tb00014.x](https://doi.org/10.1111/j.2517-6161.1948.tb00014.x)
- Sweeting, M. J., Sutton, A. J. and Lambert, P. C. (2004). What to add to nothing? Use and avoidance of continuity
  corrections in meta-analysis of sparse data. *Statistics in Medicine* 23(9), 1351–1375.
  [doi:10.1002/sim.1761](https://doi.org/10.1002/sim.1761)
- Veroniki, A. A., Jackson, D., Viechtbauer, W. et al. (2016). Methods to estimate the between-study variance and its
  uncertainty in meta-analysis. *Research Synthesis Methods* 7(1), 55–79.
  [doi:10.1002/jrsm.1164](https://doi.org/10.1002/jrsm.1164)
- Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *Journal of the
  American Statistical Association* 22(158), 209–212.
  [doi:10.1080/01621459.1927.10502953](https://doi.org/10.1080/01621459.1927.10502953)
- Woodruff, R. S. (1952). Confidence intervals for medians and other position measures. *Journal of the American
  Statistical Association* 47(260), 635–646.
  [doi:10.1080/01621459.1952.10483443](https://doi.org/10.1080/01621459.1952.10483443)
