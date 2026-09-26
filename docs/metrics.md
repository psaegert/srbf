# Metrics

Every metric is computed from a run's stored predictions by `srbf.derive_metrics`, the same way for
every method. This page defines each column it adds. For how to call it, see
[Results](results.md#deriving-metrics).

**Notation.** A problem has support targets \(y\) and validation targets \(y_{\mathrm{val}}\),
both without noise, and a ground truth with skeleton \(\bar\tau\): its expression in prefix notation
(operator first, `+ x1 sin x2` for `x1 + sin(x2)`) with every numeric constant replaced by
`<constant>`, simplified by the run's SimpliPy engine. An exponent is a constant like any other, so
`x1**2` and `x1**3` share the skeleton `pow x1 <constant>`. The method's
prediction has values \(\hat y\) and \(\hat y_{\mathrm{val}}\) on the two sets and the skeleton
\(\hat\tau\). Columns ending in `_fit` are computed on the support points and columns ending in
`_val` on the validation points. Metrics are always computed against the noise-free targets, also
when the method was given noisy ones: the question is whether the ground truth was found, not whether the
noise was reproduced.

## Success metrics and analysis metrics

The columns fall into two kinds, and a failed prediction means something different in each.

A **success metric** says whether a problem was solved, and it is defined for every problem:
`numeric_recovery_*`, `numeric_recovery_relative_*`, `symbolic_recovery`, and the share of
problems that got a prediction at all (`prediction_success`). Whenever a method errors or returns no
result it has failed the problem, and the metric is 0.

An **analysis metric** describes a prediction: how well it fits (`fvu_*`, `log10_fvu_*`, `r2_*`),
what it looks like (lengths, constants, nestedness, description lengths) and how close it comes
to the ground truth (token overlap, edit distances). What a failed problem counts there depends on the
range of the metric.

Where the range has a worst value, a failed problem takes it, so that a method cannot raise its
mean by failing on the hard expressions. These are the overlaps, which are shares in \([0, 1]\):

| column | a failed problem counts |
|---|---|
| `f1_score`, `precision_score`, `recall_score` | 0 |
| `f1_score_unique_variables`, `precision_unique_variables`, `recall_unique_variables` | 0 |

The table is `srbf.result_processing.WORST_VALUE`. The value is the end of the metric's range,
not the score of some stand-in prediction. To read these columns over the predictions that were made
instead, call `derive_metrics(..., impute_failed=False)`: a failed problem then has no value in
them either. The results explorer offers the same choice.

No other analysis metric is filled in. Most have no worst value, because a prediction can be
arbitrarily bad: \(R^2\) has no lower bound, a predicted expression no largest length, a ratio of
lengths lies in \([0, \infty)\) with its ideal at 1. A failed problem has no value there (`None`, or NaN in the
numeric columns), and none is filled in. Summaries of these columns describe the predictions a method
gave, so read them next to the share of problems it has a prediction for: the
[results explorer](https://psaegert.github.io/srbf/) draws a point hollow when that share is below
90 %.

A problem is failed when the method reports no success (`prediction_success` is false) or, where
it does not report, returned no expression. An expression that could be read and then failed to
evaluate, because it divides by zero on the data, say, is a failed problem like any other.

Placeholder rows, written when a problem could not be produced at all, carry no metric and are
left out of every summary ([Results](results.md#failures-and-placeholders)).

## Numeric fit

### `fvu_fit`, `fvu_val`

The fraction of variance unexplained,

$$
\operatorname{FVU}(y, \hat y) = \frac{\sum_i (y_i - \hat y_i)^2}{\sum_i (y_i - \bar y)^2} .
$$

0 is a perfect fit, 1 is as good as predicting the mean, and the measure does not depend on the
scale of the targets. Both sums are rescaled before they are squared, so very large and very small
targets neither overflow nor underflow. A prediction with a non-finite value has FVU \(\infty\). A
constant target has FVU 0 when it is matched exactly and \(\infty\) otherwise. FVU is never NaN.

### `log10_fvu_fit`, `log10_fvu_val`

\(\log_{10}\) of the FVU: \(-2\) means 99 % of the variance is explained, about \(-7\) is float32
precision. It is floored at the float64 epsilon, \(\log_{10} 2^{-52} \approx -15.65\)
(`srbf.metrics.numeric.LOG10_FVU_FLOOR`): below it the unexplained variance is smaller than about one rounding
unit of the variance it is divided by, so the value is rounding noise, and the same formula written two ways
lands anywhere from \(-16\) to \(-320\), or at \(-\infty\) when the arithmetic agrees bit for bit. On the
floor these are one value, and an exact fit counts in a mean like any other fit. A prediction with a
non-finite value is \(+\infty\), which a mean leaves out: report the median, or the mean next to the rate of
usable predictions.

### `r2_fit`, `r2_val`

\(R^2 = 1 - \operatorname{FVU}\): 1 is a perfect fit, 0 is as good as predicting the mean, and an
prediction worse than that is negative, without a lower bound. A prediction with a non-finite value has
\(R^2 = -\infty\). One diverging prediction would decide the mean of a whole catalog, so summarize
this column by its median: `srbf analyze` does, and with `bootstrap_report` pass
`aggregate=np.nanmedian, reduce=np.nanmedian`.

### `numeric_recovery_fit`, `numeric_recovery_val`

$$
\mathbb{1}\big[\operatorname{FVU} \le \varepsilon_{32}\big], \qquad \varepsilon_{32} = 2^{-23} \approx 1.19 \times 10^{-7} .
$$

The prediction reproduces the targets to float32 precision. On the validation points this is the
headline metric (vNRR): an expression either is the ground truth on its domain, numerically, or it is not,
and a close approximation does not count. A support recovery (fNRR) above the validation recovery
means fitting without generalizing.

`only_approx_fvu_fit`, `only_approx_fvu_val`, `only_approx_log10_fvu_fit` and
`only_approx_log10_fvu_val` repeat the FVU columns with \(-\infty\) wherever
the problem is numerically recovered, which isolates the quality of the predictions that are not exact.

### `numeric_recovery_relative_fit`, `numeric_recovery_relative_val`

Recovery relative to the reference expression. Some catalogs hold measurements: the targets are
observations, and the accepted expression \(f_{\mathrm{ref}}\) explains them only up to measurement error.
No expression reaches float32 precision on such data, and numeric recovery is 0 for every method.
The relative criterion asks instead whether the prediction fits the targets at least as well as the
accepted expression does:

$$
\mathbb{1}\Big[\operatorname{FVU}(y, \hat y) \le \max\big(\operatorname{FVU}(y, y_{\mathrm{ref}}),\; \varepsilon_{32}\big)\Big] .
$$

`reference_fvu_fit` and `reference_fvu_val` hold \(\operatorname{FVU}(y, y_{\mathrm{ref}})\), the
ground truth's own misfit. For example, if the accepted expression leaves \(10^{-3}\) of the variance of a measured
data set unexplained, a prediction with FVU \(8 \times 10^{-4}\) counts as recovered and one with
\(2 \times 10^{-3}\) does not. Where the targets are computed from the ground truth itself, the ground truth's misfit
is 0 and the criterion is numeric recovery exactly. That is the case for all 29 catalogs of the
srbf suite, so there the two columns agree on every problem.

## Symbolic agreement

All columns of this group compare the **judged skeleton** of the prediction with \(\bar\tau\),
and both come from one function. An expression is first brought into the engine's canonical form
*with its numbers*, because that is where it shows what it is: `x2 * x4 / (c * x4 ** 3)` cancels to
`x2 / (c * x4 ** 2)` only while the `3` is a number, and `pow(u, 0.5)` becomes `rootn(u, 2)`. Then
every number is masked, `pi` and `e` included, and the masked form is simplified and masked again
until nothing changes, so that a number simplification writes itself is a constant too:
`x1 * x1` becomes `pow x1 2` and then `pow x1 <constant>`, the skeleton of `x1 ** 2`. A re-ordered
sum, an unsimplified sub-term or a longer spelling of the same function therefore does not count
as a difference, whichever side wrote it.

No simplifier is complete, so the judge looks in a second place as well: the two skeletons as they
were written, settled the same way. Agreement in either place is a sound witness that the two
expressions are one family, because simplification never changes the function and masking only
forgets numbers. Agreement at a [stricter level of masking](#symbolic_recovery_mask_fittable-symbolic_recovery_mask_none)
is a witness too. `skeleton_simplified` holds \(\bar\tau\) in the judged form. When simplification changed the prediction, the skeleton as the method wrote it
is kept in `predicted_skeleton_prefix_as_emitted`.

### `symbolic_recovery`

\(\mathbb{1}[\hat\tau = \bar\tau]\): the two skeletons are the same token sequence. Constants are
masked on both sides, so the structure has to match and the values of the constants do not enter;
whether the constants are right is what numeric recovery measures.

### `symbolic_recovery_mask_fittable`, `symbolic_recovery_mask_none`

The same question with fewer numbers masked. Each level implies the one before it, and a failed
problem is a miss at all three.

| column | masked | has to be the ground truth's |
|---|---|---|
| `symbolic_recovery` | every number | the structure |
| `symbolic_recovery_mask_fittable` | the fittable constants: coefficients and offsets | the structure and its numbers: exponents, root indices |
| `symbolic_recovery_mask_none` | nothing | the structure, its numbers and the constants |

For the ground truth `x1 ** 2 + 1.5 * x2`, the prediction `x1 * x1 + 1.4 * x2` is recovered at the first two
levels, `x1 ** 3 + 1.5 * x2` at the first only, and `x1 ** 2.0000001 + 1.5 * x2`, an exponent the
method left unsnapped, at the first only, although it fits to float32 precision. Which numbers are
fittable is decided by the engine (`engine.mask(expression, 'fittable')`).

A fitted constant is right when the prediction reproduces the ground truth: `symbolic_recovery_mask_none` is
`symbolic_recovery_mask_fittable` together with `numeric_recovery_val`. Numbers are not compared
token by token, because the canonical form spreads a rational through the expression (`1.5 * x2`
is `(3 * x2) / 2`). Both columns need an engine.

### `f1_score`

Precision, recall and \(F_1\) between the *sets* of distinct tokens of \(\hat\tau\) and
\(\bar\tau\): did the prediction use the right operators and variables at all? Order and multiplicity
are ignored.

`precision_score` and `recall_score` are the two parts of `f1_score`: the share of the prediction's
distinct tokens that the ground truth uses, and the share of the ground truth's distinct tokens that the prediction
uses. The precision of an empty prediction is 0 by definition.

### `edit_distance`, `edit_distance_norm`

The Levenshtein distance between the two prefix token sequences: the number of token insertions,
deletions and substitutions that turn one into the other. `edit_distance_norm` divides it by the
length of the longer sequence, which puts it in \([0, 1]\). Both describe the predictions that were
made: a failed problem has no value in either.

### `zss_edit_distance`

The Zhang-Shasha edit distance between the two expression trees. Inserting or deleting a node costs
the length of its label, and relabeling costs the character edit distance between the labels:
`sin` to `cos` costs 3, `x1` to `x2` costs 1. Unlike the sequence distance it follows the structure
of the expression.

### Variables

`unique_variables` and `predicted_unique_variables` list the distinct variables of the ground truth and of
the prediction, `n_variables` counts the ground truth's.
`precision_unique_variables`, `recall_unique_variables` and `f1_score_unique_variables` compare the
two sets: precision falls when the prediction uses a variable the ground truth ignores, recall when it leaves out
one the ground truth needs.

## Length and complexity

| column | meaning |
|---|---|
| `skeleton_length` | number of prefix tokens of \(\bar\tau\) |
| `predicted_skeleton_prefix_length` | number of prefix tokens of \(\hat\tau\) |
| `skeleton_length_ratio` | predicted over true length; 1 is as long as the ground truth |
| `n_constants`, `predicted_n_constants` | number of `<constant>` tokens in \(\bar\tau\) and \(\hat\tau\) |
| `n_constants_delta` | predicted minus true number of constants; positive means excess free parameters |
| `total_nestedness`, `predicted_total_nestedness` | over every chain of \(m\) directly nested unary operators, the excess \(m - 1\), summed: `sin(cos(x))` counts 1, `sin(cos(exp(x)))` counts 2, `sin(x) + cos(x)` counts 0 |

### `predicted_mdl`, `ground_truth_mdl`, `mdl_ratio`

The description length of an expression, with its constants, as priced by the SimpliPy engine
(`engine.complexity`), in thousandths of a bit. Unlike a token count it charges a long constant
more than a short one: with `acj-5-4-llm`, `x1` costs 6 bits, `sin(x1)` 12 bits, `x1 + 1.5` 13.6
bits and `x1 + 3.14159` 30.8 bits. `mdl_ratio` is the prediction's length over the ground truth's; 1 is as
long as the ground truth. These columns need an engine and are added only when `derive_metrics` is given
one.

## What the method reports itself

These are raw columns, written by the adapter and not recomputed:

| column | meaning |
|---|---|
| `prediction_success` | the method returned an expression that could be parsed and evaluated |
| `fit_time` | seconds for the problem, measured around the fit; one-time loading is not included |
| `predicted_score` | the score by which the method chose its prediction, comparable only within one method |
| `predicted_log_prob` | the log-probability of the prediction under a generative model |
| `predicted_pareto_rank` | the prediction's rank on the method's own front of fit against length; `None`, or \(-1\) for Flash-ANSR, when the method does not rank that way |

## Names on the results explorer

The [results explorer](https://psaegert.github.io/srbf/) calls the success metrics *rates* and
follows the same rules: a failure counts 0 in a rate and the worst value where a metric has one,
and every other metric describes the predictions that were made. It shows the same quantities
under reader names, which say *Prediction* and *Ground Truth* for the two expressions:

| on the explorer | column |
|---|---|
| Numeric Recovery, Validation / Support; short vNRR / fNRR | `numeric_recovery_val`, `numeric_recovery_fit` |
| Fits as Well as the Ground Truth | `numeric_recovery_relative_*` |
| Successful Prediction Rate | the mean of `prediction_success` |
| Symbolic Recovery: Structure, + Exponents, + All Numbers; short SRRs / SRRe / SRRa | `symbolic_recovery`, `symbolic_recovery_mask_fittable`, `symbolic_recovery_mask_none` |
| Raw Symbolic Recovery; short SRRr | the equality of the two skeletons as written, every number masked, without simplification |
| MDL Ratio, Token Count Ratio, both prediction over ground truth | `mdl_ratio`, `skeleton_length_ratio` |
| Token Overlap, Variable Overlap | `f1_score`, `precision_score`, `recall_score` and the `*_unique_variables` columns |
| Levenshtein Edit Distance, Tree Edit Distance | `edit_distance`, `edit_distance_norm`, `zss_edit_distance` |
| Function Nesting | `total_nestedness`, `predicted_total_nestedness` |
