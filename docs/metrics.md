# Metrics

Every metric is computed from a run's stored predictions by `srbf.derive_metrics`, the same way for
every method. This page defines each column it adds. For how to call it, see
[Results](results.md#deriving-metrics).

**Notation.** A problem has support targets \(y\) and validation targets \(y_{\mathrm{val}}\),
both without noise, and a law with skeleton \(\bar\tau\): the law's expression in prefix notation
(operator first, `+ x1 sin x2` for `x1 + sin(x2)`) with every numeric constant replaced by
`<constant>`, simplified by the run's SimpliPy engine. An exponent is a constant like any other, so
`x1**2` and `x1**3` share the skeleton `pow x1 <constant>`. The method's
answer has values \(\hat y\) and \(\hat y_{\mathrm{val}}\) on the two sets and the skeleton
\(\hat\tau\). Columns ending in `_fit` are computed on the support points and columns ending in
`_val` on the validation points. Metrics are always computed against the noise-free targets, also
when the method was given noisy ones: the question is whether the law was found, not whether the
noise was reproduced.

## Rates and diagnostics

A **rate** is defined for every problem, and a problem without a usable prediction is a miss:
`numeric_recovery_*`, `numeric_recovery_relative_*`, `symbolic_recovery`, and `r2_*` (which is 0 for
a failed prediction). A **diagnostic** exists only where a prediction exists and is `None`
otherwise, so summaries of it describe the successful predictions: lengths, edit distances, token
overlap, description lengths. `fvu_*` of a failed prediction is infinite.

The numeric columns read the predicted values, the symbolic columns read the stored skeleton. An
answer that could be read as an expression and then failed to evaluate, because it divides by zero
on the data, say, is a numeric miss and still has a skeleton to compare.

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
precision. An exact fit is \(-\infty\). Summaries of this column take the finite values, so report
it next to the recovery rate, which counts the exact fits.

### `r2_fit`, `r2_val`

\(R^2 = 1 - \operatorname{FVU}\), clipped to \([0, 1]\); 0 for a failed or divergent prediction.

### `numeric_recovery_fit`, `numeric_recovery_val`

$$
\mathbb{1}\big[\operatorname{FVU} \le \varepsilon_{32}\big], \qquad \varepsilon_{32} = 2^{-23} \approx 1.19 \times 10^{-7} .
$$

The prediction reproduces the targets to float32 precision. On the validation points this is the
headline metric (vNRR): an expression either is the law on its domain, numerically, or it is not,
and a close approximation does not count. A support recovery (fNRR) above the validation recovery
means fitting without generalizing.

`only_approx_fvu_fit`, `only_approx_fvu_val`, `only_approx_log10_fvu_fit` and
`only_approx_log10_fvu_val` repeat the FVU columns with \(-\infty\) wherever
the problem is numerically recovered, which isolates the quality of the answers that are not exact.

### `numeric_recovery_relative_fit`, `numeric_recovery_relative_val`

Recovery relative to the reference law. Some catalogs hold measurements: the targets are
observations, and the accepted law \(f_{\mathrm{ref}}\) explains them only up to measurement error.
No expression reaches float32 precision on such data, and numeric recovery is 0 for every method.
The relative criterion asks instead whether the prediction fits the targets at least as well as the
accepted law does:

$$
\mathbb{1}\Big[\operatorname{FVU}(y, \hat y) \le \max\big(\operatorname{FVU}(y, y_{\mathrm{ref}}),\; \varepsilon_{32}\big)\Big] .
$$

`reference_fvu_fit` and `reference_fvu_val` hold \(\operatorname{FVU}(y, y_{\mathrm{ref}})\), the
law's own misfit. For example, if the accepted law leaves \(10^{-3}\) of the variance of a measured
data set unexplained, an answer with FVU \(8 \times 10^{-4}\) counts as recovered and one with
\(2 \times 10^{-3}\) does not. Where the targets are computed from the law itself, the law's misfit
is 0 and the criterion is numeric recovery exactly. That is the case for all 29 catalogs of the
srbf suite, so there the two columns agree on every problem.

## Symbolic agreement

All columns of this group compare the **judged skeleton** of the prediction with \(\bar\tau\). The
prediction is brought into the engine's canonical form, its constants are masked, and the masked
form is simplified once more, which is the treatment \(\bar\tau\) gets. A re-ordered sum or an
unsimplified sub-term therefore does not count as a difference. `skeleton_simplified`
holds \(\bar\tau\). When simplification changed the prediction, the skeleton as the method wrote it
is kept in `predicted_skeleton_prefix_as_emitted`.

### `symbolic_recovery`

\(\mathbb{1}[\hat\tau = \bar\tau]\): the two skeletons are the same token sequence. Constants are
masked on both sides, so the structure has to match and the values of the constants do not enter;
whether the constants are right is what numeric recovery measures.

### `f1_score`

Precision, recall and \(F_1\) between the *sets* of distinct tokens of \(\hat\tau\) and
\(\bar\tau\): did the answer use the right operators and variables at all? Order and multiplicity
are ignored.

### `edit_distance`

The Levenshtein distance between the two prefix token sequences: the number of token insertions,
deletions and substitutions that turn one into the other.

### `zss_edit_distance`

The Zhang-Shasha edit distance between the two expression trees. Inserting or deleting a node costs
the length of its label, and relabeling costs the character edit distance between the labels:
`sin` to `cos` costs 3, `x1` to `x2` costs 1. Unlike the sequence distance it follows the structure
of the expression.

### Variables

`unique_variables` and `predicted_unique_variables` list the distinct variables of the law and of
the answer, `n_variables` counts the law's.
`precision_unique_variables`, `recall_unique_variables` and `f1_score_unique_variables` compare the
two sets: precision falls when the answer uses a variable the law ignores, recall when it leaves out
one the law needs.

## Length and complexity

| column | meaning |
|---|---|
| `skeleton_length` | number of prefix tokens of \(\bar\tau\) |
| `predicted_skeleton_prefix_length` | number of prefix tokens of \(\hat\tau\) |
| `skeleton_length_ratio` | predicted over true length; 1 is as long as the law |
| `n_constants`, `predicted_n_constants` | number of `<constant>` tokens in \(\bar\tau\) and \(\hat\tau\) |
| `n_constants_delta` | predicted minus true number of constants; positive means excess free parameters |
| `total_nestedness`, `predicted_total_nestedness` | over every chain of \(m\) directly nested unary operators, the excess \(m - 1\), summed: `sin(cos(x))` counts 1, `sin(cos(exp(x)))` counts 2, `sin(x) + cos(x)` counts 0 |

### `predicted_mdl`, `ground_truth_mdl`, `mdl_ratio`

The description length of an expression, with its constants, as priced by the SimpliPy engine
(`engine.complexity`), in thousandths of a bit. Unlike a token count it charges a long constant
more than a short one: with `acj-5-4-llm`, `x1` costs 6 bits, `sin(x1)` 12 bits, `x1 + 1.5` 13.6
bits and `x1 + 3.14159` 30.8 bits. `mdl_ratio` is the prediction's length over the law's; 1 is as
long as the law. These columns need an engine and are added only when `derive_metrics` is given
one.

## What the method reports itself

These are raw columns, written by the adapter and not recomputed:

| column | meaning |
|---|---|
| `prediction_success` | the method returned an expression that could be parsed and evaluated |
| `fit_time` | seconds for the problem, measured around the fit; one-time loading is not included |
| `predicted_score` | the score by which the method chose its answer, comparable only within one method |
| `predicted_log_prob` | the log-probability of the answer under a generative model |
| `predicted_pareto_rank` | the answer's rank on the method's own front of fit against length; `None`, or \(-1\) for Flash-ANSR, when the method does not rank that way |

## Names on the results explorer

The [results explorer](https://psaegert.github.io/srbf/) shows the same quantities under reader
names: *Numeric recovery (vNRR)* is `numeric_recovery_val`, *fNRR* is `numeric_recovery_fit`,
*Expression length ratio* is `skeleton_length_ratio`, *Prediction success rate* is the mean of
`prediction_success`, and *Exact skeleton match (raw)* is the equality of the skeletons as written,
without simplification.
