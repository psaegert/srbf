# The headline index

Most of srbf's metrics read one answer per problem: the formula a method picked. The headline index reads everything a
method returns. A run usually ends with several formulas, from short ones that fit roughly to long ones that fit
closely. srbf prices each of them and measures its error, and the **front area** says how much of the trade-off
between length and error the run covers. The **index** averages the front area over the time a user is willing to
wait, from 1 ms to 1000 s.

Neither number needs a true formula, so both are defined for every problem, including the data problems that have
none. Both depend only on the method's own formulas and on constants of the problem, so adding a method to a comparison
never changes another method's value. [In the package](#in-the-package) says where the code is.

## What a run hands in

A **run** is one method at one budget on one problem. Every problem is run once per budget, and the index assumes
that every method sees the same data for a problem: the same support points, validation points and noise. A method
that keeps its own front of formulas (a hall of fame) hands in that front; other methods hand in what they fitted,
such as their samples or beams. A failed run hands in nothing.

srbf's judge gives every candidate formula \(c\) two numbers:

- \(L(c)\), its **length** in bits: the description length the SimpliPy engine prices it at, constants included
  ([Metrics](metrics.md#predicted_mdl-ground_truth_mdl-mdl_ratio));
- \(F(c)\), its **error**: the FVU on the validation points, floored at \(\varepsilon = 2^{-52}\), the float64 machine
  epsilon, as for `log10_fvu_val` ([Metrics](metrics.md#log10_fvu_fit-log10_fvu_val)).

## The box

The candidates of a run are placed in a box with logarithmic axes, one box per problem:

| axis | from | to |
|---|---|---|
| length | \(L_{\mathrm{lo}}\), the price of the mean written as a number | \(L_{\mathrm{hi}} = 26\,n\) bits, for \(n\) support points |
| error | \(\varepsilon = 2^{-52}\) | 1, the FVU of the mean |

In the box a candidate has the coordinates

$$
u(c) = \frac{\log_2 L(c) - \log_2 L_{\mathrm{lo}}}{\log_2 L_{\mathrm{hi}} - \log_2 L_{\mathrm{lo}}},
\qquad
v(c) = \frac{\log_{10} F(c) - \log_{10} \varepsilon}{0 - \log_{10} \varepsilon},
$$

both clipped to \([0, 1]\). A formula shorter than the mean counts as \(u = 0\). Each edge is set by a definition, not
by a choice of weights: the box contains no exchange rate between length and error.

### The left edge: the mean

The simplest formula is a constant, and the constant that fits the support points best is their mean, whose FVU is 1
by definition. \(L_{\mathrm{lo}}\) is the price of that formula: the mean of the support targets written as the
shortest decimal that reads back as the same float64 number, priced by the same function as every prediction. With
the suite's engine `acj-5-4-llm`, a mean of `3.5` costs 5.6 bits and a mean of `2.3333333333333335` costs 59.5 bits.

### The right edge: writing the targets out

The FVU is the mean squared error of a formula over the variance \(\sigma^2\) of the targets,

$$
\mathrm{FVU} = \frac{\sum_i (y_i - \hat y_i)^2}{\sum_i (y_i - \bar y)^2} = \frac{\mathrm{MSE}}{\sigma^2},
$$

so an FVU of \(\varepsilon\) leaves a root mean squared residual of \(\sigma \sqrt{\varepsilon} = 2^{-26} \sigma\): the
targets are reproduced to 26 significant bits relative to their spread.

The right edge is what that costs without any formula, by writing the \(n\) support targets out, given their mean and
variance. For Gaussian targets, reaching a mean squared error \(D\) takes

$$
R(D) = \frac{1}{2} \log_2 \frac{\sigma^2}{D}
\quad\text{bits per target, so}\quad
R(\varepsilon \sigma^2) = \frac{1}{2} \log_2 \frac{1}{\varepsilon} = 26 \text{ bits per target}
$$

(the rate-distortion function of the Gaussian, Cover and Thomas 2006, Thm. 10.3.2). Read the other way,
\(D = \sigma^2\, 2^{-2R}\): every bit per target divides the mean squared error by four, the six-decibels-per-bit rule
of quantization ([Gray and Neuhoff 1998](https://doi.org/10.1109/18.720541)), and 26 bits take it from \(\sigma^2\)
to \(2^{-52} \sigma^2\).

The same rate suffices for every set of targets with that variance, because under squared error the Gaussian is the
hardest case: no distribution with variance \(\sigma^2\) needs more than \(\frac{1}{2} \log_2 (\sigma^2 / D)\) bits per
target (Cover and Thomas 2006, Problem 10.8), and a code designed for Gaussian targets reaches that rate on any targets
of the same variance ([Lapidoth 1997](https://doi.org/10.1109/18.567644)). These are rates for long sequences of
targets. For one fixed dataset of \(n\) targets, a covering argument gives \(26 (n - 1)\) bits in total plus a term of
order \(\log n\), about 14 bits at \(n = 512\) ([Kostina and Verdú 2012](https://arxiv.org/abs/1102.3944), Thm. 38).
So 26 bits per target suffice for every dataset with this variance, and the worst case needs them, up to
\(O(\log n)\) bits:

$$
L_{\mathrm{hi}} = \frac{n}{2} \log_2 \frac{1}{\varepsilon} = 26\, n \text{ bits},
$$

13,312 bits at \(n = 512\) support points. A formula longer than \(L_{\mathrm{hi}}\) costs more than writing the
support targets out to FVU \(= \varepsilon\), so whatever its error, it is not a compression of the support data: it
counts as \(u = 1\).

\(L_{\mathrm{hi}}\) is a worst case over all data with the targets' variance, and a particular dataset, such as one
with heavy-tailed targets, may be compressible further. That is intended: the edge holds for every dataset. A
refinement per dataset would rest on an estimate of the data's entropy, which is an estimate and not a bound, and is
not used.

## The front area

A candidate dominates the part of the box that is longer and has a larger error than it: the rectangle
\([u(c), 1] \times [v(c), 1]\). The **front area** \(A\) of a run is the area of the union of these rectangles over its
candidates, the share of the box its front dominates (the two-dimensional hypervolume with reference point
\((1, 1)\)):

- one candidate at \((u, v)\) covers \((1 - u)(1 - v)\), and two cover the union of their rectangles;
- a candidate that another one is at most as long as and no worse than adds nothing, so more candidates never lower
  the area;
- a candidate at least \(L_{\mathrm{hi}}\) long (\(u = 1\)), or no better than the mean (\(v = 1\)), dominates nothing,
  and a candidate whose FVU is not finite is left out;
- a run without candidates, a failed run, has \(A = 0\).

\(A\) runs from 0, nothing shorter than \(L_{\mathrm{hi}}\) fits better than the mean, to 1, a formula at most as long
as the mean fits to FVU \(\varepsilon\). srbf computes it from the staircase of the front, without a grid.

## The index: the front area over time

A method's budgets buy quality with time: every budget is a separate run on the same data, and its front area grows,
as a rule, with the time it is given. The index scores the whole curve rather than one point of it, as anytime
algorithms are compared: by the hypervolume under the quality-over-time curve
([López-Ibáñez and Stützle 2014](https://doi.org/10.1016/j.ejor.2013.10.043)), by runtime distributions on a
logarithmic time axis ([Hansen et al. 2016](https://arxiv.org/abs/1605.03560)), or by the area under the learning
curve on a logarithmic time axis ([Liu et al. 2021](https://arxiv.org/abs/2201.03801)).

### A budget's time

The time \(t_b\) of budget \(b\) is the geometric mean of its calibrated per-problem times in seconds on the reference
machine: every method gets the whole machine, and the time is what a user pays, failed runs included. On a logarithmic
time axis the geometric mean is the problems' average position, the usual summary of benchmark times
([Fleming and Wallace 1986](https://doi.org/10.1145/5666.5673)), and the arithmetic mean would sit later than it. The
index changes linearly when the whole curve moves along that axis within the time range, so where the problems'
times differ by a common factor at every budget, the geometric mean gives the problems' average index.

### Which budgets count

A budget counts only if every larger budget of the same method is slower. The rule reads the budgets and their times,
never the results: small budgets whose time is dominated by start-up overhead, and that take as long as a larger
budget or longer, drop out.

### The curve and its mean

With \(\tau = \log_{10} t\) and the counted budgets' times \(\tau_1 < \dots < \tau_K\) and areas \(A_1, \dots, A_K\),
the front area over time is

$$
q(\tau) =
\begin{cases}
0 & \tau < \tau_1 \quad \text{(no answer yet)},\\
\text{linear between } (\tau_k, A_k) \text{ and } (\tau_{k+1}, A_{k+1}) & \tau_1 \le \tau \le \tau_K,\\
A_K & \tau > \tau_K \quad \text{(the largest budget can always be run again)},
\end{cases}
$$

and the index of a method on a problem is its mean over \(\tau\) from \(-3\) to \(3\), from 1 ms to
1000 s:

$$
I = \frac{1}{6} \int_{-3}^{3} q(\tau)\, d\tau \;\in\; [0, 1].
$$

Budgets outside the range are cut at its edges: \(q\) is interpolated at the edge, and a method whose first budget
takes longer than 1000 s scores 0. The integral of the piecewise-linear \(q\) is summed in closed form. Linear
interpolation in \(\tau\) moves the index little when budgets are left out, far less than a step function would, and
a monotone cubic gives almost the same values.

Read as a probability, \(I\) is the expected front area at a time budget drawn log-uniformly from 1 ms to 1000 s.
Read as a volume, it is the share of the (length, error, time) box, all three on logarithmic axes, that the method's
interpolated front dominates.

### The time range

1 ms and 1000 s are declared constants of the protocol, in seconds on the reference machine, fixed for a benchmark
version. 1 ms is the order of one fit of a formula's constants to the support points: on the reference machine,
refitting the true formula's constants on 512 points with one restart takes 1.3 ms at the 10th percentile and 4.6 ms
at the median. Below it, no answer better than the mean is possible even with the true formula. Any lower edge below
the field's first answer only multiplies every index by a common factor, so ranks, ratios and the significance of
differences stay the same; a higher edge would deny a faster future method its credit and change every published
number. 1000 s is where the ladders end: the protocol
asks every method's ladder to reach it, or to end where the method itself stops.

## Over problems

A problem's value is its run's value, one run per problem. The problems are averaged over problem sets by the
random-effects average of [How results are averaged](averaging.md), the same model as for every per-problem value,
with its 95 % interval for these problem sets. Values are reported in %. Two headlines are built this way:

1. **The index of each method**: its time indices averaged over problems, shown over the method's publication date.
2. **The front area over time**: for each budget, the front areas averaged over problems the same way, shown at the
   budget's time \(t_b\).

## Limits

- A budget's time is one geometric mean over the problems, so the interpolation switches at the same time on every
  problem. This misses the link between a problem's time and its result: a method that stops early when it succeeds
  is undervalued.
- \(L_{\mathrm{hi}}\) is a worst case over all datasets with the targets' variance; see
  [the right edge](#the-right-edge-writing-the-targets-out).

## In the package

The definitions are `srbf.headline`, and every price goes through `srbf.pricing.price_bits`, the function the judge's
MDL columns use:

```python
from srbf.headline import average_over_problems, box_edges, budget_time, front_area, time_index

# per problem: the box, from the support targets
l_lo, l_hi = box_edges(y_support, engine)

# per run: every candidate's price in bits and FVU on the validation points
area = front_area(bits, fvu, l_lo=l_lo, l_hi=l_hi)

# per budget of a method: the geometric mean of its calibrated per-problem times
t_b = budget_time(per_problem_seconds)

# per problem: the method's budgets, their times and the runs' front areas
index = time_index(budgets, budget_times, areas)

# over problems: problem set -> each problem's value (its index, or its front area at one budget)
result = average_over_problems({"feynman": [0.41, 0.12, 0.27], "nguyen": [0.63, 0.58]})
print(result.value, result.lo, result.hi)
```

`front_coordinates` gives the candidates' \((u, v)\), `dominated_area` the area of points already in the unit box,
`kept_budgets` the budgets that count, and `mean_prefix`, `lower_edge_bits` and `upper_edge_bits` the edges one at a
time.

## Sources

- Cover, T. M. and Thomas, J. A. (2006). *Elements of Information Theory*, 2nd ed. Wiley. Thm. 10.3.2 (the
  rate-distortion function of the Gaussian) and Problem 10.8 (the Gaussian as the hardest case under squared error).
- Fleming, P. J. and Wallace, J. J. (1986). How not to lie with statistics: the correct way to summarize benchmark
  results. *Communications of the ACM* 29(3), 218–221. [doi:10.1145/5666.5673](https://doi.org/10.1145/5666.5673)
- Gray, R. M. and Neuhoff, D. L. (1998). Quantization. *IEEE Transactions on Information Theory* 44(6), 2325–2383.
  [doi:10.1109/18.720541](https://doi.org/10.1109/18.720541)
- Hansen, N., Auger, A., Brockhoff, D., Tušar, D. and Tušar, T. (2016). COCO: Performance assessment.
  [arXiv:1605.03560](https://arxiv.org/abs/1605.03560)
- Kostina, V. and Verdú, S. (2012). Fixed-length lossy compression in the finite blocklength regime. *IEEE
  Transactions on Information Theory* 58(6), 3309–3338. [arXiv:1102.3944](https://arxiv.org/abs/1102.3944)
- Lapidoth, A. (1997). On the role of mismatch in rate distortion theory. *IEEE Transactions on Information Theory*
  43(1), 38–47. [doi:10.1109/18.567644](https://doi.org/10.1109/18.567644)
- Liu, Z. et al. (2021). Winning solutions and post-challenge analyses of the ChaLearn AutoDL challenge 2019.
  [arXiv:2201.03801](https://arxiv.org/abs/2201.03801)
- López-Ibáñez, M. and Stützle, T. (2014). Automatically improving the anytime behaviour of optimisation algorithms.
  *European Journal of Operational Research* 235(3), 569–582.
  [doi:10.1016/j.ejor.2013.10.043](https://doi.org/10.1016/j.ejor.2013.10.043)
