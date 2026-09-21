# Concepts

The words the rest of the documentation relies on, in the order a run meets them.

## Ground truth, problem, catalog, suite

A **ground truth** is an expression, such as `m * v**2 / 2`, together with the ranges its
variables are sampled from. A **problem** is one realization of a ground truth: a set of **support** points
`(X, y)` that the method sees, and a disjoint set of **validation** points that it never sees. A
**catalog** is a named collection of expressions (`feynman`, `nguyen`, ...), served by the
[`symbolic-data`](https://github.com/psaegert/symbolic-data) package. A **suite** is a list of
catalogs; the `srbf` suite holds all 29. See [Benchmarks](benchmarks.md).

Problems are drawn when a run starts, and the draw is not seeded: two runs see the same expressions at
different points. Numbers therefore come with intervals, and
[Benchmarks](benchmarks.md#the-same-points-for-several-methods) shows how to give several methods
identical points.

## Method, adapter, worker

A **method** is whatever turns support points into an expression. srbf talks to it through an
**adapter**. The usual adapter is a **worker**: one Python file with a `fit()` function, run in an
interpreter of your choice, so the method keeps its own dependencies. A method that installs next
to srbf without conflicts can use an **in-process adapter** class instead. See
[Adding your method](adapters.md).

## Budget, ladder, rung

Most methods have a knob that buys quality with compute: the number of candidates drawn, the beam
width, the number of search iterations. srbf calls that knob the **budget** and a list of its
values a **ladder**; one value is a **rung**. In a config a ladder is written with `!sweep`, and
every rung becomes one run with its own result file. Results are read along the ladder: recovery
as a function of the budget, or of the time that budget costs. See
[Running evaluations](running.md#ladders-and-sweeps).

## Config, experiment, run

A **config** is a YAML file with three blocks: `data_source` (which catalog, how it is sampled),
`model_adapter` (which method, how it is set up) and `runner` (where results go). A config may
hold several named **experiments**, typically one per catalog, and each experiment may hold a
ladder. One experiment at one rung is a **run**: the unit that `srbf run` executes, resumes and
shards.

## Raw results and derived metrics

A run stores **raw results**: for every problem the data, the prediction as a string, and the
predicted values. **Metrics** are derived from those files in a separate step, by one code path
for every method. A result file therefore never goes stale when a metric is added, and no method
can be scored by a different rule. See [Results](results.md).

## The judge

Two questions are asked of every prediction. **Does it fit?** The prediction is evaluated on the
validation points, and the fraction of variance it leaves unexplained (FVU) is compared with
float32 precision. **Is it the ground truth?** Prediction and ground truth are both simplified by one
[SimpliPy](https://github.com/psaegert/simplipy) engine into a canonical form, and their
**skeletons** are compared: the expressions with every numeric constant replaced by a placeholder,
so that `2.1*sin(x1)` and `3*sin(x1)` are the same structure. The engine is a fixed set of rewrite
rules, [published](https://huggingface.co/datasets/psaegert/simplipy-assets) and downloaded on first
use; judging involves no learned model. It is named in the config as `simplipy_engine`, and the
catalogs of the srbf suite are judged with `acj-5-4-llm`. The engine also measures how long an
prediction is, as a **description length** in bits (MDL) that charges for operators, variables and the
digits of constants. See [Metrics](metrics.md).

## Success metrics and analysis metrics

A **success metric**, such as numeric recovery, says whether a problem was solved and is defined
for every problem: a method that errors or returns nothing has failed it, and the metric is 0. An
**analysis metric**, such as the FVU or the length of the predicted expression, describes the
predictions that were made. Where the range of such a metric has a worst value, as a token overlap
in \([0, 1]\) has, a problem without a prediction takes it; where it has none, because a prediction can
be arbitrarily bad, the problem has no value and none is filled in, and the summary is read next
to the share of problems a method has a prediction for. The two kinds are applied the same way to every
method.

## Provenance

Every config states who chose the method's configuration (`config_provenance`), and every result
file records what ran: the config and its hash, package versions, the git state and the inputs.
See [Fairness](fairness.md).
