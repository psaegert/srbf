# srbf

**srbf**, the Symbolic Regression Benchmark Framework, runs symbolic regression methods on the same
laws, sampled the same way, and reads every answer with one judge. A method is plugged in with a
small adapter, a YAML config describes the evaluation, and four commands take it from a first
smoke test to a report:

```bash
srbf new mymethod                    # scaffold an adapter for your method
srbf check   -c config.yaml          # a few real problems, step by step
srbf run     -c config.yaml -v       # the evaluation
srbf analyze -c config.yaml -o report
```

The published results of the methods entered so far are on the
[results explorer](https://psaegert.github.io/srbf/).

## What srbf gives you

- **29 catalogs, 6,660 laws.** Feynman, FastSRB, Nguyen and the other classical suites, physics
  collections and synthetic corpora, all served by [`symbolic-data`](https://github.com/psaegert/symbolic-data)
  and fetched on first use. See [Benchmarks](benchmarks.md).
- **Any method, in its own environment.** A worker is one Python file with a `fit()` function. It
  runs in the interpreter you name, with whatever torch, Julia or NumPy version your method needs.
  See [Adding your method](adapters.md).
- **One judge.** Every answer is parsed, evaluated on held-out points and compared with the law in
  one canonical form, so a number means the same thing for every method. See [Metrics](metrics.md).
- **Evaluations that scale.** Inline `!sweep` ladders, whole-suite configs, resumable runs, and
  sharding across GPUs with a checked merge. See [Running evaluations](running.md).
- **Statistics that fit the design.** Paired comparisons on the same laws, rank analysis with
  critical differences, bootstrap intervals. See [Paired comparisons](paired.md).
- **Stated provenance.** Every configuration carries a label for who chose it, and every result file
  records what ran. See [Fairness](fairness.md).

## Install

```bash
pip install srbf                 # the framework, the metrics and the Flash-ANSR adapter
pip install "srbf[baselines]"    # plus what the PySR, NeSymReS and E2E adapters import
pip install "srbf[analysis]"     # plus matplotlib, for the figures of `srbf analyze`
```

srbf needs Python 3.12 or newer and installs `flash-ansr`, `symbolic-data` and `simplipy` with it.
The example configs live in the [repository](https://github.com/psaegert/srbf); clone it to run them.

## Where to go next

| You want to | Read |
|---|---|
| see it work in a minute, without a GPU or a model | [Quickstart](quickstart.md) |
| understand the vocabulary: law, catalog, budget, judge | [Concepts](concepts.md) |
| enter your own method | [Adding your method](adapters.md) |
| write a config, sweep a budget, run on a cluster | [Running evaluations](running.md) |
| look up a command or a flag | [Command line](cli.md) |
| read result files, derive metrics, build a report | [Results](results.md) |
| know exactly what a metric measures | [Metrics](metrics.md) |
| compare two methods, or rank many | [Paired comparisons](paired.md) |
| pick catalogs or bring your own | [Benchmarks](benchmarks.md) |
| install and configure the built-in methods | [Models](models.md) |
| see how methods are kept on equal terms | [Fairness](fairness.md) |

## Citing

srbf accompanies Saegert & Köthe, *Breaking the Simplification Bottleneck in Amortized Neural
Symbolic Regression* ([arXiv:2602.08885](https://arxiv.org/abs/2602.08885)). srbf is MIT licensed;
the catalogs keep the licenses of their sources, listed in
[THIRD_PARTY_LICENSES](https://github.com/psaegert/srbf/blob/main/THIRD_PARTY_LICENSES).
