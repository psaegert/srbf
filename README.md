<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/psaegert/srbf/main/assets/brand/srbf-wordmark-dark.svg">
    <img alt="srbf" src="https://raw.githubusercontent.com/psaegert/srbf/main/assets/brand/srbf-wordmark.svg" width="200">
  </picture>
</p>
<h3 align="center">Symbolic Regression Benchmark Framework</h3>
<p align="center">
  <a href="https://pypi.org/project/srbf/">PyPI</a> ·
  <a href="https://srbf.readthedocs.io/">Docs</a> ·
  <a href="https://psaegert.github.io/srbf/">Interactive results</a>
</p>

<p align="center">
  <a href="https://psaegert.github.io/srbf/">
    <img alt="srbf visual abstract: any datasets and any methods go through one framework (a unified dataset formalism, built-in decontamination, paired and pre-declared statistics) into the results explorer with curves, tables and ranks."
         src="https://raw.githubusercontent.com/psaegert/srbf/main/assets/brand/visual-abstract.svg" width="100%">
  </a>
</p>

`srbf` runs symbolic regression methods on the same expressions, sampled the same way, and reads every
prediction with one judge. A method is plugged in with a small adapter, a YAML config describes the
evaluation, and four commands take it from a first smoke test to a report:

```bash
srbf new mymethod                    # scaffold an adapter for your method
srbf check   -c config.yaml          # a few real problems, step by step
srbf run     -c config.yaml -v       # the evaluation
srbf analyze -c config.yaml -o report
```

- **29 catalogs, 6,660 expressions**: Feynman, FastSRB, Nguyen and the other classical suites, physics
  collections and synthetic corpora, served by [`symbolic-data`](https://github.com/psaegert/symbolic-data).
- **Any method, in its own environment**: a worker is one Python file with a `fit()` function, run
  in the interpreter you name, with whatever torch, Julia or NumPy version your method needs.
- **One judge**: every prediction is parsed, evaluated on held-out points and compared with the ground truth in
  one canonical form.
- **Evaluations that scale**: inline `!sweep` ladders, whole-suite configs, resumable runs, and
  sharding across GPUs with a checked merge.
- **Statistics that fit the design**: paired comparisons on the same expressions, rank analysis with
  critical differences, bootstrap intervals.
- **Stated provenance**: every configuration carries a label for who chose it, and every result file
  records what ran.

The published results are on the [results explorer](https://psaegert.github.io/srbf/).

## Install

```bash
pip install srbf                 # the framework, the metrics and the Flash-ANSR adapter
pip install "srbf[baselines]"    # plus what the PySR, NeSymReS and E2E adapters import
pip install "srbf[analysis]"     # plus matplotlib, for the figures of `srbf analyze`
```

srbf needs Python 3.12 or newer and installs `flash-ansr`, `symbolic-data` and `simplipy` with it.

## Quickstart

No GPU and no model are needed for a first run: `srbf new` writes an adapter whose placeholder fits
a linear model.

```bash
export FLASH_ANSR_ROOT=$PWD/bench                                   # models, results and adapters live here
srbf new mymethod --python "$(which python)"
srbf check   -c bench/adapters/mymethod/config.yaml                  # two real problems, every step named
srbf run     -c bench/adapters/mymethod/config.yaml --experiment nguyen -v
srbf analyze -c bench/adapters/mymethod/config.yaml -o report        # report/results.md and figures/
```

To evaluate a released Flash-ANSR model on one catalog at one budget:

```bash
git clone https://github.com/psaegert/srbf && cd srbf && export FLASH_ANSR_ROOT=$PWD
flash_ansr install psaegert/flash-ansr-v25.0-T8-3M
srbf run -c configs/evaluation/scaling/flash-ansr-v25.0-T8-3M_srbf.yaml --experiment nguyen --sweep-filter ladder=32 -v
```

## Documentation

[srbf.readthedocs.io](https://srbf.readthedocs.io/)

| Guide | What it covers |
|---|---|
| [Quickstart](https://srbf.readthedocs.io/en/latest/quickstart/) | both tours above, with their output |
| [Concepts](https://srbf.readthedocs.io/en/latest/concepts/) | ground truth, problem, catalog, budget, judge: the vocabulary |
| [**Adding your method**](https://srbf.readthedocs.io/en/latest/adapters/) | the worker contract and the pull request |
| [Running evaluations](https://srbf.readthedocs.io/en/latest/running/) | the config, ladders, suites, resuming, sharding, clusters |
| [Results](https://srbf.readthedocs.io/en/latest/results/) | result files, deriving metrics, intervals, the report |
| [Metrics](https://srbf.readthedocs.io/en/latest/metrics/) | the definition of every metric |
| [Paired comparisons](https://srbf.readthedocs.io/en/latest/paired/) | comparing two methods, ranking many |
| [Benchmarks](https://srbf.readthedocs.io/en/latest/benchmarks/) | the 29 catalogs and their sources, custom catalogs |
| [Models](https://srbf.readthedocs.io/en/latest/models/) | installing and configuring the built-in methods |
| [Command line](https://srbf.readthedocs.io/en/latest/cli/) | every command and flag |
| [Fairness](https://srbf.readthedocs.io/en/latest/fairness/) | one protocol, upstream defaults, provenance labels, timing |

## Citing

srbf accompanies Saegert & Köthe 2026, _Breaking the Simplification Bottleneck in Amortized Neural
Symbolic Regression_ (ICML 2026), [arXiv:2602.08885](https://arxiv.org/abs/2602.08885). If you
publish numbers on a catalog, cite its source as well
([Benchmarks](https://srbf.readthedocs.io/en/latest/benchmarks/)).

## Development

```bash
pip install -e ".[dev]"
pre-commit run --all-files
pytest tests
```

## License

MIT (see [LICENSE](LICENSE)). Third-party attributions in [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES).
