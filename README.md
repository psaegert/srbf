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

# srbf: Symbolic Regression Benchmark Framework

<p align="center">
  <a href="https://psaegert.github.io/srbf/">
    <img alt="srbf visual abstract: benchmarks and methods go through one fair protocol (same expressions, wall-clock budgets, paired statistics, pre-declared corrected comparisons) into the interactive explorer with four-state verdicts."
         src="https://raw.githubusercontent.com/psaegert/srbf/main/assets/brand/visual-abstract.svg" width="100%">
  </a>
</p>

`srbf` evaluates symbolic-regression models on shared benchmarks with shared metrics. It is the
Symbolic Regression Benchmark Framework carved out of
[flash-ansr](https://github.com/psaegert/flash-ansr): the `Benchmark` driver, model adapters, and
metrics, over `symbolic-data` catalogs. It depends one-way on `flash-ansr` (`srbf` imports
`flash-ansr`; `flash-ansr` never imports `srbf`).

**Built for contributions.** Developers of SR methods add their model by opening a PR with an
**adapter** (two methods plus a registered builder) plus install instructions. The built-in adapters (`flash_ansr`, `pysr`,
`nesymres`, `e2e`, `lample_charton`, `brute_force`) are reference examples, not a closed set. See the
[adapter contribution guide](docs/adapters.md).

> **Status: 0.6, data-layer redesign.** The benchmark seam (`srbf.core` Protocols + the `Benchmark`
> driver) is model-agnostic, the data source is always a `symbolic-data` catalog, and adapters are a
> thin mapper over each model (flash-ansr via `FlashANSR.infer()`). Inline `!sweep` config
> cross-products and multi-draw bootstrap reporting (`bootstrap_report` / `draw_distribution`) ship in
> this release.

## Install

```bash
pip install srbf                 # benchmark driver + metrics + the flash-ansr adapter (usable out of the box)
pip install "srbf[baselines]"    # + PySR and other pip baseline deps (sympy, pysr, omegaconf)
```

`srbf` pulls in `flash-ansr`, `symbolic-data`, and `simplipy` automatically, and requires
**Python >= 3.12**. The PySR adapter ships in the base wheel but the `pysr` package (plus a
Julia precompile) comes with the `[baselines]` extra, so a bare install does not include a
runnable PySR baseline. The unpackaged research baselines (NeSymReS, E2E) are provisioned
out-of-band; see [docs/models.md](docs/models.md).

## Quickstart

```bash
# 1. point srbf at a tree holding configs/, data/, and models/ (your srbf checkout works)
export FLASH_ANSR_ROOT=$(pwd)

# 2. get a model to evaluate (flash-ansr's CLI ships with srbf)
flash_ansr install psaegert/flash-ansr-v23.0-3M

# 3. run an evaluation. The config names a symbolic-data catalog (`fastsrb` / `v23-val`); it is
#    fetched from Hugging Face on first use and cached, so there is no local data-build step.
#    The config is a sweep over candidate counts; --sweep-filter picks one rung for a smoke test.
srbf run -c configs/evaluation/scaling/v23.0-3M_fastsrb.yaml --sweep-filter ladder=32 --limit 50 -v
```

Outputs land under `results/evaluation/.../*.pkl`, one row per evaluated problem with the raw
prediction columns (derive FVU / recovery / F1 in a separate step; see
[docs/running.md](docs/running.md#deriving-metrics)). Run programmatically instead:

```python
from srbf import Benchmark

# A config with inline !sweep / experiments expands to several runs; expand and run each one.
for benchmark in Benchmark.runs_from_config("configs/evaluation/scaling/v23.0-3M_fastsrb.yaml"):
    benchmark.run()  # resume-aware; a no-op if that run's configured target is already reached

# For a single, fully-resolved run (no !sweep / experiments), use from_config directly:
# Benchmark.from_config(config_dict).run()
```

## Documentation

| Guide | What it covers |
|---|---|
| [Running evaluations](docs/running.md) | the `srbf run` CLI, config anatomy (data_source / model_adapter / runner / experiments / `!sweep`), outputs, resume, reporting |
| [Benchmarks & datasets](docs/benchmarks.md) | the `data_source` catalog block, the shipped catalogs (`v23-val`, `fastsrb`, `lample-charton-v23`), custom catalogs |
| [Models & provisioning](docs/models.md) | installing/patching the built-in models; the `model_adapter` block per type |
| [**Adding your model**](docs/adapters.md) | the adapter protocol + registry, and the PR flow to contribute a new SR method |

## Development

```bash
pip install -e ".[dev]"
pre-commit run --all-files
pytest tests
```

## License

MIT (see [LICENSE](LICENSE)). Third-party attributions in [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES).
