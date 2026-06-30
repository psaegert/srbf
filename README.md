# srbf: Symbolic Regression Benchmark Framework

`srbf` evaluates symbolic-regression models on shared benchmarks with shared metrics. It is the
benchmark framework carved out of [flash-ansr](https://github.com/psaegert/flash-ansr): the
`Benchmark` driver, model adapters, and metrics, over `symbolic-data` catalogs. It depends one-way on
`flash-ansr` (`srbf` imports `flash-ansr`; `flash-ansr` never imports `srbf`).

**Built for contributions.** Developers of SR methods add their model by opening a PR with an
**adapter** (two methods) plus install instructions. The built-in adapters (`flash_ansr`, `pysr`,
`nesymres`, `e2e`, `lample_charton`, `brute_force`) are reference examples, not a closed set. See the
[adapter contribution guide](docs/adapters.md).

> **Status: 0.5, data-layer redesign.** The benchmark seam (`srbf.core` Protocols + the `Benchmark`
> driver) is model-agnostic, the data source is any `symbolic-data` catalog, and adapters are a thin
> mapper over each model (flash-ansr via `FlashANSR.infer()`). Inline `!sweep` config cross-products
> and multi-draw bootstrap reporting are planned for 0.5.1.

## Install

```bash
pip install srbf                 # benchmark driver + metrics + the pip-installable adapters (flash-ansr, PySR)
pip install "srbf[baselines]"    # + pip baseline deps (sympy, pysr, omegaconf)
```

`srbf` pulls in `flash-ansr` and `simplipy` automatically. The unpackaged research baselines
(NeSymReS, E2E) are provisioned out-of-band; see [docs/models.md](docs/models.md).

## Quickstart

```bash
# 1. point srbf at a tree holding configs/, data/, and models/ (your srbf checkout works)
export FLASH_ANSR_ROOT=$(pwd)

# 2. get a model to evaluate (flash-ansr's CLI ships with srbf)
flash_ansr install psaegert/flash-ansr-v23.0-3M

# 3. fetch a benchmark + build its skeleton pool   (see docs/benchmarks.md)
#    ...then run an evaluation:
srbf run -c configs/evaluation/scaling/v23.0-3M_val.yaml --limit 50 -v
```

Outputs land under `results/evaluation/.../*.pkl`, one row per evaluated dataset with flat metric
columns. Run programmatically instead:

```python
from srbf import Benchmark

benchmark = Benchmark.from_config(config="configs/evaluation/scaling/v23.0-3M_val.yaml")
benchmark.run()  # resume-aware; a no-op if the configured target is already reached
```

## Documentation

| Guide | What it covers |
|---|---|
| [Running evaluations](docs/running.md) | the `srbf run` CLI, config anatomy (data_source / model_adapter / runner / experiments), outputs, resume |
| [Benchmarks & datasets](docs/benchmarks.md) | fetching FastSRB, building skeleton pools with `flash_ansr import-data`, custom sets |
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
