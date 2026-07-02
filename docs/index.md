# srbf: Symbolic Regression Benchmark Framework

`srbf` evaluates symbolic-regression models on shared benchmarks with shared metrics. It is the
Symbolic Regression Benchmark Framework carved out of
[flash-ansr](https://github.com/psaegert/flash-ansr): the
`Benchmark` driver, model adapters, and metrics, over `symbolic-data` catalogs. It depends one-way on
`flash-ansr` (`srbf` imports `flash-ansr`; `flash-ansr` never imports `srbf`).

**Built for contributions.** Developers of SR methods add their model by opening a pull request with
an **adapter** (two methods plus a registered builder) plus install instructions. The built-in adapters (`flash_ansr`, `pysr`,
`nesymres`, `e2e`, `lample_charton`, `brute_force`) are reference examples, not a closed set. See the
[adapter contribution guide](adapters.md).

## Install

```bash
pip install srbf                 # benchmark driver + metrics + the flash-ansr adapter (usable out of the box)
pip install "srbf[baselines]"    # + PySR and other pip baseline deps (sympy, pysr, omegaconf)
```

`srbf` pulls in `flash-ansr`, `symbolic-data`, and `simplipy` automatically, and requires **Python
>= 3.12**. The PySR adapter ships in the base wheel, but the `pysr` package (plus a Julia
precompile) comes with the `[baselines]` extra, so a bare install does not include a runnable PySR
baseline. The example configs and benchmarks live in the
[repository](https://github.com/psaegert/srbf); clone it to run the shipped evaluations.

## Quickstart

```bash
export FLASH_ANSR_ROOT=$(pwd)                       # a tree holding configs/, models/, results/
flash_ansr install psaegert/flash-ansr-v23.0-3M     # flash-ansr's CLI ships with srbf
srbf run -c configs/evaluation/scaling/v23.0-3M_fastsrb.yaml --sweep-filter ladder=32 --limit 50 -v
```

The config names a `symbolic-data` catalog (here `fastsrb`); it is fetched from Hugging Face on first
use and cached, so there is no local data-build step. The config sweeps over candidate counts, so
`--sweep-filter ladder=32` selects a single rung for the smoke test. Outputs land under
`results/evaluation/.../*.pkl`, one row per evaluated problem.

## Documentation

| Guide | What it covers |
|---|---|
| [Running evaluations](running.md) | the `srbf run` CLI, config anatomy, `!sweep`, outputs, resume, reporting |
| [Benchmarks & datasets](benchmarks.md) | the `data_source` catalog block, the shipped catalogs, custom catalogs |
| [Models & provisioning](models.md) | installing/patching the built-in models; the `model_adapter` block per type |
| [**Adding your model**](adapters.md) | the adapter protocol + registry, and the PR flow to contribute a new SR method |

## License

MIT. See the [repository](https://github.com/psaegert/srbf) for the license and third-party attributions.
