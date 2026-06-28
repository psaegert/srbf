# srbf: Symbolic Regression Benchmark Framework

`srbf` evaluates symbolic-regression models on shared benchmarks with shared metrics. It is the
evaluation framework carved out of [flash-ansr](https://github.com/psaegert/flash-ansr): the
evaluation engine, model adapters, benchmarks, and metrics. It depends one-way on `flash-ansr`
(`srbf` imports `flash-ansr`; `flash-ansr` never imports `srbf`).

**Built for contributions.** Developers of SR methods add their model by opening a pull request with
an **adapter** (two methods) plus install instructions. The built-in adapters (`flash_ansr`, `pysr`,
`nesymres`, `e2e`, `skeleton_pool`, `brute_force`) are reference examples, not a closed set. See the
[adapter contribution guide](adapters.md).

## Install

```bash
pip install srbf                 # engine + metrics + the pip-installable adapters (flash-ansr, PySR)
pip install "srbf[baselines]"    # + pip baseline deps (sympy, pysr, omegaconf)
```

`srbf` pulls in `flash-ansr` and `simplipy` automatically. The example configs and benchmarks live in
the [repository](https://github.com/psaegert/srbf); clone it to run the shipped evaluations.

## Quickstart

```bash
export FLASH_ANSR_ROOT=$(pwd)                       # a tree holding configs/, data/, models/
flash_ansr install psaegert/flash-ansr-v23.0-3M     # flash-ansr's CLI ships with srbf
srbf run -c configs/evaluation/scaling/v23.0-3M_fastsrb.yaml --limit 50 -v
```

Outputs land under `results/evaluation/.../*.pkl`, one row per evaluated dataset.

## Documentation

| Guide | What it covers |
|---|---|
| [Running evaluations](running.md) | the `srbf run` CLI, config anatomy, outputs, resume |
| [Benchmarks & datasets](benchmarks.md) | fetching FastSRB, building skeleton pools, custom sets |
| [Models & provisioning](models.md) | installing/patching the built-in models; the `model_adapter` block per type |
| [**Adding your model**](adapters.md) | the adapter protocol + registry, and the PR flow to contribute a new SR method |

## License

MIT. See the [repository](https://github.com/psaegert/srbf) for the license and third-party attributions.
