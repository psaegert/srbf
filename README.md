# srbf — Symbolic Regression Benchmark Framework

`srbf` is the evaluation framework carved out of [flash-ansr](https://github.com/psaegert/flash-ansr):
the evaluation engine, model adapters, benchmarks, and metrics for evaluating symbolic-regression
models. It depends one-way on `flash-ansr` (`srbf` imports `flash-ansr`; `flash-ansr` never imports
`srbf`).

> **Status: 0.1 — cleanly-carved eval, not yet a general framework.** The engine seam
> (`srbf.eval.core` Protocols + `srbf.eval.engine`) is model-agnostic, but every concrete adapter
> imports `flash-ansr`, and the adapter set is a closed registry. A plugin `register_adapter()` API
> and raw-dataset (`(X, y)` CSV/parquet) ingestion are a planned follow-on.

## Install

```bash
pip install srbf                 # engine + metrics + the flash-ansr / PySR adapters
pip install "srbf[baselines]"    # + the pip-installable baseline deps (sympy, pysr, omegaconf)
```

`srbf` requires `flash-ansr` (its one-way dependency) and `simplipy`.

## Usage

Run an evaluation from a unified config:

```bash
srbf evaluate-run -c <config.yaml> -v
```

or programmatically:

```python
from srbf import build_evaluation_run

plan = build_evaluation_run(config="path/to/config.yaml")
plan.engine.run(limit=plan.remaining, output_path=plan.output_path)
```

## Baseline models (out-of-band provisioning)

The pip wheel ships the engine, metrics, and the pip-installable adapters (flash-ansr, PySR). The
**unpackaged research baselines** — NeSymReS and the E2E (`symbolicregression`) model — are NOT pip
dependencies: their upstream source trees + weights are provisioned by a clone-based bench-setup
flow (`scripts/patch_nesymres.py` patches a recursive upstream clone; weights download separately).
This keeps the wheel clean while supporting reproducible baseline comparisons.

## Asset root

`srbf` resolves config/data/model assets through `flash-ansr`'s shared project root. Point it at your
srbf checkout for eval runs:

```bash
export FLASH_ANSR_ROOT=/path/to/srbf
```

## Development

```bash
pip install -e ".[dev]"
pytest tests
```

## License

MIT (see [LICENSE](LICENSE)). Third-party attributions in [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES).
