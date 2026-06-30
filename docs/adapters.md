# Adding your model: the adapter contribution guide

srbf is a community benchmark framework. To evaluate **your** symbolic-regression method on the same
benchmarks and metrics as everyone else, you add an **adapter** and open a pull request. The built-in
adapters (`flash_ansr`, `pysr`, `nesymres`, `e2e`, `lample_charton`, `brute_force`) are reference
examples, not a closed set: pick the one closest to your model and copy it.

An adapter is the thin layer that teaches the `srbf` benchmark driver how to drive your model on one
problem at a time. You implement two methods and register a builder. That is the whole contract.

## What you submit in a PR

1. An **adapter class** in `src/srbf/model_adapters.py` (or a new module) implementing the
   `EvaluationModelAdapter` protocol.
2. A **builder function** `_build_<name>_adapter(config)` and a one-line entry in the
   `_ADAPTER_REGISTRY` in `src/srbf/config.py`.
3. An **example config** under `configs/evaluation/` with a `model_adapter: {type: <name>, ...}` block.
4. **Install instructions** for your model's dependencies (see [Provisioning](#provisioning-your-models-dependencies)).
5. Ideally a small **test** under `tests/` and a note in [docs/models.md](models.md).

## The adapter contract

An adapter implements `srbf.core.EvaluationModelAdapter`, a `@runtime_checkable` `Protocol`. You do
**not** need to subclass anything: structural typing means any object with the right methods
qualifies. The two methods:

```python
class EvaluationModelAdapter(Protocol):
    def prepare(self, *, data_source: EvaluationDataSource | None = None) -> None:
        """Run once before the first sample (load weights, start a Julia process, ...)."""

    def evaluate_sample(self, sample: EvaluationSample) -> EvaluationResult:
        """Fit + predict on ONE problem; return a normalized result."""
```

### What you receive: `EvaluationSample`

Each call to `evaluate_sample` gets one problem (`srbf.core.EvaluationSample`):

| field | shape / type | meaning |
|---|---|---|
| `x_support` | `(n_support, n_features)` | inputs to **fit** on |
| `y_support` | `(n_support, 1)` | clean targets to fit on |
| `y_support_noisy` | `(n_support, 1)` or `None` | noisy targets (use these when present and the benchmark adds noise) |
| `x_validation` | `(n_val, n_features)` | held-out inputs (may be empty) |
| `y_validation` | `(n_val, 1)` | held-out targets |
| `metadata` | `Mapping` | ground-truth info: `variables`, `skeleton`, GT expression, hashes, ... |
| `is_placeholder` | `bool` | the driver emits these when a problem could not be produced; you can skip them |

Helpers: `sample.n_support`, `sample.n_validation`, `sample.clone_metadata()` (a mutable copy of the
metadata to seed your result).

### What you return: `EvaluationResult`

Return `EvaluationResult(record)`, where `record` is a plain `dict`. Start it from
`sample.clone_metadata()` so the ground truth travels with the result, then add your outputs. The
metrics layer (`srbf.result_processing`) reads these keys:

| key | type | consumed for |
|---|---|---|
| `prediction_success` | `bool` | gating; set `False` (+ `error`) on any failure and return early |
| `y_pred` | `(n_support, 1)` | numeric fit error (FVU on support) |
| `y_pred_val` | `(n_val, 1)` | numeric recovery (FVU on the held-out set) |
| `predicted_expression` | `str` | the human-readable result expression |
| `predicted_expression_prefix` | `list[str]` | normalized prefix tokens (for exact/symbolic match) |
| `predicted_skeleton_prefix` | `list[str]` | normalized skeleton (for skeleton recovery) |
| `error` | `str` | populated on failure |
| `fit_time` | `float` | wall-clock fit time (timing comparisons) |

You do not compute FVU or recovery yourself: emit predictions + the expression, and srbf's metrics do
the rest. Prefix/skeleton normalization is available from `simplipy` (`normalize_expression`,
`normalize_skeleton`); convert an infix string to prefix with the SimpliPy engine
(`simplipy_engine.infix_to_prefix(expr)`).

## A minimal adapter (copy this)

This mirrors the built-in `PySRAdapter` (`src/srbf/model_adapters.py`), the cleanest template for
any external `fit`/`predict` regressor:

```python
import time
import numpy as np
from srbf.core import EvaluationModelAdapter, EvaluationSample, EvaluationResult
from simplipy import normalize_expression, normalize_skeleton


class MyModelAdapter(EvaluationModelAdapter):
    def __init__(self, *, simplipy_engine, **hyperparams):
        # Import your model library lazily so importing srbf never requires it.
        from mymodel import MyRegressor  # noqa: F401
        self.hyperparams = hyperparams
        self.simplipy_engine = simplipy_engine
        self._model = None

    def prepare(self, *, data_source=None):
        from mymodel import MyRegressor
        self._model = MyRegressor(**self.hyperparams)   # construct once, reuse across samples

    def evaluate_sample(self, sample: EvaluationSample) -> EvaluationResult:
        record = sample.clone_metadata()
        X = sample.x_support.copy()
        y = (sample.y_support_noisy if sample.y_support_noisy is not None else sample.y_support).copy()
        X_val = sample.x_validation.copy()

        t0 = time.time()
        try:
            self._model.fit(X, y.ravel())
            record["fit_time"] = time.time() - t0
            record["y_pred"] = self._model.predict(X).reshape(-1, 1)
            record["y_pred_val"] = (
                self._model.predict(X_val).reshape(-1, 1) if X_val.size else np.empty((0, 1))
            )
            expr = str(self._model.get_expression())          # your model's symbolic output
            record["predicted_expression"] = expr
            prefix = self.simplipy_engine.infix_to_prefix(expr)
            record["predicted_expression_prefix"] = normalize_expression(prefix).copy()
            record["predicted_skeleton_prefix"] = normalize_skeleton(prefix).copy()
            record["prediction_success"] = True
        except Exception as exc:
            record["error"] = str(exc)
            record["prediction_success"] = False
        return EvaluationResult(record)
```

> **The serial driver.** The `Benchmark` driver is a plain serial loop: it calls `evaluate_sample`
> on one problem at a time, so per-problem wall-clock timing is uncontended. Any generate-then-refine
> overlap belongs inside your model's own per-problem inference (as `flash_ansr` does internally), not
> in the adapter; the adapter contract is just `prepare` + `evaluate_sample`.

## Register your adapter

Adapters are looked up by a `type` string through a registry in `src/srbf/config.py`. Add a builder
that turns a config block into your adapter, then register it. For the SimpliPy engine, reuse the
shared `resolve_simplipy_engine` helper (it loads `model_adapter.simplipy_engine` and raises if it is
missing):

```python
from srbf.config import resolve_simplipy_engine

def _build_mymodel_adapter(config):
    return MyModelAdapter(
        simplipy_engine=resolve_simplipy_engine(config, adapter_name="mymodel"),
        **{k: v for k, v in config.items() if k not in {"type", "simplipy_engine"}},
    )

_ADAPTER_REGISTRY: dict[str, AdapterBuilder] = {
    "flash_ansr": _build_flash_ansr_adapter,
    # ...
    "mymodel": _build_mymodel_adapter,   # <- your one-line addition
}
```

Now a config can select your model:

```yaml
model_adapter:
  type: mymodel
  # ...your hyperparameters, read by _build_mymodel_adapter...
```

See [docs/running.md](running.md) for the full config anatomy and
[docs/models.md](models.md) for the per-type `model_adapter` blocks of the built-in adapters.

## Provisioning your model's dependencies

How your model is installed is part of the contribution. srbf supports three patterns (the existing
adapters are just defaults you can copy):

- **pip dependency** (cleanest): if your model is `pip install`-able, gate the import lazily (a
  `_require_<name>()` helper, like `_require_pysr`) so importing srbf never pulls it in, and document
  `pip install <yourmodel>`. Pure-pip extras can go under the `[baselines]` optional-dependency group.
- **clone + patch** (for unpackaged research code): ship a `scripts/patch_<name>.py` that pins and
  patches an upstream clone to run on the current Python, plus instructions to download weights. This
  is exactly how `nesymres` and `e2e` are provisioned today (`scripts/patch_nesymres.py`,
  `scripts/patch_symbolicregression.py`, `scripts/patch_typing_io.py`). These are **one default
  recipe**, not the only way.
- **vendored**: only if there is no upstream to track.

A wheel cannot carry submodules or weights, so clone+patch models are a *bench-setup* flow (clone the
repo, run setup), not `pip install srbf[...]`. Note that conflicting deps (Julia, old torch) often
mean each baseline wants its **own environment**; document that for your model.

## PR checklist

- [ ] adapter implements `prepare` + `evaluate_sample`, returns `EvaluationResult` with
      `prediction_success`, `y_pred`/`y_pred_val`, and `predicted_*` keys
- [ ] model library imported lazily (importing `srbf` stays light)
- [ ] `_build_<name>_adapter` + one `_ADAPTER_REGISTRY` entry
- [ ] an example `configs/evaluation/...yaml` selecting `type: <name>`
- [ ] install/provisioning instructions (pip, or `scripts/patch_<name>.py` + weights), added to
      [docs/models.md](models.md)
- [ ] a small smoke test under `tests/`
- [ ] `pre-commit run --all-files` and `pytest tests` pass
