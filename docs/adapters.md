# Adding your model: the adapter contribution guide

srbf is a community benchmark framework. To evaluate **your** symbolic-regression method on the same
benchmarks and metrics as everyone else, you add an **adapter** and open a pull request. We merge it,
run the evaluation on the calibrated reference machine, and publish the numbers. The same config runs
on your own hardware, so you can evaluate first and submit numbers you already know.

## The short version

```bash
pip install srbf
export FLASH_ANSR_ROOT=$PWD/bench       # models, results, environments and adapters live under here
srbf new mymethod                       # bench/adapters/mymethod/: worker.py, config.yaml, requirements.txt, test_worker.py
python -m venv bench/envs/mymethod && bench/envs/mymethod/bin/pip install -r bench/adapters/mymethod/requirements.txt
$EDITOR bench/adapters/mymethod/worker.py            # put your method into fit()
srbf check   -c bench/adapters/mymethod/config.yaml  # a few real problems end to end; names the failing step
srbf run     -c bench/adapters/mymethod/config.yaml -v            # the whole suite (--experiment fastsrb for one catalog)
srbf analyze -c bench/adapters/mymethod/config.yaml -o report     # the standardized report
```

The worker runs in **your own environment**, whatever torch, simplipy or Julia your method was built
against; srbf never imports it. The rest of this page is the detail behind those commands, and the
second route for methods that can live inside srbf's environment.

## Route 1: a worker in your own environment

### The contract

Your worker is a Python file (`srbf new` writes one) that defines one required function and two
optional ones:

```python
# worker.py  -- runs in YOUR venv; srbf is not installed there and need not be
import numpy as np
from mymethod import Model                       # your package, your versions

def load(options):                               # optional: once per run
    return {"model": Model.load(options["checkpoint"], device=options.get("device", "cuda"))}

def info(state):                                 # optional: provenance stored in the run's metadata
    import mymethod
    return {"mymethod": mymethod.__version__}

def fit(x, y, *, x_val, variables, meta, options, state):
    # x: list of rows (one list of floats per support point); y: targets; x_val: validation rows
    # (may be empty); variables: the column names of x, in order; the expression must use them.
    X, Y = np.asarray(x), np.asarray(y)
    expression = state["model"].fit(X, Y, variable_names=variables)   # e.g. "2.0*x1 + sin(x2)"
    return {"expression": expression}
```

`fit` returns at least `"expression"`: an infix string with numeric constants, using the operator
vocabulary of the SimpliPy engine the run is judged with (`+ - * / abs inv neg pow rootn sin cos tan
asin acos atan sinh cosh tanh asinh acosh atanh exp log`; `**` and `sqrt` are read too). srbf parses
it with that engine, evaluates it on the support and validation points, and judges it exactly as it
judges every other adapter's output. Optional keys:

| key | meaning |
|---|---|
| `y_pred`, `y_pred_val` | your method's own predictions, used instead of evaluating the expression |
| `constants` | the fitted constants, recorded as `predicted_constants` |
| `fit_time` | seconds; measured around the call when absent (timing never includes the protocol) |
| `extra` | a JSON-serializable dict merged into the result record (a Pareto front, diagnostics) |
| `error` | a message; the problem counts as failed and the worker stays up for the next one |

Anything the worker prints goes to a log file, never into the protocol. A worker that raises records
the traceback on that problem and continues; a worker that crashes or exceeds `timeout` is restarted
(`max_restarts` times per run) and the problem is recorded as an error. The worker interpreter can be
any Python from 3.8 up; the protocol side is standard library only.

If your method was trained against a simplipy older than 0.12 (the `mult2`/`pow1_3` vocabulary), do
`import srbf_worker_helpers` inside the worker (srbf puts it on the path; standard library only) and
respell your prefix tokens with `respell_legacy_prefix` before rendering them with `prefix_to_infix`.
`src/srbf/worker/models/example_worker.py` is the reference worker (it runs in a bare venv),
`pysr_worker.py` is the PySR baseline, and the protocol itself is documented in
`src/srbf/worker/runner.py`.

### The config

`srbf new` writes this; the whole srbf suite through your worker, one experiment per catalog:

```yaml
suite: srbf                           # every srbf catalog; or a list, e.g. [fastsrb, feynman]
run:
  data_source:
    sampling: {n_support: 512, n_validation: 512, noise: 0.0, problems_per_expression: 1}
  model_adapter:
    type: subprocess
    config_provenance: upstream_default   # see fairness.md
    worker: '{{ROOT}}/adapters/mymethod/worker.py'   # or a built-in name: example, pysr
    python: '{{ROOT}}/envs/mymethod/bin/python'      # the interpreter of YOUR environment
    options: {checkpoint: '{{ROOT}}/models/mymethod/best.pt'}   # forwarded verbatim to load()/fit()
    simplipy_engine: acj-5-4-llm          # the engine the catalogs are judged with; keep it
    timeout: 3600                         # seconds per problem (default: unlimited)
    drop_unused_variables: true           # hand over only the columns the ground truth uses
    worker_log: '{{ROOT}}/results/evaluation/mymethod/worker.log'
  runner:
    output: '{{ROOT}}/results/evaluation/mymethod/{catalog}.pkl'
    save_every: 20
```

`{{ROOT}}` is substituted from `FLASH_ANSR_ROOT`, so one config runs on your machine and on ours;
`{catalog}` is filled in per experiment. A single-catalog config replaces `suite:` with
`data_source.catalog: fastsrb` inside `run:`. Inline `!sweep` blocks give you an inference-time
scaling ladder ([running.md](running.md#inline-sweeps-sweep)); `--experiment`, `--sweep-filter` and
`--shard` select and split the work across a cluster. `srbf check` runs the first experiment on two
real problems and reports every step (interpreter, worker, engine, catalog, fit, metrics) as `ok` or
`FAIL` with the fix beside it.

### What you submit in a PR

Inside an srbf checkout, `srbf new mymethod --repo` writes the same four files where the pull
request wants them:

1. the **worker**, `src/srbf/worker/models/mymethod_worker.py` (referenced as `worker: mymethod`),
2. the **environment recipe**, `envs/mymethod/requirements.txt` (or a lock file) with the exact
   versions your method needs and where the weights come from,
3. the **config**, `configs/evaluation/mymethod_srbf.yaml`, with its
   [`config_provenance` label](fairness.md),
4. the **smoke test**, `tests/test_workers/test_mymethod_worker.py` (one toy problem through the
   adapter; it skips where the environment is not provisioned),

plus a section in [docs/models.md](models.md). No registry entry and no srbf code changes are needed.
See [CONTRIBUTING.md](../CONTRIBUTING.md) for the flow after the PR.

## Route 2: an in-process adapter

If your method installs into srbf's environment without conflicts (`simplipy>=0.14.6`,
`flash-ansr>=0.14`, `torch>=2`), an adapter class avoids the subprocess. The contract is
`srbf.core.EvaluationModelAdapter`, a `@runtime_checkable` `Protocol`: two methods, no base class.

```python
class EvaluationModelAdapter(Protocol):
    def prepare(self, *, data_source: EvaluationDataSource | None = None) -> None:
        """Run once before the first sample (load weights, start a backend, ...)."""

    def evaluate_sample(self, sample: EvaluationSample) -> EvaluationResult:
        """Fit + predict on ONE problem; return a normalized result."""
```

An optional `close()` is called when the run ends, however it ends.

### What you receive: `EvaluationSample`

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

Return `EvaluationResult(record)`, where `record` is a plain `dict` started from
`sample.clone_metadata()` so the ground truth travels with the result. A run records these keys
**raw**; the offline `srbf.result_processing.derive_metrics` step
([running.md](running.md#deriving-metrics)) later turns them into FVU, recovery and F1:

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

### A minimal in-process adapter

```python
import time
import numpy as np
from srbf.core import EvaluationModelAdapter, EvaluationSample, EvaluationResult
from simplipy import normalize_expression, normalize_skeleton


class MyModelAdapter(EvaluationModelAdapter):
    def __init__(self, *, simplipy_engine, **hyperparams):
        from mymodel import MyRegressor  # noqa: F401  (lazy: importing srbf never needs it)
        self.hyperparams = hyperparams
        self.simplipy_engine = simplipy_engine
        self._model = None

    def prepare(self, *, data_source=None):
        from mymodel import MyRegressor
        self._model = MyRegressor(**self.hyperparams)

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
            record["y_pred_val"] = self._model.predict(X_val).reshape(-1, 1) if X_val.size else np.empty((0, 1))
            expr = str(self._model.get_expression())
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

Register it with a builder and a one-line entry in `_ADAPTER_REGISTRY` in `src/srbf/config.py`
(`resolve_simplipy_engine(config, adapter_name="mymodel")` loads the engine), and a config selects it
with `model_adapter: {type: mymodel, ...}`. The built-in `e2e` and `nesymres` adapters in
`src/srbf/model_adapters.py` are the reference examples.

> **The serial driver.** The `Benchmark` driver is a plain serial loop: it calls `evaluate_sample`
> on one problem at a time, so per-problem wall-clock timing is uncontended. Any generate-then-refine
> overlap belongs inside your model's own per-problem inference, not in the adapter.

## Provisioning your model's dependencies

How your model is installed is part of the contribution:

- **own environment** (the worker route): a `requirements.txt`, `environment.yml` or lock file with
  the exact versions, a script that creates the environment and fetches the weights, and the
  `python:` path in the config. This is the default for anything with its own torch, simplipy or Julia.
- **pip dependency** (in-process route): gate the import lazily so importing srbf never pulls it in,
  and document `pip install <yourmodel>`. Pure-pip extras can go under the `[baselines]` group.
- **clone + patch** (unpackaged research code): a `scripts/patch_<name>.py` that pins and patches an
  upstream clone, plus instructions to download weights; how `nesymres` and `e2e` are provisioned.

A wheel cannot carry submodules or weights, so anything beyond pip is a *bench-setup* flow.

## PR checklist

- [ ] a worker (`fit`, optionally `load`/`info`) **or** an in-process adapter (`prepare` +
      `evaluate_sample`, plus a registered builder)
- [ ] an example `configs/evaluation/...yaml` declaring [`config_provenance`](fairness.md)
- [ ] provisioning: the environment recipe with pinned versions and where the weights come from,
      documented in [docs/models.md](models.md)
- [ ] `srbf check -c <config>` passes on your machine
- [ ] a smoke test under `tests/`, skipped when the method's dependencies are absent
- [ ] `pre-commit run --all-files` and `pytest tests` pass
