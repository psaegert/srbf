# Adding your method

To evaluate your symbolic regression method on the same laws and with the same judge as every
other, you write an adapter. The usual adapter is a **worker**: one Python file that runs in your
method's own environment. srbf never imports your code, so your method keeps whatever versions of
torch, NumPy or Julia it was built against.

## The short version

```bash
pip install srbf
export FLASH_ANSR_ROOT=$PWD/bench
srbf new mymethod                                     # bench/adapters/mymethod/: worker.py, config.yaml, requirements.txt, test_worker.py
python -m venv bench/envs/mymethod
bench/envs/mymethod/bin/pip install -r bench/adapters/mymethod/requirements.txt
$EDITOR bench/adapters/mymethod/worker.py             # put your method into fit()
srbf check   -c bench/adapters/mymethod/config.yaml   # a few real problems, step by step
srbf run     -c bench/adapters/mymethod/config.yaml -v
srbf analyze -c bench/adapters/mymethod/config.yaml -o report
```

The [Quickstart](quickstart.md) runs these commands with the placeholder that `srbf new` writes.

## The worker

A worker defines one function and, if it wants, two more:

```python
# worker.py: runs in YOUR environment; srbf is not installed there and need not be
import numpy as np
from mymethod import Model


def load(options):                                   # optional, once per run
    return {"model": Model.load(options["checkpoint"], device=options.get("device", "cuda"))}


def info(state):                                     # optional: what identifies this method
    import mymethod
    return {"mymethod": mymethod.__version__}


def fit(x, y, *, x_val, variables, meta, options, state):
    X, Y = np.asarray(x), np.asarray(y)
    expression = state["model"].fit(X, Y, variable_names=variables)     # e.g. "2.0*v1 + sin(v2)"
    return {"expression": expression}
```

| argument | what it holds |
|---|---|
| `x` | the support points, a list of rows, each a list of floats |
| `y` | their targets, a list of floats; the noisy ones when the run adds noise |
| `x_val` | the validation points, or an empty list. Their targets are never handed over |
| `variables` | the names of the columns of `x`, in order, as the catalog writes them (`v1, v2, ...` in the srbf catalogs). Write the expression in these names |
| `meta` | identifiers of the problem for your logs: `catalog`, `benchmark_eq_id`, `eq_id`, `eval_row_index`, `n_support`, `noise_level`. The law itself never reaches the worker |
| `options` | the `options` block of the config, with `{{ROOT}}` replaced in its strings |
| `state` | whatever `load` returned; `None` without a `load` |

Arguments are plain Python lists and numbers. The worker side of the protocol uses the standard
library only and runs on Python 3.8 and newer.

### What `fit` returns

A dict with at least `expression`, or with `error`:

| key | meaning |
|---|---|
| `expression` | the answer, an infix string with numeric constants |
| `y_pred`, `y_pred_val` | your method's own values on `x` and `x_val`. Without them srbf evaluates the expression itself. With them the numeric metrics are computed from these values, so return them only if they are what the expression computes |
| `constants` | the fitted constants, stored as `predicted_constants` |
| `fit_time` | seconds. Without it, the time around the call to `fit` is taken; the protocol and srbf's own work are never part of it |
| `extra` | a dict of anything JSON can hold, merged into the result row: a Pareto front, diagnostics. A key that a row already has is kept under `worker_extra` instead |
| `error` | a message. The problem counts as failed and the worker goes on with the next one |

An exception in `fit` is recorded as that problem's error, with its traceback, and the worker stays
up. Whatever the worker prints goes to its log file and never into the protocol. What `info`
returns is stored with every result file.

### How an expression is read

The expression is parsed by the judge's SimpliPy engine. It reads

- `+ - * /`, parentheses, and powers written as `**`, `^` or `pow(a, b)`,
- the functions `abs inv neg sin cos tan asin acos atan sinh cosh tanh asinh acosh atanh exp log`,
  `rootn(a, n)` for the n-th root, and `sqrt(a)`,
- numbers such as `2`, `-0.5` and `1e-3`, and the constants `pi` and `e`,
- the variable names it handed you.

What sympy prints is read as it is: `str(expression)` may contain `sqrt`, `Abs` and the constant
`E`. Print constants at full precision (`repr(float)`); a rounded constant costs the answer its
numeric recovery.

srbf then evaluates the expression on the support and validation points and judges it like every
other method's answer. The stored answer names the columns `x1, x2, ...` by position, whatever the
catalog called them. An expression that cannot be parsed or evaluated is a failed prediction with
the reason in `error`: a function outside the list is named in it, as in `the judge does not read
erf`. `srbf check` shows this for real problems before you start a long run.

If your method writes prefix notation, `import srbf_worker_helpers` inside the worker (srbf puts it
on the path) gives you `prefix_to_infix(tokens)` and
`substitute_placeholders(tokens, values)`, which fills `<constant>` placeholders with fitted
values.

## The config

`srbf new` writes the whole srbf suite through your worker, one experiment per catalog:

```yaml
suite: srbf                        # every srbf catalog; or a list, e.g. [fastsrb, feynman]
run:
  data_source:
    sampling:
      n_support: 512
      n_validation: 512
      noise: 0.0
      problems_per_expression: 1
  model_adapter:
    type: subprocess
    config_provenance: upstream_default
    worker: '{{ROOT}}/adapters/mymethod/worker.py'
    python: '{{ROOT}}/envs/mymethod/bin/python'
    options: {}
    simplipy_engine: acj-5-4-llm
    timeout: 3600
    worker_log: '{{ROOT}}/results/evaluation/mymethod/worker.log'
  runner:
    output: '{{ROOT}}/results/evaluation/mymethod/{catalog}.pkl'
    save_every: 20
    resume: true
```

Whatever your method needs to know goes into `options`, for example
`options: {checkpoint: "{{ROOT}}/models/mymethod/best.pt", device: cuda}`. For a budget ladder, put
the budget into `options` with a `!sweep`; [Running evaluations](running.md#on-a-cluster) has the
complete config and the job array that runs it.

### The config keys

| key | default | meaning |
|---|---|---|
| `worker` | required | the worker file, or the name of a built-in worker (`example`, `pysr`) |
| `python` | srbf's interpreter | the interpreter of your method's environment |
| `options` | `{}` | handed to `load` and to every `fit`; `{{ROOT}}` is replaced in its strings |
| `simplipy_engine` | required | the engine the answers are judged with; keep `acj-5-4-llm` for the srbf suite |
| `config_provenance` | `harness_tuned` | who chose this configuration ([Fairness](fairness.md#configuration-provenance-labels)) |
| `env` | `{}` | environment variables for the worker, added to the environment srbf runs in, for example `{CUDA_VISIBLE_DEVICES: "0"}` |
| `cwd` | none | the worker's working directory |
| `worker_log` | none | a file that receives everything the worker prints |
| `startup_timeout` | `600` | seconds the worker may take to start and run `load` |
| `timeout` | none | seconds one `fit` may take. When it passes, the worker is stopped, the problem is recorded as an error and a new worker is started |
| `max_restarts` | `1` | how many crashes or timeouts a run survives. After that every remaining problem is recorded as an error |

**Searches that stall.** Two optional policies stop a fit that hangs; both are off unless you set
them. A search that hangs usually still burns a CPU thread, so it only shows in the time it takes.
`hang_overdue_factor: 30` treats a fit as hung when it takes more than 30 times
the run's median fit, and never less than `hang_overdue_floor_s` (60 seconds), once
`hang_overdue_min_history` (10) fits have answered. `hang_after_idle_s: 120` treats a fit as hung
when the worker's processes used less than `hang_idle_cpu_s` (1) CPU second in the last 120 seconds.
A hung fit is stopped, the worker restarted without touching `max_restarts`, and the problem tried
once more; `hang_log` names a file that records every such event.

## Opening a pull request

Inside an srbf checkout that is installed with `pip install -e .`, `srbf new mymethod --repo`
writes the same four files where a pull request wants them:

1. the worker, `src/srbf/worker/models/mymethod_worker.py`, which a config then refers to as
   `worker: mymethod`,
2. the environment, `envs/mymethod/requirements.txt` or a lock file, with exact versions and where
   the weights come from,
3. the config, `configs/evaluation/mymethod_srbf.yaml`, with its `config_provenance` label,
4. a smoke test, `tests/test_workers/test_mymethod_worker.py`, which fits one toy problem and is
   skipped where the environment does not exist,

and you add a section to [Models](models.md). No other srbf code changes. Before opening it:

- [ ] `srbf check -c <config>` passes on your machine
- [ ] the environment recipe pins versions and says where the weights come from
- [ ] the config declares its `config_provenance`
- [ ] `pre-commit run --all-files` and `pytest tests` pass

After the merge we aim to run the method on the whole suite under the standard protocol and to
publish its numbers on the [results explorer](https://psaegert.github.io/srbf/), as compute
allows. Your own runs use the same config, so up to sampling noise the recovery numbers you measure
are the numbers that get published; only the times depend on the machine
([Fairness](fairness.md#how-time-is-measured)).

## An adapter inside srbf's environment

A method that installs next to srbf without conflicts (Python 3.12 or newer, `simplipy` 0.14,
`torch` 2) can skip the subprocess with an adapter class. The contract is two methods:

```python
class EvaluationModelAdapter(Protocol):
    def prepare(self, *, data_source=None) -> None: ...             # once, before the first problem
    def evaluate_sample(self, sample: EvaluationSample) -> EvaluationResult: ...
```

An optional `close()` is called when the run ends, however it ends. A sample carries `x_support`,
`y_support`, `y_support_noisy` (or `None`), `x_validation`, `y_validation` and `metadata`. The
result is a dict started from `sample.clone_metadata()`, so that the law travels with the answer.
An in-process adapter writes its expression in the names `x1, x2, ...`, by column position:

```python
import time
import numpy as np
from symbolic_data.token_ops import normalize_expression, normalize_skeleton
from srbf.core import EvaluationResult, EvaluationSample


class MyMethodAdapter:
    def __init__(self, *, simplipy_engine, **hyperparameters):
        self.simplipy_engine = simplipy_engine
        self.hyperparameters = hyperparameters
        self._model = None

    def prepare(self, *, data_source=None):
        from mymethod import Regressor                # imported here, so that importing srbf never needs it
        self._model = Regressor(**self.hyperparameters)

    def evaluate_sample(self, sample: EvaluationSample) -> EvaluationResult:
        record = sample.clone_metadata()
        y = sample.y_support_noisy if sample.y_support_noisy is not None else sample.y_support
        start = time.perf_counter()
        try:
            self._model.fit(sample.x_support, y.ravel())
            record["fit_time"] = time.perf_counter() - start
            prefix = self.simplipy_engine.read_infix(str(self._model.expression()))
            record["predicted_expression"] = str(self._model.expression())
            record["predicted_expression_prefix"] = normalize_expression(prefix)
            record["predicted_skeleton_prefix"] = normalize_skeleton(prefix)
            record["y_pred"] = self._model.predict(sample.x_support).reshape(-1, 1)
            record["y_pred_val"] = self._model.predict(sample.x_validation).reshape(-1, 1)
            record["prediction_success"] = True
        except Exception as exc:                      # a failure is a result, not a crash
            record["error"] = str(exc)
            record["prediction_success"] = False
        return EvaluationResult(record)


def build(config):                                    # the model_adapter block of the config
    from simplipy import SimpliPyEngine
    engine = SimpliPyEngine.load(config["simplipy_engine"], install=True)
    return MyMethodAdapter(simplipy_engine=engine, **config.get("hyperparameters", {}))
```

A config names the builder as `module:function`, and nothing in srbf has to change:

```yaml
model_adapter:
  type: mymethod.srbf_adapter:build
  config_provenance: upstream_default
  simplipy_engine: acj-5-4-llm
  hyperparameters: {beam: 8}
```

A prediction that failed counts as a miss. An exception that escapes the adapter is recorded the
same way, with the exception as the row's `error`
([Results](results.md#failures-and-placeholders)); catching your method's failures as above lets
you say more about them. The `e2e` and `nesymres` adapters in
`src/srbf/model_adapters.py` are worked examples. The driver fits one problem at a time, so nothing
competes with your method for the machine while it is timed.
