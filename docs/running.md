# Running evaluations

`srbf` evaluates a symbolic-regression model against a benchmark from a single YAML
config. This page covers the `srbf run` CLI and its flags, the anatomy of a config, how
to select and run experiments, what lands in the output pickle, resuming a partial run,
and the programmatic `build_evaluation_run` API.

See also:
[benchmarks.md](benchmarks.md) (preparing benchmark sets),
[models.md](models.md) (the built-in adapters and their config keys),
[adapters.md](adapters.md) (add your own model),
and the [project README](https://github.com/psaegert/srbf/blob/main/README.md).

## Prerequisites

`srbf` resolves config, data, and model paths through `flash-ansr`'s shared project
root. The `{{ROOT}}` token used throughout the configs is substituted with the
`FLASH_ANSR_ROOT` environment variable, so point it at a checkout that holds your
`configs/`, `data/`, and `models/`:

```bash
export FLASH_ANSR_ROOT=/path/to/srbf
```

You also need:

- A model to evaluate (e.g. a flash-ansr checkpoint: `flash_ansr install psaegert/flash-ansr-v23.0-120M`). See [models.md](models.md).
- A prepared benchmark at the `benchmark_path` your config points to (FastSRB ships as an `expressions.yaml`). See [benchmarks.md](benchmarks.md).

## The `srbf run` command

There is exactly one subcommand, `run`:

```bash
srbf run -c configs/evaluation/scaling/v23.0-20M_fastsrb.yaml -v
```

### Flags

| Flag | Meaning |
| --- | --- |
| `-c`, `--config` | Path to the evaluation config (required). |
| `-n`, `--limit` | Override the sample limit from the config. Useful for a quick smoke test (`-n 2`). |
| `-o`, `--output-file` | Override the output pickle path from the config. |
| `--save-every` | Override how often (in samples) results are flushed to disk. |
| `--no-resume` | Ignore any existing output pickle and start fresh (default behavior resumes). |
| `--experiment` | Name of a single experiment to run when the config defines several (see below). |
| `-v`, `--verbose` | Print a progress bar and per-experiment status. |

A quick smoke test of one experiment:

```bash
srbf run \
  -c configs/evaluation/scaling/v23.0-20M_fastsrb.yaml \
  --experiment flash_ansr_fastsrb_choices_00032 \
  --limit 2 -v
```

## Config anatomy

A config is a YAML file with a `defaults:` block (three anchored sub-blocks) followed by
an `experiments:` map. The `defaults:` block defines reusable anchors; each experiment
references them with YAML anchors (`*name`), merges them with `<<:`, and applies a few
overrides. The three sub-blocks are `data_source`, `model_adapter`, and `runner`.

Here is the structure, abridged from
[`configs/evaluation/scaling/v23.0-20M_fastsrb.yaml`](https://github.com/psaegert/srbf/blob/main/configs/evaluation/scaling/v23.0-20M_fastsrb.yaml):

```yaml
defaults:
  data_source: &flash_ansr_fastsrb_source
    type: fastsrb
    benchmark_path: "{{ROOT}}/data/ansr-data/test_set/fastsrb/expressions.yaml"
    datasets_per_expression: 10
    support_points: 512
    sample_points: 1024
    method: random
    max_trials: 100
    incremental: false
    random_state: 42
    noise_level: 0.0
  model_adapter: &flash_ansr_adapter
    type: flash_ansr
    model_path: "{{ROOT}}/models/ansr-models/v23.0-20M"
    evaluation_config:
      n_support: 512
      n_restarts: 8
      refiner_method: curve_fit_lm
      refiner_p0_noise: normal
      refiner_p0_noise_kwargs: {loc: 0.0, scale: 5}
      length_penalty: 0.05
      generation_config:
        method: softmax_sampling
        kwargs: {choices: 32, max_len: 64, batch_size: 128, temperature: 1, simplify: true}
      device: cuda
    device: cuda
  runner: &flash_ansr_fastsrb_runner
    limit: null
    save_every: 64
    resume: true

experiments:
  flash_ansr_fastsrb_choices_00032:
    run:
      data_source: *flash_ansr_fastsrb_source
      model_adapter:
        <<: *flash_ansr_adapter
        generation_overrides:
          kwargs:
            choices: 32
      runner:
        <<: *flash_ansr_fastsrb_runner
        output: "{{ROOT}}/results/evaluation/scaling/v23.0-20M/fastsrb/choices_00032.pkl"
```

### `data_source`

Describes the data to evaluate on. The example uses `type: fastsrb`, which samples
datasets from a FastSRB `expressions.yaml` at `benchmark_path`:

- `benchmark_path`: path to the FastSRB `expressions.yaml`.
- `datasets_per_expression`: how many noisy datasets to draw per benchmark equation. Total rows evaluated is `len(equation_ids) * datasets_per_expression`.
- `support_points` / `sample_points`: number of fit points and held-out validation points sampled per dataset.
- `method`, `max_trials`, `incremental`: sampling controls for drawing valid support points.
- `noise_level`: relative Gaussian noise added to the targets.
- `random_state`: seed for reproducible sampling.

The other data-source type is `type: skeleton_dataset`, used by the `*_val.yaml` configs.
It samples from a flash-ansr skeleton pool (`dataset: "{{ROOT}}/configs/.../dataset_val.yaml"`)
and typically pins the count with `target_size`. See [benchmarks.md](benchmarks.md) for
building and pinning these sets.

### `model_adapter`

Selects the model under test via `type:` plus per-type parameters. The closed registry of
adapter types is:

`flash_ansr`, `pysr`, `nesymres`, `skeleton_pool`, `brute_force`, `e2e`.

Each type reads a different set of keys (e.g. `flash_ansr` takes `model_path` +
`evaluation_config` + `device`; `pysr` takes `niterations` + `timeout_in_seconds` +
`simplipy_engine`). Experiments commonly override just the compute knob through
`generation_overrides` (flash_ansr) or a direct key like `niterations` (pysr). The full
per-adapter key reference lives in [models.md](models.md).

### `runner`

Controls the evaluation loop and persistence:

- `limit`: max samples to evaluate. `null` means "evaluate the whole data source".
- `save_every`: flush the pickle to disk every N samples. When set, `output` must be set too, or the run errors out. There is always a final save at the end regardless of `save_every`.
- `resume`: when `true` (default), continue an existing output pickle instead of recomputing it.
- `output`: destination pickle path. Defined per experiment so each writes its own file.

### Anchors and `{{ROOT}}`

The `&name` / `*name` / `<<:` syntax is plain YAML: `&` defines an anchor, `*` references
it, and `<<:` merges a mapping so an experiment can inherit a default block and override a
few keys. `{{ROOT}}` in any path field is replaced with `FLASH_ANSR_ROOT` at load time, so
configs stay portable across machines.

## Selecting experiments

A config's `experiments:` map can hold one sweep point or many (the scaling configs hold a
full sweep, e.g. `choices_00001` ... `choices_262144`).

- `srbf run -c <config> --experiment <name>` runs exactly that one experiment.
- `srbf run -c <config>` with no `--experiment` runs **all** experiments in the config, sequentially, each writing to its own `runner.output`.

List the experiment names directly from the config (they are the keys under
`experiments:`), for example by reading the file or:

```bash
python -c "import yaml,sys; print(*yaml.safe_load(open(sys.argv[1]))['experiments'], sep='\n')" \
  configs/evaluation/scaling/v23.0-20M_fastsrb.yaml
```

## Outputs

Each experiment writes a pickle to its `runner.output`, by convention under
`results/evaluation/.../*.pkl`. The pickle is a **column-oriented dict-of-lists** (load it
straight into a `pandas.DataFrame`); one index `i` across all columns is one evaluated
dataset. The row count equals `len(equation_ids) * datasets_per_expression` (capped by
`runner.limit` / `--limit` if set).

```python
import pandas as pd
df = pd.read_pickle("results/evaluation/scaling/v23.0-20M/fastsrb/choices_00032.pkl")
print(len(df), df.columns.tolist())
```

Alongside metrics, rows carry the raw sampled data (`x`, `y`, `y_pred`, `y_val`, ...) and
the pickle embeds a `__meta__` provenance entry (config path, git state). `srbf` strips
`__meta__` automatically on resume.

### Key metric columns

| Column | Meaning |
| --- | --- |
| `fvu_fit`, `fvu_val` | Fraction of variance unexplained on the fit and held-out points. |
| `log10_fvu_fit`, `log10_fvu_val` | `log10` of the above (the strict-metric distribution form). |
| `numeric_recovery_fit`, `numeric_recovery_val` | Numeric recovery (FVU below the float-precision threshold). |
| `symbolic_recovery` | Whether the predicted skeleton matches the ground truth symbolically. |
| `f1_score` | Token-level F1 between predicted and ground-truth skeleton. |
| `edit_distance`, `zss_edit_distance` | Tree/token edit distances. |
| `n_constants`, `predicted_n_constants`, `n_constants_delta` | Constant counts. |
| `placeholder`, `placeholder_reason` | Placeholder bookkeeping (see below). |

This is the load-bearing subset. The complete schema (~30 columns, including
`only_approx_fvu_*`, skeleton lengths, unique-variable precision/recall, nestedness, and
the raw `x`/`y` arrays) is the keys of the `_DEFAULTS` dict in
[`src/srbf/eval/result_processing.py`](https://github.com/psaegert/srbf/blob/main/src/srbf/eval/result_processing.py).

### Placeholders

When sample generation fails for a dataset (e.g. no valid support points within
`max_trials`, or an adapter raises), `srbf` records a **placeholder** row instead of
silently dropping it. Such a row has `placeholder=True`, a `placeholder_reason`, and
default/sentinel metric values. This keeps row counts aligned with the configured total so
sweeps remain comparable. Filter them out before any fit-based aggregation:

```python
df = df[~df["placeholder"].astype(bool)]
```

## Resuming

With `runner.resume: true` (the default), `srbf run` loads the existing output pickle, sees
how many rows are already present, and evaluates only the remaining samples before saving
again. A run interrupted partway resumes from where it stopped; a completed run is a no-op.
Because saves are atomic (write-to-temp then replace), a process killed mid-write never
leaves a corrupt pickle.

Use `--no-resume` to ignore any existing output and recompute from scratch.

## Programmatic API

`build_evaluation_run` returns a plan describing how to execute a run; you then call
`plan.engine.run(...)`. Because the scaling configs define multiple experiments and no
`default_experiment`, you must pass `experiment=` (unlike the CLI, which loops over all of
them when none is given). Guard on `plan.completed`: when a run is already finished,
`plan.engine` is `None`.

```python
from srbf import build_evaluation_run

plan = build_evaluation_run(
    config="configs/evaluation/scaling/v23.0-20M_fastsrb.yaml",
    experiment="flash_ansr_fastsrb_choices_00032",
)

if not plan.completed and plan.engine is not None:
    plan.engine.run(
        limit=plan.remaining,
        save_every=plan.save_every,
        output_path=plan.output_path,
    )
```

`build_evaluation_run` also accepts `limit_override`, `output_override`,
`save_every_override`, and `resume` (set `resume=False` for the `--no-resume` behavior),
mirroring the CLI flags. The returned `EvaluationRunPlan` exposes `engine`, `remaining`,
`save_every`, `output_path`, `completed`, `total_limit`, and `existing_results`.

## Adapters and out-of-band baselines

The built-in adapters (`flash_ansr`, `pysr`, `nesymres`, `skeleton_pool`, `brute_force`,
`e2e`) are **reference examples** of the adapter contract. `pip install srbf` ships the
engine, metrics, and the pip-installable adapters (`flash_ansr`, `pysr`); `pip install
"srbf[baselines]"` adds the baseline deps (sympy, pysr, omegaconf).

The unpackaged research baselines (`nesymres`, `e2e` / `symbolicregression`) are not pip
dependencies. Their upstream source trees and weights are provisioned out-of-band via a
clone-and-patch flow using the scripts under
[`scripts/`](https://github.com/psaegert/srbf/tree/main/scripts): `patch_nesymres.py` and `patch_symbolicregression.py` (each takes
the path to the corresponding clone) plus `patch_typing_io.py` (patches the active
environment, no args); weights download separately. This is **one default recipe**, not the
only way to wire those models in. See [models.md](models.md) for the exact per-baseline steps.

`srbf` is a community framework: a new SR method is added by PR with an adapter plus its
own install instructions. See [adapters.md](adapters.md) for the contribution guide.
