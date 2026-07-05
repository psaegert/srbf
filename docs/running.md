# Running evaluations

`srbf` evaluates a symbolic-regression model against a benchmark from a single YAML
config. This page covers the `srbf run` CLI and its flags, the anatomy of a config, inline
`!sweep` cross-products and `experiments:`, what lands in the output pickle, resuming a
partial run, multi-draw bootstrap reporting, and the programmatic `Benchmark` API.

See also:
[benchmarks.md](benchmarks.md) (the `data_source` catalog block and the shipped catalogs),
[models.md](models.md) (the built-in adapters and their config keys),
[adapters.md](adapters.md) (add your own model),
and the [project README](https://github.com/psaegert/srbf/blob/main/README.md).

## Prerequisites

`srbf` resolves config and model paths through `flash-ansr`'s shared project root. The
`{{ROOT}}` token used throughout the configs is substituted with the `FLASH_ANSR_ROOT`
environment variable, so point it at a checkout that holds your `configs/`, `models/`, and
`results/`:

```bash
export FLASH_ANSR_ROOT=/path/to/srbf
```

You also need:

- A model to evaluate (e.g. a flash-ansr checkpoint: `flash_ansr install psaegert/flash-ansr-v23.0-120M`). See [models.md](models.md).
- The benchmark the config names. The `data_source.catalog` field is a `symbolic-data` catalog (e.g. `v23-val`, `fastsrb`); it is fetched from Hugging Face on first use and cached, so there is no local data-build step. See [benchmarks.md](benchmarks.md).

## The `srbf run` command

There is exactly one subcommand, `run`:

```bash
srbf run -c configs/evaluation/scaling/v23.0-20M_fastsrb.yaml -v
```

### Flags

| Flag | Meaning |
| --- | --- |
| `-c`, `--config` | Path to the evaluation config (required). `{{ROOT}}` is resolved against `FLASH_ANSR_ROOT`. |
| `-n`, `--limit` | Override the sample limit from the config. Useful for a quick smoke test (`-n 2`). |
| `-o`, `--output-file` | Override the output pickle path from the config. |
| `--save-every` | Override how often (in samples) results are flushed to disk. |
| `--no-resume` | Ignore any existing output pickle and start fresh (default behavior resumes). |
| `--experiment` | Name of a single experiment to run when the config defines an `experiments:` map. |
| `--sweep-filter` | Run only the `!sweep` runs whose axis labels match, e.g. `--sweep-filter ladder=256` (comma-separate several: `AXIS=VALUE,AXIS=VALUE`). |
| `-v`, `--verbose` | Print a progress bar and per-run status. |

A quick smoke test of a single sweep rung:

```bash
srbf run \
  -c configs/evaluation/scaling/v23.0-20M_fastsrb.yaml \
  --sweep-filter ladder=32 \
  --limit 2 -v
```

`srbf run` expands the config into one run per resolved `(experiment, sweep-axis)`
combination via `Benchmark.runs_from_config`, then runs each one serially. The model for a
run is loaded lazily, so a run whose configured target is already complete prints
"already completed" and never loads its model.

## Config anatomy

A config has a top-level `run:` block with three sub-blocks: `data_source`,
`model_adapter`, and `runner`. (A config may instead carry an `experiments:` map of named
`run:` blocks; see [Selecting experiments](#selecting-experiments).)

Here is the structure, abridged from
[`configs/evaluation/scaling/v23.0-3M_fastsrb.yaml`](https://github.com/psaegert/srbf/blob/main/configs/evaluation/scaling/v23.0-3M_fastsrb.yaml):

```yaml
run:
  data_source:
    catalog: fastsrb              # a symbolic-data catalog name/ref (or local path / inline config)
    sampling:
      n_support: 512              # points the model fits on
      n_validation: 512           # held-out points (omit for catalogs that carry their own validation)
      noise: 0.0                  # Gaussian noise as a fraction of the target std (0 = clean)
      problems_per_expression: 10 # distinct sampled problems per ground-truth expression
  model_adapter:
    type: flash_ansr
    model_path: "{{ROOT}}/models/ansr-models/v23.0-3M"
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
  runner:
    limit: null
    save_every: 64
    resume: true
    output: "{{ROOT}}/results/evaluation/scaling/v23.0-3M/fastsrb/choices_00032.pkl"
```

### `data_source`

The data source is always a `symbolic-data` catalog. The `catalog` field names which set of
ground-truth expressions to evaluate on; `sampling` is this run's usage policy over that catalog
(how many problems to draw per expression, how many fit/validation points, noise). The catalog owns
all generation, fixed-set iteration, noise injection, and decontamination.

```yaml
data_source:
  catalog: v23-val              # a catalog name/ref, an HF 'user/repo:name' ref, a local path, or an inline config
  sampling:                     # the per-run usage policy (all fields optional)
    n_support: 512
    n_validation: 1024
    noise: 0.0
    problems_per_expression: 10
    method: iterate             # frozen catalog -> 'iterate'; open generative catalog -> 'procedural'
  holdouts:                     # optional decontamination / filters
    - exclude: lample-charton-v23
    - filter: {finite: true}
  target_size: 1000             # cap the number of rows (also honoured as the run total)
```

The shipped catalogs are `v23-val` (the frozen, sha-pinned validation set), `fastsrb` (the FastSRB
benchmark), and `lample-charton-v23` (the generative v23 training recipe). See
[benchmarks.md](benchmarks.md) for catalog references, the shipped catalogs, and pointing at your
own.

### `model_adapter`

Selects the model under test via `type:` plus per-type parameters. The registry of adapter
types is:

`flash_ansr`, `pysr`, `nesymres`, `e2e`, `lample_charton`, `brute_force`.

Each type reads a different set of keys (e.g. `flash_ansr` takes `model_path` +
`evaluation_config` + `device`; `pysr` takes `niterations` + `timeout_in_seconds` +
`simplipy_engine`). `flash_ansr` loads its SimpliPy engine from the model; every other
adapter requires an explicit `model_adapter.simplipy_engine`. The full per-adapter key
reference lives in [models.md](models.md).

### `runner`

Controls the evaluation loop and persistence:

- `limit`: max samples to evaluate. `null` defers to `data_source.target_size`, or to the catalog's own size for a frozen set, or runs unbounded for an open generative source.
- `save_every`: flush the pickle to disk every N samples. When set, `output` must be set too, or the run errors out. There is always a final save at the end regardless of `save_every`.
- `resume`: when `true` (default), continue an existing output pickle instead of recomputing it.
- `output`: destination pickle path. One per run so each writes its own file.

### `{{ROOT}}`

`{{ROOT}}` in any path field is replaced with `FLASH_ANSR_ROOT` at load time, so configs
stay portable across machines.

## Inline sweeps (`!sweep`)

A `!sweep` YAML tag marks a value that varies across runs, so one config expands to many
matched runs. There are two forms:

- `!sweep [v1, v2, ...]` is an **anonymous axis**: it forms its own dimension of the cross-product (grid) with every other anonymous sweep.
- `!sweep {name: <axis>, values: [v1, v2, ...]}` is a **named axis**: every `!sweep` sharing the same `name` advances together (element-wise zip), so co-named sweeps must be equal length. Different names form separate cross-product dimensions.

The scaling configs use a single named axis, `ladder`, carried by the swept rung field
(`choices` / `samples` / `niterations`), `problems_per_expression`, `target_size`, and the
per-rung `output` path. Because all those `!sweep`s share the name `ladder`, they zip into
one dimension: the N-rung ladder is N matched runs, not a cross-product. `--sweep-filter
ladder=<value>` runs exactly one rung.

```yaml
run:
  data_source:
    catalog: v23-val
    sampling:
      n_support: 512
      n_validation: 512
      problems_per_expression: !sweep {name: ladder, values: [10, 10, 10]}
    target_size: !sweep {name: ladder, values: [1000, 1000, 1000]}
  model_adapter:
    type: flash_ansr
    # ...
    generation_overrides:
      kwargs:
        choices: !sweep {name: ladder, values: [1, 32, 1024]}
  runner:
    save_every: 64
    output: !sweep {name: ladder, values: ['.../choices_00001.pkl', '.../choices_00032.pkl', '.../choices_01024.pkl']}
```

## Selecting experiments

A config may instead define an `experiments:` map of named `run:` blocks (e.g. one
model-variant per entry). `!sweep` and `experiments:` compose: each experiment is expanded
by `!sweep` independently.

- `srbf run -c <config> --experiment <name>` runs only that experiment.
- `srbf run -c <config>` with no `--experiment` runs **all** experiments, each writing to its own `runner.output`.

List the experiment names directly from the config (the keys under `experiments:`):

```bash
python -c "import yaml,sys; print(*yaml.safe_load(open(sys.argv[1]))['experiments'], sep='\n')" \
  <your-config>.yaml
```

## Outputs

Each run writes a pickle to its `runner.output`, by convention under
`results/evaluation/.../*.pkl`. The pickle is a **column-oriented dict-of-lists** (load it
straight into a `pandas.DataFrame`); one index `i` across all columns is one evaluated
problem. The row count equals the number of expressions in the catalog times
`problems_per_expression` (capped by `data_source.target_size`, `runner.limit`, or
`--limit` if set).

```python
import pandas as pd
df = pd.read_pickle("results/evaluation/scaling/v23.0-3M/fastsrb/choices_00032.pkl")
print(len(df), df.columns.tolist())
```

A run emits **raw results only**: each row carries the ground-truth metadata, the raw
sampled arrays (`x`, `y`, `y_noisy`, `x_val`, `y_val`, ...), and the adapter's raw prediction
(`y_pred`, `y_pred_val`, `predicted_*`, timings). It does **not** compute FVU / recovery /
F1: those are **derived metrics** produced by a separate step (see [Deriving
metrics](#deriving-metrics)). The pickle also embeds a `__meta__` provenance entry (config
path, git state, sweep labels, and the declared `config_provenance` label — see
[fairness](fairness.md)), which `srbf` strips automatically on resume.

### Raw run columns

The columns a `Benchmark.run()` snapshot actually contains:

| Column | Meaning |
| --- | --- |
| `skeleton`, `expression`, `variables`, `variable_names` | Ground-truth skeleton / expression / variables. |
| `ground_truth_prefix`, `ground_truth_infix` | Ground-truth expression in prefix / infix form. |
| `x`, `y`, `y_noisy`, `x_val`, `y_val`, `y_noisy_val` | Raw sampled fit / validation arrays. |
| `n_support`, `complexity`, `noise_level` | Per-problem sampling parameters. |
| `predicted_expression`, `predicted_expression_prefix`, `predicted_skeleton_prefix` | The normalized prediction (infix / prefix / skeleton). |
| `predicted_constants`, `predicted_score`, `predicted_log_prob` | The best candidate's fitted constants and scores. |
| `y_pred`, `y_pred_val` | Model predictions on the fit / held-out points. |
| `fit_time`, `prediction_success`, `error` | Wall-clock fit time, success flag, error string. |
| `benchmark_eq_id` | Ground-truth expression id; groups the `problems_per_expression` draws of one expression. |
| `placeholder`, `placeholder_reason` | Placeholder bookkeeping (see below). |
| `input_ids`, `labels`, `labels_decoded`, `eval_row_index` | Tokenized skeleton + resume-stable row index. |

(The `flash_ansr` adapter additionally records `generation_time` / `refinement_time`.)

### Deriving metrics

The derived metrics (`fvu_fit`, `fvu_val`, `log10_fvu_*`, `numeric_recovery_fit`,
`numeric_recovery_val`, `symbolic_recovery`, `f1_score`, `n_constants`,
`predicted_n_constants`, skeleton lengths, edit distances, unique-variable
precision/recall, ...) are computed **after** the run by `srbf.derive_metrics`. It takes one
raw run snapshot and returns a **new** snapshot with the metric columns added, without mutating
the input:

```python
from srbf import Benchmark, derive_metrics, bootstrap_report

(benchmark,) = Benchmark.runs_from_config(single_run_config_path)
snapshot = benchmark.run()      # raw dict-of-lists, no derived metrics

scored = derive_metrics(snapshot, engine=benchmark.model_adapter.get_simplipy_engine())
report = bootstrap_report(scored, "numeric_recovery_val")   # composes with the reporting helpers
```

Pass an `engine` (its `operator_arity` + `simplify` are used) **or** an explicit
`operator_arity` mapping (operator token -> arity, needed for tree-edit-distance and
nestedness); `simplify_fn` is optional (defaults to the engine's, else simplified skeletons
fall back to the raw ones). You can equally compute your own metrics directly over the raw
columns. The full set of derived keys (and the placeholder defaults used for failed rows) is
documented on `derive_metrics` / `compute_derived_metrics` and in the `DEFAULT_NEGATIVES` dict
in [`src/srbf/result_processing.py`](https://github.com/psaegert/srbf/blob/main/src/srbf/result_processing.py).

`derive_metrics` is the ergonomic wrapper over the lower-level `srbf.compute_derived_metrics`,
which mutates a nested `results[model]['results'][test_set][scaling_value]` dict in place (the
shape the cross-run analysis routine builds); reach for it directly only when you already hold
that nested structure.

### Placeholders

When a problem cannot be produced (e.g. no valid support points within `max_trials`, or an
adapter raises), `srbf` records a **placeholder** row instead of silently dropping it. Such
a row has `placeholder=True`, a `placeholder_reason`, and default/sentinel metric values.
This keeps row counts aligned with the configured total so sweeps remain comparable. Filter
them out before any fit-based aggregation:

```python
df = df[~df["placeholder"].astype(bool)]
```

## Resuming

With `runner.resume: true` (the default), `srbf run` loads the existing output pickle, sees
how many rows are already present, and evaluates only the remaining samples before saving
again. A run interrupted partway resumes from where it stopped; a completed run is a no-op
(its model is never loaded). Use `--no-resume` to ignore any existing output and recompute
from scratch.

## Reporting

`symbolic-data` sources are unseeded: reproducibility comes from fixed catalogs, not seeds.
For sampling sources the recommended report is therefore a **distribution over expressions
with a bootstrap confidence interval**, rather than a single point. A run draws
`problems_per_expression` problems per ground-truth expression; group those draws by
`benchmark_eq_id`, collapse each group to one value, then bootstrap a statistic over the
per-expression values:

A run returns only **raw** columns, so `numeric_recovery_val` (a derived metric) is not in
the snapshot yet: derive it first (see [Deriving metrics](#deriving-metrics)), then report on
the derived column.

```python
from srbf import Benchmark, compute_derived_metrics, bootstrap_report, draw_distribution

# A run returns the raw dict-of-lists snapshot directly:
(benchmark,) = Benchmark.runs_from_config(single_run_config_path)
snapshot = benchmark.run()

# Derive metrics in place (compute_derived_metrics wants the nested shape):
results = {"model": {"results": {"test": {0: snapshot}}}}
compute_derived_metrics(
    results, test_sets=["test"],
    operator_arity={"add": 2, "sub": 2, "mul": 2, "div": 2, "pow": 2},
)
derived = results["model"]["results"]["test"][0]  # now carries numeric_recovery_val, fvu_*, ...

# One value per expression (mean over that expression's draws):
per_expr = draw_distribution(derived, "numeric_recovery_val")

# Bootstrap the mean recovery across expressions, with a 95% CI:
report = bootstrap_report(derived, "numeric_recovery_val")
print(report)  # {'metric', 'n_groups', 'n_rows', 'median', 'ci_lower', 'ci_upper', 'interval'}
```

Both functions operate on a plain dict-of-lists, drop placeholder rows, and group by
`benchmark_eq_id` (override with `group_key=`). They work on **any** column present, so you
can also point them at a raw column directly (e.g. `bootstrap_report(snapshot, "fit_time")`)
without deriving metrics first. The bootstrap is unseeded, so report the interval rather than
a bit-exact point.

## Programmatic API

`Benchmark.runs_from_config(config)` expands a config (the `experiments:` map and/or inline
`!sweep`) into a list of `Benchmark`s, one per resolved run, and applies the resume/limit
math per run. Run each:

```python
from srbf import Benchmark

for benchmark in Benchmark.runs_from_config(
    "configs/evaluation/scaling/v23.0-20M_fastsrb.yaml",
    sweep_filter={"ladder": 32},   # optional: keep only matching sweep rungs
):
    benchmark.run()                # resume-aware; a no-op when that run is already complete
```

`runs_from_config` also accepts `limit_override`, `output_override`, `save_every_override`,
`resume` (set `resume=False` for `--no-resume`), and `experiment`, mirroring the CLI flags.

For a single, **fully-resolved** run config (no `!sweep`, no `experiments:` map), use
`Benchmark.from_config(config).run()` directly. `from_config` does not expand `!sweep`, so a
config that still contains `!sweep` markers must go through `runs_from_config`.

## Adapters and out-of-band baselines

The built-in adapters (`flash_ansr`, `pysr`, `nesymres`, `e2e`, `lample_charton`,
`brute_force`) are **reference examples** of the adapter contract. `pip install srbf` ships
the benchmark driver, metrics, and the `flash_ansr` adapter usable out of the box. The
`pysr` adapter code ships too, but the `pysr` package itself (plus a Julia precompile) comes
with `pip install "srbf[baselines]"` (which adds sympy, pysr, omegaconf), so a bare install
does not give a runnable PySR baseline.

The unpackaged research baselines (`nesymres`, `e2e` / `symbolicregression`) are not pip
dependencies. Their upstream source trees and weights are provisioned out-of-band via a
clone-and-patch flow using the scripts under
[`scripts/`](https://github.com/psaegert/srbf/tree/main/scripts): `patch_nesymres.py` and
`patch_symbolicregression.py` (each takes the path to the corresponding clone) plus
`patch_typing_io.py` (patches the active environment, no args); weights download separately.
This is **one default recipe**, not the only way to wire those models in. See
[models.md](models.md) for the exact per-baseline steps.

`srbf` is a community framework: a new SR method is added by PR with an adapter plus its own
install instructions. See [adapters.md](adapters.md) for the contribution guide.
