# Benchmarks and datasets

This guide covers the data `srbf` evaluates on: how the `data_source` block in a config picks
what to evaluate, how to fetch the FastSRB benchmark, how to build a skeleton pool with
flash-ansr's `import-data` CLI, and how to point at your own benchmark set. For the CLI that
*runs* an evaluation see [docs/running.md](running.md); for the model side see
[docs/models.md](models.md) and [docs/adapters.md](adapters.md); for the project overview see
[../README.md](../README.md).

`srbf` resolves `{{ROOT}}` in every config path against the `FLASH_ANSR_ROOT` environment
variable. Point it at a checkout that holds `configs/`, `data/`, and `models/`:

```bash
export FLASH_ANSR_ROOT=$(pwd)
```

## The `data_source` block

Every evaluation config carries a `data_source` block (under `defaults:` and per `experiments:`
entry, wired with YAML anchors). Its `type` selects one of two sources. These are the only two
branches in `srbf.eval.run_config._build_data_source`; anything else raises
`Unsupported data source type`.

| `type` | What it evaluates on | How the data is produced |
|---|---|---|
| `fastsrb` | the FastSRB benchmark equations | reads an `expressions.yaml` directly and samples `(X, y)` on the fly |
| `skeleton_dataset` | a flash-ansr skeleton pool (e.g. a model's held-out `val` set) | streams skeletons from a flash-ansr dataset config, sampling data per skeleton |

> **Note on terminology.** `skeleton_pool` is *not* a `data_source` type. It is a *model adapter*
> type (and a config field inside that adapter and the `brute_force` adapter), the brute-force-ish
> baseline that fits every skeleton in a pool. If you came here looking for `skeleton_pool`, see
> [docs/models.md](models.md) and [docs/adapters.md](adapters.md). The two data sources are
> `fastsrb` and `skeleton_dataset`.

### `fastsrb` fields

From `configs/evaluation/scaling/v23.0-20M_fastsrb.yaml`:

```yaml
data_source:
  type: fastsrb
  benchmark_path: "{{ROOT}}/data/ansr-data/test_set/fastsrb/expressions.yaml"
  datasets_per_expression: 10     # repeats per equation (distinct sampled datasets)
  support_points: 512             # points the model fits on
  sample_points: 1024             # total points drawn; the rest become the validation split
  method: random                  # 'random' or 'range' sampling of the input grid
  max_trials: 100                 # resample attempts before a placeholder is written
  incremental: false
  random_state: 42                # seeds sampling, so runs are reproducible
  noise_level: 0.0                # Gaussian noise as a fraction of the target std (0 = clean)
```

The `fastsrb` source loads `benchmark_path` (an `expressions.yaml`) directly and samples data for
each equation itself: there is **no skeleton-pool build step on the path to the `fastsrb` source**.
The first `support_points` of each sampled dataset are the fit split; the remainder (up to
`sample_points`) are the validation split. `datasets_per_expression` multiplies the row count: the
~unique equations times the repeats. Optional fields: `eq_ids` (a string or list to evaluate a
subset of equation ids), `n_support_override`, and `benchmark_random_state`.

### `skeleton_dataset` fields

From `configs/evaluation/scaling/v23.0-20M_val.yaml`:

```yaml
data_source:
  type: skeleton_dataset
  dataset: "{{ROOT}}/configs/v23.0-20M/dataset_val.yaml"   # a flash-ansr dataset config
  datasets_per_expression: 10
  n_support: 512
  noise_level: 0.0
  max_trials: 100
  target_size: 1000             # 100 skeletons x 10 repeats; caps the number of rows
```

The `dataset` path is a flash-ansr dataset config (it references a `skeleton_pool:` plus a
tokenizer and preprocessor). The source iterates the pool's skeletons in a deterministic, sorted
order, samples `datasets_per_expression` datasets per skeleton, and stops at `target_size`. Use
this source to evaluate on a model's own held-out validation skeletons rather than the external
FastSRB equations. Optional: `skeleton_list` (a path to a pinned JSON skeleton set, or an inline
list) freezes *which* skeletons evaluate, so "val" means the same set across models and machines;
the source hard-fails if any pinned skeleton is missing from the pool.

Both `*_fastsrb.yaml` and `*_val.yaml` variants ship for the model and baseline configs under
`configs/evaluation/{scaling,noise_sweep,support_sweep}/`. Evaluating on both is the standard dual
protocol.

## Getting the FastSRB benchmark

FastSRB ([Martinek 2025](https://arxiv.org/abs/2508.14481), MIT-licensed; attribution in
`THIRD_PARTY_LICENSES`) ships as a single raw `expressions.yaml`. Fetch it once into the path the
configs expect:

```bash
mkdir -p "$FLASH_ANSR_ROOT/data/ansr-data/test_set/fastsrb"
wget -O "$FLASH_ANSR_ROOT/data/ansr-data/test_set/fastsrb/expressions.yaml" \
  "https://raw.githubusercontent.com/viktmar/FastSRB/refs/heads/main/src/expressions.yaml"
```

Each entry in the file carries a `prepared` expression string and a `vars` block of per-variable
sampling specs. The `fastsrb` data_source reads this file as is; no further preparation is needed
to run a `*_fastsrb.yaml` config. That is all the FastSRB source requires.

## Building a skeleton pool with `flash_ansr import-data`

A *skeleton pool* is a flash-ansr artifact: a set of structural expression skeletons with compiled
sampling code. It is **not** consumed by the `fastsrb` data_source. You need one for:

- the `skeleton_dataset` data_source, whose `dataset` config references a pool (a model's `val`
  pool is usually produced during training; you can also build a pool from a benchmark), and
- the `skeleton_pool` and `brute_force` model adapters, whose `skeleton_pool:` field points at a
  pool (e.g. `{{ROOT}}/models/prior/skeleton_pool.yaml`); see [docs/models.md](models.md).

flash-ansr is a `srbf` dependency, so its CLI is already installed. Build a pool from a raw
benchmark file with `import-data`:

```bash
flash_ansr import-data \
  -i "$FLASH_ANSR_ROOT/data/ansr-data/test_set/fastsrb/expressions.yaml" \
  -p fastsrb \
  -e dev_7-3 \
  -b configs/test_set/skeleton_pool.yaml \
  -o "$FLASH_ANSR_ROOT/data/ansr-data/test_set/fastsrb/skeleton_pool" \
  -v
```

Flags (from flash-ansr's `import-data` parser):

| Flag | Meaning |
|---|---|
| `-i, --input` | path to the raw benchmark file (`.yaml`/`.yml` or `.csv`) |
| `-p, --parser` | parser name; `fastsrb`, `soose`, `feynman`, or `nguyen` |
| `-e, --simplipy-engine` | SimpliPy expression-space config (e.g. `dev_7-3`, the engine the configs use) |
| `-b, --base-skeleton-pool` | a base skeleton-pool config to extend (defines the variable budget); flash-ansr ships `configs/test_set/skeleton_pool.yaml` |
| `-o, --output-dir` | output directory for the built pool |
| `-v, --verbose` | print a progress bar and summary counts |

This writes `skeleton_pool.yaml` and `skeletons.pkl` under the output directory. Point a
`skeleton_pool` / `brute_force` adapter's `skeleton_pool:` field at the written `skeleton_pool.yaml`,
or reference it from a flash-ansr dataset config consumed by a `skeleton_dataset` source.

## Sweeps over the same data

The same equations can be re-evaluated under varied conditions by overriding `data_source` fields
per experiment with YAML anchors:

- **noise** (`configs/evaluation/noise_sweep/*.yaml`): override `noise_level` per experiment, e.g.
  `0.0`, `0.001`, `0.01`, `0.1`.
- **support size** (`configs/evaluation/support_sweep/*.yaml`): vary `support_points` /
  `sample_points`.

Example noise override (from `noise_sweep/v23.0-20M_fastsrb.yaml`):

```yaml
experiments:
  flash_ansr_fastsrb_noise_0p010:
    run:
      data_source:
        <<: *flash_ansr_fastsrb_source
        noise_level: 0.01
      # ... model_adapter, runner ...
```

## Pointing at a custom benchmark set

You have two routes, depending on the data you have.

1. **A benchmark expression list (recommended for new benchmarks).** Write an `expressions.yaml`
   in the FastSRB layout (a mapping from equation id to an entry with a `prepared` expression and a
   `vars` sampling-spec block), then set `benchmark_path` to it in a `fastsrb` data_source. The
   source samples `(X, y)` itself, so reproducibility and the support/validation split come for
   free via `random_state`, `support_points`, and `sample_points`. To restrict to a subset, set
   `eq_ids`. If your raw file is in a CSV or another benchmark's format, run `import-data` with the
   matching `-p` parser (`soose`, `feynman`, `nguyen`) to build a skeleton pool, then evaluate it
   through a `skeleton_dataset` source (or a `skeleton_pool` adapter).

2. **A skeleton pool of your own.** Build one with `import-data` (above) or flash-ansr's
   `generate-skeleton-pool`, reference it from a flash-ansr dataset config, and point a
   `skeleton_dataset` data_source's `dataset` at that config.

Copy a shipped config (e.g. `configs/evaluation/scaling/v23.0-20M_fastsrb.yaml` or
`..._val.yaml`), swap the `data_source` paths, and run it as in [docs/running.md](running.md).

## Outputs

Each run writes a pickle under `results/evaluation/.../*.pkl` (the `runner.output` path), with one
row per evaluated dataset and flat metric columns: `fvu_fit` / `fvu_val` (and
`log10_fvu_fit` / `log10_fvu_val`), `numeric_recovery_fit` / `numeric_recovery_val`,
`symbolic_recovery`, `f1_score`, and more. When sample generation fails within `max_trials`, a
`placeholder` row is written instead to keep row counts aligned across runs; filter on the
`placeholder` column before any fit-based analysis. `runner.resume` continues a partial pickle. See
[docs/running.md](running.md) for the runner and resume details.
