# Benchmarks and datasets

This guide covers the data `srbf` evaluates on: how the `data_source` block in a config picks what
to evaluate, the catalogs that ship with the framework, how catalog references are resolved, and how
to point at your own benchmark. For the CLI that *runs* an evaluation see
[docs/running.md](running.md); for the model side see [docs/models.md](models.md) and
[docs/adapters.md](adapters.md); for the project overview see
[../README.md](https://github.com/psaegert/srbf/blob/main/README.md).

`srbf` resolves `{{ROOT}}` in every config path against the `FLASH_ANSR_ROOT` environment variable.
Point it at a checkout that holds `configs/`, `models/`, and `results/`:

```bash
export FLASH_ANSR_ROOT=$(pwd)
```

## The `data_source` block

Every evaluation config carries a `data_source` block. The data source is always a `symbolic-data`
catalog: `catalog` names the set of ground-truth expressions to evaluate on, and `sampling` is this
run's usage policy over that catalog. The catalog itself owns all generation, fixed-set iteration,
noise injection, and decontamination; `srbf` just streams the resulting problems into the benchmark
driver.

```yaml
data_source:
  catalog: v23-val              # a catalog name/ref, an HF 'user/repo:name' ref, a local path, or an inline config
  sampling:                     # the symbolic-data usage policy (all fields optional)
    n_support: 512              # points the model fits on
    n_validation: 1024          # held-out points (omit for catalogs that carry their own validation)
    noise: 0.0                  # Gaussian noise as a fraction of the target std (0 = clean)
    problems_per_expression: 10 # distinct sampled problems per ground-truth expression
    method: iterate             # frozen catalog -> 'iterate'; open generative catalog -> 'procedural'
  holdouts:                     # optional decontamination / filters (see below)
    - exclude: lample-charton-v23
    - filter: {finite: true}
  target_size: 1000             # cap the number of rows (also the run total when runner.limit is null)
```

### `sampling` fields

| field | meaning |
|---|---|
| `n_support` | number of points the model fits on. `prior` (generative catalogs only) draws the support size per problem from the catalog's own prior; it **requires** `n_validation: 0` and a generative catalog, and raises a `ValueError` otherwise. |
| `n_validation` | number of held-out validation points; the first `n_support` of each sampled problem are the fit split, the rest the validation split. |
| `noise` | additive Gaussian noise as a fraction of the target std (`0.0` = clean). |
| `problems_per_expression` | how many distinct problems to draw per ground-truth expression; multiplies the row count. |
| `method` | draw mode: `iterate` over a fixed catalog, `procedural` for an open generative one. Defaults follow the catalog kind. |
| `layout` | X-point layout passed to the catalog's distribution (default `random`). |
| `max_trials` | resample attempts before a placeholder row is written. |
| `size` | number of expressions to draw from an open generative catalog (a generative usage policy). |

### `holdouts`

`holdouts` is an optional list of rules applied to every problem the catalog yields:

- `{exclude: <catalog-ref>}` — **decontamination**: drop any problem whose normalized skeleton appears in the referenced catalog (e.g. exclude the training recipe from a generated test set). The reference resolves the same way as `catalog` (a name, HF ref, local path, or inline config).
- `{filter: {...}}` — keep only problems matching a filter predicate (e.g. `{finite: true}`).

## The shipped catalogs

`srbf` configs reference these `symbolic-data` catalogs by name:

| name | what it is | kind |
|---|---|---|
| `v23-val` | the frozen, sha-pinned v23 validation set | fixed set (deterministic; iterate) |
| `fastsrb` | the FastSRB benchmark equations | fixed set (samples `(X, y)` per equation) |
| `lample-charton-v23` | the generative v23 training recipe | open generative (streams skeletons) |

`v23-val` is the drift-safe evaluation set: it is deterministic and sha256-pinned on Hugging Face, so
"val" means the same problems across models and machines without any per-run seeding or pinned-list
bookkeeping. Both a FastSRB config (`*_fastsrb.yaml`, `catalog: fastsrb`) and a validation config
(`*_val.yaml`, `catalog: v23-val`) ship for the model and baseline configs under
`configs/evaluation/{scaling,noise_sweep,support_sweep}/`. Evaluating on both is the standard dual
protocol.

FastSRB is [Martinek 2025](https://arxiv.org/abs/2508.14481) (MIT-licensed; attribution in
`THIRD_PARTY_LICENSES`).

## How a catalog reference resolves

The `catalog` field (and any `holdouts.exclude` reference) accepts four forms:

1. **A name**, e.g. `v23-val` or `fastsrb`, optionally version-pinned as `name@version`. It is looked up in the `symbolic-data` asset manifest on Hugging Face (`psaegert/symbolic-data-assets` by default), fetched with `hf_hub_download`, integrity-checked against the manifest's `sha256`, and cached. A fresh install needs network on first use; subsequent runs hit the cache.
2. **A third-party HF ref**, `user/repo:name` or `user/repo:name@version`, against another repo's manifest, so anyone can publish and load their own catalogs.
3. **A local path**, e.g. `{{ROOT}}/configs/my_catalog.yaml` — used as-is, no download, for fully offline operation.
4. **An inline config** — a mapping written directly in the YAML (e.g. a generative `{type: lample_charton, ...}` spec), so the catalog is defined in place rather than referenced.

There is no local "build a dataset" step: a named catalog is fetched and cached on demand. Use a
local path or inline config when you want full control or offline operation.

## Sweeps over the same data

The same expressions can be re-evaluated under varied conditions by sweeping `data_source` /
`sampling` fields with inline `!sweep` (see [docs/running.md](running.md) for the full `!sweep`
semantics):

- **noise** (`configs/evaluation/noise_sweep/*.yaml`): sweep `sampling.noise`, e.g. `0.0`, `0.001`, `0.01`, `0.1`.
- **support size** (`configs/evaluation/support_sweep/*.yaml`): sweep `sampling.n_support` / `sampling.n_validation`.

```yaml
data_source:
  catalog: fastsrb
  sampling:
    n_support: 512
    n_validation: 512
    noise: !sweep {name: noise, values: [0.0, 0.001, 0.01, 0.1]}
    problems_per_expression: 10
```

## Pointing at a custom benchmark set

You have three routes, depending on the data you have.

1. **Publish a catalog (recommended for a shareable benchmark).** Build a `symbolic-data` catalog and publish it to a Hugging Face dataset repo with a manifest, then reference it as `your-user/your-repo:name@version` in `data_source.catalog`. Anyone can then resolve it by ref.
2. **A local catalog config.** Write a `symbolic-data` catalog config (expressions plus their per-variable sampling spec) and point `data_source.catalog` at the local file path. The catalog samples `(X, y)` itself, so the support/validation split and noise come from `sampling`.
3. **An inline catalog.** For a one-off, write the catalog spec directly under `data_source.catalog` as a mapping.

Copy a shipped config (e.g. `configs/evaluation/scaling/v23.0-3M_fastsrb.yaml` or `..._val.yaml`),
swap the `catalog` reference, and run it as in [docs/running.md](running.md). The complete catalog
authoring reference lives in the [`symbolic-data`](https://github.com/psaegert/symbolic-data) docs.

## Outputs

Each run writes a pickle under `results/evaluation/.../*.pkl` (the `runner.output` path), with one
row per evaluated problem and the raw prediction columns (`y_pred`, `y_pred_val`, `predicted_*`,
`fit_time`, ...). The derived metrics (`fvu_fit` / `fvu_val`, `log10_fvu_*`, `numeric_recovery_*`,
`symbolic_recovery`, `f1_score`, and more) are computed by a separate `srbf.compute_derived_metrics`
step, not by the run itself. When a problem cannot be produced within `max_trials`, a `placeholder`
row is written instead to keep row counts aligned across runs; filter on the `placeholder` column
before any fit-based analysis. `runner.resume` continues a partial pickle. See
[docs/running.md](running.md) for the output columns, metric derivation, resume, and reporting
details.
