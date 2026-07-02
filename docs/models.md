# Models and provisioning

How to provision the symbolic-regression models that `srbf` evaluates, and how each is wired into a run config.

`srbf` does not bundle model weights. Each model is reached through a **model adapter**: a small wrapper, selected by `model_adapter.type` in the config, that loads a model and turns its predictions into the metrics the benchmark records. The adapters described here are the **built-in reference examples**. They live in a registry (`srbf.config._ADAPTER_REGISTRY`) with six entries:

| `type`           | Provisioning                         | Section |
| ---------------- | ------------------------------------ | ------- |
| `flash_ansr`     | pip (`flash_ansr install <repo>`)    | [FlashANSR](#flashansr-pip) |
| `pysr`           | pip (`pip install pysr` + Julia)     | [PySR](#pysr-pip) |
| `nesymres`       | clone + patch (research baseline)    | [NeSymReS](#nesymres-clone-patch) |
| `e2e`            | clone + patch (research baseline)    | [E2E / symbolicregression](#e2e-symbolicregression-clone-patch) |
| `lample_charton` | none (synthetic baseline)            | [No-provisioning baselines](#no-provisioning-baselines) |
| `brute_force`    | none (synthetic baseline)            | [No-provisioning baselines](#no-provisioning-baselines) |

`srbf` is a community framework: new SR methods are contributed by PR with their own adapter plus install instructions. The built-ins below show the pattern; to add your own model, see [docs/adapters.md](./adapters.md).

See also: [docs/running.md](./running.md) (running a config), [docs/benchmarks.md](./benchmarks.md) (the data the models are scored on), and the [project README](https://github.com/psaegert/srbf/blob/main/README.md).

## Asset root and `{{ROOT}}`

Every shipped config references assets with a `{{ROOT}}` placeholder, for example `{{ROOT}}/models/nesymres/100M.ckpt`. At load time `{{ROOT}}` is substituted with the **`FLASH_ANSR_ROOT`** environment variable (resolved through `flash-ansr`, the one-way dependency). Point it at a checkout that holds your `configs/`, `data/`, and `models/`:

```bash
export FLASH_ANSR_ROOT=/path/to/srbf
```

If `FLASH_ANSR_ROOT` is unset, `flash-ansr` falls back to its own source-checkout root, which is rarely what you want for an `srbf` eval. Set it explicitly.

## Per-model environment isolation

The four provisioned models have **mutually incompatible runtime dependencies**, and two of them are provisioned by patches that **edit installed packages in place**. Give each its own virtual environment:

- **NeSymReS** pins `hydra-core>=1.3.2,<1.4`, `omegaconf>=2.3.0,<2.4`, and `pytorch-lightning==2.5.6`, and `scripts/patch_typing_io.py` rewrites the installed Hydra / OmegaConf / ANTLR site-packages destructively.
- **E2E** drops `functorch` and rewrites its own `requirements.txt` / `pyproject.toml` from `environment.yml`.
- **PySR** pulls in JuliaCall and triggers a Julia precompile.

Installing these side by side leads to version clashes and a `torch was imported before juliacall` warning at best, and broken imports at worst. One environment per baseline keeps each reproducible. `srbf` itself plus `flash-ansr`, `simplipy`, and `symbolic-data` install cleanly into each.

> **SimpliPy engine.** Every adapter except `flash_ansr` requires an explicit `model_adapter.simplipy_engine` (a SimpliPy engine config name or path, e.g. `dev_7-3`): there is no loaded dataset to borrow an engine from, so the adapter raises if it is omitted. `flash_ansr` is the exception, it loads its engine from the model.

## The `model_adapter` config block

In a run config, the model is described by a `model_adapter` block (anchored under `defaults:` and reused per experiment). Its fields are per-type. The blocks below are adapted from the shipped configs under `configs/evaluation/scaling/` (the compute-scaling knob is shown as a single value instead of the `!sweep` tag the shipped configs use); treat them as reference examples and copy the one matching your model.

---

## FlashANSR (pip)

The main `flash-ansr` checkpoints are installed from the Hugging Face Hub with the `flash-ansr` CLI (installed as a dependency of `srbf`):

```bash
flash_ansr install psaegert/flash-ansr-v23.0-20M
```

This downloads the checkpoint to `<FLASH_ANSR_ROOT>/models/<owner>/<name>`, that is `models/psaegert/flash-ansr-v23.0-20M`.

> The shipped configs reference a curated path, for example `model_path: "{{ROOT}}/models/ansr-models/v23.0-20M"`. That is **not** the directory `flash_ansr install` writes to. After installing, point `model_adapter.model_path` at wherever the checkpoint actually landed (or symlink / move it under `models/ansr-models/`). There is no auto-wiring between the install location and the config.

`model_adapter` block (`configs/evaluation/scaling/v23.0-20M_fastsrb.yaml`):

```yaml
model_adapter:
  type: flash_ansr
  model_path: "{{ROOT}}/models/ansr-models/v23.0-20M"
  evaluation_config:
    n_support: 512
    n_restarts: 8
    refiner_method: curve_fit_lm
    refiner_p0_noise: normal
    refiner_p0_noise_kwargs: {loc: 0.0, scale: 5}
    length_penalty: 0.05
    constants_penalty: 0.0
    likelihood_penalty: 0.0
    generation_config:
      method: softmax_sampling
      kwargs: {choices: 32, top_k: 0, top_p: 1, max_len: 64, batch_size: 128, temperature: 1, valid_only: true, simplify: true, unique: true}
    device: cuda
  device: cuda
```

Key fields: `model_path` (checkpoint directory), `evaluation_config` (refiner + generation settings, including the nested `generation_config`), and `device`. The full block is ~30 lines; read the shipped config rather than retyping it. The compute-scaling knob (the candidate count `choices`) is swept with `generation_overrides` (and `evaluation_overrides` for refiner knobs). The `*_val.yaml` variants point the same adapter at the `v23-val` catalog.

`flash_ansr` is the only adapter where the wheel both ships the adapter and can fetch the model, so no extra `srbf` extra is needed.

## PySR (pip)

The PySR **adapter** ships in the base `srbf` wheel, but the `pysr` **package** is not a hard dependency: it is in the `baselines` extra and needs a Julia precompile at runtime. `pip install srbf` alone does **not** give you a working PySR baseline. Install PySR and precompile Julia inside the PySR environment:

```bash
pip install pysr            # or: pip install "srbf[baselines]"
python -c "from pysr import PySRRegressor"   # triggers Julia precompilation
```

The first import compiles the Julia backend (this can take several minutes). If Julia is not available system-wide, follow the [PySR install docs](https://astroautomata.com/PySR/install/) first.

`model_adapter` block (`configs/evaluation/scaling/pysr_fastsrb.yaml`):

```yaml
model_adapter:
  type: pysr
  timeout_in_seconds: 3600
  niterations: 1
  padding: false
  use_mult_div_operators: false
  simplipy_engine: "dev_7-3"
```

Key fields: `niterations` (the compute-scaling axis, swept per run), `timeout_in_seconds`, and `simplipy_engine` (the SimpliPy engine name; `dev_7-3` is installed on demand; required). No model weights to download: PySR fits each problem from scratch.

Two properties of the adapter worth knowing:

- **Complexity budget = PySR's own default.** Benchmark policy: baselines run at their upstream
  defaults — a method's default hyperparameters are part of the method. Note what that default
  implies here: at PySR's `maxsize=20`, 23/120 FastSRB and 743/1000 v23-val ground truths are not
  representable under the adapter vocabulary at all (largest ground truth = 40 nodes; measure it
  yourself with `python scripts/audit_pysr_maxsize.py`). This is a documented property of running
  PySR at its defaults on these benchmarks, not something srbf corrects. An optional `maxsize` key
  exists in the `model_adapter` block for side experiments only; headline results use the default.
- **Warmup fit in `prepare()`** (`warmup: true` by default, srbf 0.6.1) — the first `fit` in a
  Julia session pays a one-off precompile cost that is an order-of-magnitude timing outlier; the
  adapter burns it on a throwaway model before evaluation so problem 0's `fit_time` starts warm.
  This mirrors the other adapters, which pay their one-time model-load cost in `prepare()` too.

---

The next two are **research baselines**: upstream source trees that are not pip-installable as-is on modern Python. The default recipe is a clone of the upstream repo, a patch with the script under [scripts/](https://github.com/psaegert/srbf/tree/main/scripts), an editable install, and a separate weights download. **These patch scripts are one default recipe, not the only way.** If you maintain a fork that already builds, install that instead.

## NeSymReS (clone + patch)

Default recipe, in its own Python 3.13 environment:

```bash
# 1. srbf + its deps
pip install -e ".[baselines]"

# 2. Clone NeSymReS and install it editable
git clone https://github.com/SymposiumOrganization/NeuralSymbolicRegressionThatScales nesymres/NeuralSymbolicRegressionThatScales
pip install -e nesymres/NeuralSymbolicRegressionThatScales/src

# 3. Lightning, for the checkpoint loader
pip install pytorch-lightning==2.5.6

# 4. Patch for Python 3.13 (rerun after any reinstall of hydra/omegaconf/antlr)
python scripts/patch_typing_io.py                                   # patches installed hydra/omegaconf/antlr in place
python scripts/patch_nesymres.py nesymres/NeuralSymbolicRegressionThatScales   # dclasses + setup.py pins
```

The NeSymReS recipe uses **both** patch scripts:

- `scripts/patch_typing_io.py` rewrites the **installed** Hydra / OmegaConf / ANTLR packages (the removed `typing.io` import, plus Hydra dataclass `default_factory` fixes). It takes no arguments and operates on the active environment. Rerun it after every reinstall of those packages.
- `scripts/patch_nesymres.py <repo>` patches the **cloned** repo: it sets `bfgs: BFGSParams = field(default_factory=BFGSParams)` in `dclasses.py` and pins `hydra-core` / `omegaconf` in `setup.py`. Pass the path to your NeSymReS clone.

Then download the 100M checkpoint triplet (`eq_setting.json`, `config.yaml`, `100M.ckpt`) into `models/nesymres/` separately, as documented upstream.

`model_adapter` block (`configs/evaluation/scaling/nesymres_fastsrb.yaml`):

```yaml
model_adapter:
  type: nesymres
  eq_setting_path: "{{ROOT}}/models/nesymres/eq_setting.json"
  config_path: "{{ROOT}}/models/nesymres/config.yaml"
  weights_path: "{{ROOT}}/models/nesymres/100M.ckpt"
  simplipy_engine: "dev_7-3"
  n_restarts: 4
  device: cuda
  remove_padding: true
```

Required fields: `eq_setting_path`, `config_path`, `weights_path`, and `simplipy_engine` (the adapter raises if any is missing). `beam_width` is the compute-scaling axis, swept per run. The `nesymres_val.yaml` variant points the same adapter at the `v23-val` catalog.

> Lightning warns that the checkpoint was saved with an older release. Inference works regardless; optionally upgrade it in place with `python -m pytorch_lightning.utilities.upgrade_checkpoint models/nesymres/100M.ckpt`.

## E2E / symbolicregression (clone + patch)

The Meta "End-to-End" model (`facebookresearch/symbolicregression`). Default recipe, in its own environment:

```bash
# 1. srbf + its deps
pip install -e ".[baselines]"

# 2. Clone the upstream repo
git clone https://github.com/facebookresearch/symbolicregression e2e/symbolicregression

# 3. Patch for modern numpy/torch (drops functorch, fixes a rescale loop,
#    rewrites requirements.txt + pyproject.toml from environment.yml)
python scripts/patch_symbolicregression.py e2e/symbolicregression

# 4. Editable install of the patched tree
pip install -e e2e/symbolicregression
```

The E2E recipe uses **one** patch script, `scripts/patch_symbolicregression.py <repo>`, which takes the path to the cloned `symbolicregression` repo (the directory containing the `symbolicregression/` package). It is independent of the NeSymReS scripts; `patch_typing_io.py` is not part of this recipe.

Then download the E2E weights (`model1.pt`) into `models/e2e/` separately, per upstream.

`model_adapter` block (`configs/evaluation/scaling/e2e_fastsrb.yaml`):

```yaml
model_adapter:
  type: e2e
  model_path: "{{ROOT}}/models/e2e/model1.pt"
  simplipy_engine: "dev_7-3"
  device: cuda
  candidates_per_bag: 1
  max_input_points: 200
  max_number_bags: 100
  n_trees_to_refine: 10
  rescale: true
```

Required fields: `model_path` and `simplipy_engine` (the adapter raises if either is missing). `candidates_per_bag` is the compute-scaling axis, swept per run (some configs also tune `max_generated_output_len`). The `e2e_val.yaml` variant targets the `v23-val` catalog.

---

## No-provisioning baselines

`lample_charton` and `brute_force` are **synthetic** baselines: they fit candidate expressions drawn from (or enumerated over) a generative `symbolic-data` catalog rather than calling a trained network, so there are no weights to download. They need a generative `catalog` to draw candidates from and an explicit `simplipy_engine`.

> **Migration note.** The former `skeleton_pool` adapter is now `lample_charton`; its `skeleton_pool:` field is now `catalog:`, pointing at a generative catalog (e.g. `lample-charton-v23`).

`lample_charton` block (`configs/evaluation/scaling/lample_charton_fastsrb.yaml`):

```yaml
model_adapter:
  type: lample_charton
  simplipy_engine: "dev_7-3"
  catalog: lample-charton-v23   # the generative catalog to sample candidate expressions from
  samples: 32                   # candidate expressions sampled per problem (the compute-scaling axis)
  unique: true
  ignore_holdouts: true
  seed: 42
  n_restarts: 8
  refiner_method: curve_fit_lm
  refiner_p0_noise: normal
  refiner_p0_noise_kwargs: {loc: 0.0, scale: 5}
  numpy_errors: ignore
  length_penalty: 0.05
  constants_penalty: 0.0
  likelihood_penalty: 0.0
```

Note the **two** distinct `catalog` fields when this adapter is used: `data_source.catalog` is the evaluation set (e.g. `fastsrb` / `v23-val`), while `model_adapter.catalog` is the generative catalog the baseline samples its candidate expressions from (e.g. `lample-charton-v23`).

`brute_force` takes the same required `catalog` and `simplipy_engine`, plus enumeration limits: `max_expressions` (default 10000), `max_length`, `include_constant_token` (default true), and the same refiner / penalty fields. No shipped scaling config defines a `brute_force` run; build one by copying the `lample_charton` block, changing `type: brute_force`, and adding `max_expressions`.

---

## Where outputs land

Every run writes a pickle under `results/evaluation/.../*.pkl`, **one row per evaluated problem**, with the raw prediction columns (`y_pred`, `y_pred_val`, `predicted_*`, `fit_time`, ...); the derived metrics (`fvu_fit`, `fvu_val`, `numeric_recovery_*`, `symbolic_recovery`, `f1_score`, ...) are produced by a separate `srbf.compute_derived_metrics` step, not by the run itself. Rows where a problem could not be produced are written as `placeholder` rows so counts stay aligned across models; filter them before any fit-based analysis. `runner.resume: true` (or omitting `--no-resume`) continues a partial pickle instead of restarting. See [docs/running.md](./running.md) for the full runner, CLI, output columns, and metric-derivation / reporting behavior, and [docs/adapters.md](./adapters.md) to add a new model.
