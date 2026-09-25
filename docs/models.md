# Models

The methods that come with srbf, how to install each one, and the keys of its `model_adapter`
block. To enter a method of your own, see [Adding your method](adapters.md).

| `type` | method | installation |
|---|---|---|
| `flash_ansr` | [Flash-ANSR](#flash-ansr) and its prior reference | `flash_ansr install <checkpoint>` |
| `pysr` | [PySR](#pysr) | `pip install pysr`, in an environment of its own if you like |
| `subprocess`, `worker: operon` | [Operon](#operon) | `pip install pyoperon==0.6.1 scikit-learn`, in an environment of its own |
| `subprocess`, `worker: tisr` | [TiSR](#tisr) | `scripts/envs/build_tisr_env.sh envs/tisr`: Julia and TiSR in an environment of their own |
| `nesymres` | [NeSymReS](#nesymres) | clone, patch, download weights |
| `e2e` | [E2E](#e2e) | clone, patch, download weights |
| `lample_charton`, `brute_force` | [prior sampling and enumeration](#sampling-and-enumeration-baselines) | none |
| `subprocess` | [any method, in its own environment](adapters.md) | yours |

Two keys are common to every block. `config_provenance` states who chose the configuration
([Fairness](fairness.md#configuration-provenance-labels)). `simplipy_engine` names the engine the
predictions are judged with; the srbf suite uses `acj-5-4-llm`, which is downloaded on first use.
Every type needs it except `flash_ansr`, which takes the engine from its checkpoint.

**One environment per method.** Each method has a dependency set of its own: PySR brings Julia,
and NeSymReS and E2E were released against earlier versions of their libraries. Installed side by
side they conflict, so give each its own virtual environment; srbf installs into all of them.

## Flash-ANSR

```bash
flash_ansr install psaegert/flash-ansr-v25.0-T8-3M     # also: -T8-20M, -T8-120M
```

The command comes with srbf and puts the checkpoint under
`$FLASH_ANSR_ROOT/models/psaegert/flash-ansr-v25.0-T8-3M`. The configs
`configs/evaluation/scaling/flash-ansr-v25.0-T8-*_srbf.yaml` evaluate the three sizes on the whole
suite along a ladder of 1 to 65,536 draws.

```yaml
model_adapter:
  type: flash_ansr
  config_provenance: author_blessed
  model_path: "{{ROOT}}/models/psaegert/flash-ansr-v25.0-T8-3M"
  device: cuda
  evaluation_config:
    n_restarts: 8
    refiner_method: curve_fit_lm
    refiner_p0_noise: normal
    refiner_p0_noise_kwargs: {loc: 0.0, scale: 5}
    ranking: {mode: mdl}
    generation_config:
      method: softmax_sampling
      kwargs: {draws: 1024, max_len: 160, batch_size: 128, temperature: 1, simplify: true, unique: true}
  generation_overrides:
    kwargs:
      draws: !sweep {name: ladder, values: [1, 32, 1024]}
```

| key | default | meaning |
|---|---|---|
| `model_path` | required | the checkpoint directory |
| `evaluation_config` | required | a mapping, or the path of a YAML file that holds it |
| `evaluation_config.generation_config` | required | `method` (`softmax_sampling`, or `prior_sampling` for the prior reference) and its `kwargs`; `draws` is the budget |
| `evaluation_config.n_restarts`, `refiner_p0_noise` | required | restarts of the constant fit and the distribution of its starting points |
| `evaluation_config.refiner_method` | `curve_fit_lm` | the optimizer of the constant fit |
| `evaluation_config.ranking` | required | how the prediction is chosen among the fitted candidates; see below. A `ranking` block directly under `model_adapter` replaces it |
| `generation_overrides`, `evaluation_overrides` | none | mappings merged over `generation_config` and `evaluation_config`: the place for a `!sweep` |
| `device` | `cpu` | the device of the model |
| `refiner_workers` | flash-ansr's default | processes that fit constants in parallel |
| `emission`, `refine_scope` | `fittable` | which constants the model spells itself and which the refiner fits |
| `constant_ladder` | on | re-spell fitted constants as integers, fractions and multiples of \(\pi\) and \(e\), and let the ranking choose |
| `complexity` | `none` | condition the model on a target complexity: `none`, `ground_truth` or a number |
| `candidate_store_dir` | none | a directory that receives every candidate of every problem, one compressed `.npz` per problem, readable with `srbf.candidate_store.CandidateStoreReader` |

**Ranking.** `mode` is required and takes one of three values.

```yaml
ranking: {mode: mdl}                                    # the two-part code: (N/2) log2 FVU + the expression's description length in bits
ranking: {mode: mdl, mdl_strength: 1e-2}                 # log10 FVU + a fixed weight per bit
ranking: {mode: weighted, weights: {n_nodes: 0.05}}      # log10 FVU + weighted size measures
ranking: {mode: pareto, metrics: [fvu, n_nodes], tie_break: fvu}
```

The resolved ranking is stored in every result file. Unknown keys in the block are an error.

**The prior reference.** With `generation_config.method: prior_sampling` the model is not used: the
candidates are drawn from the training prior that ships beside the checkpoint
(`catalog_train.yaml`) and then fitted and ranked like any other. It measures what the prior alone
is worth (`configs/evaluation/scaling/flash-ansr-v25.0-T8-prior_srbf.yaml`).

## PySR

```bash
pip install pysr
python -c "from pysr import PySRRegressor"     # the first import compiles the Julia backend; this takes minutes
```

`type: pysr` runs PySR through the [worker protocol](adapters.md), so `pysr` and Julia only have to
exist in the interpreter named by `python`, which may be srbf's own.

```yaml
model_adapter:
  type: pysr
  config_provenance: upstream_default
  simplipy_engine: acj-5-4-llm
  python: "{{ROOT}}/envs/pysr/bin/python"      # default: srbf's interpreter
  niterations: 100
  timeout_in_seconds: 3600
```

| key | default | meaning |
|---|---|---|
| `niterations` | `100` | search iterations: the budget |
| `timeout_in_seconds` | `60` | PySR's own limit on one search; a guard, not the budget |
| `maxsize`, `parsimony` | PySR's defaults | passed on only when set; published results leave them unset ([Fairness](fairness.md#baselines-run-at-their-upstream-defaults)) |
| `model_selection` | `best` | which equation of PySR's hall of fame is the prediction |
| `warmup` | `true` | run a throwaway fit when the worker starts, so that Julia's one-time compilation is not part of the first problem's time |
| `python`, `env`, `timeout`, `max_restarts`, `worker_log` | | as for every worker ([the config keys](adapters.md#the-config-keys)) |

PySR searches over the operators the expressions are written in: `+ - * / ^`, `rootn`, and the unary
operators `neg abs inv sin cos tan asin acos atan sinh cosh tanh asinh acosh atanh exp log`. The
worker returns PySR's own predictions and stores the whole hall of fame in the `equations` column.
`configs/evaluation/scaling/pysr_fastsrb.yaml` sweeps the iterations from 1 to 16,384.

## Operon

```bash
python -m venv envs/operon && envs/operon/bin/pip install pyoperon==0.6.1 scikit-learn
```

Operon is a genetic-programming method with Levenberg–Marquardt constant fitting on every individual. It runs
through the [worker protocol](adapters.md) as `worker: operon`, in an environment of its own. The configuration is
the one Operon's first author published for running it as a benchmark baseline, without its hyperparameter search:
- NSGA-II on fit (R²) and length;
- one Levenberg–Marquardt step per individual;
- population 1,000, length at most 50 and depth at most 10;
- the final model picked from the Pareto front by minimum description length, with the noise level estimated by a
  random forest.

```yaml
model_adapter:
  type: subprocess
  worker: operon
  python: "{{ROOT}}/envs/operon/bin/python"
  config_provenance: author_blessed
  simplipy_engine: acj-5-4-llm
  options:
    max_evaluations: 1048576
```

| key | default | meaning |
|---|---|---|
| `options.max_evaluations` | `1000000` | evaluations of the model or its derivatives, the local search's included: the budget |
| `options.seed` | `0` | mixed with a hash of the problem's data into the run's seed |
| `python`, `env`, `timeout`, `max_restarts`, `worker_log` | | as for every worker ([the config keys](adapters.md#the-config-keys)) |

**Operators.** Operon searches over the operators it has among those the expressions are written in:
- `+ - * / ^` and `abs sin cos tan asin acos atan sinh cosh tanh exp log`;
- `rootn` for square and cube roots.

It has no node for `asinh`, `acosh`, `atanh` or other roots. It expresses `neg` and `inv` through signed weights
and division.

**What to know when reading the results:**
- Operon searches in single precision (float32).
- It always returns its model as `a * f(x) + b`, with a weight on every variable. On a law without such constants
  that shape rarely matches the ground truth symbol for symbol, so its numeric recovery is the comparable rate.
- One search runs on one thread: more threads make a run irreproducible.

The worker stores the whole Pareto front in the `front` column. `configs/evaluation/scaling/operon_fastsrb.yaml`
sweeps the evaluations in doublings from 2^10 up to about 100 s per problem on the reference machine.

## TiSR

```bash
scripts/envs/build_tisr_env.sh envs/tisr     # needs uv, curl and git; downloads Julia and compiles TiSR
```

TiSR (thermodynamics-informed symbolic regression; Martinek, Frotscher, Richter and Herzog) is genetic programming
in Julia: NSGA-II on islands, the constants of every new expression fitted by Levenberg–Marquardt, and a hall of
fame, the Pareto front of fit error against complexity. It runs through the [worker protocol](adapters.md) as
`worker: tisr` and calls Julia through juliacall. TiSR is licensed under the Apache License 2.0.

The build script creates the environment: Python 3.12 with juliacall, Julia 1.11.5, and TiSR at commit `5b541b3`,
the commit the FastSRB paper ran, with every Julia package at the version of that paper's lock file
(`scripts/envs/tisr/Manifest.toml`), all in a Julia depot of the environment's own. The worker points juliacall at
this Julia and depot and starts it with one thread.

The configuration is TiSR's own defaults at that commit:
- 20 islands of 50 expressions, each breeding about 50 new expressions per generation;
- expressions of at most 30 nodes;
- residuals weighted by 1/|y|, so that the search fits the relative error;
- constants fitted on half of the islands, by up to 10 Levenberg–Marquardt iterations (Nelder–Mead for one fit in
  ten);
- the hall of fame on the weighted squared error and a weighted node count;
- one thread.

srbf sets the operators, the budget (a number of generations) and the seed. TiSR's own budget is a wall-clock limit
of 300 s; the worker raises it to a guard that the budgets of the ladder do not reach.

```yaml
model_adapter:
  type: subprocess
  worker: tisr
  python: "{{ROOT}}/envs/tisr/bin/python"
  config_provenance: upstream_default
  simplipy_engine: acj-5-4-llm
  timeout: 7200
  options:
    generations: 64
```

| key | default | meaning |
|---|---|---|
| `options.generations` | `512` | TiSR's number of generations: the budget |
| `options.seed` | `0` | mixed with a hash of the problem's data into the run's seed |
| `options.time_guard` | `3600` | TiSR's wall-clock limit in seconds, a guard; the `hit_time_guard` column marks a run it stopped |
| `options.warmup` | `true` | run a throwaway fit when the worker starts, so that compiling TiSR is not part of the first problem's time |
| `options.config`, `options.config_by_problem` | none | TiSR settings that replace its defaults, for every problem or per problem; for side experiments only, which are then `harness_tuned` |
| `python`, `env`, `timeout`, `max_restarts`, `worker_log` | | as for every worker ([the config keys](adapters.md#the-config-keys)) |

**Operators.** TiSR takes any Julia function as an operator, so it searches over all the operators the expressions
are written in: `+ - * / ^ rootn` and `neg abs inv sin cos tan asin acos atan sinh cosh tanh asinh acosh atanh exp
log`. Its evaluator refuses a logarithm of a number that is not positive, a division by zero and a power of a
negative base before computing them, and drops the expression; any other function that fails stops the whole search.
The worker therefore gives TiSR `asin acos acosh atanh` as functions that return NaN outside their domain, as srbf
evaluates them, so that TiSR drops such an expression as well. `rootn` needs an integer index, which a fitted
constant rarely is; TiSR writes roots as powers.

**One answer per problem.** TiSR hands its user the hall of fame and has no rule that picks one expression from it;
every table and file it writes lists the hall of fame ordered by the fit objective. The worker answers with the first
row of that order: the member with the lowest weighted squared error. The whole hall of fame, with TiSR's measures of
each member, is stored in the `hall_of_fame` column.

**What to know when reading the results:**
- The weights 1/|y| make the search fit relative errors. TiSR replaces the infinite weight of a zero target by
  1e100, and such a point dominates the fit.
- The answer is the most accurate member of the front and so its most complex one. On noiseless data a longer
  expression can fit as well as the ground truth up to rounding and then is the answer.
- A run is not reproducible bit for bit: TiSR picks parents by their rank and crowding, and an expression that has
  not been through a selection yet, as all of an island's first expressions, carries values that were never set.
  The seed fixes everything else.
- The worker writes the answer from TiSR's tree with every constant at full precision; the `string_deviation`
  column records how far the string's values are from TiSR's own.

`configs/evaluation/scaling/tisr_fastsrb.yaml` sweeps the generations in doublings from 1 to 512, about 100 s per
problem on the reference machine. `configs/evaluation/panels/tisr_fastsrb2025_fastsrb.yaml` runs the protocol of the
FastSRB paper, whose complexity cap per problem is taken from the ground truth, as a check against that paper's
numbers.

## NeSymReS

In an environment of its own:

```bash
pip install -e ".[baselines]"
git clone https://github.com/SymposiumOrganization/NeuralSymbolicRegressionThatScales nesymres/NeuralSymbolicRegressionThatScales
pip install -e nesymres/NeuralSymbolicRegressionThatScales/src
pip install pytorch-lightning==2.5.6
python scripts/patch_typing_io.py
python scripts/patch_nesymres.py nesymres/NeuralSymbolicRegressionThatScales
```

`patch_typing_io.py` lets the installed Hydra, OmegaConf and ANTLR packages import on current
Python; run it again after reinstalling any of them. `patch_nesymres.py` does the same for the
clone: a dataclass default and the version pins. The model itself is untouched. Then download the
checkpoint (`100M.ckpt`,
`config.yaml`, `eq_setting.json`) into `models/nesymres/`, as the upstream repository describes.

```yaml
model_adapter:
  type: nesymres
  config_provenance: upstream_default
  simplipy_engine: acj-5-4-llm
  eq_setting_path: "{{ROOT}}/models/nesymres/eq_setting.json"
  config_path: "{{ROOT}}/models/nesymres/config.yaml"
  weights_path: "{{ROOT}}/models/nesymres/100M.ckpt"
  beam_width: 32
  device: cuda
```

The three paths and the engine are required. `beam_width` is the budget and `n_restarts` the
restarts of its constant fit; both default to the values in the checkpoint's own config. `device`
defaults to `cpu`. The model takes a fixed number of input variables. A problem with fewer is
padded with zero columns; a problem with more is given to the model with its first columns only,
with a warning, and the prediction is still judged against the full problem.

## E2E

The end-to-end transformer of Kamienny et al. (`facebookresearch/symbolicregression`), in an
environment of its own:

```bash
pip install -e ".[baselines]"
git clone https://github.com/facebookresearch/symbolicregression e2e/symbolicregression
python scripts/patch_symbolicregression.py e2e/symbolicregression
pip install -e e2e/symbolicregression
```

The patch lets the code run on current NumPy and PyTorch; the model itself is untouched. Then
download the weights (`model1.pt`)
into `models/e2e/`, as the upstream repository describes.

```yaml
model_adapter:
  type: e2e
  config_provenance: upstream_default
  simplipy_engine: acj-5-4-llm
  model_path: "{{ROOT}}/models/e2e/model1.pt"
  candidates_per_bag: 1
  device: cuda
```

| key | default | meaning |
|---|---|---|
| `model_path` | required | the weights |
| `candidates_per_bag` | `1` | candidates decoded per bag of points: the budget |
| `max_input_points` | `200` | points per bag |
| `max_number_bags` | `10` | bags per problem |
| `n_trees_to_refine` | `10` | candidates whose constants are refined |
| `rescale` | `true` | standardize the inputs, as the model expects |
| `device` | `cpu` | |

## Sampling and enumeration baselines

`lample_charton` fits expressions sampled from a generative `symbolic-data` catalog and
`brute_force` enumerates them. Neither has weights. Both need `simplipy_engine` and a `catalog` to
draw from; this is the adapter's own key and unrelated to `data_source.catalog`, which names the
expressions being evaluated.

```yaml
model_adapter:
  type: lample_charton
  config_provenance: harness_tuned
  simplipy_engine: acj-5-4-llm
  catalog: "{{ROOT}}/models/psaegert/flash-ansr-v25.0-T8-3M/catalog_train.yaml"
  samples: 32
```

| key | default | meaning |
|---|---|---|
| `samples` (`lample_charton`) | `32` | expressions sampled per problem: the budget |
| `max_expressions`, `max_length` (`brute_force`) | `10000`, none | how far the enumeration goes |
| `n_restarts`, `refiner_method`, `refiner_p0_noise` | `8`, `curve_fit_lm`, `normal` | the constant fit |
| `node_penalty` | `0.05` | the prediction minimizes log10 FVU plus this penalty per node |
| `unique`, `seed` (`lample_charton`) | `true`, none | skip duplicate samples; seed the sampler |
