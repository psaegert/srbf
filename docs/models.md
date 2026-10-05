# Models

The methods that come with srbf, how to install each one, and the keys of its `model_adapter`
block. To enter a method of your own, see [Adding your method](adapters.md).

| `type` | method | installation |
|---|---|---|
| `flash_ansr` | [Flash-ANSR](#flash-ansr) and its prior reference | `pip install "srbf[flash-ansr]"`, then download a checkpoint |
| `pysr` | [PySR](#pysr) | `pip install pysr`, in an environment of its own if you like |
| `subprocess`, `worker: operon` | [Operon](#operon) | `pip install pyoperon==0.6.1 scikit-learn`, in an environment of its own |
| `subprocess`, `worker: rilsrols` | [RILS-ROLS](#rils-rols) | `scripts/envs/build_rilsrols_env.sh`, an environment of its own |
| `subprocess`, `worker: tisr` | [TiSR](#tisr) | `scripts/envs/build_tisr_env.sh envs/tisr`: Julia and TiSR in an environment of their own |
| `nesymres` | [NeSymReS](#nesymres) | clone, patch, download weights |
| `e2e` | [E2E](#e2e) | clone, patch, download weights |
| `subprocess`, `worker: dso` | [DSR and uDSR\*](#dso) | `scripts/envs/build_dso_env.sh`: a conda environment with Python 3.7 |
| `subprocess`, `worker: gpgomea` | [GP-GOMEA](#gp-gomea) | `scripts/envs/build_gpgomea_env.sh envs/gpgomea`: a conda environment of its own, compiled from source |
| `subprocess`, `worker: qlattice` | [QLattice](#qlattice) | `scripts/envs/build_qlattice_env.sh envs/qlattice`: a virtual environment with Python 3.12 |
| `lample_charton`, `brute_force` | [prior sampling and enumeration](#sampling-and-enumeration-baselines) | `pip install "srbf[flash-ansr]"` (they fit constants with its refiner) |
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
pip install "srbf[flash-ansr]"
hf download psaegert/flash-ansr-v25.0-T8-3M --local-dir "$SRBF_ROOT/models/psaegert/flash-ansr-v25.0-T8-3M"   # also: -T8-20M, -T8-120M
```

The adapter needs `flash-ansr`, which srbf installs only with the `flash-ansr` extra, and `hf` (from
`huggingface_hub`, which comes with it) puts the checkpoint where the configs look for it. The configs
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
| `evaluation_config.generation_config` | required | `method` (`softmax_sampling`, `prior_sampling` for the prior reference or `oracle` for the oracle) and its `kwargs`; `draws` is the budget |
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

**The oracle.** With `generation_config.method: oracle` the model is not used either: the one candidate
is the problem's ground truth, in the model's own emission format (a fittable literal is a
`<constant>` for the refiner, a pow exponent or root index stays spelled), fitted and ranked like any
other. Its budget is the refiner's restarts (`configs/evaluation/scaling/flash-ansr-v25.0-T8-oracle_srbf.yaml`,
1 to 1,024). It is the ceiling of the fitting stage and runs only where a ground truth exists. The
judge is strict for the oracle as for every method: where the refitted law comes back in another form,
a constant factor or a root spelled differently, that is another structure.

## Flash-ANSR + PySR

The adapter type `flash_ansr_hybrid` evaluates flash-ansr's hybrid (`flash_ansr.hybrid`, flash-ansr >= 0.19,
`pip install srbf[hybrid]`): Flash-ANSR draws and fits candidates, its best `k_seeds` seed PySR's
populations, PySR's hall of fame is priced the way Flash-ANSR prices its own candidates, and Flash-ANSR's
ranking picks the prediction from the combined pool.

```yaml
model_adapter:
  type: flash_ansr_hybrid
  flash_ansr: {type: flash_ansr, model_path: "{{ROOT}}/models/flash-ansr-v25.0-T8-20M", ...}   # a full flash_ansr block
  hybrid:
    rungs: [[512, 16], [1024, 64], [2048, 256], [4096, 512], [8192, 1024]]   # (draws, PySR iterations) per budget
    k_seeds: 100
    snapshot_dir: "{{ROOT}}/snapshots/hybrid"   # one generation pass per problem serves every budget of the ladder
  pysr: {warmup: true}
```

The budget is a pair: Flash-ANSR's draws and PySR's iterations, paired so that both stages take the same time on
the reference machine (r* = 0.5; `flash_ansr.hybrid.R_STAR_LADDER`); the axis is labelled by the draws. The hybrid
can also run by the clock (`hybrid: {budget_s: T, ratio: r}`: Flash-ANSR gets (1 - r) T of wall time, PySR the
rest), which is how the ratio was chosen.

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
- The option `threads` sets how many threads a search runs on (default 1, the author's and pyoperon's own). The
  benchmark's configs set `threads: all`, so Operon gets the whole machine, as every method does. With more than one
  thread the evaluation budget is counted in parallel and a run is not reproducible exactly.

The worker stores the whole Pareto front in the `front` column. `configs/evaluation/scaling/operon_fastsrb.yaml`
sweeps the evaluations in doublings from 2^10 to 2^23. The author configuration also stops a search after 1,000
generations: at 2^22 a search runs about 800 generations, and at 2^23 every search ends at the cap, about 5 s per
problem on the reference machine. A larger budget adds no time, so the ladder ends there, short of the 100 s the other
methods reach.

## RILS-ROLS

```bash
PYTHON=python3.12 scripts/envs/build_rilsrols_env.sh envs/rilsrols     # needs a C++17 compiler; about a minute
```

RILS-ROLS (Kartelj and Djukanović, Journal of Big Data, 2023) is iterated local search over expression trees. From
the best expression so far it perturbs the tree, runs a local search from every perturbation, and fits the linear
coefficients of every candidate by ordinary least squares. It ranks candidates by a product of R², RMSE and size.
srbf runs release 1.6.7 (`kartelj/rils-rols`, MIT licence), the release its authors submitted to SRBench, as
`worker: rilsrols` in an environment of its own.

The build script creates a Python 3.12 environment from the pinned packages in
`scripts/envs/rilsrols-requirements.txt` (numpy 1.26.4 as in SRBench's environment for the method, the others at
their releases of the submission's date), downloads the source release from PyPI, checks its hash and compiles it.
RILS-ROLS compiles with `-march=native`, so build the environment on each machine that runs it.

The configuration is the one RILS-ROLS's first author committed for running it as a benchmark baseline (his
SRBench submission), without the hyperparameter grid that the benchmark's maintainers later searched around it:
- expressions of at most 50 nodes;
- the sample size chosen by the method, which on up to 10,000 points takes them all;
- the size penalty at its default, 0.001.

```yaml
model_adapter:
  type: subprocess
  worker: rilsrols
  python: "{{ROOT}}/envs/rilsrols/bin/python"
  config_provenance: author_blessed
  simplipy_engine: acj-5-4-llm
  timeout: 4800
  options:
    max_fit_calls: 1048576
```

| key | default | meaning |
|---|---|---|
| `options.max_fit_calls` | `1000000` | fitness evaluations of whole expressions, the local search's included: the budget |
| `options.seed` | `0` | mixed with a hash of the problem's data into the run's seed |
| `options.config` | none | settings that replace the configuration's; for side experiments only, which are then `harness_tuned` |
| `python`, `env`, `timeout`, `max_restarts`, `worker_log` | | as for every worker ([the config keys](adapters.md#the-config-keys)) |

RILS-ROLS stops by itself after 3,600 s, a guard the ladder does not reach. The worker records a fit that runs
600 s past that as the problem's error. Set srbf's `timeout` above both, as above, so that srbf does not restart
the worker for such a fit.

**Operators.** RILS-ROLS's operators are fixed in its code and cannot be chosen: `+ - * /`, `sin cos exp log sqrt`
and the square, all among the benchmark's operators. It has no `tan`, no inverse or hyperbolic functions, no `abs`
and no other powers or roots. Its starting constants are -1, 0, 0.5, 1, 2, π and 10; every other constant comes from
least squares.

**One change to the source.** RILS-ROLS prints its model's constants with six decimals, so that 6.674e-11 reads as
0 and every constant loses most of its digits. The build script changes that in one place
(`scripts/envs/patch_rilsrols.py`): the constants of the final model are printed with every digit. RILS-ROLS writes
a constant within 1e-12 of an integer as that integer, and the change keeps that rule except where the integer is
0, where it would turn a division by a small constant into a division by zero. The search compares candidates by
their printed form too, so that printing stays as it is and only the model handed back at the end changes. On the
same data and seed, the changed and the unchanged build take the same steps and return the same model.

**What to know when reading the results:**
- The prediction is the method's own answer: its final model after sympy's simplification, with every constant at
  full precision and in srbf's variable names. RILS-ROLS gives the simplification two seconds and otherwise returns
  the model as it printed it; a fit whose `fit_time` exceeds the search's own `total_time` by two seconds took that
  path. When the simplified model holds a function the benchmark does not read, the worker writes the model as
  printed instead and sets `answer_form` to `unsimplified`. `model_string` keeps the model as RILS-ROLS printed it.
- RILS-ROLS drops every least-squares coefficient smaller than 1e-12 in magnitude, so on a law whose values are
  that small it returns a constant.
- It evaluates without protected operators, as srbf does. The `string_deviation` column records the largest
  deviation of the written prediction from the method's own values on the support points, relative to the largest
  of those values.
- It does not stop early: a fit spends its whole budget, also after it has fit the data exactly. The `fit_calls`
  column records what a fit spent, a few calls more than the budget.
- Every fit runs in a process forked for it. RILS-ROLS has no parallelism, so a fit runs on one thread; the
  benchmark's configs set `threads: all` as for every method, and the worker caps no library's thread pool. The same
  data and seed give the same answer on the same machine.

`configs/evaluation/scaling/rilsrols_fastsrb.yaml` sweeps the fitness evaluations in doublings from 2^6, the first
power of two above the method's first step (scoring the perturbations of its starting model), up to 2^22, about
100 s per problem on the reference machine.

## TiSR

```bash
scripts/envs/build_tisr_env.sh envs/tisr     # needs uv, curl and git; downloads Julia and compiles TiSR
```

TiSR (thermodynamics-informed symbolic regression; Martinek, Frotscher, Richter and Herzog) is genetic programming
in Julia: NSGA-II on islands, the constants of new expressions fitted by Levenberg–Marquardt, and a hall of fame,
the Pareto front of fit error against complexity. It runs through the [worker protocol](adapters.md) as
`worker: tisr` and calls Julia through juliacall. TiSR is licensed under the Apache License 2.0.

The build script creates the environment: Python 3.12 with juliacall, Julia 1.12.7, and TiSR at commit `9e628e6` of
its main branch (January 2026), with every Julia package at the version of the lock file
(`scripts/envs/tisr/Manifest.toml`), resolved from TiSR's own compatibility bounds, all in a Julia depot of the
environment's own. The worker points juliacall at this Julia and depot and starts it with one thread.

The configuration is TiSR's own defaults at that commit:
- 20 islands of 50 expressions, each breeding about 50 new expressions per generation;
- expressions of at most 30 nodes;
- residuals weighted by 1/|y|, so that the search fits the relative error;
- constants fitted on half of the islands, by up to 10 Levenberg–Marquardt iterations (Nelder–Mead for one fit in
  ten);
- an expression seen recently is rejected with probability 0.9;
- the hall of fame on the weighted squared error and a weighted node count.

On a system whose `libstdc++` is older than Julia 1.12 needs (`GLIBCXX_3.4.26`, missing on RHEL 8 and its
relatives), put Julia's own first for the worker: `env: {LD_LIBRARY_PATH: <prefix>/julia-1.12.7/lib/julia}` in the
config. Without it juliacall loads the system's and stops with "version `GLIBCXX_3.4.26' not found".

srbf sets the operators, the budget (a number of generations), the seed and the threads. TiSR runs on one thread
unless it is told otherwise; given more, it breeds and fits the islands of a generation in parallel, as its
documentation describes, and the benchmark gives it the whole machine (`threads: all`). BLAS stays at one thread so
that the islands do not each start a BLAS pool on the same cores.

**Every run spends its whole budget.** TiSR ends a search at the first of four conditions: the number of
generations, a wall-clock limit (300 s by default), a user callback, or `q` typed on its input. The worker keeps only
the number of generations: it raises the wall-clock limit to a guard far above every budget of the ladder, leaves the
callback at TiSR's default, which never stops, and gives the search no input. A constant fit inside a generation ends
at TiSR's own iteration limits and convergence test, which end that fit, not the search.

```yaml
model_adapter:
  type: subprocess
  worker: tisr
  python: "{{ROOT}}/envs/tisr/bin/python"
  config_provenance: upstream_default
  simplipy_engine: acj-5-4-llm
  timeout: 7200
  selection: {mode: mdl}       # TiSR picks no answer; without this every problem counts as failed
  options:
    generations: 64
    threads: all
```

| key | default | meaning |
|---|---|---|
| `options.generations` | `512` | TiSR's number of generations: the budget |
| `options.threads` | `1` | the threads Julia starts with, or `all`; with more than one, TiSR breeds and fits its islands in parallel |
| `options.seed` | `0` | mixed with a hash of the problem's data into the run's seed |
| `options.time_guard` | `3600` | TiSR's wall-clock limit in seconds, a guard; the `hit_time_guard` column marks a run it stopped |
| `options.warmup` | `true` | run a throwaway fit when the worker starts, so that compiling TiSR is not part of the first problem's time |
| `options.config`, `options.config_by_problem` | none | TiSR settings that replace its defaults, for every problem or per problem; for side experiments only, which are then `harness_tuned`; they cannot set a stop |
| `python`, `env`, `timeout`, `max_restarts`, `worker_log` | | as for every worker ([the config keys](adapters.md#the-config-keys)) |

**Operators.** TiSR takes any Julia function as an operator, so it searches over all the operators the expressions
are written in: `+ - * / ^ rootn` and `neg abs inv sin cos tan asin acos atan sinh cosh tanh asinh acosh atanh exp
log`. Its evaluator refuses a logarithm of a number that is not positive, a division by zero and a power of a
negative base before computing them, and drops the expression; any other function that fails stops the whole search.
The worker therefore gives TiSR `asin acos acosh atanh` as functions that return NaN outside their domain, as srbf
evaluates them, so that TiSR drops such an expression as well. `rootn` needs an integer index, which a fitted
constant rarely is; TiSR writes roots as powers.

**No single answer of its own.** A TiSR search returns its hall of fame, and TiSR defines no rule that picks one
expression from it: its README and example sort the hall of fame for the user to inspect, and its export functions
order their tables by the fit error. The worker therefore returns the whole hall of fame as the problem's candidates,
and srbf picks one only when the config sets `selection` ([Picking among candidates](adapters.md#the-config-keys)):
`{mode: mdl}`, the two-part code srbf ranks Flash-ANSR's candidates with, is srbf's rule and not TiSR's. Without it
every problem is recorded as failed. Either way the row carries the whole hall of fame in the `hall_of_fame` column,
each member written in the benchmark's syntax with every constant at full precision, with TiSR's measures of it and
with its `string_deviation`, how far the string's values are from TiSR's own.

**What to know when reading the results:**
- The weights 1/|y| make the search fit relative errors. TiSR replaces the infinite weight of a zero target by
  1e100, and such a point dominates the fit.
- TiSR's runs are not reproducible from the seed: two fits of the same data with the same seed can return different
  halls of fame. TiSR picks parents by their rank and crowding, and an expression that has not been through a
  selection yet, as all of an island's first expressions, carries values that TiSR never set. srbf runs TiSR as it
  is.

`configs/evaluation/scaling/tisr_fastsrb.yaml` sweeps the generations in doublings from 1 to 2048, about 100 s per
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

## DSO

```bash
scripts/envs/build_dso_env.sh envs/dso        # needs conda or mamba, git and a C compiler
```

DSO (deep symbolic optimization) runs through the [worker protocol](adapters.md) as `worker: dso`, in an
environment of its own: its release v3.0.0 needs Python 3.6 or 3.7 and TensorFlow 1.14. The script creates that
environment with conda from pinned package lists (`scripts/envs/dso-conda-linux-64.txt`,
`scripts/envs/dso-requirements.txt`) and installs DSO from its release commit in editable mode, which DSO requires.
One worker runs two methods, chosen by `options.arm`:

- **DSR** (`arm: dsr`): deep symbolic regression, an RNN trained with a risk-seeking policy gradient (Petersen et
  al., ICLR 2021), in the regression configuration DSO ships: batch 1,000, learning rate 0.0005, entropy weight 0.03
  with a decay of 0.7 along the expression, the soft length and uniform arity priors, at most 64 tokens. The constant
  token is added, as DSO's authors advise for data with constants.
- **uDSR\*** (`arm: udsr`): unified deep symbolic regression (Landajuela et al., NeurIPS 2022) as far as it was
  released.[^udsr] It adds GP-meld, a genetic-programming inner loop seeded with each batch of the RNN (Mundhenk
  et al., NeurIPS 2021), and a polynomial token whose coefficients are fitted by least squares. The configuration is
  the uDSR paper's Table 3: priority queue training with learning rate 0.0025, batch and GP population 500, 25 GP
  generations per iteration, entropy weight 0.03 with decay 0.7, the polynomial token of degree 3 with at most 10
  terms, and the constant token and the literal 1.

[^udsr]: Public release: DSR, GP-meld and the polynomial token; the paper's AI Feynman step and pre-training were
    never released.

```yaml
model_adapter:
  type: subprocess
  worker: dso
  python: "{{ROOT}}/envs/dso/bin/python"
  config_provenance: author_blessed        # upstream_default for arm: dsr
  simplipy_engine: acj-5-4-llm
  timeout: 7200
  options:
    arm: udsr
    n_samples: 13000
```

| key | default | meaning |
|---|---|---|
| `options.arm` | required | `dsr` or `udsr` |
| `options.n_samples` | `2000000` | expressions the RNN samples and GP-meld breeds, repeats included: the budget |
| `options.seed` | `0` | mixed with a hash of the problem's data into the run's seed |
| `options.max_seconds` | `3600` | a wall-clock guard far above every budget of the ladders; when it passes, the best expression so far is returned and `guard_hit` is set |
| `options.warmup` | `true` | run a small throwaway fit when the worker starts, so that compiling DSO's numba functions is not part of the first problem's time |
| `python`, `env`, `timeout`, `max_restarts`, `worker_log` | | as for every worker ([the config keys](adapters.md#the-config-keys)) |

**Budget.** DSO checks `n_samples` after each iteration: an iteration is 1,000 expressions for DSR and 500 + 25 ×
500 = 13,000 for uDSR\*, so the ladders count whole iterations and double them: DSR's `n_samples` is 1,000 × 2^k,
uDSR\*'s 13,000 × 2^k. A budget between two multiples of an iteration would run to the next one. A search stops
early once an expression fits the data to a normalized mean squared error below 1e-12, which happens only on
noiseless data.

**Operators.** DSO searches over the operators it has among those the expressions are written in:

- `+ - * /` and `neg abs inv sin cos tan tanh exp log`;
- `^` as squares, cubes and fourth powers, and `rootn` as the square root.

It has no `asin acos atan sinh cosh asinh acosh atanh` and no general power or root.

**Patches.** The worker applies three patches to DSO v3.0.0, each the smallest change that restores the released
method; its docstring gives the details.

- GP-meld evaluates the expressions it breeds. The release scores every bred expression as the RNN sample it came
  from, which makes GP-meld inert; the patch restores what the GP-meld paper's own release evaluates.
- The polynomial token is fitted under `neg` and `n4` too. DSO fits it by inverting the operators above it and had
  no inverse for these two, which stopped the search.
- A GP-meld copy of an expression shares the primitive set instead of copying it. The search is unchanged
  expression for expression; the copies took most of an iteration's time.

**What to know when reading the results:**

- The prediction is the expression with the highest reward on the data, the inverse of the normalized root mean
  squared error; DSO adds no complexity penalty, so predictions can be long.
- An expression that raises a floating-point error on any data point, underflow included, gets the lowest reward.
- The polynomial token needs a data point per monomial: `C(d + 3, 3)` for `d` variables. With fewer points it is
  the constant 1, which the result records as `linear_underdetermined`; at 512 points this happens from 13
  variables on.
- The prior that keeps the polynomial token out of `sin cos tan abs` and the even powers binds the RNN only:
  GP-meld's check of it looks at one of these operators, as in the authors' own code, so GP-meld breeds such
  expressions.
- The option `threads` sets DSO's reward pool (`n_cores_batch`, the processes that compute the rewards of each
  batch's new expressions, their constant fits included). The default 1 is DSO's own, no pool. DSR's benchmark
  config sets `threads: all`, so DSR gets the whole machine, as every method does. uDSR* runs in DSO's default
  configuration, its one configuration, with the whole machine available to it. DSO pins its TensorFlow session to
  one thread itself. GP-meld's own evaluation pool (`parallel_eval`) stays off: it sends every new expression
  through the pool and back, and made uDSR* slower in every run we timed.
- DSO forks its reward pool anew for every fit, from a process that already runs TensorFlow's and the numerical
  libraries' threads, and a forked process can then wait forever on a lock one of those threads held. DSR's config
  therefore uses srbf's hang policy ([Adapters](adapters.md)), as PySR's does: a problem whose worker stops using the
  CPU for 60 seconds, or runs 30 times longer than the run's median problem, is tried once more in a fresh worker,
  and a second hang fails it. Every hang is logged.

The worker stores the Pareto front of complexity against reward over every expression evaluated in the `front`
column, and the expressions and iterations used in `nevals` and `iterations`.
`configs/evaluation/scaling/dso_dsr_fastsrb.yaml` sweeps DSR's budget in doublings of an iteration, from 1,000 to
16,000 expressions, up to about 100 s per problem on the reference machine.
`configs/evaluation/scaling/dso_udsr_fastsrb.yaml` runs uDSR\* for one iteration, 13,000 expressions: a single
iteration already takes about that long, because GP-meld fits the constants of thousands of new expressions in it.

## GP-GOMEA

```bash
scripts/envs/build_gpgomea_env.sh envs/gpgomea     # needs conda or mamba; a few minutes
```

GP-GOMEA (Virgolin, Alderliesten, Witteveen and Bosman) is genetic programming with gene-pool optimal
mixing: every tree fills a fixed template, and each generation the method learns which template
positions belong together and recombines them as blocks, keeping a change only when the tree does
not get worse. srbf runs the original code (`marcovirgolin/GP-GOMEA`) at commit `6a92cb6`, the
commit SRBench 2021 ran, through its Python bindings, as `worker: gpgomea` in an environment of its
own.

The build script creates a conda environment with the toolchain of that time (Python 3.8, Boost
1.74, Armadillo 9.9, gcc 11.2, scikit-learn 0.24), clones the commit and compiles it. The C++
source is compiled unchanged. The build system gets four adjustments, each explained in the
script:
- the Boost.Python and Boost.NumPy libraries are named for Python 3.8, as SRBench 2021's install
  script does;
- the environment's include and library directories are added to the compile and link lines, as
  SRBench 2021's install script does;
- the library directory is written into the module's run path, so the module loads without an
  activated environment;
- the compiler's sysroot is pinned to glibc 2.17, because the linker of gcc 11.2 cannot read the
  newer one that conda resolves by default.

The configuration is the one GP-GOMEA's first author committed for running it as a benchmark
baseline (SRBench 2021), without the hyperparameter grid that SRBench's maintainers searched around it:
- GP-GOMEA with the linkage-tree FOS, linear scaling and ephemeral random constants;
- the interleaved multistart scheme off, population 500, initial tree height 4, elitism 1;
- one thread, which the option `threads` raises: the benchmark's configs set `threads: all`, so GP-GOMEA gets the
  whole machine, as every method does.

The interleaved multistart scheme is the authors' way to run GP-GOMEA without choosing a population
size. It is off because the first author turned it off in his benchmark configuration.

```yaml
model_adapter:
  type: subprocess
  worker: gpgomea
  python: "{{ROOT}}/envs/gpgomea/bin/python"
  config_provenance: author_blessed
  simplipy_engine: acj-5-4-llm
  timeout: 9600
  options:
    max_evaluations: 1048576
```

| key | default | meaning |
|---|---|---|
| `options.max_evaluations` | `500000` | fitness evaluations of whole trees, the initial population's included: the budget |
| `options.seed` | `0` | mixed with a hash of the problem's data into the run's seed |
| `options.config` | none | settings that replace the configuration's; for side experiments only, which are then `harness_tuned` |
| `python`, `env`, `timeout`, `max_restarts`, `worker_log` | | as for every worker ([the config keys](adapters.md#the-config-keys)) |

GP-GOMEA stops by itself after 7,200 s, a guard the ladder does not reach. The worker records a fit
that runs 1,800 s past that as the problem's error. Set srbf's `timeout` above both, as above, so
that srbf does not restart the worker for such a fit.

**Operators.** GP-GOMEA has `+ - *`, `exp sin cos` and the square `(u)^2` directly. It has
division, logarithm and square root only in protected form, and each is written as the function it
computes, so that srbf evaluates what GP-GOMEA evaluated:

| GP-GOMEA | computes | written as |
|---|---|---|
| `p/(u, v)` | `sign(v) * u / (abs(v) + 1e-6)`, with `sign(0) = 1` | `u/(v + 1e-06)` when `v >= 0` at every support point, `u/(v - 1e-06)` when `v < 0` at every support point, otherwise `u/(v*(1 + 1e-06/abs(v)))` |
| `plog(u)` | `log(abs(u))`, and 0 where that is not finite | `log(abs(u))`, or `0` when `u` is 0 at every support point |
| `sqrt(u)` | `sqrt(abs(u))` | `sqrt(abs(u))` |
| `(u)^2` | `u**2` | `(u)**2` |

On every support point each spelling is GP-GOMEA's function. GP-GOMEA has no node for other
powers and roots, `abs`, `tan`, or the inverse and hyperbolic functions. It expresses `neg` and
`inv` through `-` and `p/`.

**What to know when reading the results:**
- GP-GOMEA returns its model as `a + b * f(x)`, with the intercept and slope fitted by least
  squares. It prints them with six decimals. The worker computes them again in double precision
  with GP-GOMEA's own least-squares step, which restores the values the method computed.
- Its random constants are multiples of 0.001 drawn from ±5 times the largest absolute input, and
  it does not optimize them.
- The intercept, the slope and the guards of the protected operators are constants of the
  prediction, so on a law without such constants the prediction rarely matches the ground truth
  symbol for symbol. Its numeric recovery is the comparable rate.
- The budget is checked between generations, so a run overshoots it by up to one generation (about
  10,000 evaluations at the start of a run). The `evaluations` column records what a fit spent.
- Every fit runs in a process forked for it and starts the random number generators as a fresh process would. On
  one thread the same data and seed give the same answer; with more (`threads`, GP-GOMEA's `parallel`) the
  evaluation count races and a run is not reproducible exactly.
- GP-GOMEA evaluates a formula only when a change reaches a part of the tree that the formula uses. Once
  every member of the population expresses the same formula, nothing it changes is evaluated: generations go
  on, the evaluation count stands still, and the answer does not change. The run moves on only when the unused
  parts have become identical too, which makes GP-GOMEA start a new population. A budget that a run reaches
  before its population settles costs a fraction of a second; one above that point waits out at least one such
  stall, ten seconds or more on the reference machine. Between 2^16 and 2^18 evaluations more and more runs stall, and the time per
  problem grows about tenfold per doubling of the budget.

The worker also stores GP-GOMEA's own printed model in the `model_string` column.
`configs/evaluation/scaling/gpgomea_fastsrb.yaml` sweeps the evaluations in doublings, from 2^14,
the first power of two above the cost of one generation, up to about 100 s per problem on the
reference machine. Between 2^16 and 2^18 it takes quarter steps (2^16.25, 2^16.5, ...), and between
2^16.75 and 2^17, where the time grows tenfold within one quarter step, sixteenth steps (2^16.8125, 2^16.875,
2^16.9375), so that its points stay about evenly spaced in time where the stalls set in.
`configs/evaluation/panels/gpgomea_srbench2021_feynman.yaml` runs the configuration that SRBench 2021
published its GP-GOMEA results with, on the Feynman catalogs.

## QLattice

```bash
scripts/envs/build_qlattice_env.sh envs/qlattice     # Python 3.12, feyn 3.5.0 and pinned dependencies
```

QLattice samples models from a probability distribution over expression graphs, fits every sampled model's
parameters by gradient descent and moves the distribution towards the best models, epoch after epoch. It is
distributed as the Python package `feyn` (Abzu), whose core is closed source; it runs locally, without a licence key
or network access. **Licence:** feyn is licensed CC BY-NC-ND 4.0, for research and other non-commercial use. The
script installs it from PyPI; srbf redistributes none of it.

QLattice runs through the [worker protocol](adapters.md) as `worker: qlattice`, in an environment of its own. The
configuration is the one QLattice's authors submitted to SRBench:
- their own epoch loop: sample new models, fit the whole pool, prune it, update the distribution;
- 200 epochs, at most 10 edges per model;
- models ranked by feyn's `wide_parsimony` criterion, fitted to the squared error, every input numerical;
- the prediction is the first model the loop returns, its best by that ranking.

```yaml
model_adapter:
  type: subprocess
  worker: qlattice
  python: "{{ROOT}}/envs/qlattice/bin/python"
  config_provenance: author_blessed
  simplipy_engine: acj-5-4-llm
  timeout: 7200
  options:
    n_epochs: 16
```

| key | default | meaning |
|---|---|---|
| `options.n_epochs` | `200` | epochs of the loop: the budget |
| `options.seed` | `0` | mixed with a hash of the problem's data into the run's seed |
| `options.max_time` | `3600` | the authors' wall-clock stop in seconds, SRBench's limit for this method; when it passes, the best models so far are returned and `hit_time_guard` is set |
| `python`, `env`, `timeout`, `max_restarts`, `worker_log` | | as for every worker ([the config keys](adapters.md#the-config-keys)) |

**Budget.** An epoch samples about a thousand new models and refits the whole pool, each model on 20,000 rows
resampled from the data, so its cost hardly depends on the number of data points.

**Operators.** QLattice searches over the functions it has among the benchmark's operators:
- `+ *` and `exp log tanh`;
- `^` as the square, `rootn` as the square root and `inv` as `1/u`;
- the affine node `w*u + b`, through which it places constants inside a model.

It has no `sin`, `cos` or other trigonometric function, no `abs`, and no general power or root; subtraction and
negation it expresses through signed weights. Its `gaussian` function, `exp(-2u²)`, is a compound of the
benchmark's operators and is left out.

**Patches.** The method is unchanged. The authors' loop gets one crash fix: it passes `stypes=None` to feyn's data
validation, which the version the authors submitted with (3.0.1) accepted and 3.5.0 rejects; the worker passes an
empty mapping there, which states the same thing, every input numerical.

**What to know when reading the results:**
- Every input enters through an affine map and the output leaves through one, so a prediction reads
  `A*f(a1*x1 + c1, ...) + B`. Like Operon's, that shape rarely matches a law symbol for symbol, so its numeric
  recovery is the comparable rate.
- A model has at most 10 edges.
- QLattice's `exp`, `log`, square root, square and `1/u` are protected: the model clips their argument. Where a
  clip is active at a support or validation point, the prediction writes it out with `abs`, so that the expression
  computes the model's own prediction; the `protections` column names these functions.
- The option `threads` sets the threads feyn fits its models on (the authors' "auto" is the cores minus one). The
  benchmark's configs set `threads: all`, so QLattice gets the whole machine, as every method does.

The worker stores the other models the method returned in the `diverse` column, and the largest deviation of the
expression from the model's own predictions in `string_deviation` and `string_deviation_val`.
`configs/evaluation/scaling/qlattice_fastsrb.yaml` sweeps the epochs in doublings from 1 to 256, about 100 s per
problem on the reference machine.

## Sampling and enumeration baselines

`lample_charton` fits expressions sampled from a generative `symbolic-data` catalog and
`brute_force` enumerates them. Neither has weights, and both fit their constants with flash-ansr's refiner
(`pip install "srbf[flash-ansr]"`). Both need `simplipy_engine` and a `catalog` to
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
