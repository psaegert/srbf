# Running evaluations

An evaluation is described by one YAML config and run with `srbf run -c <config>`. This page covers
the config, ladders, experiments and suites, and how a long evaluation is checked, resumed and
split across GPUs. Every flag is listed in the [command reference](cli.md).

## The root directory

srbf keeps models, results, adapters and environments under one directory, named by the
`FLASH_ANSR_ROOT` environment variable. In a config the token `{{ROOT}}` stands for it, so one
config runs on any machine:

```bash
export FLASH_ANSR_ROOT=/path/to/bench
```

`{{ROOT}}` is replaced wherever a config names a file or a directory: in the config path itself,
in `runner.output`, in `data_source.catalog` and `holdouts`, in the path fields of the
`model_adapter` block (`model_path`, `worker`, `python`, `worker_log`, `simplipy_engine`, and the
like) and in every string of a worker's `options` and `env`. Set the variable explicitly: without
it the token resolves to a directory that depends on how `flash-ansr` was installed.

## The config

```yaml
run:
  data_source:
    catalog: fastsrb                 # which laws
    sampling:
      n_support: 512                 # points the method fits on
      n_validation: 512              # held-out points
      noise: 0.0
      problems_per_expression: 1
  model_adapter:
    type: subprocess                 # which method, and how it is set up
    config_provenance: upstream_default
    worker: "{{ROOT}}/adapters/mymethod/worker.py"
    python: "{{ROOT}}/envs/mymethod/bin/python"
    simplipy_engine: acj-5-4-llm
  runner:
    output: "{{ROOT}}/results/evaluation/mymethod/fastsrb.pkl"
    save_every: 20
    resume: true
```

`data_source` is described in [Benchmarks](benchmarks.md#the-data_source-block) and
`model_adapter` in [Models](models.md) and [Adding your method](adapters.md). The `runner` block:

| key | default | meaning |
|---|---|---|
| `output` | none | the result file. Without it the results stay in memory |
| `save_every` | none | write the file every N problems; needs `output`. A final save always happens when the run completes |
| `resume` | `true` | continue an existing result file instead of starting over |
| `limit` | `null` | the number of problems to evaluate |

The number of problems of a run is the first of these that is set: `--limit` on the command line,
`runner.limit`, `data_source.target_size`, the size of the catalog. Saves are atomic: a run that is
killed leaves the last saved file intact and loses the problems since then, so set `save_every` on
anything long.

## Ladders and sweeps

A `!sweep` tag marks a value that varies across runs, so one config expands into many:

```yaml
run:
  data_source:
    catalog: fastsrb
    sampling: {n_support: 512, n_validation: 512}
  model_adapter:
    type: pysr
    config_provenance: upstream_default
    simplipy_engine: acj-5-4-llm
    niterations: !sweep {name: ladder, values: [1, 4, 16, 64]}
  runner:
    output: !sweep
      name: ladder
      values:
        - "{{ROOT}}/results/pysr/fastsrb/iterations_0001.pkl"
        - "{{ROOT}}/results/pysr/fastsrb/iterations_0004.pkl"
        - "{{ROOT}}/results/pysr/fastsrb/iterations_0016.pkl"
        - "{{ROOT}}/results/pysr/fastsrb/iterations_0064.pkl"
```

- `!sweep {name: <axis>, values: [...]}` is a **named axis**. Sweeps that share a name advance
  together, element by element, and must have the same length. Here the budget and the output path
  move together: four rungs, four runs, four files.
- `!sweep [v1, v2, ...]` is an **anonymous axis**, a dimension of its own.

Different axes are combined as a grid. `runner.output` has to be swept along with the budget,
otherwise every rung writes to the same file. A named axis is labeled by the first of its sweeps
whose values are all different, here the iteration counts, and that label selects a rung:

```bash
srbf run -c pysr_fastsrb.yaml --sweep-filter ladder=16
```

The label is also stored in the result file. A `!sweep` can stand for any value in the config, such
as a noise level or a support size ([Benchmarks](benchmarks.md#sweeping-data-conditions)).

## Experiments

A config may hold several named runs under `experiments:` instead of one `run:` block, typically
one per catalog. Each entry has the three blocks of a run and may carry its own sweeps.

```bash
srbf run -c config.yaml                          # every experiment, every rung
srbf run -c config.yaml --experiment feynman     # one experiment
```

Experiments do not inherit from each other or from a top-level `run:` block. To share settings,
use YAML anchors; any top-level key srbf does not know is ignored and can hold them.

## Whole suites

`suite:` writes the experiments for you. It names a suite or lists catalogs and gives one `run:`
template; srbf makes one experiment per catalog, fills in `data_source.catalog` and replaces
`{catalog}` in every string of the template:

```yaml
suite: srbf                        # all 29 catalogs; or a list: [fastsrb, feynman]
run:
  data_source:
    sampling: {n_support: 512, n_validation: 512, noise: 0.0}
  model_adapter:
    type: subprocess
    config_provenance: upstream_default
    worker: "{{ROOT}}/adapters/mymethod/worker.py"
    python: "{{ROOT}}/envs/mymethod/bin/python"
    simplipy_engine: acj-5-4-llm
  runner:
    output: "{{ROOT}}/results/evaluation/mymethod/{catalog}.pkl"
    save_every: 20
```

`--experiment feynman` then runs one catalog, and a `!sweep` in the template applies to every
catalog. A config carries either `suite:` or `experiments:`, not both. This is the config that
`srbf new` writes.

## Checking a config first

```bash
srbf check -c config.yaml                                   # the first experiment, its first rung, two problems
srbf check -c config.yaml --experiment feynman -n 5         # one experiment, five problems
srbf check -c config.yaml --all --sweep-filter ladder=128   # every experiment at one rung
```

`srbf check` walks the path a run takes, step by step: the config, the sweep rung, the provenance
label, the output directory, the worker file, its interpreter and options, the judge's engine, the
catalog, the adapter's start, and then real fits with their metrics. Each step prints `ok` or
`FAIL` with the fix beside it, and the walk ends at a step that the later ones depend on. It writes
no result file, and its exit code is non-zero when a step failed, so the error a run would hit
after hours shows up within a minute. The [Quickstart](quickstart.md) shows its output.

## Resuming

With `resume: true`, `srbf run` loads the existing result file, counts its rows and evaluates only
the problems after them. A finished run is recognized before the method is loaded, so re-running a
whole suite costs nothing for the parts that are done. `--no-resume` ignores the existing file and
overwrites it.

Resuming is positional: it relies on the catalog yielding the laws in the same order, which holds
for the default `method: iterate`.

A problem whose fit failed is a result like any other: it is stored with its error, counts as a
miss and is not tried again. To evaluate a run afresh, for example after fixing a bug in your
worker, delete its result file or pass `--no-resume`.

## Progress

`srbf status` reads the result files of a config and reports how far every run is. It loads no
model and no worker:

```console
$ srbf status -c config.yaml
not started       0 / 12      experiment=nguyen, ladder=1
done             12 / 12      experiment=nguyen, ladder=4
started           1 / 2       experiment=koza, ladder=1
not started       0 / 2       experiment=koza, ladder=4
1 of 4 run(s) done
```

It takes the selectors of `srbf run` (`--experiment`, `--sweep-filter`, `--shard`), and its exit
code is 0 only when every selected run is done, so a script can wait on it. `srbf analyze` reads
whatever is on disk: a run that is only started enters the report with the problems it has, so
check the status before you read a report as final.

## Sharding a run across GPUs

One run can be too long for one GPU: `erbench-syneq` has 5,301 problems. `--shard K/N` makes a run
evaluate every N-th problem starting at the K-th, write its own file
`<output>.shard-K-of-N.<ext>`, and resume on its own. Submit the N shards as N jobs and put them
back together:

```bash
srbf run -c config.yaml --experiment erbench-syneq --sweep-filter ladder=1024 --shard 0/8
# ... shards 1/8 to 7/8, each on its own GPU
srbf merge -o results/erbench-syneq/draws_001024.pkl results/erbench-syneq/draws_001024.shard-*-of-8.pkl
```

`srbf merge` refuses shards that do not belong together: it checks that they share one shard
count, that no index repeats, that the columns are identical and that no problem appears twice.
It orders the rows by their position in the catalog and writes the file an unsharded run would
have written. Missing shards are an error unless `--allow-partial` is given, which records the gap
in the merged file. `--shard` applies to every run that the command selects, and a catalog with
fewer laws than shards leaves some shards empty.

## On a cluster

A whole-suite ladder is many independent runs, which is what a job array wants. A config like the
following gives one run per catalog and rung; `{catalog}` is filled in inside the sweep values too:

```yaml
suite: srbf
run:
  data_source:
    sampling: {n_support: 512, n_validation: 512, noise: 0.0}
  model_adapter:
    type: subprocess
    config_provenance: upstream_default
    worker: "{{ROOT}}/adapters/mymethod/worker.py"
    python: "{{ROOT}}/envs/mymethod/bin/python"
    simplipy_engine: acj-5-4-llm
    options:
      budget: !sweep {name: ladder, values: [1, 4, 16]}
  runner:
    output: !sweep
      name: ladder
      values:
        - "{{ROOT}}/results/evaluation/mymethod/{catalog}/budget_0001.pkl"
        - "{{ROOT}}/results/evaluation/mymethod/{catalog}/budget_0004.pkl"
        - "{{ROOT}}/results/evaluation/mymethod/{catalog}/budget_0016.pkl"
    save_every: 20
```

One array task then takes one catalog at one rung, here with Slurm:

```bash
#!/bin/bash
#SBATCH --array=0-28
#SBATCH --gres=gpu:1
CATALOGS=(fastsrb feynman feynman-bonus nguyen)   # ... the catalogs of your suite
export FLASH_ANSR_ROOT=/path/to/bench
srbf run -c config.yaml --experiment "${CATALOGS[$SLURM_ARRAY_TASK_ID]}" --sweep-filter ladder=16 -v
```

The worker inherits the environment of the job, so `CUDA_VISIBLE_DEVICES` and the modules the job
loaded reach your method. A task that hits its time limit loses nothing but the problems since the
last save: submit it again and it resumes. For a long catalog, add `--shard K/N` to N tasks and
merge them afterwards, and let `srbf status -c config.yaml` tell you when everything is in.

## From Python

```python
from srbf import Benchmark

for benchmark in Benchmark.runs_from_config(
    "configs/evaluation/scaling/flash-ansr-v25.0-T8-3M_srbf.yaml",
    experiment="nguyen",
    sweep_filter={"ladder": 32},
):
    snapshot = benchmark.run()      # a dict of columns, one entry per problem
```

A run started from Python resumes, saves and records what ran exactly like `srbf run`.
`Benchmark.runs_from_config` expands experiments, suites and sweeps into one `Benchmark` per run
and takes the command line's options as keyword arguments: `limit_override`, `output_override`,
`save_every_override`, `resume`, `experiment`, `sweep_filter` and `shard`. Each benchmark carries
its `label`, for example `{"experiment": "nguyen", "ladder": 32}`. `Benchmark.from_config` builds a
single run from a config without sweeps.
