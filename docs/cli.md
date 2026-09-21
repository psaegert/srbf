# Command line

```text
srbf {new, check, run, status, merge, analyze, decontamination}
```

`python -m srbf` is the same program. Every command prints its own `--help`.

## `srbf run`

Evaluate a config: every experiment and every sweep rung it defines, one after the other.

| flag | meaning |
|---|---|
| `-c`, `--config` | the config file (required) |
| `--experiment` | run one experiment of the config |
| `--sweep-filter` | run the rungs whose axis labels match, `AXIS=VALUE`; several as `AXIS=VALUE,AXIS=VALUE`, all of which must match |
| `--shard` | `K/N`: evaluate every N-th problem starting at K (from 0) and write `<output>.shard-K-of-N.<ext>` |
| `-n`, `--limit` | the number of problems, including those already in a resumed file |
| `-o`, `--output-file` | the result file, overriding `runner.output` |
| `--save-every` | write the result file every N problems |
| `--no-resume` | start over and overwrite an existing result file |
| `-v`, `--verbose` | a progress bar and one line per run |

A filter that matches no rung runs nothing. See [Running evaluations](running.md).

## `srbf check`

Walk a config end to end on a few real problems before the long run. The exit code is 1 when a
step failed.

| flag | meaning |
|---|---|
| `-c`, `--config` | the config file (required) |
| `--experiment` | the experiment to check; by default the first |
| `--all` | check every experiment |
| `--sweep-filter` | the rung to check, as for `run`; by default the first |
| `-n`, `--problems` | problems to fit per experiment; by default 2 |

## `srbf status`

Report how far every run of a config is, from its result files. No model and no worker is loaded.

| flag | meaning |
|---|---|
| `-c`, `--config` | the config file (required) |
| `--experiment` | report one experiment |
| `--sweep-filter` | report the rungs whose axis labels match, as for `run` |
| `--shard` | `K/N`: report the shard files of that shard |

Each run is `done`, `started` or `not started`, with the problems on disk out of its total. The
exit code is 0 only when every selected run is done. See
[Running evaluations](running.md#progress).

## `srbf new`

Scaffold an adapter for a method: `srbf new <name>`, where the name is a lowercase identifier.

| flag | meaning |
|---|---|
| `--dir` | where the adapter directory goes; by default `$FLASH_ANSR_ROOT/adapters`, else `./adapters` |
| `--python` | the interpreter of the method's environment, written into the config; by default `{{ROOT}}/envs/<name>/bin/python` |
| `--repo` | write into the layout of an srbf checkout, for a pull request |
| `--force` | overwrite files that exist |

It writes `worker.py`, `config.yaml`, `requirements.txt` and `test_worker.py`, and prints the next
steps. With `--repo`, run inside an srbf checkout, the four files go to
`src/srbf/worker/models/<name>_worker.py`, `configs/evaluation/<name>_srbf.yaml`,
`envs/<name>/requirements.txt` and `tests/test_workers/test_<name>_worker.py`.
See [Adding your method](adapters.md).

## `srbf merge`

Put the shard files of one run back together: `srbf merge -o <output> <shards...>`.

| flag | meaning |
|---|---|
| `-o`, `--output` | the unsharded result file to write (required) |
| `--allow-partial` | merge although shards are missing, and record which |

## `srbf analyze`

Derive the metrics and render the standard report, `results.md` and `figures/`, from the result
files of one or several configs.

| flag | meaning |
|---|---|
| `-c`, `--config` | a config; every experiment and rung whose result file exists becomes a run. Repeat the flag to compare methods |
| `--model` | the method's name for the `-c` in the same position; by default derived from the adapter block |
| `-o`, `--out-dir` | the output directory (required) |
| `--engine` | the SimpliPy engine the expressions are judged with; by default `acj-5-4-llm` |
| `--title` | the title of the report |

Instead of `-c`, a manifest file can list the runs by hand:

```yaml
runs:
  - {model: mymethod, benchmark: nguyen, scaling: 32, path: results/mymethod/nguyen.pkl}
```

Figures need the `analysis` extra. See [Results](results.md#the-standard-report).

## `srbf decontamination`

Verify that the holdout of a training catalog covers the benchmark expressions.

| flag | meaning |
|---|---|
| `-t`, `--training-catalog` | the training catalog, a generative `symbolic-data` config (required) |
| `-b`, `--benchmarks` | the catalogs to verify; by default the training catalog's `holdout_pools` |
| `-o`, `--output-file` | write the coverage report as JSON |
| `-v`, `--verbose` | print the coverage of each catalog as it is computed |

For every benchmark problem the command asks the question the training sampler asks before it
rejects a draw: is this expression, in every way the catalog writes it, held out? A problem counts
as held out only when every rendering is. One that cannot be probed is reported as `UNVERIFIED` and
counts against the coverage. The exit code is 0 only when every problem is verified as held out; a
miss and an unverified problem both give 1. See [Fairness](fairness.md#verifying-decontamination).
