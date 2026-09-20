# Contributing

## Adding your symbolic-regression method

The [adapter contribution guide](docs/adapters.md) is the full reference. The short version:

```bash
git clone https://github.com/psaegert/srbf && cd srbf
pip install -e .                    # from the checkout, so that the worker you add is found by its name
srbf new mymethod --repo            # the worker, its suite config, an environment recipe and a test, where the PR wants them
```

Then put your method into `fit()` in `src/srbf/worker/models/mymethod_worker.py`, pin its
environment in `envs/mymethod/requirements.txt`, and run

```bash
export FLASH_ANSR_ROOT=$PWD                                     # models, results and environments live under here
python -m venv envs/mymethod && envs/mymethod/bin/pip install -r envs/mymethod/requirements.txt
srbf check -c configs/evaluation/mymethod_srbf.yaml             # a few real problems end to end
srbf run -c configs/evaluation/mymethod_srbf.yaml -v            # the whole suite, or --experiment fastsrb
srbf analyze -c configs/evaluation/mymethod_srbf.yaml -o report # the standardized report
```

The config's `python:` already points at that interpreter. The worker runs in its own environment,
so any torch, simplipy or Julia version is fine; srbf never imports your code.

A pull request carries:

1. the worker (`src/srbf/worker/models/<name>_worker.py`),
2. the environment recipe (`envs/<name>/requirements.txt` or an equivalent lock file, plus where
   the weights come from),
3. the config (`configs/evaluation/<name>_srbf.yaml`) with its
   [`config_provenance` label](docs/fairness.md),
4. a section in [docs/models.md](docs/models.md),
5. the smoke test (`tests/test_workers/test_<name>_worker.py`; it skips where the environment is
   not provisioned).

After the merge we aim to provision the environment from your recipe on the reference machine,
run the full suite and publish the numbers on the results site, as compute allows. Your own runs
use the identical config, so up to sampling noise the recovery metrics you measure are the ones
that get published; only the timings depend on the machine.

## Working on srbf itself

```bash
pip install -e '.[analysis]' pytest pytest-cov pre-commit
pre-commit install
pytest tests
```

Tests that need provisioned assets or a method's environment skip when those are absent. Keep
`pre-commit run --all-files` green (flake8, mypy). An experiment belongs on a branch or in a
config, not behind a switch in the package.
