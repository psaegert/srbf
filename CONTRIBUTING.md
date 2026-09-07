# Contributing

## Adding your symbolic-regression method

The [adapter contribution guide](docs/adapters.md) is the full reference. The short version:

```bash
pip install srbf
git clone https://github.com/psaegert/srbf && cd srbf
srbf new mymethod --repo            # the worker, its suite config, an environment recipe and a test, where the PR wants them
```

Then put your method into `fit()` in `src/srbf/worker/models/mymethod_worker.py`, pin its
environment in `envs/mymethod/requirements.txt`, and run

```bash
export FLASH_ANSR_ROOT=$PWD                                     # models, results and environments live under here
python -m venv envs/mymethod-venv && envs/mymethod-venv/bin/pip install -r envs/mymethod/requirements.txt
srbf check -c configs/evaluation/mymethod_srbf.yaml             # a few real problems end to end
srbf run -c configs/evaluation/mymethod_srbf.yaml -v            # the whole suite, or --experiment fastsrb
srbf analyze -c configs/evaluation/mymethod_srbf.yaml -o report # the standardized report
```

(with `python:` in the config pointing at that interpreter). The worker runs in its own environment,
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

We merge, provision the environment from your recipe on the calibrated reference machine, run the
full suite, and publish the numbers on the results site. Your own runs use the identical config,
so the recovery metrics you measure are the ones we publish; only the timings depend on the
machine.

## Working on srbf itself

```bash
pip install -e '.[analysis]' pytest pytest-cov pre-commit
pre-commit install
pytest tests
```

Tests that need provisioned assets or a method's environment skip when those are absent. Keep
`pre-commit run --all-files` green (flake8, mypy); production code carries no experiment switches.
