<!-- For a new method, see CONTRIBUTING.md and docs/adapters.md. `srbf new <name> --repo` writes the files below. -->

## What this adds

<!-- the method, a link to its paper/repository, and what the adapter runs (checkpoint, defaults) -->

## Checklist for a new method

- [ ] worker: `src/srbf/worker/models/<name>_worker.py` (`fit`, optionally `load` / `info`)
- [ ] environment recipe: `envs/<name>/requirements.txt` (versions pinned) and where the weights come from
- [ ] config: `configs/evaluation/<name>_srbf.yaml` with its `config_provenance` label (docs/fairness.md)
- [ ] `srbf check -c configs/evaluation/<name>_srbf.yaml` passes on my machine
- [ ] a section in `docs/models.md`
- [ ] `tests/test_workers/test_<name>_worker.py` (skips where the environment is absent)
- [ ] `pre-commit run --all-files` and `pytest tests` pass

## Results you already have (optional)

<!-- `srbf analyze -c ... -o report` output, or the numbers from your own run; we re-run on the calibrated machine before publishing -->
