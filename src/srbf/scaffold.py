"""``srbf new <name>``: the four files a new adapter starts from.

Bench layout (the default), under ``<dir>/<name>/``: ``worker.py`` (the contract with a placeholder
``fit``), ``config.yaml`` (the whole srbf suite through that worker), ``requirements.txt`` (the
method's own environment) and ``test_worker.py`` (one toy problem through the adapter). Repo layout
(``--repo``, run inside an srbf checkout): the same four files where a pull request wants them.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from flash_ansr.utils.paths import get_root

_NAME = re.compile(r"^[a-z][a-z0-9_]*$")

WORKER_TEMPLATE = '''"""%NAME%: an srbf worker. It runs in ITS OWN interpreter (the config's ``python:``); srbf is not
imported here, so any torch / simplipy / numpy version goes.

The contract (docs/adapters.md):
    load(options) -> state      optional, once per run: load weights, start a backend
    info(state)   -> dict       optional: versions and provenance, recorded with the run
    fit(x, y, *, x_val, variables, meta, options, state) -> {"expression": "<infix>", ...}
"""
import numpy as np


def load(options):
    """``options`` is the config's ``model_adapter.options`` block, verbatim."""
    # return {"model": MyModel.load(options["checkpoint"], device=options.get("device", "cuda"))}
    return {}


def info(state):
    """Anything worth recording with the results: package versions, checkpoint hashes."""
    return {"worker": "%NAME%", "numpy": np.__version__}


def fit(x, y, *, x_val, variables, meta, options, state):
    """Fit ONE problem and return its expression.

    x: the support rows (one list of floats per point); y: their targets; x_val: the validation
    rows (may be empty); variables: the names of x's columns, in order (write the expression in
    them); meta: the problem's metadata (eq_id, ...); options and state: see load().

    Return {"expression": "<infix string>"} at least. Optional keys: y_pred and y_pred_val (your
    own predictions), constants, fit_time (seconds), extra (a JSON-able dict), or
    {"error": "<message>"} to fail this problem and go on with the next.
    """
    X = np.asarray(x, dtype=float)
    Y = np.asarray(y, dtype=float).reshape(-1)
    # -- replace from here: the placeholder fits a linear model -----------------------------------
    design = np.column_stack([X, np.ones(len(X))])
    coef, *_ = np.linalg.lstsq(design, Y, rcond=None)
    terms = ["%r*%s" % (c, v) for c, v in zip(coef[:-1].tolist(), variables)]
    expression = " + ".join(terms + [repr(coef[-1].item())])
    # -- to here ----------------------------------------------------------------------------------
    return {"expression": expression}
'''

CONFIG_TEMPLATE = '''# %NAME% on the srbf suite.
#   srbf check -c %CONFIG_NAME%              # a few real problems end to end, before the long run
#   srbf run   -c %CONFIG_NAME% -v           # the suite (--experiment fastsrb for one catalog)
#   srbf analyze -c %CONFIG_NAME% -o report  # the standardized report
# {{ROOT}} is FLASH_ANSR_ROOT; {catalog} is filled in per experiment. See docs/adapters.md.
suite: srbf                        # every srbf catalog; or a list, e.g. [fastsrb, feynman]
run:
  data_source:
    sampling:
      n_support: 512
      n_validation: 512
      noise: 0.0
      problems_per_expression: 1
  model_adapter:
    type: subprocess
    config_provenance: upstream_default   # docs/fairness.md: upstream_default | author_blessed | harness_tuned
    worker: '%WORKER%'
    python: '%PYTHON%'                    # the interpreter of the method's OWN environment
    options: {}                           # forwarded verbatim to load() and fit()
    simplipy_engine: acj-5-4-llm          # the engine the catalogs are judged with; keep it
    timeout: 3600                         # seconds per problem
    drop_unused_variables: true
    worker_log: '{{ROOT}}/results/evaluation/%NAME%/worker.log'
  runner:
    output: '{{ROOT}}/results/evaluation/%NAME%/{catalog}.pkl'
    save_every: 20
    resume: true
'''

REQUIREMENTS_TEMPLATE = '''# The environment %NAME%'s worker runs in (the config's python:). Pin what the method needs;
# srbf is NOT installed here. Create it with
#   python -m venv "$FLASH_ANSR_ROOT/envs/%NAME%" && "$FLASH_ANSR_ROOT/envs/%NAME%/bin/pip" install -r %REQ_PATH%
numpy
'''

TEST_TEMPLATE = '''"""One toy problem through the %NAME% worker in its own interpreter; skipped where that
environment does not exist."""
from pathlib import Path

import pytest

from srbf.testing import fit_once

CONFIG = %CONFIG_EXPR%


def test_%NAME%_worker_fits_a_toy_problem():
    try:
        record = fit_once(config=CONFIG)
    except FileNotFoundError as missing:  # the method's interpreter is not provisioned here
        pytest.skip(str(missing))
    assert record["prediction_success"], record.get("error")
    assert record["predicted_expression"]
    assert len(record["y_pred_val"]) == 12
'''


@dataclass
class Scaffold:
    name: str
    files: list[Path] = field(default_factory=list)
    config: Path | None = None
    next_steps: list[str] = field(default_factory=list)


def _render(template: str, **values: str) -> str:
    text = template
    for key, value in values.items():
        text = text.replace(f"%{key}%", value)
    return text


def _repo_root() -> Path:
    """The srbf checkout the command runs in (``--repo``)."""
    for candidate in (Path.cwd(), *Path.cwd().parents):
        pyproject = candidate / "pyproject.toml"
        if pyproject.is_file() and 'name = "srbf"' in pyproject.read_text() and (candidate / "src/srbf/worker/models").is_dir():
            return candidate
    raise ValueError("--repo needs to run inside an srbf checkout (no pyproject.toml naming srbf above the working directory)")


def _from_cwd(path: Path) -> str:
    """``path`` relative to the working directory when it lies below it, else absolute."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _rooted(path: Path) -> str:
    """``path`` as a ``{{ROOT}}`` reference when it lies under the asset root, else absolute."""
    root = Path(get_root()).resolve()
    resolved = path.resolve()
    try:
        return "{{ROOT}}/" + resolved.relative_to(root).as_posix()
    except ValueError:
        return str(resolved)


def scaffold_adapter(name: str, *, directory: str | None = None, python: str | None = None,
                     repo: bool = False, force: bool = False) -> Scaffold:
    """Write the worker, config, requirements and test for ``name``; refuses to overwrite unless ``force``."""
    if not _NAME.match(name):
        raise ValueError(f"the adapter name must be a lowercase identifier (letters, digits, _), got {name!r}")
    python_ref = python or f"{{{{ROOT}}}}/envs/{name}/bin/python"
    if repo:
        root = _repo_root()
        worker_path = root / "src/srbf/worker/models" / f"{name}_worker.py"
        config_path = root / "configs/evaluation" / f"{name}_srbf.yaml"
        requirements_path = root / "envs" / name / "requirements.txt"
        test_path = root / "tests/test_workers" / f"test_{name}_worker.py"
        worker_ref = name  # a built-in name: srbf/worker/models/<name>_worker.py
        config_expr = f'Path(__file__).resolve().parents[2] / "configs" / "evaluation" / "{name}_srbf.yaml"'
        config_name = config_path.relative_to(root).as_posix()
        requirements_ref = requirements_path.relative_to(root).as_posix()
    else:
        base = Path(directory) if directory else Path(os.environ.get("FLASH_ANSR_ROOT") or ".") / "adapters"
        adapter_dir = base / name
        worker_path = adapter_dir / "worker.py"
        config_path = adapter_dir / "config.yaml"
        requirements_path = adapter_dir / "requirements.txt"
        test_path = adapter_dir / "test_worker.py"
        worker_ref = _rooted(worker_path)
        config_expr = 'Path(__file__).resolve().parent / "config.yaml"'
        config_name = _from_cwd(config_path)
        requirements_ref = _from_cwd(requirements_path)
    files = {
        worker_path: _render(WORKER_TEMPLATE, NAME=name),
        config_path: _render(CONFIG_TEMPLATE, NAME=name, WORKER=worker_ref, PYTHON=python_ref, CONFIG_NAME=config_name),
        requirements_path: _render(REQUIREMENTS_TEMPLATE, NAME=name, REQ_PATH=requirements_ref),
        test_path: _render(TEST_TEMPLATE, NAME=name, CONFIG_EXPR=config_expr),
    }
    existing = [path for path in files if path.exists()]
    if existing and not force:
        raise FileExistsError(f"already there (pass --force to overwrite): {', '.join(str(p) for p in existing)}")
    for path, text in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    env_dir = python_ref.rsplit("/bin/python", 1)[0] if python is None else None
    steps = [f"write your method into fit() in {_from_cwd(worker_path)}"]
    if env_dir is not None:
        steps.append(f"create its environment: python -m venv \"{env_dir}\" && \"{env_dir}/bin/pip\" install -r {requirements_ref}"
                     f"   (or set python: in {config_name} to an interpreter you have)")
    steps += [f"srbf check -c {config_name}", f"srbf run -c {config_name} -v", f"srbf analyze -c {config_name} -o report"]
    if repo:
        steps.append("for the PR: a section in docs/models.md, then pytest tests/test_workers and pre-commit run --all-files")
    return Scaffold(name=name, files=list(files), config=config_path, next_steps=steps)


__all__ = ["Scaffold", "scaffold_adapter"]
