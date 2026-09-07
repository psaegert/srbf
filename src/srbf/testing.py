"""One toy problem through a worker, the way a run would do it: for a worker's own test.

::

    from srbf.testing import fit_once
    record = fit_once(config="adapters/mymethod/config.yaml")   # or fit_once(worker=..., python=...)
    assert record["prediction_success"], record.get("error")

``fit_once`` raises :class:`FileNotFoundError` when the method's interpreter does not exist on this
machine, so a test can ``pytest.skip`` instead of failing where the environment is not provisioned.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from srbf.core import EvaluationSample

TOY_VARIABLES = ["x1", "x2"]
TOY_EXPRESSION = ["+", "-", "*", "2", "x1", "*", "0.5", "x2", "1"]
"""``2*x1 - 0.5*x2 + 1``, the toy problem's ground truth, in prefix notation."""


def _toy_target(rows: np.ndarray) -> np.ndarray:
    return (2.0 * rows[:, 0] - 0.5 * rows[:, 1] + 1.0).reshape(-1, 1)


def toy_sample(seed: int = 0, *, n_support: int = 48, n_validation: int = 12) -> EvaluationSample:
    """A two-variable linear problem with the metadata a catalog problem carries (arrays, ids, ground truth)."""
    from srbf.sample_metadata import build_base_metadata

    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, size=(n_support, 2))
    x_val = rng.uniform(-2.0, 2.0, size=(n_validation, 2))
    y, y_val = _toy_target(x), _toy_target(x_val)
    metadata = build_base_metadata(
        skeleton=TOY_EXPRESSION, expression=TOY_EXPRESSION, variables=TOY_VARIABLES,
        x_support=x, y_support=y, x_validation=x_val, y_validation=y_val,
        y_support_noisy=y, y_validation_noisy=y_val, noise_level=0.0,
        extra_fields={"benchmark_eq_id": f"toy-{seed}", "eval_row_index": seed, "gt_kind": "exact"},
    )
    return EvaluationSample(x_support=x, y_support=y, x_validation=x_val, y_validation=y_val, metadata=metadata)


def fit_once(
    *,
    config: str | Path | None = None,
    experiment: str | None = None,
    worker: str | Path | None = None,
    python: str | None = None,
    options: Mapping[str, Any] | None = None,
    engine: Any = "acj-4-3",
    sample: EvaluationSample | None = None,
    timeout: float = 600.0,
) -> dict[str, Any]:
    """Run one problem through a worker exactly as a benchmark run would; returns the result record.

    Give either ``config`` (a run config; the ``model_adapter`` of its first experiment, or of
    ``experiment``) or ``worker`` with ``python``, ``options`` and ``engine`` (an engine name or a
    loaded ``SimpliPyEngine``). ``options`` overrides the config's. The record carries the keys a
    run records: ``prediction_success``, ``predicted_expression``, ``y_pred_val``, ``error``, ...
    """
    from flash_ansr.utils.paths import substitute_root_path
    from simplipy import SimpliPyEngine

    from srbf.config import (
        _resolve_worker_ref,
        _subprocess_common_kwargs,
        extract_run_section,
        load_run_config,
        select_experiment,
    )
    from srbf.subprocess_adapter import SubprocessAdapter

    kwargs: dict[str, Any]
    if config is not None:
        raw = load_run_config(str(config))
        experiments = raw.get("experiments")
        chosen = experiment if experiment is not None else (next(iter(experiments)) if experiments else None)
        section = select_experiment(raw, chosen) if chosen is not None else dict(raw)
        model_cfg = extract_run_section(section).get("model_adapter") or {}
        if str(model_cfg.get("type", "")).lower() != "subprocess" or model_cfg.get("worker") is None:
            raise ValueError("fit_once(config=...) needs a worker-backed model_adapter (type: subprocess, worker: ...)")
        worker = _resolve_worker_ref(model_cfg["worker"])
        kwargs = _subprocess_common_kwargs(model_cfg)
        if options is not None:
            kwargs["options"] = dict(options)
        engine = model_cfg.get("simplipy_engine", engine)
    elif worker is not None:
        kwargs = dict(python=python, options=dict(options or {}), timeout=timeout)
    else:
        raise ValueError("fit_once needs config= or worker=")
    interpreter = str(kwargs.get("python") or sys.executable)
    if not Path(interpreter).exists():
        raise FileNotFoundError(
            f"interpreter not found: {interpreter} (create the method's environment there, or edit model_adapter.python)")
    kwargs["python"] = interpreter
    if kwargs.get("timeout") is None:
        kwargs["timeout"] = timeout
    loaded = SimpliPyEngine.load(substitute_root_path(str(engine)), install=True) if isinstance(engine, str) else engine
    adapter = SubprocessAdapter(worker=worker, simplipy_engine=loaded, **kwargs)
    try:
        adapter.prepare()
        return dict(adapter.evaluate_sample(sample or toy_sample()).to_mapping())
    finally:
        adapter.close()


__all__ = ["TOY_EXPRESSION", "TOY_VARIABLES", "fit_once", "toy_sample"]
