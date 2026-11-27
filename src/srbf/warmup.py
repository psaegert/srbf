"""Setup before the clock: srbf's warm-up problem.

The timing rule (docs/fairness.md, "How time is measured"): anything that does not depend on the data may happen
before the clock starts -- loading weights, compiling, warming up on a fixed synthetic problem. It happens once per
worker process and again after every worker restart. Everything after a problem's data arrives counts toward its fit
time. The same rule applies to every method.

This module holds the fixed problem srbf sends through a method's fit before the first timed problem when the run's
``runner.warmup`` is on, and the function that runs it on an in-process adapter. Its answer is discarded: it is never
judged, stored or counted, and its time goes into the run's ``__meta__['setup']``, never into a ``fit_time``. A
worker-backed adapter runs the same problem through its worker after every start (``SubprocessAdapter.warm_up``).
"""
from __future__ import annotations

import time
from typing import Any, Callable, Mapping

import numpy as np

from srbf.core import EvaluationSample

#: The seed the warm-up problem's points are drawn with: the problem is the same in every run, on every machine.
WARMUP_SEED = 0
#: The warm-up law, ``1.5*x1*sin(x2) + 0.5``, in prefix notation over the columns ``x1, x2``.
WARMUP_EXPRESSION = ("+", "*", "1.5", "*", "x1", "sin", "x2", "0.5")
WARMUP_VARIABLES = ("x1", "x2")
#: The sizes of the srbf suite's problems: 512 support and 512 validation points.
WARMUP_N_SUPPORT = 512
WARMUP_N_VALIDATION = 512
#: What the warm-up problem is called in its metadata (``benchmark_eq_id``, ``catalog``).
WARMUP_ID = "srbf-warmup"


def warmup_sample() -> EvaluationSample:
    """The warm-up problem: ``1.5*x1*sin(x2) + 0.5`` on points drawn uniformly from [-5, 5]^2 with :data:`WARMUP_SEED`,
    without noise. Its metadata says ``warmup: True``, names its two columns and carries its law (an oracle reads it),
    and has no ``eval_row_index``: it is no row of any run."""
    rng = np.random.default_rng(WARMUP_SEED)
    x = rng.uniform(-5.0, 5.0, size=(WARMUP_N_SUPPORT + WARMUP_N_VALIDATION, len(WARMUP_VARIABLES)))
    y = (1.5 * x[:, 0] * np.sin(x[:, 1]) + 0.5).reshape(-1, 1)
    metadata = {
        "warmup": True,
        "benchmark_eq_id": WARMUP_ID,
        "eq_id": WARMUP_ID,
        "catalog": WARMUP_ID,
        "variables": list(WARMUP_VARIABLES),
        "variable_names": list(WARMUP_VARIABLES),
        "n_support": WARMUP_N_SUPPORT,
        "noise_level": 0.0,
        "expression": list(WARMUP_EXPRESSION),
        "ground_truth_prefix": list(WARMUP_EXPRESSION),
    }
    return EvaluationSample(
        x_support=x[:WARMUP_N_SUPPORT], y_support=y[:WARMUP_N_SUPPORT],
        x_validation=x[WARMUP_N_SUPPORT:], y_validation=y[WARMUP_N_SUPPORT:], metadata=metadata)


def warmup_outcome(seconds: float, record: Mapping[str, Any] | None, error: str | None = None) -> dict[str, Any]:
    """The record of one warm-up: ``warmup_time`` (seconds around the fit, srbf's side included),
    ``warmup_fit_time`` (the fit time the method reported for it) and ``warmup_error`` (None, or why the method did
    not answer the warm-up problem: an exception, a reported error or no prediction)."""
    record = record or {}
    if error is None and not record.get("prediction_success", False):
        error = str(record.get("error") or "the method returned no prediction")
    fit_time = record.get("fit_time")
    return {"warmup_time": float(seconds), "warmup_fit_time": None if fit_time is None else float(fit_time),
            "warmup_error": error}


def run_warmup(evaluate: Callable[[EvaluationSample], Any], sample: EvaluationSample | None = None) -> dict[str, Any]:
    """Send the warm-up problem through ``evaluate`` (an in-process adapter's ``evaluate_sample``), discard the answer
    and return :func:`warmup_outcome`. An exception is the warm-up's error, not the run's."""
    sample = warmup_sample() if sample is None else sample
    started = time.perf_counter()
    try:
        result = evaluate(sample)
    except Exception as exc:  # noqa: BLE001 - reported in the run's metadata and as a warning by the caller
        return warmup_outcome(time.perf_counter() - started, None, f"{type(exc).__name__}: {exc}")
    seconds = time.perf_counter() - started
    record = result.to_mapping() if hasattr(result, "to_mapping") else result
    return warmup_outcome(seconds, record if isinstance(record, Mapping) else None)


__all__ = [
    "WARMUP_EXPRESSION", "WARMUP_ID", "WARMUP_N_SUPPORT", "WARMUP_N_VALIDATION", "WARMUP_SEED", "WARMUP_VARIABLES",
    "run_warmup", "warmup_outcome", "warmup_sample",
]
