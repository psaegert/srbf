"""Helpers for constructing evaluation sample metadata."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


COMMON_PREDICTION_FIELDS: Mapping[str, Any] = {
    "length_penalty": None,
    "constants_penalty": None,
    "likelihood_penalty": None,
    "fit_time": None,
    "predicted_expression": None,
    "predicted_expression_prefix": None,
    "predicted_skeleton_prefix": None,
    "predicted_constants": None,
    "predicted_score": None,
    "predicted_log_prob": None,
    "y_pred": None,
    "y_pred_val": None,
    "prediction_success": False,
    "error": None,
    "placeholder": False,
    "placeholder_reason": None,
}


def _ensure_list(value: Sequence[str] | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    return list(value)


def build_base_metadata(
    *,
    skeleton: Sequence[str] | None,
    expression: Sequence[str] | None,
    variables: Sequence[str] | None,
    x_support: np.ndarray,
    y_support: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    y_support_noisy: np.ndarray,
    y_validation_noisy: np.ndarray,
    noise_level: float,
    skeleton_hash: Sequence[str] | None = None,
    labels_decoded: Sequence[str] | None = None,
    complexity: int | None = None,
    y_reference_support: np.ndarray | None = None,
    y_reference_validation: np.ndarray | None = None,
    extra_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a metadata dictionary with consistent core fields."""

    skeleton_list = _ensure_list(skeleton)
    expression_list = _ensure_list(expression)
    variable_list = _ensure_list(variables)
    skeleton_hash_value = tuple(_ensure_list(skeleton_hash) or (skeleton_list or ()))
    if not skeleton_hash_value:
        skeleton_hash_value = None  # type: ignore[assignment]

    metadata: dict[str, Any] = {
        "skeleton": skeleton_list,
        "skeleton_hash": skeleton_hash_value,
        "expression": expression_list,
        "constants": [],
        "variables": variable_list,
        "variable_names": variable_list.copy() if variable_list else None,
        "x": x_support.copy(),
        "y": y_support.copy(),
        "y_noisy": y_support_noisy.copy(),
        "x_val": x_validation.copy(),
        "y_val": y_validation.copy(),
        "y_noisy_val": y_validation_noisy.copy(),
        # Reference-law predictions on the same points (real-data catalogs, WP7). For synthetic
        # problems the generating expression IS the reference, so the default is the clean y:
        # reference_fvu is then exactly 0 and reference-relative recovery reduces to the
        # machine-precision criterion.
        "y_ref": (y_reference_support if y_reference_support is not None else y_support).copy(),
        "y_ref_val": (y_reference_validation if y_reference_validation is not None else y_validation).copy(),
        "n_support": int(x_support.shape[0]),
        "labels_decoded": list(labels_decoded) if labels_decoded is not None else None,
        "complexity": complexity,
        "noise_level": noise_level,
    }

    metadata.update(COMMON_PREDICTION_FIELDS)
    if extra_fields:
        metadata.update(extra_fields)
    return metadata
