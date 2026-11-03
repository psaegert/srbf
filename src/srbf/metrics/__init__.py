"""Evaluation metric helpers for srbf, carved from flash-ansr."""

from srbf.metrics.bootstrap import bootstrapped_metric_ci
from srbf.metrics.improvability import ERROR_FLOOR, improvability, pool_best, validation_error
from srbf.metrics.numeric import (
    fvu,
    fvu_exact,
    is_perfect_fit,
    log10_fvu, r2,
    naninfmean,
    safe_divide,
)
from srbf.metrics.paired import (
    McNemarResult,
    mcnemar_exact,
    paired_difference_ci,
    wilson_interval,
)
from srbf.metrics.symbolic import total_nestedness
from srbf.metrics.zss import build_tree, zss_tree_edit_distance

__all__ = [
    "ERROR_FLOOR",
    "McNemarResult",
    "bootstrapped_metric_ci",
    "build_tree",
    "fvu",
    "fvu_exact",
    "improvability",
    "is_perfect_fit",
    "log10_fvu",
    "mcnemar_exact",
    "paired_difference_ci",
    "pool_best",
    "naninfmean",
    "safe_divide",
    "total_nestedness",
    "validation_error",
    "wilson_interval",
    "zss_tree_edit_distance",
]
