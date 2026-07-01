"""Multi-draw reporting: group per-problem metrics by expression, bootstrap a CI.

The 0.5.x reproducibility story for SAMPLING sources (no seeding): a benchmark draws
``problems_per_expression`` problems per ground-truth expression, and we report the metric as a
DISTRIBUTION over expressions with a bootstrap confidence interval, rather than relying on a fixed
seed. ``draw_distribution`` collapses the per-draw rows to one value per expression (grouped by
``benchmark_eq_id``); ``bootstrap_report`` bootstraps a statistic (default: the mean) over those
per-expression values via :func:`srbf.metrics.bootstrap.bootstrapped_metric_ci`.

Operates on the plain dict-of-lists snapshot a ``Benchmark`` run returns (``ResultStore.snapshot()``):
no dependency on the (deferred) columnar store or a typed ``Result`` projection. Placeholder rows
(failed/exhausted draws) are dropped before aggregation.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

from srbf.metrics.bootstrap import bootstrapped_metric_ci


def _row_count(snapshot: Mapping[str, Sequence[Any]]) -> int:
    return max((len(v) for v in snapshot.values()), default=0)


def _valid_rows(snapshot: Mapping[str, Sequence[Any]]) -> list[int]:
    """Row indices that are NOT placeholders (failed/exhausted draws are excluded from stats)."""
    placeholders = snapshot.get("placeholder", [])
    return [
        i for i in range(_row_count(snapshot))
        if not (i < len(placeholders) and bool(placeholders[i]))
    ]


def draw_distribution(
    snapshot: Mapping[str, Sequence[Any]],
    metric_key: str,
    *,
    group_key: str = "benchmark_eq_id",
    aggregate: Callable[[np.ndarray], float] = np.nanmean,
) -> dict[Any, float]:
    """Collapse per-draw metric rows to ONE value per expression.

    Groups non-placeholder rows by ``group_key`` (default ``benchmark_eq_id``) and reduces each
    group's metric values with ``aggregate`` (default mean over the expression's draws). When no
    ``group_key`` column exists, each row is its own group (one draw per expression). ``None`` metric
    values are skipped; a group with no finite values is omitted.
    """
    return _draw_distribution(snapshot, metric_key, _valid_rows(snapshot), group_key=group_key, aggregate=aggregate)


def _draw_distribution(
    snapshot: Mapping[str, Sequence[Any]],
    metric_key: str,
    rows: Sequence[int],
    *,
    group_key: str,
    aggregate: Callable[[np.ndarray], float],
) -> dict[Any, float]:
    """``draw_distribution`` body over a PRECOMPUTED valid-row index list, so ``bootstrap_report``
    can share one ``_valid_rows`` scan for both the distribution and its ``n_rows`` count."""
    if metric_key not in snapshot:
        raise KeyError(f"metric '{metric_key}' not in snapshot (columns: {sorted(snapshot.keys())})")
    values = snapshot[metric_key]
    groups = snapshot.get(group_key)

    by_group: dict[Any, list[float]] = {}
    for i in rows:
        value = values[i] if i < len(values) else None
        if value is None:
            continue
        key = groups[i] if (groups is not None and i < len(groups) and groups[i] is not None) else i
        by_group.setdefault(key, []).append(float(value))

    out: dict[Any, float] = {}
    for key, vals in by_group.items():
        reduced = float(aggregate(np.asarray(vals, dtype=float)))
        if np.isfinite(reduced):
            out[key] = reduced
    return out


def bootstrap_report(
    snapshot: Mapping[str, Sequence[Any]],
    metric_key: str,
    *,
    group_key: str = "benchmark_eq_id",
    aggregate: Callable[[np.ndarray], float] = np.nanmean,
    reduce: Callable[[np.ndarray], float] = np.nanmean,
    n: int = 10_000,
    interval: float = 0.95,
) -> dict[str, Any]:
    """Bootstrap ``reduce`` over the per-expression metric distribution; return median + CI.

    Pipeline: ``draw_distribution`` (one value per ``group_key``) -> ``bootstrapped_metric_ci`` over
    those values. ``reduce`` is the statistic estimated (default the mean recovery across
    expressions); ``aggregate`` collapses draws within an expression. Returns
    ``{metric, n_groups, n_rows, median, ci_lower, ci_upper, interval}``. The bootstrap is UNSEEDED
    (per the no-seeding policy), so CIs vary run-to-run; report the interval, not point bit-equality.
    """
    valid_rows = _valid_rows(snapshot)   # single scan, reused for the distribution and n_rows
    dist = _draw_distribution(snapshot, metric_key, valid_rows, group_key=group_key, aggregate=aggregate)
    arr = np.asarray(list(dist.values()), dtype=float)
    n_rows = len(valid_rows)
    if arr.size == 0:
        return {
            "metric": metric_key, "n_groups": 0, "n_rows": n_rows,
            "median": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan"),
            "interval": interval,
        }
    median, lower, upper = bootstrapped_metric_ci(arr, reduce, n=n, interval=interval)
    return {
        "metric": metric_key, "n_groups": int(arr.size), "n_rows": n_rows,
        "median": float(median), "ci_lower": float(lower), "ci_upper": float(upper),
        "interval": interval,
    }


__all__ = ["draw_distribution", "bootstrap_report"]
