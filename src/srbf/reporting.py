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


def draw_values(
    snapshot: Mapping[str, Sequence[Any]],
    metric_key: str,
    *,
    group_key: str = "benchmark_eq_id",
) -> dict[Any, np.ndarray]:
    """Per-expression arrays of PER-DRAW metric values (the uncollapsed sibling of
    :func:`draw_distribution`).

    Groups non-placeholder rows by ``group_key`` and returns each group's raw draw values as a
    float array (``None`` values skipped; empty groups omitted). This is the primitive under the
    paired layer and the noise-margin derivation, both of which need draws, not collapsed means.

    Unlike :func:`draw_distribution`, a MISSING ``group_key`` column raises ``KeyError`` instead
    of falling back to row identity: pairing joins on these keys, and a row-order join between
    two snapshots with non-seed-matched draws is meaningless-but-plausible — the exact silent
    failure the paired design must exclude.
    """
    if metric_key not in snapshot:
        raise KeyError(f"metric '{metric_key}' not in snapshot (columns: {sorted(snapshot.keys())})")
    if group_key not in snapshot:
        raise KeyError(
            f"group key '{group_key}' not in snapshot — refusing the row-identity fallback: "
            f"paired statistics must join on expression ids, never on row order")
    values = snapshot[metric_key]
    groups = snapshot[group_key]

    by_group: dict[Any, list[float]] = {}
    for i in _valid_rows(snapshot):
        value = values[i] if i < len(values) else None
        key = groups[i] if i < len(groups) else None
        if value is None or key is None:
            continue
        by_group.setdefault(key, []).append(float(value))
    return {key: np.asarray(vals, dtype=float) for key, vals in by_group.items() if vals}


def paired_expression_deltas(
    values_a: Mapping[Any, np.ndarray],
    values_b: Mapping[Any, np.ndarray],
    *,
    aggregate: Callable[[np.ndarray], float | np.ndarray] = np.nanmean,
) -> dict[str, Any]:
    """Join two per-expression value dicts on their KEYS and form per-expression deltas.

    For every expression id present in both inputs: ``delta = aggregate(a) - aggregate(b)``.
    ``aggregate`` collapses that expression's draws; it may return a scalar (the standard paired
    delta) or a 1-D profile (vector values stack to an ``(n_common, k)`` delta matrix, feeding
    :func:`srbf.metrics.bootstrap.bootstrap_band` for the WP2 profile difference).

    The join is BY ID — never positional — so a row-permuted snapshot yields identical deltas.
    Ids present on one side only are returned in the diagnostics, not silently dropped:
    ``n_only_a`` / ``n_only_b`` are the pairing-transparency counters every report must surface.

    Returns
    -------
    dict with keys ``keys`` (sorted common ids), ``deltas`` (``(n,)`` or ``(n, k)`` float array),
    ``n_pairs``, ``n_only_a``, ``n_only_b``, ``only_a``, ``only_b`` (sorted id lists).
    """
    common = sorted(set(values_a) & set(values_b), key=str)
    only_a = sorted(set(values_a) - set(values_b), key=str)
    only_b = sorted(set(values_b) - set(values_a), key=str)

    deltas = np.asarray(
        [np.asarray(aggregate(values_a[k]), dtype=float) - np.asarray(aggregate(values_b[k]), dtype=float)
         for k in common],
        dtype=float,
    ) if common else np.empty((0,), dtype=float)

    return {
        "keys": common,
        "deltas": deltas,
        "n_pairs": len(common),
        "n_only_a": len(only_a),
        "n_only_b": len(only_b),
        "only_a": only_a,
        "only_b": only_b,
    }


def bootstrap_report(
    snapshot: Mapping[str, Sequence[Any]],
    metric_key: str,
    *,
    group_key: str = "benchmark_eq_id",
    aggregate: Callable[[np.ndarray], float] = np.nanmean,
    reduce: Callable[[np.ndarray], float] = np.nanmean,
    n: int = 10_000,
    interval: float = 0.95,
    rng: np.random.Generator | int | None = 0,
) -> dict[str, Any]:
    """Bootstrap ``reduce`` over the per-expression metric distribution; return median + CI.

    Pipeline: ``draw_distribution`` (one value per ``group_key``) -> ``bootstrapped_metric_ci`` over
    those values. ``reduce`` is the statistic estimated (default the mean recovery across
    expressions); ``aggregate`` collapses draws within an expression. Returns
    ``{metric, n_groups, n_rows, median, ci_lower, ci_upper, interval}``. The bootstrap resampling
    is SEEDED by default (``rng=0``) so reports are bit-reproducible; pass ``rng=None`` for fresh
    entropy per call. Either way the estimate is the interval, not the point: interpret results by
    the CI, never by bit-equality of the point value between configurations.
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
    median, lower, upper = bootstrapped_metric_ci(arr, reduce, n=n, interval=interval, rng=rng)
    return {
        "metric": metric_key, "n_groups": int(arr.size), "n_rows": n_rows,
        "median": float(median), "ci_lower": float(lower), "ci_upper": float(upper),
        "interval": interval,
    }


__all__ = ["draw_distribution", "draw_values", "paired_expression_deltas", "bootstrap_report"]
