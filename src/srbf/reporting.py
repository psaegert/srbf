"""Multi-draw reporting: group per-problem metrics by expression, bootstrap a CI; paired layer.

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
from srbf.provenance import META_KEY


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


class PairingContractError(ValueError):
    """Two snapshots may not be paired: provenance mismatch, missing metadata, or a broken join."""


def resolve_group_key(snapshot: Mapping[str, Sequence[Any]]) -> str:
    """Expression-id column of a snapshot: ``benchmark_eq_id`` (curated benchmarks), falling back
    to ``skeleton_hash`` for older/generative snapshots (v23-val), where the ground-truth skeleton
    tuple IS the stable expression identity. Raises if neither exists — never row order."""
    for key in ("benchmark_eq_id", "skeleton_hash"):
        column = snapshot.get(key)
        if column is not None and any(v is not None for v in column):
            return key
    raise PairingContractError(
        "snapshot has neither 'benchmark_eq_id' nor 'skeleton_hash' — cannot pair on expression ids")


def pairing_fingerprint(snapshot: Mapping[str, Sequence[Any]]) -> dict[str, Any] | None:
    """Benchmark-identity fingerprint from a snapshot's embedded ``__meta__`` provenance.

    Returns the sha256 of every provenance INPUT whose name marks it as benchmark/catalog data
    (model weights legitimately differ between the two sides of a pair; the benchmark inputs
    must not). ``None`` when the snapshot predates embedded provenance — the caller decides
    whether unverified pairing is acceptable (``allow_unverified``)."""
    meta = snapshot.get(META_KEY) if hasattr(snapshot, "get") else None
    if not isinstance(meta, dict):
        return None
    inputs = meta.get("inputs") or {}
    marks = ("benchmark", "catalog", "dataset", "test_set", "val")
    data_inputs = {
        name: (info or {}).get("sha256")
        for name, info in inputs.items()
        if any(mark in name.lower() for mark in marks)
    }
    return {"data_inputs": data_inputs, "timestamp": meta.get("timestamp")}


def _verify_pairing(
    snapshot_a: Mapping[str, Sequence[Any]],
    snapshot_b: Mapping[str, Sequence[Any]],
    *,
    allow_unverified: bool,
) -> dict[str, Any]:
    """Layer 1 of the pairing contract: benchmark-identity provenance must match."""
    fp_a, fp_b = pairing_fingerprint(snapshot_a), pairing_fingerprint(snapshot_b)
    if fp_a is None or fp_b is None:
        if not allow_unverified:
            raise PairingContractError(
                "snapshot(s) carry no embedded provenance (__meta__) — pairing cannot be verified. "
                "Pass allow_unverified=True to pair anyway (archived pre-provenance snapshots).")
        return {"verified": False, "reason": "missing __meta__ provenance"}
    shared = set(fp_a["data_inputs"]) & set(fp_b["data_inputs"])
    mismatched = sorted(name for name in shared if fp_a["data_inputs"][name] != fp_b["data_inputs"][name])
    if mismatched:
        raise PairingContractError(
            f"benchmark data inputs differ between snapshots: {mismatched} — these are not the "
            f"same benchmark; pairing would compare different populations.")
    return {"verified": True, "shared_data_inputs": sorted(shared)}


def paired_report(
    snapshot_a: Mapping[str, Sequence[Any]],
    snapshot_b: Mapping[str, Sequence[Any]],
    metric_key: str,
    *,
    group_key: str | None = None,
    aggregate: Callable[[np.ndarray], float] = np.nanmean,
    n: int = 10_000,
    interval: float = 0.95,
    rng: np.random.Generator | int | None = 0,
    higher_is_better: bool = True,
    margin: float | Mapping[str, float] | None = None,
    zero_method: str = "pratt",
    expect_total: bool = False,
    allow_unverified: bool = False,
) -> dict[str, Any]:
    """Paired comparison of two SAME-BENCHMARK snapshots: per-expression deltas, bootstrap CI,
    effect sizes, power honesty, and (given a measurement-noise ``margin``) a four-state verdict.

    Per expression e (joined on ids, never row order): ``delta_e = aggregate(A_e) - aggregate(B_e)``
    over each side's draws. All statistics are computed from ONE expression-bootstrap pass.
    Positive deltas favor A when ``higher_is_better`` (flip labels otherwise). Marginal CIs are
    never combined or subtracted — that is the point of this function.

    Missing-data semantics follow the metric's two-regime column encoding: RATE metrics carry
    failures as 0.0 (total columns — pass ``expect_total=True`` to make any one-sided expression
    id a contract error); DIAGNOSTIC metrics carry failures as ``None`` and pair over the
    both-have-values intersection — a conditional-on-both-succeed estimand, disclosed via
    ``n_only_a`` / ``n_only_b`` in the output (never silently).

    ``margin`` is the pair-specific measurement-noise margin m_AB (:func:`pair_margin`; pass the
    dict or the float). Verdicts: ``better`` = CI entirely beyond +margin (in A's favor),
    ``worse`` = mirrored, ``equivalent`` = CI entirely inside [-margin, +margin] (a difference
    smaller than the benchmark can measure), ``undecided`` = anything else.
    ``equivalence_attainable`` reports whether the CI is even narrow enough for ``equivalent``
    to be reachable — when False, an 'undecided' is resolution-limited, not evidence of parity.
    """
    contract = _verify_pairing(snapshot_a, snapshot_b, allow_unverified=allow_unverified)

    key_a = group_key or resolve_group_key(snapshot_a)
    key_b = group_key or resolve_group_key(snapshot_b)
    if key_a != key_b:
        raise PairingContractError(f"group-key mismatch: {key_a!r} vs {key_b!r}")

    values_a = draw_values(snapshot_a, metric_key, group_key=key_a)
    values_b = draw_values(snapshot_b, metric_key, group_key=key_b)
    pairs = paired_expression_deltas(values_a, values_b, aggregate=aggregate)
    if expect_total and (pairs["n_only_a"] or pairs["n_only_b"]):
        raise PairingContractError(
            f"metric '{metric_key}' declared total (rate metric) but the expression-id join is "
            f"asymmetric: only_a={pairs['only_a'][:5]}..., only_b={pairs['only_b'][:5]}... — "
            f"a rate metric with missing ids means the snapshots cover different populations.")

    deltas = np.asarray(pairs["deltas"], dtype=float)
    n_pairs = int(deltas.shape[0])
    report: dict[str, Any] = {
        "metric": metric_key,
        "group_key": key_a,
        "n_pairs": n_pairs,
        "n_only_a": pairs["n_only_a"],
        "n_only_b": pairs["n_only_b"],
        "only_a": pairs["only_a"],
        "only_b": pairs["only_b"],
        "interval": interval,
        "n_bootstrap": int(n),
        "higher_is_better": higher_is_better,
        "pairing": contract,
    }
    if n_pairs == 0:
        report.update({k: float("nan") for k in
                       ("delta_mean", "ci_lower", "ci_upper", "delta_median", "mde_80")})
        report.update({"win_rate": None, "prob_superiority": float("nan"), "wilcoxon": None,
                       "verdict": None, "margin": None, "equivalence_attainable": None,
                       "variance_decomposition": None})
        return report

    # One expression-resampling pass serves every bootstrap statistic (house convention:
    # point estimate = median of the bootstrap distribution; report rounding handles wobble).
    if not hasattr(rng, "integers"):
        rng = np.random.default_rng(rng)
    indices = rng.integers(0, n_pairs, size=(int(n), n_pairs))
    resampled = deltas[indices]                                   # (n, n_pairs)
    boot_means = np.nanmean(resampled, axis=1)
    boot_medians = np.nanmedian(resampled, axis=1)
    lo_q, hi_q = (1 - interval) / 2 * 100, (1 + interval) / 2 * 100

    report["delta_mean"] = float(np.nanmedian(boot_means))
    report["ci_lower"] = float(np.nanpercentile(boot_means, lo_q))
    report["ci_upper"] = float(np.nanpercentile(boot_means, hi_q))
    report["delta_median"] = float(np.nanmedian(boot_medians))
    report["median_ci_lower"] = float(np.nanpercentile(boot_medians, lo_q))
    report["median_ci_upper"] = float(np.nanpercentile(boot_medians, hi_q))

    # Effect sizes: per-expression win/tie/loss and the probability of superiority (ties half).
    n_a_better = int(np.sum(deltas > 0)) if higher_is_better else int(np.sum(deltas < 0))
    n_b_better = int(np.sum(deltas < 0)) if higher_is_better else int(np.sum(deltas > 0))
    n_tied = int(np.sum(deltas == 0))
    report["win_rate"] = {"a_better": n_a_better, "b_better": n_b_better, "tied": n_tied}
    signed = deltas if higher_is_better else -deltas
    report["prob_superiority"] = float((np.sum(signed > 0) + 0.5 * n_tied) / n_pairs)

    # Wilcoxon signed-rank companion, zeros COUNTED (pratt) — scipy's default drops them, which
    # on rate metrics silently conditions the test on the discordant minority.
    from scipy.stats import wilcoxon  # scipy is a declared dependency
    nonzero = int(np.sum(deltas != 0))
    if nonzero == 0:
        report["wilcoxon"] = {"statistic": float("nan"), "p": 1.0, "n_zero": n_tied,
                              "n_nonzero": 0, "zero_method": zero_method}
    else:
        w = wilcoxon(deltas, zero_method=zero_method, method="approx")
        report["wilcoxon"] = {"statistic": float(w.statistic), "p": float(w.pvalue),
                              "n_zero": n_tied, "n_nonzero": nonzero, "zero_method": zero_method}

    # Power honesty: the mean-delta magnitude detectable at 80% power (two-sided alpha from
    # `interval`) — printed next to every verdict so "undecided" is never read as "equal".
    se = float(np.nanstd(deltas, ddof=1) / np.sqrt(n_pairs)) if n_pairs > 1 else float("nan")
    report["mde_80"] = float((1.959963985 + 0.8416212336) * se)

    # Draws-vs-expressions variance decomposition (WP7's "grow expressions or draws?" observable).
    within = [
        float(np.nanvar(values_a[k], ddof=1)) / len(values_a[k])
        + float(np.nanvar(values_b[k], ddof=1)) / len(values_b[k])
        for k in pairs["keys"]
        if len(values_a[k]) >= 2 and len(values_b[k]) >= 2
    ]
    between = float(np.nanvar(deltas, ddof=1)) if n_pairs > 1 else float("nan")
    mean_within = float(np.mean(within)) if within else float("nan")
    report["variance_decomposition"] = {
        "between_expression_var": between,
        "mean_within_expression_var": mean_within,
        "draw_noise_share": (mean_within / between) if between and np.isfinite(between) and between > 0
        else float("nan"),
    }

    # Verdict vs the pair-specific measurement-noise margin.
    m = margin.get("margin") if isinstance(margin, Mapping) else margin
    report["margin"] = float(m) if m is not None else None
    if m is None:
        report["verdict"] = None
        report["equivalence_attainable"] = None
    else:
        lo, hi = report["ci_lower"], report["ci_upper"]
        if not higher_is_better:
            lo, hi = -hi, -lo  # orient so positive favors A
        if lo > m:
            verdict = "better"
        elif hi < -m:
            verdict = "worse"
        elif -m <= lo and hi <= m:
            verdict = "equivalent"
        else:
            verdict = "undecided"
        report["verdict"] = verdict
        report["equivalence_attainable"] = bool((hi - lo) / 2 <= m)
    return report


def self_noise(
    values: Mapping[Any, np.ndarray],
    *,
    n_null: int = 1000,
    reduce: Callable[..., float | np.ndarray] = np.nanmean,
    method: str = "split-half",
    rng: np.random.Generator | int | None = None,
) -> dict[str, Any]:
    """A series' draw-noise contribution to an aggregate paired delta (its NOISE NULL).

    In a real paired comparison, ``delta_hat = reduce_e(mean(A_e) - mean(B_e))``; under the
    per-expression null its distribution is that of ``N_A - N_B`` where ``N_X`` is the aggregate
    draw-noise of series X alone at its OWN per-expression draw counts. This function estimates
    ``N_X`` from one series' draws; :func:`pair_margin` then combines two such nulls into the
    measurement-noise margin (MRD) for a specific pair.

    Methods (both scale-exact per expression; they must agree unless draws are not i.i.d.):
    - ``'split-half'``: for each null sample, randomly split each expression's k draws into
      halves, take the half-mean difference, and rescale by the exact per-expression factor
      ``sqrt((1/k) / (1/k1 + 1/k2))`` to the full-k noise scale (``1/2`` for even k).
    - ``'bootstrap'``: centered draw resampling per expression, rescaled by
      ``sqrt(k/(k-1))`` to correct the bootstrap's small-k variance deflation.

    Expressions with fewer than 2 draws carry no noise information and are skipped (counted in
    ``n_skipped``). Returns ``{null, sd, q95, n_expressions, n_skipped, method}`` where ``null``
    is the ``(n_null,)`` sample of ``N_X`` used by :func:`pair_margin`.
    """
    if method not in ("split-half", "bootstrap"):
        raise ValueError(f"unknown method {method!r}")
    if not hasattr(rng, "integers"):
        rng = np.random.default_rng(rng)

    usable = {k: np.asarray(v, dtype=float) for k, v in values.items() if len(v) >= 2}
    n_skipped = len(values) - len(usable)
    if not usable:
        return {"null": np.empty((0,)), "sd": float("nan"), "q95": float("nan"),
                "n_expressions": 0, "n_skipped": n_skipped, "method": method}

    null = np.empty(n_null, dtype=float)
    keys = list(usable.keys())
    for s in range(n_null):
        contributions = np.empty(len(keys), dtype=float)
        for j, key in enumerate(keys):
            draws = usable[key]
            k = len(draws)
            if method == "split-half":
                perm = rng.permutation(k)
                k1 = k // 2
                half_delta = float(np.mean(draws[perm[:k1]]) - np.mean(draws[perm[k1:]]))
                scale = np.sqrt((1.0 / k) / (1.0 / k1 + 1.0 / (k - k1)))
                contributions[j] = half_delta * scale
            else:  # bootstrap
                resampled = draws[rng.integers(0, k, size=k)]
                centered = float(np.mean(resampled) - np.mean(draws))
                contributions[j] = centered * np.sqrt(k / (k - 1.0))
        null[s] = float(reduce(contributions))

    return {
        "null": null,
        "sd": float(np.std(null)),
        "q95": float(np.percentile(np.abs(null), 95)),
        "n_expressions": len(keys),
        "n_skipped": n_skipped,
        "method": method,
    }


def pair_margin(
    noise_a: Mapping[str, Any],
    noise_b: Mapping[str, Any],
    *,
    level: float = 0.95,
    n_pairs: int = 100_000,
    rng: np.random.Generator | int | None = None,
) -> dict[str, float]:
    """Measurement-noise margin (MRD) for a SPECIFIC pair of series.

    Convolves the two stored noise nulls by sampling (``N_A - N_B`` with independent draws from
    each null) and returns the ``level``-quantile of the magnitude: the largest aggregate delta
    that is indistinguishable from re-running the benchmark on two equally-good models with
    these two series' noise levels. Pair-specific by construction — a global max-over-models
    margin either starves quiet pairs of attainable 'equivalent' verdicts or dilutes the
    regression gate (referee round, 2026-07-02).

    Returns ``{margin, sd}``; ``sd`` is the combined null SD for diagnostics (``margin/sd`` far
    from ~1.96 signals heavy tails).
    """
    a, b = np.asarray(noise_a["null"], dtype=float), np.asarray(noise_b["null"], dtype=float)
    if a.size == 0 or b.size == 0:
        return {"margin": float("nan"), "sd": float("nan")}
    if not hasattr(rng, "integers"):
        rng = np.random.default_rng(rng)
    diff = a[rng.integers(0, a.size, size=n_pairs)] - b[rng.integers(0, b.size, size=n_pairs)]
    return {"margin": float(np.percentile(np.abs(diff), level * 100)), "sd": float(np.std(diff))}


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


__all__ = ["draw_distribution", "draw_values", "paired_expression_deltas", "paired_report",
           "self_noise", "pair_margin", "resolve_group_key", "pairing_fingerprint",
           "PairingContractError", "bootstrap_report"]
