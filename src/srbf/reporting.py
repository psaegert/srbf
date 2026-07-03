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
    worst_rank: bool = False,
    hierarchical: bool = False,
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

    ``worst_rank=True`` additionally reports the RANK statistics (median delta, win/tie/loss,
    prob_superiority, Wilcoxon) over the UNION of expression ids, imputing one-sided failures as
    sign-only sentinels ("worse than every observed value" — worst-rank composite scoring, cf.
    Lachin 1999). Sound for ranks only, never applied to the mean. Reported medians/CI bounds
    may be ±inf when they land on a sentinel; with >= 50% imputed pairs the block is flagged
    ``degenerate`` and its estimates suppressed (read the success rate instead).

    ``hierarchical=True`` switches the rank statistics' CIs (median, prob_superiority) to a
    two-stage bootstrap (resample expressions, then draws within each) so draw-level uncertainty
    is propagated where collapsing attenuates it. The mean-delta CI keeps collapse-first
    resampling (exactly the cluster bootstrap for the mean).
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
                       ("delta_mean", "ci_lower", "ci_upper", "delta_median",
                        "median_ci_lower", "median_ci_upper", "mde_80", "p_value")})
        report.update({"win_rate": None, "prob_superiority": float("nan"),
                       "prob_superiority_ci": None, "wilcoxon": None, "verdict": None,
                       "margin": None, "equivalence_attainable": None,
                       "variance_decomposition": None,
                       "rank_ci_method": "hierarchical" if hierarchical else "expression-bootstrap"})
        if worst_rank:
            report["worst_rank"] = {"n_imputed_a": pairs["n_only_a"],
                                    "n_imputed_b": pairs["n_only_b"],
                                    "n_union": pairs["n_only_a"] + pairs["n_only_b"],
                                    "degenerate": True, "median_delta": None, "median_ci": None,
                                    "win_rate": None, "prob_superiority": None, "wilcoxon": None,
                                    "note": "no common expressions — nothing to pair"}
        return report

    # One expression-resampling pass serves every bootstrap statistic (house convention:
    # point estimate = median of the bootstrap distribution; report rounding handles wobble).
    if not hasattr(rng, "integers"):
        rng = np.random.default_rng(rng)
    indices = rng.integers(0, n_pairs, size=(int(n), n_pairs))
    resampled = deltas[indices]                                   # (n, n_pairs)
    boot_means = np.nanmean(resampled, axis=1)
    lo_q, hi_q = (1 - interval) / 2 * 100, (1 + interval) / 2 * 100

    report["delta_mean"] = float(np.nanmedian(boot_means))
    report["ci_lower"] = float(np.nanpercentile(boot_means, lo_q))
    report["ci_upper"] = float(np.nanpercentile(boot_means, hi_q))
    # The gate's p: two-sided bootstrap percentile-inversion test on the MEAN delta (the
    # estimand-matched primary procedure; Wilcoxon below is the robustness companion, never
    # the gate). Floored at 1/(n+1): a bootstrap cannot certify smaller p than its resolution.
    p_low = float(np.mean(boot_means <= 0.0))
    p_high = float(np.mean(boot_means >= 0.0))
    report["p_value"] = float(max(2.0 * min(p_low, p_high), 1.0 / (int(n) + 1)))

    # Effect sizes: per-expression win/tie/loss and the probability of superiority (ties half).
    n_a_better = int(np.sum(deltas > 0)) if higher_is_better else int(np.sum(deltas < 0))
    n_b_better = int(np.sum(deltas < 0)) if higher_is_better else int(np.sum(deltas > 0))
    n_tied = int(np.sum(deltas == 0))
    report["win_rate"] = {"a_better": n_a_better, "b_better": n_b_better, "tied": n_tied}
    signed = deltas if higher_is_better else -deltas
    report["prob_superiority"] = float((np.sum(signed > 0) + 0.5 * n_tied) / n_pairs)

    # P(superiority) CI + median CI: plain expression resampling by default; hierarchical=True
    # propagates draw-level noise via a two-stage scheme (expressions, then draws within — the
    # draw stage sampled from a pregenerated per-expression pool, the standard bootstrap-pool
    # construction) for the rank statistics ONLY (the mean keeps collapse-first: exact).
    if hierarchical:
        pool_m = 256
        pools = np.empty((n_pairs, pool_m))
        for i, k in enumerate(pairs["keys"]):
            a_draws, b_draws = values_a[k], values_b[k]
            a_idx = rng.integers(0, len(a_draws), size=(pool_m, len(a_draws)))
            b_idx = rng.integers(0, len(b_draws), size=(pool_m, len(b_draws)))
            pools[i] = (np.asarray([aggregate(a_draws[j]) for j in a_idx])
                        - np.asarray([aggregate(b_draws[j]) for j in b_idx]))
        pool_pick = rng.integers(0, pool_m, size=(int(n), n_pairs))
        rank_resampled = pools[indices, pool_pick]           # (n, n_pairs), two-stage
    else:
        rank_resampled = resampled
    signed_resampled = rank_resampled if higher_is_better else -rank_resampled
    psup_boot = (np.sum(signed_resampled > 0, axis=1)
                 + 0.5 * np.sum(signed_resampled == 0, axis=1)) / n_pairs
    report["prob_superiority_ci"] = (float(np.nanpercentile(psup_boot, lo_q)),
                                     float(np.nanpercentile(psup_boot, hi_q)))
    rank_medians = np.nanmedian(rank_resampled, axis=1)
    report["delta_median"] = float(np.nanmedian(rank_medians))
    report["median_ci_lower"] = float(np.nanpercentile(rank_medians, lo_q))
    report["median_ci_upper"] = float(np.nanpercentile(rank_medians, hi_q))
    report["rank_ci_method"] = "hierarchical" if hierarchical else "expression-bootstrap"

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

    # Worst-rank composite scoring over the UNION of ids: one-sided failures become sign-only
    # sentinels ("worse than every observed value"). Rank statistics only — never the mean.
    if worst_rank:
        n_imputed = pairs["n_only_a"] + pairs["n_only_b"]
        n_union = n_pairs + n_imputed
        # signed space: positive favors A; a side that produced values beats a side that failed
        signed_union = np.concatenate([
            signed,
            np.full(pairs["n_only_a"], np.inf),
            np.full(pairs["n_only_b"], -np.inf),
        ])
        degenerate = n_union == 0 or (n_imputed / n_union) >= 0.5
        block: dict[str, Any] = {
            "n_imputed_a": pairs["n_only_a"], "n_imputed_b": pairs["n_only_b"],
            "n_union": n_union, "degenerate": bool(degenerate),
        }
        if degenerate:
            block.update({"median_delta": None, "median_ci": None, "win_rate": None,
                          "prob_superiority": None, "wilcoxon": None,
                          "note": ">=50% imputed pairs — rank statistics are the sentinel; "
                                  "read the success-rate metric instead"})
        else:
            u_idx = rng.integers(0, n_union, size=(int(n), n_union))
            u_medians = np.median(signed_union[u_idx], axis=1)  # ±inf-tolerant
            sign = 1.0 if higher_is_better else -1.0
            block["median_delta"] = float(sign * np.median(u_medians))
            m_lo, m_hi = np.percentile(u_medians, [lo_q, hi_q])
            block["median_ci"] = tuple(sorted((float(sign * m_lo), float(sign * m_hi))))
            block["win_rate"] = {
                "a_better": int(np.sum(signed_union > 0)),
                "b_better": int(np.sum(signed_union < 0)),
                "tied": int(np.sum(signed_union == 0)),
            }
            block["prob_superiority"] = float(
                (np.sum(signed_union > 0) + 0.5 * np.sum(signed_union == 0)) / n_union)
            finite = np.abs(signed_union[np.isfinite(signed_union)])
            top = float(finite.max()) * 1.000001 + 1.0 if finite.size else 1.0
            ranked = np.clip(signed_union, -top, top)  # sentinels tie at the top |delta| rank
            if np.any(ranked != 0):
                w = wilcoxon(ranked, zero_method=zero_method, method="approx")
                block["wilcoxon"] = {"statistic": float(w.statistic), "p": float(w.pvalue),
                                     "n_zero": int(np.sum(ranked == 0)),
                                     "n_nonzero": int(np.sum(ranked != 0)),
                                     "zero_method": zero_method}
            else:
                block["wilcoxon"] = {"statistic": float("nan"), "p": 1.0,
                                     "n_zero": n_union, "n_nonzero": 0,
                                     "zero_method": zero_method}
        report["worst_rank"] = block

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


def series_x_positions(
    snapshots: Mapping[int, Mapping[str, Sequence[Any]]],
    *,
    time_key: str = "fit_time",
) -> dict[int, float]:
    """Compute-axis position of every rung of a series: the median wall-clock ``fit_time`` over
    the rung's valid rows (the site's convention — the axis is a per-rung summary, so a Δ(t)
    curve compares configurations whose MEDIAN cost is t, not per-expression equal budgets)."""
    positions: dict[int, float] = {}
    for rung, snapshot in snapshots.items():
        times = [
            float(snapshot[time_key][i])
            for i in _valid_rows(snapshot)
            if i < len(snapshot.get(time_key, [])) and snapshot[time_key][i] is not None
        ]
        if times:
            positions[rung] = float(np.median(times))
    return positions


def _expression_matrix(
    snapshots: Mapping[int, Mapping[str, Sequence[Any]]],
    metric_key: str,
    group_key: str,
    aggregate: Callable[[np.ndarray], float],
) -> tuple[list[Any], list[int], np.ndarray]:
    """(expression x rung) matrix of per-expression aggregates for one series; NaN = no value."""
    per_rung = {rung: draw_values(snap, metric_key, group_key=group_key)
                for rung, snap in snapshots.items()}
    keys = sorted({k for values in per_rung.values() for k in values}, key=str)
    rungs = sorted(per_rung)
    matrix = np.full((len(keys), len(rungs)), np.nan)
    for j, rung in enumerate(rungs):
        values = per_rung[rung]
        for i, key in enumerate(keys):
            if key in values:
                matrix[i, j] = float(aggregate(values[key]))
    return keys, rungs, matrix


def _interpolate_series(
    matrix: np.ndarray,
    x_by_rung: list[float],
    t: float,
) -> tuple[np.ndarray, tuple[int, int], bool]:
    """Per-expression values of one series at time t: linear in log10(t) between the bracketing
    measured rungs. An expression gets a value ONLY if it is valid at BOTH bracketing rungs
    (composition-drift guard); at a measured rung the bracket is that rung itself. Assumes
    x_by_rung sorted ascending. Returns (values, (j_lo, j_hi), measured_exactly)."""
    x = np.asarray(x_by_rung, dtype=float)
    exact = np.nonzero(np.isclose(x, t, rtol=1e-9))[0]
    if exact.size:
        j = int(exact[0])
        return matrix[:, j], (j, j), True
    j_hi = int(np.searchsorted(x, t))
    j_lo = j_hi - 1
    if j_lo < 0 or j_hi >= len(x):
        raise ValueError(f"t={t} outside the measured range [{x[0]}, {x[-1]}]")
    w = (np.log10(t) - np.log10(x[j_lo])) / (np.log10(x[j_hi]) - np.log10(x[j_lo]))
    values = (1.0 - w) * matrix[:, j_lo] + w * matrix[:, j_hi]  # NaN if either bracket is NaN
    return values, (j_lo, j_hi), False


def paired_delta_curve(
    series_a: Mapping[int, Mapping[str, Sequence[Any]]],
    series_b: Mapping[int, Mapping[str, Sequence[Any]]],
    metric_key: str,
    *,
    x_policy: str = "time",
    group_key: str | None = None,
    aggregate: Callable[[np.ndarray], float] = np.nanmean,
    n: int = 10_000,
    interval: float = 0.95,
    rng: np.random.Generator | int | None = 0,
    allow_unverified: bool = False,
) -> dict[str, Any]:
    """Paired Δ curve of two series over the compute axis, under a declared ``x_policy``.

    ``series_*`` map rung value -> snapshot (one snapshot per measured rung).

    - ``x_policy='rung'`` (same-method variants): Δ is computed ONLY at shared rung values —
      exact configuration match, no interpolation. The natural policy when both series ran the
      same ladder (ablation vs parent, size ladder, version vs predecessor).
    - ``x_policy='time'`` (any pair): Δ(t) on the union grid of both series' measured median-time
      positions within the OVERLAP window, with per-expression LINEAR interpolation in log10-time
      between bracketing rungs. Never extrapolates: grid points outside either series' measured
      span are returned in ``out_of_range`` (loud), not silently dropped. An expression
      contributes at t only if valid at both bracketing rungs of BOTH series; per-point ``n_pairs``
      makes composition drift visible. Bands are POINTWISE ``interval`` bootstrap bands over
      expressions (label them as such).

    Each point carries ``(rung_a, rung_b, x_a, x_b, measured_a, measured_b)`` — the display marks
    measured points (dots) vs interpolated segments (line). Statistics are conditional on the
    measured rung placements (x treated as fixed design points).
    """
    first_a = next(iter(series_a.values()))
    first_b = next(iter(series_b.values()))
    contract = _verify_pairing(first_a, first_b, allow_unverified=allow_unverified)
    key = group_key or resolve_group_key(first_a)

    keys_a, rungs_a, matrix_a = _expression_matrix(series_a, metric_key, key, aggregate)
    keys_b, rungs_b, matrix_b = _expression_matrix(series_b, metric_key, key, aggregate)
    common = sorted(set(keys_a) & set(keys_b), key=str)
    index_a = [keys_a.index(k) for k in common]
    index_b = [keys_b.index(k) for k in common]
    matrix_a, matrix_b = matrix_a[index_a], matrix_b[index_b]

    if not hasattr(rng, "integers"):
        rng = np.random.default_rng(rng)

    def point_stats(deltas: np.ndarray) -> tuple[float, float, float, int]:
        valid = deltas[np.isfinite(deltas)]
        if valid.size == 0:
            return float("nan"), float("nan"), float("nan"), 0
        est, lo, hi = _bootstrap_triple(valid, n=n, interval=interval, rng=rng)
        return est, lo, hi, int(valid.size)

    points: list[dict[str, Any]] = []
    out_of_range: dict[str, list[float]] = {"a": [], "b": []}

    if x_policy == "rung":
        x_a = series_x_positions(series_a)
        x_b = series_x_positions(series_b)
        for rung in sorted(set(rungs_a) & set(rungs_b)):
            j_a, j_b = rungs_a.index(rung), rungs_b.index(rung)
            deltas = matrix_a[:, j_a] - matrix_b[:, j_b]
            est, lo, hi, n_pairs = point_stats(deltas)
            points.append({
                "x": x_a.get(rung), "rung_a": rung, "rung_b": rung,
                "x_a": x_a.get(rung), "x_b": x_b.get(rung),
                "measured_a": True, "measured_b": True,
                "delta": est, "lo": lo, "hi": hi, "n_pairs": n_pairs,
            })
    elif x_policy == "time":
        pos_a = series_x_positions(series_a)
        pos_b = series_x_positions(series_b)
        # sort by x; keep only rungs with a time position; columns follow the sort
        order_a = sorted((x, rung) for rung, x in pos_a.items())
        order_b = sorted((x, rung) for rung, x in pos_b.items())
        xs_a = [x for x, _ in order_a]
        xs_b = [x for x, _ in order_b]
        cols_a = matrix_a[:, [rungs_a.index(r) for _, r in order_a]]
        cols_b = matrix_b[:, [rungs_b.index(r) for _, r in order_b]]

        lo_t, hi_t = max(xs_a[0], xs_b[0]), min(xs_a[-1], xs_b[-1])
        for t in sorted(set(xs_a) | set(xs_b)):
            if not (lo_t <= t <= hi_t):
                side = "a" if (t < xs_a[0] or t > xs_a[-1]) else "b"
                out_of_range[side].append(t)
                continue
            values_a, (a_lo, a_hi), measured_a = _interpolate_series(cols_a, xs_a, t)
            values_b, (b_lo, b_hi), measured_b = _interpolate_series(cols_b, xs_b, t)
            deltas = values_a - values_b
            est, lo, hi, n_pairs = point_stats(deltas)
            points.append({
                "x": t,
                "rung_a": order_a[a_hi][1] if measured_a else (order_a[a_lo][1], order_a[a_hi][1]),
                "rung_b": order_b[b_hi][1] if measured_b else (order_b[b_lo][1], order_b[b_hi][1]),
                "x_a": t if measured_a else (xs_a[a_lo], xs_a[a_hi]),
                "x_b": t if measured_b else (xs_b[b_lo], xs_b[b_hi]),
                "measured_a": measured_a, "measured_b": measured_b,
                "delta": est, "lo": lo, "hi": hi, "n_pairs": n_pairs,
            })
    else:
        raise ValueError(f"unknown x_policy {x_policy!r} (use 'time' or 'rung')")

    n_values = [p["n_pairs"] for p in points if p["n_pairs"]]
    return {
        "metric": metric_key,
        "x_policy": x_policy,
        "group_key": key,
        "band": "pointwise",
        "interval": interval,
        "points": points,
        "out_of_range": out_of_range,
        "n_pairs_range": (min(n_values), max(n_values)) if n_values else (0, 0),
        "pairing": contract,
    }


def _side_at_time(
    matrix: np.ndarray,
    xs: list[float],
    rungs: list[int],
    t: float,
    ladder_policy: str,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    """One series' per-expression values at time ``t`` + a status block, or ``None`` when the
    series cannot run within ``t`` (t below its cheapest measured configuration).

    ``ladder_policy='plateau'``: for t beyond the most expensive measured configuration the LAST
    measured values are carried forward with ``status='plateau'`` — never a trend extrapolation.
    The carried value is a LOWER bound on the series' value at t under the monotone
    quality-in-compute assumption; callers must guard verdicts accordingly.
    ``ladder_policy='strict'``: beyond-ladder returns ``None`` (the point does not exist).
    Columns of ``matrix`` follow ``xs``/``rungs`` (ascending in x)."""
    if t < xs[0] * (1 - 1e-9):
        return None
    if t > xs[-1] * (1 + 1e-9):
        if ladder_policy != "plateau":
            return None
        j = len(xs) - 1
        return matrix[:, j], {"status": "plateau", "rung_lo": rungs[j], "rung_hi": rungs[j],
                              "x_lo": xs[j], "x_hi": xs[j], "x_effective": xs[j]}
    values, (j_lo, j_hi), measured = _interpolate_series(matrix, xs, t)
    return values, {"status": "measured" if measured else "interpolated",
                    "rung_lo": rungs[j_lo], "rung_hi": rungs[j_hi],
                    "x_lo": xs[j_lo], "x_hi": xs[j_hi], "x_effective": t}


def _sorted_time_columns(
    series: Mapping[int, Mapping[str, Sequence[Any]]],
    metric_key: str,
    key: str,
    aggregate: Callable[[np.ndarray], float],
) -> tuple[np.ndarray, list[float], list[int], list[Any]]:
    """(matrix, xs, rungs, keys) for one series, columns sorted ascending by median time."""
    keys, rungs, matrix = _expression_matrix(series, metric_key, key, aggregate)
    pos = series_x_positions(series)
    order = sorted((x, rung) for rung, x in pos.items() if rung in rungs)
    xs = [x for x, _ in order]
    cols = [rungs.index(r) for _, r in order]
    return matrix[:, cols], xs, [r for _, r in order], keys


def series_values_at_time(
    series: Mapping[int, Mapping[str, Sequence[Any]]],
    metric_key: str,
    t: float,
    *,
    group_key: str | None = None,
    aggregate: Callable[[np.ndarray], float] = np.nanmean,
    ladder_policy: str = "plateau",
) -> dict[str, Any] | None:
    """PER-PROBLEM values of one series at time ``t`` — the vector behind
    :func:`series_report_at_time`, exposed for cross-method constructions (rank leagues).

    Same interpolation model and boundary policy as the other at-time reports (linear in
    log10-time between bracketing configurations; ``'plateau'`` carries the last measured
    values forward, flagged). Returns ``{'values': {problem_id: value}, status fields...}``
    or ``None`` when the series cannot run within ``t``. Only finite per-problem values are
    returned (a problem missing at either bracketing configuration is absent)."""
    key = group_key or resolve_group_key(next(iter(series.values())))
    matrix, xs, rungs, keys = _sorted_time_columns(series, metric_key, key, aggregate)
    if not xs:
        return None
    side = _side_at_time(matrix, xs, rungs, t, ladder_policy)
    if side is None:
        return None
    values, status = side
    out = {"values": {k: float(v) for k, v in zip(keys, values) if np.isfinite(v)}}
    out.update(status)
    return out


def rank_league(
    values_by_method: Mapping[str, Mapping[Any, float]],
    *,
    higher_is_better: bool = True,
    mode: str = "worst-rank",
    alpha: float = 0.05,
) -> dict[str, Any] | None:
    """k-method rank league over shared problems: tie-corrected Friedman omnibus + Nemenyi
    critical difference + indistinguishability cliques (the CD-diagram statistics, Demšar 2006).

    ``values_by_method`` maps method name -> {problem_id: value} (e.g. from
    :func:`series_values_at_time`). Within each problem, methods are ranked 1 (best) .. k,
    ties averaged. ``mode='worst-rank'`` keeps every problem where at least one method has a
    value and ranks MISSING methods strictly worst in that problem (ties among them averaged)
    — sound for quality-to-GT axes, where "no usable prediction" is worse than any observed
    value; ``mode='all-present'`` keeps only problems where every method has a value (the
    conditional league for output-property metrics — a different estimand, label it).

    Returns ``None`` when fewer than 2 problems or 2 methods survive. The Friedman statistic
    uses the tie-corrected form (Conover); ``cd`` is the Nemenyi critical difference at
    ``alpha``; ``cliques`` are the maximal groups of methods whose mean ranks span <= cd
    (report them only when the omnibus rejects — that discipline is the caller's job)."""
    from scipy.stats import chi2, rankdata, studentized_range

    methods = list(values_by_method)
    k = len(methods)
    if k < 2:
        return None
    if mode == "all-present":
        ids = sorted(set.intersection(*(set(values_by_method[m]) for m in methods)), key=str)
    elif mode == "worst-rank":
        ids = sorted(set.union(*(set(values_by_method[m]) for m in methods)), key=str)
    else:
        raise ValueError(f"unknown mode {mode!r} (use 'worst-rank' or 'all-present')")
    n = len(ids)
    if n < 2:
        return None

    oriented = np.full((n, k), -np.inf)   # missing = strictly worst in the oriented score
    n_missing = np.zeros(k, dtype=int)
    for j, m in enumerate(methods):
        col = values_by_method[m]
        for i, pid in enumerate(ids):
            if pid in col:
                v = float(col[pid])
                oriented[i, j] = v if higher_is_better else -v
            else:
                n_missing[j] += 1
    ranks = np.vstack([rankdata(-oriented[i], method="average") for i in range(n)])

    mean_ranks = ranks.mean(axis=0)
    # tie-corrected Friedman (Conover): chi2 = (k-1) * sum_j (R_j - n(k+1)/2)^2 / (A1 - C1)
    column_sums = ranks.sum(axis=0)
    a1 = float(np.sum(ranks ** 2))
    c1 = n * k * (k + 1) ** 2 / 4.0
    if a1 <= c1:                          # every problem fully tied — no information
        chi2_stat, p_value = 0.0, 1.0
    else:
        chi2_stat = float((k - 1) * np.sum((column_sums - n * (k + 1) / 2.0) ** 2) / (a1 - c1))
        p_value = float(chi2.sf(chi2_stat, k - 1))
    cd = float(studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2.0)
               * np.sqrt(k * (k + 1) / (6.0 * n)))

    order = np.argsort(mean_ranks)
    clique_indices: list[list[int]] = []
    for i in range(k):
        group = [int(j) for j in order if 0 <= mean_ranks[j] - mean_ranks[order[i]] <= cd]
        if len(group) > 1 and not any(set(group) <= set(g) for g in clique_indices):
            clique_indices.append(group)
    tie_share = float(np.mean([1.0 - len(set(row)) / k for row in ranks]))

    return {
        "methods": methods,
        "n_problems": n,
        "mean_ranks": {m: float(mean_ranks[j]) for j, m in enumerate(methods)},
        "n_missing": {m: int(n_missing[j]) for j, m in enumerate(methods)},
        "friedman_chi2": chi2_stat,
        "friedman_p": p_value,
        "cd": cd,
        "alpha": alpha,
        "mode": mode,
        "tie_share": tie_share,
        "cliques": [[methods[j] for j in group] for group in clique_indices],
    }


def series_report_at_time(
    series: Mapping[int, Mapping[str, Sequence[Any]]],
    metric_key: str,
    t: float,
    *,
    group_key: str | None = None,
    aggregate: Callable[[np.ndarray], float] = np.nanmean,
    n: int = 10_000,
    interval: float = 0.95,
    rng: np.random.Generator | int | None = 0,
    ladder_policy: str = "plateau",
) -> dict[str, Any] | None:
    """MARGINAL value of one series at time ``t``: per-expression linear interpolation in
    log10-time between the bracketing measured configurations (the same model as
    :func:`paired_delta_curve`), then the house bootstrap triple over expressions.

    Returns ``None`` when the series cannot run within ``t``. ``status`` is ``'measured'`` /
    ``'interpolated'`` / ``'plateau'`` (see ``ladder_policy``); a plateau value is a lower bound
    under the monotone quality-in-compute assumption and must be displayed flagged. Marginal
    numbers: never difference two of these — that comparison belongs to
    :func:`paired_report_at_time`."""
    key = group_key or resolve_group_key(next(iter(series.values())))
    matrix, xs, rungs, _ = _sorted_time_columns(series, metric_key, key, aggregate)
    if not xs:
        return None
    side = _side_at_time(matrix, xs, rungs, t, ladder_policy)
    if side is None:
        return None
    values, status = side
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return None
    if not hasattr(rng, "integers"):
        rng = np.random.default_rng(rng)
    est, lo, hi = _bootstrap_triple(valid, n=n, interval=interval, rng=rng)
    report = {"metric": metric_key, "t": float(t), "value": est, "ci_lower": lo, "ci_upper": hi,
              "n_groups": int(valid.size), "interval": interval, "group_key": key}
    report.update(status)
    return report


def paired_report_at_time(
    series_a: Mapping[int, Mapping[str, Sequence[Any]]],
    series_b: Mapping[int, Mapping[str, Sequence[Any]]],
    metric_key: str,
    t: float,
    *,
    higher_is_better: bool = True,
    margin: float | Mapping[str, Any] | None = None,
    group_key: str | None = None,
    aggregate: Callable[[np.ndarray], float] = np.nanmean,
    n: int = 10_000,
    interval: float = 0.95,
    rng: np.random.Generator | int | None = 0,
    zero_method: str = "pratt",
    allow_unverified: bool = False,
    ladder_policy: str = "plateau",
) -> dict[str, Any] | None:
    """Paired comparison of two series AT THE SAME wall-clock time ``t`` per problem.

    Each side is brought to exactly ``t`` by per-expression linear interpolation in log10-time
    between its bracketing measured configurations (never extrapolated); an expression
    contributes only if valid at both bracketing rungs of BOTH sides. Statistics follow
    :func:`paired_report`'s conventions (expression bootstrap of the mean delta, median of the
    bootstrap distribution, floored two-sided p, win/tie/loss, probability of superiority,
    Wilcoxon-pratt companion, MDE₈₀, four-state verdict vs ``margin``).

    Ladder boundaries (``ladder_policy='plateau'``): a side whose ladder ends below ``t``
    carries its last measured values forward (``status='plateau'`` — a LOWER bound under the
    monotone quality-in-compute assumption, never a trend extrapolation). A verdict stands only
    if no plateau side could overturn it by improving: 'better'/'worse' in favor of a
    non-plateau side over a plateau side, and 'equivalent' involving any plateau side, are
    downgraded to 'undecided' with ``verdict_note='ladder-limited'``. Returns ``None`` when
    either side cannot run within ``t`` at all. Draw-level extras (variance decomposition,
    hierarchical rank CIs, worst-rank) are undefined for interpolated values and not reported."""
    first_a = next(iter(series_a.values()))
    first_b = next(iter(series_b.values()))
    contract = _verify_pairing(first_a, first_b, allow_unverified=allow_unverified)
    key = group_key or resolve_group_key(first_a)

    matrix_a, xs_a, rungs_a, keys_a = _sorted_time_columns(series_a, metric_key, key, aggregate)
    matrix_b, xs_b, rungs_b, keys_b = _sorted_time_columns(series_b, metric_key, key, aggregate)
    if not xs_a or not xs_b:
        return None
    common = sorted(set(keys_a) & set(keys_b), key=str)
    matrix_a = matrix_a[[keys_a.index(k) for k in common]]
    matrix_b = matrix_b[[keys_b.index(k) for k in common]]

    side_a = _side_at_time(matrix_a, xs_a, rungs_a, t, ladder_policy)
    side_b = _side_at_time(matrix_b, xs_b, rungs_b, t, ladder_policy)
    if side_a is None or side_b is None:
        return None
    values_a, status_a = side_a
    values_b, status_b = side_b

    finite_a, finite_b = np.isfinite(values_a), np.isfinite(values_b)
    both = finite_a & finite_b
    deltas = (values_a - values_b)[both]
    n_pairs = int(deltas.size)
    only_a = [common[i] for i in np.nonzero(finite_a & ~finite_b)[0]]
    only_b = [common[i] for i in np.nonzero(finite_b & ~finite_a)[0]]

    report: dict[str, Any] = {
        "metric": metric_key, "group_key": key, "t": float(t),
        "n_pairs": n_pairs, "n_only_a": len(only_a), "n_only_b": len(only_b),
        "only_a": only_a, "only_b": only_b,
        "interval": interval, "n_bootstrap": int(n),
        "higher_is_better": higher_is_better, "pairing": contract,
        "side_a": status_a, "side_b": status_b,
    }
    if n_pairs == 0:
        report.update({k: float("nan") for k in
                       ("delta_mean", "ci_lower", "ci_upper", "delta_median",
                        "median_ci_lower", "median_ci_upper", "mde_80", "p_value")})
        report.update({"win_rate": None, "prob_superiority": float("nan"),
                       "prob_superiority_ci": None, "wilcoxon": None, "verdict": None,
                       "verdict_note": None, "margin": None, "equivalence_attainable": None})
        return report

    if not hasattr(rng, "integers"):
        rng = np.random.default_rng(rng)
    indices = rng.integers(0, n_pairs, size=(int(n), n_pairs))
    resampled = deltas[indices]
    boot_means = np.nanmean(resampled, axis=1)
    lo_q, hi_q = (1 - interval) / 2 * 100, (1 + interval) / 2 * 100
    report["delta_mean"] = float(np.nanmedian(boot_means))
    report["ci_lower"] = float(np.nanpercentile(boot_means, lo_q))
    report["ci_upper"] = float(np.nanpercentile(boot_means, hi_q))
    p_low = float(np.mean(boot_means <= 0.0))
    p_high = float(np.mean(boot_means >= 0.0))
    report["p_value"] = float(max(2.0 * min(p_low, p_high), 1.0 / (int(n) + 1)))

    n_tied = int(np.sum(deltas == 0))
    report["win_rate"] = {
        "a_better": int(np.sum(deltas > 0)) if higher_is_better else int(np.sum(deltas < 0)),
        "b_better": int(np.sum(deltas < 0)) if higher_is_better else int(np.sum(deltas > 0)),
        "tied": n_tied,
    }
    signed = deltas if higher_is_better else -deltas
    report["prob_superiority"] = float((np.sum(signed > 0) + 0.5 * n_tied) / n_pairs)
    signed_resampled = resampled if higher_is_better else -resampled
    psup_boot = (np.sum(signed_resampled > 0, axis=1)
                 + 0.5 * np.sum(signed_resampled == 0, axis=1)) / n_pairs
    report["prob_superiority_ci"] = (float(np.nanpercentile(psup_boot, lo_q)),
                                     float(np.nanpercentile(psup_boot, hi_q)))
    rank_medians = np.nanmedian(resampled, axis=1)
    report["delta_median"] = float(np.nanmedian(rank_medians))
    report["median_ci_lower"] = float(np.nanpercentile(rank_medians, lo_q))
    report["median_ci_upper"] = float(np.nanpercentile(rank_medians, hi_q))

    from scipy.stats import wilcoxon
    nonzero = int(np.sum(deltas != 0))
    if nonzero == 0:
        report["wilcoxon"] = {"statistic": float("nan"), "p": 1.0, "n_zero": n_tied,
                              "n_nonzero": 0, "zero_method": zero_method}
    else:
        w = wilcoxon(deltas, zero_method=zero_method, method="approx")
        report["wilcoxon"] = {"statistic": float(w.statistic), "p": float(w.pvalue),
                              "n_zero": n_tied, "n_nonzero": nonzero, "zero_method": zero_method}

    se = float(np.nanstd(deltas, ddof=1) / np.sqrt(n_pairs)) if n_pairs > 1 else float("nan")
    report["mde_80"] = float((1.959963985 + 0.8416212336) * se)

    m = margin.get("margin") if isinstance(margin, Mapping) else margin
    report["margin"] = float(m) if m is not None else None
    report["verdict_note"] = None
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
        # Ladder guard: a plateau side's true value at t is >= its carried value, so any verdict
        # a plateau side could overturn by improving is not an at-t claim — downgrade it.
        limited_a = status_a["status"] == "plateau"
        limited_b = status_b["status"] == "plateau"
        overturnable = ((verdict == "better" and limited_b)
                        or (verdict == "worse" and limited_a)
                        or (verdict == "equivalent" and (limited_a or limited_b)))
        if overturnable:
            verdict = "undecided"
            report["verdict_note"] = "ladder-limited"
        elif verdict == "undecided" and (limited_a or limited_b):
            report["verdict_note"] = "ladder-limited"
        report["verdict"] = verdict
        report["equivalence_attainable"] = bool((hi - lo) / 2 <= m)
    return report


def _bootstrap_triple(
    data: np.ndarray, *, n: int, interval: float, rng: np.random.Generator,
) -> tuple[float, float, float]:
    """House (estimate, lo, hi) for a 1-D sample: median of bootstrap means + percentile CI."""
    indices = rng.integers(0, data.size, size=(int(n), data.size))
    boot = np.nanmean(data[indices], axis=1)
    return (float(np.nanmedian(boot)),
            float(np.nanpercentile(boot, (1 - interval) / 2 * 100)),
            float(np.nanpercentile(boot, (1 + interval) / 2 * 100)))


def significant_round(value: float, ci_width: float) -> float:
    """Round ``value`` to the decimals justified by its CI width (two significant digits of the
    width): a number with a ±0.03 interval has no business displaying five decimals, and the
    rounding hides bootstrap recomputation wobble at the display layer (WP1 ✓dec: no fixed-seed
    requirement — precision limited to what the uncertainty supports instead)."""
    if not np.isfinite(value):
        return value
    if not np.isfinite(ci_width) or ci_width <= 0:
        return value
    decimals = max(0, 1 - int(np.floor(np.log10(ci_width))))
    return round(value, decimals)


def rounded_triple(value: float, lo: float, hi: float) -> tuple[float, float, float]:
    """(value, lo, hi) display-rounded to the precision the interval width justifies."""
    width = hi - lo
    return (significant_round(value, width), significant_round(lo, width),
            significant_round(hi, width))


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
           "paired_report_at_time", "series_report_at_time", "series_values_at_time",
           "rank_league",
           "paired_delta_curve", "series_x_positions", "self_noise", "pair_margin",
           "resolve_group_key", "pairing_fingerprint", "PairingContractError",
           "significant_round", "rounded_triple", "bootstrap_report"]
