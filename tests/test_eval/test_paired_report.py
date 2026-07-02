"""Paired-comparison layer: report shape, verdicts, contract violations, and the core property —
pairing must beat naive difference-of-marginals when difficulty is shared."""
import numpy as np
import pytest

from srbf.metrics.bootstrap import bootstrapped_metric_ci
from srbf.reporting import PairingContractError, paired_report


def _snapshot_from(values_by_eq: dict[str, list[float]]) -> dict:
    """Build a dict-of-lists snapshot (one row per draw) from {eq_id: draw values (None = failed)}."""
    eq_ids: list[str] = []
    metric: list[float | None] = []
    for eq, draws in values_by_eq.items():
        for value in draws:
            eq_ids.append(eq)
            metric.append(value)
    return {"benchmark_eq_id": eq_ids, "m": metric, "placeholder": [False] * len(eq_ids)}


def _shifted_snapshots(rng, n_expr=60, k=8, shift=0.1, difficulty_sd=1.0, noise_sd=0.1):
    """Two snapshots sharing per-expression difficulty; B is `shift` worse than A everywhere."""
    a, b = {}, {}
    for i in range(n_expr):
        difficulty = rng.normal(0.0, difficulty_sd)
        a[f"E{i}"] = list(difficulty + rng.normal(0, noise_sd, size=k))
        b[f"E{i}"] = list(difficulty - shift + rng.normal(0, noise_sd, size=k))
    return _snapshot_from(a), _snapshot_from(b)


def test_paired_report_detects_a_shift_shared_difficulty_cancels():
    rng = np.random.default_rng(0)
    snap_a, snap_b = _shifted_snapshots(rng, shift=0.1)
    report = paired_report(snap_a, snap_b, "m", allow_unverified=True)
    assert report["n_pairs"] == 60 and report["n_only_a"] == report["n_only_b"] == 0
    # the shift is recovered and the CI excludes zero despite difficulty_sd >> shift
    assert report["ci_lower"] > 0
    assert report["delta_mean"] == pytest.approx(0.1, abs=0.05)
    assert report["prob_superiority"] > 0.8
    assert report["win_rate"]["a_better"] > report["win_rate"]["b_better"]
    assert report["wilcoxon"]["p"] < 0.01


def test_paired_ci_narrower_than_difference_of_marginals():
    # THE property that justifies WP1: shared difficulty inflates marginal CIs but cancels in
    # the paired delta. The advantage must vanish when difficulty is not shared.
    rng = np.random.default_rng(1)
    snap_a, snap_b = _shifted_snapshots(rng, n_expr=80, shift=0.05, difficulty_sd=1.0, noise_sd=0.1)
    report = paired_report(snap_a, snap_b, "m", allow_unverified=True, rng=2)

    from srbf.reporting import draw_distribution
    per_expr_a = np.array(list(draw_distribution(snap_a, "m").values()))
    per_expr_b = np.array(list(draw_distribution(snap_b, "m").values()))
    _, a_lo, a_hi = bootstrapped_metric_ci(per_expr_a, np.nanmean, n=4000, rng=3)
    _, b_lo, b_hi = bootstrapped_metric_ci(per_expr_b, np.nanmean, n=4000, rng=4)
    naive_width = (a_hi - a_lo) + (b_hi - b_lo)  # width of the (forbidden) subtracted interval

    paired_width = report["ci_upper"] - report["ci_lower"]
    assert paired_width < 0.5 * naive_width  # pairing must be dramatically tighter here

    # Control: independent difficulties -> pairing gains ~nothing (widths comparable).
    snap_c = _snapshot_from({f"E{i}": list(np.random.default_rng(100 + i).normal(0, 1.0, 8))
                             for i in range(80)})
    snap_d = _snapshot_from({f"E{i}": list(np.random.default_rng(500 + i).normal(0, 1.0, 8))
                             for i in range(80)})
    report_indep = paired_report(snap_c, snap_d, "m", allow_unverified=True, rng=5)
    per_c = np.array(list(draw_distribution(snap_c, "m").values()))
    per_d = np.array(list(draw_distribution(snap_d, "m").values()))
    _, c_lo, c_hi = bootstrapped_metric_ci(per_c, np.nanmean, n=4000, rng=6)
    _, d_lo, d_hi = bootstrapped_metric_ci(per_d, np.nanmean, n=4000, rng=7)
    naive_indep = (c_hi - c_lo) + (d_hi - d_lo)
    paired_indep = report_indep["ci_upper"] - report_indep["ci_lower"]
    assert paired_indep > 0.5 * naive_indep  # no shared difficulty -> no dramatic gain


def test_verdicts_four_states_and_attainability():
    rng = np.random.default_rng(2)
    snap_a, snap_b = _shifted_snapshots(rng, shift=0.2, noise_sd=0.05)
    better = paired_report(snap_a, snap_b, "m", allow_unverified=True, margin=0.05)
    assert better["verdict"] == "better"
    worse = paired_report(snap_b, snap_a, "m", allow_unverified=True, margin=0.05)
    assert worse["verdict"] == "worse"

    snap_e, snap_f = _shifted_snapshots(rng, shift=0.0, noise_sd=0.05)
    equivalent = paired_report(snap_e, snap_f, "m", allow_unverified=True, margin=0.05)
    assert equivalent["verdict"] == "equivalent"
    assert equivalent["equivalence_attainable"] is True

    # Tiny margin makes 'equivalent' unattainable: the verdict must be 'undecided' and the
    # attainability diagnostic must say so (resolution-limited, not evidence of parity).
    undecided = paired_report(snap_e, snap_f, "m", allow_unverified=True, margin=1e-6)
    assert undecided["verdict"] == "undecided"
    assert undecided["equivalence_attainable"] is False

    # Lower-is-better metrics flip the labels, not the sign convention.
    flipped = paired_report(snap_a, snap_b, "m", allow_unverified=True, margin=0.05,
                            higher_is_better=False)
    assert flipped["verdict"] == "worse"


def test_rate_metric_zeros_are_counted_not_dropped():
    # 30 concordant expressions (delta 0) + 8 where A wins: pratt keeps the zeros visible.
    values_a = {f"C{i}": [1.0, 1.0] for i in range(30)} | {f"W{i}": [1.0, 1.0] for i in range(8)}
    values_b = {f"C{i}": [1.0, 1.0] for i in range(30)} | {f"W{i}": [0.0, 0.0] for i in range(8)}
    report = paired_report(_snapshot_from(values_a), _snapshot_from(values_b), "m",
                           allow_unverified=True)
    assert report["wilcoxon"]["n_zero"] == 30
    assert report["wilcoxon"]["n_nonzero"] == 8
    assert report["win_rate"] == {"a_better": 8, "b_better": 0, "tied": 30}
    assert report["prob_superiority"] == pytest.approx((8 + 15) / 38)


def test_diagnostic_missing_data_is_disclosed_and_total_mode_raises():
    values_a = {"E1": [0.5, 0.6], "E2": [0.1], "ONLY_A": [0.3]}
    values_b = {"E1": [0.4, 0.4], "E2": [0.2]}
    snap_a, snap_b = _snapshot_from(values_a), _snapshot_from(values_b)
    report = paired_report(snap_a, snap_b, "m", allow_unverified=True)
    assert report["n_pairs"] == 2 and report["n_only_a"] == 1 and report["only_a"] == ["ONLY_A"]
    with pytest.raises(PairingContractError, match="total"):
        paired_report(snap_a, snap_b, "m", allow_unverified=True, expect_total=True)


def test_contract_requires_provenance_unless_waived():
    snap_a, snap_b = _shifted_snapshots(np.random.default_rng(3))
    with pytest.raises(PairingContractError, match="allow_unverified"):
        paired_report(snap_a, snap_b, "m")

    # matching benchmark provenance verifies; mismatched raises
    meta = {"inputs": {"benchmark_file": {"sha256": "abc"}, "model_weights": {"sha256": "X"}}}
    snap_a["__meta__"], snap_b["__meta__"] = meta, {"inputs": {"benchmark_file": {"sha256": "abc"},
                                                               "model_weights": {"sha256": "Y"}}}
    verified = paired_report(snap_a, snap_b, "m")
    assert verified["pairing"]["verified"] is True

    snap_b["__meta__"] = {"inputs": {"benchmark_file": {"sha256": "DIFFERENT"}}}
    with pytest.raises(PairingContractError, match="not the same benchmark"):
        paired_report(snap_a, snap_b, "m")


def test_mde_and_variance_decomposition_present():
    rng = np.random.default_rng(4)
    snap_a, snap_b = _shifted_snapshots(rng, n_expr=100, k=10, shift=0.0,
                                        difficulty_sd=0.5, noise_sd=0.3)
    report = paired_report(snap_a, snap_b, "m", allow_unverified=True)
    # MDE ~ 2.8 * sd(delta)/sqrt(n); sd(delta) ~ noise_sd * sqrt(2/k)
    expected_sd = 0.3 * np.sqrt(2 / 10)
    assert report["mde_80"] == pytest.approx(2.8016 * expected_sd / np.sqrt(100), rel=0.3)
    vd = report["variance_decomposition"]
    # no real shift + shared difficulty cancelled -> delta variance is ~all draw noise
    assert vd["draw_noise_share"] == pytest.approx(1.0, rel=0.35)
