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


# --- Section C: paired_delta_curve (x_policy machinery) ---

def _rung_snapshot(values_by_eq: dict[str, list[float]], fit_time: float) -> dict:
    snap = _snapshot_from(values_by_eq)
    snap["fit_time"] = [fit_time] * len(snap["m"])
    return snap


def _linear_series(eqs, rung_xs: dict[int, float], slope: float, offset: float) -> dict[int, dict]:
    """Per-expression metric value = offset + eq_index + slope * log10(x): exactly linear in
    log-time, so interpolation must be EXACT at any midpoint."""
    series = {}
    for rung, x in rung_xs.items():
        series[rung] = _rung_snapshot(
            {eq: [offset + i + slope * np.log10(x)] * 2 for i, eq in enumerate(eqs)}, fit_time=x)
    return series


def test_delta_curve_rung_policy_matches_shared_rungs():
    from srbf.reporting import paired_delta_curve
    eqs = [f"E{i}" for i in range(20)]
    a = _linear_series(eqs, {64: 1.0, 1024: 10.0}, slope=0.0, offset=0.5)
    b = _linear_series(eqs, {64: 2.0, 1024: 20.0, 4096: 50.0}, slope=0.0, offset=0.3)
    curve = paired_delta_curve(a, b, "m", x_policy="rung", allow_unverified=True)
    assert [p["rung_a"] for p in curve["points"]] == [64, 1024]
    for p in curve["points"]:
        assert p["delta"] == pytest.approx(0.2, abs=1e-9)
        assert p["measured_a"] and p["measured_b"]
        assert p["n_pairs"] == 20


def test_delta_curve_time_policy_interpolates_exactly_in_log_time():
    from srbf.reporting import paired_delta_curve
    eqs = [f"E{i}" for i in range(15)]
    # A measured at x = 1 and 100; B at x = 10 (the log-midpoint) plus endpoints.
    a = _linear_series(eqs, {1: 1.0, 100: 100.0}, slope=0.4, offset=1.0)
    b = _linear_series(eqs, {1: 1.0, 10: 10.0, 100: 100.0}, slope=0.4, offset=0.0)
    curve = paired_delta_curve(a, b, "m", x_policy="time", allow_unverified=True)
    # grid = union {1, 10, 100}; at t=10 A is interpolated between 1 and 100 — since the metric
    # is exactly linear in log10(t), the interpolated delta must equal the true offset 1.0.
    by_x = {p["x"]: p for p in curve["points"]}
    assert set(by_x) == {1.0, 10.0, 100.0}
    for t, p in by_x.items():
        assert p["delta"] == pytest.approx(1.0, abs=1e-9), f"at t={t}"
    assert by_x[10.0]["measured_a"] is False and by_x[10.0]["measured_b"] is True
    assert isinstance(by_x[10.0]["x_a"], tuple)  # bracketing positions recorded
    assert curve["out_of_range"] == {"a": [], "b": []}


def test_delta_curve_out_of_range_is_loud_not_silent():
    from srbf.reporting import paired_delta_curve
    eqs = [f"E{i}" for i in range(10)]
    a = _linear_series(eqs, {64: 1.0, 1024: 10.0}, slope=0.0, offset=0.0)
    b = _linear_series(eqs, {64: 5.0, 1024: 50.0, 4096: 500.0}, slope=0.0, offset=0.0)
    curve = paired_delta_curve(a, b, "m", x_policy="time", allow_unverified=True)
    # overlap window is [5, 10]: b's x=50 and x=500 and a's x=1 fall outside and MUST be reported
    assert 1.0 in curve["out_of_range"]["a"] or 1.0 in curve["out_of_range"]["b"]
    assert 50.0 in curve["out_of_range"]["a"]   # beyond a's measured span
    assert 500.0 in curve["out_of_range"]["a"]
    assert all(5.0 <= p["x"] <= 10.0 for p in curve["points"])


def test_delta_curve_composition_drift_guard():
    from srbf.reporting import paired_delta_curve
    eqs = [f"E{i}" for i in range(10)]
    a = _linear_series(eqs, {1: 1.0, 100: 100.0}, slope=0.0, offset=1.0)
    # E0 has NO value at b's x=100 rung -> at interpolated t=10 (bracket 1..100) E0 must drop,
    # so n_pairs at t=10 is 9 while at t=1 it is 10.
    b_low = {eq: [0.0, 0.0] for eq in eqs}
    b_high = {eq: [0.0, 0.0] for eq in eqs if eq != "E0"}
    b = {1: _rung_snapshot(b_low, 1.0), 100: _rung_snapshot(b_high, 100.0),
         10: _rung_snapshot(b_low, 10.0)}
    curve = paired_delta_curve(a, b, "m", x_policy="time", allow_unverified=True)
    by_x = {p["x"]: p for p in curve["points"]}
    assert by_x[1.0]["n_pairs"] == 10
    assert by_x[10.0]["n_pairs"] == 10   # measured for B at t=10 (E0 valid there)
    assert by_x[100.0]["n_pairs"] == 9   # E0 missing at B's 100-rung
    assert curve["n_pairs_range"] == (9, 10)


# --- A fast-follows: worst-rank, hierarchical bootstrap, display rounding ---

def test_worst_rank_imputes_one_sided_failures_as_sentinels():
    from srbf.reporting import paired_report
    # 20 finite pairs (A slightly better) + 4 expressions where only A produced values
    values_a = {f"E{i}": [0.5, 0.6] for i in range(20)} | {f"F{i}": [0.9] for i in range(4)}
    values_b = {f"E{i}": [0.4, 0.5] for i in range(20)}
    report = paired_report(_snapshot_from(values_a), _snapshot_from(values_b), "m",
                           allow_unverified=True, worst_rank=True)
    block = report["worst_rank"]
    assert block["n_union"] == 24 and block["n_imputed_a"] == 4 and not block["degenerate"]
    assert block["win_rate"] == {"a_better": 24, "b_better": 0, "tied": 0}
    assert block["prob_superiority"] == 1.0
    # median over 20 finite +0.1 deltas and 4 +inf sentinels is finite
    assert block["median_delta"] == pytest.approx(0.1, abs=1e-9)
    assert block["wilcoxon"]["p"] < 0.01
    # and the mean path is untouched by imputation (still both-succeed only)
    assert report["n_pairs"] == 20


def test_worst_rank_degenerate_when_half_imputed():
    from srbf.reporting import paired_report
    values_a = {f"E{i}": [0.5] for i in range(4)} | {f"F{i}": [0.9] for i in range(6)}
    values_b = {f"E{i}": [0.4] for i in range(4)}
    report = paired_report(_snapshot_from(values_a), _snapshot_from(values_b), "m",
                           allow_unverified=True, worst_rank=True)
    block = report["worst_rank"]
    assert block["degenerate"] is True and block["median_delta"] is None
    assert "success-rate" in block["note"]


def test_worst_rank_flips_sentinels_for_lower_is_better():
    from srbf.reporting import paired_report
    # A produced values where B failed -> A better; for a lower-is-better metric the reported
    # median delta must favor A with NEGATIVE sign (A - B < 0 = A better).
    values_a = {f"E{i}": [0.5] for i in range(10)} | {f"F{i}": [0.9] for i in range(3)}
    values_b = {f"E{i}": [0.8] for i in range(10)}
    report = paired_report(_snapshot_from(values_a), _snapshot_from(values_b), "m",
                           allow_unverified=True, worst_rank=True, higher_is_better=False)
    block = report["worst_rank"]
    assert block["win_rate"]["a_better"] == 13
    assert block["median_delta"] == pytest.approx(-0.3, abs=1e-9)


def test_hierarchical_widens_rank_cis_under_draw_noise():
    from srbf.reporting import paired_report
    rng = np.random.default_rng(7)
    snap_a, snap_b = _shifted_snapshots(rng, n_expr=40, k=4, shift=0.0,
                                        difficulty_sd=0.05, noise_sd=0.6)
    flat = paired_report(snap_a, snap_b, "m", allow_unverified=True, rng=1)
    hier = paired_report(snap_a, snap_b, "m", allow_unverified=True, rng=1, hierarchical=True)
    width_flat = flat["median_ci_upper"] - flat["median_ci_lower"]
    width_hier = hier["median_ci_upper"] - hier["median_ci_lower"]
    assert hier["rank_ci_method"] == "hierarchical"
    assert width_hier >= width_flat * 0.95  # never meaningfully narrower
    # P(sup) CI present in both
    assert len(flat["prob_superiority_ci"]) == 2
    lo, hi = hier["prob_superiority_ci"]
    assert 0.0 <= lo <= hi <= 1.0


def test_significant_round_matches_ci_width():
    from srbf.reporting import rounded_triple, significant_round
    # width ~0.143 -> two significant digits of the width = 2 decimals
    assert rounded_triple(0.064312, -0.00701, 0.135702) == (0.06, -0.01, 0.14)
    # width ~0.0143 -> 3 decimals
    assert rounded_triple(0.064312, 0.057, 0.0713) == (0.064, 0.057, 0.071)
    assert significant_round(0.123456, 0.5) == 0.12
    assert significant_round(42.1234, 12.0) == 42.0
    assert significant_round(float("nan"), 0.1) != significant_round(0.0, 0.1)  # nan passes through
    assert significant_round(1.234567, 0.0) == 1.234567  # zero width: no information, no rounding


def test_p_value_matches_ci_semantics():
    from srbf.reporting import paired_report
    rng = np.random.default_rng(9)
    shifted_a, shifted_b = _shifted_snapshots(rng, shift=0.15, noise_sd=0.05)
    strong = paired_report(shifted_a, shifted_b, "m", allow_unverified=True, rng=1)
    assert strong["p_value"] <= 2e-4  # floored at 1/(n+1)
    null_a, null_b = _shifted_snapshots(rng, shift=0.0, noise_sd=0.3)
    null = paired_report(null_a, null_b, "m", allow_unverified=True, rng=2)
    assert null["p_value"] > 0.05
    # CI excludes zero  <=>  p < alpha (percentile duality, same resamples)
    assert (strong["ci_lower"] > 0) == (strong["p_value"] < 0.05)
