"""Tests for srbf.analysis: the standardized results-page views over raw run snapshots."""
import os

import numpy as np
import pytest

from srbf.analysis import RunResult, leaderboard, build_report, DEFAULT_METRICS

ARITY = {"add": 2, "mul": 2, "sin": 1}
GT = ["add", "mul", "x1", "x1", "sin", "x1"]
WRONG = ["mul", "x1", "sin", "x1"]


def _snapshot(n_expr, draws, recovery_rate, rng):
    n = 8
    cols = {k: [] for k in ("y", "y_pred", "y_val", "y_pred_val", "skeleton", "predicted_skeleton_prefix", "benchmark_eq_id", "placeholder")}
    for e in range(n_expr):
        recovered = rng.random() < recovery_rate
        for _ in range(draws):
            base = np.linspace(-1, 1, n) + e * 0.01
            cols["y"].append(base.copy())
            cols["y_val"].append(base.copy())
            if recovered:
                cols["y_pred"].append(base.copy())
                cols["y_pred_val"].append(base.copy())
                cols["predicted_skeleton_prefix"].append(list(GT))
            else:
                cols["y_pred"].append(base + rng.normal(scale=0.6, size=n))
                cols["y_pred_val"].append(base + rng.normal(scale=0.6, size=n))
                cols["predicted_skeleton_prefix"].append(list(WRONG))
            cols["skeleton"].append(list(GT))
            cols["benchmark_eq_id"].append(f"eq{e}")
            cols["placeholder"].append(False)
    return cols


def _runs():
    rng = np.random.default_rng(0)
    runs = []
    profiles = {"strong": {64: 0.4, 512: 0.85}, "weak": {64: 0.1, 512: 0.2}}
    for model, prof in profiles.items():
        for bench in ("fastsrb", "feynman"):
            for s, rate in prof.items():
                runs.append(RunResult(model=model, benchmark=bench, scaling=s,
                                      snapshot=_snapshot(10, 4, rate, rng)))
    return runs


def test_leaderboard_ranks_models_with_ci_columns():
    lb = leaderboard(_runs(), operator_arity=ARITY, n_bootstrap=300)
    assert list(lb["model"]) == ["strong", "weak"]  # ranked by numeric recovery, strong first
    # the strong model recovers more at its max scaling than the weak one
    m = DEFAULT_METRICS[0].label
    strong = lb.loc[lb["model"] == "strong", f"{m} median"].iloc[0]
    weak = lb.loc[lb["model"] == "weak", f"{m} median"].iloc[0]
    assert strong > weak
    # CI columns present and ordered lo <= median <= hi
    for _, r in lb.iterrows():
        assert r[f"{m} lo"] <= r[f"{m} median"] <= r[f"{m} hi"]


def test_build_report_writes_page_and_figures(tmp_path):
    out = build_report(_runs(), str(tmp_path), operator_arity=ARITY, n_bootstrap=300)
    assert os.path.isfile(out)
    text = open(out).read()
    for section in ("## Leaderboard", "## Scaling", "## Per-benchmark breakdown", "## Distribution"):
        assert section in text
    for fig in ("scaling.png", "per_benchmark.png", "distribution.png"):
        assert os.path.isfile(os.path.join(tmp_path, "figures", fig))


def test_leaderboard_requires_engine_or_arity():
    with pytest.raises(ValueError):
        leaderboard(_runs(), n_bootstrap=50)
