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
    pytest.importorskip("matplotlib")  # figures live behind the [analysis] extra
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


def test_export_data_writes_tidy_json(tmp_path):
    import json

    from srbf.analysis import export_data

    rng = np.random.default_rng(2)
    runs = [
        RunResult(model="strong", benchmark="fastsrb", axis="compute", scaling=1.0, version="v23.0",
                  snapshot=_snapshot(8, 3, 0.8, rng)),
        RunResult(model="strong", benchmark="fastsrb", axis="compute", scaling=10.0, version="v23.0",
                  snapshot=_snapshot(8, 3, 0.9, rng)),
        RunResult(model="pysr", benchmark="fastsrb", axis="compute", scaling=5.0, version="-",
                  snapshot=_snapshot(8, 3, 0.3, rng)),
    ]
    out = export_data(runs, str(tmp_path / "results_data.json"), operator_arity=ARITY, n_bootstrap=200)
    payload = json.loads(open(out).read())
    assert payload["axes"] == ["compute"]
    assert {m["key"] for m in payload["metrics"]} >= {"numeric_recovery_val"}
    assert len(payload["records"]) == 3
    rec = payload["records"][0]
    for key in ("series", "version", "benchmark", "axis", "x"):
        assert key in rec
    cell = rec["numeric_recovery_val"]
    assert set(cell) == {"median", "lo", "hi", "n"}
    # versions travel through so archived + fresh series coexist
    assert {r["version"] for r in payload["records"]} == {"v23.0", "-"}


def test_load_runs_from_manifest(tmp_path):
    import pickle

    import yaml

    from srbf.analysis import load_runs

    rng = np.random.default_rng(1)
    entries = []
    for i, (model, bench, s, rate) in enumerate([
        ("strong", "fastsrb", 512, 0.8), ("weak", "fastsrb", 512, 0.2),
    ]):
        path = tmp_path / f"run{i}.pkl"
        with open(path, "wb") as h:
            pickle.dump(_snapshot(8, 3, rate, rng), h)
        entries.append({"model": model, "benchmark": bench, "scaling": s, "path": f"run{i}.pkl"})
    with open(tmp_path / "manifest.yaml", "w") as h:
        yaml.safe_dump({"runs": entries}, h)

    runs = load_runs(str(tmp_path / "manifest.yaml"))
    assert len(runs) == 2
    assert {r.model for r in runs} == {"strong", "weak"}
    assert all(r.scaling == 512 and "y_pred_val" in r.snapshot for r in runs)
    # the loaded runs render a report end-to-end (figures need the [analysis] extra)
    pytest.importorskip("matplotlib")
    out = build_report(runs, str(tmp_path / "out"), operator_arity=ARITY, n_bootstrap=200)
    assert os.path.isfile(out)
