"""Tests for srbf.analysis: the standardized results-page views over raw run snapshots, averaged over the benchmarks as
the results site averages (srbf.aggregate)."""
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


def test_leaderboard_ranks_models_with_interval_columns():
    lb = leaderboard(_runs(), operator_arity=ARITY)
    assert list(lb["model"]) == ["strong", "weak"]  # ranked by numeric recovery, strong first
    # the strong model recovers more at its max scaling than the weak one
    m = DEFAULT_METRICS[0].label
    strong = lb.loc[lb["model"] == "strong", m].iloc[0]
    weak = lb.loc[lb["model"] == "weak", m].iloc[0]
    assert strong > weak
    # interval columns present and ordered lo <= value <= hi; two benchmarks of ten problems each
    for _, r in lb.iterrows():
        assert r[f"{m} lo"] <= r[m] <= r[f"{m} hi"]
        assert r["n_benchmarks"] == 2 and r["n_expressions"] == 20


def test_build_report_writes_page_and_figures(tmp_path):
    pytest.importorskip("matplotlib")  # figures live behind the [analysis] extra
    out = build_report(_runs(), str(tmp_path), operator_arity=ARITY)
    assert os.path.isfile(out)
    text = open(out).read()
    for section in ("## Leaderboard", "## Scaling", "## Per-benchmark breakdown", "## Distribution"):
        assert section in text
    for fig in ("scaling.png", "per_benchmark.png", "distribution.png"):
        assert os.path.isfile(os.path.join(tmp_path, "figures", fig))


def test_leaderboard_requires_engine_or_arity():
    with pytest.raises(ValueError):
        leaderboard(_runs())


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
    out = export_data(runs, str(tmp_path / "results_data.json"), operator_arity=ARITY)
    payload = json.loads(open(out).read())
    assert payload["axes"] == ["compute"]
    assert {m["key"] for m in payload["metrics"]} >= {"numeric_recovery_val"}
    assert len(payload["records"]) == 3
    rec = payload["records"][0]
    for key in ("series", "version", "benchmark", "axis", "x"):
        assert key in rec
    cell = rec["numeric_recovery_val"]
    assert set(cell) == {"value", "lo", "hi", "n"} and cell["n"] == 8
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
    out = build_report(runs, str(tmp_path / "out"), operator_arity=ARITY)
    assert os.path.isfile(out)


def test_a_metric_is_named_by_its_column():
    """Every table and figure takes a derived column's name as well as a Metric object."""
    from srbf.analysis import per_benchmark_table, scaling_table
    runs = _runs()
    by_name = per_benchmark_table(runs, "numeric_recovery_val", operator_arity=ARITY)
    by_object = per_benchmark_table(runs, DEFAULT_METRICS[0], operator_arity=ARITY)
    assert by_name.equals(by_object) and list(by_name.columns) == ["fastsrb", "feynman"]
    assert len(scaling_table(runs, "symbolic_recovery", operator_arity=ARITY)) == 4
    assert "Symbolic recovery" in leaderboard(runs, metrics=["symbolic_recovery"], operator_arity=ARITY).columns


def _fixed(recovered, draws=2):
    """A snapshot whose expressions are recovered exactly as listed (every draw alike)."""
    rng = np.random.default_rng(5)
    n = 8
    cols = {k: [] for k in ("y", "y_pred", "y_val", "y_pred_val", "skeleton", "predicted_skeleton_prefix", "benchmark_eq_id", "placeholder")}
    for e, ok in enumerate(recovered):
        for _ in range(draws):
            base = np.linspace(-1, 1, n) + e * 0.01
            cols["y"].append(base.copy())
            cols["y_val"].append(base.copy())
            cols["y_pred"].append(base.copy() if ok else base + rng.normal(scale=0.6, size=n))
            cols["y_pred_val"].append(base.copy() if ok else base + rng.normal(scale=0.6, size=n))
            cols["predicted_skeleton_prefix"].append(list(GT if ok else WRONG))
            cols["skeleton"].append(list(GT))
            cols["benchmark_eq_id"].append(f"eq{e}")
            cols["placeholder"].append(False)
    return cols


def test_a_large_benchmark_does_not_decide_the_average():
    """500 problems at 10 % and 10 at 90 %: over all problems that is 11.6 %, but each benchmark counts about the same,
    so the average is the results site's, from srbf.aggregate on the same problems."""
    from srbf.aggregate import average
    big, small = [i % 10 == 0 for i in range(500)], [i != 0 for i in range(10)]
    runs = [RunResult(model="m", benchmark="big", snapshot=_fixed(big)),
            RunResult(model="m", benchmark="small", snapshot=_fixed(small))]
    lb = leaderboard(runs, metrics=["numeric_recovery_val"], operator_arity=ARITY)
    want = average({"big": [[float(ok)] * 2 for ok in big], "small": [[float(ok)] * 2 for ok in small]}, "rate")
    label = DEFAULT_METRICS[0].label    # the columns carry the metric's label
    value = lb[label].iloc[0]
    assert value == pytest.approx(want.value, abs=1e-12)
    assert (lb[f"{label} lo"].iloc[0], lb[f"{label} hi"].iloc[0]) == pytest.approx((want.lo, want.hi), abs=1e-12)
    assert value > 0.3 > (50 + 9) / 510


def test_one_benchmark_shows_its_plain_rate():
    from srbf.analysis import per_benchmark_table
    runs = [RunResult(model="m", benchmark="b", snapshot=_fixed([True] * 8))]
    assert per_benchmark_table(runs, "numeric_recovery_val", operator_arity=ARITY).loc["m", "b"] == 1.0


def test_each_column_is_averaged_by_its_kind():
    from srbf.analysis import Metric, metric_kind, as_metric
    assert [metric_kind(as_metric(k)) for k in ("numeric_recovery_val", "symbolic_recovery", "mdl_ratio", "r2_val", "f1_score")] \
        == ["rate", "rate", "ratio", "median", "mean"]
    assert metric_kind(Metric("anything", "Anything", True, kind="median")) == "median"


def test_the_sites_scripts_use_the_package_averaging():
    """scripts/site_random_effects.py loads src/srbf/aggregate.py, so the site and srbf analyze average alike."""
    import importlib.util
    import srbf.aggregate as ag
    path = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "site_random_effects.py")
    spec = importlib.util.spec_from_file_location("site_random_effects_check", path)
    site = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(site)
    sets = [(20, 7.0), (3, 0.0), (8, 8.0), (100, 41.5)]
    a = ag.combine([ag.SetStat(n, k, k) for n, k in sets], "logit")
    b = site.combine([site.SetStat(n, k, k) for n, k in sets], "logit")
    assert (a.mu, a.lo, a.hi, a.pi_lo, a.pi_hi) == (b.mu, b.lo, b.hi, b.pi_lo, b.pi_hi)


# ---- benchmarks: the catalogs of one benchmark are one problem set ------------------------------------------------
_B, _B_POINTS, _C = [True] * 7 + [False], [False] * 4, [i % 3 == 0 for i in range(12)]


def _split_runs():
    """A benchmark B published as two catalogs (b and b-points; their rows both count from 0) and a benchmark C."""
    return [RunResult(model="m", benchmark="B", catalog="b", snapshot=_fixed(_B)),
            RunResult(model="m", benchmark="B", catalog="b-points", snapshot=_fixed(_B_POINTS)),
            RunResult(model="m", benchmark="C", snapshot=_fixed(_C))]


def test_the_catalogs_of_one_benchmark_are_averaged_as_one_problem_set():
    from srbf.aggregate import average
    lb = leaderboard(_split_runs(), metrics=["numeric_recovery_val"], operator_arity=ARITY)
    hits = lambda flags: [[float(ok)] * 2 for ok in flags]   # noqa: E731 - two draws per problem
    want = average({"B": hits(_B) + hits(_B_POINTS), "C": hits(_C)}, "rate")
    label = DEFAULT_METRICS[0].label
    assert (lb[label].iloc[0], lb[f"{label} lo"].iloc[0], lb[f"{label} hi"].iloc[0]) == pytest.approx(
        (want.value, want.lo, want.hi), abs=1e-12)
    assert lb["n_benchmarks"].iloc[0] == 2 and lb["n_expressions"].iloc[0] == 24   # row 0 of b and of b-points: two problems
    # the catalogs as sets of their own weigh B's problems differently
    apart = average({"b": hits(_B), "b-points": hits(_B_POINTS), "C": hits(_C)}, "rate")
    assert abs(apart.value - want.value) > 0.03


def test_every_catalog_its_own_benchmark_averages_as_before():
    runs = _runs()
    named = [RunResult(model=r.model, benchmark=r.benchmark, catalog=r.benchmark, scaling=r.scaling, snapshot=r.snapshot)
             for r in runs]
    assert leaderboard(named, operator_arity=ARITY).equals(leaderboard(runs, operator_arity=ARITY))
    assert all(r.catalog == r.benchmark for r in runs)


def test_the_per_benchmark_view_splits_by_benchmark_or_by_catalog():
    from srbf.analysis import per_benchmark_table
    runs = _split_runs()
    by_benchmark = per_benchmark_table(runs, "numeric_recovery_val", operator_arity=ARITY)
    assert list(by_benchmark.columns) == ["B", "C"] and by_benchmark.loc["m", "B"] == pytest.approx(7 / 12)
    by_catalog = per_benchmark_table(runs, "numeric_recovery_val", operator_arity=ARITY, by="catalog")
    assert list(by_catalog.columns) == ["C", "b", "b-points"] and by_catalog.loc["m", "b"] == pytest.approx(7 / 8)
    with pytest.raises(ValueError, match="by must be one of"):
        per_benchmark_table(runs, "numeric_recovery_val", operator_arity=ARITY, by="file")


def test_a_catalog_in_two_benchmarks_is_an_error():
    runs = [RunResult(model="m", benchmark="B", catalog="b", snapshot=_fixed([True])),
            RunResult(model="m", benchmark="C", catalog="b", snapshot=_fixed([False]))]
    with pytest.raises(ValueError, match="two benchmarks"):
        leaderboard(runs, metrics=["numeric_recovery_val"], operator_arity=ARITY)


def test_records_and_manifests_carry_the_catalog(tmp_path):
    import json
    import pickle

    import yaml

    from srbf.analysis import export_data, load_runs
    out = export_data(_split_runs(), str(tmp_path / "d.json"), metrics=["numeric_recovery_val"], operator_arity=ARITY)
    records = json.loads(open(out).read())["records"]
    assert [(r["benchmark"], r["catalog"]) for r in records] == [("B", "b"), ("B", "b-points"), ("C", "C")]
    with open(tmp_path / "r.pkl", "wb") as h:
        pickle.dump(_fixed([True]), h)
    with open(tmp_path / "manifest.yaml", "w") as h:
        yaml.safe_dump({"runs": [{"model": "m", "benchmark": "B", "catalog": "b-points", "path": "r.pkl"},
                                 {"model": "m", "benchmark": "C", "path": "r.pkl"}]}, h)
    runs = load_runs(str(tmp_path / "manifest.yaml"))
    assert [(r.benchmark, r.catalog) for r in runs] == [("B", "b-points"), ("C", "C")]
