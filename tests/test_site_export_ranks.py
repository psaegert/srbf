"""The Ranks view of the results site ships PAIRWISE outcomes (scripts/site_export_v2.py); the page turns them into
mean ranks for whatever methods and catalogs are selected. That only works if the pairwise identity holds exactly."""
import importlib.util
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent.parent


def _exporter() -> Any:
    spec = importlib.util.spec_from_file_location("site_export_v2", ROOT / "scripts" / "site_export_v2.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rows(values: list[float | None], key: str) -> dict[int, dict[str, float | None]]:
    sx = _exporter()
    return {i: {k: (v if k == key else None) for k in sx.RANK_KEYS} for i, v in enumerate(values)}


def test_a_law_without_an_answer_ranks_last_and_an_exact_recovery_first() -> None:
    sx = _exporter()
    assert sx.rank_score(None, False) == -math.inf and sx.rank_score(float("nan"), True) == -math.inf
    assert sx.rank_score(-math.inf, False) == math.inf                      # log10 FVU of an exact recovery
    assert sx.rank_score(0.5, None) == sx.rank_score(2.0, None) < sx.rank_score(1.0, None)   # a ratio: closer to 1 is better


def test_mean_ranks_follow_from_the_pairwise_outcomes_exactly() -> None:
    sx = _exporter()
    key, rng = "log10_fvu_val", np.random.default_rng(0)
    raw = {m: [None if rng.random() < 0.2 else (-math.inf if rng.random() < 0.1 else float(np.round(rng.normal(-3, 3), 0))) for _ in range(400)]
           for m in "abcd"}                                                 # rounded: plenty of ties, and laws nobody answers
    rows = {m: _rows(v, key) for m, v in raw.items()}
    direct = np.mean([rankdata([-sx.rank_score(raw[m][i], False) for m in "abcd"], method="average") for i in range(400)], axis=0)
    ki = sx.RANK_KEYS.index(key)
    for j, a in enumerate("abcd"):
        lost = 0.0
        for b in "abcd":
            if a != b:
                n, *w = sx.rank_pair_cell(rows[a], rows[b])
                wa, wb = w[2 * ki], w[2 * ki + 1]
                lost += (wb + 0.5 * (n - wa - wb)) / n
        assert abs(1 + lost - direct[j]) < 1e-12


def test_a_time_budget_buys_the_largest_rung_timed_within_it() -> None:
    sx = _exporter()
    timing = {"m": {"1": 0.2, "2": 0.4, "4": 0.9, "8": 2.5}}
    assert sx.rung_within(timing, "m", 1.0, {1, 2, 4, 8}) == 4
    assert sx.rung_within(timing, "m", 1.0, {1, 2}) == 2                     # a rung without results cannot be read
    assert sx.rung_within(timing, "m", 0.1, {1, 2, 4, 8}) is None            # cannot finish within the budget: sits out
    assert sx.rung_within(timing, "untimed", 10.0, {1}) is None


def test_a_metric_that_copies_another_in_every_cell_is_not_listed() -> None:
    """Recovery relative to the reference law IS numeric recovery wherever the targets are computed from the law;
    the menu lists it only once some cell tells the two apart (a catalog of measured data)."""
    sx = _exporter()
    same = {"m": {"numeric_recovery_val": [4, 10], "numeric_recovery_relative_val": [4, 10],
                  "numeric_recovery_fit": [5, 10], "numeric_recovery_relative_fit": [5, 10]}}
    apart = {"m": {"numeric_recovery_val": [0, 10], "numeric_recovery_relative_val": [7, 10],
                   "numeric_recovery_fit": [5, 10], "numeric_recovery_relative_fit": [5, 10]}}
    keys = lambda cells: {m["key"] for m in sx.listed_metrics(cells)}   # noqa: E731
    everything = {m["key"] for m in sx.registry_json()}
    assert keys({"a": {"nguyen": {"1": same, "2": same}}}) == everything - set(sx.COPY_OF)
    # one cell that tells validation apart brings that metric back, and only that one
    assert keys({"a": {"nguyen": {"1": same}, "measured": {"1": apart}}}) == everything - {"numeric_recovery_relative_fit"}
    # nothing published yet: nothing is known to be a copy, so nothing is dropped
    assert keys({}) == everything


def test_a_difference_ranks_by_its_distance_from_zero() -> None:
    """Predicted minus true counts are signed: five constants too few must not beat an exact count."""
    sx = _exporter()
    for key in ("n_constants_delta", "total_nestedness_delta"):
        higher, ideal = sx.METRIC_HIGHER[key], sx.IDEAL[key]
        assert higher is None and ideal == 0.0
        assert sx.rank_score(-5.0, higher, ideal) == sx.rank_score(5.0, higher, ideal) < sx.rank_score(0.0, higher, ideal)


def test_every_ranked_metric_says_what_is_better() -> None:
    sx = _exporter()
    kinds = {m[0]: m for m in sx.METRICS}
    assert sx.RANK_KEYS[0] == "log10_fvu_val"                              # the primary league
    assert len(set(sx.RANK_KEYS)) == len(sx.RANK_KEYS)
    for key in sx.RANK_KEYS:
        assert key in kinds, key
        assert sx.METRIC_HIGHER[key] is not None or key in sx.IDEAL, f"{key} has neither a direction nor an ideal"
        assert key not in sx.EVERY_PROBLEM, f"{key} is a property of the ground truth: every method would tie"
    assert "r2_val" not in sx.RANK_KEYS and "predicted_log_prob" not in sx.RANK_KEYS


def test_an_overlay_written_with_other_rank_keys_maps_onto_the_release_keys(tmp_path: Path) -> None:
    """ranks.js files merge in whichever order they load; the release's key list wins, the overlay maps onto it."""
    import json
    import shutil
    import subprocess

    import pytest
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    sx = _exporter()

    def write(name: str, keys: list[str], main: bool, pairs: dict[str, Any]) -> None:
        (tmp_path / name).write_text(sx.RANKS_JS % ('"r"', json.dumps(keys), "true" if main else "false", '["t1"]', "[1]", json.dumps(pairs), "{}", "{}"))

    write("release.js", ["a", "b", "c"], True, {"x|y": {"cat": {"t1": [10, 6, 3, 2, 1, 5, 4]}}})
    write("overlay.js", ["a", "c", "d"], False, {"x|z": {"cat": {"t1": [9, 7, 2, 4, 5, 1, 1]}}})
    probe = ("global.window={};require(process.argv[1]);require(process.argv[2]);"
             "process.stdout.write(JSON.stringify(window.RESULTS_V2_RANKS.r))")
    for first, second in (("release.js", "overlay.js"), ("overlay.js", "release.js")):
        merged = json.loads(subprocess.run(["node", "-e", probe, str(tmp_path / first), str(tmp_path / second)],
                                           capture_output=True, text=True, check=True).stdout)
        assert merged["keys"] == ["a", "b", "c"], first
        assert merged["pairs"]["x|y"]["cat"]["t1"] == [10, 6, 3, 2, 1, 5, 4], first
        assert merged["pairs"]["x|z"]["cat"]["t1"] == [9, 7, 2, None, None, 4, 5], first


def test_the_public_guard_reads_every_method_a_written_ranks_js_names() -> None:
    """The guard refuses to publish a ranks.js that names a private method; it must parse what the exporter writes."""
    import json
    sx = _exporter()
    spec = importlib.util.spec_from_file_location("public_guard", ROOT / "results-site" / "tests" / "public_guard.py")
    assert spec is not None and spec.loader is not None
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    text = sx.RANKS_JS % ('"r"', json.dumps(["a"]), "true", '["t1"]', "[1]",
                          json.dumps({"hidden-a|e2e": {"cat": {"t1": [3, 1, 1]}}}), json.dumps({"e2e": {"t1": 4}}),
                          json.dumps({"hidden-b|e2e": {"t1": [4, 4]}}))
    assert guard.rank_methods(text) == {"e2e", "hidden-a", "hidden-b"}
