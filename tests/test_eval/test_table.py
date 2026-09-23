"""srbf.table: result files -> one row per problem and rung, the per-file cache, the variable spellings."""
import csv
import os
import pickle

import numpy as np
import pytest

from srbf.table import COLUMNS, ResultTree, build_table, judge_result_file, parse_index_bases

ENGINE = "acj-4-3"


@pytest.fixture(scope="module")
def engine():
    from simplipy import SimpliPyEngine

    return SimpliPyEngine.load(ENGINE, install=True)


def _snapshot(predictions, success=None):
    """Four problems whose ground truth is x1 + x2 (columns v1, v2); `predictions` are the stored prefixes."""
    n = len(predictions)
    x = np.linspace(0.5, 2.0, 16)
    y = x + 2.0 * x
    success = [p is not None for p in predictions] if success is None else success
    pred_y = [y.copy() if ok else None for ok in success]
    return {
        "eval_row_index": list(range(n)),
        "skeleton": [["+", "x1", "x2"]] * n,
        "ground_truth_prefix": [["+", "x1", "x2"]] * n,
        "variables": [["v1", "v2"]] * n,
        "predicted_skeleton_prefix": list(predictions),
        "predicted_expression_prefix": list(predictions),
        "prediction_success": list(success),
        "y": [y.copy() for _ in range(n)], "y_pred": pred_y,
        "y_val": [y.copy() for _ in range(n)], "y_pred_val": [None if p is None else p.copy() for p in pred_y],
        "fit_time": [0.25] * n, "n_support": [16] * n,
        "placeholder": [False] * n, "benchmark_eq_id": [f"p{i}" for i in range(n)],
    }


def _write(tmp_path, catalog, name, snapshot):
    d = tmp_path / "tree" / catalog
    d.mkdir(parents=True, exist_ok=True)
    with open(d / name, "wb") as fh:
        pickle.dump(snapshot, fh)
    return str(d / name)


def _col(rows, name):
    return [r[COLUMNS.index(name)] for r in rows]


def test_a_file_becomes_one_row_per_problem_with_the_identity_columns(tmp_path, engine):
    path = _write(tmp_path, "toy", "choices_000064.pkl", _snapshot([["+", "x1", "x2"], ["+", "x2", "x1"], ["*", "x1", "x2"], None]))
    rows = judge_result_file(path, method="m", draw=2, engine=engine)
    assert len(rows) == 4 and all(len(r) == len(COLUMNS) for r in rows)
    assert _col(rows, "model") == ["m"] * 4 and _col(rows, "draw") == [2] * 4
    assert _col(rows, "catalog") == ["toy"] * 4 and _col(rows, "rung") == [64] * 4 and _col(rows, "row") == [0, 1, 2, 3]
    assert _col(rows, "symbolic_recovery") == [1, 1, 0, 0]


def test_a_failed_problem_is_a_miss_in_the_rates_and_takes_the_worst_value_only_where_there_is_one(tmp_path, engine):
    path = _write(tmp_path, "toy", "choices_000001.pkl", _snapshot([["+", "x1", "x2"], None]))
    ok, failed = judge_result_file(path, method="m", engine=engine)
    at = COLUMNS.index
    assert failed[at("success")] == 0 and failed[at("numeric_recovery_val")] == 0 and failed[at("symbolic_recovery")] == 0
    assert failed[at("f1_score")] == 0.0          # a worst value exists: a failure takes it
    assert failed[at("log10_fvu_val")] == ""      # none exists: nothing is filled in
    assert failed[at("edit_distance")] == ""      # nor for the edit distances
    assert failed[at("ground_truth_mdl")] != ""   # the ground truth is described for every problem
    assert ok[at("success")] == 1 and ok[at("log10_fvu_val")] != ""


def test_index_spelled_variables_are_renamed_when_the_tree_says_how_they_are_counted(tmp_path, engine):
    e2e_style = _snapshot([["+", "x_0", "x_1"]])
    nesymres_style = _snapshot([["+", "x_1", "x_2"]])
    named = _snapshot([["+", "v1", "v2"]])   # a worker that answered in the names it was handed
    for first, snap in ((0, e2e_style), (1, nesymres_style), (None, named)):
        path = _write(tmp_path, f"toy{first}", "choices_000001.pkl", snap)
        (row,) = judge_result_file(path, method="m", engine=engine, first_index=first)
        assert row[COLUMNS.index("symbolic_recovery")] == 1
    path = _write(tmp_path, "toy_unmapped", "choices_000001.pkl", _snapshot([["+", "x_0", "x_1"]]))
    (row,) = judge_result_file(path, method="m", engine=engine)
    assert row[COLUMNS.index("symbolic_recovery")] == 0   # without the count the spelling is left alone


def _read(path):
    with open(path) as fh:
        return list(csv.reader(fh))


def test_the_cache_serves_a_file_until_it_changes(tmp_path):
    path = _write(tmp_path, "toy", "choices_000002.pkl", _snapshot([["+", "x1", "x2"], ["*", "x1", "x2"]]))
    _write(tmp_path, "toy", "choices_000004.shard-0-of-2.pkl", _snapshot([["+", "x1", "x2"]]))
    tree = ResultTree("m", 1, str(tmp_path / "tree"))
    cache, out = str(tmp_path / "cache"), str(tmp_path / "t.csv")
    first = build_table([tree], out, engine=ENGINE, workers=1, cache_dir=cache, log=None)
    rows_first = _read(out)
    assert (first.files, first.judged, first.from_cache, first.rows, first.errors) == (2, 2, 0, 3, [])
    assert rows_first[0] == COLUMNS and len(rows_first) == 4
    assert {r[COLUMNS.index("shard")] for r in rows_first[1:]} == {"", "0/2"}
    again = build_table([tree], out, engine=ENGINE, workers=1, cache_dir=cache, log=None)
    assert (again.judged, again.from_cache) == (0, 2) and _read(out) == rows_first
    with open(path, "wb") as fh:
        pickle.dump(_snapshot([["*", "x1", "x2"], ["*", "x1", "x2"]]), fh)
    os.utime(path, ns=(os.stat(path).st_atime_ns, os.stat(path).st_mtime_ns + 1_000_000))
    changed = build_table([tree], out, engine=ENGINE, workers=1, cache_dir=cache, log=None)
    assert (changed.judged, changed.from_cache) == (1, 1)
    assert sum(int(r[COLUMNS.index("symbolic_recovery")]) for r in _read(out)[1:]) == 1


def test_an_unreadable_file_is_reported_and_left_out(tmp_path):
    _write(tmp_path, "toy", "choices_000001.pkl", _snapshot([["+", "x1", "x2"]]))
    bad = tmp_path / "tree" / "toy" / "choices_000002.pkl"
    bad.write_bytes(b"not a pickle")
    report = build_table([ResultTree("m", 1, str(tmp_path / "tree"))], str(tmp_path / "t.csv"), engine=ENGINE, workers=1, log=None)
    assert report.rows == 1 and len(report.errors) == 1 and "choices_000002.pkl" in report.errors[0]


def test_tree_and_index_specs_parse_as_the_command_line_takes_them():
    assert ResultTree.parse("e2e:2:/data/e2e") == ResultTree("e2e", 2, "/data/e2e")
    assert ResultTree.parse("m:1:C:/x") == ResultTree("m", 1, "C:/x")
    with pytest.raises(ValueError, match="METHOD:DRAW:PATH"):
        ResultTree.parse("e2e/data")
    assert parse_index_bases(["e2e=0", "nesymres=1"]) == {"e2e": 0, "nesymres": 1}
    with pytest.raises(ValueError, match="METHOD=FIRST"):
        parse_index_bases(["e2e"])
