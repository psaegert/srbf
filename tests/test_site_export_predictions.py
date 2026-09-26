"""The Predictions view's files: the formulas as the judge read them, rounded for reading, one file per finished run."""
import csv
import importlib.util
import json
import os

SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "site_export_v2.py")
spec = importlib.util.spec_from_file_location("site_export_v2", SCRIPT)
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)

HEAD = ["model", "draw", "catalog", "rung", "row", "success", "numeric_recovery_val", "symbolic_recovery",
        "predicted_expression", "ground_truth_expression"]


def _rows(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEAD)
        w.writerows(rows)


def test_numbers_are_rounded_for_reading_and_integers_stay():
    assert export.round_prefix("* 0.39894228 exp / neg pow x1 2 2") == "* 0.3989 exp / neg pow x1 2 2"
    assert export.round_prefix("+ 1.6018267823270094e-08 x1") == "+ 1.602e-08 x1"
    assert export.round_prefix("* np.pi x1") == "* np.pi x1"


def test_only_a_finished_run_is_published_and_a_failure_stays_visible(tmp_path):
    gt = "* x1 x2"
    _rows(tmp_path / "rows_full_all.csv", [
        ["m", 1, "toy", 8, 0, 1, 1, 1, "* 30.84302119 x1", gt],
        ["m", 1, "toy", 8, 1, 0, 0, 0, "", gt],             # no usable formula: shown as such, and it completes the run
        ["m", 2, "toy", 8, 0, 1, 0, 0, "+ x1 x2", gt],      # run 2 has one of two problems: still in progress
        ["private", 1, "toy", 8, 0, 1, 1, 1, "* x1 x2", gt],
    ])
    pred, truth = export.load_expressions(str(tmp_path), ["m"])
    index = export.write_predictions(str(tmp_path / "out"), "rel", pred, truth, {"toy": 2})
    assert index == {"m": {"toy|8": [1]}}
    text = (tmp_path / "out" / "pred" / "m" / "toy" / "8.1.0.js").read_text()
    body = json.loads(text[text.index('"m|toy|8|1|0"]=') + len('"m|toy|8|1|0"]='):text.rindex(";})();")])
    assert body == {"0": ["* 30.84 x1", 3], "1": None}
    assert not (tmp_path / "out" / "pred" / "m" / "toy" / "8.2.0.js").exists()
    assert not (tmp_path / "out" / "pred" / "private").exists()
    assert (tmp_path / "out" / "pred" / "truth" / "toy.0.js").exists()


def test_rows_without_expressions_publish_nothing(tmp_path):
    with open(tmp_path / "rows_full_all.csv", "w") as fh:
        fh.write("model,draw,catalog,rung,row,success\nm,1,toy,8,0,1\n")
    pred, truth = export.load_expressions(str(tmp_path), ["m"])
    assert export.write_predictions(str(tmp_path / "out"), "rel", pred, truth, {"toy": 1}) == {}
