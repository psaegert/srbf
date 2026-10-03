"""Metrics a method reports about its own choice: a placeholder is no value, a metric without values is not listed, and
the selection score's histogram covers the scale the score is on."""
import csv
import importlib.util
import math
import os

SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "site_export_v2.py")
spec = importlib.util.spec_from_file_location("site_export_v2", SCRIPT)
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)

HEAD = ["model", "draw", "catalog", "rung", "row", "success", "numeric_recovery_val", "log10_fvu_val", "predicted_pareto_rank"]


def _cells(tmp_path, rows):
    with open(tmp_path / "rows_full_all.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEAD)
        w.writerows(rows)
    data = export.load_rows(str(tmp_path))
    return {m: {c: {str(r): export.summarize_cell(rows_, 2)} for (c, r), rows_ in per.items()} for m, per in data.items()}


def test_a_pareto_rank_never_computed_is_no_value_and_is_not_listed(tmp_path):
    # Flash-ANSR ranks by its score here and writes -1 for the front it never computed (PARETO_RANK_NOT_COMPUTED)
    cells = _cells(tmp_path, [["m", 1, "toy", 8, 0, 1, 1, -3.0, -1], ["m", 1, "toy", 8, 1, 1, 0, -1.0, -1]])
    assert "predicted_pareto_rank" not in cells["m"]["toy"]["8"]["m"]
    keys = [m["key"] for m in export.listed_metrics(cells)]
    assert "predicted_pareto_rank" not in keys
    assert "log10_fvu_val" in keys and "numeric_recovery_val" in keys


def test_a_pareto_rank_that_was_computed_is_kept_and_listed(tmp_path):
    cells = _cells(tmp_path, [["m", 1, "toy", 8, 0, 1, 1, -3.0, 0], ["m", 1, "toy", 8, 1, 1, 0, -1.0, 2]])
    assert cells["m"]["toy"]["8"]["m"]["predicted_pareto_rank"][:2] == [2, 2]
    assert "predicted_pareto_rank" in [m["key"] for m in export.listed_metrics(cells)]


def test_the_selection_score_histogram_covers_the_scale_the_score_is_on():
    # Flash-ANSR's stored score: log10 FVU on the n given points + 2 / (n log2 10) per bit of the formula's length
    lo, hi, tf = export.HIST_SPECS["predicted_score"]
    per_bit = 2.0 / (512 * math.log2(10.0))
    exact_80_bits = export.FVU_FLOOR + 80 * per_bit          # an exact fit 80 bits long: about -15.56
    mean_300_bits = 0.0 + 300 * per_bit                       # no better than the mean, 300 bits long: about 0.35
    assert tf is None and lo < exact_80_bits and mean_300_bits < hi
    assert hi - lo <= 25                                      # on the scale of log10 FVU, not of bits
    assert export.pp_spec("predicted_score")["lo"] == lo and export.pp_spec("predicted_score")["hi"] == hi
    desc = [m for m in export.METRICS if m[0] == "predicted_score"][0][9]
    assert "log10 FVU" in desc and "in bits;" not in desc
