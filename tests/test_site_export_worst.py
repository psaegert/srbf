"""The results site reads what srbf computes: the metrics a failed problem counts in, and how an unbounded one is
summarized."""
import importlib.util
import math
import os

from srbf.result_processing import WORST_VALUE

_SPEC = importlib.util.spec_from_file_location(
    "site_export_v2", os.path.join(os.path.dirname(__file__), "..", "scripts", "site_export_v2.py"))
export = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(export)


def _row(**values):
    row = {k: 0.0 for k in export.RATE_KEYS}
    row.update({k: None for k in export.CONT_KEYS})
    row.update(values)
    return row


def test_the_site_counts_a_failed_problem_where_srbf_does():
    assert export.WORST == WORST_VALUE
    registry = {m["key"]: m for m in export.registry_json()}
    assert {k: m["worst"] for k, m in registry.items() if "worst" in m} == WORST_VALUE


def test_an_unbounded_metric_ships_no_sums_and_names_the_histogram_its_median_is_read_from():
    registry = {m["key"]: m for m in export.registry_json()}
    assert registry["r2_val"]["median_via"] == "log10_fvu_val" and registry["r2_fit"]["median_via"] == "log10_fvu_fit"
    assert registry["r2_val"]["hist"]["lo"] < 0                           # no floor at 0
    assert "r2_val" not in export.RANK_KEYS and "r2_val" not in export.PAIRED_KEYS   # it orders answers as the FVU does
    rows = {0: _row(success=1.0, r2_val=0.5), 1: _row(success=1.0, r2_val=-1e300), 2: _row(success=1.0, r2_val=-math.inf), 3: _row()}
    cell = export.summarize_cell(rows, None)
    assert cell["m"]["r2_val"] == [3, 2, 0.0, 0.0]                        # 3 answers, 2 finite, and no sum to overflow


def test_a_cell_names_the_laws_a_metric_can_be_defined_for():
    rows = {0: _row(success=1.0, n_constants=2.0, n_constants_ratio=1.0), 1: _row(n_constants=1.0), 2: _row(success=1.0, n_constants=0.0)}
    cell = export.summarize_cell(rows, None)
    assert cell["n"] == 3 and cell["e"] == {"n_constants_ratio": 2}        # the third law has no constant to count
    assert cell["m"]["n_constants_ratio"][0] == 1                         # one of the two has a value: the other failed


def test_a_cell_names_how_many_of_its_values_were_filled_in():
    """The page can then leave the failed predictions out again: off the sums, out of the worst value's bin."""
    rows = {0: _row(success=1.0, f1_score=0.5, recall_score=0.25, edit_distance_norm=0.5), 1: _row(f1_score=0.0, recall_score=0.0),
            2: _row(success=1.0, f1_score=0.0, recall_score=0.0, edit_distance_norm=1.0), 3: _row()}   # an answer may score the worst value itself
    cell = export.summarize_cell(rows, None)
    assert cell["w"] == {"f1_score": 1, "recall_score": 1}
    assert cell["m"]["f1_score"][:3] == [3, 3, 0.5] and cell["m"]["recall_score"][:3] == [3, 3, 0.25]
    assert cell["m"]["edit_distance_norm"][:3] == [2, 2, 1.5]             # the answers that were made, nothing filled in
    assert "w" not in export.summarize_cell({0: rows[0], 2: rows[2]}, None)


def test_a_paired_contrast_ships_in_both_readings():
    a = {0: _row(success=1.0, f1_score=0.8), 1: _row(success=1.0, f1_score=0.6), 2: _row(f1_score=0.0)}
    b = {0: _row(success=1.0, f1_score=0.5), 1: _row(f1_score=0.0), 2: _row(success=1.0, f1_score=0.4)}
    paired = export.paired_cell(a, b)["m"]
    n, total = paired["f1_score"][:2]
    assert n == 3 and math.isclose(total, 0.3 + 0.6 - 0.4)               # every law, a failure at 0
    n, total = paired["f1_score" + export.ANSWERED][:2]
    assert n == 1 and math.isclose(total, 0.3)                             # the one law both answered
    assert "log10_fvu_val" + export.ANSWERED not in paired                 # a metric without a worst value has one reading


def test_a_rate_the_rows_do_not_carry_is_absent_and_not_zero():
    """Rows judged before a metric existed have no column for it: the cell then has no such rate, where a rate
    of 0 would say that the method recovered nothing."""
    rows = {0: _row(success=1.0, symbolic_recovery=1.0), 1: _row(success=1.0)}
    for row in rows.values():
        row["symbolic_recovery_mask_fittable"] = None
        row["symbolic_recovery_mask_none"] = None
    cell = export.summarize_cell(rows, None)
    assert cell["m"]["symbolic_recovery"] == [1, 2]
    assert "symbolic_recovery_mask_fittable" not in cell["m"] and "symbolic_recovery_mask_none" not in cell["m"]
    rows[0]["symbolic_recovery_mask_fittable"], rows[1]["symbolic_recovery_mask_fittable"] = 1.0, 0.0
    assert export.summarize_cell(rows, None)["m"]["symbolic_recovery_mask_fittable"] == [1, 2]


def test_the_three_levels_of_symbolic_recovery_are_in_the_registry():
    keys = [m["key"] for m in export.registry_json()]
    at = keys.index("symbolic_recovery")
    assert keys[at:at + 3] == ["symbolic_recovery", "symbolic_recovery_mask_fittable", "symbolic_recovery_mask_none"]
