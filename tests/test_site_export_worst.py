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
    registry = export.registry_json()
    keys = [m["key"] for m in registry]
    at = keys.index("symbolic_recovery")
    assert keys[at:at + 4] == ["symbolic_recovery", "symbolic_recovery_mask_fittable", "symbolic_recovery_mask_none", "skeleton_match_raw"]
    assert [m["short"] for m in registry[at:at + 4]] == ["SRRs", "SRRe", "SRRa", "SRRr"]


def test_metric_names_are_written_in_title_caps_with_one_word_for_each_side():
    """A name says Prediction and Ground Truth, never a second word for either, and every word of it is capitalized
    except the small ones and the mathematics."""
    import re
    small = {"of", "the", "to", "as", "a", "and", "log10", "log2"}
    for metric in export.registry_json():
        for text in (metric["label"], metric["group"], metric["short"].replace("GT", "Ground Truth")):
            words = [w for w in re.split(r"[^A-Za-z0-9²]+", text) if w and not w.isdigit()]
            lowered = [w for w in words if w[0].islower() and w not in small and w not in ("vNRR", "fNRR")]
            assert not lowered, (text, lowered)
            assert not re.search(r"\b(law|answer|pred|GT|skeleton)\b", text, re.I) or metric["key"] == "skeleton_match_raw", text


def test_a_property_of_one_expression_is_not_grouped_with_the_comparisons():
    registry = {m["key"]: m for m in export.registry_json()}
    alone = {k for k, m in registry.items() if m["group"] == "Expression Properties"}
    assert alone == {"predicted_mdl", "ground_truth_mdl", "predicted_skeleton_prefix_length", "skeleton_length", "predicted_n_constants",
                     "n_constants", "predicted_total_nestedness", "total_nestedness", "n_variables"}
    assert {k for k, m in registry.items() if m.get("every")} == export.EVERY_PROBLEM


def test_paired_wins_and_losses_count_better_and_worse_not_higher_and_lower():
    """The paired table's wins and losses are read as better / worse. For an error (lower is better) that is the
    opposite of the difference's sign, and for a ratio it is the distance from the ideal 1, which the sign cannot say."""
    a = {0: _row(success=1.0, log10_fvu_val=-6.0, mdl_ratio=1.1), 1: _row(success=1.0, log10_fvu_val=-2.0, mdl_ratio=0.5)}
    b = {0: _row(success=1.0, log10_fvu_val=-3.0, mdl_ratio=2.0), 1: _row(success=1.0, log10_fvu_val=-1.0, mdl_ratio=1.5)}
    paired = export.paired_cell(a, b)["m"]
    n, total, _, better, worse = paired["log10_fvu_val"]
    assert n == 2 and math.isclose(total, -3.0 - 1.0)                      # a's values are lower ...
    assert (better, worse) == (2, 0)                                       # ... which is better for an error
    n, _, _, better, worse = paired["mdl_ratio"]
    assert n == 2 and (better, worse) == (1, 1)                            # 1.1 beats 2.0; 0.5 loses to 1.5 (|log 0.5| > |log 1.5|)
