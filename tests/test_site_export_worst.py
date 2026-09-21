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
