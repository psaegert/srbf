"""Every model is run twice with different seeds (two uncalibrated draws). A cell pools the draws that are COMPLETE for
that catalog and rung, and only those: a draw still running is not a random subset of the problems. Rows are keyed by
(draw, row), so a paired contrast pairs a problem with itself within a draw and never across draws."""
import importlib.util
import os

_SPEC = importlib.util.spec_from_file_location(
    "site_export_v2", os.path.join(os.path.dirname(__file__), "..", "scripts", "site_export_v2.py"))
export = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(export)


def _row(**values):
    row = {k: 0.0 for k in export.RATE_KEYS}
    row.update({k: None for k in export.CONT_KEYS})
    row.update(values)
    return row


def _draw(d, hits, n=4):
    return {(d, i): _row(success=1.0, numeric_recovery_val=1.0 if i in hits else 0.0) for i in range(n)}


def test_a_cell_pools_the_complete_draws_only():
    both = {**_draw(1, {0, 1}), **_draw(2, {0, 1, 2})}
    cell = export.summarize_cell(both, 4)
    assert cell["state"] == "complete" and cell["d"] == 2 and cell["n"] == 8
    assert cell["m"]["numeric_recovery_val"] == [5, 8]
    one_running = {**_draw(1, {0, 1}), **dict(list(_draw(2, {0, 1, 2}).items())[:3])}   # draw 2 has 3 of 4 problems
    cell = export.summarize_cell(one_running, 4)
    assert cell["state"] == "complete" and cell["d"] == 1 and cell["n"] == 4
    assert cell["m"]["numeric_recovery_val"] == [2, 4]                                  # draw 1 alone
    none = {**dict(list(_draw(1, {0, 1}).items())[:2]), **dict(list(_draw(2, {0, 1, 2}).items())[:3])}
    cell = export.summarize_cell(none, 4)
    assert cell["state"] == "partial" and cell["d"] == 1 and cell["n"] == 3               # the fullest draw, marked partial
    assert export.summarize_cell(_draw(1, {0}), None)["d"] == 1                          # no expected count: every draw counts


def test_a_paired_contrast_pairs_within_a_draw_and_skips_a_running_one():
    a = {**_draw(1, {0, 1}), **_draw(2, {0, 1, 2})}
    b = {**_draw(1, {1, 2}), **dict(list(_draw(2, {0}).items())[:2])}                    # b's draw 2 is running
    pc = export.paired_cell(a, b, 4)
    assert pc["n"] == 4                                                                   # draw 1 only, four problems
    assert pc["m"]["numeric_recovery_val"] == [1, 1, 1, 1]                                # n11, n10, n01, n00 of draw 1
    b_done = {**_draw(1, {1, 2}), **_draw(2, {0})}
    pc = export.paired_cell(a, b_done, 4)
    assert pc["n"] == 8 and pc["m"]["numeric_recovery_val"] == [2, 3, 1, 2]
    rk = export.rank_pair_cell(a, b, 4)
    assert rk[0] == 4


def test_progress_counts_finished_catalog_rungs_per_draw_against_the_plan(tmp_path):
    # the plan: draw 1 holds catalog a at rungs 1 and 2 (a split in two shards at rung 2), draw 2 holds a at rung 1
    (tmp_path / "units_hyb_d1.txt").write_text("1 a 1\n1 a 2 0/2\n1 a 2 1/2\n1 a 4096\n")
    (tmp_path / "units_hyb_d2.txt").write_text("2 a 1\n")
    plan = export.planned_cells(str(tmp_path), "hyb", lambda r: r <= 256)       # rung 4096 is not published
    assert plan == {(1, "a", 1), (1, "a", 2), (2, "a", 1)}
    rows = {("a", 1): {**_draw(1, set(), n=4), **dict(list(_draw(2, set(), n=4).items())[:3])},   # draw 2 not finished
            ("a", 2): _draw(1, set(), n=4)}
    assert export.status_of(rows, {"a": 4}, plan) == [2, 3]
    assert export.status_of(rows, {"a": 4}, None) == [2, None]                  # no plan: no total, never a guess
    assert export.progress_of(rows, {"a": 4}, plan) == {"1": [1, 2], "2": [1, 1]}   # per budget: draw 2 of rung 1 is open
    assert export.progress_of(rows, {"a": 4}, None) == {"1": [1, None], "2": [1, None]}
    assert export.progress_of({}, {"a": 4}, plan) == {"1": [0, 2], "2": [0, 1]}    # a planned budget without data is shown
    assert export.planned_cells(str(tmp_path / "none"), "hyb", lambda r: True) is None


def test_a_cell_does_not_depend_on_the_order_its_rows_were_read_in():
    # both draws half done: a tie for the fullest draw, decided by the draw number, not by which draw was read first
    d1 = {(1, i): _row(success=1.0, numeric_recovery_val=1.0, log10_fvu_val=0.1 * (i + 1)) for i in range(2)}
    d2 = {(2, i): _row(success=1.0, numeric_recovery_val=0.0, log10_fvu_val=1.0 / (i + 3)) for i in range(2)}
    first, second = export.summarize_cell({**d1, **d2}, 4), export.summarize_cell({**d2, **d1}, 4)
    assert first == second and first["m"]["numeric_recovery_val"] == [2, 2]            # draw 1 stands in
    rows = {(1, i): _row(success=1.0, log10_fvu_val=v) for i, v in enumerate([0.1, 1e16, -1e16, 0.3])}
    assert export.summarize_cell(rows, 4) == export.summarize_cell(dict(reversed(list(rows.items()))), 4)
