"""A data catalog without a test split (owner 2026-10-17): its problems have no validation points, srbf's table leaves
their validation rates empty, and every summary of the exporter leaves them out of those rates -- never a miss (0),
never a hit (an empty value read as true)."""
import csv
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = importlib.util.spec_from_file_location("site_export_v2", os.path.join(HERE, "..", "scripts", "site_export_v2.py"))
sx = importlib.util.module_from_spec(SPEC)
sys.modules["site_export_v2"] = sx
SPEC.loader.exec_module(sx)
from srbf.table import COLUMNS  # noqa: E402


def _write(tmp_path, rows):
    path = tmp_path / "rows_full_test.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        for r in rows:
            w.writerow([r.get(c, "") for c in COLUMNS])
    return str(tmp_path)


def _row(model, catalog, row, draw, *, val, fit, n_val):
    return {"model": model, "draw": draw, "catalog": catalog, "rung": 8, "row": row, "success": 1,
            "numeric_recovery_val": val, "numeric_recovery_fit": fit, "symbolic_recovery": 0, "symbolic_recovery_mask_fittable": 0,
            "symbolic_recovery_mask_none": "" if val == "" else 0, "numeric_recovery_relative_val": val,
            "numeric_recovery_relative_fit": fit, "skeleton_match_raw": 0, "n_validation": n_val, "n_support": 512}


def test_rows_without_validation_points_are_left_out_of_the_validation_rates(tmp_path):
    rows = [_row("m", "with-test", 0, d, val=1, fit=1, n_val=512) for d in (1, 2)]
    rows += [_row("m", "with-test", 1, d, val=0, fit=1, n_val=512) for d in (1, 2)]
    rows += [_row("m", "no-test", i, d, val="", fit=f, n_val=0) for i, f in ((0, 1), (1, 0)) for d in (1, 2)]
    data = sx.load_rows(_write(tmp_path, rows))
    no_test = data["m"][("no-test", 8)]
    assert all(r["numeric_recovery_val"] is None and r["numeric_recovery_fit"] is not None for r in no_test.values())
    cell = sx.summarize_cell(no_test, expected=None)
    assert cell["m"].get("numeric_recovery_val") is None            # no problem with the rate: no reading at all
    assert cell["m"]["numeric_recovery_fit"][:2] == [2, 1.0]        # the fit split is summarized as always
    with_test = sx.summarize_cell(data["m"][("with-test", 8)], expected=None)
    assert with_test["m"]["numeric_recovery_val"][:2] == [2, 1.0]
    ids, shares = sx.rate_matrix(no_test, "numeric_recovery_val")
    assert np.isnan(shares).all()
    _, fit_shares = sx.rate_matrix(no_test, "numeric_recovery_fit")
    assert sorted(fit_shares.tolist()) == [0.0, 1.0]


def test_an_empty_rate_with_validation_points_is_still_a_miss(tmp_path):
    rows = [_row("m", "with-test", 0, 1, val="", fit=1, n_val=512)]
    data = sx.load_rows(_write(tmp_path, rows))
    (r,) = data["m"][("with-test", 8)].values()
    assert r["numeric_recovery_val"] == 0.0
