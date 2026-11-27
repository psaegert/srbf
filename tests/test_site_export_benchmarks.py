"""The export names every catalog's benchmark, the unit the page averages over: from the rows' benchmark column
(srbf table), a catalog whose rows name none being a benchmark of its own."""
import csv
import importlib.util
import os

import pytest

from srbf.table import COLUMNS

_SPEC = importlib.util.spec_from_file_location(
    "site_export_v2", os.path.join(os.path.dirname(__file__), "..", "scripts", "site_export_v2.py"))
export = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(export)


def _rows(tmp_path, rows, columns=COLUMNS, name="rows_full_all.csv"):
    with open(tmp_path / name, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(columns)
        for r in rows:
            w.writerow([r.get(c, "") for c in columns])
    return str(tmp_path)


def _row(catalog, row, benchmark=""):
    return {"model": "m", "draw": 1, "catalog": catalog, "benchmark": benchmark, "rung": 1, "row": row, "success": 1}


def test_the_catalogs_carry_the_benchmark_the_rows_name(tmp_path):
    root = _rows(tmp_path, [_row("feynman", 0, "feynman"), _row("feynman-points", 0, "feynman"),
                            _row("feynman-points", 1, "feynman"), _row("nguyen", 0, "")])
    benchmarks: dict[str, str] = {}
    data = export.load_rows(root, benchmarks)
    assert benchmarks == {"feynman": "feynman", "feynman-points": "feynman", "nguyen": "nguyen"}
    assert set(data["m"]) == {("feynman", 1), ("feynman-points", 1), ("nguyen", 1)}   # the cells stay per catalog
    cats = export.catalog_meta(None, {"feynman": 1, "feynman-points": 2, "nguyen": 1}, benchmarks)
    assert {c["key"]: c["benchmark"] for c in cats} == benchmarks


def test_rows_without_the_column_make_every_catalog_its_own_benchmark(tmp_path):
    root = _rows(tmp_path, [_row("feynman", 0), _row("nguyen", 0)], columns=[c for c in COLUMNS if c != "benchmark"])
    benchmarks: dict[str, str] = {}
    export.load_rows(root, benchmarks)
    assert benchmarks == {"feynman": "feynman", "nguyen": "nguyen"}
    assert [c["benchmark"] for c in export.catalog_meta(None, {"feynman": 1, "nguyen": 1})] == ["feynman", "nguyen"]


def test_a_catalog_in_two_benchmarks_is_an_error(tmp_path):
    root = _rows(tmp_path, [_row("feynman", 0, "feynman"), _row("feynman", 1, "ai-feynman")])
    with pytest.raises(ValueError, match="two benchmarks"):
        export.load_rows(root, {})
    assert export.load_rows(root)                                     # without the mapping, the rows load as before
