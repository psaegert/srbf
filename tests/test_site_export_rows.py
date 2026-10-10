"""The export holds every judged row in memory: a compact read-only Row instead of a dict per row (2.85 million dicts
took 5.3 GB, and the machine's memory guard stopped the export, 2026-10-09). A Row must read exactly like the dict."""
import csv
import importlib.util
import math
import os
import sys

_SPEC = importlib.util.spec_from_file_location(
    "site_export_v2", os.path.join(os.path.dirname(__file__), "..", "scripts", "site_export_v2.py"))
export = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(export)


def _values():
    vals = {k: float(i) for i, k in enumerate(export.ROW_KEYS)}
    first, second, third, fourth, fifth = export.ROW_KEYS[:5]
    vals.update({first: None, second: math.nan, third: -0.0, fourth: math.inf, fifth: -math.inf})
    return vals


def test_a_row_reads_like_the_dict_it_replaces():
    vals = _values()
    row = export.Row(vals)
    assert list(row) == list(vals) and len(row) == len(vals)
    nan_key = export.ROW_KEYS[1]                                     # a NaN equals no other NaN object: compared below
    assert {k: v for k, v in row.items() if k != nan_key} == {k: v for k, v in vals.items() if k != nan_key}
    first, second, third, fourth, fifth = export.ROW_KEYS[:5]
    assert row[first] is None and row.get(first) is None              # missing stays missing
    assert math.isnan(row[second])                                    # a NaN stays a NaN, not None
    assert math.copysign(1.0, row[third]) == -1.0                     # the sign of zero is kept
    assert row[fourth] == math.inf and row[fifth] == -math.inf
    for k, v in vals.items():
        if k not in (first, second):
            assert row[k] == v and type(row[k]) is float and row.get(k) == v
    assert "not a metric" not in row and row.get("not a metric") is None and row.get("not a metric", 7) == 7
    assert first in row
    try:
        row["not a metric"]
    except KeyError:
        pass
    else:
        raise AssertionError("a key the row does not hold must raise KeyError, like a dict")


def test_a_row_takes_a_fraction_of_the_dict():
    row = export.Row(_values())
    assert not hasattr(row, "__dict__")
    size = sys.getsizeof(row) + sys.getsizeof(row._values)
    as_dict = sys.getsizeof(dict(_values())) + 24 * len(export.ROW_KEYS)   # the dict and its float objects
    assert size * 3 < as_dict, (size, as_dict)


def test_load_rows_gives_rows_with_the_values_the_csv_holds(tmp_path):
    cols = ["model", "draw", "catalog", "rung", "row"] + export.ROW_KEYS
    with open(tmp_path / "rows_full_all.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        w.writerow(["m", 1, "cat", 64, 0] + ["" if i == 3 else str(i / 4) for i in range(len(export.ROW_KEYS))])
    data = export.load_rows(str(tmp_path))
    row = data["m"][("cat", 64)][(1, 0)]
    assert isinstance(row, export.Row)
    k3 = export.ROW_KEYS[3]
    expected = 0.0 if k3 in export.RATE_KEYS else None                 # an empty rate is a miss, an empty value is missing
    assert row[k3] == expected and row[export.ROW_KEYS[4]] == 1.0
