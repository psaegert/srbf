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


# ---- subsets: the level between a benchmark and its problems -------------------------------------------------------
def _sub_row(catalog, row, subset, success=1, draw=1, rung=1):
    return {"model": "m", "draw": draw, "catalog": catalog, "benchmark": "srsd", "rung": rung, "row": row,
            "subset": subset, "success": success}


def test_the_rows_subsets_split_a_catalog_into_cells_per_subset(tmp_path):
    """A catalog whose rows name several subsets gets "subsets" and "subset_rows", and cells per subset keyed
    "<catalog>/<subset>" from its rows restricted to the subset; a catalog in one subset only its name; a catalog
    without the column is one subset, named after itself."""
    rows = [_sub_row("srsd", i, "easy" if i < 3 else ("hard" if i < 5 else ""), success=int(i < 3)) for i in range(6)]
    rows += [_sub_row("srsd-points", i, "points") for i in range(2)] + [_sub_row("nguyen", 0, "")]
    root = _rows(tmp_path, rows, columns=COLUMNS)
    benchmarks: dict[str, str] = {}
    subsets: dict[str, dict[int, str]] = {}
    data = export.load_rows(root, benchmarks, subsets)
    assert subsets == {"srsd": {0: "easy", 1: "easy", 2: "easy", 3: "hard", 4: "hard"}, "srsd-points": {0: "points", 1: "points"}}
    parts = export.catalog_subsets(subsets, {"srsd": 6, "srsd-points": 2, "nguyen": 1})
    assert parts == {"srsd": {"subsets": ["easy", "hard", "srsd"], "rows": [0, 0, 0, 1, 1, 2]}, "srsd-points": {"subset": "points"}}
    cats = {c["key"]: c for c in export.catalog_meta(None, {"srsd": 6, "srsd-points": 2, "nguyen": 1}, benchmarks, parts)}
    assert (cats["srsd"]["subsets"], cats["srsd"]["subset_rows"]) == (["easy", "hard", "srsd"], [0, 0, 0, 1, 1, 2])
    assert cats["srsd-points"]["subset"] == "points" and "subsets" not in cats["nguyen"] and "subset" not in cats["nguyen"]
    pdata, psizes = export.part_cells(data, parts)
    assert psizes == {"srsd/easy": 3, "srsd/hard": 2, "srsd/srsd": 1}
    assert sorted(pdata["m"]) == [("srsd/easy", 1), ("srsd/hard", 1), ("srsd/srsd", 1)]
    easy = export.summarize_cell(pdata["m"][("srsd/easy", 1)], psizes["srsd/easy"])
    hard = export.summarize_cell(pdata["m"][("srsd/hard", 1)], psizes["srsd/hard"])
    assert easy["state"] == "complete" and easy["m"]["success"][:2] == [3, 3.0] and hard["m"]["success"][:2] == [2, 0.0]
    # every subset sits where its catalog sits at a slot
    at = {"m": {"4": {"srsd": [1, 1, 0.0], "nguyen": [1, 1, 0.0]}}}
    assert export.with_part_places(at, parts)["m"]["4"]["srsd/hard"] == [1, 1, 0.0]


def test_a_problem_in_two_subsets_is_an_error(tmp_path):
    root = _rows(tmp_path, [_sub_row("srsd", 0, "easy"), _sub_row("srsd", 0, "hard", draw=2)], columns=COLUMNS)
    with pytest.raises(ValueError, match="two subsets"):
        export.load_rows(root, {}, {})


def test_the_release_carries_a_split_catalogs_subsets_and_their_cells(tmp_path, monkeypatch):
    """End to end: a catalog of two subsets comes out with "subsets" and "subset_rows", and with cells, histograms and
    pair cells per subset next to its own; the per-problem files stay per catalog."""
    import json
    import sys
    root, site = tmp_path / "root", tmp_path / "site"
    root.mkdir()
    head = ["model", "draw", "catalog", "benchmark", "subset", "rung", "row", "success", "numeric_recovery_val",
            "symbolic_recovery", "log10_fvu_val", "skeleton_length"]
    with open(root / "rows_full_all.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(head)
        for m in ("pa", "pb"):
            for i in range(4):
                ok = int(m == "pa" or i < 2)
                w.writerow([m, 1, "toy", "toy", "easy" if i < 2 else "hard", 8, i, 1, ok, ok, -3.0 + i, 3])
    (root / "methods.json").write_text(json.dumps([[k, k.upper(), "draws", "#000", "other", "harness_tuned", None, None]
                                                   for k in ("pa", "pb")]))
    monkeypatch.setattr(export, "METHODS", list(export.METHODS))
    monkeypatch.setattr(export, "canonical_truths", lambda *a, **k: {})
    monkeypatch.setattr(sys, "argv", ["site_export_v2.py", str(root), "rel", str(site / "data" / "rel" / "results.js"),
                                      "--sizes", str(root / "none.json"), "--site-dir", str(site), "--public", "pa,pb",
                                      "--methods-file", str(root / "methods.json")])
    export.main()
    out = site / "data" / "rel"
    release = json.loads((out / "results.js").read_text()[len("window.RESULTS_V2 = "):-2])
    toy = next(c for c in release["catalogs"] if c["key"] == "toy")
    assert (toy["subsets"], toy["subset_rows"]) == (["easy", "hard"], [0, 0, 1, 1])
    assert [c["key"] for c in release["catalogs"]] == ["toy"]                    # a subset is no problem set of the picker
    cells = release["cells"]["pb"]
    assert sorted(cells) == ["toy", "toy/easy", "toy/hard"]
    assert cells["toy/easy"]["8"]["m"]["numeric_recovery_val"][:2] == [2, 2.0]
    assert cells["toy/hard"]["8"]["m"]["numeric_recovery_val"][:2] == [2, 0.0] and cells["toy/hard"]["8"]["state"] == "complete"
    hist = (out / "hist" / "log10_fvu_val.js").read_text()
    assert '"toy/easy"' in hist and '"toy/hard"' in hist
    paired = (out / "paired" / "numeric_recovery_val.js").read_text()
    assert '"toy/easy"' in paired and '"toy/hard"' in paired
    assert not any(part in str(p.relative_to(out)) for p in out.rglob("*") for part in ("easy", "hard"))   # no files of its own
    assert release["pv"] == {"pa": {"8": ["toy"]}, "pb": {"8": ["toy"]}}


def test_a_subset_with_a_slash_is_refused():
    """The page keys a subset's cells "<catalog>/<subset>"."""
    with pytest.raises(ValueError, match="may not contain"):
        export.catalog_subsets({"srsd": {0: "easy", 1: "a/b"}}, {"srsd": 2})


def test_a_release_refuses_rows_that_name_no_benchmark_or_subset(tmp_path):
    """Strict by default where it publishes: a row without its benchmark or subset (an older table, or one written with
    srbf table --lenient-metadata for a catalog it could not read) is refused, every such catalog named; lenient
    loading takes each catalog's name, as local builds may."""
    rows = [_sub_row("srsd", 0, "easy"), _sub_row("srsd", 1, "hard"), _sub_row("gone", 0, ""),
            {**_sub_row("lost", 0, "x"), "benchmark": ""}]
    root = _rows(tmp_path, rows, columns=COLUMNS)
    with pytest.raises(ValueError, match=r"2 catalog\(s\) name no benchmark or subset: gone \(no subset\); lost \(no benchmark\)"):
        export.load_rows(root, {}, {}, strict=True)
    benchmarks: dict[str, str] = {}
    subsets: dict[str, dict[int, str]] = {}
    export.load_rows(root, benchmarks, subsets)                      # lenient: the catalog's own name
    assert benchmarks["lost"] == "lost" and "gone" not in subsets
    (tmp_path / "old").mkdir()
    old = _rows(tmp_path / "old", [_row("feynman", 0, "feynman")], columns=[c for c in COLUMNS if c != "subset"])
    with pytest.raises(ValueError, match="feynman \\(no subset\\)"):
        export.load_rows(old, {}, {}, strict=True)                   # a table written before the subset column
