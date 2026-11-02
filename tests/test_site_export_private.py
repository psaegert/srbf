"""A private overlay carries its methods' per-problem files (the Predictions and Correlations views'), and the release
it sits beside comes out exactly as it does without one."""
import csv
import importlib.util
import json
import os
import sys

SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "site_export_v2.py")
spec = importlib.util.spec_from_file_location("site_export_v2", SCRIPT)
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)
METHODS = list(export.METHODS)   # main() adds the methods file's entries to the list it is given

HEAD = ["model", "draw", "catalog", "rung", "row", "success", "numeric_recovery_val", "symbolic_recovery",
        "predicted_expression", "ground_truth_expression", "log10_fvu_val", "skeleton_length"]


def _export(monkeypatch, root, site, *extra):
    monkeypatch.setattr(export, "METHODS", list(METHODS))
    monkeypatch.setattr(export, "canonical_truths", lambda *a, **k: {})   # no SimpliPy engine for a toy
    monkeypatch.setattr(sys, "argv", ["site_export_v2.py", str(root), "rel", str(site / "data" / "rel" / "results.js"),
                                      "--sizes", str(root / "none.json"), "--site-dir", str(site), "--public", "pub",
                                      "--methods-file", str(root / "methods.json"), *extra])
    export.main()


def _tree(path):
    return {str(p.relative_to(path)): p.read_bytes() for p in sorted(path.rglob("*.js"))}


def _unit(path):
    """The overlay a unit's unit.js pushes onto window.RESULTS_V2_PRIVATE_UNITS."""
    text = (path / "unit.js").read_text()
    head = "(window.RESULTS_V2_PRIVATE_UNITS=window.RESULTS_V2_PRIVATE_UNITS||[]).push("
    assert text.startswith(head) and text.endswith(");\n")
    return json.loads(text[len(head):-3])


def _rows(root, methods):
    with open(root / "rows_full_all.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEAD)
        for m, expr in methods:
            w.writerow([m, 1, "toy", 8, 0, 1, 1, 1, expr, "* x1 x2", -3.0, 3])
            w.writerow([m, 1, "toy", 8, 1, 1, 0, 0, expr, "x1", -1.0, 1])
            w.writerow([m, 2, "toy", 8, 0, 1, 0, 0, expr, "* x1 x2", -2.0, 3])   # run 2 has one of two problems: not published
    entries = [[k, k.upper(), "draws", "#000", "other", "harness_tuned", None, None] for k, _ in methods]
    (root / "methods.json").write_text(json.dumps(entries))


def test_a_private_overlay_carries_its_methods_files_and_leaves_the_release_alone(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    _rows(root, (("pub", "* x1 x2"), ("priv", "+ x1 x2")))

    alone, beside = tmp_path / "alone", tmp_path / "beside"
    _export(monkeypatch, root, alone)
    pdir = beside / "private" / "rel"
    _export(monkeypatch, root, beside, "--private", "priv", "--private-dir", str(pdir))

    # the release's per-problem files, the ground truth's included, are the same bytes with or without the overlay
    for sub in ("pp", "pred"):
        assert _tree(alone / "data" / "rel" / sub) == _tree(beside / "data" / "rel" / sub)
    assert (beside / "data" / "rel" / "pp" / "truth" / "toy.js").exists()
    assert (beside / "data" / "rel" / "pred" / "truth" / "toy.0.js").exists()
    assert not any("priv" in p for p in _tree(beside / "data" / "rel"))

    # the method's unit holds its files and their indices, shaped like the release's, and no ground truth
    unit = pdir / "m" / "priv"
    overlay = _unit(unit)
    assert [m["key"] for m in overlay["methods"]] == ["priv"]
    assert overlay["pp"] == {"priv": {"toy|8": [1]}}
    assert overlay["pred"] == {"priv": {"toy|8": [1]}}
    assert overlay["pred_block"] == export.PRED_BLOCK
    assert sorted(_tree(unit / "pp")) == ["priv/toy/8.1.js"]
    assert sorted(_tree(unit / "pred")) == ["priv/toy/8.1.0.js"]
    body = (unit / "pred" / "priv" / "toy" / "8.1.0.js").read_text()
    assert '"priv|toy|8|1|0"]={"0":["+ x1 x2",3],"1":["+ x1 x2",0]}' in body
    assert not (pdir / "p").exists() and not (pdir / "results_v2_private.js").exists()
    release = json.loads((beside / "data" / "rel" / "results.js").read_text()[len("window.RESULTS_V2 = "):-2])
    assert set(release["pp"]) == set(release["pred"]) == {"pub"}


def test_every_private_method_is_a_unit_of_its_own_and_a_private_pair_is_a_unit_of_both(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    _rows(root, (("pub", "* x1 x2"), ("alpha", "+ x1 x2"), ("beta", "- x1 x2")))
    site = tmp_path / "site"
    pdir = site / "private" / "rel"
    (pdir / "pp").mkdir(parents=True)   # the one-overlay layout the units replace: removed
    (pdir / "results_v2_private.js").write_text("window.RESULTS_V2_PRIVATE = {};\n")
    _export(monkeypatch, root, site, "--private", "beta,alpha", "--private-dir", str(pdir))
    assert sorted(p.name for p in pdir.iterdir()) == ["m", "p"]
    assert sorted(p.name for p in (pdir / "m").iterdir()) == ["alpha", "beta"]
    assert [str(p.relative_to(pdir / "p")) for p in sorted((pdir / "p").glob("*/*"))] == ["alpha/beta"]

    def pairs(unit):
        found = set()
        for sub in ("ranks", "paired"):
            for f in (unit / sub).glob("*.js"):
                text = f.read_text()
                found |= {pk for pk in json.loads(text[text.index("var P=") + 6:text.index(";Object.keys(P)")]).get(f.stem, {})}
        return found

    # a method's unit names no other private method anywhere, and holds its cells against the release only
    for m, other in (("alpha", "beta"), ("beta", "alpha")):
        unit = pdir / "m" / m
        assert [x["key"] for x in _unit(unit)["methods"]] == [m]
        assert {pk for pk in pairs(unit)} == {f"{m}|pub"}
        assert not any(other in str(p) or other.encode() in p.read_bytes() for p in unit.rglob("*.js")), (m, other)
    # the pair's unit: no method of its own, and only the two methods' cells against each other
    pair = pdir / "p" / "alpha" / "beta"
    assert _unit(pair)["methods"] == [] and _unit(pair)["pair"] == ["alpha", "beta"]
    assert pairs(pair) == {"beta|alpha"} or pairs(pair) == {"alpha|beta"}
    assert not (pair / "pp").exists() and not (pair / "pv").exists()

    # a method that stops being private takes its units with it
    _export(monkeypatch, root, site, "--private", "alpha", "--private-dir", str(pdir))
    assert sorted(p.name for p in (pdir / "m").iterdir()) == ["alpha"] and not (pdir / "p").exists()
