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

HEAD = ["model", "draw", "catalog", "rung", "row", "success", "numeric_recovery_val", "symbolic_recovery",
        "predicted_expression", "ground_truth_expression", "log10_fvu_val", "skeleton_length"]


def _export(monkeypatch, root, site, *extra):
    monkeypatch.setattr(export, "METHODS", list(export.METHODS))   # main() adds the methods file's entries to it
    monkeypatch.setattr(export, "canonical_truths", lambda *a, **k: {})   # no SimpliPy engine for a toy
    monkeypatch.setattr(sys, "argv", ["site_export_v2.py", str(root), "rel", str(site / "data" / "rel" / "results.js"),
                                      "--sizes", str(root / "none.json"), "--site-dir", str(site), "--public", "pub",
                                      "--methods-file", str(root / "methods.json"), *extra])
    export.main()


def _tree(path):
    return {str(p.relative_to(path)): p.read_bytes() for p in sorted(path.rglob("*.js"))}


def test_a_private_overlay_carries_its_methods_files_and_leaves_the_release_alone(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    with open(root / "rows_full_all.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEAD)
        for m, expr in (("pub", "* x1 x2"), ("priv", "+ x1 x2")):
            w.writerow([m, 1, "toy", 8, 0, 1, 1, 1, expr, "* x1 x2", -3.0, 3])
            w.writerow([m, 1, "toy", 8, 1, 1, 0, 0, expr, "x1", -1.0, 1])
            w.writerow([m, 2, "toy", 8, 0, 1, 0, 0, expr, "* x1 x2", -2.0, 3])   # run 2 has one of two problems: not published
    entries = [[k, k.upper(), "draws", "#000", "other", "harness_tuned", None, None] for k in ("pub", "priv")]
    (root / "methods.json").write_text(json.dumps(entries))

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

    # the overlay holds its methods' files and their indices, shaped like the release's, and no ground truth
    text = (pdir / "results_v2_private.js").read_text()
    overlay = json.loads(text[len("window.RESULTS_V2_PRIVATE = "):text.rindex(";")])
    assert overlay["pp"] == {"priv": {"toy|8": [1]}}
    assert overlay["pred"] == {"priv": {"toy|8": [1]}}
    assert overlay["pred_block"] == export.PRED_BLOCK
    assert sorted(_tree(pdir / "pp")) == ["priv/toy/8.1.js"]
    assert sorted(_tree(pdir / "pred")) == ["priv/toy/8.1.0.js"]
    body = (pdir / "pred" / "priv" / "toy" / "8.1.0.js").read_text()
    assert '"priv|toy|8|1|0"]={"0":["+ x1 x2",3],"1":["+ x1 x2",0]}' in body
    release = json.loads((beside / "data" / "rel" / "results.js").read_text()[len("window.RESULTS_V2 = "):-2])
    assert set(release["pp"]) == set(release["pred"]) == {"pub"}
