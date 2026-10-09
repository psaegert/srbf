"""scripts/rekey_table_cache.py: a change that leaves judging alone carries the table cache over instead of re-judging
every file, and its check catches a carried entry that a fresh judgment would not reproduce."""
import csv
import importlib.util
import os
import pickle

import numpy as np

import srbf.table as table
from srbf.table import ResultTree, build_table

_SPEC = importlib.util.spec_from_file_location(
    "rekey_table_cache", os.path.join(os.path.dirname(__file__), "..", "scripts", "rekey_table_cache.py"))
rekey = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rekey)

ENGINE = "acj-4-3"


def _write(root, catalog, name, predictions):
    n, x = len(predictions), np.linspace(0.5, 2.0, 16)
    y = 3.0 * x
    snap = {"eval_row_index": list(range(n)), "skeleton": [["+", "x1", "x2"]] * n,
            "ground_truth_prefix": [["+", "x1", "x2"]] * n, "variables": [["v1", "v2"]] * n,
            "predicted_skeleton_prefix": list(predictions), "predicted_expression_prefix": list(predictions),
            "prediction_success": [True] * n, "y": [y.copy()] * n, "y_pred": [y.copy()] * n,
            "y_val": [y.copy()] * n, "y_pred_val": [y.copy()] * n, "fit_time": [0.25] * n, "n_support": [16] * n,
            "placeholder": [False] * n, "benchmark_eq_id": [f"p{i}" for i in range(n)]}
    snap["predicted_log_prob"] = [float("nan")] * n                         # no log probability: NaN, as the prior's
    d = root / "tree" / catalog
    d.mkdir(parents=True, exist_ok=True)
    with open(d / name, "wb") as fh:
        pickle.dump(snap, fh)


def _read(path):
    with open(path) as fh:
        return list(csv.reader(fh))


def test_a_cache_is_carried_to_a_new_fingerprint_and_checked(tmp_path, monkeypatch):
    for k, preds in enumerate(([["+", "x1", "x2"]], [["*", "x1", "x2"], ["+", "x2", "x1"]], [["-", "x1", "x2"]])):
        _write(tmp_path, f"c{k}", "choices_000001.pkl", preds)
    tree = ResultTree("m", 1, str(tmp_path / "tree"))
    cache, out = str(tmp_path / "cache"), str(tmp_path / "t.csv")
    old = table.judge_fingerprint(ENGINE)
    build_table([tree], out, engine=ENGINE, workers=1, cache_dir=cache, log=None)
    rows = _read(out)
    _write(tmp_path, "c3", "choices_000001.pkl", [["+", "x1", "x2"]])        # judged after the change: no old entry

    for mod in (table, rekey):                                                # srbf's source moved
        monkeypatch.setattr(mod, "judge_fingerprint", lambda engine_name: "f" * 16)
    r = rekey.rekey([tree], cache, old, engine=ENGINE)
    assert (r["files"], len(r["carried"]), r["present"], r["missing"]) == (4, 3, 0, 1)
    assert rekey.check(r["carried"], cache, ENGINE, 10) == []
    again = build_table([tree], out, engine=ENGINE, workers=1, cache_dir=cache, log=None)
    assert (again.judged, again.from_cache) == (1, 3)
    assert [row for row in _read(out) if row[table.COLUMNS.index("catalog")] != "c3"] == rows

    twice = rekey.rekey([tree], cache, old, engine=ENGINE)                    # never replaces an entry
    assert (len(twice["carried"]), twice["present"]) == (0, 4)

    # a carried entry a fresh judgment does not reproduce (as after a change that DID touch judging) is caught
    tree_path, entry = r["carried"][0][1], table._cache_path(cache, table._cache_key(tree, r["carried"][0][1], "f" * 16))
    stored = pickle.loads(open(entry, "rb").read())
    stored[0][table.COLUMNS.index("symbolic_recovery")] = 1 - stored[0][table.COLUMNS.index("symbolic_recovery")]
    os.remove(entry)                                                          # a new inode: the old entry stays as it was
    with open(entry, "wb") as fh:
        pickle.dump(stored, fh)
    assert rekey.check(r["carried"], cache, ENGINE, 10) == [tree_path]
    assert rekey.main(["--cache", cache, "--from", old, "--tree", f"m:1:{tmp_path / 'tree'}", "--engine", ENGINE]) == 0
