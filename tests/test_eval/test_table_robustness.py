"""srbf.table: a file that hangs does not hold up the table; the judge's fingerprint names only what judges."""
import csv
import os
import pickle
import time
from pathlib import Path

import numpy as np

from srbf.table import ResultTree, _judge_job, build_table, judge_dependencies

ENGINE = "acj-4-3"


def _snapshot(n):
    x = np.linspace(0.5, 2.0, 16)
    y = 3.0 * x
    return {
        "eval_row_index": list(range(n)),
        "skeleton": [["+", "x1", "x2"]] * n, "ground_truth_prefix": [["+", "x1", "x2"]] * n,
        "variables": [["v1", "v2"]] * n,
        "predicted_skeleton_prefix": [["+", "x1", "x2"]] * n, "predicted_expression_prefix": [["+", "x1", "x2"]] * n,
        "prediction_success": [True] * n,
        "y": [y.copy() for _ in range(n)], "y_pred": [y.copy() for _ in range(n)],
        "y_val": [y.copy() for _ in range(n)], "y_pred_val": [y.copy() for _ in range(n)],
        "fit_time": [0.25] * n, "n_support": [16] * n,
        "placeholder": [False] * n, "benchmark_eq_id": [f"p{i}" for i in range(n)],
    }


def _write(tmp_path, catalog, name, snapshot):
    d = tmp_path / "tree" / catalog
    d.mkdir(parents=True, exist_ok=True)
    with open(d / name, "wb") as fh:
        pickle.dump(snapshot, fh)
    return str(d / name)


def _hanging_judge(job):
    """The real job, except for a file under a catalog named ``hang-once`` (it hangs the first time it is judged) or
    ``hang-always`` (it hangs every time). Module level, so the pool's worker processes can import it."""
    _index, _tree, path, _engine, _progress = job
    tried = path + ".tried"
    if "hang-always" in path or ("hang-once" in path and not os.path.exists(tried)):
        if "hang-once" in path:
            open(tried, "w").close()
        from srbf.table import _mark
        _mark(_progress, "S", _index, path)
        time.sleep(600)
    return _judge_job(job)


def _rows(path):
    with open(path) as fh:
        return list(csv.reader(fh))[1:]


def test_a_file_that_hangs_is_judged_once_more_in_a_fresh_worker(tmp_path):
    _write(tmp_path, "toy", "choices_000001.pkl", _snapshot(2))
    _write(tmp_path, "hang-once", "choices_000001.pkl", _snapshot(3))
    tree = ResultTree("m", 1, str(tmp_path / "tree"))
    out, said = str(tmp_path / "t.csv"), []
    report = build_table([tree], out, engine=ENGINE, workers=2, cache_dir=str(tmp_path / "cache"), stall_timeout=3,
                         log=said.append, _judge=_hanging_judge)
    assert not report.stalled and report.errors == []
    assert report.rows == 5 and len(_rows(out)) == 5
    assert any(m.startswith("RETRY") and "hang-once" in m for m in said)
    assert not any(m.startswith("STALLED") for m in said)


def test_a_file_that_hangs_twice_is_left_out_and_the_rest_is_judged(tmp_path):
    for k in range(3):
        _write(tmp_path, "toy", f"choices_{2 ** k:06d}.pkl", _snapshot(2))
    _write(tmp_path, "hang-always", "choices_000001.pkl", _snapshot(4))
    tree = ResultTree("m", 1, str(tmp_path / "tree"))
    out, said = str(tmp_path / "t.csv"), []
    report = build_table([tree], out, engine=ENGINE, workers=2, cache_dir=str(tmp_path / "cache"), stall_timeout=3,
                         log=said.append, _judge=_hanging_judge)
    assert report.stalled and len(report.errors) == 1 and "hang-always" in report.errors[0]
    assert report.rows == 6 and len(_rows(out)) == 6                    # every other file's rows
    assert [m for m in said if m.startswith("STALLED")] and "hang-always" in [m for m in said if m.startswith("STALLED")][0]


def test_the_rows_keep_the_files_order_whatever_order_they_finish_in(tmp_path):
    for k in range(5):
        _write(tmp_path, f"c{k}", "choices_000001.pkl", _snapshot(k + 1))
    tree = ResultTree("m", 1, str(tmp_path / "tree"))
    a, b = str(tmp_path / "a.csv"), str(tmp_path / "b.csv")
    build_table([tree], a, engine=ENGINE, workers=3, log=None)
    build_table([tree], b, engine=ENGINE, workers=1, cache_dir=str(tmp_path / "cache"), log=None)
    assert _rows(a) == _rows(b) and len(_rows(a)) == 15


def _module(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def test_the_fingerprint_follows_the_judges_imports_only(tmp_path):
    """The judge's sources are the modules srbf.table reaches through its imports, at any depth and inside functions;
    the package __init__ that re-exports everything, the workers and the rung names are not among them."""
    root = tmp_path / "srbf"
    _module(root, "__init__.py", "from srbf.report import r\nfrom srbf.a import f\n")
    _module(root, "table.py", "import srbf\nimport numpy as np\nfrom srbf.rungs import RUNGS\n\n"
                              "def judge():\n    from srbf.a import f\n    return f()\n")
    _module(root, "a.py", "import editdistance\nfrom . import b\n\ndef f():\n    return b.g()\n")
    _module(root, "b.py", "import os\n\ndef g():\n    return os.sep\n")
    _module(root, "report.py", "from srbf.c import h\n")
    _module(root, "c.py", "import matplotlib\n")
    _module(root, "rungs.py", "RUNGS = ()\n")
    _module(root, "worker/__init__.py", "")
    _module(root, "worker/runner.py", "import torch\n")
    sources, libraries = judge_dependencies(root)
    assert sorted(p.relative_to(root).as_posix() for p in sources) == ["a.py", "b.py", "table.py"]
    assert libraries == ["editdistance", "numpy", "sympy"]
