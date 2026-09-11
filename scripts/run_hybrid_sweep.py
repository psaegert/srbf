"""Run the hybrid sweep: every ratio cell over the stratified subset, one problem at a time on the
whole machine, sequentially. The subset is the calibrated protocol's: shard 0 of N per catalog,
N from the catalog's size (`--subset-rule 50:10,10:2`: 1 in 10 for catalogs of >= 50 problems,
1 in 2 for >= 10, every problem below).

The subset is FROZEN once per catalog (a materialized symbolic_data catalog under
<root>/hybrid_data/) and every cell reads the frozen file through a derived config: the data
source re-draws support points on every iteration, and the generation snapshots the adapter caches
per problem -- the first cell that touches a problem pays the generation pass for all cells -- are
only valid on the exact data they were generated on. Freezing also makes the cells paired by
INSTANCE, which the pre-registered McNemar test assumes. `eval_row_index` in the results counts
inside the frozen subset; the problem's `meta.source_row_index` is its row in the source catalog.

Usage: run_hybrid_sweep.py --config hybrid.yaml --root ROOT [--ratios 0,0.5,1] [--experiments a,b]
       [--subset-rule 50:10,10:2] [--dry-run]
Markers and logs under <root>/hybrid_marks/; the frozen data and config under <root>/hybrid_data/.
"""
from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_calibrated_ladder import catalog_size, parse_rule, subset_count  # noqa: E402

from srbf.config import build_catalog_source, load_config, select_experiment  # noqa: E402
from srbf.sweep import Sweep  # noqa: E402


def _sweep_representer(dumper, data):
    body = {"values": list(data.values)}
    if data.name is not None:
        body = {"name": data.name, **body}
    return dumper.represent_mapping("!sweep", body)


yaml.add_representer(Sweep, _sweep_representer)


def freeze_subset(cfg, experiment: str, n: int, path: Path, engine_ref: str) -> int:
    """Materialize shard 0 of ``n`` of the experiment's catalog into a frozen catalog at ``path``
    (once: an existing file is reused). Returns the number of problems in the subset."""
    from simplipy import SimpliPyEngine
    from symbolic_data.catalog import ProblemCatalog, load_catalog

    if path.exists():
        return len(load_catalog(str(path)).problems or [])
    exp = select_experiment(cfg, experiment)
    source = build_catalog_source(exp["data_source"], target_size=None, skip=0)
    engine = SimpliPyEngine.load(engine_ref, install=True)
    source.prepare(adapter=SimpleNamespace(get_simplipy_engine=lambda: engine))
    problems = []
    for index, problem in enumerate(source.problem_source):
        if index % n:
            continue
        problem.meta["source_row_index"] = int(index)
        problems.append(problem)
    catalog = ProblemCatalog.from_problems(
        problems, name=f"{experiment}-hybrid-1-of-{n}",
        meta={"source_catalog": str(exp["data_source"].get("catalog")), "shard": [0, int(n)],
              "sampling": dict(exp["data_source"].get("sampling") or {}), "frozen_by": "scripts/run_hybrid_sweep.py"})
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    catalog.save(tmp)
    tmp.replace(path)
    return len(problems)


def frozen_config(cfg, experiments: list[str], data_dir: Path) -> dict:
    """The run config with every experiment's data source pointed at its frozen subset."""
    out = copy.deepcopy(cfg)
    for e in experiments:
        block = out["experiments"][e]
        if "data_source" not in block:
            raise ValueError(f"experiment {e!r} carries no data_source of its own; the hybrid config generator writes one per catalog")
        block["data_source"] = dict(block["data_source"])
        block["data_source"]["catalog"] = str(data_dir / f"{e}.npz")
    return out


def _engine_ref(cfg, experiment: str) -> str:
    """The simplipy engine the cells share (the PySR block's, which the config generator sets)."""
    adapter = select_experiment(cfg, experiment)["model_adapter"]
    return str((adapter.get("pysr") or {}).get("simplipy_engine") or adapter.get("simplipy_engine") or "acj-5-4-llm")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--ratios", default=None, help="subset of the config's ratio axis, comma separated")
    ap.add_argument("--experiments", default=None)
    ap.add_argument("--subset-rule", default="50:10,10:2")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--freeze-only", action="store_true", help="freeze the subset and write the frozen config, run nothing")
    a = ap.parse_args()

    os.environ["FLASH_ANSR_ROOT"] = a.root
    cfg = load_config(a.config)
    experiments = a.experiments.split(",") if a.experiments else list(cfg["experiments"].keys())
    rule = parse_rule(a.subset_rule)
    ratio_axis = None
    for name, exp in cfg["experiments"].items():
        block = exp["model_adapter"]["hybrid"]
        ratio_axis = block.get("ratios") if isinstance(block, dict) else None
        break
    ratios = [str(float(r)) for r in (a.ratios.split(",") if a.ratios else (ratio_axis or []))]   # the sweep filter compares str(value)
    mark_dir = Path(a.root) / "hybrid_marks"
    mark_dir.mkdir(parents=True, exist_ok=True)
    sizes = {e: catalog_size(select_experiment(cfg, e)) for e in experiments}
    plan = [(r, e, subset_count(sizes[e], rule)) for r in ratios for e in experiments]
    n_problems = sum(-(-sizes[e] // n) for _, e, n in plan[: len(experiments)])
    print(f"{len(plan)} cells: {len(ratios)} ratios x {len(experiments)} catalogs, ~{n_problems} problems per ratio; root {a.root}", flush=True)
    if a.dry_run:
        for r, e, n in plan:
            print("   srbf run -c <frozen> --experiment", e, "--sweep-filter", f"ratio={r}", f"({sizes[e]} problems, 1 in {n})")
        return 0

    # freeze the subset once per catalog; every cell reads the frozen file
    data_dir = Path(a.root) / "hybrid_data"
    for e in experiments:
        n = subset_count(sizes[e], rule)
        t0 = time.time()
        count = freeze_subset(cfg, e, n, data_dir / f"{e}.npz", _engine_ref(cfg, e))
        print(time.strftime("%H:%M:%S"), f"frozen {e}: {count} problems (1 in {n} of {sizes[e]}) {time.time() - t0:.0f}s", flush=True)
    frozen_path = data_dir / (Path(a.config).stem + ".frozen.yaml")
    frozen_path.write_text(yaml.dump(frozen_config(cfg, experiments, data_dir), sort_keys=False, width=200))
    if a.freeze_only:
        print("frozen config:", frozen_path)
        return 0

    env = dict(os.environ, PYTHONUNBUFFERED="1")
    for r, e, n in plan:
        tag = f"ratio_{r}_{e}" + (f".subset-1-of-{n}" if n > 1 else "")
        marker = mark_dir / f"{tag}.done"
        if marker.exists():
            continue
        cmd = ["srbf", "run", "-c", str(frozen_path), "--experiment", e, "--sweep-filter", f"ratio={r}", "-v"]
        t0 = time.time()
        with open(mark_dir / f"{tag}.log", "a") as log:
            rc = subprocess.call(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        print(time.strftime("%H:%M:%S"), f"ratio={r} {e} rc={rc} {time.time() - t0:.0f}s", flush=True)
        if rc == 0:
            marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S\n"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
