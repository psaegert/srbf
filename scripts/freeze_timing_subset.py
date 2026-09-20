"""Freeze a timing subset of the srbf suite: about 260 problems stratified by catalog. Per catalog it is shard 0
of N, with N from the catalog's size (``--subset-rule 50:40,20:4,10:2``: 1 in 40 for catalogs of 50 problems or
more, 1 in 4 from 20, 1 in 2 from 10, every problem below). The subset is materialized once, so every time
measurement runs on the same instances and is paired by instance across methods and budgets.

  freeze_timing_subset.py -c configs/evaluation/scaling/flash-ansr-v25.0-T8-20M_srbf.yaml --out-dir DIR
  freeze_timing_subset.py -c ... --from-frozen OTHER_DIR --out-dir DIR     # nest it in an existing frozen subset
  freeze_timing_subset.py --check --out-dir DIR                            # verify the files against the manifest

With ``--from-frozen`` the rows are taken from an existing frozen subset (``source_row_index % N == 0``, N a
multiple of that subset's shard count): the very same instances. Without it the problems are drawn fresh from the
catalog, which samples new support points for the same laws.

The manifest ``<out-dir>/timing_subset.json`` records the rule, every catalog's full size (the stratum weight the
read-out uses), shard, count and source rows. Existing files are never overwritten: they are verified instead.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_calibrated_ladder import catalog_size, parse_rule, subset_count  # noqa: E402

DEFAULT_RULE = "50:40,20:4,10:2"


def _rows(problems: Any) -> list[int]:
    rows = []
    for p in problems:
        r = (p.meta or {}).get("source_row_index")
        if r is None:
            raise RuntimeError("a frozen problem carries no meta.source_row_index; cannot nest or verify")
        rows.append(int(r))
    return rows


def freeze_from_frozen(src: Path, experiment: str, n: int, out: Path, sampling: Any) -> list[int]:
    """An existing frozen catalog filtered to ``source_row_index % n == 0`` (n a multiple of its shard)."""
    from symbolic_data.catalog import ProblemCatalog, load_catalog

    catalog = load_catalog(str(src))
    shard = list((catalog.meta or {}).get("shard") or [0, 1])
    if shard[0] != 0 or n % int(shard[1]):
        raise RuntimeError(f"{experiment}: timing shard 1/{n} does not nest in the frozen shard {shard}")
    keep = [p for p in (catalog.problems or []) if int(p.meta["source_row_index"]) % n == 0]
    if not keep:
        raise RuntimeError(f"{experiment}: no rows left at 1 in {n}")
    new = ProblemCatalog.from_problems(
        keep, name=f"{experiment}-timing-1-of-{n}",
        meta={"source_catalog": (catalog.meta or {}).get("source_catalog", experiment), "shard": [0, int(n)],
              "nested_in": {"path": str(src), "shard": shard, "name": (catalog.meta or {}).get("name")},
              "sampling": dict((catalog.meta or {}).get("sampling") or sampling or {}),
              "frozen_by": "scripts/freeze_timing_subset.py"})
    _save(new, out)
    return _rows(keep)


def freeze_from_source(cfg: Any, experiment: str, n: int, out: Path, engine_ref: str) -> list[int]:
    """Fresh instances: shard 0 of n of the experiment's catalog source ."""
    from simplipy import SimpliPyEngine
    from srbf.config import build_catalog_source, select_experiment
    from symbolic_data.catalog import ProblemCatalog

    exp = select_experiment(cfg, experiment)
    source = build_catalog_source(exp["data_source"], target_size=None, skip=0)
    engine = SimpliPyEngine.load(engine_ref, install=True)
    source.prepare(adapter=SimpleNamespace(get_simplipy_engine=lambda: engine))
    keep = []
    for index, problem in enumerate(source.problem_source):
        if index % n:
            continue
        problem.meta["source_row_index"] = int(index)
        keep.append(problem)
    new = ProblemCatalog.from_problems(
        keep, name=f"{experiment}-timing-1-of-{n}",
        meta={"source_catalog": str(exp["data_source"].get("catalog")), "shard": [0, int(n)], "nested_in": None,
              "sampling": dict(exp["data_source"].get("sampling") or {}), "frozen_by": "scripts/freeze_timing_subset.py"})
    _save(new, out)
    return _rows(keep)


def _save(catalog: Any, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    catalog.save(tmp)
    tmp.replace(out)


def verify(out_dir: Path, manifest: dict) -> int:
    from symbolic_data.catalog import load_catalog

    bad = 0
    for e, m in manifest["catalogs"].items():
        path = out_dir / f"{e}.npz"
        if not path.exists():
            print(f"  MISSING {path}")
            bad += 1
            continue
        cat = load_catalog(str(path))
        rows = _rows(cat.problems or [])
        if rows != list(m["source_row_index"]):
            print(f"  ROWS DIFFER {e}: file {len(rows)} rows, manifest {len(m['source_row_index'])}")
            bad += 1
        elif any(r % m["shard"] for r in rows):
            print(f"  NOT A 1-IN-{m['shard']} SHARD {e}")
            bad += 1
    total = sum(m["count"] for m in manifest["catalogs"].values())
    print(f"{'OK' if not bad else 'FAILED'}: {len(manifest['catalogs'])} catalogs, {total} problems, {bad} problems found")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", help="a run config naming every catalog (sizes come from its data sources)")
    ap.add_argument("--out-dir", required=True, help="where <catalog>.npz and timing_subset.json go")
    ap.add_argument("--from-frozen", help="an existing frozen subset: nest the timing rows in its files")
    ap.add_argument("--subset-rule", default=DEFAULT_RULE)
    ap.add_argument("--experiments", help="comma-separated subset of the config's experiments (default: all)")
    ap.add_argument("--engine", default="acj-5-4-llm", help="simplipy engine for a fresh (non-nested) freeze")
    ap.add_argument("--check", action="store_true", help="verify --out-dir against its manifest and exit")
    a = ap.parse_args()
    out_dir = Path(a.out_dir)
    manifest_path = out_dir / "timing_subset.json"
    if a.check:
        return verify(out_dir, json.loads(manifest_path.read_text()))
    if not a.config:
        sys.exit("-c/--config is required to freeze (catalog sizes)")
    from srbf.config import load_run_config, select_experiment

    cfg = load_run_config(a.config)
    experiments = list(cfg["experiments"])
    if a.experiments:
        wanted = a.experiments.split(",")
        experiments = [e for e in experiments if e in wanted]
    rule = parse_rule(a.subset_rule)
    sizes = {e: catalog_size(select_experiment(cfg, e)) for e in experiments}
    suite = sum(sizes.values())
    existing = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    catalogs = dict(existing["catalogs"]) if existing else {}
    for e in experiments:
        n = subset_count(sizes[e], rule)
        out = out_dir / f"{e}.npz"
        t0 = time.time()
        if out.exists():
            if e not in catalogs:
                raise RuntimeError(f"{out} exists without a manifest entry; remove it or restore the manifest")
            print(f"  kept   {e}: {catalogs[e]['count']} problems (1 in {catalogs[e]['shard']} of {sizes[e]})", flush=True)
            continue
        sampling = dict(select_experiment(cfg, e)["data_source"].get("sampling") or {})
        if a.from_frozen:
            rows = freeze_from_frozen(Path(a.from_frozen) / f"{e}.npz", e, n, out, sampling)
        else:
            rows = freeze_from_source(cfg, e, n, out, a.engine)
        catalogs[e] = {"size": sizes[e], "shard": n, "count": len(rows), "weight": sizes[e] / suite,
                       "nested_in": str(Path(a.from_frozen) / f"{e}.npz") if a.from_frozen else None,
                       "source_row_index": rows}
        print(f"  frozen {e}: {len(rows)} problems (1 in {n} of {sizes[e]}) {time.time() - t0:.1f}s", flush=True)
    manifest = {"rule": a.subset_rule, "built_from": f"frozen:{a.from_frozen}" if a.from_frozen else "source",
                "config": str(a.config), "suite_size": suite, "total_problems": sum(m["count"] for m in catalogs.values()),
                "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "catalogs": catalogs}
    manifest_path.write_text(json.dumps(manifest, indent=1))
    print(f"{manifest['total_problems']} problems in {len(catalogs)} catalogs (suite {suite}); manifest {manifest_path}")
    return verify(out_dir, manifest)


if __name__ == "__main__":
    sys.exit(main())
