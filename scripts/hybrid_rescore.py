"""Re-pick the hybrid arm's predictions offline: PySR's hall of fame joins the Flash-ANSR candidates and
Flash-ANSR's sorting picks rank 0 -- from the stored rows (PySR's whole hall of fame rides in every row)
and the generation snapshots (the Flash-ANSR candidates at the cell's candidate count). Rows of the r = 0
cells and rows without a hall of fame are copied unchanged.

    python scripts/hybrid_rescore.py --results <root>/results/evaluation/hybrid/<arm> \
        --snapshots <root>/snapshots --out <root>/results/evaluation/hybrid/<arm>_ranked [--engine acj-5-4-llm]
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from flash_ansr.scoring import RankingConfig
from simplipy import SimpliPyEngine

from srbf.hybrid_adapter import pick_prediction


def _row(columns: dict, i: int) -> dict:
    return {k: v[i] for k, v in columns.items() if k != "__meta__" and isinstance(v, (list, np.ndarray)) and len(v) == len(columns["expression"])}


def rescore_file(path: Path, snapshots: Path, out: Path, engine: SimpliPyEngine) -> tuple[int, int, dict[str, int]]:
    with path.open("rb") as fh:
        columns = pickle.load(fh)
    n = len(columns["expression"])
    winners: dict[str, int] = {}
    changed = 0
    rows = [_row(columns, i) for i in range(n)]
    for i, row in enumerate(rows):
        equations = row.get("equations")
        choices = int(row.get("hybrid_choices") or 0)
        if not equations and not choices:
            continue
        if int(row.get("hybrid_niterations") or 0) <= 0:
            continue                                        # an r = 0 cell: Flash-ANSR alone, nothing to re-select
        flash: list = []
        if choices > 0:
            snap_path = snapshots / f"problem_{int(row['eval_row_index']):06d}.pkl"
            with snap_path.open("rb") as fh:
                snap = pickle.load(fh)
            target = snap["targets"][choices]
            best = target["best"]
            flash = list(target.get("candidates") or ([best] if best is not None else []))
        ranking = row.get("ranking")
        weights = RankingConfig.from_dict(dict(ranking)).effective_weights
        y_fit = row["y_noisy"] if row.get("y_noisy") is not None else row["y"]
        y_val = row.get("y_noisy_val") if row.get("y_noisy_val") is not None else row.get("y_val")
        variables = list(row.get("variables") or row.get("variable_names") or [])   # the full list, by column of x
        before = row.get("predicted_expression")
        pick_prediction(row, flash=flash, equations=equations, engine=engine, weights=weights,
                        x_support=np.asarray(row["x"], dtype=float), y_fit=np.asarray(y_fit, dtype=float).reshape(-1),
                        x_val=np.asarray(row["x_val"], dtype=float) if row.get("x_val") is not None else None,
                        y_val=np.asarray(y_val, dtype=float).reshape(-1) if y_val is not None else None,
                        variables=variables)
        winners[row["predicted_source"]] = winners.get(row["predicted_source"], 0) + 1
        changed += int(row.get("predicted_expression") != before)
        for k, v in row.items():
            if k not in columns:
                columns[k] = [None] * n
            columns[k][i] = v
    meta = dict(columns.get("__meta__") or {})
    meta["hybrid_ranking"] = "PySR's hall of fame added to the Flash-ANSR candidates, Flash-ANSR's sorting picked rank 0 (offline, scripts/hybrid_rescore.py)"
    columns["__meta__"] = meta
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fh:
        pickle.dump(columns, fh)
    return n, changed, winners


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, type=Path, help="<root>/results/evaluation/hybrid/<arm>")
    ap.add_argument("--snapshots", required=True, type=Path, help="<root>/snapshots (one subdirectory per catalog)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--engine", default="acj-5-4-llm")
    args = ap.parse_args()
    engine = SimpliPyEngine.load(args.engine, install=True)
    for path in sorted(args.results.glob("*/ratio_*.pkl")):
        catalog = path.parent.name
        n, changed, winners = rescore_file(path, args.snapshots / catalog, args.out / catalog / path.name, engine)
        print(f"{catalog:<18s} {path.name:<14s} rows {n:>4d}  changed {changed:>4d}  rank 0 from {winners}")


if __name__ == "__main__":
    main()
