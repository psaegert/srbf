"""Derive measurement-noise margins (MRD) from archived evaluation snapshots.

The margin table that powers paired verdicts ("equivalent" = the difference is smaller than the
benchmark can measure) is DERIVED, not hand-picked: for every series we estimate the aggregate
draw-noise null of the paired statistic by comparing the series TO ITSELF (split-half across its
~10 draws per expression, exact per-expression rescaling to full-draw scale; a centered
draw-bootstrap serves as a cross-check — the two must agree unless draws are not i.i.d.).
Margins for a specific pair (A, B) combine the two series' nulls (`srbf.reporting.pair_margin`);
they are pair-specific by design.

Tier-1 metrics: `numeric_recovery_val` (rate; failures scored 0.0 per the two-regime rule) and
`expr_length_ratio` (diagnostic; failures dropped). Snapshots are the archived per-draw eval
pickles (dict-of-lists with `benchmark_eq_id`, `y_val`, `y_pred_val`, `prediction_success`,
`predicted_skeleton_prefix`, `skeleton`, `placeholder`).

Derive margins (top rung per series x benchmark; flagship over all rungs):
    python scripts/derive_noise_margins.py --results-root <...>/results/evaluation/scaling \\
        --out noise_margins.json

Validate on the same-weights near-replicate pair (v23.0-120M vs v23.3-120M = identical weights,
KV-cache decode): their observed paired delta must fall within the pair margin:
    python scripts/derive_noise_margins.py --results-root <...> --validate v23.0-120M v23.3-120M
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np

from srbf.metrics.numeric import is_perfect_fit
from srbf.reporting import draw_values, paired_expression_deltas, pair_margin, self_noise

TIER1_METRICS = ("numeric_recovery_val", "expr_length_ratio")


def load_snapshot(path: Path) -> dict:
    with open(path, "rb") as f:
        snapshot = pickle.load(f)
    snapshot.pop("__meta__", None)
    return snapshot


def score_tier1(snapshot: dict) -> dict:
    """Attach per-draw tier-1 metric columns (strict srbf scorers, two-regime failure policy)."""
    n = max((len(v) for v in snapshot.values() if isinstance(v, list)), default=0)

    def row(key: str, i: int):
        col = snapshot.get(key)
        return col[i] if col is not None and i < len(col) else None

    recovery: list[float | None] = []
    length_ratio: list[float | None] = []
    for i in range(n):
        success = bool(row("prediction_success", i))
        y_val, y_pred_val = row("y_val", i), row("y_pred_val", i)
        if not success or y_val is None or y_pred_val is None:
            recovery.append(0.0)      # rate metric: failure counts as 0.0
            length_ratio.append(None)  # diagnostic metric: failure is dropped
            continue
        recovery.append(float(is_perfect_fit(np.asarray(y_val), np.asarray(y_pred_val))))
        predicted, truth = row("predicted_skeleton_prefix", i), row("skeleton", i)
        if predicted and truth:
            length_ratio.append(float(len(predicted)) / float(len(truth)))
        else:
            length_ratio.append(None)

    snapshot["numeric_recovery_val"] = recovery
    snapshot["expr_length_ratio"] = length_ratio
    return snapshot


def resolve_group_key(snapshot: dict) -> str:
    """Expression-id column: `benchmark_eq_id` (curated benchmarks) with `skeleton_hash` as the
    fallback for older/generative snapshots (v23-val), where the ground-truth skeleton tuple IS
    the stable expression identity. Raises if neither exists — never row order."""
    for key in ("benchmark_eq_id", "skeleton_hash"):
        column = snapshot.get(key)
        if column is not None and any(v is not None for v in column):
            return key
    raise KeyError("snapshot has neither 'benchmark_eq_id' nor 'skeleton_hash' — cannot pair")


def discover_rungs(results_root: Path, series: str, benchmark: str) -> dict[int, Path]:
    """Map rung value -> pkl path for one (series, benchmark); prefix-agnostic (choices_/niter_/...)."""
    rungs: dict[int, Path] = {}
    for path in sorted((results_root / series / benchmark).glob("*.pkl")):
        match = re.search(r"_(\d+)\.pkl$", path.name)
        if match:
            rungs[int(match.group(1))] = path
    return rungs


def derive_cell(path: Path, *, n_null: int, rng: np.random.Generator) -> dict:
    """Noise nulls for one (series, benchmark, rung) snapshot, per tier-1 metric x method."""
    snapshot = score_tier1(load_snapshot(path))
    group_key = resolve_group_key(snapshot)
    cell: dict = {"group_key": group_key}
    for metric in TIER1_METRICS:
        values = draw_values(snapshot, metric, group_key=group_key)
        split = self_noise(values, n_null=n_null, method="split-half", rng=rng)
        boot = self_noise(values, n_null=n_null, method="bootstrap", rng=rng)
        agreement = split["sd"] / boot["sd"] if boot["sd"] else float("nan")
        cell[metric] = {
            "sd": split["sd"], "q95": split["q95"],
            "n_expressions": split["n_expressions"], "n_skipped": split["n_skipped"],
            "bootstrap_sd": boot["sd"], "method_sd_ratio": agreement,
            "null": [round(float(x), 8) for x in split["null"]],
        }
    return cell


def cmd_derive(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    results_root = Path(args.results_root)
    series_list = args.series or sorted(p.name for p in results_root.iterdir() if p.is_dir())

    table: dict = {"metrics": list(TIER1_METRICS), "n_null": args.n_null, "cells": {}}
    for series in series_list:
        for benchmark in args.benchmarks:
            rungs = discover_rungs(results_root, series, benchmark)
            if not rungs:
                continue
            selected = sorted(rungs) if series in (args.all_rungs_for or []) else [max(rungs)]
            for rung in selected:
                key = f"{series}|{benchmark}|{rung}"
                print(f"[derive] {key}", flush=True)
                table["cells"][key] = derive_cell(rungs[rung], n_null=args.n_null, rng=rng)

    out = Path(args.out)
    out.write_text(json.dumps(table))
    print(f"\n[out] {out}  ({len(table['cells'])} cells)")

    print(f"\n== noise SD (aggregate delta scale) + q95, split-half [bootstrap agreement ratio] ==")
    for metric in TIER1_METRICS:
        print(f"\n-- {metric}")
        for key, cell in sorted(table["cells"].items()):
            m = cell[metric]
            print(f"  {key:42s} sd={m['sd']:.5f} q95={m['q95']:.5f} "
                  f"n={m['n_expressions']:>3} [ratio {m['method_sd_ratio']:.2f}]")

    # Pair-margin preview for a few canonical pairs at their top rungs.
    def top_key(series: str, benchmark: str) -> str | None:
        keys = [k for k in table["cells"] if k.startswith(f"{series}|{benchmark}|")]
        return max(keys, key=lambda k: int(k.rsplit("|", 1)[1])) if keys else None

    print("\n== example pair margins (numeric_recovery_val, top rungs) ==")
    anchors = [s for s in ("v23.0-120M", "pysr", "nesymres", "v23.0-3M", "v23.3-120M") if s in series_list]
    for a in anchors:
        for b in anchors:
            if a >= b:
                continue
            for benchmark in args.benchmarks:
                ka, kb = top_key(a, benchmark), top_key(b, benchmark)
                if not ka or not kb:
                    continue
                margin = pair_margin(
                    {"null": np.asarray(table["cells"][ka]["numeric_recovery_val"]["null"])},
                    {"null": np.asarray(table["cells"][kb]["numeric_recovery_val"]["null"])},
                    rng=rng)
                print(f"  {a} vs {b} [{benchmark}]: m_AB = {margin['margin']:.4f}")


def cmd_validate(args: argparse.Namespace) -> None:
    """Same-weights near-replicate check: the observed paired delta must sit within m_AB."""
    rng = np.random.default_rng(args.seed)
    results_root = Path(args.results_root)
    series_a, series_b = args.validate

    for benchmark in args.benchmarks:
        rungs_a = discover_rungs(results_root, series_a, benchmark)
        rungs_b = discover_rungs(results_root, series_b, benchmark)
        shared = sorted(set(rungs_a) & set(rungs_b))
        if not shared:
            print(f"[{benchmark}] no shared rungs")
            continue
        rung = max(shared)
        snap_a = score_tier1(load_snapshot(rungs_a[rung]))
        snap_b = score_tier1(load_snapshot(rungs_b[rung]))
        key_a, key_b = resolve_group_key(snap_a), resolve_group_key(snap_b)
        if key_a != key_b:
            print(f"[{benchmark}] group-key mismatch ({key_a} vs {key_b}) — skipping")
            continue
        print(f"\n== {series_a} vs {series_b} [{benchmark}] at shared rung {rung} (join on {key_a}) ==")
        for metric in TIER1_METRICS:
            va = draw_values(snap_a, metric, group_key=key_a)
            vb = draw_values(snap_b, metric, group_key=key_b)
            pairs = paired_expression_deltas(va, vb)
            delta = float(np.mean(pairs["deltas"])) if pairs["n_pairs"] else float("nan")
            noise_a = self_noise(va, n_null=args.n_null, rng=rng)
            noise_b = self_noise(vb, n_null=args.n_null, rng=rng)
            margin = pair_margin(noise_a, noise_b, rng=rng)
            inside = abs(delta) <= margin["margin"]
            print(f"  {metric:24s} delta={delta:+.5f}  m_AB={margin['margin']:.5f}  "
                  f"n_pairs={pairs['n_pairs']}  (only_a={pairs['n_only_a']}, only_b={pairs['n_only_b']})  "
                  f"-> {'WITHIN margin (equivalent)' if inside else 'OUTSIDE margin'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--series", nargs="*", default=None, help="default: every dir under the root")
    parser.add_argument("--benchmarks", nargs="*", default=["fastsrb", "val"])
    parser.add_argument("--n-null", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--all-rungs-for", nargs="*", default=["v23.0-120M"],
                        help="series to derive at EVERY rung (margin-vs-rung sensitivity)")
    parser.add_argument("--out", default="noise_margins.json")
    parser.add_argument("--validate", nargs=2, metavar=("SERIES_A", "SERIES_B"),
                        help="near-replicate check instead of deriving the table")
    args = parser.parse_args()

    if args.validate:
        cmd_validate(args)
    else:
        cmd_derive(args)


if __name__ == "__main__":
    main()
