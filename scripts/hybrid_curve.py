"""The hybrid r-curve: recovery against the share r of a fixed budget given to PySR, from the
`ratio_<r>.pkl` result files one `run_hybrid_sweep.py` produced.

Usage: hybrid_curve.py --results <root>/results/evaluation/hybrid/<tag> [--out DIR] [--engine acj-5-4-llm]
Writes hybrid_curve.csv (pooled + per catalog, bootstrap intervals), hybrid_tests.txt (paired
McNemar of r = 0.5 against both endpoints, pre-registered), hybrid_times.csv (target vs achieved
seconds per stage) and hybrid_curve.png (dots, whiskers, band; legend below the axes). Symbolic
recovery is reported raw (`exact`: the masked skeleton equals the law's) and up to simplipy's
canonical form (`exact_canonical`); the MDL of every arm's answer is priced uniformly here.
"""
from __future__ import annotations

import argparse
import glob
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from simplipy import SimpliPyEngine
from srbf.metrics.numeric import fvu as srbf_fvu, is_perfect_fit
from symbolic_data.token_ops import normalize_skeleton


def load(results_dir: str) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(f"{results_dir}/*/ratio_*.pkl")):
        cat = Path(f).parent.name
        r = float(Path(f).name.split("ratio_")[1].split(".shard")[0].removesuffix(".pkl"))   # ratio_0.5[.shard-0-of-60].pkl
        d = pickle.load(open(f, "rb"))
        n = len(d["skeleton"])
        for i in range(n):
            if d["placeholder"][i]:
                continue
            ok = bool(d["prediction_success"][i])
            yv = np.asarray(d["y_val"][i], float).reshape(-1); ys = np.asarray(d["y"][i], float).reshape(-1)
            rec_val = rec_fit = False; fvu_val = np.nan
            if ok and d["y_pred_val"][i] is not None and len(np.asarray(d["y_pred_val"][i]).reshape(-1)) == len(yv):
                ypv = np.asarray(d["y_pred_val"][i], float).reshape(-1); yp = np.asarray(d["y_pred"][i], float).reshape(-1)
                rec_val = bool(is_perfect_fit(yv, ypv)); rec_fit = bool(is_perfect_fit(ys, yp)); fvu_val = float(srbf_fvu(yv, ypv))
            rows.append({
                "catalog": cat, "ratio": r, "pid": int(d["eval_row_index"][i]) if "eval_row_index" in d else i,
                "ok": ok, "rec_val": rec_val, "rec_fit": rec_fit, "fvu_val": fvu_val,
                "skeleton": tuple(d["skeleton"][i]), "pred_skeleton": tuple(d["predicted_skeleton_prefix"][i] or ()) if ok else (),
                "pred_prefix": list(d["predicted_expression_prefix"][i] or []) if ok else [],
                "pred_mdl": d.get("predicted_mdl", [None] * n)[i] if ok else None,
                "fit_time": d["fit_time"][i], "gen_s": d.get("hybrid_generation_s", [np.nan] * n)[i],
                "gp_s": d.get("hybrid_gp_s", [np.nan] * n)[i], "target_gen_s": d.get("hybrid_target_generation_s", [np.nan] * n)[i],
                "target_gp_s": d.get("hybrid_target_gp_s", [np.nan] * n)[i], "n_seeds": d.get("hybrid_n_seeds", [np.nan] * n)[i],
                "choices": d.get("hybrid_choices", [np.nan] * n)[i], "niterations": d.get("hybrid_niterations", [np.nan] * n)[i],
                "law_prefix": list(d["ground_truth_prefix"][i]) if "ground_truth_prefix" in d else list(d["expression"][i]),
            })
    return pd.DataFrame(rows)


def boot(v: np.ndarray, fn=np.mean, n=1000, seed=0):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    if not len(v): return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(v), size=(n, len(v)))
    s = np.array([fn(v[i]) for i in idx]); return (fn(v), np.percentile(s, 2.5), np.percentile(s, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--results", required=True); ap.add_argument("--out", default=None)
    ap.add_argument("--engine", default="acj-5-4-llm"); ap.add_argument("--primary", type=float, default=0.5)
    a = ap.parse_args(); out = Path(a.out or a.results); out.mkdir(parents=True, exist_ok=True)
    df = load(a.results)
    engine = SimpliPyEngine.load(a.engine, install=True)
    from simplipy.engine import Mode
    laws = {}
    for key, grp in df.groupby(["catalog", "pid"]):
        sk = grp.iloc[0]["skeleton"]; lp = grp.iloc[0]["law_prefix"]
        try: law_sk = tuple(engine.simplify(list(sk)))
        except Exception: law_sk = tuple(sk)
        try: gt_mdl = float(engine.complexity(list(lp), certified=True, mode=Mode.f64, canon="default"))
        except Exception: gt_mdl = np.nan
        laws[key] = (law_sk, gt_mdl)
    df["exact"] = [bool(r.rec_val and laws[(r.catalog, r.pid)][0] == tuple(normalize_skeleton(list(r.pred_prefix)))) if r.ok else False for r in df.itertuples()]
    # symbolic recovery up to simplipy's canonical form (a GP answer is often a bloated equivalent)
    def _canonical_exact(r):
        if not (r.ok and r.rec_val):
            return False
        try:
            return laws[(r.catalog, r.pid)][0] == tuple(normalize_skeleton(list(engine.simplify(list(r.pred_prefix)))))
        except Exception:
            return False
    df["exact_canonical"] = [_canonical_exact(r) for r in df.itertuples()]
    # the MDL of every arm's answer priced the same way (the PySR worker records none)
    def _mdl(r):
        if not r.ok:
            return np.nan
        if r.pred_mdl is not None and np.isfinite(float(r.pred_mdl)):
            return float(r.pred_mdl)
        try:
            return float(engine.complexity(list(r.pred_prefix), certified=True, mode=Mode.f64, canon="default"))
        except Exception:
            return np.nan
    df["pred_mdl_uniform"] = [_mdl(r) for r in df.itertuples()]
    df["mdl_ratio"] = [r.pred_mdl_uniform / laws[(r.catalog, r.pid)][1] if (r.rec_val and np.isfinite(r.pred_mdl_uniform) and np.isfinite(laws[(r.catalog, r.pid)][1])) else np.nan for r in df.itertuples()]
    df["achieved_s"] = df["gen_s"].fillna(0) + df["gp_s"].fillna(0)
    df["target_s"] = df["target_gen_s"].fillna(0) + df["target_gp_s"].fillna(0)

    ratios = sorted(df.ratio.unique())
    rows = []
    for r in ratios:
        g = df[df.ratio == r]
        for key, fn in [("rec_val", np.mean), ("rec_fit", np.mean), ("exact", np.mean), ("exact_canonical", np.mean), ("mdl_ratio", np.nanmedian), ("achieved_s", np.mean)]:
            m, lo, hi = boot(g[key].astype(float).values, fn)
            rows.append({"scope": "pooled", "ratio": r, "metric": key, "value": m, "lo": lo, "hi": hi, "n": len(g)})
        for cat, gc in g.groupby("catalog"):
            m, lo, hi = boot(gc["rec_val"].astype(float).values, np.mean)
            rows.append({"scope": cat, "ratio": r, "metric": "rec_val", "value": m, "lo": lo, "hi": hi, "n": len(gc)})
    table = pd.DataFrame(rows); table.to_csv(out / "hybrid_curve.csv", index=False)
    pooled = table[table.scope == "pooled"]
    print(f"{df.pid.nunique()} problems x {len(ratios)} ratios")
    print(f"{'r':>5s} {'vNRR':>20s} {'fNRR':>8s} {'exact':>8s} {'canon.':>8s} {'MDL ratio':>10s} {'achieved s':>11s}")
    for r in ratios:
        p = pooled[pooled.ratio == r].set_index("metric")
        print(f"{r:5.2f} {100*p.loc['rec_val','value']:6.1f} [{100*p.loc['rec_val','lo']:5.1f}, {100*p.loc['rec_val','hi']:5.1f}] {100*p.loc['rec_fit','value']:7.1f} {100*p.loc['exact','value']:7.1f} {100*p.loc['exact_canonical','value']:7.1f} {p.loc['mdl_ratio','value']:9.2f} {p.loc['achieved_s','value']:10.1f}")

    # pre-registered test: r = primary vs each endpoint, paired McNemar on rec_val
    lines = []
    piv = df.pivot_table(index=["catalog", "pid"], columns="ratio", values="rec_val", aggfunc="first")
    for other in (min(ratios), max(ratios)):
        if a.primary not in piv.columns or other not in piv.columns: continue
        x, y = piv[a.primary].astype(bool), piv[other].astype(bool)
        b01, b10 = int((~x & y).sum()), int((x & ~y).sum())
        p = binomtest(min(b01, b10), b01 + b10, 0.5).pvalue if b01 + b10 else 1.0
        lines.append(f"r={a.primary} vs r={other}: vNRR {100*x.mean():.1f}% vs {100*y.mean():.1f}%; won {b10}, lost {b01}, McNemar p={p:.3g}")
    best = pooled[pooled.metric == "rec_val"].sort_values("value", ascending=False).iloc[0]
    lines.append(f"best ratio {best.ratio:g}: vNRR {100*best.value:.1f}% [{100*best.lo:.1f}, {100*best.hi:.1f}]")
    (out / "hybrid_tests.txt").write_text("\n".join(lines) + "\n"); print("\n".join(lines))
    df.groupby("ratio")[["target_gen_s", "gen_s", "target_gp_s", "gp_s", "achieved_s", "target_s", "choices", "niterations", "n_seeds"]].mean().to_csv(out / "hybrid_times.csv")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, (key, title, pct) in zip(axes, [("rec_val", "Numeric recovery", True), ("exact_canonical", "Symbolic recovery", True), ("mdl_ratio", "MDL ratio to the law", False)]):
        p = pooled[pooled.metric == key].sort_values("ratio"); sc = 100 if pct else 1
        ax.fill_between(p.ratio, p.lo * sc, p.hi * sc, alpha=0.12, linewidth=0, color="#4c72b0")
        ax.errorbar(p.ratio, p.value * sc, yerr=[(p.value - p.lo) * sc, (p.hi - p.value) * sc], fmt="o", ms=4, capsize=2, color="#4c72b0")
        ax.set_xlabel("share of the budget for PySR (r)"); ax.set_title(title, fontsize=11); ax.grid(alpha=0.25)
        if pct: ax.set_ylabel("%")
    fig.suptitle(f"Flash-ANSR seeds + PySR, {df.pid.nunique()} problems, {df.target_s.max():.0f} s per problem", fontsize=11)
    fig.tight_layout(); fig.savefig(out / "hybrid_curve.png", dpi=150, bbox_inches="tight"); print("figure:", out / "hybrid_curve.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
