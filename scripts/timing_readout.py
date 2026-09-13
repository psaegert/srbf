"""Bootstrap read-out of the timing subset (scripts/freeze_timing_subset.py): per model and rung the POOLED mean
fit time -- catalog strata weighted by the FULL catalog sizes from the manifest, so the estimate targets the
whole suite's pooled mean, not the subset's -- with its 95 % bootstrap CI (resampling within strata), the
weighted median, the macro (per-catalog) mean, the generation / refinement split, and paired ratios against a
reference model on the same instances. Secondary: the pooled fit time as a power law a * draws^beta across the rungs.

  timing_readout.py --root ROOT --manifest ROOT/hybrid_data/timing_subset.json --models t8-20m,t8-3m,t8-120m \\
      [--reference t8-20m] [--results-dir results/evaluation/timing] [--rungs 1,2,...] [--file-pattern choices_{rung:06d}.pkl]
      [--n-boot 4000] [--seed 0] [--out report.md] [--json report.json] [--strict]

One joint resample of problems per replicate (the same rows for every model and rung), so the paired ratios and
the across-rung fit see the instance pairing. Rows with a missing or infinite fit_time (failed fits) are counted
and excluded. --strict fails when a file holds fewer rows than the manifest's count for that catalog.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np


def load_rung(root: Path, results_dir: str, model: str, catalogs: dict, rung: int, pattern: str, strict: bool):
    """Per catalog: arrays of fit / generation / refinement time (NaN where missing) in eval_row_index order."""
    out, missing_files, short = {}, [], []
    for e, m in catalogs.items():
        path = root / results_dir / model / e / pattern.format(rung=rung)
        if not path.exists():
            missing_files.append(e); continue
        with open(path, "rb") as fh:
            snap = pickle.load(fh)
        n = int(m["count"])
        idx = snap.get("eval_row_index") or list(range(len(snap["fit_time"])))
        cols = {}
        for key in ("fit_time", "generation_time", "refinement_time"):
            arr = np.full(n, np.nan)
            for i, v in zip(idx, snap.get(key) or []):
                if i is None or int(i) >= n:
                    continue
                try:
                    f = float(v)
                except (TypeError, ValueError):
                    continue
                arr[int(i)] = f if np.isfinite(f) else np.nan
            cols[key] = arr
        if len(snap["fit_time"]) < n:
            short.append(f"{e}:{len(snap['fit_time'])}/{n}")
        out[e] = cols
    if strict and (missing_files or short):
        raise SystemExit(f"{model} rung {rung}: missing {missing_files}, short {short}")
    return out, missing_files, short


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values); v, w = values[order], weights[order]
    cum = np.cumsum(w); cum /= cum[-1]
    return float(v[np.searchsorted(cum, q)])


def estimates(cols_by_cat: dict, weights: dict, draws: dict) -> dict:
    """Point estimates on the identity draw: pooled (size-weighted) mean, weighted median, macro mean, subset
    mean, the generation share and the row counts."""
    b = boot_estimates(cols_by_cat, weights, {e: d[None, :] for e, d in draws.items()})
    n_all = sum(c["fit_time"].size for c in cols_by_cat.values())
    n_ok = int(sum(np.isfinite(c["fit_time"]).sum() for c in cols_by_cat.values()))
    return {"pooled_mean": float(b["pooled_mean"][0]), "median": float(b["median"][0]), "macro_mean": float(b["macro_mean"][0]),
            "subset_mean": float(b["subset_mean"][0]), "gen_share": float(b["gen_share"][0]), "n_ok": n_ok, "n": n_all}


def boot_estimates(cols_by_cat: dict, weights: dict, draws: dict) -> dict:
    """Per replicate (``draws[e]``: a (B, n_e) matrix of row indices into catalog e): the pooled mean, the
    size-weighted median, the macro mean, the subset mean and the generation share, as length-B arrays."""
    B = next(iter(draws.values())).shape[0]
    pooled = np.zeros(B); wsum = np.zeros(B); macro = np.zeros(B); n_cat = np.zeros(B)
    gen = np.zeros(B); tot = np.zeros(B); cnt = np.zeros(B)
    vals, wts = [], []
    for e, cols in cols_by_cat.items():
        t = cols["fit_time"][draws[e]]                       # (B, n_e)
        ok = np.isfinite(t); k = ok.sum(axis=1)               # rows with a finite fit time per replicate
        m = np.where(k > 0, np.nansum(np.where(ok, t, 0.0), axis=1) / np.maximum(k, 1), np.nan)
        w = weights[e]; has = k > 0
        pooled[has] += w * m[has]; wsum[has] += w; macro[has] += m[has]; n_cat[has] += 1
        tot[has] += np.nansum(np.where(ok, t, 0.0), axis=1)[has]; cnt[has] += k[has]
        g = cols["generation_time"][draws[e]]; gok = ok & np.isfinite(g)
        gm = np.where(gok.sum(axis=1) > 0, np.nansum(np.where(gok, g, 0.0), axis=1) / np.maximum(gok.sum(axis=1), 1), 0.0)
        gen[has] += w * gm[has]
        vals.append(np.where(ok, t, np.nan)); wts.append(np.where(ok, w / np.maximum(k, 1)[:, None], 0.0))
    V = np.concatenate(vals, axis=1); W = np.concatenate(wts, axis=1)   # (B, N)
    order = np.argsort(np.where(np.isfinite(V), V, np.inf), axis=1)
    Vs = np.take_along_axis(V, order, axis=1); Ws = np.take_along_axis(W, order, axis=1)
    cum = np.cumsum(Ws, axis=1); cum /= np.maximum(cum[:, -1:], 1e-300)
    pos = (cum >= 0.5).argmax(axis=1)
    median = Vs[np.arange(B), pos]
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"pooled_mean": pooled / wsum, "median": median, "macro_mean": macro / n_cat, "subset_mean": tot / cnt,
                "gen_share": (gen / wsum) / (pooled / wsum)}


def ci(samples: np.ndarray) -> tuple[float, float]:
    s = samples[np.isfinite(samples)]
    return (float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))) if s.size else (np.nan, np.nan)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--models", required=True, help="comma-separated result directory names")
    ap.add_argument("--reference", help="model the paired ratios are taken against")
    ap.add_argument("--results-dir", default="results/evaluation/timing")
    ap.add_argument("--rungs", help="comma-separated (default: 1..65536 by doubling, those with files)")
    ap.add_argument("--file-pattern", default="choices_{rung:06d}.pkl")
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="markdown report path (default: stdout)")
    ap.add_argument("--json", help="machine-readable dump")
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()
    root = Path(a.root)
    manifest = json.loads(Path(a.manifest).read_text())
    catalogs = manifest["catalogs"]
    weights = {e: float(m["weight"]) for e, m in catalogs.items()}
    models = a.models.split(",")
    rungs = [int(r) for r in a.rungs.split(",")] if a.rungs else [2 ** k for k in range(17)]
    rng = np.random.default_rng(a.seed)
    # one joint resample per replicate: the same rows for every model and rung
    draws = {e: rng.integers(0, int(m["count"]), (a.n_boot, int(m["count"]))) for e, m in catalogs.items()}
    ident = {e: np.arange(int(m["count"])) for e, m in catalogs.items()}

    data = {}  # (model, rung) -> cols_by_cat
    notes = []
    for model in models:
        for rung in rungs:
            cols, missing, short = load_rung(root, a.results_dir, model, catalogs, rung, a.file_pattern, a.strict)
            if not cols:
                continue
            if missing:
                notes.append(f"{model} rung {rung}: {len(missing)} catalogs without a file ({', '.join(missing[:5])}{'...' if len(missing) > 5 else ''})")
            if short:
                notes.append(f"{model} rung {rung}: partial files {' '.join(short)}")
            data[(model, rung)] = cols

    report, dump = [], {"manifest": a.manifest, "n_boot": a.n_boot, "models": {}, "ratios": {}, "notes": notes}
    lines = report.append
    lines(f"# Timing read-out: {len(models)} models, {manifest['total_problems']} frozen problems, {a.n_boot} joint resamples\n")
    lines("Pooled mean = catalog means weighted by the full catalog sizes (target: the whole suite's pooled mean); "
          "CI = 95 % percentile bootstrap within strata; median = size-weighted; macro = plain mean over catalogs.\n")
    point = {}
    for model in models:
        lines(f"## {model}\n")
        lines("| rung | n ok / n | pooled mean [s] | 95 % CI | +- % | median [s] | macro mean [s] | subset mean [s] | gen share |")
        lines("|---:|---:|---:|---|---:|---:|---:|---:|---:|")
        dump["models"][model] = {}
        for rung in rungs:
            cols = data.get((model, rung))
            if cols is None:
                continue
            est = estimates(cols, weights, ident)
            bo = boot_estimates(cols, weights, draws)
            boot = np.stack([bo["pooled_mean"], bo["median"], bo["macro_mean"]], axis=1)
            lo, hi = ci(boot[:, 0]); mlo, mhi = ci(boot[:, 1])
            rel = 100 * (hi - lo) / 2 / est["pooled_mean"] if est["pooled_mean"] > 0 else np.nan
            point[(model, rung)] = (est, boot[:, 0])
            lines(f"| {rung} | {est['n_ok']} / {est['n']} | {est['pooled_mean']:.3f} | [{lo:.3f}, {hi:.3f}] | {rel:.1f} | "
                  f"{est['median']:.3f} [{mlo:.3f}, {mhi:.3f}] | {est['macro_mean']:.3f} | {est['subset_mean']:.3f} | "
                  f"{est['gen_share']:.2f} |")
            dump["models"][model][rung] = {**{k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in est.items()},
                                           "pooled_ci": [lo, hi], "median_ci": [mlo, mhi], "rel_halfwidth_pct": rel}
        # secondary: pooled time ~ a * draws^beta across the rungs measured for this model (log-log least squares)
        rs = [r for r in rungs if (model, r) in point]
        if len(rs) >= 3:
            x = np.log(np.array(rs, float)); y = np.log(np.array([point[(model, r)][0]["pooled_mean"] for r in rs]))
            A = np.vstack([np.ones_like(x), x]).T
            (la, beta), *_ = np.linalg.lstsq(A, y, rcond=None)
            Y = np.log(np.stack([point[(model, r)][1] for r in rs], axis=1))   # (B, n_rungs)
            bs = np.linalg.lstsq(A, Y.T, rcond=None)[0].T
            alo, ahi = ci(np.exp(bs[:, 0])); blo, bhi = ci(bs[:, 1])
            resid = 100 * np.abs(np.exp(y) - np.exp(la + beta * x)) / np.exp(y)
            lines(f"\nPooled time ~ a * draws^beta over rungs {rs[0]}..{rs[-1]}: a = {np.exp(la):.3f} s [{alo:.3f}, {ahi:.3f}], "
                  f"beta = {beta:.3f} [{blo:.3f}, {bhi:.3f}]; worst rung residual {resid.max():.1f} % "
                  f"(a summary, not the estimate: use the per-rung rows).")
            dump["models"][model]["power_law"] = {"a": float(np.exp(la)), "beta": float(beta), "a_ci": [alo, ahi], "beta_ci": [blo, bhi],
                                                  "max_resid_pct": float(resid.max())}
        lines("")
    if a.reference and a.reference in models:
        lines(f"## Paired ratios of the pooled mean against {a.reference} (same instances, joint resamples)\n")
        lines("| model | rung | ratio | 95 % CI |")
        lines("|---|---:|---:|---|")
        for model in models:
            if model == a.reference:
                continue
            for rung in rungs:
                if (model, rung) not in point or (a.reference, rung) not in point:
                    continue
                num, den = point[(model, rung)], point[(a.reference, rung)]
                ratio = num[0]["pooled_mean"] / den[0]["pooled_mean"]
                rlo, rhi = ci(num[1] / den[1])
                lines(f"| {model} | {rung} | {ratio:.3f} | [{rlo:.3f}, {rhi:.3f}] |")
                dump["ratios"][f"{model}/{a.reference}@{rung}"] = {"ratio": ratio, "ci": [rlo, rhi]}
        lines("")
    if notes:
        lines("## Notes\n")
        for n in notes:
            lines(f"- {n}")
    text = "\n".join(report)
    if a.out:
        Path(a.out).write_text(text + "\n"); print("wrote", a.out)
    else:
        print(text)
    if a.json:
        Path(a.json).write_text(json.dumps(dump, indent=1, default=float)); print("wrote", a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
