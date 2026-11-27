"""The simulations behind docs/averaging.md, "How well it works": the three-level averaging of srbf.aggregate
(benchmarks > subsets > problems, combine_nested) on two structures of subset sizes, "release" (145 benchmarks of 255
subsets) and "small" (29 benchmarks of 38 subsets).

  means      coverage of the interval for these subsets (the weighted average of the subsets' true means, with the
             weights the interval used), and agreement of the mean and both variances with three-level REML
  twolevel   the two-level average's coverage on the same subset sizes, every subset a benchmark of its own, with and
             without the subsets of fewer than five problems
  rates      coverage of the score-test interval near 0 and 100 % too, against what the average estimates for these
             subsets (the beta-binomial fit to every redraw's effective counts: the expected likelihood's maximum),
             agreement of the two-stage estimate with the full nested beta-binomial model fitted by maximum likelihood
             (FullModel), and rho_S against its true value
  rho        rho_S by Williams' rule against a maximum-likelihood fit of rho_S with every benchmark's rate fitted too
  moderated  the subsets' variances moderated by an empirical-Bayes prior (srbf.aggregate.set_estimates) against the
             rule before it (a subset with one problem or one value took the pooled variance, every other its own),
             on the gate below

THE GATE FOR THE MODERATED VARIANCES (fixed before it was run). Settings: three structures ("release", "small", and
"tiny": 145 benchmarks of 255 subsets, four in five of 1 to 4 problems) x the five spreads of "means" x four kinds of
within-subset spread (every subset sd 1; sd log-normal across subsets, log sd ~ N(0, 0.5^2) and N(0, 1); log sd = 0.5
times the subset's standardised true mean), 4 draws of the subsets x 200 redraws of the problems each, R = 800 redraws
per setting. Per setting and estimator: coverage of what the interval claims (the weighted average of these subsets'
true means, with the estimator's own weights) with its Monte Carlo standard error, the bias and the RMSE of the mean
(against the generating mean, 0, and against that weighted average), and the mean interval width. The moderated
variances are adopted only if, in every one of the 60 settings, (1) their coverage minus the rule's is at least -2 times
the paired Monte Carlo standard error, sqrt(b + c - (b - c)^2 / R) / R with b and c the redraws only one of the two
covers, and (2) where the rule's coverage is below 0.95 - 2 sqrt(0.95 0.05 / R), theirs is not. Otherwise they are not
adopted. The interval's degrees of freedom stay sum (n - 1), EPS_VAR stays, and n0 keeps the pooled variance.

usage: python scripts/averaging_simulation.py {means,twolevel,rates,rho,moderated} OUT.json [--workers K] [--reps R]
       [--full small:6,release:2]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool
from typing import Any

import numpy as np
from scipy import optimize, special

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))   # this checkout's srbf
from srbf import aggregate as ag  # noqa: E402


# ---- structures -------------------------------------------------------------------------------------------------------
def structure(kind: str) -> list[list[int]]:
    """Subset sizes per benchmark. "release": 145 benchmarks, 255 subsets; "small": 29 benchmarks, 38 subsets;
    "tiny": the release's nesting with four subsets in five of 1 to 4 problems, the rest of 10 to 1,000."""
    if kind == "tiny":
        rng = np.random.default_rng(20261128)
        counts = [1] * 95 + [2] * 25 + [3] * 12 + [4] * 8 + [6] * 3 + [12] * 2
        return [[int(rng.integers(1, 5)) if rng.random() < 0.8 else int(round(10 ** rng.uniform(1, 3))) for _ in range(c)]
                for c in counts]
    rng = np.random.default_rng(20261127 if kind == "release" else 20260927)
    if kind == "release":
        counts = [1] * 95 + [2] * 25 + [3] * 12 + [4] * 8 + [6] * 3 + [12] * 2
    else:
        counts = [1] * 22 + [2] * 5 + [3] * 2
    sizes = [[int(round(10 ** rng.uniform(0, 3))) for _ in range(c)] for c in counts]
    sizes[0][0] = 5301
    return sizes


# ---- means ------------------------------------------------------------------------------------------------------------
def reml(groups_y: Any, groups_v: Any) -> tuple[float, float, float]:
    """Three-level REML of (mu, tau_B^2, tau_S^2) for subset estimates y with sampling variances v."""
    def nll(x: Any) -> tuple[float, float]:
        tb, ts = math.exp(x[0]), math.exp(x[1])
        logdet, A, Ay = 0.0, 0.0, 0.0
        parts = []
        for y, v in zip(groups_y, groups_v):
            d = np.asarray(v) + ts
            w = 1.0 / d
            W = w.sum()
            logdet += np.log(d).sum() + math.log1p(tb * W)
            a = W / (1.0 + tb * W)          # 1' V^-1 1
            ay = (w * y).sum() / (1.0 + tb * W)
            A += a
            Ay += ay
            parts.append((np.asarray(y), w, W))
        mu = Ay / A
        q = 0.0
        for y, w, W in parts:
            r = y - mu
            q += (w * r * r).sum() - tb * ((w * r).sum()) ** 2 / (1.0 + tb * W)
        return 0.5 * (logdet + math.log(A) + q), mu
    best: Any = None
    for x0 in ([0.0, 0.0], [-3.0, 0.0], [0.0, -3.0], [-5.0, -5.0]):
        r = optimize.minimize(lambda x: nll(x)[0], x0, method="L-BFGS-B", bounds=[(-25, 5), (-25, 5)])
        if best is None or r.fun < best.fun:
            best = r
    return nll(best.x)[1], math.exp(best.x[0]), math.exp(best.x[1])


def means_task(args: Any) -> dict[str, Any]:
    kind, tb, ts, draw, reps = args
    sizes = structure(kind)
    rng = np.random.default_rng([int(1000 * tb), int(1000 * ts), draw, 7])
    true_mu = [[rng.normal(ub, ts) for _ in g] for g, ub in ((g, rng.normal(0.0, tb)) for g in sizes)]
    cover, width, d_reml, t2b, t2s, r_t2b, r_t2s = 0, [], [], [], [], [], []
    for _ in range(reps):
        groups = []
        for g, mus in zip(sizes, true_mu):
            gs = []
            for n, mu in zip(g, mus):
                x = rng.normal(mu, 1.0, n)
                gs.append(ag.SetStat(n, float(x.sum()), float((x * x).sum())))
            groups.append(gs)
        c = ag.combine_nested(groups, "normal")
        target = sum(w * mu for ws, mus in zip(c.subset_weights, true_mu) for w, mu in zip(ws, mus))
        cover += c.lo <= target <= c.hi
        width.append(c.hi - c.lo)
        t2b.append(c.tau2)
        t2s.append(c.tau2_within)
        flat = [s for g in groups for s in g]
        y, v = ag.set_estimates(flat, "normal")
        gy, gv, at = [], [], 0
        for g in groups:
            gy.append(y[at:at + len(g)])
            gv.append(v[at:at + len(g)])
            at += len(g)
        mu_r, rb, rs = reml(gy, gv)
        d_reml.append(c.mu - mu_r)
        r_t2b.append(rb)
        r_t2s.append(rs)
    return {"kind": kind, "tau_B": tb, "tau_S": ts, "draw": draw, "reps": reps, "cover": cover / reps,
            "width": float(np.median(width)), "mean_abs_vs_reml": float(np.mean(np.abs(d_reml))),
            "max_abs_vs_reml": float(np.max(np.abs(d_reml))), "tau2_B": float(np.mean(t2b)), "tau2_S": float(np.mean(t2s)),
            "reml_tau2_B": float(np.mean(r_t2b)), "reml_tau2_S": float(np.mean(r_t2s))}


# ---- rates ------------------------------------------------------------------------------------------------------------
def draw_rates(sizes: list[list[int]], m: float, nu_b: float, nu_s: float | None, rng: Any) -> list[list[float]]:
    out = []
    for g in sizes:
        pb = rng.beta(max(m * nu_b, 1e-300), max((1 - m) * nu_b, 1e-300))
        out.append([pb if nu_s is None else rng.beta(max(pb * nu_s, 1e-300), max((1 - pb) * nu_s, 1e-300)) for _ in g])
    return out


def draw_counts(sizes: list[list[int]], rates: list[list[float]], rng: Any) -> list[list[Any]]:
    return [[ag.SetStat(n, float(k), float(k)) for n, k in zip(g, rng.binomial(g, ps))] for g, ps in zip(sizes, rates)]


class FullModel:
    """The nested beta-binomial: p_b ~ Beta(m nu_B, (1 - m) nu_B), p_bs | p_b ~ Beta(p_b nu_S, (1 - p_b) nu_S),
    k_bs ~ Binomial(n_bs, p_bs); each benchmark's likelihood integrates over p_b as a Stieltjes sum over cells of the
    logit scale (a coarse grid over [-40, 40] and a fine one around the benchmark's pooled rate), each cell's beta mass
    exact (the regularised incomplete beta function), the rest of the integrand at the cell's middle."""

    def __init__(self, groups: Any, coarse: float = 0.1, fine_cells: int = 300) -> None:
        self.parts = []
        for g in groups:
            n = np.array([st.n for st in g], float)
            k = np.array([st.s1 for st in g], float)
            N, K = n.sum(), k.sum()
            p0 = min(max((K + 0.5) / (N + 1.0), 1e-12), 1 - 1e-12)
            e0, w = math.log(p0 / (1 - p0)), 1.0 / math.sqrt(N * p0 * (1 - p0) + 1.0)
            edges = np.unique(np.concatenate([np.arange(-40.0, 40.0 + 1e-9, coarse),
                                              np.linspace(e0 - 12 * w, e0 + 12 * w, fine_cells + 1)]))
            edges = edges[(edges >= -40.0) & (edges <= 40.0)]
            mids = np.concatenate([[edges[0] - 1.0], 0.5 * (edges[:-1] + edges[1:]), [edges[-1] + 1.0]])
            low = edges <= 0      # the cell masses from below on the lower half, from above on the upper half
            lbin = special.gammaln(n + 1) - special.gammaln(k + 1) - special.gammaln(n - k + 1)
            self.parts.append((n, k, special.expit(edges), special.expit(-edges), low, special.expit(mids), special.expit(-mids), lbin.sum()))

    def loglik(self, x: Any) -> float:
        m, nu_b, nu_s = special.expit(x[0]), math.exp(x[1]), math.exp(x[2])
        a, b = m * nu_b, (1 - m) * nu_b
        tot = 0.0
        for n, k, pe, qe, low, pm, qm, lbin in self.parts:
            cdf = np.where(low, special.betainc(a, b, pe), 0.0)           # P(p <= edge), below the middle
            sf = np.where(~low, special.betainc(b, a, qe), 0.0)           # P(p > edge), above it
            inner = np.diff(np.where(low, cdf, 1.0 - sf))
            # a cell that straddles 0 or lies above it: its mass from the survival function, exactly
            up = ~low[:-1]
            inner[up] = sf[:-1][up] - sf[1:][up]
            mass = np.concatenate([[cdf[0] if low[0] else 1.0 - sf[0]], inner, [sf[-1] if not low[-1] else 1.0 - cdf[-1]]])
            with np.errstate(divide="ignore"):
                lm = np.log(np.maximum(mass, 0.0))
            pa, pb = pm[:, None] * nu_s, qm[:, None] * nu_s
            lf = (special.betaln(k[None, :] + pa, n[None, :] - k[None, :] + pb) - special.betaln(pa, pb)).sum(axis=1)
            tot += special.logsumexp(lm + lf) + lbin
        return tot

    def fit(self, x0: Any) -> tuple[Any, float]:
        best: Any = None
        for start in x0:
            r = optimize.minimize(lambda x: -self.loglik(x), start, method="L-BFGS-B",
                                  bounds=[(-30.0, 30.0), (-12.0, 16.0), (-12.0, 16.0)], options={"maxiter": 500})
            if best is None or r.fun < best.fun:
                best = r
        return best.x, -best.fun


def rates_task(args: Any) -> dict[str, Any]:
    kind, m, nu_b, nu_s, draw, reps, full = args
    sizes = structure(kind)
    rng = np.random.default_rng([int(1e4 * m), int(10 * nu_b), int(10 * (nu_s or 0)), draw, 11])
    rates = draw_rates(sizes, m, nu_b, nu_s, rng)
    fits, effs, rhos, full_d, full_rel, flat_fits, flat_sets = [], [], [], [], [], [], []
    t0 = time.time()
    for r in range(reps):
        groups = draw_counts(sizes, rates, rng)
        c = ag.combine_nested(groups, "logit")
        rho = ag.within_rho(groups)
        effs.extend(ag.effective_sets(groups, rho))
        fits.append((c.mu, c.lo, c.hi))
        rhos.append(rho)
        flat = [s for g in groups for s in g]           # the two-level average: every subset a benchmark of its own
        f = ag.combine(flat, "logit")
        assert f is not None
        flat_fits.append((f.lo, f.hi))
        flat_sets.extend(flat)
        if r < full:
            fm = FullModel(groups)
            eta = math.log(max(c.mu, 1e-9) / max(1 - c.mu, 1e-9))
            x, _ = fm.fit([[eta, math.log(max(nu_b, 0.1)), math.log(nu_s or 1e3)], [eta, 0.0, 0.0], [eta, 2.0, 5.0]])
            mf = float(special.expit(x[0]))
            full_d.append(c.mu - mf)
            full_rel.append(abs(c.mu - mf) / max((c.hi - c.lo) / 2, 1e-12))
    eta_t, lam_t = ag.rate_fit(effs)     # the two-stage estimator's pseudo-true value for these subsets (all redraws)
    target = 1.0 if eta_t == math.inf else 0.0 if eta_t == -math.inf else float(special.expit(eta_t))
    cover = sum(lo <= target <= hi for _, lo, hi in fits) / reps
    eta_f, _ = ag.rate_fit(flat_sets)
    target_f = 1.0 if eta_f == math.inf else 0.0 if eta_f == -math.inf else float(special.expit(eta_f))
    flat_cover = sum(lo <= target_f <= hi for lo, hi in flat_fits) / reps
    return {"kind": kind, "m": m, "nu_B": nu_b, "nu_S": nu_s, "draw": draw, "reps": reps, "target": target,
            "true_mean_of_sets": float(np.mean([np.mean(ps) for ps in rates])),
            "cover": cover, "width": float(np.median([hi - lo for _, lo, hi in fits])), "flat_cover": flat_cover,
            "bias": float(np.mean([mu for mu, _, _ in fits]) - target),
            "rho_true": 0.0 if nu_s is None else 1.0 / (1.0 + nu_s), "rho_mean": float(np.mean(rhos)),
            "full_n": len(full_d), "full_mean_abs_pp": float(100 * np.mean(np.abs(full_d))) if full_d else None,
            "full_max_abs_pp": float(100 * np.max(np.abs(full_d))) if full_d else None,
            "full_mean_rel_halfwidth": float(np.mean(full_rel)) if full_rel else None,
            "full_max_rel_halfwidth": float(np.max(full_rel)) if full_rel else None,
            "seconds": time.time() - t0}


def twolevel_task(args: Any) -> dict[str, Any]:
    kind, drop_small, draw, reps = args
    sizes = [n for g in structure(kind) for n in g if n >= (5 if drop_small else 1)]
    rng = np.random.default_rng([draw, int(drop_small), 3])
    mus = rng.normal(0.0, 1.0, len(sizes))
    hits = 0
    for _ in range(reps):
        sets = [ag.SetStat(n, float(x.sum()), float((x * x).sum())) for n, x in ((n, rng.normal(mu, 1.0, n)) for n, mu in zip(sizes, mus))]
        c = ag.combine(sets, "normal")
        hits += c.lo <= sum(w * mu for w, mu in zip(c.weights, mus)) <= c.hi
    return {"kind": kind, "drop_small": drop_small, "sets": len(sizes), "draw": draw, "reps": reps, "cover": hits / reps}


def profile_ml_rho(groups: Any) -> float:
    """rho_S by maximum likelihood with every benchmark's rate p_b fitted alongside (the subsets' counts beta-binomial
    around p_b): the estimate the moment rule is compared with."""
    data = [(np.array([s.n for s in g], float), np.array([s.s1 for s in g], float)) for g in groups if len(g) > 1]

    def nll(lnu: float) -> float:
        nu, tot = math.exp(lnu), 0.0
        for n, k in data:
            def f(e: float) -> float:
                a, b = special.expit(e) * nu, special.expit(-e) * nu
                return -(special.betaln(k + a, n - k + b) - special.betaln(a, b)).sum()
            tot += optimize.minimize_scalar(f, bounds=(-25, 25), method="bounded", options={"xatol": 1e-6}).fun
        return tot
    r = optimize.minimize_scalar(nll, bounds=(-8, 14), method="bounded", options={"xatol": 1e-4})
    return 1 / (1 + math.exp(r.x))


def rho_task(args: Any) -> dict[str, Any]:
    kind, m, nu_b, nu_s, reps = args
    sizes = structure(kind)
    rng = np.random.default_rng([int(1e4 * m), int(10 * nu_b), int(10 * nu_s), 5])
    w, ml = [], []
    for _ in range(reps):
        groups = draw_counts(sizes, draw_rates(sizes, m, nu_b, nu_s, rng), rng)
        w.append(ag.within_rho(groups))
        ml.append(profile_ml_rho(groups))
    return {"kind": kind, "m": m, "nu_B": nu_b, "nu_S": nu_s, "reps": reps, "rho_true": 1 / (1 + nu_s),
            "williams": float(np.mean(w)), "profile_ml": float(np.mean(ml))}


def set_estimates_before(sets: Any, scale: str) -> tuple[list[float], list[float]]:
    """srbf.aggregate.set_estimates before the moderated variances: a set with one problem, or whose problems all have
    one value, takes the variance pooled over the sets; every other set its own."""
    sets = [st for st in sets if st.n > 0]
    pooled = ag.pooled_variance(sets)
    ys, vs = [], []
    for st in sets:
        own = st.var if st.var is not None and st.var > 0 else None
        ys.append(st.mean)
        vs.append(max(ag.EPS_VAR, (own if own is not None else pooled) / st.n))
    return ys, vs


SPREADS = [(1.0, 0.0), (1.0, 0.3), (1.0, 1.0), (0.3, 1.0), (0.0, 1.0)]
SDS = ("equal", "lognormal-0.5", "lognormal-1", "mean-correlated")


def moderated_task(args: Any) -> dict[str, Any]:
    kind, tb, ts, sd_kind, draw, reps = args
    sizes = structure(kind)
    rng = np.random.default_rng([int(1000 * tb), int(1000 * ts), SDS.index(sd_kind), draw, 13, ["release", "small", "tiny"].index(kind)])
    truth = [[rng.normal(ub, ts) for _ in g] for g, ub in ((g, rng.normal(0.0, tb)) for g in sizes)]
    flat = np.array([m for g in truth for m in g])
    if sd_kind == "equal":
        sds = np.ones(flat.size)
    elif sd_kind.startswith("lognormal"):
        sds = np.exp(rng.normal(0.0, float(sd_kind.split("-")[1]), flat.size))
    else:
        sds = np.exp(0.5 * (flat - flat.mean()) / (flat.std() or 1.0))
    sd_of, at = [], 0
    for g in sizes:
        sd_of.append(list(sds[at:at + len(g)]))
        at += len(g)
    out: dict[str, Any] = {"kind": kind, "tau_B": tb, "tau_S": ts, "sd": sd_kind, "draw": draw, "reps": reps}
    hits = {"before": [], "moderated": []}
    est = {"before": [], "moderated": []}
    target = {"before": [], "moderated": []}
    width = {"before": [], "moderated": []}
    original = ag.set_estimates
    for _ in range(reps):
        groups = [[ag.SetStat(n, float(x.sum()), float((x * x).sum())) for n, x in
                   ((n, rng.normal(mu, sd, n)) for n, mu, sd in zip(g, mus, sdg))] for g, mus, sdg in zip(sizes, truth, sd_of)]
        for name in ("before", "moderated"):
            ag.set_estimates = set_estimates_before if name == "before" else original
            try:
                c = ag.combine_nested(groups, "normal")
            finally:
                ag.set_estimates = original
            assert c is not None and c.subset_weights is not None
            t = sum(w * mu for ws, mus in zip(c.subset_weights, truth) for w, mu in zip(ws, mus))
            hits[name].append(bool(c.lo <= t <= c.hi))
            est[name].append(c.mu)
            target[name].append(t)
            width[name].append(c.hi - c.lo)
    for name in hits:
        e, t = np.array(est[name]), np.array(target[name])
        out[name] = {"cover": float(np.mean(hits[name])), "bias": float(e.mean()), "rmse": float(np.sqrt((e ** 2).mean())),
                     "bias_target": float((e - t).mean()), "rmse_target": float(np.sqrt(((e - t) ** 2).mean())),
                     "width": float(np.mean(width[name]))}
    b = sum(1 for x, y in zip(hits["moderated"], hits["before"]) if x and not y)
    c_ = sum(1 for x, y in zip(hits["moderated"], hits["before"]) if y and not x)
    out["only_moderated"], out["only_before"] = b, c_
    return out


def gate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """The gate of the docstring over the draws of every setting: per setting the coverages with their Monte Carlo
    standard errors, the paired difference, and whether (1) and (2) hold."""
    by: dict[tuple[str, float, float, str], list[dict[str, Any]]] = {}
    for r in results:
        by.setdefault((r["kind"], r["tau_B"], r["tau_S"], r["sd"]), []).append(r)
    rows, ok = [], True
    for key, rs in sorted(by.items()):
        R = sum(r["reps"] for r in rs)
        cov = {name: sum(r[name]["cover"] * r["reps"] for r in rs) / R for name in ("before", "moderated")}
        b, c = sum(r["only_moderated"] for r in rs), sum(r["only_before"] for r in rs)
        se_diff = math.sqrt(max(0.0, b + c - (b - c) ** 2 / R)) / R
        floor = 0.95 - 2 * math.sqrt(0.95 * 0.05 / R)
        not_worse = cov["moderated"] - cov["before"] >= -2 * se_diff
        nominal = cov["before"] >= floor or cov["moderated"] >= floor
        ok = ok and not_worse and nominal
        rows.append({"setting": list(key), "R": R, "cover_before": cov["before"], "cover_moderated": cov["moderated"],
                     "se_before": math.sqrt(cov["before"] * (1 - cov["before"]) / R),
                     "se_moderated": math.sqrt(cov["moderated"] * (1 - cov["moderated"]) / R), "se_diff": se_diff,
                     "not_worse": not_worse, "nominal_where_short": nominal,
                     **{f"{f}_{name}": float(np.mean([r[name][f] for r in rs])) for name in ("before", "moderated")
                        for f in ("bias", "rmse", "bias_target", "rmse_target", "width")}})
    return {"adopt": ok, "settings": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["means", "twolevel", "rates", "rho", "moderated"])
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--reps", type=int, default=0)
    ap.add_argument("--full", default="", help="kind:count, redraws per draw fitted by the full model too (rates)")
    ap.add_argument("--kinds", default="small,release")
    a = ap.parse_args()
    kinds = a.kinds.split(",")
    if a.what == "means":
        reps = a.reps or 200
        settings = [(1.0, 0.0), (1.0, 0.3), (1.0, 1.0), (0.3, 1.0), (0.0, 1.0)]
        tasks: list[Any] = [(k, tb, ts, d, reps) for k in kinds for tb, ts in settings for d in range(4)]
        fn: Any = means_task
    elif a.what == "twolevel":
        reps = a.reps or 150
        tasks = [(k, drop, d, reps) for k in kinds for drop in (False, True) for d in range(4)]
        fn = twolevel_task
    elif a.what == "moderated":
        reps = a.reps or 200
        tasks = [(k, tb, ts, sd, d, reps) for k in ("release", "small", "tiny") for tb, ts in SPREADS for sd in SDS
                 for d in range(4)]
        fn = moderated_task
    elif a.what == "rho":
        reps = a.reps or 15
        tasks = [(k, m, nb, ns, reps) for k in kinds for m, nb, ns in ((0.3, 3.0, 3.0), (0.3, 20.0, 1.0), (0.05, 3.0, 3.0))]
        fn = rho_task
    else:
        reps = a.reps or 60
        full = {kv.split(":")[0]: int(kv.split(":")[1]) for kv in a.full.split(",") if kv}
        tasks = [(k, m, nb, ns, d, reps, full.get(k, 0)) for k in kinds for m in (0.002, 0.01, 0.05, 0.3, 0.95, 0.99, 0.998)
                 for nb, ns in [(3.0, None), (3.0, 3.0), (20.0, 1.0)] for d in range(3)]
        fn = rates_task
    with Pool(a.workers) as pool:
        res = []
        for i, r in enumerate(pool.imap_unordered(fn, tasks)):
            res.append(r)
            print(f"{i + 1}/{len(tasks)}", json.dumps(r), flush=True)
    if a.what == "moderated":
        verdict = gate(res)
        res = {"draws": res, "gate": verdict}
        print("gate:", "ADOPT" if verdict["adopt"] else "DO NOT ADOPT", flush=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
