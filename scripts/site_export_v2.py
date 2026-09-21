"""Export a srbf benchmark release for the results site's explorer (results-site/explorer_v2.js), schema 2.

Reads the per-law judged rows of a campaign root (rows_full_<method>.csv written by the full-metric readout:
srbf's derive_metrics plus the 2026-07 site's derived columns, one row per law x rung) and writes

  <out.js>                       window.RESULTS_V2: release, catalogs, rungs, methods, the METRIC REGISTRY, and per
                                 method x catalog x rung cell: n laws, n successful, and for every metric
                                 [n defined, n finite, sum, sum of squares] (rates: [hits, n]); "e": the laws a
                                 metric can be defined for, where that is not every law; "w": how many values of a
                                 worst-value metric were filled in for failed predictions; status; timing.
  <out dir>/hist/<metric>.js     per-metric histograms of the same cells (pooled medians and the distribution view),
                                 loaded by the page on demand.
  <out dir>/paired.js            draw-1 paired contrasts per method pair x catalog x rung: 2x2 tables for the rate
                                 metrics (exact McNemar on the client), [n, sum d, sum d^2, wins, losses] for the
                                 continuous ones; a worst-value metric also as "<key>@answered", over the laws both
                                 methods answered.

LOCAL-ONLY METHODS. The methods named in --public go into the release files the site ships. A method named in
--private is written to --private-dir only, a directory outside the deployed tree (results-site/README.md,
"Local-only methods"), with the same schema; the page merges it when a local build loads it. Such a method is
described in --methods-file, a JSON list of entries shaped like METHODS below, kept outside the repository.

usage: site_export_v2.py <root> <release id> <out.js> [--title ...] [--notes ...] [--sizes suite_law_mu.json]
       [--public e2e,nesymres-100M,...] [--private KEY[,KEY] --private-dir DIR --methods-file FILE]"""
import argparse
import csv
import datetime as dt
import glob
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any
import numpy as np

# ---- registries -------------------------------------------------------------------------------------------------
# key, label, param, color, group, provenance, unit-file key, selection rule (how the method picks the
# one answer it submits; that is the method's own business, and the page says whose rule it is)
METHODS = [
    # a learned baseline carries its size in its name, like the Flash-ANSR series: parameters counted from the released
    # checkpoints (E2E model1.pt: embedder 6.2 M + encoder 12.6 M + decoder 74.6 M = 93.5 M; NeSymReS 100M.ckpt: 26.4 M --
    # the "100M" of that file name is the 100 million equations it was trained on, not its size)
    ("e2e", "E2E 93M", "candidates per bag", "#2f6fd0", "baseline", "upstream_default", "e2e",
     "Refines its decoded trees with BFGS and submits the one with the lowest error on the data it was given."),
    ("nesymres-100M", "NeSymReS 26M", "beam width", "#e8842a", "baseline", "upstream_default", "nesymres",
     "Beam search, then BFGS on the constants; submits the beam candidate that fits the data best."),
    ("PySR", "PySR", "iterations", "#d62728", "baseline", "upstream_default", "pysr",
     "Evolutionary search; submits the pick of its own hall of fame, its own accuracy-versus-complexity rule."),
    ("T8-3M", "Flash-ANSR T8-3M", "draws", "#8fcf8a", "flash-ansr", "author_blessed", None, None),
    ("T8-20M", "Flash-ANSR T8-20M", "draws", "#3e9b4a", "flash-ansr", "author_blessed", None, None),
    ("T8-120M", "Flash-ANSR T8-120M", "draws", "#1b5e20", "flash-ansr", "author_blessed", None, None),
    ("prior", "Flash-ANSR prior", "draws", "#9a9a9a", "reference", "author_blessed", None,
     "Draws skeletons from Flash-ANSR's training prior with no model and no data, then refines and picks them the way Flash-ANSR does: what the prior alone is worth.")]
FLASH_ANSR_SELECTION = ("Fits the constants of every candidate it draws and submits the one with the best two-part code: "
                        "(n/2) log2 FVU plus the description length of the expression in bits.")
RUNGS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 65536]
E2E_DEFAULT_MAX_RUNG = 256   # E2E is reported at its default settings only
CATALOG_GROUPS = {
    "physics": ["fastsrb", "feynman", "feynman-bonus", "srsd-dummy", "erbench-phybench", "erbench-densities", "physo-astro", "physo-class"],
    "classical": ["nguyen", "keijzer", "korns", "koza", "livermore", "livermore2", "vladislavleva", "jin", "neat", "pagie", "poly", "nonic", "sine", "meier", "r-rationals", "constant", "grammarvae"],
    "synthetic": ["erbench-syneq", "soose-fc", "soose-nc", "soose-wc"]}
NB = 128

# key, label, short, group, kind, higher_is_better (None = 1 is ideal / descriptive), tier, format, histogram (lo, hi, transform), description
# Rate metrics are defined for EVERY law (a failed prediction is a miss). So are the continuous metrics whose range
# has a worst value (WORST below): a failed prediction takes it. Every other continuous metric has no worst value --
# R^2, a length, a ratio of lengths can be arbitrarily bad -- and describes the answers that were made; the page
# marks a point as hollow when fewer laws than the reader's threshold have a value. Ground-truth descriptors cover
# every law.
# NO WALL-CLOCK METRIC IS PUBLISHED. Seconds measured where a unit happened to run depend on the node, its GPU
# and whatever shared it, so they are not comparable between methods. The only time this benchmark publishes is
# the reference-machine ladder in timing.json, which the site uses for the time AXIS and nothing else.
METRICS = [
    ("numeric_recovery_val", "Numeric recovery (vNRR)", "vNRR", "Recovery", "rate", True, "main", "pct", None,
     "Share of laws whose prediction reproduces the validation targets to float32 precision: FVU on the validation split at or below 2^-23. A failed prediction is a miss."),
    ("symbolic_recovery", "Symbolic recovery (SRR)", "SRR", "Recovery", "rate", True, "main", "pct", None,
     "Share of laws whose predicted expression has the same certified canonical form (SimpliPy, f64) as the law after both are simplified: structurally the same law, constants matched up to the judge's tolerance."),
    ("numeric_recovery_fit", "Numeric recovery on support (fNRR)", "fNRR", "Recovery", "rate", True, "more", "pct", None,
     "The float32-precision indicator on the support points the method was fitted on. fNRR above vNRR means fitting without generalizing."),
    ("skeleton_match_raw", "Exact skeleton match (raw)", "raw match", "Recovery", "rate", True, "more", "pct", None,
     "Share of laws whose predicted skeleton equals the law's skeleton token for token, without simplification: the 2026-07 site's symbolic recovery. Sensitive to spelling, kept for comparability."),
    ("numeric_recovery_relative_val", "Recovery relative to the reference law (val)", "ref. vNRR", "Recovery", "rate", True, "more", "pct", None,
     "Share of laws whose prediction fits the validation targets at least as well as the reference law itself does: FVU at or below the reference law's own FVU on the same targets, and never below the float32 bar. It differs from numeric recovery only where the targets are measurements that the accepted law does not reproduce exactly."),
    ("numeric_recovery_relative_fit", "Recovery relative to the reference law (support)", "ref. fNRR", "Recovery", "rate", True, "more", "pct", None,
     "The same criterion on the support points: FVU at or below the reference law's own FVU there, and never below the float32 bar."),
    ("success", "Prediction success rate", "success", "Recovery", "rate", True, "more", "pct", None,
     "Share of laws for which the method returned any evaluable expression at all: decoding, parsing, compiling and constant fitting completed."),
    ("log10_fvu_val", "log10 FVU (validation)", "log10 FVU val", "Fit quality", "cont", False, "main", "num2", (-17.0, 3.0, None),
     "Fraction of variance unexplained on the validation split, log10. -inf is a perfect fit and counts in the median; the mean is over finite values only."),
    ("log10_fvu_fit", "log10 FVU (support)", "log10 FVU fit", "Fit quality", "cont", False, "more", "num2", (-17.0, 3.0, None),
     "Fraction of variance unexplained on the support points, log10."),
    ("r2_val", "R² (validation)", "R² val", "Fit quality", "cont", True, "more", "num3", (-1.0, 1.0, None),
     "1 - FVU on the validation points: 1 is a perfect fit, 0 is as good as predicting the mean, and there is no lower bound. One diverging answer would decide a mean, so the median is shown whichever statistic is chosen."),
    ("r2_fit", "R² (support)", "R² fit", "Fit quality", "cont", True, "more", "num3", (-1.0, 1.0, None),
     "1 - FVU on the support points, without a lower bound; the median is shown whichever statistic is chosen."),
    ("mdl_ratio", "MDL ratio (pred / law)", "MDL ratio", "Length and complexity", "cont", None, "main", "ratio", (-4.0, 4.0, "log2"),
     "Description length of the prediction over the law's, both priced in the certified f64 canon (SimpliPy mu). 1 = as long as the law; the median is taken on the log2 scale."),
    ("predicted_mdl", "Predicted description length (bits)", "pred. MDL", "Length and complexity", "cont", False, "more", "num1", (0.0, 256.0, None),
     "Description length of the predicted expression in bits (SimpliPy mu, f64 canon)."),
    ("expr_length_ratio", "Expression length ratio (pred / law)", "length ratio", "Length and complexity", "cont", None, "main", "ratio", (-4.0, 4.0, "log2"),
     "Prefix-token length of the predicted skeleton over the simplified law's. 1 = same length; the median is taken on the log2 scale."),
    ("expr_length_ratio_abserr", "Length ratio |log2| error", "|log2 ratio|", "Length and complexity", "cont", False, "more", "num2", (0.0, 4.0, None),
     "Absolute log2 of the length ratio: 0 when the lengths match, 1 at twice or half the length."),
    ("predicted_skeleton_prefix_length", "Predicted skeleton length", "pred. length", "Length and complexity", "cont", False, "more", "num1", (0.0, 64.0, None),
     "Prefix-token length of the predicted skeleton."),
    ("predicted_n_constants", "Predicted constant count", "pred. constants", "Length and complexity", "cont", False, "more", "num1", (0.0, 32.0, None),
     "Number of fitted constants in the predicted skeleton."),
    ("n_constants_ratio", "Constant-count ratio (pred / law)", "constants ratio", "Length and complexity", "cont", None, "more", "ratio", (-4.0, 4.0, "log2"),
     "Predicted over true constant count, for laws with at least one constant; the median is taken on the log2 scale."),
    ("n_constants_delta", "Constant-count delta (pred - law)", "constants delta", "Length and complexity", "cont", False, "more", "num1", (-16.0, 16.0, None),
     "Predicted minus true constant count."),
    ("predicted_total_nestedness", "Predicted unary nestedness", "pred. nesting", "Length and complexity", "cont", False, "more", "num1", (0.0, 16.0, None),
     "Excess depth of directly nested unary operators in the prediction, summed over maximal chains: sin(log(x)) counts 1."),
    ("total_nestedness_delta", "Unary-nestedness delta (pred - law)", "nesting delta", "Length and complexity", "cont", False, "more", "num1", (-8.0, 8.0, None),
     "Predicted minus true unary nestedness."),
    ("f1_score", "Skeleton token F1", "token F1", "Skeleton similarity", "cont", True, "more", "num3", (0.0, 1.0, None),
     "F1 between the sets of distinct tokens of the predicted skeleton and of the simplified law's skeleton. A failed prediction counts 0, the end of the range."),
    ("precision_score", "Skeleton token precision", "token precision", "Skeleton similarity", "cont", True, "more", "num3", (0.0, 1.0, None),
     "Share of the prediction's distinct tokens that occur in the law's skeleton. A failed prediction counts 0: the precision of an empty answer is 0 by definition."),
    ("recall_score", "Skeleton token recall", "token recall", "Skeleton similarity", "cont", True, "more", "num3", (0.0, 1.0, None),
     "Share of the law's distinct tokens that occur in the prediction. A failed prediction counts 0."),
    ("edit_distance_norm", "Skeleton edit distance (normalized)", "edit dist. norm", "Skeleton similarity", "cont", False, "more", "num3", (0.0, 1.0, None),
     "Levenshtein distance between the prefix token sequences over the longer length, in [0, 1]. A failed prediction counts 1, the end of the range: the value an answer tends to as it grows without bound."),
    ("edit_distance", "Skeleton edit distance", "edit dist.", "Skeleton similarity", "cont", False, "more", "num1", (0.0, 64.0, None),
     "Levenshtein distance between the prefix token sequences."),
    ("zss_edit_distance", "Tree edit distance (ZSS)", "tree edit dist.", "Skeleton similarity", "cont", False, "more", "num1", (0.0, 128.0, None),
     "Zhang-Shasha tree edit distance between the expression trees."),
    ("f1_score_unique_variables", "Variable-set F1", "variables F1", "Skeleton similarity", "cont", True, "more", "num3", (0.0, 1.0, None),
     "F1 between the sets of input variables the prediction and the law use. A failed prediction counts 0."),
    ("precision_unique_variables", "Variable-set precision", "variables prec.", "Skeleton similarity", "cont", True, "more", "num3", (0.0, 1.0, None),
     "Share of the prediction's variables that the law uses. A failed prediction counts 0."),
    ("recall_unique_variables", "Variable-set recall", "variables recall", "Skeleton similarity", "cont", True, "more", "num3", (0.0, 1.0, None),
     "Share of the law's variables that the prediction uses. A failed prediction counts 0."),
    ("predicted_log_prob", "Predicted log-probability", "log-prob", "Model internals", "cont", True, "more", "num1", (-64.0, 0.0, None),
     "Log-probability of the selected candidate's token sequence under the model's decoder. Sampling methods only."),
    ("predicted_score", "Selection score", "score", "Model internals", "cont", False, "more", "num1", (-2048.0, 512.0, None),
     "The ranking score of the selected candidate as the method computed it (Flash-ANSR: the two-part code, lower is better). Only comparable within one method."),
    ("predicted_pareto_rank", "Pareto rank of the selection", "Pareto rank", "Model internals", "cont", False, "more", "num1", (0.0, 64.0, None),
     "Rank of the selected candidate on the method's FVU / length front (0 = on the front)."),
    ("skeleton_length", "Law skeleton length", "law length", "Ground truth", "cont", None, "more", "num1", (0.0, 64.0, None),
     "Prefix-token length of the simplified law: a property of the catalog, the same for every method."),
    ("ground_truth_mdl", "Law description length (bits)", "law MDL", "Ground truth", "cont", None, "more", "num1", (0.0, 256.0, None),
     "Description length of the law in bits (SimpliPy mu, f64 canon)."),
    ("n_constants", "Law constant count", "law constants", "Ground truth", "cont", None, "more", "num1", (0.0, 32.0, None),
     "Number of constants in the law's skeleton."),
    ("total_nestedness", "Law unary nestedness", "law nesting", "Ground truth", "cont", None, "more", "num1", (0.0, 16.0, None),
     "Excess depth of directly nested unary operators in the law."),
    ("n_variables", "Law variable count", "law variables", "Ground truth", "cont", None, "more", "num1", (0.0, 16.0, None),
     "Number of input variables the law uses.")]
# A metric that repeats another in EVERY published cell says nothing of its own, so the menu does not list it (its
# numbers stay in the cells). Recovery relative to the reference law is numeric recovery wherever the targets are
# computed from the law (reference FVU = 0); it is a metric of its own only once a catalog of measured data is in.
# The continuous metrics whose range has a worst value, and that value: a failed prediction takes it, so these are
# read over every law (the rows carry the value already: srbf.result_processing.WORST_VALUE, checked by the tests).
# The reader may leave the failed predictions out instead. Nothing is exported twice for that: a cell names how many
# of its values were filled in ("w"), and the page takes that many off the sums and out of the histogram bin of the
# worst value. Only a paired contrast cannot be undone that way, so it ships in both readings (key + ANSWERED).
WORST = {"f1_score": 0.0, "precision_score": 0.0, "recall_score": 0.0, "edit_distance_norm": 1.0,
         "f1_score_unique_variables": 0.0, "precision_unique_variables": 0.0, "recall_unique_variables": 0.0}
# Metrics without a bound on the bad side AND with a heavy tail there: one answer decides a mean, so the page reads
# the median whatever the reader chose. R^2 = 1 - FVU is a monotone map of the FVU, so near 1, where its own linear
# histogram cannot tell 0.99 from 0.9999, and below the histogram's lower end, the page reads the order statistic from
# the log10 FVU histogram instead (bins of 0.16 decades).
MEDIAN_ONLY = {"r2_val": "log10_fvu_val", "r2_fit": "log10_fvu_fit"}
# A metric that is undefined for some laws whatever the method does: the laws it could be defined for are the base of
# its share of valid results. (The constant-count ratio needs a law with at least one constant.)
ELIGIBLE = {"n_constants_ratio": lambda row: bool(row.get("n_constants"))}
ANSWERED = "@answered"   # suffix of a paired contrast taken over the laws BOTH methods answered
COPY_OF = {"numeric_recovery_relative_val": "numeric_recovery_val", "numeric_recovery_relative_fit": "numeric_recovery_fit"}


def listed_metrics(cells: dict[str, Any]) -> list[dict[str, Any]]:
    """The metric registry without the entries that copy another metric in every cell of `cells`."""
    def copies(key: str, of: str) -> bool:
        seen = False
        for per_catalog in cells.values():
            for per_rung in per_catalog.values():
                for cell in per_rung.values():
                    m = cell.get("m") or {}
                    if key in m or of in m:
                        seen = True
                        if m.get(key) != m.get(of):
                            return False
        return seen
    return [m for m in registry_json() if not (m["key"] in COPY_OF and copies(m["key"], COPY_OF[m["key"]]))]


RATE_KEYS = [m[0] for m in METRICS if m[4] == "rate"]
CONT_KEYS = [m[0] for m in METRICS if m[4] == "cont"]
HIST_SPECS = {m[0]: m[8] for m in METRICS if m[8]}
METRIC_HIGHER = {m[0]: m[5] for m in METRICS}
PAIRED_KEYS = ["numeric_recovery_val", "symbolic_recovery", "success", "log10_fvu_val", "mdl_ratio", "expr_length_ratio", "f1_score"]
# The Ranks view: within every law the methods are placed 1st, 2nd, ... on one continuous metric, a method without a
# usable answer last. A mean rank over any set of laws and any roster of methods follows from PAIRWISE outcomes alone
# (rank_i = 1 + sum_j [j beats i] + 0.5 [j ties i]), and pairwise counts add up over catalogs, so that is what ships:
# per pair x catalog x slot, [n laws, then (wins of the first, wins of the second) per rank key]. A slot is a rung
# ("64": both methods at that rung) or a time budget ("t3": each method at its largest rung the reference machine
# timed at or under 3 s per problem). The first key is the primary league.
RANK_KEYS = ["log10_fvu_val", "mdl_ratio", "expr_length_ratio", "f1_score"]   # R^2 orders the answers exactly as the FVU does
TIME_BUDGETS = [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]


def registry_json() -> list[dict[str, Any]]:
    out = []
    for k, label, short, group, kind, higher, tier, fmt, hist, desc in METRICS:
        m: dict[str, Any] = {"key": k, "label": label, "short": short, "group": group, "kind": kind, "higher": higher, "tier": tier, "fmt": fmt, "desc": desc}
        if hist:
            m["hist"] = {"lo": hist[0], "hi": hist[1], "tf": hist[2]}
        if k in WORST:
            m["worst"] = WORST[k]
        if k in MEDIAN_ONLY:
            m["median_via"] = MEDIAN_ONLY[k]
        out.append(m)
    return out


# ---- reading rows -----------------------------------------------------------------------------------------------
def fnum(s: str | None) -> float | None:
    if s is None or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_rows(root: str) -> dict[str, dict[tuple[str, int], dict[int, dict[str, Any]]]]:
    """{method: {(catalog, rung): {row: {metric: value}}}}, draw 1 only."""
    data: dict[str, dict[tuple[str, int], dict[int, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    files = sorted(set(glob.glob(os.path.join(root, "rows_full_*.csv")) + glob.glob(os.path.join(root, "*_rows_full.csv"))))
    for path in files:
        with open(path) as fh:
            for r in csv.DictReader(fh):
                if r.get("draw", "1") != "1":
                    continue
                vals: dict[str, float | None] = {}
                for k in RATE_KEYS:
                    v = fnum(r.get(k))
                    vals[k] = 0.0 if v is None else v
                for k in CONT_KEYS:
                    vals[k] = fnum(r.get(k))
                data[r["model"]][(r["catalog"], int(r["rung"]))][int(r["row"])] = vals
    return data


def transform(v: float | None, tf: str | None) -> float | None:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if tf == "log2":
        return math.log2(v) if v > 0 else (-math.inf if v == 0 else None)
    if tf == "log10":
        return math.log10(v) if v > 0 else (-math.inf if v == 0 else None)
    return v


def hist_of(values: list[float | None], lo: float, hi: float) -> list[Any] | None:
    """values already transformed; +-inf clipped into the edge bins; sparse [bin, count] pairs when few bins are used."""
    v = np.asarray([x for x in values if x is not None], float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return None
    idx = np.clip(np.floor((np.clip(v, lo, hi) - lo) / (hi - lo) * NB).astype(int), 0, NB - 1)
    h = np.bincount(idx, minlength=NB)
    nz = np.nonzero(h)[0]
    if nz.size <= NB // 4:
        return [[int(i), int(h[i])] for i in nz]
    return h.tolist()


def summarize_cell(rows: dict[int, dict[str, Any]], expected: int | None) -> dict[str, Any]:
    vals = list(rows.values())
    cell: dict[str, Any] = {
        "state": "complete" if expected is None or len(rows) >= expected else "partial",
        "n": len(vals), "ok": int(sum(1 for x in vals if x["success"])), "m": {}}
    for k in RATE_KEYS:
        cell["m"][k] = [int(sum(1 for x in vals if x[k])), len(vals)]
    for k in CONT_KEYS:
        xs = [x[k] for x in vals if x[k] is not None and not math.isnan(x[k])]
        if not xs:
            continue
        fin = np.asarray([x for x in xs if math.isfinite(x)], float)
        if k in MEDIAN_ONLY:   # no mean is read, and the sums of an unbounded metric overflow
            cell["m"][k] = [len(xs), int(fin.size), 0.0, 0.0]
        else:
            cell["m"][k] = [len(xs), int(fin.size), float(fin.sum()) if fin.size else 0.0, float((fin * fin).sum()) if fin.size else 0.0]
    for k, eligible in ELIGIBLE.items():
        cell.setdefault("e", {})[k] = int(sum(1 for x in vals if eligible(x)))
    for k in WORST:
        filled = int(sum(1 for x in vals if not x["success"] and x[k] is not None))
        if filled:
            cell.setdefault("w", {})[k] = filled
    return cell


def paired_cell(rows_a: dict[int, dict[str, Any]], rows_b: dict[int, dict[str, Any]]) -> dict[str, Any] | None:
    common = sorted(set(rows_a) & set(rows_b))
    if not common:
        return None
    out: dict[str, Any] = {}
    for k in PAIRED_KEYS:
        if k in RATE_KEYS:
            n11 = n10 = n01 = n00 = 0
            for i in common:
                a, b = bool(rows_a[i][k]), bool(rows_b[i][k])
                if a and b:
                    n11 += 1
                elif a:
                    n10 += 1
                elif b:
                    n01 += 1
                else:
                    n00 += 1
            out[k] = [n11, n10, n01, n00]
        else:
            tf = HIST_SPECS[k][2] if k in HIST_SPECS else None
            for name, laws in ((k, common),) + (((k + ANSWERED, [i for i in common if rows_a[i]["success"] and rows_b[i]["success"]]),) if k in WORST else ()):
                ds = []
                for i in laws:
                    va, vb = transform(rows_a[i][k], tf), transform(rows_b[i][k], tf)
                    if va is None or vb is None or not (math.isfinite(va) and math.isfinite(vb)):
                        continue
                    ds.append(va - vb)
                d = np.asarray(ds, float)
                out[name] = [int(d.size), float(d.sum()) if d.size else 0.0, float((d * d).sum()) if d.size else 0.0, int((d > 0).sum()), int((d < 0).sum())]
    return {"n": len(common), "m": out}


def budget_key(t: float) -> str:
    return "t" + ("%g" % t)


def rank_score(value: float | None, higher: bool | None) -> float:
    """Oriented so that larger is better; a law without a usable value scores -inf (placed last). `higher` None
    marks a ratio whose ideal is 1: closer to 1 on the log scale is better."""
    if value is None or math.isnan(value):
        return -math.inf
    if higher is True:
        return value
    if higher is False:
        return -value
    return -abs(math.log(value)) if value > 0 and math.isfinite(value) else -math.inf


def rank_pair_cell(rows_a: dict[int, dict[str, Any]], rows_b: dict[int, dict[str, Any]]) -> list[int] | None:
    common = sorted(set(rows_a) & set(rows_b))
    if not common:
        return None
    out = [len(common)]
    for k in RANK_KEYS:
        higher = METRIC_HIGHER[k]
        wa = wb = 0
        for i in common:
            sa, sb = rank_score(rows_a[i][k], higher), rank_score(rows_b[i][k], higher)
            if sa > sb:
                wa += 1
            elif sb > sa:
                wb += 1
        out += [wa, wb]
    return out


def rung_within(timing: dict[str, Any], key: str, budget: float, have: set[int]) -> int | None:
    """The largest rung of `key` the reference machine timed at or under `budget` seconds, among the rungs in `have`
    (the caller passes the rungs the method has finished)."""
    fits = [int(r) for r, sec in (timing.get(key) or {}).items() if sec is not None and sec <= budget and int(r) in have]
    return max(fits) if fits else None


# ---- status / catalogs / timing ---------------------------------------------------------------------------------
def load_units(root: str, ukey: str | None, key: str) -> int | None:
    for cand in ([os.path.join(root, f"units_{ukey}_d1.txt")] if ukey else []) + [os.path.join(root, "t8s1_units_draw1.txt")]:
        if not os.path.exists(cand):
            continue
        n = 0
        for line in open(cand):
            p = line.split()
            if not p:
                continue
            if ukey or p[0] == key or (len(p) > 1 and p[1] == key):
                n += 1
        if n:
            return n
    return None


def status_of(root: str, key: str, ukey: str | None, cells_done: int) -> list[int | None]:
    total = load_units(root, ukey, key)
    marks = os.path.join(root, "markers", f"{key}.txt")
    done = sum(1 for line in open(marks) if line.strip()) if os.path.exists(marks) else cells_done
    return [done, total if total is not None else (664 if not ukey else None)]


def catalog_meta(sizes_path: str | None, present: dict[str, int]) -> list[dict[str, Any]]:
    groups = {c: g for g, cs in CATALOG_GROUPS.items() for c in cs}
    per = json.load(open(sizes_path)).get("per_catalog", {}) if sizes_path and os.path.exists(sizes_path) else {}
    cats: list[dict[str, Any]] = []
    for c in sorted(set(per) | set(present), key=lambda c: (-(len(per.get(c, [])) or present.get(c, 0)), c)):   # size, then name: deterministic
        mu = np.asarray(per.get(c, []), float)
        cats.append({"key": c, "laws": int(mu.size) if mu.size else int(present.get(c, 0)), "group": groups.get(c, "other"),
                     "mu": [round(float(np.percentile(mu, q)), 1) for q in (25, 50, 75)] if mu.size else None})
    return cats


def rel_base(out_dir: str, site_dir: str) -> str:
    return os.path.relpath(out_dir, site_dir).replace(os.sep, "/") + "/"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("release")
    ap.add_argument("out")
    ap.add_argument("--title", default=None)
    ap.add_argument("--notes", default="")
    ap.add_argument("--sizes", default=None)
    ap.add_argument("--site-dir", default=None, help="results-site directory (default: two levels above out.js); base paths are relative to it")
    ap.add_argument("--public", default=",".join(m[0] for m in METHODS))
    ap.add_argument("--private", default="")
    ap.add_argument("--private-dir", default=None)
    ap.add_argument("--methods-file", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results-site", "private", "methods.json"),
                    help="JSON list of local-only method entries, shaped like METHODS (default: results-site/private/methods.json, if it exists)")
    a = ap.parse_args()
    if a.methods_file and os.path.exists(a.methods_file):
        METHODS.extend(tuple(entry) for entry in json.load(open(a.methods_file)))
    public = [k for k in a.public.split(",") if k]
    private = [k for k in a.private.split(",") if k]
    known = {m[0] for m in METHODS}
    for k in public + private:
        if k not in known:
            sys.exit(f"unknown method key {k!r}; add it to METHODS")
    if set(public) & set(private):
        sys.exit(f"a method cannot be both public and private: {sorted(set(public) & set(private))}")
    if private and not a.private_dir:
        sys.exit("--private needs --private-dir")
    out_dir = os.path.dirname(os.path.abspath(a.out))
    site_dir = os.path.abspath(a.site_dir) if a.site_dir else os.path.abspath(os.path.join(out_dir, "..", ".."))
    data = load_rows(a.root)
    present: dict[str, int] = defaultdict(int)
    for mk in data:
        for (c, r), rows in data[mk].items():
            present[c] = max(present[c], len(rows))
    cats = catalog_meta(a.sizes, present)
    sizes = {c["key"]: c["laws"] for c in cats}

    def usable(key: str, r: int) -> bool:
        return r in RUNGS and not (key == "e2e" and r > E2E_DEFAULT_MAX_RUNG)

    def contrasts(pairs: list[tuple[str, str]], paired: dict[str, Any]) -> None:
        for ka, kb in pairs:
            for (c, r), rows_a in data.get(ka, {}).items():
                rows_b = data.get(kb, {}).get((c, r))
                if not rows_b or not usable(ka, r) or not usable(kb, r):
                    continue
                pc = paired_cell(rows_a, rows_b)
                if pc:
                    paired.setdefault(ka + "|" + kb, {}).setdefault(c, {})[str(r)] = pc

    tpath0 = os.path.join(a.root, "timing.json")
    timing_all: dict[str, Any] = {}
    if os.path.exists(tpath0):
        timing_all = {k: v for k, v in json.load(open(tpath0)).items() if isinstance(v, dict) and not k.startswith("__")}

    def leagues(pairs: list[tuple[str, str]], keys: list[str]) -> dict[str, Any]:
        """Pairwise rank outcomes for `pairs`, and for every method of `keys` the rung a time budget buys it."""
        rungs_of = {k: {r for (_c, r) in data.get(k, {}) if usable(k, r)} for k in {x for p in pairs for x in p} | set(keys)}
        # a time budget buys a rung the method has FINISHED: every catalog, every law (the site shows no pooled number
        # for a rung that is still running, so a budget must not point at one)
        finished = {k: {r for r in rungs_of[k] if all(len(data[k].get((c, r), {})) >= n for c, n in sizes.items())} for k in rungs_of}
        at: dict[str, dict[str, int]] = {k: {} for k in rungs_of}
        for k in rungs_of:
            for t in TIME_BUDGETS:
                r = rung_within(timing_all, k, t, finished[k])
                if r is not None:
                    at[k][budget_key(t)] = r
        out: dict[str, Any] = {}
        # which rungs a time-budget outcome compared: an overlay is sealed less often than the release is refreshed,
        # and its outcomes against a public method are only valid while that method still sits on the same rung
        compared: dict[str, dict[str, list[int]]] = {}
        for ka, kb in pairs:
            slots = [(str(r), r, r) for r in sorted(rungs_of[ka] & rungs_of[kb])]
            timed = [(b, at[ka][b], at[kb][b]) for b in (budget_key(t) for t in TIME_BUDGETS) if b in at[ka] and b in at[kb]]
            compared[ka + "|" + kb] = {b: [ra, rb] for b, ra, rb in timed}
            slots += timed
            for slot, ra, rb in slots:
                for c in sizes:
                    rows_a, rows_b = data.get(ka, {}).get((c, ra)), data.get(kb, {}).get((c, rb))
                    if not rows_a or not rows_b:
                        continue
                    pc = rank_pair_cell(rows_a, rows_b)
                    if pc:
                        out.setdefault(ka + "|" + kb, {}).setdefault(c, {})[slot] = pc
        return {"keys": RANK_KEYS, "budgets": [budget_key(t) for t in TIME_BUDGETS], "seconds": TIME_BUDGETS,
                "at": {k: at[k] for k in keys}, "pairs": out, "rungs": {k: v for k, v in compared.items() if k in out and v}}

    def build(keys: list[str], base: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        methods = [m for m in METHODS if m[0] in keys]
        cells: dict[str, Any] = {}
        hists: dict[str, Any] = {k: {} for k in HIST_SPECS}
        status: dict[str, Any] = {}
        for key, label, param, color, group, prov, ukey, _sel in methods:
            cells[key] = {}
            for (c, r), rows in sorted(data.get(key, {}).items()):
                if not usable(key, r):
                    continue
                cells[key].setdefault(c, {})[str(r)] = summarize_cell(rows, sizes.get(c))
                for hk, (lo, hi, tf) in HIST_SPECS.items():
                    h = hist_of([transform(x[hk], tf) for x in rows.values()], lo, hi)
                    if h is not None:
                        hists[hk].setdefault(key, {}).setdefault(c, {})[str(r)] = h
            status[key] = status_of(a.root, key, ukey, sum(len(v) for v in cells[key].values()))
        paired: dict[str, Any] = {}
        mkeys = [m[0] for m in methods]
        contrasts([(ka, kb) for i, ka in enumerate(mkeys) for kb in mkeys[i + 1:]], paired)
        timing: dict[str, Any] = {}
        timing_note = ""
        tpath = os.path.join(a.root, "timing.json")
        if os.path.exists(tpath):
            t = json.load(open(tpath))
            timing = {k: v for k, v in t.items() if k in keys and isinstance(v, dict)}
            timing_note = t.get("note", "")
        payload = {"schema": 2, "base": base,
                   "release": {"id": a.release, "title": a.title or a.release, "notes": a.notes, "generated": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
                               "scoring": "Every method submits one answer per problem and chooses it by its own rule; the rule is named next to the method, along with who chose its configuration.",
                               "judge": "One judge for every answer: the predicted expression and the law are compared in one certified canonical form (SimpliPy acj-5-4-llm, f64), and numeric recovery is float32 precision on 512 held-out points."},
                   "catalogs": cats, "rungs": RUNGS, "nb": NB, "metrics": listed_metrics(cells), "paired_keys": PAIRED_KEYS, "rank_keys": RANK_KEYS,
                   # budget: what one rung of the ladder buys. "candidates" is a count a generative method draws;
                   # PySR's rungs are search iterations, which have no place on the candidate axis of the site.
                   "methods": [{"key": k, "label": l, "param": p, "budget": p if p in ("iterations", "seconds") else "candidates",
                                "color": col, "group": g, "provenance": prov,
                                "selection": sel or (FLASH_ANSR_SELECTION if g == "flash-ansr" else "")}
                               for k, l, p, col, g, prov, _, sel in methods],
                   "cells": cells, "status": status, "timing": timing, "timing_note": timing_note}
        return payload, hists, paired

    def write_set(payload: dict[str, Any], hists: dict[str, Any], paired: dict[str, Any], ranks: dict[str, Any], out_js: str, out_dir: str, var: str, note: str) -> None:
        os.makedirs(os.path.join(out_dir, "hist"), exist_ok=True)
        with open(out_js, "w") as fh:
            fh.write(f"window.{var} = " + json.dumps(payload, separators=(",", ":")) + ";\n")
        rel = json.dumps(payload["release"]["id"])
        for hk, per in hists.items():
            lo, hi, _ = HIST_SPECS[hk]
            with open(os.path.join(out_dir, "hist", hk + ".js"), "w") as fh:
                fh.write("window.RESULTS_V2_HIST=window.RESULTS_V2_HIST||{};(function(){var R=window.RESULTS_V2_HIST;R[%s]=R[%s]||{};var H=R[%s];H[%s]=H[%s]||{lo:%s,hi:%s,nb:%d,cells:{}};Object.assign(H[%s].cells,%s);})();\n" % (
                    rel, rel, rel, json.dumps(hk), json.dumps(hk), lo, hi, NB, json.dumps(hk), json.dumps(per, separators=(",", ":"))))
        with open(os.path.join(out_dir, "paired.js"), "w") as fh:
            fh.write("window.RESULTS_V2_PAIRED=window.RESULTS_V2_PAIRED||{};(function(){var R=window.RESULTS_V2_PAIRED;R[%s]=R[%s]||{};Object.assign(R[%s],%s);})();\n" % (
                rel, rel, rel, json.dumps(paired, separators=(",", ":"))))
        with open(os.path.join(out_dir, "ranks.js"), "w") as fh:   # merged like paired.js: an overlay adds its pairs and its own budget rungs
            fh.write("window.RESULTS_V2_RANKS=window.RESULTS_V2_RANKS||{};(function(){var R=window.RESULTS_V2_RANKS;R[%s]=R[%s]||{keys:%s,budgets:%s,seconds:%s,at:{},pairs:{}};R[%s].rungs=R[%s].rungs||{};Object.assign(R[%s].at,%s);Object.assign(R[%s].pairs,%s);Object.assign(R[%s].rungs,%s);})();\n" % (
                rel, rel, json.dumps(ranks["keys"]), json.dumps(ranks["budgets"]), json.dumps(ranks["seconds"]), rel, rel, rel, json.dumps(ranks["at"], separators=(",", ":")),
                rel, json.dumps(ranks["pairs"], separators=(",", ":")), rel, json.dumps(ranks.get("rungs", {}), separators=(",", ":"))))
        with_data = [k for k in payload["cells"] if payload["cells"][k]]
        print(f"{note}: {out_js} ({os.path.getsize(out_js) // 1024} kB), hist/ {len(hists)} files, paired {len(paired)} pairs; methods with data: {with_data}; status {payload['status']}")

    payload, hists, paired = build(public, rel_base(out_dir, site_dir))
    ranks = leagues([(ka, kb) for i, ka in enumerate(public) for kb in public[i + 1:]], public)
    write_set(payload, hists, paired, ranks, a.out, out_dir, "RESULTS_V2", f"public release {a.release}")
    if private:
        pdir = os.path.abspath(a.private_dir)
        os.makedirs(pdir, exist_ok=True)
        ppayload, phists, ppaired = build(private, rel_base(pdir, site_dir))
        contrasts([(ka, kb) for ka in private for kb in public], ppaired)   # private-vs-public contrasts stay private
        pranks = leagues([(ka, kb) for i, ka in enumerate(private) for kb in private[i + 1:]] + [(ka, kb) for ka in private for kb in public], private)
        write_set(ppayload, phists, ppaired, pranks, os.path.join(pdir, "results_v2_private.js"), pdir, "RESULTS_V2_PRIVATE", f"private overlay ({len(private)} method(s), never inside the deployed tree)")


if __name__ == "__main__":
    main()
