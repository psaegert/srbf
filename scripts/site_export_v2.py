"""Export a srbf benchmark release for the results site's explorer (results-site/explorer_v2.js), schema 2.

Reads the per-problem judged rows of a campaign root (rows_full_<name>.csv or <name>_rows_full.csv, written by
`srbf table`: one row per problem and rung, every metric) and writes

  <out.js>                       window.RESULTS_V2: release, catalogs, rungs, methods, the METRIC REGISTRY, and per
                                 method x catalog x rung cell: n problems, n successful, and for every metric
                                 [n defined, n finite, sum, sum of squares] (rates: [hits, n]); "e": the problems a
                                 metric can be defined for, where that is not every problem; "w": how many values of a
                                 worst-value metric were filled in for failed predictions; status; timing.
  <out dir>/hist/<metric>.js     per-metric histograms of the same cells (pooled medians and the distribution view),
                                 loaded by the page on demand.
  <out dir>/paired.js            paired contrasts (a problem with itself within a draw, over the draws complete on both
                                 sides) per method pair x catalog x rung: 2x2 tables for the rate
                                 metrics (exact McNemar on the client), [n, sum d, sum d^2, better, worse] for the
                                 continuous ones; a worst-value metric also as "<key>@answered", over the problems both
                                 methods have a prediction for.

LOCAL-ONLY METHODS. The methods named in --public go into the release files the site ships. A method named in
--private is written to --private-dir only, a directory outside the deployed tree (results-site/README.md,
"Local-only methods"), with the same schema; the page merges it when a local build loads it. Such a method is
described in --methods-file, a JSON list of entries shaped like METHODS below, kept outside the repository.

usage: site_export_v2.py <root> <release id> <out.js> [--title ...] [--notes ...] [--sizes catalog_mu.json]
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
from typing import Any, Mapping
import numpy as np

# ---- registries -------------------------------------------------------------------------------------------------
# key, label, param, color, group, provenance, unit-file key, selection rule (how the method picks the
# one answer it submits; that is the method's own business, and the page says whose rule it is)
METHODS = [
    # a learned baseline carries its size in its name, like the Flash-ANSR series: parameters counted from the released
    # checkpoints (E2E model1.pt: embedder 6.2 M + encoder 12.6 M + decoder 74.6 M = 93.5 M; NeSymReS 100M.ckpt: 26.4 M --
    # the "100M" of that file name is the 100 million equations it was trained on, not its size)
    ("e2e", "E2E 93M", "candidates per bag", "#2f6fd0", "baseline", "upstream_default", "e2e",
     "A neural network generates candidate formulas from the data. E2E fits the numbers in each with the BFGS optimizer "
     "and returns the one with the smallest error on the 512 given points."),
    ("nesymres-100M", "NeSymReS 26M", "beam width", "#e8842a", "baseline", "upstream_default", "nesymres",
     "A neural network generates candidate formulas from the data with beam search (it keeps the most probable partial formulas at every step). "
     "NeSymReS fits the numbers in each with the BFGS optimizer and returns the one that fits the 512 given points best."),
    ("PySR", "PySR", "iterations", "#d62728", "baseline", "upstream_default", "pysr",
     "PySR evolves a population of formulas, as a genetic algorithm does. It keeps the best formula of every length found "
     "so far and returns the one its own rule picks from these, a rule that weighs error against length."),
    ("T8-3M", "Flash-ANSR T8-3M", "draws", "#8fcf8a", "flash-ansr", "author_blessed", None, None),
    ("T8-20M", "Flash-ANSR T8-20M", "draws", "#3e9b4a", "flash-ansr", "author_blessed", None, None),
    ("T8-120M", "Flash-ANSR T8-120M", "draws", "#1b5e20", "flash-ansr", "author_blessed", None, None),
    # the hybrid: a rung is a pair (D draws, I iterations) chosen so that both halves take the same time on the reference
    # machine; the ladder is labelled by its draws
    ("T8-20M-pysr", "Flash-ANSR T8-20M + PySR", "draws", "#7b1fa2", "hybrid", "author_blessed", "hybrid",
     "Flash-ANSR T8-20M generates candidate formulas, and up to 100 of those with the best Flash-ANSR score become PySR's starting population. "
     "PySR's best formulas then join Flash-ANSR's candidates, and Flash-ANSR's rule picks one. At a budget of B, Flash-ANSR generates B "
     "candidates and PySR runs as many iterations as take the same time on our timing workstation, so each budget costs about twice "
     "what Flash-ANSR alone spends at it."),
    ("prior", "Flash-ANSR prior", "draws", "#9a9a9a", "reference", "author_blessed", None,
     "Draws random formulas of the kind Flash-ANSR was trained on, without looking at the data, then fits their constants and "
     "picks one with Flash-ANSR's rule. It shows what guessing plus fitting and selection achieves, and so how much Flash-ANSR "
     "gains by reading the data."),
    # the ceiling: the ground truth itself as the one candidate, fitted by Flash-ANSR's refiner; its rungs are restarts
    ("oracle", "Oracle", "restarts", "#000000", "reference", "author_blessed", "oracle",
     "Is given the true formula with its constants blanked out (exponents are kept) and only has to fit the constants, "
     "the way Flash-ANSR does. It shows the best result that fitting alone can reach. Its budget is the number of fitting "
     "attempts, each from new random starting values.")]
# How a method is drawn when its colour alone is not the point: the oracle is the ceiling, a dashed line in the ink colour
# of the page (black, or white in the dark theme), like the ground truth's own reference line.
METHOD_STYLE: dict[str, dict[str, bool]] = {"oracle": {"dash": True, "ink": True}}
# Where two methods share a component at different versions, the release says so (Protocol, "Versions").
RELEASE_VERSIONS = ("PySR: version 2.3.0, with SymbolicRegression.jl 2.4.0. The PySR part of Flash-ANSR T8-20M + PySR: PySR 2.4.0, "
                    "with SymbolicRegression.jl 2.4.1 (2.4.2 on our timing workstation). The settings it uses have the same defaults "
                    "in both PySR versions, and the SymbolicRegression.jl versions in between change speed, not results. "
                    "Simplification: SimpliPy (a formula-simplification library), with its rule set acj-5-4-llm.")
FLASH_ANSR_SELECTION = ("A neural network generates candidate formulas from the data. Flash-ANSR fits the numbers in each and "
                        "returns the one that best balances error and length: the smallest (n/2) log2 FVU plus the formula's length "
                        "in bits, where n is the number of given points and FVU is the share of their variation the formula leaves "
                        "unexplained.")
# How each budget unit reads next to the method's name.
PARAM_LABEL = {"candidates per bag": "budget: candidate formulas", "beam width": "budget: beam width",
               "iterations": "budget: search iterations", "draws": "budget: candidate formulas",
               "restarts": "budget: fitting attempts"}
RUNGS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 65536]
E2E_DEFAULT_MAX_RUNG = 256   # E2E is reported at its default settings only
CATALOG_GROUPS = {
    "physics": ["fastsrb", "feynman", "feynman-bonus", "srsd-dummy", "erbench-phybench", "erbench-densities", "physo-astro", "physo-class"],
    "classical": ["nguyen", "keijzer", "korns", "koza", "livermore", "livermore2", "vladislavleva", "jin", "neat", "pagie", "poly", "nonic", "sine", "meier", "r-rationals", "constant", "grammarvae"],
    "synthetic": ["erbench-syneq", "soose-fc", "soose-nc", "soose-wc"]}
NB = 128

# key, label, short, group, kind, higher_is_better (None = 1 is ideal / descriptive), tier, format, histogram (lo, hi, transform), description
# Rate metrics are defined for EVERY problem (a failed prediction is a miss). So are the continuous metrics whose range
# has a worst value (WORST below, the overlaps): a failed prediction takes it. Every other continuous metric
# describes the answers that were made -- R^2, a length, a ratio of lengths can be arbitrarily bad --; the page
# marks a point as hollow when fewer problems than the reader's threshold have a value. Ground-truth descriptors cover
# every problem.
# NO WALL-CLOCK METRIC IS PUBLISHED. Seconds measured where a unit happened to run depend on the node, its GPU
# and whatever shared it, so they are not comparable between methods. The only time this benchmark publishes is
# the reference-machine ladder in timing.json, which the site uses for the time AXIS and nothing else.
METRICS = [
    # ---- was the problem solved? rates over every problem ----
    ("numeric_recovery_val", "Numeric Recovery, Validation", "vNRR", "Numeric Recovery", "rate", True, "main", "pct", None,
     "The share of problems where the predicted formula reproduces the 512 held-out points, which the method never saw, almost exactly: its FVU on those points is at most 2^-23 (about 1.2e-7, the precision of a 32-bit float), where FVU, the fraction of variance unexplained, is the mean squared error divided by the variance of the true values. So its typical error is at most 0.035 % of the true values' standard deviation. Short name: vNRR (v for validation, the held-out points)."),
    ("numeric_recovery_fit", "Numeric Recovery, Support", "fNRR", "Numeric Recovery", "rate", True, "more", "pct", None,
     "The same test on the 512 points the method was given (the support points, which it fits). When fNRR is higher than vNRR, some formulas fit the given points but not the held-out ones. Short name: fNRR (f for fit)."),
    ("numeric_recovery_relative_val", "Fits as Well as the Ground Truth, Validation", "GT-Level vNRR", "Numeric Recovery", "rate", True, "more", "pct", None,
     "The share of problems where the predicted formula fits the held-out points at least as well as the true formula does: its FVU is at most the true formula's, and the bar is never stricter than Numeric Recovery's. It differs from Numeric Recovery only where the data are measurements that the true formula does not reproduce exactly."),
    ("numeric_recovery_relative_fit", "Fits as Well as the Ground Truth, Support", "GT-Level fNRR", "Numeric Recovery", "rate", True, "more", "pct", None,
     "The same test on the 512 given points."),
    ("success", "Successful Prediction Rate", "Success", "Numeric Recovery", "rate", True, "more", "pct", None,
     "The share of problems where the method returned a formula that could be evaluated at all: it was produced and parsed, and its numbers were fitted, without an error."),
    ("symbolic_recovery", "Symbolic Recovery: Structure", "SRRs", "Symbolic Recovery", "rate", True, "main", "pct", None,
     "The share of problems where the predicted formula has the same form as the true formula once every number in both is ignored, exponents included: 2.1 sin(x) matches 5 sin(x), and x^2 also matches x^3. Both formulas are first simplified into a standard form (with SimpliPy), so x + y also matches y + x. Short name: SRRs (s for structure)."),
    ("symbolic_recovery_mask_fittable", "Symbolic Recovery: Structure + Exponents", "SRRe", "Symbolic Recovery", "rate", True, "more", "pct", None,
     "Like SRRs, but only the constants a method fits (coefficients, added terms, constants inside functions) are ignored. Exponents and root indices must match exactly: x^2 and x^3 differ, and an exponent of 1.9999 where the true formula has 2 is a miss. Short name: SRRe (e for exponents)."),
    ("symbolic_recovery_mask_none", "Symbolic Recovery: Structure + All Numbers", "SRRa", "Symbolic Recovery", "rate", True, "more", "pct", None,
     "An SRRe match whose formula, with its fitted numbers, also passes Numeric Recovery on the held-out points: the formula itself was found. Short name: SRRa (a for all numbers)."),
    ("skeleton_match_raw", "Raw Symbolic Recovery", "SRRr", "Symbolic Recovery", "rate", True, "more", "pct", None,
     "Like SRRs, but without simplifying first: the predicted formula must be written exactly like the true one, symbol by symbol, with every number ignored. x*x instead of x^2, or a sum in a different order, is a miss. It is the strictest about how a formula is written, and like SRRs it ignores all numbers. Short name: SRRr (r for raw)."),
    # ---- how good is the fit? the predictions that were made ----
    ("log10_fvu_val", "log10 FVU, Validation", "log10 FVU Val", "Fit Error", "cont", False, "main", "num2", (-17.0, 3.0, None),
     "How much of the variation in the 512 held-out points the formula leaves unexplained (FVU: the mean squared error divided by the variance of the true values), on a log10 scale. 0 is no better than always predicting the average, -2 leaves 1 % unexplained, -7 leaves 0.00001 %. The scale stops at -15.65, the precision of a 64-bit float: an exact fit counts there. A formula that blows up has no finite value: the median counts it as worst, the mean leaves it out."),
    ("log10_fvu_fit", "log10 FVU, Support", "log10 FVU Fit", "Fit Error", "cont", False, "more", "num2", (-17.0, 3.0, None),
     "The same on the 512 given points."),
    ("r2_val", "R², Validation", "R² Val", "Fit Error", "cont", True, "more", "num3", (-1.0, 1.0, None),
     "1 - FVU on the held-out points: 1 is a perfect fit, 0 is no better than always predicting the average, and there is no lower limit. A single formula that blows up could decide an average, so the median is shown whichever statistic you choose."),
    ("r2_fit", "R², Support", "R² Fit", "Fit Error", "cont", True, "more", "num3", (-1.0, 1.0, None),
     "The same on the 512 given points. The median is shown whichever statistic you choose."),
    # ---- comparisons of the prediction with the ground truth ----
    ("mdl_ratio", "MDL Ratio", "MDL Ratio", "Size Compared to the Ground Truth", "cont", None, "main", "ratio", (-4.0, 4.0, "log2"),
     "The length of the predicted formula in bits divided by the length of the true formula. Length in bits (MDL, minimum description length) is how many bits it takes to write the formula down, including the digits of its numbers, measured by SimpliPy after simplification. 1 means as long as the true formula, 2 twice as long. Means are taken on a log scale, so ×2 and ×0.5 cancel. Flash-ANSR picks its prediction with this same length measure."),
    ("expr_length_ratio", "Token Count Ratio", "Token Ratio", "Size Compared to the Ground Truth", "cont", None, "main", "ratio", (-4.0, 4.0, "log2"),
     "The number of symbols in the predicted formula divided by the number in the simplified true formula; every operator, variable and number counts as one symbol. 1 means the same length. Means are taken on a log scale, so ×2 and ×0.5 cancel."),
    ("expr_length_ratio_abserr", "Token Count Mismatch", "|log2 Ratio|", "Size Compared to the Ground Truth", "cont", False, "more", "num2", (0.0, 4.0, None),
     "How far the Token Count Ratio is from 1, in doublings: 0 when both formulas have the same number of symbols, 1 when one has twice as many as the other."),
    ("n_constants_ratio", "Constant Count Ratio", "Constants Ratio", "Size Compared to the Ground Truth", "cont", None, "more", "ratio", (-4.0, 4.0, "log2"),
     "The number of constants (fitted numbers) in the predicted formula divided by the number in the true formula, for problems whose true formula has at least one constant. Means are taken on a log scale, so ×2 and ×0.5 cancel."),
    ("n_constants_delta", "Constant Count Difference", "Constants Diff.", "Size Compared to the Ground Truth", "cont", None, "more", "num1", (-16.0, 16.0, None),
     "The number of constants in the predicted formula minus the number in the true formula."),
    ("total_nestedness_delta", "Function Nesting Difference", "Nesting Diff.", "Size Compared to the Ground Truth", "cont", None, "more", "num1", (-8.0, 8.0, None),
     "The function nesting of the predicted formula minus that of the true formula (see Function Nesting of the Prediction)."),
    ("f1_score", "Token Overlap, F1", "Token F1", "Similarity to the Ground Truth", "cont", True, "more", "num3", (0.0, 1.0, None),
     "How well the symbols of the two formulas overlap, ignoring order and repeats, with every number counted as the same placeholder symbol. F1 combines Precision and Recall below (their harmonic mean); 1 means both formulas use exactly the same symbols."),
    ("precision_score", "Token Overlap, Precision", "Token Precision", "Similarity to the Ground Truth", "cont", True, "more", "num3", (0.0, 1.0, None),
     "The share of the distinct symbols in the predicted formula that also occur in the simplified true formula."),
    ("recall_score", "Token Overlap, Recall", "Token Recall", "Similarity to the Ground Truth", "cont", True, "more", "num3", (0.0, 1.0, None),
     "The share of the distinct symbols in the simplified true formula that also occur in the predicted formula."),
    ("f1_score_unique_variables", "Variable Overlap, F1", "Variables F1", "Similarity to the Ground Truth", "cont", True, "more", "num3", (0.0, 1.0, None),
     "The same as Token Overlap, F1, for the input variables alone: 1 means both formulas use exactly the same variables."),
    ("precision_unique_variables", "Variable Overlap, Precision", "Variables Prec.", "Similarity to the Ground Truth", "cont", True, "more", "num3", (0.0, 1.0, None),
     "The share of the variables in the predicted formula that the true formula also uses."),
    ("recall_unique_variables", "Variable Overlap, Recall", "Variables Recall", "Similarity to the Ground Truth", "cont", True, "more", "num3", (0.0, 1.0, None),
     "The share of the variables in the true formula that the predicted formula also uses."),
    ("edit_distance", "Levenshtein Edit Distance", "Levenshtein", "Similarity to the Ground Truth", "cont", False, "more", "num1", (0.0, 64.0, None),
     "How many symbols must be inserted, deleted or replaced to turn the predicted formula into the simplified true formula (Levenshtein distance). Both are written as sequences of symbols with each operator before its arguments, and every number as the same placeholder."),
    ("edit_distance_norm", "Levenshtein Edit Distance, Normalized", "Levenshtein Norm.", "Similarity to the Ground Truth", "cont", False, "more", "num3", (0.0, 1.0, None),
     "The Levenshtein distance divided by the number of symbols in the longer formula: 0 means identical, 1 means every symbol differs."),
    ("zss_edit_distance", "Tree Edit Distance", "Tree Edit Dist.", "Similarity to the Ground Truth", "cont", False, "more", "num1", (0.0, 128.0, None),
     "The cost of turning the predicted formula's tree into the simplified true formula's tree by inserting, deleting or renaming nodes (Zhang-Shasha tree edit distance). In such a tree each operator is a node, and its arguments are its children. The cost follows the spelling of the node names: each step costs the number of characters it changes, so deleting sin costs 3 and renaming sin to tan costs 2."),
    # ---- properties of ONE expression, the prediction or the ground truth: no comparison ----
    ("predicted_mdl", "MDL of the Prediction", "Prediction MDL", "Expression Properties", "cont", False, "more", "num1", (0.0, 256.0, None),
     "The length of the predicted formula in bits: how many bits it takes to write the formula down (MDL, minimum description length), measured by SimpliPy after simplification."),
    ("ground_truth_mdl", "MDL of the Ground Truth", "GT MDL", "Expression Properties", "cont", None, "more", "num1", (0.0, 256.0, None),
     "The length of the true formula in bits, measured the same way."),
    ("predicted_skeleton_prefix_length", "Token Count of the Prediction", "Prediction Tokens", "Expression Properties", "cont", False, "more", "num1", (0.0, 64.0, None),
     "The number of symbols in the predicted formula; every operator, variable and number counts as one."),
    ("skeleton_length", "Token Count of the Ground Truth", "GT Tokens", "Expression Properties", "cont", None, "more", "num1", (0.0, 64.0, None),
     "The number of symbols in the simplified true formula."),
    ("predicted_n_constants", "Constant Count of the Prediction", "Prediction Constants", "Expression Properties", "cont", False, "more", "num1", (0.0, 32.0, None),
     "How many constants the method fitted in its formula."),
    ("n_constants", "Constant Count of the Ground Truth", "GT Constants", "Expression Properties", "cont", None, "more", "num1", (0.0, 32.0, None),
     "How many constants the true formula has."),
    ("predicted_total_nestedness", "Function Nesting of the Prediction", "Prediction Nesting", "Expression Properties", "cont", False, "more", "num1", (0.0, 16.0, None),
     "How deeply functions sit directly inside each other in the predicted formula: sin(x) counts 0, sin(log(x)) counts 1, sin(log(exp(x))) counts 2, and separate chains add up."),
    ("total_nestedness", "Function Nesting of the Ground Truth", "GT Nesting", "Expression Properties", "cont", None, "more", "num1", (0.0, 16.0, None),
     "The same for the true formula."),
    ("n_variables", "Variable Count of the Ground Truth", "GT Variables", "Expression Properties", "cont", None, "more", "num1", (0.0, 16.0, None),
     "How many input variables the true formula uses."),
    # ---- what the method reports about its own choice ----
    ("predicted_log_prob", "Log-Probability of the Prediction", "Log-Prob", "Method Internals", "cont", True, "more", "num1", (-64.0, 0.0, None),
     "How probable the chosen formula was under the Flash-ANSR model itself, given the data: the logarithm of the probability of its sequence of symbols. Longer formulas have lower values. Only the Flash-ANSR models report it."),
    ("predicted_score", "Selection Score", "Score", "Method Internals", "cont", False, "more", "num1", (-2048.0, 512.0, None),
     "The score by which Flash-ANSR chose its formula: the error-plus-length score described under its name, in bits; lower is better. Only the Flash-ANSR entries report it."),
    ("predicted_pareto_rank", "Pareto Rank of the Prediction", "Pareto Rank", "Method Internals", "cont", False, "more", "num1", (0.0, 64.0, None),
     "Where the chosen formula stands among Flash-ANSR's candidates when they are compared by error and length together. 0 means no other candidate is both more accurate and shorter. 1 means that holds once the candidates at 0 are set aside, and so on. Only the Flash-ANSR entries report it.")]
# A metric that repeats another in EVERY published cell says nothing of its own, so the menu does not list it (its
# numbers stay in the cells). Recovery relative to the ground truth is numeric recovery wherever the targets are
# computed from the ground truth (reference FVU = 0); it is a metric of its own only once a catalog of measured data is in.
# The continuous metrics whose range has a worst value, and that value: a failed prediction takes it, so these are
# read over every problem (the rows carry the value already: srbf.result_processing.WORST_VALUE, checked by the tests).
# The reader may leave the failed predictions out instead. Nothing is exported twice for that: a cell names how many
# of its values were filled in ("w"), and the page takes that many off the sums and out of the histogram bin of the
# worst value. Only a paired contrast cannot be undone that way, so it ships in both readings (key + ANSWERED).
WORST = {"f1_score": 0.0, "precision_score": 0.0, "recall_score": 0.0,
         "f1_score_unique_variables": 0.0, "precision_unique_variables": 0.0, "recall_unique_variables": 0.0}
# Metrics without a bound on the bad side AND with a heavy tail there: one answer decides a mean, so the page reads
# the median whatever the reader chose. R^2 = 1 - FVU is a monotone map of the FVU, so near 1, where its own linear
# histogram cannot tell 0.99 from 0.9999, and below the histogram's lower end, the page reads the order statistic from
# the log10 FVU histogram instead (bins of 0.16 decades).
MEDIAN_ONLY = {"r2_val": "log10_fvu_val", "r2_fit": "log10_fvu_fit"}
# A metric that is undefined for some problems whatever the method does: the problems it could be defined for are the base of
# its share of valid results. (The constant-count ratio needs a ground truth with at least one constant.)
ELIGIBLE = {"n_constants_ratio": lambda row: bool(row.get("n_constants"))}
# Properties of the ground truth alone: defined for every problem, whatever the method did.
EVERY_PROBLEM = {"ground_truth_mdl", "skeleton_length", "n_constants", "total_nestedness", "n_variables"}
ANSWERED = "@answered"   # suffix of a paired contrast taken over the problems BOTH methods have a prediction for
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
# The Ranks view: within every problem the methods are placed 1st, 2nd, ... on one continuous metric, a method without a
# usable prediction last. A mean rank over any set of problems and any roster of methods follows from PAIRWISE outcomes alone
# (rank_i = 1 + sum_j [j beats i] + 0.5 [j ties i]), and pairwise counts add up over catalogs, so that is what ships:
# per pair x catalog x slot, [n problems, then (wins of the first, wins of the second) per rank key]. A slot is a rung
# ("64": both methods at that rung) or a time budget ("t3": each method at its largest rung the reference machine
# timed at or under 3 s per problem). The first key is the primary league.
# Every metric that says how good a prediction is can rank: the rates (a hit beats a miss), and every continuous metric
# with a direction or an ideal. Not ranked: R^2, which orders the predictions exactly as the FVU does; the properties of
# the ground truth, the same for every method; and the scores a method gives its own predictions (log-probability,
# selection score, Pareto rank), which mean something else for every method.
RANK_KEYS = ["log10_fvu_val",
             "numeric_recovery_val", "numeric_recovery_fit", "numeric_recovery_relative_val", "numeric_recovery_relative_fit", "success",
             "symbolic_recovery", "symbolic_recovery_mask_fittable", "symbolic_recovery_mask_none", "skeleton_match_raw",
             "log10_fvu_fit",
             "mdl_ratio", "expr_length_ratio", "expr_length_ratio_abserr", "n_constants_ratio", "n_constants_delta", "total_nestedness_delta",
             "f1_score", "precision_score", "recall_score", "f1_score_unique_variables", "precision_unique_variables", "recall_unique_variables",
             "edit_distance", "edit_distance_norm", "zss_edit_distance",
             "predicted_mdl", "predicted_skeleton_prefix_length", "predicted_n_constants", "predicted_total_nestedness"]
# The value a comparison with the ground truth ideally takes, for the metrics without a better direction: a ratio is best
# at 1, a difference at 0.
IDEAL = {"mdl_ratio": 1.0, "expr_length_ratio": 1.0, "n_constants_ratio": 1.0, "n_constants_delta": 0.0, "total_nestedness_delta": 0.0}
TIME_BUDGETS = [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]


def registry_json() -> list[dict[str, Any]]:
    out = []
    for k, label, short, group, kind, higher, tier, fmt, hist, desc in METRICS:
        m: dict[str, Any] = {"key": k, "label": label, "short": short, "group": group, "kind": kind, "higher": higher, "tier": tier, "fmt": fmt, "desc": desc}
        if hist:
            m["hist"] = {"lo": hist[0], "hi": hist[1], "tf": hist[2]}
        if k in WORST:
            m["worst"] = WORST[k]
        if k in IDEAL:
            m["ideal"] = IDEAL[k]
        if k in MEDIAN_ONLY:
            m["median_via"] = MEDIAN_ONLY[k]
        if k in EVERY_PROBLEM:
            m["every"] = True
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


Rows = dict[tuple[int, int], dict[str, Any]]   # (draw, row) -> {metric: value}


def load_rows(root: str) -> dict[str, dict[tuple[str, int], Rows]]:
    """{method: {(catalog, rung): {(draw, row): {metric: value}}}}: every draw the rows carry (a model is run twice
    with different seeds; which draws a cell pools is decided per cell, see pooled_rows)."""
    data: dict[str, dict[tuple[str, int], Rows]] = defaultdict(lambda: defaultdict(dict))
    files = sorted(set(glob.glob(os.path.join(root, "rows_full_*.csv")) + glob.glob(os.path.join(root, "*_rows_full.csv"))))
    for path in files:
        with open(path) as fh:
            for r in csv.DictReader(fh):
                vals: dict[str, float | None] = {}
                for k in RATE_KEYS:
                    v = fnum(r.get(k))
                    vals[k] = (0.0 if v is None else v) if k in r else None   # a column these rows do not have is no rate of 0
                for k in CONT_KEYS:
                    vals[k] = fnum(r.get(k))
                data[r["model"]][(r["catalog"], int(r["rung"]))][(int(r.get("draw") or 1), int(r["row"]))] = vals
    return data


# ---- the formulas themselves: the Predictions view ----------------------------------------------------------------
# One file per method x problem set x budget x run x block of PRED_BLOCK problems, written only once that run of the
# cell is complete: a finished file never changes, so the repository grows by each formula once. The ground truth has
# its own files per problem set and block.
PRED_BLOCK = 500
PRED_NUMERIC, PRED_SYMBOLIC = 1, 2   # the flags stored with a prediction


def round_prefix(expr: str) -> str:
    """A prefix expression with every number at 4 significant digits (integers as written): what the page shows."""
    out = []
    for t in expr.split():
        if t.lstrip("-").isdigit():
            out.append(t)
            continue
        try:
            v = float(t)
        except ValueError:
            out.append(t)
            continue
        out.append(f"{v:.4g}" if math.isfinite(v) else t)
    return " ".join(out)


Expressions = dict[tuple[str, str, int, int], dict[int, list[Any] | None]]


def load_expressions(root: str, keys: list[str]) -> tuple[Expressions, dict[str, dict[int, str]]]:
    """(pred, truth): pred[(method, catalog, rung, draw)][row] = [expression, flags] for the methods in `keys`, and
    truth[catalog][row] = the ground truth's expression. Both are empty when the rows carry no expressions (a table
    written before `srbf table` stored them)."""
    pred: Expressions = defaultdict(dict)
    truth: dict[str, dict[int, str]] = defaultdict(dict)
    want = set(keys)
    files = sorted(set(glob.glob(os.path.join(root, "rows_full_*.csv")) + glob.glob(os.path.join(root, "*_rows_full.csv"))))
    for path in files:
        with open(path) as fh:
            rd = csv.reader(fh)
            head = next(rd, [])
            if "predicted_expression" not in head:
                continue
            names = ("model", "draw", "catalog", "rung", "row", "success", "numeric_recovery_val", "symbolic_recovery",
                     "predicted_expression", "ground_truth_expression")
            at = {k: head.index(k) for k in names}
            for r in rd:
                cat, row = r[at["catalog"]], int(r[at["row"]])
                gt = r[at["ground_truth_expression"]]
                if gt and row not in truth[cat]:
                    truth[cat][row] = round_prefix(gt)
                if r[at["model"]] not in want:
                    continue
                expr = r[at["predicted_expression"]] if r[at["success"]] in ("1", "1.0") else ""
                numeric, symbolic = r[at["numeric_recovery_val"]] in ("1", "1.0"), r[at["symbolic_recovery"]] in ("1", "1.0")
                flags = (PRED_NUMERIC if numeric else 0) | (PRED_SYMBOLIC if symbolic else 0)
                key = (r[at["model"]], cat, int(r[at["rung"]]), int(r[at["draw"]] or 1))
                pred[key][row] = [round_prefix(expr), flags] if expr else None
    return pred, truth


def write_predictions(out_dir: str, rel: str, pred: Expressions, truth: dict[str, dict[int, str]],
                      sizes: dict[str, int]) -> dict[str, dict[str, list[int]]]:
    """Write the Predictions view's files next to the release; returns the index the page reads:
    {method: {"catalog|rung": [the runs whose files exist]}}. A run still in progress is left out."""
    def put(path: str, key: str, obj: Any) -> None:
        text = "window.RESULTS_V2_PRED=window.RESULTS_V2_PRED||{};(function(){var R=window.RESULTS_V2_PRED;R[%s]=R[%s]||{};R[%s][%s]=%s;})();\n" % (
            json.dumps(rel), json.dumps(rel), json.dumps(rel), json.dumps(key), json.dumps(obj, separators=(",", ":")))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path) and open(path).read() == text:
            return
        with open(path, "w") as fh:
            fh.write(text)

    def blocks(rows: Mapping[int, Any]) -> dict[int, dict[str, Any]]:
        out: dict[int, dict[str, Any]] = defaultdict(dict)
        for i in sorted(rows):
            out[i // PRED_BLOCK][str(i)] = rows[i]
        return out

    index: dict[str, dict[str, list[int]]] = defaultdict(dict)
    for (m, c, r, d), got in sorted(pred.items()):
        if c not in sizes or len(got) < sizes[c]:
            continue
        index[m].setdefault(f"{c}|{r}", []).append(d)
        for b, chunk in blocks(got).items():
            put(os.path.join(out_dir, "pred", m, c, f"{r}.{d}.{b}.js"), f"{m}|{c}|{r}|{d}|{b}", chunk)
    for c, formulas in sorted(truth.items()):
        if c in sizes:
            for b, chunk in blocks(formulas).items():
                put(os.path.join(out_dir, "pred", "truth", f"{c}.{b}.js"), f"truth|{c}|{b}", chunk)
    return dict(index)


def by_draw(rows: dict[Any, dict[str, Any]]) -> dict[int, dict[int, dict[str, Any]]]:
    """Rows keyed by (draw, row) split per draw; a bare row key counts as draw 1."""
    out: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    for k, v in rows.items():
        d, i = k if isinstance(k, tuple) else (1, k)
        out[d][i] = v
    return out


def complete_draws(rows: dict[Any, dict[str, Any]], expected: int | None) -> list[int]:
    """The draws that cover every problem of the cell (all of them when no count is expected)."""
    return sorted(d for d, rs in by_draw(rows).items() if expected is None or len(rs) >= expected)


def pooled_rows(rows: dict[Any, dict[str, Any]], expected: int | None) -> tuple[Rows, list[int]]:
    """What a cell pools: every complete draw, and only those (a draw still running is not a random subset of the
    problems). With no complete draw the fullest draw stands in (the first draw on a tie), and the cell is partial.
    The rows come back in (draw, problem) order, so the cell's sums do not depend on the order the table was read in."""
    per = by_draw(rows)
    done = complete_draws(rows, expected)
    if not done:
        fullest = max(sorted(per), key=lambda k: len(per[k])) if per else 1
        return {(fullest, i): per[fullest][i] for i in sorted(per.get(fullest, {}))}, []
    return {(d, i): per[d][i] for d in done for i in sorted(per[d])}, done


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


def summarize_cell(rows: dict[Any, dict[str, Any]], expected: int | None) -> dict[str, Any]:
    pool, done = pooled_rows(rows, expected)
    vals = list(pool.values())
    cell: dict[str, Any] = {
        "state": "complete" if done else "partial", "d": max(1, len(done)),   # d: how many draws the cell pools
        "n": len(vals), "ok": int(sum(1 for x in vals if x["success"])), "m": {}}
    for k in RATE_KEYS:
        if any(x[k] is not None for x in vals):
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


def paired_cell(rows_a: dict[Any, dict[str, Any]], rows_b: dict[Any, dict[str, Any]], expected: int | None = None) -> dict[str, Any] | None:
    """A problem is paired with itself within a draw, over the draws complete on both sides."""
    rows_a, rows_b = pooled_rows(rows_a, expected)[0], pooled_rows(rows_b, expected)[0]
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
                # d is the first method's value minus the second's; the wins and losses count where the first is BETTER
                # or WORSE on the metric's own terms (rank_score: lower is better for an error, closer to the ideal
                # for a ratio), which the sign of d alone does not say
                ds, better, worse = [], 0, 0
                higher, ideal = METRIC_HIGHER[k], IDEAL.get(k)
                for i in laws:
                    va, vb = transform(rows_a[i][k], tf), transform(rows_b[i][k], tf)
                    if va is None or vb is None or not (math.isfinite(va) and math.isfinite(vb)):
                        continue
                    ds.append(va - vb)
                    sa, sb = rank_score(rows_a[i][k], higher, ideal), rank_score(rows_b[i][k], higher, ideal)
                    better, worse = better + (sa > sb), worse + (sb > sa)
                d = np.asarray(ds, float)
                out[name] = [int(d.size), float(d.sum()) if d.size else 0.0, float((d * d).sum()) if d.size else 0.0, int(better), int(worse)]
    return {"n": len(common), "m": out}


def budget_key(t: float) -> str:
    return "t" + ("%g" % t)


# ranks.js: the release's pairwise outcomes, merged with a key-opened overlay's in whichever order the two load. The
# release's file sets the key list; a file written with another list (an overlay sealed before RANK_KEYS changed) is
# mapped onto it, whichever loaded first, and a key it lacks is left empty, which the page reads as no outcome.
RANKS_JS = ("window.RESULTS_V2_RANKS=window.RESULTS_V2_RANKS||{};(function(){var R=window.RESULTS_V2_RANKS,rel=%s,K=%s,MAIN=%s;"
            "var X=R[rel]=R[rel]||{keys:K,budgets:%s,seconds:%s,at:{},pairs:{}};X.rungs=X.rungs||{};var P=%s;"
            "var map=function(Q,from,to){var ix=to.map(function(k){return from.indexOf(k);});Object.keys(Q).forEach(function(pk){"
            "Object.keys(Q[pk]).forEach(function(c){Object.keys(Q[pk][c]).forEach(function(sl){var t=Q[pk][c][sl],u=[t[0]];"
            "ix.forEach(function(j){u.push(j<0?null:t[1+2*j],j<0?null:t[2+2*j]);});Q[pk][c][sl]=u;});});});};"
            "if(X.keys.join()!==K.join()){if(MAIN){map(X.pairs,X.keys,K);X.keys=K;}else{map(P,K,X.keys);}}"
            "Object.assign(X.at,%s);Object.assign(X.pairs,P);Object.assign(X.rungs,%s);})();\n")


def rank_score(value: float | None, higher: bool | None, ideal: float | None = None) -> float:
    """Oriented so that larger is better; a problem without a usable value scores -inf (placed last). `higher` None
    marks a comparison with an ideal value (IDEAL): a ratio is better the closer it is to 1 on the log scale, a
    difference the closer it is to 0."""
    if value is None or math.isnan(value):
        return -math.inf
    if higher is True:
        return value
    if higher is False:
        return -value
    if ideal == 0.0:
        return -abs(value) if math.isfinite(value) else -math.inf
    return -abs(math.log(value)) if value > 0 and math.isfinite(value) else -math.inf


def rank_pair_cell(rows_a: dict[Any, dict[str, Any]], rows_b: dict[Any, dict[str, Any]], expected: int | None = None,
                   keys: list[str] | None = None) -> list[int] | None:
    rows_a, rows_b = pooled_rows(rows_a, expected)[0], pooled_rows(rows_b, expected)[0]
    common = sorted(set(rows_a) & set(rows_b))
    if not common:
        return None
    out = [len(common)]
    for k in RANK_KEYS if keys is None else keys:
        higher, ideal = METRIC_HIGHER[k], IDEAL.get(k)
        wa = wb = 0
        for i in common:
            sa, sb = rank_score(rows_a[i].get(k), higher, ideal), rank_score(rows_b[i].get(k), higher, ideal)
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
def planned_cells(root: str, ukey: str | None, publishes: Any) -> set[tuple[int, str, int]] | None:
    """Every (draw, catalog, rung) a method's run plan holds: the unit lists in the root (`units_<ukey>_d<draw>.txt`, or
    the Flash-ANSR series' shared `t8s1_units_draw<draw>.txt`; lines "draw catalog rung [shard]", the shards of a split
    catalog are one cell), over the rungs the release publishes (`publishes(rung)`). None without a plan."""
    plan: set[tuple[int, str, int]] = set()
    for draw in (1, 2):
        path = os.path.join(root, f"units_{ukey}_d{draw}.txt" if ukey else f"t8s1_units_draw{draw}.txt")
        if not os.path.exists(path):
            continue
        for line in open(path):
            p = line.split()
            if len(p) >= 3 and p[2].isdigit() and publishes(int(p[2])):
                plan.add((draw, p[1], int(p[2])))
    return plan or None


def status_of(rows_by_cell: dict[tuple[str, int], Any], sizes: dict[str, int], plan: set[tuple[int, str, int]] | None) -> list[int | None]:
    """[finished, planned]: a catalog at one rung in one draw is finished once its rows cover the catalog, the same
    test that puts a rung in the plots; counted against the plan when there is one."""
    done = {(d, c, r) for (c, r), rows in rows_by_cell.items() for d, rs in by_draw(rows).items()
            if c in sizes and len(rs) >= sizes[c]}
    if plan is not None:
        done &= plan
    return [len(done), len(plan) if plan is not None else None]


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
    ap.add_argument("--sizes", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results-site", "data", "catalog_mu.json"),
                    help="every catalog's ground-truth description lengths (scripts/catalog_mu.py; default: results-site/data/catalog_mu.json)")
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
            present[c] = max([present[c]] + [len(rs) for rs in by_draw(rows).values()])
    cats = catalog_meta(a.sizes, present)
    sizes = {c["key"]: c["laws"] for c in cats}

    def usable(key: str, r: int) -> bool:
        return r in RUNGS and not (key == "e2e" and r > E2E_DEFAULT_MAX_RUNG)

    def contrasts(pairs: list[tuple[str, str]], paired: dict[str, Any]) -> None:
        for ka, kb in pairs:
            for (c, r), rows_a in sorted(data.get(ka, {}).items(), key=lambda kv: kv[0]):   # a fixed order, whatever the table's
                rows_b = data.get(kb, {}).get((c, r))
                if not rows_b or not usable(ka, r) or not usable(kb, r):
                    continue
                pc = paired_cell(rows_a, rows_b, sizes.get(c))
                if pc:
                    paired.setdefault(ka + "|" + kb, {}).setdefault(c, {})[str(r)] = pc

    tpath0 = os.path.join(a.root, "timing.json")
    timing_all: dict[str, Any] = {}
    if os.path.exists(tpath0):
        timing_all = {k: v for k, v in json.load(open(tpath0)).items() if isinstance(v, dict) and not k.startswith("__")}

    def leagues(pairs: list[tuple[str, str]], keys: list[str], rank_keys: list[str]) -> dict[str, Any]:
        """Pairwise rank outcomes for `pairs`, and for every method of `keys` the rung a time budget buys it."""
        rungs_of = {k: {r for (_c, r) in data.get(k, {}) if usable(k, r)} for k in {x for p in pairs for x in p} | set(keys)}
        # a time budget buys a rung the method has FINISHED: every catalog, every problem (the site shows no pooled number
        # for a rung that is still running, so a budget must not point at one)
        finished = {k: {r for r in rungs_of[k] if all(complete_draws(data[k].get((c, r), {}), n) for c, n in sizes.items())} for k in rungs_of}
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
                    pc = rank_pair_cell(rows_a, rows_b, sizes.get(c), rank_keys)
                    if pc:
                        out.setdefault(ka + "|" + kb, {}).setdefault(c, {})[slot] = pc
        return {"keys": rank_keys, "budgets": [budget_key(t) for t in TIME_BUDGETS], "seconds": TIME_BUDGETS,
                "at": {k: at[k] for k in keys}, "pairs": out, "rungs": {k: v for k, v in compared.items() if k in out and v}}

    def build(keys: list[str], base: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        methods = [m for m in METHODS if m[0] in keys]
        cells: dict[str, Any] = {}
        hists: dict[str, Any] = {k: {} for k in HIST_SPECS}
        status: dict[str, Any] = {}
        plans: dict[str, set[tuple[int, str, int]] | None] = {}
        for key, label, param, color, group, prov, ukey, _sel in methods:
            cells[key] = {}
            for (c, r), rows in sorted(data.get(key, {}).items()):
                if not usable(key, r):
                    continue
                cells[key].setdefault(c, {})[str(r)] = summarize_cell(rows, sizes.get(c))
                pool = pooled_rows(rows, sizes.get(c))[0]
                for hk, (lo, hi, tf) in HIST_SPECS.items():
                    h = hist_of([transform(x[hk], tf) for x in pool.values()], lo, hi)
                    if h is not None:
                        hists[hk].setdefault(key, {}).setdefault(c, {})[str(r)] = h
            plans[key] = planned_cells(a.root, ukey, lambda r, k=key: usable(k, r))
            status[key] = status_of({cr: rows for cr, rows in data.get(key, {}).items() if usable(key, cr[1])}, sizes, plans[key])
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
        listed = listed_metrics(cells)
        payload = {"schema": 2, "base": base,
                   "release": {"id": a.release, "title": a.title or a.release, "notes": a.notes, "versions": RELEASE_VERSIONS, "generated": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
                               "updated": dt.datetime.now().astimezone().isoformat(timespec="minutes"),   # with its offset: shown in the reader's time zone
                               "scoring": "Every method returns one formula per problem, its prediction, and picks it by its own rule. The ? after a method's name describes that rule; the label beside it says who chose the method's settings.",
                               "judge": "Every prediction is checked the same way against the true formula. Numeric Recovery: it reproduces the 512 held-out points almost exactly (FVU at most 2^-23: a typical error of at most 0.035 % of the true values' spread). Symbolic Recovery: once both formulas are simplified into a standard form, they are identical when their numbers are ignored (so x^2 matches x^3; stricter versions also check exponents and all numbers)."},
                   "catalogs": cats, "rungs": RUNGS, "nb": NB, "metrics": listed, "paired_keys": PAIRED_KEYS, "rank_keys": [k for k in RANK_KEYS if k in {m["key"] for m in listed}],
                   # budget: what one rung of the ladder buys. "candidates" is a count a generative method draws;
                   # PySR's rungs are search iterations, which have no place on the candidate axis of the site.
                   "methods": [{"key": k, "label": l, "param": PARAM_LABEL.get(p, p), "budget": p if p in ("iterations", "seconds", "restarts") else "candidates",
                                "color": col, "group": g, "provenance": prov,
                                "selection": sel or (FLASH_ANSR_SELECTION if g == "flash-ansr" else ""),
                                # the budgets its run plan holds (None without a plan): a budget outside them is never run
                                "budgets": sorted({r for _, _, r in plan}) if (plan := plans.get(k)) else None, **METHOD_STYLE.get(k, {})}
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
            fh.write(RANKS_JS % (rel, json.dumps(ranks["keys"]), "true" if var == "RESULTS_V2" else "false", json.dumps(ranks["budgets"]), json.dumps(ranks["seconds"]), json.dumps(ranks["pairs"], separators=(",", ":")),
                                 json.dumps(ranks["at"], separators=(",", ":")), json.dumps(ranks.get("rungs", {}), separators=(",", ":"))))
        with_data = [k for k in payload["cells"] if payload["cells"][k]]
        print(f"{note}: {out_js} ({os.path.getsize(out_js) // 1024} kB), hist/ {len(hists)} files, paired {len(paired)} pairs; methods with data: {with_data}; status {payload['status']}")

    payload, hists, paired = build(public, rel_base(out_dir, site_dir))
    pred, truth = load_expressions(a.root, public)
    pred = {k: v for k, v in pred.items() if usable(k[0], k[2])}   # the budgets the release publishes, and no others
    payload["pred"] = write_predictions(out_dir, a.release, pred, truth, sizes)
    payload["pred_block"] = PRED_BLOCK
    ranks = leagues([(ka, kb) for i, ka in enumerate(public) for kb in public[i + 1:]], public, payload["rank_keys"])
    write_set(payload, hists, paired, ranks, a.out, out_dir, "RESULTS_V2", f"public release {a.release}")
    if private:
        pdir = os.path.abspath(a.private_dir)
        os.makedirs(pdir, exist_ok=True)
        ppayload, phists, ppaired = build(private, rel_base(pdir, site_dir))
        contrasts([(ka, kb) for ka in private for kb in public], ppaired)   # private-vs-public contrasts stay private
        pranks = leagues([(ka, kb) for i, ka in enumerate(private) for kb in private[i + 1:]] + [(ka, kb) for ka in private for kb in public], private, payload["rank_keys"])   # the release's keys
        write_set(ppayload, phists, ppaired, pranks, os.path.join(pdir, "results_v2_private.js"), pdir, "RESULTS_V2_PRIVATE", f"private overlay ({len(private)} method(s), never inside the deployed tree)")


if __name__ == "__main__":
    main()
