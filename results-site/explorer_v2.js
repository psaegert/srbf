/* Results explorer for benchmark releases from 2026-09 on (srbf 0.20 / flash-ansr 0.18, the two-part code).
 *
 * Reads window.RESULTS_V2 (schema 2): the metric REGISTRY and per method x catalog x rung the problems' counts, sums
 * and sums of squares (a problem's value is the mean of its runs), so any catalog subset is averaged on the client,
 * over the catalogs by random effects (owner 2026-09-27; scripts/site_random_effects.py is the reference).
 * Per-metric histograms (pooled medians, the Distribution view) are fetched on demand from <base>hist/<metric>.js;
 * the paired contrasts and pairwise rank outcomes along the budgets and times from <base>paired/<metric>.js and
 * <base>ranks/<metric>.js, and at any position they are computed here from <base>pv/ (pairstats.js).
 * window.RESULTS_V2_PRIVATE, if a LOCAL build provides it, is merged in (results-site/README.md, "Local-only
 * methods"); the public page never references such a file.
 * Views: Curves (every plotted metric vs budget or time) | Table (rungs, or catalogs at one rung) | Catalogs (matrix
 * of one metric at one rung) | Distribution (per-method histograms, cumulative curves, a box per catalog, quartiles
 * along the ladder; per-catalog rates for rate metrics) | Ranks (average places from the pairwise chances to beat,
 * pairwise tests, head-to-head table) | Paired Δ (each method against a baseline, problem by
 * problem: the mean difference, tested on the difference for rates and on the superiority otherwise) | Correlations
 * (two metrics problem by problem, from <base>pp/: contours, heat maps, points, trend, every pair, method against method).
 * State lives in the URL (?release=...&v=...) for sharing and in localStorage for convenience; colours in the same
 * first-party cookie srbf_colors, written only on an explicit change. The site's 2026-07 release (explorer.js) was
 * retired on 2026-09-26; its links' keys are dropped from the URL and its ?release= opens this page. */
(function () {
  "use strict";
  // Two pages mount this script: the home page holds the headline's two fixed charts, the explorer's page (explorer.html) the explorer.
  // Without the explorer's mount, only the headline is drawn: no controls, and the URL is left alone.
  var root = document.getElementById("results-explorer-v2");
  var headRoot = document.getElementById("results-headline-v2");
  if ((!root && !headRoot) || typeof window.RESULTS_V2 === "undefined") { return; }
  var D = JSON.parse(JSON.stringify(window.RESULTS_V2));
  var REL = D.release.id;
  // another release is on screen (the routing script decided before this file ran): leave the page and its URL alone
  D.status = D.status || {}; D.timing = D.timing || {};
  var SOURCES = [{ base: D.base, local: false }];
  // An overlay adds methods to the release: the same schema, merged into the tables every view reads. A local build
  // supplies one as a plain script and serves its lazy files from a base of its own (results-site/README.md,
  // "Local-only methods"); an overlay that arrives with a key carries those files inside it, so it has no base and
  // its methods are shown like any other.
  function mergeOverlay(P, withBase) {
    if (!P) { return []; }
    var added = [];
    (P.methods || []).forEach(function (m) {
      if (!D.methods.some(function (x) { return x.key === m.key; })) { D.methods.push(Object.assign({}, m, { local: !!withBase, overlay: true })); added.push(m.key); }
    });
    Object.assign(D.cells, P.cells || {}); Object.assign(D.status, P.status || {}); Object.assign(D.timing, P.timing || {});
    // its values and formulas problem by problem: indexed like the release's, and, when it arrived with a key, each file
    // sealed on its own under the overlay's file keys (sealedFile below)
    ["pp", "pred", "pv"].forEach(function (k) { if (P[k]) { D[k] = Object.assign(D[k] || {}, P[k]); } });
    if (P.pred_block && !D.pred_block) { D.pred_block = P.pred_block; }
    if (P.sealed_files && !withBase) { (P.methods || []).forEach(function (m) { SEALKEY[m.key] = P.sealed_files; }); OVERLAYKEYS.push(P.sealed_files); }
    // where each of its methods sits at every slot of the per-key cells, and against which brackets of the release's
    // methods its cells were computed (a release refreshed since the overlay was sealed has moved on: those are stale)
    if (P.slots) {
      D.slots = D.slots || { rungs: [], budgets: [], seconds: [], at: {}, stamp: {}, basis: {} };
      Object.assign(D.slots.at, P.slots.at || {}); Object.assign(D.slots.basis, P.slots.basis || {});
      (P.slots.rungs || []).forEach(function (r) { if (D.slots.rungs.indexOf(r) < 0) { D.slots.rungs.push(r); } }); D.slots.rungs.sort(function (a, b) { return a - b; });
    }
    if (withBase && P.base) { SOURCES.push({ base: P.base, local: true }); }
    return added;
  }
  var SEALKEY = {}, OVERLAYKEYS = [];   // a method that arrived with a key -> the file keys its overlay carries; every such overlay's keys
  mergeOverlay(window.RESULTS_V2_PRIVATE, true);
  var Z = 1.959964, LN2 = Math.log(2);
  var CATS = D.catalogs.map(function (c) { return c.key; });
  var CAT = {}; D.catalogs.forEach(function (c) { CAT[c.key] = c; });
  var GROUPS = { physics: "physics laws", classical: "classic benchmark sets", synthetic: "machine-generated formulas", other: "other" };
  var METRIC = {}; D.metrics.forEach(function (m) { METRIC[m.key] = m; });
  var MGROUPS = []; D.metrics.forEach(function (m) { if (MGROUPS.indexOf(m.group) < 0) { MGROUPS.push(m.group); } });
  var PAIRED_KEYS = D.paired_keys || [];
  // A plot is {x, y}: y is always a metric, x is a budget axis ("time", "rung") or a metric of its own (the
  // trade-off plot). The old shape was a bare metric key and still parses, from a link or from a saved state.
  function defaultAxis() { return anyTime() ? "time" : "rung"; }   // never default to a time the release does not publish
  function asPlot(v) {
    if (v && typeof v === "object") { return METRIC[v.y] ? { x: v.x || defaultAxis(), y: v.y } : null; }
    var s = String(v), i = s.indexOf("~");
    var x = i < 0 ? null : s.slice(0, i), y = i < 0 ? s : s.slice(i + 1);
    if (!METRIC[y]) { return null; }
    if (x && x !== "time" && x !== "rung" && !METRIC[x]) { x = null; }
    return { x: x || defaultAxis(), y: y };
  }
  function plotKey(p) { return p.x + "~" + p.y; }
  function plotAxes() { var seen = {}, out = []; state.plots.forEach(function (p) { [p.x, p.y].forEach(function (k) { if (METRIC[k] && !seen[k]) { seen[k] = 1; out.push(k); } }); }); return out; }
  function plotMetrics() { var seen = {}, out = []; state.plots.forEach(function (p) { if (!seen[p.y]) { seen[p.y] = 1; out.push(p.y); } }); return out; }
  function isBudgetAxis(x) { return x === "time" || x === "rung"; }

  // ---- One name, one definition, everywhere ----------------------------------------------------------------------
  // A metric carries its label wherever there is room (plot header, axis, table column, view title) and its short
  // form only where space is not negotiable (a point tooltip, a narrow screen). Its definition is ONE string, shown
  // by every surface that names it. The two budget axes are described here in the same shape, so the pickers and the
  // axis labels read them the same way.
  var AXIS = {
    time: { key: "time", label: "time per problem", short: "time", get desc() { return TERMS.time; } },
    rung: { key: "rung", label: "candidates", short: "candidates", get desc() { return TERMS.candidates; } }
  };
  function axisOf(k) { return AXIS[k] || METRIC[k]; }
  function mname(m) { return m ? m.label : ""; }
  function mdef(m) {
    if (!m) { return ""; }
    if (AXIS[m.key]) { return m.desc; }
    return m.desc + (m.every ? " It is the same for every method." : m.kind === "rate" ? " Every problem counts; a problem without a usable formula counts as a miss." : m.worst !== undefined ? " A problem without a usable formula counts as 0, unless you choose to leave such problems out." : " Measured only on problems where the method returned a usable formula.") +
      (m.higher === true ? " Higher is better." : m.higher === false ? " Lower is better." : m.ideal !== undefined ? " Closer to " + m.ideal + " is better." : "");
  }
  function mhelp(m) { return help(mdef(m), "What is " + mname(m) + "?"); }
  function axisName(m) { return narrow() ? m.short : mname(m) + (tfOf(m) ? " (log scale)" : ""); }
  function lastAxis() { return state.plots.length ? state.plots[state.plots.length - 1].x : defaultAxis(); }   // a new plot joins the last one
  var PROV = { upstream_default: "upstream defaults", author_blessed: "author-blessed", harness_tuned: "maintainer-chosen" };
  var PROV_NOTE = {
    upstream_default: "Settings: the defaults the method's own release ships with. srbf sets only the budget and, where the method takes a list of operators, the 23 the problems are written in. Nothing was tuned.",
    author_blessed: "Settings: the ones the method's authors use when they benchmark it themselves, which can differ from its library's defaults. The Flash-ANSR authors also run this benchmark, and this label says so.",
    harness_tuned: "Settings: chosen by the benchmark's maintainers, in whole or in part, such as the rule that picks the method's one answer."
  };
  var TERMS = {
    posbudget: "Every method is read at the budget set here, in its own unit. Where a method was run at that budget, its numbers are the measured ones. Between two budgets it was run at, they are interpolated by where the budget lies between the two on a logarithmic scale, and marked \u2248. Outside the budgets a method was run at, it has no value. Slide to any budget, or type one into the box; the arrows step to the next budget a shown method was run at.",
    postime: "Every method is read at the time per problem set here, on our timing workstation. Each budget a method was run at took a measured time; between two of them, its numbers are interpolated by where the time lies between theirs on a logarithmic scale, and marked \u2248. A method has no value at a time shorter than its smallest budget took or longer than its largest. Slide to any time, or type one into the box; the arrows step to the next of 0.1, 0.3, 1, 3, 10, 30, 100, 300 and 1,000 s.",
    between: "\u2248 marks a number read between two budgets a method was run at. Every sum over the problems is blended from the two budgets, in proportion to where the position lies between them on a logarithmic scale: a mean is the mean of the blended problems, and an interval is, if anything, a little too wide.",
    complete: "Each point is one method at one budget. Each method is run over its own range of budgets, and some runs are still in progress. A point appears only once the method has at least one finished run on every problem of the problem sets you selected (in the headline charts: all 29), so every point covers the same problems. Every problem is run twice, and its value is the average of its finished runs, so a point can rest partly on problems with one run so far. The page \u201cHow to read the results\u201d explains more, and how to see partial results by selecting fewer problem sets.",
    interval: "The 95 % interval shows how precisely a value is known on the problem sets you selected, from how much the problems within each set vary. It is about srbf's problem sets, not about problem sets in general: the range for one more problem set, in tables and tooltips, shows how far single sets spread. Bands shade the interval around each point and between neighbouring points; crosses draw it as bars through each point. You can show either, both or neither.",
    average: "First each problem's runs are averaged, then the problems of each problem set. Then the problem sets are averaged, each weighted by how precisely its own average is known, plus an allowance for how much problem sets differ from each other. Because they differ a lot, each counts about the same whatever its size, and a set of only a few problems counts less. A set of 5,000 problems therefore cannot decide the result on its own.",
    setrange: "Where the value of one more problem set would fall, with 95 % probability, if it were like the selected ones. It is wider than the 95 % interval: the interval says how precisely the average is known, this range how far single problem sets spread around it. It needs at least three problem sets.",
    median: "The median is read from a histogram with 128 bins, so it is accurate to the width of one bin (for log10 FVU, about 0.16). Each problem set weighs as much in the histogram as it does in an average, and a problem's runs share its weight.",
    mean: "The mean averages over the problems where the value is finite; a problem's value is the average of its runs. log10 FVU stops at -15.65, the precision of a 64-bit float, so an exact fit counts there like any other fit. A formula that blows up has an infinite value: the mean leaves it out, while the median counts it as the worst. Ratios, such as the MDL ratio, are averaged as geometric means, so a ratio of 2 and one of 0.5 average to 1. The tooltip of each point shows how many problems it averages.",
    regime: "How a problem without a usable formula counts. In a rate, such as Numeric Recovery, it counts as a miss. In the overlap metrics (Token Overlap, Variable Overlap) it counts as 0, unless you leave such problems out with the switch that appears in the side panel for these metrics. In log10 FVU and R\u00b2 it counts as the worst value in medians, distributions, places and comparisons, and means leave it out. All other metrics are measured only over the problems where the method returned a usable formula; most of them have no worst value, since a formula can always be worse.",
    impute: "What should a problem without a usable formula count in the overlap metrics? Counted (the default): it counts as 0, the lowest possible value, so a method cannot look better by failing on hard problems. Left out: the average covers only the problems where the method returned a usable formula. Rates always count such a problem as a miss.",
    valid: "Most metrics can only be measured on some problems: where the method returned a usable formula, and the value could be computed for it (a formula can, for example, give infinite values, or be one SimpliPy cannot simplify or price in bits). A method that fails on the hard problems then looks better than it is, because only its easier problems are measured. A point is drawn hollow when it is based on fewer than this share of the problems. The setting is under Reading in the explorer's side panel; 0 % turns the marking off.",
    pairdiff: "Each problem gives one difference: the method's value minus the baseline's, over every combination of their runs. The differences are averaged over problem sets like any value, with a 95 % interval. p asks whether the average difference could be zero. For a rate it tests the difference itself. For other metrics it tests how often the method does better than the baseline on a problem, so a few very large differences cannot decide it; \u0394 shows the size. A p-value below 0.05 means a difference this large is unlikely to be chance.",
    pairtest: "Each pair of methods is tested: on a problem, is one of them better more often than it is worse, averaged over the problem sets? The p-values are corrected for testing every pair (Holm's method), so the chance that chance alone separates any pair stays below 5 %. A method that is shown only with an access key is tested among its own pairs, so it never changes the verdict between the published methods.",
    draw1: "Every method is run twice on each problem, each time on newly sampled points from the same ranges. This view shows one run's prediction at a time. Everywhere else a problem's value is the average of its finished runs, and a comparison sets every run of one method against every run of the other.",
    time: "Seconds per problem, measured for every method on the same workstation (16 CPU cores, one RTX 4090 GPU), one method at a time, on a fixed sample of 262 problems. Timings from the compute cluster that produces most of the results depend on which machine a job ran on, so they are never shown. A method that has not been timed on the workstation yet has no place on a time axis; it is named below the chart instead.",
    candidates: "The budget of the methods that generate candidate formulas: how many candidates they may generate per problem. NeSymReS is placed here by its beam width, the number of partial formulas its search keeps. PySR counts search iterations, which cannot be placed on this axis, so it appears on the time axis only.",
    rungs: "A budget is how much search a method may spend on one problem, in the method's own unit: candidate formulas, beam width or search iterations. Most methods double their budget (1, 2, 4, 8 and so on), each over its own range; a method that works in batches doubles from its batch (DSR samples 1,000 formulas at a time). Each point on a curve is one budget a method was run at. Because the units differ, the same budget number means different amounts of work for different methods; the time axis shows what each budget costs.",
    predat: "A formula cannot be blended between two budgets, so each method shows the formula from its last budget at or below the position: the largest budget it was run at that is no larger than the budget set here, or, by time, the largest whose measured time per problem fits within the time set here. The budget column names it, with its time. Slide to any time or budget, or type one into the box; the arrows step to the next time limit or budget.",
    worstrank: "For every pair of methods and every problem: the chance that one does better than the other on the chosen metric, setting every run of one against every run of the other, with ties counting half. A method without a usable formula counts as worst, and on log10 FVU two predictions that both meet Numeric Recovery tie. These chances are averaged over problem sets like any value. A method's average place is 1 plus the chances that each other method does better than it: the place it takes on average. Only who is better counts, so a narrow win counts as much as a wide one.",
    winshare: "The chance that a method does better than another method on a problem, averaged over the other methods; a tie counts as half. 100 % means it beats every other method on every problem, and 50 % means it wins as often as it loses. Unlike the average place, it stays on the same 0 to 100 % scale when the number of methods changes; its value still depends on which methods are compared.",
    provenance: "Who chose each method's settings. Upstream defaults: the settings the method's own release ships with; nothing was tuned. Author-blessed: the settings the method's authors use when they benchmark it themselves; for Flash-ANSR, these are also the authors of this benchmark. Maintainer-chosen: settings chosen, in whole or in part, by the benchmark's maintainers."
  };
  var COOKIE = "srbf_colors";
  function readCookie() { var m = document.cookie.match(new RegExp("(?:^|; )" + COOKIE + "=([^;]*)")); if (!m) { return {}; } try { return JSON.parse(decodeURIComponent(m[1])) || {}; } catch (e) { return {}; } }
  function writeCookie(obj) { if (!obj || !Object.keys(obj).length) { document.cookie = COOKIE + "=;path=/;max-age=0;SameSite=Lax"; return; } document.cookie = COOKIE + "=" + encodeURIComponent(JSON.stringify(obj)) + ";path=/;max-age=" + (60 * 60 * 24 * 365) + ";SameSite=Lax"; }
  var userColors = readCookie();
  // A method drawn in the page's ink (the oracle) follows the theme: black on the light page, white on the dark one.
  function ink() { var v = window.getComputedStyle(document.documentElement).getPropertyValue("--ink"); return (v && v.trim()) || "#000000"; }
  // Three states per method (owner 2026-10-01): shown, faded, hidden. The headline starts every visit at its own default
  // -- the three methods below shown, the smaller Flash-ANSR models and their hybrid hidden (owner 2026-10-07: they
  // clutter the plots), every other one faded (GP-GOMEA too since 2026-10-02) -- and a click on a name in
  // its legend cycles that method, shown -> hidden -> faded -> shown; nothing of it is stored. The explorer keeps its
  // own states, in its method boxes (the same cycle), with an adjustable strength for the faded ones, and leaves the
  // headline alone. A faded method is drawn in its colour blended into the chart's background (so 20 % of the colour,
  // 80 % background), still opaque: an overlap of its own line and points does not darken, and it is drawn behind every
  // shown method.
  var FULL_DEFAULT = ["T8-120M", "T8-120M-pysr", "PySR"], HIDDEN_DEFAULT = ["T8-3M", "T8-20M", "T8-20M-pysr"], HL_FADE = 0.2, FADE_DEFAULT = 0.2;
  var hlVis = {}, HL_DRAWING = false;
  function hlDefault(k) { return FULL_DEFAULT.indexOf(k) >= 0 ? "full" : HIDDEN_DEFAULT.indexOf(k) >= 0 ? "hidden" : "dim"; }
  function hlVisOf(k) { return hlVis[k] || hlDefault(k); }
  function exVisOf(k) { return state.methods.indexOf(k) < 0 ? "hidden" : (state.dim || []).indexOf(k) >= 0 ? "dim" : "full"; }
  function visOf(k) { return HL_DRAWING ? hlVisOf(k) : exVisOf(k); }
  function nextVis(v) { return v === "full" ? "hidden" : v === "hidden" ? "dim" : "full"; }
  function alphaOf(k) { return visOf(k) === "dim" ? (HL_DRAWING ? HL_FADE : state.fade) : 1; }
  function tint(c, a) { return "color-mix(in srgb, " + String(c).trim() + " " + Math.round(a * 100) + "%, var(--surface))"; }
  function isFaded(sr) { return !!(sr && sr.key) && visOf(sr.key) === "dim"; }
  function behind(list) { return list.filter(isFaded).concat(list.filter(function (sr) { return !isFaded(sr); })); }   // faded first, so shown ones draw on top
  function baseColorOf(m) { return userColors[m.key] || (m.ink ? ink() : m.color); }
  function colorOf(m) { var c = baseColorOf(m), a = alphaOf(m.key); return a < 1 ? tint(c, a) : c; }
  function dashOf(sr) { return sr && sr.dash ? ' stroke-dasharray="7 4"' : ""; }
  function esc(s) { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/"/g, "&quot;"); }
  function term(key, text) { return '<span class="v2term" data-term="' + key + '" role="button" tabindex="0">' + text + "</span>"; }
  function help(text, label) { return '<button type="button" class="v2help" data-help="' + esc(text) + '" aria-label="' + esc(label || "What does this mean?") + '">?</button>'; }

  // ---- state: URL > localStorage > defaults --------------------------------------------------------------------
  var withData = function (m) { return D.cells[m.key] && Object.keys(D.cells[m.key]).length; };
  var VALID_DEFAULT = 90;
  var DEFAULTS = function () {
    return { view: "curves", cats: CATS.slice(), methods: D.methods.filter(function (m) { return withData(m) && !m.off && HIDDEN_DEFAULT.indexOf(m.key) < 0; }).map(function (m) { return m.key; }),
      dim: D.methods.filter(function (m) { return withData(m) && !m.off && FULL_DEFAULT.indexOf(m.key) < 0 && HIDDEN_DEFAULT.indexOf(m.key) < 0; }).map(function (m) { return m.key; }), fade: FADE_DEFAULT,
      plots: D.metrics.filter(function (m) { return m.tier === "main"; }).map(function (m) { return { x: defaultAxis(), y: m.key }; }),
      focus: "numeric_recovery_val", stat: "mean", band: true, cross: false, xaxis: anyTime() ? "time" : "rung", rung: 64, base: null, tier: "main", q: "", rows: "rungs", pset: null, prun: 1, pprob: 0,
      // the Distribution view reads a continuous metric by default (a rate has no distribution over problems), the Ranks
      // view the primary ranking metric; each display remembers its own
      dmetric: "log10_fvu_val", dmode: "hist", dnorm: "ok", rmetric: (D.rank_keys || ["log10_fvu_val"])[0], pm: anyTime() ? "time" : "budget", pt: null,
      // a point that rests on fewer than this share of the problems is drawn hollow
      valid: VALID_DEFAULT,
      // a metric whose range has a worst value: does a failed prediction count that value, or is it left out?
      impute: true,
      // the Correlations view: the two metrics, how they are drawn, and the two methods its pair displays compare
      cx: "mdl_ratio", cy: "log10_fvu_val", cv: "contour", ca: null, cb: null };
  };
  var state = DEFAULTS();
  var LS = "srbf-v2-" + REL + ".13";   // bumped whenever a default changes (.13 the 3M and 20M Flash-ANSR models and their hybrid hidden by default; .12 every display reads at a time by default; .11 GP-GOMEA faded by default; .10 shown, faded and hidden methods; .2 time axis, .3 mean, .4 bands, .5 per-view metrics, .6 complete pools only, .7 hollow markers, .8 failed predictions counted or left out, .9 methods marked off start unchecked), so a saved state cannot pin the old one
  var rungChosen = false;   // a budget from a link or from storage is kept; otherwise the first render picks one that has data
  function loadState() {
    try { var s = JSON.parse(localStorage.getItem(LS) || "null"); if (s) { rungChosen = s.rung !== undefined; Object.keys(state).forEach(function (k) { if (s[k] !== undefined) { state[k] = s[k]; } }); } } catch (e) { /* no storage */ }
    var q = new URLSearchParams(window.location.search); var any = false;
    var preset = { all: CATS, phys: CATS.filter(function (c) { return CAT[c].group === "physics"; }), classic: CATS.filter(function (c) { return CAT[c].group === "classical"; }), synth: CATS.filter(function (c) { return CAT[c].group === "synthetic"; }), none: [] };
    if (q.has("v")) { state.view = q.get("v"); any = true; }
    if (q.has("c")) { var c = q.get("c"); state.cats = preset[c] ? preset[c].slice() : c.split(",").filter(function (x) { return CAT[x]; }); any = true; }
    if (q.has("m")) {   // a faded method is written with a tilde: m=T8-120M,~e2e
      var mlist = q.get("m").split(","), known = function (x) { return D.methods.some(function (mm) { return mm.key === x; }); };
      state.methods = mlist.map(function (x) { return x.replace(/^~/, ""); }).filter(known);
      state.dim = mlist.filter(function (x) { return x.charAt(0) === "~"; }).map(function (x) { return x.slice(1); }).filter(known); any = true;
    }
    if (q.has("fa")) { var fa = parseFloat(q.get("fa")); if (fa > 0 && fa < 1) { state.fade = fa; } any = true; }
    if (q.has("p")) { state.plots = q.get("p").split(",").map(asPlot).filter(Boolean); any = true; }
    if (q.has("f") && METRIC[q.get("f")]) { state.focus = q.get("f"); any = true; }
    if (q.has("s")) { state.stat = q.get("s") === "median" ? "median" : "mean"; any = true; }
    if (q.has("ci")) { state.band = q.get("ci") !== "0"; state.cross = false; any = true; }   // the old single flag
    if (q.has("band")) { state.band = q.get("band") !== "0"; any = true; }
    if (q.has("cross")) { state.cross = q.get("cross") !== "0"; any = true; }
    if (q.has("x")) { state.xaxis = q.get("x") === "time" ? "time" : "rung"; any = true; }
    if (q.has("r")) { var r = parseFloat(q.get("r")); if (r > 0) { state.rung = r; rungChosen = true; } any = true; }
    if (q.has("pm")) { state.pm = q.get("pm") === "time" ? "time" : "budget"; any = true; }
    else if (q.has("t") && /^t[0-9.]+$/.test(q.get("t"))) { state.pm = "time"; state.pt = parseFloat(q.get("t").slice(1)); }   // a link from before 2026-10-05:
    else if ((state.view === "ranks" || state.view === "corr") && q.has("x")) { state.pm = q.get("x") === "time" ? "time" : "budget"; }   // Ranks and Correlations
    else if (q.has("r")) { state.pm = "budget"; }   // kept their own mode in x and t, and a link read at a budget had no pm
    if (q.has("pt")) { var pt = parseFloat(q.get("pt")); if (pt > 0) { state.pt = pt; } any = true; }
    if (q.has("dm") && METRIC[q.get("dm")]) { state.dmetric = q.get("dm"); any = true; }
    if (q.has("dv")) { state.dmode = q.get("dv"); any = true; }
    if (q.has("dn")) { state.dnorm = q.get("dn") === "all" ? "all" : "ok"; any = true; }
    if (q.has("rm") && METRIC[q.get("rm")]) { state.rmetric = q.get("rm"); any = true; }
    if (q.has("t")) { any = true; }
    if (q.has("b")) { state.base = q.get("b"); any = true; }
    if (q.has("rows")) { state.rows = q.get("rows") === "cats" ? "cats" : "rungs"; any = true; }
    if (q.has("ok")) { state.valid = parseInt(q.get("ok"), 10); any = true; }
    if (q.has("imp")) { state.impute = q.get("imp") !== "0"; any = true; }
    if (q.has("tier")) { state.tier = q.get("tier") === "all" ? "all" : "main"; }
    if (q.has("ps") && CAT[q.get("ps")]) { state.pset = q.get("ps"); any = true; }
    if (q.has("pr")) { state.prun = q.get("pr") === "2" ? 2 : 1; any = true; }
    if (q.has("pn")) { state.pprob = Math.max(0, (parseInt(q.get("pn"), 10) || 1) - 1); any = true; }
    ["cx", "cy", "cv", "ca", "cb"].forEach(function (k) { if (q.has(k)) { state[k] = q.get(k); any = true; } });
    if (["curves", "table", "matrix", "dist", "corr", "ranks", "paired", "preds"].indexOf(state.view) < 0) { state.view = "curves"; }
    var isPP = function (k) { return !!(METRIC[k] && METRIC[k].pp); };
    if (!isPP(state.cx)) { state.cx = isPP("mdl_ratio") ? "mdl_ratio" : (D.metrics.filter(function (m) { return m.pp; })[0] || {}).key || null; }
    if (!isPP(state.cy)) { state.cy = isPP("log10_fvu_val") ? "log10_fvu_val" : state.cx; }
    if (["contour", "heat", "points", "trend", "matrix", "vs"].indexOf(state.cv) < 0) { state.cv = "contour"; }
    if (state.prun !== 2) { state.prun = 1; }
    state.pprob = Math.max(0, state.pprob | 0);
    if (["hist", "ecdf", "cats", "rungs"].indexOf(state.dmode) < 0) { state.dmode = "hist"; }
    if (!METRIC[state.dmetric]) { state.dmetric = "log10_fvu_val"; }
    if (!METRIC[state.rmetric]) { state.rmetric = (D.rank_keys || ["log10_fvu_val"])[0]; }
    if (!Array.isArray(state.cats)) { state.cats = CATS.slice(); }
    if (!Array.isArray(state.methods)) { state.methods = DEFAULTS().methods; }
    if (!Array.isArray(state.plots)) { state.plots = DEFAULTS().plots; }
    state.cats = state.cats.filter(function (x) { return CAT[x]; });
    state.plots = state.plots.map(asPlot).filter(Boolean);   // from a link, from storage, or from an older shape
    state.methods = state.methods.filter(function (x) { return D.methods.some(function (mm) { return mm.key === x; }); });
    if (!Array.isArray(state.dim)) { state.dim = []; }
    state.dim = state.dim.filter(function (x) { return state.methods.indexOf(x) >= 0; });
    if (!(state.fade > 0 && state.fade < 1)) { state.fade = FADE_DEFAULT; }
    if (!METRIC[state.focus]) { state.focus = "numeric_recovery_val"; }
    if (!state.base || state.methods.indexOf(state.base) < 0) { state.base = state.methods[0] || null; }
    if (!(state.rung > 0)) { state.rung = 64; }
    if (state.pm !== "time" || !anyTime()) { state.pm = "budget"; }
    if (!(state.pt > 0)) { state.pt = null; }
    state.impute = state.impute !== false;
    state.valid = isFinite(state.valid) ? Math.min(100, Math.max(0, Math.round(state.valid))) : VALID_DEFAULT;
    return any;
  }
  loadState();
  function catsParam() {
    var set = {}; state.cats.forEach(function (c) { set[c] = 1; });
    var is = function (g) { var gs = CATS.filter(function (c) { return CAT[c].group === g; }); return gs.length === state.cats.length && gs.every(function (c) { return set[c]; }); };
    if (state.cats.length === CATS.length) { return "all"; } if (!state.cats.length) { return "none"; }
    if (is("physics")) { return "phys"; } if (is("classical")) { return "classic"; } if (is("synthetic")) { return "synth"; }
    return state.cats.join(",");
  }
  // A link is meant to be passed around, so it names only methods the release publishes; a method that arrived
  // with a key is remembered for this browser (localStorage, this device only) and re-added by the key, not by a URL.
  var byKey = {};
  function sharedMethods() { return state.methods.filter(function (k) { return !byKey[k]; }); }
  function save() {
    try { localStorage.setItem(LS, JSON.stringify(state)); } catch (e) { /* no storage */ }
    var q = new URLSearchParams(window.location.search);
    ["view", "bench", "baseline", "metric", "budget"].forEach(function (k) { q.delete(k); });   // never carry the retired 2026-07 explorer's keys
    q.set("release", REL); q.set("v", state.view); q.set("c", catsParam()); q.set("m", sharedMethods().map(function (k) { return (state.dim.indexOf(k) >= 0 ? "~" : "") + k; }).join(",")); q.delete("fa"); if (state.fade !== FADE_DEFAULT) { q.set("fa", String(state.fade)); } q.set("p", state.plots.map(plotKey).join(","));
    q.set("f", state.focus); q.set("s", state.stat); q.delete("pool"); q.delete("thin"); q.set("band", state.band ? "1" : "0"); q.set("cross", state.cross ? "1" : "0");
    q.set("x", state.xaxis); q.set("r", String(state.rung)); q.set("pm", state.pm); q.delete("pt"); if (state.pm === "time" && state.pt) { q.set("pt", String(state.pt)); } if (state.base) { q.set("b", state.base); } q.set("rows", state.rows); q.set("ok", String(state.valid)); q.set("imp", state.impute ? "1" : "0");
    ["dm", "dv", "dn", "rm", "t", "ps", "pr", "pn", "pp", "cx", "cy", "cv", "ca", "cb"].forEach(function (k) { q.delete(k); });   // a link carries only what its display reads
    if (state.view === "preds") { if (state.pset) { q.set("ps", state.pset); } q.set("pr", String(state.prun)); q.set("pn", String(state.pprob + 1)); }
    if (state.view === "dist") { q.set("dm", state.dmetric); q.set("dv", state.dmode); q.set("dn", state.dnorm); }
    if (state.view === "ranks") { q.set("rm", state.rmetric); }
    if (state.view === "corr") { q.set("cv", state.cv); if (state.cv !== "matrix") { q.set("cy", state.cy); } if (state.cv !== "matrix" && state.cv !== "vs") { q.set("cx", state.cx); } if ((state.cv === "matrix" || state.cv === "vs") && state.ca) { q.set("ca", state.ca); q.set("cb", state.cb || state.ca); } }
    try { window.history.replaceState(null, "", "?" + q.toString() + window.location.hash); } catch (e) { /* file:// */ }
  }

  // ---- lazy payloads: histograms per metric, paired contrasts ----------------------------------------------------
  var loading = {}, failed = {};
  // a file of a method that arrived with a key: its values or formulas problem by problem, sealed on its own
  function sealedOwner(file) { var m = /^(pp|pred|pv)\/([^/]+)\//.exec(file); return m && SEALKEY[m[2]] ? SEALKEY[m[2]] : null; }
  // a file of per-key cells or histograms: the release's, and each keyed overlay's own, sealed under that overlay's keys
  function shared(file) { return /^(ranks|paired|hist)\//.test(file); }
  function ensure(file, cb) {
    var sk = sealedOwner(file); if (sk) { sealedFile(file, sk, cb); return; }
    if (shared(file)) { OVERLAYKEYS.forEach(function (k) { sealedFile(file, k, cb); }); }
    SOURCES.forEach(function (s) {
      var key = s.base + file; if (loading[key] || failed[key]) { return; }
      loading[key] = "pending";
      var sc = document.createElement("script"); sc.src = s.base + file; sc.async = true;
      sc.onload = function () { loading[key] = "done"; cb(); };
      sc.onerror = function () { failed[key] = true; delete loading[key]; cb(); };
      document.head.appendChild(sc);
    });
  }
  function sealedDone(file, sk) { var key = "sealed:" + sk.name + ":" + file; return loading[key] === "done" || !!failed[key]; }
  function ready(file) {
    var sk = sealedOwner(file); if (sk) { return sealedDone(file, sk); }
    return (!shared(file) || OVERLAYKEYS.every(function (k) { return sealedDone(file, k); })) &&
      SOURCES.every(function (s) { return loading[s.base + file] === "done" || failed[s.base + file]; });
  }
  function histOf(k) { var H = window.RESULTS_V2_HIST && window.RESULTS_V2_HIST[REL]; return H && H[k]; }

  // ---- adding a method from a key ---------------------------------------------------------------------------------
  // A release may ship sealed payloads next to the public ones. Each holds the same kind of payload scripts as the
  // rest of the release -- an overlay, its histograms and its paired contrasts -- gzipped and encrypted with
  // AES-256-GCM under a key derived by PBKDF2-HMAC-SHA256, each under a key of its own (tools/seal.mjs). No key is in
  // this repository and nothing here derives one; the payloads are fetched only when someone asks, a key is tried on
  // every one, and a key that opens none leaves the page exactly as it was. What the page shows without them is
  // complete on its own terms.
  var sealedLoaded = false, sealedWaiting = null;
  function bytesOf(b64) { var s = atob(b64), u = new Uint8Array(s.length); for (var i = 0; i < s.length; i++) { u[i] = s.charCodeAt(i); } return u; }
  function withSealed(cb) {   // cb once the sealed payloads are here (or known to be absent); callers queue meanwhile
    if (sealedLoaded || window.RESULTS_V2_SEALED) { cb(); return; }
    if (sealedWaiting) { sealedWaiting.push(cb); return; }
    sealedWaiting = [cb];
    var done = function () { sealedLoaded = true; var w = sealedWaiting; sealedWaiting = null; w.forEach(function (f) { f(); }); };
    var sc = document.createElement("script"); sc.src = D.base + "sealed.js"; sc.async = true;
    sc.onload = done; sc.onerror = done; document.head.appendChild(sc);
  }
  function envelopes() {   // the first where a single one has always been, the others after it
    var S = window.RESULTS_V2_SEALED, M = window.RESULTS_V2_SEALED_MORE;
    return [S && S[REL]].concat((M && M[REL]) || []).filter(Boolean);
  }
  function openOne(env, key, subtle) {   // -> the payload's source, or null when the key does not open it
    return subtle.importKey("raw", new TextEncoder().encode(key), "PBKDF2", false, ["deriveKey"])
      .then(function (base) {
        return subtle.deriveKey({ name: "PBKDF2", salt: bytesOf(env.salt), iterations: env.iter, hash: "SHA-256" },
          base, { name: "AES-GCM", length: 256 }, false, ["decrypt"]);
      })
      .then(function (k) { return subtle.decrypt({ name: "AES-GCM", iv: bytesOf(env.iv) }, k, bytesOf(env.ct)); })
      .then(function (gz) { return new Response(new Blob([gz]).stream().pipeThrough(new DecompressionStream("gzip"))).text(); })
      .catch(function () { return null; });   // a key that does not fit is not told apart from a release without one
  }
  function openWithKey(key) {   // -> the keys of the methods it added, or [] when nothing opened
    return new Promise(function (res) { withSealed(res); }).then(function () {
      var subtle = window.crypto && window.crypto.subtle, envs = envelopes();
      if (!envs.length || !subtle || typeof DecompressionStream === "undefined") { return []; }
      return Promise.all(envs.map(function (env) { return openOne(env, key, subtle); })).then(function (sources) {
        var added = [];
        sources.forEach(function (src) {   // one payload at a time: each sets RESULTS_V2_PRIVATE and is merged at once
          if (src == null) { return; }
          try {
            (new Function(src))();   // the payload scripts, run exactly as a <script> tag would run them
            added = added.concat(mergeOverlay(window.RESULTS_V2_PRIVATE, false));
          } catch (e) { /* a payload that does not run adds nothing */ }
        });
        added.forEach(function (k2) { byKey[k2] = true; if (state.methods.indexOf(k2) < 0) { state.methods.push(k2); } });
        return added;
      });
    });
  }
  // A sealed overlay's per-problem files are sealed one by one (tools/seal.mjs): each at sealed/<name>.js, where the name
  // is an HMAC of the file's path under the overlay's name key, encrypted with AES-256-GCM under its file key, the
  // path bound in as associated data. The overlay carries both keys, so a file costs no key derivation; a file is
  // fetched only when a display asks for it, and one that does not open counts as missing.
  var SEALKEYS = {};
  function sealedKeys(sk, subtle) {
    var id = sk.enc + "|" + sk.name;
    return SEALKEYS[id] || (SEALKEYS[id] = Promise.all([
      subtle.importKey("raw", bytesOf(sk.enc), { name: "AES-GCM" }, false, ["decrypt"]),
      subtle.importKey("raw", bytesOf(sk.name), { name: "HMAC", hash: "SHA-256" }, false, ["sign"])]).then(function (ks) { return { enc: ks[0], name: ks[1] }; }));
  }
  function loadScript(src) { return new Promise(function (res, rej) { var sc = document.createElement("script"); sc.src = src; sc.async = true; sc.onload = res; sc.onerror = rej; document.head.appendChild(sc); }); }
  function sealedFile(file, sk, cb) {
    var key = "sealed:" + sk.name + ":" + file; if (loading[key] || failed[key]) { return; }
    loading[key] = "pending";
    var fail = function () { failed[key] = true; delete loading[key]; cb(); }, subtle = window.crypto && window.crypto.subtle, te = new TextEncoder();
    if (!subtle || typeof DecompressionStream === "undefined") { fail(); return; }
    sealedKeys(sk, subtle).then(function (k) {
      return subtle.sign("HMAC", k.name, te.encode("name\u0000" + file)).then(function (mac) {
        var name = Array.prototype.map.call(new Uint8Array(mac), function (b) { return (b < 16 ? "0" : "") + b.toString(16); }).join("").slice(0, 32);
        return loadScript(D.base + "sealed/" + name + ".js").then(function () {
          var env = (window.RESULTS_V2_SEALED_FILES || {})[name]; if (!env) { throw new Error("no such sealed file"); }
          return subtle.decrypt({ name: "AES-GCM", iv: bytesOf(env.iv), additionalData: te.encode(REL + "/" + file) }, k.enc, bytesOf(env.ct));
        });
      });
    }).then(function (gz) { return new Response(new Blob([gz]).stream().pipeThrough(new DecompressionStream("gzip"))).text(); })
      .then(function (src) { (new Function(src))(); loading[key] = "done"; cb(); })   // run exactly as a <script> tag would run it
      .catch(fail);
  }
  // The keys that opened something, remembered for this browser session (the first version kept one, as srbf.k).
  function savedKeys() {
    var out = [];
    try {
      out = JSON.parse(window.sessionStorage.getItem("srbf.keys") || "[]") || [];
      var one = window.sessionStorage.getItem("srbf.k");
      if (one && out.indexOf(one) < 0) { out.push(one); }
    } catch (e) { /* storage off */ }
    return out;
  }
  function saveKey(key) {
    var keys = savedKeys();
    if (keys.indexOf(key) < 0) { keys.push(key); }
    try { window.sessionStorage.setItem("srbf.keys", JSON.stringify(keys)); window.sessionStorage.removeItem("srbf.k"); } catch (e) { /* storage off */ }
  }
  function labelsOf(keys) {
    return keys.map(function (k) { var m = D.methods.filter(function (x) { return x.key === k; })[0]; return m ? m.label : k; }).join(", ");
  }
  function tryKey(key, quiet) {
    var msg = root.querySelector(".v2addmmsg");
    if (msg && !quiet) { msg.textContent = "checking…"; }
    openWithKey(key).then(function (added) {
      if (added.length) {
        saveKey(key);
        shell(); render();
        var after = root.querySelector(".v2addmmsg");
        if (after && !quiet) { after.textContent = "Added " + labelsOf(added) + "."; }
        return;
      }
      if (!quiet && msg) { msg.textContent = "No method found for that key."; }
    });
  }

  // ---- pooling ---------------------------------------------------------------------------------------------------
  // ---- positions between the budgets a method was run at (owner 2026-10-01) ------------------------------------------
  // Tables, problem sets and distributions read every method at one position: a budget, or a time per problem on our
  // timing workstation ("t" + seconds). Where a method was run at that budget (or that time), its numbers are the
  // measured ones. Between two budgets it was run at, they are interpolated, linearly in the logarithm of the budget (or
  // of the time): every sum a problem set carries -- problems, values, squares -- is (1 - w) times the lower budget's plus
  // w times the upper one's. A mean is then exactly the mean of every problem's interpolated value, and a spread is never
  // smaller than that of the interpolated values (an interval is, if anything, too wide). A histogram is blended the same
  // way. Outside the budgets a method was run at, nothing is extrapolated: the method has no value there.
  var BETWEEN = false;   // on only while a display that reads at a position draws; everywhere else a budget is a budget a method was run at
  function isTimePos(r) { return typeof r === "string" && r.charAt(0) === "t"; }
  function posValue(r) { return isTimePos(r) ? parseFloat(r.slice(1)) : +r; }
  function cellRaw(m, c, r) { var x = D.cells[m] && D.cells[m][c] && D.cells[m][c][String(r)]; return x && x.state === "complete" ? x : null; }
  // the finished budgets of method m on problem set c, as [position, budget], by budget or by time, in order
  function points(m, c, time) {
    var cs = (D.cells[m] || {})[c] || {}, out = [];
    Object.keys(cs).forEach(function (k) { if (cs[k].state !== "complete") { return; } var v = time ? refTime(m, k) : +k; if (v > 0) { out.push([v, +k]); } });
    return out.sort(function (a, b) { return a[0] - b[0]; });
  }
  // where x lies among the points [position, budget] (in order): at one of them (w = 0), or between two neighbours.
  // pairstats.js, where the page has it, takes the logarithms with arithmetic of its own, the same as the exporter's:
  // the browser's Math.log can differ from Python's in the last bit, and the weights must agree with the exporter's.
  function bracketIn(ps, x) {
    if (typeof window !== "undefined" && window.PAIRSTATS && window.PAIRSTATS.bracketIn) { return window.PAIRSTATS.bracketIn(ps, x); }
    var i; if (!(x > 0)) { return null; }
    for (i = 0; i < ps.length; i++) { if (Math.abs(ps[i][0] - x) <= 1e-9 * x) { return { r1: ps[i][1], r2: ps[i][1], w: 0 }; } }
    for (i = 0; i + 1 < ps.length; i++) { if (ps[i][0] < x && x < ps[i + 1][0]) { return { r1: ps[i][1], r2: ps[i + 1][1], w: Math.log(x / ps[i][0]) / Math.log(ps[i + 1][0] / ps[i][0]) }; } }
    return null;
  }
  function bracket(m, c, r) { return bracketIn(points(m, c, isTimePos(r)), posValue(r)); }
  function blendArr(a, b, w) { return a && b && a.length === b.length ? a.map(function (v, i) { return (1 - w) * v + w * b[i]; }) : null; }
  function blendMap(A, B, w) { var out = {}; Object.keys(A || {}).forEach(function (k) { var x = Array.isArray(A[k]) ? blendArr(A[k], (B || {})[k], w) : (B || {})[k] !== undefined ? (1 - w) * A[k] + w * B[k] : null; if (x !== null) { out[k] = x; } }); return out; }
  var BLENDED = {};
  function cell(m, c, r) {
    if (!isTimePos(r)) { var exact = cellRaw(m, c, r); if (exact || !BETWEEN) { return exact; } }
    var b = bracket(m, c, r); if (!b) { return null; }
    if (b.r1 === b.r2) { return cellRaw(m, c, b.r1); }
    var key = m + "|" + c + "|" + r; if (BLENDED[key]) { return BLENDED[key]; }
    var a = cellRaw(m, c, b.r1), z = cellRaw(m, c, b.r2), w = b.w;
    return (BLENDED[key] = { state: "complete", d: Math.min(a.d || 1, z.d || 1), n: (1 - w) * a.n + w * z.n, ok: (1 - w) * (a.ok || 0) + w * (z.ok || 0),
      m: blendMap(a.m, z.m, w), a: blendMap(a.a, z.a, w), e: blendMap(a.e, z.e, w), between: [b.r1, b.r2] });
  }
  function histCell(H, m, c, r) {
    var cs = H.cells[m] && H.cells[m][c]; if (!cs) { return null; }
    if (!isTimePos(r) && (cs[String(r)] || !BETWEEN)) { return cs[String(r)] || null; }
    var b = bracket(m, c, r); if (!b) { return null; }
    var h1 = cs[String(b.r1)]; if (b.r1 === b.r2) { return h1 || null; }
    var h2 = cs[String(b.r2)]; if (!h1 || !h2) { return null; }
    var a = new Array(H.nb).fill(0), z = new Array(H.nb).fill(0); addHist(a, h1, H.nb); addHist(z, h2, H.nb);
    return a.map(function (v, i) { return (1 - b.w) * v + b.w * z[i]; });
  }
  function curPos() { return state.pm === "time" && state.pt ? "t" + state.pt : state.rung; }
  function fmtSec(x) { return x >= 100 ? Math.round(x).toLocaleString() : x >= 10 ? x.toFixed(1).replace(/\.0$/, "") : x >= 1 ? x.toFixed(2).replace(/\.?0+$/, "") : String(+x.toPrecision(2)); }
  function fmtBudget(x) { return x >= 10 ? Math.round(x).toLocaleString() : String(+x.toPrecision(3)); }
  function posText(r) { return isTimePos(r) ? fmtSec(posValue(r)) + " s per problem" : "budget " + fmtBudget(+r); }
  function hasRung(m, r) { return CATS.some(function (c) { return cell(m, c, r); }); }
  function shownMethods() { return D.methods.filter(function (m) { return state.methods.indexOf(m.key) >= 0 && withData(m); }); }
  // A pooled number stands for the selected catalogs only if it holds ALL of their problems: a budget is shown
  // once it is entirely complete, never as a thin rung. The catalogs differ far too much for a
  // part to speak for the whole -- one synthetic corpus is four fifths of the problems, and recovery on it is a fraction
  // of recovery elsewhere -- and a rung that is still running would move as its catalogs land. So a method has a
  // pooled value at a rung when it has finished every selected catalog there, and none otherwise; every method that
  // has one is then pooled over the same problems, which is what "matched" used to arrange.
  function poolCats(m, r) {
    return state.cats.length && state.cats.every(function (c) { return cell(m, c, r); }) ? state.cats.slice() : [];
  }
  function laws(cs) { return cs.reduce(function (a, c) { return a + CAT[c].laws; }, 0); }
  // ---- averaging over problem sets (owner 2026-09-27) --------------------------------------------------------------
  // The results are problem sets > problems > runs. A problem's value is the mean of its runs; a set is its number of
  // problems with a value, their sum and their sum of squares (the exporter's cells). Sets are combined by a random-
  // effects model: weights 1 / (tau^2 + the set's own noise), tau^2 by Paule-Mandel, the 95 % interval over these problem
  // sets (combineScale), the prediction interval for a new problem set with t on S - 2. Rates on the logit
  // scale, with the continuity-corrected rate (x n + 0.5) / (n + 1). scripts/site_random_effects.py is the reference:
  // the same arithmetic, step for step, and the site suite checks that the two agree.
  var EPS_VAR = 1e-12;
  function lgamma(x) { var g = 7, c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]; if (x < 0.5) { return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x); } x -= 1; var a = c[0], t = x + g + 0.5; for (var i = 1; i < g + 2; i++) { a += c[i] / (x + i); } return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a); }
  function betacf(a, b, x) {
    var qab = a + b, qap = a + 1, qam = a - 1, c = 1, d = 1 - qab * x / qap, h, aa, del, m, m2;
    d = 1 / (Math.abs(d) > 1e-300 ? d : 1e-300); h = d;
    for (m = 1; m < 300; m++) {
      m2 = 2 * m; aa = m * (b - m) * x / ((qam + m2) * (a + m2));
      d = 1 + aa * d; d = 1 / (Math.abs(d) > 1e-300 ? d : 1e-300); c = 1 + aa / c; c = Math.abs(c) > 1e-300 ? c : 1e-300; h *= d * c;
      aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
      d = 1 + aa * d; d = 1 / (Math.abs(d) > 1e-300 ? d : 1e-300); c = 1 + aa / c; c = Math.abs(c) > 1e-300 ? c : 1e-300;
      del = d * c; h *= del; if (Math.abs(del - 1) < 1e-15) { break; }
    }
    return h;
  }
  function betai(a, b, x) {
    if (x <= 0) { return 0; } if (x >= 1) { return 1; }
    var bt = Math.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log(1 - x));
    return x < (a + 1) / (a + b + 2) ? bt * betacf(a, b, x) / a : 1 - bt * betacf(b, a, 1 - x) / b;
  }
  function tCdf(t, df) { var tail = 0.5 * betai(df / 2, 0.5, df / (df + t * t)); return t >= 0 ? 1 - tail : tail; }
  var TQ = {};
  function tq975(df) {
    if (TQ[df]) { return TQ[df]; }
    var lo = 0, hi = 1, i, mid; while (tCdf(hi, df) < 0.975) { hi *= 2; }
    for (i = 0; i < 100; i++) { mid = (lo + hi) / 2; if (tCdf(mid, df) < 0.975) { lo = mid; } else { hi = mid; } }
    return (TQ[df] = (lo + hi) / 2);
  }
  function tP(t, df) { return Math.min(1, 2 * (1 - tCdf(Math.abs(t), df))); }
  function expit(x) { return 1 / (1 + Math.exp(-x)); }
  function setVar(st) { if (st.n < 2) { return null; } var m = st.s1 / st.n; return Math.max(0, (st.s2 - st.n * m * m) / (st.n - 1)); }
  // each set's estimate on the combining scale and its squared standard error; a set with one problem, or whose problems
  // all have the same value, borrows a spread: a rate its continuity-corrected p(1 - p), any other metric the pooled one
  function setEstimates(sets, scale) {
    var num = 0, den = 0, y = [], v = [];
    sets.forEach(function (st) { var vv = setVar(st); if (vv !== null) { num += (st.n - 1) * vv; den += st.n - 1; } });
    var pooled = den > 0 ? num / den : 0;
    sets.forEach(function (st) {
      var own = setVar(st), m = st.s1 / st.n; own = own !== null && own > 0 ? own : null;
      if (scale === "logit") { var pc = (st.n * m + 0.5) / (st.n + 1), vr = own !== null ? own : pc * (1 - pc); y.push(Math.log(pc / (1 - pc))); v.push(Math.max(EPS_VAR, vr / st.n / Math.pow(pc * (1 - pc), 2))); }
      else { y.push(m); v.push(Math.max(EPS_VAR, (own !== null ? own : pooled) / st.n)); }
    });
    return { y: y, v: v, pooled: pooled };
  }
  function pmExcess(y, v, t2) {
    var sw = 0, swy = 0, i, w, mu, q = 0;
    for (i = 0; i < y.length; i++) { w = 1 / (v[i] + t2); sw += w; swy += w * y[i]; }
    mu = swy / sw; for (i = 0; i < y.length; i++) { q += (1 / (v[i] + t2)) * Math.pow(y[i] - mu, 2); }
    return q - (y.length - 1);
  }
  function pauleMandel(y, v) {
    if (y.length < 2 || pmExcess(y, v, 0) <= 0) { return 0; }
    var lo = 0, hi = 1, i, mid; while (pmExcess(y, v, hi) > 0) { hi *= 2; }
    for (i = 0; i < 80; i++) { mid = (lo + hi) / 2; if (pmExcess(y, v, mid) > 0) { lo = mid; } else { hi = mid; } }
    return (lo + hi) / 2;
  }
  // The interval covers these problem sets (owner 2026-09-27): the weights held fixed, only each set's own noise,
  // Var(mu) = sum w^2 v / (sum w)^2, t on the problems' degrees of freedom (sum of n - 1). One set is its own mean with
  // the t interval on its problems. The range for one more problem set is the prediction interval, t with S - 2.
  function combineScale(y, v, df) {
    var S = y.length, i, tau2 = S > 1 ? pauleMandel(y, v) : 0, w = [], sw = 0, mu = 0, q = 0;
    for (i = 0; i < S; i++) { w.push(1 / (v[i] + tau2)); sw += w[i]; mu += w[i] * y[i]; }
    mu /= sw; for (i = 0; i < S; i++) { q += w[i] * w[i] * v[i]; }
    df = Math.max(1, df || 1);
    var se = Math.sqrt(q) / sw, tq = tq975(df), half = S >= 3 ? tq975(S - 2) * Math.sqrt(tau2 + 1 / sw) : null;
    return { mu: mu, lo: mu - tq * se, hi: mu + tq * se, piLo: half === null ? null : mu - half, piHi: half === null ? null : mu + half,
             tau2: tau2, w: w.map(function (x) { return x / sw; }), S: S, p: se > 0 ? tP(mu / se, df) : (mu !== 0 ? 0 : 1) };
  }
  // sets: [{n, s1, s2, ...}]; a rate ("logit") comes back on its own 0-1 scale. Also: the problems behind it, and n0,
  // the number of problems at which a set counts half as much as a very large one (sigma^2 / tau^2)
  function reCombine(sets, scale) {
    sets = sets.filter(function (st) { return st.n > 0; }); if (!sets.length) { return null; }
    var e = setEstimates(sets, scale), c = combineScale(e.y, e.v, sets.reduce(function (a, st) { return a + st.n - 1; }, 0));
    if (scale === "logit") { ["mu", "lo", "hi", "piLo", "piHi"].forEach(function (k) { if (c[k] !== null) { c[k] = expit(c[k]); } }); }
    c.n = sets.reduce(function (a, st) { return a + st.n; }, 0); c.sets = sets; c.n0 = c.tau2 > 0 ? e.pooled / c.tau2 : Infinity;
    return c;
  }
  function holm(ps) {
    var order = ps.map(function (_p, i) { return i; }).sort(function (a, b) { return ps[a] - ps[b]; }), out = ps.slice(), run = 0;
    order.forEach(function (i, rank) { run = Math.max(run, Math.min(1, (ps.length - rank) * ps[i])); out[i] = run; });
    return out;
  }
  function binVal(h, i) { return h.lo + (i + 0.5) * (h.hi - h.lo) / h.nb; }
  function addHist(acc, hc, nb) { if (hc.length && Array.isArray(hc[0])) { hc.forEach(function (p) { acc[p[0]] += p[1]; }); } else { for (var i = 0; i < nb; i++) { acc[i] += hc[i] || 0; } } }
  // A metric whose range has a worst value ships with the failed predictions counted at it, and every cell says how
  // many of its values were filled in that way ("w"). Leaving them out again is exact: that many come off the sums
  // and out of the bin the worst value falls into.
  function leftOut(k) { return METRIC[k] && METRIC[k].worst !== undefined && !state.impute; }
  function reading(c, k) { return c ? (leftOut(k) ? c.a && c.a[k] : c.m[k]) : null; }   // [problems with a value, with a finite one, sum, sum of squares]
  function histKey(k) { return leftOut(k) ? k + "@answered" : k; }
  function histFile(k) { return "hist/" + histKey(k).replace("@answered", "_answered") + ".js"; }
  // A distribution over the selected problem sets weighs each set the way the average does: n / (n + n0), with n0 from the
  // fit of the metric's mean (its median_via for R^2), so a set counts about once whatever its size, and a set of a few
  // problems less. Within a set every problem weighs 1 (the exporter shares it among its runs). Returned scaled to the
  // number of problems, so a share is h / n as before.
  function setWeights(k, m, r, cs) {
    var mk = METRIC[k] && METRIC[k].median_via ? METRIC[k].median_via : k, sets = [];
    cs.forEach(function (c) { var t = reading(cell(m, c, r), mk); if (t && t[1] > 0) { sets.push({ c: c, n: t[1], s1: t[2], s2: t[3] }); } });
    var fit = sets.length >= 2 ? reCombine(sets, "normal") : null;
    return function (n) { return fit && isFinite(fit.n0) ? n / (n + fit.n0) : n; };
  }
  function pooledHist(k, m, r, cs) {
    var H = histOf(histKey(k)); if (!H || !H.cells[m]) { return null; }
    var weigh = setWeights(k, m, r, cs), acc = new Array(H.nb).fill(0), N = 0, parts = [];
    cs.forEach(function (c) { var hc = histCell(H, m, c, r); if (!hc) { return; } var h = new Array(H.nb).fill(0); addHist(h, hc, H.nb); var tot = h.reduce(function (a, b) { return a + b; }, 0); if (tot > 0) { parts.push({ c: c, h: h, tot: tot, w: weigh(tot) }); N += tot; } });
    var W = parts.reduce(function (a, pt) { return a + pt.w; }, 0); if (!N || !W) { return null; }
    parts.forEach(function (pt) { for (var i = 0; i < H.nb; i++) { acc[i] += (pt.w / W) * pt.h[i] / pt.tot * N; } });
    return { h: acc, n: N, lo: H.lo, hi: H.hi, nb: H.nb, parts: parts, W: W };
  }
  // The interval of a median (Woodruff): the share of each set below the pooled median, averaged over the sets like any
  // rate, gives an interval for that share; the pooled distribution maps it back to the metric.
  function medianShareHalf(ph, mid) {
    var sets = ph.parts.map(function (pt) { var below = 0; for (var i = 0; i <= mid; i++) { below += pt.h[i]; } var p = Math.min(1, below / pt.tot); return { n: Math.max(1, Math.round(pt.tot)), s1: p * Math.max(1, Math.round(pt.tot)), s2: p * Math.max(1, Math.round(pt.tot)) }; });
    var c = reCombine(sets, "normal"); return c ? Math.max(0, (c.hi - c.lo) / 2) : 0;
  }
  function quantileBin(h, kth) { var cum = 0; for (var i = 0; i < h.nb; i++) { cum += h.h[i]; if (cum >= kth) { return i; } } return h.nb - 1; }
  function tfOf(metric) { return metric.hist && metric.hist.tf; }
  function fwd(metric, x) { var tf = tfOf(metric); return tf === "log2" ? Math.log2(Math.max(1e-300, x)) : tf === "log10" ? Math.log10(Math.max(1e-300, x)) : x; }
  function back(metric, x) { var tf = tfOf(metric); return tf === "log2" ? Math.pow(2, x) : tf === "log10" ? Math.pow(10, x) : x; }
  // every statistic is returned in the metric's TRANSFORMED space (log2 for ratios, log10 for seconds), where the charts live
  // A rate is read over every problem, and so is a metric whose range has a worst value: a failed prediction takes it.
  // Every other metric has no worst value and describes the predictions that were made, so a method that fails on the
  // hard problems looks better there than it is. `share` is the part of the problems behind a point (of the problems the metric
  // can be defined for); below the reader's threshold the point is drawn hollow.
  function validShare(metric, cells) {   // the share of the problems (that the metric can be defined for) with a finite value
    if (metric.kind === "rate") { return 1; }
    var d = 0, e = 0; cells.forEach(function (c) { var t = reading(c, metric.key); d += t ? t[1] : 0; e += c.e && c.e[metric.key] !== undefined ? c.e[metric.key] : c.n; });
    return e ? d / e : null;
  }
  // where one more problem set would fall (at least three sets); shown in tooltips and tables, never drawn (owner 2026-09-27)
  function rangeText(metric, st) { return st && st.piLo !== null && st.piLo !== undefined ? "; one problem set: " + fmt(metric, st.piLo) + " to " + fmt(metric, st.piHi) : ""; }
  function problemsText(st, use) {
    var n = Math.round(st.n), S = st.S || use.length;
    var more = st.ndef ? st.ndef - n : 0;
    return "From " + n.toLocaleString() + " problems in " + S + (S === 1 ? " problem set" : " problem sets") + (more > 0 ? "; " + more.toLocaleString() + " more have no finite value, which the mean leaves out" : "") + (st.d === 1 ? "; some problems have one finished run so far" : "");
  }
  function thin(st) { return !!st && !st.pending && typeof st.share === "number" && st.share < state.valid / 100; }
  function shareText(st) { return Math.floor(100 * st.share) + " % of the problems have a value"; }   // shown in tooltips
  var THIN_MARK = '<span class="v2hollow" aria-hidden="true">\u25cb</span> ';
  // An unbounded metric with a heavy tail has no usable mean (one diverging prediction decides it): it is read by its
  // median whatever the reader chose. R^2 = 1 - FVU is read from two histograms, each where it is the finer one:
  // its own linear bins (0.016 wide) up to 0.96, and the log10 FVU bins (0.16 decades) above that, where the linear
  // ones cannot tell 0.99 from 0.9999, and below -1, where they end.
  function statOf(metric) { return metric.median_via ? "median" : state.stat; }
  function histKeys(metric) { return metric.median_via ? [metric.key, metric.median_via] : [metric.key]; }
  function needsHist(metric) { return metric.kind === "cont" && statOf(metric) === "median"; }
  function ensureHists(metric) { histKeys(metric).forEach(function (k) { ensure(histFile(k), scheduleRender); }); }
  var R2_FINE = 0.96;
  function r2Quantile(own, fvu, kth) {   // the kth smallest R^2 is the kth largest FVU
    var i = quantileBin(own, kth), v = binVal(own, i);
    if (i > 0 && v <= R2_FINE) { return { v: v, edge: 0 }; }
    var j = quantileBin(fvu, Math.min(fvu.n, Math.max(1, fvu.n + 1 - kth)));
    return { v: j === 0 ? 1 : 1 - Math.pow(10, binVal(fvu, j)), edge: j === fvu.nb - 1 ? -1 : 0 };
  }
  // a cell reports the fewest runs any of its problems has (c.d); a pooled statistic reports the fewest among its cells
  function stat(metric, m, r, cs) {
    var out = stat0(metric, m, r, cs);
    if (out && !out.pending) { var used = cs.map(function (c) { return cell(m, c, r); }).filter(Boolean), ds = used.map(function (c) { return c.d || 1; }); out.d = ds.length ? Math.min.apply(null, ds) : 1;
      var tween = used.filter(function (c) { return c.between; }); if (tween.length) { out.between = tween[0].between; } }
    return out;
  }
  function stat0(metric, m, r, cs) {
    var cells = cs.map(function (c) { return cell(m, c, r); }).filter(Boolean); if (!cells.length) { return null; }
    var fit, out;
    if (metric.kind === "rate") {
      fit = reCombine(cells.map(function (c) { var t = c.m[metric.key]; return t ? { n: t[0], s1: t[1], s2: t[2] } : { n: 0 }; }), "logit");
      return fit ? { v: fit.mu, lo: fit.lo, hi: fit.hi, piLo: fit.piLo, piHi: fit.piHi, n: fit.n, S: fit.S, share: 1 } : null;
    }
    var share = validShare(metric, cells);
    if (statOf(metric) === "mean") {
      fit = reCombine(cells.map(function (c) { var t = reading(c, metric.key); return t ? { n: t[1], s1: t[2], s2: t[3] } : { n: 0 }; }), "normal");
      if (!fit) { return null; }
      var nd = cells.reduce(function (a, c) { var t = reading(c, metric.key); return a + (t ? t[0] : 0); }, 0);
      return { v: fit.mu, lo: fit.lo, hi: fit.hi, piLo: fit.piLo, piHi: fit.piHi, n: fit.n, ndef: nd, S: fit.S, share: share };
    }
    if (!histKeys(metric).every(function (k) { return ready(histFile(k)); })) { return { pending: true }; }
    var ph = pooledHist(metric.key, m, r, cs); if (!ph) { return null; }
    if (metric.median_via) {
      var pf = pooledHist(metric.median_via, m, r, cs); if (!pf) { return null; }
      var q = function (share_) { return r2Quantile(ph, pf, Math.max(1e-9, Math.min(ph.n, share_ * ph.n))); }, half0 = medianShareHalf(pf, quantileBin(pf, pf.n / 2)), qm = q(0.5);
      out = { v: qm.v, lo: q(Math.max(0, 0.5 - half0)).v, hi: q(Math.min(1, 0.5 + half0)).v, n: ph.n, S: ph.parts.length, edge: qm.edge, share: share };
      return out;
    }
    var mid = quantileBin(ph, ph.n / 2), half = medianShareHalf(ph, mid);
    var lo = quantileBin(ph, Math.max(1e-9, (0.5 - half) * ph.n)), hi = quantileBin(ph, Math.min(ph.n, (0.5 + half) * ph.n));
    return { v: binVal(ph, mid), lo: binVal(ph, lo), hi: binVal(ph, hi), n: ph.n, S: ph.parts.length, edge: mid === 0 ? -1 : (mid === ph.nb - 1 ? 1 : 0), share: share };
  }

  function fmt(metric, x, edge) {
    if (x === null || x === undefined || !isFinite(x)) { return "–"; }
    var v = back(metric, x), s;
    if (metric.fmt === "pct") { s = (100 * v).toFixed(1) + " %"; }
    else if (metric.fmt === "ratio") { s = v < 0.01 ? v.toExponential(1) : v.toFixed(2); }
    else if (metric.fmt === "sec") { s = v < 1 ? v.toFixed(2) + " s" : v < 100 ? v.toFixed(1) + " s" : Math.round(v) + " s"; }
    else { s = v.toFixed(metric.fmt === "num1" ? 1 : metric.fmt === "num3" ? 3 : 2); }
    return (edge === -1 ? "≤ " : edge === 1 ? "≥ " : "") + s;
  }
  // ---- axes: ticks at round values, a range that follows what is drawn, labels that never crowd ------------------
  function niceStep(span, target) { var raw = span / Math.max(1, target), p = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / p; return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * p; }
  function roundNum(v) { if (!isFinite(v)) { return "–"; } var a = Math.abs(v); if (a >= 1000) { return Math.round(v).toLocaleString(); } var t = String(+v.toPrecision(a >= 100 ? 4 : 3)); return t === "-0" ? "0" : t; }
  function linearTicks(lo, hi, target) {   // the round step (1, 2 or 5 times a power of ten) whose tick count comes closest to five, the coarser on a tie
    var want = target || 5, span = Math.max(hi - lo, 1e-12), p = Math.pow(10, Math.floor(Math.log10(span))), best = null, bestD = Infinity;
    [10, 5, 2, 1, 0.5, 0.2, 0.1, 0.05].forEach(function (f) { var st = f * p, n = Math.floor(hi / st + 1e-9) - Math.ceil(lo / st - 1e-9) + 1, d = Math.abs(n - want) + (n < 3 ? 100 : 0); if (d < bestD) { bestD = d; best = st; } });
    var step = best || niceStep(span, want), out = []; for (var t = Math.ceil(lo / step - 1e-9) * step; t <= hi + step * 1e-9; t += step) { out.push(+t.toFixed(10)); } return out;
  }
  // a log axis, positions in exponents of `base`: powers of the base over a wide range, round VALUES over a narrow one
  function logTicks(lo, hi, base, target) {
    var span = hi - lo, out = [], k, lb = Math.log(base);
    if (span * lb / Math.LN2 < 3.2) { return linearTicks(Math.pow(base, lo), Math.pow(base, hi), target || 5).filter(function (v) { return v > 0; }).map(function (v) { return Math.log(v) / lb; }); }
    if (base === 10 && span <= 3.2) { for (k = Math.floor(lo); k <= Math.ceil(hi); k++) { [1, 2, 5].forEach(function (m) { var e = k + Math.log10(m); if (e >= lo - 1e-9 && e <= hi + 1e-9) { out.push(e); } }); } return out; }
    var step = Math.max(1, Math.ceil(span / (target || 6)));
    for (k = Math.ceil(lo / step - 1e-9) * step; k <= hi + 1e-9; k += step) { out.push(k); } return out;
  }
  function tickLabel(metric, x) { var tf = tfOf(metric); if (metric.kind === "rate") { return roundNum(100 * x) + "%"; } if (tf === "log2") { return roundNum(Math.pow(2, x)); } if (tf === "log10") { return roundNum(Math.pow(10, x)) + " s"; } return roundNum(x); }
  function ticksFor(metric, ymin, ymax, target) { var tf = tfOf(metric); return tf === "log2" ? logTicks(ymin, ymax, 2, target) : tf === "log10" ? logTicks(ymin, ymax, 10, target) : linearTicks(ymin, ymax, target); }
  // labels that would sit closer than `gap` pixels are dropped, the rounder value kept (an integer before a fraction)
  function thinTicks(ticks, pos, gap, weight) {
    var order = ticks.slice().sort(function (a, b) { return (weight ? weight(b) - weight(a) : 0) || a - b; }), kept = [];
    order.forEach(function (t) { if (kept.every(function (k) { return Math.abs(pos(k) - pos(t)) >= gap; })) { kept.push(t); } });
    return kept.sort(function (a, b) { return a - b; });
  }
  function roundness(v) { var a = Math.abs(v); if (a < 1e-12) { return 9; } var e = Math.floor(Math.log10(a)), m = a / Math.pow(10, e); return Math.abs(m - 1) < 1e-9 ? 3 : (Math.abs(m - 5) < 1e-9 || Math.abs(m - 2) < 1e-9 ? 2 : 1); }
  // a reference value (the problem itself, no difference) joins the range only when it is near what is drawn: far away
  // it would leave most of the chart empty
  function nearRange(lo, hi, ref, frac) { var span = Math.max(hi - lo, 1e-9); return ref >= lo - frac * span && ref <= hi + frac * span; }

  // ---- SVG chart: series of points {x, v, lo, hi, hollow, title} on a rung/time x axis -----------------------------
  function coarse() { return !!(window.matchMedia && window.matchMedia("(pointer: coarse)").matches); }   // a finger, not a mouse: no keyboard until one is asked for
  function narrow() { return window.innerWidth < 700; }   // the viewport, like the CSS breakpoints: a container reflows, this does not
  var CHART_MIN = 520, CHART_GAP = 20;   // must match the grid in styles.css (.v2charts, .v2hlcharts)
  var TITLE_Y = 30, YLABEL_X = 24;   // a chart's title baseline and its y label's centre: room from the frame
  function inner(el) {   // the width the grid actually has: clientWidth still counts the padding
    if (!el || !el.clientWidth) { return 0; }
    var cs = window.getComputedStyle(el);
    return el.clientWidth - (parseFloat(cs.paddingLeft) || 0) - (parseFloat(cs.paddingRight) || 0);
  }
  function chartWidth(host, count) {
    var avail = inner(host) || inner(root) || window.innerWidth;
    var fits = Math.max(1, Math.min(3, Math.floor((avail + CHART_GAP) / (CHART_MIN + CHART_GAP))));
    var cols = Math.max(1, Math.min(fits, count || fits));
    return Math.max(300, Math.floor((avail - CHART_GAP * (cols - 1)) / cols));
  }
  function plotHeight(w) { return Math.max(250, Math.min(400, Math.round(w * 0.52))); }
  // A legend or label column is as wide as its longest name, measured in the chart's own font (.leg, 12 px): a fixed
  // width cuts a long method name at the chart's edge.
  var measurer = null;
  function textWidth(text) {
    if (!measurer) { measurer = document.createElement("canvas").getContext("2d"); }
    measurer.font = "12px " + (window.getComputedStyle(root || headRoot).fontFamily || "sans-serif");
    return measurer.measureText(text).width;
  }
  function widest(labels) { return labels.reduce(function (w, t) { return Math.max(w, textWidth(t)); }, 0); }
  function legendRight(labels) { return Math.max(160, Math.ceil(LEG_GAP + 26 + widest(labels) + 16)); }   // gap, swatch, name, margin
  var LEG_GAP = 14;   // between the plot area and its legend
  var chartHost = null, chartCount = 0;   // set while a block renders: its charts are built for THAT container
  function hostWidth() { return chartWidth(chartHost || (root && root.querySelector(".v2main")) || root || headRoot, chartCount); }
  // A display of one chart (Distribution, Ranks) spans the column up to WIDE_MAX and is drawn at the width it is shown
  // at: drawn at a grid cell's width and stretched, its text and marks would grow with the window.
  var WIDE_MAX = 980;   // must match .v2distwide and .v2one in styles.css
  function wideWidth() { return Math.min(WIDE_MAX, chartWidth(chartHost || (root && root.querySelector(".v2main")) || root || headRoot, 1)); }
  function inBlock(host, count, fn) { chartHost = host; chartCount = count; try { return fn(); } finally { chartHost = null; chartCount = 0; } }
  // ---- the 95 % interval, drawn -----------------------------------------------------------------------------------
  // Every point carries an uncertainty BOX: [xlo, xhi] x [lo, hi] in data units, whatever the axes are. The band is
  // the region a ladder could occupy -- the boxes, plus the convex hull of each consecutive pair, so the shape
  // follows the path instead of assuming it runs left to right. The pieces are drawn in one group at a single
  // opacity, so overlaps do not darken. Where the x is a budget the boxes are vertical segments and the union is
  // exactly the familiar envelope; where both axes are measured the band is genuinely two-dimensional.
  function anyCI() { return state.band || state.cross; }
  function hullOf(ps) {   // monotone chain, on the few corners of one or two boxes
    var q = ps.slice().sort(function (a, b) { return a[0] - b[0] || a[1] - b[1]; }), i, lo = [], up = [];
    var turn = function (o, a, b) { return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]); };
    for (i = 0; i < q.length; i++) { while (lo.length > 1 && turn(lo[lo.length - 2], lo[lo.length - 1], q[i]) <= 0) { lo.pop(); } lo.push(q[i]); }
    for (i = q.length - 1; i >= 0; i--) { while (up.length > 1 && turn(up[up.length - 2], up[up.length - 1], q[i]) <= 0) { up.pop(); } up.push(q[i]); }
    return lo.slice(0, -1).concat(up.slice(0, -1));
  }
  function boxOf(p, xs, y, clx, cly) {
    var a = xs(clx(isFinite(p.xlo) ? p.xlo : p.x)), b = xs(clx(isFinite(p.xhi) ? p.xhi : p.x));
    var c = y(cly(isFinite(p.hi) ? p.hi : p.v)), d = y(cly(isFinite(p.lo) ? p.lo : p.v));
    return [[a, c], [b, c], [b, d], [a, d]];
  }
  function ciSVG(pts, col, xs, y, clx, cly) {
    if (!pts.length || !anyCI()) { return ""; }
    var out = "";
    if (state.band) {
      var polys = pts.map(function (p, i) {
        var corners = boxOf(p, xs, y, clx, cly);
        if (i + 1 < pts.length) { corners = corners.concat(boxOf(pts[i + 1], xs, y, clx, cly)); }
        return '<polygon points="' + hullOf(corners).map(function (q) { return q[0].toFixed(1) + "," + q[1].toFixed(1); }).join(" ") + '"/>';
      });
      out += '<g fill="' + col + '" opacity="0.13" stroke="none">' + polys.join("") + "</g>";
    }
    if (state.cross) {
      out += pts.map(function (p) {
        var px = xs(clx(p.x)).toFixed(1), py = y(cly(p.v)).toFixed(1), bar = "";
        if (isFinite(p.lo) && isFinite(p.hi)) { bar += '<line x1="' + px + '" y1="' + y(cly(p.hi)).toFixed(1) + '" x2="' + px + '" y2="' + y(cly(p.lo)).toFixed(1) + '" stroke="' + col + '" stroke-width="1.5" stroke-opacity="0.45"/>'; }
        if (isFinite(p.xlo) && isFinite(p.xhi)) { bar += '<line x1="' + xs(clx(p.xlo)).toFixed(1) + '" y1="' + py + '" x2="' + xs(clx(p.xhi)).toFixed(1) + '" y2="' + py + '" stroke="' + col + '" stroke-width="1.5" stroke-opacity="0.45"/>'; }
        return bar;
      }).join("");
    }
    return out;
  }
  // A legend entry: its line and its name. On the headline (cycle) it is a button that cycles the method's state, and a
  // hidden method keeps an entry, dotted, so that it can be brought back.
  function ghostLabels(ghosts) { return ghosts.map(function (m) { return m.label + (m.local ? " (local)" : ""); }); }
  function ghostSeries(m) { return { key: m.key, label: m.label + (m.local ? " (local)" : ""), color: baseColorOf(m), dash: !!m.dash }; }
  // A legend keeps one order, the methods' own: a method stays in its place whether it is shown, faded or hidden, so
  // cycling one never moves the others.
  function legendEntries(series, ghosts) {
    var pos = {};
    D.methods.forEach(function (m, i) { pos[m.key] = i; });
    var at = function (e) { return pos[e.sr.key] === undefined ? D.methods.length : pos[e.sr.key]; };
    return series.map(function (sr) { return { sr: sr, gone: false }; })
      .concat(ghosts.map(function (m) { return { sr: ghostSeries(m), gone: true }; }))
      .sort(function (a, b) { return at(a) - at(b); });
  }
  function legendItem(sr, lx, ly, cycle, gone) {
    var vis = gone ? "hidden" : sr.key ? visOf(sr.key) : "full", lab = esc(sr.label);
    var mark = '<line x1="' + lx + '" y1="' + ly + '" x2="' + (lx + 20) + '" y2="' + ly + '" stroke="' + sr.color + '"' + (gone ? ' stroke-width="2" stroke-dasharray="2 3"' : ' stroke-width="3"' + dashOf(sr)) + "/>";
    var text = '<text x="' + (lx + 26) + '" y="' + (ly + 4) + '" class="leg' + (vis === "hidden" ? " v2leggone" : vis === "dim" ? " v2legdim" : "") + '">' + lab + "</text>";
    if (!cycle || !sr.key) { return mark + text; }
    var word = vis === "hidden" ? "hidden" : vis === "dim" ? "faded" : "shown";
    return '<g class="v2legitem" data-cycle="' + esc(sr.key) + '" data-vis="' + vis + '" role="button" tabindex="0" aria-label="' + lab + ", " + word + '; a click changes it"><title>' + lab + ": " + word + " (a click cycles shown, hidden, faded)</title>" +
      '<rect x="' + (lx - 4) + '" y="' + (ly - 10) + '" width="' + Math.ceil(34 + 7 * sr.label.length) + '" height="20" fill="transparent"/>' + mark + text + "</g>";
  }
  function chartSVG(opts) {
    var series = opts.series, nr = narrow(), W = opts.width || hostWidth(), L = 76, T = opts.title ? 52 : 18, ghosts = opts.ghosts || [];
    var R = nr ? 18 : legendRight(series.map(function (sr) { return sr.label; }).concat(ghostLabels(ghosts)));
    var B = nr ? 60 + 20 * Math.max(1, series.length + ghosts.length) : 58, H = plotHeight(W) + B;
    // A narrow chart shortens the visible axis label to "(s, ref)"; the calibration is named in full here, so
    // every width -- and every screen reader -- says which clock these seconds came from.
    var aria = (opts.aria || opts.title || "") + (opts.timeAxis && opts.timeSource === "ref" ? ", timed on one workstation" : "");
    var s = '<svg viewBox="0 0 ' + W + ' ' + H + '" class="v2chart" role="img" aria-label="' + esc(aria) + '">' + (opts.title ? '<text x="' + L + '" y="' + TITLE_Y + '" class="ct">' + esc(opts.title) + "</text>" : "");
    if (opts.empty) { return s + '<text x="' + W / 2 + '" y="' + H / 2 + '" class="tick" text-anchor="middle">' + esc(opts.empty) + '</text></svg>'; }
    var ymin = opts.ymin, ymax = opts.ymax; if (!(ymax > ymin)) { ymax = ymin + 1; }
    var timeAxis = opts.timeAxis, tmin = opts.tmin, tmax = opts.tmax;
    var rmax = 4; series.forEach(function (sr) { sr.pts.forEach(function (p) { if (!timeAxis && p.x > rmax) { rmax = p.x; } }); });
    var xs = function (x) { return timeAxis ? L + (Math.log10(x) - Math.log10(tmin)) / (Math.log10(tmax) - Math.log10(tmin)) * (W - L - R) : L + 8 + Math.log2(x) / Math.log2(rmax) * (W - L - R - 16); };
    var y = function (v) { return T + (1 - (v - ymin) / (ymax - ymin)) * (H - T - B); };
    thinTicks(opts.ticks, y, 20, roundness).forEach(function (g) { s += '<line x1="' + L + '" y1="' + y(g).toFixed(1) + '" x2="' + (W - R) + '" y2="' + y(g).toFixed(1) + '" class="grid"/><text x="' + (L - 6) + '" y="' + (y(g) + 4).toFixed(1) + '" class="tick" text-anchor="end">' + esc(opts.tick(g)) + '</text>'; });
    if (opts.zero !== undefined && opts.zero >= ymin && opts.zero <= ymax) { s += '<line x1="' + L + '" y1="' + y(opts.zero).toFixed(1) + '" x2="' + (W - R) + '" y2="' + y(opts.zero).toFixed(1) + '" class="grid zero" stroke-dasharray="4 4"/>'; }
    var xtext = function (a, label) {   // the outermost tick label leans inwards instead of over the edge
      var anchor = a > W - 30 ? "end" : (a < L + 14 ? "start" : "middle"), x = anchor === "end" ? Math.min(a, W - 2) : (anchor === "start" ? Math.max(a, 2) : a);
      return '<text x="' + x.toFixed(1) + '" y="' + (H - B + 16) + '" class="tick" text-anchor="' + anchor + '">' + label + "</text>";
    };
    if (timeAxis) {
      var tt = logTicks(Math.log10(tmin), Math.log10(tmax), 10, nr ? 4 : 7).map(function (e) { return Math.pow(10, e); });
      thinTicks(tt, xs, nr ? 46 : 58, roundness).forEach(function (t) { s += '<line x1="' + xs(t).toFixed(1) + '" y1="' + T + '" x2="' + xs(t).toFixed(1) + '" y2="' + (H - B) + '" class="grid"/>' + xtext(xs(t), roundNum(t) + " s"); });
    } else {
      var rr = D.rungs.filter(function (r) { return r <= rmax; });
      rr.forEach(function (r) { s += '<line x1="' + xs(r).toFixed(1) + '" y1="' + (H - B) + '" x2="' + xs(r).toFixed(1) + '" y2="' + (H - B + 4) + '" class="grid"/>'; });
      thinTicks(rr, xs, nr ? 34 : 44, function (r) { return Math.round(Math.log2(r)) % 2 ? 1 : 2; }).forEach(function (r) { s += xtext(xs(r), String(r >= 1024 ? (r / 1024) + "k" : r)); });
    }
    var xlab = opts.xlabel || (timeAxis
      ? (nr ? "time (s)" : "time per problem (s, log scale)")
      : (nr ? "budget" : "budget per problem (log scale)"));
    s += '<text x="' + ((L + W - R) / 2).toFixed(0) + '" y="' + (H - B + 32) + '" class="tick" text-anchor="middle">' + esc(xlab) + '</text>';
    if (opts.ylabel) { s += '<text transform="translate(' + YLABEL_X + ',' + ((T + H - B) / 2).toFixed(0) + ') rotate(-90)" class="tick" text-anchor="middle">' + esc(opts.ylabel) + "</text>"; }
    var ly = nr ? H - B + 46 : T + 6, lx = nr ? L : W - R + LEG_GAP;
    // A series' points come in its method's budget order and the line keeps it: on a time axis a larger budget can be
    // timed faster than a smaller one (PySR at 1 and 2 iterations: 2.58 and 2.03 s), and re-sorting by x joined the
    // points in time order, a dip that is not in the data. The band follows the same path (ciSVG).
    behind(series).forEach(function (sr) { var col = sr.color, pts = sr.pts; var cl = function (v) { return Math.min(ymax, Math.max(ymin, v)); };
      s += ciSVG(pts, col, xs, y, function (v) { return v; }, cl);
      s += '<polyline fill="none" stroke="' + col + '" stroke-width="2"' + dashOf(sr) + ' points="' + pts.map(function (p) { return xs(p.x).toFixed(1) + "," + y(cl(p.v)).toFixed(1); }).join(" ") + '"/>';
      pts.forEach(function (p) { s += markerSVG(xs(p.x), y(cl(p.v)), p, col); }); });
    legendEntries(series, ghosts).forEach(function (e) { s += legendItem(e.sr, lx, ly, opts.cycle, e.gone); ly += 20; });
    return s + "</svg>";
  }
  // x positions. The budget axis is the method's own ladder. The time axis is seconds per problem, always from
  // the reference machine -- one chart uses one source for all of its methods, never a mixture.
  // The x position of a time axis is a CALIBRATED time or it does not exist. Seconds measured where a unit
  // happened to run -- a cluster node, its GPU, whatever else was on it -- are not comparable between methods, so
  // they are never an axis and never published. A method the reference machine has not timed therefore has no
  // position on this axis: it is left out of the chart and named underneath, exactly as a method whose budget is
  // a time limit is left out of the candidate axis. The rest of the chart is unaffected.
  function refTime(m, r) { var t = (D.timing[m] || {})[String(r)]; return t > 0 ? t : null; }
  function hasRefTime(k) { return !!(D.timing[k] && Object.keys(D.timing[k]).length); }
  function timeSource(keys) { return keys.length && keys.every(hasRefTime) ? "ref" : null; }
  function timeOf(m, r, cs, src) { return src === "ref" ? refTime(m, r) : null; }

  // A chart whose x is a metric, not a budget: each method's ladder walks a path through the plane
  // (here: how long its prediction is against how well it fits), so the points keep their rung order.
  function frontSVG(opts) {
    var nr = narrow(), W = opts.width || hostWidth(), L = 76, T = opts.title ? 52 : 18, ghosts = opts.ghosts || [];
    var R = nr ? 18 : legendRight(opts.series.map(function (sr) { return sr.label; }).concat(ghostLabels(ghosts)));
    var B = nr ? 60 + 20 * Math.max(1, opts.series.length + ghosts.length) : 58, H = plotHeight(W) + B;
    var s = '<svg viewBox="0 0 ' + W + ' ' + H + '" class="v2chart" role="img" aria-label="' + esc(opts.aria || opts.title) + '">' + (opts.title ? '<text x="' + L + '" y="' + TITLE_Y + '" class="ct">' + esc(opts.title) + "</text>" : "");
    if (opts.empty) { return s + '<text x="' + W / 2 + '" y="' + H / 2 + '" class="tick" text-anchor="middle">' + esc(opts.empty) + "</text></svg>"; }
    var xmin = opts.xmin, xmax = opts.xmax, ymin = opts.ymin, ymax = opts.ymax;
    if (!(xmax > xmin)) { xmax = xmin + 1; } if (!(ymax > ymin)) { ymax = ymin + 1; }
    var xs = function (x) { return L + (x - xmin) / (xmax - xmin) * (W - L - R); };
    var y = function (v) { return T + (1 - (v - ymin) / (ymax - ymin)) * (H - T - B); };
    var clx = function (v) { return Math.min(xmax, Math.max(xmin, v)); }, cly = function (v) { return Math.min(ymax, Math.max(ymin, v)); };
    thinTicks(opts.yticks, y, 20, roundness).forEach(function (g) { s += '<line x1="' + L + '" y1="' + y(g).toFixed(1) + '" x2="' + (W - R) + '" y2="' + y(g).toFixed(1) + '" class="grid"/><text x="' + (L - 6) + '" y="' + (y(g) + 4).toFixed(1) + '" class="tick" text-anchor="end">' + esc(opts.ytick(g)) + "</text>"; });
    thinTicks(opts.xticks, xs, nr ? 40 : 52, roundness).forEach(function (g) { s += '<line x1="' + xs(g).toFixed(1) + '" y1="' + T + '" x2="' + xs(g).toFixed(1) + '" y2="' + (H - B) + '" class="grid"/><text x="' + xs(g).toFixed(1) + '" y="' + (H - B + 16) + '" class="tick" text-anchor="middle">' + esc(opts.xtick(g)) + "</text>"; });
    if (opts.xzero !== undefined && opts.xzero > xmin && opts.xzero < xmax) { s += '<line x1="' + xs(opts.xzero).toFixed(1) + '" y1="' + T + '" x2="' + xs(opts.xzero).toFixed(1) + '" y2="' + (H - B) + '" class="grid zero" stroke-dasharray="4 4"/>'; }
    s += '<text x="' + ((L + W - R) / 2).toFixed(0) + '" y="' + (H - B + 32) + '" class="tick" text-anchor="middle">' + esc(opts.xlabel) + "</text>";
    s += '<text transform="translate(' + YLABEL_X + ',' + ((T + H - B) / 2).toFixed(0) + ') rotate(-90)" class="tick" text-anchor="middle">' + esc(opts.ylabel) + "</text>";
    var ly = nr ? H - B + 46 : T + 6, lx = nr ? L : W - R + LEG_GAP;
    behind(opts.series).forEach(function (sr) {
      var col = sr.color;
      s += ciSVG(sr.pts, col, xs, y, clx, cly);
      s += '<polyline fill="none" stroke="' + col + '" stroke-width="1.6" stroke-opacity="0.65"' + dashOf(sr) + ' points="' + sr.pts.map(function (p) { return xs(clx(p.x)).toFixed(1) + "," + y(cly(p.v)).toFixed(1); }).join(" ") + '"/>';
      sr.pts.forEach(function (p) { s += markerSVG(xs(clx(p.x)), y(cly(p.v)), p, col); });
    });
    legendEntries(opts.series, ghosts).forEach(function (e) { s += legendItem(e.sr, lx, ly, opts.cycle, e.gone); ly += 20; });
    return s + "</svg>";
  }
  function frontChart(xm, ym, shown, title, aria, xlabel, ylabel, legend) {
    var keys = shown.map(function (m) { return m.key; }), series = [], pending = false;
    var xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
    shown.forEach(function (m) {
      var pts = [];
      D.rungs.forEach(function (r) {
        var use = poolCats(m.key, r, keys); if (!use.length) { return; }
        var sx = stat(xm, m.key, r, use), sy = stat(ym, m.key, r, use); if (!sx || !sy) { return; }
        if (sx.pending || sy.pending) { pending = true; return; }
        if (!isFinite(sx.v) || !isFinite(sy.v)) { return; }
        var least = thin(sx) && (!thin(sy) || sx.share < sy.share) ? sx : sy;
        if (thin(sx) || thin(sy)) { thinDrawn = true; }
        pts.push({ x: sx.v, xlo: sx.lo, xhi: sx.hi, v: sy.v, lo: sy.lo, hi: sy.hi, hollow: thin(sx) || thin(sy), end: endsAt(m.key, r),
          title: withEnd(m.key, r, m.label + " @ " + r + ": " + fmt(xm, sx.v, sx.edge) + " " + xm.short + ", " + fmt(ym, sy.v, sy.edge) + " " + ym.short + ", " + Math.round(sy.n).toLocaleString() + " problems in " + sy.S + (sy.S === 1 ? " problem set" : " problem sets") + (least.share < 1 ? ", " + shareText(least) : "")) });
        [sx.v, anyCI() ? sx.lo : sx.v, anyCI() ? sx.hi : sx.v].forEach(function (v) { if (isFinite(v)) { xmin = Math.min(xmin, v); xmax = Math.max(xmax, v); } });
        [sy.v, anyCI() ? sy.lo : sy.v, anyCI() ? sy.hi : sy.v].forEach(function (v) { if (isFinite(v)) { ymin = Math.min(ymin, v); ymax = Math.max(ymax, v); } });
      });
      if (pts.length) { series.push({ key: m.key, label: m.label + (m.local ? " (local)" : ""), color: colorOf(m), dash: !!m.dash, pts: pts }); }
    });
    if (title == null) { title = xm.short + " vs " + ym.short; }
    if (pending && !series.length) { return frontSVG({ title: title, aria: aria, series: [], empty: "loading the distributions\u2026" }); }
    if (!series.length) { return frontSVG({ title: title, aria: aria, series: [], empty: "no budget has finished for this selection yet" }); }
    var xpad = (xmax - xmin) * 0.06 || 0.3, ypad = (ymax - ymin) * 0.06 || 0.3;
    if (nearRange(xmin, xmax, 0, 0.35)) { xmin = Math.min(xmin, 0); xmax = Math.max(xmax, 0); }   // the problem's own length, when it is near
    xmin -= xpad; xmax += xpad; ymin -= ypad; ymax += ypad;
    return frontSVG({ title: title, aria: aria, series: series, cycle: !!(legend && legend.cycle), ghosts: legend && legend.ghosts, xmin: xmin, xmax: xmax, ymin: ymin, ymax: ymax,
      xticks: ticksFor(xm, xmin, xmax), xtick: function (g) { return tickLabel(xm, g); },
      yticks: ticksFor(ym, ymin, ymax), ytick: function (g) { return tickLabel(ym, g); },
      xlabel: xlabel || axisName(xm), ylabel: ylabel == null ? axisName(ym) : ylabel, xzero: 0 });
  }
  function xOf(m, r, cs, src) { return state.xaxis === "time" ? timeOf(m, r, cs, src) : r; }
  function hasCandidateBudget(m) { return (m.budget || "candidates") === "candidates"; }   // PySR's budget is search iterations
  function onAxis(m) { return state.xaxis === "time" ? hasRefTime(m.key) : hasCandidateBudget(m); }
  function axisMethods(shown) { return shown.filter(onAxis); }
  function offAxis(shown) { return shown.filter(function (m) { return !onAxis(m); }); }
  function anyTime() { return Object.keys(D.timing).some(hasRefTime); }   // a reference row, or no time axis
  function timeRange(tmin, tmax) {   // what is drawn plus a margin, not whole decades: a decade of nothing is a third of a chart
    var lo = Math.log10(tmin), hi = Math.log10(tmax); if (!(hi - lo > 0.3)) { var mid = (lo + hi) / 2; lo = mid - 0.15; hi = mid + 0.15; }
    var pad = 0.05 * (hi - lo) + 0.04; return [Math.pow(10, lo - pad), Math.pow(10, hi + pad)];
  }

  // ---- Curves ----------------------------------------------------------------------------------------------------
  var thinDrawn = false;   // set by a chart that drew a hollow point, so the block around it can say what that means
  function thinNote() { return "A hollow point is based on fewer than " + state.valid + " % of the problems"; }
  // A method that cannot run a larger budget -- its defaults, a resource limit, a bug: the release says why
  // (summary.ladder, "declared") -- ends its curve with a square. A ladder that reached the end of the time axis does not.
  var endDrawn = false;   // set by a chart that drew a square, so the block around it can say what that means
  function lastRung(key) {
    var c = D.cells[key] || {}, mx = 0;
    Object.keys(c).forEach(function (cat) { Object.keys(c[cat]).forEach(function (r) { mx = Math.max(mx, +r); }); });
    return mx;
  }
  function ladderEnd(key) { var e = D.summary && D.summary.ladder && D.summary.ladder[key]; return e && e.end === "declared" ? e : null; }
  function endsAt(key, r) { return !!ladderEnd(key) && r === lastRung(key); }
  function withEnd(key, r, title) { return endsAt(key, r) ? title.replace(/\.?$/, ". ") + "Its largest budget: " + ladderEnd(key).reason : title; }
  function endNote() { return "A square marks the largest budget a method can run; its tooltip says why"; }
  function markerSVG(cx, cy, p, col) {
    var fill = p.hollow ? "var(--surface)" : col, tip = "<title>" + esc(p.title) + "</title>";
    if (p.end) {
      endDrawn = true;
      return '<rect x="' + (cx - 3).toFixed(1) + '" y="' + (cy - 3).toFixed(1) + '" width="6" height="6" fill="' + fill + '" stroke="' + col + '" stroke-width="1.5" class="v2end">' + tip + "</rect>";
    }
    return '<circle cx="' + cx.toFixed(1) + '" cy="' + cy.toFixed(1) + '" r="3.2" fill="' + fill + '" stroke="' + col + '" stroke-width="1.5">' + tip + "</circle>";
  }
  function curveChart(metric, shown, title, aria, legend) {
    var keys = shown.map(function (x) { return x.key; }), series = [], ymin = Infinity, ymax = -Infinity, pending = false, tmin = Infinity, tmax = -Infinity;
    var src = timeSource(keys);
    shown.forEach(function (m) { var pts = [];
      D.rungs.forEach(function (r) { var use = poolCats(m.key, r, keys); if (!use.length) { return; } var x = xOf(m.key, r, use, src); if (x === null) { return; }
        var st = stat(metric, m.key, r, use); if (!st) { return; } if (st.pending) { pending = true; return; } if (!isFinite(st.v)) { return; }
        var title = m.label + " at budget " + r + (state.xaxis === "time" ? " (" + x.toFixed(2) + " s per problem)" : "") + ": " + fmt(metric, st.v, st.edge) + " (95 % interval " + fmt(metric, st.lo) + " to " + fmt(metric, st.hi) + rangeText(metric, st) + "). " + problemsText(st, use) + (st.share < 1 ? "; " + shareText(st) : "") + ".";
        if (thin(st)) { thinDrawn = true; }
        pts.push({ x: x, v: st.v, lo: st.lo, hi: st.hi, hollow: thin(st), end: endsAt(m.key, r), title: withEnd(m.key, r, title) });   // a budget has no interval: the candidate count is exact, and the measured time is within a pixel of its mean (0.4-1.1 px, measured)
        [st.v, anyCI() ? st.lo : st.v, anyCI() ? st.hi : st.v].forEach(function (v) { if (isFinite(v)) { ymin = Math.min(ymin, v); ymax = Math.max(ymax, v); } });
        if (state.xaxis === "time") { tmin = Math.min(tmin, x); tmax = Math.max(tmax, x); } });
      if (pts.length) { series.push({ key: m.key, label: m.label + (m.local ? " (local)" : ""), color: colorOf(m), dash: !!m.dash, pts: pts }); } });
    if (title == null) { title = metric.label; }   // the statistic is named once per block, not on every chart
    var ylabel = axisName(metric) + (metric.median_via && state.stat === "mean" ? ", median" : "");
    if (pending && !series.length) { return chartSVG({ title: title, aria: aria, series: [], empty: "loading the distribution…" }); }
    if (!series.length) { return chartSVG({ title: title, aria: aria, series: [], empty: state.cats.length ? (shown.length ? (state.xaxis === "time" ? "none of the methods shown has been timed yet" : "no method has results for this selection yet") : "no method selected") : "no problem set selected" }); }
    var pad = (ymax - ymin) * 0.06 || 0.05;
    if (metric.kind === "rate") { var floor0 = nearRange(ymin, ymax, 0, 0.6); ymin = floor0 ? 0 : Math.max(0, ymin - pad); ymax = Math.min(1, Math.max(ymin + 0.02, ymax + pad)); }
    else { if (tfOf(metric) === "log2" && nearRange(ymin, ymax, 0, 0.35)) { ymin = Math.min(ymin, 0); ymax = Math.max(ymax, 0); } ymin -= pad; ymax += pad; }
    var tr = state.xaxis === "time" ? timeRange(tmin, tmax) : [0, 0];
    return chartSVG({ title: title, aria: aria, series: series, cycle: !!(legend && legend.cycle), ghosts: legend && legend.ghosts, ymin: ymin, ymax: ymax, ticks: ticksFor(metric, ymin, ymax), tick: function (g) { return tickLabel(metric, g); }, ylabel: ylabel, timeAxis: state.xaxis === "time", timeSource: src, tmin: tr[0], tmax: tr[1], zero: tfOf(metric) === "log2" ? 0 : undefined });
  }
  // ---- Curves: one card per plot, each carrying its own two axes -------------------------------------------------
  // A picker, not a select: forty metrics in one column is a scroll, in grouped columns it is a menu. One control
  // serves every place a metric is chosen, and every entry carries the same name and the same definition.
  function pickButton(cls, attrs, k) {
    var m = axisOf(k);
    return '<button type="button" class="v2pick ' + cls + '" ' + attrs + ' data-k="' + esc(k) + '" aria-haspopup="dialog" aria-expanded="false" title="' + esc(mdef(m)) + '">' +
      '<span class="v2picklab">' + esc(mname(m)) + '</span><span class="v2caret" aria-hidden="true">\u25be</span></button>';
  }
  function pickerGroups(axis) {
    var gs = axis === "x" ? [{ title: "budget spent", items: [AXIS.time, AXIS.rung] }] : [];
    var only = axis === "focus" && state.view === "ranks" ? (D.rank_keys || []) : null;   // the metrics the release ranked on
    var perProblem = axis === "cx" || axis === "cy";   // the Correlations view: the continuous metrics, whose values per problem ship
    MGROUPS.forEach(function (g) {
      var ms = D.metrics.filter(function (m) { return m.group === g && (!only || only.indexOf(m.key) >= 0) && (!perProblem || m.pp); });
      if (ms.length) { gs.push({ title: g, items: ms }); }
    });
    return gs;
  }
  function pickerHTML(axis, cur) {
    var cols = pickerGroups(axis).map(function (g) {
      return '<div class="v2pickgroup"><h5>' + esc(g.title) + "</h5>" + g.items.map(function (m) {
        var off = m.key === "time" && !anyTime();
        return '<button type="button" class="v2pickitem' + (m.key === cur ? " on" : "") + '" data-k="' + esc(m.key) + '" data-alt="' + esc((m.short || "").toLowerCase()) + '"' + (off ? " disabled" : "") +
          ' title="' + esc(mdef(m)) + '"><span>' + esc(mname(m)) + "</span>" + (off ? ' <span class="v2hint">no reference timing yet</span>' : "") + "</button>";
      }).join("") + "</div>";
    }).join("");
    return '<input type="search" class="v2pickq" placeholder="filter" aria-label="filter the list"><div class="v2pickcols">' + cols + "</div>";
  }
  function setPick(el, k) {
    if (!el) { return; }
    var m = axisOf(k);
    el.dataset.k = k; el.title = mdef(m); el.querySelector(".v2picklab").textContent = mname(m);
  }
  function plotCard(p, i, shown) {
    var ym = METRIC[p.y], svg;
    if (isBudgetAxis(p.x)) { svg = withState({ xaxis: p.x }, function () { return curveChart(ym, axisMethods(shown), "", mname(ym) + " against " + mname(AXIS[p.x])); }); }
    else { svg = frontChart(METRIC[p.x], ym, shown, "", mname(ym) + " against " + mname(METRIC[p.x])); }
    return '<figure class="v2plot">' +
      '<div class="v2plothead">' + pickButton("v2ysel", 'data-i="' + i + '" data-axis="y" aria-label="metric on the y axis"', p.y) +
      // two metrics can trade places; a budget axis cannot become the y axis, so that plot keeps its plain "vs"
      (isBudgetAxis(p.x) ? '<span class="v2vs">vs</span>'
        : '<button type="button" class="v2swap" data-i="' + i + '" aria-label="Swap the two axes" title="Swap the two axes">\u21c6</button>') +
      pickButton("v2xsel", 'data-i="' + i + '" data-axis="x" aria-label="axis on the x axis"', p.x) +
      '<button type="button" class="v2rmplot" data-i="' + i + '" aria-label="Remove this plot" title="Remove this plot">\u00d7</button></div>' +
      svg + "</figure>";
  }
  function renderCurves(shown) {
    var main = root.querySelector(".v2main");
    var add = '<button type="button" class="v2addplot" data-act="add-plot" aria-label="Add a plot"><span class="v2plus" aria-hidden="true">+</span><span>add plot</span></button>';
    if (!state.plots.length) { return '<div class="v2charts">' + add + '</div><p class="v2hint">No plot yet: add one, then choose what goes on each axis.</p>'; }
    // Each budget axis leaves some methods without a position. They are named here rather than dropped in silence.
    var note = ["rung", "time"].map(function (ax) {
      if (!state.plots.some(function (p) { return p.x === ax; })) { return ""; }
      var off = withState({ xaxis: ax }, function () { return offAxis(shown); });
      if (!off.length) { return ""; }
      var names = esc(off.map(function (m) { return m.label; }).join(", ")), many = off.length > 1;
      return '<p class="v2hint">' + names + (ax === "rung"
        ? (many ? " count their budget in " : " counts its budget in ") + (off.every(function (m) { return m.budget === "iterations"; }) ? "search iterations" : "a unit of " + (many ? "their" : "its") + " own") + ", not in " + term("candidates", "candidates") + ", so " + (many ? "they are" : "it is") + " not drawn on this axis.</p>"
        : (many ? " have" : " has") + " not been " + term("time", "timed") + " yet, so " + (many ? "they are" : "it is") + " not drawn on the time axis.</p>");
    }).join("");
    thinDrawn = false; endDrawn = false;
    var cards = inBlock(main, state.plots.length + 1, function () {
      return state.plots.map(function (p, i) { return plotCard(p, i, shown); }).join("");
    });
    if (thinDrawn) { note += '<p class="v2hint v2hollownote">' + term("valid", thinNote()) + ": on the others this metric has no value, because the method returned no usable formula or the value could not be computed.</p>"; }
    if (endDrawn) { note += '<p class="v2hint v2endnote">' + endNote() + ".</p>"; }
    return note + '<div class="v2charts">' + cards + add + "</div>";
  }

  // ---- Headline ---------------------------------------------------------------------------------------------------
  // Two charts that are the same for every visitor: what a method recovers, and how long its answer is, against
  // what it costs. They are deliberately not wired to the controls below -- everything adjustable is the explorer.
  var HEADLINE = [
    { key: "numeric_recovery_val", title: function () { return anyTime() ? "Recovery vs time" : "Recovery vs budget"; },
      caption: "Numeric Recovery: the share of problems where the method's formula reproduces the held-out points (the validation set, which the method never saw) almost exactly: its typical (root-mean-square) error is at most 0.035 % of the true values' standard deviation. Up and to the left is better: more problems solved in less time." },
    { x: "mdl_ratio", y: "log10_fvu_val", title: "Fit vs length",
      caption: "Down is a better fit: log10 FVU is the share of the variation in the held-out points that the formula leaves unexplained, on a log scale (-2 means 1 %). Across is the formula's length: the MDL Ratio divides its length in bits (its minimum description length, MDL) by the true formula's, so the dashed line at 1 is the true length, left of it shorter and right of it longer. Best is low and close to the line. Both are averaged over the problems where the method's formula has a value; an exact fit counts at -15.65, the precision of a 64-bit float." }];
  function hlText(v) { return typeof v === "function" ? v() : v; }
  function withState(over, fn) { var prev = state; state = Object.assign({}, prev, over); try { return fn(); } finally { state = prev; } }
  function renderHeadline() {
    if (!headRoot) { return; }
    var ghosts = D.methods.filter(function (m) { return withData(m) && hlVisOf(m.key) === "hidden"; });
    var changed = Object.keys(hlVis).some(function (k) { return hlVis[k] !== hlDefault(k); });
    HL_DRAWING = true;
    try {
    headRoot.innerHTML = inBlock(headRoot, HEADLINE.length, function () { return withState(
      { cats: CATS.slice(), methods: D.methods.filter(function (m) { return withData(m) && hlVisOf(m.key) !== "hidden"; }).map(function (m) { return m.key; }),
        stat: "mean", band: true, cross: false, xaxis: anyTime() ? "time" : "rung", valid: VALID_DEFAULT, impute: true },
      function () {
        var shown = shownMethods();
        if (!shown.length) { return ghosts.length ? '<p class="v2hint">Every method is hidden. <button type="button" class="v2btn" data-hlreset="1">show the default again</button></p>' : '<p class="v2hint">No method has finished a budget in this release yet.</p>'; }
        var drawn = axisMethods(shown), off = offAxis(shown);
        var keys = drawn.map(function (m) { return m.key; }), src = timeSource(keys);
        thinDrawn = false; endDrawn = false;
        var charts = HEADLINE.map(function (h) {
          var ms = (h.key ? [h.key] : [h.x, h.y]).map(function (k) { return METRIC[k]; });
          if (ms.some(function (m) { return !m; })) { return ""; }
          ms.forEach(function (m) { if (needsHist(m)) { ensureHists(m); } });
          var svg = h.key ? curveChart(ms[0], drawn, hlText(h.title), undefined, { cycle: true, ghosts: ghosts.filter(onAxis) }) : frontChart(ms[0], ms[1], shown, hlText(h.title), undefined, undefined, undefined, { cycle: true, ghosts: ghosts });
          return '<figure class="v2hlfig">' + svg + "<figcaption>" + esc(hlText(h.caption)) + "</figcaption></figure>";
        }).join("");
        return '<h2 class="v2hltitle">Accuracy, cost and formula length</h2>' +
          '<p class="v2hlsub">Each point is one method at one ' + term("rungs", "budget") + ': the ' + (state.stat === "mean" ? "mean" : "median") + ' over ' + laws(CATS).toLocaleString() + ' problems from ' + CATS.length + ' problem sets, ' + term("average", "with the problem sets weighted about equally") + ', and a ' + term("interval", "95 % interval") + '. Every method is run twice on every problem; a problem\u2019s value is the average of its finished runs. ' + term("complete", "Why do some methods have fewer points?") + '</p>' +
          '<div class="v2hlcharts">' + charts + "</div>" +
          '<p class="v2hint v2hlvis">A click on a name in a legend cycles that method: shown, hidden, faded.' + (changed ? ' <button type="button" class="v2btn" data-hlreset="1">show the default again</button>' : "") + "</p>" +
          '<p class="v2hint">' + (src === "ref" ? term("time", "Time is measured on one workstation for every method") +
              (off.length ? ". " + esc(off.map(function (m) { return m.label; }).join(", ")) + (off.length > 1 ? " have" : " has") + " not been timed yet, so " +
                (off.length > 1 ? "they are" : "it is") + " not in the first chart" : "")
            : "The x-axis is the budget per problem. " + term("time", "A time axis appears once a method shown has been timed") +
              ": seconds measured wherever a unit happened to run are not comparable between methods, so this release does not publish them") + "." +
            (thinDrawn ? " " + term("valid", thinNote()) + "." : "") + (endDrawn ? " " + endNote() + "." : "") + "</p>";
      }); });
    } finally { HL_DRAWING = false; }
  }
  if (headRoot) {   // the headline's legend: a click or Enter on a name cycles that method for this visit
    // The headline is drawn again with the new state. The entry that was clicked stays where it was on the screen, and
    // the focus goes to the same entry of the same chart: focusing another chart's entry scrolled the page to that
    // chart (on a phone, where the charts are stacked, to the top).
    var cycleHeadline = function (g) {
      var k = g.getAttribute("data-cycle"), charts = Array.prototype.slice.call(headRoot.querySelectorAll("svg.v2chart"));
      var at = charts.indexOf(g.closest("svg")), top = g.getBoundingClientRect().top;
      hlVis[k] = nextVis(hlVisOf(k)); renderHeadline();
      var chart = headRoot.querySelectorAll("svg.v2chart")[at], sel = '[data-cycle="' + k + '"]';
      var again = (chart && chart.querySelector(sel)) || headRoot.querySelector(sel);
      if (!again) { return; }
      window.scrollBy(0, again.getBoundingClientRect().top - top);
      if (again.focus) { try { again.focus({ preventScroll: true }); } catch (err) { again.focus(); } }
    };
    headRoot.addEventListener("click", function (e) {
      var g = e.target.closest ? e.target.closest("[data-cycle]") : null; if (g) { cycleHeadline(g); return; }
      if (e.target.closest && e.target.closest("[data-hlreset]")) { hlVis = {}; renderHeadline(); }
    });
    headRoot.addEventListener("keydown", function (e) {
      if (e.key !== "Enter" && e.key !== " ") { return; }
      var g = e.target.closest ? e.target.closest("[data-cycle]") : null; if (g) { e.preventDefault(); cycleHeadline(g); }
    });
  }

  // ---- Table -----------------------------------------------------------------------------------------------------
  var lastTable = null;
  var rangeShown = false, tweenShown = false;   // tweenShown: a table cell held a number between two budgets
  function cellText(st, p) { if (!st) { return ""; } if (st.pending) { return "…"; } if (thin(st)) { thinDrawn = true; } var pi = anyCI() && st.piLo !== null && st.piLo !== undefined; if (pi) { rangeShown = true; }
    if (st.between) { tweenShown = true; }
    return (st.between ? '<span class="v2tween" title="' + esc(tweenText(st.between)) + '">\u2248</span>' : "") + (thin(st) ? '<span title="' + esc(shareText(st)) + '">' + THIN_MARK + "</span>" : "") + fmt(p, st.v, st.edge) + (anyCI() ? ' <span class="v2ci-txt">[' + fmt(p, st.lo) + ", " + fmt(p, st.hi) + "]</span>" : "") + (pi ? '<br><span class="v2ci-txt">one set: ' + fmt(p, st.piLo) + " to " + fmt(p, st.piHi) + "</span>" : ""); }
  function renderTable(shown) {
    var plots = plotMetrics().map(function (k) { return METRIC[k]; }); var keys = shown.map(function (m) { return m.key; });
    if (!plots.length || !shown.length) { return '<p class="v2hint">Select at least one method and one metric.</p>'; }
    var head = '<tr><th>' + (state.rows === "cats" ? "problem set" : "budget") + '</th><th>problems</th>' + shown.map(function (m) { return '<th colspan="' + plots.length + '"><span class="v2sw" style="background:' + colorOf(m) + '"></span>' + esc(m.label) + '</th>'; }).join("") + '</tr>' +
      '<tr><th></th><th></th>' + shown.map(function () { return plots.map(function (p) { return '<th>' + esc(mname(p)) + " " + mhelp(p) + '</th>'; }).join(""); }).join("") + '</tr>';
    var body = "", rowsOut = []; thinDrawn = false; rangeShown = false; tweenShown = false;
    var emit = function (label, nl, tds) { body += '<tr><td>' + esc(label) + '</td><td>' + esc(nl) + '</td>' + tds.map(function (x) { return '<td>' + x.t + '</td>'; }).join("") + '</tr>'; rowsOut.push([label, nl].concat(tds.map(function (x) { return x.raw; }))); };
    if (state.rows === "rungs") {
      D.rungs.forEach(function (r) { var cells = shown.map(function (m) { return { m: m, use: poolCats(m.key, r, keys) }; }); if (!cells.some(function (c) { return c.use.length; })) { return; }
        var ref = cells.filter(function (c) { return c.use.length; })[0]; var nl = laws(ref.use).toLocaleString() + " (" + ref.use.length + " problem sets)";
        var tds = []; cells.forEach(function (c) { plots.forEach(function (p) { if (!c.use.length) { tds.push({ t: "", raw: "" }); return; } var st = stat(p, c.m.key, r, c.use); tds.push({ t: cellText(st, p), raw: st && !st.pending ? fmt(p, st.v, st.edge) : "" }); }); });
        emit(String(r), nl, tds); });
    } else {
      var r = curPos();
      state.cats.slice().sort(function (a, b) { return CAT[b].laws - CAT[a].laws; }).forEach(function (c) { if (!shown.some(function (m) { return cell(m.key, c, r); })) { return; }
        var tds = []; shown.forEach(function (m) { plots.forEach(function (p) { if (!cell(m.key, c, r)) { tds.push({ t: "", raw: "" }); return; } var st = stat(p, m.key, r, [c]); tds.push({ t: cellText(st, p), raw: st && !st.pending ? fmt(p, st.v, st.edge) : "" }); }); });
        emit(c, String(CAT[c].laws), tds); });
      var tds2 = []; shown.forEach(function (m) { plots.forEach(function (p) { var use = poolCats(m.key, r); var st = use.length ? stat(p, m.key, r, use) : null; tds2.push({ t: cellText(st, p), raw: st && !st.pending ? fmt(p, st.v, st.edge) : "" }); }); });
      body += '<tr class="v2total"><td>all selected</td><td>' + laws(state.cats) + '</td>' + tds2.map(function (x) { return '<td>' + x.t + '</td>'; }).join("") + '</tr>';
      rowsOut.push(["all selected", String(laws(state.cats))].concat(tds2.map(function (x) { return x.raw; })));
    }
    lastTable = { header: [state.rows === "cats" ? "problem set" : "budget", "problems"].concat(shown.reduce(function (a, m) { return a.concat(plots.map(function (p) { return m.label + " · " + mname(p); })); }, [])), rows: rowsOut };
    var ctl = '<div class="v2row v2tablectl"><span class="v2lab">rows</span><label><input type="radio" name="v2rows" value="rungs"' + (state.rows === "rungs" ? " checked" : "") + '> every budget</label><label><input type="radio" name="v2rows" value="cats"' + (state.rows === "cats" ? " checked" : "") + '> problem sets at one budget or time</label>' + (state.rows === "cats" ? posControl(shown) : "") +
      '<span class="v2spacer"></span><button type="button" class="v2btn" data-act="copy-tsv">copy as TSV</button><button type="button" class="v2btn" data-act="csv">download CSV</button></div>';
    return ctl + '<div class="v2table-wrap"><table class="v2table"><thead>' + head + '</thead><tbody>' + body + '</tbody></table></div>' + (state.rows === "cats" ? tweenNote(shown, curPos()) : "") + '<p class="v2hint">' + (anyCI() ? "Brackets: the " + term("interval", "95 % interval") + ". " + (rangeShown ? "One set: " + term("setrange", "where one more problem set would fall") + ". " : "") : "") + term("regime", "How problems without a usable formula count") + ". " + (thinDrawn ? term("valid", "\u25cb marks a number based on fewer than " + state.valid + " % of the problems") + ". " : "") + term("complete", "A value appears once a method has results for all selected problem sets at that budget") + ".</p>";
  }

  // ---- Catalog matrix --------------------------------------------------------------------------------------------
  function accentRGB() { var v = (getComputedStyle(document.documentElement).getPropertyValue("--accent") || "#4f46e5").trim(); var m = v.match(/^#([0-9a-f]{6})$/i); if (!m) { return [79, 70, 229]; } return [parseInt(m[1].slice(0, 2), 16), parseInt(m[1].slice(2, 4), 16), parseInt(m[1].slice(4, 6), 16)]; }
  function renderMatrix(shown) {
    var p = METRIC[state.focus], r = curPos(), rgb = accentRGB();
    if (!shown.length) { return '<p class="v2hint">Select at least one method.</p>'; }
    var bar = '<div class="v2viewbar"><span class="v2segwrap"><span class="v2lab">metric</span>' + pickButton("v2viewpick", 'data-axis="focus" aria-label="metric shown in the matrix"', p.key) + " " + mhelp(p) + "</span>" + posControl(shown) + "</div>";
    var asked = shown; shown = withAt(shown, r);
    var cats = state.cats.slice().sort(function (a, b) { return CAT[b].laws - CAT[a].laws; }).filter(function (c) { return shown.some(function (m) { return cell(m.key, c, r); }); });
    if (!cats.length) { return bar + '<p class="v2hint">No selected method has a value at ' + esc(posText(r)) + " for this selection: move the budget or time above.</p>"; }
    var vals = {}, all = [], pending = false; thinDrawn = false; tweenShown = false;
    cats.forEach(function (c) { vals[c] = {}; shown.forEach(function (m) { if (!cell(m.key, c, r)) { return; } var st = stat(p, m.key, r, [c]); if (st && st.pending) { pending = true; return; } if (st && isFinite(st.v)) { vals[c][m.key] = st; all.push(st.v); } }); });
    if (pending && !all.length) { return bar + '<p class="v2hint">Loading the distribution…</p>'; }
    var lo = Math.min.apply(null, all), hi = Math.max.apply(null, all), ideal = p.ideal === undefined ? (tfOf(p) === "log2" ? 0 : 1) : tfOf(p) === "log2" ? Math.log2(p.ideal) : p.ideal;
    var score = function (v) { if (!(hi > lo)) { return 0.5; } if (p.higher === null) { var dm = Math.max(Math.abs(lo - ideal), Math.abs(hi - ideal)); return dm ? 1 - Math.abs(v - ideal) / dm : 1; } var t = (v - lo) / (hi - lo); return p.higher ? t : 1 - t; };
    var h = '<div class="v2table-wrap"><table class="v2table v2matrix"><thead><tr><th>problem set</th><th>problems</th>' + shown.map(function (m) { return '<th><span class="v2sw" style="background:' + colorOf(m) + '"></span>' + esc(m.label) + '</th>'; }).join("") + '</tr></thead><tbody>';
    cats.forEach(function (c) { h += '<tr><td>' + esc(c) + ' <span class="v2hint">' + GROUPS[CAT[c].group] + '</span></td><td>' + CAT[c].laws + '</td>' + shown.map(function (m) { var st = vals[c][m.key]; if (!st) { return '<td class="v2na">' + (cell(m.key, c, r) ? "…" : "") + '</td>'; } var a = 0.06 + 0.5 * score(st.v); return '<td style="background:rgba(' + rgb.join(",") + "," + a.toFixed(2) + ')" title="' + esc("95 % interval " + fmt(p, st.lo) + " to " + fmt(p, st.hi) + ", from " + Math.round(st.n).toLocaleString() + " problems" + (st.share < 1 ? ", " + shareText(st) : "")) + '">' + (thin(st) ? (thinDrawn = true, THIN_MARK) : "") + fmt(p, st.v, st.edge) + '</td>'; }).join("") + '</tr>'; });
    var pooled = shown.map(function (m) { var use = poolCats(m.key, r); var st = use.length ? stat(p, m.key, r, use) : null; return '<td>' + (st && !st.pending ? cellText(st, p) : "") + '</td>'; }).join("");
    h += '<tr class="v2total"><td>all selected ' + help(TERMS.complete, "When is this row filled in?") + '</td><td>' + laws(state.cats).toLocaleString() + '</td>' + pooled + '</tr></tbody></table></div>';
    var thinHint = thinDrawn ? '<p class="v2hint v2hollownote">' + term("valid", "\u25cb marks a number based on fewer than " + state.valid + " % of the problems") + ".</p>" : "";
    return bar + missingNote(asked, shown, "any selected problem set at " + posText(r), r, true) + tweenNote(shown, r) + thinHint + '<p class="v2hint">' + esc(p.label) + " at " + posText(r) + ", one square per problem set; darker is better" + (p.higher === null && p.ideal !== undefined ? " (closer to " + p.ideal + ")" : "") + ". " + (p.kind === "cont" ? (statOf(p) === "mean" ? term("mean", "Means") : term("median", "Medians")) + (p.worst !== undefined && state.impute ? " over all problems; a problem without a usable formula counts as " + p.worst + "." : " over the problems where the method returned a usable formula.") : term("regime", "Every problem counts; a problem without a usable formula is a miss") + ".") + "</p>" + h;
  }

  // ---- controls that live on the display itself -----------------------------------------------------------------
  // The sidebar holds what every display shares. What only ONE display can use -- which budget a snapshot is taken
  // at, how a distribution is drawn -- sits on that display, next to what it changes.
  // opts: [value, label, tooltip?, short label?]; a short label takes the label's place on a narrow screen (styles.css)
  function seg(key, cur, opts, labelHtml, aria) {
    return '<span class="v2segwrap">' + (labelHtml ? '<span class="v2lab">' + labelHtml + "</span>" : "") + '<span class="v2seg" role="group" aria-label="' + esc(aria) + '">' +
      opts.map(function (o) { return '<button type="button" class="v2segbtn' + (o[0] === cur ? " on" : "") + '" data-set="' + key + ":" + o[0] + '" aria-pressed="' + (o[0] === cur ? "true" : "false") + '"' + (o[2] ? ' title="' + esc(o[2]) + '"' : "") + ">" + (o[3] ? '<span class="v2long">' + esc(o[1]) + '</span><span class="v2short">' + esc(o[3]) + "</span>" : esc(o[1])) + "</button>"; }).join("") + "</span></span>";
  }
  function stepper(key, cur, values, labelOf, labelHtml, aria) {   // values in order; cur may sit between them
    var i = values.indexOf(cur), prev = null, next = null;
    if (i >= 0) { prev = i > 0 ? values[i - 1] : null; next = i < values.length - 1 ? values[i + 1] : null; }
    else { values.forEach(function (v) { if (v < cur) { prev = v; } else if (v > cur && next === null) { next = v; } }); }
    var all = i >= 0 ? values : values.concat([cur]);
    var btn = function (to, glyph, word) { return '<button type="button" class="v2stepbtn" ' + (to === null ? "disabled" : 'data-set="' + key + ":" + to + '"') + ' aria-label="' + esc(word + " " + aria) + '">' + glyph + "</button>"; };
    return '<span class="v2segwrap"><span class="v2lab">' + labelHtml + '</span><span class="v2step">' + btn(prev, "◀", "lower") +
      '<select class="v2stepsel" data-state="' + key + '" aria-label="' + esc(aria) + '">' + all.map(function (v) { return '<option value="' + v + '"' + (v === cur ? " selected" : "") + ">" + esc(labelOf(v) + (values.indexOf(v) < 0 ? " (nothing finished)" : "")) + "</option>"; }).join("") + "</select>" +
      btn(next, "▶", "higher") + "</span></span>";
  }
  // the budgets a stepper offers: where a selected method has finished a selected catalog, or -- for a display that
  // pools -- every selected catalog
  function rungsWith(shown, whole) { return D.rungs.filter(function (r) { return shown.some(function (m) { return whole ? poolCats(m.key, r).length : state.cats.some(function (c) { return cell(m.key, c, r); }); }); }); }
  function nearestOf(values, x) { var best = null; values.forEach(function (v) { if (best === null || Math.abs(Math.log(v / x)) < Math.abs(Math.log(best / x))) { best = v; } }); return best === null ? x : best; }
  function rungStepper(shown, whole, cur) { return stepper("rung", cur !== undefined ? cur : state.rung, rungsWith(shown, whole), function (r) { return String(r); }, "budget " + help(TERMS.rungs, "What is a budget?"), "budget per problem"); }
  // The position every display that reads one position reads every method at (owner 2026-10-05: every display but
  // Curves): a time per problem on our timing workstation -- the default, since time is the one budget every method
  // shares -- or a budget in each method's own unit. The slider runs over the logarithm of the range the shown methods
  // cover and is continuous, and a number typed into its box is taken as it is: neither is rounded to a mark. The
  // arrows -- the buttons beside it, and the arrow keys on the slider -- step to the next of the release's time limits
  // (TIME_BUDGETS), or to the next budget a shown method was run at.
  function posRange(shown, time) {
    var lo = Infinity, hi = -Infinity;
    shown.forEach(function (m) { state.cats.forEach(function (c) { points(m.key, c, time).forEach(function (pt) { if (pt[0] < lo) { lo = pt[0]; } if (pt[0] > hi) { hi = pt[0]; } }); }); });
    return isFinite(lo) ? (hi > lo ? [lo, hi] : [lo / 2, hi * 2]) : null;
  }
  function posMarks(shown, rg, time) {
    if (time) { return (D.time_budgets || TIME_BUDGETS).filter(function (t) { return t >= rg[0] * (1 - 1e-9) && t <= rg[1] * (1 + 1e-9); }); }
    var seen = {};
    shown.forEach(function (m) { state.cats.forEach(function (c) { points(m.key, c, false).forEach(function (pt) { seen[pt[1]] = 1; }); }); });
    return Object.keys(seen).map(Number).sort(function (a, b) { return a - b; });
  }
  function posLabel(x, time) { return time ? fmtSec(x) + " s" : fmtBudget(x); }
  // the number in the box: grouped the same way in every locale, so that it reads back as the number it shows
  function posNum(x, time) { return time ? (x >= 100 ? Math.round(x).toLocaleString("en-US") : fmtSec(x)) : x >= 10 ? Math.round(x).toLocaleString("en-US") : String(+x.toPrecision(3)); }
  function parsePos(text) { var x = parseFloat(String(text).replace(/[,\s]/g, "").replace(/s(ec(onds?)?)?$/i, "")); return isFinite(x) && x > 0 ? x : null; }
  function posControl(shown, why) {
    var time = state.pm === "time", rg = posRange(shown, time);
    var mode = anyTime() ? seg("pm", state.pm, [["time", "time", "Each method at the same time per problem on our timing workstation"], ["budget", "budget", "Each method at the same budget, in its own unit"]], "at", "read every method at a time or at a budget")
      : '<span class="v2segwrap"><span class="v2lab">at budget</span></span>';
    var tip = why || help(time ? TERMS.postime : TERMS.posbudget, time ? "How is a method read at a time?" : "How is a method read at a budget?");
    if (!rg) { return mode + tip + '<span class="v2hint">' + (time ? "No shown method has a measured time yet." : "No shown method has finished a selected problem set yet.") + "</span>"; }
    var x = time ? state.pt : state.rung;
    var marks = posMarks(shown, rg, time), prev = null, next = null, span = Math.log(rg[1] / rg[0]);
    marks.forEach(function (v) { if (v < x * (1 - 1e-9)) { prev = v; } else if (v > x * (1 + 1e-9) && next === null) { next = v; } });
    var at = function (v) { return Math.min(1, Math.max(0, Math.log(v / rg[0]) / span)); };
    var btn = function (to, glyph, word) { return '<button type="button" class="v2stepbtn" ' + (to === null ? "disabled" : 'data-set="pos:' + to + '"') + ' aria-label="' + word + '">' + glyph + "</button>"; };
    return mode + tip + '<span class="v2step v2posstep">' + btn(prev, "◀", time ? "shorter time" : "smaller budget") +
      '<span class="v2posrail"><input type="range" class="v2pos" min="0" max="1" step="any" value="' + at(x) + '" data-lo="' + rg[0] + '" data-hi="' + rg[1] + '"' +
      (prev !== null ? ' data-prev="' + prev + '"' : "") + (next !== null ? ' data-next="' + next + '"' : "") +
      ' aria-label="' + (time ? "time per problem" : "budget per problem") + '" aria-valuetext="' + esc(posLabel(x, time)) + '">' +
      marks.map(function (v) { return '<i style="left:calc(8px + (100% - 16px) * ' + at(v).toFixed(4) + ')"></i>'; }).join("") + "</span>" +
      '<span class="v2posbox"><input type="text" class="v2posval" inputmode="decimal" autocomplete="off" spellcheck="false" value="' + esc(posNum(x, time)) + '" aria-label="' + (time ? "seconds per problem: type a number" : "budget per problem: type a number") + '">' +
      (time ? '<span class="v2posunit" aria-hidden="true">s</span>' : "") + "</span>" + btn(next, "▶", time ? "longer time" : "larger budget") + "</span>";
  }
  // the mark at which the most shown methods have a value over every selected problem set; of several such marks, the
  // middle one (the upper of the two middle ones)
  function defaultPos(shown, rg, time) {
    var tops = [], top = -1, was = BETWEEN; BETWEEN = true;
    try { posMarks(shown, rg, time).forEach(function (v) { var n = shown.filter(function (m) { return poolCats(m.key, time ? "t" + v : v).length; }).length; if (n > top) { top = n; tops = []; } if (n === top) { tops.push(v); } }); } finally { BETWEEN = was; }
    return tops.length ? tops[Math.floor(tops.length / 2)] : +Math.sqrt(rg[0] * rg[1]).toPrecision(3);
  }
  // a position inside the range the shown methods cover: kept where it is, moved to the nearer end when it lies outside,
  // and chosen afresh when there is none yet
  function settlePos(shown) {
    var time = state.pm === "time", rg = posRange(shown, time); if (!rg) { return; }
    var x = time ? state.pt : state.rung;
    if (!(x > 0)) { x = defaultPos(shown, rg, time); } else if (x < rg[0] * (1 - 1e-9)) { x = rg[0]; } else if (x > rg[1] * (1 + 1e-9)) { x = rg[1]; }
    if (time) { state.pt = x; } else { state.rung = x; }
  }
  function posFromSlider(t) { var lo = +t.dataset.lo, hi = +t.dataset.hi, f = Math.min(1, Math.max(0, +t.value)); return +(lo * Math.pow(hi / lo, f)).toPrecision(6); }
  // which shown methods sit between two of their budgets at this position, and between which
  function tweenText(between) { return "interpolated between budgets " + fmtBudget(between[0]) + " and " + fmtBudget(between[1]); }
  function tweenNote(shown, r) {
    var parts = [];
    shown.forEach(function (m) { var x = null; state.cats.some(function (c) { x = cell(m.key, c, r); return !!x; }); if (x && x.between) { parts.push(esc(m.label) + " between budgets " + fmtBudget(x.between[0]) + " and " + fmtBudget(x.between[1])); } });
    return parts.length ? '<p class="v2hint"><span class="v2tween">≈</span> ' + term("between", "Interpolated") + ": " + parts.join("; ") + ".</p>" : "";
  }
  function bestRung(shown) {   // the largest budget that the most methods have finished
    var best = null, top = 0;
    D.rungs.forEach(function (r) { var n = shown.filter(function (m) { return poolCats(m.key, r).length; }).length; if (n && n >= top) { top = n; best = r; } });
    return best;
  }
  function withAt(shown, r) { return shown.filter(function (m) { return state.cats.some(function (c) { return cell(m.key, c, r); }); }); }
  // a method missing at budget r either never runs there (outside the budgets its plan holds) or has not finished yet
  function notRun(m, r) { return !!(m.budgets && r !== undefined && m.budgets.indexOf(+r) < 0); }
  function budgetRange(m) { return m.budgets.length > 1 ? "budgets " + fmtBudget(m.budgets[0]) + " to " + fmtBudget(m.budgets[m.budgets.length - 1]) : "budget " + fmtBudget(m.budgets[0]); }
  // at a position (a budget or a time anywhere on the slider): a method has no value outside the budgets it runs at,
  // or, by time, outside the times its finished budgets took
  function timesOf(m) { return (m.budgets || []).map(function (b) { return refTime(m.key, b); }).filter(function (t) { return t > 0; }); }
  function outside(m, r) {
    var x = posValue(r);
    if (isTimePos(r)) { var ts = timesOf(m); return !ts.length || x < Math.min.apply(null, ts) * (1 - 1e-9) || x > Math.max.apply(null, ts) * (1 + 1e-9); }
    return !!(m.budgets && m.budgets.length && (x < m.budgets[0] * (1 - 1e-9) || x > m.budgets[m.budgets.length - 1] * (1 + 1e-9)));
  }
  function outsideText(m, r) {
    if (!isTimePos(r)) { return esc(m.label) + " has no value at " + esc(posText(r)) + " (it runs at " + budgetRange(m) + ")."; }
    var ts = timesOf(m);
    var lo = Math.min.apply(null, ts), hi = Math.max.apply(null, ts);
    return esc(m.label) + (!ts.length ? " has no time measured on our timing workstation yet." : " has no value at " + esc(posText(r)) +
      (hi > lo ? " (its budgets take " + fmtSec(lo) + " to " + fmtSec(hi) + " s per problem)." : " (its one timed budget takes " + fmtSec(lo) + " s per problem)."));
  }
  function missingNote(shown, present, what, r, atPos) {
    var gone = shown.filter(function (m) { return present.indexOf(m) < 0; }), off = atPos ? outside : notRun;
    var never = gone.filter(function (m) { return off(m, r); }), open = gone.filter(function (m) { return !off(m, r); });
    var names = function (ms) { return esc(ms.map(function (m) { return m.label; }).join(", ")); };
    return (never.length ? '<p class="v2hint">' + never.map(function (m) { return atPos ? outsideText(m, r) : esc(m.label) + " is not run at budget " + r + " (it runs at " + budgetRange(m) + ")."; }).join(" ") + "</p>" : "") +
      (open.length ? '<p class="v2hint">' + names(open) + (open.length > 1 ? " have" : " has") + " not finished " + what + ".</p>" : "");
  }

  // ---- Distribution ----------------------------------------------------------------------------------------------
  // Four readings of the same pooled histograms: one histogram per method, the cumulative curves of all of them on
  // one axis, a box per catalog, and the quartiles along the ladder. A rate has no distribution over laws (every
  // problem is a hit or a miss), so a rate shows how it is spread over the catalogs instead.
  var DMODES = [["hist", "histograms", "One histogram per method, on a shared axis"], ["ecdf", "cumulative", "The share of problems at or below each value, all methods on one axis"],
    ["cats", "by problem set", "A box per problem set and method: where the middle half and the middle 90 % of the problems lie", "by set"], ["rungs", "along the budgets", "The median and the middle half of the problems at every budget", "by budget"]];
  function qv(ph, q) { return binVal(ph, quantileBin(ph, Math.max(1, Math.ceil(q * ph.n)))); }
  function five(ph) { return [0.05, 0.25, 0.5, 0.75, 0.95].map(function (q) { return qv(ph, q); }); }
  function viewRange(phs) {   // the bins that hold 99 % of what is shown, so an empty tail does not squeeze the rest
    var lo = Infinity, hi = -Infinity;
    phs.forEach(function (ph) { lo = Math.min(lo, quantileBin(ph, Math.max(1, Math.ceil(0.005 * ph.n)))); hi = Math.max(hi, quantileBin(ph, Math.ceil(0.995 * ph.n))); });
    var nb = phs[0].nb, pad = Math.max(2, Math.round(0.04 * (hi - lo + 1)));
    lo = Math.max(0, lo - pad); hi = Math.min(nb - 1, hi + pad);
    if (hi - lo < 8) { lo = Math.max(0, lo - 4); hi = Math.min(nb - 1, hi + 4); }
    var w = (phs[0].hi - phs[0].lo) / nb;
    return { b0: lo, b1: hi, lo: phs[0].lo + lo * w, hi: phs[0].lo + (hi + 1) * w, w: w };
  }
  function distHead(shown, p) {
    var cont = p.kind !== "rate";
    return '<div class="v2viewbar">' + '<span class="v2segwrap"><span class="v2lab">metric</span>' + pickButton("v2viewpick", 'data-axis="focus" aria-label="metric whose distribution is shown"', p.key) + " " + mhelp(p) + "</span>" +
      (cont ? seg("dmode", state.dmode, DMODES, "show", "how the distribution is drawn") : "") +
      (cont && state.dmode === "rungs" ? "" : posControl(shown)) +
      (cont && state.dmode === "ecdf" ? seg("dnorm", state.dnorm, [["ok", "predicted problems", "100 % is every problem the method has a finite value for"], ["all", "all problems", "100 % is every problem in the pool: a problem without a prediction never enters the curve"]], "out of", "what the cumulative share is a share of") : "") +
      (cont && state.dmode === "rungs" ? seg("xaxis", state.xaxis, [["time", "time"], ["rung", "candidates"]].filter(function (o) { return o[0] !== "time" || anyTime(); }), "x axis", "what the budgets are drawn against") : "") + "</div>";
  }
  function boxSVG(f, xs, yc, h, col, title) {   // whiskers 5-95, box 25-75, median tick
    var x = f.map(function (v) { return xs(v).toFixed(1); });
    return '<g><title>' + esc(title) + '</title><line x1="' + x[0] + '" y1="' + yc + '" x2="' + x[4] + '" y2="' + yc + '" stroke="' + col + '" stroke-width="1.5" stroke-opacity="0.7"/>' +
      '<rect x="' + x[1] + '" y="' + (yc - h / 2).toFixed(1) + '" width="' + Math.max(1.5, xs(f[3]) - xs(f[1])).toFixed(1) + '" height="' + h + '" fill="' + col + '" fill-opacity="0.28" stroke="' + col + '" stroke-width="1.2"/>' +
      '<line x1="' + x[2] + '" y1="' + (yc - h / 2 - 2).toFixed(1) + '" x2="' + x[2] + '" y2="' + (yc + h / 2 + 2).toFixed(1) + '" stroke="' + col + '" stroke-width="2.4"/></g>';
  }
  function fiveText(p, f) { return "median " + fmt(p, f[2]) + "; middle 50 % of the problems " + fmt(p, f[1]) + " to " + fmt(p, f[3]) + "; middle 90 % " + fmt(p, f[0]) + " to " + fmt(p, f[4]); }
  function xAxisSVG(p, vr, xs, T, yb, W, L, R) {
    var s = "";
    ticksFor(p, vr.lo, vr.hi).forEach(function (g) { if (g < vr.lo - 1e-9 || g > vr.hi + 1e-9) { return; } s += '<line x1="' + xs(g).toFixed(1) + '" y1="' + T + '" x2="' + xs(g).toFixed(1) + '" y2="' + yb + '" class="grid"/><text x="' + xs(g).toFixed(1) + '" y="' + (yb + 16) + '" class="tick" text-anchor="middle">' + esc(tickLabel(p, g)) + "</text>"; });
    return s + '<text x="' + ((L + W - R) / 2).toFixed(0) + '" y="' + (yb + 33) + '" class="tick" text-anchor="middle">' + esc(axisName(p)) + "</text>";
  }
  function renderDistRate(shown, p, r) {   // per-catalog rates: a dot plot with each set's own interval
    var present = withAt(shown, r), nr = narrow(), W = wideWidth(), T = 46, R = 16;
    var cats = state.cats.slice().sort(function (a, b) { return CAT[b].laws - CAT[a].laws; }).filter(function (c) { return present.some(function (m) { return cell(m.key, c, r); }); });
    var swap = '<p class="v2hint">A rate is a hit or a miss on each run, so a histogram over problems would have only a few bars. This view shows how the rate varies between problem sets instead. ' + '<button type="button" class="v2btn" data-set="dmetric:log10_fvu_val">show the distribution of log10 FVU instead</button></p>';
    if (!cats.length) { return swap + '<p class="v2hint">No selected method has a value at ' + esc(posText(r)) + " for this selection: move the budget or time above.</p>"; }
    var rowH = Math.max(20, 7 * present.length + 8), B1 = 44 + 18 * present.length, Hh = T + cats.length * rowH + B1, Lw = nr ? 110 : 150;
    var s = '<svg viewBox="0 0 ' + W + " " + Hh + '" class="v2chart v2dist v2distwide" role="img" aria-label="' + esc(p.label) + ' per problem set"><text x="' + Lw + '" y="' + TITLE_Y + '" class="ct">' + esc(p.label) + " per problem set at " + posText(r) + "</text>";
    var xs = function (v) { return Lw + v * (W - Lw - R); };
    [0, 0.25, 0.5, 0.75, 1].forEach(function (g) { s += '<line x1="' + xs(g) + '" y1="' + T + '" x2="' + xs(g) + '" y2="' + (Hh - B1) + '" class="grid"/><text x="' + xs(g) + '" y="' + (Hh - B1 + 16) + '" class="tick" text-anchor="middle">' + (100 * g) + "%</text>"; });
    cats.forEach(function (c, i) { var yy = T + (i + 0.5) * rowH; s += '<text x="' + (Lw - 8) + '" y="' + (yy + 4) + '" class="tick" text-anchor="end">' + esc(c) + " · " + CAT[c].laws + "</text>";
      if (i % 2) { s += '<rect x="' + Lw + '" y="' + (yy - rowH / 2) + '" width="' + (W - Lw - R) + '" height="' + rowH + '" class="v2stripe"/>'; }
      present.forEach(function (m, j) { var st = cell(m.key, c, r) ? stat(p, m.key, r, [c]) : null; if (!st) { return; } var yj = yy + (j - (present.length - 1) / 2) * 7, col = colorOf(m);
        s += '<line x1="' + xs(st.lo).toFixed(1) + '" y1="' + yj.toFixed(1) + '" x2="' + xs(st.hi).toFixed(1) + '" y2="' + yj.toFixed(1) + '" stroke="' + col + '" stroke-width="2" stroke-opacity="0.4"/>';
        s += '<circle cx="' + xs(st.v).toFixed(1) + '" cy="' + yj.toFixed(1) + '" r="3.2" fill="' + col + '"><title>' + esc(m.label + " on " + c + ": " + fmt(p, st.v) + ", 95 % interval " + fmt(p, st.lo) + " to " + fmt(p, st.hi) + ", from " + Math.round(st.n).toLocaleString() + " problems") + "</title></circle>"; }); });
    var ly = Hh - B1 + 34; present.forEach(function (m) { s += '<circle cx="' + (Lw + 6) + '" cy="' + ly + '" r="4" fill="' + colorOf(m) + '"/><text x="' + (Lw + 16) + '" y="' + (ly + 4) + '" class="leg">' + esc(m.label + (m.local ? " (local)" : "")) + "</text>"; ly += 18; });
    return swap + s + "</svg>" + '<p class="v2hint">One row per problem set, largest first; one dot per method with its ' + term("interval", "95 % interval") + ".</p>" + missingNote(shown, present, "any selected problem set at " + posText(r), r, true) + tweenNote(present, r);
  }
  function renderDist(shown) {
    var p = METRIC[state.dmetric], r = curPos(), keys = shown.map(function (m) { return m.key; });
    if (!shown.length) { return '<p class="v2hint">Select at least one method.</p>'; }
    var head = distHead(shown, p);
    if (p.kind === "rate") { return head + renderDistRate(shown, p, r); }
    if (!ready(histFile(p.key))) { ensure(histFile(p.key), scheduleRender); return head + '<p class="v2hint">Loading the distribution…</p>'; }
    if (!histOf(p.key)) { return head + '<p class="v2hint">This release holds no distribution for ' + esc(p.label) + ".</p>"; }
    if (state.dmode === "rungs") { return head + renderDistLadder(shown, p); }
    var series = [];
    shown.forEach(function (m) { var use = state.dmode === "cats" ? state.cats.filter(function (c) { return cell(m.key, c, r); }) : poolCats(m.key, r, keys); if (!use.length) { return; }
      var ph = pooledHist(p.key, m.key, r, use); if (!ph) { return; }
      var rows = use.reduce(function (a, c) { return a + cell(m.key, c, r).n; }, 0); series.push({ m: m, ph: ph, f: five(ph), use: use, laws: laws(use), rows: rows }); });
    var gone = missingNote(shown, series.map(function (sr) { return sr.m; }), state.dmode === "cats" ? "any selected problem set at " + posText(r) : "all selected problem sets at " + posText(r) + ", so " + term("complete", "its distribution is not shown yet") + " (the problem sets it has finished are under \u201cby problem set\u201d)", r, true);
    if (!series.length) { return head + '<p class="v2hint">No selected method has finished ' + (state.dmode === "cats" ? "any selected problem set" : "all selected problem sets") + " at " + posText(r) + " yet. Choose another budget or time above, or select fewer problem sets.</p>" + gone; }
    var pooled = state.dmode !== "cats" ? " The problems are those of the selected problem sets (" + series[0].laws.toLocaleString() + "), the same for every method; the count under each histogram says on how many runs the method's formula has a value." : "";
    var body = state.dmode === "ecdf" ? distEcdf(series, p, r) : state.dmode === "cats" ? distCats(series, p, r) : distHists(series, p, r);
    return head + body + tweenNote(series.map(function (sr) { return sr.m; }), r) + '<p class="v2hint">' + (state.dmode === "cats" ? "" : (state.dmode === "ecdf" && state.dnorm === "all" ? "" : (p.worst !== undefined && state.impute ? "All problems: a problem without a usable formula counts as " + p.worst + "." : "Only problems where the method's formula has a value are counted.")) + pooled + " ") + term("median", "Medians are read from histograms with 128 bins") + "; values outside the shown range, including plus infinity, are counted in the outermost bins.</p>" + gone;
  }
  function distHists(series, p, r) {
    var nr = narrow(), W = wideWidth(), R = 18, T = 52, ph0 = series[0].ph, vr = viewRange(series.map(function (sr) { return sr.ph; }));
    var L = nr ? 14 : Math.max(168, Math.ceil(25 + widest(series.map(function (sr) { return sr.m.label + (sr.m.local ? " (local)" : ""); })) + 16));   // the name column
    var nmax = Math.max.apply(null, series.map(function (sr) { return sr.ph.n; })), f = nmax < 150 ? 4 : nmax < 600 ? 2 : 1;   // fewer problems, wider bins
    var panelH = nr ? 118 : 80, gap = 16, boxH = 22, capH = nr ? 34 : 0, step = panelH + boxH + capH + gap, H = T + series.length * step + 34;
    var xs = function (x) { return L + (Math.min(vr.hi, Math.max(vr.lo, x)) - vr.lo) / (vr.hi - vr.lo) * (W - L - R); };
    var groups = series.map(function (sr) { var g = []; for (var b = vr.b0; b <= vr.b1; b += f) { var c = 0; for (var j = b; j < Math.min(b + f, vr.b1 + 1); j++) { c += sr.ph.h[j]; } g.push({ b: b, share: c / sr.ph.n, edge: (b === 0 && vr.b0 === 0) || (b + f > ph0.nb - 1 && vr.b1 === ph0.nb - 1) }); } return g; });
    // One scale for every panel, set by the body of the distributions: a lone spike (every problem that fits no better
    // than the mean lands in one bin) would flatten everything else, so it is cut at the scale and labelled instead.
    var top = 0; groups.forEach(function (g) { var v = g.filter(function (x) { return !x.edge; }).map(function (x) { return x.share; }).sort(function (a, b) { return b - a; }); if (v.length) { top = Math.max(top, v.length > 1 && v[0] > 1.6 * v[1] ? v[1] : v[0]); } });
    if (!top) { groups.forEach(function (g) { g.forEach(function (x) { top = Math.max(top, x.share); }); }); }
    var s = '<svg viewBox="0 0 ' + W + " " + H + '" class="v2chart v2distwide" role="img" aria-label="' + esc(p.label) + ' distribution, one histogram per method"><text x="' + L + '" y="' + TITLE_Y + '" class="ct">' + esc(p.label) + " at " + posText(r) + "</text>";
    s += xAxisSVG(p, vr, xs, T, H - 34 - gap + 6, W, L, R);
    series.forEach(function (sr, i) { var y0 = T + i * step, yb = y0 + panelH, col = colorOf(sr.m), pts = [], cl = [];
      var yOf = function (v) { return yb - Math.min(1, v / (top * 1.08)) * (panelH - 14); };
      groups[i].forEach(function (x) { var x0 = xs(vr.lo + (x.b - vr.b0) * vr.w), x1 = xs(vr.lo + (Math.min(x.b + f, vr.b1 + 1) - vr.b0) * vr.w); pts.push(x0.toFixed(1) + "," + yOf(x.share).toFixed(1), x1.toFixed(1) + "," + yOf(x.share).toFixed(1)); if (x.share > top * 1.08) { cl.push({ x0: x0, x1: x1, x: (x0 + x1) / 2, share: x.share, edge: x.edge }); } });
      s += '<line x1="' + L + '" y1="' + yb + '" x2="' + (W - R) + '" y2="' + yb + '" class="grid"/>';
      s += '<polygon points="' + xs(vr.lo).toFixed(1) + "," + yb + " " + pts.join(" ") + " " + xs(vr.hi).toFixed(1) + "," + yb + '" fill="' + col + '" fill-opacity="0.22" stroke="none"/><polyline fill="none" stroke="' + col + '" stroke-width="1.6" points="' + pts.join(" ") + '"/>';
      // a bin taller than the scale is a broken bar: a slanted gap near its top, and its share written beside the gap
      var nl = 0, nrr = 0; cl.forEach(function (c) { var right = c.x > (L + W - R) / 2, row = right ? nrr++ : nl++, by = y0 + 30, xa = (c.x0 - 2).toFixed(1), xb = (c.x1 + 2).toFixed(1);
        s += '<path class="v2break" d="M' + xa + " " + by + " L" + xb + " " + (by - 4) + " L" + xb + " " + (by - 9) + " L" + xa + " " + (by - 5) + 'Z" fill="var(--surface)" stroke="none"/>' +
          '<line x1="' + xa + '" y1="' + by + '" x2="' + xb + '" y2="' + (by - 4) + '" stroke="' + col + '" stroke-width="1.4"/><line x1="' + xa + '" y1="' + (by - 5) + '" x2="' + xb + '" y2="' + (by - 9) + '" stroke="' + col + '" stroke-width="1.4"/>';
        s += '<text x="' + (right ? c.x0 - 8 : c.x1 + 8).toFixed(1) + '" y="' + (by + 14 * row) + '" class="tick" text-anchor="' + (right ? "end" : "start") + '">' + (100 * c.share).toFixed(0) + (c.edge ? (right ? " % in the right-most bin" : " % in the left-most bin") : " % in this bin") + "</text>"; });
      s += boxSVG(sr.f, xs, yb + boxH / 2 + 3, 9, col, sr.m.label + ": " + fiveText(p, sr.f));
      var name = sr.m.label + (sr.m.local ? " (local)" : ""), cap = "median " + fmt(p, sr.f[2]) + " · n = " + Math.round(sr.ph.n).toLocaleString() + " of " + sr.rows.toLocaleString() + " problems";
      if (nr) { s += '<rect x="' + L + '" y="' + (yb + boxH + 9) + '" width="10" height="10" rx="2" fill="' + col + '"/><text x="' + (L + 15) + '" y="' + (yb + boxH + 18) + '" class="leg">' + esc(name) + '</text><text x="' + L + '" y="' + (yb + boxH + 33) + '" class="tick">' + esc(cap) + "</text>"; }
      else { s += '<rect x="10" y="' + (y0 + 8) + '" width="10" height="10" rx="2" fill="' + col + '"/><text x="25" y="' + (y0 + 17) + '" class="leg">' + esc(name) + '</text><text x="10" y="' + (y0 + 36) + '" class="tick">median ' + esc(fmt(p, sr.f[2])) + '</text><text x="10" y="' + (y0 + 52) + '" class="tick">n = ' + Math.round(sr.ph.n).toLocaleString() + " of " + sr.rows.toLocaleString() + "</text>"; } });
    return s + "</svg>" + '<p class="v2hint">Bar height: the share of the method’s problems in each bin, with the problem sets weighted as in an average, on the same scale in every panel. A bar too tall for the scale is cut, with its share written beside it. Under each histogram, the line spans the middle 90 % of the problems, the box the middle 50 %, and the tick marks the median.</p>';
  }
  function distEcdf(series, p, r) {
    var nr = narrow(), W = wideWidth(), L = 66, T = 52, B = nr ? 60 + 20 * series.length : 58, H = plotHeight(W) + B;
    var R = nr ? 18 : legendRight(series.map(function (sr) { return sr.m.label + (sr.m.local ? " (local)" : ""); }));
    var vr = viewRange(series.map(function (sr) { return sr.ph; })), all = state.dnorm === "all", low = p.higher === true;   // a problem without a prediction sits at the worse end
    var xs = function (x) { return L + (Math.min(vr.hi, Math.max(vr.lo, x)) - vr.lo) / (vr.hi - vr.lo) * (W - L - R); }, y = function (v) { return T + (1 - v) * (H - T - B); };
    var s = '<svg viewBox="0 0 ' + W + " " + H + '" class="v2chart v2distwide" role="img" aria-label="' + esc(p.label) + ' cumulative distribution"><text x="' + (nr ? 10 : L) + '" y="' + TITLE_Y + '" class="ct">' + esc(nr ? p.short + " @ " + (isTimePos(r) ? fmtSec(posValue(r)) + " s" : fmtBudget(+r)) + ": share at or below" : p.label + " at " + posText(r) + ": share of " + (all ? "all" : "predicted") + " problems at or below") + "</text>";
    [0, 0.25, 0.5, 0.75, 1].forEach(function (g) { s += '<line x1="' + L + '" y1="' + y(g).toFixed(1) + '" x2="' + (W - R) + '" y2="' + y(g).toFixed(1) + '" class="grid' + (g === 0.5 ? " zero" : "") + '"/><text x="' + (L - 6) + '" y="' + (y(g) + 4).toFixed(1) + '" class="tick" text-anchor="end">' + (100 * g) + "%</text>"; });
    s += xAxisSVG(p, vr, xs, T, H - B, W, L, R);
    var ly = nr ? H - B + 50 : T + 6, lx = nr ? L : W - R + LEG_GAP;
    var lines = [], keys2 = [];
    series.forEach(function (sr) { var col = colorOf(sr.m), den = all ? sr.rows : sr.ph.n, cum = all && low ? sr.rows - sr.ph.n : 0, pts = [], line = "";
      for (var b = 0; b < sr.ph.nb; b++) { var before = cum; cum += sr.ph.h[b]; if (b < vr.b0) { continue; } if (b > vr.b1) { break; } var x0 = vr.lo + (b - vr.b0) * vr.w; if (!pts.length) { pts.push(xs(x0).toFixed(1) + "," + y(before / den).toFixed(1)); } pts.push(xs(x0 + vr.w).toFixed(1) + "," + y(before / den).toFixed(1), xs(x0 + vr.w).toFixed(1) + "," + y(cum / den).toFixed(1)); }
      line = '<polyline fill="none" stroke="' + col + '" stroke-width="2"' + dashOf(sr.m) + ' points="' + pts.join(" ") + '"><title>' + esc(sr.m.label + ": " + fiveText(p, sr.f) + "; " + Math.round(sr.ph.n).toLocaleString() + " of " + sr.rows.toLocaleString() + " problems have a value") + "</title></polyline>";
      lines.push({ key: sr.m.key, line: line });
      keys2.push('<line x1="' + lx + '" y1="' + ly + '" x2="' + (lx + 20) + '" y2="' + ly + '" stroke="' + col + '" stroke-width="3"' + dashOf(sr.m) + '/><text x="' + (lx + 26) + '" y="' + (ly + 4) + '" class="leg' + (visOf(sr.m.key) === "dim" ? " v2legdim" : "") + '">' + esc(sr.m.label + (sr.m.local ? " (local)" : "")) + "</text>"); ly += 20; });
    s += behind(lines).map(function (x) { return x.line; }).join("") + keys2.join("");
    return s + "</svg>" + '<p class="v2hint">' + (p.higher === true ? "Further right is better: a curve that stays low longer holds more of its problems at high values." : p.higher === false ? "Further left is better: a curve that rises early holds more of its problems at low values." : p.ideal !== undefined ? "Closer to " + p.ideal + " is better: a curve that rises steeply around " + p.ideal + " is the tighter one." : "") +
      (all ? " Out of all problems, a method that leaves problems without a prediction " + (low ? "starts above 0 %." : "ends below 100 %.") : "") + "</p>";
  }
  function distCats(series, p, r) {
    var nr = narrow(), W = wideWidth(), T = 52, R = 18, Lw = nr ? 110 : 160, per = {}, phs = [];
    var cats = state.cats.slice().sort(function (a, b) { return CAT[b].laws - CAT[a].laws; });
    cats.forEach(function (c) { series.forEach(function (sr) { if (sr.use.indexOf(c) < 0) { return; } var ph = pooledHist(p.key, sr.m.key, r, [c]); if (!ph) { return; } (per[c] = per[c] || {})[sr.m.key] = ph; phs.push(ph); }); });
    cats = cats.filter(function (c) { return per[c]; });
    if (!phs.length) { return '<p class="v2hint">No selected problem set has a finite value at ' + esc(posText(r)) + ".</p>"; }
    var vr = viewRange(phs), rowH = Math.max(22, 11 * series.length + 8), B = 44 + 18 * series.length, H = T + (cats.length + 1) * rowH + B;
    var xs = function (x) { return Lw + (Math.min(vr.hi, Math.max(vr.lo, x)) - vr.lo) / (vr.hi - vr.lo) * (W - Lw - R); };
    var s = '<svg viewBox="0 0 ' + W + " " + H + '" class="v2chart v2distwide" role="img" aria-label="' + esc(p.label) + ' per problem set"><text x="' + Lw + '" y="' + TITLE_Y + '" class="ct">' + esc(p.label) + " per problem set at " + posText(r) + "</text>";
    s += xAxisSVG(p, vr, xs, T, H - B, W, Lw, R);
    var row = function (label, i, get, bold) { var yy = T + (i + 0.5) * rowH;
      if (i % 2) { s += '<rect x="' + Lw + '" y="' + (yy - rowH / 2) + '" width="' + (W - Lw - R) + '" height="' + rowH + '" class="v2stripe"/>'; }
      s += '<text x="' + (Lw - 8) + '" y="' + (yy + 4) + '" class="' + (bold ? "leg" : "tick") + '" text-anchor="end">' + esc(label) + "</text>";
      series.forEach(function (sr, j) { var ph = get(sr); if (!ph) { return; } var yj = yy + (j - (series.length - 1) / 2) * 11; s += boxSVG(five(ph), xs, yj, 7, colorOf(sr.m), sr.m.label + " on " + label + ": " + fiveText(p, five(ph)) + ", n = " + ph.n); }); };
    cats.forEach(function (c, i) { row(c + " · " + CAT[c].laws, i, function (sr) { return per[c][sr.m.key]; }, false); });
    row("all listed", cats.length, function (sr) { return sr.ph; }, true);
    var ly = H - B + 46; series.forEach(function (sr) { s += '<rect x="' + (Lw + 1) + '" y="' + (ly - 6) + '" width="10" height="10" rx="2" fill="' + colorOf(sr.m) + '"/><text x="' + (Lw + 16) + '" y="' + (ly + 4) + '" class="leg">' + esc(sr.m.label + (sr.m.local ? " (local)" : "")) + "</text>"; ly += 18; });
    return s + "</svg>" + '<p class="v2hint">One row per problem set, largest first. For each method, the line spans the middle 90 % of its problems, the box the middle 50 %, and the tick marks the median. Tap or hover a box for its numbers.</p>';
  }
  function renderDistLadder(shown, p) {
    var drawn = axisMethods(shown), keys = drawn.map(function (m) { return m.key; }), src = timeSource(keys), series = [], ymin = Infinity, ymax = -Infinity, tmin = Infinity, tmax = -Infinity;
    drawn.forEach(function (m) { var pts = [];
      D.rungs.forEach(function (r) { var use = poolCats(m.key, r, keys); if (!use.length) { return; } var x = xOf(m.key, r, use, src); if (x === null) { return; } var ph = pooledHist(p.key, m.key, r, use); if (!ph) { return; } var f = five(ph);
        pts.push({ x: x, v: f[2], lo: f[1], hi: f[3], title: m.label + " @ " + r + ": " + fiveText(p, f) + ", n = " + ph.n });
        ymin = Math.min(ymin, f[1]); ymax = Math.max(ymax, f[3]); if (state.xaxis === "time") { tmin = Math.min(tmin, x); tmax = Math.max(tmax, x); } });
      if (pts.length) { series.push({ key: m.key, label: m.label + (m.local ? " (local)" : ""), color: colorOf(m), dash: !!m.dash, pts: pts }); } });
    var title = narrow() ? p.short + ": median, middle half" : p.label + ": median and middle half, by budget", off = offAxis(shown);
    var note = off.length ? '<p class="v2hint">' + esc(off.map(function (m) { return m.label; }).join(", ")) + (off.length > 1 ? " have" : " has") + " no place on this axis; choose the other x-axis above.</p>" : "";
    if (!series.length) { return chartSVG({ title: title, aria: title, series: [], empty: "nothing to draw on this axis yet", width: wideWidth() }) + note; }
    var pad = (ymax - ymin) * 0.08 || 0.1; ymin -= pad; ymax += pad; var tr = state.xaxis === "time" ? timeRange(tmin, tmax) : [0, 0];
    var svg = withState({ band: true, cross: false }, function () { return chartSVG({ title: title, aria: title, width: wideWidth(), series: series, ymin: ymin, ymax: ymax, ticks: ticksFor(p, ymin, ymax), tick: function (g) { return tickLabel(p, g); }, ylabel: axisName(p), timeAxis: state.xaxis === "time", timeSource: src, tmin: tr[0], tmax: tr[1], zero: tfOf(p) === "log2" ? 0 : undefined }); });
    return '<div class="v2charts v2one">' + svg + "</div>" + '<p class="v2hint">Line: the median over the problems with a usable formula. Band: the middle 50 % of those problems; it shows their spread, not the uncertainty of the median. Tap or hover a point for more percentiles.</p>' + note;
  }

  // ---- Paired ----------------------------------------------------------------------------------------------------
  function fmtP(p) { if (p === null || p === undefined) { return "–"; } return p < 0.001 ? "< 0.001" : p.toFixed(3); }
  // Two methods on the problems both have, one problem at a time (over every combination of their runs): each problem's
  // difference, averaged over the problem sets like any value, with its interval. The test asks whether the average
  // difference is zero (t, like the interval); for a continuous metric it is taken on each problem's superiority (the
  // share of run pairs one method wins minus the share it loses), so a few large differences cannot decide it.
  // a contrast at a slot from the exporter's cells (paired/<key>.js): both methods placed on every selected problem set
  function pairedSlotStat(metric, a, b, slot) {
    if (!slotPlaces(a, slot) || !slotPlaces(b, slot) || !freshAt(a, b, slot)) { return null; }
    var names = [metric.key, metric.key + "@answered"], key = a + "|" + b, flip = false;
    if (!names.some(function (nm) { return (pairCellsOf(nm) || {})[key]; })) { key = b + "|" + a; flip = true; }
    return pairedPool(metric, function (c) { var o = {}; names.forEach(function (nm) { var e = ((pairCellsOf(nm) || {})[key] || {})[c]; if (e && e[slot]) { o[nm] = e[slot]; } }); return { m: o }; }, state.cats, flip);
  }
  // a contrast pooled over the problem sets cs from their cells (cellOf(c): paired_cell's {n, m}), flipped when the
  // cells hold the other method first
  function pairedPool(metric, cellOf, cs, flip) {
    var pk = metric.key + (leftOut(metric.key) ? "@answered" : ""), sgn = flip ? -1 : 1, rate = metric.kind === "rate", dsets = [], ssets = [], wins = 0, losses = 0;
    cs.forEach(function (c) { var pc = cellOf(c), t = pc && pc.m[pk]; if (!t) { return; }
      dsets.push({ n: t[0], s1: sgn * t[1], s2: t[2] });
      if (rate) { wins += flip ? t[4] : t[3]; losses += flip ? t[3] : t[4]; }
      else { ssets.push({ n: t[3], s1: sgn * t[4], s2: t[5] }); wins += flip ? t[7] : t[6]; losses += flip ? t[6] : t[7]; } });
    var fd = reCombine(dsets, "normal"); if (!fd) { return null; }
    var ft = rate ? fd : reCombine(ssets, "normal");
    return { v: fd.mu, lo: fd.lo, hi: fd.hi, piLo: fd.piLo, piHi: fd.piHi, n: ft ? ft.n : fd.n, S: fd.S, p: ft ? ft.p : null, wins: wins, losses: losses };
  }
  function fmtDelta(metric, d) { if (!isFinite(d)) { return "–"; } var tf = tfOf(metric); if (metric.kind === "rate") { return (d >= 0 ? "+" : "") + (100 * d).toFixed(1) + " pp"; } if (tf === "log2") { return "× " + Math.pow(2, d).toFixed(2); } if (tf === "log10") { return "× " + Math.pow(10, d).toFixed(2); } return (d >= 0 ? "+" : "") + d.toFixed(metric.fmt === "num3" ? 3 : 2); }
  function renderPaired(shown) {
    if (shown.length < 2) { return '<p class="v2hint">Select at least two methods; one of them is the baseline.</p>'; }
    if (!state.base || !shown.some(function (m) { return m.key === state.base; })) { state.base = shown[0].key; }
    var base = D.methods.filter(function (m) { return m.key === state.base; })[0], others = shown.filter(function (m) { return m.key !== state.base; }), keys = shown.map(function (m) { return m.key; });
    var plots = plotMetrics().filter(function (k) { return PAIRED_KEYS.indexOf(k) >= 0; }).map(function (k) { return METRIC[k]; });
    var ctl = '<p class="v2hint">Each method is compared with ' + esc(base.label) + ' problem by problem: every number is the method\u2019s value minus ' + esc(base.label) + '\u2019s. ' + term("pairdiff", "How the difference is taken and tested") + '.</p>';
    if (!plots.length) { return ctl + '<p class="v2hint">None of the plotted metrics has paired contrasts. Paired contrasts exist for: ' + esc(PAIRED_KEYS.map(function (k) { return METRIC[k] ? mname(METRIC[k]) : k; }).join(", ")) + '.</p>'; }
    // The charts run along the budgets or the times, as the position above reads: at every slot, each method is read
    // as at the position (between two of its budgets too), from the exporter's cells.
    var time = state.pm === "time", files = plots.map(function (p) { return "paired/" + p.key + ".js"; }), slots = slotsOf(time);
    if (!filesReady(files)) { return ctl + '<div class="v2viewbar">' + posControl(shown) + '</div><p class="v2hint">Loading the paired contrasts…</p>'; }
    var charts = inBlock(root.querySelector(".v2main"), plots.length, function () { return plots.map(function (p) { var series = [], ymin = Infinity, ymax = -Infinity, tmin = Infinity, tmax = -Infinity;
      var ctx = p.key + (leftOut(p.key) ? "@answered" : "") + "|" + base.key + "|" + slotContext(pairCellsOf(p.key));
      others.forEach(function (m) { var pts = []; slots.forEach(function (sl) { var x = sl.x, st = slotMemo("pair|" + ctx + "|" + m.key + "|" + sl.slot, function () { return pairedSlotStat(p, m.key, base.key, sl.slot); }); if (!st || !isFinite(st.v)) { return; }
          pts.push({ x: x, v: st.v, lo: st.lo, hi: st.hi, title: m.label + " − " + base.label + " at " + posText(time ? "t" + x : x) + ": " + fmtDelta(p, st.v) + " [" + fmtDelta(p, st.lo) + ", " + fmtDelta(p, st.hi) + "], over " + st.n.toLocaleString() + " problems in " + st.S + " problem sets, p = " + fmtP(st.p) });
          [st.v, anyCI() ? st.lo : st.v, anyCI() ? st.hi : st.v].forEach(function (v) { if (isFinite(v)) { ymin = Math.min(ymin, v); ymax = Math.max(ymax, v); } }); if (time) { tmin = Math.min(tmin, x); tmax = Math.max(tmax, x); } });
        if (pts.length) { series.push({ key: m.key, label: m.label + (m.local ? " (local)" : ""), color: colorOf(m), dash: !!m.dash, pts: pts }); } });
      var title = "Δ " + mname(p) + " vs " + base.label;
      if (!series.length) { return chartSVG({ title: title, aria: title, series: [], empty: "no matched cells with the baseline yet" }); }
      if (nearRange(ymin, ymax, 0, 0.35)) { ymin = Math.min(ymin, 0); ymax = Math.max(ymax, 0); } var pad = (ymax - ymin) * 0.08 || 0.05; ymin -= pad; ymax += pad;
      var tr = time ? timeRange(tmin, tmax) : [0, 0];
      var ptf = tfOf(p), dticks = ptf === "log2" ? logTicks(ymin, ymax, 2) : ptf === "log10" ? logTicks(ymin, ymax, 10) : linearTicks(ymin, ymax);
      var dtick = function (g) { return p.kind === "rate" ? (g > 0 ? "+" : "") + roundNum(100 * g) + " pp" : ptf === "log2" ? "× " + roundNum(Math.pow(2, g)) : ptf === "log10" ? "× " + roundNum(Math.pow(10, g)) : (g > 0 ? "+" : "") + roundNum(g); };
      return chartSVG({ title: title, aria: title, series: series, ymin: ymin, ymax: ymax, ticks: dticks, tick: dtick, ylabel: "", timeAxis: time, timeSource: "ref", tmin: tr[0], tmax: tr[1], zero: 0 }); }); });
    var pos = curPos(), rows = "", pend = false, anyRow = false;
    others.forEach(function (m) { rows += '<tr><td><span class="v2sw" style="background:' + colorOf(m) + '"></span>' + esc(m.label) + '</td>' + plots.map(function (p) {
      var st = pairedAt(p, m, base, pos);
      if (st && st.wait) { pend = true; return '<td class="v2na">…</td><td class="v2na">…</td><td class="v2na">…</td>'; }
      if (!st) { return '<td class="v2na">–</td><td class="v2na">–</td><td class="v2na">–</td>'; }
      anyRow = true; var sig = st.p !== null && st.p < 0.05;
      return '<td' + (sig ? ' class="v2sig"' : "") + '>' + fmtDelta(p, st.v) + ' <span class="v2ci-txt">[' + fmtDelta(p, st.lo) + ", " + fmtDelta(p, st.hi) + ']</span></td><td>' + fmtP(st.p) + '</td><td class="v2hint">' + st.wins.toLocaleString() + " / " + st.losses.toLocaleString() + " of " + st.n.toLocaleString() + '</td>'; }).join("") + '</tr>'; });
    var gone = others.concat([base]).filter(function (m) { return !pvPlaces(m, pos); });
    var table = '<h3 class="v2h">At ' + esc(posText(pos)) + "</h3>" +
      (pend ? '<p class="v2hint">Loading the values problem by problem…</p>' : anyRow ? "" : '<p class="v2hint">No method has a value at ' + esc(posText(pos)) + " on every selected problem set together with the baseline. Move the position above, or select fewer problem sets.</p>") +
      (gone.length ? '<p class="v2hint">' + gone.map(function (m) { return !(D.pv || {})[m.key] ? esc(m.label) + ": its values per problem are not published in this release." : outside(m, pos) ? outsideText(m, pos) : esc(m.label) + " has not finished all selected problem sets at the budgets around " + esc(posText(pos)) + "."; }).join(" ") + "</p>" : "") +
      '<div class="v2table-wrap"><table class="v2table"><thead><tr><th>method − ' + esc(base.label) + ", at " + esc(posText(pos)) + '</th>' + plots.map(function (p) { return '<th colspan="3">' + esc(mname(p)) + " " + mhelp(p) + '</th>'; }).join("") + '</tr><tr><th></th>' + plots.map(function () { return '<th>Δ [95 %]</th><th>p</th><th>better / worse</th>'; }).join("") + '</tr></thead><tbody>' + rows + '</tbody></table></div>' +
      '<p class="v2hint">\u0394: the method\u2019s value minus the baseline\u2019s, with its 95 % interval; pp are percentage points, and ratios are compared as factors (\u00d7 0.5 means half). p: the probability of a difference at least this large if both methods were equally good, from ' + term("pairdiff", "a test over the problem sets") + '; for metrics other than rates it tests how often the method does better, not \u0394. Bold: p below 0.05, so the direction of the difference is unlikely to be chance; the \u0394 column shows its size. Better / worse: on how many problems the method does better or worse than the baseline on that metric; for a ratio, these counts say which is closer to 1. Between two budgets, a method is read problem by problem from both. Choose the baseline above. <a href="paired.html">How paired comparisons work</a></p>';
    return ctl + '<div class="v2viewbar">' + posControl(shown) + "</div>" + '<div class="v2charts">' + charts.join("") + "</div>" + table;
  }

  // ---- Predictions: the formulas themselves ------------------------------------------------------------------------
  // The formula each method returned for each problem, as the judge read it (the ground truth's variable names, the
  // engine's spelling), next to the ground truth, typeset with the page's KaTeX. The release ships one file per method
  // x problem set x budget x finished run x block of problems (D.pred indexes them), loaded as a page needs them.
  var PRED_ARITY = { "+": 2, "-": 2, "*": 2, "/": 2, pow: 2, rootn: 2, abs: 1, acos: 1, acosh: 1, asin: 1, asinh: 1, atan: 1, atanh: 1,
    cos: 1, cosh: 1, exp: 1, inv: 1, log: 1, neg: 1, sin: 1, sinh: 1, tan: 1, tanh: 1 };
  var PRED_FN = { sin: "\\sin", cos: "\\cos", tan: "\\tan", sinh: "\\sinh", cosh: "\\cosh", tanh: "\\tanh", asin: "\\arcsin", acos: "\\arccos",
    atan: "\\arctan", asinh: "\\operatorname{arsinh}", acosh: "\\operatorname{arcosh}", atanh: "\\operatorname{artanh}", log: "\\log" };
  function prefixTree(tokens) {   // a prefix expression as a tree; a token outside the vocabulary is a leaf
    var i = 0;
    function node() { var t = tokens[i++]; if (t === undefined) { throw new Error("incomplete"); } var kids = []; for (var k = 0; k < (PRED_ARITY[t] || 0); k++) { kids.push(node()); } return { t: t, k: kids }; }
    var top = node(); return i === tokens.length ? top : null;
  }
  // p is how tightly a piece binds: 1 a sum or a leading minus, 2 a product, 5 a function call, 9 an atom
  function leafTex(t) {
    if (t === "np.pi" || t === "pi") { return { s: "\\pi", p: 9 }; }
    if (t === "np.e" || t === "E") { return { s: "e", p: 9 }; }
    if (t === "<constant>") { return { s: "c", p: 9 }; }
    var v = /^x_?(\d+)$/.exec(t); if (v) { return { s: "x_{" + v[1] + "}", p: 9 }; }
    if (/^-?(\d+\.?\d*|\.\d+)(e[-+]?\d+)?$/i.test(t)) {
      var neg = t.charAt(0) === "-", body = neg ? t.slice(1) : t, e = /^(.*)e([-+]?\d+)$/i.exec(body);
      return { s: (neg ? "-" : "") + (e ? e[1] + " \\cdot 10^{" + parseInt(e[2], 10) + "}" : body), p: neg ? 1 : e ? 2 : 9, num: true };
    }
    return { s: "\\mathrm{" + t.replace(/[^A-Za-z0-9.]/g, "") + "}", p: 9 };
  }
  function texOf(n) {
    var a = n.k[0] && texOf(n.k[0]), b = n.k[1] && texOf(n.k[1]);
    var par = function (x, min) { return x.p < min ? { s: "\\left(" + x.s + "\\right)", p: 9, num: x.num } : x; };
    switch (n.t) {
      case "+": return { s: a.s + (b.s.charAt(0) === "-" ? " - " + b.s.slice(1) : " + " + b.s), p: 1 };
      case "-": return { s: a.s + " - " + par(b, 2).s, p: 1 };
      case "*": { var l = a.num ? a : par(a, 2), r = par(b, 2), s2 = l.s + (l.num && !r.num ? " \\, " : " \\cdot ") + r.s; return { s: s2, p: s2.charAt(0) === "-" ? 1 : 2 }; }
      case "/": return { s: "\\frac{" + a.s + "}{" + b.s + "}", p: 9 };
      case "pow": return { s: (a.sup ? "\\left(" + a.s + "\\right)" : par(a, 9).s) + "^{" + b.s + "}", p: 9, sup: true };   // a power of a power keeps its brackets
      case "rootn": return { s: (b.s === "2" ? "\\sqrt{" : "\\sqrt[" + b.s + "]{") + a.s + "}", p: 9 };
      case "neg": return { s: "-" + par(a, 2).s, p: 1 };
      case "inv": return { s: "\\frac{1}{" + a.s + "}", p: 9 };
      case "abs": return { s: "\\left|" + a.s + "\\right|", p: 9 };
      case "exp": return { s: "e^{" + a.s + "}", p: 9, sup: true };
      default: return PRED_FN[n.t] && a ? { s: PRED_FN[n.t] + "\\left(" + a.s + "\\right)", p: 5 } : leafTex(n.t);
    }
  }
  function typeset(expr) {   // KaTeX when the page has it and the expression parses; the prefix as written otherwise
    var t = null;
    try { var tree = prefixTree(String(expr).split(" ")); t = tree ? texOf(tree).s : null; } catch (e) { t = null; }
    if (t !== null && window.katex) { try { return window.katex.renderToString("\\displaystyle " + t, { throwOnError: false }); } catch (e) { /* as written */ } }
    return "<code>" + esc(expr) + "</code>";
  }
  function predMarks(f) {
    return (f & 1 ? '<span class="v2tag v2predmark" title="Numeric Recovery: reproduces the held-out points almost exactly">numeric</span>' : "") +
      (f & 2 ? '<span class="v2tag v2predmark" title="Symbolic Recovery: Structure: the same form as the true formula once numbers are ignored">structure</span>' : "");
  }
  // The true formula as the benchmark states it, and its canonical form: what SimpliPy, the engine the judge uses, makes of
  // it with its numbers kept -- the form Symbolic Recovery compares a prediction with once the numbers are masked. A
  // release written before the canonical forms shipped holds the stated formula alone.
  function predTruth(gt, wait) {
    var stated = Array.isArray(gt) ? gt[0] : gt, canon = Array.isArray(gt) ? gt[1] : undefined;
    var html = '<div class="v2predtruth"><span class="v2lab">true formula</span><span class="v2predtruthf">' + (gt === undefined ? wait : typeset(stated)) + "</span></div>";
    if (canon === undefined) { return html; }
    return html + '<div class="v2predtruth v2predcanon"><span class="v2lab">simplified ' + help("The true formula simplified by SimpliPy, the engine the judge uses, with its numbers kept. Symbolic Recovery compares a prediction with this form once the numbers are masked (the stricter versions keep exponents, or all numbers).", "What is the simplified form?") +
      '</span><span class="v2predtruthf">' + (canon === null ? '<span class="v2predna">none: the engine cannot read this formula</span>' : canon === stated ? '<span class="v2hint">the same as stated</span>' : typeset(canon)) + "</span></div>";
  }
  // A formula cannot be blended, so at a position each method shows the formula of its last budget at or below it whose
  // run is published: the largest such budget, or by time, the largest whose measured time per problem fits within it.
  // Its row names that budget.
  function predAt(m, c, run, pos) {
    var idx = (D.pred || {})[m.key] || {}, time = isTimePos(pos), x = posValue(pos), best = null;
    Object.keys(idx).forEach(function (k) { var cut = k.lastIndexOf("|"); if (k.slice(0, cut) !== c || idx[k].indexOf(run) < 0) { return; }
      var r = +k.slice(cut + 1), v = time ? refTime(m.key, r) : r; if (v > 0 && v <= x * (1 + 1e-9) && (best === null || r > best)) { best = r; } });
    return best;
  }
  function predWhy(m, c, run, pos) {
    if (!(D.pred || {})[m.key]) { return "its formulas are not published in this release"; }
    if (isTimePos(pos)) { var ts = timesOf(m); return !ts.length ? "not timed yet" : Math.min.apply(null, ts) > posValue(pos) * (1 + 1e-9) ? "its smallest budget takes " + fmtSec(Math.min.apply(null, ts)) + " s" : "run " + run + " not finished within " + fmtSec(posValue(pos)) + " s yet"; }
    return m.budgets && m.budgets.length && m.budgets[0] > posValue(pos) * (1 + 1e-9) ? "it runs from budget " + fmtBudget(m.budgets[0]) : "run " + run + " not finished at or below budget " + fmtBudget(posValue(pos)) + " yet";
  }
  function renderPreds(shown) {   // one problem at a time: the true formula, then one row per method
    var P = D.pred || {}, B = D.pred_block || 500;
    if (!Object.keys(P).length) { return '<p class="v2hint">The formulas are not published in this release yet.</p>'; }
    var sets = state.cats.slice().sort();
    if (!sets.length) { return '<p class="v2hint">Select a problem set in the side panel.</p>'; }
    if (sets.indexOf(state.pset) < 0) { state.pset = sets.indexOf("feynman") >= 0 ? "feynman" : sets[0]; }
    var c = state.pset, run = state.prun, n = CAT[c].laws, pos = curPos(), time = isTimePos(pos);
    state.pprob = Math.min(n - 1, Math.max(0, state.pprob));
    var i = state.pprob, blk = Math.floor(i / B), at = {};
    shown.forEach(function (m) { at[m.key] = predAt(m, c, run, pos); });
    var have = shown.filter(function (m) { return at[m.key] !== null; });
    var files = ["pred/truth/" + c + "." + blk + ".js"].concat(have.map(function (m) { return "pred/" + m.key + "/" + c + "/" + at[m.key] + "." + run + "." + blk + ".js"; }));
    var pending = files.filter(function (f) { return !ready(f); });
    pending.forEach(function (f) { ensure(f, render); });
    var R = (window.RESULTS_V2_PRED || {})[REL] || {};
    var get = function (key) { var got = R[key + "|" + blk]; return got ? got[String(i)] : undefined; };
    var wait = pending.length ? "\u2026" : "";
    var probBtn = function (to, glyph, word) { return '<button type="button" class="v2stepbtn" ' + (to === null ? "disabled" : 'data-set="pprob:' + to + '"') + ' aria-label="' + word + ' problem">' + glyph + "</button>"; };
    var bar = '<div class="v2viewbar"><span class="v2segwrap"><span class="v2lab">problem set</span><select class="v2stepsel" data-state="pset" aria-label="problem set">' +
      sets.map(function (x) { return '<option value="' + esc(x) + '"' + (x === c ? " selected" : "") + ">" + esc(x) + "</option>"; }).join("") + "</select></span>" +
      '<span class="v2segwrap"><span class="v2lab">problem</span><span class="v2step">' + probBtn(i > 0 ? i - 1 : null, "◀", "previous") +
      '<input type="number" class="v2predprob" data-state="pnum" min="1" max="' + n + '" value="' + (i + 1) + '" aria-label="problem number">' +
      '<span class="v2hint v2predof">of ' + n + "</span>" + probBtn(i < n - 1 ? i + 1 : null, "▶", "next") + "</span></span>" +
      seg("prun", String(run), [["1", "run 1"], ["2", "run 2"]], "run " + help(TERMS.draw1, "What is a run?"), "which run") + "</div>" +
      '<div class="v2viewbar">' + posControl(shown, help(TERMS.predat, "Which budget does each method show?")) + "</div>";
    var gt = get("truth|" + c);
    var rows = shown.map(function (m) {
      var r = at[m.key], why = r === null ? predWhy(m, c, run, pos) : "";
      var v = why ? undefined : get(m.key + "|" + c + "|" + r + "|" + run);
      var formula = why ? '<span class="v2predna">' + esc(why) + "</span>" : v === undefined ? wait : v === null ? '<span class="v2predna">no usable formula</span>' : typeset(v[0]);
      var budget = r === null ? '<span class="v2predna">–</span>' : fmtBudget(r) + (time && refTime(m.key, r) ? ' <span class="v2ci-txt">' + fmtSec(refTime(m.key, r)) + " s</span>" : "");
      return '<tr><th><span class="v2sw" style="background:' + colorOf(m) + '"></span>' + esc(m.label) + '</th><td class="v2predat">' + budget + '</td><td class="v2predmarks">' + (v ? predMarks(v[1]) : "") + '</td><td class="v2predf">' + formula + "</td></tr>";
    }).join("");
    return bar + '<p class="v2hint">The formula each method returned for one problem of ' + esc(c) + ", run " + run + ", from its last budget " + (time ? "that takes at most " + fmtSec(posValue(pos)) + " s per problem" : "at or below budget " + fmtBudget(posValue(pos))) + " (named in each row)" +
      ". Step through the problems with \u25c0 \u25b6 or type a number. Numbers are rounded to 4 significant digits, and variables are named by their input column (x\u2081 is the first). " +
      "The column \u201crecovered\u201d says whether the formula passes Numeric Recovery (numeric) and Symbolic Recovery: Structure (structure). The problem sets to choose from are the ones selected in the side panel.</p>" +
      predTruth(gt, wait) +
      '<div class="v2table-wrap"><table class="v2table v2predtable"><thead><tr><th>method</th><th>budget</th><th>recovered</th><th>its formula</th></tr></thead><tbody>' + rows + "</tbody></table></div>";
  }

  // ---- Ranks and Paired differences at a position, computed in the browser (owner 2026-10-05) --------------------------
  // Both views read every method at the position, between two of its budgets too. What the exporter's paired_cell and
  // rank_pair_cell read of every problem ships per method x budget (pv/: a frame with the problems and their runs, and
  // one file per metric, so only the metric on screen is fetched), and pairstats.js computes the cells those functions
  // compute: bit for bit at a budget a method ran, and between two budgets by the rule at its top. A method is placed
  // on each problem set by the bracket every display uses, over its finished budgets there.
  function pvBudgets(m, c) { var idx = (D.pv || {})[m] || {}, out = []; Object.keys(idx).forEach(function (r) { if (idx[r].indexOf(c) >= 0) { out.push(+r); } }); return out; }
  function pvAt(m, c, pos) {
    var time = isTimePos(pos), ps = [];
    pvBudgets(m, c).forEach(function (r) { var v = time ? refTime(m, r) : r; if (v > 0) { ps.push([v, r]); } });
    return bracketIn(ps.sort(function (a, z) { return a[0] - z[0]; }), posValue(pos));
  }
  // where a method sits on every selected problem set, or null when it has no value on one of them
  function pvPlaces(m, pos) {
    if (!(D.pv || {})[m.key] || !state.cats.length || !window.PAIRSTATS) { return null; }
    var out = {};
    for (var i = 0; i < state.cats.length; i++) { var b = pvAt(m.key, state.cats[i], pos); if (!b) { return null; } out[state.cats[i]] = b; }
    return out;
  }
  function baseKey(k) { return String(k).replace(/@answered$/, ""); }
  function pvNeed(m, places, keys) {   // the files a method's places need: a frame and the metrics' files per budget
    var rs = {}, out = [];
    Object.keys(places).forEach(function (c) { rs[places[c].r1] = 1; rs[places[c].r2] = 1; });
    Object.keys(rs).forEach(function (r) { out.push(PAIRSTATS.framePath(m, r)); keys.forEach(function (k) { out.push(PAIRSTATS.keyPath(m, r, baseKey(k))); }); });
    return out;
  }
  // the methods' positions on the problem sets, decoded once per drawing
  function pvReader(entries) {
    var memo = {};
    return function (x, c) {
      var id = x.m.key + "|" + c; if (id in memo) { return memo[id]; }
      var b = x.places[c], lo = PAIRSTATS.file(REL, x.m.key, b.r1, c), hi = b.r2 === b.r1 ? lo : PAIRSTATS.file(REL, x.m.key, b.r2, c);
      return (memo[id] = lo && hi ? PAIRSTATS.at(lo, hi, b.w) : null);
    };
  }
  // the budget (or the two) a method is read at, as the tables name it
  function placeText(m, places, time) {
    var seen = {}, list = [];
    Object.keys(places).forEach(function (c) { var b = places[c], t = b.r1 === b.r2 ? fmtBudget(b.r1) : "≈ " + fmtBudget(b.r1) + "–" + fmtBudget(b.r2); if (!seen[t]) { seen[t] = 1; list.push(b); } });
    if (list.length > 1) { return "varies by problem set"; }
    var b = list[0], one = function (r) { return fmtBudget(r) + (time && refTime(m.key, r) ? ' <span class="v2ci-txt">' + fmtSec(refTime(m.key, r)) + " s</span>" : ""); };
    return b.r1 === b.r2 ? one(b.r1) : "≈ " + one(b.r1) + " – " + one(b.r2);
  }
  // A ranking at the position on one metric. out.wait: files are still arriving (they are requested here).
  function rankingAt(shown, pos, key) {
    var out = { roster: [], out: [], blind: [], cats: [], n: 0, wait: false }, files = [];
    shown.forEach(function (m) { var pl = pvPlaces(m, pos); if (pl) { out.roster.push({ m: m, places: pl }); } else { out.out.push(m); } });
    if (out.roster.length < 2) { return out; }
    out.roster.forEach(function (x) { files = files.concat(pvNeed(x.m.key, x.places, [key])); });
    if (!filesReady(files)) { out.wait = true; return out; }
    out.cats = state.cats.slice();
    var read = pvReader(), was = BETWEEN; BETWEEN = true;
    try {
      return rankStats(out, function (x, z, c) {
        var A = read(x, c), B = read(z, c), t = A && B ? PAIRSTATS.rankPairCell(A, B, [key]) : null;
        return t ? { n: t[0], s1: t[1], s2: t[2] } : null;
      }, function (x) {
        x.blank = out.cats.reduce(function (a, c) { var ce = cell(x.m.key, c, pos); return a + (ce ? ce.n - (ce.ok === undefined ? ce.n : ce.ok) : 0); }, 0);
        x.capped = false; });
    } finally { BETWEEN = was; }
  }
  // A paired contrast at the position: metric p, method a minus method b, pooled over the selected problem sets as the
  // charts pool the exporter's cells. null: no value; {wait: true}: files are still arriving.
  function pairedAt(p, a, b, pos) {
    var pa = pvPlaces(a, pos), pb = pvPlaces(b, pos); if (!pa || !pb) { return null; }
    var key = p.key, files = pvNeed(a.key, pa, [key]).concat(pvNeed(b.key, pb, [key]));
    if (!filesReady(files)) { return { wait: true }; }
    var read = pvReader(), xa = { m: a, places: pa }, xb = { m: b, places: pb };
    return pairedPool(p, function (c) { var A = read(xa, c), B = read(xb, c); return A && B ? PAIRSTATS.pairedCell(A, B, { keys: [key] }) : null; }, state.cats, false);
  }

  // ---- Ranks -----------------------------------------------------------------------------------------------------
  // Within every problem the methods are placed 1st, 2nd, ... on one metric; a failed prediction is placed last, two
  // recovered predictions tie, ties share a place. Per problem set, the PAIRWISE outcomes -- the sums of each pair's
  // per-problem superiority -- averaged over the problem sets give the chance that one method beats another on a
  // problem, and a method's average place is 1 + the chances that each other method beats it, so any roster of methods
  // over any selection of problem sets is ranked here. The snapshot is taken at the position, every method read there
  // (between two of its budgets too), its outcomes computed here (rankingAt); the chart along the budgets or times
  // reads the exporter's (rankingAtSlot). Pairs are tested one by one, Holm-corrected over the release's methods
  // (owner 2026-09-27; the Friedman test and the Nemenyi critical difference are gone).
  // ---- the per-key cells along the slots (ranks/<key>.js, paired/<key>.js) ----------------------------------------
  // The charts along the budgets or the times read cells the exporter computed at fixed positions, the slots: every
  // budget of the release, and the time limits TIME_BUDGETS. A slot is a position like any other: every method is
  // placed on each problem set by the same bracket (D.slots.at), so a chart at 10 s agrees with the snapshot at 10 s.
  function slotsOf(time) {
    var S = D.slots; if (!S) { return []; }
    return time ? S.budgets.map(function (b, i) { return { slot: b, x: S.seconds[i] }; }) : S.rungs.map(function (r) { return { slot: String(r), x: r }; });
  }
  function slotPlaces(m, slot) {   // {problem set: [r1, r2, w]} on every selected problem set, or null
    var at = (((D.slots || {}).at || {})[m] || {})[slot]; if (!at || !state.cats.length) { return null; }
    return state.cats.every(function (c) { return at[c]; }) ? at : null;
  }
  // an overlay's cells against a release method count only if they were computed against that method's brackets as
  // they are now: an overlay is sealed less often than the release is refreshed
  function freshAt(a, b, slot) {
    var S = D.slots || {}, B = S.basis || {}, stamp = S.stamp || {};
    var ok = function (own, partner) { var bs = B[own] && B[own][partner]; return !bs || bs[slot] === undefined || bs[slot] === (stamp[partner] || {})[slot]; };
    return ok(a, b) && ok(b, a);
  }
  // The charts along the slots do not move with the position: their points are kept between drawings, for as long as
  // the methods, the problem sets, the metric and the cells loaded (a keyed overlay adds pairs) stay the same.
  var SLOTMEMO = {}, SLOTMEMO_N = 0;
  function slotMemo(id, fn) {
    if (id in SLOTMEMO) { return SLOTMEMO[id]; }
    if (++SLOTMEMO_N > 4000) { SLOTMEMO = {}; SLOTMEMO_N = 1; }
    return (SLOTMEMO[id] = fn());
  }
  function slotContext(cells) { return state.cats.join(",") + "|" + Object.keys(cells || {}).length + "|" + OVERLAYKEYS.length; }
  function rankCellsOf(key) { var R = window.RESULTS_V2_RANKCELLS && window.RESULTS_V2_RANKCELLS[REL]; return (R && R[key]) || null; }
  function pairCellsOf(name) { var R = window.RESULTS_V2_PAIRCELLS && window.RESULTS_V2_PAIRCELLS[REL]; return (R && R[name]) || null; }
  // A ranking at a slot from the exporter's cells. A method without cells against another (an overlay sealed before
  // the other's results changed) sits out instead of emptying the ranking: the one missing the most partners goes
  // first, a release method before one the reader added with a key.
  function rankingAtSlot(shown, slot, key) {
    var R = rankCellsOf(key) || {}, out = { roster: [], out: [], blind: [], cats: [], n: 0 }, cand = [];
    shown.forEach(function (m) { var pl = slotPlaces(m.key, slot); if (pl) { cand.push({ m: m, places: pl }); } else { out.out.push(m); } });
    var pair = function (x, z, c) {
      var e = R[x.m.key + "|" + z.m.key], flip = false; if (!e) { e = R[z.m.key + "|" + x.m.key]; flip = true; }
      var t = e && e[c] && e[c][slot]; if (!t || !freshAt(x.m.key, z.m.key, slot)) { return null; }
      return { n: t[0], s1: flip ? -t[1] : t[1], s2: t[2] };
    };
    var shared = function (ro) { return state.cats.filter(function (c) { return ro.every(function (x, i) { return ro.every(function (z, j) { return j <= i || pair(x, z, c); }); }); }); };
    var ro = cand;
    while (ro.length >= 2 && !shared(ro).length) {
      var lone = ro.map(function (x) { return ro.filter(function (z) { return z !== x && !state.cats.some(function (c) { return pair(x, z, c); }); }).length; });
      var worst = Math.max.apply(null, lone); if (!worst) { break; }
      var drop = -1; ro.forEach(function (x, i) { if (lone[i] === worst && (drop < 0 || (ro[drop].m.overlay && !x.m.overlay) || (!!ro[drop].m.overlay === !!x.m.overlay))) { drop = i; } });
      out.blind.push(ro[drop].m); ro = ro.filter(function (_x, i) { return i !== drop; });
    }
    out.roster = ro; if (ro.length < 2) { return out; }
    out.cats = shared(ro); if (!out.cats.length) { return out; }
    return rankStats(out, pair, function (x) { x.blank = 0; x.capped = false; });
  }
  // The statistics of a ranking, whatever the outcomes per problem set come from: out.roster the methods (two or more),
  // out.cats the problem sets, pair(x, z, c) the problems x and z share on c with the sum and the sum of squares of x's
  // superiority over z (or null), more(x) whatever else a method's entry carries.
  function rankStats(out, pair, more) {
    var ro = out.roster, k = ro.length;
    // On a problem, the chance that one method beats another is (1 + its superiority) / 2 (ties count half); averaged
    // over the problem sets like any value. A method's average place is 1 + the chances that each other method beats it.
    var beat = ro.map(function () { return ro.map(function () { return null; }); }), pairs = [];
    ro.forEach(function (x, i) { ro.forEach(function (z, j) { if (j <= i) { return; }
      var f = reCombine(out.cats.map(function (c) { return pair(x, z, c); }).filter(Boolean), "normal");
      if (!f) { f = { mu: 0, lo: 0, hi: 0, p: 1, n: 0, S: 0 }; }
      beat[i][j] = { p: (1 + f.mu) / 2, lo: (1 + f.lo) / 2, hi: (1 + f.hi) / 2, pv: f.p, n: f.n, S: f.S };
      beat[j][i] = { p: (1 - f.mu) / 2, lo: (1 - f.hi) / 2, hi: (1 - f.lo) / 2, pv: f.p, n: f.n, S: f.S };
      pairs.push([i, j]); }); });
    out.n = beat[0][1].n; out.beat = beat; out.k = k;
    ro.forEach(function (x, i) { var lost = 0; ro.forEach(function (_z, j) { if (j !== i) { lost += beat[j][i].p; } });
      x.rank = 1 + lost; x.share = 1 - lost / (k - 1); more(x); });
    // Which pairs differ: each pair's test, Holm-corrected over the pairs of the release's methods at this budget or time
    // limit; a method added with a key is tested in a family of its own, so it never changes a verdict between the others.
    var pub = pairs.filter(function (pr) { return !ro[pr[0]].m.overlay && !ro[pr[1]].m.overlay; }), own = pairs.filter(function (pr) { return ro[pr[0]].m.overlay || ro[pr[1]].m.overlay; });
    [pub, own].forEach(function (fam) { var adj = holm(fam.map(function (pr) { return beat[pr[0]][pr[1]].pv; }));
      fam.forEach(function (pr, t) { [beat[pr[0]][pr[1]], beat[pr[1]][pr[0]]].forEach(function (e) { e.padj = adj[t]; e.sep = adj[t] < 0.05; }); }); });
    out.pairs = pairs.length; out.separated = pairs.filter(function (pr) { return beat[pr[0]][pr[1]].sep; }).length;
    out.order = ro.slice().sort(function (a, b) { return a.rank - b.rank; });
    // bands: runs of neighbours in the order that no test separates
    var sepOf = function (x, z) { var i = ro.indexOf(x), j = ro.indexOf(z); return beat[i][j].sep; };
    out.cliques = [];
    out.order.forEach(function (x, i) { var g = [x]; for (var j = i + 1; j < out.order.length; j++) { var z = out.order[j]; if (g.some(function (y) { return sepOf(y, z); })) { break; } g.push(z); }
      if (g.length > 1 && !out.cliques.some(function (q) { return g.every(function (z) { return q.indexOf(z) >= 0; }); })) { out.cliques.push(g); } });
    return out;
  }
  function renderRanks(shown) {
    var keys = D.rank_keys || []; if (!keys.length) { return '<p class="v2hint">This release holds no rankings yet.</p>'; }
    if (keys.indexOf(state.rmetric) < 0) { state.rmetric = keys[0]; }
    var ki = keys.indexOf(state.rmetric), p = METRIC[state.rmetric], pos = curPos();
    var head = '<div class="v2viewbar"><span class="v2segwrap"><span class="v2lab">ranked on</span>' + pickButton("v2viewpick", 'data-axis="focus" aria-label="metric the methods are ranked on"', p.key) + " " + mhelp(p) +
      (ki === 0 ? ' <span class="v2tag v2tag-primary" title="The ranking metric chosen before the results were read. Which time or budget to compare at depends on the use case.">primary</span>' : ' <span class="v2tag" title="For browsing; the primary ranking metric is log10 FVU on the held-out points.">exploratory</span>') + "</span></div>" +
      '<div class="v2viewbar">' + posControl(shown) + "</div>";
    if (shown.length < 2) { return head + '<p class="v2hint">Select at least two methods to rank.</p>'; }
    var lg = rankingAt(shown, pos, state.rmetric);
    if (lg.wait) { return head + '<p class="v2hint">Loading the values of ' + lg.roster.length + " methods, problem by problem…</p>"; }
    var why = function (m) {   // one reason per method
      if (!(D.pv || {})[m.key]) { return esc(m.label) + " is not ranked: its values per problem are not published in this release."; }
      if (outside(m, pos)) { return outsideText(m, pos).replace(" has no value at ", " is not ranked: it has no value at "); }
      return esc(m.label) + " is not ranked: it has not finished all selected problem sets at the budgets around " + esc(posText(pos)) + ".";
    };
    var sitOut = lg.out.length ? '<p class="v2hint">' + lg.out.map(why).join(" ") + "</p>" : "";
    if (lg.roster.length < 2 || !lg.cats.length) { return head + '<p class="v2hint">Fewer than two of the selected methods have ' + term("complete", "finished all selected problem sets") + " at " + esc(posText(pos)) + ". Move the position above, or select fewer problem sets.</p>" + sitOut + rankLadder(shown, p, ki); }
    return head + rankDiagram(lg, p, pos) + sitOut + rankTables(lg, p, pos) + rankLadder(shown, p, ki);
  }
  function rankDiagram(lg, p, pos) {
    var nr = narrow(), W = wideWidth(), k = lg.k, Rr = nr ? 44 : 56, T = 76, rowH = 30, B = 42, H = T + k * rowH + B;
    var Lw = nr ? 138 : Math.max(190, Math.ceil(widest(lg.order.map(function (x) { return x.m.label + (x.m.local ? " (local)" : ""); })) + 28));
    var xs = function (v) { return Lw + (v - 1) / (k - 1) * (W - Lw - Rr); };
    var title = "Average place on " + p.label + " · " + posText(pos);
    var s = '<svg viewBox="0 0 ' + W + " " + H + '" class="v2chart v2distwide v2rankchart" role="img" aria-label="' + esc(title) + '"><text x="' + (nr ? 10 : Lw) + '" y="' + TITLE_Y + '" class="ct">' + esc(nr ? "Average place · " + posText(pos) : title) + "</text>";
    for (var g = 1; g <= k; g++) { s += '<line x1="' + xs(g).toFixed(1) + '" y1="' + T + '" x2="' + xs(g).toFixed(1) + '" y2="' + (H - B) + '" class="grid"/><text x="' + xs(g).toFixed(1) + '" y="' + (H - B + 16) + '" class="tick" text-anchor="middle">' + g + "</text>"; }
    s += '<text x="' + ((Lw + W - Rr) / 2).toFixed(0) + '" y="' + (H - B + 33) + '" class="tick" text-anchor="middle">average place (1 = best of ' + k + ")</text>";
    if (lg.cliques.length) { lg.cliques.forEach(function (q) { var is = q.map(function (x) { return lg.order.indexOf(x); }), rs = q.map(function (x) { return x.rank; });
      s += '<rect x="' + (xs(Math.min.apply(null, rs)) - 9).toFixed(1) + '" y="' + (T + Math.min.apply(null, is) * rowH + 3) + '" width="' + (xs(Math.max.apply(null, rs)) - xs(Math.min.apply(null, rs)) + 18).toFixed(1) + '" height="' + ((Math.max.apply(null, is) - Math.min.apply(null, is) + 1) * rowH - 6) + '" rx="6" class="v2clique"/>'; }); }
    lg.order.forEach(function (x, i) { var yy = T + (i + 0.5) * rowH, col = colorOf(x.m);
      s += '<text x="' + (Lw - 12) + '" y="' + (yy + 4) + '" class="leg" text-anchor="end">' + esc(x.m.label + (x.m.local ? " (local)" : "")) + "</text>";
      s += '<line x1="' + xs(1).toFixed(1) + '" y1="' + yy + '" x2="' + xs(x.rank).toFixed(1) + '" y2="' + yy + '" stroke="' + col + '" stroke-width="2" stroke-opacity="0.3"/>';
      s += '<circle cx="' + xs(x.rank).toFixed(1) + '" cy="' + yy + '" r="6.5" fill="' + col + '" stroke="' + col + '" stroke-width="2.5"><title>' + esc(x.m.label + ": average place " + x.rank.toFixed(2) + " of " + k + " at " + posText(pos) + "; wins " + (100 * x.share).toFixed(1) + " % of its comparisons; no usable formula in any run on " + Math.round(x.blank).toLocaleString() + " problems") + "</title></circle>";
      s += '<text x="' + (xs(x.rank) + 12).toFixed(1) + '" y="' + (yy + 4) + '" class="tick">' + x.rank.toFixed(2) + "</text>"; });
    return s + "</svg>" + '<p class="v2hint">' + k + " methods, each placed against the others on " + lg.n.toLocaleString() + " problems from " + lg.cats.length + " problem sets, averaged over the problem sets; " + term("worstrank", "how places are given") + ". " +
      (lg.cliques.length ? "Methods joined by a shaded band are not told apart by the " + term("pairtest", "pairwise tests") + "." : "The " + term("pairtest", "pairwise tests") + " tell every pair apart.") +
      (lg.order.some(function (x) { return Object.keys(x.places).some(function (c) { return x.places[c].r1 !== x.places[c].r2; }); }) ? " " + term("between", "Between two budgets") + " a method is read problem by problem from both." : "") + ' <a href="ranks.html">How ranks work</a></p>';
  }
  function rankTables(lg, p, pos) {
    var k = lg.k, ord = lg.order, idx = ord.map(function (x) { return lg.roster.indexOf(x); }), time = isTimePos(pos);
    var t1 = '<h3 class="v2h">Standings</h3><div class="v2table-wrap"><table class="v2table v2ranktable"><thead><tr><th>method</th><th>budget</th><th>average place</th><th>comparisons won ' + help(TERMS.winshare, "What is the share of comparisons won?") + "</th><th>problems with no usable formula in any run</th></tr></thead><tbody>" +
      ord.map(function (x) { return '<tr><td><span class="v2sw" style="background:' + colorOf(x.m) + '"></span>' + esc(x.m.label) + "</td><td>" + placeText(x.m, x.places, time) + "</td><td><b>" + x.rank.toFixed(2) + "</b></td><td>" + (100 * x.share).toFixed(1) + " %</td><td>" + Math.round(x.blank).toLocaleString() + ' <span class="v2ci-txt">of ' + lg.n.toLocaleString() + "</span></td></tr>"; }).join("") + "</tbody></table></div>";
    var t2 = '<h3 class="v2h">Head to head</h3><div class="v2table-wrap"><table class="v2table v2matrix v2h2h"><thead><tr><th>row beats column on</th>' + ord.map(function (x) { return '<th><span class="v2sw" style="background:' + colorOf(x.m) + '"></span>' + esc(x.m.label) + "</th>"; }).join("") + "</tr></thead><tbody>" +
      ord.map(function (x, a) { return '<tr><td><span class="v2sw" style="background:' + colorOf(x.m) + '"></span>' + esc(x.m.label) + "</td>" + ord.map(function (z, b) { if (a === b) { return '<td class="v2na">·</td>'; } var e = lg.beat[idx[a]][idx[b]], w = e.p, rgb = accentRGB();
        return '<td style="background:rgba(' + rgb.join(",") + "," + (0.04 + 0.5 * w).toFixed(2) + ')" title="' + esc(x.m.label + " does better than " + z.m.label + " on " + (100 * e.p).toFixed(1) + " % of the problems (95 % interval " + (100 * e.lo).toFixed(1) + " to " + (100 * e.hi).toFixed(1) + " %), ties counting half, averaged over " + e.S + " problem sets; " + (e.sep ? "the pairwise test tells them apart" : "the pairwise test does not tell them apart") + " (p = " + fmtP(e.padj) + ")") + '">' + (w > 0.5 && e.sep ? "<b>" : "") + (100 * w).toFixed(1) + " %" + (w > 0.5 && e.sep ? "</b>" : "") + ' <span class="v2ci-txt">[' + (100 * e.lo).toFixed(0) + ", " + (100 * e.hi).toFixed(0) + "]</span></td>"; }).join("") + "</tr>"; }).join("") + "</tbody></table></div>" +
      '<p class="v2hint">Each cell: the share of problems on which the row\u2019s method does better than the column\u2019s on ' + esc(p.label) + ", ties counting half, averaged over the problem sets, with its 95 % interval. Bold: the row\u2019s method does better, and the " + term("pairtest", "pairwise test") + " tells the two apart.</p>";
    return t1 + t2;
  }
  function rankLadder(shown, p, ki) {
    var time = state.pm === "time", file = "ranks/" + p.key + ".js", head = '<h3 class="v2h">By ' + (time ? "time" : "budget") + "</h3>";
    if (!ready(file)) { ensure(file, scheduleRender); return head + '<p class="v2hint">Loading the rankings along the ' + (time ? "times" : "budgets") + "…</p>"; }
    var by = {}, tmin = Infinity, tmax = -Infinity;
    var ctx = p.key + "|" + shown.map(function (m) { return m.key; }).join(",") + "|" + slotContext(rankCellsOf(p.key));
    slotsOf(time).forEach(function (sl) { var lg = slotMemo("rank|" + ctx + "|" + sl.slot, function () { return rankingAtSlot(shown, sl.slot, p.key); }); if (lg.roster.length < 2 || !lg.cats.length) { return; }
      if (time) { tmin = Math.min(tmin, sl.x); tmax = Math.max(tmax, sl.x); }
      lg.roster.forEach(function (e) { (by[e.m.key] = by[e.m.key] || { m: e.m, pts: [] }).pts.push({ x: sl.x, v: e.share, lo: NaN, hi: NaN, title: e.m.label + " at " + posText(time ? "t" + sl.x : sl.x) + ": wins " + (100 * e.share).toFixed(1) + " % of its comparisons, average place " + e.rank.toFixed(2) + " of " + lg.k + ", " + lg.n.toLocaleString() + " problems" }); }); });
    var series = shown.filter(function (m) { return by[m.key]; }).map(function (m) { return { label: m.label + (m.local ? " (local)" : ""), color: colorOf(m), dash: !!m.dash, pts: by[m.key].pts }; });
    if (!series.length) { return ""; }
    var title = narrow() ? "Comparisons won" : "Comparisons won, by " + (time ? "time" : "budget"), rate = { kind: "rate", fmt: "pct", hist: null }, tr = time ? timeRange(tmin, tmax) : [0, 0];
    var svg = withState({ band: false, cross: false }, function () { return chartSVG({ title: title, aria: title, width: wideWidth(), series: series, ymin: 0, ymax: 1, ticks: [0, 0.25, 0.5, 0.75, 1], tick: function (g) { return tickLabel(rate, g); }, ylabel: narrow() ? "won" : "comparisons won", timeAxis: time, timeSource: "ref", tmin: tr[0], tmax: tr[1], zero: 0.5, xlabel: time ? (narrow() ? "time (s)" : "time per problem (s, log scale)") : (narrow() ? "budget" : "budget per problem (log scale)") }); });
    return head + '<div class="v2charts v2one">' + svg + "</div>" +
      '<p class="v2hint">' + term("winshare", "Comparisons won") + ": the share of one-on-one comparisons a method wins. 100 % means it beats every other method on every problem; 50 % means it wins as often as it loses. At every " + (time ? "time" : "budget") + ", each method is read as at the position above, between two of its budgets too.</p>";
  }


  // ---- Correlations: two metrics, problem by problem ----------------------------------------------------------------
  // Every other display summarises the problems; this one shows them. The release ships every problem's value of every
  // continuous metric, one byte per run, per method x problem set x budget x finished run (D.pp indexes the files;
  // scripts/site_export_v2.py, write_values): the value's place on its metric's grid (METRIC.pp: lo and hi in the space
  // the metric's histograms use, whole numbers exact), 253 below the grid, 254 above it, 255 no value. A run weighs what
  // it weighs in a distribution: its problem's share of its problem set, the set weighted as in an average, and a
  // problem's runs share its weight. The set weights are those of log10 FVU, whatever the axes show, so that changing an
  // axis never re-weighs the problems.
  // A value that piles up at a bound of its metric (an exact fit's log10 FVU, an overlap of 0 or 1), and a value beyond
  // the range on screen, is drawn in a strip along the frame instead of inside it: inside, a smoothed density or a colour
  // scale would turn a pile-up into a blob, or wash out everything else.
  var PP_STEPS = 252, PP_BELOW = 253, PP_ABOVE = 254, PP_NONE = 255, PP_OK = 1, PP_NUM = 2, PP_SYM = 4;
  var TIME_BUDGETS = [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000];   // the exporter's TIME_BUDGETS, as the Ranks view uses them
  var CMODES = [["contour", "contours", "Where each method's runs lie: the smallest regions that hold half and nine tenths of its runs, every method on one chart"],
    ["heat", "heat maps", "One panel per method: the share of its runs in each cell of a grid"],
    ["points", "points", "Every run as a point; tap or hover one to see its problem"],
    ["trend", "trend", "Along the x axis, in bins: the median of the y axis and the middle half of the runs"],
    ["matrix", "all pairs", "The rank correlation of every pair of metrics, for two methods at once", "pairs"],
    ["vs", "method vs method", "One metric with one method on each axis, problem by problem", "A vs B"]];
  var CTERMS = {
    weights: "Each point is one run of a method on one problem. A run weighs what it weighs in a distribution: every problem set counts as it does in an average of log10 FVU (each about once, whatever its size, and a set of a few problems less), every problem of a set the same, and a problem's two runs share its weight. Without this, the largest problem set would make up four fifths of every chart.",
    strips: "Values that pile up at a bound of their metric, such as the log10 FVU of an exact fit or an overlap of exactly 0 or 1, and values beyond the range on screen, are drawn in a strip along the frame instead of inside it. Inside, a smoothed region or a colour scale would turn a pile-up into a blob, or wash out everything else. The table below the chart gives the share of each method's runs in every strip.",
    rho: "Spearman's rank correlation over the runs, weighted as in the charts: 1 when the two metrics put the runs in the same order, -1 in the opposite order, and near 0 when the order on one does not follow the order on the other. Ties share a rank, and a value in a strip counts as lower or higher than every value inside the frame. The 95 % interval treats each problem as one observation.",
    cbetween: "Between two budgets a method was run at, its runs at both are drawn: each run of the lower budget weighs (1 \u2212 w) of its usual weight and each run of the upper one w, where w is how far the position lies from the lower budget towards the upper one on a logarithmic scale. Every point is a run the method made, as in a distribution that blends the two budgets' histograms. The legend names the two budgets (\u2248). Slide to any time or budget, or type one into the box; the arrows step to the next time limit or budget.",
    hdr: "The darker region is the smallest part of the plane that holds half of a method's runs inside the frame, the outline the smallest that holds nine tenths. They are drawn from a smoothed density, so their edges are approximate."
  };
  Object.assign(TERMS, CTERMS);   // a dotted term looks its text up in TERMS
  var csel = null;   // the run picked in the points display: {m, c, row, d, r}
  function ppStore() { var R = window.RESULTS_V2_PP; return (R && R[REL]) || {}; }
  var PPDEC = {};
  function ppPart(key) {
    var raw = ppStore()[key]; if (!raw) { return null; }
    var e = PPDEC[key]; if (!e || e.raw !== raw) { e = PPDEC[key] = { raw: raw, n: raw.n, s: raw.s ? bytesOf(raw.s) : null, v: {} }; }
    return e;
  }
  function ppCol(part, k) { if (!part || !part.raw.v || !part.raw.v[k]) { return null; } return part.v[k] || (part.v[k] = bytesOf(part.raw.v[k])); }
  function ppMetrics() { return D.metrics.filter(function (m) { return m.kind === "cont" && m.pp; }); }
  function ppValue(sp, c) { return sp.int ? sp.lo + c : sp.lo + c / PP_STEPS * (sp.hi - sp.lo); }
  function ppBound(sp, side) { var b = sp.bounds && sp.bounds[side]; return b === null || b === undefined || sp.int ? -1 : Math.round((b - sp.lo) / (sp.hi - sp.lo) * PP_STEPS); }
  function ppHas(k, r) { var idx = (D.pp || {})[k]; return !!idx && state.cats.length > 0 && state.cats.every(function (c) { return (idx[c + "|" + r] || []).length > 0; }); }
  function ppRungs(k) { return D.rungs.filter(function (r) { return ppHas(k, r); }); }
  // Where a method is read at the position: at a budget whose values are published for every selected problem set, or
  // between two of them. Between two, every run of the lower budget weighs (1 - w) of what it weighs there and every run
  // of the upper one w, w by where the position lies between them on a logarithmic scale: the two budgets' runs are
  // blended as a distribution blends their histograms, and every point drawn is a run the method made.
  function corrAt(m, pos) {
    var time = isTimePos(pos), ps = [];
    ppRungs(m.key).forEach(function (r) { var v = time ? refTime(m.key, r) : r; if (v > 0) { ps.push([v, r]); } });
    var b = bracketIn(ps.sort(function (a, z) { return a[0] - z[0]; }), posValue(pos));
    return b ? { r1: b.r1, r2: b.r2, w: b.w, pos: pos } : null;
  }
  function corrWhy(m, pos) {
    if (!(D.pp || {})[m.key]) { return esc(m.label) + ": its values per problem are not published in this release."; }
    if (outside(m, pos)) { return outsideText(m, pos); }
    return esc(m.label) + " has not finished every run of the selected problem sets at the budgets around " + esc(posText(pos)) + ".";
  }
  function corrFiles(k, r, ks) {
    var idx = D.pp[k], out = [];
    state.cats.forEach(function (c) { (idx[c + "|" + r] || []).forEach(function (d) { out.push("pp/" + k + "/" + c + "/" + r + "." + d + ".js"); }); });
    if (ks.some(function (q) { return METRIC[q].every; })) { state.cats.forEach(function (c) { out.push("pp/truth/" + c + ".js"); }); }
    return out;
  }
  function filesReady(files) { var all = true; files.forEach(function (f) { if (!ready(f)) { all = false; ensure(f, scheduleRender); } }); return all; }
  // One method's runs at the position b (corrAt), with the columns of the metrics ks: its runs at one budget, or at the
  // two around the position, each run's weight scaled by its budget's share (rr: each run's budget).
  function ppCloudAt(m, b, ks) {
    if (b.r1 === b.r2) { return ppCloud(m, b.r1, ks); }
    var lo = ppCloud(m, b.r1, ks), hi = ppCloud(m, b.r2, ks), n = lo.n + hi.n, out = { m: m, r: b.pos, between: [b.r1, b.r2], cs: lo.cs, n: n, cols: {}, rr: new Float64Array(n) }, i;
    ["cat", "row", "drw", "flag", "share"].forEach(function (k) { var a = new lo[k].constructor(n); a.set(lo[k].subarray(0, lo.n)); a.set(hi[k].subarray(0, hi.n), lo.n); out[k] = a; });
    ks.forEach(function (k) { var a = new Uint8Array(n); a.set(lo.cols[k].subarray(0, lo.n)); a.set(hi.cols[k].subarray(0, hi.n), lo.n); out.cols[k] = a; });
    for (i = 0; i < n; i++) { var up = i >= lo.n; out.share[i] *= up ? b.w : 1 - b.w; out.rr[i] = up ? b.r2 : b.r1; }
    return out;
  }
  function runRung(cl, i) { return cl.rr ? cl.rr[i] : cl.r; }
  // One method's runs at budget r over the selected problem sets, with the columns of the metrics ks.
  function ppCloud(m, r, ks) {
    var idx = D.pp[m.key], cs = state.cats.slice(), n = 0, at = 0, cols = {};
    cs.forEach(function (c) { n += (idx[c + "|" + r] || []).length * CAT[c].laws; });
    var cat = new Uint16Array(n), row = new Int32Array(n), drw = new Uint8Array(n), flag = new Uint8Array(n), share = new Float64Array(n);
    ks.forEach(function (k) { cols[k] = new Uint8Array(n).fill(PP_NONE); });
    cs.forEach(function (c, ci) {
      var ds = idx[c + "|" + r] || [], truth = ppPart("truth|" + c);
      ds.forEach(function (d) {
        var part = ppPart(m.key + "|" + c + "|" + r + "|" + d); if (!part) { return; }
        var N = Math.min(part.n, n - at);
        ks.forEach(function (k) { var col = METRIC[k].every ? ppCol(truth, k) : ppCol(part, k); if (col) { cols[k].set(col.subarray(0, N), at); } });
        for (var i = 0; i < N; i++) { cat[at + i] = ci; row[at + i] = i; drw[at + i] = d; flag[at + i] = part.s ? part.s[i] : PP_OK; share[at + i] = 1 / ds.length; }
        at += N;
      });
    });
    return { m: m, r: r, cs: cs, n: at, cat: cat, row: row, drw: drw, flag: flag, share: share, cols: cols };
  }
  // whether a run has a value on metric k as the reader counts it (a failed run left out of an overlap metric on request)
  function ppOk(cl, k, i) { return cl.cols[k][i] !== PP_NONE && !(leftOut(k) && !(cl.flag[i] & PP_OK)); }
  // Every run's weight (0 where a metric of ks has no value), summing to 1; n: the problems behind them; neff: the number
  // of equally weighted problems that would be as precise (Kish), for the intervals.
  function ppWeights(cl, ks) {
    var w = new Float64Array(cl.n), cnt = cl.cs.map(function () { return 0; }), ok = new Uint8Array(cl.n), i, j;
    for (i = 0; i < cl.n; i++) { var good = true; for (j = 0; j < ks.length; j++) { if (!ppOk(cl, ks[j], i)) { good = false; break; } } if (good) { ok[i] = 1; cnt[cl.cat[i]] += cl.share[i]; } }
    var weigh = setWeights("log10_fvu_val", cl.m.key, cl.r, cl.cs), W = cnt.map(function (x) { return x > 0 ? weigh(x) : 0; });
    var tot = W.reduce(function (a, b) { return a + b; }, 0), q = 0;
    W.forEach(function (x, ci) { if (cnt[ci] > 0) { q += x * x / cnt[ci]; } });
    for (i = 0; i < cl.n; i++) { if (ok[i] && tot > 0) { w[i] = W[cl.cat[i]] * cl.share[i] / cnt[cl.cat[i]] / tot; } }
    var all = cl.cs.reduce(function (a, c) { return a + CAT[c].laws; }, 0), have = cnt.reduce(function (a, b) { return a + b; }, 0);
    return { w: w, n: have, all: all, neff: q > 0 ? tot * tot / q : 0 };
  }
  // An axis: the range on screen, from where 99 % of the values inside the frame lie (as in the Distribution view), and
  // the bytes of the metric's bounds.
  function ppAxis(k, clouds, ws) {
    var sp = METRIC[k].pp, bl = ppBound(sp, 0), bh = ppBound(sp, 1), h = new Float64Array(PP_STEPS + 1), tot = 0;
    clouds.forEach(function (cl, j) { var col = cl.cols[k], w = ws[j].w; for (var i = 0; i < cl.n; i++) { var c = col[i]; if (w[i] > 0 && c <= PP_STEPS && c !== bl && c !== bh) { h[c] += w[i]; tot += w[i]; } } });
    var lo = sp.lo, hi = sp.hi;
    if (tot > 0) {
      var q = function (p) { var cum = 0; for (var c = 0; c <= PP_STEPS; c++) { cum += h[c]; if (cum >= p * tot - 1e-12) { return c; } } return PP_STEPS; };
      var a = ppValue(sp, q(0.005)), b = ppValue(sp, q(0.995));
      if (sp.int) { lo = a - 0.5; hi = b + 0.5; }
      else { var step = (sp.hi - sp.lo) / PP_STEPS, pad = Math.max(0.04 * (b - a), 2 * step); lo = Math.max(sp.lo, a - pad); hi = Math.min(sp.hi, b + pad); }
    }
    if (!(hi > lo)) { hi = lo + 1; }
    return { k: k, m: METRIC[k], sp: sp, lo: lo, hi: hi, bl: bl, bh: bh };
  }
  // where a run falls on an axis: -1 the low strip, 1 the high strip, 0 inside the frame, null no value
  function ppPlace(A, c) {
    if (c === PP_NONE) { return null; }
    if (c === PP_BELOW || c === A.bl) { return -1; }
    if (c === PP_ABOVE || c === A.bh) { return 1; }
    var v = ppValue(A.sp, c);
    return v < A.lo ? -1 : v > A.hi ? 1 : 0;
  }
  function vText(A, x) { return A.sp.int ? String(Math.round(back(A.m, x))) : fmt(A.m, x); }
  function codeText(A, c) {
    if (c === PP_NONE) { return "no value"; }
    if (c === PP_BELOW) { return A.m.key.indexOf("log10_fvu") === 0 ? "below " + vText(A, A.sp.lo) : "below " + vText(A, A.sp.lo) + " or minus infinity"; }
    if (c === PP_ABOVE) { return "above " + vText(A, A.sp.hi) + ", or no finite value"; }
    return (c === A.bl || c === A.bh ? "exactly " : "") + vText(A, ppValue(A.sp, c));
  }
  // what a strip holds: the bound, when only values at the bound are in it, else everything beyond the range on screen
  function stripLabel(A, side, onlyBound) {
    if (onlyBound) { return "= " + vText(A, A.sp.bounds[side < 0 ? 0 : 1]); }
    return (side < 0 ? "≤ " : "≥ ") + vText(A, side < 0 ? A.lo : A.hi);
  }
  function hexRGB(s) {
    s = String(s || "").trim(); var m6 = /^#([0-9a-f]{6})$/i.exec(s), m3 = /^#([0-9a-f]{3})$/i.exec(s);
    if (m6) { return [0, 2, 4].map(function (i) { return parseInt(m6[1].slice(i, i + 2), 16); }); }
    if (m3) { return [0, 1, 2].map(function (i) { return parseInt(m3[1].charAt(i) + m3[1].charAt(i), 16); }); }
    return [128, 128, 128];
  }
  function cssVar(name) { return window.getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
  function mixRGB(a, b, t) { return [0, 1, 2].map(function (i) { return Math.round(t * a[i] + (1 - t) * b[i]); }); }
  function rgbCss(c, alpha) { return alpha === undefined ? "rgb(" + c.join(",") + ")" : "rgba(" + c.join(",") + "," + alpha.toFixed(3) + ")"; }
  function methodRGB(m) { var c = hexRGB(baseColorOf(m)), a = alphaOf(m.key); return a < 1 ? mixRGB(c, hexRGB(cssVar("--surface")), a) : c; }
  // A raster drawn on a canvas and placed in the chart as an image: points and grids by the thousand stay one element.
  function rasterHref(w, h, draw) {
    var dpr = Math.min(2, window.devicePixelRatio || 1), cv = document.createElement("canvas");
    cv.width = Math.max(1, Math.round(w * dpr)); cv.height = Math.max(1, Math.round(h * dpr));
    var g = cv.getContext("2d"); if (!g) { return ""; } g.scale(dpr, dpr); draw(g);
    try { return cv.toDataURL("image/png"); } catch (e) { return ""; }
  }
  // ---- the frame: the plot area, a strip on each side that holds values, ticks that keep clear of the strip labels ----
  function corrFrame(W, H, A, B, has, opt) {
    var nr = narrow(), Lm = opt.left || (nr ? 52 : 70), Tm = opt.top || 14, Rm = opt.right || 14, Bm = 50 + (opt.below || 0), ST = opt.strip || 16, gap = 4;
    var f = { W: W, H: H, A: A, B: B, has: has, ST: ST };
    f.x0 = Lm + (has.xl ? ST + gap : 0); f.x1 = W - Rm - (has.xh ? ST + gap : 0);
    f.yb = H - Bm; f.y0 = Tm + (has.yh ? ST + gap : 0); f.y1 = f.yb - (has.yl ? ST + gap : 0);
    f.sxl = [Lm, Lm + ST]; f.sxh = [W - Rm - ST, W - Rm]; f.syh = [Tm, Tm + ST]; f.syl = [f.yb - ST, f.yb];
    f.xs = function (v) { return f.x0 + (v - A.lo) / (A.hi - A.lo) * (f.x1 - f.x0); };
    f.ys = function (v) { return f.y1 - (v - B.lo) / (B.hi - B.lo) * (f.y1 - f.y0); };
    f.xv = function (px) { return A.lo + (px - f.x0) / (f.x1 - f.x0) * (A.hi - A.lo); };
    f.yv = function (py) { return B.lo + (f.y1 - py) / (f.y1 - f.y0) * (B.hi - B.lo); };
    return f;
  }
  // a run's spot: inside, its values; in a strip, the strip's middle across and its value along (u in [0, 1) spreads it across)
  // inside the frame, d in [0, 1) places a continuous value within its grid step (it is known to that precision)
  function dither(sp, c, d) { return d === undefined || sp.int || c > PP_STEPS ? 0 : (d - 0.5) * (sp.hi - sp.lo) / PP_STEPS; }
  function spotX(f, px, c, u, d) { return px === 0 ? f.xs(ppValue(f.A.sp, c) + dither(f.A.sp, c, d)) : px < 0 ? f.sxl[0] + (u === undefined ? 0.5 : u) * f.ST : f.sxh[0] + (u === undefined ? 0.5 : u) * f.ST; }
  function spotY(f, py, c, u, d) { return py === 0 ? f.ys(ppValue(f.B.sp, c) + dither(f.B.sp, c, d)) : py < 0 ? f.syl[0] + (u === undefined ? 0.5 : u) * f.ST : f.syh[0] + (u === undefined ? 0.5 : u) * f.ST; }
  function frameSVG(f, labels) {
    var A = f.A, B = f.B, s = "", nr = narrow();
    var strip = function (x, y, w, h) { return '<rect x="' + x.toFixed(1) + '" y="' + y.toFixed(1) + '" width="' + w.toFixed(1) + '" height="' + h.toFixed(1) + '" class="v2cstrip"/>'; };
    if (f.has.xl) { s += strip(f.sxl[0], f.y0, f.ST, f.y1 - f.y0); } if (f.has.xh) { s += strip(f.sxh[0], f.y0, f.ST, f.y1 - f.y0); }
    if (f.has.yl) { s += strip(f.x0, f.syl[0], f.x1 - f.x0, f.ST); } if (f.has.yh) { s += strip(f.x0, f.syh[0], f.x1 - f.x0, f.ST); }
    // the strips' labels take their places first; a tick label that would crowd one gives way
    var xres = [], yres = [];
    if (f.has.xl) { xres.push({ at: f.sxl[0] + f.ST / 2, t: labels.xl, anchor: "end" }); } if (f.has.xh) { xres.push({ at: f.sxh[0] + f.ST / 2, t: labels.xh, anchor: "start" }); }
    if (f.has.yl) { yres.push({ at: f.syl[0] + f.ST / 2, t: labels.yl }); } if (f.has.yh) { yres.push({ at: f.syh[0] + f.ST / 2, t: labels.yh }); }
    var xt = thinTicks(ticksFor(A.m, A.lo, A.hi, nr ? 4 : 6).filter(function (g) { return g >= A.lo - 1e-9 && g <= A.hi + 1e-9; }), f.xs, nr ? 40 : 52, roundness)
      .filter(function (g) { return xres.every(function (r) { return Math.abs(f.xs(g) - r.at) >= 12 + textWidth(r.t) / 2 + textWidth(tickLabel(A.m, g)) / 2; }); });
    var yt = thinTicks(ticksFor(B.m, B.lo, B.hi, nr ? 4 : 5).filter(function (g) { return g >= B.lo - 1e-9 && g <= B.hi + 1e-9; }), f.ys, 20, roundness)
      .filter(function (g) { return yres.every(function (r) { return Math.abs(f.ys(g) - r.at) >= 15; }); });
    xt.forEach(function (g) { var x = f.xs(g).toFixed(1); s += '<line x1="' + x + '" y1="' + f.y0 + '" x2="' + x + '" y2="' + f.y1 + '" class="grid"/><text x="' + x + '" y="' + (f.yb + 16) + '" class="tick" text-anchor="middle">' + esc(A.sp.int ? String(Math.round(back(A.m, g))) : tickLabel(A.m, g)) + "</text>"; });
    yt.forEach(function (g) { var y = f.ys(g).toFixed(1); s += '<line x1="' + f.x0 + '" y1="' + y + '" x2="' + f.x1 + '" y2="' + y + '" class="grid"/><text x="' + (Math.min(f.x0, f.has.xl ? f.sxl[0] : f.x0) - 6) + '" y="' + (+y + 4) + '" class="tick" text-anchor="end">' + esc(B.sp.int ? String(Math.round(back(B.m, g))) : tickLabel(B.m, g)) + "</text>"; });
    xres.forEach(function (r) { s += '<text x="' + r.at.toFixed(1) + '" y="' + (f.yb + 16) + '" class="tick v2cstriplab" text-anchor="middle">' + esc(r.t) + "</text>"; });
    yres.forEach(function (r) { s += '<text x="' + ((f.has.xl ? f.sxl[0] : f.x0) - 6) + '" y="' + (r.at + 4).toFixed(1) + '" class="tick v2cstriplab" text-anchor="end">' + esc(r.t) + "</text>"; });
    s += '<rect x="' + f.x0 + '" y="' + f.y0 + '" width="' + (f.x1 - f.x0) + '" height="' + (f.y1 - f.y0) + '" class="v2cframe"/>';
    s += '<text x="' + ((f.x0 + f.x1) / 2).toFixed(0) + '" y="' + (f.yb + 34) + '" class="tick" text-anchor="middle">' + esc(labels.x || axisName(A.m)) + "</text>";
    s += '<text transform="translate(' + (narrow() ? 12 : 16) + "," + ((f.y0 + f.y1) / 2).toFixed(0) + ') rotate(-90)" class="tick" text-anchor="middle">' + esc(labels.y || axisName(B.m)) + "</text>";
    return s;
  }
  // which strips the runs of these clouds need, and what each holds
  function stripsOf(A, B, clouds, ws) {
    var has = { xl: false, xh: false, yl: false, yh: false }, bound = { xl: true, xh: true, yl: true, yh: true };
    clouds.forEach(function (cl, j) { var ca = cl.cols[A.k], cb = cl.cols[B.k], w = ws[j].w;
      for (var i = 0; i < cl.n; i++) { if (!(w[i] > 0)) { continue; } var px = ppPlace(A, ca[i]), py = ppPlace(B, cb[i]);
        if (px < 0) { has.xl = true; if (ca[i] !== A.bl) { bound.xl = false; } } else if (px > 0) { has.xh = true; if (ca[i] !== A.bh) { bound.xh = false; } }
        if (py < 0) { has.yl = true; if (cb[i] !== B.bl) { bound.yl = false; } } else if (py > 0) { has.yh = true; if (cb[i] !== B.bh) { bound.yh = false; } } } });
    return { has: has, labels: { xl: stripLabel(A, -1, bound.xl), xh: stripLabel(A, 1, bound.xh), yl: stripLabel(B, -1, bound.yl), yh: stripLabel(B, 1, bound.yh) } };
  }
  // where a strip on each axis meets, a square holds the runs that are in both (both exact fits, say)
  function cornersSVG(f, corner, color, title) {
    var s = "";
    Object.keys(corner).forEach(function (k) { var v = corner[k], xk = k.charAt(0) === "l" ? "xl" : "xh", yk = k.charAt(1) === "l" ? "yl" : "yh";
      if (!(v > 0) || !f.has[xk] || !f.has[yk]) { return; }
      var x = xk === "xl" ? f.sxl[0] : f.sxh[0], y = yk === "yl" ? f.syl[0] : f.syh[0];
      s += '<rect x="' + x.toFixed(1) + '" y="' + y.toFixed(1) + '" width="' + f.ST + '" height="' + f.ST + '" rx="2" fill="' + color(v) + '" class="v2ccorner"><title>' + esc(title(v)) + "</title></rect>"; });
    return s;
  }
  // ---- statistics on the runs ------------------------------------------------------------------------------------
  function ppOrder(c) { return c === PP_BELOW ? 0 : c === PP_ABOVE ? 254 : c + 1; }   // value order, strips at the ends
  function midRanks(h, tot) { var out = new Float64Array(h.length), cum = 0; for (var i = 0; i < h.length; i++) { out[i] = (cum + h[i] / 2) / tot; cum += h[i]; } return out; }
  // Spearman's rho of two metrics over the runs that have both, with the weights w (renormalised over those runs), and
  // a 95 % interval from Fisher's z with the Bonett-Wright variance on the effective number of problems.
  function spearman(cl, w, ka, kb, neff) {
    var ca = cl.cols[ka], cb = cl.cols[kb], ha = new Float64Array(255), hb = new Float64Array(255), tot = 0, i;
    for (i = 0; i < cl.n; i++) { if (w[i] > 0 && ppOk(cl, ka, i) && ppOk(cl, kb, i)) { ha[ppOrder(ca[i])] += w[i]; hb[ppOrder(cb[i])] += w[i]; tot += w[i]; } }
    if (!(tot > 0)) { return null; }
    var ra = midRanks(ha, tot), rb = midRanks(hb, tot), ma = 0, mb = 0, sab = 0, saa = 0, sbb = 0;
    for (i = 0; i < cl.n; i++) { if (w[i] > 0 && ppOk(cl, ka, i) && ppOk(cl, kb, i)) { var x = ra[ppOrder(ca[i])], y = rb[ppOrder(cb[i])], wi = w[i] / tot; ma += wi * x; mb += wi * y; } }
    for (i = 0; i < cl.n; i++) { if (w[i] > 0 && ppOk(cl, ka, i) && ppOk(cl, kb, i)) { var dx = ra[ppOrder(ca[i])] - ma, dy = rb[ppOrder(cb[i])] - mb, wj = w[i] / tot; sab += wj * dx * dy; saa += wj * dx * dx; sbb += wj * dy * dy; } }
    if (!(saa > 0 && sbb > 0)) { return null; }
    var rho = Math.max(-1, Math.min(1, sab / Math.sqrt(saa * sbb))), n = neff * tot, out = { rho: rho, n: n, lo: null, hi: null };
    if (n > 4 && Math.abs(rho) < 1) { var z = 0.5 * Math.log((1 + rho) / (1 - rho)), se = Math.sqrt((1 + rho * rho / 2) / (n - 3)); out.lo = Math.tanh(z - Z * se); out.hi = Math.tanh(z + Z * se); }
    return out;
  }
  function rhoText(st) { return st ? "ρ = " + st.rho.toFixed(2) + (st.lo !== null ? " [" + st.lo.toFixed(2) + ", " + st.hi.toFixed(2) + "]" : "") : "ρ: –"; }
  function wQuantiles(vals, ws, qs) {   // weighted quantiles of values that may be +-Infinity
    var ix = vals.map(function (_v, i) { return i; }).sort(function (a, b) { return vals[a] - vals[b]; }), tot = 0;
    ix.forEach(function (i) { tot += ws[i]; });
    return qs.map(function (q) { var cum = 0; for (var j = 0; j < ix.length; j++) { cum += ws[ix[j]]; if (cum >= q * tot - 1e-12) { return vals[ix[j]]; } } return vals[ix[ix.length - 1]]; });
  }
  function hash01(a, b, c) { var h = (a * 73856093) ^ (b * 19349663) ^ (c * 83492791); h = (h ^ (h >>> 13)) * 1274126177; return ((h ^ (h >>> 16)) >>> 0) / 4294967296; }
  // ---- hover: one handler per chart, by id; it maps a position in the chart to a sentence, or null ----------------------
  var HOVERS = {}, hoverSeq = 0, tipEl = null;
  function hoverId(fn) { var id = "c" + (++hoverSeq); HOVERS[id] = fn; return id; }
  function hideTip() { if (tipEl) { tipEl.remove(); tipEl = null; } }
  function chartPoint(svg, e) { var r = svg.getBoundingClientRect(), vb = svg.viewBox.baseVal; return { x: (e.clientX - r.left) * vb.width / r.width, y: (e.clientY - r.top) * vb.height / r.height }; }
  function showTip(text, e) {
    if (!tipEl) { tipEl = document.createElement("div"); tipEl.className = "v2pop v2ctip"; tipEl.setAttribute("role", "status"); document.body.appendChild(tipEl); }
    tipEl.textContent = text; var margin = 8, w = tipEl.offsetWidth, h = tipEl.offsetHeight;
    var left = Math.min(Math.max(margin, e.clientX + 14), window.innerWidth - w - margin), top = e.clientY + 16;
    if (top + h > window.innerHeight - margin) { top = Math.max(margin, e.clientY - h - 12); }
    tipEl.style.left = left + "px"; tipEl.style.top = top + "px";
  }
  // ---- the view --------------------------------------------------------------------------------------------------------
  function corrHead(shown, used) {
    var mode = state.cv, x = METRIC[state.cx], y = METRIC[state.cy];
    var bar = '<div class="v2viewbar">' + seg("cv", mode, CMODES, "show", "how the runs are drawn") + "</div>";
    var axes;
    if (mode === "matrix" || mode === "vs") {
      var opts = function (cur) { return used.map(function (u) { return '<option value="' + esc(u.m.key) + '"' + (u.m.key === cur ? " selected" : "") + ">" + esc(u.m.label) + "</option>"; }).join(""); };
      axes = (mode === "vs" ? '<span class="v2segwrap"><span class="v2lab">metric</span>' + pickButton("v2viewpick", 'data-axis="cy" aria-label="metric compared"', y.key) + " " + mhelp(y) + "</span>" : "") +
        '<span class="v2segwrap"><span class="v2lab">' + (mode === "vs" ? "x axis" : "lower left") + '</span><select class="v2stepsel" data-state="ca" aria-label="' + (mode === "vs" ? "method on the x axis" : "method in the lower left triangle") + '">' + opts(state.ca) + "</select></span>" +
        '<span class="v2segwrap"><span class="v2lab">' + (mode === "vs" ? "y axis" : "upper right") + '</span><select class="v2stepsel" data-state="cb" aria-label="' + (mode === "vs" ? "method on the y axis" : "method in the upper right triangle") + '">' + opts(state.cb) + "</select></span>";
    } else {
      axes = '<span class="v2segwrap"><span class="v2lab">x</span>' + pickButton("v2viewpick", 'data-axis="cx" aria-label="metric on the x axis"', x.key) + " " + mhelp(x) + "</span>" +
        '<button type="button" class="v2swap" data-act="corr-swap" aria-label="Swap the two axes" title="Swap the two axes">⇆</button>' +
        '<span class="v2segwrap"><span class="v2lab">y</span>' + pickButton("v2viewpick", 'data-axis="cy" aria-label="metric on the y axis"', y.key) + " " + mhelp(y) + "</span>";
    }
    return bar + '<div class="v2viewbar">' + axes + "</div>" + '<div class="v2viewbar">' + posControl(shown, help(CTERMS.cbetween, "How is a method read between two budgets here?")) + "</div>";
  }
  function renderCorr(shown) {
    HOVERS = {}; hideTip();
    if (!D.pp || !Object.keys(D.pp).length) { return '<p class="v2hint">The values per problem are not published in this release yet.</p>'; }
    if (!shown.length) { return '<p class="v2hint">Select at least one method.</p>'; }
    if (!state.cats.length) { return '<p class="v2hint">Select at least one problem set in the side panel.</p>'; }
    var pos = curPos(), used = [], gone = [];
    shown.forEach(function (m) { var b = corrAt(m, pos); if (b === null) { gone.push(m); } else { used.push({ m: m, b: b }); } });
    if (state.cv === "matrix" || state.cv === "vs") {
      var keys = used.map(function (u) { return u.m.key; });
      if (keys.indexOf(state.ca) < 0) { state.ca = keys[0] || null; }
      if (keys.indexOf(state.cb) < 0 || (state.cb === state.ca && keys.length > 1)) { state.cb = keys.filter(function (k) { return k !== state.ca; })[0] || state.ca; }
    }
    var head = corrHead(shown, used), missing = gone.length ? '<p class="v2hint">' + gone.map(function (m) { return corrWhy(m, pos); }).join(" ") + "</p>" : "";
    if (!used.length) { return head + '<p class="v2hint">No selected method has values for every run of the selected problem sets at ' + esc(posText(pos)) + ". Move the position above, or select fewer problem sets.</p>" + missing; }
    var ks = state.cv === "matrix" ? ppMetrics().map(function (m) { return m.key; }) : state.cv === "vs" ? [state.cy] : [state.cx, state.cy];
    var pick = state.cv === "matrix" || state.cv === "vs" ? used.filter(function (u) { return u.m.key === state.ca || u.m.key === state.cb; }) : used;
    var files = []; pick.forEach(function (u) { files = files.concat(corrFiles(u.m.key, u.b.r1, ks)); if (u.b.r2 !== u.b.r1) { files = files.concat(corrFiles(u.m.key, u.b.r2, ks)); } });
    if (!filesReady(files)) { return head + '<p class="v2hint">Loading the values of ' + pick.length + (pick.length === 1 ? " method" : " methods") + "…</p>" + missing; }
    var clouds = pick.map(function (u) { return ppCloudAt(u.m, u.b, ks); });
    var body = state.cv === "matrix" ? corrMatrix(clouds, ks) : state.cv === "vs" ? corrVs(clouds) : corrPlane(clouds);
    return head + body + missing;
  }
  function corrLabel(cl) { return cl.m.label + (cl.m.local ? " (local)" : "") + (cl.between ? " · ≈ budgets " + fmtBudget(cl.between[0]) + "–" + fmtBudget(cl.between[1]) : isTimePos(curPos()) ? " · budget " + fmtBudget(cl.r) : ""); }
  // The four displays of two metrics: contours, heat maps, points, trend.
  function corrPlane(clouds) {
    var ka = state.cx, kb = state.cy;
    var ws = clouds.map(function (cl) { return ppWeights(cl, [ka, kb]); });
    var A = ppAxis(ka, clouds, ws), B = ppAxis(kb, clouds, ws), st = stripsOf(A, B, clouds, ws);
    var rhos = clouds.map(function (cl, j) { return spearman(cl, ws[j].w, ka, kb, ws[j].neff); });
    var chart = state.cv === "heat" ? corrHeat(clouds, ws, A, B, st) : state.cv === "points" ? corrPoints(clouds, ws, A, B, st) : state.cv === "trend" ? corrTrend(clouds, ws, A, B, st) : corrContours(clouds, ws, A, B, st);
    return chart + corrTable(clouds, ws, A, B, st, rhos) + corrSelected(A, B);
  }
  function legendSVG(clouds, f, W) {   // the methods' names beside the chart, or under it on a narrow screen
    var s = "", nr = narrow(), lx = nr ? f.x0 : W - legendRight(clouds.map(corrLabel)) + LEG_GAP, ly = nr ? f.yb + 58 : f.y0 + 6;
    behind(clouds.map(function (cl) { return { key: cl.m.key, label: corrLabel(cl), color: rgbCss(methodRGB(cl.m)), dash: !!cl.m.dash }; })).slice().sort(function (a, b) { return clouds.map(function (c) { return c.m.key; }).indexOf(a.key) - clouds.map(function (c) { return c.m.key; }).indexOf(b.key); })
      .forEach(function (sr) { s += legendItem(sr, lx, ly, false, false); ly += 20; });
    return s;
  }
  function planeSize(clouds, legend) {
    var nr = narrow(), W = wideWidth(), R = legend && !nr ? legendRight(clouds.map(corrLabel)) : (nr ? 14 : 18);
    var below = legend && nr ? 20 * clouds.length + 12 : 0, H = Math.round(Math.min(560, Math.max(300, (W - R) * 0.62))) + below;
    return { W: W, H: H, right: R, below: below };
  }
  function corrContours(clouds, ws, A, B, st) {
    var k = clouds.length, sz = planeSize(clouds, true), ST = Math.max(16, 6 * k + 6);
    var f = corrFrame(sz.W, sz.H, A, B, st.has, { right: sz.right, strip: ST, below: sz.below });
    var s = '<svg viewBox="0 0 ' + sz.W + " " + sz.H + '" class="v2chart v2corr" role="img" aria-label="' + esc(B.m.label + " against " + A.m.label + ", where each method's runs lie") + '">' + frameSVG(f, st.labels);
    var G = 72, clip = "cc" + (++hoverSeq), regions = [];
    s += '<defs><clipPath id="' + clip + '"><rect x="' + f.x0 + '" y="' + f.y0 + '" width="' + (f.x1 - f.x0) + '" height="' + (f.y1 - f.y0) + '"/></clipPath></defs>';
    clouds.forEach(function (cl, j) {
      var ca = cl.cols[A.k], cb = cl.cols[B.k], w = ws[j].w, grid = new Float64Array(G * G), xs = [], ys = [], wv = [], i;
      for (i = 0; i < cl.n; i++) { if (!(w[i] > 0)) { continue; } if (ppPlace(A, ca[i]) !== 0 || ppPlace(B, cb[i]) !== 0) { continue; }
        var vx = ppValue(A.sp, ca[i]), vy = ppValue(B.sp, cb[i]); xs.push(vx); ys.push(vy); wv.push(w[i]);
        var gx = (vx - A.lo) / (A.hi - A.lo) * G - 0.5, gy = (vy - B.lo) / (B.hi - B.lo) * G - 0.5, ix = Math.floor(gx), iy = Math.floor(gy), fx = gx - ix, fy = gy - iy;
        [[0, 0, (1 - fx) * (1 - fy)], [1, 0, fx * (1 - fy)], [0, 1, (1 - fx) * fy], [1, 1, fx * fy]].forEach(function (q) { var a = Math.min(G - 1, Math.max(0, ix + q[0])), b = Math.min(G - 1, Math.max(0, iy + q[1])); grid[b * G + a] += w[i] * q[2]; }); }
      if (!xs.length) { return; }
      var hx = bandwidth(xs, wv, A.sp.int) / (A.hi - A.lo) * G, hy = bandwidth(ys, wv, B.sp.int) / (B.hi - B.lo) * G;
      var sm = blur(grid, G, Math.max(0.6, hx), Math.max(0.6, hy)), lv = hdrLevels(sm, [0.5, 0.9]);
      var gxp = function (gx) { return f.x0 + (gx + 0.5) / G * (f.x1 - f.x0); }, gyp = function (gy) { return f.y1 - (gy + 0.5) / G * (f.y1 - f.y0); };
      var path = function (t) { return contourRings(sm, G, t).map(function (ring) { return "M" + ring.map(function (p) { return gxp(p[0]).toFixed(1) + " " + gyp(p[1]).toFixed(1); }).join("L") + "Z"; }).join(""); };
      regions.push({ key: cl.m.key, col: rgbCss(methodRGB(cl.m)), p50: path(lv[0]), p90: path(lv[1]) });
    });
    s += '<g clip-path="url(#' + clip + ')">';
    behind(regions).forEach(function (rg) { s += '<path d="' + rg.p50 + '" fill="' + rg.col + '" fill-opacity="0.28" fill-rule="evenodd" stroke="' + rg.col + '" stroke-width="2"/><path d="' + rg.p90 + '" fill="none" stroke="' + rg.col + '" stroke-width="1.4" fill-rule="evenodd"/>'; });
    s += "</g>" + stripIntervals(clouds, ws, A, B, f) + legendSVG(clouds, f, sz.W);
    return '<div class="v2charts v2one">' + s + "</svg></div>" + '<p class="v2hint">For each method, the ' + term("hdr", "darker region holds half of its runs inside the frame, the outline nine tenths") + ". " + term("strips", "Runs at a bound or beyond the range are in the strips along the frame") + "; there, a thick bar spans the middle half of a method's runs and a thin one the middle nine tenths. " + term("weights", "How the runs are weighted") + ".</p>";
  }
  function bandwidth(vals, ws, integer) {   // Scott's rule in two dimensions on a robust spread; a whole number never under half a step
    var tot = 0, m = 0, v = 0, i; for (i = 0; i < vals.length; i++) { tot += ws[i]; m += ws[i] * vals[i]; } m /= tot;
    for (i = 0; i < vals.length; i++) { v += ws[i] * (vals[i] - m) * (vals[i] - m); } var sd = Math.sqrt(v / tot);
    var q = wQuantiles(vals, ws, [0.25, 0.75]), iqr = (q[1] - q[0]) / 1.349, sig = iqr > 0 ? Math.min(sd, iqr) : sd;
    var s2 = 0; for (i = 0; i < ws.length; i++) { s2 += ws[i] * ws[i]; } var neff = tot * tot / s2;
    var h = sig * Math.pow(Math.max(2, neff), -1 / 6); return integer ? Math.max(0.5, h) : h;
  }
  function blur(grid, G, sx, sy) {   // a separable Gaussian, in cells
    var kern = function (s) { var r = Math.ceil(3 * s), k = [], t = 0; for (var i = -r; i <= r; i++) { var v = Math.exp(-0.5 * i * i / (s * s)); k.push(v); t += v; } return { r: r, k: k.map(function (v) { return v / t; }) }; };
    var kx = kern(sx), ky = kern(sy), tmp = new Float64Array(G * G), out = new Float64Array(G * G), x, y, i;
    for (y = 0; y < G; y++) { for (x = 0; x < G; x++) { var a = 0; for (i = -kx.r; i <= kx.r; i++) { var xx = x + i; if (xx >= 0 && xx < G) { a += grid[y * G + xx] * kx.k[i + kx.r]; } } tmp[y * G + x] = a; } }
    for (y = 0; y < G; y++) { for (x = 0; x < G; x++) { var b = 0; for (i = -ky.r; i <= ky.r; i++) { var yy = y + i; if (yy >= 0 && yy < G) { b += tmp[yy * G + x] * ky.k[i + ky.r]; } } out[y * G + x] = b; } }
    return out;
  }
  function hdrLevels(d, shares) {   // the density above which the given shares of the mass lie
    var v = Array.prototype.slice.call(d).sort(function (a, b) { return b - a; }), tot = v.reduce(function (a, b) { return a + b; }, 0);
    return shares.map(function (p) { var cum = 0; for (var i = 0; i < v.length; i++) { cum += v[i]; if (cum >= p * tot) { return v[i]; } } return v[v.length - 1]; });
  }
  // Marching squares over the grid padded with zeros (so every outline closes), with the segments joined into rings.
  // Corner values sit at cell centres; a saddle is resolved by the mean of its four corners.
  function contourRings(d, G, t) {
    var val = function (x, y) { return x < 0 || y < 0 || x >= G || y >= G ? 0 : d[y * G + x]; };
    var segs = {}, ends = {}, id = 0, x, y;
    var pt = function (k) { var p = k.split(","), kind = p[0], i = +p[1], j = +p[2], a, b; if (kind === "h") { a = val(i, j); b = val(i + 1, j); return [i + (t - a) / (b - a), j]; } a = val(i, j); b = val(i, j + 1); return [i, j + (t - a) / (b - a)]; };
    var add = function (k1, k2) { var s = { a: k1, b: k2, used: false }; segs[id] = s; (ends[k1] = ends[k1] || []).push(id); (ends[k2] = ends[k2] || []).push(id); id++; };
    for (y = -1; y < G; y++) {
      for (x = -1; x < G; x++) {
        var v0 = val(x, y), v1 = val(x + 1, y), v2 = val(x + 1, y + 1), v3 = val(x, y + 1);
        var c = (v0 >= t ? 1 : 0) | (v1 >= t ? 2 : 0) | (v2 >= t ? 4 : 0) | (v3 >= t ? 8 : 0);
        if (c === 0 || c === 15) { continue; }
        var B = "h," + x + "," + y, R = "v," + (x + 1) + "," + y, T = "h," + x + "," + (y + 1), Lk = "v," + x + "," + y;
        var ctr = (v0 + v1 + v2 + v3) / 4 >= t;
        switch (c) {
          case 1: case 14: add(Lk, B); break;
          case 2: case 13: add(B, R); break;
          case 3: case 12: add(Lk, R); break;
          case 4: case 11: add(R, T); break;
          case 6: case 9: add(B, T); break;
          case 7: case 8: add(Lk, T); break;
          case 5: if (ctr) { add(Lk, T); add(B, R); } else { add(Lk, B); add(R, T); } break;
          case 10: if (ctr) { add(Lk, B); add(R, T); } else { add(Lk, T); add(B, R); } break;
        }
      }
    }
    var rings = [];
    Object.keys(segs).forEach(function (sid) {
      var s = segs[sid]; if (s.used) { return; }
      s.used = true; var ring = [s.a, s.b], cur = s.b, guard = 0;
      while (cur !== s.a && guard++ < 100000) {
        var nxt = (ends[cur] || []).map(function (q) { return segs[q]; }).filter(function (q) { return !q.used; })[0];
        if (!nxt) { break; } nxt.used = true; cur = nxt.a === cur ? nxt.b : nxt.a; ring.push(cur);
      }
      if (ring.length > 3) { rings.push(ring.map(pt)); }
    });
    return rings;
  }
  // in the contour display, a strip shows each method's middle half (thick) and middle nine tenths (thin) along it
  function stripIntervals(clouds, ws, A, B, f) {
    var s = "", k = clouds.length, lanes = [["xl", -1, null], ["xh", 1, null], ["yl", null, -1], ["yh", null, 1]];
    lanes.forEach(function (ln) { if (!f.has[ln[0]]) { return; }
      clouds.forEach(function (cl, j) {
        var ca = cl.cols[A.k], cb = cl.cols[B.k], w = ws[j].w, vals = [], wv = [], tot = 0, i;
        for (i = 0; i < cl.n; i++) { if (!(w[i] > 0)) { continue; } var px = ppPlace(A, ca[i]), py = ppPlace(B, cb[i]);
          if (ln[1] !== null && px === ln[1] && py === 0) { vals.push(ppValue(B.sp, cb[i])); wv.push(w[i]); }
          if (ln[2] !== null && py === ln[2] && px === 0) { vals.push(ppValue(A.sp, ca[i])); wv.push(w[i]); } }
        if (!vals.length) { return; } wv.forEach(function (x) { tot += x; });
        var q = wQuantiles(vals, wv, [0.05, 0.25, 0.5, 0.75, 0.95]), col = rgbCss(methodRGB(cl.m)), off = 3 + (j + 0.5) * (f.ST - 6) / k, title = "<title>" + esc(cl.m.label + ": " + (100 * tot).toFixed(1) + " % of its runs in this strip and inside the frame on the other axis; median " + (ln[1] !== null ? vText(B, q[2]) : vText(A, q[2]))) + "</title>";
        if (ln[1] !== null) { var sx = (ln[1] < 0 ? f.sxl[0] : f.sxh[0]) + off;
          s += '<g>' + title + '<line x1="' + sx.toFixed(1) + '" y1="' + f.ys(q[0]).toFixed(1) + '" x2="' + sx.toFixed(1) + '" y2="' + f.ys(q[4]).toFixed(1) + '" stroke="' + col + '" stroke-width="1.5"/><line x1="' + sx.toFixed(1) + '" y1="' + f.ys(q[1]).toFixed(1) + '" x2="' + sx.toFixed(1) + '" y2="' + f.ys(q[3]).toFixed(1) + '" stroke="' + col + '" stroke-width="4"/><circle cx="' + sx.toFixed(1) + '" cy="' + f.ys(q[2]).toFixed(1) + '" r="2.6" fill="var(--surface)" stroke="' + col + '" stroke-width="1.5"/></g>'; }
        else { var sy = (ln[2] < 0 ? f.syl[0] : f.syh[0]) + off;
          s += '<g>' + title + '<line x1="' + f.xs(q[0]).toFixed(1) + '" y1="' + sy.toFixed(1) + '" x2="' + f.xs(q[4]).toFixed(1) + '" y2="' + sy.toFixed(1) + '" stroke="' + col + '" stroke-width="1.5"/><line x1="' + f.xs(q[1]).toFixed(1) + '" y1="' + sy.toFixed(1) + '" x2="' + f.xs(q[3]).toFixed(1) + '" y2="' + sy.toFixed(1) + '" stroke="' + col + '" stroke-width="4"/><circle cx="' + f.xs(q[2]).toFixed(1) + '" cy="' + sy.toFixed(1) + '" r="2.6" fill="var(--surface)" stroke="' + col + '" stroke-width="1.5"/></g>'; }
      }); });
    return s;
  }
  // grid bins: one per whole number where there are few enough, else about eight pixels each
  function binsOf(Ax, px) { var span = Ax.hi - Ax.lo; if (Ax.sp.int && span <= 64) { return Math.max(1, Math.round(span)); } return Math.max(12, Math.min(60, Math.round(px / 8))); }
  function corrHeat(clouds, ws, A, B, st) {
    var nr = narrow(), avail = wideWidth(), gapW = 16, cols = Math.max(1, Math.min(3, clouds.length, Math.floor((avail + gapW) / (280 + gapW))));
    var W = Math.floor((avail - gapW * (cols - 1)) / cols), H = Math.round(Math.max(250, Math.min(400, W * 0.85)));
    var surf = hexRGB(cssVar("--surface")), panels = [], top = 0, stop = 0;
    clouds.forEach(function (cl, j) {
      var f = corrFrame(W, H, A, B, st.has, { top: 12, left: nr ? 48 : 58 }), nx = binsOf(A, f.x1 - f.x0), ny = binsOf(B, f.y1 - f.y0);
      var cells = new Float64Array(nx * ny), strips = { xl: new Float64Array(ny), xh: new Float64Array(ny), yl: new Float64Array(nx), yh: new Float64Array(nx) }, corner = { ll: 0, lh: 0, hl: 0, hh: 0 }, ca = cl.cols[A.k], cb = cl.cols[B.k], w = ws[j].w;
      var bx = function (v) { return Math.min(nx - 1, Math.max(0, Math.floor((v - A.lo) / (A.hi - A.lo) * nx))); }, by = function (v) { return Math.min(ny - 1, Math.max(0, Math.floor((v - B.lo) / (B.hi - B.lo) * ny))); };
      for (var i = 0; i < cl.n; i++) { if (!(w[i] > 0)) { continue; } var px = ppPlace(A, ca[i]), py = ppPlace(B, cb[i]);
        if (px === 0 && py === 0) { cells[by(ppValue(B.sp, cb[i])) * nx + bx(ppValue(A.sp, ca[i]))] += w[i]; }
        else if (px !== 0 && py === 0) { strips[px < 0 ? "xl" : "xh"][by(ppValue(B.sp, cb[i]))] += w[i]; }
        else if (py !== 0 && px === 0) { strips[py < 0 ? "yl" : "yh"][bx(ppValue(A.sp, ca[i]))] += w[i]; }
        else { corner[(px < 0 ? "l" : "h") + (py < 0 ? "l" : "h")] += w[i]; } }
      cells.forEach(function (v) { top = Math.max(top, v); }); ["xl", "xh", "yl", "yh"].forEach(function (k) { strips[k].forEach(function (v) { stop = Math.max(stop, v); }); });
      Object.keys(corner).forEach(function (k) { stop = Math.max(stop, corner[k]); });
      panels.push({ cl: cl, f: f, nx: nx, ny: ny, cells: cells, strips: strips, corner: corner, col: methodRGB(cl.m) });
    });
    var out = panels.map(function (P) {
      var f = P.f, cw = (f.x1 - f.x0) / P.nx, ch = (f.y1 - f.y0) / P.ny, shade = function (v, mx) { return mixRGB(P.col, surf, Math.sqrt(v / mx)); };
      var href = rasterHref(f.x1 - f.x0, f.y1 - f.y0, function (g) { for (var yb = 0; yb < P.ny; yb++) { for (var xb = 0; xb < P.nx; xb++) { var v = P.cells[yb * P.nx + xb]; if (v > 0) { g.fillStyle = rgbCss(shade(v, top)); g.fillRect(xb * cw, (P.ny - 1 - yb) * ch, cw + 0.5, ch + 0.5); } } } });
      var s = '<svg viewBox="0 0 ' + W + " " + H + '" class="v2chart v2corr" role="img" aria-label="' + esc(P.cl.m.label + ": " + B.m.label + " against " + A.m.label + ", share of runs per cell") + '" data-hover="' + hoverId(function (pt) {
        var txt = null;
        if (pt.x >= f.x0 && pt.x <= f.x1 && pt.y >= f.y0 && pt.y <= f.y1) { var xb = Math.min(P.nx - 1, Math.floor((pt.x - f.x0) / cw)), yb = Math.min(P.ny - 1, Math.floor((f.y1 - pt.y) / ch)), v = P.cells[yb * P.nx + xb];
          var x0 = A.lo + xb * (A.hi - A.lo) / P.nx, x1 = x0 + (A.hi - A.lo) / P.nx, y0 = B.lo + yb * (B.hi - B.lo) / P.ny, y1 = y0 + (B.hi - B.lo) / P.ny;
          txt = P.cl.m.label + ": " + A.m.short + " " + vText(A, x0) + " to " + vText(A, x1) + ", " + B.m.short + " " + vText(B, y0) + " to " + vText(B, y1) + ": " + (100 * v).toFixed(2) + " % of its runs"; }
        return txt ? { text: txt } : null; }) + '">' + frameSVG(f, st.labels);
      if (href) { s += '<image href="' + href + '" x="' + f.x0 + '" y="' + f.y0 + '" width="' + (f.x1 - f.x0) + '" height="' + (f.y1 - f.y0) + '" preserveAspectRatio="none"/>'; }
      ["xl", "xh", "yl", "yh"].forEach(function (k) { if (!f.has[k]) { return; } var arr = P.strips[k], n = arr.length, vert = k.charAt(0) === "x";
        for (var b = 0; b < n; b++) { if (!(arr[b] > 0)) { continue; } var c = rgbCss(shade(arr[b], stop)), lab = esc(P.cl.m.label + ": " + (100 * arr[b]).toFixed(2) + " % of its runs in this cell of the strip");
          if (vert) { var sx = k === "xl" ? f.sxl[0] : f.sxh[0], y1b = f.y1 - b * ch; s += '<rect x="' + sx.toFixed(1) + '" y="' + (y1b - ch).toFixed(1) + '" width="' + f.ST + '" height="' + (ch + 0.4).toFixed(1) + '" fill="' + c + '"><title>' + lab + "</title></rect>"; }
          else { var sy = k === "yl" ? f.syl[0] : f.syh[0]; s += '<rect x="' + (f.x0 + b * cw).toFixed(1) + '" y="' + sy.toFixed(1) + '" width="' + (cw + 0.4).toFixed(1) + '" height="' + f.ST + '" fill="' + c + '"><title>' + lab + "</title></rect>"; } } });
      s += cornersSVG(f, P.corner, function (v) { return rgbCss(shade(v, stop)); }, function (v) { return P.cl.m.label + ": " + (100 * v).toFixed(2) + " % of its runs in both strips"; });
      // the panel's name is HTML above the chart, so that a long one wraps instead of running off the frame
      return '<figure class="v2cpanel"><figcaption class="v2cpanelname">' + esc(corrLabel(P.cl)) + "</figcaption>" + s + "</svg></figure>";
    }).join("");
    return '<div class="v2cpanels" style="grid-template-columns:repeat(' + cols + ', minmax(0, 1fr))">' + out + "</div>" +
      '<p class="v2hint">Colour: the share of the method’s runs in each cell, on a square-root scale, the same in every panel; the darkest cell holds ' + (100 * top).toFixed(1) + " % of a method’s runs. " + term("strips", "The strips along the frame") + " have a scale of their own" + (stop > 0 ? " (darkest: " + (100 * stop).toFixed(1) + " %)" : "") + ". " + term("weights", "How the runs are weighted") + ".</p>";
  }
  function corrPoints(clouds, ws, A, B, st) {
    var sz = planeSize(clouds, true), f = corrFrame(sz.W, sz.H, A, B, st.has, { right: sz.right, below: sz.below }), nr = narrow(), rad = nr ? 2.1 : 1.7;
    var pts = [];   // drawn order: faded methods first, then shown ones in the methods' order
    var order = behind(clouds.map(function (cl) { return { key: cl.m.key, cl: cl }; }));
    order.forEach(function (o) {
      var cl = o.cl, j = clouds.indexOf(cl), w = ws[j].w, ca = cl.cols[A.k], cb = cl.cols[B.k], pos = [], i;
      for (i = 0; i < cl.n; i++) { if (w[i] > 0) { pos.push(w[i]); } }
      pos.sort(function (a, b) { return a - b; }); var med = pos.length ? pos[Math.floor(pos.length / 2)] : 1;
      for (i = 0; i < cl.n; i++) { if (!(w[i] > 0)) { continue; } var px = ppPlace(A, ca[i]), py = ppPlace(B, cb[i]), u = hash01(cl.cat[i] + 1, cl.row[i] + 1, cl.drw[i] + 7 * j);
        var u2 = hash01(cl.row[i] + 3, cl.cat[i] + 5, cl.drw[i] + 11 * j), bx0 = ca[i] === A.bl || ca[i] === A.bh, by0 = cb[i] === B.bl || cb[i] === B.bh;
        pts.push({ j: j, i: i, x: spotX(f, px, ca[i], u, bx0 ? undefined : u2), y: spotY(f, py, cb[i], u2, by0 ? undefined : u), a: Math.min(1, 0.35 * w[i] / med) }); }
    });
    var cols = clouds.map(function (cl) { return methodRGB(cl.m); }), L = f.has.xl ? f.sxl[0] : f.x0, T = f.has.yh ? f.syh[0] : f.y0, Rr = f.has.xh ? f.sxh[1] : f.x1, Bb = f.has.yl ? f.syl[1] : f.y1;
    var href = rasterHref(Rr - L, Bb - T, function (g) { pts.forEach(function (p) { g.fillStyle = rgbCss(cols[p.j], p.a); g.beginPath(); g.arc(p.x - L, p.y - T, rad, 0, 2 * Math.PI); g.fill(); }); });
    var cellPx = 8, buckets = {};
    pts.forEach(function (p, n) { var key = Math.floor(p.x / cellPx) + "," + Math.floor(p.y / cellPx); (buckets[key] = buckets[key] || []).push(n); });
    var near = function (pt) { var best = -1, bd = 36, bx = Math.floor(pt.x / cellPx), by = Math.floor(pt.y / cellPx);
      for (var dx = -1; dx <= 1; dx++) { for (var dy = -1; dy <= 1; dy++) { (buckets[(bx + dx) + "," + (by + dy)] || []).forEach(function (n) { var p = pts[n], d = (p.x - pt.x) * (p.x - pt.x) + (p.y - pt.y) * (p.y - pt.y); if (d <= bd) { bd = d; best = n; } }); } }
      return best < 0 ? null : pts[best]; };
    var hid = hoverId(function (pt, click) {
      var p = near(pt); if (!p) { return null; }
      var cl = clouds[p.j], i = p.i, c = cl.cs[cl.cat[i]], fl = cl.flag[i];
      if (click) { csel = { m: cl.m.key, c: c, row: cl.row[i], d: cl.drw[i], r: runRung(cl, i), xa: codeText(A, cl.cols[A.k][i]), yb: codeText(B, cl.cols[B.k][i]) }; render(); }   // at once: a deferred redraw is pushed back by every further click
      return { text: cl.m.label + " · " + c + ", problem " + (cl.row[i] + 1) + ", run " + cl.drw[i] + " (budget " + fmtBudget(runRung(cl, i)) + "): " + A.m.short + " " + codeText(A, cl.cols[A.k][i]) + ", " + B.m.short + " " + codeText(B, cl.cols[B.k][i]) +
        (fl & PP_OK ? ((fl & PP_NUM) || (fl & PP_SYM) ? "; recovered: " + [fl & PP_NUM ? "numeric" : "", fl & PP_SYM ? "structure" : ""].filter(Boolean).join(" and ") : "") : "; no usable formula") + ". A click shows its formula." };
    });
    var s = '<svg viewBox="0 0 ' + sz.W + " " + sz.H + '" class="v2chart v2corr v2cpoints" role="img" aria-label="' + esc(B.m.label + " against " + A.m.label + ", every run as a point") + '" data-hover="' + hid + '">' + frameSVG(f, st.labels);
    if (href) { s += '<image href="' + href + '" x="' + L + '" y="' + T + '" width="' + (Rr - L) + '" height="' + (Bb - T) + '" preserveAspectRatio="none"/>'; }
    s += legendSVG(clouds, f, sz.W) + "</svg>";
    return '<div class="v2charts v2one">' + s + "</div>" + '<p class="v2hint">One point per run, ' + pts.length.toLocaleString() + " in all. A point’s opacity follows its " + term("weights", "weight") + ", so the runs of a small problem set stand out from those of a large one.  In the " + term("strips", "strips") + ", points are spread across the strip at random; their position across it means nothing. Inside the frame, a point is placed at random within the precision its value is published to (one 252nd of the metric’s range), so that values do not line up in stripes. Tap or hover a point for its problem; a click shows its formula below.</p>";
  }
  function corrTrend(clouds, ws, A, B, st) {
    var sz = planeSize(clouds, true), f = corrFrame(sz.W, sz.H, A, B, st.has, { right: sz.right, below: sz.below }), nb = binsOf(A, f.x1 - f.x0);
    if (!(A.sp.int && A.hi - A.lo <= 40)) { nb = Math.max(8, Math.min(24, Math.round((f.x1 - f.x0) / 36))); }
    var bw = (A.hi - A.lo) / nb, series = [], dots = [];
    clouds.forEach(function (cl, j) {
      var ca = cl.cols[A.k], cb = cl.cols[B.k], w = ws[j].w, bins = {}, i;
      for (i = 0; i < cl.n; i++) { if (!(w[i] > 0)) { continue; } var px = ppPlace(A, ca[i]), py = ppPlace(B, cb[i]);
        var key = px < 0 ? "lo" : px > 0 ? "hi" : String(Math.min(nb - 1, Math.floor((ppValue(A.sp, ca[i]) - A.lo) / bw)));
        var yv = py < 0 ? -Infinity : py > 0 ? Infinity : ppValue(B.sp, cb[i]);
        (bins[key] = bins[key] || { v: [], w: [] }).v.push(yv); bins[key].w.push(w[i]); }
      var pts = [];
      Object.keys(bins).forEach(function (key) { var b = bins[key], share = b.w.reduce(function (a, x) { return a + x; }, 0), q = wQuantiles(b.v, b.w, [0.25, 0.5, 0.75]);
        var x = key === "lo" ? f.sxl[0] + f.ST / 2 : key === "hi" ? f.sxh[0] + f.ST / 2 : f.xs(A.lo + (+key + 0.5) * bw);
        var yOf = function (v) { return v === -Infinity ? f.syl[0] + f.ST / 2 : v === Infinity ? f.syh[0] + f.ST / 2 : f.ys(Math.min(B.hi, Math.max(B.lo, v))); };
        var tv = function (v) { return v === -Infinity ? st.labels.yl : v === Infinity ? st.labels.yh : vText(B, v); };
        var range = key === "lo" ? st.labels.xl : key === "hi" ? st.labels.xh : vText(A, A.lo + +key * bw) + " to " + vText(A, A.lo + (+key + 1) * bw);
        pts.push({ x: x, inside: key !== "lo" && key !== "hi", y: yOf(q[1]), y0: yOf(q[0]), y1: yOf(q[2]), r: Math.max(2.2, Math.min(6, 2 + 18 * Math.sqrt(share / 10))), share: share,
          text: cl.m.label + ", " + A.m.short + " " + range + ": median " + B.m.short + " " + tv(q[1]) + ", middle half " + tv(q[0]) + " to " + tv(q[2]) + "; " + (100 * share).toFixed(1) + " % of its runs" }); });
      pts.sort(function (a, b) { return a.x - b.x; });
      series.push({ key: cl.m.key, col: rgbCss(methodRGB(cl.m)), pts: pts }); pts.forEach(function (p) { dots.push(p); });
    });
    var hid = hoverId(function (pt) { var best = null, bd = 400; dots.forEach(function (p) { var d = (p.x - pt.x) * (p.x - pt.x) + (p.y - pt.y) * (p.y - pt.y); if (d < bd) { bd = d; best = p; } }); return best ? { text: best.text } : null; });
    var s = '<svg viewBox="0 0 ' + sz.W + " " + sz.H + '" class="v2chart v2corr" role="img" aria-label="' + esc("median " + B.m.label + " along " + A.m.label) + '" data-hover="' + hid + '">' + frameSVG(f, st.labels);
    behind(series).forEach(function (sr) {
      var inside = sr.pts.filter(function (p) { return p.inside; });
      if (inside.length > 1) { s += '<polygon points="' + inside.map(function (p) { return p.x.toFixed(1) + "," + p.y1.toFixed(1); }).concat(inside.slice().reverse().map(function (p) { return p.x.toFixed(1) + "," + p.y0.toFixed(1); })).join(" ") + '" fill="' + sr.col + '" fill-opacity="0.13" stroke="none"/>'; }
      sr.pts.forEach(function (p) { s += '<line x1="' + p.x.toFixed(1) + '" y1="' + p.y0.toFixed(1) + '" x2="' + p.x.toFixed(1) + '" y2="' + p.y1.toFixed(1) + '" stroke="' + sr.col + '" stroke-width="1.5" stroke-opacity="0.45"/><circle cx="' + p.x.toFixed(1) + '" cy="' + p.y.toFixed(1) + '" r="' + p.r.toFixed(1) + '" fill="' + sr.col + '"/>'; });
    });
    s += legendSVG(clouds, f, sz.W) + "</svg>";
    return '<div class="v2charts v2one">' + s + "</div>" + '<p class="v2hint">The x axis is cut into bins. In each, a dot marks the median ' + esc(B.m.label) + " of a method’s runs, and the bar and the band span their middle half; a larger dot holds more of the method’s runs. A bin in a " + term("strips", "strip") + " holds the runs at a bound or beyond the range. " + term("weights", "How the runs are weighted") + ".</p>";
  }
  // Where each method's runs fall, strip by strip, and the rank correlation of the two metrics.
  function corrTable(clouds, ws, A, B, st, rhos) {
    var lanes = [["in", "inside the frame"]].concat([["xl", A.m.short + " " + st.labels.xl], ["xh", A.m.short + " " + st.labels.xh], ["yl", B.m.short + " " + st.labels.yl], ["yh", B.m.short + " " + st.labels.yh]].filter(function (l) { return st.has[l[0]]; }));
    var rows = clouds.map(function (cl, j) {
      var ca = cl.cols[A.k], cb = cl.cols[B.k], w = ws[j].w, tot = { in: 0, xl: 0, xh: 0, yl: 0, yh: 0 }, i;
      for (i = 0; i < cl.n; i++) { if (!(w[i] > 0)) { continue; } var px = ppPlace(A, ca[i]), py = ppPlace(B, cb[i]);
        if (px === 0 && py === 0) { tot.in += w[i]; } if (px < 0) { tot.xl += w[i]; } if (px > 0) { tot.xh += w[i]; } if (py < 0) { tot.yl += w[i]; } if (py > 0) { tot.yh += w[i]; } }
      var none = ws[j].all - ws[j].n;
      return "<tr><td>" + '<span class="v2sw" style="background:' + rgbCss(methodRGB(cl.m)) + '"></span>' + esc(cl.m.label + (cl.m.local ? " (local)" : "")) + "</td><td>" + (cl.between ? "≈ " + fmtBudget(cl.between[0]) + "–" + fmtBudget(cl.between[1]) : fmtBudget(cl.r)) + "</td>" + lanes.map(function (l) { return "<td>" + (100 * tot[l[0]]).toFixed(1) + " %</td>"; }).join("") +
        "<td>" + Math.round(ws[j].n).toLocaleString() + (none > 0.5 ? ' <span class="v2ci-txt">of ' + ws[j].all.toLocaleString() + "</span>" : "") + "</td><td>" + rhoText(rhos[j]) + "</td></tr>";
    }).join("");
    return '<div class="v2table-wrap"><table class="v2table v2ctable"><thead><tr><th>method</th><th>budget</th>' + lanes.map(function (l) { return "<th>" + esc(l[1]) + "</th>"; }).join("") + "<th>problems with both values</th><th>rank correlation " + help(CTERMS.rho, "What is the rank correlation?") + "</th></tr></thead><tbody>" + rows + "</tbody></table></div>" +
      '<p class="v2hint">The share of each method’s runs inside the frame and in each strip, weighted as in the chart; a run in a corner counts in both of its strips. A problem without a value on either metric is left out' + (leftOut(A.k) || leftOut(B.k) ? ", and so is a run without a usable formula in an overlap metric, as chosen in the side panel" : "") + ".</p>";
  }
  // the run picked in the points display: its problem's true formula and the method's, from the Predictions view's files
  function corrSelected(A, B) {
    if (!csel || state.cv !== "points") { return ""; }
    var m = D.methods.filter(function (x) { return x.key === csel.m; })[0]; if (!m) { return ""; }
    var blk = Math.floor(csel.row / (D.pred_block || 500)), have = ((D.pred || {})[m.key] || {})[csel.c + "|" + csel.r] || [];
    var files = ["pred/truth/" + csel.c + "." + blk + ".js"].concat(have.indexOf(csel.d) >= 0 ? ["pred/" + m.key + "/" + csel.c + "/" + csel.r + "." + csel.d + "." + blk + ".js"] : []);
    var wait = filesReady(files) ? "" : "…", Rp = (window.RESULTS_V2_PRED || {})[REL] || {};
    var gt = (Rp["truth|" + csel.c + "|" + blk] || {})[String(csel.row)], v = (Rp[m.key + "|" + csel.c + "|" + csel.r + "|" + csel.d + "|" + blk] || {})[String(csel.row)];
    var formula = have.indexOf(csel.d) < 0 ? '<span class="v2predna">not published</span>' : v === undefined ? wait : v === null ? '<span class="v2predna">no usable formula</span>' : typeset(v[0]);
    return '<section class="v2csel" aria-label="the run picked in the chart"><h3 class="v2h">' + esc(m.label) + " · " + esc(csel.c) + ", problem " + (csel.row + 1) + ", run " + csel.d + ", budget " + fmtBudget(csel.r) + "</h3>" +
      '<p class="v2hint">' + esc(A.m.label) + ": " + esc(csel.xa) + "; " + esc(B.m.label) + ": " + esc(csel.yb) + "</p>" + predTruth(gt, wait) +
      '<div class="v2predtruth"><span class="v2lab">its formula</span><span class="v2predtruthf">' + formula + (v ? " " + predMarks(v[1]) : "") + "</span></div>" +
      '<div class="v2row"><button type="button" class="v2btn" data-act="corr-open">open this problem in Predictions</button><button type="button" class="v2btn" data-act="corr-clear">clear</button></div></section>';
  }
  // Every pair of metrics: one method's rank correlations below the diagonal, the other's above it.
  function corrMatrix(clouds, ks) {
    var cA = clouds.filter(function (cl) { return cl.m.key === state.ca; })[0], cB = clouds.filter(function (cl) { return cl.m.key === state.cb; })[0] || cA;
    var wA = ppWeights(cA, []), wB = cB === cA ? wA : ppWeights(cB, []);
    var has = function (cl, k) { for (var i = 0; i < cl.n; i++) { if (cl.cols[k][i] !== PP_NONE) { return true; } } return false; };
    var mk = ks.filter(function (k) { return has(cA, k) || has(cB, k); }), n = mk.length, nr = narrow();
    var labW = Math.ceil(widest(mk.map(function (k) { return METRIC[k].short; }))) + 14, cell = Math.max(nr ? 20 : 22, Math.min(34, Math.floor((wideWidth() - labW - 20) / n)));
    var topH = Math.ceil(widest(mk.map(function (k) { return METRIC[k].short; })) * 0.72) + 24, W = labW + n * cell + topH, H = topH + n * cell + 10;
    var memo = {}, rho = function (cl, w, a, b) { var key = cl.m.key + "|" + a + "|" + b; if (!(key in memo)) { memo[key] = spearman(cl, w.w, a, b, w.neff); } return memo[key]; };
    var s = '<svg viewBox="0 0 ' + W + " " + H + '" class="v2chart v2cmatrix" style="min-width:' + W + 'px" role="img" aria-label="' + esc("rank correlations of every pair of metrics: " + cA.m.label + " below the diagonal, " + cB.m.label + " above it") + '">';
    mk.forEach(function (k, i) { var x = labW + (i + 0.5) * cell, y = topH + (i + 0.5) * cell;
      s += '<text x="' + (labW - 8) + '" y="' + (y + 4).toFixed(1) + '" class="tick" text-anchor="end">' + esc(METRIC[k].short) + "<title>" + esc(METRIC[k].label) + "</title></text>";
      s += '<text transform="translate(' + (x + 3).toFixed(1) + "," + (topH - 8) + ') rotate(-45)" class="tick">' + esc(METRIC[k].short) + "<title>" + esc(METRIC[k].label) + "</title></text>"; });
    mk.forEach(function (ky, i) { mk.forEach(function (kx, j) {
      var x = labW + j * cell, y = topH + i * cell;
      if (i === j) { s += '<rect x="' + x + '" y="' + y + '" width="' + cell + '" height="' + cell + '" class="v2cdiag"/>'; return; }
      var lower = i > j, cl = lower ? cA : cB, w = lower ? wA : wB, st = rho(cl, w, kx, ky);
      var fill = st ? "color-mix(in srgb, var(" + (st.rho >= 0 ? "--rho-pos" : "--rho-neg") + ") " + Math.round(70 * Math.abs(st.rho)) + "%, var(--surface))" : "var(--surface)";
      s += '<g class="v2ccell" data-cx="' + esc(kx) + '" data-cy="' + esc(ky) + '" role="button" tabindex="0"><title>' + esc(cl.m.label + ": " + METRIC[ky].label + " and " + METRIC[kx].label + ", " + (st ? rhoText(st) + " (95 % interval), from " + Math.round(st.n).toLocaleString() + " problems" : "no runs with both values") + ". A click draws the two.") + "</title>" +
        '<rect x="' + x + '" y="' + y + '" width="' + cell + '" height="' + cell + '" fill="' + fill + '" class="v2ccellbox"/>' +
        (st && cell >= 26 ? '<text x="' + (x + cell / 2) + '" y="' + (y + cell / 2 + 3.5) + '" class="v2ccelltxt" text-anchor="middle">' + (st.rho < 0 ? "−" : "") + Math.abs(st.rho).toFixed(2).replace(/^0/, "") + "</text>" : "") + "</g>"; }); });
    s += '<line x1="' + labW + '" y1="' + topH + '" x2="' + (labW + n * cell) + '" y2="' + (topH + n * cell) + '" class="v2cdiagline"/></svg>';
    return '<p class="v2hint">Below the diagonal: <span class="v2sw" style="background:' + rgbCss(methodRGB(cA.m)) + '"></span>' + esc(corrLabel(cA)) + "; above it: " + '<span class="v2sw" style="background:' + rgbCss(methodRGB(cB.m)) + '"></span>' + esc(corrLabel(cB)) + ".</p>" +
      '<div class="v2table-wrap v2cmatwrap">' + s + "</div>" +
      '<p class="v2hint">Each cell is the ' + term("rho", "rank correlation") + " of two metrics over the method’s runs: blue where they rise together, orange where one falls as the other rises, faint where the two are close to unrelated. Hover a cell for its 95 % interval; a click draws the two metrics against each other. Properties of the true formula are the same for every method, so their correlations with each other agree on both sides.</p>";
  }
  // One metric, two methods: each problem's value under one method against its value under the other.
  function corrVs(clouds) {
    var k = state.cy, cA = clouds.filter(function (cl) { return cl.m.key === state.ca; })[0], cB = clouds.filter(function (cl) { return cl.m.key === state.cb; })[0] || cA;
    if (cA === cB) { return '<p class="v2hint">Select a second method in the side panel to compare two.</p>'; }
    var wA = ppWeights(cA, [k]), wB = ppWeights(cB, [k]), A = ppAxis(k, [cA, cB], [wA, wB]);
    var key = function (cl, i) { return cl.cat[i] + ":" + cl.row[i]; }, runsB = {}, i;
    // pairs of runs on the same problem, every combination; a problem weighs the mean of its weights under the two methods,
    // shared among its pairs by the product of the two runs' shares of their problem (equal shares at one budget)
    for (i = 0; i < cB.n; i++) { if (wB.w[i] > 0) { (runsB[cB.cs[cB.cat[i]] + ":" + cB.row[i]] = runsB[cB.cs[cB.cat[i]] + ":" + cB.row[i]] || []).push(i); } }
    var probA = {}; for (i = 0; i < cA.n; i++) { if (wA.w[i] > 0) { var pk = cA.cs[cA.cat[i]] + ":" + cA.row[i]; (probA[pk] = probA[pk] || []).push(i); } }
    var pairs = [], tot = 0;
    Object.keys(probA).forEach(function (pk) { var ra = probA[pk], rb = runsB[pk]; if (!rb) { return; }
      var wa = ra.reduce(function (a, q) { return a + wA.w[q]; }, 0), wb = rb.reduce(function (a, q) { return a + wB.w[q]; }, 0), wp = (wa + wb) / 2;
      ra.forEach(function (qa) { rb.forEach(function (qb) { var wv = wp * (wA.w[qa] / wa) * (wB.w[qb] / wb); pairs.push([cA.cols[k][qa], cB.cols[k][qb], wv]); tot += wv; }); }); });
    if (!pairs.length) { return '<p class="v2hint">The two methods share no problem with a value on ' + esc(METRIC[k].label) + ".</p>"; }
    var has = { xl: false, xh: false, yl: false, yh: false }, onlyB = { xl: true, xh: true, yl: true, yh: true };
    pairs.forEach(function (p) { var px = ppPlace(A, p[0]), py = ppPlace(A, p[1]);
      if (px < 0) { has.xl = true; if (p[0] !== A.bl) { onlyB.xl = false; } } if (px > 0) { has.xh = true; if (p[0] !== A.bh) { onlyB.xh = false; } }
      if (py < 0) { has.yl = true; if (p[1] !== A.bl) { onlyB.yl = false; } } if (py > 0) { has.yh = true; if (p[1] !== A.bh) { onlyB.yh = false; } } });
    var labels = { xl: stripLabel(A, -1, onlyB.xl), xh: stripLabel(A, 1, onlyB.xh), yl: stripLabel(A, -1, onlyB.yl), yh: stripLabel(A, 1, onlyB.yh), x: cA.m.label + ": " + axisName(A.m), y: cB.m.label + ": " + axisName(A.m) };
    var W = Math.min(wideWidth(), 640), H = Math.round(W * 0.9), f = corrFrame(W, H, A, A, has, {}), nx = binsOf(A, f.x1 - f.x0), ny = binsOf(A, f.y1 - f.y0);
    var cells = new Float64Array(nx * ny), strips = { xl: new Float64Array(ny), xh: new Float64Array(ny), yl: new Float64Array(nx), yh: new Float64Array(nx) }, corner = { ll: 0, lh: 0, hl: 0, hh: 0 }, below = 0, above = 0, same = 0;
    var bx = function (v) { return Math.min(nx - 1, Math.max(0, Math.floor((v - A.lo) / (A.hi - A.lo) * nx))); }, by = function (v) { return Math.min(ny - 1, Math.max(0, Math.floor((v - A.lo) / (A.hi - A.lo) * ny))); };
    pairs.forEach(function (p) { var px = ppPlace(A, p[0]), py = ppPlace(A, p[1]), wv = p[2] / tot, oa = ppOrder(p[0]), ob = ppOrder(p[1]);
      if (ob < oa) { below += wv; } else if (ob > oa) { above += wv; } else { same += wv; }
      if (px === 0 && py === 0) { cells[by(ppValue(A.sp, p[1])) * nx + bx(ppValue(A.sp, p[0]))] += wv; }
      else if (px !== 0 && py === 0) { strips[px < 0 ? "xl" : "xh"][by(ppValue(A.sp, p[1]))] += wv; }
      else if (py !== 0 && px === 0) { strips[py < 0 ? "yl" : "yh"][bx(ppValue(A.sp, p[0]))] += wv; }
      else if (px !== null && py !== null) { corner[(px < 0 ? "l" : "h") + (py < 0 ? "l" : "h")] += wv; } });
    var top = 0, stop = 0; cells.forEach(function (v) { top = Math.max(top, v); }); ["xl", "xh", "yl", "yh"].forEach(function (q) { strips[q].forEach(function (v) { stop = Math.max(stop, v); }); });
    Object.keys(corner).forEach(function (q) { stop = Math.max(stop, corner[q]); });
    var ink3 = hexRGB(cssVar("--accent")), surf = hexRGB(cssVar("--surface")), cw = (f.x1 - f.x0) / nx, ch = (f.y1 - f.y0) / ny, shade = function (v, mx) { return rgbCss(mixRGB(ink3, surf, Math.sqrt(v / mx))); };
    var href = rasterHref(f.x1 - f.x0, f.y1 - f.y0, function (g) { for (var yb = 0; yb < ny; yb++) { for (var xb = 0; xb < nx; xb++) { var v = cells[yb * nx + xb]; if (v > 0) { g.fillStyle = shade(v, top); g.fillRect(xb * cw, (ny - 1 - yb) * ch, cw + 0.5, ch + 0.5); } } } });
    var s = '<svg viewBox="0 0 ' + W + " " + H + '" class="v2chart v2corr" role="img" aria-label="' + esc(METRIC[k].label + ": " + cB.m.label + " against " + cA.m.label + ", problem by problem") + '" data-hover="' + hoverId(function (pt) {
      if (!(pt.x >= f.x0 && pt.x <= f.x1 && pt.y >= f.y0 && pt.y <= f.y1)) { return null; }
      var xb = Math.min(nx - 1, Math.floor((pt.x - f.x0) / cw)), yb = Math.min(ny - 1, Math.floor((f.y1 - pt.y) / ch)), v = cells[yb * nx + xb], x0 = A.lo + xb * (A.hi - A.lo) / nx, y0 = A.lo + yb * (A.hi - A.lo) / ny;
      return { text: cA.m.label + " " + vText(A, x0) + " to " + vText(A, x0 + (A.hi - A.lo) / nx) + ", " + cB.m.label + " " + vText(A, y0) + " to " + vText(A, y0 + (A.hi - A.lo) / ny) + ": " + (100 * v).toFixed(2) + " % of the run pairs" }; }) + '">' + frameSVG(f, labels);
    if (href) { s += '<image href="' + href + '" x="' + f.x0 + '" y="' + f.y0 + '" width="' + (f.x1 - f.x0) + '" height="' + (f.y1 - f.y0) + '" preserveAspectRatio="none"/>'; }
    ["xl", "xh", "yl", "yh"].forEach(function (q) { if (!f.has[q]) { return; } var arr = strips[q], vert = q.charAt(0) === "x";
      for (var b = 0; b < arr.length; b++) { if (!(arr[b] > 0)) { continue; } var c = shade(arr[b], stop), lab = "<title>" + (100 * arr[b]).toFixed(2) + " % of the run pairs</title>";
        if (vert) { var sx = q === "xl" ? f.sxl[0] : f.sxh[0], yy = f.y1 - (b + 1) * ch; s += '<rect x="' + sx + '" y="' + yy.toFixed(1) + '" width="' + f.ST + '" height="' + (ch + 0.4).toFixed(1) + '" fill="' + c + '">' + lab + "</rect>"; }
        else { var sy = q === "yl" ? f.syl[0] : f.syh[0]; s += '<rect x="' + (f.x0 + b * cw).toFixed(1) + '" y="' + sy + '" width="' + (cw + 0.4).toFixed(1) + '" height="' + f.ST + '" fill="' + c + '">' + lab + "</rect>"; } } });
    s += cornersSVG(f, corner, function (v) { return shade(v, stop); }, function (v) { return (100 * v).toFixed(2) + " % of the run pairs in both strips"; });
    s += '<line x1="' + f.x0 + '" y1="' + f.y1 + '" x2="' + f.x1 + '" y2="' + f.y0 + '" class="grid zero" stroke-dasharray="4 4"/></svg>';
    var lowName = METRIC[k].higher === false ? " (lower is better)" : METRIC[k].higher === true ? " (higher is better)" : "";
    return '<div class="v2charts v2one v2cvs">' + s + "</div>" +
      '<p class="v2hint">Each problem’s ' + esc(METRIC[k].label) + " under " + esc(cA.m.label) + " (across) and under " + esc(cB.m.label) + " (up), every run of one against every run of the other. On the dashed line both have the same value; below it " + esc(cB.m.label) + " has the lower value" + lowName + ". Of the run pairs, weighted as drawn: " + esc(cB.m.label) + " lower on " + (100 * below).toFixed(1) + " %, " + esc(cA.m.label) + " lower on " + (100 * above).toFixed(1) + " %, the same on " + (100 * same).toFixed(1) + " %. These shares describe the chart; the " + '<button type="button" class="v2linkbtn" data-view="paired">Paired differences</button> view tests the difference. ' + term("strips", "Strips along the frame") + ". " + term("weights", "How the runs are weighted") + ".</p>";
  }

  // ---- shell -----------------------------------------------------------------------------------------------------
  var VIEWS = [["curves", "Curves"], ["table", "Tables"], ["matrix", "Problem sets"], ["dist", "Distribution"], ["corr", "Correlations"], ["ranks", "Ranks"], ["paired", "Paired differences"], ["preds", "Predictions"]];
  // the metric a single-metric display shows: each of them keeps its own
  function focusKey() { return state.view === "dist" ? "dmetric" : state.view === "ranks" ? "rmetric" : "focus"; }
  // Each display carries its own controls. A control that cannot change what is on screen is not shown, and a
  // control has one place: the metric, the budget and the row layout of a snapshot are chosen on the display itself
  // (its bar), so the side panel never repeats them.
  var USES = {
    curves: { stat: 1, ci: 1, valid: 1 },
    table: { plots: 1, stat: 1, ci: 1, valid: 1 },
    matrix: { stat: 1, valid: 1 },
    dist: {},
    corr: {},
    ranks: {},
    paired: { plots: 1, base: 1, ci: 1 },
    preds: {}
  };
  function shownMetricKeys(view) { return view === "matrix" ? [state.focus] : view === "dist" ? [state.dmetric] : view === "corr" ? (state.cv === "matrix" ? D.metrics.filter(function (m) { return m.pp; }).map(function (m) { return m.key; }) : state.cv === "vs" ? [state.cy] : [state.cx, state.cy]) : view === "ranks" || view === "preds" ? [] : plotAxes(); }
  function usesFor(view) {
    var u = {}, src = USES[view] || {};
    Object.keys(src).forEach(function (k) { u[k] = src[k]; });
    if (shownMetricKeys(view).some(function (k) { return METRIC[k] && METRIC[k].worst !== undefined; })) { u.impute = 1; }
    if (view === "table" && state.rows !== "cats") { delete u.rung; }   // the budget only binds the by-catalog table
    if (view === "dist" && METRIC[state.dmetric].kind !== "rate" && state.dmode === "rungs") { delete u.rung; }
    if (view === "ranks" && state.xaxis === "time" && anyTime()) { delete u.rung; }
    return u;
  }
  // One line under the release's title: how many methods are finished, in progress and scheduled. The Progress page
  // (progress.html, pages.js) shows each method budget by budget; the counts are the exporter's (progress_summary).
  function progressLine() {
    var s = D.summary;
    if (!s) { return ""; }
    var parts = [[s.finished.length, "finished"], [s.in_progress.length, "in progress"], [s.scheduled.length, "scheduled"]]
      .filter(function (p) { return p[0] > 0; }).map(function (p, i) { return p[0] + (i ? " " : p[0] === 1 ? " method " : " methods ") + p[1]; });
    return '<p class="v2progline"><span class="v2kicker">Progress</span> ' + parts.join(" · ") + ' · <a href="progress.html">details by method and budget</a></p>';
  }
  // When the release was last refreshed: stored with its offset, shown in the reader's own time zone, with how long ago
  function releaseStamp(rel) {
    var t = rel.updated ? new Date(rel.updated) : null;
    if (!t || isNaN(t.getTime())) { return rel.generated ? '<span class="v2updated">Updated ' + esc(rel.generated) + '</span>' : ""; }
    var abs = t.toLocaleString("en-GB", { day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit", timeZoneName: "short" });
    var mins = Math.round((Date.now() - t.getTime()) / 60000), ago;
    if (mins < 1) { ago = "just now"; } else if (mins < 60) { ago = mins + (mins === 1 ? " minute" : " minutes") + " ago"; } else if (mins < 48 * 60) { var h = Math.round(mins / 60); ago = h + (h === 1 ? " hour" : " hours") + " ago"; } else { ago = Math.round(mins / 1440) + " days ago"; }
    return '<time class="v2updated" datetime="' + esc(rel.updated) + '">Updated ' + esc(abs) + ' <span class="v2ago">· ' + ago + '</span></time>';
  }
  function shell() {
    var rel = D.release;
    var stamp = releaseStamp(rel);
    var catList = CATS.map(function (c) { var m = CAT[c]; return '<label title="' + esc(GROUPS[m.group] + (m.mu ? " · typical formula length " + m.mu[1] + " bits (middle half: " + m.mu[0] + " to " + m.mu[2] + ")" : "")) + '"><input type="checkbox" data-c="' + c + '"> ' + esc(c) + ' <span class="v2hint">' + m.laws + '</span></label>'; }).join("");
    var methList = D.methods.filter(withData).map(function (m) { return '<div class="v2meth"><label><input type="checkbox" data-m="' + m.key + '"><input type="color" class="v2swatch" data-m="' + m.key + '" value="' + baseColorOf(m) + '" title="Colour for ' + esc(m.label) + '"><span class="v2mname">' + esc(m.label) + '</span></label>' + (m.local ? ' <span class="v2tag v2tag-local">local only</span>' : "") + ' <span class="v2hint">' + esc(m.param) + '</span>' + (m.selection ? " " + help(m.selection, "How does " + m.label + " choose its prediction?") : "") + ' <span class="v2tag" title="' + esc(PROV_NOTE[m.provenance] || "") + '">' + esc(PROV[m.provenance] || m.provenance || "") + '</span><button type="button" class="v2reset" data-m="' + m.key + '" title="Reset colour to default" hidden>↺</button></div>'; }).join("") || '<span class="v2hint">no method has finished a budget yet</span>';
    var metricList = MGROUPS.map(function (g) { var ms = D.metrics.filter(function (m) { return m.group === g; }); return '<div class="v2mgroup" data-group="' + esc(g) + '"><h4>' + esc(g) + '</h4>' + ms.map(function (m) { return '<div class="v2metric" data-tier="' + m.tier + '" data-key="' + m.key + '"><label><input type="checkbox" data-p="' + m.key + '"> ' + esc(m.label) + '</label> ' + mhelp(m) + '</div>'; }).join("") + "</div>"; }).join("");
    root.innerHTML =
      '<div class="v2relhead"><div><h2 class="v2hltitle">Release ' + esc(rel.id) + (rel.title !== rel.id ? ' · ' + esc(rel.title) : '') + '</h2><p class="v2hlsub">' + stamp + (rel.notes ? ' · ' + esc(rel.notes) : '') + '</p></div><div class="v2row"><button type="button" class="v2btn" data-act="link">copy link to this view</button><span class="v2linkok v2hint" hidden>link copied</span></div></div>' +
      progressLine() +
      '<div class="v2tabs" role="tablist">' + VIEWS.map(function (v) { return '<button type="button" class="v2tab" role="tab" data-view="' + v[0] + '">' + v[1] + '</button>'; }).join("") + '</div>' +
      '<div class="v2layout"><aside class="v2side">' +
      // 1. what this display shows. Every row declares the views it belongs to; the rest stay out of the way.
      '<div class="v2panel v2panel-show"><h3><span class="v2showtitle">Plots</span> <span class="v2hint v2metcount"></span></h3>' +
      '<div class="v2row" data-uses="plots"><input type="search" class="v2q" placeholder="filter metrics" aria-label="filter metrics"><label><input type="checkbox" class="v2tier"> show all ' + D.metrics.length + '</label></div>' +
      '<div class="v2metrics" data-uses="plots">' + metricList + '</div>' +
      '<div class="v2row" data-uses="focus"><span class="v2lab">metric</span>' + pickButton("v2focus", 'data-axis="focus" aria-label="metric shown in this view"', state[focusKey()]) + '</div>' +
      '<div class="v2row" data-uses="rows"><span class="v2lab">rows</span><label><input type="radio" name="v2rows" value="rungs"> one per budget</label><label><input type="radio" name="v2rows" value="cats"> one per problem set</label></div>' +
      '<div class="v2row" data-uses="base"><span class="v2lab">baseline</span><select class="v2base" aria-label="baseline method">' + D.methods.filter(withData).map(function (m) { return '<option value="' + m.key + '">' + esc(m.label) + '</option>'; }).join("") + '</select></div>' +
      '<div class="v2row" data-uses="rung"><span class="v2lab">budget</span><select class="v2rung" aria-label="budget per problem">' + D.rungs.map(function (r) { return '<option value="' + r + '">' + r + '</option>'; }).join("") + '</select><span class="v2hint">in each method\u2019s own unit</span></div></div>' +
      // 2. what it is shown for
      '<div class="v2panel"><h3>Methods</h3><div class="v2methods">' + methList + '</div>' +
      '<div class="v2row v2faderow"><span class="v2lab">faded at ' + help("A click on a method\u2019s box cycles it: shown (ticked), hidden (empty), faded (a dash). A faded method stays in every chart, behind the shown ones, in its colour blended into the background: this much of the colour, the rest background.", "What does a box\u2019s dash mean?") + '</span><input type="range" class="v2fade" min="0.05" max="0.9" step="0.05" aria-label="how much of its colour a faded method keeps"><output class="v2fadeval"></output></div>' +
      '<div class="v2row v2addm"><button type="button" class="v2btn v2addmopen" data-act="add-method">open a method with a key</button>' +
      '<span class="v2addmbox" hidden><input type="text" class="v2addmkey" placeholder="key" autocomplete="off" autocapitalize="off" spellcheck="false" aria-label="key for a method shared with you">' +
      '<button type="button" class="v2btn" data-act="add-method-go">add</button></span>' +
      '<span class="v2hint v2addmmsg" role="status"></span></div>' +
      '<div class="v2row v2colour"><button type="button" class="v2btn" data-act="reset-colours">reset all colours</button><span class="v2hint v2cookie" hidden>A single functional cookie remembers your colour choices on this device: no tracking, no third parties. It is written only when you change a colour; \u201creset all colours\u201d deletes it.</span></div></div>' +
      '<div class="v2panel"><h3>Problem sets <span class="v2hint v2catcount"></span> <a class="v2hint" href="https://srbf.readthedocs.io/en/latest/benchmarks/#the-srbf-suite" target="_blank" rel="noopener">what are these?</a></h3><div class="v2row"><button type="button" data-act="all">all</button><button type="button" data-act="none">none</button><button type="button" data-act="phys">physics</button><button type="button" data-act="classic">classic</button><button type="button" data-act="synth">machine-generated</button></div><div class="v2cats">' + catList + '</div></div>' +
      // 3. how the numbers are read
      '<div class="v2panel"><h3>Reading</h3>' +
      '<div class="v2row" data-uses="stat"><span class="v2lab">statistic ' + help("Mean: " + TERMS.mean + " Median: " + TERMS.median, "How are the mean and the median taken?") + '</span><label><input type="radio" name="v2stat" value="mean"> mean</label><label><input type="radio" name="v2stat" value="median"> median</label></div>' +
      '<div class="v2row" data-uses="xaxis"><span class="v2lab">x axis ' + help("Time: " + TERMS.time + " Candidates: " + TERMS.candidates, "What do the two x-axes measure?") + '</span><label><input type="radio" name="v2xaxis" value="time" class="v2xtime"> time</label><label><input type="radio" name="v2xaxis" value="rung"> candidates</label><span class="v2hint v2xtimehint"></span></div>' +
      '<div class="v2row v2checks" data-uses="ci"><span class="v2lab" data-uses="ci">95 % intervals ' + help(TERMS.interval) + '</span>' +
      '<label data-uses="ci"><input type="checkbox" class="v2band"> bands</label><label data-uses="ci"><input type="checkbox" class="v2cross"> crosses</label></div>' +
      '<div class="v2row" data-uses="impute"><span class="v2lab">no usable formula ' + help(TERMS.impute, "How does a problem without a usable formula count?") + '</span><label><input type="checkbox" class="v2impute"> count as 0 in overlap metrics</label></div>' +
      '<div class="v2row" data-uses="valid"><span class="v2lab">hollow if under ' + help(TERMS.valid, "When is a point drawn hollow?") + '</span><input type="range" class="v2valid" min="0" max="100" step="5" aria-label="share of the problems a point must be based on to be drawn solid, in percent"><output class="v2validout"></output></div></div>' +
      '</aside><section class="v2main"><p class="v2err" role="alert"></p><div class="v2view"></div></section></div>';
    var hasTiming = anyTime();
    root.querySelector(".v2xtime").disabled = !hasTiming; root.querySelector(".v2xtimehint").textContent = hasTiming ? "" : "(no measurements yet)";
    if (!hasTiming && state.xaxis === "time") { state.xaxis = "rung"; }
  }
  function syncControls() {
    root.querySelectorAll(".v2cats input").forEach(function (i) { i.checked = state.cats.indexOf(i.dataset.c) >= 0; });
    root.querySelectorAll(".v2methods input[type=checkbox]").forEach(function (i) { var v = exVisOf(i.dataset.m); i.checked = v === "full"; i.indeterminate = v === "dim"; i.title = v === "full" ? "shown (a click hides it)" : v === "dim" ? "faded (a click shows it)" : "hidden (a click fades it in)";
      var row = i.closest(".v2meth"); if (row) { row.classList.toggle("v2faded", v === "dim"); row.classList.toggle("v2hidden", v === "hidden"); row.style.setProperty("--fade", String(state.fade)); } });
    var fd = root.querySelector(".v2fade"); if (fd && document.activeElement !== fd) { fd.value = String(state.fade); } var fo = root.querySelector(".v2fadeval"); if (fo) { fo.textContent = Math.round(100 * state.fade) + " %"; }
    root.querySelectorAll(".v2metrics input[type=checkbox]").forEach(function (i) { i.checked = plotMetrics().indexOf(i.dataset.p) >= 0; });
    var q = state.q.toLowerCase();
    root.querySelectorAll(".v2metric").forEach(function (l) { var mm = METRIC[l.dataset.key]; var show = state.tier === "all" || l.dataset.tier === "main" || plotMetrics().indexOf(l.dataset.key) >= 0; if (q) { show = mm.label.toLowerCase().indexOf(q) >= 0 || l.dataset.key.indexOf(q) >= 0 || mm.group.toLowerCase().indexOf(q) >= 0; } l.hidden = !show; });
    root.querySelectorAll(".v2mgroup").forEach(function (g) { g.hidden = !Array.prototype.some.call(g.querySelectorAll(".v2metric"), function (l) { return !l.hidden; }); });
    root.querySelector(".v2tier").checked = state.tier === "all"; if (root.querySelector(".v2q").value !== state.q) { root.querySelector(".v2q").value = state.q; }
    root.querySelectorAll("input[name=v2stat]").forEach(function (i) { i.checked = i.value === state.stat; });
    root.querySelectorAll("input[name=v2xaxis]").forEach(function (i) { i.checked = i.value === state.xaxis; });
    root.querySelector(".v2band").checked = state.band; root.querySelector(".v2cross").checked = state.cross;
    root.querySelector(".v2impute").checked = state.impute;
    var vs = root.querySelector(".v2valid"); if (document.activeElement !== vs) { vs.value = String(state.valid); } root.querySelector(".v2validout").textContent = state.valid + " % of the problems";
    setPick(root.querySelector(".v2focus"), state[focusKey()]); root.querySelector(".v2rung").value = String(state.rung);
    root.querySelectorAll("input[name=v2rows]").forEach(function (i) { i.checked = i.value === state.rows; });
    if (state.base) { root.querySelector(".v2base").value = state.base; }
    root.querySelectorAll(".v2tab").forEach(function (b) { b.classList.toggle("active", b.dataset.view === state.view); b.setAttribute("aria-selected", b.dataset.view === state.view ? "true" : "false"); });
    root.querySelectorAll(".v2swatch").forEach(function (i) { var m = D.methods.filter(function (x) { return x.key === i.dataset.m; })[0]; if (m && document.activeElement !== i) { i.value = baseColorOf(m); } });
    root.querySelectorAll(".v2reset").forEach(function (b) { b.hidden = !userColors[b.dataset.m]; });
    var single = state.view === "matrix" || state.view === "dist" || state.view === "ranks";
    root.querySelector(".v2metcount").textContent = single ? "" : plotMetrics().length + " plotted";
    root.querySelector(".v2showtitle").textContent = single ? "Metric" : state.view === "paired" ? "Compared metrics" : "Plots";
    var uses = usesFor(state.view);   // hide every control this display cannot use, then the panels left empty
    root.querySelectorAll("[data-uses]").forEach(function (el) { el.hidden = !el.dataset.uses.split(" ").some(function (k) { return uses[k]; }); });
    root.querySelectorAll(".v2panel").forEach(function (p) {
      var rows = p.querySelectorAll("[data-uses]");
      if (rows.length) { p.hidden = !Array.prototype.some.call(rows, function (r) { return !r.hidden; }); }
    });
    root.querySelector(".v2catcount").textContent = state.cats.length + " of " + CATS.length + " \u00b7 " + laws(state.cats).toLocaleString() + " problems";
  }
  var rt = null;
  function scheduleRender() { clearTimeout(rt); rt = setTimeout(render, 30); }
  function render() {
    if (!root) {   // the home page: the headline, and the release's line of progress where the page has a place for it
      renderHeadline();
      var pl = document.getElementById("results-progress-v2");
      if (pl) { pl.innerHTML = progressLine(); }
      var up = document.getElementById("results-updated-v2");
      if (up) { up.innerHTML = releaseStamp(D.release); }
      return;
    }
    try {
      syncControls();
      var shown = shownMethods(), view = root.querySelector(".v2view");
      if (!rungChosen) { rungChosen = true; var br = bestRung(shown); if (br) { state.rung = br; } syncControls(); }
      if (state.view !== "dist" && state.view !== "paired" && state.view !== "corr") {   // histograms the current view needs
        (state.view === "matrix" ? [METRIC[state.focus]] : D.metrics.filter(function (m) { return plotAxes().indexOf(m.key) >= 0; })).forEach(function (m) { if (needsHist(m)) { ensureHists(m); } });
      }
      renderHeadline();
      // Every display but Curves reads every method at one position (between its budgets too): a table by problem set,
      // problem sets, a distribution, correlations, predictions, and the snapshots of Ranks and Paired differences. Their
      // charts along the budgets, a table by budget and a distribution along the budgets stay on budgets a method was
      // run at, and so do Ranks and Paired differences until they turn BETWEEN on for their snapshot.
      var atPos = state.view === "matrix" || (state.view === "table" && state.rows === "cats") || (state.view === "dist" && !(METRIC[state.dmetric].kind !== "rate" && state.dmode === "rungs")) ||
        state.view === "corr" || state.view === "preds" || state.view === "ranks" || state.view === "paired";
      if (atPos) { settlePos(shown); }
      BETWEEN = atPos && state.view !== "ranks" && state.view !== "paired";
      try {
      view.innerHTML = state.view === "table" ? renderTable(shown) : state.view === "matrix" ? renderMatrix(shown) : state.view === "dist" ? renderDist(shown) : state.view === "corr" ? renderCorr(shown) : state.view === "ranks" ? renderRanks(shown) : state.view === "paired" ? renderPaired(shown) : state.view === "preds" ? renderPreds(shown) : renderCurves(shown);
      } finally { BETWEEN = false; }
      // A redraw replaces the button an open menu hangs on (a histogram that arrives, a container that settles).
      // The menu moves to the button's successor; it closes only when the control itself is gone.
      if (pickerFor && !document.body.contains(pickerFor)) { var again = samePick(pickerFor); if (again) { pickerFor = again; again.setAttribute("aria-expanded", "true"); placePicker(again); } else { closePicker(); } }
      root.querySelector(".v2err").textContent = ""; save();
    } catch (e) { root.querySelector(".v2err").textContent = "The explorer hit an error while drawing: " + (e && e.message ? e.message : e) + ". Reload the page, or press “all” under Problem sets to reset the selection."; if (window.console) { console.error(e); } }
  }

  // ---- events ----------------------------------------------------------------------------------------------------
  var SETTABLE = { pos: 1, pm: 1, rung: 1, dmode: 1, dnorm: 1, dmetric: 1, xaxis: 1, pset: 1, prun: 1, pprob: 1, pnum: 1, cv: 1, ca: 1, cb: 1 };
  function cycleMethod(k) {
    var v = nextVis(exVisOf(k));
    state.methods = state.methods.filter(function (c) { return c !== k; }); state.dim = state.dim.filter(function (c) { return c !== k; });
    if (v !== "hidden") { state.methods.push(k); } if (v === "dim") { state.dim.push(k); }
  }
  function setState(key, val) {
    if (!SETTABLE[key]) { return; }
    if (key === "rung") { var r = parseInt(val, 10); if (D.rungs.indexOf(r) >= 0) { state.rung = r; } return; }
    if (key === "pos") { var x = parseFloat(val); if (x > 0) { if (state.pm === "time") { state.pt = x; } else { state.rung = x; } } return; }
    if (key === "pm") { state.pm = val === "time" && anyTime() ? "time" : "budget"; return; }
    if (key === "dmetric" && !METRIC[val]) { return; }
    if (key === "pset") { if (CAT[val]) { state.pset = val; state.pprob = 0; } return; }
    if (key === "prun") { state.prun = val === "2" ? 2 : 1; return; }
    if (key === "pprob") { state.pprob = Math.max(0, parseInt(val, 10) || 0); return; }
    if (key === "pnum") { state.pprob = Math.max(0, (parseInt(val, 10) || 1) - 1); return; }
    if (key === "cv") { if (CMODES.some(function (o) { return o[0] === val; })) { state.cv = val; csel = null; } return; }
    if (key === "ca" || key === "cb") { if (D.methods.some(function (m) { return m.key === val; })) { state[key] = val; } return; }
    state[key] = val;
  }
  // the explorer's controls, where the explorer is mounted
  var on = root ? root.addEventListener.bind(root) : function () { /* the home page: no controls */ };
  if (root) { shell(); }
  on("change", function (e) {
    var t = e.target;
    if (t.dataset.c) { if (t.checked) { state.cats.push(t.dataset.c); } else { state.cats = state.cats.filter(function (c) { return c !== t.dataset.c; }); } }
    else if (t.dataset.m && t.type === "checkbox") { cycleMethod(t.dataset.m); }
    else if (t.dataset.p) { if (t.checked) { state.plots.push({ x: lastAxis(), y: t.dataset.p }); } else { state.plots = state.plots.filter(function (c) { return c.y !== t.dataset.p; }); } }
    else if (t.name === "v2stat") { state.stat = t.value; } else if (t.name === "v2xaxis") { state.xaxis = t.value; } else if (t.name === "v2rows") { state.rows = t.value; }
    else if (t.classList.contains("v2band")) { state.band = t.checked; } else if (t.classList.contains("v2cross")) { state.cross = t.checked; } else if (t.classList.contains("v2impute")) { state.impute = t.checked; } else if (t.classList.contains("v2tier")) { state.tier = t.checked ? "all" : "main"; }
    else if (t.classList.contains("v2rung")) { state.rung = parseInt(t.value, 10); } else if (t.classList.contains("v2base")) { state.base = t.value; }
    else if (t.classList.contains("v2pos")) { setState("pos", String(posFromSlider(t))); }
    else if (t.classList.contains("v2posval")) { var typed = parsePos(t.value); if (typed !== null) { setState("pos", String(typed)); } }
    else if (t.dataset.state) { setState(t.dataset.state, t.value); }
    else if (t.classList.contains("v2swatch")) { userColors[t.dataset.m] = t.value; writeCookie(userColors); root.querySelector(".v2cookie").hidden = false; }
    else { return; }
    render();
  });
  on("input", function (e) { var t = e.target; if (t.classList.contains("v2fade")) { state.fade = Math.min(0.9, Math.max(0.05, parseFloat(t.value) || FADE_DEFAULT)); var fo = root.querySelector(".v2fadeval"); if (fo) { fo.textContent = Math.round(100 * state.fade) + " %"; } scheduleRender(); return; } if (t.classList.contains("v2pos")) { var o = t.closest(".v2posstep").querySelector(".v2posval"), v = posFromSlider(t); if (o) { o.value = posNum(v, state.pm === "time"); } return; } if (t.classList.contains("v2valid")) { state.valid = Math.min(100, Math.max(0, parseInt(t.value, 10) || 0)); syncControls(); scheduleRender(); } else if (t.classList.contains("v2q")) { state.q = t.value; syncControls(); } else if (t.classList.contains("v2swatch")) { userColors[t.dataset.m] = t.value; render(); } });
  // the arrow keys on the position's slider step to the next time limit or budget, as its buttons do; Home and End
  // keep their own meaning (the two ends of the range)
  on("keydown", function (e) {
    var t = e.target; if (!t.classList || !t.classList.contains("v2pos")) { return; }
    var dir = e.key === "ArrowRight" || e.key === "ArrowUp" ? "next" : e.key === "ArrowLeft" || e.key === "ArrowDown" ? "prev" : null; if (!dir) { return; }
    e.preventDefault(); if (!t.dataset[dir]) { return; }
    setState("pos", t.dataset[dir]); render();
    var again = root.querySelector(".v2viewbar input.v2pos"); if (again) { again.focus(); }
  });
  // a cell of the Correlations view's matrix draws its two metrics
  function openPair(g) { state.cx = g.getAttribute("data-cx"); state.cy = g.getAttribute("data-cy"); state.cv = "contour"; render(); }
  on("click", function (e) {
    var cellG = e.target.closest ? e.target.closest(".v2ccell") : null; if (cellG && root.contains(cellG)) { openPair(cellG); return; }
    var b = e.target.closest ? e.target.closest("button") : null; if (!b || !root.contains(b) || b.classList.contains("v2help")) { return; }
    if (b.dataset.view) { state.view = b.dataset.view; render(); return; }
    if (b.dataset.set) { var kv = b.dataset.set.split(":"); setState(kv[0], kv.slice(1).join(":")); render(); return; }
    if (b.classList.contains("v2reset")) { delete userColors[b.dataset.m]; writeCookie(userColors); render(); return; }
    if (b.classList.contains("v2rmplot")) { state.plots.splice(+b.dataset.i, 1); render(); return; }
    if (b.dataset.act === "corr-swap") { var cxy = state.cx; state.cx = state.cy; state.cy = cxy; render(); return; }
    if (b.dataset.act === "corr-clear") { csel = null; render(); return; }
    if (b.dataset.act === "corr-open" && csel) { state.view = "preds"; state.pset = csel.c; state.pprob = csel.row; if (state.pm === "time" && refTime(csel.m, csel.r)) { state.pt = refTime(csel.m, csel.r); } else { state.pm = "budget"; state.rung = csel.r; } state.prun = csel.d === 2 ? 2 : 1; if (state.cats.indexOf(csel.c) < 0) { state.cats.push(csel.c); } render(); return; }
    if (b.classList.contains("v2swap")) { var sp = state.plots[+b.dataset.i]; if (sp && METRIC[sp.x]) { state.plots[+b.dataset.i] = { x: sp.y, y: sp.x }; render(); } return; }
    var act = b.dataset.act; if (!act) { return; }
    if (act === "add-plot") {   // a plot the reader does not have yet, on the axis the last one uses
      var taken = plotMetrics();
      var next = D.metrics.filter(function (m) { return m.tier === "main" && taken.indexOf(m.key) < 0; })[0]
        || D.metrics.filter(function (m) { return taken.indexOf(m.key) < 0; })[0] || D.metrics[0];
      state.plots.push({ x: lastAxis(), y: next.key }); render(); return;
    }
    if (act === "add-method") {
      var box = root.querySelector(".v2addmbox"), open = root.querySelector(".v2addmopen");
      if (box) { box.hidden = false; } if (open) { open.hidden = true; }
      var f = root.querySelector(".v2addmkey"); if (f) { f.focus(); }
      return;
    }
    if (act === "add-method-go") { var fk = root.querySelector(".v2addmkey"); if (fk && fk.value.trim()) { tryKey(fk.value.trim(), false); } return; }
    if (act === "all") { state.cats = CATS.slice(); } else if (act === "none") { state.cats = []; }
    else if (act === "phys" || act === "classic" || act === "synth") { var g = { phys: "physics", classic: "classical", synth: "synthetic" }[act]; state.cats = CATS.filter(function (c) { return CAT[c].group === g; }); }
    else if (act === "reset-colours") { userColors = {}; writeCookie(userColors); root.querySelector(".v2cookie").hidden = true; }
    else if (act === "link") { save(); var url = window.location.href; var ok = root.querySelector(".v2linkok"); var done = function () { ok.hidden = false; setTimeout(function () { ok.hidden = true; }, 1800); }; if (navigator.clipboard && navigator.clipboard.writeText) { navigator.clipboard.writeText(url).then(done, function () { window.prompt("Link to this view", url); }); } else { window.prompt("Link to this view", url); } return; }
    else if (act === "copy-tsv" || act === "csv") { if (!lastTable) { return; } var sep = act === "csv" ? "," : "\t"; var qf = function (s) { s = String(s); return act === "csv" && /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s; }; var text = [lastTable.header].concat(lastTable.rows).map(function (r) { return r.map(qf).join(sep); }).join("\n");
      if (act === "csv") { var a = document.createElement("a"); a.href = "data:text/csv;charset=utf-8," + encodeURIComponent(text); a.download = "srbf-" + REL + "-table-rung" + state.rung + ".csv"; document.body.appendChild(a); a.click(); a.remove(); } else if (navigator.clipboard && navigator.clipboard.writeText) { navigator.clipboard.writeText(text).then(null, function () { window.prompt("Copy the table", text); }); } else { window.prompt("Copy the table", text); } return; }
    render();
  });
  // floating one-sentence explanations (fixed at body level: never reflow the layout)
  var popArmed = false;
  function closePop() { var p = document.querySelector(".v2pop"); if (p) { p.remove(); } }
  function closeArmed() { if (popArmed) { closePop(); } }
  document.addEventListener("click", function (e) {
    var el = e.target.closest ? e.target.closest(".v2term, .v2help") : null; var open = document.querySelector(".v2pop"); var prev = open && open._anchor; closePop();
    if (!el || prev === el || !((root && root.contains(el)) || (headRoot && headRoot.contains(el)))) { return; }
    e.preventDefault();
    var pop = document.createElement("div"); pop.className = "v2pop"; var tipText = el.classList.contains("v2help") ? el.dataset.help : TERMS[el.dataset.term]; if (!tipText) { if (window.console) { console.warn("srbf: no text for hint", el.dataset.term); } return; } pop.textContent = tipText; pop._anchor = el; document.body.appendChild(pop);
    var margin = 8; pop.style.maxWidth = Math.min(520, window.innerWidth - 2 * margin) + "px"; var r = el.getBoundingClientRect();
    var left = Math.min(Math.max(r.left, margin), window.innerWidth - pop.offsetWidth - margin), top = r.bottom + 6;
    if (top + pop.offsetHeight > window.innerHeight - margin && r.top - pop.offsetHeight - 6 > margin) { top = r.top - pop.offsetHeight - 6; }
    pop.style.left = left + "px"; pop.style.top = top + "px";
    popArmed = false; window.requestAnimationFrame(function () { window.requestAnimationFrame(function () { popArmed = true; }); });
  });
  // ---- the metric picker: a menu in columns, at body level like the explanations ---------------------------------
  var pickerEl = null, pickerFor = null;
  function closePicker() {
    if (pickerEl) { pickerEl.remove(); pickerEl = null; }
    if (pickerFor) { pickerFor.setAttribute("aria-expanded", "false"); pickerFor = null; }
  }
  function samePick(old) {   // the control a redraw has put where `old` was
    return Array.prototype.filter.call(document.querySelectorAll(".v2pick"), function (b) { return b.className === old.className && b.dataset.axis === old.dataset.axis && b.dataset.i === old.dataset.i; })[0] || null;
  }
  function placePicker(btn) {
    var margin = 8, r = btn.getBoundingClientRect(), h = pickerEl.offsetHeight, w = pickerEl.offsetWidth;
    var viewH = window.visualViewport ? Math.min(window.innerHeight, window.visualViewport.height) : window.innerHeight;   // what a keyboard leaves
    var left = Math.min(Math.max(r.left, margin), Math.max(margin, window.innerWidth - w - margin));
    var top = r.bottom + 6;
    if (top + h > viewH - margin) { top = Math.max(margin, viewH - h - margin); }
    pickerEl.style.left = left + "px"; pickerEl.style.top = top + "px";
  }
  function openPicker(btn) {
    var axis = btn.dataset.axis;
    closePicker(); closePop();
    pickerEl = document.createElement("div");
    pickerEl.className = "v2picker"; pickerEl.setAttribute("role", "dialog");
    pickerEl.setAttribute("aria-label", axis === "x" ? "What the x axis shows" : "Which metric to show");
    pickerEl.innerHTML = pickerHTML(axis, btn.dataset.k);
    document.body.appendChild(pickerEl);
    pickerFor = btn; btn.setAttribute("aria-expanded", "true"); placePicker(btn);
    var q = pickerEl.querySelector(".v2pickq");
    q.addEventListener("input", function () {
      var v = q.value.trim().toLowerCase();
      pickerEl.querySelectorAll(".v2pickitem").forEach(function (it) {   // the short name answers the filter too ("vnrr")
        it.hidden = !!v && it.textContent.toLowerCase().indexOf(v) < 0 && it.dataset.k.indexOf(v) < 0 && (it.dataset.alt || "").indexOf(v) < 0;
      });
      pickerEl.querySelectorAll(".v2pickgroup").forEach(function (g) { g.hidden = !Array.prototype.some.call(g.querySelectorAll(".v2pickitem"), function (it) { return !it.hidden; }); });
    });
    pickerEl.addEventListener("click", function (e) {
      var it = e.target.closest ? e.target.closest(".v2pickitem") : null;
      if (!it || it.disabled) { return; }
      if (axis === "focus") { state[focusKey()] = it.dataset.k; }
      else if (axis === "cx" || axis === "cy") { state[axis] = it.dataset.k; }
      else if (btn.dataset.i !== undefined) { state.plots[+btn.dataset.i][axis] = it.dataset.k; }
      closePicker(); render();
    });
    // The cursor goes into the search field only where typing costs nothing. On a touch screen a focused field
    // summons the on-screen keyboard over the menu the reader has just asked for.
    if (!narrow() && !coarse()) { q.focus(); }
  }
  document.addEventListener("click", function (e) {
    if (!e.target.closest) { return; }
    if (e.target.closest(".v2picker")) { return; }
    var trigger = e.target.closest(".v2pick");
    if (!trigger || trigger === pickerFor) { closePicker(); return; }
    if (root && root.contains(trigger)) { openPicker(trigger); }
  });
  document.addEventListener("keydown", function (e) { if (e.key === "Escape") { closePop(); closePicker(); } });
  // Only a change of WIDTH changes the layout. An on-screen keyboard (or a browser bar that slides away) takes
  // height and nothing else: the open menu stays, and moves back into what is left of the window.
  var lastInnerW = window.innerWidth;
  window.addEventListener("resize", function () {
    if (window.innerWidth === lastInnerW) { if (pickerEl && pickerFor) { placePicker(pickerFor); } return; }
    lastInnerW = window.innerWidth; closeArmed(); closePicker(); scheduleRender();
  });
  if (window.visualViewport) { window.visualViewport.addEventListener("resize", function () { if (pickerEl && pickerFor) { placePicker(pickerFor); } }); }
  // The first paint can measure a container that has not settled (fonts, the sidebar, a scrollbar), and a
  // chart built for the wrong width is a chart whose labels are the wrong size. Watch and redraw.
  if (window.ResizeObserver) {
    var lastW = 0;
    var ro = new ResizeObserver(function () { var w = hostWidth(); if (Math.abs(w - lastW) > 8) { lastW = w; scheduleRender(); } });
    [root && root.querySelector(".v2main"), headRoot].forEach(function (el) { if (el) { ro.observe(el); } });
  }
  document.addEventListener("scroll", closeArmed, true);
  // The Correlations view's charts answer the pointer: each registers what a position in it means (HOVERS), a click
  // in the points display picks the run under it. A touch is a click.
  function hoverAt(e, click) {
    var svg = e.target.closest ? e.target.closest("svg[data-hover]") : null, fn = svg && HOVERS[svg.getAttribute("data-hover")];
    if (!fn) { hideTip(); return; }
    var got = fn(chartPoint(svg, e), click); if (got) { showTip(got.text, e); } else { hideTip(); }
  }
  on("mousemove", function (e) { hoverAt(e, false); });
  on("mouseleave", hideTip);
  on("click", function (e) { if (e.target.closest && e.target.closest("svg[data-hover]")) { hoverAt(e, true); } });
  document.addEventListener("scroll", hideTip, true);
  // A canvas does not follow the theme by itself: the Correlations view draws its rasters again when the theme changes.
  if (window.MutationObserver) { new MutationObserver(function () { if (state.view === "corr") { scheduleRender(); } }).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] }); }
  if (window.matchMedia) { var dark = window.matchMedia("(prefers-color-scheme: dark)"); if (dark.addEventListener) { dark.addEventListener("change", function () { if (state.view === "corr") { scheduleRender(); } }); } }
  on("keydown", function (e) {
    if ((e.key === "Enter" || e.key === " ") && e.target.classList && e.target.classList.contains("v2ccell")) { e.preventDefault(); openPair(e.target); return; }
    if (e.key !== "Enter" || !e.target.classList || !e.target.classList.contains("v2addmkey")) { return; }
    e.preventDefault(); var v = e.target.value.trim(); if (v) { tryKey(v, false); }
  });
  render();
  // a key given earlier in this tab opens the same payload again without asking for it. (The page does not scroll to
  // the explorer on load: it is the top of its own page, and the address carries the state after every change, so a
  // reload scrolled the release's title under the sticky navigation bar.)
  if (root) {
    savedKeys().forEach(function (k) { tryKey(k, true); });
  }
})();
