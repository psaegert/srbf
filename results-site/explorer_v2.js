/* Results explorer for benchmark releases from 2026-09 on (srbf 0.20 / flash-ansr 0.18, the two-part code).
 *
 * Reads window.RESULTS_V2 (schema 2): the metric REGISTRY (every metric of the 2026-07 site and the new ones) and per
 * method x catalog x rung cell counts, sums and sums of squares, so any catalog subset is pooled on the client with
 * confidence bands. Per-metric histograms (pooled medians, the Distribution view) and the draw-1 paired contrasts
 * (the Paired view) are fetched on demand from <base>hist/<metric>.js and <base>paired.js.
 * window.RESULTS_V2_PRIVATE, if a LOCAL build provides it, is merged in (results-site/README.md, "Local-only
 * methods"); the public page never references such a file.
 * Views: Curves (every plotted metric vs budget or time) | Table (rungs, or catalogs at one rung) | Catalogs (matrix
 * of one metric at one rung) | Distribution (histograms at one rung; per-catalog rates for rate metrics) | Paired Δ
 * (each method against a baseline on the same laws: exact McNemar for rates, paired t + sign test otherwise).
 * State lives in the URL (?release=...&v=...) for sharing and in localStorage for convenience; colours in the same
 * first-party cookie the 2026-07 explorer uses (srbf_colors), written only on an explicit change. */
(function () {
  "use strict";
  var root = document.getElementById("results-explorer-v2");
  if (!root || typeof window.RESULTS_V2 === "undefined") { return; }
  var headRoot = document.getElementById("results-headline-v2");   // the two fixed charts above the explorer
  var D = JSON.parse(JSON.stringify(window.RESULTS_V2));
  var REL = D.release.id;
  // another release is on screen (the routing script decided before this file ran): leave the page and its URL alone
  if (window.SRBF_RELEASE && window.SRBF_RELEASE !== REL) { return; }
  var SOURCES = [{ base: D.base, local: false }];
  if (typeof window.RESULTS_V2_PRIVATE !== "undefined") {   // local overlay: additional methods, same schema
    var P = window.RESULTS_V2_PRIVATE;
    (P.methods || []).forEach(function (m) { if (!D.methods.some(function (x) { return x.key === m.key; })) { D.methods.push(Object.assign({}, m, { local: true })); } });
    Object.assign(D.cells, P.cells || {}); Object.assign(D.status, P.status || {}); Object.assign(D.timing, P.timing || {});
    SOURCES.push({ base: P.base, local: true });
  }
  var Z = 1.959964, LN2 = Math.log(2);
  var CATS = D.catalogs.map(function (c) { return c.key; });
  var CAT = {}; D.catalogs.forEach(function (c) { CAT[c.key] = c; });
  var GROUPS = { physics: "physics laws", classical: "classical GP suites", synthetic: "synthetic corpora", other: "other" };
  var METRIC = {}; D.metrics.forEach(function (m) { METRIC[m.key] = m; });
  var MGROUPS = []; D.metrics.forEach(function (m) { if (MGROUPS.indexOf(m.group) < 0) { MGROUPS.push(m.group); } });
  var PAIRED_KEYS = D.paired_keys || [];
  var PROV = { upstream_default: "upstream defaults", author_blessed: "author-blessed", harness_tuned: "maintainer-chosen" };
  var PROV_NOTE = {
    upstream_default: "Configuration: the method's own upstream defaults; nothing was tuned in either direction.",
    author_blessed: "Configuration: author-blessed, chosen by the method's authors. For Flash-ANSR entries the benchmark and the method share authors; that is what this label discloses.",
    harness_tuned: "Configuration: maintainer-chosen, set by the benchmark maintainers."
  };
  var TERMS = {
    matched: "Matched pooling: at every rung only the catalogs that EVERY shown method has finished are pooled, so the methods are compared on the same laws. Own pooling: each method over whatever it has finished.",
    thin: "A rung is thin when its pool holds fewer than half the laws of the method's largest pool (units still running). Thin rungs are hidden unless you show them; they draw with hollow markers.",
    wilson: "95 % Wilson score interval for a rate; t-interval for a mean; order-statistic interval for a median (from the pooled histogram).",
    median: "The median is read from a 128-bin histogram per cell, so it is exact to a bin. Ratios and times are binned on a log scale.",
    mean: "The default. A mean is taken over the finite values of the pooled laws, so an exactly recovered law (log10 FVU = -inf) is counted by the recovery rates and by the median, but not by the mean; every point reports how many finite values it averaged and how many it had. Switch to the median where that matters.",
    regime: "Rate metrics are defined for every law: a failed prediction is a miss. Continuous metrics describe successful predictions only.",
    mcnemar: "Exact McNemar test on the laws the two methods disagree on (one recovered, the other did not): two-sided binomial p-value, no asymptotics. The difference of paired rates carries a 95 % Wald interval.",
    signtest: "Paired mean difference with a t-interval over laws where both methods have a finite value, plus a two-sided exact sign test on the wins and losses.",
    draw1: "One draw per problem so far. These are paired contrasts on the same laws, not the repeated-draw noise margins of the 2026-07 release; those follow when draws 2 and up exist.",
    time: "Seconds per problem, averaged over the pooled laws. Where the reference machine (solomon: RTX 4090, 16 refiner workers, a frozen 262-problem subset) has measured every shown method, that is the x position; until then it is the time measured where each unit ran, on the cluster's mixed GPUs. One chart never mixes the two, and its x label says which it is.",
    candidates: "Samples, beam width, candidates or draws per problem: the budget a generative method spends. PySR's budget is a time limit rather than a count, so it has no position on this axis and appears on the time axis only.",
    provenance: "Who chose each method's configuration. Upstream defaults: nothing tuned. Author-blessed: the method's authors chose it (Flash-ANSR shares authors with the benchmark). Maintainer-chosen: the benchmark maintainers set it."
  };
  var COOKIE = "srbf_colors";
  function readCookie() { var m = document.cookie.match(new RegExp("(?:^|; )" + COOKIE + "=([^;]*)")); if (!m) { return {}; } try { return JSON.parse(decodeURIComponent(m[1])) || {}; } catch (e) { return {}; } }
  function writeCookie(obj) { if (!obj || !Object.keys(obj).length) { document.cookie = COOKIE + "=;path=/;max-age=0;SameSite=Lax"; return; } document.cookie = COOKIE + "=" + encodeURIComponent(JSON.stringify(obj)) + ";path=/;max-age=" + (60 * 60 * 24 * 365) + ";SameSite=Lax"; }
  var userColors = readCookie();
  function colorOf(m) { return userColors[m.key] || m.color; }
  function esc(s) { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/"/g, "&quot;"); }
  function term(key, text) { return '<span class="v2term" data-term="' + key + '" role="button" tabindex="0">' + text + "</span>"; }
  function help(text, label) { return '<button type="button" class="v2help" data-help="' + esc(text) + '" aria-label="' + esc(label || "What does this mean?") + '">?</button>'; }

  // ---- state: URL > localStorage > defaults --------------------------------------------------------------------
  var withData = function (m) { return D.cells[m.key] && Object.keys(D.cells[m.key]).length; };
  var DEFAULTS = function () {
    return { view: "curves", cats: CATS.slice(), methods: D.methods.filter(withData).map(function (m) { return m.key; }),
      plots: D.metrics.filter(function (m) { return m.tier === "main"; }).map(function (m) { return m.key; }),
      focus: "numeric_recovery_val", stat: "mean", pool: "matched", ci: true, thin: false, xaxis: "time", rung: 64, base: null, tier: "main", q: "", rows: "rungs" };
  };
  var state = DEFAULTS();
  var LS = "srbf-v2-" + REL + ".3";   // bumped whenever a default changes (.2 time axis, .3 mean), so a saved state cannot pin the old one
  function loadState() {
    try { var s = JSON.parse(localStorage.getItem(LS) || "null"); if (s) { Object.keys(state).forEach(function (k) { if (s[k] !== undefined) { state[k] = s[k]; } }); } } catch (e) { /* no storage */ }
    var q = new URLSearchParams(window.location.search); var any = false;
    var preset = { all: CATS, phys: CATS.filter(function (c) { return CAT[c].group === "physics"; }), classic: CATS.filter(function (c) { return CAT[c].group === "classical"; }), synth: CATS.filter(function (c) { return CAT[c].group === "synthetic"; }), none: [] };
    if (q.has("v")) { state.view = q.get("v"); any = true; }
    if (q.has("c")) { var c = q.get("c"); state.cats = preset[c] ? preset[c].slice() : c.split(",").filter(function (x) { return CAT[x]; }); any = true; }
    if (q.has("m")) { state.methods = q.get("m").split(",").filter(function (x) { return D.methods.some(function (mm) { return mm.key === x; }); }); any = true; }
    if (q.has("p")) { state.plots = q.get("p").split(",").filter(function (x) { return METRIC[x]; }); any = true; }
    if (q.has("f") && METRIC[q.get("f")]) { state.focus = q.get("f"); any = true; }
    if (q.has("s")) { state.stat = q.get("s") === "median" ? "median" : "mean"; any = true; }
    if (q.has("pool")) { state.pool = q.get("pool") === "own" ? "own" : "matched"; any = true; }
    if (q.has("ci")) { state.ci = q.get("ci") !== "0"; any = true; }
    if (q.has("thin")) { state.thin = q.get("thin") === "1"; any = true; }
    if (q.has("x")) { state.xaxis = q.get("x") === "time" ? "time" : "rung"; any = true; }
    if (q.has("r")) { var r = parseInt(q.get("r"), 10); if (D.rungs.indexOf(r) >= 0) { state.rung = r; } any = true; }
    if (q.has("b")) { state.base = q.get("b"); any = true; }
    if (q.has("rows")) { state.rows = q.get("rows") === "cats" ? "cats" : "rungs"; any = true; }
    if (q.has("tier")) { state.tier = q.get("tier") === "all" ? "all" : "main"; }
    if (["curves", "table", "matrix", "dist", "paired"].indexOf(state.view) < 0) { state.view = "curves"; }
    if (!Array.isArray(state.cats)) { state.cats = CATS.slice(); }
    if (!Array.isArray(state.methods)) { state.methods = DEFAULTS().methods; }
    if (!Array.isArray(state.plots)) { state.plots = DEFAULTS().plots; }
    state.cats = state.cats.filter(function (x) { return CAT[x]; }); state.plots = state.plots.filter(function (x) { return METRIC[x]; });
    state.methods = state.methods.filter(function (x) { return D.methods.some(function (mm) { return mm.key === x; }); });
    if (!METRIC[state.focus]) { state.focus = "numeric_recovery_val"; }
    if (!state.base || state.methods.indexOf(state.base) < 0) { state.base = state.methods[0] || null; }
    if (D.rungs.indexOf(state.rung) < 0) { state.rung = 64; }
    return any;
  }
  var fromUrl = loadState();
  function catsParam() {
    var set = {}; state.cats.forEach(function (c) { set[c] = 1; });
    var is = function (g) { var gs = CATS.filter(function (c) { return CAT[c].group === g; }); return gs.length === state.cats.length && gs.every(function (c) { return set[c]; }); };
    if (state.cats.length === CATS.length) { return "all"; } if (!state.cats.length) { return "none"; }
    if (is("physics")) { return "phys"; } if (is("classical")) { return "classic"; } if (is("synthetic")) { return "synth"; }
    return state.cats.join(",");
  }
  function save() {
    try { localStorage.setItem(LS, JSON.stringify(state)); } catch (e) { /* no storage */ }
    var q = new URLSearchParams(window.location.search);
    ["view", "bench", "baseline", "metric", "budget"].forEach(function (k) { q.delete(k); });   // never carry 2026-07 keys
    q.set("release", REL); q.set("v", state.view); q.set("c", catsParam()); q.set("m", state.methods.join(",")); q.set("p", state.plots.join(","));
    q.set("f", state.focus); q.set("s", state.stat); q.set("pool", state.pool); q.set("ci", state.ci ? "1" : "0"); q.set("thin", state.thin ? "1" : "0");
    q.set("x", state.xaxis); q.set("r", String(state.rung)); if (state.base) { q.set("b", state.base); } q.set("rows", state.rows);
    try { window.history.replaceState(null, "", "?" + q.toString() + window.location.hash); } catch (e) { /* file:// */ }
  }

  // ---- lazy payloads: histograms per metric, paired contrasts ----------------------------------------------------
  var loading = {}, failed = {};
  function ensure(file, cb) {
    SOURCES.forEach(function (s) {
      var key = s.base + file; if (loading[key] || failed[key]) { return; }
      loading[key] = "pending";
      var sc = document.createElement("script"); sc.src = s.base + file; sc.async = true;
      sc.onload = function () { loading[key] = "done"; cb(); };
      sc.onerror = function () { failed[key] = true; delete loading[key]; cb(); };
      document.head.appendChild(sc);
    });
  }
  function ready(file) { return SOURCES.every(function (s) { return loading[s.base + file] === "done" || failed[s.base + file]; }); }
  function histOf(k) { var H = window.RESULTS_V2_HIST && window.RESULTS_V2_HIST[REL]; return H && H[k]; }
  function pairedOf() { var Pd = window.RESULTS_V2_PAIRED && window.RESULTS_V2_PAIRED[REL]; return Pd || null; }

  // ---- pooling ---------------------------------------------------------------------------------------------------
  function cell(m, c, r) { var x = D.cells[m] && D.cells[m][c] && D.cells[m][c][String(r)]; return x && x.state === "complete" ? x : null; }
  function hasRung(m, r) { return CATS.some(function (c) { return cell(m, c, r); }); }
  function shownMethods() { return D.methods.filter(function (m) { return state.methods.indexOf(m.key) >= 0 && withData(m); }); }
  function poolCats(m, r, shown) {
    if (!hasRung(m, r)) { return []; }
    var sel = state.cats.filter(function (c) { return cell(m, c, r); });
    if (state.pool !== "matched") { return sel; }
    var present = shown.filter(function (k) { return hasRung(k, r); });
    return sel.filter(function (c) { return present.every(function (k) { return cell(k, c, r); }); });
  }
  function laws(cs) { return cs.reduce(function (a, c) { return a + CAT[c].laws; }, 0); }
  function thinMap(m, shown) { var pools = {}, mx = 0; D.rungs.forEach(function (r) { var n = laws(poolCats(m, r, shown)); pools[r] = n; if (n > mx) { mx = n; } }); var thin = {}; D.rungs.forEach(function (r) { thin[r] = pools[r] > 0 && pools[r] < 0.5 * mx; }); return { pools: pools, thin: thin }; }
  function wilson(a, b) { if (!b) { return null; } var p = a / b, z2 = Z * Z; var ctr = (p + z2 / (2 * b)) / (1 + z2 / b), half = Z * Math.sqrt(p * (1 - p) / b + z2 / (4 * b * b)) / (1 + z2 / b); return { v: p, lo: ctr - half, hi: ctr + half, n: b }; }
  function binVal(h, i) { return h.lo + (i + 0.5) * (h.hi - h.lo) / h.nb; }
  function addHist(acc, hc, nb) { if (hc.length && Array.isArray(hc[0])) { hc.forEach(function (p) { acc[p[0]] += p[1]; }); } else { for (var i = 0; i < nb; i++) { acc[i] += hc[i] || 0; } } }
  function pooledHist(k, m, r, cs) { var H = histOf(k); if (!H || !H.cells[m]) { return null; } var acc = new Array(H.nb).fill(0); cs.forEach(function (c) { var hc = H.cells[m][c] && H.cells[m][c][String(r)]; if (hc) { addHist(acc, hc, H.nb); } }); var n = acc.reduce(function (a, b) { return a + b; }, 0); return n ? { h: acc, n: n, lo: H.lo, hi: H.hi, nb: H.nb } : null; }
  function quantileBin(h, kth) { var cum = 0; for (var i = 0; i < h.nb; i++) { cum += h.h[i]; if (cum >= kth) { return i; } } return h.nb - 1; }
  function tfOf(metric) { return metric.hist && metric.hist.tf; }
  function fwd(metric, x) { var tf = tfOf(metric); return tf === "log2" ? Math.log2(Math.max(1e-300, x)) : tf === "log10" ? Math.log10(Math.max(1e-300, x)) : x; }
  function back(metric, x) { var tf = tfOf(metric); return tf === "log2" ? Math.pow(2, x) : tf === "log10" ? Math.pow(10, x) : x; }
  // every statistic is returned in the metric's TRANSFORMED space (log2 for ratios, log10 for seconds), where the charts live
  function stat(metric, m, r, cs) {
    var cells = cs.map(function (c) { return cell(m, c, r); }).filter(Boolean); if (!cells.length) { return null; }
    if (metric.kind === "rate") { var a = 0, b = 0; cells.forEach(function (c) { var t = c.m[metric.key]; if (t) { a += t[0]; b += t[1]; } }); return wilson(a, b); }
    if (state.stat === "mean") {
      var nd = 0, n = 0, s = 0, ss = 0; cells.forEach(function (c) { var t = c.m[metric.key]; if (t) { nd += t[0]; n += t[1]; s += t[2]; ss += t[3]; } });
      if (!n) { return null; } var mean = s / n, varr = Math.max(0, (ss - n * mean * mean) / Math.max(1, n - 1)), se = Math.sqrt(varr / n);
      return { v: fwd(metric, mean), lo: fwd(metric, mean - Z * se), hi: fwd(metric, mean + Z * se), n: n, ndef: nd };
    }
    if (!ready("hist/" + metric.key + ".js")) { return { pending: true }; }
    var ph = pooledHist(metric.key, m, r, cs); if (!ph) { return null; }
    var half = Z * Math.sqrt(ph.n) / 2, mid = quantileBin(ph, ph.n / 2);
    var lo = quantileBin(ph, Math.max(1, Math.floor(ph.n / 2 - half))), hi = quantileBin(ph, Math.min(ph.n, Math.ceil(ph.n / 2 + half)));
    return { v: binVal(ph, mid), lo: binVal(ph, lo), hi: binVal(ph, hi), n: ph.n, edge: mid === 0 ? -1 : (mid === ph.nb - 1 ? 1 : 0) };
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
  function tickLabel(metric, x) { var tf = tfOf(metric); if (metric.kind === "rate") { return (100 * x).toFixed(0) + "%"; } if (tf === "log2") { var v = Math.pow(2, x); return v >= 1 ? String(v) : "1/" + String(Math.pow(2, -x)); } if (tf === "log10") { return Math.pow(10, x) + " s"; } return Math.abs(x) >= 100 ? x.toFixed(0) : x.toFixed(metric.fmt === "num3" ? 2 : 1); }
  function ticksFor(metric, ymin, ymax) {
    var tf = tfOf(metric), out = [];
    if (tf === "log2" || tf === "log10") { for (var i = Math.ceil(ymin); i <= Math.floor(ymax); i++) { out.push(i); } if (out.length > 8) { out = out.filter(function (v) { return v % 2 === 0; }); } return out; }
    if (metric.kind === "rate") { return [0, 1, 2, 3, 4].map(function (i) { return ymin + (ymax - ymin) * i / 4; }); }
    var span = ymax - ymin, step = Math.pow(10, Math.floor(Math.log10(span / 4))); [1, 2, 5].some(function (f) { if (span / (step * f) <= 6) { step = step * f; return true; } return false; });
    for (var t = Math.ceil(ymin / step) * step; t <= ymax + 1e-9; t += step) { out.push(t); } return out;
  }

  // ---- SVG chart: series of points {x, v, lo, hi, thin, title} on a rung/time x axis -----------------------------
  function narrow() { return window.innerWidth < 700; }   // the viewport, like the CSS breakpoints: a container reflows, this does not
  var CHART_MIN = 520, CHART_GAP = 14;   // must match the grid in styles.css (.v2charts, .v2hlcharts)
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
  var chartHost = null, chartCount = 0;   // set while a block renders: its charts are built for THAT container
  function hostWidth() { return chartWidth(chartHost || root.querySelector(".v2main") || root, chartCount); }
  function inBlock(host, count, fn) { chartHost = host; chartCount = count; try { return fn(); } finally { chartHost = null; chartCount = 0; } }
  function chartSVG(opts) {
    var nr = narrow(), W = opts.width || hostWidth(), L = 62, T = 34, R = nr ? 16 : 180, series = opts.series;
    var B = nr ? 56 + 20 * Math.max(1, series.length) : 52, H = plotHeight(W) + B;
    var s = '<svg viewBox="0 0 ' + W + ' ' + H + '" class="v2chart" role="img" aria-label="' + esc(opts.title) + '"><text x="' + L + '" y="18" class="ct">' + esc(opts.title) + '</text>';
    if (opts.empty) { return s + '<text x="' + W / 2 + '" y="' + H / 2 + '" class="tick" text-anchor="middle">' + esc(opts.empty) + '</text></svg>'; }
    var ymin = opts.ymin, ymax = opts.ymax; if (!(ymax > ymin)) { ymax = ymin + 1; }
    var timeAxis = opts.timeAxis, tmin = opts.tmin, tmax = opts.tmax;
    var xs = function (x) { return timeAxis ? L + (Math.log10(x) - Math.log10(tmin)) / (Math.log10(tmax) - Math.log10(tmin)) * (W - L - R) : L + Math.log2(x) / Math.log2(65536) * (W - L - R); };
    var y = function (v) { return T + (1 - (v - ymin) / (ymax - ymin)) * (H - T - B); };
    opts.ticks.forEach(function (g) { s += '<line x1="' + L + '" y1="' + y(g).toFixed(1) + '" x2="' + (W - R) + '" y2="' + y(g).toFixed(1) + '" class="grid"/><text x="' + (L - 6) + '" y="' + (y(g) + 4).toFixed(1) + '" class="tick" text-anchor="end">' + esc(opts.tick(g)) + '</text>'; });
    if (opts.zero !== undefined && opts.zero >= ymin && opts.zero <= ymax) { s += '<line x1="' + L + '" y1="' + y(opts.zero).toFixed(1) + '" x2="' + (W - R) + '" y2="' + y(opts.zero).toFixed(1) + '" class="grid zero" stroke-dasharray="4 4"/>'; }
    if (timeAxis) { for (var t = tmin; t <= tmax * 1.0001; t *= 10) { s += '<line x1="' + xs(t).toFixed(1) + '" y1="' + T + '" x2="' + xs(t).toFixed(1) + '" y2="' + (H - B) + '" class="grid"/><text x="' + xs(t).toFixed(1) + '" y="' + (H - B + 16) + '" class="tick" text-anchor="middle">' + (t >= 1 ? t : t.toPrecision(1)) + ' s</text>'; } }
    else { D.rungs.forEach(function (r) { var e = Math.round(Math.log2(r)); s += '<line x1="' + xs(r).toFixed(1) + '" y1="' + (H - B) + '" x2="' + xs(r).toFixed(1) + '" y2="' + (H - B + (e % 2 ? 3 : 5)) + '" class="grid"/>'; if (e % 2) { return; } s += '<text x="' + xs(r).toFixed(1) + '" y="' + (H - B + 16) + '" class="tick" text-anchor="middle">' + (r >= 1024 ? (r / 1024) + "k" : r) + '</text>'; }); }
    var xlab = timeAxis
      ? (opts.timeSource === "ref"
        ? (nr ? "fit time (s, ref)" : "fit time per problem (s, log, reference machine)")
        : (nr ? "fit time (s, as run)" : "fit time per problem (s, log, as run)"))
      : (nr ? "candidates / problem" : "candidates per problem (log scale)");
    s += '<text x="' + ((L + W - R) / 2).toFixed(0) + '" y="' + (H - B + 32) + '" class="tick" text-anchor="middle">' + esc(xlab) + '</text>';
    if (opts.ylabel) { s += '<text transform="translate(14,' + ((T + H - B) / 2).toFixed(0) + ') rotate(-90)" class="tick" text-anchor="middle">' + esc(opts.ylabel) + "</text>"; }
    var ly = nr ? H - B + 46 : T + 6, lx = nr ? L : W - R + 10;
    series.forEach(function (sr) { var col = sr.color, pts = sr.pts.slice().sort(function (a, b) { return a.x - b.x; }); var cl = function (v) { return Math.min(ymax, Math.max(ymin, v)); };
      if (state.ci && pts.some(function (p) { return isFinite(p.lo) && isFinite(p.hi); })) { var up = pts.map(function (p) { return xs(p.x).toFixed(1) + "," + y(cl(isFinite(p.hi) ? p.hi : p.v)).toFixed(1); }); var dn = pts.slice().reverse().map(function (p) { return xs(p.x).toFixed(1) + "," + y(cl(isFinite(p.lo) ? p.lo : p.v)).toFixed(1); }); s += '<polygon points="' + up.concat(dn).join(" ") + '" fill="' + col + '" fill-opacity="0.13" stroke="none"/>'; }
      s += '<polyline fill="none" stroke="' + col + '" stroke-width="2" points="' + pts.map(function (p) { return xs(p.x).toFixed(1) + "," + y(cl(p.v)).toFixed(1); }).join(" ") + '"/>';
      pts.forEach(function (p) { s += '<circle cx="' + xs(p.x).toFixed(1) + '" cy="' + y(cl(p.v)).toFixed(1) + '" r="3.2" fill="' + (p.thin ? "var(--surface)" : col) + '" stroke="' + col + '" stroke-width="1.5"><title>' + esc(p.title) + '</title></circle>'; });
      s += '<line x1="' + lx + '" y1="' + ly + '" x2="' + (lx + 20) + '" y2="' + ly + '" stroke="' + col + '" stroke-width="3"/><text x="' + (lx + 26) + '" y="' + (ly + 4) + '" class="leg">' + esc(sr.label) + '</text>'; ly += 20; });
    return s + "</svg>";
  }
  // x positions. The budget axis is the method's own ladder. The time axis is seconds per problem: the
  // reference machine where it has measured every shown method, the runs themselves otherwise -- one
  // chart uses one source for all of its methods, never a mixture.
  function refTime(m, r) { var t = (D.timing[m] || {})[String(r)]; return t > 0 ? t : null; }
  function measuredTime(m, r, cs) { var n = 0, s = 0; cs.forEach(function (c) { var x = cell(m, c, r), t = x && x.m.fit_time; if (t) { n += t[1]; s += t[2]; } }); return n ? s / n : null; }
  function timeSource(keys) { return keys.length && keys.every(function (k) { return D.timing[k] && Object.keys(D.timing[k]).length; }) ? "ref" : "run"; }
  function timeOf(m, r, cs, src) { return src === "ref" ? refTime(m, r) : measuredTime(m, r, cs); }

  // A chart whose x is a metric, not a budget: each method's ladder walks a path through the plane
  // (here: how long its answer is against how well it fits), so the points keep their rung order.
  function frontSVG(opts) {
    var nr = narrow(), W = opts.width || hostWidth(), L = 62, T = 34, R = nr ? 16 : 180;
    var B = nr ? 56 + 20 * Math.max(1, opts.series.length) : 52, H = plotHeight(W) + B;
    var s = '<svg viewBox="0 0 ' + W + ' ' + H + '" class="v2chart" role="img" aria-label="' + esc(opts.title) + '"><text x="' + L + '" y="18" class="ct">' + esc(opts.title) + "</text>";
    if (opts.empty) { return s + '<text x="' + W / 2 + '" y="' + H / 2 + '" class="tick" text-anchor="middle">' + esc(opts.empty) + "</text></svg>"; }
    var xmin = opts.xmin, xmax = opts.xmax, ymin = opts.ymin, ymax = opts.ymax;
    if (!(xmax > xmin)) { xmax = xmin + 1; } if (!(ymax > ymin)) { ymax = ymin + 1; }
    var xs = function (x) { return L + (x - xmin) / (xmax - xmin) * (W - L - R); };
    var y = function (v) { return T + (1 - (v - ymin) / (ymax - ymin)) * (H - T - B); };
    var clx = function (v) { return Math.min(xmax, Math.max(xmin, v)); }, cly = function (v) { return Math.min(ymax, Math.max(ymin, v)); };
    opts.yticks.forEach(function (g) { s += '<line x1="' + L + '" y1="' + y(g).toFixed(1) + '" x2="' + (W - R) + '" y2="' + y(g).toFixed(1) + '" class="grid"/><text x="' + (L - 6) + '" y="' + (y(g) + 4).toFixed(1) + '" class="tick" text-anchor="end">' + esc(opts.ytick(g)) + "</text>"; });
    opts.xticks.forEach(function (g) { s += '<line x1="' + xs(g).toFixed(1) + '" y1="' + T + '" x2="' + xs(g).toFixed(1) + '" y2="' + (H - B) + '" class="grid"/><text x="' + xs(g).toFixed(1) + '" y="' + (H - B + 16) + '" class="tick" text-anchor="middle">' + esc(opts.xtick(g)) + "</text>"; });
    if (opts.xzero !== undefined && opts.xzero > xmin && opts.xzero < xmax) { s += '<line x1="' + xs(opts.xzero).toFixed(1) + '" y1="' + T + '" x2="' + xs(opts.xzero).toFixed(1) + '" y2="' + (H - B) + '" class="grid zero" stroke-dasharray="4 4"/>'; }
    s += '<text x="' + ((L + W - R) / 2).toFixed(0) + '" y="' + (H - B + 32) + '" class="tick" text-anchor="middle">' + esc(opts.xlabel) + "</text>";
    s += '<text transform="translate(14,' + ((T + H - B) / 2).toFixed(0) + ') rotate(-90)" class="tick" text-anchor="middle">' + esc(opts.ylabel) + "</text>";
    var ly = nr ? H - B + 46 : T + 6, lx = nr ? L : W - R + 10;
    opts.series.forEach(function (sr) {
      var col = sr.color;
      if (state.ci) { sr.pts.forEach(function (p) { if (isFinite(p.lo) && isFinite(p.hi)) { s += '<line x1="' + xs(clx(p.x)).toFixed(1) + '" y1="' + y(cly(p.hi)).toFixed(1) + '" x2="' + xs(clx(p.x)).toFixed(1) + '" y2="' + y(cly(p.lo)).toFixed(1) + '" stroke="' + col + '" stroke-width="1.5" stroke-opacity="0.4"/>'; } }); }
      s += '<polyline fill="none" stroke="' + col + '" stroke-width="1.6" stroke-opacity="0.65" points="' + sr.pts.map(function (p) { return xs(clx(p.x)).toFixed(1) + "," + y(cly(p.v)).toFixed(1); }).join(" ") + '"/>';
      sr.pts.forEach(function (p) { s += '<circle cx="' + xs(clx(p.x)).toFixed(1) + '" cy="' + y(cly(p.v)).toFixed(1) + '" r="3.2" fill="' + (p.thin ? "var(--surface)" : col) + '" stroke="' + col + '" stroke-width="1.5"><title>' + esc(p.title) + "</title></circle>"; });
      s += '<line x1="' + lx + '" y1="' + ly + '" x2="' + (lx + 20) + '" y2="' + ly + '" stroke="' + col + '" stroke-width="3"/><text x="' + (lx + 26) + '" y="' + (ly + 4) + '" class="leg">' + esc(sr.label) + "</text>"; ly += 20;
    });
    return s + "</svg>";
  }
  function frontChart(xm, ym, shown, title) {
    var keys = shown.map(function (m) { return m.key; }), series = [], pending = false;
    var xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
    shown.forEach(function (m) {
      var tm = thinMap(m.key, keys), pts = [];
      D.rungs.forEach(function (r) {
        var use = poolCats(m.key, r, keys); if (!use.length || (tm.thin[r] && !state.thin)) { return; }
        var sx = stat(xm, m.key, r, use), sy = stat(ym, m.key, r, use); if (!sx || !sy) { return; }
        if (sx.pending || sy.pending) { pending = true; return; }
        if (!isFinite(sx.v) || !isFinite(sy.v)) { return; }
        pts.push({ x: sx.v, v: sy.v, lo: sy.lo, hi: sy.hi, thin: tm.thin[r],
          title: m.label + " @ " + r + ": " + fmt(xm, sx.v, sx.edge) + " " + xm.short + ", " + fmt(ym, sy.v, sy.edge) + " " + ym.short + ", n = " + sy.n });
        xmin = Math.min(xmin, sx.v); xmax = Math.max(xmax, sx.v);
        [sy.v, state.ci ? sy.lo : sy.v, state.ci ? sy.hi : sy.v].forEach(function (v) { if (isFinite(v)) { ymin = Math.min(ymin, v); ymax = Math.max(ymax, v); } });
      });
      if (pts.length) { series.push({ label: m.label + (m.local ? " (local)" : ""), color: colorOf(m), pts: pts }); }
    });
    title = title || (xm.short + " vs " + ym.short);
    if (pending && !series.length) { return frontSVG({ title: title, series: [], empty: "loading the distributions\u2026" }); }
    if (!series.length) { return frontSVG({ title: title, series: [], empty: "no finished units for this selection yet" }); }
    var xpad = (xmax - xmin) * 0.08 || 0.3, ypad = (ymax - ymin) * 0.08 || 0.3;
    xmin = Math.min(xmin - xpad, -0.15); xmax += xpad; ymin -= ypad; ymax += ypad;   // keep the law's own length in view
    return frontSVG({ title: title, series: series, xmin: xmin, xmax: xmax, ymin: ymin, ymax: ymax,
      xticks: ticksFor(xm, xmin, xmax), xtick: function (g) { return tickLabel(xm, g); },
      yticks: ticksFor(ym, ymin, ymax), ytick: function (g) { return tickLabel(ym, g); },
      xlabel: narrow() ? "length / law (log)" : "answer length / law length (log scale)",
      ylabel: narrow() ? ym.short : ym.label, xzero: 0 });
  }
  function xOf(m, r, cs, src) { return state.xaxis === "time" ? timeOf(m, r, cs, src) : r; }
  function hasCandidateBudget(m) { return (m.budget || "candidates") !== "seconds"; }   // PySR's budget is seconds
  function axisMethods(shown) { return state.xaxis === "time" ? shown : shown.filter(hasCandidateBudget); }
  function anyTime() {
    if (Object.keys(D.timing).some(function (k) { return Object.keys(D.timing[k]).length; })) { return true; }
    return D.methods.some(function (m) { var cs = D.cells[m.key] || {};
      return Object.keys(cs).some(function (c) { return Object.keys(cs[c]).some(function (r) { return cs[c][r].m && cs[c][r].m.fit_time; }); }); });
  }
  function timeRange(tmin, tmax) { tmin = Math.pow(10, Math.floor(Math.log10(tmin))); tmax = Math.pow(10, Math.ceil(Math.log10(tmax))); if (tmax <= tmin) { tmax = tmin * 10; } return [tmin, tmax]; }

  // ---- Curves ----------------------------------------------------------------------------------------------------
  function curveChart(metric, shown, title) {
    var keys = shown.map(function (x) { return x.key; }), series = [], ymin = Infinity, ymax = -Infinity, pending = false, tmin = Infinity, tmax = -Infinity;
    var src = timeSource(keys);
    shown.forEach(function (m) { var tm = thinMap(m.key, keys), pts = [];
      D.rungs.forEach(function (r) { var use = poolCats(m.key, r, keys); if (!use.length || (tm.thin[r] && !state.thin)) { return; } var x = xOf(m.key, r, use, src); if (x === null) { return; }
        var st = stat(metric, m.key, r, use); if (!st) { return; } if (st.pending) { pending = true; return; } if (!isFinite(st.v)) { return; }
        var title = m.label + " @ " + r + (state.xaxis === "time" ? " (" + x.toFixed(2) + " s)" : "") + ": " + fmt(metric, st.v, st.edge) + " [" + fmt(metric, st.lo) + ", " + fmt(metric, st.hi) + "], n = " + st.n + (st.ndef ? " finite of " + st.ndef : "") + ", pool " + tm.pools[r] + " laws" + (tm.thin[r] ? " (thin)" : "");
        pts.push({ x: x, v: st.v, lo: st.lo, hi: st.hi, thin: tm.thin[r], title: title });
        [st.v, state.ci ? st.lo : st.v, state.ci ? st.hi : st.v].forEach(function (v) { if (isFinite(v)) { ymin = Math.min(ymin, v); ymax = Math.max(ymax, v); } });
        if (state.xaxis === "time") { tmin = Math.min(tmin, x); tmax = Math.max(tmax, x); } });
      if (pts.length) { series.push({ label: m.label + (m.local ? " (local)" : ""), color: colorOf(m), pts: pts }); } });
    title = title || metric.label;   // the statistic is named once per block, not on every chart
    if (pending && !series.length) { return chartSVG({ title: title, series: [], empty: "loading the distribution…" }); }
    if (!series.length) { return chartSVG({ title: title, series: [], empty: state.cats.length ? (shown.length ? (state.xaxis === "time" ? "no time measurement on the reference machine yet for the shown methods" : "no finished units for this selection yet") : "no method selected") : "no catalog selected" }); }
    if (metric.kind === "rate") { ymin = 0; ymax = Math.min(1, Math.max(0.05, ymax * 1.05)); }
    else if (tfOf(metric) === "log2") { ymin = Math.min(ymin, -0.3); ymax = Math.max(ymax, 0.3); }
    else { var pad = (ymax - ymin) * 0.08 || 0.1; ymin -= pad; ymax += pad; }
    var tr = state.xaxis === "time" ? timeRange(tmin, tmax) : [0, 0];
    return chartSVG({ title: title, series: series, ymin: ymin, ymax: ymax, ticks: ticksFor(metric, ymin, ymax), tick: function (g) { return tickLabel(metric, g); }, ylabel: "", timeAxis: state.xaxis === "time", timeSource: src, tmin: tr[0], tmax: tr[1], zero: tfOf(metric) === "log2" ? 0 : undefined });
  }
  function renderCurves(shown) {
    var plots = D.metrics.filter(function (m) { return state.plots.indexOf(m.key) >= 0; });
    if (!plots.length) { return '<p class="v2hint">No metric selected: tick one in the Metrics panel.</p>'; }
    var drawn = axisMethods(shown), dropped = shown.filter(function (m) { return drawn.indexOf(m) < 0; });
    var main = root.querySelector(".v2main");
    var note = dropped.length ? '<p class="v2hint">' + esc(dropped.map(function (m) { return m.label; }).join(", ")) + ' spends a time budget, not ' + term("candidates", "a candidate count") + ': switch the x axis to time to see ' + (dropped.length > 1 ? "them" : "it") + '.</p>' : "";
    return note + '<div class="v2charts">' + inBlock(main, plots.length, function () { return plots.map(function (m) { return curveChart(m, drawn); }).join(""); }) + "</div>";
  }

  // ---- Headline ---------------------------------------------------------------------------------------------------
  // Two charts that are the same for every visitor: what a method recovers, and how long its answer is, against
  // what it costs. They are deliberately not wired to the controls below -- everything adjustable is the explorer.
  var HEADLINE = [
    { key: "numeric_recovery_val", title: "Recovery vs time",
      caption: "Laws reproduced to float32 precision on held-out points. Up and left is better." },
    { x: "mdl_ratio", y: "log10_fvu_val", title: "Fit vs length",
      caption: "Description length in the certified canon; dashed line: the law itself. Down and left is better." }];
  function withState(over, fn) { var prev = state; state = Object.assign({}, prev, over); try { return fn(); } finally { state = prev; } }
  function renderHeadline() {
    if (!headRoot) { return; }
    headRoot.innerHTML = inBlock(headRoot, HEADLINE.length, function () { return withState(
      { cats: CATS.slice(), methods: D.methods.filter(withData).map(function (m) { return m.key; }),
        pool: "matched", stat: "mean", ci: true, thin: false, xaxis: "time" },
      function () {
        var shown = shownMethods();
        if (!shown.length) { return '<p class="v2hint">No method has finished units in this release yet.</p>'; }
        var keys = shown.map(function (m) { return m.key; }), src = timeSource(keys);
        var charts = HEADLINE.map(function (h) {
          var ms = (h.key ? [h.key] : [h.x, h.y]).map(function (k) { return METRIC[k]; });
          if (ms.some(function (m) { return !m; })) { return ""; }
          if (state.stat === "median") { ms.forEach(function (m) { if (m.kind === "cont") { ensure("hist/" + m.key + ".js", scheduleRender); } }); }
          var svg = h.key ? curveChart(ms[0], shown, h.title) : frontChart(ms[0], ms[1], shown, h.title);
          return '<figure class="v2hlfig">' + svg + "<figcaption>" + esc(h.caption) + "</figcaption></figure>";
        }).join("");
        return '<h2 class="v2hltitle">Recovery, cost and length</h2>' +
          '<p class="v2hlsub">One point per budget, ' + (state.stat === "mean" ? "mean" : "median") + ' over all ' + CATS.length + ' catalogs (' +
          term("matched", "matched") + '), with ' + term("wilson", "95 % intervals") + '.</p>' +
          '<div class="v2hlcharts">' + charts + "</div>" +
          '<p class="v2hint">' + (src === "ref" ? term("time", "Fit time on the reference machine")
            : term("time", "Fit time as each unit ran") + ", on mixed cluster GPUs until the reference machine has timed every method shown") + ".</p>";
      }); });
  }

  // ---- Table -----------------------------------------------------------------------------------------------------
  var lastTable = null;
  function cellText(st, p) { if (!st) { return ""; } if (st.pending) { return "…"; } return fmt(p, st.v, st.edge) + (state.ci ? ' <span class="v2ci-txt">[' + fmt(p, st.lo) + ", " + fmt(p, st.hi) + "]</span>" : ""); }
  function renderTable(shown) {
    var plots = D.metrics.filter(function (m) { return state.plots.indexOf(m.key) >= 0; }); var keys = shown.map(function (m) { return m.key; });
    if (!plots.length || !shown.length) { return '<p class="v2hint">Select at least one method and one metric.</p>'; }
    var head = '<tr><th>' + (state.rows === "cats" ? "catalog" : "rung") + '</th><th>laws</th>' + shown.map(function (m) { return '<th colspan="' + plots.length + '"><span class="v2sw" style="background:' + colorOf(m) + '"></span>' + esc(m.label) + '</th>'; }).join("") + '</tr>' +
      '<tr><th></th><th></th>' + shown.map(function () { return plots.map(function (p) { return '<th>' + esc(p.short) + '</th>'; }).join(""); }).join("") + '</tr>';
    var body = "", rowsOut = [];
    var emit = function (label, nl, tds) { body += '<tr><td>' + esc(label) + '</td><td>' + esc(nl) + '</td>' + tds.map(function (x) { return '<td>' + x.t + '</td>'; }).join("") + '</tr>'; rowsOut.push([label, nl].concat(tds.map(function (x) { return x.raw; }))); };
    if (state.rows === "rungs") {
      var tms = {}; shown.forEach(function (m) { tms[m.key] = thinMap(m.key, keys); });
      D.rungs.forEach(function (r) { var cells = shown.map(function (m) { return { m: m, use: poolCats(m.key, r, keys), thin: tms[m.key].thin[r] }; }); if (!cells.some(function (c) { return c.use.length && (state.thin || !c.thin); })) { return; }
        var ref = cells.filter(function (c) { return c.use.length; })[0]; var nl = state.pool === "matched" ? laws(ref.use) + " (" + ref.use.length + " catalogs)" : "own";
        var tds = []; cells.forEach(function (c) { plots.forEach(function (p) { if (!c.use.length || (c.thin && !state.thin)) { tds.push({ t: "", raw: "" }); return; } var st = stat(p, c.m.key, r, c.use); tds.push({ t: cellText(st, p), raw: st && !st.pending ? fmt(p, st.v, st.edge) : "" }); }); });
        emit(String(r), nl, tds); });
    } else {
      var r = state.rung;
      state.cats.slice().sort(function (a, b) { return CAT[b].laws - CAT[a].laws; }).forEach(function (c) { if (!shown.some(function (m) { return cell(m.key, c, r); })) { return; }
        var tds = []; shown.forEach(function (m) { plots.forEach(function (p) { if (!cell(m.key, c, r)) { tds.push({ t: "", raw: "" }); return; } var st = stat(p, m.key, r, [c]); tds.push({ t: cellText(st, p), raw: st && !st.pending ? fmt(p, st.v, st.edge) : "" }); }); });
        emit(c, String(CAT[c].laws), tds); });
      var tds2 = []; shown.forEach(function (m) { plots.forEach(function (p) { var use = state.cats.filter(function (c) { return cell(m.key, c, r); }); var st = use.length ? stat(p, m.key, r, use) : null; tds2.push({ t: cellText(st, p), raw: st && !st.pending ? fmt(p, st.v, st.edge) : "" }); }); });
      body += '<tr class="v2total"><td>all selected</td><td>' + laws(state.cats) + '</td>' + tds2.map(function (x) { return '<td>' + x.t + '</td>'; }).join("") + '</tr>';
      rowsOut.push(["all selected", String(laws(state.cats))].concat(tds2.map(function (x) { return x.raw; })));
    }
    lastTable = { header: [state.rows === "cats" ? "catalog" : "rung", "laws"].concat(shown.reduce(function (a, m) { return a.concat(plots.map(function (p) { return m.label + " · " + p.short; })); }, [])), rows: rowsOut };
    var ctl = '<div class="v2row v2tablectl"><span class="v2lab">rows</span><label><input type="radio" name="v2rows" value="rungs"' + (state.rows === "rungs" ? " checked" : "") + '> every rung</label><label><input type="radio" name="v2rows" value="cats"' + (state.rows === "cats" ? " checked" : "") + '> catalogs at rung ' + state.rung + '</label>' +
      '<span class="v2spacer"></span><button type="button" class="v2btn" data-act="copy-tsv">copy as TSV</button><button type="button" class="v2btn" data-act="csv">download CSV</button></div>';
    return ctl + '<div class="v2table-wrap"><table class="v2table"><thead>' + head + '</thead><tbody>' + body + '</tbody></table></div><p class="v2hint">' + (state.ci ? "Brackets: 95 % interval (" + term("wilson", "Wilson / t / order statistic") + "). " : "") + term("regime", "Rates count every law; continuous metrics describe successful predictions") + ".</p>";
  }

  // ---- Catalog matrix --------------------------------------------------------------------------------------------
  function accentRGB() { var v = (getComputedStyle(document.documentElement).getPropertyValue("--accent") || "#4f46e5").trim(); var m = v.match(/^#([0-9a-f]{6})$/i); if (!m) { return [79, 70, 229]; } return [parseInt(m[1].slice(0, 2), 16), parseInt(m[1].slice(2, 4), 16), parseInt(m[1].slice(4, 6), 16)]; }
  function renderMatrix(shown) {
    var p = METRIC[state.focus], r = state.rung, rgb = accentRGB();
    if (!shown.length) { return '<p class="v2hint">Select at least one method.</p>'; }
    var cats = state.cats.slice().sort(function (a, b) { return CAT[b].laws - CAT[a].laws; }).filter(function (c) { return shown.some(function (m) { return cell(m.key, c, r); }); });
    if (!cats.length) { return '<p class="v2hint">No finished units at rung ' + r + ' for this selection.</p>'; }
    var vals = {}, all = [], pending = false;
    cats.forEach(function (c) { vals[c] = {}; shown.forEach(function (m) { if (!cell(m.key, c, r)) { return; } var st = stat(p, m.key, r, [c]); if (st && st.pending) { pending = true; return; } if (st && isFinite(st.v)) { vals[c][m.key] = st; all.push(st.v); } }); });
    if (pending && !all.length) { return '<p class="v2hint">Loading the distribution…</p>'; }
    var lo = Math.min.apply(null, all), hi = Math.max.apply(null, all), ideal = tfOf(p) === "log2" ? 0 : 1;
    var score = function (v) { if (!(hi > lo)) { return 0.5; } if (p.higher === null) { var dm = Math.max(Math.abs(lo - ideal), Math.abs(hi - ideal)); return dm ? 1 - Math.abs(v - ideal) / dm : 1; } var t = (v - lo) / (hi - lo); return p.higher ? t : 1 - t; };
    var h = '<div class="v2table-wrap"><table class="v2table v2matrix"><thead><tr><th>catalog</th><th>laws</th>' + shown.map(function (m) { return '<th><span class="v2sw" style="background:' + colorOf(m) + '"></span>' + esc(m.label) + '</th>'; }).join("") + '</tr></thead><tbody>';
    cats.forEach(function (c) { h += '<tr><td>' + esc(c) + ' <span class="v2hint">' + GROUPS[CAT[c].group] + '</span></td><td>' + CAT[c].laws + '</td>' + shown.map(function (m) { var st = vals[c][m.key]; if (!st) { return '<td class="v2na">' + (cell(m.key, c, r) ? "…" : "") + '</td>'; } var a = 0.06 + 0.5 * score(st.v); return '<td style="background:rgba(' + rgb.join(",") + "," + a.toFixed(2) + ')" title="' + esc(fmt(p, st.lo) + " to " + fmt(p, st.hi) + ", n = " + st.n) + '">' + fmt(p, st.v, st.edge) + '</td>'; }).join("") + '</tr>'; });
    var pooled = shown.map(function (m) { var use = cats.filter(function (c) { return cell(m.key, c, r); }); var st = use.length ? stat(p, m.key, r, use) : null; return '<td>' + (st && !st.pending ? cellText(st, p) : "") + '</td>'; }).join("");
    h += '<tr class="v2total"><td>all listed</td><td>' + laws(cats) + '</td>' + pooled + '</tr></tbody></table></div>';
    return '<p class="v2hint">' + esc(p.label) + " at rung " + r + ", one cell per catalog; darker = better" + (p.higher === null ? " (closer to 1)" : "") + ". " + (p.kind === "cont" ? (state.stat === "mean" ? term("mean", "Means") : term("median", "Medians")) + " over successful predictions." : term("regime", "Rates over every law") + ".") + "</p>" + h;
  }

  // ---- Distribution ----------------------------------------------------------------------------------------------
  function renderDist(shown) {
    var p = METRIC[state.focus], r = state.rung, keys = shown.map(function (m) { return m.key; });
    if (!shown.length) { return '<p class="v2hint">Select at least one method.</p>'; }
    var nr = narrow(), W = hostWidth(), L = 62, T = 34, R = 16;
    if (p.kind === "rate") {   // per-catalog rates: a dot plot with Wilson intervals
      var cats = state.cats.slice().sort(function (a, b) { return CAT[b].laws - CAT[a].laws; }).filter(function (c) { return shown.some(function (m) { return cell(m.key, c, r); }); });
      if (!cats.length) { return '<p class="v2hint">No finished units at rung ' + r + '.</p>'; }
      var rowH = 20, B1 = 44 + 18 * shown.length, Hh = T + cats.length * rowH + B1, Lw = nr ? 110 : 150;
      var s = '<svg viewBox="0 0 ' + W + ' ' + Hh + '" class="v2chart v2dist" role="img" aria-label="' + esc(p.label) + ' per catalog"><text x="' + Lw + '" y="18" class="ct">' + esc(p.label) + " per catalog at rung " + r + '</text>';
      var xs = function (v) { return Lw + v * (W - Lw - R); };
      [0, 0.25, 0.5, 0.75, 1].forEach(function (g) { s += '<line x1="' + xs(g) + '" y1="' + T + '" x2="' + xs(g) + '" y2="' + (Hh - B1) + '" class="grid"/><text x="' + xs(g) + '" y="' + (Hh - B1 + 16) + '" class="tick" text-anchor="middle">' + (100 * g) + '%</text>'; });
      cats.forEach(function (c, i) { var yy = T + (i + 0.5) * rowH; s += '<text x="' + (Lw - 8) + '" y="' + (yy + 4) + '" class="tick" text-anchor="end">' + esc(c) + ' · ' + CAT[c].laws + '</text>';
        shown.forEach(function (m, j) { var st = cell(m.key, c, r) ? stat(p, m.key, r, [c]) : null; if (!st) { return; } var yj = yy + (j - (shown.length - 1) / 2) * Math.min(6, rowH / (shown.length + 1)); var col = colorOf(m);
          if (state.ci) { s += '<line x1="' + xs(st.lo).toFixed(1) + '" y1="' + yj.toFixed(1) + '" x2="' + xs(st.hi).toFixed(1) + '" y2="' + yj.toFixed(1) + '" stroke="' + col + '" stroke-width="2" stroke-opacity="0.45"/>'; }
          s += '<circle cx="' + xs(st.v).toFixed(1) + '" cy="' + yj.toFixed(1) + '" r="3.5" fill="' + col + '"><title>' + esc(m.label + " on " + c + ": " + fmt(p, st.v) + " [" + fmt(p, st.lo) + ", " + fmt(p, st.hi) + "], n = " + st.n) + '</title></circle>'; }); });
      var ly = Hh - B1 + 34; shown.forEach(function (m) { s += '<circle cx="' + (Lw + 6) + '" cy="' + ly + '" r="4" fill="' + colorOf(m) + '"/><text x="' + (Lw + 16) + '" y="' + (ly + 4) + '" class="leg">' + esc(m.label + (m.local ? " (local)" : "")) + '</text>'; ly += 20; });
      return s + "</svg>" + '<p class="v2hint">Each dot is one catalog; bars are ' + term("wilson", "95 % Wilson intervals") + '. Catalogs sorted by size. Pick a continuous metric under Focus for a histogram.</p>';
    }
    if (!ready("hist/" + p.key + ".js")) { ensure("hist/" + p.key + ".js", scheduleRender); return '<p class="v2hint">Loading the distribution…</p>'; }
    var H0 = histOf(p.key); if (!H0) { return '<p class="v2hint">No distribution available for ' + esc(p.label) + '.</p>'; }
    var series = [], ymax = 0;
    shown.forEach(function (m) { var use = poolCats(m.key, r, keys); if (!use.length) { return; } var ph = pooledHist(p.key, m.key, r, use); if (!ph) { return; } var bw = (ph.hi - ph.lo) / ph.nb; var dens = ph.h.map(function (c) { return c / ph.n / bw; }); ymax = Math.max(ymax, Math.max.apply(null, dens)); series.push({ m: m, ph: ph, dens: dens, med: binVal(ph, quantileBin(ph, ph.n / 2)), laws: laws(use) }); });
    if (!series.length) { return '<p class="v2hint">No finished units at rung ' + r + ' for this selection.</p>'; }
    var B = 44 + 18 * series.length, H = 300 + B, lo = H0.lo, hi = H0.hi, xs2 = function (x) { return L + (x - lo) / (hi - lo) * (W - L - R); }, y2 = function (v) { return T + (1 - v / (ymax * 1.05)) * (H - T - B); };
    var s2 = '<svg viewBox="0 0 ' + W + ' ' + H + '" class="v2chart v2dist" role="img" aria-label="' + esc(p.label) + ' distribution"><text x="' + L + '" y="18" class="ct">' + esc(p.label) + " at rung " + r + '</text>';
    ticksFor(p, lo, hi).forEach(function (g) { s2 += '<line x1="' + xs2(g).toFixed(1) + '" y1="' + T + '" x2="' + xs2(g).toFixed(1) + '" y2="' + (H - B) + '" class="grid"/><text x="' + xs2(g).toFixed(1) + '" y="' + (H - B + 16) + '" class="tick" text-anchor="middle">' + esc(tickLabel(p, g)) + '</text>'; });
    s2 += '<text x="' + ((L + W - R) / 2) + '" y="' + (H - B + 32) + '" class="tick" text-anchor="middle">' + esc(p.short) + (tfOf(p) ? " (log scale)" : "") + '</text><text transform="translate(14,' + ((T + H - B) / 2) + ') rotate(-90)" class="tick" text-anchor="middle">density</text>';
    var ly2 = H - B + 46;
    series.forEach(function (sr) { var col = colorOf(sr.m), pts = []; for (var i = 0; i < sr.ph.nb; i++) { var x0 = lo + i * (hi - lo) / sr.ph.nb, x1 = x0 + (hi - lo) / sr.ph.nb; pts.push(xs2(x0).toFixed(1) + "," + y2(sr.dens[i]).toFixed(1)); pts.push(xs2(x1).toFixed(1) + "," + y2(sr.dens[i]).toFixed(1)); }
      s2 += '<polygon points="' + xs2(lo).toFixed(1) + "," + y2(0).toFixed(1) + " " + pts.join(" ") + " " + xs2(hi).toFixed(1) + "," + y2(0).toFixed(1) + '" fill="' + col + '" fill-opacity="0.12" stroke="none"/><polyline fill="none" stroke="' + col + '" stroke-width="1.8" points="' + pts.join(" ") + '"/>';
      s2 += '<line x1="' + xs2(sr.med).toFixed(1) + '" y1="' + T + '" x2="' + xs2(sr.med).toFixed(1) + '" y2="' + (H - B) + '" stroke="' + col + '" stroke-width="1.5" stroke-dasharray="5 4"><title>' + esc(sr.m.label + " median " + fmt(p, sr.med)) + '</title></line>';
      s2 += '<line x1="' + L + '" y1="' + ly2 + '" x2="' + (L + 20) + '" y2="' + ly2 + '" stroke="' + col + '" stroke-width="3"/><text x="' + (L + 26) + '" y="' + (ly2 + 4) + '" class="leg">' + esc(sr.m.label + (sr.m.local ? " (local)" : "") + " · median " + fmt(p, sr.med) + " · n = " + sr.ph.n + " over " + sr.laws + " laws") + '</text>'; ly2 += 18; });
    return s2 + "</svg>" + '<p class="v2hint">Histograms over the successful predictions of the pooled catalogs (128 bins; values outside the axis are pooled into the edge bins). Dashed lines: ' + term("median", "medians") + '.</p>';
  }

  // ---- Paired ----------------------------------------------------------------------------------------------------
  function lgamma(x) { var g = 7, c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]; if (x < 0.5) { return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x); } x -= 1; var a = c[0], t = x + g + 0.5; for (var i = 1; i < g + 2; i++) { a += c[i] / (x + i); } return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a); }
  function binomTwoSided(k, n) { if (!n) { return null; } var kk = Math.min(k, n - k), s = 0; for (var i = 0; i <= kk; i++) { s += Math.exp(lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1) - n * LN2); } return Math.min(1, 2 * s); }
  function fmtP(p) { if (p === null || p === undefined) { return "–"; } return p < 0.001 ? "< 0.001" : p.toFixed(3); }
  function pairedStat(metric, a, b, r, cs) {
    var Pd = pairedOf(); if (!Pd) { return null; } var key = a + "|" + b, flip = false; if (!Pd[key]) { key = b + "|" + a; flip = true; } if (!Pd[key]) { return null; }
    var x = [0, 0, 0, 0, 0], any = false;
    cs.forEach(function (c) { var pc = Pd[key][c] && Pd[key][c][String(r)]; if (!pc || !pc.m[metric.key]) { return; } any = true; var t = pc.m[metric.key]; for (var i = 0; i < t.length; i++) { x[i] += t[i]; } });
    if (!any) { return null; }
    if (metric.kind === "rate") { var n10 = flip ? x[2] : x[1], n01 = flip ? x[1] : x[2], N = x[0] + x[1] + x[2] + x[3]; if (!N) { return null; } var d = (n10 - n01) / N, se = Math.sqrt(Math.max(0, (n10 + n01) - (n10 - n01) * (n10 - n01) / N)) / N; return { v: d, lo: d - Z * se, hi: d + Z * se, n: N, p: binomTwoSided(n10, n10 + n01), wins: n10, losses: n01 }; }
    if (!x[0]) { return null; }
    var m = x[1] / x[0], varr = x[0] > 1 ? Math.max(0, (x[2] - x[0] * m * m) / (x[0] - 1)) : 0, se2 = Math.sqrt(varr / x[0]); if (flip) { m = -m; }
    var wins = flip ? x[4] : x[3], losses = flip ? x[3] : x[4];
    return { v: m, lo: m - Z * se2, hi: m + Z * se2, n: x[0], p: binomTwoSided(wins, wins + losses), wins: wins, losses: losses };
  }
  function fmtDelta(metric, d) { if (!isFinite(d)) { return "–"; } var tf = tfOf(metric); if (metric.kind === "rate") { return (d >= 0 ? "+" : "") + (100 * d).toFixed(1) + " pp"; } if (tf === "log2") { return "× " + Math.pow(2, d).toFixed(2); } if (tf === "log10") { return "× " + Math.pow(10, d).toFixed(2); } return (d >= 0 ? "+" : "") + d.toFixed(metric.fmt === "num3" ? 3 : 2); }
  function renderPaired(shown) {
    if (!ready("paired.js")) { ensure("paired.js", scheduleRender); return '<p class="v2hint">Loading the paired contrasts…</p>'; }
    if (shown.length < 2) { return '<p class="v2hint">Select at least two methods; one of them is the baseline.</p>'; }
    if (!state.base || !shown.some(function (m) { return m.key === state.base; })) { state.base = shown[0].key; }
    var base = D.methods.filter(function (m) { return m.key === state.base; })[0], others = shown.filter(function (m) { return m.key !== state.base; }), keys = shown.map(function (m) { return m.key; });
    var plots = D.metrics.filter(function (m) { return state.plots.indexOf(m.key) >= 0 && PAIRED_KEYS.indexOf(m.key) >= 0; });
    var ctl = '<p class="v2hint">Every method is read as method \u2212 ' + esc(base.label) + ' on the same laws \u00b7 ' + term("draw1", "draw 1") + '</p>';
    if (!plots.length) { return ctl + '<p class="v2hint">None of the plotted metrics has paired contrasts. Paired contrasts exist for: ' + PAIRED_KEYS.map(function (k) { return METRIC[k] ? METRIC[k].short : k; }).join(", ") + '.</p>'; }
    var src = timeSource(keys);
    var charts = inBlock(root.querySelector(".v2main"), plots.length, function () { return plots.map(function (p) { var series = [], ymin = Infinity, ymax = -Infinity, tmin = Infinity, tmax = -Infinity;
      axisMethods(others).forEach(function (m) { var pts = []; D.rungs.forEach(function (r) { var use = poolCats(m.key, r, keys).filter(function (c) { return cell(base.key, c, r); }); if (!use.length) { return; } var x = xOf(m.key, r, use, src); if (x === null) { return; } var st = pairedStat(p, m.key, base.key, r, use); if (!st || !isFinite(st.v)) { return; }
          pts.push({ x: x, v: st.v, lo: st.lo, hi: st.hi, thin: false, title: m.label + " − " + base.label + " @ " + r + ": " + fmtDelta(p, st.v) + " [" + fmtDelta(p, st.lo) + ", " + fmtDelta(p, st.hi) + "], n = " + st.n + " laws, p = " + fmtP(st.p) });
          [st.v, state.ci ? st.lo : st.v, state.ci ? st.hi : st.v].forEach(function (v) { if (isFinite(v)) { ymin = Math.min(ymin, v); ymax = Math.max(ymax, v); } }); if (state.xaxis === "time") { tmin = Math.min(tmin, x); tmax = Math.max(tmax, x); } });
        if (pts.length) { series.push({ label: m.label + (m.local ? " (local)" : ""), color: colorOf(m), pts: pts }); } });
      var title = "Δ " + p.short + " vs " + base.label;
      if (!series.length) { return chartSVG({ title: title, series: [], empty: "no matched cells with the baseline yet" }); }
      ymin = Math.min(ymin, 0); ymax = Math.max(ymax, 0); var pad = (ymax - ymin) * 0.1 || 0.05; ymin -= pad; ymax += pad;
      var tr = state.xaxis === "time" ? timeRange(tmin, tmax) : [0, 0];
      var lin = { kind: "cont", fmt: "num2", hist: null };
      return chartSVG({ title: title, series: series, ymin: ymin, ymax: ymax, ticks: ticksFor(lin, ymin, ymax), tick: function (g) { return fmtDelta(p, g); }, ylabel: "", timeAxis: state.xaxis === "time", timeSource: src, tmin: tr[0], tmax: tr[1], zero: 0 }); }); });
    var r = state.rung, rows = "";
    others.forEach(function (m) { rows += '<tr><td><span class="v2sw" style="background:' + colorOf(m) + '"></span>' + esc(m.label) + '</td>' + plots.map(function (p) { var use = poolCats(m.key, r, keys).filter(function (c) { return cell(base.key, c, r); }); var st = use.length ? pairedStat(p, m.key, base.key, r, use) : null; if (!st) { return '<td class="v2na">–</td><td class="v2na">–</td><td class="v2na">–</td>'; } var sig = st.p !== null && st.p < 0.05; return '<td' + (sig ? ' class="v2sig"' : "") + '>' + fmtDelta(p, st.v) + ' <span class="v2ci-txt">[' + fmtDelta(p, st.lo) + ", " + fmtDelta(p, st.hi) + ']</span></td><td>' + fmtP(st.p) + '</td><td class="v2hint">' + st.wins + " / " + st.losses + " of " + st.n + '</td>'; }).join("") + '</tr>'; });
    var anyRow = others.some(function (m) { return plots.some(function (p) { var use = poolCats(m.key, r, keys).filter(function (c) { return cell(base.key, c, r); }); return use.length && pairedStat(p, m.key, base.key, r, use); }); });
    var table = '<h3 class="v2h">At rung ' + r + '</h3>' + (anyRow ? "" : '<p class="v2hint">No matched cells with the baseline at rung ' + r + ' yet: pick another rung under Options.</p>') + '<div class="v2table-wrap"><table class="v2table"><thead><tr><th>method − ' + esc(base.label) + '</th>' + plots.map(function (p) { return '<th colspan="3">' + esc(p.short) + '</th>'; }).join("") + '</tr><tr><th></th>' + plots.map(function () { return '<th>Δ [95 %]</th><th>p</th><th>wins / losses</th>'; }).join("") + '</tr></thead><tbody>' + rows + '</tbody></table></div>' +
      '<p class="v2hint">Rates: ' + term("mcnemar", "exact McNemar") + ' on the laws the pair disagrees on; wins / losses count those laws. Continuous metrics: ' + term("signtest", "paired mean difference and exact sign test") + ' over laws where both have a finite value; ratios and times are read as multiplicative factors. Bold: p below 0.05, uncorrected.</p>';
    return ctl + '<div class="v2charts">' + charts.join("") + "</div>" + table;
  }

  // ---- shell -----------------------------------------------------------------------------------------------------
  var VIEWS = [["curves", "Curves"], ["table", "Tables"], ["matrix", "Catalogs"], ["dist", "Distribution"], ["paired", "Paired Δ"]];
  // Each display carries its own controls. A control that cannot change what is on screen is not shown.
  var USES = {
    curves: { plots: 1, stat: 1, pool: 1, ci: 1, thin: 1, xaxis: 1 },
    table: { plots: 1, rows: 1, stat: 1, pool: 1, ci: 1, rung: 1 },
    matrix: { focus: 1, stat: 1, rung: 1 },
    dist: { focus: 1, stat: 1, rung: 1 },
    paired: { plots: 1, base: 1, ci: 1, xaxis: 1, rung: 1 }
  };
  function usesFor(view) {
    var u = {}, src = USES[view] || {};
    Object.keys(src).forEach(function (k) { u[k] = src[k]; });
    if (view === "table" && state.rows !== "cats") { delete u.rung; }   // the budget only binds the by-catalog table
    return u;
  }
  function shell() {
    var rel = D.release;
    var strip = D.methods.filter(function (m) { return D.status[m.key]; }).map(function (m) { var d = D.status[m.key][0], t = D.status[m.key][1]; return '<div class="v2tile" title="' + esc(m.label) + '"><b><span class="v2sw" style="background:' + colorOf(m) + '"></span>' + esc(m.label) + (m.local ? " (local)" : "") + '</b><span>' + d + '<small> / ' + (t == null ? "?" : t) + ' units</small></span><div class="v2bar"><i style="width:' + (t ? 100 * d / t : 0) + '%"></i></div></div>'; }).join("");
    var catList = CATS.map(function (c) { var m = CAT[c]; return '<label title="' + esc(GROUPS[m.group] + (m.mu ? " · median law complexity " + m.mu[1] + " bits (IQR " + m.mu[0] + " to " + m.mu[2] + ")" : "")) + '"><input type="checkbox" data-c="' + c + '"> ' + esc(c) + ' <span class="v2hint">' + m.laws + '</span></label>'; }).join("");
    var methList = D.methods.filter(withData).map(function (m) { return '<div class="v2meth"><label><input type="checkbox" data-m="' + m.key + '"><input type="color" class="v2swatch" data-m="' + m.key + '" value="' + colorOf(m) + '" title="Colour for ' + esc(m.label) + '"><span class="v2mname">' + esc(m.label) + '</span></label>' + (m.local ? ' <span class="v2tag v2tag-local">local only</span>' : "") + ' <span class="v2hint">' + esc(m.param) + '</span>' + (m.selection ? " " + help(m.selection, "How does " + m.label + " choose its answer?") : "") + ' <span class="v2tag" title="' + esc(PROV_NOTE[m.provenance] || "") + '">' + esc(PROV[m.provenance] || m.provenance || "") + '</span><button type="button" class="v2reset" data-m="' + m.key + '" title="Reset colour to default" hidden>↺</button></div>'; }).join("") || '<span class="v2hint">no method has finished units yet</span>';
    var metricList = MGROUPS.map(function (g) { var ms = D.metrics.filter(function (m) { return m.group === g; }); return '<div class="v2mgroup" data-group="' + esc(g) + '"><h4>' + esc(g) + '</h4>' + ms.map(function (m) { return '<div class="v2metric" data-tier="' + m.tier + '" data-key="' + m.key + '"><label><input type="checkbox" data-p="' + m.key + '"> ' + esc(m.label) + '</label> ' + help(m.desc + (m.kind === "rate" ? " Defined for every law." : " Successful predictions only.") + (m.higher === true ? " Higher is better." : m.higher === false ? " Lower is better." : ""), "What is " + m.label + "?") + '</div>'; }).join("") + "</div>"; }).join("");
    root.innerHTML =
      '<div class="v2head"><div><div class="v2kicker">benchmark release ' + esc(rel.id) + '</div><p class="v2sub">' + esc(rel.title !== rel.id ? rel.title + " · " : "") + 'generated ' + esc(rel.generated) + '. ' + esc(rel.notes || "") + '</p></div><div class="v2row"><button type="button" class="v2btn" data-act="link">copy link to this view</button><span class="v2linkok v2hint" hidden>link copied</span></div></div>' +
      '<details class="v2release"><summary>Protocol of this release</summary><ul><li><b>Choosing an answer.</b> ' + esc(rel.scoring || "") + '</li><li><b>Judging it.</b> ' + esc(rel.judge || "") + '</li><li><b>Data.</b> One problem per law: 512 support points and 512 validation points from the catalog\'s own ranges, no noise; ' + CATS.length + ' catalogs, ' + laws(CATS) + ' laws.</li><li><b>Configurations.</b> ' + term("provenance", "Who chose each method\'s configuration") + ' is shown next to every method.</li><li><b>Time axis.</b> ' + term("time", "Reference-machine timing") + (D.timing_note ? " · " + esc(D.timing_note) : "") + '</li><li><b>Statistics.</b> ' + term("regime", "Two regimes") + ', ' + term("matched", "matched pooling") + ', ' + term("wilson", "95 % intervals") + '.</li></ul></details>' +
      '<div class="v2strip">' + strip + '</div>' +
      '<div class="v2tabs" role="tablist">' + VIEWS.map(function (v) { return '<button type="button" class="v2tab" role="tab" data-view="' + v[0] + '">' + v[1] + '</button>'; }).join("") + '</div>' +
      '<div class="v2layout"><aside class="v2side">' +
      // 1. what this display shows. Every row declares the views it belongs to; the rest stay out of the way.
      '<div class="v2panel v2panel-show"><h3><span class="v2showtitle">Plots</span> <span class="v2hint v2metcount"></span></h3>' +
      '<div class="v2row" data-uses="plots"><input type="search" class="v2q" placeholder="filter metrics" aria-label="filter metrics"><label><input type="checkbox" class="v2tier"> show all ' + D.metrics.length + '</label></div>' +
      '<div class="v2metrics" data-uses="plots">' + metricList + '</div>' +
      '<div class="v2row" data-uses="focus"><span class="v2lab">metric</span><select class="v2focus" aria-label="metric shown in this view">' + D.metrics.map(function (m) { return '<option value="' + m.key + '">' + esc(m.label) + '</option>'; }).join("") + '</select></div>' +
      '<div class="v2row" data-uses="rows"><span class="v2lab">rows</span><label><input type="radio" name="v2rows" value="rungs"> one per budget</label><label><input type="radio" name="v2rows" value="cats"> one per catalog</label></div>' +
      '<div class="v2row" data-uses="base"><span class="v2lab">baseline</span><select class="v2base" aria-label="baseline method">' + D.methods.filter(withData).map(function (m) { return '<option value="' + m.key + '">' + esc(m.label) + '</option>'; }).join("") + '</select></div>' +
      '<div class="v2row" data-uses="rung"><span class="v2lab">budget</span><select class="v2rung" aria-label="budget per problem">' + D.rungs.map(function (r) { return '<option value="' + r + '">' + r + '</option>'; }).join("") + '</select><span class="v2hint">candidates or seconds, per method</span></div></div>' +
      // 2. what it is shown for
      '<div class="v2panel"><h3>Methods</h3><div class="v2methods">' + methList + '</div><div class="v2row v2colour"><button type="button" class="v2btn" data-act="reset-colours">reset all colours</button><span class="v2hint v2cookie" hidden>A single functional cookie remembers your colour choices on this device: no tracking, no third parties. It is written only when you change a colour; \u201creset all colours\u201d deletes it.</span></div></div>' +
      '<div class="v2panel"><h3>Catalogs <span class="v2hint v2catcount"></span></h3><div class="v2row"><button type="button" data-act="all">all</button><button type="button" data-act="none">none</button><button type="button" data-act="phys">physics</button><button type="button" data-act="classic">classical</button><button type="button" data-act="synth">synthetic</button></div><div class="v2cats">' + catList + '</div></div>' +
      // 3. how the numbers are read
      '<div class="v2panel"><h3>Reading</h3>' +
      '<div class="v2row" data-uses="stat"><span class="v2lab">statistic</span><label><input type="radio" name="v2stat" value="mean"> ' + term("mean", "mean") + '</label><label><input type="radio" name="v2stat" value="median"> ' + term("median", "median") + '</label></div>' +
      '<div class="v2row" data-uses="pool"><span class="v2lab">' + term("matched", "pooling") + '</span><label><input type="radio" name="v2pool" value="matched"> matched</label><label><input type="radio" name="v2pool" value="own"> own</label></div>' +
      '<div class="v2row" data-uses="xaxis"><span class="v2lab">x axis</span><label><input type="radio" name="v2xaxis" value="time" class="v2xtime"> ' + term("time", "time") + '</label><label><input type="radio" name="v2xaxis" value="rung"> ' + term("candidates", "candidates") + '</label><span class="v2hint v2xtimehint"></span></div>' +
      '<div class="v2row v2checks" data-uses="ci thin"><label data-uses="ci"><input type="checkbox" class="v2ci"> 95 % intervals ' + help(TERMS.wilson) + '</label><label data-uses="thin"><input type="checkbox" class="v2thin"> show ' + term("thin", "thin rungs") + '</label></div></div>' +
      '</aside><section class="v2main"><p class="v2err" role="alert"></p><div class="v2view"></div></section></div>';
    var hasTiming = anyTime();
    root.querySelector(".v2xtime").disabled = !hasTiming; root.querySelector(".v2xtimehint").textContent = hasTiming ? "" : "(no measurements yet)";
    if (!hasTiming && state.xaxis === "time") { state.xaxis = "rung"; }
  }
  function syncControls() {
    root.querySelectorAll(".v2cats input").forEach(function (i) { i.checked = state.cats.indexOf(i.dataset.c) >= 0; });
    root.querySelectorAll(".v2methods input[type=checkbox]").forEach(function (i) { i.checked = state.methods.indexOf(i.dataset.m) >= 0; });
    root.querySelectorAll(".v2metrics input[type=checkbox]").forEach(function (i) { i.checked = state.plots.indexOf(i.dataset.p) >= 0; });
    var q = state.q.toLowerCase();
    root.querySelectorAll(".v2metric").forEach(function (l) { var mm = METRIC[l.dataset.key]; var show = state.tier === "all" || l.dataset.tier === "main" || state.plots.indexOf(l.dataset.key) >= 0; if (q) { show = mm.label.toLowerCase().indexOf(q) >= 0 || l.dataset.key.indexOf(q) >= 0 || mm.group.toLowerCase().indexOf(q) >= 0; } l.hidden = !show; });
    root.querySelectorAll(".v2mgroup").forEach(function (g) { g.hidden = !Array.prototype.some.call(g.querySelectorAll(".v2metric"), function (l) { return !l.hidden; }); });
    root.querySelector(".v2tier").checked = state.tier === "all"; if (root.querySelector(".v2q").value !== state.q) { root.querySelector(".v2q").value = state.q; }
    root.querySelectorAll("input[name=v2stat]").forEach(function (i) { i.checked = i.value === state.stat; });
    root.querySelectorAll("input[name=v2pool]").forEach(function (i) { i.checked = i.value === state.pool; });
    root.querySelectorAll("input[name=v2xaxis]").forEach(function (i) { i.checked = i.value === state.xaxis; });
    root.querySelector(".v2ci").checked = state.ci; root.querySelector(".v2thin").checked = state.thin;
    root.querySelector(".v2focus").value = state.focus; root.querySelector(".v2rung").value = String(state.rung);
    root.querySelectorAll("input[name=v2rows]").forEach(function (i) { i.checked = i.value === state.rows; });
    if (state.base) { root.querySelector(".v2base").value = state.base; }
    root.querySelectorAll(".v2tab").forEach(function (b) { b.classList.toggle("active", b.dataset.view === state.view); b.setAttribute("aria-selected", b.dataset.view === state.view ? "true" : "false"); });
    root.querySelectorAll(".v2swatch").forEach(function (i) { var m = D.methods.filter(function (x) { return x.key === i.dataset.m; })[0]; if (m && document.activeElement !== i) { i.value = colorOf(m); } });
    root.querySelectorAll(".v2reset").forEach(function (b) { b.hidden = !userColors[b.dataset.m]; });
    root.querySelector(".v2metcount").textContent = state.view === "matrix" || state.view === "dist" ? "" : state.plots.length + " plotted";
    root.querySelector(".v2showtitle").textContent = state.view === "matrix" || state.view === "dist" ? "Metric" : state.view === "paired" ? "Contrast" : "Plots";
    var uses = usesFor(state.view);   // hide every control this display cannot use, then the panels left empty
    root.querySelectorAll("[data-uses]").forEach(function (el) { el.hidden = !el.dataset.uses.split(" ").some(function (k) { return uses[k]; }); });
    root.querySelectorAll(".v2panel").forEach(function (p) {
      var rows = p.querySelectorAll("[data-uses]");
      if (rows.length) { p.hidden = !Array.prototype.some.call(rows, function (r) { return !r.hidden; }); }
    });
    root.querySelector(".v2catcount").textContent = state.cats.length + " of " + CATS.length + " \u00b7 " + laws(state.cats).toLocaleString() + " laws";
  }
  var rt = null;
  function scheduleRender() { clearTimeout(rt); rt = setTimeout(render, 30); }
  function render() {
    try {
      syncControls();
      var shown = shownMethods(), view = root.querySelector(".v2view");
      if (state.stat === "median" && state.view !== "dist" && state.view !== "paired") {   // histograms the current view needs
        (state.view === "matrix" ? [METRIC[state.focus]] : D.metrics.filter(function (m) { return state.plots.indexOf(m.key) >= 0; })).forEach(function (m) { if (m.kind === "cont") { ensure("hist/" + m.key + ".js", scheduleRender); } });
      }
      renderHeadline();
      view.innerHTML = state.view === "table" ? renderTable(shown) : state.view === "matrix" ? renderMatrix(shown) : state.view === "dist" ? renderDist(shown) : state.view === "paired" ? renderPaired(shown) : renderCurves(shown);
      root.querySelector(".v2err").textContent = ""; save();
    } catch (e) { root.querySelector(".v2err").textContent = "The explorer hit an error while drawing: " + (e && e.message ? e.message : e) + ". Reload the page, or press “all” under Catalogs to reset the selection."; if (window.console) { console.error(e); } }
  }

  // ---- events ----------------------------------------------------------------------------------------------------
  shell();
  root.addEventListener("change", function (e) {
    var t = e.target;
    if (t.dataset.c) { if (t.checked) { state.cats.push(t.dataset.c); } else { state.cats = state.cats.filter(function (c) { return c !== t.dataset.c; }); } }
    else if (t.dataset.m && t.type === "checkbox") { if (t.checked) { state.methods.push(t.dataset.m); } else { state.methods = state.methods.filter(function (c) { return c !== t.dataset.m; }); } }
    else if (t.dataset.p) { if (t.checked) { state.plots.push(t.dataset.p); } else { state.plots = state.plots.filter(function (c) { return c !== t.dataset.p; }); } }
    else if (t.name === "v2stat") { state.stat = t.value; } else if (t.name === "v2pool") { state.pool = t.value; } else if (t.name === "v2xaxis") { state.xaxis = t.value; } else if (t.name === "v2rows") { state.rows = t.value; }
    else if (t.classList.contains("v2ci")) { state.ci = t.checked; } else if (t.classList.contains("v2thin")) { state.thin = t.checked; } else if (t.classList.contains("v2tier")) { state.tier = t.checked ? "all" : "main"; }
    else if (t.classList.contains("v2focus")) { state.focus = t.value; } else if (t.classList.contains("v2rung")) { state.rung = parseInt(t.value, 10); } else if (t.classList.contains("v2base")) { state.base = t.value; }
    else if (t.classList.contains("v2swatch")) { userColors[t.dataset.m] = t.value; writeCookie(userColors); root.querySelector(".v2cookie").hidden = false; }
    else { return; }
    render();
  });
  root.addEventListener("input", function (e) { var t = e.target; if (t.classList.contains("v2q")) { state.q = t.value; syncControls(); } else if (t.classList.contains("v2swatch")) { userColors[t.dataset.m] = t.value; render(); } });
  root.addEventListener("click", function (e) {
    var b = e.target.closest ? e.target.closest("button") : null; if (!b || !root.contains(b) || b.classList.contains("v2help")) { return; }
    if (b.dataset.view) { state.view = b.dataset.view; render(); return; }
    if (b.classList.contains("v2reset")) { delete userColors[b.dataset.m]; writeCookie(userColors); render(); return; }
    var act = b.dataset.act; if (!act) { return; }
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
    if (!el || prev === el || !(root.contains(el) || (headRoot && headRoot.contains(el)))) { return; }
    e.preventDefault();
    var pop = document.createElement("div"); pop.className = "v2pop"; pop.textContent = el.classList.contains("v2help") ? el.dataset.help : (TERMS[el.dataset.term] || ""); pop._anchor = el; document.body.appendChild(pop);
    var margin = 8; pop.style.maxWidth = Math.min(520, window.innerWidth - 2 * margin) + "px"; var r = el.getBoundingClientRect();
    var left = Math.min(Math.max(r.left, margin), window.innerWidth - pop.offsetWidth - margin), top = r.bottom + 6;
    if (top + pop.offsetHeight > window.innerHeight - margin && r.top - pop.offsetHeight - 6 > margin) { top = r.top - pop.offsetHeight - 6; }
    pop.style.left = left + "px"; pop.style.top = top + "px";
    popArmed = false; window.requestAnimationFrame(function () { window.requestAnimationFrame(function () { popArmed = true; }); });
  });
  document.addEventListener("keydown", function (e) { if (e.key === "Escape") { closePop(); } });
  window.addEventListener("resize", function () { closeArmed(); scheduleRender(); });
  // The first paint can measure a container that has not settled (fonts, the sidebar, a scrollbar), and a
  // chart built for the wrong width is a chart whose labels are the wrong size. Watch and redraw.
  if (window.ResizeObserver) {
    var lastW = 0;
    var ro = new ResizeObserver(function () { var w = hostWidth(); if (Math.abs(w - lastW) > 8) { lastW = w; scheduleRender(); } });
    [root.querySelector(".v2main"), headRoot].forEach(function (el) { if (el) { ro.observe(el); } });
  }
  document.addEventListener("scroll", closeArmed, true);
  render();
  if (fromUrl) { try { root.scrollIntoView({ block: "start" }); } catch (e) { /* ignore */ } }
})();
