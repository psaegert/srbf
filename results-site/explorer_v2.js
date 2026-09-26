/* Results explorer for benchmark releases from 2026-09 on (srbf 0.20 / flash-ansr 0.18, the two-part code).
 *
 * Reads window.RESULTS_V2 (schema 2): the metric REGISTRY and per
 * method x catalog x rung cell counts, sums and sums of squares, so any catalog subset is pooled on the client with
 * confidence bands. Per-metric histograms (pooled medians, the Distribution view) and the within-draw paired contrasts
 * (the Paired view) are fetched on demand from <base>hist/<metric>.js and <base>paired.js.
 * window.RESULTS_V2_PRIVATE, if a LOCAL build provides it, is merged in (results-site/README.md, "Local-only
 * methods"); the public page never references such a file.
 * Views: Curves (every plotted metric vs budget or time) | Table (rungs, or catalogs at one rung) | Catalogs (matrix
 * of one metric at one rung) | Distribution (per-method histograms, cumulative curves, a box per catalog, quartiles
 * along the ladder; per-catalog rates for rate metrics) | Ranks (mean rank within each problem, critical difference,
 * head-to-head table; from the pairwise outcomes in <base>ranks.js) | Paired Δ (each method against a baseline on
 * the same laws: exact McNemar for rates, paired t + sign test otherwise).
 * State lives in the URL (?release=...&v=...) for sharing and in localStorage for convenience; colours in the same
 * first-party cookie srbf_colors, written only on an explicit change. The site's 2026-07 release (explorer.js) was
 * retired on 2026-09-26; its links' keys are dropped from the URL and its ?release= opens this page. */
(function () {
  "use strict";
  var root = document.getElementById("results-explorer-v2");
  if (!root || typeof window.RESULTS_V2 === "undefined") { return; }
  var headRoot = document.getElementById("results-headline-v2");   // the two fixed charts above the explorer
  var D = JSON.parse(JSON.stringify(window.RESULTS_V2));
  var REL = D.release.id;
  // another release is on screen (the routing script decided before this file ran): leave the page and its URL alone
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
    if (withBase && P.base) { SOURCES.push({ base: P.base, local: true }); }
    return added;
  }
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
    upstream_default: "Settings: the defaults the method's own release ships with. Nothing was tuned.",
    author_blessed: "Settings: chosen by the method's authors. The Flash-ANSR authors also run this benchmark, and this label says so.",
    harness_tuned: "Settings: chosen by the benchmark's maintainers."
  };
  var TERMS = {
    complete: "Each point is one method at one budget. Each method is run over its own range of budgets, and some runs are still in progress. A point appears only once the method has results for all the problem sets you selected (in the headline charts: all 29), so every point covers the same problems. On some problem sets only one of the two runs may be finished yet, so the number of runs behind a point can differ between methods. \u201cHow to read the results\u201d below explains why, and how to see partial results by selecting fewer problem sets.",
    wilson: "The 95 % interval shows how precisely a value is known, given how much the results vary from problem to problem. It treats the problems as a random sample of similar problems, and each of a problem's two runs as a separate problem. Bands shade the interval around each point and between neighbouring points; crosses draw it as bars through each point. You can show either, both or neither.",
    median: "The median is read from a histogram with 128 bins, so it is accurate to the width of one bin (for log10 FVU, about 0.16).",
    mean: "The mean averages over the problems where the value is finite. log10 FVU stops at -15.65, the precision of a 64-bit float, so an exact fit counts there like any other fit. A formula that blows up has an infinite value: the mean leaves it out, while the median counts it as the worst. The tooltip of each point shows how many values it averages.",
    regime: "How a problem without a usable formula counts. In a rate, such as Numeric Recovery, it counts as a miss. In the overlap metrics (Token Overlap, Variable Overlap) it counts as 0, unless you leave such problems out with the switch that appears in the side panel for these metrics. All other metrics are measured only over the problems where the method returned a usable formula; most of them have no worst value, since a formula can always be worse.",
    impute: "What should a problem without a usable formula count in the overlap metrics? Counted (the default): it counts as 0, the lowest possible value, so a method cannot look better by failing on hard problems. Left out: the average covers only the problems where the method returned a usable formula. Rates always count such a problem as a miss.",
    valid: "Most metrics can only be measured on some problems: where the method returned a usable formula, and the value could be computed for it (a formula can, for example, give infinite values, or be one SimpliPy cannot simplify or price in bits). A method that fails on the hard problems then looks better than it is, because only its easier problems are measured. A point is drawn hollow when it is based on fewer than this share of the problems. The setting is under Reading in the explorer's side panel; 0 % turns the marking off.",
    mcnemar: "For a rate, only the problems where the two methods disagree (one recovered the formula, the other did not) can tell them apart. The p-value comes from the exact McNemar test on those problems. A p-value below 0.05 means a difference this large is unlikely to be chance. The interval is the 95 % interval of the difference in rates.",
    signtest: "Over the problems where both methods have a finite value: the mean difference with its 95 % interval, and a sign test that asks whether one method is better on more problems than chance would give. The two answer different questions and can disagree: a few large differences move the mean, while the sign test only counts better and worse. A p-value below 0.05 means a difference this large is unlikely to be chance.",
    draw1: "Every method is run twice on each problem, with newly sampled points each time. A comparison pairs the two methods on the same problem in the same run, so both had exactly the same data. Every run that both methods have finished counts, so a problem usually counts twice.",
    time: "Seconds per problem, measured for every method on the same workstation (16 CPU cores, one RTX 4090 GPU), one method at a time, on a fixed sample of 262 problems. Timings from the compute cluster that produces most of the results depend on which machine a job ran on, so they are never shown. A method that has not been timed on the workstation yet has no place on a time axis; it is named below the chart instead.",
    candidates: "The budget of the methods that generate candidate formulas: how many candidates they may generate per problem. NeSymReS is placed here by its beam width, the number of partial formulas its search keeps. PySR counts search iterations, which cannot be placed on this axis, so it appears on the time axis only.",
    rungs: "A budget is how much search a method may spend on one problem, in the method's own unit: candidate formulas, beam width or search iterations. Budgets are powers of two (1, 2, 4, 8 and so on), each method over its own range, and each point on a curve is one of them. Because the units differ, the same budget number means different amounts of work for different methods; the time axis shows what each budget costs.",
    tbudget: "Each method is compared at its largest finished budget that takes at most this many seconds per problem on our timing workstation. A method is left out when it has not been timed yet, or when its smallest budget already takes longer. A hollow dot marks a method whose largest budget still stays under the limit: with more budget it might do better.",
    worstrank: "On each problem the methods are ordered from best to worst on the chosen metric: the best gets place 1, the next place 2, and so on. A method without a usable formula for that problem gets the last place. Methods with equal values share the average of their places. Each of a problem's two runs is ordered separately, and every run counts the same. Only the order matters, so a narrow win counts as much as a wide one.",
    friedman: "Friedman's test checks whether the methods' average places differ more than chance would make them differ if all methods were equally good. Groups are only drawn when the test finds such a difference at the 5 % level. It is computed without a correction for ties, which only makes it more cautious.",
    cd: "The critical difference is the smallest gap between two average places that is unlikely to be chance (Nemenyi test, 5 % level), allowing for the fact that every pair of methods is compared. It gets smaller with more problems and larger with more methods. It counts each of a problem's two runs as a separate problem.",
    winshare: "The share of one-on-one comparisons a method wins, against every other method on every problem; a tie counts as half a win. 100 % means it beats every other method on every problem, and 50 % means it wins as often as it loses. Unlike the average place, it stays on the same 0 to 100 % scale when the number of methods changes; its value still depends on which methods are compared.",
    provenance: "Who chose each method's settings. Upstream defaults: the settings the method's own release ships with; nothing was tuned. Author-blessed: settings chosen by the method's authors; for Flash-ANSR, these are also the authors of this benchmark. Maintainer-chosen: settings chosen by the benchmark's maintainers."
  };
  var COOKIE = "srbf_colors";
  function readCookie() { var m = document.cookie.match(new RegExp("(?:^|; )" + COOKIE + "=([^;]*)")); if (!m) { return {}; } try { return JSON.parse(decodeURIComponent(m[1])) || {}; } catch (e) { return {}; } }
  function writeCookie(obj) { if (!obj || !Object.keys(obj).length) { document.cookie = COOKIE + "=;path=/;max-age=0;SameSite=Lax"; return; } document.cookie = COOKIE + "=" + encodeURIComponent(JSON.stringify(obj)) + ";path=/;max-age=" + (60 * 60 * 24 * 365) + ";SameSite=Lax"; }
  var userColors = readCookie();
  // A method drawn in the page's ink (the oracle) follows the theme: black on the light page, white on the dark one.
  function ink() { var v = window.getComputedStyle(document.documentElement).getPropertyValue("--ink"); return (v && v.trim()) || "#000000"; }
  function colorOf(m) { return userColors[m.key] || (m.ink ? ink() : m.color); }
  function dashOf(sr) { return sr && sr.dash ? ' stroke-dasharray="7 4"' : ""; }
  function esc(s) { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/"/g, "&quot;"); }
  function term(key, text) { return '<span class="v2term" data-term="' + key + '" role="button" tabindex="0">' + text + "</span>"; }
  function help(text, label) { return '<button type="button" class="v2help" data-help="' + esc(text) + '" aria-label="' + esc(label || "What does this mean?") + '">?</button>'; }

  // ---- state: URL > localStorage > defaults --------------------------------------------------------------------
  var withData = function (m) { return D.cells[m.key] && Object.keys(D.cells[m.key]).length; };
  var VALID_DEFAULT = 90;
  var DEFAULTS = function () {
    return { view: "curves", cats: CATS.slice(), methods: D.methods.filter(withData).map(function (m) { return m.key; }),
      plots: D.metrics.filter(function (m) { return m.tier === "main"; }).map(function (m) { return { x: defaultAxis(), y: m.key }; }),
      focus: "numeric_recovery_val", stat: "mean", band: true, cross: false, xaxis: anyTime() ? "time" : "rung", rung: 64, base: null, tier: "main", q: "", rows: "rungs",
      // the Distribution view reads a continuous metric by default (a rate has no distribution over problems), the Ranks
      // view the primary ranking metric; each display remembers its own
      dmetric: "log10_fvu_val", dmode: "hist", dnorm: "ok", rmetric: (D.rank_keys || ["log10_fvu_val"])[0], tbudget: null,
      // a point that rests on fewer than this share of the problems is drawn hollow
      valid: VALID_DEFAULT,
      // a metric whose range has a worst value: does a failed prediction count that value, or is it left out?
      impute: true };
  };
  var state = DEFAULTS();
  var LS = "srbf-v2-" + REL + ".8";   // bumped whenever a default changes (.2 time axis, .3 mean, .4 bands, .5 per-view metrics, .6 complete pools only, .7 hollow markers, .8 failed predictions counted or left out), so a saved state cannot pin the old one
  var rungChosen = false;   // a budget from a link or from storage is kept; otherwise the first render picks one that has data
  function loadState() {
    try { var s = JSON.parse(localStorage.getItem(LS) || "null"); if (s) { rungChosen = s.rung !== undefined; Object.keys(state).forEach(function (k) { if (s[k] !== undefined) { state[k] = s[k]; } }); } } catch (e) { /* no storage */ }
    var q = new URLSearchParams(window.location.search); var any = false;
    var preset = { all: CATS, phys: CATS.filter(function (c) { return CAT[c].group === "physics"; }), classic: CATS.filter(function (c) { return CAT[c].group === "classical"; }), synth: CATS.filter(function (c) { return CAT[c].group === "synthetic"; }), none: [] };
    if (q.has("v")) { state.view = q.get("v"); any = true; }
    if (q.has("c")) { var c = q.get("c"); state.cats = preset[c] ? preset[c].slice() : c.split(",").filter(function (x) { return CAT[x]; }); any = true; }
    if (q.has("m")) { state.methods = q.get("m").split(",").filter(function (x) { return D.methods.some(function (mm) { return mm.key === x; }); }); any = true; }
    if (q.has("p")) { state.plots = q.get("p").split(",").map(asPlot).filter(Boolean); any = true; }
    if (q.has("f") && METRIC[q.get("f")]) { state.focus = q.get("f"); any = true; }
    if (q.has("s")) { state.stat = q.get("s") === "median" ? "median" : "mean"; any = true; }
    if (q.has("ci")) { state.band = q.get("ci") !== "0"; state.cross = false; any = true; }   // the old single flag
    if (q.has("band")) { state.band = q.get("band") !== "0"; any = true; }
    if (q.has("cross")) { state.cross = q.get("cross") !== "0"; any = true; }
    if (q.has("x")) { state.xaxis = q.get("x") === "time" ? "time" : "rung"; any = true; }
    if (q.has("r")) { var r = parseInt(q.get("r"), 10); if (D.rungs.indexOf(r) >= 0) { state.rung = r; rungChosen = true; } any = true; }
    if (q.has("dm") && METRIC[q.get("dm")]) { state.dmetric = q.get("dm"); any = true; }
    if (q.has("dv")) { state.dmode = q.get("dv"); any = true; }
    if (q.has("dn")) { state.dnorm = q.get("dn") === "all" ? "all" : "ok"; any = true; }
    if (q.has("rm") && METRIC[q.get("rm")]) { state.rmetric = q.get("rm"); any = true; }
    if (q.has("t")) { state.tbudget = q.get("t"); any = true; }
    if (q.has("b")) { state.base = q.get("b"); any = true; }
    if (q.has("rows")) { state.rows = q.get("rows") === "cats" ? "cats" : "rungs"; any = true; }
    if (q.has("ok")) { state.valid = parseInt(q.get("ok"), 10); any = true; }
    if (q.has("imp")) { state.impute = q.get("imp") !== "0"; any = true; }
    if (q.has("tier")) { state.tier = q.get("tier") === "all" ? "all" : "main"; }
    if (["curves", "table", "matrix", "dist", "ranks", "paired"].indexOf(state.view) < 0) { state.view = "curves"; }
    if (["hist", "ecdf", "cats", "rungs"].indexOf(state.dmode) < 0) { state.dmode = "hist"; }
    if (!METRIC[state.dmetric]) { state.dmetric = "log10_fvu_val"; }
    if (!METRIC[state.rmetric]) { state.rmetric = (D.rank_keys || ["log10_fvu_val"])[0]; }
    if (!Array.isArray(state.cats)) { state.cats = CATS.slice(); }
    if (!Array.isArray(state.methods)) { state.methods = DEFAULTS().methods; }
    if (!Array.isArray(state.plots)) { state.plots = DEFAULTS().plots; }
    state.cats = state.cats.filter(function (x) { return CAT[x]; });
    state.plots = state.plots.map(asPlot).filter(Boolean);   // from a link, from storage, or from an older shape
    state.methods = state.methods.filter(function (x) { return D.methods.some(function (mm) { return mm.key === x; }); });
    if (!METRIC[state.focus]) { state.focus = "numeric_recovery_val"; }
    if (!state.base || state.methods.indexOf(state.base) < 0) { state.base = state.methods[0] || null; }
    if (D.rungs.indexOf(state.rung) < 0) { state.rung = 64; }
    state.impute = state.impute !== false;
    state.valid = isFinite(state.valid) ? Math.min(100, Math.max(0, Math.round(state.valid))) : VALID_DEFAULT;
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
  // A link is meant to be passed around, so it names only methods the release publishes; a method that arrived
  // with a key is remembered for this browser (localStorage, this device only) and re-added by the key, not by a URL.
  var byKey = {};
  function sharedMethods() { return state.methods.filter(function (k) { return !byKey[k]; }); }
  function save() {
    try { localStorage.setItem(LS, JSON.stringify(state)); } catch (e) { /* no storage */ }
    var q = new URLSearchParams(window.location.search);
    ["view", "bench", "baseline", "metric", "budget"].forEach(function (k) { q.delete(k); });   // never carry the retired 2026-07 explorer's keys
    q.set("release", REL); q.set("v", state.view); q.set("c", catsParam()); q.set("m", sharedMethods().join(",")); q.set("p", state.plots.map(plotKey).join(","));
    q.set("f", state.focus); q.set("s", state.stat); q.delete("pool"); q.delete("thin"); q.set("band", state.band ? "1" : "0"); q.set("cross", state.cross ? "1" : "0");
    q.set("x", state.xaxis); q.set("r", String(state.rung)); if (state.base) { q.set("b", state.base); } q.set("rows", state.rows); q.set("ok", String(state.valid)); q.set("imp", state.impute ? "1" : "0");
    ["dm", "dv", "dn", "rm", "t"].forEach(function (k) { q.delete(k); });   // a link carries only what its display reads
    if (state.view === "dist") { q.set("dm", state.dmetric); q.set("dv", state.dmode); q.set("dn", state.dnorm); }
    if (state.view === "ranks") { q.set("rm", state.rmetric); if (state.tbudget) { q.set("t", state.tbudget); } }
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

  // ---- adding a method from a key ---------------------------------------------------------------------------------
  // A release may ship one sealed payload next to the public ones. It holds the same kind of payload scripts as the
  // rest of the release -- an overlay, its histograms and its paired contrasts -- gzipped and encrypted with
  // AES-256-GCM under a key derived by PBKDF2-HMAC-SHA256. The key is not in this repository and nothing here
  // derives it; the payload is fetched only when someone asks for it, and a key that does not open it leaves the
  // page exactly as it was. What the page shows without it is complete on its own terms.
  var sealedAsked = false;
  function bytesOf(b64) { var s = atob(b64), u = new Uint8Array(s.length); for (var i = 0; i < s.length; i++) { u[i] = s.charCodeAt(i); } return u; }
  function withSealed(cb) {
    if (sealedAsked || window.RESULTS_V2_SEALED) { cb(); return; }
    sealedAsked = true;
    var sc = document.createElement("script"); sc.src = D.base + "sealed.js"; sc.async = true;
    sc.onload = cb; sc.onerror = cb; document.head.appendChild(sc);
  }
  function openWithKey(key) {   // -> the keys of the methods it added, or [] when nothing opened
    return new Promise(function (res) { withSealed(res); }).then(function () {
      var S = window.RESULTS_V2_SEALED, env = S && S[REL];
      var subtle = window.crypto && window.crypto.subtle;
      if (!env || !subtle || typeof DecompressionStream === "undefined") { return []; }
      return subtle.importKey("raw", new TextEncoder().encode(key), "PBKDF2", false, ["deriveKey"])
        .then(function (base) {
          return subtle.deriveKey({ name: "PBKDF2", salt: bytesOf(env.salt), iterations: env.iter, hash: "SHA-256" },
            base, { name: "AES-GCM", length: 256 }, false, ["decrypt"]);
        })
        .then(function (k) { return subtle.decrypt({ name: "AES-GCM", iv: bytesOf(env.iv) }, k, bytesOf(env.ct)); })
        .then(function (gz) { return new Response(new Blob([gz]).stream().pipeThrough(new DecompressionStream("gzip"))).text(); })
        .then(function (src) {
          (new Function(src))();   // the payload scripts, run exactly as a <script> tag would run them
          var added = mergeOverlay(window.RESULTS_V2_PRIVATE, false);
          added.forEach(function (k2) { byKey[k2] = true; if (state.methods.indexOf(k2) < 0) { state.methods.push(k2); } });
          return added;
        })
        .catch(function () { return []; });   // a key that does not fit is not told apart from a release without one
    });
  }
  function labelsOf(keys) {
    return keys.map(function (k) { var m = D.methods.filter(function (x) { return x.key === k; })[0]; return m ? m.label : k; }).join(", ");
  }
  function tryKey(key, quiet) {
    var msg = root.querySelector(".v2addmmsg");
    if (msg && !quiet) { msg.textContent = "checking…"; }
    openWithKey(key).then(function (added) {
      if (added.length) {
        try { window.sessionStorage.setItem("srbf.k", key); } catch (e) { /* storage off */ }
        shell(); render();
        var after = root.querySelector(".v2addmmsg");
        if (after && !quiet) { after.textContent = "Added " + labelsOf(added) + "."; }
        return;
      }
      if (!quiet && msg) { msg.textContent = "No method found for that key."; }
    });
  }

  // ---- pooling ---------------------------------------------------------------------------------------------------
  function cell(m, c, r) { var x = D.cells[m] && D.cells[m][c] && D.cells[m][c][String(r)]; return x && x.state === "complete" ? x : null; }
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
  function wilson(a, b) { if (!b) { return null; } var p = a / b, z2 = Z * Z; var ctr = (p + z2 / (2 * b)) / (1 + z2 / b), half = Z * Math.sqrt(p * (1 - p) / b + z2 / (4 * b * b)) / (1 + z2 / b); return { v: p, lo: ctr - half, hi: ctr + half, n: b }; }
  function binVal(h, i) { return h.lo + (i + 0.5) * (h.hi - h.lo) / h.nb; }
  function addHist(acc, hc, nb) { if (hc.length && Array.isArray(hc[0])) { hc.forEach(function (p) { acc[p[0]] += p[1]; }); } else { for (var i = 0; i < nb; i++) { acc[i] += hc[i] || 0; } } }
  // A metric whose range has a worst value ships with the failed predictions counted at it, and every cell says how
  // many of its values were filled in that way ("w"). Leaving them out again is exact: that many come off the sums
  // and out of the bin the worst value falls into.
  function leftOut(k) { return METRIC[k] && METRIC[k].worst !== undefined && !state.impute; }
  function filled(c, k) { return c && c.w && c.w[k] ? c.w[k] : 0; }
  function worstBin(H, k) { return Math.min(H.nb - 1, Math.max(0, Math.floor((METRIC[k].worst - H.lo) / (H.hi - H.lo) * H.nb))); }
  function pooledHist(k, m, r, cs) { var H = histOf(k); if (!H || !H.cells[m]) { return null; } var acc = new Array(H.nb).fill(0); cs.forEach(function (c) { var hc = H.cells[m][c] && H.cells[m][c][String(r)]; if (hc) { addHist(acc, hc, H.nb); if (leftOut(k)) { var wb = worstBin(H, k); acc[wb] = Math.max(0, acc[wb] - filled(cell(m, c, r), k)); } } }); var n = acc.reduce(function (a, b) { return a + b; }, 0); return n ? { h: acc, n: n, lo: H.lo, hi: H.hi, nb: H.nb } : null; }
  function quantileBin(h, kth) { var cum = 0; for (var i = 0; i < h.nb; i++) { cum += h.h[i]; if (cum >= kth) { return i; } } return h.nb - 1; }
  function tfOf(metric) { return metric.hist && metric.hist.tf; }
  function fwd(metric, x) { var tf = tfOf(metric); return tf === "log2" ? Math.log2(Math.max(1e-300, x)) : tf === "log10" ? Math.log10(Math.max(1e-300, x)) : x; }
  function back(metric, x) { var tf = tfOf(metric); return tf === "log2" ? Math.pow(2, x) : tf === "log10" ? Math.pow(10, x) : x; }
  // every statistic is returned in the metric's TRANSFORMED space (log2 for ratios, log10 for seconds), where the charts live
  // A rate is read over every problem, and so is a metric whose range has a worst value: a failed prediction takes it.
  // Every other metric has no worst value and describes the predictions that were made, so a method that fails on the
  // hard problems looks better there than it is. `share` is the part of the problems behind a point (of the problems the metric
  // can be defined for); below the reader's threshold the point is drawn hollow.
  function validShare(metric, cells) {
    if (metric.kind === "rate") { return 1; }
    var d = 0, e = 0, out = leftOut(metric.key); cells.forEach(function (c) { var t = c.m[metric.key]; d += t ? t[0] - (out ? filled(c, metric.key) : 0) : 0; e += c.e && c.e[metric.key] !== undefined ? c.e[metric.key] : c.n; });
    return e ? d / e : null;
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
  function ensureHists(metric) { histKeys(metric).forEach(function (k) { ensure("hist/" + k + ".js", scheduleRender); }); }
  var R2_FINE = 0.96;
  function r2Quantile(own, fvu, kth) {   // the kth smallest R^2 is the kth largest FVU
    var i = quantileBin(own, kth), v = binVal(own, i);
    if (i > 0 && v <= R2_FINE) { return { v: v, edge: 0 }; }
    var j = quantileBin(fvu, Math.min(fvu.n, Math.max(1, fvu.n + 1 - kth)));
    return { v: j === 0 ? 1 : 1 - Math.pow(10, binVal(fvu, j)), edge: j === fvu.nb - 1 ? -1 : 0 };
  }
  // a cell pools the draws complete for it (c.d, 1 when absent); a pooled statistic reports the fewest among its cells
  function stat(metric, m, r, cs) {
    var out = stat0(metric, m, r, cs);
    if (out && !out.pending) { var ds = cs.map(function (c) { return cell(m, c, r); }).filter(Boolean).map(function (c) { return c.d || 1; }); out.d = ds.length ? Math.min.apply(null, ds) : 1; }
    return out;
  }
  function stat0(metric, m, r, cs) {
    var cells = cs.map(function (c) { return cell(m, c, r); }).filter(Boolean); if (!cells.length) { return null; }
    if (metric.kind === "rate") { var a = 0, b = 0; cells.forEach(function (c) { var t = c.m[metric.key]; if (t) { a += t[0]; b += t[1]; } }); var w = wilson(a, b); if (w) { w.share = 1; } return w; }
    var share = validShare(metric, cells);
    if (statOf(metric) === "mean") {
      var nd = 0, n = 0, s = 0, ss = 0, out = leftOut(metric.key), w = out ? metric.worst : 0;
      cells.forEach(function (c) { var t = c.m[metric.key]; if (t) { var k = out ? filled(c, metric.key) : 0; nd += t[0] - k; n += t[1] - k; s += t[2] - k * w; ss += t[3] - k * w * w; } });
      if (n <= 0) { return null; } var mean = s / n, varr = Math.max(0, (ss - n * mean * mean) / Math.max(1, n - 1)), se = Math.sqrt(varr / n);
      return { v: fwd(metric, mean), lo: fwd(metric, mean - Z * se), hi: fwd(metric, mean + Z * se), n: n, ndef: nd, share: share };
    }
    if (!histKeys(metric).every(function (k) { return ready("hist/" + k + ".js"); })) { return { pending: true }; }
    var ph = pooledHist(metric.key, m, r, cs); if (!ph) { return null; }
    var half = Z * Math.sqrt(ph.n) / 2;
    if (metric.median_via) {
      var pf = pooledHist(metric.median_via, m, r, cs); if (!pf) { return null; }
      var q = function (kth) { return r2Quantile(ph, pf, kth); }, qm = q(Math.max(1, Math.ceil(ph.n / 2)));
      return { v: qm.v, lo: q(Math.max(1, Math.floor(ph.n / 2 - half))).v, hi: q(Math.min(ph.n, Math.ceil(ph.n / 2 + half))).v, n: ph.n, edge: qm.edge, share: share };
    }
    var mid = quantileBin(ph, ph.n / 2);
    var lo = quantileBin(ph, Math.max(1, Math.floor(ph.n / 2 - half))), hi = quantileBin(ph, Math.min(ph.n, Math.ceil(ph.n / 2 + half)));
    return { v: binVal(ph, mid), lo: binVal(ph, lo), hi: binVal(ph, hi), n: ph.n, edge: mid === 0 ? -1 : (mid === ph.nb - 1 ? 1 : 0), share: share };
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
    measurer.font = "12px " + (window.getComputedStyle(root).fontFamily || "sans-serif");
    return measurer.measureText(text).width;
  }
  function widest(labels) { return labels.reduce(function (w, t) { return Math.max(w, textWidth(t)); }, 0); }
  function legendRight(labels) { return Math.max(160, Math.ceil(LEG_GAP + 26 + widest(labels) + 16)); }   // gap, swatch, name, margin
  var LEG_GAP = 14;   // between the plot area and its legend
  var chartHost = null, chartCount = 0;   // set while a block renders: its charts are built for THAT container
  function hostWidth() { return chartWidth(chartHost || root.querySelector(".v2main") || root, chartCount); }
  // A display of one chart (Distribution, Ranks) spans the column up to WIDE_MAX and is drawn at the width it is shown
  // at: drawn at a grid cell's width and stretched, its text and marks would grow with the window.
  var WIDE_MAX = 980;   // must match .v2distwide and .v2one in styles.css
  function wideWidth() { return Math.min(WIDE_MAX, chartWidth(chartHost || root.querySelector(".v2main") || root, 1)); }
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
  function chartSVG(opts) {
    var series = opts.series, nr = narrow(), W = opts.width || hostWidth(), L = 76, T = opts.title ? 52 : 18;
    var R = nr ? 18 : legendRight(series.map(function (sr) { return sr.label; }));
    var B = nr ? 60 + 20 * Math.max(1, series.length) : 58, H = plotHeight(W) + B;
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
      : (nr ? "candidates" : "candidates per problem (log scale)"));
    s += '<text x="' + ((L + W - R) / 2).toFixed(0) + '" y="' + (H - B + 32) + '" class="tick" text-anchor="middle">' + esc(xlab) + '</text>';
    if (opts.ylabel) { s += '<text transform="translate(' + YLABEL_X + ',' + ((T + H - B) / 2).toFixed(0) + ') rotate(-90)" class="tick" text-anchor="middle">' + esc(opts.ylabel) + "</text>"; }
    var ly = nr ? H - B + 46 : T + 6, lx = nr ? L : W - R + LEG_GAP;
    series.forEach(function (sr) { var col = sr.color, pts = sr.pts.slice().sort(function (a, b) { return a.x - b.x; }); var cl = function (v) { return Math.min(ymax, Math.max(ymin, v)); };
      s += ciSVG(pts, col, xs, y, function (v) { return v; }, cl);
      s += '<polyline fill="none" stroke="' + col + '" stroke-width="2"' + dashOf(sr) + ' points="' + pts.map(function (p) { return xs(p.x).toFixed(1) + "," + y(cl(p.v)).toFixed(1); }).join(" ") + '"/>';
      pts.forEach(function (p) { s += '<circle cx="' + xs(p.x).toFixed(1) + '" cy="' + y(cl(p.v)).toFixed(1) + '" r="3.2" fill="' + (p.hollow ? "var(--surface)" : col) + '" stroke="' + col + '" stroke-width="1.5"><title>' + esc(p.title) + '</title></circle>'; });
      s += '<line x1="' + lx + '" y1="' + ly + '" x2="' + (lx + 20) + '" y2="' + ly + '" stroke="' + col + '" stroke-width="3"' + dashOf(sr) + '/><text x="' + (lx + 26) + '" y="' + (ly + 4) + '" class="leg">' + esc(sr.label) + '</text>'; ly += 20; });
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
    var nr = narrow(), W = opts.width || hostWidth(), L = 76, T = opts.title ? 52 : 18;
    var R = nr ? 18 : legendRight(opts.series.map(function (sr) { return sr.label; }));
    var B = nr ? 60 + 20 * Math.max(1, opts.series.length) : 58, H = plotHeight(W) + B;
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
    opts.series.forEach(function (sr) {
      var col = sr.color;
      s += ciSVG(sr.pts, col, xs, y, clx, cly);
      s += '<polyline fill="none" stroke="' + col + '" stroke-width="1.6" stroke-opacity="0.65"' + dashOf(sr) + ' points="' + sr.pts.map(function (p) { return xs(clx(p.x)).toFixed(1) + "," + y(cly(p.v)).toFixed(1); }).join(" ") + '"/>';
      sr.pts.forEach(function (p) { s += '<circle cx="' + xs(clx(p.x)).toFixed(1) + '" cy="' + y(cly(p.v)).toFixed(1) + '" r="3.2" fill="' + (p.hollow ? "var(--surface)" : col) + '" stroke="' + col + '" stroke-width="1.5"><title>' + esc(p.title) + "</title></circle>"; });
      s += '<line x1="' + lx + '" y1="' + ly + '" x2="' + (lx + 20) + '" y2="' + ly + '" stroke="' + col + '" stroke-width="3"' + dashOf(sr) + '/><text x="' + (lx + 26) + '" y="' + (ly + 4) + '" class="leg">' + esc(sr.label) + "</text>"; ly += 20;
    });
    return s + "</svg>";
  }
  function frontChart(xm, ym, shown, title, aria, xlabel, ylabel) {
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
        pts.push({ x: sx.v, xlo: sx.lo, xhi: sx.hi, v: sy.v, lo: sy.lo, hi: sy.hi, hollow: thin(sx) || thin(sy),
          title: m.label + " @ " + r + ": " + fmt(xm, sx.v, sx.edge) + " " + xm.short + ", " + fmt(ym, sy.v, sy.edge) + " " + ym.short + ", n = " + sy.n + (least.share < 1 ? ", " + shareText(least) : "") });
        [sx.v, anyCI() ? sx.lo : sx.v, anyCI() ? sx.hi : sx.v].forEach(function (v) { if (isFinite(v)) { xmin = Math.min(xmin, v); xmax = Math.max(xmax, v); } });
        [sy.v, anyCI() ? sy.lo : sy.v, anyCI() ? sy.hi : sy.v].forEach(function (v) { if (isFinite(v)) { ymin = Math.min(ymin, v); ymax = Math.max(ymax, v); } });
      });
      if (pts.length) { series.push({ label: m.label + (m.local ? " (local)" : ""), color: colorOf(m), dash: !!m.dash, pts: pts }); }
    });
    if (title == null) { title = xm.short + " vs " + ym.short; }
    if (pending && !series.length) { return frontSVG({ title: title, aria: aria, series: [], empty: "loading the distributions\u2026" }); }
    if (!series.length) { return frontSVG({ title: title, aria: aria, series: [], empty: "no budget has finished for this selection yet" }); }
    var xpad = (xmax - xmin) * 0.06 || 0.3, ypad = (ymax - ymin) * 0.06 || 0.3;
    if (nearRange(xmin, xmax, 0, 0.35)) { xmin = Math.min(xmin, 0); xmax = Math.max(xmax, 0); }   // the problem's own length, when it is near
    xmin -= xpad; xmax += xpad; ymin -= ypad; ymax += ypad;
    return frontSVG({ title: title, aria: aria, series: series, xmin: xmin, xmax: xmax, ymin: ymin, ymax: ymax,
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
  function curveChart(metric, shown, title, aria) {
    var keys = shown.map(function (x) { return x.key; }), series = [], ymin = Infinity, ymax = -Infinity, pending = false, tmin = Infinity, tmax = -Infinity;
    var src = timeSource(keys);
    shown.forEach(function (m) { var pts = [];
      D.rungs.forEach(function (r) { var use = poolCats(m.key, r, keys); if (!use.length) { return; } var x = xOf(m.key, r, use, src); if (x === null) { return; }
        var st = stat(metric, m.key, r, use); if (!st) { return; } if (st.pending) { pending = true; return; } if (!isFinite(st.v)) { return; }
        var title = m.label + " at budget " + r + (state.xaxis === "time" ? " (" + x.toFixed(2) + " s per problem)" : "") + ": " + fmt(metric, st.v, st.edge) + " (95 % interval " + fmt(metric, st.lo) + " to " + fmt(metric, st.hi) + "). From " + st.n.toLocaleString() + (st.ndef && st.ndef !== st.n ? " finite values of " + st.ndef.toLocaleString() : " values") + ", over " + laws(use).toLocaleString() + " problems" + (st.d > 1 ? " run " + st.d + " times each" : "") + (st.share < 1 ? "; " + shareText(st) : "") + ".";
        if (thin(st)) { thinDrawn = true; }
        pts.push({ x: x, v: st.v, lo: st.lo, hi: st.hi, hollow: thin(st), title: title });   // a budget has no interval: the candidate count is exact, and the measured time is within a pixel of its mean (0.4-1.1 px, measured)
        [st.v, anyCI() ? st.lo : st.v, anyCI() ? st.hi : st.v].forEach(function (v) { if (isFinite(v)) { ymin = Math.min(ymin, v); ymax = Math.max(ymax, v); } });
        if (state.xaxis === "time") { tmin = Math.min(tmin, x); tmax = Math.max(tmax, x); } });
      if (pts.length) { series.push({ label: m.label + (m.local ? " (local)" : ""), color: colorOf(m), dash: !!m.dash, pts: pts }); } });
    if (title == null) { title = metric.label; }   // the statistic is named once per block, not on every chart
    var ylabel = axisName(metric) + (metric.median_via && state.stat === "mean" ? ", median" : "");
    if (pending && !series.length) { return chartSVG({ title: title, aria: aria, series: [], empty: "loading the distribution…" }); }
    if (!series.length) { return chartSVG({ title: title, aria: aria, series: [], empty: state.cats.length ? (shown.length ? (state.xaxis === "time" ? "none of the methods shown has been timed yet" : "no method has results for this selection yet") : "no method selected") : "no problem set selected" }); }
    var pad = (ymax - ymin) * 0.06 || 0.05;
    if (metric.kind === "rate") { var floor0 = nearRange(ymin, ymax, 0, 0.6); ymin = floor0 ? 0 : Math.max(0, ymin - pad); ymax = Math.min(1, Math.max(ymin + 0.02, ymax + pad)); }
    else { if (tfOf(metric) === "log2" && nearRange(ymin, ymax, 0, 0.35)) { ymin = Math.min(ymin, 0); ymax = Math.max(ymax, 0); } ymin -= pad; ymax += pad; }
    var tr = state.xaxis === "time" ? timeRange(tmin, tmax) : [0, 0];
    return chartSVG({ title: title, aria: aria, series: series, ymin: ymin, ymax: ymax, ticks: ticksFor(metric, ymin, ymax), tick: function (g) { return tickLabel(metric, g); }, ylabel: ylabel, timeAxis: state.xaxis === "time", timeSource: src, tmin: tr[0], tmax: tr[1], zero: tfOf(metric) === "log2" ? 0 : undefined });
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
    var only = axis === "focus" && state.view === "ranks" ? ((ranksOf() || {}).keys || D.rank_keys || []) : null;   // the metrics the release ranked on
    MGROUPS.forEach(function (g) {
      var ms = D.metrics.filter(function (m) { return m.group === g && (!only || only.indexOf(m.key) >= 0); });
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
    thinDrawn = false;
    var cards = inBlock(main, state.plots.length + 1, function () {
      return state.plots.map(function (p, i) { return plotCard(p, i, shown); }).join("");
    });
    if (thinDrawn) { note += '<p class="v2hint v2hollownote">' + term("valid", thinNote()) + ": on the others this metric has no value, because the method returned no usable formula or the value could not be computed.</p>"; }
    return note + '<div class="v2charts">' + cards + add + "</div>";
  }

  // ---- Headline ---------------------------------------------------------------------------------------------------
  // Two charts that are the same for every visitor: what a method recovers, and how long its answer is, against
  // what it costs. They are deliberately not wired to the controls below -- everything adjustable is the explorer.
  var HEADLINE = [
    { key: "numeric_recovery_val", title: function () { return anyTime() ? "Recovery vs time" : "Recovery vs budget"; },
      caption: "Numeric Recovery: the share of problems where the method's formula reproduces the held-out points (the validation set, which the method never saw) almost exactly: its typical (root-mean-square) error is at most 0.035 % of the true values' standard deviation. Up and to the left is better: more problems solved in less time." },
    { x: "mdl_ratio", y: "log10_fvu_val", title: "Fit vs length",
      caption: "Down is a better fit: log10 FVU is the share of the variation in the held-out points that the formula leaves unexplained, on a log scale (-2 means 1 %). Across is the formula's length: the MDL Ratio divides its length in bits by the true formula's, so the dashed line at 1 is the true length, left of it shorter and right of it longer. Best is low and close to the line. Both are averaged over the problems where the method's formula has a value; an exact fit counts at -15.65, the precision of a 64-bit float." }];
  function hlText(v) { return typeof v === "function" ? v() : v; }
  function withState(over, fn) { var prev = state; state = Object.assign({}, prev, over); try { return fn(); } finally { state = prev; } }
  function renderHeadline() {
    if (!headRoot) { return; }
    headRoot.innerHTML = inBlock(headRoot, HEADLINE.length, function () { return withState(
      { cats: CATS.slice(), methods: D.methods.filter(withData).map(function (m) { return m.key; }),
        stat: "mean", band: true, cross: false, xaxis: anyTime() ? "time" : "rung", valid: VALID_DEFAULT, impute: true },
      function () {
        var shown = shownMethods();
        if (!shown.length) { return '<p class="v2hint">No method has finished a budget in this release yet.</p>'; }
        var drawn = axisMethods(shown), off = offAxis(shown);
        var keys = drawn.map(function (m) { return m.key; }), src = timeSource(keys);
        thinDrawn = false;
        var charts = HEADLINE.map(function (h) {
          var ms = (h.key ? [h.key] : [h.x, h.y]).map(function (k) { return METRIC[k]; });
          if (ms.some(function (m) { return !m; })) { return ""; }
          ms.forEach(function (m) { if (needsHist(m)) { ensureHists(m); } });
          var svg = h.key ? curveChart(ms[0], drawn, hlText(h.title)) : frontChart(ms[0], ms[1], shown, hlText(h.title));
          return '<figure class="v2hlfig">' + svg + "<figcaption>" + esc(hlText(h.caption)) + "</figcaption></figure>";
        }).join("");
        return '<h2 class="v2hltitle">Accuracy, cost and formula length</h2>' +
          '<p class="v2hlsub">Each point is one method at one ' + term("rungs", "budget") + ', the ' + (state.stat === "mean" ? "mean" : "median") + ' over both runs of all ' + laws(CATS).toLocaleString() + ' problems of the ' + CATS.length + ' problem sets (' + (CATS.length ? CAT[CATS[0]].laws.toLocaleString() + ' of them from one problem set, ' + esc(CATS[0]) : '') + '), with a ' + term("wilson", "95 % interval") + '. ' + term("complete", "Why do some methods have fewer points?") + '</p>' +
          '<div class="v2hlcharts">' + charts + "</div>" +
          '<p class="v2hint">' + (src === "ref" ? term("time", "Time is measured on one workstation for every method") +
              (off.length ? ". " + esc(off.map(function (m) { return m.label; }).join(", ")) + (off.length > 1 ? " have" : " has") + " not been timed yet, so " +
                (off.length > 1 ? "they are" : "it is") + " not in the first chart" : "")
            : "The x-axis is the budget per problem. " + term("time", "A time axis appears once a method shown has been timed") +
              ": seconds measured wherever a unit happened to run are not comparable between methods, so this release does not publish them") + "." +
            (thinDrawn ? " " + term("valid", thinNote()) + "." : "") + "</p>";
      }); });
  }

  // ---- Table -----------------------------------------------------------------------------------------------------
  var lastTable = null;
  function cellText(st, p) { if (!st) { return ""; } if (st.pending) { return "…"; } if (thin(st)) { thinDrawn = true; } return (thin(st) ? '<span title="' + esc(shareText(st)) + '">' + THIN_MARK + "</span>" : "") + fmt(p, st.v, st.edge) + (anyCI() ? ' <span class="v2ci-txt">[' + fmt(p, st.lo) + ", " + fmt(p, st.hi) + "]</span>" : ""); }
  function renderTable(shown) {
    var plots = plotMetrics().map(function (k) { return METRIC[k]; }); var keys = shown.map(function (m) { return m.key; });
    if (!plots.length || !shown.length) { return '<p class="v2hint">Select at least one method and one metric.</p>'; }
    var head = '<tr><th>' + (state.rows === "cats" ? "problem set" : "budget") + '</th><th>problems</th>' + shown.map(function (m) { return '<th colspan="' + plots.length + '"><span class="v2sw" style="background:' + colorOf(m) + '"></span>' + esc(m.label) + '</th>'; }).join("") + '</tr>' +
      '<tr><th></th><th></th>' + shown.map(function () { return plots.map(function (p) { return '<th>' + esc(mname(p)) + " " + mhelp(p) + '</th>'; }).join(""); }).join("") + '</tr>';
    var body = "", rowsOut = []; thinDrawn = false;
    var emit = function (label, nl, tds) { body += '<tr><td>' + esc(label) + '</td><td>' + esc(nl) + '</td>' + tds.map(function (x) { return '<td>' + x.t + '</td>'; }).join("") + '</tr>'; rowsOut.push([label, nl].concat(tds.map(function (x) { return x.raw; }))); };
    if (state.rows === "rungs") {
      D.rungs.forEach(function (r) { var cells = shown.map(function (m) { return { m: m, use: poolCats(m.key, r, keys) }; }); if (!cells.some(function (c) { return c.use.length; })) { return; }
        var ref = cells.filter(function (c) { return c.use.length; })[0]; var nl = laws(ref.use).toLocaleString() + " (" + ref.use.length + " problem sets)";
        var tds = []; cells.forEach(function (c) { plots.forEach(function (p) { if (!c.use.length) { tds.push({ t: "", raw: "" }); return; } var st = stat(p, c.m.key, r, c.use); tds.push({ t: cellText(st, p), raw: st && !st.pending ? fmt(p, st.v, st.edge) : "" }); }); });
        emit(String(r), nl, tds); });
    } else {
      var r = state.rung;
      state.cats.slice().sort(function (a, b) { return CAT[b].laws - CAT[a].laws; }).forEach(function (c) { if (!shown.some(function (m) { return cell(m.key, c, r); })) { return; }
        var tds = []; shown.forEach(function (m) { plots.forEach(function (p) { if (!cell(m.key, c, r)) { tds.push({ t: "", raw: "" }); return; } var st = stat(p, m.key, r, [c]); tds.push({ t: cellText(st, p), raw: st && !st.pending ? fmt(p, st.v, st.edge) : "" }); }); });
        emit(c, String(CAT[c].laws), tds); });
      var tds2 = []; shown.forEach(function (m) { plots.forEach(function (p) { var use = poolCats(m.key, r); var st = use.length ? stat(p, m.key, r, use) : null; tds2.push({ t: cellText(st, p), raw: st && !st.pending ? fmt(p, st.v, st.edge) : "" }); }); });
      body += '<tr class="v2total"><td>all selected</td><td>' + laws(state.cats) + '</td>' + tds2.map(function (x) { return '<td>' + x.t + '</td>'; }).join("") + '</tr>';
      rowsOut.push(["all selected", String(laws(state.cats))].concat(tds2.map(function (x) { return x.raw; })));
    }
    lastTable = { header: [state.rows === "cats" ? "problem set" : "budget", "problems"].concat(shown.reduce(function (a, m) { return a.concat(plots.map(function (p) { return m.label + " · " + mname(p); })); }, [])), rows: rowsOut };
    var ctl = '<div class="v2row v2tablectl"><span class="v2lab">rows</span><label><input type="radio" name="v2rows" value="rungs"' + (state.rows === "rungs" ? " checked" : "") + '> every budget</label><label><input type="radio" name="v2rows" value="cats"' + (state.rows === "cats" ? " checked" : "") + '> problem sets at one budget</label>' + (state.rows === "cats" ? rungStepper(shown) : "") +
      '<span class="v2spacer"></span><button type="button" class="v2btn" data-act="copy-tsv">copy as TSV</button><button type="button" class="v2btn" data-act="csv">download CSV</button></div>';
    return ctl + '<div class="v2table-wrap"><table class="v2table"><thead>' + head + '</thead><tbody>' + body + '</tbody></table></div><p class="v2hint">' + (anyCI() ? "Brackets: the " + term("wilson", "95 % interval") + ". " : "") + term("regime", "How problems without a usable formula count") + ". " + (thinDrawn ? term("valid", "\u25cb marks a number based on fewer than " + state.valid + " % of the problems") + ". " : "") + term("complete", "A value appears once a method has results for all selected problem sets at that budget") + ".</p>";
  }

  // ---- Catalog matrix --------------------------------------------------------------------------------------------
  function accentRGB() { var v = (getComputedStyle(document.documentElement).getPropertyValue("--accent") || "#4f46e5").trim(); var m = v.match(/^#([0-9a-f]{6})$/i); if (!m) { return [79, 70, 229]; } return [parseInt(m[1].slice(0, 2), 16), parseInt(m[1].slice(2, 4), 16), parseInt(m[1].slice(4, 6), 16)]; }
  function renderMatrix(shown) {
    var p = METRIC[state.focus], r = state.rung, rgb = accentRGB();
    if (!shown.length) { return '<p class="v2hint">Select at least one method.</p>'; }
    var bar = '<div class="v2viewbar"><span class="v2segwrap"><span class="v2lab">metric</span>' + pickButton("v2viewpick", 'data-axis="focus" aria-label="metric shown in the matrix"', p.key) + " " + mhelp(p) + "</span>" + rungStepper(shown) + "</div>";
    var asked = shown; shown = withAt(shown, r);
    var cats = state.cats.slice().sort(function (a, b) { return CAT[b].laws - CAT[a].laws; }).filter(function (c) { return shown.some(function (m) { return cell(m.key, c, r); }); });
    if (!cats.length) { return bar + '<p class="v2hint">Nothing finished at budget ' + r + " for this selection: step to another budget above.</p>"; }
    var vals = {}, all = [], pending = false; thinDrawn = false;
    cats.forEach(function (c) { vals[c] = {}; shown.forEach(function (m) { if (!cell(m.key, c, r)) { return; } var st = stat(p, m.key, r, [c]); if (st && st.pending) { pending = true; return; } if (st && isFinite(st.v)) { vals[c][m.key] = st; all.push(st.v); } }); });
    if (pending && !all.length) { return bar + '<p class="v2hint">Loading the distribution…</p>'; }
    var lo = Math.min.apply(null, all), hi = Math.max.apply(null, all), ideal = p.ideal === undefined ? (tfOf(p) === "log2" ? 0 : 1) : tfOf(p) === "log2" ? Math.log2(p.ideal) : p.ideal;
    var score = function (v) { if (!(hi > lo)) { return 0.5; } if (p.higher === null) { var dm = Math.max(Math.abs(lo - ideal), Math.abs(hi - ideal)); return dm ? 1 - Math.abs(v - ideal) / dm : 1; } var t = (v - lo) / (hi - lo); return p.higher ? t : 1 - t; };
    var h = '<div class="v2table-wrap"><table class="v2table v2matrix"><thead><tr><th>problem set</th><th>problems</th>' + shown.map(function (m) { return '<th><span class="v2sw" style="background:' + colorOf(m) + '"></span>' + esc(m.label) + '</th>'; }).join("") + '</tr></thead><tbody>';
    cats.forEach(function (c) { h += '<tr><td>' + esc(c) + ' <span class="v2hint">' + GROUPS[CAT[c].group] + '</span></td><td>' + CAT[c].laws + '</td>' + shown.map(function (m) { var st = vals[c][m.key]; if (!st) { return '<td class="v2na">' + (cell(m.key, c, r) ? "…" : "") + '</td>'; } var a = 0.06 + 0.5 * score(st.v); return '<td style="background:rgba(' + rgb.join(",") + "," + a.toFixed(2) + ')" title="' + esc(fmt(p, st.lo) + " to " + fmt(p, st.hi) + ", n = " + st.n + (st.share < 1 ? ", " + shareText(st) : "")) + '">' + (thin(st) ? (thinDrawn = true, THIN_MARK) : "") + fmt(p, st.v, st.edge) + '</td>'; }).join("") + '</tr>'; });
    var pooled = shown.map(function (m) { var use = poolCats(m.key, r); var st = use.length ? stat(p, m.key, r, use) : null; return '<td>' + (st && !st.pending ? cellText(st, p) : "") + '</td>'; }).join("");
    h += '<tr class="v2total"><td>all selected ' + help(TERMS.complete, "When is this row filled in?") + '</td><td>' + laws(state.cats).toLocaleString() + '</td>' + pooled + '</tr></tbody></table></div>';
    var thinHint = thinDrawn ? '<p class="v2hint v2hollownote">' + term("valid", "\u25cb marks a number based on fewer than " + state.valid + " % of the problems") + ".</p>" : "";
    return bar + missingNote(asked, shown, "any selected problem set at budget " + r, r) + thinHint + '<p class="v2hint">' + esc(p.label) + " at budget " + r + ", one square per problem set; darker is better" + (p.higher === null && p.ideal !== undefined ? " (closer to " + p.ideal + ")" : "") + ". " + (p.kind === "cont" ? (statOf(p) === "mean" ? term("mean", "Means") : term("median", "Medians")) + (p.worst !== undefined && state.impute ? " over all problems; a problem without a usable formula counts as " + p.worst + "." : " over the problems where the method returned a usable formula.") : term("regime", "Every problem counts; a problem without a usable formula is a miss") + ".") + "</p>" + h;
  }

  // ---- controls that live on the display itself -----------------------------------------------------------------
  // The sidebar holds what every display shares. What only ONE display can use -- which budget a snapshot is taken
  // at, how a distribution is drawn -- sits on that display, next to what it changes.
  function seg(key, cur, opts, labelHtml, aria) {
    return '<span class="v2segwrap">' + (labelHtml ? '<span class="v2lab">' + labelHtml + "</span>" : "") + '<span class="v2seg" role="group" aria-label="' + esc(aria) + '">' +
      opts.map(function (o) { return '<button type="button" class="v2segbtn' + (o[0] === cur ? " on" : "") + '" data-set="' + key + ":" + o[0] + '" aria-pressed="' + (o[0] === cur ? "true" : "false") + '"' + (o[2] ? ' title="' + esc(o[2]) + '"' : "") + ">" + esc(o[1]) + "</button>"; }).join("") + "</span></span>";
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
  function rungStepper(shown, whole) { return stepper("rung", state.rung, rungsWith(shown, whole), function (r) { return String(r); }, "budget " + help(TERMS.rungs, "What is a budget?"), "budget per problem"); }
  function bestRung(shown) {   // the largest budget that the most methods have finished
    var best = null, top = 0;
    D.rungs.forEach(function (r) { var n = shown.filter(function (m) { return poolCats(m.key, r).length; }).length; if (n && n >= top) { top = n; best = r; } });
    return best;
  }
  function withAt(shown, r) { return shown.filter(function (m) { return state.cats.some(function (c) { return cell(m.key, c, r); }); }); }
  // a method missing at budget r either never runs there (outside the budgets its plan holds) or has not finished yet
  function notRun(m, r) { return !!(m.budgets && r !== undefined && m.budgets.indexOf(+r) < 0); }
  function budgetRange(m) { return m.budgets.length > 1 ? "budgets " + m.budgets[0] + " to " + m.budgets[m.budgets.length - 1] : "budget " + m.budgets[0]; }
  function missingNote(shown, present, what, r) {
    var gone = shown.filter(function (m) { return present.indexOf(m) < 0; });
    var never = gone.filter(function (m) { return notRun(m, r); }), open = gone.filter(function (m) { return !notRun(m, r); });
    var names = function (ms) { return esc(ms.map(function (m) { return m.label; }).join(", ")); };
    return (never.length ? '<p class="v2hint">' + never.map(function (m) { return esc(m.label) + " is not run at budget " + r + " (it runs at " + budgetRange(m) + ")."; }).join(" ") + "</p>" : "") +
      (open.length ? '<p class="v2hint">' + names(open) + (open.length > 1 ? " have" : " has") + " not finished " + what + ".</p>" : "");
  }

  // ---- Distribution ----------------------------------------------------------------------------------------------
  // Four readings of the same pooled histograms: one histogram per method, the cumulative curves of all of them on
  // one axis, a box per catalog, and the quartiles along the ladder. A rate has no distribution over laws (every
  // problem is a hit or a miss), so a rate shows how it is spread over the catalogs instead.
  var DMODES = [["hist", "histograms", "One histogram per method, on a shared axis"], ["ecdf", "cumulative", "The share of problems at or below each value, all methods on one axis"],
    ["cats", "by problem set", "A box per problem set and method: where the middle half and the middle 90 % of the problems lie"], ["rungs", "along the budgets", "The median and the middle half of the problems at every budget"]];
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
      (cont && state.dmode === "rungs" ? "" : rungStepper(shown, cont && state.dmode !== "cats")) +
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
  function renderDistRate(shown, p, r) {   // per-catalog rates: a dot plot with Wilson intervals
    var present = withAt(shown, r), nr = narrow(), W = wideWidth(), T = 46, R = 16;
    var cats = state.cats.slice().sort(function (a, b) { return CAT[b].laws - CAT[a].laws; }).filter(function (c) { return present.some(function (m) { return cell(m.key, c, r); }); });
    var swap = '<p class="v2hint">A rate is a hit or a miss on each problem, so a histogram over problems would have only two bars. This view shows how the rate varies between problem sets instead. ' + '<button type="button" class="v2btn" data-set="dmetric:log10_fvu_val">show the distribution of log10 FVU instead</button></p>';
    if (!cats.length) { return swap + '<p class="v2hint">Nothing finished at budget ' + r + " for this selection: step to another budget above.</p>"; }
    var rowH = Math.max(20, 7 * present.length + 8), B1 = 44 + 18 * present.length, Hh = T + cats.length * rowH + B1, Lw = nr ? 110 : 150;
    var s = '<svg viewBox="0 0 ' + W + " " + Hh + '" class="v2chart v2dist v2distwide" role="img" aria-label="' + esc(p.label) + ' per problem set"><text x="' + Lw + '" y="' + TITLE_Y + '" class="ct">' + esc(p.label) + " per problem set at budget " + r + "</text>";
    var xs = function (v) { return Lw + v * (W - Lw - R); };
    [0, 0.25, 0.5, 0.75, 1].forEach(function (g) { s += '<line x1="' + xs(g) + '" y1="' + T + '" x2="' + xs(g) + '" y2="' + (Hh - B1) + '" class="grid"/><text x="' + xs(g) + '" y="' + (Hh - B1 + 16) + '" class="tick" text-anchor="middle">' + (100 * g) + "%</text>"; });
    cats.forEach(function (c, i) { var yy = T + (i + 0.5) * rowH; s += '<text x="' + (Lw - 8) + '" y="' + (yy + 4) + '" class="tick" text-anchor="end">' + esc(c) + " · " + CAT[c].laws + "</text>";
      if (i % 2) { s += '<rect x="' + Lw + '" y="' + (yy - rowH / 2) + '" width="' + (W - Lw - R) + '" height="' + rowH + '" class="v2stripe"/>'; }
      present.forEach(function (m, j) { var st = cell(m.key, c, r) ? stat(p, m.key, r, [c]) : null; if (!st) { return; } var yj = yy + (j - (present.length - 1) / 2) * 7, col = colorOf(m);
        s += '<line x1="' + xs(st.lo).toFixed(1) + '" y1="' + yj.toFixed(1) + '" x2="' + xs(st.hi).toFixed(1) + '" y2="' + yj.toFixed(1) + '" stroke="' + col + '" stroke-width="2" stroke-opacity="0.4"/>';
        s += '<circle cx="' + xs(st.v).toFixed(1) + '" cy="' + yj.toFixed(1) + '" r="3.2" fill="' + col + '"><title>' + esc(m.label + " on " + c + ": " + fmt(p, st.v) + ", 95 % interval " + fmt(p, st.lo) + " to " + fmt(p, st.hi) + ", from " + st.n.toLocaleString() + " values") + "</title></circle>"; }); });
    var ly = Hh - B1 + 34; present.forEach(function (m) { s += '<circle cx="' + (Lw + 6) + '" cy="' + ly + '" r="4" fill="' + colorOf(m) + '"/><text x="' + (Lw + 16) + '" y="' + (ly + 4) + '" class="leg">' + esc(m.label + (m.local ? " (local)" : "")) + "</text>"; ly += 18; });
    return swap + s + "</svg>" + '<p class="v2hint">One row per problem set, largest first; one dot per method with its ' + term("wilson", "95 % interval") + ".</p>" + missingNote(shown, present, "any selected problem set at budget " + r, r);
  }
  function renderDist(shown) {
    var p = METRIC[state.dmetric], r = state.rung, keys = shown.map(function (m) { return m.key; });
    if (!shown.length) { return '<p class="v2hint">Select at least one method.</p>'; }
    var head = distHead(shown, p);
    if (p.kind === "rate") { return head + renderDistRate(shown, p, r); }
    if (!ready("hist/" + p.key + ".js")) { ensure("hist/" + p.key + ".js", scheduleRender); return head + '<p class="v2hint">Loading the distribution…</p>'; }
    if (!histOf(p.key)) { return head + '<p class="v2hint">This release holds no distribution for ' + esc(p.label) + ".</p>"; }
    if (state.dmode === "rungs") { return head + renderDistLadder(shown, p); }
    var series = [];
    shown.forEach(function (m) { var use = state.dmode === "cats" ? state.cats.filter(function (c) { return cell(m.key, c, r); }) : poolCats(m.key, r, keys); if (!use.length) { return; }
      var ph = pooledHist(p.key, m.key, r, use); if (!ph) { return; }
      var rows = use.reduce(function (a, c) { return a + cell(m.key, c, r).n; }, 0); series.push({ m: m, ph: ph, f: five(ph), use: use, laws: laws(use), rows: rows }); });
    var gone = missingNote(shown, series.map(function (sr) { return sr.m; }), state.dmode === "cats" ? "any selected problem set at budget " + r : "all selected problem sets at budget " + r + ", so " + term("complete", "its distribution is not shown yet") + " (the problem sets it has finished are under \u201cby problem set\u201d)", r);
    if (!series.length) { return head + '<p class="v2hint">No selected method has finished ' + (state.dmode === "cats" ? "any selected problem set" : "all selected problem sets") + " at budget " + r + " yet. Choose another budget above, or select fewer problem sets.</p>" + gone; }
    var pooled = state.dmode !== "cats" ? " The problems are those of the selected problem sets (" + series[0].laws.toLocaleString() + "), the same for every method; the count under each histogram says on how many runs the method's formula has a value." : "";
    var body = state.dmode === "ecdf" ? distEcdf(series, p, r) : state.dmode === "cats" ? distCats(series, p, r) : distHists(series, p, r);
    return head + body + '<p class="v2hint">' + (state.dmode === "cats" ? "" : (state.dmode === "ecdf" && state.dnorm === "all" ? "" : (p.worst !== undefined && state.impute ? "All problems: a problem without a usable formula counts as " + p.worst + "." : "Only problems where the method's formula has a value are counted.")) + pooled + " ") + term("median", "Medians are read from histograms with 128 bins") + "; values outside the shown range, including plus infinity, are counted in the outermost bins.</p>" + gone;
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
    var s = '<svg viewBox="0 0 ' + W + " " + H + '" class="v2chart v2distwide" role="img" aria-label="' + esc(p.label) + ' distribution, one histogram per method"><text x="' + L + '" y="' + TITLE_Y + '" class="ct">' + esc(p.label) + " at budget " + r + "</text>";
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
      var name = sr.m.label + (sr.m.local ? " (local)" : ""), cap = "median " + fmt(p, sr.f[2]) + " · n = " + sr.ph.n.toLocaleString() + " of " + sr.rows.toLocaleString() + " problems";
      if (nr) { s += '<rect x="' + L + '" y="' + (yb + boxH + 9) + '" width="10" height="10" rx="2" fill="' + col + '"/><text x="' + (L + 15) + '" y="' + (yb + boxH + 18) + '" class="leg">' + esc(name) + '</text><text x="' + L + '" y="' + (yb + boxH + 33) + '" class="tick">' + esc(cap) + "</text>"; }
      else { s += '<rect x="10" y="' + (y0 + 8) + '" width="10" height="10" rx="2" fill="' + col + '"/><text x="25" y="' + (y0 + 17) + '" class="leg">' + esc(name) + '</text><text x="10" y="' + (y0 + 36) + '" class="tick">median ' + esc(fmt(p, sr.f[2])) + '</text><text x="10" y="' + (y0 + 52) + '" class="tick">n = ' + sr.ph.n.toLocaleString() + " of " + sr.rows.toLocaleString() + "</text>"; } });
    return s + "</svg>" + '<p class="v2hint">Bar height: the share of the method’s problems in each bin, on the same scale in every panel. A bar too tall for the scale is cut, with its share written beside it. Under each histogram, the line spans the middle 90 % of the problems, the box the middle 50 %, and the tick marks the median.</p>';
  }
  function distEcdf(series, p, r) {
    var nr = narrow(), W = wideWidth(), L = 66, T = 52, B = nr ? 60 + 20 * series.length : 58, H = plotHeight(W) + B;
    var R = nr ? 18 : legendRight(series.map(function (sr) { return sr.m.label + (sr.m.local ? " (local)" : ""); }));
    var vr = viewRange(series.map(function (sr) { return sr.ph; })), all = state.dnorm === "all", low = p.higher === true;   // a problem without a prediction sits at the worse end
    var xs = function (x) { return L + (Math.min(vr.hi, Math.max(vr.lo, x)) - vr.lo) / (vr.hi - vr.lo) * (W - L - R); }, y = function (v) { return T + (1 - v) * (H - T - B); };
    var s = '<svg viewBox="0 0 ' + W + " " + H + '" class="v2chart v2distwide" role="img" aria-label="' + esc(p.label) + ' cumulative distribution"><text x="' + (nr ? 10 : L) + '" y="' + TITLE_Y + '" class="ct">' + esc(nr ? p.short + " @ " + r + ": share at or below" : p.label + " at budget " + r + ": share of " + (all ? "all" : "predicted") + " problems at or below") + "</text>";
    [0, 0.25, 0.5, 0.75, 1].forEach(function (g) { s += '<line x1="' + L + '" y1="' + y(g).toFixed(1) + '" x2="' + (W - R) + '" y2="' + y(g).toFixed(1) + '" class="grid' + (g === 0.5 ? " zero" : "") + '"/><text x="' + (L - 6) + '" y="' + (y(g) + 4).toFixed(1) + '" class="tick" text-anchor="end">' + (100 * g) + "%</text>"; });
    s += xAxisSVG(p, vr, xs, T, H - B, W, L, R);
    var ly = nr ? H - B + 50 : T + 6, lx = nr ? L : W - R + LEG_GAP;
    series.forEach(function (sr) { var col = colorOf(sr.m), den = all ? sr.rows : sr.ph.n, cum = all && low ? sr.rows - sr.ph.n : 0, pts = [];
      for (var b = 0; b < sr.ph.nb; b++) { var before = cum; cum += sr.ph.h[b]; if (b < vr.b0) { continue; } if (b > vr.b1) { break; } var x0 = vr.lo + (b - vr.b0) * vr.w; if (!pts.length) { pts.push(xs(x0).toFixed(1) + "," + y(before / den).toFixed(1)); } pts.push(xs(x0 + vr.w).toFixed(1) + "," + y(before / den).toFixed(1), xs(x0 + vr.w).toFixed(1) + "," + y(cum / den).toFixed(1)); }
      s += '<polyline fill="none" stroke="' + col + '" stroke-width="2"' + dashOf(sr.m) + ' points="' + pts.join(" ") + '"><title>' + esc(sr.m.label + ": " + fiveText(p, sr.f) + "; " + sr.ph.n + " of " + sr.rows + " problems have a usable formula") + "</title></polyline>";
      s += '<line x1="' + lx + '" y1="' + ly + '" x2="' + (lx + 20) + '" y2="' + ly + '" stroke="' + col + '" stroke-width="3"' + dashOf(sr.m) + '/><text x="' + (lx + 26) + '" y="' + (ly + 4) + '" class="leg">' + esc(sr.m.label + (sr.m.local ? " (local)" : "")) + "</text>"; ly += 20; });
    return s + "</svg>" + '<p class="v2hint">' + (p.higher === true ? "Further right is better: a curve that stays low longer holds more of its problems at high values." : p.higher === false ? "Further left is better: a curve that rises early holds more of its problems at low values." : p.ideal !== undefined ? "Closer to " + p.ideal + " is better: a curve that rises steeply around " + p.ideal + " is the tighter one." : "") +
      (all ? " Out of all problems, a method that leaves problems without a prediction " + (low ? "starts above 0 %." : "ends below 100 %.") : "") + "</p>";
  }
  function distCats(series, p, r) {
    var nr = narrow(), W = wideWidth(), T = 52, R = 18, Lw = nr ? 110 : 160, per = {}, phs = [];
    var cats = state.cats.slice().sort(function (a, b) { return CAT[b].laws - CAT[a].laws; });
    cats.forEach(function (c) { series.forEach(function (sr) { if (sr.use.indexOf(c) < 0) { return; } var ph = pooledHist(p.key, sr.m.key, r, [c]); if (!ph) { return; } (per[c] = per[c] || {})[sr.m.key] = ph; phs.push(ph); }); });
    cats = cats.filter(function (c) { return per[c]; });
    if (!phs.length) { return '<p class="v2hint">No selected problem set has a finite value at budget ' + r + ".</p>"; }
    var vr = viewRange(phs), rowH = Math.max(22, 11 * series.length + 8), B = 44 + 18 * series.length, H = T + (cats.length + 1) * rowH + B;
    var xs = function (x) { return Lw + (Math.min(vr.hi, Math.max(vr.lo, x)) - vr.lo) / (vr.hi - vr.lo) * (W - Lw - R); };
    var s = '<svg viewBox="0 0 ' + W + " " + H + '" class="v2chart v2distwide" role="img" aria-label="' + esc(p.label) + ' per problem set"><text x="' + Lw + '" y="' + TITLE_Y + '" class="ct">' + esc(p.label) + " per problem set at budget " + r + "</text>";
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
      if (pts.length) { series.push({ label: m.label + (m.local ? " (local)" : ""), color: colorOf(m), dash: !!m.dash, pts: pts }); } });
    var title = narrow() ? p.short + ": median, middle half" : p.label + ": median and middle half, by budget", off = offAxis(shown);
    var note = off.length ? '<p class="v2hint">' + esc(off.map(function (m) { return m.label; }).join(", ")) + (off.length > 1 ? " have" : " has") + " no place on this axis; choose the other x-axis above.</p>" : "";
    if (!series.length) { return chartSVG({ title: title, aria: title, series: [], empty: "nothing to draw on this axis yet", width: wideWidth() }) + note; }
    var pad = (ymax - ymin) * 0.08 || 0.1; ymin -= pad; ymax += pad; var tr = state.xaxis === "time" ? timeRange(tmin, tmax) : [0, 0];
    var svg = withState({ band: true, cross: false }, function () { return chartSVG({ title: title, aria: title, width: wideWidth(), series: series, ymin: ymin, ymax: ymax, ticks: ticksFor(p, ymin, ymax), tick: function (g) { return tickLabel(p, g); }, ylabel: axisName(p), timeAxis: state.xaxis === "time", timeSource: src, tmin: tr[0], tmax: tr[1], zero: tfOf(p) === "log2" ? 0 : undefined }); });
    return '<div class="v2charts v2one">' + svg + "</div>" + '<p class="v2hint">Line: the median over the problems with a usable formula. Band: the middle 50 % of those problems; it shows their spread, not the uncertainty of the median. Tap or hover a point for more percentiles.</p>' + note;
  }

  // ---- Paired ----------------------------------------------------------------------------------------------------
  function lgamma(x) { var g = 7, c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]; if (x < 0.5) { return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x); } x -= 1; var a = c[0], t = x + g + 0.5; for (var i = 1; i < g + 2; i++) { a += c[i] / (x + i); } return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a); }
  function binomTwoSided(k, n) { if (!n) { return null; } var kk = Math.min(k, n - k), s = 0; for (var i = 0; i <= kk; i++) { s += Math.exp(lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1) - n * LN2); } return Math.min(1, 2 * s); }
  function fmtP(p) { if (p === null || p === undefined) { return "–"; } return p < 0.001 ? "< 0.001" : p.toFixed(3); }
  function pairedStat(metric, a, b, r, cs) {
    var Pd = pairedOf(); if (!Pd) { return null; } var key = a + "|" + b, flip = false; if (!Pd[key]) { key = b + "|" + a; flip = true; } if (!Pd[key]) { return null; }
    var x = [0, 0, 0, 0, 0], any = false;
    var pk = metric.key + (leftOut(metric.key) ? "@answered" : "");
    cs.forEach(function (c) { var pc = Pd[key][c] && Pd[key][c][String(r)]; if (!pc || !pc.m[pk]) { return; } any = true; var t = pc.m[pk]; for (var i = 0; i < t.length; i++) { x[i] += t[i]; } });
    if (!any) { return null; }
    if (metric.kind === "rate") { var n10 = flip ? x[2] : x[1], n01 = flip ? x[1] : x[2], N = x[0] + x[1] + x[2] + x[3]; if (!N) { return null; } var d = (n10 - n01) / N, se = Math.sqrt(Math.max(0, (n10 + n01) - (n10 - n01) * (n10 - n01) / N)) / N; return { v: d, lo: d - Z * se, hi: d + Z * se, n: N, p: binomTwoSided(n10, n10 + n01), wins: n10, losses: n01 }; }
    if (!x[0]) { return null; }
    var m = x[1] / x[0], varr = x[0] > 1 ? Math.max(0, (x[2] - x[0] * m * m) / (x[0] - 1)) : 0, se2 = Math.sqrt(varr / x[0]); if (flip) { m = -m; }
    var wins = flip ? x[4] : x[3], losses = flip ? x[3] : x[4];
    return { v: m, lo: m - Z * se2, hi: m + Z * se2, n: x[0], p: binomTwoSided(wins, wins + losses), wins: wins, losses: losses };
  }
  function fmtDelta(metric, d) { if (!isFinite(d)) { return "–"; } var tf = tfOf(metric); if (metric.kind === "rate") { return (d >= 0 ? "+" : "") + (100 * d).toFixed(1) + " pp"; } if (tf === "log2") { return "× " + Math.pow(2, d).toFixed(2); } if (tf === "log10") { return "× " + Math.pow(10, d).toFixed(2); } return (d >= 0 ? "+" : "") + d.toFixed(metric.fmt === "num3" ? 3 : 2); }
  function bothDone(a, b, r) { return poolCats(a, r).length && poolCats(b, r).length ? state.cats.slice() : []; }
  function renderPaired(shown) {
    if (!ready("paired.js")) { ensure("paired.js", scheduleRender); return '<p class="v2hint">Loading the paired contrasts…</p>'; }
    if (shown.length < 2) { return '<p class="v2hint">Select at least two methods; one of them is the baseline.</p>'; }
    if (!state.base || !shown.some(function (m) { return m.key === state.base; })) { state.base = shown[0].key; }
    var base = D.methods.filter(function (m) { return m.key === state.base; })[0], others = shown.filter(function (m) { return m.key !== state.base; }), keys = shown.map(function (m) { return m.key; });
    var plots = plotMetrics().filter(function (k) { return PAIRED_KEYS.indexOf(k) >= 0; }).map(function (k) { return METRIC[k]; });
    var ctl = '<p class="v2hint">Each method is compared with ' + esc(base.label) + ' problem by problem: every number is the method\u2019s value minus ' + esc(base.label) + '\u2019s. ' + term("draw1", "Both methods are compared on the same data") + '.</p>';
    if (!plots.length) { return ctl + '<p class="v2hint">None of the plotted metrics has paired contrasts. Paired contrasts exist for: ' + esc(PAIRED_KEYS.map(function (k) { return METRIC[k] ? mname(METRIC[k]) : k; }).join(", ")) + '.</p>'; }
    var src = timeSource(axisMethods(others).map(function (m) { return m.key; }));   // the drawn set, not the baseline
    var charts = inBlock(root.querySelector(".v2main"), plots.length, function () { return plots.map(function (p) { var series = [], ymin = Infinity, ymax = -Infinity, tmin = Infinity, tmax = -Infinity;
      axisMethods(others).forEach(function (m) { var pts = []; D.rungs.forEach(function (r) { var use = bothDone(m.key, base.key, r); if (!use.length) { return; } var x = xOf(m.key, r, use, src); if (x === null) { return; } var st = pairedStat(p, m.key, base.key, r, use); if (!st || !isFinite(st.v)) { return; }
          pts.push({ x: x, v: st.v, lo: st.lo, hi: st.hi, title: m.label + " − " + base.label + " @ " + r + ": " + fmtDelta(p, st.v) + " [" + fmtDelta(p, st.lo) + ", " + fmtDelta(p, st.hi) + "], n = " + st.n + " problems, p = " + fmtP(st.p) });
          [st.v, anyCI() ? st.lo : st.v, anyCI() ? st.hi : st.v].forEach(function (v) { if (isFinite(v)) { ymin = Math.min(ymin, v); ymax = Math.max(ymax, v); } }); if (state.xaxis === "time") { tmin = Math.min(tmin, x); tmax = Math.max(tmax, x); } });
        if (pts.length) { series.push({ label: m.label + (m.local ? " (local)" : ""), color: colorOf(m), dash: !!m.dash, pts: pts }); } });
      var title = "Δ " + mname(p) + " vs " + base.label;
      if (!series.length) { return chartSVG({ title: title, aria: title, series: [], empty: "no matched cells with the baseline yet" }); }
      if (nearRange(ymin, ymax, 0, 0.35)) { ymin = Math.min(ymin, 0); ymax = Math.max(ymax, 0); } var pad = (ymax - ymin) * 0.08 || 0.05; ymin -= pad; ymax += pad;
      var tr = state.xaxis === "time" ? timeRange(tmin, tmax) : [0, 0];
      var ptf = tfOf(p), dticks = ptf === "log2" ? logTicks(ymin, ymax, 2) : ptf === "log10" ? logTicks(ymin, ymax, 10) : linearTicks(ymin, ymax);
      var dtick = function (g) { return p.kind === "rate" ? (g > 0 ? "+" : "") + roundNum(100 * g) + " pp" : ptf === "log2" ? "× " + roundNum(Math.pow(2, g)) : ptf === "log10" ? "× " + roundNum(Math.pow(10, g)) : (g > 0 ? "+" : "") + roundNum(g); };
      return chartSVG({ title: title, aria: title, series: series, ymin: ymin, ymax: ymax, ticks: dticks, tick: dtick, ylabel: "", timeAxis: state.xaxis === "time", timeSource: src, tmin: tr[0], tmax: tr[1], zero: 0 }); }); });
    var r = state.rung, rows = "";
    others.forEach(function (m) { rows += '<tr><td><span class="v2sw" style="background:' + colorOf(m) + '"></span>' + esc(m.label) + '</td>' + plots.map(function (p) { var use = bothDone(m.key, base.key, r); var st = use.length ? pairedStat(p, m.key, base.key, r, use) : null; if (!st) { return '<td class="v2na">–</td><td class="v2na">–</td><td class="v2na">–</td>'; } var sig = st.p !== null && st.p < 0.05; return '<td' + (sig ? ' class="v2sig"' : "") + '>' + fmtDelta(p, st.v) + ' <span class="v2ci-txt">[' + fmtDelta(p, st.lo) + ", " + fmtDelta(p, st.hi) + ']</span></td><td>' + fmtP(st.p) + '</td><td class="v2hint">' + st.wins.toLocaleString() + " / " + st.losses.toLocaleString() + " of " + st.n.toLocaleString() + '</td>'; }).join("") + '</tr>'; });
    var anyRow = others.some(function (m) { return plots.some(function (p) { var use = bothDone(m.key, base.key, r); return use.length && pairedStat(p, m.key, base.key, r, use); }); });
    var table = '<h3 class="v2h">At one budget</h3><div class="v2viewbar">' + rungStepper(shown, true) + "</div>" + (anyRow ? "" : '<p class="v2hint">No method has ' + term("complete", "finished all selected problem sets") + " at budget " + r + " together with the baseline yet. Choose another budget above, or select fewer problem sets.</p>") + '<div class="v2table-wrap"><table class="v2table"><thead><tr><th>method − ' + esc(base.label) + ', budget ' + r + '</th>' + plots.map(function (p) { return '<th colspan="3">' + esc(mname(p)) + " " + mhelp(p) + '</th>'; }).join("") + '</tr><tr><th></th>' + plots.map(function () { return '<th>Δ [95 %]</th><th>p</th><th>better / worse</th>'; }).join("") + '</tr></thead><tbody>' + rows + '</tbody></table></div>' +
      '<p class="v2hint">\u0394: the method\u2019s value minus the baseline\u2019s, with its 95 % interval; pp are percentage points, and ratios are compared as factors (\u00d7 0.5 means half). p: the probability of a difference at least this large if both methods were equally good; for rates it comes from ' + term("mcnemar", "the McNemar test") + ', for other metrics from ' + term("signtest", "a sign test on the better and worse counts") + ', not from \u0394. Bold means below 0.05. Better / worse: on how many problems the method does better or worse than the baseline on that metric; for a ratio, only these counts say which is closer to 1. Choose the baseline above. <a href="#paired">How paired comparisons work</a></p>';
    return ctl + '<div class="v2charts">' + charts.join("") + "</div>" + table;
  }

  // ---- Ranks -----------------------------------------------------------------------------------------------------
  // Within every problem the methods are placed 1st, 2nd, ... on one continuous metric; a method without a usable answer
  // is placed last, ties share a place. The release ships PAIRWISE outcomes per catalog (ranks.js), and a mean rank
  // is 1 + sum over opponents of (their wins + half the ties) / problems, so any roster of methods over any set of
  // catalogs is ranked here, on the same problems for every method. A snapshot is taken at one budget: every method at
  // the same rung of its own ladder, or every method at the largest rung the reference machine timed within a time
  // budget. The Friedman omnibus test is computed without the tie correction, which can only understate it; the
  // critical difference is Nemenyi's at 5 %.
  function ranksOf() { var R = window.RESULTS_V2_RANKS && window.RESULTS_V2_RANKS[REL]; return R && R.keys ? R : null; }
  var NEMENYI = [0, 0, 1.960, 2.343, 2.569, 2.728, 2.850, 2.949, 3.031, 3.102, 3.164, 3.219, 3.268, 3.313, 3.354, 3.391, 3.426, 3.458, 3.489, 3.517, 3.544];   // q(0.05, k, inf) / sqrt(2)
  function gammaQ(a, x) {   // regularised upper incomplete gamma: the chi-square tail
    if (!(x > 0)) { return 1; }
    var gl = lgamma(a), i;
    if (x < a + 1) { var ap = a, sum = 1 / a, del = sum; for (i = 0; i < 500; i++) { ap += 1; del *= x / ap; sum += del; if (Math.abs(del) < Math.abs(sum) * 1e-14) { break; } } return Math.max(0, 1 - sum * Math.exp(-x + a * Math.log(x) - gl)); }
    var b = x + 1 - a, c = 1e300, d = 1 / b, h = d;
    for (i = 1; i < 500; i++) { var an = -i * (i - a); b += 2; d = an * d + b; if (Math.abs(d) < 1e-300) { d = 1e-300; } c = b + an / c; if (Math.abs(c) < 1e-300) { c = 1e-300; } d = 1 / d; var dl = d * c; h *= dl; if (Math.abs(dl - 1) < 1e-14) { break; } }
    return Math.min(1, Math.exp(-x + a * Math.log(x) - gl) * h);
  }
  function isTimeSlot(slot) { return String(slot).charAt(0) === "t"; }
  function slotRung(R, key, slot) { if (!isTimeSlot(slot)) { return +slot; } var r = (R.at[key] || {})[slot]; return r === undefined ? null : r; }
  function slotSeconds(R, slot) { return R.seconds[R.budgets.indexOf(slot)]; }
  function slotLabel(R, slot) { return isTimeSlot(slot) ? slotSeconds(R, slot) + " s per problem" : "budget " + slot; }
  // held back by the time limit: a larger rung of its ladder is timed, and takes longer than the limit allows
  function timeBound(key, r, seconds) { var t = D.timing[key] || {}; return Object.keys(t).some(function (r2) { return +r2 > r && t[r2] > seconds; }); }
  function ranking(R, shown, slot, ki) {
    var out = { roster: [], out: [], blind: [], cats: [], n: 0 };
    var cand = [], paired = function (a, b) { return R.pairs[a + "|" + b] || R.pairs[b + "|" + a]; };
    shown.forEach(function (m) { var r = slotRung(R, m.key, slot);   // ranked: a method that has finished every selected catalog at its budget
      if (r !== null && poolCats(m.key, r).length) { cand.push({ m: m, r: r }); } else { out.out.push(m); } });
    cand = cand.filter(function (x) { var ok = cand.length < 2 || cand.some(function (z) { return z !== x && paired(x.m.key, z.m.key); }); if (!ok) { out.blind.push(x.m); } return ok; });   // an overlay sealed before rankings existed
    // An outcome at a time limit compared two rungs. An overlay is sealed less often than the release is refreshed, so
    // its outcomes against a method that has since moved to another rung are stale: they count as missing.
    var fresh = function (a, b) { var g = R.rungs || {}, e = g[a.m.key + "|" + b.m.key], flip = false; if (!e) { e = g[b.m.key + "|" + a.m.key]; flip = true; }
      var q = e && e[String(slot)]; return !isTimeSlot(slot) || !q || (q[flip ? 1 : 0] === a.r && q[flip ? 0 : 1] === b.r); };
    var pair = function (a, b, c) { var e = R.pairs[a + "|" + b], flip = false; if (!e) { e = R.pairs[b + "|" + a]; flip = true; } var t = e && e[c] && e[c][String(slot)]; if (!t) { return null; } var wa = t[1 + 2 * ki], wb = t[2 + 2 * ki]; if (wa == null || wb == null) { return null; } return { n: t[0], wa: flip ? wb : wa, wb: flip ? wa : wb }; };
    var meets = function (x, z, c) { return fresh(x, z) && pair(x.m.key, z.m.key, c); };
    var shared = function (ro) { return state.cats.filter(function (c) { return ro.every(function (x) { return cell(x.m.key, c, x.r); }) && ro.every(function (x, i) { return ro.every(function (z, j) { return j <= i || meets(x, z, c); }); }); }); };
    // Rank the methods that can all be compared with one another here. One without outcomes against another sits out
    // instead of emptying the whole ranking: the one missing the most partners goes first, a release method before
    // one the reader added with a key.
    var ro = cand;
    while (ro.length >= 2 && !shared(ro).length) {
      var lone = ro.map(function (x) { return ro.filter(function (z) { return z !== x && !state.cats.some(function (c) { return meets(x, z, c); }); }).length; });
      var worst = Math.max.apply(null, lone); if (!worst) { break; }
      var drop = -1; ro.forEach(function (x, i) { if (lone[i] === worst && (drop < 0 || (ro[drop].m.overlay && !x.m.overlay) || (!!ro[drop].m.overlay === !!x.m.overlay))) { drop = i; } });
      out.blind.push(ro[drop].m); ro = ro.filter(function (_x, i) { return i !== drop; });
    }
    out.roster = ro;
    var k = ro.length; if (k < 2) { return out; }
    out.cats = shared(ro);
    if (!out.cats.length) { return out; }
    var beat = ro.map(function () { return ro.map(function () { return { w: 0, t: 0, n: 0 }; }); });
    ro.forEach(function (x, i) { ro.forEach(function (z, j) { if (j <= i) { return; } out.cats.forEach(function (c) { var t = pair(x.m.key, z.m.key, c); beat[i][j].w += t.wa; beat[j][i].w += t.wb; beat[i][j].t += t.n - t.wa - t.wb; beat[j][i].t += t.n - t.wa - t.wb; beat[i][j].n += t.n; beat[j][i].n += t.n; }); }); });
    out.n = beat[0][1].n; out.beat = beat; out.k = k;
    var key = R.keys[ki];
    ro.forEach(function (x, i) { var lost = 0; ro.forEach(function (z, j) { if (j !== i) { lost += (beat[j][i].w + 0.5 * beat[j][i].t) / beat[j][i].n; } });
      x.rank = 1 + lost; x.share = 1 - lost / (k - 1);
      x.blank = out.cats.reduce(function (a, c) { var ce = cell(x.m.key, c, x.r), t = ce.m[key]; return a + ce.n - (t ? t[0] : 0); }, 0);
      x.capped = isTimeSlot(slot) && !timeBound(x.m.key, x.r, slotSeconds(R, slot)); });
    var dev = ro.reduce(function (a, x) { return a + Math.pow(x.rank - (k + 1) / 2, 2); }, 0);
    out.chi2 = 12 * out.n / (k * (k + 1)) * dev; out.p = gammaQ((k - 1) / 2, out.chi2 / 2);
    out.cd = (NEMENYI[k] || NEMENYI[NEMENYI.length - 1]) * Math.sqrt(k * (k + 1) / (6 * out.n));
    out.order = ro.slice().sort(function (a, b) { return a.rank - b.rank; });
    out.cliques = [];
    out.order.forEach(function (x, i) { var g = out.order.filter(function (z, j) { return j >= i && z.rank - x.rank <= out.cd; }); if (g.length > 1 && !out.cliques.some(function (q) { return g.every(function (z) { return q.indexOf(z) >= 0; }); })) { out.cliques.push(g); } });
    return out;
  }
  function rankSlots(R) { return state.xaxis === "time" ? R.budgets.filter(function (b) { return D.methods.filter(function (m) { return (R.at[m.key] || {})[b] !== undefined; }).length >= 2; }) : null; }
  function bestBudget(R, shown) {   // the smallest time budget that the most methods can run within
    var best = null, top = 0; R.budgets.forEach(function (b) { var n = shown.filter(function (m) { return slotRung(R, m.key, b) !== null; }).length; if (n > top) { top = n; best = b; } }); return best;
  }
  function renderRanks(shown) {
    if (!ready("ranks.js")) { ensure("ranks.js", scheduleRender); return '<p class="v2hint">Loading the rankings…</p>'; }
    var R = ranksOf(); if (!R) { return '<p class="v2hint">This release holds no rankings yet.</p>'; }
    if (R.keys.indexOf(state.rmetric) < 0) { state.rmetric = R.keys[0]; }
    var timed = state.xaxis === "time" && anyTime(), budgets = timed ? rankSlots(R) : [];
    if (timed && budgets.indexOf(state.tbudget) < 0) { state.tbudget = bestBudget(R, shown) || budgets[0] || null; }
    var slot = timed ? state.tbudget : String(state.rung), ki = R.keys.indexOf(state.rmetric), p = METRIC[state.rmetric];
    var head = '<div class="v2viewbar"><span class="v2segwrap"><span class="v2lab">ranked on</span>' + pickButton("v2viewpick", 'data-axis="focus" aria-label="metric the methods are ranked on"', p.key) + " " + mhelp(p) +
      (ki === 0 ? ' <span class="v2tag v2tag-primary" title="The ranking metric chosen before the results were read. Which time limit or budget to compare at depends on the use case.">primary</span>' : ' <span class="v2tag" title="For browsing; the primary ranking metric is log10 FVU on the held-out points.">exploratory</span>') + "</span>" +
      seg("xaxis", timed ? "time" : "rung", [["time", "the same time", "Each method at its largest finished budget that fits within the time limit per problem"], ["rung", "the same budget", "Each method at the same budget number, in its own unit: a rough comparison, since the units differ"]].filter(function (o) { return o[0] !== "time" || anyTime(); }), "every method at", "what the methods are held equal on") +
      (timed ? stepper("tbudget", state.tbudget, budgets, function (b) { return slotSeconds(R, b) + " s"; }, "time limit " + help(TERMS.tbudget, "How is the time limit applied?"), "time limit per problem") : rungStepper(shown, true)) + "</div>";
    if (shown.length < 2) { return head + '<p class="v2hint">Select at least two methods to rank.</p>'; }
    if (slot === null) { return head + '<p class="v2hint">Fewer than two of the selected methods have been timed yet.</p>'; }
    var lg = ranking(R, shown, slot, ki);
    var why = function (m) {   // one reason per method: never timed, too slow, never run at this budget, or not finished
      if (timed) { return Object.keys(D.timing[m.key] || {}).length ? "none of its finished budgets takes " + slotSeconds(R, slot) + " s or less per problem" : "it has not been timed yet"; }
      return notRun(m, slot) ? "it is not run at budget " + slot + " (it runs at " + budgetRange(m) + ")" : "it has not finished all selected problem sets at budget " + slot;
    };
    var sitOut = (lg.out.length ? '<p class="v2hint">' + lg.out.map(function (m) { return esc(m.label) + " is not ranked: " + why(m) + "."; }).join(" ") + "</p>" : "") +
      (lg.blind.length ? '<p class="v2hint">' + esc(lg.blind.map(function (m) { return m.label; }).join(", ")) + (lg.blind.length > 1 ? " are" : " is") + " not ranked: " + (lg.blind.length > 1 ? "they have" : "it has") + " no results on the same problems as all other selected methods at " + esc(slotLabel(R, slot)) + ".</p>" : "");
    if (lg.roster.length < 2 || !lg.cats.length) { return head + '<p class="v2hint">Fewer than two of the selected methods have ' + term("complete", "finished all selected problem sets") + " at " + esc(slotLabel(R, slot)) + ". Choose another setting above, or select fewer problem sets.</p>" + sitOut; }
    return head + rankDiagram(R, lg, p, slot) + sitOut + rankTables(R, lg, p, slot, timed) + rankLadder(R, shown, p, ki, timed);
  }
  function rankDiagram(R, lg, p, slot) {
    var nr = narrow(), W = wideWidth(), k = lg.k, Rr = nr ? 44 : 56, T = 76, rowH = 30, B = 42, H = T + k * rowH + B;
    var Lw = nr ? 138 : Math.max(190, Math.ceil(widest(lg.order.map(function (x) { return x.m.label + (x.m.local ? " (local)" : ""); })) + 28));
    var xs = function (v) { return Lw + (v - 1) / (k - 1) * (W - Lw - Rr); }, reject = lg.p < 0.05;
    var title = "Average place on " + p.label + " · " + slotLabel(R, slot);
    var s = '<svg viewBox="0 0 ' + W + " " + H + '" class="v2chart v2distwide v2rankchart" role="img" aria-label="' + esc(title) + '"><text x="' + (nr ? 10 : Lw) + '" y="' + TITLE_Y + '" class="ct">' + esc(nr ? "Average place · " + slotLabel(R, slot) : title) + "</text>";
    for (var g = 1; g <= k; g++) { s += '<line x1="' + xs(g).toFixed(1) + '" y1="' + T + '" x2="' + xs(g).toFixed(1) + '" y2="' + (H - B) + '" class="grid"/><text x="' + xs(g).toFixed(1) + '" y="' + (H - B + 16) + '" class="tick" text-anchor="middle">' + g + "</text>"; }
    s += '<text x="' + ((Lw + W - Rr) / 2).toFixed(0) + '" y="' + (H - B + 33) + '" class="tick" text-anchor="middle">average place (1 = best of ' + k + ")</text>";
    var cdw = Math.max(2, xs(1 + lg.cd) - xs(1));
    s += '<g><title>' + esc("Critical difference: " + lg.cd.toFixed(3) + " places") + '</title><line x1="' + xs(1).toFixed(1) + '" y1="46" x2="' + (xs(1) + cdw).toFixed(1) + '" y2="46" class="v2cd"/><line x1="' + xs(1).toFixed(1) + '" y1="41" x2="' + xs(1).toFixed(1) + '" y2="51" class="v2cd"/><line x1="' + (xs(1) + cdw).toFixed(1) + '" y1="41" x2="' + (xs(1) + cdw).toFixed(1) + '" y2="51" class="v2cd"/>' +
      '<text x="' + (xs(1) + cdw + 8).toFixed(1) + '" y="50" class="tick">critical difference ' + lg.cd.toFixed(2) + "</text></g>";
    if (reject) { lg.cliques.forEach(function (q) { var is = q.map(function (x) { return lg.order.indexOf(x); }), rs = q.map(function (x) { return x.rank; });
      s += '<rect x="' + (xs(Math.min.apply(null, rs)) - 9).toFixed(1) + '" y="' + (T + Math.min.apply(null, is) * rowH + 3) + '" width="' + (xs(Math.max.apply(null, rs)) - xs(Math.min.apply(null, rs)) + 18).toFixed(1) + '" height="' + ((Math.max.apply(null, is) - Math.min.apply(null, is) + 1) * rowH - 6) + '" rx="6" class="v2clique"/>'; }); }
    lg.order.forEach(function (x, i) { var yy = T + (i + 0.5) * rowH, col = colorOf(x.m);
      s += '<text x="' + (Lw - 12) + '" y="' + (yy + 4) + '" class="leg" text-anchor="end">' + esc(x.m.label + (x.m.local ? " (local)" : "")) + "</text>";
      s += '<line x1="' + xs(1).toFixed(1) + '" y1="' + yy + '" x2="' + xs(x.rank).toFixed(1) + '" y2="' + yy + '" stroke="' + col + '" stroke-width="2" stroke-opacity="0.3"/>';
      s += '<circle cx="' + xs(x.rank).toFixed(1) + '" cy="' + yy + '" r="6.5" fill="' + (x.capped ? "var(--surface)" : col) + '" stroke="' + col + '" stroke-width="2.5"><title>' + esc(x.m.label + ": average place " + x.rank.toFixed(2) + " of " + k + " at budget " + x.r + (x.capped ? " (its largest budget, still under the time limit)" : "") + "; wins " + (100 * x.share).toFixed(1) + " % of its comparisons; no usable formula on " + x.blank + " problem runs") + "</title></circle>";
      s += '<text x="' + (xs(x.rank) + 12).toFixed(1) + '" y="' + (yy + 4) + '" class="tick">' + x.rank.toFixed(2) + "</text>"; });
    return s + "</svg>" + '<p class="v2hint">' + k + " methods, each placed against the others on " + lg.n.toLocaleString() + " problem runs (each problem is run twice, and each run counts) from " + lg.cats.length + " problem sets; " + term("worstrank", "how places are given") + ". " +
      (reject ? (lg.cliques.length ? "Methods joined by a shaded band are closer than the " + term("cd", "critical difference") + ": the data cannot tell them apart." : "Every gap is larger than the " + term("cd", "critical difference") + ", so no difference in average place is likely to be chance.")
        : "<b>" + term("friedman", "The places do not differ more than chance would make them") + " (p = " + lg.p.toFixed(3) + "), so no groups are drawn.</b>") +
      (lg.order.some(function (x) { return x.capped; }) ? " A hollow dot marks a method whose largest budget stays below the time limit; with more budget it might move up." : "") + ' <a href="#ranks">How ranks work</a></p>';
  }
  function rankTables(R, lg, p, slot, timed) {
    var k = lg.k, ord = lg.order, idx = ord.map(function (x) { return lg.roster.indexOf(x); });
    var t1 = '<h3 class="v2h">Standings</h3><div class="v2table-wrap"><table class="v2table v2ranktable"><thead><tr><th>method</th><th>' + (timed ? "budget within " + slotSeconds(R, slot) + " s" : "budget") + "</th><th>average place</th><th>comparisons won " + help(TERMS.winshare, "What is the share of comparisons won?") + "</th><th>problem runs without a usable formula</th></tr></thead><tbody>" +
      ord.map(function (x) { return '<tr><td><span class="v2sw" style="background:' + colorOf(x.m) + '"></span>' + esc(x.m.label) + "</td><td>" + x.r + (timed && refTime(x.m.key, x.r) ? ' <span class="v2ci-txt">' + refTime(x.m.key, x.r).toFixed(2) + " s</span>" : "") + "</td><td><b>" + x.rank.toFixed(2) + "</b></td><td>" + (100 * x.share).toFixed(1) + " %</td><td>" + x.blank.toLocaleString() + ' <span class="v2ci-txt">of ' + lg.n.toLocaleString() + "</span></td></tr>"; }).join("") + "</tbody></table></div>";
    var t2 = '<h3 class="v2h">Head to head</h3><div class="v2table-wrap"><table class="v2table v2matrix v2h2h"><thead><tr><th>row beats column on</th>' + ord.map(function (x) { return '<th><span class="v2sw" style="background:' + colorOf(x.m) + '"></span>' + esc(x.m.label) + "</th>"; }).join("") + "</tr></thead><tbody>" +
      ord.map(function (x, a) { return '<tr><td><span class="v2sw" style="background:' + colorOf(x.m) + '"></span>' + esc(x.m.label) + "</td>" + ord.map(function (z, b) { if (a === b) { return '<td class="v2na">·</td>'; } var e = lg.beat[idx[a]][idx[b]], w = e.w / e.n, l = lg.beat[idx[b]][idx[a]].w / e.n, rgb = accentRGB();
        return '<td style="background:rgba(' + rgb.join(",") + "," + (0.04 + 0.5 * w).toFixed(2) + ')" title="' + esc(x.m.label + " beats " + z.m.label + " on " + e.w + ", ties on " + e.t + " and loses on " + lg.beat[idx[b]][idx[a]].w + " of " + e.n + " problems") + '">' + (w > l ? "<b>" : "") + (100 * w).toFixed(1) + " %" + (w > l ? "</b>" : "") + ' <span class="v2ci-txt">' + (100 * e.t / e.n).toFixed(0) + " % tied</span></td>"; }).join("") + "</tr>"; }).join("") + "</tbody></table></div>" +
      '<p class="v2hint">Each cell: the share of problems where the row\u2019s method does better than the column\u2019s on ' + esc(p.label) + "; the rest are ties or losses. Bold: the row\u2019s method wins more often than it loses.</p>";
    return t1 + t2;
  }
  function rankLadder(R, shown, p, ki, timed) {
    var slots = timed ? rankSlots(R) : D.rungs.map(String), by = {}, tmin = Infinity, tmax = -Infinity;
    slots.forEach(function (slot) { var lg = ranking(R, shown, slot, ki); if (lg.roster.length < 2 || !lg.cats.length) { return; }
      var x = timed ? slotSeconds(R, slot) : +slot; if (timed) { tmin = Math.min(tmin, x); tmax = Math.max(tmax, x); }
      lg.roster.forEach(function (e) { (by[e.m.key] = by[e.m.key] || { m: e.m, pts: [] }).pts.push({ x: x, v: e.share, lo: NaN, hi: NaN, hollow: e.capped, title: e.m.label + " at " + slotLabel(R, slot) + ": wins " + (100 * e.share).toFixed(1) + " % of its comparisons, average place " + e.rank.toFixed(2) + " of " + lg.k + ", " + lg.n.toLocaleString() + " problem runs" }); }); });
    var series = shown.filter(function (m) { return by[m.key]; }).map(function (m) { return { label: m.label + (m.local ? " (local)" : ""), color: colorOf(m), dash: !!m.dash, pts: by[m.key].pts }; });
    if (!series.length) { return ""; }
    var title = narrow() ? "Comparisons won" : "Comparisons won, by " + (timed ? "time limit" : "budget"), rate = { kind: "rate", fmt: "pct", hist: null }, tr = timed ? timeRange(tmin, tmax) : [0, 0];
    var svg = withState({ band: false, cross: false }, function () { return chartSVG({ title: title, aria: title, width: wideWidth(), series: series, ymin: 0, ymax: 1, ticks: [0, 0.25, 0.5, 0.75, 1], tick: function (g) { return tickLabel(rate, g); }, ylabel: narrow() ? "won" : "comparisons won", timeAxis: timed, timeSource: "budget", tmin: tr[0], tmax: tr[1], zero: 0.5, xlabel: timed ? (narrow() ? "time limit (s)" : "time limit per problem (s, log scale)") : (narrow() ? "budget" : "budget per problem (log scale)") }); });
    return '<h3 class="v2h">By ' + (timed ? "time limit" : "budget") + '</h3><div class="v2charts v2one">' + svg + "</div>" +
      '<p class="v2hint">' + term("winshare", "Comparisons won") + ": the share of one-on-one comparisons a method wins. 100 % means it beats every other method on every problem; 50 % means it wins as often as it loses." + (timed ? " A hollow marker means the method\u2019s largest budget stays below this time limit." : "") + "</p>";
  }

  // ---- shell -----------------------------------------------------------------------------------------------------
  var VIEWS = [["curves", "Curves"], ["table", "Tables"], ["matrix", "Problem sets"], ["dist", "Distribution"], ["ranks", "Ranks"], ["paired", "Paired differences"]];
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
    ranks: {},
    paired: { plots: 1, base: 1, ci: 1, xaxis: 1 }
  };
  function shownMetricKeys(view) { return view === "matrix" ? [state.focus] : view === "dist" ? [state.dmetric] : view === "ranks" ? [] : plotAxes(); }
  function usesFor(view) {
    var u = {}, src = USES[view] || {};
    Object.keys(src).forEach(function (k) { u[k] = src[k]; });
    if (shownMetricKeys(view).some(function (k) { return METRIC[k] && METRIC[k].worst !== undefined; })) { u.impute = 1; }
    if (view === "table" && state.rows !== "cats") { delete u.rung; }   // the budget only binds the by-catalog table
    if (view === "dist" && METRIC[state.dmetric].kind !== "rate" && state.dmode === "rungs") { delete u.rung; }
    if (view === "ranks" && state.xaxis === "time" && anyTime()) { delete u.rung; }
    return u;
  }
  function shell() {
    var rel = D.release;
    var strip = D.methods.filter(function (m) { return D.status[m.key]; }).map(function (m) { var d = D.status[m.key][0], t = D.status[m.key][1]; return '<div class="v2tile" title="' + esc(m.label) + '"><b><span class="v2sw" style="background:' + colorOf(m) + '"></span>' + esc(m.label) + (m.local ? " (local)" : "") + '</b><span>' + d + '<small> / ' + (t == null ? "?" : t) + '</small></span><div class="v2bar"><i style="width:' + (t ? 100 * d / t : 0) + '%"></i></div></div>'; }).join("");
    // When the release was last refreshed: stored with its offset, shown in the reader's own time zone, with how long ago
    var stamp = (function () {
      var t = rel.updated ? new Date(rel.updated) : null;
      if (!t || isNaN(t.getTime())) { return rel.generated ? '<span class="v2updated">Updated ' + esc(rel.generated) + '</span>' : ""; }
      var abs = t.toLocaleString("en-GB", { day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit", timeZoneName: "short" });
      var mins = Math.round((Date.now() - t.getTime()) / 60000), ago;
      if (mins < 1) { ago = "just now"; } else if (mins < 60) { ago = mins + (mins === 1 ? " minute" : " minutes") + " ago"; } else if (mins < 48 * 60) { var h = Math.round(mins / 60); ago = h + (h === 1 ? " hour" : " hours") + " ago"; } else { ago = Math.round(mins / 1440) + " days ago"; }
      return '<time class="v2updated" datetime="' + esc(rel.updated) + '">Updated ' + esc(abs) + ' <span class="v2ago">· ' + ago + '</span></time>';
    })();
    var progressHint = "How much of each method's evaluation is done. Every method is run twice on every problem set, at each of its budgets. One count is one problem set at one budget in one run; it is counted once every problem in it has a result. A budget appears in the plots once every problem set is done at that budget, in at least one of the two runs.";
    var catList = CATS.map(function (c) { var m = CAT[c]; return '<label title="' + esc(GROUPS[m.group] + (m.mu ? " · typical formula length " + m.mu[1] + " bits (middle half: " + m.mu[0] + " to " + m.mu[2] + ")" : "")) + '"><input type="checkbox" data-c="' + c + '"> ' + esc(c) + ' <span class="v2hint">' + m.laws + '</span></label>'; }).join("");
    var methList = D.methods.filter(withData).map(function (m) { return '<div class="v2meth"><label><input type="checkbox" data-m="' + m.key + '"><input type="color" class="v2swatch" data-m="' + m.key + '" value="' + colorOf(m) + '" title="Colour for ' + esc(m.label) + '"><span class="v2mname">' + esc(m.label) + '</span></label>' + (m.local ? ' <span class="v2tag v2tag-local">local only</span>' : "") + ' <span class="v2hint">' + esc(m.param) + '</span>' + (m.selection ? " " + help(m.selection, "How does " + m.label + " choose its prediction?") : "") + ' <span class="v2tag" title="' + esc(PROV_NOTE[m.provenance] || "") + '">' + esc(PROV[m.provenance] || m.provenance || "") + '</span><button type="button" class="v2reset" data-m="' + m.key + '" title="Reset colour to default" hidden>↺</button></div>'; }).join("") || '<span class="v2hint">no method has finished a budget yet</span>';
    var metricList = MGROUPS.map(function (g) { var ms = D.metrics.filter(function (m) { return m.group === g; }); return '<div class="v2mgroup" data-group="' + esc(g) + '"><h4>' + esc(g) + '</h4>' + ms.map(function (m) { return '<div class="v2metric" data-tier="' + m.tier + '" data-key="' + m.key + '"><label><input type="checkbox" data-p="' + m.key + '"> ' + esc(m.label) + '</label> ' + mhelp(m) + '</div>'; }).join("") + "</div>"; }).join("");
    root.innerHTML =
      '<div class="v2relhead"><div><h2 class="v2hltitle">Release ' + esc(rel.id) + (rel.title !== rel.id ? ' · ' + esc(rel.title) : '') + '</h2><p class="v2hlsub">' + stamp + (rel.notes ? ' · ' + esc(rel.notes) : '') + '</p></div><div class="v2row"><button type="button" class="v2btn" data-act="link">copy link to this view</button><span class="v2linkok v2hint" hidden>link copied</span></div></div>' +
      '<details class="v2release"><summary class="v2kicker">Protocol</summary><ul><li><b>Choosing a prediction.</b> ' + esc(rel.scoring || "") + '</li><li><b>Judging it.</b> ' + esc(rel.judge || "") + '</li><li><b>Problems.</b> Each problem is one known formula, with 512 data points given to the method and 512 held-out points used only to check the formula it returns. The points are sampled from the ranges the problem set specifies, without noise. Every method is run twice on each problem, with new points each time. ' + CATS.length + ' problem sets, ' + laws(CATS).toLocaleString() + ' problems.</li><li><b>Settings.</b> ' + term("provenance", "Who chose each method\'s settings") + ' is shown next to its name.</li>' + (rel.versions ? '<li><b>Versions.</b> ' + esc(rel.versions) + '</li>' : '') + '<li><b>Time.</b> Seconds per problem. ' + (D.timing_note ? esc(D.timing_note) + " " : "") + term("time", "Why only these times?") + '</li><li><b>Statistics.</b> ' + term("regime", "How problems without a usable formula count") + ', ' + term("complete", "why some points are missing") + ', ' + term("wilson", "what the 95 % intervals mean") + '.</li></ul></details>' +
      '<section class="v2status" aria-label="Progress of this release"><h3 class="v2kicker">Progress ' + help(progressHint, "What is counted?") + '</h3><div class="v2strip">' + strip + '</div></section>' +
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
      '<div class="v2row v2checks" data-uses="ci"><span class="v2lab" data-uses="ci">95 % intervals ' + help(TERMS.wilson) + '</span>' +
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
    root.querySelectorAll(".v2methods input[type=checkbox]").forEach(function (i) { i.checked = state.methods.indexOf(i.dataset.m) >= 0; });
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
    root.querySelectorAll(".v2swatch").forEach(function (i) { var m = D.methods.filter(function (x) { return x.key === i.dataset.m; })[0]; if (m && document.activeElement !== i) { i.value = colorOf(m); } });
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
    try {
      syncControls();
      var shown = shownMethods(), view = root.querySelector(".v2view");
      if (!rungChosen) { rungChosen = true; var br = bestRung(shown); if (br) { state.rung = br; } syncControls(); }
      if (state.view !== "dist" && state.view !== "paired") {   // histograms the current view needs
        (state.view === "matrix" ? [METRIC[state.focus]] : D.metrics.filter(function (m) { return plotAxes().indexOf(m.key) >= 0; })).forEach(function (m) { if (needsHist(m)) { ensureHists(m); } });
      }
      renderHeadline();
      view.innerHTML = state.view === "table" ? renderTable(shown) : state.view === "matrix" ? renderMatrix(shown) : state.view === "dist" ? renderDist(shown) : state.view === "ranks" ? renderRanks(shown) : state.view === "paired" ? renderPaired(shown) : renderCurves(shown);
      // A redraw replaces the button an open menu hangs on (a histogram that arrives, a container that settles).
      // The menu moves to the button's successor; it closes only when the control itself is gone.
      if (pickerFor && !document.body.contains(pickerFor)) { var again = samePick(pickerFor); if (again) { pickerFor = again; again.setAttribute("aria-expanded", "true"); placePicker(again); } else { closePicker(); } }
      root.querySelector(".v2err").textContent = ""; save();
    } catch (e) { root.querySelector(".v2err").textContent = "The explorer hit an error while drawing: " + (e && e.message ? e.message : e) + ". Reload the page, or press “all” under Catalogs to reset the selection."; if (window.console) { console.error(e); } }
  }

  // ---- events ----------------------------------------------------------------------------------------------------
  var SETTABLE = { rung: 1, dmode: 1, dnorm: 1, dmetric: 1, xaxis: 1, tbudget: 1 };
  function setState(key, val) {
    if (!SETTABLE[key]) { return; }
    if (key === "rung") { var r = parseInt(val, 10); if (D.rungs.indexOf(r) >= 0) { state.rung = r; } return; }
    if (key === "dmetric" && !METRIC[val]) { return; }
    state[key] = val;
  }
  shell();
  root.addEventListener("change", function (e) {
    var t = e.target;
    if (t.dataset.c) { if (t.checked) { state.cats.push(t.dataset.c); } else { state.cats = state.cats.filter(function (c) { return c !== t.dataset.c; }); } }
    else if (t.dataset.m && t.type === "checkbox") { if (t.checked) { state.methods.push(t.dataset.m); } else { state.methods = state.methods.filter(function (c) { return c !== t.dataset.m; }); } }
    else if (t.dataset.p) { if (t.checked) { state.plots.push({ x: lastAxis(), y: t.dataset.p }); } else { state.plots = state.plots.filter(function (c) { return c.y !== t.dataset.p; }); } }
    else if (t.name === "v2stat") { state.stat = t.value; } else if (t.name === "v2xaxis") { state.xaxis = t.value; } else if (t.name === "v2rows") { state.rows = t.value; }
    else if (t.classList.contains("v2band")) { state.band = t.checked; } else if (t.classList.contains("v2cross")) { state.cross = t.checked; } else if (t.classList.contains("v2impute")) { state.impute = t.checked; } else if (t.classList.contains("v2tier")) { state.tier = t.checked ? "all" : "main"; }
    else if (t.classList.contains("v2rung")) { state.rung = parseInt(t.value, 10); } else if (t.classList.contains("v2base")) { state.base = t.value; }
    else if (t.dataset.state) { setState(t.dataset.state, t.value); }
    else if (t.classList.contains("v2swatch")) { userColors[t.dataset.m] = t.value; writeCookie(userColors); root.querySelector(".v2cookie").hidden = false; }
    else { return; }
    render();
  });
  root.addEventListener("input", function (e) { var t = e.target; if (t.classList.contains("v2valid")) { state.valid = Math.min(100, Math.max(0, parseInt(t.value, 10) || 0)); syncControls(); scheduleRender(); } else if (t.classList.contains("v2q")) { state.q = t.value; syncControls(); } else if (t.classList.contains("v2swatch")) { userColors[t.dataset.m] = t.value; render(); } });
  root.addEventListener("click", function (e) {
    var b = e.target.closest ? e.target.closest("button") : null; if (!b || !root.contains(b) || b.classList.contains("v2help")) { return; }
    if (b.dataset.view) { state.view = b.dataset.view; render(); return; }
    if (b.dataset.set) { var kv = b.dataset.set.split(":"); setState(kv[0], kv.slice(1).join(":")); render(); return; }
    if (b.classList.contains("v2reset")) { delete userColors[b.dataset.m]; writeCookie(userColors); render(); return; }
    if (b.classList.contains("v2rmplot")) { state.plots.splice(+b.dataset.i, 1); render(); return; }
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
    if (!el || prev === el || !(root.contains(el) || (headRoot && headRoot.contains(el)))) { return; }
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
    if (root.contains(trigger)) { openPicker(trigger); }
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
    [root.querySelector(".v2main"), headRoot].forEach(function (el) { if (el) { ro.observe(el); } });
  }
  document.addEventListener("scroll", closeArmed, true);
  root.addEventListener("keydown", function (e) {
    if (e.key !== "Enter" || !e.target.classList || !e.target.classList.contains("v2addmkey")) { return; }
    e.preventDefault(); var v = e.target.value.trim(); if (v) { tryKey(v, false); }
  });
  render();
  // a key given earlier in this tab opens the same payload again without asking for it
  try { var saved = window.sessionStorage.getItem("srbf.k"); if (saved) { tryKey(saved, true); } } catch (e) { /* storage off */ }
  if (fromUrl) { try { root.scrollIntoView({ block: "start" }); } catch (e) { /* ignore */ } }
})();
