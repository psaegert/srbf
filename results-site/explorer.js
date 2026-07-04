/* srbf results explorer — standalone frontend build.
 *
 * Reads window.RESULTS_DATA (emitted by srbf.analysis.export_data + augmented by
 * build_results_data.py): { metrics:[{key,label,higher_is_better,tier}], axes:[...],
 * colors:{series:hex}, series_meta:{series:{group,default_on,is_ablation,parent,predecessor}},
 * records:[...] }. Views form a registry over {Display: Curves|Table|Ranks} × {Values:
 * Absolute|Paired}: curves×absolute (marginal line + CI band), curves×paired (Δ-vs-baseline
 * curves with pointwise bands + verdict), table×absolute (each series read at exactly the
 * selected budget, interpolated in log-time, plateau-flagged), table×paired (k×k four-state
 * verdict matrix), ranks (critical-difference league of the fixed roster; Values n/a).
 * Everything except curves×absolute reads paired_data.json (lazy-fetched; ALL statistics precomputed in Python by
 * srbf's paired layer — this file only renders). Per-series colour customisation persists in a
 * single functional cookie; two-tier metric menu; theme-aware (light/dark).
 */
(function () {
  "use strict";

  var AXIS_META = {
    compute: { label: "Inference compute (median fit time)", unit: "s", log: true },
    n_support: { label: "Support size (n samples)", unit: "", log: true },
    noise: { label: "Noise level (relative)", unit: "", log: false }
  };
  var FALLBACK = ["#4f46e5", "#f97316", "#16a34a", "#dc2626", "#9333ea", "#0891b2", "#db2777", "#65788a", "#ca8a04", "#0ea5e9"];
  var GROUP_ORDER = ["Models", "Baselines"];
  var COOKIE = "srbf_colors";

  // --- cookie helpers (single first-party functional cookie; set ONLY on an explicit colour change) ---
  function readCookie() {
    var m = document.cookie.match(new RegExp("(?:^|; )" + COOKIE + "=([^;]*)"));
    if (!m) { return {}; }
    try { return JSON.parse(decodeURIComponent(m[1])) || {}; } catch (e) { return {}; }
  }
  function writeCookie(obj) {
    if (!obj || !Object.keys(obj).length) { deleteCookie(); return; }
    var v = encodeURIComponent(JSON.stringify(obj));
    document.cookie = COOKIE + "=" + v + ";path=/;max-age=" + (60 * 60 * 24 * 365) + ";SameSite=Lax";
  }
  function deleteCookie() { document.cookie = COOKIE + "=;path=/;max-age=0;SameSite=Lax"; }

  function init() {
    var root = document.getElementById("results-explorer");
    if (!root || typeof window.RESULTS_DATA === "undefined") { return; }
    if (typeof Plotly === "undefined") { root.innerHTML = "<div class='loading'>Plotly failed to load.</div>"; return; }
    root.innerHTML = "";

    var data = window.RESULTS_DATA;
    var records = data.records;
    var metrics = data.metrics;                       // [{key, label, higher_is_better, tier}]
    var axes = data.axes;
    var defaults = data.colors || {};                 // paper-consistent default palette
    var meta = data.series_meta || {};
    var series = uniqueSorted(records.map(function (r) { return r.series; }));
    var benches = uniqueSorted(records.map(function (r) { return r.benchmark; }));

    var userColors = readCookie();
    function metaOf(s) { return meta[s] || {}; }
    function defaultColor(s, i) { return defaults[s] || FALLBACK[i % FALLBACK.length]; }
    function colorOf(s, i) { return userColors[s] || defaultColor(s, i); }
    var idx = {}; series.forEach(function (s, i) { idx[s] = i; });

    // --- metric select, two-tier via optgroups ---
    var metricSel = document.createElement("select");
    var tiers = [["main", "Main metrics"], ["more", "More metrics"]];
    tiers.forEach(function (t) {
      var ms = metrics.filter(function (m) { return (m.tier || "more") === t[0]; });
      if (!ms.length) { return; }
      var og = document.createElement("optgroup"); og.label = t[1];
      ms.forEach(function (m) { var o = document.createElement("option"); o.value = m.key; o.textContent = m.label; og.appendChild(o); });
      metricSel.appendChild(og);
    });
    var firstMain = metrics.filter(function (m) { return (m.tier || "more") === "main"; })[0];
    if (firstMain) { metricSel.value = firstMain.key; }

    var axisSel = selectFrom(axes.map(function (a) { return [a, (AXIS_META[a] || {}).label || a]; }));
    var benchSel = selectFrom(benches.map(function (b) { return [b, b]; }));

    // --- view registry: a 2×2 — {Display: Curves|Table} × {Values: Absolute|Paired} ----------
    // Legacy deep-link keys (curves/paired/matrix) stay stable; 'table' is the fourth quadrant.
    var VIEWS = {
      curves: { display: "curves", values: "absolute", usesAxis: true, usesSeries: true },
      paired: { display: "curves", values: "paired", usesBaseline: true, usesCorrection: true, usesBudget: true, usesSeries: true },
      table:  { display: "table",  values: "absolute", usesBudget: true, usesSeries: true },
      matrix: { display: "table",  values: "paired", usesCorrection: true, usesBudget: true, usesSeries: true },
      ranks:  { display: "ranks",  values: null, usesBudget: true, usesSeries: false }   // k-model view: fixed roster
    };
    function viewFor(display, values) {
      if (display === "ranks") { return "ranks"; }
      return Object.keys(VIEWS).filter(function (v) {
        return VIEWS[v].display === display && VIEWS[v].values === values;
      })[0];
    }
    var currentView = "curves";
    var lastValues = "absolute";   // restored when leaving the Ranks display
    var pairedData = null, pairedLoading = false;

    // baseline select grouped like the pills: Models, Baselines, then ablation groups
    var baselineSel = document.createElement("select");
    (function () {
      var groups = [];
      series.forEach(function (s) {
        var g = (meta[s] || {}).group || "Other";
        if (groups.indexOf(g) < 0) { groups.push(g); }
      });
      groups.sort(function (a, b) {
        var ia = GROUP_ORDER.indexOf(a), ib = GROUP_ORDER.indexOf(b);
        return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib) || a.localeCompare(b);
      });
      groups.forEach(function (g) {
        var og = document.createElement("optgroup"); og.label = g;
        series.forEach(function (s) {
          if (((meta[s] || {}).group || "Other") !== g) { return; }
          var o = document.createElement("option"); o.value = s; o.textContent = s;
          og.appendChild(o);
        });
        baselineSel.appendChild(og);
      });
    })();
    if (series.indexOf("v23.0-120M") >= 0) { baselineSel.value = "v23.0-120M"; }

    var correctionSel = selectFrom([["corrected", "corrected (default)"], ["raw", "raw p — exploratory only"]]);

    // compute-budget slider: CONTINUOUS in log-time over the measured span, with magnetic
    // snap points at the PRE-DECLARED budget grid. Only the marks carry precomputed verdicts,
    // margins and corrected p-values (computed in Python, pre-registered); any position
    // between them renders a DESCRIPTIVE read of the plotted curves at t — value/Δ + CI
    // interpolated along the drawn segments, never a verdict (free-t verdicts would be an
    // uncorrectable infinite family: "slide until significant").
    var budgetWrap = document.createElement("div"); budgetWrap.className = "budget-slider";
    var budgetValue = document.createElement("span"); budgetValue.className = "budget-value";
    var budgetRange = document.createElement("input");
    budgetRange.type = "range"; budgetRange.min = "0"; budgetRange.max = "1000";
    budgetRange.step = "1"; budgetRange.value = "1000";
    budgetRange.setAttribute("aria-label", "Compute budget");
    var budgetTicks = document.createElement("div"); budgetTicks.className = "budget-ticks";
    budgetWrap.appendChild(budgetValue);
    budgetWrap.appendChild(budgetRange);
    budgetWrap.appendChild(budgetTicks);

    // log-t domain = exactly the marked grid (payload budgets), so the 10^k marks sit
    // equally spaced with no dead track outside them; set once the payload arrives
    var budgetLog = null;
    var SNAP_FRACTION = 0.015;                      // snap radius as a fraction of the log span
    var currentBudget = null, budgetSnapped = false;
    function positionOf(t) { return (Math.log10(t) - budgetLog.lo) / budgetLog.span; }
    function tOf(position) { return Math.pow(10, budgetLog.lo + position * budgetLog.span); }
    function selectedBudget() { return currentBudget; }
    function budgetIsSnapped() { return budgetSnapped; }

    function syncBudget() {
      var budgets = (pairedData && pairedData.budgets) || [];
      if (!budgetLog) { budgetValue.textContent = "…"; return; }
      var t = tOf(parseInt(budgetRange.value, 10) / 1000);
      budgetSnapped = false;
      if (currentView === "ranks" && budgets.length) {
        // leagues exist at marks only — the slider is snap-only here (nearest mark wins)
        t = budgets.reduce(function (a, b) {
          return Math.abs(Math.log10(t) - Math.log10(a)) <= Math.abs(Math.log10(t) - Math.log10(b)) ? a : b;
        });
        budgetSnapped = true;
        budgetRange.value = String(Math.round(positionOf(t) * 1000));
      }
      for (var i = 0; !budgetSnapped && i < budgets.length; i++) {
        if (Math.abs(Math.log10(t) - Math.log10(budgets[i])) <= SNAP_FRACTION * budgetLog.span) {
          t = budgets[i]; budgetSnapped = true;     // magnetic snap near a mark
          budgetRange.value = String(Math.round(positionOf(t) * 1000));
          break;
        }
      }
      currentBudget = t;
      budgetValue.innerHTML = budgetSnapped
        ? "≤ " + t + " s per problem"
        : "t ≈ " + Number(t.toPrecision(3)) + " s per problem <span class='budget-desc-tag'>descriptive</span>";
      budgetTicks.querySelectorAll(".budget-tick").forEach(function (el, i) {
        el.classList.toggle("active", budgetSnapped && budgets[i] === currentBudget);
      });
    }
    function fillBudgets() {
      if (!pairedData || pairedData.error || !pairedData.budgets || budgetTicks.childNodes.length) { return; }
      var budgets = pairedData.budgets;
      budgetLog = { lo: Math.log10(budgets[0]),
                    span: Math.log10(budgets[budgets.length - 1]) - Math.log10(budgets[0]) };
      budgets.forEach(function (b, i) {
        var tick = document.createElement("button");
        tick.type = "button"; tick.className = "budget-tick";
        tick.textContent = String(b);   // unit lives in the readout
        var frac = Math.min(1, Math.max(0, positionOf(b)));
        tick.style.left = (frac * 100) + "%";
        if (frac < 0.06) { tick.classList.add("first"); }
        if (frac > 0.94) { tick.classList.add("last"); }
        tick.addEventListener("click", function () {
          budgetRange.value = String(Math.round(positionOf(b) * 1000));
          syncBudget(); render();
        });
        budgetTicks.appendChild(tick);
      });
      budgetRange.value = String(Math.round(positionOf(budgets[budgets.length - 1]) * 1000));
      syncBudget();
    }
    var budgetRenderPending = false;
    budgetRange.addEventListener("input", function () {
      syncBudget();
      if (!budgetRenderPending) {                    // one render per frame while dragging
        budgetRenderPending = true;
        window.requestAnimationFrame(function () { budgetRenderPending = false; render(); });
      }
    });

    // two segmented toggles exposing the 2×2: what to draw × what the numbers mean
    var tabs = document.createElement("div"); tabs.className = "view-tabs";
    function segmented(labelText, axis, options) {
      var field = document.createElement("div"); field.className = "view-toggle";
      var lab = document.createElement("span"); lab.className = "results-field-label";
      lab.textContent = labelText; field.appendChild(lab);
      var group = document.createElement("div"); group.className = "view-toggle-group";
      options.forEach(function (opt) {
        var b = document.createElement("button"); b.type = "button"; b.className = "view-tab";
        b.textContent = opt[1]; b.dataset.axis = axis; b.dataset.value = opt[0];
        b.addEventListener("click", function () {
          var vm = VIEWS[currentView];
          var next = axis === "display"
            ? viewFor(opt[0], vm.values || lastValues)
            : viewFor(vm.display, opt[0]);
          if (next) { switchView(next); }
        });
        group.appendChild(b);
      });
      field.appendChild(group);
      return field;
    }
    tabs.appendChild(segmented("Display", "display", [["curves", "Curves"], ["table", "Table"], ["ranks", "Ranks"]]));
    tabs.appendChild(segmented("Values", "values", [["absolute", "Absolute"], ["paired", "Paired Δ"]]));

    var controls = document.createElement("div");
    controls.className = "results-controls";
    var axisField = labelled("X-axis", axisSel);
    var baselineField = labelled("Baseline — every checked series is compared against it", baselineSel);
    var baselineHint = document.createElement("span");
    baselineHint.className = "results-field-hint";
    baselineHint.textContent = "The zero line. Any series can serve — it does not have to be checked above.";
    baselineField.appendChild(baselineHint);
    var correctionField = labelled("Multiple-comparison correction", correctionSel);
    var budgetField = labelled("Compute budget", budgetWrap);
    var budgetHint = document.createElement("span");
    budgetHint.className = "results-field-hint";
    budgetHint.textContent = "Marks = release-grade numbers · between marks = descriptive curve read (snaps near marks).";
    budgetField.appendChild(budgetHint);
    controls.appendChild(axisField);
    controls.appendChild(labelled("Metric", metricSel));
    controls.appendChild(labelled("Benchmark", benchSel));
    controls.appendChild(budgetField);
    controls.appendChild(baselineField);
    controls.appendChild(correctionField);
    [axisSel, metricSel, benchSel, baselineSel, correctionSel].forEach(function (s) { s.addEventListener("change", render); });
    metricSel.addEventListener("change", function () {
      ranksSwitchedFrom = null; ranksRestoreKey = null; ranksAutoKey = null;
    });

    function switchView(v) {
      var leavingRanks = currentView === "ranks" && v !== "ranks";
      currentView = v;
      var vm = VIEWS[v];
      if (leavingRanks && ranksRestoreKey && metricSel.value === ranksAutoKey) {
        metricSel.value = ranksRestoreKey;   // the Ranks metric switch is view-scoped
        ranksRestoreKey = null; ranksAutoKey = null; ranksSwitchedFrom = null;
      }
      if (vm.values) { lastValues = vm.values; }
      tabs.querySelectorAll(".view-tab").forEach(function (b) {
        b.classList.toggle("active", vm[b.dataset.axis] === b.dataset.value);
        if (b.dataset.axis === "values") { b.classList.toggle("disabled", vm.display === "ranks"); }
      });
      axisField.style.display = vm.usesAxis ? "" : "none";
      baselineField.style.display = vm.usesBaseline ? "" : "none";
      budgetField.style.display = vm.usesBudget ? "" : "none";
      correctionField.style.display = vm.usesCorrection ? "" : "none";
      seriesArea.style.display = vm.usesSeries === false ? "none" : "";
      tools.style.display = vm.usesSeries === false ? "none" : "";
      if (v === "ranks") { ranksAutoPicked = false; syncBudget(); }
      var inRanks = vm.display === "ranks";
      for (var oi = 0; oi < metricSel.options.length; oi++) {
        var opt2 = metricSel.options[oi];
        opt2.disabled = inRanks && pairedData && !rankMetricInfo(opt2.value);
      }
      rosterNote.style.display = inRanks ? "" : "none";
      if (v !== "curves" && !pairedData && !pairedLoading) { loadPaired(); }
      render();
    }

    function loadPaired() {
      pairedLoading = true;
      fetch("paired_data.json").then(function (r) { return r.json(); }).then(function (d) {
        pairedData = d; pairedLoading = false; fillBudgets(); render();
      }).catch(function () {
        pairedLoading = false; pairedData = { error: true }; render();
      });
    }

    // --- series pills, grouped; primary groups visible, ablations collapsed ---
    var seriesChecks = {};
    var swatchInputs = {};

    function pill(s) {
      var i = idx[s];
      var wrap = document.createElement("span"); wrap.className = "series-pill";
      var lbl = document.createElement("label"); lbl.className = "series-toggle";
      var cb = document.createElement("input"); cb.type = "checkbox";
      cb.checked = metaOf(s).default_on === true;
      cb.addEventListener("change", render);
      seriesChecks[s] = cb;
      var color = document.createElement("input"); color.type = "color"; color.className = "series-swatch";
      color.value = toHex6(colorOf(s, i)); color.title = "Colour for " + s;
      color.addEventListener("input", function () {
        userColors[s] = color.value; writeCookie(userColors); syncReset(s); render(); showCookieNotice();
      });
      swatchInputs[s] = color;
      var name = document.createElement("span"); name.className = "series-name"; name.textContent = s;
      var reset = document.createElement("button"); reset.type = "button"; reset.className = "series-reset";
      reset.textContent = "↺"; reset.title = "Reset colour to default";
      reset.addEventListener("click", function () {
        delete userColors[s]; writeCookie(userColors);
        color.value = toHex6(defaultColor(s, i)); syncReset(s); render();
      });
      lbl.appendChild(cb); lbl.appendChild(color); lbl.appendChild(name); lbl.appendChild(reset);
      wrap.appendChild(lbl);
      syncReset(s);
      return wrap;
      function syncReset(k) { reset.style.display = userColors[k] ? "inline-flex" : "none"; }
    }

    function groupsFor(pred) {
      var gs = [];
      series.forEach(function (s) { if (pred(s)) { var g = metaOf(s).group || "Other"; if (gs.indexOf(g) < 0) { gs.push(g); } } });
      gs.sort(function (a, b) { var ia = GROUP_ORDER.indexOf(a), ib = GROUP_ORDER.indexOf(b);
        return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib) || a.localeCompare(b); });
      return gs;
    }
    function groupBlock(g, pred) {
      var block = document.createElement("div"); block.className = "series-group";
      var h = document.createElement("div"); h.className = "series-group-label"; h.textContent = g; block.appendChild(h);
      var row = document.createElement("div"); row.className = "results-series";
      series.forEach(function (s) { if (pred(s) && (metaOf(s).group || "Other") === g) { row.appendChild(pill(s)); } });
      block.appendChild(row); return block;
    }

    var seriesArea = document.createElement("div"); seriesArea.className = "series-area";
    var rosterNote = document.createElement("div"); rosterNote.className = "results-field-hint roster-note";
    rosterNote.textContent = "Ranks always compares the full pre-declared roster (all sizes + all baselines) — series selection does not apply here.";
    rosterNote.style.display = "none";
    var primary = function (s) { return !metaOf(s).is_ablation; };
    var ablation = function (s) { return metaOf(s).is_ablation === true; };
    groupsFor(primary).forEach(function (g) { seriesArea.appendChild(groupBlock(g, primary)); });

    var ablGroups = groupsFor(ablation);
    if (ablGroups.length) {
      var det = document.createElement("details"); det.className = "ablation-details";
      var sum = document.createElement("summary");
      var nAbl = series.filter(ablation).length;
      sum.textContent = "Ablations (" + nAbl + ") — off by default";
      det.appendChild(sum);
      ablGroups.forEach(function (g) { det.appendChild(groupBlock(g, ablation)); });
      seriesArea.appendChild(det);
    }

    // --- colour tools + cookie notice ---
    var tools = document.createElement("div"); tools.className = "colour-tools";
    var resetAll = document.createElement("button"); resetAll.type = "button"; resetAll.className = "reset-all";
    resetAll.textContent = "Reset all colours";
    resetAll.addEventListener("click", function () {
      userColors = {}; deleteCookie();
      series.forEach(function (s) { if (swatchInputs[s]) { swatchInputs[s].value = toHex6(defaultColor(s, idx[s])); } });
      seriesArea.querySelectorAll(".series-reset").forEach(function (b) { b.style.display = "none"; });
      hideCookieNotice(); render();
    });
    var notice = document.createElement("span"); notice.className = "cookie-notice"; notice.id = "cookie-notice";
    notice.innerHTML = "A single functional cookie remembers your colour choices on this device — " +
      "no tracking, no third parties. It is written only when you change a colour, and “Reset all colours” deletes it.";
    tools.appendChild(resetAll); tools.appendChild(notice);
    function showCookieNotice() { notice.classList.add("visible"); }
    function hideCookieNotice() { notice.classList.remove("visible"); }
    if (Object.keys(userColors).length) { showCookieNotice(); }

    var plot = document.createElement("div"); plot.id = "results-plot";
    plot.style.width = "100%"; plot.style.height = "500px";
    var pairedFoot = document.createElement("div"); pairedFoot.className = "paired-foot";
    // matrix cells: tap/click (and Enter/Space) pins the cell's full record below the table —
    // hover titles are invisible on touch devices, so this is the primary affordance on mobile
    pairedFoot.addEventListener("click", function (e) {
      var td = e.target.closest ? e.target.closest("td[data-cell]") : null;
      if (td) { selectMatrixCell(parseInt(td.dataset.cell, 10)); }
    });
    pairedFoot.addEventListener("keydown", function (e) {
      if (e.key !== "Enter" && e.key !== " ") { return; }
      var td = e.target.closest ? e.target.closest("td[data-cell]") : null;
      if (td) { e.preventDefault(); selectMatrixCell(parseInt(td.dataset.cell, 10)); }
    });

    root.appendChild(tabs);
    root.appendChild(controls);
    root.appendChild(seriesArea);
    root.appendChild(rosterNote);
    root.appendChild(tools);
    root.appendChild(plot);
    root.appendChild(pairedFoot);

    // deep links: ?view=paired&baseline=PySR&bench=FastSRB&metric=numeric_recovery_val
    try {
      var params = new URLSearchParams(window.location.search);
      if (params.get("metric")) { metricSel.value = params.get("metric"); }
      if (params.get("bench")) { benchSel.value = params.get("bench"); }
      if (params.get("baseline")) { baselineSel.value = params.get("baseline"); }
      if (params.get("view") && VIEWS[params.get("view")]) { currentView = params.get("view"); }
    } catch (e) { /* older browsers: defaults */ }
    switchView(currentView);

    function theme() {
      var dark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
      return dark ? { grid: "#232838", ink: "#b7bdcc", zero: "#333a4f" } : { grid: "#eceef4", ink: "#4a4f5e", zero: "#d7dae4" };
    }

    function render() {
      pairedFoot.innerHTML = "";
      if (currentView === "paired") { return renderPaired(); }
      if (currentView === "matrix") { return renderMatrix(); }
      if (currentView === "table") { return renderTable(); }
      if (currentView === "ranks") { return renderRanks(); }
      plot.style.display = "";
      return renderCurves();
    }

    function pairedReady() {
      if (pairedData && !pairedData.error) { return true; }
      plot.style.display = "";
      Plotly.react(plot, [], {
        annotations: [{ text: pairedLoading ? "Loading comparison data…" :
          (pairedData && pairedData.error ? "The comparison data failed to load — refresh to retry." : "Loading comparison data…"),
          showarrow: false, font: { size: 15, color: theme().ink } }],
        paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)"
      }, { responsive: true, displayModeBar: false });
      return false;
    }

    // orientation helper: paired records/curves are stored canonically (a, b); flip for display
    function orient(entry, selected, baseline) {
      if (entry.a === selected && entry.b === baseline) { return 1; }
      if (entry.a === baseline && entry.b === selected) { return -1; }
      return 0;
    }
    function flipVerdict(v) { return v === "better" ? "worse" : v === "worse" ? "better" : v; }
    // prob_superiority is stored as P(a better) + 0.5·P(tie), so the row-oriented flip is exactly
    // 1 − value; re-round to the payload's own display precision
    function flipPsup(v) {
      if (v === null || v === undefined || v !== v) { return v; }
      var d = (String(v).split(".")[1] || "").length;
      return Number((1 - v).toFixed(Math.max(d, 1)));
    }
    var VERDICT_COLORS = { better: "#16a34a", worse: "#dc2626", equivalent: "#2563eb", undecided: "#8a8f9e" };

    // --- descriptive reads at free slider positions: the plotted curves, quantified ---------
    // Linear interpolation in log10-time along the DRAWN segments (curves are piecewise-linear
    // in log-t by construction), so a free-t number is exactly what a ruler on the figure gives.
    // Never a verdict: margins and corrections exist only at the pre-declared marks.
    function fmtDesc(v) { return (v === null || v === undefined || v !== v) ? "–" : String(Number(Number(v).toPrecision(3))); }
    function logW(xLo, xHi, t) { return (Math.log10(t) - Math.log10(xLo)) / (Math.log10(xHi) - Math.log10(xLo)); }

    function marginalReadAt(s, bench, metricKey, t) {
      var pts = records.filter(function (r) {
        return r.series === s && r.axis === "compute" && r.benchmark === bench &&
               r[metricKey] && r[metricKey].median !== null && r.x !== null;
      }).sort(function (a, b) { return a.x - b.x; });
      if (!pts.length) { return null; }
      if (t < pts[0].x * (1 - 1e-9)) { return { status: "below" }; }
      var last = pts[pts.length - 1];
      if (t > last.x * (1 + 1e-9)) {
        var lm = last[metricKey];
        return { status: "plateau", x: last.x, value: lm.median, lo: lm.lo, hi: lm.hi, n: lm.n };
      }
      var j = 0;
      while (j < pts.length - 1 && pts[j].x < t) { j++; }
      if (Math.abs(Math.log10(pts[j].x) - Math.log10(t)) < 1e-9) {
        var em = pts[j][metricKey];
        return { status: "measured", x: pts[j].x, value: em.median, lo: em.lo, hi: em.hi, n: em.n };
      }
      var a = pts[j - 1][metricKey], b = pts[j][metricKey];
      var w = logW(pts[j - 1].x, pts[j].x, t);
      return { status: "interpolated", x: t,
               value: (1 - w) * a.median + w * b.median,
               lo: (1 - w) * a.lo + w * b.lo, hi: (1 - w) * a.hi + w * b.hi,
               n: Math.min(a.n, b.n) };
    }

    function pairedReadAt(curve, sign, t) {
      var pts = curve.points.filter(function (p) { return p.delta !== null; });
      if (!pts.length) { return null; }
      if (t < pts[0].x * (1 - 1e-9) || t > pts[pts.length - 1].x * (1 + 1e-9)) {
        return { status: "outside" };
      }
      var j = 0;
      while (j < pts.length - 1 && pts[j].x < t) { j++; }
      var pLo = pts[Math.max(0, j - 1)], pHi = pts[j];
      var w = pHi.x === pLo.x ? 1 : logW(pLo.x, pHi.x, t);
      var d = (1 - w) * pLo.delta + w * pHi.delta;
      var lo = (1 - w) * pLo.lo + w * pHi.lo;
      var hi = (1 - w) * pLo.hi + w * pHi.hi;
      return { status: "in",
               delta: sign * d,
               lo: sign > 0 ? lo : -hi, hi: sign > 0 ? hi : -lo,
               n: Math.min(pLo.n, pHi.n) };
    }

    function descBanner() {
      var t = Number(selectedBudget().toPrecision(3));
      return "<div class='desc-banner'>Descriptive slice at t ≈ " + t + " s per problem — " +
        "values are read off the plotted curves (interpolated along the drawn segments). " +
        "There are <b>no verdicts, margins or corrected p-values between the marked budgets</b> (" +
        ((pairedData && pairedData.budgets) || []).join(", ") + " s, the pre-declared grid): " +
        "sliding until a difference looks convincing and quoting it is exactly the " +
        "multiple-comparisons trap the marks prevent. Snap to a mark for release-grade " +
        "numbers.</div>";
    }

    function metricInfo(key) { return (pairedData.metrics || []).filter(function (m) { return m.key === key; })[0]; }

    // payload numbers are pre-rounded to the precision their CI justifies — render them verbatim
    function fmt(v) { if (v === null || v === undefined || v !== v) { return "–"; } var x = (v === 0 ? 0 : v); return (x > 0 ? "+" : "") + String(x); }
    function fmtAbs(v) { return (v === null || v === undefined || v !== v) ? "–" : String(v); }
    function fmtP(p) { return (p === null || p === undefined) ? "–" : Number(p).toPrecision(2); }
    var VERDICT_GLYPHS = { better: "▲", worse: "▼", equivalent: "≈", undecided: "?" };
    function setName(key) { return ({ ablations: "ablation vs parent", size_ladder: "adjacent model sizes", baselines: "model sizes vs baselines" })[key] || key; }

    function pairedMetricGuard(metricKey) {
      if (metricInfo(metricKey)) { return true; }
      plot.style.display = "";
      Plotly.react(plot, [], {
        annotations: [{ text: "Paired comparisons are available for the Main metrics — pick one from the “Main metrics” group in the Metric menu.",
          showarrow: false, font: { size: 14, color: theme().ink } }],
        paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)"
      }, { responsive: true, displayModeBar: false });
      return false;
    }

    function renderPaired() {
      if (!pairedReady()) { return; }
      var metricKey = metricSel.value, bench = benchSel.value, baseline = baselineSel.value;
      if (!pairedMetricGuard(metricKey)) { return; }
      plot.style.display = "";
      var t = theme();
      var mi = metricInfo(metricKey);
      var traces = [];
      var notes = [];
      series.forEach(function (s) {
        if (!seriesChecks[s].checked || s === baseline) { return; }
        var col = colorOf(s, idx[s]);
        var curve = (pairedData.curves || []).filter(function (c) {
          return c.benchmark === bench && c.metric === metricKey && orient(c, s, baseline) !== 0;
        })[0];
        if (!curve) { return; }
        var sign = orient(curve, s, baseline);
        var pts = curve.points.filter(function (p) { return p.delta !== null; });
        if (!pts.length) { return; }
        var xs = pts.map(function (p) { return p.x; });
        var d = pts.map(function (p) { return sign * p.delta; });
        var lo = pts.map(function (p) { return sign > 0 ? p.lo : -p.hi; });
        var hi = pts.map(function (p) { return sign > 0 ? p.hi : -p.lo; });
        traces.push({ x: xs.concat(xs.slice().reverse()), y: hi.concat(lo.slice().reverse()),
          fill: "toself", fillcolor: hexA(col, 0.13), line: { width: 0 },
          hoverinfo: "skip", showlegend: false, legendgroup: s, type: "scatter", mode: "lines" });
        traces.push({ x: xs, y: d, name: s, legendgroup: s,
          line: { color: col, width: 2.5 },
          marker: { color: col, size: pts.map(function (p) { return (p.mA && p.mB) ? 8 : 4; }),
                    symbol: pts.map(function (p) { return (p.mA && p.mB) ? "circle" : "circle-open"; }) },
          type: "scatter", mode: "lines+markers",
          customdata: pts.map(function (p) {
            var favors = p.delta === 0 ? "neither (tie)"
              : ((sign * p.delta > 0) === (mi.higher_is_better !== false) ? s : baseline);
            return [sign > 0 ? p.lo : -p.hi, sign > 0 ? p.hi : -p.lo, p.n,
                    (p.mA && p.mB) ? "measured" : "interpolated", favors];
          }),
          hovertemplate: "<b>" + s + " − " + baseline + "</b><br>t: %{x:.3g}s (%{customdata[3]})<br>" +
            "Δ: %{y:.3f} [%{customdata[0]:.3f}, %{customdata[1]:.3f}] — point estimate favors %{customdata[4]}" +
            "<br>n=%{customdata[2]} expressions<extra></extra>" });
        if (!budgetIsSnapped()) {
          var read = pairedReadAt(curve, sign, selectedBudget());
          if (read) {
            notes.push("<b>" + s + "</b> vs " + baseline + ": " + (read.status === "outside"
              ? "<i>t is outside this pair's measured overlap — no read</i>"
              : "Δ ≈ " + fmtDesc(read.delta) + " [" + fmtDesc(read.lo) + ", " + fmtDesc(read.hi) +
                "] <span class='fam'>(descriptive curve read · n ≈ " + read.n + " expressions)</span>"));
          }
        }
        var rec = budgetIsSnapped() ? (pairedData.records || []).filter(function (r) {
          return r.benchmark === bench && r.metric === metricKey && r.budget === selectedBudget() &&
                 orient(r, s, baseline) !== 0;
        })[0] : null;
        if (rec) {
          var rSign = orient(rec, s, baseline);
          var verdict = rSign > 0 ? rec.verdict : flipVerdict(rec.verdict);
          var corrected = correctionSel.value === "corrected";
          var p = corrected ? (rec.p_adj !== undefined ? rec.p_adj : rec.q_bh) : rec.p_raw;
          var pLabel = corrected ? (rec.p_adj !== undefined ? "Holm-corrected p = " : "BH-corrected q = ") : "raw p = ";
          var onlySided = (rec.n_only_a || 0) + (rec.n_only_b || 0);
          notes.push("<b>" + s + "</b> vs " + baseline + ": " +
            "<span style='color:" + (VERDICT_COLORS[verdict] || "#888") + "'><b>" + verdictLabel(rec, verdict) + "</b></span> — " +
            "Δ = " + fmt(rSign * rec.delta) + " [" + fmt(rSign > 0 ? rec.lo : -rec.hi) + ", " +
            fmt(rSign > 0 ? rec.hi : -rec.lo) + "] vs noise margin ±" + rec.margin +
            (rec.verdict_note === "ladder-limited"
              ? " — <i>the losing side has no measurements at this budget; its carried value is a lower bound, so no at-budget verdict can be issued</i>"
              : (verdict === "undecided" && rec.equivalence_attainable === false
                 ? " — <i>too few paired expressions to ever certify “equivalent” at this margin; limited resolution, not evidence of parity</i>" : "")) +
            "<br><span class='fam'>" + pLabel + fmtP(p) +
            (rec.family_id ? " · confirmatory — pre-declared " + setName(rec.confirmatory_set) + " set, Holm-corrected over " + rec.family_size + " comparisons"
                           : " · exploratory (BH-corrected)") +
            " · n = " + rec.n_pairs + " paired expressions" +
            (onlySided ? " (+" + onlySided + " solved by one side only)" : "") +
            " · smallest reliably detectable Δ (MDE₈₀) ≈ " + rec.mde_80 + "</span>" +
            (rec.notes && rec.notes.length ? "<br><i>" + rec.notes.join(" ") + "</i>" : ""));
        }
      });
      var layout = {
        font: { family: "Inter, system-ui, sans-serif", color: t.ink, size: 13 },
        margin: { l: 62, r: 20, t: 16, b: 58 },
        xaxis: { title: "Compute budget t (s, log scale)", dtick: 1,
                 type: "log", gridcolor: t.grid, zerolinecolor: t.zero, ticks: "outside", tickcolor: t.grid },
        yaxis: { title: "Δ " + (mi ? mi.label : metricKey),
                 gridcolor: t.grid, zerolinewidth: 2, zerolinecolor: t.ink, ticks: "outside", tickcolor: t.grid },
        legend: { orientation: "h", y: -0.28, yanchor: "top", font: { size: 12 } },
        hovermode: "closest", paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
        shapes: traces.length ? [{ type: "line", x0: selectedBudget(), x1: selectedBudget(),
          yref: "paper", y0: 0, y1: 1, line: { color: t.ink, width: 1, dash: "dot" } }] : [],
        annotations: traces.length ? [] : [{ text: (pairedData.records || []).some(function (r) { return r.benchmark === bench && (r.a === baseline || r.b === baseline); })
          ? "No paired curves for this selection — toggle series above (the baseline itself is the zero line)."
          : "No paired comparisons are available for baseline " + baseline + " on " + bench + " — pick another baseline or benchmark.",
          showarrow: false, font: { size: 14, color: t.ink } }]
      };
      Plotly.react(plot, traces, layout, { responsive: true, displayModeBar: false });
      // axis titles stay short (they clip on narrow screens); the reading guide lives here instead
      var caption = traces.length
        ? "<div class='plot-caption'>Δ = selected − baseline; " +
          (mi && mi.higher_is_better === false ? "below" : "above") + " 0 favors the selected series. " +
          "x: each series' median wall-clock time per problem — solid markers are measured configurations, " +
          "hollow are interpolated between them; the dotted line marks the selected budget.</div>" : "";
      var rawWarn = correctionSel.value !== "corrected"
        ? "<div class='matrix-warning'>⚠ Uncorrected p-values — exploratory browsing only; never quote these as claims. Switch back to “corrected”.</div>" : "";
      var panelTitle = budgetIsSnapped()
        ? "Verdicts at the standardized budget of ≤ " + selectedBudget() + " s per problem (dotted line) — " +
          "every series at exactly t, plateaus flagged (95% CIs · release " +
          (pairedData.results_release_id || "?") + " · α = " + (pairedData.alpha || 0.05) + ")"
        : "Curve reads at t ≈ " + Number(selectedBudget().toPrecision(3)) + " s per problem (dotted line)";
      pairedFoot.innerHTML = notes.length
        ? caption + (budgetIsSnapped() ? rawWarn : descBanner()) +
          "<div class='paired-verdicts'><div class='paired-verdicts-title'>" + panelTitle + "</div>" +
          notes.map(function (n) { return "<div class='paired-verdict-row'>" + n + "</div>"; }).join("") + "</div>"
        : caption + (budgetIsSnapped() ? rawWarn : descBanner());
    }

    // row-oriented view of a canonical (a, b) record: everything a cell or detail panel displays
    function orientRecord(rec, rowS) {
      var sign = rec.a === rowS ? 1 : -1;
      return {
        sign: sign,
        verdict: sign > 0 ? rec.verdict : flipVerdict(rec.verdict),
        delta: sign * rec.delta,
        lo: sign > 0 ? rec.lo : -rec.hi,
        hi: sign > 0 ? rec.hi : -rec.lo,
        psup: sign > 0 ? rec.prob_superiority : flipPsup(rec.prob_superiority),
        xRow: sign > 0 ? rec.x_a : rec.x_b,
        xCol: sign > 0 ? rec.x_b : rec.x_a,
        statusRow: sign > 0 ? rec.status_a : rec.status_b,
        statusCol: sign > 0 ? rec.status_b : rec.status_a,
        bracketRow: sign > 0 ? rec.bracket_a : rec.bracket_b,
        bracketCol: sign > 0 ? rec.bracket_b : rec.bracket_a
      };
    }

    // one side's basis at the budget, for titles/detail panels ("interpolated", "plateau"...)
    function sideBasis(name, status, x, bracket) {
      if (status === "plateau") { return name + ": ladder ends at ≈" + x + " s (value carried forward — a lower bound)"; }
      if (status === "interpolated") { return name + ": interpolated at " + x + " s (between ≈" + bracket[0] + " and ≈" + bracket[1] + " s)"; }
      return name + ": measured at ≈" + x + " s";
    }
    function verdictLabel(rec, verdict) {
      return verdict + (rec.verdict_note ? " (" + rec.verdict_note + ")" : "");
    }

    var matrixCells = [], matrixSelectedKey = null;

    function matrixDetailHtml(cell) {
      if (cell.desc) {
        var dHead = "<div class='matrix-detail-title'><b>" + cell.rowS + "</b> − <b>" + cell.colS +
          "</b> · " + benchSel.value + " · t ≈ " + Number(selectedBudget().toPrecision(3)) +
          " s per problem</div>";
        if (cell.desc.status !== "in") {
          return dHead + "<div class='fam'>t is outside this pair's measured overlap — no read.</div>";
        }
        return dHead + "<div>Δ ≈ " + fmtDesc(cell.desc.delta) + " [" + fmtDesc(cell.desc.lo) +
          ", " + fmtDesc(cell.desc.hi) + "] <span class='fam'>(descriptive curve read · n ≈ " +
          cell.desc.n + " expressions)</span></div>" +
          "<div class='fam'>No verdict at unmarked times — noise margins and corrections exist " +
          "only at the pre-declared budgets; snap the slider to a mark.</div>";
      }
      var head = "<div class='matrix-detail-title'><b>" + cell.rowS + "</b> − <b>" + cell.colS +
        "</b> · " + benchSel.value + " · budget ≤ " + selectedBudget() + " s per problem</div>";
      var rec = cell.rec;
      if (!rec) {
        return head + "<div class='fam'>n/a — this pair was not measurable within this budget, " +
          "or was not evaluated on this benchmark.</div>";
      }
      var o = orientRecord(rec, cell.rowS);
      var corrected = correctionSel.value === "corrected";
      var p = corrected ? (rec.family_id ? rec.p_adj : rec.q_bh) : rec.p_raw;
      var pLabel = corrected ? (rec.family_id ? "Holm-corrected p = " : "BH-corrected q = ") : "raw p = ";
      var onlySided = (rec.n_only_a || 0) + (rec.n_only_b || 0);
      return head +
        "<div><span style='color:" + (VERDICT_COLORS[o.verdict] || "#888") + "'><b>" + verdictLabel(rec, o.verdict) +
        "</b></span> — Δ = " + fmt(o.delta) + " [" + fmt(o.lo) + ", " + fmt(o.hi) +
        "] vs noise margin ±" + rec.margin +
        (rec.verdict_note === "ladder-limited"
          ? " — <i>a plateau side has no measurements at this budget; its carried value is a lower bound, so no at-budget verdict can be issued</i>"
          : (o.verdict === "undecided" && rec.equivalence_attainable === false
             ? " — <i>too few paired expressions to ever certify “equivalent” at this margin; limited resolution, not evidence of parity</i>" : "")) +
        "</div>" +
        "<div class='fam'>" + pLabel + fmtP(p) + " (uncorrected p = " + fmtP(rec.p_raw) + ")" +
        (rec.family_id
          ? " · confirmatory — pre-declared " + setName(rec.confirmatory_set) + " set, Holm-corrected over " + rec.family_size + " comparisons"
          : " · exploratory — BH-corrected over the " + rec.exploratory_family_size + "-cell matrix") + "</div>" +
        "<div class='fam'>n = " + rec.n_pairs + " paired expressions" +
        (onlySided ? " (+" + onlySided + " solved by one side only)" : "") +
        " · smallest reliably detectable Δ (MDE₈₀) ≈ " + rec.mde_80 +
        " · P(" + cell.rowS + " better on a random expression) = " + o.psup + "</div>" +
        "<div class='fam'>" + (rec.same_configuration
          ? "same configuration on both sides (≈" + o.xRow + " s / ≈" + o.xCol + " s per problem)"
          : "both sides at the budget — " + sideBasis(cell.rowS, o.statusRow, o.xRow, o.bracketRow) +
            " · " + sideBasis(cell.colS, o.statusCol, o.xCol, o.bracketCol)) + "</div>" +
        (rec.notes && rec.notes.length ? "<div class='fam'><i>" + rec.notes.join(" ") + "</i></div>" : "");
    }

    function selectMatrixCell(i) {
      var cell = matrixCells[i];
      if (!cell) { return; }
      var key = cell.rowS + "|" + cell.colS;
      matrixSelectedKey = (matrixSelectedKey === key) ? null : key;   // tap again to unpin
      var detail = pairedFoot.querySelector(".matrix-detail");
      pairedFoot.querySelectorAll("td.selected").forEach(function (td) { td.classList.remove("selected"); });
      if (!detail) { return; }
      if (matrixSelectedKey === null) { detail.innerHTML = ""; return; }
      var td = pairedFoot.querySelector("td[data-cell='" + i + "']");
      if (td) { td.classList.add("selected"); }
      detail.innerHTML = matrixDetailHtml(cell);
      var box = detail.getBoundingClientRect();
      if (box.bottom > window.innerHeight || box.top < 0) {
        detail.scrollIntoView({ block: "nearest", behavior: "smooth" });
      }
    }

    function renderMatrixDescriptive() {
      var metricKey = metricSel.value, bench = benchSel.value;
      var t = selectedBudget();
      var active = series.filter(function (s) { return seriesChecks[s].checked; });
      var curveFor = {};
      (pairedData.curves || []).forEach(function (c) {
        if (c.benchmark === bench && c.metric === metricKey) { curveFor[c.a + "|" + c.b] = c; }
      });
      matrixCells = [];
      var selectedIdx = -1;
      var html = [descBanner()];
      html.push("<div class='matrix-wrap'>");
      html.push("<table class='paired-matrix'><thead><tr><th>row − column</th>");
      active.forEach(function (s) { html.push("<th>" + s + "</th>"); });
      html.push("</tr></thead><tbody>");
      active.forEach(function (rowS) {
        html.push("<tr><th>" + rowS + "</th>");
        active.forEach(function (colS) {
          if (rowS === colS) { html.push("<td class='diag'>—</td>"); return; }
          var curve = curveFor[rowS + "|" + colS] || curveFor[colS + "|" + rowS] || null;
          var sign = curve ? (curve.a === rowS ? 1 : -1) : 0;
          var read = curve ? pairedReadAt(curve, sign, t) : null;
          var cellIdx = matrixCells.length;
          var isSel = matrixSelectedKey === rowS + "|" + colS;
          if (isSel) { selectedIdx = cellIdx; }
          matrixCells.push({ desc: read || { status: "none" }, rowS: rowS, colS: colS });
          var tap = " data-cell='" + cellIdx + "' tabindex='0' role='button' aria-expanded='" + isSel + "'";
          if (!read || read.status !== "in") {
            html.push("<td class='missing" + (isSel ? " selected" : "") + "'" + tap + ">n/a</td>");
            return;
          }
          html.push("<td class='desc" + (isSel ? " selected" : "") + "'" + tap + ">" +
            "<span class='cell-delta'>" + fmtDesc(read.delta) + "</span></td>");
        });
        html.push("</tr>");
      });
      html.push("</tbody></table></div>");
      html.push("<div class='tap-hint'>Tap any cell to pin its read below.</div>");
      if (selectedIdx < 0) { matrixSelectedKey = null; }
      html.push("<div class='matrix-detail'>" +
        (selectedIdx >= 0 ? matrixDetailHtml(matrixCells[selectedIdx]) : "") + "</div>");
      html.push("<div class='matrix-legend'>Cells read row − column: Δ read off the paired " +
        "Δ(t) curve at t ≈ " + Number(t.toPrecision(3)) + " s per problem — descriptive only, " +
        "no verdict colours or confirmatory badges at unmarked times. <i>n/a</i> = t outside " +
        "the pair's measured overlap. Tap a cell for its interval; snap the slider to a mark " +
        "for the verdict matrix.</div>");
      pairedFoot.innerHTML = html.join("");
    }

    function renderMatrix() {
      if (!pairedReady()) { return; }
      var metricKey = metricSel.value, bench = benchSel.value;
      if (!pairedMetricGuard(metricKey)) { return; }
      plot.style.display = "none";
      if (!budgetIsSnapped()) { return renderMatrixDescriptive(); }
      var corrected = correctionSel.value === "corrected";
      var active = series.filter(function (s) { return seriesChecks[s].checked; });
      var recFor = {};
      (pairedData.records || []).forEach(function (r) {
        if (r.benchmark === bench && r.metric === metricKey && r.budget === selectedBudget()) {
          recFor[r.a + "|" + r.b] = r;
        }
      });
      matrixCells = [];
      var selectedIdx = -1;
      var html = [];
      if (!corrected) {
        html.push("<div class='matrix-warning'>⚠ Uncorrected p-values — exploratory browsing only; " +
          "never quote these as claims. Switch back to “corrected”.</div>");
      }
      html.push("<div class='matrix-wrap'>");
      html.push("<table class='paired-matrix'><thead><tr><th>row − column</th>");
      active.forEach(function (s) { html.push("<th>" + s + "</th>"); });
      html.push("</tr></thead><tbody>");
      active.forEach(function (rowS) {
        html.push("<tr><th>" + rowS + "</th>");
        active.forEach(function (colS) {
          if (rowS === colS) { html.push("<td class='diag'>—</td>"); return; }
          var rec = recFor[rowS + "|" + colS] || recFor[colS + "|" + rowS] || null;
          var cellIdx = matrixCells.length;
          matrixCells.push({ rec: rec, rowS: rowS, colS: colS });
          var isSel = matrixSelectedKey === rowS + "|" + colS;
          if (isSel) { selectedIdx = cellIdx; }
          var tap = " data-cell='" + cellIdx + "' tabindex='0' role='button' aria-expanded='" + isSel + "'";
          if (!rec) {
            html.push("<td class='missing" + (isSel ? " selected" : "") + "'" + tap +
              " title='not measurable within this budget, or not evaluated on this benchmark'>n/a</td>");
            return;
          }
          var o = orientRecord(rec, rowS);
          var verdict = o.verdict;
          var delta = o.delta;
          var p = corrected ? (rec.family_id ? rec.p_adj : rec.q_bh) : rec.p_raw;
          var pLabel = corrected ? (rec.family_id ? "Holm p" : "BH q") : "raw p";
          var onlySided = (rec.n_only_a || 0) + (rec.n_only_b || 0);
          var title = rowS + " − " + colS + " [" + bench + "]\n" +
            "Δ = " + fmt(delta) + "   95% CI [" + fmt(o.lo) + ", " + fmt(o.hi) + "]\n" +
            "noise margin ±" + rec.margin + "  →  " + verdictLabel(rec, verdict) + "\n" +
            pLabel + fmtP(p) + "  (uncorrected p = " + fmtP(rec.p_raw) + ")\n" +
            (rec.family_id ? "confirmatory: pre-declared " + setName(rec.confirmatory_set) + " set, Holm-corrected over " + rec.family_size + " comparisons"
                           : "exploratory: BH-corrected over the " + rec.exploratory_family_size + "-cell matrix") +
            "\nn = " + rec.n_pairs + " paired expressions" +
            (onlySided ? " (+" + onlySided + " solved by one side only)" : "") +
            "\nsmallest reliably detectable Δ (MDE₈₀) ≈ " + rec.mde_80 +
            "  ·  P(row better on a random expression) = " + o.psup +
            "\nbudget \u2264 " + rec.budget + "s: " + (rec.same_configuration
              ? "same configuration on both sides (\u2248" + o.xRow + "s / \u2248" + o.xCol + "s)"
              : sideBasis("row", o.statusRow, o.xRow, o.bracketRow) + " \u00b7 " +
                sideBasis("column", o.statusCol, o.xCol, o.bracketCol)) +
            (rec.notes && rec.notes.length ? "\n" + rec.notes.join("\n") : "");
          html.push("<td class='v-" + verdict + (rec.family_id ? " confirmatory" : "") + (isSel ? " selected" : "") +
            "'" + tap + " title='" + title.replace(/'/g, "&#39;") + "'>" +
            "<span class='cell-delta'>" + fmt(delta) + "</span>" +
            "<span class='cell-verdict' aria-label='" + verdict + "'>" + (VERDICT_GLYPHS[verdict] || "") + "</span>" +
            (rec.family_id ? "<span class='cell-badge'>C</span>" : "") + "</td>");
        });
        html.push("</tr>");
      });
      html.push("</tbody></table></div>");
      html.push("<div class='tap-hint'>Tap any cell to pin its full statistical record below.</div>");
      if (selectedIdx < 0) { matrixSelectedKey = null; }   // selected pair no longer on screen
      html.push("<div class='matrix-detail'>" +
        (selectedIdx >= 0 ? matrixDetailHtml(matrixCells[selectedIdx]) : "") + "</div>");
      html.push("<div class='matrix-legend'>" +
        "<span class='v-better'>better ▲</span> / <span class='v-worse'>worse ▼</span> (CI clear of the noise margin) · " +
        "<span class='v-equivalent'>equivalent ≈</span> (CI inside the margin: any difference is smaller than the benchmark can measure) · " +
        "<span class='v-undecided'>undecided ?</span> (hatched fill; not enough data — the full record shows the smallest detectable difference) · " +
        "<i>n/a</i> = not measurable within this budget, or not evaluated on this benchmark. " +
        "A small <b>C</b> marks a confirmatory cell: one of the comparisons declared in advance (each ablation vs its parent, adjacent model sizes, " +
        "and each model size vs each baseline), held to the stricter Holm correction — every other cell is exploratory screening " +
        "with the lenient Benjamini–Hochberg correction; see <a href='#paired'>Paired comparisons</a>. " +
        "Cells read row − column at the selected compute budget; tap or hover any cell to pin its full record below the table. Release " +
        (pairedData.results_release_id || "?") + ".</div>");
      pairedFoot.innerHTML = html.join("");
    }

    function renderTable() {
      if (!pairedReady()) { return; }
      var metricKey = metricSel.value, bench = benchSel.value;
      if (!pairedMetricGuard(metricKey)) { return; }
      plot.style.display = "none";
      var mi = metricInfo(metricKey);
      var hib = !(mi && mi.higher_is_better === false);
      if (!pairedData.leaderboard) {
        pairedFoot.innerHTML = "<div class='matrix-warning'>The Table data is not part of " +
          "this results release yet — check back after the next release.</div>";
        return;
      }
      var snapped = budgetIsSnapped();
      var byMetricSeries = {};
      pairedData.leaderboard.forEach(function (r) {
        if (r.benchmark === bench && r.metric === metricKey && r.budget === selectedBudget()) {
          byMetricSeries[r.series] = r;
        }
      });
      var active = series.filter(function (s) { return seriesChecks[s].checked; });
      var rows = active.map(function (s) {
        if (snapped) { return { s: s, e: byMetricSeries[s] || null }; }
        var r = marginalReadAt(s, bench, metricKey, selectedBudget());
        return { s: s, e: (r && r.status !== "below") ? r : null };
      });
      rows.sort(function (a, b) {
        if (!a.e && !b.e) { return 0; }
        if (!a.e) { return 1; }               // unmeasured rows sink to the bottom
        if (!b.e) { return -1; }
        return hib ? b.e.value - a.e.value : a.e.value - b.e.value;
      });
      var noteIndex = {};                      // note text -> footnote number
      var html = [];
      html.push("<div class='matrix-wrap'>");
      html.push("<table class='paired-matrix abs-table'><thead><tr>" +
        "<th>series</th><th>" + (mi ? mi.label : metricKey) + " [95% CI]</th>" +
        "<th>at the " + (snapped ? "≤ " + selectedBudget() : "t ≈ " + Number(selectedBudget().toPrecision(3))) +
        " s budget</th><th>n</th></tr></thead><tbody>");
      rows.forEach(function (row) {
        var col = colorOf(row.s, idx[row.s]);
        var marks = "";
        (row.e && row.e.notes ? row.e.notes : []).forEach(function (note) {
          if (!(note in noteIndex)) { noteIndex[note] = Object.keys(noteIndex).length + 1; }
          marks += "<sup>" + noteIndex[note] + "</sup>";
        });
        html.push("<tr><th><span class='lb-swatch' style='background:" + col + "'></span>" +
          row.s + marks + "</th>");
        if (!row.e) {
          html.push("<td class='missing' colspan='3'>n/a — cannot run within ≤ " +
            selectedBudget() + " s per problem on this benchmark</td></tr>");
          return;
        }
        var fv = snapped ? fmtAbs : fmtDesc;
        var basis = row.e.status === "plateau"
          ? "ladder ends ≈" + fmtDesc(row.e.x) + " s <span class='lb-flag'>plateau</span>"
          : row.e.status === "interpolated"
            ? (snapped ? "= " + row.e.x + " s (interpolated)"
                       : "≈ " + Number(selectedBudget().toPrecision(3)) + " s (curve read)")
            : "measured ≈" + fmtDesc(row.e.x) + " s";
        html.push("<td class='lb-value'><b>" + fv(row.e.value) + "</b> [" + fv(row.e.lo) +
          ", " + fv(row.e.hi) + "]" + (row.e.status === "plateau" ? " <span class='lb-flag'>plateau</span>" : "") + "</td>" +
          "<td class='lb-x'>" + basis + "</td>" +
          "<td class='lb-n'>" + row.e.n + "</td></tr>");
      });
      html.push("</tbody></table></div>");
      var notes = Object.keys(noteIndex).map(function (note) {
        return "<div class='fam'><sup>" + noteIndex[note] + "</sup> " + note + "</div>";
      }).join("");
      if (!snapped) {
        html.push(descBanner() +
          "<div class='matrix-legend'>" + (hib ? "Higher" : "Lower") + " is better, best first. " +
          "<b>These are marginal numbers: never read a difference between two rows off their " +
          "CIs</b> — head-to-head questions belong to the Paired Δ views (<a href='#paired'>why</a>). " +
          "Rankings are per benchmark only.</div>");
        pairedFoot.innerHTML = html.join("");
        return;
      }
      html.push("<div class='desc-banner'><b>Not a leaderboard:</b> these are marginal numbers " +
        "— overlapping intervals are NOT evidence of no difference, and a small gap between " +
        "rows proves nothing. For any A-vs-B question use the Paired Δ views " +
        "(<a href='#paired'>why</a>).</div>");
      html.push("<div class='matrix-legend'>" +
        "Each row: the series evaluated at exactly the selected budget — interpolated per " +
        "problem, linearly in log-time, between its two bracketing measured configurations " +
        "(the same model as the Δ(t) curves, so the value sits on the plotted segment; " +
        "measured points keep the exact Curves-view value). " +
        "<span class='lb-flag'>plateau</span> = the method's measurements end below the " +
        "budget (its largest tested configuration is shown); the value is carried forward as " +
        "a lower bound, since more compute could only help it — never extrapolated. " +
        (hib ? "Higher" : "Lower") + " is better, best first; rankings are per benchmark " +
        "only — there is deliberately no combined number. Release " +
        (pairedData.results_release_id || "?") + ".</div>" +
        (notes ? "<div class='matrix-legend'>" + notes + "</div>" : ""));
      pairedFoot.innerHTML = html.join("");
    }

    var ranksAutoPicked = false, ranksSwitchedFrom = null;
    var ranksRestoreKey = null, ranksAutoKey = null;
    function rankMetricInfo(key) {
      if (!pairedData || pairedData.error) { return null; }
      return (pairedData.rank_metrics || []).filter(function (m) { return m.key === key; })[0] || null;
    }

    function renderRanks() {
      if (!pairedReady()) { return; }
      var bench = benchSel.value, metricKey = metricSel.value;
      if (!pairedData.ranks) {
        plot.style.display = "none";
        pairedFoot.innerHTML = "<div class='matrix-warning'>The rank leagues are not part of " +
          "this results release yet — check back after the next release.</div>";
        return;
      }
      var rmi = rankMetricInfo(metricKey);
      if (!rmi && !ranksAutoPicked) {
        // entering the view with a non-league metric (e.g. the default vNRR): jump to the
        // primary league once — VISIBLY (a silent control change misattributes the ranking);
        // a deliberate ineligible pick afterwards shows the guard instead
        ranksAutoPicked = true;
        var prim = (pairedData.rank_metrics || []).filter(function (m) { return m.primary; })[0];
        if (prim) {
          ranksSwitchedFrom = (metrics.filter(function (m) { return m.key === metricKey; })[0] || {}).label || metricKey;
          ranksRestoreKey = metricKey;      // put the user's metric back when they leave Ranks
          ranksAutoKey = prim.key;
          metricSel.value = prim.key;
          return renderRanks();
        }
      }
      if (!rmi) {
        plot.style.display = "none";
        pairedFoot.innerHTML = "<div class='matrix-warning'>Rank leagues exist for the " +
          "continuous metrics only — rate metrics take so few values that most methods tie " +
          "on most expressions, and model-internal metrics do not exist for the baselines. " +
          "Pick one of: " + (pairedData.rank_metrics || [])
            .map(function (m) { return m.label; }).join(" · ") + ".</div>";
        return;
      }
      if (!budgetIsSnapped()) {
        plot.style.display = "none";
        pairedFoot.innerHTML = "<div class='desc-banner'>Rank leagues exist at the marked " +
          "budgets only — snap the slider to a mark to see one.</div>";
        return;
      }
      var league = (pairedData.ranks || []).filter(function (r) {
        return r.benchmark === bench && r.metric === metricKey && r.budget === selectedBudget();
      })[0];
      if (!league) {
        plot.style.display = "none";
        pairedFoot.innerHTML = "<div class='matrix-warning'>No league for this metric at this " +
          "budget — fewer than two methods can run within it, or too few expressions have " +
          "values for every method.</div>";
        return;
      }
      plot.style.display = "";
      var t = theme();
      var order = Object.keys(league.mean_ranks).sort(function (a, b) {
        return league.mean_ranks[a] - league.mean_ranks[b];
      });
      var k = order.length;
      var reject = league.friedman_p < league.alpha;
      var shapes = [];
      if (reject) {
        league.cliques.forEach(function (clique) {
          var rs = clique.map(function (m) { return league.mean_ranks[m]; });
          var ys = clique.map(function (m) { return order.indexOf(m); });
          shapes.push({ type: "rect",
            x0: Math.min.apply(null, rs) - 0.06, x1: Math.max.apply(null, rs) + 0.06,
            y0: Math.min.apply(null, ys) - 0.38, y1: Math.max.apply(null, ys) + 0.38,
            fillcolor: hexA(t.ink, 0.06), line: { width: 1, color: hexA(t.ink, 0.25) } });
        });
      }
      // CD ruler above the plot area
      shapes.push({ type: "line", x0: 1, x1: 1 + league.cd, yref: "paper", y0: 1.04, y1: 1.04,
                    line: { color: t.ink, width: 2 } });
      var traces = [{
        x: order.map(function (m) { return league.mean_ranks[m]; }),
        y: order.map(function (m, i) { return i; }),
        type: "scatter", mode: "markers",
        marker: { size: 13,
                  color: order.map(function (m) { return colorOf(m, idx[m] || 0); }),
                  symbol: order.map(function (m) {
                    return league.ladder_limited.indexOf(m) >= 0 ? "circle-open" : "circle"; }),
                  line: { width: 2,
                          color: order.map(function (m) { return colorOf(m, idx[m] || 0); }) } },
        customdata: order.map(function (m) {
          return [league.n_missing[m] || 0,
                  league.ladder_limited.indexOf(m) >= 0 ? " · ladder-limited (worst-case position)" : ""];
        }),
        hovertemplate: "<b>%{text}</b><br>mean rank %{x:.2f} of " + k +
          "<br>%{customdata[0]} failures ranked worst%{customdata[1]}<extra></extra>",
        text: order,
        showlegend: false
      }];
      var layout = {
        font: { family: "Inter, system-ui, sans-serif", color: t.ink, size: 13 },
        margin: { l: 130, r: 30, t: 46, b: 52 },
        xaxis: { title: "mean rank (1 = best of " + k + " methods)",
                 range: [0.7, k + 0.3], gridcolor: t.grid, ticks: "outside", tickcolor: t.grid },
        yaxis: { tickvals: order.map(function (m, i) { return i; }), ticktext: order,
                 autorange: "reversed", gridcolor: t.grid, zeroline: false },
        shapes: shapes,
        annotations: [{ text: "CD = " + league.cd, xref: "x", x: 1 + league.cd / 2,
                        yref: "paper", y: 1.055, yanchor: "bottom", showarrow: false,
                        font: { size: 12, color: t.ink } }],
        paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", hovermode: "closest"
      };
      Plotly.react(plot, traces, layout, { responsive: true, displayModeBar: false });
      var missing = order.filter(function (m) { return (league.n_missing[m] || 0) > 0; })
        .map(function (m) { return m + " (" + league.n_missing[m] + ")"; });
      var excludedNote = league.excluded.length
        ? "<div class='fam'>" + league.excluded.join(" and ") +
          " cannot run within this budget and sit out this league.</div>" : "";
      var switchNote = ranksSwitchedFrom
        ? "<div class='desc-banner'>Showing <b>" + rmi.label + "</b>, the primary league " +
          "metric — " + ranksSwitchedFrom + " has no rank league (rate metrics tie on most " +
          "expressions). Your metric choice comes back when you leave Ranks.</div>" : "";
      pairedFoot.innerHTML =
        switchNote +
        // decode key: the two glyphs and the ruler, right where the eye lands
        "<div class='plot-caption'><span class='key-band'></span> shaded band = no reliable " +
        "rank difference (gap smaller than the critical difference — absence of evidence, not " +
        "equality) · <b>○</b> hollow dot = ladder-limited: measurements end below the budget, " +
        "so the true position could only improve · the CD ruler (top) shows the smallest " +
        "mean-rank gap that counts.</div>" +
        "<div class='matrix-legend'>" +
        "<span class='" + (league.primary ? "rank-primary" : "rank-exploratory") + "'>" +
        (league.primary ? "PRIMARY league" : "exploratory league") + "</span> — " +
        (league.primary
          ? "the one pre-declared rank claim per benchmark and budget. "
          : "browse freely, quote only the primary (log10 FVU, validation) league. ") +
        "Within each of the n = " + league.n_problems + " expressions, the " + k +
        " methods are ranked 1 (best) to " + k + " on <b>" + rmi.label + "</b> at the ≤ " +
        selectedBudget() + " s per problem budget (every method evaluated at exactly t); " +
        "hard and easy expressions count equally, because each hands out the same placings. " +
        (league.mode === "worst-rank"
          ? "A method that fails an expression ranks last on it" +
            (missing.length ? " (failures: " + missing.join(", ") + ")" : "") + "."
          : "<b>Conditional league</b>: only expressions every method solved — a smaller, " +
            "easier population; the property being ranked is undefined for failures.") + " " +
        (reject
          ? "A <a href='#ranks'>Friedman omnibus</a> confirms the spread of mean ranks is " +
            "real before any grouping is drawn; the Nemenyi critical difference corrects for " +
            "comparing all " + (k * (k - 1) / 2) + " pairs at once. A pre-declared Paired " +
            "verdict can legitimately separate two methods that share a band — it asks a " +
            "magnitude question, ranks ask a consistency question (a hair's win counts like " +
            "a mile's)."
          : "<b>The Friedman omnibus does not reject</b> — no rank separations to report; " +
            "positions are shown without groupings.") +
        "<div class='fam'>Full record: Friedman (tie-corrected) χ² = " + league.friedman_chi2 +
        ", p = " + fmtP(league.friedman_p) + " over n = " + league.n_problems +
        " expressions · CD = " + league.cd + " at α = " + league.alpha +
        " · release " + (pairedData.results_release_id || "?") + ".</div>" +
        excludedNote +
        "</div>";
    }

    function renderCurves() {
      var t = theme();
      var axis = axisSel.value, metricKey = metricSel.value, bench = benchSel.value;
      var metric = metrics.filter(function (m) { return m.key === metricKey; })[0];
      var am = AXIS_META[axis] || { label: axis, log: false };
      var traces = [];
      series.forEach(function (s) {
        if (!seriesChecks[s].checked) { return; }
        var col = colorOf(s, idx[s]);
        var pts = records.filter(function (r) {
          return r.series === s && r.axis === axis && r.benchmark === bench &&
                 r[metricKey] && r[metricKey].median !== null && r.x !== null;
        }).sort(function (a, b) { return a.x - b.x; });
        if (!pts.length) { return; }
        var xs = pts.map(function (r) { return r.x; });
        var med = pts.map(function (r) { return r[metricKey].median; });
        var lo = pts.map(function (r) { return r[metricKey].lo; });
        var hi = pts.map(function (r) { return r[metricKey].hi; });
        var version = pts[0].version;
        traces.push({
          x: xs.concat(xs.slice().reverse()), y: hi.concat(lo.slice().reverse()),
          fill: "toself", fillcolor: hexA(col, 0.13), line: { width: 0 },
          hoverinfo: "skip", showlegend: false, legendgroup: s, type: "scatter", mode: "lines"
        });
        traces.push({
          x: xs, y: med, name: s + (version && version !== "-" ? " (" + version + ")" : ""), legendgroup: s,
          line: { color: col, width: 2.5 }, marker: { color: col, size: 7 },
          type: "scatter", mode: "lines+markers",
          customdata: pts.map(function (r) { return [r[metricKey].lo, r[metricKey].hi, r[metricKey].n]; }),
          hovertemplate: "<b>" + s + "</b><br>" + am.label + ": %{x}<br>" +
            metric.label + ": %{y:.3f} [%{customdata[0]:.3f}, %{customdata[1]:.3f}]<br>n=%{customdata[2]}<extra></extra>"
        });
      });
      var layout = {
        font: { family: "Inter, system-ui, sans-serif", color: t.ink, size: 13 },
        margin: { l: 62, r: 20, t: 16, b: 58 },
        xaxis: { title: am.label + (am.unit ? " (" + am.unit + ")" : ""), type: am.log ? "log" : "linear", dtick: am.log ? 1 : undefined,
                 gridcolor: t.grid, zerolinecolor: t.zero, ticks: "outside", tickcolor: t.grid },
        yaxis: { title: metric.label, gridcolor: t.grid, zerolinecolor: t.zero, ticks: "outside", tickcolor: t.grid },
        legend: { orientation: "h", y: -0.22, font: { size: 12 } },
        hovermode: "closest", hoverlabel: { font: { family: "Inter, system-ui, sans-serif" } },
        paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
        annotations: traces.length ? [] : [{ text: "No data for this selection", showarrow: false, font: { size: 15, color: t.ink } }]
      };
      Plotly.react(plot, traces, layout, { responsive: true, displayModeBar: false });
      pairedFoot.innerHTML = traces.length
        ? "<div class='plot-caption'>Each line: one series' <b>" + (metric ? metric.label : metricKey) +
          "</b> (the median of bootstrapped means over expressions); the shaded band is its 95% " +
          "bootstrap confidence interval. Tap or hover a point for its exact values and n. " +
          "<a href='#metrics'>What this metric means</a>.</div>" : "";
    }

    render();
    if (window.matchMedia) {
      var mq = window.matchMedia("(prefers-color-scheme: dark)");
      (mq.addEventListener ? mq.addEventListener.bind(mq, "change") : mq.addListener.bind(mq))(render);
    }
  }

  // --- helpers ---
  function uniqueSorted(a) { return Array.prototype.filter.call(a, function (v, i) { return a.indexOf(v) === i; }).sort(); }
  function selectFrom(pairs) {
    var s = document.createElement("select");
    pairs.forEach(function (p) { var o = document.createElement("option"); o.value = p[0]; o.textContent = p[1]; s.appendChild(o); });
    return s;
  }
  function labelled(text, node) {
    var w = document.createElement("label"); w.className = "results-field";
    var t = document.createElement("span"); t.className = "results-field-label"; t.textContent = text;
    w.appendChild(t); w.appendChild(node); return w;
  }
  function toHex6(c) {
    if (typeof c !== "string") { return "#000000"; }
    if (/^#[0-9a-fA-F]{6}$/.test(c)) { return c.toLowerCase(); }
    if (/^#[0-9a-fA-F]{3}$/.test(c)) { return "#" + c.slice(1).split("").map(function (ch) { return ch + ch; }).join("").toLowerCase(); }
    return "#000000";
  }
  function hexA(hex, a) {
    var h = toHex6(hex); var n = parseInt(h.slice(1), 16);
    return "rgba(" + ((n >> 16) & 255) + "," + ((n >> 8) & 255) + "," + (n & 255) + "," + a + ")";
  }

  if (document.readyState === "loading") { document.addEventListener("DOMContentLoaded", init); }
  else { init(); }
})();
