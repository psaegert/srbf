/* srbf results explorer — standalone frontend build.
 *
 * Reads window.RESULTS_DATA (emitted by srbf.analysis.export_data + augmented by
 * build_results_data.py): { metrics:[{key,label,higher_is_better,tier}], axes:[...],
 * colors:{series:hex}, series_meta:{series:{group,default_on,is_ablation,parent,predecessor}},
 * records:[...] }. Three views (registry): Curves (marginal line + CI band), Paired (Δ-vs-baseline
 * curves with pointwise bands + verdict), Matrix (k×k four-state verdict cells). Paired/Matrix data
 * comes from paired_data.json (lazy-fetched; ALL statistics precomputed in Python by srbf's paired
 * layer — this file only renders). Per-series colour customisation persists in a single functional
 * cookie; two-tier metric menu; theme-aware (light/dark).
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

    // --- view registry: Curves | Paired | Matrix ---------------------------------------------
    var VIEWS = {
      curves: { label: "Curves", usesAxis: true, usesBaseline: false, usesCorrection: false },
      paired: { label: "Paired Δ", usesAxis: false, usesBaseline: true, usesCorrection: true, usesBudget: true },
      matrix: { label: "Matrix", usesAxis: false, usesBaseline: false, usesCorrection: true, usesBudget: true }
    };
    var currentView = "curves";
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

    // standardized compute budgets (populated from the paired payload on first load)
    var budgetSel = document.createElement("select");
    function fillBudgets() {
      if (!pairedData || pairedData.error || !pairedData.budgets || budgetSel.options.length) { return; }
      pairedData.budgets.forEach(function (b) {
        var o = document.createElement("option"); o.value = String(b);
        o.textContent = "≤ " + b + " s per expression";
        budgetSel.appendChild(o);
      });
      budgetSel.value = String(pairedData.budgets[pairedData.budgets.length - 1]);
    }
    budgetSel.addEventListener("change", render);
    function selectedBudget() { return parseFloat(budgetSel.value); }

    var tabs = document.createElement("div"); tabs.className = "view-tabs";
    Object.keys(VIEWS).forEach(function (v) {
      var b = document.createElement("button"); b.type = "button"; b.className = "view-tab";
      b.textContent = VIEWS[v].label; b.dataset.view = v;
      if (v === currentView) { b.classList.add("active"); }
      b.addEventListener("click", function () { switchView(v); });
      tabs.appendChild(b);
    });

    var controls = document.createElement("div");
    controls.className = "results-controls";
    var axisField = labelled("X-axis", axisSel);
    var baselineField = labelled("Baseline — every checked series is compared against it", baselineSel);
    var baselineHint = document.createElement("span");
    baselineHint.className = "results-field-hint";
    baselineHint.textContent = "The baseline is the flat zero line. It does not have to be checked above — any series can serve, including hidden and ablation series.";
    baselineField.appendChild(baselineHint);
    var correctionField = labelled("Multiple-comparison correction", correctionSel);
    var budgetField = labelled("Compute budget (verdicts)", budgetSel);
    var budgetHint = document.createElement("span");
    budgetHint.className = "results-field-hint";
    budgetHint.textContent = "Verdicts compare each method's best measured configuration within this budget (median time per expression).";
    budgetField.appendChild(budgetHint);
    controls.appendChild(axisField);
    controls.appendChild(labelled("Metric", metricSel));
    controls.appendChild(labelled("Benchmark", benchSel));
    controls.appendChild(baselineField);
    controls.appendChild(budgetField);
    controls.appendChild(correctionField);
    [axisSel, metricSel, benchSel, baselineSel, correctionSel].forEach(function (s) { s.addEventListener("change", render); });

    function switchView(v) {
      currentView = v;
      tabs.querySelectorAll(".view-tab").forEach(function (b) {
        b.classList.toggle("active", b.dataset.view === v);
      });
      var vm = VIEWS[v];
      axisField.style.display = vm.usesAxis ? "" : "none";
      baselineField.style.display = vm.usesBaseline ? "" : "none";
      budgetField.style.display = vm.usesBudget ? "" : "none";
      correctionField.style.display = vm.usesCorrection ? "" : "none";
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

    root.appendChild(tabs);
    root.appendChild(controls);
    root.appendChild(seriesArea);
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
      plot.style.display = "";
      return renderCurves();
    }

    function pairedReady() {
      if (pairedData && !pairedData.error) { return true; }
      plot.style.display = "";
      Plotly.react(plot, [], {
        annotations: [{ text: pairedLoading ? "Loading paired data…" :
          (pairedData && pairedData.error ? "paired_data.json failed to load." : "Loading paired data…"),
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
    var VERDICT_COLORS = { better: "#16a34a", worse: "#dc2626", equivalent: "#2563eb", undecided: "#8a8f9e" };

    function metricInfo(key) { return (pairedData.metrics || []).filter(function (m) { return m.key === key; })[0]; }

    // payload numbers are pre-rounded to the precision their CI justifies — render them verbatim
    function fmt(v) { if (v === null || v === undefined || v !== v) { return "–"; } var x = (v === 0 ? 0 : v); return (x > 0 ? "+" : "") + String(x); }
    function fmtP(p) { return (p === null || p === undefined) ? "–" : Number(p).toPrecision(2); }
    var VERDICT_GLYPHS = { better: "▲", worse: "▼", equivalent: "≈", undecided: "?" };
    function setName(key) { return ({ ablations: "ablation vs parent", size_ladder: "adjacent model sizes", baselines: "model sizes vs baselines" })[key] || key; }

    function pairedMetricGuard(metricKey) {
      if (metricInfo(metricKey)) { return true; }
      plot.style.display = "";
      Plotly.react(plot, [], {
        annotations: [{ text: "Paired data is available for the Main metrics — pick one from the “Main metrics” group in the Metric menu.",
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
        var rec = (pairedData.records || []).filter(function (r) {
          return r.benchmark === bench && r.metric === metricKey && r.budget === selectedBudget() &&
                 orient(r, s, baseline) !== 0;
        })[0];
        if (rec) {
          var rSign = orient(rec, s, baseline);
          var verdict = rSign > 0 ? rec.verdict : flipVerdict(rec.verdict);
          var corrected = correctionSel.value === "corrected";
          var p = corrected ? (rec.p_adj !== undefined ? rec.p_adj : rec.q_bh) : rec.p_raw;
          var pLabel = corrected ? (rec.p_adj !== undefined ? "Holm-corrected p = " : "BH-corrected q = ") : "raw p = ";
          var onlySided = (rec.n_only_a || 0) + (rec.n_only_b || 0);
          notes.push("<b>" + s + "</b> vs " + baseline + ": " +
            "<span style='color:" + (VERDICT_COLORS[verdict] || "#888") + "'><b>" + verdict + "</b></span> — " +
            "Δ = " + fmt(rSign * rec.delta) + " [" + fmt(rSign > 0 ? rec.lo : -rec.hi) + ", " +
            fmt(rSign > 0 ? rec.hi : -rec.lo) + "] vs noise margin ±" + rec.margin +
            (verdict === "undecided" && rec.equivalence_attainable === false
              ? " — <i>too few paired expressions to ever certify “equivalent” at this margin; limited resolution, not evidence of parity</i>" : "") +
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
        xaxis: { title: "Compute budget t (s, log scale) — each series' median time per expression · solid = measured, hollow = interpolated",
                 type: "log", gridcolor: t.grid, zerolinecolor: t.zero, ticks: "outside", tickcolor: t.grid },
        yaxis: { title: "Δ " + (mi ? mi.label : metricKey) + "  (selected − baseline; " +
                 (mi && mi.higher_is_better === false ? "below" : "above") + " 0 favors selected)",
                 gridcolor: t.grid, zerolinewidth: 2, zerolinecolor: t.ink, ticks: "outside", tickcolor: t.grid },
        legend: { orientation: "h", y: -0.22, font: { size: 12 } },
        hovermode: "closest", paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
        shapes: traces.length ? [{ type: "line", x0: selectedBudget(), x1: selectedBudget(),
          yref: "paper", y0: 0, y1: 1, line: { color: t.ink, width: 1, dash: "dot" } }] : [],
        annotations: traces.length ? [] : [{ text: (pairedData.records || []).some(function (r) { return r.benchmark === bench && (r.a === baseline || r.b === baseline); })
          ? "No paired curves for this selection — toggle series above (the baseline itself is the zero line)."
          : "No precomputed pairs for baseline " + baseline + " on " + bench + " — pick another baseline or benchmark.",
          showarrow: false, font: { size: 14, color: t.ink } }]
      };
      Plotly.react(plot, traces, layout, { responsive: true, displayModeBar: false });
      var rawWarn = correctionSel.value !== "corrected"
        ? "<div class='matrix-warning'>⚠ Uncorrected p-values — exploratory browsing only; never quote these as claims. Switch back to “corrected”.</div>" : "";
      pairedFoot.innerHTML = notes.length
        ? rawWarn + "<div class='paired-verdicts'><div class='paired-verdicts-title'>Verdicts at the standardized budget of ≤ " + selectedBudget() + " s per expression (dotted line) — " +
          "each method's best measured configuration within it (95% CIs · release " +
          (pairedData.results_release_id || "?") + " · α = " + (pairedData.alpha || 0.05) + ")</div>" +
          notes.map(function (n) { return "<div class='paired-verdict-row'>" + n + "</div>"; }).join("") + "</div>"
        : rawWarn;
    }

    function renderMatrix() {
      if (!pairedReady()) { return; }
      var metricKey = metricSel.value, bench = benchSel.value;
      if (!pairedMetricGuard(metricKey)) { return; }
      plot.style.display = "none";
      var corrected = correctionSel.value === "corrected";
      var active = series.filter(function (s) { return seriesChecks[s].checked; });
      var recFor = {};
      (pairedData.records || []).forEach(function (r) {
        if (r.benchmark === bench && r.metric === metricKey && r.budget === selectedBudget()) {
          recFor[r.a + "|" + r.b] = r;
        }
      });
      var html = ["<div class='matrix-wrap'>"];
      if (!corrected) {
        html.push("<div class='matrix-warning'>⚠ Uncorrected p-values — exploratory browsing only; " +
          "never quote these as claims. Switch back to “corrected”.</div>");
      }
      html.push("<table class='paired-matrix'><thead><tr><th>row − column</th>");
      active.forEach(function (s) { html.push("<th>" + s + "</th>"); });
      html.push("</tr></thead><tbody>");
      active.forEach(function (rowS) {
        html.push("<tr><th>" + rowS + "</th>");
        active.forEach(function (colS) {
          if (rowS === colS) { html.push("<td class='diag'>—</td>"); return; }
          var rec = recFor[rowS + "|" + colS] || recFor[colS + "|" + rowS];
          if (!rec) { html.push("<td class='missing' title='not measurable within this budget, or not evaluated on this benchmark'>n/a</td>"); return; }
          var sign = rec.a === rowS ? 1 : -1;
          var verdict = sign > 0 ? rec.verdict : flipVerdict(rec.verdict);
          var delta = (sign * rec.delta);
          var p = corrected ? (rec.family_id ? rec.p_adj : rec.q_bh) : rec.p_raw;
          var pLabel = corrected ? (rec.family_id ? "Holm p" : "BH q") : "raw p";
          var onlySided = (rec.n_only_a || 0) + (rec.n_only_b || 0);
          var title = rowS + " − " + colS + " [" + bench + "]\n" +
            "Δ = " + fmt(delta) + "   95% CI [" + fmt(sign > 0 ? rec.lo : -rec.hi) + ", " +
            fmt(sign > 0 ? rec.hi : -rec.lo) + "]\n" +
            "noise margin ±" + rec.margin + "  →  " + verdict + "\n" +
            pLabel + fmtP(p) + "  (uncorrected p = " + fmtP(rec.p_raw) + ")\n" +
            (rec.family_id ? "confirmatory: pre-declared " + setName(rec.confirmatory_set) + " set, Holm-corrected over " + rec.family_size + " comparisons"
                           : "exploratory: BH-corrected over the " + rec.exploratory_family_size + "-cell matrix") +
            "\nn = " + rec.n_pairs + " paired expressions" +
            (onlySided ? " (+" + onlySided + " solved by one side only)" : "") +
            "\nsmallest reliably detectable Δ (MDE₈₀) ≈ " + rec.mde_80 +
            "  ·  P(row better on a random expression) = " + rec.prob_superiority +
            "\nbudget \u2264 " + rec.budget + "s: " + (rec.same_configuration
              ? "same configuration on both sides (\u2248" + rec.x_a + "s / \u2248" + rec.x_b + "s)"
              : "each side\u2019s best within budget (row \u2248" + rec.x_a + "s \u00b7 column \u2248" + rec.x_b + "s)") +
            (rec.notes && rec.notes.length ? "\n" + rec.notes.join("\n") : "");
          html.push("<td class='v-" + verdict + (rec.family_id ? " confirmatory" : "") +
            "' title='" + title.replace(/'/g, "&#39;") + "'>" +
            "<span class='cell-delta'>" + fmt(delta) + "</span>" +
            "<span class='cell-verdict' aria-label='" + verdict + "'>" + (VERDICT_GLYPHS[verdict] || "") + "</span>" +
            (rec.family_id ? "<span class='cell-badge'>C</span>" : "") + "</td>");
        });
        html.push("</tr>");
      });
      html.push("</tbody></table>");
      html.push("<div class='matrix-legend'>" +
        "<span class='v-better'>better ▲</span> / <span class='v-worse'>worse ▼</span> (CI clear of the noise margin) · " +
        "<span class='v-equivalent'>equivalent ≈</span> (CI inside the margin: any difference is smaller than the benchmark can measure) · " +
        "<span class='v-undecided'>undecided ?</span> (hatched fill; not enough data — hover for the smallest detectable difference) · " +
        "<i>n/a</i> = this pair was not evaluated on this benchmark. " +
        "<b>C</b> (blue outline) = confirmatory cell: one of the comparisons declared in advance (each ablation vs its parent, adjacent model sizes, " +
        "and each model size vs each baseline), held to the stricter Holm correction — every other cell is exploratory screening " +
        "with the lenient Benjamini–Hochberg correction; see <a href='#paired'>Paired comparisons</a>. " +
        "Cells read row − column at the selected compute budget; hover any cell for its full record. Release " +
        (pairedData.results_release_id || "?") + ".</div>");
      html.push("</div>");
      pairedFoot.innerHTML = html.join("");
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
        xaxis: { title: am.label + (am.unit ? " (" + am.unit + ")" : ""), type: am.log ? "log" : "linear",
                 gridcolor: t.grid, zerolinecolor: t.zero, ticks: "outside", tickcolor: t.grid },
        yaxis: { title: metric.label, gridcolor: t.grid, zerolinecolor: t.zero, ticks: "outside", tickcolor: t.grid },
        legend: { orientation: "h", y: -0.22, font: { size: 12 } },
        hovermode: "closest", hoverlabel: { font: { family: "Inter, system-ui, sans-serif" } },
        paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
        annotations: traces.length ? [] : [{ text: "No data for this selection", showarrow: false, font: { size: 15, color: t.ink } }]
      };
      Plotly.react(plot, traces, layout, { responsive: true, displayModeBar: false });
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
