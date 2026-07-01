/* srbf results explorer — standalone frontend build.
 *
 * Reads window.RESULTS_DATA (emitted by srbf.analysis.export_data + augmented by
 * build_results_data.py): { metrics:[{key,label,higher_is_better,tier}], axes:[...],
 * colors:{series:hex}, series_meta:{series:{group,default_on,is_ablation}}, records:[...] }.
 * Renders a Plotly line + CI-band chart with axis / metric / benchmark / series controls, per-series
 * colour customisation (persisted in a single functional cookie, resettable), and a two-tier metric
 * menu. No server, no PNGs; theme-aware (light/dark).
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

    var controls = document.createElement("div");
    controls.className = "results-controls";
    controls.appendChild(labelled("X-axis", axisSel));
    controls.appendChild(labelled("Metric", metricSel));
    controls.appendChild(labelled("Benchmark", benchSel));
    [axisSel, metricSel, benchSel].forEach(function (s) { s.addEventListener("change", render); });

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

    root.appendChild(controls);
    root.appendChild(seriesArea);
    root.appendChild(tools);
    root.appendChild(plot);

    function theme() {
      var dark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
      return dark ? { grid: "#232838", ink: "#b7bdcc", zero: "#333a4f" } : { grid: "#eceef4", ink: "#4a4f5e", zero: "#d7dae4" };
    }

    function render() {
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
