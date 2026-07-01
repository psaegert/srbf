/* srbf results explorer — standalone frontend build.
 *
 * Reads window.RESULTS_DATA (emitted by srbf.analysis.export_data -> results_data.js): a flat list of
 * records {series, version, benchmark, axis, x, <metric>: {median, lo, hi, n}}. Renders a Plotly
 * line+CI-band chart and lets the reader pick the x-axis (sweep), metric, benchmark, and series.
 * No server, no PNGs: everything is client-side. Theme-aware (light/dark).
 */
(function () {
  "use strict";

  var AXIS_META = {
    compute: { label: "Inference compute (median fit time)", unit: "s", log: true },
    n_support: { label: "Support size (n samples)", unit: "", log: true },
    noise: { label: "Noise level (relative)", unit: "", log: false }
  };
  var PALETTE = ["#4f46e5", "#f97316", "#16a34a", "#dc2626", "#9333ea", "#0891b2", "#db2777", "#65788a", "#ca8a04", "#0ea5e9"];

  function theme() {
    var dark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
    return dark
      ? { grid: "#232838", ink: "#b7bdcc", zero: "#333a4f" }
      : { grid: "#eceef4", ink: "#4a4f5e", zero: "#d7dae4" };
  }

  function init() {
    var root = document.getElementById("results-explorer");
    if (!root || typeof window.RESULTS_DATA === "undefined") { return; }
    if (typeof Plotly === "undefined") { root.innerHTML = "<div class='loading'>Plotly failed to load.</div>"; return; }
    root.innerHTML = "";

    var data = window.RESULTS_DATA;
    var records = data.records;
    var metrics = data.metrics;                       // [{key, label, higher_is_better}]
    var axes = data.axes;                             // e.g. ["compute", "n_support", "noise"]
    var series = uniqueSorted(records.map(function (r) { return r.series; }));
    var benches = uniqueSorted(records.map(function (r) { return r.benchmark; }));
    var colour = {};
    series.forEach(function (s, i) { colour[s] = PALETTE[i % PALETTE.length]; });

    var axisSel = selectFrom(axes.map(function (a) { return [a, (AXIS_META[a] || {}).label || a]; }));
    var metricSel = selectFrom(metrics.map(function (m) { return [m.key, m.label]; }));
    var benchSel = selectFrom(benches.map(function (b) { return [b, b]; }));
    var seriesBox = document.createElement("div");
    seriesBox.className = "results-series";
    var seriesChecks = {};
    series.forEach(function (s) {
      var lbl = document.createElement("label");
      var cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = /^v23\.0-/.test(s);              // flash-ansr sizes on by default
      cb.addEventListener("change", render);
      seriesChecks[s] = cb;
      var sw = document.createElement("span");
      sw.className = "results-swatch";
      sw.style.background = colour[s];
      sw.style.color = colour[s];
      lbl.appendChild(cb); lbl.appendChild(sw); lbl.appendChild(document.createTextNode(" " + s));
      seriesBox.appendChild(lbl);
    });

    var controls = document.createElement("div");
    controls.className = "results-controls";
    controls.appendChild(labelled("X-axis", axisSel));
    controls.appendChild(labelled("Metric", metricSel));
    controls.appendChild(labelled("Benchmark", benchSel));
    [axisSel, metricSel, benchSel].forEach(function (s) { s.addEventListener("change", render); });

    var plot = document.createElement("div");
    plot.id = "results-plot";
    plot.style.width = "100%";
    plot.style.height = "500px";

    root.appendChild(controls);
    root.appendChild(labelled("Series", seriesBox));
    root.appendChild(plot);

    function render() {
      var t = theme();
      var axis = axisSel.value, metricKey = metricSel.value, bench = benchSel.value;
      var metric = metrics.filter(function (m) { return m.key === metricKey; })[0];
      var am = AXIS_META[axis] || { label: axis, log: false };
      var traces = [];
      series.forEach(function (s) {
        if (!seriesChecks[s].checked) { return; }
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
          x: xs.concat(xs.slice().reverse()),
          y: hi.concat(lo.slice().reverse()),
          fill: "toself", fillcolor: hexA(colour[s], 0.14), line: { width: 0 },
          hoverinfo: "skip", showlegend: false, legendgroup: s, type: "scatter", mode: "lines"
        });
        traces.push({
          x: xs, y: med, name: s + " (" + version + ")", legendgroup: s,
          line: { color: colour[s], width: 2.5, shape: "linear" },
          marker: { color: colour[s], size: 7, line: { width: 0 } },
          type: "scatter", mode: "lines+markers",
          customdata: pts.map(function (r) { return [r[metricKey].lo, r[metricKey].hi, r[metricKey].n]; }),
          hovertemplate: "<b>" + s + "</b><br>" + am.label + ": %{x}<br>" +
            metric.label + ": %{y:.3f} [%{customdata[0]:.3f}, %{customdata[1]:.3f}]<br>" +
            "n=%{customdata[2]}<extra></extra>"
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
  function hexA(hex, a) {
    var n = parseInt(hex.slice(1), 16);
    return "rgba(" + ((n >> 16) & 255) + "," + ((n >> 8) & 255) + "," + (n & 255) + "," + a + ")";
  }

  if (document.readyState === "loading") { document.addEventListener("DOMContentLoaded", init); }
  else { init(); }
})();
