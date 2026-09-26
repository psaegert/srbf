/* The pages around the explorer: the Progress page (progress.html) and the Protocol at the top of the guide
 * (guide.html). Both read window.RESULTS_V2_SUMMARY, the few kB that scripts/site_export_v2.py writes next to a
 * release's results.js as data/<release>/summary.js, so neither page loads the release's full data. Which methods are
 * finished, in progress and scheduled is the exporter's call (progress_summary), the same one the Results page's
 * progress line shows. Method colours follow the reader's own choices from the explorer (the srbf_colors cookie). */
(function () {
  "use strict";
  var S = window.RESULTS_V2_SUMMARY;
  if (!S) { return; }

  function esc(s) { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/"/g, "&quot;"); }
  function roundNum(v) { if (!isFinite(v)) { return "–"; } var a = Math.abs(v); if (a >= 1000) { return Math.round(v).toLocaleString(); } var t = String(+v.toPrecision(a >= 100 ? 4 : 3)); return t === "-0" ? "0" : t; }
  var COOKIE = "srbf_colors";
  function readCookie() { var m = document.cookie.match(new RegExp("(?:^|; )" + COOKIE + "=([^;]*)")); if (!m) { return {}; } try { return JSON.parse(decodeURIComponent(m[1])) || {}; } catch (e) { return {}; } }
  var userColors = readCookie();
  function ink() { return getComputedStyle(document.documentElement).getPropertyValue("--ink").trim() || "#111"; }
  function colorOf(m) { return userColors[m.key] || (m.ink ? ink() : m.color); }
  // when the release was last refreshed: stored with its offset, shown in the reader's own time zone, with how long ago
  function stamp(rel) {
    var t = rel.updated ? new Date(rel.updated) : null;
    if (!t || isNaN(t.getTime())) { return rel.generated ? "Updated " + esc(rel.generated) + "." : ""; }
    var abs = t.toLocaleString("en-GB", { day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit", timeZoneName: "short" });
    var mins = Math.round((Date.now() - t.getTime()) / 60000), ago;
    if (mins < 1) { ago = "just now"; } else if (mins < 60) { ago = mins + (mins === 1 ? " minute" : " minutes") + " ago"; } else if (mins < 48 * 60) { var h = Math.round(mins / 60); ago = h + (h === 1 ? " hour" : " hours") + " ago"; } else { ago = Math.round(mins / 1440) + " days ago"; }
    return '<time datetime="' + esc(rel.updated) + '">Updated ' + esc(abs) + "</time> (" + ago + ").";
  }

  // ---- the guide's Protocol: the release's own texts ------------------------------------------------------------
  Array.prototype.forEach.call(document.querySelectorAll("[data-fill]"), function (el) {
    var k = el.getAttribute("data-fill");
    var v = k === "counts" ? S.problem_sets + " problem sets, " + S.problems.toLocaleString() + " problems."
      : k === "timing_note" ? S.timing_note : S.release[k];
    var row = el.closest("[data-optional]");
    if (v) { el.textContent = v; } else if (row) { row.hidden = true; }
  });

  // ---- the Progress page ----------------------------------------------------------------------------------------
  var root = document.getElementById("progress-root");
  if (!root) { return; }
  var upd = document.getElementById("progress-updated");
  if (upd) { upd.innerHTML = stamp(S.release); }
  var BY = {};
  S.methods.forEach(function (m) { BY[m.key] = m; });
  function budgetsOf(m) {
    var per = S.progress[m.key] || {};
    return (m.budgets || Object.keys(per).map(Number)).slice().sort(function (a, b) { return a - b; });
  }
  function swatch(m) { return '<span class="v2sw" style="background:' + colorOf(m) + '"></span>'; }
  // A method in progress: a row of blocks for its results (each block fills with the finished share of that budget's
  // runs) and one for its times (a block fills once the budget is timed), one block per budget, with their counts.
  function tile(m) {
    var st = S.status[m.key] || [0, null], per = S.progress[m.key] || {}, tm = S.timing[m.key] || {}, budgets = budgetsOf(m), n = budgets.length;
    var perKnown = Object.keys(per).length > 0;   // data without per-budget counts: the results row is one bar of the total
    var runs = perKnown ? budgets.map(function (b) {
      var c = per[String(b)] || [0, null], share = c[1] ? Math.min(1, c[0] / c[1]) : 0;
      return '<i class="v2seg" title="Budget ' + b.toLocaleString() + ": " + c[0] + " of " + (c[1] == null ? "?" : c[1]) + ' runs finished"><i style="width:' + (100 * share).toFixed(1) + '%"></i></i>';
    }).join("") : '<i class="v2seg" title="' + st[0] + " of " + (st[1] == null ? "?" : st[1]) + ' runs finished"><i style="width:' + (st[1] ? 100 * Math.min(1, st[0] / st[1]) : 0).toFixed(1) + '%"></i></i>';
    var timed = budgets.filter(function (b) { return tm[String(b)] != null; }).length;
    var times = budgets.map(function (b) {
      var s = tm[String(b)];
      return '<i class="v2seg" title="Budget ' + b.toLocaleString() + ": " + (s != null ? roundNum(s) + " s per problem" : "time not measured yet") + '"><i style="width:' + (s != null ? 100 : 0) + '%"></i></i>';
    }).join("");
    return '<div class="v2tile" data-m="' + esc(m.key) + '" style="--n:' + n + '"><b>' + swatch(m) + esc(m.label) + '</b>' +
      '<span class="v2plab">Results</span><span class="v2segs" aria-hidden="true"' + (perKnown ? "" : ' style="--n:1"') + '>' + runs + '</span><span class="v2pnum">' + st[0] + '<small> / ' + (st[1] == null ? "?" : st[1]) + '</small></span>' +
      '<span class="v2plab">Times</span><span class="v2segs" aria-hidden="true">' + times + '</span><span class="v2pnum">' + timed + '<small> / ' + n + '</small></span>' +
      '<span class="v2paxis" aria-hidden="true"><span>' + (n ? budgets[0].toLocaleString() : "") + '</span><span>' + (n > 1 ? budgets[n - 1].toLocaleString() : "") + '</span></span></div>';
  }
  var sum = S.summary, html = "";
  var finished = sum.finished.map(function (k) { return BY[k]; }).filter(Boolean);
  if (finished.length) {
    html += '<div class="v2done"><span class="v2lab">Finished</span>' + finished.map(function (m) {
      var b = budgetsOf(m);
      return '<span class="v2chip" data-m="' + esc(m.key) + '" title="' + esc(m.label + ": both runs on every problem set and a measured time, at every budget from " + b[0].toLocaleString() + " to " + b[b.length - 1].toLocaleString() + ".") + '">' + swatch(m) + esc(m.label) + "</span>";
    }).join("") + "</div>";
  }
  var going = sum.in_progress.map(function (k) { return BY[k]; }).filter(Boolean);
  if (going.length) { html += '<div class="v2strip">' + going.map(tile).join("") + "</div>"; }
  if (sum.scheduled.length) {
    html += '<div class="v2sched"><span class="v2lab">Scheduled</span>' + sum.scheduled.map(function (x) { return '<span class="v2chip" title="' + esc(x.note) + '">' + esc(x.label) + "</span>"; }).join("") + "</div>" +
      '<details class="v2schednotes"><summary>About the scheduled methods</summary><dl>' + sum.scheduled.map(function (x) { return "<dt>" + esc(x.label) + "</dt><dd>" + esc(x.note) + "</dd>"; }).join("") + "</dl></details>";
  }
  root.innerHTML = html || '<p class="v2hint">No method has been planned yet.</p>';
})();
