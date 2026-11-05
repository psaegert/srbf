// The Benchmarks page's filter: one origin (or all) and any number of tags; a row shows when it has the origin and
// every chosen tag. Without this script the filter stays hidden and the list shows every problem set.
(function () {
  var box = document.querySelector(".bm-filter");
  var rows = Array.prototype.slice.call(document.querySelectorAll(".bm-list tbody tr"));
  if (!box || !rows.length) { return; }
  var origin = "";
  var tags = [];
  var count = box.querySelector(".bm-count");
  function fmt(n) { return n.toLocaleString("en-US"); }
  function apply() {
    var sets = 0;
    var problems = 0;
    rows.forEach(function (tr) {
      var have = (tr.getAttribute("data-tags") || "").split("|");
      var show = (!origin || tr.getAttribute("data-origin") === origin) &&
        tags.every(function (t) { return have.indexOf(t) >= 0; });
      tr.hidden = !show;
      if (show) { sets += 1; problems += Number(tr.getAttribute("data-problems")); }
    });
    count.textContent = (sets === rows.length ? "All " : "") + fmt(sets) + (sets === 1 ? " problem set, " : " problem sets, ") +
      fmt(problems) + (problems === 1 ? " problem." : " problems.");
  }
  box.querySelector('[data-filter="origin"]').addEventListener("click", function (e) {
    var b = e.target.closest("button");
    if (!b) { return; }
    origin = b.getAttribute("data-value");
    Array.prototype.forEach.call(this.querySelectorAll("button"), function (x) { x.setAttribute("aria-pressed", String(x === b)); });
    apply();
  });
  box.querySelector('[data-filter="tags"]').addEventListener("click", function (e) {
    var b = e.target.closest("button");
    if (!b) { return; }
    var t = b.getAttribute("data-value");
    var on = b.getAttribute("aria-pressed") !== "true";
    b.setAttribute("aria-pressed", String(on));
    tags = on ? tags.concat([t]) : tags.filter(function (x) { return x !== t; });
    apply();
  });
  box.hidden = false;
  apply();
})();
