/* The header's theme toggle: Auto follows the device; Light and Dark are explicit overrides, stored in ONE
 * localStorage entry that is set only on an explicit choice (Auto stores nothing). A pre-paint script in index.html
 * applies the stored choice before the first render. Charts read the theme from CSS tokens, so they follow on their
 * own; a "srbf-theme" event is dispatched for anything that must redraw. */
(function () {
  "use strict";
  var THEME_KEY = "srbf_theme";
  function themeMode() {
    try { var t = localStorage.getItem(THEME_KEY); return (t === "light" || t === "dark") ? t : "auto"; }
    catch (e) { return "auto"; }
  }
  function effectiveDark() {
    var attr = document.documentElement.getAttribute("data-theme");
    if (attr === "dark") { return true; }
    if (attr === "light") { return false; }
    return !!(window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches);
  }
  function applyTheme(mode) {
    try {
      if (mode === "auto") { localStorage.removeItem(THEME_KEY); }
      else { localStorage.setItem(THEME_KEY, mode); }
    } catch (e) { /* storage blocked: the choice still applies to this page view */ }
    if (mode === "auto") { document.documentElement.removeAttribute("data-theme"); }
    else { document.documentElement.setAttribute("data-theme", mode); }
    document.dispatchEvent(new CustomEvent("srbf-theme"));
  }
  function initThemeToggle() {
    var btn = document.getElementById("theme-toggle");
    if (!btn) { return; }
    var ORDER = ["auto", "dark", "light"];
    var ICONS = {
      auto: "<svg viewBox='0 0 24 24' width='15' height='15' aria-hidden='true'>" +
        "<circle cx='12' cy='12' r='8.6' fill='none' stroke='currentColor' stroke-width='1.8'/>" +
        "<path fill='currentColor' d='M12 3.4a8.6 8.6 0 0 1 0 17.2Z'/></svg>",
      dark: "<svg viewBox='0 0 24 24' width='15' height='15' aria-hidden='true'>" +
        "<path fill='currentColor' d='M20.5 14.6A8.6 8.6 0 0 1 9.4 3.5a8.6 8.6 0 1 0 11.1 11.1Z'/></svg>",
      light: "<svg viewBox='0 0 24 24' width='15' height='15' aria-hidden='true'>" +
        "<circle cx='12' cy='12' r='4.6' fill='currentColor'/>" +
        "<g stroke='currentColor' stroke-width='1.8' stroke-linecap='round'>" +
        "<line x1='12' y1='2.5' x2='12' y2='5.2'/><line x1='12' y1='18.8' x2='12' y2='21.5'/>" +
        "<line x1='2.5' y1='12' x2='5.2' y2='12'/><line x1='18.8' y1='12' x2='21.5' y2='12'/>" +
        "<line x1='5.3' y1='5.3' x2='7.2' y2='7.2'/><line x1='16.8' y1='16.8' x2='18.7' y2='18.7'/>" +
        "<line x1='5.3' y1='18.7' x2='7.2' y2='16.8'/><line x1='16.8' y1='7.2' x2='18.7' y2='5.3'/></g></svg>"
    };
    var NAMES = { auto: "Auto", dark: "Dark", light: "Light" };
    var HINTS = {
      auto: "Theme: Auto, follows your device. Tap for dark.",
      dark: "Theme: Dark. Tap for light.",
      light: "Theme: Light. Tap for auto."
    };
    function show(mode) {
      btn.innerHTML = ICONS[mode] + "<span class='theme-name'>" + NAMES[mode] + "</span>";
      btn.title = HINTS[mode];
      btn.setAttribute("aria-label", HINTS[mode]);
    }
    btn.addEventListener("click", function () {
      var next = ORDER[(ORDER.indexOf(themeMode()) + 1) % ORDER.length];
      applyTheme(next);
      show(next);
    });
    show(themeMode());
  }
  if (document.readyState === "loading") { document.addEventListener("DOMContentLoaded", initThemeToggle); }
  else { initThemeToggle(); }
})();
