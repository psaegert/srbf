// Functionality suite for the results explorer. Philosophy: general quality gates, not
// pixel-diff snapshots — every view renders its primary content, every view TRANSITION lands
// cleanly (the class of bug screenshots of single states cannot catch), core interactions
// work, no state presents as an empty/ghost plot, and no page state produces console errors.
import { test, expect } from '@playwright/test';

const VIEWS = ['curves', 'paired', 'table', 'matrix', 'ranks'];

// how to reach a view via the Display × Values toggles
const TOGGLES = {
  curves: ['curves', 'absolute'],
  paired: ['curves', 'paired'],
  table: ['table', 'absolute'],
  matrix: ['table', 'paired'],
  ranks: ['ranks', null],
};

// what "this view rendered" means, per view
const CONTENT = {
  curves: { locator: '#results-plot .main-svg', plotVisible: true },
  paired: { locator: '#results-plot .main-svg', plotVisible: true },
  table: { locator: 'table.abs-table tbody tr', plotVisible: false },
  matrix: { locator: 'table.paired-matrix:not(.abs-table) tbody tr', plotVisible: false },
  ranks: { locator: '.rank-primary, .rank-exploratory', plotVisible: true },
};

function collectErrors(page) {
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  page.on('console', (msg) => { if (msg.type() === 'error') { errors.push(msg.text()); } });
  return errors;
}

async function gotoView(page, view) {
  const extra = view === 'paired' ? '&baseline=v23.0-120M' : '';
  await page.goto(`/?view=${view}&bench=FastSRB${extra}`);
  await expectViewRendered(page, view);
}

async function switchTo(page, view) {
  const [display, values] = TOGGLES[view];
  await page.click(`.view-tab[data-axis="display"][data-value="${display}"]`);
  if (values) { await page.click(`.view-tab[data-axis="values"][data-value="${values}"]`); }
  await expectViewRendered(page, view);
}

async function expectViewRendered(page, view) {
  const spec = CONTENT[view];
  await expect(page.locator(spec.locator).first()).toBeVisible({ timeout: 20_000 });
  const plot = page.locator('#results-plot');
  if (spec.plotVisible) {
    await expect(plot).toBeVisible();
  } else {
    // the ghost-plot regression (ranks -> table, 2026-07-04): a table-style view must
    // NEVER leave the plot div visible
    await expect(plot).toBeHidden();
  }
}

function metricSelect(page) {
  return page.locator('select:has(option[value="numeric_recovery_val"])');
}

// ---- every view renders from a deep link -----------------------------------------------

for (const view of VIEWS) {
  test(`deep link renders: ${view}`, async ({ page }) => {
    const errors = collectErrors(page);
    await gotoView(page, view);
    expect(errors).toEqual([]);
  });
}

// ---- the full ordered view-transition matrix -------------------------------------------

for (const from of VIEWS) {
  for (const to of VIEWS) {
    if (from === to) { continue; }
    test(`transition: ${from} -> ${to}`, async ({ page }) => {
      const errors = collectErrors(page);
      await gotoView(page, from);
      await switchTo(page, to);
      expect(errors).toEqual([]);
    });
  }
}

// ---- budget slider: marks are release-grade, free positions are descriptive -------------

test('slider: free position shows the descriptive state, tick click snaps back', async ({ page }) => {
  const errors = collectErrors(page);
  await gotoView(page, 'table');
  await page.evaluate(() => {
    const r = document.querySelector('.budget-slider input[type=range]');
    r.value = '500'; r.dispatchEvent(new Event('input'));
  });
  await expect(page.locator('.budget-value')).toContainText('t ≈');
  await expect(page.locator('.desc-banner').first()).toContainText('Descriptive slice');
  await page.click('.budget-tick >> nth=2');
  await expect(page.locator('.budget-value')).toContainText('≤ 100 s');
  await expect(page.locator('table.abs-table th', { hasText: '≤ 100 s' })).toBeVisible();
  expect(errors).toEqual([]);
});

test('slider: Ranks is snap-only (no dead free positions)', async ({ page }) => {
  const errors = collectErrors(page);
  await gotoView(page, 'ranks');
  await page.evaluate(() => {
    const r = document.querySelector('.budget-slider input[type=range]');
    r.value = '500'; r.dispatchEvent(new Event('input'));
  });
  await expect(page.locator('.budget-value')).toContainText('≤');
  await expectViewRendered(page, 'ranks');
  expect(errors).toEqual([]);
});

// ---- matrix: tap-to-pin the full record --------------------------------------------------

test('matrix: tapping a cell pins its record, tapping again unpins', async ({ page }) => {
  const errors = collectErrors(page);
  await gotoView(page, 'matrix');
  await page.click('td[data-cell="2"]');
  await expect(page.locator('.matrix-detail')).toContainText('noise margin');
  await page.click('td[data-cell="2"]');
  await expect(page.locator('.matrix-detail')).toBeEmpty();
  expect(errors).toEqual([]);
});

// ---- metric eligibility guards ------------------------------------------------------------

test('table: a metric without paired data shows a banner, never a ghost plot', async ({ page }) => {
  const errors = collectErrors(page);
  await gotoView(page, 'table');
  await metricSelect(page).selectOption('log10_fvu_val');
  await expect(page.locator('.matrix-warning')).toContainText('Main metrics');
  await expect(page.locator('#results-plot')).toBeHidden();
  expect(errors).toEqual([]);
});

test('ranks: entering with vNRR switches to the primary metric VISIBLY and restores on leave', async ({ page }) => {
  const errors = collectErrors(page);
  await gotoView(page, 'ranks');
  await expect(metricSelect(page)).toHaveValue('log10_fvu_val');
  await expect(page.locator('.desc-banner')).toContainText('primary ranking metric');
  await switchTo(page, 'table');                       // the 2026-07-04 regression path
  await expect(metricSelect(page)).toHaveValue('numeric_recovery_val');
  expect(errors).toEqual([]);
});

test('ranks: the metric menu shows the league structure at first glance', async ({ page }) => {
  await gotoView(page, 'ranks');
  await expect(metricSelect(page)
    .locator('optgroup[label="Primary"] option[value="log10_fvu_val"]')).toHaveCount(1);
  expect(await metricSelect(page)
    .locator('optgroup[label="Exploratory"] option').count()).toBeGreaterThan(5);
  await expect(page.locator('.metric-badge')).toHaveText('primary');
  await gotoView(page, 'curves');
  await expect(metricSelect(page).locator('optgroup[label="Main metrics"]')).toHaveCount(1);
  await expect(page.locator('.metric-badge')).toBeHidden();
});

test('ranks: ineligible metrics are disabled in the menu', async ({ page }) => {
  await gotoView(page, 'ranks');
  expect(await metricSelect(page).locator('option[value="numeric_recovery_val"]')
    .evaluate((o) => o.disabled)).toBe(true);
  await gotoView(page, 'curves');
  expect(await metricSelect(page).locator('option[value="numeric_recovery_val"]')
    .evaluate((o) => o.disabled)).toBe(false);
});

// ---- benchmark switch keeps every view alive ----------------------------------------------

for (const view of ['table', 'matrix', 'ranks']) {
  test(`benchmark switch stays functional: ${view}`, async ({ page }) => {
    const errors = collectErrors(page);
    await gotoView(page, view);
    await page.locator('select:has(option[value="v23-val"])').selectOption('v23-val');
    await expectViewRendered(page, view);
    expect(errors).toEqual([]);
  });
}

// ---- layout sanity: the page must never scroll horizontally -------------------------------

for (const view of VIEWS) {
  test(`no horizontal page overflow: ${view}`, async ({ page }) => {
    await gotoView(page, view);
    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    expect(overflow).toBeLessThanOrEqual(1);
  });
}

// ---- contextual help: ?-links, collapsed fine print, tap-to-define terms -------------------

test('help: the ?-link opens a hint with a Read-more link, no jump', async ({ page }) => {
  await gotoView(page, 'curves');
  await page.locator('.help-link[href="#metrics"]').click();
  await expect(page.locator('.term-pop')).toContainText('defined precisely');
  await expect(page.locator('.term-pop a[href="#metrics"]')).toContainText('Read more');
  expect(new URL(page.url()).hash).not.toBe('#metrics');   // no navigation happened
  await page.locator('h1').click();
  await expect(page.locator('.term-pop')).toHaveCount(0);
});

test('help: legends open with a key line, fine print collapsed by default', async ({ page }) => {
  await gotoView(page, 'matrix');
  const fine = page.locator('.matrix-legend details.fine-print');
  await expect(fine).toBeVisible();
  expect(await fine.evaluate((d) => d.open)).toBe(false);
  await fine.locator('summary').click();
  expect(await fine.evaluate((d) => d.open)).toBe(true);
});

test('help: tapping a term shows its one-sentence definition', async ({ page }) => {
  await gotoView(page, 'ranks');
  await page.locator('.term[data-term="cd"]').first().click();
  await expect(page.locator('.term-pop')).toContainText('critical difference');
  await page.locator('h1').click();          // click elsewhere closes it
  await expect(page.locator('.term-pop')).toHaveCount(0);
});

// ---- theme: Auto by default, manual override, persistence ---------------------------------

const DARK_BG = 'rgb(12, 14, 20)';     // --bg #0c0e14
const LIGHT_BG = 'rgb(246, 247, 251)'; // --bg #f6f7fb

function bodyBg(page) {
  return page.evaluate(() => getComputedStyle(document.body).backgroundColor);
}

test('theme: Auto follows the device scheme and stores nothing', async ({ page }) => {
  const errors = collectErrors(page);
  await page.emulateMedia({ colorScheme: 'dark' });
  await gotoView(page, 'curves');
  expect(await bodyBg(page)).toBe(DARK_BG);
  expect(await page.evaluate(() => localStorage.getItem('srbf_theme'))).toBe(null);
  await page.emulateMedia({ colorScheme: 'light' });
  await expect.poll(() => bodyBg(page)).toBe(LIGHT_BG);
  expect(errors).toEqual([]);
});

test('theme: the toggle forces dark on a light device, persists, and cycles back to Auto', async ({ page }) => {
  const errors = collectErrors(page);
  await page.emulateMedia({ colorScheme: 'light' });
  await gotoView(page, 'curves');
  await page.click('#theme-toggle');                       // Auto -> Dark
  await expect.poll(() => bodyBg(page)).toBe(DARK_BG);
  expect(await page.evaluate(() => localStorage.getItem('srbf_theme'))).toBe('dark');
  await page.reload();                                     // pre-paint script applies it
  await expectViewRendered(page, 'curves');
  expect(await bodyBg(page)).toBe(DARK_BG);
  await page.click('#theme-toggle');                       // Dark -> Light
  await expect.poll(() => bodyBg(page)).toBe(LIGHT_BG);
  await page.click('#theme-toggle');                       // Light -> Auto (entry removed)
  expect(await page.evaluate(() => localStorage.getItem('srbf_theme'))).toBe(null);
  expect(errors).toEqual([]);
});

test('theme: the visual abstract follows the manual override', async ({ page }) => {
  await page.emulateMedia({ colorScheme: 'light' });
  await gotoView(page, 'curves');
  await page.click('#theme-toggle');
  await expect.poll(() => page.locator('#va rect.bg')
    .evaluate((r) => getComputedStyle(r).fill)).toBe(DARK_BG);
});

test('theme: every view survives a toggle without errors', async ({ page }) => {
  const errors = collectErrors(page);
  for (const view of VIEWS) {
    await gotoView(page, view);
    await page.click('#theme-toggle');                     // -> dark: Plotly re-renders
    await expectViewRendered(page, view);
    await page.click('#theme-toggle');                     // -> light
    await page.click('#theme-toggle');                     // -> auto (clean for next view)
  }
  expect(errors).toEqual([]);
});

// ---- regressions from user phone reports (2026-07-04) --------------------------------------

test('help: term hints render as soft prose, not label styling (ALL-CAPS regression)', async ({ page }) => {
  await gotoView(page, 'ranks');
  await page.locator('.term[data-term="cd"]').first().click();
  const style = await page.locator('.term-pop').evaluate((el) => {
    const s = getComputedStyle(el);
    return { weight: s.fontWeight, transform: s.textTransform, spacing: s.letterSpacing };
  });
  expect(style.weight).toBe('400');
  expect(style.transform).toBe('none');
  expect(style.spacing).toBe('normal');
});

test('mobile: toggles keep headings, share one row with a divider, slider spans the panel', async ({ page }) => {
  test.skip(page.viewportSize().width > 560, 'compact controls only apply below 560px');
  await gotoView(page, 'table');
  const layout = await page.evaluate(() => {
    const toggles = document.querySelectorAll('.view-toggle');
    const a = toggles[0].getBoundingClientRect();
    const b = toggles[1].getBoundingClientRect();
    const labelsVisible = [...document.querySelectorAll('.view-toggle .results-field-label')]
      .every((l) => l.offsetHeight > 0);
    const slider = document.querySelector('.budget-slider').getBoundingClientRect();
    const panel = document.querySelector('.results-controls').getBoundingClientRect();
    return { sameRow: Math.abs(a.top - b.top) < 2, labelsVisible,
             divider: getComputedStyle(toggles[1]).borderLeftWidth,
             sliderFill: slider.width / panel.width };
  });
  expect(layout.sameRow).toBe(true);
  expect(layout.labelsVisible).toBe(true);
  expect(layout.divider).toBe('1px');
  expect(layout.sliderFill).toBeGreaterThan(0.97);
});

// ---- Curves × Paired: the baseline pill is the parked reference ----------------------------

test('paired: the baseline pill is marked, parked, follows the selector, and unparks on leave', async ({ page }) => {
  const errors = collectErrors(page);
  await gotoView(page, 'paired');
  const basePill = page.locator('.series-pill.is-baseline');
  await expect(basePill).toHaveCount(1);
  await expect(basePill.locator('.series-name')).toHaveText('v23.0-120M');
  await expect(basePill.locator('.baseline-tag')).toBeVisible();
  expect(await basePill.locator('input[type=checkbox]').evaluate((c) => c.disabled)).toBe(true);
  await page.locator('select:has(option[value="PySR"])').last().selectOption('PySR');
  await expect(basePill.locator('.series-name')).toHaveText('PySR');
  await switchTo(page, 'table');
  await expect(page.locator('.series-pill.is-baseline')).toHaveCount(0);
  expect(await page.evaluate(() =>
    [...document.querySelectorAll('.series-pill input[type=checkbox]')].some((c) => c.disabled))).toBe(false);
  expect(errors).toEqual([]);
});

test('paired: the zero line carries the baseline colour and an explicit label', async ({ page }) => {
  await gotoView(page, 'paired');
  await expect(page.locator('#results-plot .annotation-text', { hasText: 'baseline' }).first())
    .toBeVisible({ timeout: 15_000 });
});

// ---- table: interpolated / curve read are tap-to-define -----------------------------------

test('table: the interpolated status explains itself on tap', async ({ page }) => {
  await gotoView(page, 'table');
  await page.click('.budget-tick >> nth=2');                       // ≤ 100 s: interpolated rows exist
  const t = page.locator('.term[data-term="interpolated"]').first();
  await expect(t).toBeVisible();
  await t.click();
  await expect(page.locator('.term-pop')).toContainText('straight line');
});

// ---- hints float: revealed content must never push the layout around ----------------------

// document-relative layout box: immune to the page scrolling between measurements
function layoutBox(page, selector) {
  return page.evaluate((sel) => {
    const r = document.querySelector(sel).getBoundingClientRect();
    return { top: r.top + window.scrollY, height: r.height };
  }, selector);
}

test('help: popovers float and do not reflow the table they are opened from', async ({ page }) => {
  await gotoView(page, 'table');
  await page.click('.budget-tick >> nth=2');
  const before = await layoutBox(page, 'table.abs-table');
  await page.locator('.term[data-term="interpolated"]').first().click();
  const pop = page.locator('.term-pop');
  await expect(pop).toBeVisible();
  expect(await pop.evaluate((el) => getComputedStyle(el).position)).toBe('fixed');
  const box = await pop.boundingBox();
  const vp = page.viewportSize();
  expect(box.x).toBeGreaterThanOrEqual(0);
  expect(box.x + box.width).toBeLessThanOrEqual(vp.width + 1);
  const after = await layoutBox(page, 'table.abs-table');
  expect(after.height).toBe(before.height);   // nothing was squeezed
  expect(after.top).toBe(before.top);         // nothing was pushed
});

test('help: a floating popover dismisses on scroll', async ({ page }) => {
  await gotoView(page, 'ranks');
  await page.locator('.term[data-term="cd"]').first().click();
  await expect(page.locator('.term-pop')).toBeVisible();
  await page.waitForTimeout(100);                       // let the dismiss handler arm
  await page.evaluate(() => window.scrollBy(0, 120));   // wheel does not scroll in touch emulation
  await expect(page.locator('.term-pop')).toHaveCount(0);
});

// ---- config provenance: the fairness labels reach every surface ---------------------------

const PROVENANCE_VALUES = ['upstream_default', 'author_blessed', 'harness_tuned'];

test('provenance: every series carries a valid config_provenance label', async ({ request }) => {
  const data = await (await request.get('/results_data.json')).json();
  expect(data.schema_version).toBeGreaterThanOrEqual(3);
  const entries = Object.entries(data.series_meta);
  expect(entries.length).toBeGreaterThan(20);
  for (const [series, m] of entries) {
    expect(PROVENANCE_VALUES, `${series} label`).toContain(m.config_provenance);
  }
});

test('provenance: the About section explains the three labels', async ({ page }) => {
  await page.goto('/');
  const about = page.locator('#about');
  await expect(about).toContainText('Who chose each configuration');
  await expect(about).toContainText('Upstream defaults');
  await expect(about).toContainText('Author-blessed');
  await expect(about).toContainText('Maintainer-chosen');
  await expect(about).toContainText('The benchmark and Flash-ANSR share authors');
});

test('provenance: table rows footnote who chose each configuration', async ({ page }) => {
  await gotoView(page, 'table');
  await expect(page.locator('.paired-foot')).toContainText('Configuration: the method\'s own upstream defaults');
  await expect(page.locator('.paired-foot')).toContainText('author-blessed');
});

test('provenance: a pinned matrix cell states both sides\' labels', async ({ page }) => {
  await gotoView(page, 'matrix');
  await page.click('td[data-cell="2"]');
  await expect(page.locator('.matrix-detail')).toContainText('configs:');
  await expect(page.locator('.matrix-detail a[href="#about"]')).toContainText('who chose what');
});

test('provenance: footnotes appear only at snapped budgets (descriptive slices stay clean)', async ({ page }) => {
  await gotoView(page, 'table');
  await expect(page.locator('.paired-foot')).toContainText('Configuration:');
  await page.evaluate(() => {
    const r = document.querySelector('.budget-slider input[type=range]');
    r.value = '500'; r.dispatchEvent(new Event('input'));
  });
  await expect(page.locator('.desc-banner').first()).toContainText('Descriptive slice');
  await expect(page.locator('.paired-foot')).not.toContainText('Configuration:');
});

test('provenance: an old cached payload without labels degrades gracefully', async ({ page }) => {
  const errors = collectErrors(page);
  await page.route('**/results_data.js', async (route) => {
    const body = (await (await fetch(new URL('/results_data.js', 'http://localhost:8123'))).text())
      .replace(/"config_provenance":\s*"[a-z_]+",?/g, '');
    await route.fulfill({ status: 200, contentType: 'application/javascript', body });
  });
  await gotoView(page, 'table');
  await expect(page.locator('.paired-foot')).not.toContainText('Configuration:');
  await switchTo(page, 'matrix');
  await page.click('td[data-cell="2"]');
  await expect(page.locator('.matrix-detail')).toContainText('noise margin');
  await expect(page.locator('.matrix-detail')).not.toContainText('configs:');
  expect(errors).toEqual([]);
});

// ---- payload contract ----------------------------------------------------------------------

test('payload: schema and blocks the frontend depends on', async ({ request }) => {
  const payload = await (await request.get('/paired_data.json')).json();
  expect(payload.schema_version).toBeGreaterThanOrEqual(5);
  expect(payload.budgets.length).toBeGreaterThan(0);
  expect(payload.records.length).toBeGreaterThan(0);
  expect(payload.leaderboard.length).toBeGreaterThan(0);
  expect(payload.ranks.length).toBeGreaterThan(0);
  expect(payload.rank_metrics.some((m) => m.primary)).toBe(true);
});
