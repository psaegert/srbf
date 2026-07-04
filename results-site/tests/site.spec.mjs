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

test('ranks: entering with vNRR switches to the primary league VISIBLY and restores on leave', async ({ page }) => {
  const errors = collectErrors(page);
  await gotoView(page, 'ranks');
  await expect(metricSelect(page)).toHaveValue('log10_fvu_val');
  await expect(page.locator('.desc-banner')).toContainText('primary league metric');
  await switchTo(page, 'table');                       // the 2026-07-04 regression path
  await expect(metricSelect(page)).toHaveValue('numeric_recovery_val');
  expect(errors).toEqual([]);
});

test('ranks: the metric menu shows the league structure at first glance', async ({ page }) => {
  await gotoView(page, 'ranks');
  await expect(metricSelect(page)
    .locator('optgroup[label="Primary league"] option[value="log10_fvu_val"]')).toHaveCount(1);
  expect(await metricSelect(page)
    .locator('optgroup[label="Exploratory leagues"] option').count()).toBeGreaterThan(5);
  await expect(page.locator('.metric-badge')).toContainText('primary league');
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

test('help: the metric ?-link routes to the metrics reference', async ({ page }) => {
  await gotoView(page, 'curves');
  await expect(page.locator('.help-link[href="#metrics"]')).toBeVisible();
  await expect(page.locator('.help-link[href="#paired"]').first()).toBeAttached();
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
