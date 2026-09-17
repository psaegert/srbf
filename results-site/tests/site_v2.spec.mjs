// The 2026-09 release explorer (explorer_v2.js) and the release switch. Same philosophy as site.spec.mjs:
// content renders, transitions land, no console errors, no horizontal overflow. Every view, the metric registry,
// the lazily loaded histograms and paired contrasts, deep links, popovers, colours.
import { test, expect } from '@playwright/test';

function collectErrors(page) {
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  page.on('console', (msg) => { if (msg.type() === 'error') { errors.push(msg.text()); } });
  return errors;
}
const V2 = '#results-explorer-v2';
const VIEWS = ['curves', 'table', 'matrix', 'dist', 'paired'];
// the 2026-07 site's 21 metrics under their schema-2 keys: none may be missing from a release
const LEGACY_METRICS = ['numeric_recovery_val', 'expr_length_ratio', 'log10_fvu_val', 'log10_fvu_fit', 'numeric_recovery_fit', 'success',
  'skeleton_match_raw', 'f1_score', 'precision_score', 'recall_score', 'edit_distance_norm', 'zss_edit_distance', 'expr_length_ratio_abserr',
  'predicted_skeleton_prefix_length', 'skeleton_length', 'n_constants_ratio', 'n_constants_delta', 'total_nestedness_delta', 'predicted_log_prob',
  'predicted_score', 'fit_time'];

test('the newest release is the default and renders its curves', async ({ page }) => {
  const errors = collectErrors(page);
  await page.goto('/');
  await expect(page.locator(V2)).toBeVisible();
  await expect(page.locator('#results-explorer')).toBeHidden();
  await expect(page.locator(V2 + ' svg.v2chart').first()).toBeVisible();
  await expect(page.locator('#release-switch a[aria-current="true"]')).toHaveAttribute('data-release', '2026-09');
  expect(errors).toEqual([]);
});

test('the release switch reaches the 2026-07 explorer and back', async ({ page }) => {
  const errors = collectErrors(page);
  await page.goto('/');
  await page.locator('#release-switch a[data-release="2026-07"]').click();
  await expect(page.locator('#results-explorer')).toBeVisible();
  await expect(page.locator(V2)).toBeHidden();
  await expect(page.locator('#results-plot .main-svg').first()).toBeVisible();
  await page.locator('#release-switch a[data-release="2026-09"]').click();
  await expect(page.locator(V2 + ' svg.v2chart').first()).toBeVisible();
  expect(errors).toEqual([]);
});

test('a 2026-07 deep link still opens the 2026-07 explorer', async ({ page }) => {
  await page.goto('/?view=curves&bench=FastSRB');
  await expect(page.locator('#results-explorer')).toBeVisible();
  await expect(page.locator(V2)).toBeHidden();
});

test('the metric registry carries every 2026-07 metric and the new headline ones', async ({ page }) => {
  await page.goto('/');
  const keys = await page.evaluate(() => window.RESULTS_V2.metrics.map((m) => m.key));
  for (const k of LEGACY_METRICS.concat(['symbolic_recovery', 'mdl_ratio', 'r2_val'])) { expect(keys, k).toContain(k); }
  expect(keys.length).toBeGreaterThanOrEqual(24);
  const count = await page.evaluate(() => document.querySelectorAll('#results-explorer-v2 .v2metric').length);
  expect(count).toBe(keys.length);
});

for (const view of VIEWS) {
  test(`the ${view} view renders from a deep link without errors or overflow`, async ({ page }) => {
    const errors = collectErrors(page);
    await page.goto(`/?release=2026-09&v=${view}&f=log10_fvu_val&r=32`);
    await expect(page.locator(V2 + ' .v2tab.active')).toHaveAttribute('data-view', view);
    const content = page.locator(V2 + ' .v2view svg.v2chart, ' + V2 + ' .v2view table');
    await expect(content.first()).toBeVisible({ timeout: 15000 });
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    expect(overflow).toBeLessThanOrEqual(0);
    expect(errors).toEqual([]);
  });
}

test('pooled medians load their histograms on demand', async ({ page }) => {
  const errors = collectErrors(page);
  await page.goto('/?release=2026-09&v=table&p=log10_fvu_val,mdl_ratio&s=median');
  await expect(page.locator(V2 + ' table')).toBeVisible();
  await expect.poll(async () => page.evaluate(() => Object.keys((window.RESULTS_V2_HIST || {})['2026-09'] || {}).length), { timeout: 15000 }).toBeGreaterThanOrEqual(2);
  await expect.poll(async () => (await page.locator(V2 + ' table tbody td').allTextContents()).filter((t) => /^-?\d/.test(t.trim())).length, { timeout: 15000 }).toBeGreaterThan(0);
  expect(errors).toEqual([]);
});

test('the paired view loads its contrasts and shows a baseline selector', async ({ page }) => {
  const errors = collectErrors(page);
  await page.goto('/?release=2026-09&v=paired&p=numeric_recovery_val');
  await expect(page.locator(V2 + ' select.v2base')).toBeVisible({ timeout: 15000 });
  await expect.poll(async () => page.evaluate(() => Object.keys((window.RESULTS_V2_PAIRED || {})['2026-09'] || {}).length), { timeout: 15000 }).toBeGreaterThanOrEqual(1);
  await expect(page.locator(V2 + ' .v2view svg.v2chart').first()).toBeVisible();
  expect(errors).toEqual([]);
});

test('catalog and method controls change the pooled charts', async ({ page }) => {
  const errors = collectErrors(page);
  await page.goto('/?release=2026-09&v=curves');
  const count = page.locator(V2 + ' .v2catcount');
  const before = await count.textContent();
  await page.locator(V2 + ' button[data-act="phys"]').click();
  await expect(count).not.toHaveText(before);
  await page.locator(V2 + ' button[data-act="none"]').click();
  await expect(page.locator(V2 + ' svg.v2chart').first()).toContainText('no catalog selected');
  await page.locator(V2 + ' button[data-act="all"]').click();
  await expect(count).toHaveText(before);
  expect(errors).toEqual([]);
});

test('the view state round-trips through the URL', async ({ page }) => {
  await page.goto('/?release=2026-09&v=matrix&f=mdl_ratio&r=16&c=phys&s=mean&pool=own&ci=0');
  await expect(page.locator(V2 + ' .v2tab.active')).toHaveAttribute('data-view', 'matrix');
  await expect(page.locator(V2 + ' select.v2focus')).toHaveValue('mdl_ratio');
  await expect(page.locator(V2 + ' select.v2rung')).toHaveValue('16');
  await expect(page.locator(V2 + ' input[name="v2stat"][value="mean"]')).toBeChecked();
  await expect(page.locator(V2 + ' input[name="v2pool"][value="own"]')).toBeChecked();
  await expect(page.locator(V2 + ' .v2ci')).not.toBeChecked();
  await expect(page.locator(V2 + ' .v2catcount')).toContainText('8 of');
  await page.locator(V2 + ' .v2tab[data-view="table"]').click();
  expect(page.url()).toContain('v=table');
});

test('terms and metric help open a floating explanation', async ({ page }) => {
  await page.goto('/?release=2026-09&v=curves');
  await page.locator(V2 + ' .v2metrics .v2help').first().click();
  await expect(page.locator('.v2pop')).toBeVisible();
  await expect(page.locator('.v2pop')).toContainText('Share of laws');
  await page.keyboard.press('Escape');
  await expect(page.locator('.v2pop')).toHaveCount(0);
  await page.locator(V2 + ' .v2panel .v2term[data-term="matched"]').first().click();
  await expect(page.locator('.v2pop')).toContainText('Matched pooling');
});

test('time is the default x axis wherever a time was measured', async ({ page }) => {
  await page.goto('/');
  const radio = page.locator(V2 + ' input.v2xtime');
  // a time exists when the reference machine has measured a method, or when the runs themselves carry fit_time
  const hasTime = await page.evaluate(() => {
    const D = window.RESULTS_V2 || {};
    if (Object.keys(D.timing || {}).some((k) => Object.keys(D.timing[k]).length)) { return true; }
    return Object.keys(D.cells || {}).some((m) => Object.keys(D.cells[m]).some((c) => Object.keys(D.cells[m][c]).some((r) => D.cells[m][c][r].m && D.cells[m][c][r].m.fit_time)));
  });
  if (!hasTime) { await expect(radio).toBeDisabled(); return; }
  await expect(radio).toBeEnabled();
  await expect(radio).toBeChecked();
  await expect(page.locator(V2 + ' .v2view svg.v2chart').first()).toContainText('fit time');
});

test('the candidate axis names itself and is offered beside time', async ({ page }) => {
  const errors = collectErrors(page);
  await page.goto('/?release=2026-09&v=curves');
  await page.locator(V2 + ' input[name="v2xaxis"][value="rung"]').check();
  await expect(page.locator(V2 + ' .v2view svg.v2chart').first()).toContainText('candidates');
  // a method whose budget is a time limit has no position on this axis and is named instead of dropped silently
  const seconds = await page.evaluate(() => (window.RESULTS_V2.methods || []).filter((m) => m.budget === 'seconds' && window.RESULTS_V2.cells[m.key] && Object.keys(window.RESULTS_V2.cells[m.key]).length).map((m) => m.label));
  for (const label of seconds) { await expect(page.locator(V2 + ' .v2view')).toContainText(label); }
  await page.locator(V2 + ' input.v2xtime').check();
  await expect(page.locator(V2 + ' .v2view svg.v2chart').first()).toContainText('fit time');
  expect(errors).toEqual([]);
});

test('the headline stands above the explorer with its two fixed charts', async ({ page }) => {
  const errors = collectErrors(page);
  await page.goto('/');
  const head = page.locator('#results-headline-v2');
  await expect(head).toBeVisible();
  await expect(head.locator('.v2hltitle')).toBeVisible();
  await expect(head.locator('svg.v2chart')).toHaveCount(2);
  await expect(head.locator('svg.v2chart').first()).toContainText('Recovery vs time');
  await expect(head.locator('svg.v2chart').first()).toContainText('fit time');
  // the second headline chart is the trade-off: description length on x, fit error on y
  await expect(head.locator('svg.v2chart').nth(1)).toContainText('Fit vs length');
  await expect(head.locator('svg.v2chart').nth(1)).toContainText(/length \/ law/);
  await expect(head.locator('svg.v2chart').nth(1)).toContainText('FVU');
  // fixed: the explorer's own controls do not move it
  await page.locator(V2 + ' button[data-act="none"]').click();
  await expect(head.locator('svg.v2chart').first()).not.toContainText('no catalog selected');
  expect(errors).toEqual([]);
});

test('the headline belongs to the 2026-09 release only', async ({ page }) => {
  await page.goto('/?release=2026-07');
  await expect(page.locator('#results-headline-v2')).toBeHidden();
});

test('the public page carries no private overlay', async ({ page }) => {
  await page.goto('/');
  expect(await page.evaluate(() => typeof window.RESULTS_V2_PRIVATE)).toBe('undefined');
  const html = await page.content();
  expect(html).not.toContain('private/');
});

test('every method says how it picks the answer it submits', async ({ page }) => {
  await page.goto('/?release=2026-09&v=curves');
  const withData = await page.evaluate(() => (window.RESULTS_V2.methods || []).filter((m) => window.RESULTS_V2.cells[m.key] && Object.keys(window.RESULTS_V2.cells[m.key]).length));
  expect(withData.length).toBeGreaterThan(0);
  for (const m of withData) { expect(m.selection, m.key).toBeTruthy(); }
  // the release protocol no longer states one method's ranking rule as if it were the benchmark's
  const scoring = await page.evaluate(() => window.RESULTS_V2.release.scoring);
  expect(scoring).not.toMatch(/two-part code/i);
  await page.locator(V2 + ' .v2methods .v2help').first().click();
  await expect(page.locator('.v2pop')).toBeVisible();
});

test('the view that shows tables is called Tables', async ({ page }) => {
  await page.goto('/?release=2026-09&v=curves');
  await expect(page.locator(V2 + ' .v2tab[data-view="table"]')).toHaveText('Tables');
  await expect(page.locator('#va')).toContainText('Tables');
});

test('the mean is the default statistic, and the median stays one click away', async ({ page }) => {
  await page.goto('/?release=2026-09&v=curves&p=log10_fvu_val');
  await expect(page.locator(V2 + ' input[name="v2stat"][value="mean"]')).toBeChecked();
  await expect(page.locator('#results-headline-v2 .v2hlsub')).toContainText('mean');
  const chart = page.locator(V2 + ' .v2view svg.v2chart').first();
  const headline = page.locator('#results-headline-v2 svg.v2chart').first();
  await expect(chart).toBeVisible();
  await expect(headline).toBeVisible();   // drawn from the cell sums: a mean needs no histogram
  // switching the statistic moves the data, never the wording, here or in the headline
  const xLabel = async (svg) => (await svg.locator('text').allTextContents()).find((t) => t.includes('fit time'));
  const before = await xLabel(chart);
  const headBefore = await headline.textContent();
  await page.locator(V2 + ' input[name="v2stat"][value="median"]').check();
  await expect(chart).not.toContainText('loading', { timeout: 15000 });
  expect(await xLabel(chart)).toBe(before);
  expect(await headline.textContent()).toBe(headBefore);
});

test('each display carries only the controls it can use', async ({ page }) => {
  const shown = () => page.evaluate(() => [...document.querySelectorAll('#results-explorer-v2 [data-uses]')]
    .filter((e) => !e.hidden).flatMap((e) => e.dataset.uses.split(' ')));
  await page.goto('/?release=2026-09&v=curves');
  const curves = await shown();
  for (const k of ['plots', 'stat', 'xaxis']) { expect(curves, k).toContain(k); }
  for (const k of ['focus', 'rung', 'base', 'rows']) { expect(curves, k).not.toContain(k); }
  await page.locator(V2 + ' .v2tab[data-view="matrix"]').click();
  await expect.poll(shown).toContain('focus');
  const matrix = await shown();
  for (const k of ['rung', 'stat']) { expect(matrix, k).toContain(k); }
  for (const k of ['plots', 'xaxis', 'thin']) { expect(matrix, k).not.toContain(k); }   // one budget, one metric
  await page.locator(V2 + ' .v2tab[data-view="paired"]').click();
  await expect.poll(shown).toContain('base');
});

test('a chart is drawn at the width it is given', async ({ page }) => {
  await page.setViewportSize({ width: 1600, height: 1000 });
  await page.goto('/?release=2026-09&v=curves');
  const chart = page.locator('#results-headline-v2 svg.v2chart').first();
  await expect(chart).toBeVisible();
  const box = await chart.boundingBox();
  const viewBox = await chart.getAttribute('viewBox');
  expect(box.width).toBeGreaterThan(500);                                  // a chart, not a thumbnail
  expect(Math.abs(Number(viewBox.split(' ')[2]) - box.width)).toBeLessThan(2);   // 1 unit = 1 px: 12 px of label is 12 px
});
