// The 2026-09 release explorer (explorer_v2.js) and the release switch. Same philosophy as site.spec.mjs:
// content renders, transitions land, no console errors, no horizontal overflow. Every view, the metric registry,
// the lazily loaded histograms and paired contrasts, deep links, popovers, colours.
import { test, expect } from '@playwright/test';
import { readFileSync } from 'node:fs';

function collectErrors(page) {
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  page.on('console', (msg) => { if (msg.type() === 'error') { errors.push(msg.text()); } });
  return errors;
}
const V2 = '#results-explorer-v2';
// every metric is chosen through the same picker: open the control, click the entry
async function pick(page, trigger, key) {
  await trigger.click();
  await expect(page.locator('.v2picker')).toBeVisible();
  await page.locator(`.v2picker .v2pickitem[data-k="${key}"]`).click();
  await expect(page.locator('.v2picker')).toHaveCount(0);
}
const VIEWS = ['curves', 'table', 'matrix', 'dist', 'ranks', 'paired'];
// the 2026-07 site's 21 metrics under their schema-2 keys: none may be missing from a release
const LEGACY_METRICS = ['numeric_recovery_val', 'expr_length_ratio', 'log10_fvu_val', 'log10_fvu_fit', 'numeric_recovery_fit', 'success',
  'skeleton_match_raw', 'f1_score', 'precision_score', 'recall_score', 'edit_distance_norm', 'zss_edit_distance', 'expr_length_ratio_abserr',
  'predicted_skeleton_prefix_length', 'skeleton_length', 'n_constants_ratio', 'n_constants_delta', 'total_nestedness_delta', 'predicted_log_prob',
  'predicted_score'];   // fit_time is NOT here: the 2026-07 site published an as-run wall clock, this release publishes none

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
  await page.goto('/?release=2026-09&v=matrix&f=mdl_ratio&r=16&c=phys&s=mean&pool=own&thin=1&ci=0');   // pool= and thin= are old links: read, ignored
  await expect(page.locator(V2 + ' .v2tab.active')).toHaveAttribute('data-view', 'matrix');
  await expect(page.locator(V2 + ' .v2focus')).toHaveAttribute('data-k', 'mdl_ratio');
  await expect(page.locator(V2 + ' select.v2rung')).toHaveValue('16');
  await expect(page.locator(V2 + ' input[name="v2stat"][value="mean"]')).toBeChecked();
  await expect(page.locator(V2 + ' input[name="v2pool"]')).toHaveCount(0);
  await expect(page.locator(V2 + ' .v2thin')).toHaveCount(0);
  await expect(page.locator(V2 + ' .v2band')).not.toBeChecked();   // the legacy ci=0 link still means "no interval"
  await expect(page.locator(V2 + ' .v2catcount')).toContainText('8 of');
  await page.locator(V2 + ' .v2tab[data-view="table"]').click();
  expect(page.url()).toContain('v=table');
});

test('terms and metric help open a floating explanation', async ({ page }) => {
  await page.goto('/?release=2026-09&v=table');
  await page.locator(V2 + ' .v2metrics .v2help').first().click();
  await expect(page.locator('.v2pop')).toBeVisible();
  await expect(page.locator('.v2pop')).toContainText('Share of laws');
  await page.keyboard.press('Escape');
  await expect(page.locator('.v2pop')).toHaveCount(0);
  await page.locator(V2 + ' .v2view .v2term[data-term="complete"]').first().click();
  await expect(page.locator('.v2pop')).toContainText('EVERY selected catalog');
});

// A published time is a CALIBRATED time. Seconds measured wherever a unit happened to run are not comparable
// between methods, so the release must not carry them and no chart may draw them.
const hasRefTiming = (page) => page.evaluate(() => Object.keys(window.RESULTS_V2.timing || {})
  .some((k) => Object.keys(window.RESULTS_V2.timing[k]).length));

// A chart on the time axis says so at every width: the visible label shortens to "fit time (s, ref)" on a narrow
// screen, so the calibration itself is asserted on the aria-label, which is width-independent.
async function calibrated(chart) {
  await expect(chart).toContainText('fit time');
  await expect(chart).toHaveAttribute('aria-label', /reference machine/);
}

test('a time axis needs one calibrated method, and an uncalibrated one costs only its own points', async ({ page }) => {
  await page.goto('/');
  const axis = page.locator(V2 + ' .v2plot .v2xsel').first();
  if (!await hasRefTiming(page)) {
    await expect(axis).toHaveAttribute('data-k', 'rung');                 // the budget, never an uncalibrated time
    await expect(page.locator(V2 + ' .v2view svg.v2chart').first()).not.toContainText('fit time');
    await expect(page.locator('#results-headline-v2')).not.toContainText('fit time');
    await axis.click();
    await expect(page.locator('.v2pickitem[data-k="time"]')).toBeDisabled();
    await page.keyboard.press('Escape');
  } else {
    await expect(axis).toHaveAttribute('data-k', 'time');
    await calibrated(page.locator(V2 + ' .v2view svg.v2chart').first());
  }
  // whatever the state, nothing anywhere may say a time was measured "as run"
  expect(await page.content()).not.toContain('as run');
});

test('the release publishes no as-run wall clock', async ({ page }) => {
  await page.goto('/');
  const banned = await page.evaluate(() => {
    const D = window.RESULTS_V2, ban = ['fit_time', 'generation_time'], hits = [];
    for (const k of ban) {
      if ((D.metrics || []).some((m) => m.key === k)) { hits.push('registry:' + k); }
      if ((D.paired_keys || []).includes(k)) { hits.push('paired_keys:' + k); }
      for (const m of Object.keys(D.cells || {})) {
        for (const c of Object.keys(D.cells[m])) {
          for (const r of Object.keys(D.cells[m][c])) {
            const cell = D.cells[m][c][r];
            if ((cell.m && k in cell.m) || (cell.r && k in cell.r)) { hits.push('cell:' + k); }
          }
        }
      }
    }
    return [...new Set(hits)];
  });
  expect(banned, 'the payload carries an as-run wall-clock metric').toEqual([]);
});

test('the candidate axis names itself and is offered beside time', async ({ page }) => {
  const errors = collectErrors(page);
  await page.goto('/?release=2026-09&v=curves');
  await pick(page, page.locator(V2 + ' .v2plot .v2xsel').first(), 'rung');
  await expect(page.locator(V2 + ' .v2view svg.v2chart').first()).toContainText('candidates');
  // a method whose budget is not a candidate count has no position on this axis and is named instead of dropped silently
  const seconds = await page.evaluate(() => (window.RESULTS_V2.methods || []).filter((m) => (m.budget || 'candidates') !== 'candidates' && window.RESULTS_V2.cells[m.key] && Object.keys(window.RESULTS_V2.cells[m.key]).length).map((m) => m.label));
  for (const label of seconds) { await expect(page.locator(V2 + ' .v2view')).toContainText(label); }
  if (await hasRefTiming(page)) {
    await pick(page, page.locator(V2 + ' .v2plot .v2xsel').first(), 'time');
    await calibrated(page.locator(V2 + ' .v2view svg.v2chart').first());
  }
  expect(errors).toEqual([]);
});

test('the headline stands above the explorer with its two fixed charts', async ({ page }) => {
  const errors = collectErrors(page);
  await page.goto('/');
  const head = page.locator('#results-headline-v2');
  await expect(head).toBeVisible();
  await expect(head.locator('.v2hltitle')).toBeVisible();
  await expect(head.locator('svg.v2chart')).toHaveCount(2);
  await expect(head.locator('svg.v2chart').first()).toContainText(await hasRefTiming(page) ? 'Recovery vs time' : 'Recovery vs budget');
  if (await hasRefTiming(page)) { await calibrated(head.locator('svg.v2chart').first()); }
  else { await expect(head.locator('svg.v2chart').first()).toContainText('candidates'); }
  // the second headline chart is the trade-off: description length on x, fit error on y
  await expect(head.locator('svg.v2chart').nth(1)).toContainText('Fit vs length');
  await expect(head.locator('svg.v2chart').nth(1)).toContainText('MDL ratio');   // the metric's own name, as everywhere else
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
  for (const k of ['stat', 'ci']) { expect(curves, k).toContain(k); }
  for (const k of ['pool', 'thin']) { expect(curves, k).not.toContain(k); }   // a pooled number is complete or absent: nothing to switch
  for (const k of ['plots', 'xaxis', 'focus', 'rung', 'base', 'rows']) { expect(curves, k).not.toContain(k); }
  await page.locator(V2 + ' .v2tab[data-view="table"]').click();
  await expect.poll(shown).toContain('plots');
  await page.locator(V2 + ' .v2tab[data-view="matrix"]').click();
  await expect.poll(shown).toContain('focus');
  const matrix = await shown();
  for (const k of ['rung', 'stat']) { expect(matrix, k).toContain(k); }
  for (const k of ['plots', 'xaxis', 'thin']) { expect(matrix, k).not.toContain(k); }   // one budget, one metric
  await page.locator(V2 + ' .v2tab[data-view="paired"]').click();
  await expect.poll(shown).toContain('base');
});

test('every plot carries its own two axes, and plots are added and removed', async ({ page }) => {
  const errors = collectErrors(page);
  await page.setViewportSize({ width: 1600, height: 1100 });
  await page.goto('/?release=2026-09&v=curves&p=time~numeric_recovery_val');
  const cards = page.locator(V2 + ' .v2plot');
  await expect(cards).toHaveCount(1);
  await expect(cards.first().locator('.v2ysel')).toHaveAttribute('data-k', 'numeric_recovery_val');
  // a metric on x makes the plot a trade-off: the budget is gone from both axes
  await pick(page, cards.first().locator('.v2xsel'), 'mdl_ratio');
  await expect(cards.first().locator('svg.v2chart')).toContainText('MDL ratio');
  expect(decodeURIComponent(page.url())).toContain('mdl_ratio~numeric_recovery_val');
  // the dashed tile adds a plot, the x removes one
  await page.locator(V2 + ' .v2addplot').click();
  await expect(cards).toHaveCount(2);
  await page.locator(V2 + ' .v2rmplot').first().click();
  await expect(cards).toHaveCount(1);
  await page.locator(V2 + ' .v2rmplot').first().click();
  await expect(cards).toHaveCount(0);
  await expect(page.locator(V2 + ' .v2addplot')).toBeVisible();   // never a dead end
  await page.locator(V2 + ' .v2addplot').click();
  await expect(cards).toHaveCount(1);
  expect(errors).toEqual([]);
});

test('a metric carries one name and one definition wherever it appears', async ({ page }) => {
  await page.setViewportSize({ width: 1600, height: 1100 });
  await page.goto('/?release=2026-09&v=curves&p=time~numeric_recovery_val');
  const M = await page.evaluate(() => (window.RESULTS_V2.metrics || []).find((m) => m.key === 'numeric_recovery_val'));
  // the plot header, the picker entry and the table column all use the registry label
  await expect(page.locator(V2 + ' .v2plot .v2ysel .v2picklab')).toHaveText(M.label);
  await page.locator(V2 + ' .v2plot .v2ysel').click();
  await expect(page.locator(`.v2picker .v2pickitem[data-k="${M.key}"]`)).toContainText(M.label);
  const def = await page.locator(`.v2picker .v2pickitem[data-k="${M.key}"]`).getAttribute('title');
  expect(def).toContain(M.desc);
  await page.keyboard.press('Escape');
  await page.locator(V2 + ' .v2tab[data-view="table"]').click();
  const th = page.locator(V2 + ' table thead th', { hasText: M.label }).first();
  await expect(th).toBeVisible();
  expect(await th.locator('.v2help').getAttribute('data-help')).toBe(def);   // the same sentence, not a paraphrase
});

test('every chart names both of its axes', async ({ page }) => {
  await page.setViewportSize({ width: 1600, height: 1100 });
  await page.goto('/?release=2026-09&v=curves&p=rung~numeric_recovery_val,mdl_ratio~log10_fvu_val');
  const labels = async (svg) => (await svg.locator('text').allTextContents()).join(' | ');
  const charts = page.locator('svg.v2chart');
  await expect(charts.first()).toBeVisible();
  const n = await charts.count();
  expect(n).toBeGreaterThanOrEqual(4);   // two headline panels and two plots
  for (let i = 0; i < n; i++) {
    const t = await labels(charts.nth(i));
    expect(t, `chart ${i} x label`).toMatch(/fit time per problem|MDL ratio|candidates per problem/);   // a budget or a metric, never an uncalibrated time
    expect(t, `chart ${i} y label`).toMatch(/Numeric recovery|log10 FVU/);
  }
});

test('the picker lists every metric in columns and filters', async ({ page }) => {
  const errors = collectErrors(page);
  await page.setViewportSize({ width: 1600, height: 1100 });
  await page.goto('/?release=2026-09&v=curves&p=time~numeric_recovery_val');
  const all = await page.evaluate(() => (window.RESULTS_V2.metrics || []).length);
  await page.locator(V2 + ' .v2plot .v2xsel').click();
  const picker = page.locator('.v2picker');
  await expect(picker).toBeVisible();
  await expect(picker.locator('.v2pickitem')).toHaveCount(all + 2);        // every metric, plus the two budgets
  expect(await picker.locator('.v2pickgroup').count()).toBeGreaterThan(3);  // grouped, not one long column
  expect(await picker.evaluate((el) => getComputedStyle(el.querySelector('.v2pickcols')).columnCount)).not.toBe('1');
  await picker.locator('.v2pickq').fill('recovery');
  await expect(picker.locator('.v2pickitem:visible')).toHaveCount(await page.evaluate(() => (window.RESULTS_V2.metrics || []).filter((m) => (m.label + m.key).toLowerCase().includes('recovery')).length));
  await page.keyboard.press('Escape');
  await expect(picker).toHaveCount(0);
  expect(errors).toEqual([]);
});

// the fixture overlay and the key it was sealed with (results-site/tools/seal.mjs, tests/fixtures/overlay)
const FIXTURE_KEY = 'fixture-key-for-the-tests';
const FIXTURE = readFileSync(new URL('./fixtures/sealed_fixture.js', import.meta.url), 'utf8');

test('a method is added with its key, and a key that does not fit adds nothing', async ({ page }) => {
  const errors = collectErrors(page);
  await page.addInitScript({ content: FIXTURE });
  await page.goto('/?release=2026-09&v=curves');
  const methods = page.locator(V2 + ' .v2methods .v2meth');
  await expect(methods.first()).toBeVisible();
  const before = await methods.count();
  await page.locator(V2 + ' .v2addmopen').click();
  await page.locator(V2 + ' .v2addmkey').fill('not-the-key-at-all');
  await page.locator(V2 + ' [data-act="add-method-go"]').click();
  await expect(page.locator(V2 + ' .v2addmmsg')).toHaveText(/No method found/, { timeout: 20000 });
  expect(await methods.count(), 'a key that does not fit changes nothing').toBe(before);
  await page.locator(V2 + ' .v2addmkey').fill(FIXTURE_KEY);
  await page.locator(V2 + ' [data-act="add-method-go"]').click();
  await expect(page.locator(V2 + ' .v2methods')).toContainText('Fixture Method', { timeout: 20000 });
  expect(await methods.count()).toBe(before + 1);
  // and it is an ordinary method from there on: selectable, and carried by the URL like the rest
  await expect(page.locator(V2 + ' .v2methods input[type=checkbox][data-m="fixture-method"]')).toBeChecked();
  expect(errors).toEqual([]);
});

test('an untimed method leaves the time axis standing and is named under it', async ({ page }) => {
  const errors = collectErrors(page);
  await page.addInitScript({ content: FIXTURE });
  await page.goto('/?release=2026-09&v=curves');
  test.skip(!await hasRefTiming(page), 'no reference timing in this release yet');
  await page.locator(V2 + ' .v2addmopen').click();
  await page.locator(V2 + ' .v2addmkey').fill(FIXTURE_KEY);
  await page.locator(V2 + ' [data-act="add-method-go"]').click();
  await expect(page.locator(V2 + ' .v2methods')).toContainText('Fixture Method', { timeout: 20000 });
  // the fixture overlay carries timing {}, so it has no position on the time axis
  expect(await page.evaluate(() => !!(window.RESULTS_V2.timing || {})['fixture-method'])).toBe(false);

  await pick(page, page.locator(V2 + ' .v2plot .v2xsel').first(), 'time');
  const chart = page.locator(V2 + ' .v2view svg.v2chart').first();
  await calibrated(chart);                                            // the axis stands, on the reference machine
  await expect(chart).toContainText('E2E');                           // and the calibrated methods are still drawn
  await expect(chart, 'an untimed method must not be drawn on a calibrated axis').not.toContainText('Fixture Method');
  await expect(page.locator(V2 + ' .v2view'), 'and it must be named, not dropped in silence').toContainText('Fixture Method');
  await expect(page.locator(V2 + ' .v2view .v2hint').first()).toContainText(/reference-machine time/);
  expect(errors).toEqual([]);
});

test('an untimed method changes nothing else on the time axis', async ({ page }) => {
  await page.addInitScript({ content: FIXTURE });
  await page.goto('/?release=2026-09&v=curves');
  test.skip(!await hasRefTiming(page), 'no reference timing in this release yet');
  await pick(page, page.locator(V2 + ' .v2plot .v2xsel').first(), 'time');
  const chart = page.locator(V2 + ' .v2view svg.v2chart').first();
  await calibrated(chart);
  const before = await chart.textContent();                           // every drawn point, with its value and its pool

  await page.locator(V2 + ' .v2addmopen').click();
  await page.locator(V2 + ' .v2addmkey').fill(FIXTURE_KEY);
  await page.locator(V2 + ' [data-act="add-method-go"]').click();
  await expect(page.locator(V2 + ' .v2methods')).toContainText('Fixture Method', { timeout: 20000 });
  // the point of the whole arrangement: an uncalibrated method costs its own points and nothing else -- same
  // methods, same values, same matched pool, because it never enters the set the axis is drawn over
  await expect(page.locator(V2 + ' .v2view svg.v2chart').first()).toHaveText(before);
});

test('the page says nothing about what it does not show', async ({ page }) => {
  await page.goto('/?release=2026-09&v=curves');
  // the control is plain, and neither it nor the payload names anything the release does not publish
  await expect(page.locator(V2 + ' .v2addmopen')).toHaveText('add method');
  const html = await page.content();
  expect(html).not.toMatch(/private|sealed|decrypt|password/i);
});

test('the interval is a band by default, and crosses are a separate switch', async ({ page }) => {
  const errors = collectErrors(page);
  await page.setViewportSize({ width: 1600, height: 1100 });
  await page.goto('/?release=2026-09&v=curves&p=rung~numeric_recovery_val,mdl_ratio~log10_fvu_val');
  // one shape for both kinds of x axis: the band is drawn from the interval boxes, so a trade-off plot gets a
  // two-dimensional region rather than a bar through each point
  const shapes = () => page.evaluate(() => [...document.querySelectorAll('#results-explorer-v2 .v2plot svg.v2chart')].map((s) => ({
    polys: s.querySelectorAll('polygon').length,
    bars: [...s.querySelectorAll('line')].filter((l) => l.getAttribute('stroke-opacity') === '0.45').length,
  })));
  await expect(page.locator(V2 + ' .v2band')).toBeChecked();
  await expect(page.locator(V2 + ' .v2cross')).not.toBeChecked();
  for (const c of await shapes()) { expect(c.polys).toBeGreaterThan(0); expect(c.bars).toBe(0); }
  await page.locator(V2 + ' .v2cross').check();
  await expect.poll(async () => (await shapes())[0].bars).toBeGreaterThan(0);
  for (const c of await shapes()) { expect(c.polys, 'both shapes at once').toBeGreaterThan(0); }
  await page.locator(V2 + ' .v2band').uncheck();
  await expect.poll(async () => (await shapes())[0].polys).toBe(0);
  for (const c of await shapes()) { expect(c.bars).toBeGreaterThan(0); }
  await page.locator(V2 + ' .v2cross').uncheck();
  for (const c of await shapes()) { expect(c.polys + c.bars, 'neither shape').toBe(0); }
  expect(page.url()).toContain('band=0');
  expect(errors).toEqual([]);
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

// ---- Distribution: a distribution is what the view shows, in four readings ------------------------------------
test('the distribution view opens on histograms of a continuous metric', async ({ page }) => {
  const errors = collectErrors(page);
  await page.goto('/?release=2026-09');
  await page.locator(V2 + ' .v2tab[data-view="dist"]').click();
  const chart = page.locator(V2 + ' .v2view svg.v2chart').first();
  await expect(chart).toBeVisible({ timeout: 15000 });
  await expect(chart).toHaveAttribute('aria-label', /one histogram per method/);
  // the metric it opens on is continuous, whatever the Catalogs view is looking at
  const kind = await page.evaluate(() => { const k = document.querySelector('#results-explorer-v2 .v2viewbar .v2pick').dataset.k; return window.RESULTS_V2.metrics.find((m) => m.key === k).kind; });
  expect(kind).toBe('cont');
  // one filled histogram and one box per method drawn, and every panel states how many laws it holds
  const panels = await chart.locator('polygon').count();
  expect(panels).toBeGreaterThanOrEqual(2);
  expect(await chart.locator('rect[fill-opacity="0.28"]').count()).toBe(panels);
  await expect(chart).toContainText(/n = [\d,]+ of [\d,]+/);
  expect(errors).toEqual([]);
});

test('every reading of a distribution draws, and the choice travels in the link', async ({ page }) => {
  const errors = collectErrors(page);
  await page.goto('/?release=2026-09&v=dist&dm=log10_fvu_val');
  const chart = page.locator(V2 + ' .v2view svg.v2chart').first();
  await expect(chart).toBeVisible({ timeout: 15000 });
  for (const [mode, label] of [['ecdf', /cumulative distribution/], ['cats', /per catalog/], ['rungs', /along the ladder|median, middle half/], ['hist', /one histogram per method/]]) {
    await page.locator(V2 + ` .v2viewbar button[data-set="dmode:${mode}"]`).click();
    await expect(page.locator(V2 + ` .v2viewbar button[data-set="dmode:${mode}"]`)).toHaveAttribute('aria-pressed', 'true');
    await expect(chart).toHaveAttribute('aria-label', label);
    expect(page.url()).toContain('dv=' + mode);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    expect(overflow, mode).toBeLessThanOrEqual(0);
  }
  // the cumulative curves can be read out of all laws: a method that leaves laws unanswered then ends below 100 %
  await page.goto('/?release=2026-09&v=dist&dm=log10_fvu_val&dv=ecdf&dn=all');
  await expect(page.locator(V2 + ' .v2viewbar button[data-set="dnorm:all"]')).toHaveAttribute('aria-pressed', 'true');
  await expect(page.locator(V2 + ' .v2view')).toContainText('ends below 100');
  expect(errors).toEqual([]);
});

test('the budget of a snapshot is stepped on the display itself', async ({ page }) => {
  await page.goto('/?release=2026-09&v=dist&dm=log10_fvu_val&r=16');
  const sel = page.locator(V2 + ' .v2viewbar select[data-state="rung"]');
  await expect(sel).toHaveValue('16');
  await expect(page.locator(V2 + ' .v2view svg.v2chart').first()).toContainText('at budget 16');
  await page.locator(V2 + ' .v2viewbar .v2stepbtn[aria-label^="higher"]').click();
  await expect(sel).toHaveValue('32');
  await expect(page.locator(V2 + ' .v2view svg.v2chart').first()).toContainText('at budget 32');
  await expect(page.locator(V2 + ' select.v2rung')).toHaveValue('32');   // one budget, two places to set it
  await sel.selectOption('8');
  await expect(page.locator(V2 + ' .v2view svg.v2chart').first()).toContainText('at budget 8');
  // the Catalogs matrix and the by-catalog table carry the same stepper
  await page.locator(V2 + ' .v2tab[data-view="matrix"]').click();
  await expect(page.locator(V2 + ' .v2viewbar select[data-state="rung"]')).toHaveValue('8');
});

test('a rate is shown per catalog, with a way to a distribution', async ({ page }) => {
  await page.goto('/?release=2026-09&v=dist&dm=numeric_recovery_val&r=16');
  await expect(page.locator(V2 + ' .v2view svg.v2chart').first()).toHaveAttribute('aria-label', /per catalog/);
  await expect(page.locator(V2 + ' .v2viewbar button[data-set^="dmode:"]')).toHaveCount(0);   // no reading applies to a hit-or-miss
  await page.locator(V2 + ' .v2view button[data-set="dmetric:log10_fvu_val"]').click();
  await expect(page.locator(V2 + ' .v2view svg.v2chart').first()).toHaveAttribute('aria-label', /one histogram per method/);
});

// ---- Ranks ---------------------------------------------------------------------------------------------------------
test('the ranks view places the methods, names what it ranks on and how it holds them equal', async ({ page }) => {
  const errors = collectErrors(page);
  await page.goto('/?release=2026-09&v=ranks&x=rung&r=16');
  const chart = page.locator(V2 + ' .v2view svg.v2rankchart');
  await expect(chart).toBeVisible({ timeout: 15000 });
  await expect(chart).toHaveAttribute('aria-label', /Mean rank on .* budget 16/);
  await expect(chart).toContainText('critical difference');
  await expect(page.locator(V2 + ' .v2viewbar .v2tag-primary')).toBeVisible();
  // mean ranks: one per ranked method, each within [1, k], and they sum to k (k + 1) / 2 as ranks must
  const ranks = await page.locator(V2 + ' .v2ranktable tbody tr td:nth-child(3)').allTextContents();
  const k = ranks.length;
  expect(k).toBeGreaterThanOrEqual(2);
  const vals = ranks.map(Number);
  for (const v of vals) { expect(v).toBeGreaterThanOrEqual(1); expect(v).toBeLessThanOrEqual(k); }
  expect(Math.abs(vals.reduce((a, b) => a + b, 0) - k * (k + 1) / 2)).toBeLessThan(0.02 * k);
  expect(vals.slice().sort((a, b) => a - b)).toEqual(vals);   // the standings are in order
  // head to head: k x k, and a pair's two shares plus their ties make 100 %
  await expect(page.locator(V2 + ' .v2h2h tbody tr')).toHaveCount(k);
  const cell = (r, c) => page.locator(V2 + ` .v2h2h tbody tr:nth-child(${r}) td:nth-child(${c + 1})`).textContent();
  const num = (t) => parseFloat(t), tied = (t) => parseFloat(t.split('%')[1]);
  const ab = await cell(1, 2), ba = await cell(2, 1);
  expect(Math.abs(num(ab) + num(ba) + tied(ab) - 100)).toBeLessThan(1.6);
  // the standings along the ladder
  await expect(page.locator(V2 + ' .v2view svg.v2chart').last()).toContainText('Comparisons won');
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(overflow).toBeLessThanOrEqual(0);
  expect(errors).toEqual([]);
});

test('ranks follow the selection: another metric, fewer methods, fewer catalogs', async ({ page }) => {
  await page.goto('/?release=2026-09&v=ranks&x=rung&r=16');
  const rows = page.locator(V2 + ' .v2ranktable tbody tr');
  await expect(rows.first()).toBeVisible({ timeout: 15000 });
  const k = await rows.count();
  const laws = () => page.locator(V2 + ' .v2view').textContent().then((t) => +t.match(/within each of ([\d,]+) laws/)[1].replace(/,/g, ''));
  const all = await laws();
  await page.locator(V2 + ' button[data-act="phys"]').click();
  await expect.poll(laws).toBeLessThan(all);
  await page.locator(V2 + ' button[data-act="all"]').click();
  if (k > 2) {
    await page.locator(V2 + ' .v2methods input[type=checkbox]:checked').first().uncheck();
    await expect(rows).toHaveCount(k - 1);
  }
  // only metrics that can be ranked are offered, and an exploratory one says so
  await page.locator(V2 + ' .v2viewbar .v2pick').click();
  const offered = await page.locator('.v2picker .v2pickitem').count();
  expect(offered).toBe(await page.evaluate(() => window.RESULTS_V2_RANKS['2026-09'].keys.length));
  await page.locator('.v2picker .v2pickitem[data-k="mdl_ratio"]').click();
  await expect(page.locator(V2 + ' .v2view svg.v2rankchart')).toHaveAttribute('aria-label', /MDL ratio/);
  await expect(page.locator(V2 + ' .v2viewbar .v2tag-primary')).toHaveCount(0);
  expect(page.url()).toContain('rm=mdl_ratio');
});

test('ranks hold the methods equal on reference-machine time when it exists', async ({ page }) => {
  await page.goto('/?release=2026-09&v=ranks&x=time');
  if (!(await hasRefTiming(page))) { test.skip(); }
  const chart = page.locator(V2 + ' .v2view svg.v2rankchart');
  await expect(chart).toBeVisible({ timeout: 15000 });
  await expect(chart).toHaveAttribute('aria-label', /s per problem/);
  await expect(page.locator(V2 + ' .v2viewbar select[data-state="tbudget"]')).toBeVisible();
  await expect(page.locator(V2 + ' .v2viewbar select[data-state="rung"]')).toHaveCount(0);
  // every ranked method's budget was timed within the limit
  const limit = await page.locator(V2 + ' .v2viewbar select[data-state="tbudget"]').evaluate((s) => parseFloat(s.options[s.selectedIndex].text));
  const secs = await page.locator(V2 + ' .v2ranktable tbody tr td:nth-child(2) .v2ci-txt').allTextContents();
  expect(secs.length).toBeGreaterThanOrEqual(2);
  for (const t of secs) { expect(parseFloat(t)).toBeLessThanOrEqual(limit); }
  await page.locator(V2 + ' .v2viewbar button[data-set="xaxis:rung"]').click();
  await expect(chart).toHaveAttribute('aria-label', /budget \d+/);
});

test('a pooled number appears only where a method has finished every selected catalog', async ({ page }) => {
  // the release payload, with one more method that has finished a single catalog at budget 16: the one corpus that
  // is four fifths of the laws, which is as close to the whole as a part can come
  await page.route('**/data/2026-09/results.js', async (route) => {
    const res = await route.fetch();
    const add = `;(function () { var D = window.RESULTS_V2;
      var donor = D.methods.filter(function (m) { return D.cells[m.key] && Object.keys(D.cells[m.key]).length > 3; })[0];
      var cat = D.catalogs.slice().sort(function (a, b) { return b.laws - a.laws; }).map(function (c) { return c.key; })
        .filter(function (c) { return D.cells[donor.key][c] && D.cells[donor.key][c]['16']; })[0];
      D.methods.push({ key: 'fixture-begun', label: 'Fixture just begun', param: 'draws', budget: 'candidates', color: '#555555', group: 'baseline', provenance: 'upstream_default', selection: '' });
      D.cells['fixture-begun'] = {}; D.cells['fixture-begun'][cat] = { '16': D.cells[donor.key][cat]['16'] }; D.status['fixture-begun'] = [1, 100];
      window.FIXTURE_CAT = cat; })();`;
    await route.fulfill({ response: res, body: (await res.text()) + add });
  });
  await page.goto('/?release=2026-09&v=table&rows=rungs&p=numeric_recovery_val&x=rung');
  await expect(page.locator(V2 + ' .v2methods')).toContainText('Fixture just begun');
  const row16 = page.locator(V2 + ' .v2table tbody tr').filter({ has: page.locator('td:first-child', { hasText: /^16$/ }) });
  await expect(row16).toBeVisible({ timeout: 15000 });
  // the others are still pooled over everything, and the newcomer shows nothing at all
  const all = await page.evaluate(() => window.RESULTS_V2.catalogs.length);
  expect(+(await row16.locator('td:nth-child(2)').textContent()).match(/\((\d+) catalogs\)/)[1]).toBe(all);
  await expect(row16.locator('td').last()).toHaveText('');
  await expect(page.locator(V2 + ' .v2view svg circle[fill="var(--surface)"]')).toHaveCount(0);   // no hollow markers anywhere
  // narrowed to the catalog it HAS finished, it is complete there and gets its number
  const cat = await page.evaluate(() => window.FIXTURE_CAT);
  await page.locator(V2 + ' button[data-act="none"]').click();
  await page.locator(V2 + ` .v2cats input[data-c="${cat}"]`).check();
  await expect(row16.locator('td').last()).not.toHaveText('');
  // and it is ranked there, but not over the whole selection
  await page.goto('/?release=2026-09&v=ranks&x=rung&r=16&c=all');
  await expect(page.locator(V2 + ' .v2ranktable')).toBeVisible({ timeout: 15000 });
  await expect(page.locator(V2 + ' .v2ranktable')).not.toContainText('Fixture just begun');
  await expect(page.locator(V2 + ' .v2view')).toContainText(/Fixture just begun ha(s|ve) not finished every selected catalog/);
});


test('a method without outcomes against another sits out of the ranking instead of emptying it', async ({ page }) => {
  // the release rankings with every outcome between two ranked methods at budget 16 removed: what an overlay sealed
  // before another method had results looks like
  await page.route('**/data/2026-09/ranks.js', async (route) => {
    const res = await route.fetch();
    const cut = `;(function () { var R = window.RESULTS_V2_RANKS['2026-09'], D = window.RESULTS_V2;
      var full = D.methods.map(function (m) { return m.key; }).filter(function (k) { return D.catalogs.every(function (c) { return D.cells[k] && D.cells[k][c.key] && D.cells[k][c.key]['16']; }); });
      var key = Object.keys(R.pairs).filter(function (p) { var ab = p.split('|'); return full.indexOf(ab[0]) >= 0 && full.indexOf(ab[1]) >= 0; })[0];
      Object.keys(R.pairs[key]).forEach(function (c) { delete R.pairs[key][c]['16']; });
      window.FIXTURE_PAIR = key.split('|'); })();`;
    await route.fulfill({ response: res, body: (await res.text()) + cut });
  });
  await page.goto('/?release=2026-09&v=ranks&x=rung&r=16&c=all');
  const table = page.locator(V2 + ' .v2ranktable').first();
  await expect(table).toBeVisible({ timeout: 15000 });
  const pair = await page.evaluate(() => window.FIXTURE_PAIR.map((k) => window.RESULTS_V2.methods.filter((m) => m.key === k)[0].label));
  const names = await table.locator('tbody tr td:first-child').allTextContents();
  const ranked = pair.filter((label) => names.some((n) => n.trim() === label));
  expect(ranked.length).toBe(1);                                   // one of the two is ranked with everybody else
  expect(names.length).toBeGreaterThanOrEqual(2);
  const out = pair.filter((label) => ranked.indexOf(label) < 0)[0];
  await expect(page.locator(V2 + ' .v2view')).toContainText(out + ' carries no pairwise outcomes against all of the other selected methods at budget 16 and sits out.');
  const vals = (await table.locator('tbody tr td:nth-child(3)').allTextContents()).map(Number);
  expect(Math.abs(vals.reduce((a, b) => a + b, 0) - vals.length * (vals.length + 1) / 2)).toBeLessThan(0.02 * vals.length);
});

test('outcomes at a time limit that compared another rung than the method sits on now count as missing', async ({ page }) => {
  await page.route('**/data/2026-09/ranks.js', async (route) => {
    const res = await route.fetch();
    const stale = `;(function () { var R = window.RESULTS_V2_RANKS['2026-09'];
      var a = Object.keys(R.at).filter(function (k) { return Object.keys(R.at[k]).length > 3; })[0]; R.rungs = R.rungs || {};
      Object.keys(R.pairs).forEach(function (p) { if (p.split('|').indexOf(a) < 0) { return; } R.rungs[p] = {}; R.budgets.forEach(function (b) { R.rungs[p][b] = [-1, -1]; }); });
      window.FIXTURE_STALE = a; })();`;
    await route.fulfill({ response: res, body: (await res.text()) + stale });
  });
  await page.goto('/?release=2026-09&v=ranks&x=time&c=all');
  if (!(await hasRefTiming(page))) { test.skip(); }
  const table = page.locator(V2 + ' .v2ranktable').first();
  await expect(table).toBeVisible({ timeout: 15000 });
  const label = await page.evaluate(() => window.RESULTS_V2.methods.filter((m) => m.key === window.FIXTURE_STALE)[0].label);
  const names = (await table.locator('tbody tr td:first-child').allTextContents()).map((n) => n.trim());
  expect(names).not.toContain(label);
  expect(names.length).toBeGreaterThanOrEqual(2);
  await expect(page.locator(V2 + ' .v2view')).toContainText(new RegExp(label.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ' carries no pairwise outcomes against all of the other selected methods at [\\d.]+ s per problem and sits out'));
});
