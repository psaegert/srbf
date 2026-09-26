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
// the metric floor: the 20 metrics every release carries (those of the site's first release, under schema-2 keys)
const LEGACY_METRICS = ['numeric_recovery_val', 'expr_length_ratio', 'log10_fvu_val', 'log10_fvu_fit', 'numeric_recovery_fit', 'success',
  'skeleton_match_raw', 'f1_score', 'precision_score', 'recall_score', 'edit_distance_norm', 'zss_edit_distance', 'expr_length_ratio_abserr',
  'predicted_skeleton_prefix_length', 'skeleton_length', 'n_constants_ratio', 'n_constants_delta', 'total_nestedness_delta', 'predicted_log_prob',
  'predicted_score'];   // fit_time is NOT here: the first release published an as-run wall clock, this one publishes none

test('the Results page opens on the explorer and renders its curves', async ({ page }) => {
  const errors = collectErrors(page);
  await page.goto('/results.html');
  await expect(page.locator(V2)).toBeVisible();
  await expect(page.locator(V2 + ' svg.v2chart').first()).toBeVisible();
  await expect(page.locator('#release-switch')).toHaveCount(0);   // one release: nothing to switch
  expect(errors).toEqual([]);
});

test('the metric registry carries its floor and the headline metrics', async ({ page }) => {
  await page.goto('/results.html');
  const keys = await page.evaluate(() => window.RESULTS_V2.metrics.map((m) => m.key));
  for (const k of LEGACY_METRICS.concat(['symbolic_recovery', 'mdl_ratio', 'r2_val'])) { expect(keys, k).toContain(k); }
  expect(keys.length).toBeGreaterThanOrEqual(24);
  const count = await page.evaluate(() => document.querySelectorAll('#results-explorer-v2 .v2metric').length);
  expect(count).toBe(keys.length);
});

for (const view of VIEWS) {
  test(`the ${view} view renders from a deep link without errors or overflow`, async ({ page }) => {
    const errors = collectErrors(page);
    await page.goto(`/results.html?release=2026-09&v=${view}&f=log10_fvu_val&r=32`);
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
  await page.goto('/results.html?release=2026-09&v=table&p=log10_fvu_val,mdl_ratio&s=median');
  await expect(page.locator(V2 + ' table')).toBeVisible();
  await expect.poll(async () => page.evaluate(() => Object.keys((window.RESULTS_V2_HIST || {})['2026-09'] || {}).length), { timeout: 15000 }).toBeGreaterThanOrEqual(2);
  await expect.poll(async () => (await page.locator(V2 + ' table tbody td').allTextContents()).filter((t) => /^-?\d/.test(t.trim())).length, { timeout: 15000 }).toBeGreaterThan(0);
  expect(errors).toEqual([]);
});

test('the paired view loads its contrasts and shows a baseline selector', async ({ page }) => {
  const errors = collectErrors(page);
  await page.goto('/results.html?release=2026-09&v=paired&p=numeric_recovery_val');
  await expect(page.locator(V2 + ' select.v2base')).toBeVisible({ timeout: 15000 });
  await expect.poll(async () => page.evaluate(() => Object.keys((window.RESULTS_V2_PAIRED || {})['2026-09'] || {}).length), { timeout: 15000 }).toBeGreaterThanOrEqual(1);
  await expect(page.locator(V2 + ' .v2view svg.v2chart').first()).toBeVisible();
  expect(errors).toEqual([]);
});

test('catalog and method controls change the pooled charts', async ({ page }) => {
  const errors = collectErrors(page);
  await page.goto('/results.html?release=2026-09&v=curves');
  const count = page.locator(V2 + ' .v2catcount');
  const before = await count.textContent();
  await page.locator(V2 + ' button[data-act="phys"]').click();
  await expect(count).not.toHaveText(before);
  await page.locator(V2 + ' button[data-act="none"]').click();
  await expect(page.locator(V2 + ' svg.v2chart').first()).toContainText('no problem set selected');
  await page.locator(V2 + ' button[data-act="all"]').click();
  await expect(count).toHaveText(before);
  expect(errors).toEqual([]);
});

test('the view state round-trips through the URL', async ({ page }) => {
  await page.goto('/results.html?release=2026-09&v=matrix&f=mdl_ratio&r=16&c=phys&s=mean&pool=own&thin=1&ci=0');   // pool= and thin= are old links: read, ignored
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
  await page.goto('/results.html?release=2026-09&v=table');
  await page.locator(V2 + ' .v2metrics .v2help').first().click();
  await expect(page.locator('.v2pop')).toBeVisible();
  await expect(page.locator('.v2pop')).toContainText('The share of problems');
  await page.keyboard.press('Escape');
  await expect(page.locator('.v2pop')).toHaveCount(0);
  await page.locator(V2 + ' .v2view .v2term[data-term="complete"]').first().click();
  await expect(page.locator('.v2pop')).toContainText('all the problem sets you selected');
});

// A published time is a CALIBRATED time. Seconds measured wherever a unit happened to run are not comparable
// between methods, so the release must not carry them and no chart may draw them.
const hasRefTiming = (page) => page.evaluate(() => Object.keys(window.RESULTS_V2.timing || {})
  .some((k) => Object.keys(window.RESULTS_V2.timing[k]).length));

// A chart on the time axis says so at every width: the visible label shortens to "time (s)" on a narrow screen, so
// the calibration itself is asserted on the aria-label, which is width-independent.
async function calibrated(chart) {
  await expect(chart).toContainText(/time (per problem )?\(s/);
  await expect(chart).toHaveAttribute('aria-label', /timed on one workstation/);
}

test('a time axis needs one calibrated method, and an uncalibrated one costs only its own points', async ({ page }) => {
  await page.goto('/results.html');
  const axis = page.locator(V2 + ' .v2plot .v2xsel').first();
  if (!await hasRefTiming(page)) {
    await expect(axis).toHaveAttribute('data-k', 'rung');                 // the budget, never an uncalibrated time
    await expect(page.locator(V2 + ' .v2view svg.v2chart').first()).not.toContainText(/time (per problem )?\(s/);
    await axis.click();
    await expect(page.locator('.v2pickitem[data-k="time"]')).toBeDisabled();
    await page.keyboard.press('Escape');
  } else {
    await expect(axis).toHaveAttribute('data-k', 'time');
    await calibrated(page.locator(V2 + ' .v2view svg.v2chart').first());
  }
  // whatever the state, nothing anywhere may say a time was measured "as run"
  expect(await page.content()).not.toContain('as run');
  // the headline on the home page follows the same rule
  const timed = await hasRefTiming(page);
  await page.goto('/');
  const head = page.locator('#results-headline-v2 svg.v2chart').first();
  await expect(head).toBeVisible();
  if (timed) { await calibrated(head); } else { await expect(page.locator('#results-headline-v2')).not.toContainText(/time (per problem )?\(s/); }
  expect(await page.content()).not.toContain('as run');
});

test('the release publishes no as-run wall clock', async ({ page }) => {
  await page.goto('/results.html');
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
  await page.goto('/results.html?release=2026-09&v=curves');
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

test('the home page holds the headline\'s two fixed charts, and the explorer its own page', async ({ page }) => {
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
  await expect(head.locator('svg.v2chart').nth(1)).toContainText('MDL Ratio');   // the metric's own name, as everywhere else
  await expect(head.locator('svg.v2chart').nth(1)).toContainText('FVU');
  // fixed: no control on the home page, and the page's address stays its own
  await expect(page.locator(V2)).toHaveCount(0);
  await expect(page.locator('.explore-cta a')).toHaveAttribute('href', 'results.html');
  expect(new URL(page.url()).search).toBe('');
  // and the explorer's page has no headline
  await page.goto('/results.html');
  await expect(page.locator(V2 + ' svg.v2chart').first()).toBeVisible();
  await expect(page.locator('#results-headline-v2')).toHaveCount(0);
  expect(errors).toEqual([]);
});

test('a link to a view of the explorer on the home page opens the Results page with the same settings', async ({ page }) => {
  const q = '?release=2026-09&v=ranks&x=rung&r=16';
  await page.goto('/' + q);
  await expect(page).toHaveURL(/\/results\.html\?/);
  await expect(page.locator(V2 + ' .v2tab.active')).toHaveAttribute('data-view', 'ranks');
});

test('the public pages carry no private overlay', async ({ page }) => {
  for (const url of ['/', '/results.html']) {
    await page.goto(url);
    expect(await page.evaluate(() => typeof window.RESULTS_V2_PRIVATE), url).toBe('undefined');
    expect(await page.content(), url).not.toContain('private/');
  }
});

test('every method says how it picks the prediction it submits', async ({ page }) => {
  await page.goto('/results.html?release=2026-09&v=curves');
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
  await page.goto('/results.html?release=2026-09&v=curves');
  await expect(page.locator(V2 + ' .v2tab[data-view="table"]')).toHaveText('Tables');
  await page.goto('/');                                                   // the visual abstract on the home page says so too
  await expect(page.locator('#va')).toContainText('Tables');
});

test('the mean is the default statistic, and the median stays one click away', async ({ page }) => {
  await page.goto('/results.html?release=2026-09&v=curves&p=log10_fvu_val');
  await expect(page.locator(V2 + ' input[name="v2stat"][value="mean"]')).toBeChecked();
  const chart = page.locator(V2 + ' .v2view svg.v2chart').first();
  await expect(chart).toBeVisible();
  // switching the statistic moves the data, never the wording
  const xLabel = async (svg) => (await svg.locator('text').allTextContents()).find((t) => /^time (per problem )?\(s/.test(t));
  const before = await xLabel(chart);
  await page.locator(V2 + ' input[name="v2stat"][value="median"]').check();
  await expect(chart).not.toContainText('loading', { timeout: 15000 });
  expect(await xLabel(chart)).toBe(before);
  // the headline shows the mean, drawn from the cell sums: a mean needs no histogram
  await page.goto('/');
  await expect(page.locator('#results-headline-v2 .v2hlsub')).toContainText('mean');
  await expect(page.locator('#results-headline-v2 svg.v2chart').first()).toBeVisible();
});

test('each display carries only the controls it can use', async ({ page }) => {
  const shown = () => page.evaluate(() => [...document.querySelectorAll('#results-explorer-v2 [data-uses]')]
    .filter((e) => !e.hidden).flatMap((e) => e.dataset.uses.split(' ')));
  await page.goto('/results.html?release=2026-09&v=curves');
  const curves = await shown();
  for (const k of ['stat', 'ci']) { expect(curves, k).toContain(k); }
  for (const k of ['pool', 'thin']) { expect(curves, k).not.toContain(k); }   // a pooled number is complete or absent: nothing to switch
  for (const k of ['plots', 'xaxis', 'focus', 'rung', 'base', 'rows']) { expect(curves, k).not.toContain(k); }
  await page.locator(V2 + ' .v2tab[data-view="table"]').click();
  await expect.poll(shown).toContain('plots');
  await page.locator(V2 + ' .v2tab[data-view="matrix"]').click();
  await expect.poll(shown).not.toContain('plots');
  const matrix = await shown();
  expect(matrix, 'stat').toContain('stat');
  // one budget, one metric, each chosen in one place: on the display's own bar, never repeated in the side panel
  for (const k of ['plots', 'xaxis', 'thin', 'focus', 'rung', 'rows']) { expect(matrix, k).not.toContain(k); }
  await expect(page.locator(V2 + ' .v2viewbar .v2viewpick')).toBeVisible();
  await expect(page.locator(V2 + ' .v2viewbar select[data-state="rung"]')).toBeVisible();
  await page.locator(V2 + ' .v2tab[data-view="paired"]').click();
  await expect.poll(shown).toContain('base');
});

test('every plot carries its own two axes, and plots are added and removed', async ({ page }) => {
  const errors = collectErrors(page);
  await page.setViewportSize({ width: 1600, height: 1100 });
  await page.goto('/results.html?release=2026-09&v=curves&p=time~numeric_recovery_val');
  const cards = page.locator(V2 + ' .v2plot');
  await expect(cards).toHaveCount(1);
  await expect(cards.first().locator('.v2ysel')).toHaveAttribute('data-k', 'numeric_recovery_val');
  // a metric on x makes the plot a trade-off: the budget is gone from both axes
  await pick(page, cards.first().locator('.v2xsel'), 'mdl_ratio');
  await expect(cards.first().locator('svg.v2chart')).toContainText('MDL Ratio');
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
  await page.goto('/results.html?release=2026-09&v=curves&p=time~numeric_recovery_val');
  const M = await page.evaluate(() => (window.RESULTS_V2.metrics || []).find((m) => m.key === 'numeric_recovery_val'));
  // the plot header, the picker entry and the table column all use the registry label: one plain name, no
  // parenthesized qualifier (the abbreviation lives in the short name, the definition in the hint)
  expect(M.label).not.toMatch(/[()]/);
  expect(M.short).toBe('vNRR');
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
  const labels = async (svg) => (await svg.locator('text').allTextContents()).join(' | ');
  // the explorer's two plots, then the home page's two headline charts
  for (const url of ['/results.html?release=2026-09&v=curves&p=rung~numeric_recovery_val,mdl_ratio~log10_fvu_val', '/']) {
    await page.goto(url);
    const charts = page.locator('svg.v2chart');
    await expect(charts.first()).toBeVisible();
    const n = await charts.count();
    expect(n, url).toBeGreaterThanOrEqual(2);
    for (let i = 0; i < n; i++) {
      const t = await labels(charts.nth(i));
      expect(t, `${url} chart ${i} x label`).toMatch(/time per problem|MDL Ratio|candidates per problem/);   // a budget or a metric, never an uncalibrated time
      expect(t, `${url} chart ${i} y label`).toMatch(/Numeric Recovery|log10 FVU/);
    }
  }
});

test('the picker lists every metric in columns and filters', async ({ page }) => {
  const errors = collectErrors(page);
  await page.setViewportSize({ width: 1600, height: 1100 });
  await page.goto('/results.html?release=2026-09&v=curves&p=time~numeric_recovery_val');
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

test('the picker lists plain names, and the filter still answers to the dropped qualifier', async ({ page }) => {
  const errors = collectErrors(page);
  await page.setViewportSize({ width: 1600, height: 1100 });
  await page.goto('/results.html?release=2026-09&v=curves&p=time~numeric_recovery_val');
  await page.locator(V2 + ' .v2plot .v2ysel').click();
  const picker = page.locator('.v2picker');
  await expect(picker).toBeVisible();
  const names = await picker.locator('.v2pickitem > span:first-child').allTextContents();
  expect(names.length).toBeGreaterThan(30);
  expect(names.filter((t) => /[()]/.test(t))).toEqual([]);          // no "(vNRR)", no "(Prediction / Ground Truth)"
  expect(new Set(names).size).toBe(names.length);                    // the names stay distinct without their qualifiers
  await picker.locator('.v2pickq').fill('vnrr');                     // the short name is not in the text, but it answers the filter
  await expect(picker.locator('.v2pickitem:visible')).toHaveCount(1);
  await expect(picker.locator('.v2pickitem:visible')).toHaveAttribute('data-k', 'numeric_recovery_val');
  await page.keyboard.press('Escape');
  expect(errors).toEqual([]);
});

test('a long plot title wraps onto more lines; no title is cut to an ellipsis', async ({ page }) => {
  const errors = collectErrors(page);
  const url = '/results.html?release=2026-09&v=curves&p=time~symbolic_recovery_mask_none,symbolic_recovery_mask_none~edit_distance_norm';
  const read = () => page.locator(V2 + ' .v2plothead .v2picklab').evaluateAll((els) => els.map((el) => {
    const cs = getComputedStyle(el);
    return { text: el.textContent, nowrap: cs.whiteSpace === 'nowrap', clipped: el.scrollWidth > el.clientWidth + 1,
      lines: Math.round(el.getBoundingClientRect().height / parseFloat(cs.lineHeight)) };
  }));
  // wide: the plain names fit on one line, and nothing is set to be cut
  await page.setViewportSize({ width: 1400, height: 900 });
  await page.goto(url);
  await expect(page.locator(V2 + ' .v2plothead .v2picklab')).toHaveCount(4);
  for (const l of await read()) { expect(l.nowrap, l.text).toBe(false); expect(l.clipped, l.text).toBe(false); }
  // narrow: the long names take a second line instead of an ellipsis
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto(url);
  await expect(page.locator(V2 + ' .v2plothead .v2picklab')).toHaveCount(4);
  const narrow = await read();
  for (const l of narrow) { expect(l.nowrap, l.text).toBe(false); expect(l.clipped, l.text).toBe(false); }
  expect(narrow.filter((l) => l.lines >= 2).length).toBeGreaterThan(0);
  expect(errors).toEqual([]);
});

// the fixture overlay and the key it was sealed with (results-site/tools/seal.mjs, tests/fixtures/overlay)
const FIXTURE_KEY = 'fixture-key-for-the-tests';
const FIXTURE = readFileSync(new URL('./fixtures/sealed_fixture.js', import.meta.url), 'utf8');

test('a method is added with its key, and a key that does not fit adds nothing', async ({ page }) => {
  const errors = collectErrors(page);
  await page.addInitScript({ content: FIXTURE });
  await page.goto('/results.html?release=2026-09&v=curves');
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
  await page.goto('/results.html?release=2026-09&v=curves');
  test.skip(!await hasRefTiming(page), 'no reference timing in this release yet');
  await page.locator(V2 + ' .v2addmopen').click();
  await page.locator(V2 + ' .v2addmkey').fill(FIXTURE_KEY);
  await page.locator(V2 + ' [data-act="add-method-go"]').click();
  await expect(page.locator(V2 + ' .v2methods')).toContainText('Fixture Method', { timeout: 20000 });
  // the fixture overlay carries timing {}, so it has no position on the time axis
  expect(await page.evaluate(() => !!(window.RESULTS_V2.timing || {})['fixture-method'])).toBe(false);

  await pick(page, page.locator(V2 + ' .v2plot .v2xsel').first(), 'time');
  const chart = page.locator(V2 + ' .v2view svg.v2chart').first();
  await calibrated(chart);                                            // the axis stands, timed on the workstation
  await expect(chart).toContainText('E2E');                           // and the calibrated methods are still drawn
  await expect(chart, 'an untimed method must not be drawn on a calibrated axis').not.toContainText('Fixture Method');
  await expect(page.locator(V2 + ' .v2view'), 'and it must be named, not dropped in silence').toContainText('Fixture Method');
  await expect(page.locator(V2 + ' .v2view .v2hint').first()).toContainText(/ha(s|ve) not been timed yet/);
  expect(errors).toEqual([]);
});

test('an untimed method changes nothing else on the time axis', async ({ page }) => {
  await page.addInitScript({ content: FIXTURE });
  await page.goto('/results.html?release=2026-09&v=curves');
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
  await page.goto('/results.html?release=2026-09&v=curves');
  // the control is plain, and neither it nor the payload names anything the release does not publish
  await expect(page.locator(V2 + ' .v2addmopen')).toHaveText('open a method with a key');
  const html = await page.content();
  expect(html).not.toMatch(/private|sealed|decrypt|password/i);
});

test('the interval is a band by default, and crosses are a separate switch', async ({ page }) => {
  const errors = collectErrors(page);
  await page.setViewportSize({ width: 1600, height: 1100 });
  await page.goto('/results.html?release=2026-09&v=curves&p=rung~numeric_recovery_val,mdl_ratio~log10_fvu_val');
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
  await page.goto('/');
  const chart = page.locator('#results-headline-v2 svg.v2chart').first();
  await expect(chart).toBeVisible();
  const box = await chart.boundingBox();
  const viewBox = await chart.getAttribute('viewBox');
  expect(box.width).toBeGreaterThan(500);                                  // a chart, not a thumbnail
  expect(Math.abs(Number(viewBox.split(' ')[2]) - box.width)).toBeLessThan(2);   // 1 unit = 1 px: 12 px of label is 12 px
});

test('a display of one chart is drawn at the width it is shown, not stretched from a grid cell', async ({ page }) => {
  await page.setViewportSize({ width: 1600, height: 1000 });
  const stretch = () => page.evaluate(() => [...document.querySelectorAll('#results-explorer-v2 .v2main svg.v2chart')].map((s) => s.getBoundingClientRect().width / s.viewBox.baseVal.width));
  for (const view of ['dist', 'dist&dm=ecdf', 'dist&dm=cats', 'dist&dm=ladder', 'dist&dmetric=numeric_recovery_val', 'ranks']) {
    await page.goto('/results.html?release=2026-09&v=' + view);
    await expect(page.locator(V2 + ' .v2main svg.v2chart').first()).toBeVisible();
    // 1 unit = 1 px: drawn at a grid cell's width and stretched to the column, 12 px of label grew to 20 px
    await expect.poll(async () => { const r = await stretch(); return r.length && r.every((x) => Math.abs(x - 1) < 0.02); }, { message: view }).toBe(true);
  }
});

test('a method marked as the ceiling is drawn dashed in the page ink, legend included', async ({ page }) => {
  // mark the prior as the oracle is marked (dash, ink), in the payload as it is served
  await page.route('**/data/2026-09/results.js*', async (route) => {
    const response = await route.fetch();
    const body = (await response.text()).replace('"key":"prior",', '"key":"prior","dash":true,"ink":true,');
    await route.fulfill({ response, body });
  });
  await page.goto('/results.html?release=2026-09&v=curves&p=rung~numeric_recovery_val');   // the budget axis: the prior has no reference time
  const chart = page.locator(V2 + ' .v2main svg.v2chart').first();
  await expect(chart).toBeVisible();
  const drawn = await chart.evaluate((svg) => {
    const ink = getComputedStyle(document.documentElement).getPropertyValue('--ink').trim();
    const label = [...svg.querySelectorAll('text.leg')].find((t) => t.textContent === 'Flash-ANSR prior');
    const key = label && label.previousElementSibling;
    const lines = [...svg.querySelectorAll('polyline')].filter((l) => l.getAttribute('stroke') === ink);
    return { ink, keyDash: key && key.getAttribute('stroke-dasharray'), keyStroke: key && key.getAttribute('stroke'),
             dashedLines: lines.filter((l) => l.getAttribute('stroke-dasharray')).length,
             otherDashed: [...svg.querySelectorAll('polyline[stroke-dasharray]')].filter((l) => l.getAttribute('stroke') !== ink).length };
  });
  expect(drawn.keyStroke, 'the legend key takes the page ink').toBe(drawn.ink);
  expect(drawn.keyDash, 'the legend key is dashed').toBeTruthy();
  expect(drawn.dashedLines, 'the line is dashed in the ink').toBeGreaterThan(0);
  expect(drawn.otherDashed, 'no other method is dashed').toBe(0);
});

test('a headline chart keeps its title and its y label off the frame', async ({ page }) => {
  await page.setViewportSize({ width: 1400, height: 900 });
  await page.goto('/');
  await expect(page.locator('.headline-v2 svg.v2chart').first()).toBeVisible();
  const gaps = await page.evaluate(() => [...document.querySelectorAll('.headline-v2 svg.v2chart')].map((svg) => {
    const box = svg.getBoundingClientRect(), title = svg.querySelector('text.ct').getBoundingClientRect();
    const ylabel = [...svg.querySelectorAll('text')].find((t) => (t.getAttribute('transform') || '').includes('rotate(-90)'));
    return { top: title.top - box.top, left: ylabel ? ylabel.getBoundingClientRect().left - box.left : null };
  }));
  expect(gaps.length).toBe(2);
  for (const g of gaps) {
    expect(g.top, 'room above the title').toBeGreaterThanOrEqual(12);
    expect(g.left, 'room left of the y label').toBeGreaterThanOrEqual(8);
  }
});

// ---- Distribution: a distribution is what the view shows, in four readings ------------------------------------
test('the distribution view opens on histograms of a continuous metric', async ({ page }) => {
  const errors = collectErrors(page);
  await page.goto('/results.html?release=2026-09');
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
  // a bin taller than the shared scale is a broken bar with its share beside it: no arrow that could point at the panel above
  if (await chart.locator('path.v2break').count()) { await expect(chart).toContainText(/\d+ % in (this|the (left|right)-most) bin/); }
  await expect(chart).not.toContainText('\u25b2');
  expect(errors).toEqual([]);
});

test('every reading of a distribution draws, and the choice travels in the link', async ({ page }) => {
  const errors = collectErrors(page);
  await page.goto('/results.html?release=2026-09&v=dist&dm=log10_fvu_val');
  const chart = page.locator(V2 + ' .v2view svg.v2chart').first();
  await expect(chart).toBeVisible({ timeout: 15000 });
  for (const [mode, label] of [['ecdf', /cumulative distribution/], ['cats', /per problem set/], ['rungs', /by budget|median, middle half/], ['hist', /one histogram per method/]]) {
    await page.locator(V2 + ` .v2viewbar button[data-set="dmode:${mode}"]`).click();
    await expect(page.locator(V2 + ` .v2viewbar button[data-set="dmode:${mode}"]`)).toHaveAttribute('aria-pressed', 'true');
    await expect(chart).toHaveAttribute('aria-label', label);
    expect(page.url()).toContain('dv=' + mode);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    expect(overflow, mode).toBeLessThanOrEqual(0);
  }
  // the cumulative curves can be read out of all laws: a method that leaves laws unanswered then ends below 100 %
  await page.goto('/results.html?release=2026-09&v=dist&dm=log10_fvu_val&dv=ecdf&dn=all');
  await expect(page.locator(V2 + ' .v2viewbar button[data-set="dnorm:all"]')).toHaveAttribute('aria-pressed', 'true');
  await expect(page.locator(V2 + ' .v2view')).toContainText('ends below 100');
  expect(errors).toEqual([]);
});

test('the budget of a snapshot is stepped on the display itself', async ({ page }) => {
  await page.goto('/results.html?release=2026-09&v=dist&dm=log10_fvu_val&r=16');
  const sel = page.locator(V2 + ' .v2viewbar select[data-state="rung"]');
  await expect(sel).toHaveValue('16');
  await expect(page.locator(V2 + ' .v2view svg.v2chart').first()).toContainText('at budget 16');
  await page.locator(V2 + ' .v2viewbar .v2stepbtn[aria-label^="higher"]').click();
  await expect(sel).toHaveValue('32');
  await expect(page.locator(V2 + ' .v2view svg.v2chart').first()).toContainText('at budget 32');
  await expect(page.locator(V2 + ' select.v2rung')).toBeHidden();   // the budget has one place: the display's bar
  await sel.selectOption('8');
  await expect(page.locator(V2 + ' .v2view svg.v2chart').first()).toContainText('at budget 8');
  // the Catalogs matrix and the by-catalog table carry the same stepper
  await page.locator(V2 + ' .v2tab[data-view="matrix"]').click();
  await expect(page.locator(V2 + ' .v2viewbar select[data-state="rung"]')).toHaveValue('8');
});

test('a rate is shown per problem set, with a way to a distribution', async ({ page }) => {
  await page.goto('/results.html?release=2026-09&v=dist&dm=numeric_recovery_val&r=16');
  await expect(page.locator(V2 + ' .v2view svg.v2chart').first()).toHaveAttribute('aria-label', /per problem set/);
  await expect(page.locator(V2 + ' .v2viewbar button[data-set^="dmode:"]')).toHaveCount(0);   // no reading applies to a hit-or-miss
  await page.locator(V2 + ' .v2view button[data-set="dmetric:log10_fvu_val"]').click();
  await expect(page.locator(V2 + ' .v2view svg.v2chart').first()).toHaveAttribute('aria-label', /one histogram per method/);
});

// ---- Ranks ---------------------------------------------------------------------------------------------------------
test('the ranks view places the methods, names what it ranks on and how it holds them equal', async ({ page }) => {
  const errors = collectErrors(page);
  await page.goto('/results.html?release=2026-09&v=ranks&x=rung&r=16');
  const chart = page.locator(V2 + ' .v2view svg.v2rankchart');
  await expect(chart).toBeVisible({ timeout: 15000 });
  await expect(chart).toHaveAttribute('aria-label', /Average place on .* budget 16/);
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
  await page.goto('/results.html?release=2026-09&v=ranks&x=rung&r=16');
  const rows = page.locator(V2 + ' .v2ranktable tbody tr');
  await expect(rows.first()).toBeVisible({ timeout: 15000 });
  const k = await rows.count();
  const laws = () => page.locator(V2 + ' .v2view').textContent().then((t) => +t.match(/on ([\d,]+) problem runs/)[1].replace(/,/g, ''));
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
  await expect(page.locator(V2 + ' .v2view svg.v2rankchart')).toHaveAttribute('aria-label', /MDL Ratio/);
  await expect(page.locator(V2 + ' .v2viewbar .v2tag-primary')).toHaveCount(0);
  expect(page.url()).toContain('rm=mdl_ratio');
});

test('ranks hold the methods equal on reference-machine time when it exists', async ({ page }) => {
  await page.goto('/results.html?release=2026-09&v=ranks&x=time');
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
  await page.goto('/results.html?release=2026-09&v=table&rows=rungs&p=numeric_recovery_val&x=rung');
  await expect(page.locator(V2 + ' .v2methods')).toContainText('Fixture just begun');
  const row16 = page.locator(V2 + ' .v2table tbody tr').filter({ has: page.locator('td:first-child', { hasText: /^16$/ }) });
  await expect(row16).toBeVisible({ timeout: 15000 });
  // the others are still pooled over everything, and the newcomer shows nothing at all
  const all = await page.evaluate(() => window.RESULTS_V2.catalogs.length);
  expect(+(await row16.locator('td:nth-child(2)').textContent()).match(/\((\d+) problem sets\)/)[1]).toBe(all);
  await expect(row16.locator('td').last()).toHaveText('');
  await expect(page.locator(V2 + ' .v2view svg circle[fill="var(--surface)"]')).toHaveCount(0);   // no hollow markers anywhere
  // narrowed to the catalog it HAS finished, it is complete there and gets its number
  const cat = await page.evaluate(() => window.FIXTURE_CAT);
  await page.locator(V2 + ' button[data-act="none"]').click();
  await page.locator(V2 + ` .v2cats input[data-c="${cat}"]`).check();
  await expect(row16.locator('td').last()).not.toHaveText('');
  // and it is ranked there, but not over the whole selection
  await page.goto('/results.html?release=2026-09&v=ranks&x=rung&r=16&c=all');
  await expect(page.locator(V2 + ' .v2ranktable')).toBeVisible({ timeout: 15000 });
  await expect(page.locator(V2 + ' .v2ranktable')).not.toContainText('Fixture just begun');
  await expect(page.locator(V2 + ' .v2view')).toContainText(/Fixture just begun (is|are) not ranked: (it has|they have) not finished all selected problem sets/);
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
  await page.goto('/results.html?release=2026-09&v=ranks&x=rung&r=16&c=all');
  const table = page.locator(V2 + ' .v2ranktable').first();
  await expect(table).toBeVisible({ timeout: 15000 });
  const pair = await page.evaluate(() => window.FIXTURE_PAIR.map((k) => window.RESULTS_V2.methods.filter((m) => m.key === k)[0].label));
  const names = await table.locator('tbody tr td:first-child').allTextContents();
  const ranked = pair.filter((label) => names.some((n) => n.trim() === label));
  expect(ranked.length).toBe(1);                                   // one of the two is ranked with everybody else
  expect(names.length).toBeGreaterThanOrEqual(2);
  const out = pair.filter((label) => ranked.indexOf(label) < 0)[0];
  await expect(page.locator(V2 + ' .v2view')).toContainText(out + ' is not ranked: it has no results on the same problems as all other selected methods at budget 16.');
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
  await page.goto('/results.html?release=2026-09&v=ranks&x=time&c=all');
  if (!(await hasRefTiming(page))) { test.skip(); }
  const table = page.locator(V2 + ' .v2ranktable').first();
  await expect(table).toBeVisible({ timeout: 15000 });
  const label = await page.evaluate(() => window.RESULTS_V2.methods.filter((m) => m.key === window.FIXTURE_STALE)[0].label);
  const names = (await table.locator('tbody tr td:first-child').allTextContents()).map((n) => n.trim());
  expect(names).not.toContain(label);
  expect(names.length).toBeGreaterThanOrEqual(2);
  await expect(page.locator(V2 + ' .v2view')).toContainText(new RegExp(label.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ' is not ranked: it has no results on the same problems as all other selected methods at [\\d.]+ s per problem'));
});

// ---- design contract (2026-09-20): hints, labels, axes, the pinned column, the header ---------------------------
test('no hint opens empty, in any display', async ({ page }) => {
  for (const view of VIEWS) {
    await page.goto(`/results.html?release=2026-09&v=${view}&c=all`);
    await expect(page.locator(V2 + ' .v2tab.active')).toHaveAttribute('data-view', view);
    await page.waitForTimeout(600);
    await page.evaluate(() => document.querySelectorAll('.explorer-v2 details').forEach((d) => { d.open = true; }));   // collapsed sections hold hints too
    // every dotted term and every "?" on screen (the headline, the side panel and the display), each distinct one once
    const hints = page.locator('.explorer-v2 .v2term:visible, .explorer-v2 .v2help:visible');
    const n = await hints.count();
    expect(n, view).toBeGreaterThan(0);
    const seen = new Set();
    for (let i = 0; i < n; i++) {
      const h = hints.nth(i);
      const key = (await h.getAttribute('data-term')) || (await h.getAttribute('data-help')) || '';
      if (seen.has(key)) { continue; }
      seen.add(key);
      await h.scrollIntoViewIfNeeded();
      await h.click();
      const pop = page.locator('.v2pop');
      await expect(pop, `${view}: hint ${i} (${key.slice(0, 40)})`).toBeVisible();
      expect((await pop.textContent()).trim().length, `${view}: hint ${i} (${key.slice(0, 40)})`).toBeGreaterThan(20);
      await page.keyboard.press('Escape');
      await expect(pop).toHaveCount(0);
    }
  }
});

test('the text of an option selects the option: hints never sit inside a label', async ({ page }) => {
  await page.goto('/results.html?release=2026-09&v=curves');
  await page.locator(V2 + ' label', { hasText: /^\s*median\s*$/ }).click();
  await expect(page.locator(V2 + ' input[name="v2stat"][value="median"]')).toBeChecked();
  await expect(page.locator('.v2pop')).toHaveCount(0);
  for (const view of VIEWS) {
    await page.goto(`/results.html?release=2026-09&v=${view}`);
    await expect(page.locator(V2 + ' .v2tab.active')).toHaveAttribute('data-view', view);
    await expect(page.locator(V2 + ' label .v2term, ' + V2 + ' label .v2help, ' + V2 + ' button .v2term, ' + V2 + ' .v2lab .v2term'), view).toHaveCount(0);
  }
});

test('axis ticks are round values with room between their labels', async ({ page }) => {
  await page.setViewportSize({ width: 1360, height: 1000 });
  for (const view of ['curves', 'dist', 'ranks', 'paired']) {
    await page.goto(`/results.html?release=2026-09&v=${view}&c=all`);
    await expect(page.locator(V2 + ' .v2tab.active')).toHaveAttribute('data-view', view);
    await expect(page.locator('.explorer-v2 svg.v2chart').first()).toBeVisible({ timeout: 15000 });
    await page.waitForTimeout(800);
    const report = await page.evaluate(() => {
      const bad = [], num = (t) => parseFloat(t.replace(/\u2212/g, '-').replace(/[^\d.+-]/g, ''));
      const mant = (v) => { const a = Math.abs(v); return a / Math.pow(10, Math.floor(Math.log10(a) + 1e-9)); };
      const near = (a, b) => Math.abs(a - b) < 1e-6 * Math.max(1, Math.abs(a), Math.abs(b));
      // a round axis: equal steps of 1, 2, 2.5 or 5 times a power of ten, or values that are each 1, 2 or 5 times one, or powers of two
      const round = (vs) => {
        if (vs.length < 2) { return true; }
        const d = vs.slice(1).map((v, i) => v - vs[i]);
        if (d.every((x) => near(x, d[0])) && [1, 2, 2.5, 5].some((m) => near(mant(d[0]), m))) { return true; }
        if (vs.every((v) => v > 0 && near(Math.log2(v), Math.round(Math.log2(v))))) { return true; }
        return vs.every((v) => v === 0 || [1, 2, 5].some((m) => near(mant(v), m)));
      };
      document.querySelectorAll('.explorer-v2 svg.v2chart').forEach((svg, ci) => {
        const ticks = [...svg.querySelectorAll('text.tick')].filter((t) => /^[-+\u2212\u00d7\s]*[\d.,]+\s*(%|s|pp)?$/.test(t.textContent.trim()))
          .map((t) => ({ r: t.getBoundingClientRect(), s: t.textContent.trim(), a: t.getAttribute('text-anchor') }));
        const ys = ticks.filter((t) => t.a === 'end').sort((a, b) => a.r.top - b.r.top), xs = ticks.filter((t) => t.a === 'middle').sort((a, b) => a.r.left - b.r.left);
        for (const [axis, list, gap] of [['y', ys, (a, b) => b.r.top - a.r.bottom], ['x', xs, (a, b) => b.r.left - a.r.right]]) {
          if (!round(list.map((t) => num(t.s.replace(/,/g, ''))).sort((a, b) => a - b))) { bad.push(`chart ${ci} ${axis}: "${list.map((t) => t.s).join(' | ')}" is not a round axis`); }
          for (let i = 1; i < list.length; i++) { if (gap(list[i - 1], list[i]) < 12) { bad.push(`chart ${ci} ${axis}: "${list[i - 1].s}" and "${list[i].s}" are ${Math.round(gap(list[i - 1], list[i]))} px apart`); } }
        }
      });
      return bad;
    });
    expect(report, view).toEqual([]);
  }
});

test('a chart hugs its data: neither a reference line nor a snapped axis leaves it mostly empty', async ({ page }) => {
  await page.setViewportSize({ width: 1360, height: 1000 });
  for (const view of ['curves', 'paired']) {
    await page.goto(`/results.html?release=2026-09&v=${view}&c=all`);
    await expect(page.locator(V2 + ' .v2view svg.v2chart').first()).toBeVisible({ timeout: 15000 });
    await page.waitForTimeout(800);
    const fills = await page.evaluate(() => [...document.querySelectorAll('.explorer-v2 svg.v2chart')].map((svg, ci) => {
      const marks = [...svg.querySelectorAll('circle')].map((c) => c.getBoundingClientRect()).filter((r) => r.width > 0);
      const grid = [...svg.querySelectorAll('line.grid')].map((l) => l.getBoundingClientRect());
      if (marks.length < 4 || !grid.length) { return null; }
      const gx0 = Math.min(...grid.map((r) => r.left)), gx1 = Math.max(...grid.map((r) => r.right)), gy0 = Math.min(...grid.map((r) => r.top)), gy1 = Math.max(...grid.map((r) => r.bottom));
      return { ci, w: (Math.max(...marks.map((r) => r.right)) - Math.min(...marks.map((r) => r.left))) / (gx1 - gx0), h: (Math.max(...marks.map((r) => r.bottom)) - Math.min(...marks.map((r) => r.top))) / (gy1 - gy0) };
    }).filter(Boolean));
    expect(fills.length, view).toBeGreaterThan(2);
    for (const f of fills) { expect(f.w, `${view} chart ${f.ci} width`).toBeGreaterThan(0.8); expect(f.h, `${view} chart ${f.ci} height`).toBeGreaterThan(0.6); }
  }
});

test('the pinned first column of a wide table sits flush left at any scroll position', async ({ page }) => {
  await page.setViewportSize({ width: 900, height: 900 });
  for (const view of ['table', 'matrix', 'ranks']) {
    await page.goto(`/results.html?release=2026-09&v=${view}&c=all`);
    const wrap = page.locator(V2 + ' .v2view .v2table-wrap').last();
    await expect(wrap).toBeVisible({ timeout: 15000 });
    const gaps = await wrap.evaluate((w) => {
      const out = [];
      for (const x of [0, 60, w.scrollWidth]) {
        w.scrollLeft = x;
        const box = w.getBoundingClientRect(), left = box.left + w.clientLeft;
        w.querySelectorAll('tbody tr td:first-child, thead tr th:first-child').forEach((c) => { out.push(Math.abs(c.getBoundingClientRect().left - left)); });
      }
      return Math.max(...out);
    });
    expect(gaps, view).toBeLessThanOrEqual(0.5);   // no sliver of the scrolled columns to the left of the pinned one
  }
});

test('the page names itself once: the bar carries the name, the title says what the page is', async ({ page }) => {
  await page.goto('/');
  const name = 'Symbolic Regression Benchmark Framework';
  await expect(page.locator('.site-header .brand')).toContainText(name);
  await expect(page.locator('.site-header .brand .brand-mark')).toHaveCount(0);   // the icon already reads "srbf"
  await expect(page.locator('main h1')).not.toContainText(name);
  await expect(page.locator('main h1')).toHaveText('Benchmark results');
});

test('the guide explains the results from scratch', async ({ page }) => {
  await page.goto('/guide.html');
  for (const h of ['Problems', 'Checking a prediction', 'Budgets and time', 'Why some points are missing', 'Problems without a usable formula', 'Intervals']) {
    await expect(page.locator('#about h3', { hasText: new RegExp('^' + h + '$') })).toBeVisible();
  }
  await expect(page.locator('[data-for]')).toHaveCount(0);   // one release: no text is written for another one
});

// ---- how much of a point is there ---------------------------------------------------------------------------------
// A method that has answered half of the laws at budget 16 and 95 % of them at budget 32, in every catalog. Its
// constant-count ratio exists for 40 laws, of the 42 that have a constant at all.
async function withHalfAnswered(page) {
  await page.route('**/data/2026-09/results.js', async (route) => {
    const res = await route.fetch();
    const add = `;(function () { var D = window.RESULTS_V2;
      // its answers have a token F1 of 0.8; the laws it failed are counted at 0, and the cell says how many those are
      var mk = function (k) { return { state: 'complete', n: 100, ok: k, e: { n_constants_ratio: 42 }, w: { f1_score: 100 - k }, m: {
        numeric_recovery_val: [10, 100], success: [k, 100], mdl_ratio: [k, k, 1.5 * k, 2.5 * k], log10_fvu_val: [k, k, -2 * k, 5 * k],
        f1_score: [100, 100, 0.8 * k, 0.64 * k], n_constants_ratio: [40, 40, 44, 50] } }; };
      D.methods.push({ key: 'fixture-half', label: 'Fixture half answered', param: 'draws', budget: 'candidates', color: '#555555', group: 'baseline', provenance: 'upstream_default', selection: '' });
      D.cells['fixture-half'] = {}; D.catalogs.forEach(function (c) { D.cells['fixture-half'][c.key] = { '16': mk(50), '32': mk(95) }; });
      D.status['fixture-half'] = [2, 2]; })();`;
    await route.fulfill({ response: res, body: (await res.text()) + add });
  });
}
const HOLLOW = 'circle[fill="var(--surface)"]';
const HALF = '/results.html?release=2026-09&m=fixture-half&x=rung&s=mean&p=rung~mdl_ratio,rung~f1_score,rung~n_constants_ratio,rung~numeric_recovery_val';
async function setThreshold(page, value) {
  await page.locator(V2 + ' .v2valid').fill(String(value));
  await expect(page.locator(V2 + ' .v2validout')).toHaveText(value + ' % of the problems');
}

test('a point that rests on too few problems is drawn hollow, and the reader sets how few', async ({ page }) => {
  const errors = collectErrors(page);
  await withHalfAnswered(page);
  await page.goto(HALF + '&v=curves');
  const plot = (i) => page.locator(V2 + ' .v2plot').nth(i);
  await expect(plot(0).locator('svg circle')).toHaveCount(2);
  // the default threshold is 90 %: half of the laws is too few, 95 % is enough
  await expect(page.locator(V2 + ' .v2valid')).toHaveValue('90');
  await expect(plot(0).locator(HOLLOW)).toHaveCount(1);
  await expect(plot(0).locator(HOLLOW + ' title')).toHaveText(/at budget 16.*50 % of the problems have a value/);
  await expect(page.locator(V2 + ' .v2hollownote')).toContainText('fewer than 90 % of the problems');
  // a metric with a worst value counts every law, and so does a rate: never hollow
  await expect(plot(1).locator('svg circle')).toHaveCount(2);
  await expect(plot(1).locator(HOLLOW)).toHaveCount(0);
  await expect(plot(3).locator(HOLLOW)).toHaveCount(0);
  // 40 of the 42 laws that HAVE a constant: the base is the laws the metric can be defined for
  await expect(plot(2).locator(HOLLOW)).toHaveCount(0);
  await setThreshold(page, 100);
  await expect(plot(0).locator(HOLLOW)).toHaveCount(2);
  await expect(plot(2).locator(HOLLOW)).toHaveCount(2);
  await expect(plot(1).locator(HOLLOW)).toHaveCount(0);
  await expect(plot(3).locator(HOLLOW)).toHaveCount(0);
  expect(page.url()).toContain('ok=100');
  await setThreshold(page, 0);   // off
  await expect(page.locator(V2 + ' .v2view ' + HOLLOW)).toHaveCount(0);
  await expect(page.locator(V2 + ' .v2hollownote')).toHaveCount(0);
  // a link carries the threshold
  await page.goto(HALF + '&v=curves&ok=40');
  await expect(page.locator(V2 + ' .v2valid')).toHaveValue('40');
  await expect(plot(0).locator('svg circle')).toHaveCount(2);
  await expect(plot(0).locator(HOLLOW)).toHaveCount(0);
  // the headline's two fixed charts on the home page do not follow the control: they keep the default
  await page.goto('/');
  await expect(page.locator('#results-headline-v2 ' + HOLLOW).first()).toBeVisible();
  await expect(page.locator('#results-headline-v2')).toContainText('fewer than 90 % of the problems');
  expect(errors, errors.join('\n')).toEqual([]);
});

test('a table marks the same numbers, and a display without such numbers has no threshold', async ({ page }) => {
  await withHalfAnswered(page);
  await page.goto(HALF + '&v=table&rows=rungs');
  const row = (r) => page.locator(V2 + ' .v2table tbody tr').filter({ has: page.locator('td:first-child', { hasText: new RegExp('^' + r + '$') }) });
  await expect(row(16)).toBeVisible();
  await expect(row(16).locator('.v2hollow')).toHaveCount(1);   // the description-length ratio, not the overlap, the ratio of constants or the rate
  await expect(row(32).locator('.v2hollow')).toHaveCount(0);
  await expect(page.locator(V2 + ' .v2view')).toContainText('fewer than 90 % of the problems');
  for (const v of ['dist', 'ranks', 'paired']) {
    await page.locator(V2 + ` .v2tab[data-view="${v}"]`).click();
    await expect(page.locator(V2 + ' .v2valid')).toBeHidden();
  }
  await page.locator(V2 + ' .v2tab[data-view="matrix"]').click();
  await expect(page.locator(V2 + ' .v2valid')).toBeVisible();
});

test('R² has no floor and is read by its median', async ({ page }) => {
  const errors = collectErrors(page);
  const asked = [];
  page.on('request', (r) => { if (r.url().includes('/hist/')) { asked.push(r.url().split('/hist/')[1]); } });
  await page.goto('/results.html?release=2026-09&v=curves&x=rung&s=mean&p=rung~r2_val');
  const reg = await page.evaluate(() => { const D = window.RESULTS_V2; const m = D.metrics.filter((x) => x.key === 'r2_val')[0];
    return { label: m.label, via: m.median_via, lo: m.hist.lo, ranks: D.rank_keys, paired: D.paired_keys,
      worst: D.metrics.filter((x) => x.worst !== undefined).map((x) => x.key + '=' + x.worst).sort() }; });
  expect(reg.label).toBe('R², Validation');
  expect(reg.via).toBe('log10_fvu_val');
  expect(reg.lo).toBeLessThan(0);
  expect(reg.ranks).not.toContain('r2_val');   // it orders the answers exactly as the FVU does
  expect(reg.paired).not.toContain('r2_val');
  expect(reg.worst).toEqual(['f1_score=0', 'f1_score_unique_variables=0', 'precision_score=0',
    'precision_unique_variables=0', 'recall_score=0', 'recall_unique_variables=0']);
  // the mean is chosen, the median is drawn, and the axis says so
  const chart = page.locator(V2 + ' .v2plot svg').first();
  await expect(chart.locator('circle').first()).toBeVisible({ timeout: 15000 });
  await expect(chart).toContainText(/R²(, Validation| Val), median/);
  // near 1 the linear bins of R² are too coarse, so the FVU's log bins are read there
  expect(asked).toContain('log10_fvu_val.js');
  expect(asked).toContain('r2_val.js');
  const titles = await chart.locator('circle title').allTextContents();
  const values = titles.map((t) => parseFloat(t.split(': ')[1].replace(/^[≤≥] /, '')));
  expect(values.length).toBeGreaterThan(3);
  for (const v of values) { expect(v).toBeLessThanOrEqual(1); }
  expect(values.some((v) => v > 0.99 && v < 1)).toBe(true);        // resolved next to 1 ...
  expect(new Set(values.filter((v) => v < 0.9)).size).toBeGreaterThan(3);   // ... and not in a handful of coarse steps below it
  // switching the statistic changes nothing for this metric, except that the axis no longer has to say it
  await page.locator(V2 + ' input[name="v2stat"][value="median"]').check();
  await expect(chart).not.toContainText(/, median/);
  expect(await chart.locator('circle title').allTextContents()).toEqual(titles);
  expect(errors, errors.join('\n')).toEqual([]);
});

test('a failed prediction counts the worst value, or is left out: the reader chooses', async ({ page }) => {
  const errors = collectErrors(page);
  await withHalfAnswered(page);
  await page.goto('/results.html?release=2026-09&m=fixture-half&x=rung&s=mean&band=0&p=rung~f1_score,rung~mdl_ratio&v=table&rows=rungs');
  const row = (r) => page.locator(V2 + ' .v2table tbody tr').filter({ has: page.locator('td:first-child', { hasText: new RegExp('^' + r + '$') }) });
  const f1 = (r) => row(r).locator('td').nth(2);
  // counted, the default: half of the laws at 0.8 and half at 0
  await expect(page.locator(V2 + ' .v2impute')).toBeChecked();
  await expect(f1(16)).toHaveText('0.400');
  await expect(f1(32)).toHaveText('0.760');
  await expect(row(16).locator('td').nth(3)).toHaveText(/1\.50/);          // a metric without a worst value: answers only, either way
  // left out: the answers that were made, and half of the laws is too few for a solid number
  await page.locator(V2 + ' .v2impute').uncheck();
  await expect(f1(16)).toHaveText(/^○ 0\.800$/);
  await expect(f1(32)).toHaveText('0.800');
  await expect(row(16).locator('td').nth(3)).toHaveText(/1\.50/);
  expect(page.url()).toContain('imp=0');
  // the same in a chart: the marker of the thin point turns hollow
  await page.locator(V2 + ' .v2tab[data-view="curves"]').click();
  const plot = page.locator(V2 + ' .v2plot').first();
  await expect(plot.locator('svg circle')).toHaveCount(2);
  await expect(plot.locator(HOLLOW)).toHaveCount(1);
  await page.locator(V2 + ' .v2impute').check();
  await expect(plot.locator(HOLLOW)).toHaveCount(0);
  // a link carries the choice, and a display without such a metric does not offer it
  await page.goto('/results.html?release=2026-09&m=fixture-half&x=rung&s=mean&band=0&p=rung~f1_score&v=table&rows=rungs&imp=0');
  await expect(page.locator(V2 + ' .v2impute')).not.toBeChecked();
  await expect(f1(16)).toHaveText(/0\.800/);
  await page.goto('/results.html?release=2026-09&m=fixture-half&x=rung&s=mean&p=rung~mdl_ratio&v=table&rows=rungs');
  await expect(page.locator(V2 + ' .v2impute')).toBeHidden();
  expect(errors, errors.join('\n')).toEqual([]);
});

test('leaving failed predictions out is exact for the median, the distribution and a paired contrast', async ({ page }) => {
  const errors = collectErrors(page);
  // the release's own numbers: a method that answers under half of the laws has a median overlap of 0 when its
  // failures count, and the median of its answers when they do not
  const low = await page.goto('/results.html?release=2026-09&v=table&rows=rungs&x=rung&s=median&band=0&p=rung~f1_score').then(() => page.evaluate(() => {
    const D = window.RESULTS_V2; let best = null;
    D.methods.forEach((m) => { const per = D.cells[m.key] || {}; let n = 0, ok = 0; Object.keys(per).forEach((c) => { const x = per[c]['1']; if (x) { n += x.n; ok += x.ok; } }); if (n && ok / n < 0.45 && (!best || ok / n < best.share)) { best = { key: m.key, share: ok / n }; } });
    return best; }));
  test.skip(!low, 'no method of this release has a prediction for under half of the problems');
  const url = (extra) => '/results.html?release=2026-09&rows=rungs&x=rung&s=median&band=0&p=rung~f1_score&m=' + low.key + extra;
  const first = () => page.locator(V2 + ' .v2table tbody tr').first().locator('td').nth(2);
  await page.goto(url('&v=table&imp=1'));
  await expect(first()).toHaveText(/\d/, { timeout: 15000 });
  expect(parseFloat((await first().textContent()).replace(/^[≤≥○ ]+/, ''))).toBeLessThan(0.05);
  await page.locator(V2 + ' .v2impute').uncheck();
  await expect(first()).toHaveText(/○/);
  expect(parseFloat((await first().textContent()).replace(/^[≤≥○ ]+/, ''))).toBeGreaterThan(0.3);
  // the distribution loses exactly the laws that were filled in
  await page.goto(url('&v=dist&dm=f1_score&dv=hist&r=1&imp=1'));   // the choice is remembered, so the link states it
  await expect(page.locator(V2 + ' .v2view')).toContainText('All problems: a problem without a usable formula counts as 0', { timeout: 15000 });
  await page.locator(V2 + ' .v2impute').uncheck();
  await expect(page.locator(V2 + ' .v2view')).toContainText("Only problems where the method's formula has a value are counted");
  // a paired contrast is then taken over the laws both methods answered
  const other = await page.evaluate((k) => window.RESULTS_V2.methods.filter((m) => m.key !== k && window.RESULTS_V2.cells[m.key] && Object.keys(window.RESULTS_V2.cells[m.key]).length)[0].key, low.key);
  await page.goto('/results.html?release=2026-09&v=paired&x=rung&p=rung~f1_score&m=' + low.key + ',' + other + '&b=' + other + '&imp=1');
  await expect(page.locator(V2 + ' .v2view')).toContainText(/\d/, { timeout: 15000 });
  const counted = await page.locator(V2 + ' .v2view').innerText();
  await page.locator(V2 + ' .v2impute').uncheck();
  await expect(page.locator(V2 + ' .v2view')).not.toHaveText(counted);
  expect(errors, errors.join('\n')).toEqual([]);
});

test('symbolic recovery is asked at three levels of masking, each implying the one before it', async ({ page }) => {
  const errors = collectErrors(page);
  await page.goto('/results.html?release=2026-09&v=table&rows=rungs&x=rung&p=rung~symbolic_recovery,rung~symbolic_recovery_mask_fittable,rung~symbolic_recovery_mask_none');
  const seen = await page.evaluate(() => {
    const D = window.RESULTS_V2; let cells = 0, missing = 0, broken = 0, strict = 0;
    Object.keys(D.cells).forEach((m) => Object.keys(D.cells[m]).forEach((c) => Object.keys(D.cells[m][c]).forEach((r) => {
      const x = D.cells[m][c][r].m, all = x.symbolic_recovery, exp = x.symbolic_recovery_mask_fittable, num = x.symbolic_recovery_mask_none;
      cells += 1;
      if (!all || !exp || !num) { missing += 1; return; }
      if (!(num[0] <= exp[0] && exp[0] <= all[0] && num[0] <= x.numeric_recovery_val[0] && exp[1] === all[1] && num[1] === all[1])) { broken += 1; }
      if (exp[0] < all[0]) { strict += 1; }
    })));
    const labels = D.metrics.filter((m) => m.key.indexOf('symbolic_recovery') === 0).map((m) => m.label);
    return { cells, missing, broken, strict, labels };
  });
  expect(seen.cells).toBeGreaterThan(100);
  expect(seen.missing).toBe(0);
  expect(seen.broken).toBe(0);
  expect(seen.strict).toBeGreaterThan(0);   // the stricter level is a level of its own: somewhere an exponent is not the law's
  expect(seen.labels).toEqual(['Symbolic Recovery: Structure', 'Symbolic Recovery: Structure + Exponents', 'Symbolic Recovery: Structure + All Numbers']);
  // all three are rates over every law, so none of their numbers is marked as resting on too few
  const row = page.locator(V2 + ' .v2table tbody tr').first();
  await expect(row).toBeVisible({ timeout: 15000 });
  await expect(page.locator(V2 + ' .v2table .v2hollow')).toHaveCount(0);
  expect(errors, errors.join('\n')).toEqual([]);
});

test('metric names say Prediction and Ground Truth, in title caps, and single-expression properties sit apart', async ({ page }) => {
  await page.goto('/results.html?release=2026-09');
  const reg = await page.evaluate(() => window.RESULTS_V2.metrics.map((m) => ({ key: m.key, label: m.label, short: m.short, group: m.group })));
  const small = ['of', 'the', 'to', 'as', 'a', 'and', 'log10', 'log2', 'vNRR', 'fNRR'];
  for (const m of reg) {
    for (const text of [m.label, m.group, m.short.replace('GT', 'Ground Truth')]) {
      const lowered = text.split(/[^A-Za-z0-9²]+/).filter((w) => w && /^[a-z]/.test(w) && small.indexOf(w) < 0);
      expect(lowered, text).toEqual([]);
      expect(text, text).not.toMatch(/\b(law|answer|pred|GT)\b/i);
    }
  }
  const by = (k) => reg.filter((m) => m.key === k)[0];
  expect(by('success').label).toBe('Successful Prediction Rate');
  expect(by('mdl_ratio').label).toBe('MDL Ratio');
  expect(by('edit_distance').label).toContain('Levenshtein');
  expect(by('edit_distance_norm').label).toContain('Levenshtein');
  expect(['symbolic_recovery', 'symbolic_recovery_mask_fittable', 'symbolic_recovery_mask_none', 'skeleton_match_raw'].map((k) => by(k).short)).toEqual(['SRRs', 'SRRe', 'SRRa', 'SRRr']);
  // a property of one expression is not a comparison: those have a group of their own
  const alone = reg.filter((m) => m.group === 'Expression Properties').map((m) => m.key).sort();
  expect(alone).toEqual(['ground_truth_mdl', 'n_constants', 'n_variables', 'predicted_mdl', 'predicted_n_constants', 'predicted_skeleton_prefix_length',
    'predicted_total_nestedness', 'skeleton_length', 'total_nestedness']);
  expect(reg.filter((m) => /Ground Truth$/.test(m.group)).every((m) => alone.indexOf(m.key) < 0)).toBe(true);
});

// A tablet is wide enough for the desktop layout and has no keyboard until a text field takes the focus. The
// keyboard changes the height of the window, never its width: the picker must survive that, and must not summon it.
test.describe('the picker on a touch tablet', () => {
  test.use({ viewport: { width: 820, height: 1180 }, hasTouch: true, isMobile: true });
  test('opening it does not put the cursor into the search field', async ({ page }) => {
    await page.goto('/results.html?release=2026-09&v=curves&p=time~numeric_recovery_val');
    expect(await page.evaluate(() => window.matchMedia('(pointer: coarse)').matches)).toBe(true);   // the premise of this test
    await page.locator(`${V2} .v2plothead .v2pick[data-axis="y"]`).first().tap();
    await expect(page.locator('.v2picker')).toBeVisible();
    expect(await page.evaluate(() => document.activeElement && document.activeElement.className)).not.toContain('v2pickq');
  });
  test('it stays open when the on-screen keyboard takes height away, and closes when the width changes', async ({ page }) => {
    const errors = collectErrors(page);
    await page.goto('/results.html?release=2026-09&v=curves&p=time~numeric_recovery_val');
    await page.locator(`${V2} .v2plothead .v2pick[data-axis="y"]`).first().tap();
    const picker = page.locator('.v2picker');
    await expect(picker).toBeVisible();
    await picker.locator('.v2pickq').tap();                      // the reader asks for the keyboard
    await page.setViewportSize({ width: 820, height: 640 });      // ... and it takes the lower half of the window
    await page.waitForTimeout(400);
    await expect(picker).toBeVisible();
    const box = await picker.boundingBox();
    expect(box.y).toBeGreaterThanOrEqual(0);
    await picker.locator('.v2pickq').fill('recovery');
    await expect(picker.locator('.v2pickitem:visible').first()).toBeVisible();
    await picker.locator('.v2pickitem[data-k="symbolic_recovery"]').tap();
    await expect(picker).toHaveCount(0);
    await expect(page.locator(`${V2} .v2plothead .v2pick[data-axis="y"]`).first()).toContainText('Symbolic Recovery');
    await page.locator(`${V2} .v2plothead .v2pick[data-axis="y"]`).first().tap();
    await expect(picker).toBeVisible();
    await page.setViewportSize({ width: 1180, height: 820 });     // the tablet is turned: the layout changes under the menu
    await expect(picker).toHaveCount(0);
    expect(errors).toEqual([]);
  });
});

test('a redraw under an open picker does not close it', async ({ page }) => {
  const errors = collectErrors(page);
  await page.setViewportSize({ width: 1400, height: 900 });
  // a median metric needs its histogram, which arrives late here: the redraw lands while the menu is open
  await page.route('**/data/2026-09/hist/**', async (route) => { await new Promise((r) => setTimeout(r, 1500)); await route.continue(); });
  const arrived = page.waitForResponse((r) => r.url().includes('/hist/') && r.status() === 200);
  await page.goto('/results.html?release=2026-09&v=curves&p=time~r2_val', { waitUntil: 'domcontentloaded' });   // `load` would wait for the histogram
  const trigger = page.locator(`${V2} .v2plothead .v2pick[data-axis="y"]`).first();
  await trigger.click();
  const picker = page.locator('.v2picker');
  await expect(picker).toBeVisible();
  await page.evaluate(() => { window.__pickBefore = document.querySelector('#results-explorer-v2 .v2plothead .v2pick[data-axis="y"]'); });
  await arrived;
  await expect.poll(() => page.evaluate(() => document.body.contains(window.__pickBefore))).toBe(false);   // the premise: the redraw replaced the button
  await expect(picker).toBeVisible();
  await expect(page.locator(`${V2} .v2plothead .v2pick[data-axis="y"]`).first()).toHaveAttribute('aria-expanded', 'true');
  await picker.locator('.v2pickitem[data-k="symbolic_recovery"]').click();
  await expect(picker).toHaveCount(0);
  await expect(page.locator(`${V2} .v2plothead .v2pick[data-axis="y"]`).first()).toContainText('Symbolic Recovery');
  expect(errors).toEqual([]);
});

// A plot of two metrics can trade its axes with one control, where "vs" stands on a budget plot.
test('the axis swap trades the two metrics of a plot; a budget plot keeps its plain vs', async ({ page }) => {
  const errors = collectErrors(page);
  await page.setViewportSize({ width: 1400, height: 900 });
  await page.goto('/results.html?release=2026-09&v=curves&p=log10_fvu_val~numeric_recovery_val,time~numeric_recovery_val');
  const heads = page.locator(`${V2} .v2plothead`);
  await expect(heads).toHaveCount(2);
  const tradeoff = heads.nth(0), budget = heads.nth(1);
  await expect(tradeoff.locator('.v2swap')).toHaveCount(1);
  await expect(tradeoff.locator('.v2swap')).toHaveText('⇆');
  await expect(tradeoff.locator('.v2vs')).toHaveCount(0);
  await expect(budget.locator('.v2swap')).toHaveCount(0);
  await expect(budget.locator('.v2vs')).toHaveText('vs');
  const yBefore = await tradeoff.locator('.v2pick[data-axis="y"]').getAttribute('data-k');
  const xBefore = await tradeoff.locator('.v2pick[data-axis="x"]').getAttribute('data-k');
  expect([yBefore, xBefore]).toEqual(['numeric_recovery_val', 'log10_fvu_val']);
  await tradeoff.locator('.v2swap').click();
  const head = page.locator(`${V2} .v2plothead`).nth(0);
  await expect(head.locator('.v2pick[data-axis="y"]')).toHaveAttribute('data-k', 'log10_fvu_val');
  await expect(head.locator('.v2pick[data-axis="x"]')).toHaveAttribute('data-k', 'numeric_recovery_val');
  await expect(head.locator('.v2pick[data-axis="y"]')).toContainText('FVU');
  expect(page.url()).toContain('p=numeric_recovery_val%7Elog10_fvu_val%2Ctime%7Enumeric_recovery_val');
  await head.locator('.v2swap').click();                                                    // and back
  await expect(page.locator(`${V2} .v2plothead`).nth(0).locator('.v2pick[data-axis="y"]')).toHaveAttribute('data-k', 'numeric_recovery_val');
  // the control is the quiet grey of the label it replaces, outlined and rounded
  const style = await page.locator(`${V2} .v2swap`).first().evaluate((el) => { const c = getComputedStyle(el); return { color: c.color, border: c.borderTopWidth + ' ' + c.borderTopStyle, radius: c.borderTopLeftRadius }; });
  const faint = await page.locator(`${V2} .v2vs`).first().evaluate((el) => getComputedStyle(el).color);
  expect(style.color).toBe(faint);
  expect(style.border).toBe('1px solid');
  expect(parseFloat(style.radius)).toBeGreaterThan(0);
  expect(errors).toEqual([]);
});

test('the release has one home: a title and its update time in the headline roles, then one line of progress', async ({ page }) => {
  const errors = collectErrors(page);
  // one role each: the title and its caption compute exactly like the headline's on the home page
  const style = (sel) => page.locator(sel).first().evaluate((el) => { const c = getComputedStyle(el); return [c.fontSize, c.fontWeight, c.color, c.textTransform, c.letterSpacing].join(' '); });
  await page.goto('/');
  await expect(page.locator('.headline-v2 .v2hltitle')).toBeVisible();
  const [hlTitle, hlSub] = [await style('.headline-v2 .v2hltitle'), await style('.headline-v2 .v2hlsub')];
  await page.goto('/results.html?release=2026-09&v=curves');
  const head = page.locator(V2 + ' .v2relhead');
  await expect(head).toHaveCount(1);
  await expect(head.locator('h2.v2hltitle')).toHaveText(/^Release 2026-09/);
  expect(await style(V2 + ' .v2relhead .v2hltitle')).toBe(hlTitle);
  expect(await style(V2 + ' .v2relhead .v2hlsub')).toBe(hlSub);
  // the update time: the caption of the title, with its offset in the markup and a time zone on screen
  const rel = await page.evaluate(() => window.RESULTS_V2.release);
  const upd = head.locator('.v2hlsub .v2updated');
  await expect(upd).toHaveCount(1);
  await expect(upd).toHaveText(/^Updated /);
  if (rel.updated) {
    expect(rel.updated).toMatch(/^\d{4}-\d\d-\d\dT\d\d:\d\d[+-]\d\d:\d\d$/);
    await expect(head.locator('time.v2updated')).toHaveAttribute('datetime', rel.updated);
    await expect(upd).toHaveText(/\b20\d\d\b.* · (just now|\d+ (minute|minutes|hour|hours|days) ago)$/);
  }
  await expect(page.locator(V2 + ' .v2updated')).toHaveCount(1);                  // one place for the time
  await expect(page.locator(V2)).not.toContainText(/(?<!machine-)generated|benchmark release/i); // the release head above names the release
  // then one line of progress: the counts the exporter decided, and the way to the Progress page
  const line = page.locator(V2 + ' .v2progline');
  const summary = await page.evaluate(() => window.RESULTS_V2.summary);
  if (summary) {
    await expect(line).toHaveCount(1);
    await expect(line.locator('.v2kicker')).toHaveText('Progress');
    for (const [n, word] of [[summary.finished.length, 'finished'], [summary.in_progress.length, 'in progress'], [summary.scheduled.length, 'scheduled']]) {
      if (n) { await expect(line).toContainText(n + ' ' + word); } else { await expect(line).not.toContainText(word); }
    }
    await expect(line.locator('a')).toHaveAttribute('href', 'progress.html');
  } else {
    await expect(line).toHaveCount(0);                                              // data without a summary: no line
  }
  await expect(page.locator(V2 + ' .v2tile')).toHaveCount(0);                       // the details live on the Progress page
  await expect(page.locator(V2 + ' details.v2release')).toHaveCount(0);             // the protocol lives on the guide
  expect(errors).toEqual([]);
});

// Every text the 2026-09 page shows or offers (hints, popovers, tooltips, labels, the protocol), in every view and
// mode, uses the reader's words, not the pipeline's. The same list as copy_lint.py's BANNED_V2, which checks the prose.
const PIPELINE_WORDS = [/\bcatalogs?\b/i, /\brungs?\b/i, /\bdraws? (\d|per\b|of\b)|\b(\d+|two|its|their|one|per|of|over|the) draws?\b/i,
  /\bpooled\b|\bpooling\b/i, /reference[- ]machine/i, /\bladder\b/i, /\bcanon\b|canonical form/i, /\bstrat(um|a)\b|stratified/i,
  /\bmu\b/i];
test('every text of the 2026-09 explorer uses the reader\'s words', async ({ page }) => {
  test.setTimeout(120_000);
  const urls = ['v=curves&x=time', 'v=curves&x=rung', 'v=table&rows=rungs', 'v=table&rows=cats', 'v=matrix',
    'v=dist&dm=log10_fvu_val&dv=hist', 'v=dist&dm=log10_fvu_val&dv=ecdf', 'v=dist&dm=log10_fvu_val&dv=cats',
    'v=dist&dm=log10_fvu_val&dv=rungs', 'v=dist&dm=numeric_recovery_val&dv=cats', 'v=ranks&x=rung&r=16', 'v=ranks&x=time', 'v=paired', 'v=preds'];
  const found = [];
  for (const u of [...urls, 'home']) {   // every display of the explorer, then the home page's headline
    if (u === 'home') {
      await page.goto('/');
      await expect(page.locator('#results-headline-v2 svg.v2chart').first()).toBeVisible();
    } else {
      await page.goto('/results.html?release=2026-09&' + u);
      await expect(page.locator(V2 + ' .v2view')).toBeVisible();
      await expect(page.locator(V2 + ' .v2view')).not.toContainText('loading', { timeout: 15000 });
    }
    const texts = await page.evaluate(() => {
      const roots = ['#results-headline-v2', '#results-explorer-v2'].map((s) => document.querySelector(s)).filter(Boolean);
      const out = roots.map((r) => r.textContent);
      roots.forEach((r) => r.querySelectorAll('[data-help], [title], [aria-label]').forEach((e) =>
        ['data-help', 'title', 'aria-label'].forEach((a) => { if (e.getAttribute(a)) { out.push(e.getAttribute(a)); } })));
      return out;
    });
    for (const t of texts) {
      for (const w of PIPELINE_WORDS) {
        const m = t.match(w);
        if (m) { found.push(`${u}: ${w} … ${t.slice(Math.max(0, m.index - 50), m.index + 50).replace(/\s+/g, ' ')} …`); }
      }
    }
  }
  expect([...new Set(found)]).toEqual([]);
});

// ---- Predictions: the formulas themselves ---------------------------------------------------------------------------
test('the predictions view shows one problem: its true formula, and one row per method with its formula', async ({ page }) => {
  const errors = collectErrors(page);
  const wrap = (key, obj) => `window.RESULTS_V2_PRED=window.RESULTS_V2_PRED||{};(function(){var R=window.RESULTS_V2_PRED;R["2026-09"]=R["2026-09"]||{};R["2026-09"][${JSON.stringify(key)}]=${JSON.stringify(obj)};})();`;
  const truth = {}, preds = {};
  for (let i = 0; i < 100; i++) { truth[String(i)] = '* x1 x2'; preds[String(i)] = ['+ x1 1.5', 0]; }
  preds['0'] = ['* x1 x2', 3];           // recovered, numerically and in structure
  preds['1'] = null;                     // no usable formula
  preds['2'] = ['sqrt x1 x2', 0];        // a token outside the vocabulary: shown as written, never an error
  await page.route('**/data/2026-09/results.js', async (route) => {
    const res = await route.fetch();
    await route.fulfill({ response: res, body: (await res.text()) + ';(function(){var D=window.RESULTS_V2;D.pred={"T8-20M":{"feynman|16":[1]}};D.pred_block=500;})();' });
  });
  await page.route('**/pred/truth/feynman.0.js', (route) => route.fulfill({ contentType: 'text/javascript', body: wrap('truth|feynman|0', truth) }));
  await page.route('**/pred/T8-20M/feynman/16.1.0.js', (route) => route.fulfill({ contentType: 'text/javascript', body: wrap('T8-20M|feynman|16|1|0', preds) }));
  await page.goto('/results.html?release=2026-09&v=preds&ps=feynman&r=16&pr=1&pn=1&m=T8-20M,prior');
  const rows = page.locator(V2 + ' .v2predtable tbody tr');
  await expect(rows).toHaveCount(2, { timeout: 15000 });                                   // one row per shown method
  await expect(page.locator(V2 + ' .v2predtruth .katex')).toHaveCount(1);                  // the true formula, typeset
  await expect(rows.nth(0).locator('td.v2predf .katex')).toHaveCount(1);                   // the method's formula, typeset
  await expect(rows.nth(0).locator('.v2predmark')).toHaveText(['numeric', 'structure']);
  await expect(rows.nth(1)).toContainText(/not finished yet|not run at budget 16/);         // a method without this run says so
  await page.locator(V2 + ' .v2viewbar .v2stepbtn[aria-label="next problem"]').click();
  await expect(rows.nth(0).locator('td.v2predf')).toHaveText('no usable formula');
  expect(page.url()).toContain('pn=2');
  await page.locator(V2 + ' .v2predprob').fill('3');
  await page.locator(V2 + ' .v2predprob').press('Enter');
  await expect(rows.nth(0).locator('td.v2predf code')).toHaveText('sqrt x1 x2');
  await expect(page.locator(V2 + ' .v2predof')).toHaveText('of 100');
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(overflow).toBeLessThanOrEqual(0);
  expect(errors).toEqual([]);
});

test('without published formulas the predictions view says so', async ({ page }) => {
  await page.route('**/data/2026-09/results.js', async (route) => {
    const res = await route.fetch();
    await route.fulfill({ response: res, body: (await res.text()) + ';(function(){var D=window.RESULTS_V2;D.pred={};})();' });
  });
  await page.goto('/results.html?release=2026-09&v=preds');
  await expect(page.locator(V2 + ' .v2view')).toContainText('The formulas are not published in this release yet.');
});
