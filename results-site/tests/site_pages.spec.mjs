// The pages of the site (owner 2026-09-26/27: a home page that says what srbf is, with the headline's charts; the
// explorer on a page of its own; everything else on a page of its own): one navigation on every page, each page marked
// in it, the old section links opening their pages, the Progress page and the guide's Protocol. The explorer itself is
// covered by site_v2.spec.mjs, the theme and the visual abstract by site.spec.mjs.
import { test, expect } from '@playwright/test';

function collectErrors(page) {
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  page.on('console', (msg) => { if (msg.type() === 'error') { errors.push(msg.text()); } });
  return errors;
}

// the header names the pages; Docs, GitHub and PyPI have one place, the footer of every page. The Ranks and Paired
// pages explain two of the explorer's displays and are reached from those displays (owner 2026-09-26), not the header.
// The home page is the brand's link; Explorer is the explorer's page.
const NAV = [['explorer.html', 'Explorer'], ['progress.html', 'Progress'], ['guide.html', 'How to read'], ['metrics.html', 'Metrics'],
  ['reproductions.html', 'Reproductions']];
const FOOTER = ['https://srbf.readthedocs.io/', 'https://github.com/psaegert/srbf', 'https://pypi.org/project/srbf/', 'privacy.html'];
const PAGES = [['/', null], ['/explorer.html', 'Explorer'], ['/progress.html', 'Progress'], ['/guide.html', 'How to read'], ['/metrics.html', 'Metrics'],
  ['/reproductions.html', 'Reproductions'], ['/ranks.html', null], ['/paired.html', null], ['/privacy.html', null]];

for (const [url, current] of PAGES) {
  test(`${url}: the one navigation, this page marked in it, one title, nothing wider than the screen`, async ({ page }) => {
    const errors = collectErrors(page);
    await page.goto(url);
    await page.waitForLoadState('networkidle');
    expect(await page.locator('.site-nav a').evaluateAll((as) => as.map((a) => [a.getAttribute('href'), a.textContent]))).toEqual(NAV);
    expect(await page.locator('.site-nav a[aria-current="page"]').allTextContents()).toEqual(current ? [current] : []);
    await expect(page.locator('main h1')).toHaveCount(1);
    for (const href of FOOTER) { await expect(page.locator(`.site-footer a[href="${href}"]`).first(), href).toBeVisible(); }
    expect(await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)).toBeLessThanOrEqual(0);
    await expect(page.locator('.katex-error')).toHaveCount(0);                   // every formula typesets
    expect(errors).toEqual([]);
  });
}

test('every page the navigation names is there', async ({ page, request }) => {
  for (const [href] of NAV) {
    expect((await request.get('/' + href)).status(), href).toBe(200);
  }
  await page.goto('/metrics.html');
  await expect(page.locator('.site-header a.brand')).toHaveAttribute('href', './');   // the brand leads home
  await page.goto('/metrics.html');
  await page.waitForLoadState('networkidle');
  expect(await page.locator('main .katex').count()).toBeGreaterThan(20);        // the definitions are typeset
});

test('the home page\'s example is real: its link opens the explorer on that problem, with that formula', async ({ page }) => {
  const errors = collectErrors(page);
  await page.goto('/');
  await expect(page.locator('.home-example .katex').first()).toBeVisible();             // the formulas are typeset
  await expect(page.locator('#results-updated-v2 time')).toHaveCount(1);                  // the release says when it was updated
  const href = await page.locator('.home-example figcaption a').getAttribute('href');
  await page.goto('/' + href);
  const view = page.locator('#results-explorer-v2 .v2view');
  await expect(page.locator('#results-explorer-v2 .v2tab.active')).toHaveAttribute('data-view', 'preds');
  await expect(view).toContainText('0.04775', { timeout: 20_000 });                         // PySR's fitted number, as on the card
  expect(errors).toEqual([]);
});

test('the explorer\'s former address opens it with the same settings', async ({ page }) => {
  await page.goto('/results.html?release=2026-09&v=ranks&x=rung&r=16');
  await expect(page).toHaveURL(/\/explorer\.html\?release=2026-09&v=ranks/);
  await expect(page.locator('#results-explorer-v2 .v2tab.active')).toHaveAttribute('data-view', 'ranks');
});

for (const [view, target] of [['ranks', 'ranks.html'], ['paired', 'paired.html']]) {
  test(`the ${view} display links to ${target}, the page that explains it`, async ({ page, request }) => {
    const errors = collectErrors(page);
    await page.goto('/explorer.html?release=2026-09&v=' + view);
    await expect(page.locator(`#results-explorer-v2 .v2main a[href="${target}"]`).first()).toBeVisible({ timeout: 20_000 });
    expect((await request.get('/' + target)).status()).toBe(200);
    expect(errors).toEqual([]);
  });
}

for (const [hash, target] of [['#about', 'guide.html'], ['#paired', 'paired.html'], ['#ranks', 'ranks.html'], ['#metrics', 'metrics.html']]) {
  test(`a link to the former section ${hash} opens ${target}`, async ({ page }) => {
    await page.goto('/' + hash);
    await expect(page).toHaveURL(new RegExp('/' + target.replace('.', '\\.') + '$'));
    await expect(page.locator('main h1')).toHaveCount(1);
  });
}

test('the Progress page: the finished names, a tile per method in progress, the scheduled names', async ({ page }) => {
  const errors = collectErrors(page);
  await page.goto('/progress.html');
  const S = await page.evaluate(() => window.RESULTS_V2_SUMMARY);
  expect(S).toBeTruthy();
  const root = page.locator('#progress-root');
  expect(await root.locator('.v2done .v2chip').evaluateAll((cs) => cs.map((c) => c.dataset.m))).toEqual(S.summary.finished);
  expect(await root.locator('.v2tile').evaluateAll((ts) => ts.map((t) => t.dataset.m))).toEqual(S.summary.in_progress);
  expect(await root.locator('.v2sched .v2chip').allTextContents()).toEqual(S.summary.scheduled.map((x) => x.label));
  expect(await root.locator('.v2schednotes dd').allTextContents()).toEqual(S.summary.scheduled.map((x) => x.note));
  await expect(page.locator('#progress-updated time')).toHaveCount(S.release.updated ? 1 : 0);
  // a tile: Results and Times, each a count and one block per budget of the method
  const shape = await page.evaluate(() => {
    const S = window.RESULTS_V2_SUMMARY, out = [];
    document.querySelectorAll('#progress-root .v2tile').forEach((t) => {
      const m = S.methods.find((x) => x.key === t.dataset.m);
      out.push({ key: m.key, budgets: (m.budgets || []).length, perKnown: Object.keys(S.progress[m.key] || {}).length > 0,
        timed: (m.budgets || []).filter((b) => (S.timing[m.key] || {})[String(b)] != null).length,
        rows: [...t.querySelectorAll('.v2segs')].map((r) => r.children.length), nums: [...t.querySelectorAll('.v2pnum')].map((n) => n.textContent),
        labels: [...t.querySelectorAll('.v2plab')].map((l) => l.textContent), status: S.status[m.key] });
    });
    return out;
  });
  for (const t of shape) {
    expect(t.labels, t.key).toEqual(['Results', 'Times']);
    expect(t.rows, t.key).toEqual([t.perKnown ? t.budgets : 1, t.budgets]);
    expect(t.nums[0], t.key).toBe(t.status[0] + ' / ' + (t.status[1] == null ? '?' : t.status[1]));
    expect(t.nums[1], t.key).toBe(t.timed + ' / ' + t.budgets);
  }
  expect(errors).toEqual([]);
});

test('a method with every result and every time in moves to the Finished line', async ({ page }) => {
  const errors = collectErrors(page);
  // fixture: the exporter's summary with the first method in progress declared finished
  await page.addInitScript(() => { let r; Object.defineProperty(window, 'RESULTS_V2_SUMMARY', { configurable: true, get() { return r; }, set(v) {
    const key = v.summary.in_progress[0];
    if (key) { window.__finished = key; v.summary.in_progress = v.summary.in_progress.slice(1); v.summary.finished = v.summary.finished.concat([key]); }
    r = v; } }); });
  await page.goto('/progress.html');
  const key = await page.evaluate(() => window.__finished);
  expect(key).toBeTruthy();
  const root = page.locator('#progress-root');
  await expect(root.locator('.v2done .v2lab')).toHaveText('Finished');
  await expect(root.locator(`.v2done .v2chip[data-m="${key}"]`)).toHaveCount(1);
  await expect(root.locator(`.v2tile[data-m="${key}"]`)).toHaveCount(0);        // no tile of its own any more
  expect(errors).toEqual([]);
});

test('a tile names its method in full: a long name takes a second line instead of being cut', async ({ page }) => {
  for (const width of [1400, 390]) {
    await page.setViewportSize({ width, height: 900 });
    await page.goto('/progress.html');
    const cut = await page.locator('#progress-root .v2tile b').evaluateAll((els) => els.map((el) => getComputedStyle(el).whiteSpace === 'nowrap' || el.scrollWidth > el.clientWidth + 1));
    expect(cut.length).toBeGreaterThan(0);
    expect(cut.filter(Boolean), String(width)).toEqual([]);
  }
});

test('the guide opens with the protocol, in the release\'s own words, and its links land on the page', async ({ page }) => {
  const errors = collectErrors(page);
  await page.goto('/guide.html');
  const S = await page.evaluate(() => window.RESULTS_V2_SUMMARY);
  await expect(page.locator('#protocol [data-fill="scoring"]')).toHaveText(S.release.scoring);
  await expect(page.locator('#protocol [data-fill="judge"]')).toHaveText(S.release.judge);
  const counts = await page.evaluate(() => { const S = window.RESULTS_V2_SUMMARY; return S.problem_sets + ' problem sets, ' + S.problems.toLocaleString() + ' problems.'; });
  await expect(page.locator('#protocol [data-fill="counts"]')).toHaveText(counts);
  await expect(page.locator('#protocol [data-fill="timing_note"]')).toHaveText(S.timing_note);
  for (const href of await page.locator('#protocol a[href^="#"]').evaluateAll((as) => as.map((a) => a.getAttribute('href')))) {
    await expect(page.locator(href + ' > h3'), href).toHaveCount(1);
  }
  expect(errors).toEqual([]);
});

// The Reproductions page (owner 2026-10-01): srbf's runs of each method next to the results its own publications
// report, one exact two-sided test per row; scripts/site_reproductions.py writes the rows from data/reproductions.yaml.
test('the Reproductions page: every compared method with its source, published value, srbf runs and p; the rest with a reason', async ({ page }) => {
  const errors = collectErrors(page);
  await page.goto('/reproductions.html');
  const compared = page.locator('#compared > section.topic');
  expect(await compared.locator('h3').allTextContents()).toEqual(['E2E', 'GP-GOMEA', 'NeSymReS', 'Operon', 'QLattice', 'RILS-ROLS']);
  expect(await page.locator('#not-compared > section.topic h3').allTextContents()).toEqual(['DSR', 'PySR', 'uDSR*']);
  for (const topic of await compared.all()) {
    await expect(topic.locator('.repro-src a[href^="https://"]')).toHaveCount(1);              // the publication, linked
    for (const row of await topic.locator('.repro-row').all()) {
      const cells = await row.locator('tbody tr').evaluateAll((trs) => trs.map((tr) => [tr.querySelector('th').textContent, tr.querySelector('td:last-child').textContent]));
      expect(cells[0][0]).toBe('Published');
      expect(cells.length).toBeGreaterThan(1);
      for (const [, p] of cells.slice(1)) { expect(p).toMatch(/^(< 0\.001|0\.\d+|1\.00)$/); }   // every srbf run has its p
      await expect(row.locator('.repro-reading')).toHaveText(/^(No significant difference\.|srbf's (runs succeed (less|more) often than the published ones|fits score (lower|higher) than the published fits)\.)/);
      expect(await row.locator('.repro-protocol dt').allTextContents()).toEqual(['Published', 'srbf']);
    }
  }
  // a difference has its explanation next to it
  for (const row of await page.locator('.repro-row').all()) {
    if ((await row.locator('.repro-reading').textContent()).startsWith("srbf's ")) { await expect(row.locator('.repro-note')).toHaveCount(1); }
  }
  await expect(page.locator('#how h2')).toHaveText('How to read the comparisons');
  expect(errors).toEqual([]);
});
