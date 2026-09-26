// The page around the explorer: the theme toggle (theme.js), the visual abstract, the prose, and links from the
// retired 2026-07 explorer. The explorer itself is covered by site_v2.spec.mjs.
import { test, expect } from '@playwright/test';

const V2 = '#results-explorer-v2';

function collectErrors(page) {
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  page.on('console', (msg) => { if (msg.type() === 'error') { errors.push(msg.text()); } });
  return errors;
}

// a page is open once its first chart is drawn: the headline's on the home page, the explorer's on the Results page
const CHART = '#results-headline-v2 svg.v2chart, #results-explorer-v2 svg.v2chart';
async function open(page, url = '/') {
  await page.goto(url);
  await expect(page.locator(CHART).first()).toBeVisible({ timeout: 20_000 });
}

// ---- theme: Auto follows the device; the toggle cycles Auto -> Dark -> Light -> Auto -----------------------------

const DARK_BG = 'rgb(12, 14, 20)';     // --bg #0c0e14
const LIGHT_BG = 'rgb(246, 247, 251)'; // --bg #f6f7fb
function bodyBg(page) {
  return page.evaluate(() => getComputedStyle(document.body).backgroundColor);
}

test('theme: Auto follows the device scheme and stores nothing', async ({ page }) => {
  const errors = collectErrors(page);
  await page.emulateMedia({ colorScheme: 'dark' });
  await open(page);
  expect(await bodyBg(page)).toBe(DARK_BG);
  expect(await page.evaluate(() => localStorage.getItem('srbf_theme'))).toBe(null);
  await page.emulateMedia({ colorScheme: 'light' });
  await expect.poll(() => bodyBg(page)).toBe(LIGHT_BG);
  expect(errors).toEqual([]);
});

test('theme: the toggle forces dark on a light device, persists, and cycles back to Auto', async ({ page }) => {
  const errors = collectErrors(page);
  await page.emulateMedia({ colorScheme: 'light' });
  await open(page);
  await expect(page.locator('#theme-toggle')).toHaveAttribute('aria-label', /Auto/);   // theme.js has wired it
  await page.click('#theme-toggle');                       // Auto -> Dark
  await expect.poll(() => bodyBg(page)).toBe(DARK_BG);
  expect(await page.evaluate(() => localStorage.getItem('srbf_theme'))).toBe('dark');
  await page.reload();                                     // the pre-paint script applies it
  await expect(page.locator(CHART).first()).toBeVisible({ timeout: 20_000 });
  expect(await bodyBg(page)).toBe(DARK_BG);
  await page.click('#theme-toggle');                       // Dark -> Light
  await expect.poll(() => bodyBg(page)).toBe(LIGHT_BG);
  await page.click('#theme-toggle');                       // Light -> Auto (entry removed)
  expect(await page.evaluate(() => localStorage.getItem('srbf_theme'))).toBe(null);
  expect(errors).toEqual([]);
});

test('theme: the visual abstract follows the manual override', async ({ page }) => {
  await page.emulateMedia({ colorScheme: 'light' });
  await open(page);
  await page.click('#theme-toggle');
  await expect.poll(() => page.locator('#va rect.bg')
    .evaluate((r) => getComputedStyle(r).fill)).toBe(DARK_BG);
});

// ---- prose ----------------------------------------------------------------------------------------------------------

test('provenance: the prose explains the three labels and discloses the shared authors', async ({ page }) => {
  await page.goto('/guide.html');
  const about = page.locator('#about');
  await expect(about).toContainText("Who chose each method's settings");
  await expect(about).toContainText('Upstream defaults');
  await expect(about).toContainText('Author-blessed');
  await expect(about).toContainText('Maintainer-chosen');
  await expect(about).toContainText("Here they are also the benchmark's authors");
  await page.goto('/');
  await expect(page.locator('.hero')).toContainText('The people who run this benchmark also develop Flash-ANSR');
});

// ---- the retired 2026-07 explorer -----------------------------------------------------------------------------------

for (const url of ['/?release=2026-07', '/?view=curves&bench=FastSRB', '/?view=ranks&metric=log10_fvu_val&budget=10']) {
  test(`a link of the retired 2026-07 explorer opens the current explorer: ${url}`, async ({ page }) => {
    const errors = collectErrors(page);
    await open(page, url);
    await expect(page).toHaveURL(/\/results\.html\?/);
    await expect(page.locator(V2 + ' svg.v2chart').first()).toBeVisible();
    for (const k of ['view', 'bench', 'baseline', 'metric', 'budget']) { expect(new URL(page.url()).searchParams.has(k), k).toBe(false); }
    expect(errors).toEqual([]);
  });
}
