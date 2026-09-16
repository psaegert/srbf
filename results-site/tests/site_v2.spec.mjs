// The 2026-09 release explorer (explorer_v2.js) and the release switch. Same philosophy as site.spec.mjs:
// content renders, transitions land, no console errors, no horizontal overflow.
import { test, expect } from '@playwright/test';

function collectErrors(page) {
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  page.on('console', (msg) => { if (msg.type() === 'error') { errors.push(msg.text()); } });
  return errors;
}

test('the newest release is the default and renders its charts', async ({ page }) => {
  const errors = collectErrors(page);
  await page.goto('/');
  await expect(page.locator('#results-explorer-v2')).toBeVisible();
  await expect(page.locator('#results-explorer')).toBeHidden();
  await expect(page.locator('#results-explorer-v2 svg.v2chart').first()).toBeVisible();
  await expect(page.locator('#release-switch a[aria-current="true"]')).toHaveAttribute('data-release', '2026-09');
  expect(errors).toEqual([]);
});

test('the release switch reaches the 2026-07 explorer and back', async ({ page }) => {
  const errors = collectErrors(page);
  await page.goto('/');
  await page.locator('#release-switch a[data-release="2026-07"]').click();
  await expect(page.locator('#results-explorer')).toBeVisible();
  await expect(page.locator('#results-explorer-v2')).toBeHidden();
  await expect(page.locator('#results-plot .main-svg').first()).toBeVisible();
  await page.locator('#release-switch a[data-release="2026-09"]').click();
  await expect(page.locator('#results-explorer-v2 svg.v2chart').first()).toBeVisible();
  expect(errors).toEqual([]);
});

test('a 2026-07 deep link still opens the 2026-07 explorer', async ({ page }) => {
  await page.goto('/?view=curves&bench=FastSRB');
  await expect(page.locator('#results-explorer')).toBeVisible();
  await expect(page.locator('#results-explorer-v2')).toBeHidden();
});

test('catalog and method controls change the pooled charts', async ({ page }) => {
  const errors = collectErrors(page);
  await page.goto('/');
  const count = page.locator('#results-explorer-v2 .v2catcount');
  const before = await count.textContent();
  await page.locator('#results-explorer-v2 button[data-act="phys"]').click();
  await expect(count).not.toHaveText(before);
  await page.locator('#results-explorer-v2 button[data-act="none"]').click();
  await expect(page.locator('#results-explorer-v2 svg.v2chart').first()).toContainText('no catalog selected');
  await page.locator('#results-explorer-v2 button[data-act="reset"]').click();
  await expect(count).toHaveText(before);
  expect(errors).toEqual([]);
});

test('the time axis is offered only with reference-machine measurements', async ({ page }) => {
  await page.goto('/');
  const radio = page.locator('#results-explorer-v2 input.v2xtime');
  const hasTiming = await page.evaluate(() => Object.keys((window.RESULTS_V2 || {}).timing || {}).length > 0);
  if (hasTiming) { await expect(radio).toBeEnabled(); } else { await expect(radio).toBeDisabled(); }
});

test('no horizontal page overflow on the 2026-09 release', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('#results-explorer-v2 svg.v2chart').first()).toBeVisible();
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(overflow).toBeLessThanOrEqual(0);
});
