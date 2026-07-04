import { defineConfig } from '@playwright/test';

// Serves the static results-site and runs the functionality suite against it.
// Two projects: desktop (1400px) and mobile (390x844) — the site must work on both.
export default defineConfig({
  testDir: '.',
  timeout: 30_000,
  retries: 1,                       // static site; a retry absorbs rare CDN hiccups (Plotly)
  workers: 4,
  reporter: [['list']],
  webServer: {
    command: 'python3 -m http.server 8123 --directory ..',
    port: 8123,
    reuseExistingServer: true,
  },
  use: {
    baseURL: 'http://localhost:8123',
  },
  projects: [
    { name: 'desktop', use: { viewport: { width: 1400, height: 900 } } },
    { name: 'mobile', use: { viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: true } },
  ],
});
