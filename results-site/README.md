# Results site (github.io frontend)

This directory is the **standalone results frontend** deployed to GitHub Pages at
<https://psaegert.github.io/srbf/>. It is deliberately **not** part of the MkDocs documentation
(the docs live at <https://srbf.readthedocs.io/>): the results site is a polished, client-side
explorer for the benchmark numbers, nothing more.

## Files

| File | Purpose |
|------|---------|
| `index.html` | The Results page: the intro, the visual abstract, the headline charts, the release's title with one line of progress, and the explorer. A link to one of its former sections (`#about`, `#paired`, `#ranks`, `#metrics`) opens that section's page. |
| `progress.html`, `guide.html`, `metrics.html`, `ranks.html`, `paired.html`, `privacy.html` | The pages around it, one per topic, each with the same header navigation (Results, Progress, How to read, Metrics; the page itself marked; Ranks and Paired are reached from the explorer's Ranks and Paired displays, whose hints link to them): the progress of every method (finished, in progress budget by budget, scheduled), how to read the results (the protocol, then the topics), the metric definitions (typeset with KaTeX), ranks, paired comparisons, and what the site stores. Written for a first-time reader: every term is defined where it first appears, and the explorer's short hints link here for the longer explanations. |
| `pages.js` | The Progress page and the guide's protocol, from `data/<release>/summary.js`. |
| `styles.css` | Standalone styling (light/dark aware), including the prose/metric-card styles for the prose sections. |
| `theme.js` | The header's theme toggle (Auto / Dark / Light; one `srbf_theme` localStorage entry, set only on an explicit choice). |
| `explorer_v2.js` | The explorer for benchmark releases from 2026-09 on (srbf 0.20 / flash-ansr 0.18, candidates ranked by the two-part code). Reads `window.RESULTS_V2` (schema 2) from `data/<release>/results.js`: the full metric registry (the metric floor of the site's first release plus SRR, MDL ratio, R², relative recovery, variable-set P/R, Pareto rank, ground-truth descriptors) and per method × catalog × rung counts, sums and sums of squares, so any catalog subset is pooled on the client with 95 % bands (Wilson / t / order statistic). Six views: Curves (every plotted metric vs budget or reference-machine time), Table (every rung, or catalogs at one rung; TSV / CSV export), Catalogs (a shaded matrix of one metric), Distribution (one histogram per method, cumulative curves, a box per catalog, quartiles along the ladder; per-catalog rates for rate metrics), Ranks (mean rank within each law at one budget or within one reference-machine time limit, Friedman + Nemenyi critical difference, standings, head-to-head table, standings along the ladder), Paired Δ (each method against a baseline on the same laws: exact McNemar for rates, paired t + sign test otherwise). Predictions (the formulas themselves, typeset with KaTeX from their prefix form, page by page, next to the true formula). Histograms (`data/<release>/hist/<metric>.js`), paired contrasts (`data/<release>/paired.js`) and the pairwise rank outcomes (`data/<release>/ranks.js`: per pair × catalog × slot `[n laws, wins of the first, wins of the second, ...]` per rank metric, a slot being a rung or a time budget; mean ranks for any roster and any catalog subset follow from them on the client) load on demand. Controls that only one view can use (the budget a snapshot is taken at, how a distribution is drawn) sit on that view. State in the URL (shareable) and localStorage; colours in one `srbf_colors` cookie, written only on an explicit change; one-sentence popovers on every metric and term. No library, plain SVG. Each view declares the controls it uses (`USES`), and the sidebar hides the rest, so a control on screen can always change what is on screen. Charts are built for the width of the container they land in (`chartWidth` + a `ResizeObserver`), so one SVG unit is one CSS pixel and a 12 px label is 12 px at any window size. |
| `data/<release>/results.js`, `hist/`, `paired.js`, `ranks.js` | One release, written by `scripts/site_export_v2.py` (see "Releases"). |
| `data/<release>/summary.js` | The few kB the Progress page and the guide read (the protocol's texts, every method's progress and times per budget, and which methods are finished, in progress and scheduled), written next to `results.js` by the same exporter. |
| `data/<release>/pred/` | The Predictions view's files: every method's formula for every problem, as the judge read it (rounded to 4 significant digits), one file per method × problem set × budget × finished run × block of 500 problems (`pred/<method>/<set>/<budget>.<run>.<block>.js`), and the ground truth's (`pred/truth/<set>.<block>.js`). A file is written once its run is complete and never changes after. They come from the `predicted_expression` / `ground_truth_expression` columns of `srbf table`. |
| `srbf-icon.svg`, `apple-touch-icon.png`, `srbf-social.png` | Brand assets (favicon, touch icon, `og:image` social card); canonical sources live in `../assets/brand/`. |

## Releases

The page shows one release, **2026-09**: 29 catalogs, the Flash-ANSR T8 series under the two-part code, the
Flash-ANSR + PySR hybrid, PySR, NeSymReS, E2E at its default settings and the Flash-ANSR prior reference, with fit
times from the one timing workstation once measured. `explorer_v2.js` reads it from `data/2026-09/results.js`.

The paper-era release 2026-07 (`explorer.js`, `results_data.js`, `paired_data.js`) was retired on 2026-09-26; it
remains in the git history. Its links (`?release=2026-07`, `?view=`, `?bench=`, `?baseline=`, `?metric=`,
`?budget=`) open the current page, and the explorer drops their keys from the URL.

The 2026-09 release is rebuilt from the result files with srbf alone, in three steps:

```bash
# 1. judge every result file: one row per problem and rung, every metric (a --tree per method and draw)
srbf table --tree T8-20M:1:<draw 1>/flash-ansr-v25.0-T8-20M --tree T8-20M:2:<draw 2>/flash-ansr-v25.0-T8-20M ... \
           --index-variables e2e=0 --index-variables nesymres-100M=1 -o <root>/rows_full_all.csv
# 2. the time axis, from the reference machine's timing result files
python scripts/site_timing.py --manifest configs/timing/timing_subset.json \
    --subset T8-20M=<timing results>/t8-20m ... --suite PySR=<PySR results> --out <root>/timing.json
# 3. the release
python scripts/site_export_v2.py <root> 2026-09 results-site/data/2026-09/results.js \
    --public e2e,nesymres-100M,PySR,T8-3M,T8-20M,T8-120M,T8-20M-pysr,prior
```

The method names in `--tree`, `--subset` and `--suite` are the site's method keys (`METHODS` in the exporter).
`--pattern KEY=PATTERN` names one method's rung files when they are not `choices_<rung>.pkl`: a ladder that counts
iterations keeps its `niter_{rung:05d}.pkl` files on the timing subset too.
`--index-variables` is for the E2E and NeSymReS result files written before srbf renamed their variables at the
source. The catalog table reads `data/catalog_mu.json` (`scripts/catalog_mu.py`). A cell pools the draws that are
complete for it. The progress shown per method reads optional unit lists and completion marks in the root
(`units_<key>_d<draw>.txt`, `markers/<key>.txt`); without them it counts the finished cells.

The exporter writes `results.js` (registry + cells), `hist/<metric>.js` and `paired.js` next to it.

A new release is a new `data/<id>/results.js`, and the script tag in `index.html` that loads it.

## Local-only methods

Some methods are evaluated but **not published** (they are not ours to release). The site keeps
them out of every public channel by construction, not by convention:

- `scripts/site_export_v2.py --private <keys> --private-dir results-site/private/<release>`
  writes those methods to a separate directory (payload, histograms, paired contrasts, including contrasts
  against public methods). The public payload, `index.html`, `explorer_v2.js` and the
  repository carry no reference to them; the exporter refuses a key that is in both lists.
- `results-site/private/` and `results-site/index.local.html` are git-ignored. `build_local.sh`
  writes `index.local.html` (the public page plus one script tag that loads the private file) and
  refuses to run if either path is not ignored. Serve it locally:
  `cd results-site && python3 -m http.server 8765` then open `/index.local.html`. The explorer marks
  such methods "local only".
- `tests/public_guard.py` fails the deploy if `index.html` mentions `private/` or `index.local`, if
  any method key in `data/*/results.js`, `data/*/hist/*.js` or `data/*/paired.js` is outside the public
  allowlist in that file, if a release payload lacks any metric of the floor, or if the private
  directory or the local page exists in the CI checkout. It runs before the Playwright suite.
- Screenshots, artifacts, PR descriptions and issues are made from the public page only.

## Sharing an overlay with the people it belongs to

An overlay can also travel with the site instead of staying on one machine, for the collaborators whose
results they are. `tools/seal.mjs` concatenates the overlay's payload scripts, gzips them and encrypts the
result with AES-256-GCM under a key derived by PBKDF2-HMAC-SHA256 (600k iterations, fresh salt and IV per
run); the page fetches `data/<release>/sealed.js` only when someone enters a key under the method list, runs
the payload exactly as a `<script>` would, and from then on treats its methods like any other. A key that
does not fit is indistinguishable from a release that ships no such file.

    SRBF_SEAL_KEY="$(cat ~/.config/srbf/seal-<release>.key)" node tools/seal.mjs <release>

Three things this rests on, none of them optional:

- **The key is generated, never chosen.** The sealed file is public, so guessing is offline and unlimited;
  the KDF is the only brake. Keep the key outside the repository, out of commit messages and issues, and
  hand it over through a channel that is not this site.
- **Publishing a sealed file cannot be undone.** It enters git history, forks and archives. If the key
  leaks later, everything inside it is exposed retroactively. Seal only what may live in public in that
  form, and only with the agreement of whoever owns the results.
- **The guard checks it is sealed** (`tests/public_guard.py`): envelope shape, KDF strength, ciphertext
  entropy and no plaintext left inside. The checker is run against a deliberately unsealed blob on every
  invocation, so it cannot pass vacuously.

`tests/fixtures/` holds a stand-in overlay and its sealed form, so the Playwright suite exercises the whole
path — wrong key, right key, method merged — without any real payload.

## Testing

`tests/` holds the functionality suite (Playwright, desktop + mobile projects) and a copy lint
for viewer-facing wording. Both gate the Pages deploy in CI. Locally:

```bash
cd results-site/tests
npm install && npx playwright install chromium
python3 copy_lint.py && python3 public_guard.py && npx playwright test --config playwright.config.mjs
```

`site_v2.spec.mjs` covers the explorer: the metric registry floor, every view from a deep link (no errors, no
overflow at 390 px), on-demand histograms and paired contrasts, problem-set and method controls, URL round trips,
popovers, the two fixed headline charts (present and unmoved by the explorer's controls), the time and candidate
axes, the reader's vocabulary in every rendered text, and the absence of any private overlay. `site.spec.mjs`
covers the page around it: the theme toggle, the visual abstract, the prose, and links of the retired 2026-07
explorer. `copy_lint.py` bans fixed wording bugs, and our pipeline's vocabulary in the prose and the data texts. When a
wording bug is fixed, add its pattern to `tests/copy_lint.py` so it cannot return.

## Deploying

Deploys are automatic: pushing to `main` with changes under `results-site/` triggers
`.github/workflows/pages.yaml`, which uploads this directory as the Pages artifact and deploys
it (Pages `build_type` is `workflow`). Manual redeploy: `gh workflow run pages.yaml`.

The `gh-pages` branch is not consumed. Always verify the SERVED site after deploying, at desktop
AND mobile widths.
