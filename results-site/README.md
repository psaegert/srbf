# Results site (github.io frontend)

This directory is the **standalone results frontend** deployed to GitHub Pages at
<https://psaegert.github.io/srbf/>. It is deliberately **not** part of the MkDocs documentation
(the docs live at <https://srbf.readthedocs.io/>): the results site is a polished, client-side
explorer for the benchmark numbers, nothing more.

## Files

| File | Purpose |
|------|---------|
| `index.html` | The home page: what srbf is, a real example from the release (Feynman II.8.7, PySR's answer, both checks; its link opens that problem in the explorer), the visual abstract, six points on the framework, the current release with its line of progress and the headline's two fixed charts (drawn by `explorer_v2.js` in its headline-only mode), how to run it, and how to cite it. A link to one of its former sections (`#about`, `#paired`, `#ranks`, `#metrics`) opens that section's page; a link with explorer settings in its query opens `explorer.html` with the same settings. |
| `explorer.html` | The explorer, and nothing else (the release's title with one line of progress, then the displays). `results.html`, its address for a few hours on 2026-09-26/27, forwards to it with the query. |
| `progress.html`, `guide.html`, `metrics.html`, `ranks.html`, `paired.html`, `privacy.html` | The pages around it, one per topic, each with the same header navigation (Explorer, Progress, How to read, Metrics, the brand leading home; the page itself marked; Ranks and Paired are reached from the explorer's Ranks and Paired displays, whose hints link to them): the progress of every method (finished, in progress budget by budget, scheduled), how to read the results (the protocol, then the topics), the metric definitions (typeset with KaTeX), ranks, paired comparisons, and what the site stores. Written for a first-time reader: every term is defined where it first appears, and the explorer's short hints link here for the longer explanations. |
| `pages.js` | The Progress page and the guide's protocol, from `data/<release>/summary.js`. |
| `styles.css` | Standalone styling (light/dark aware), including the prose/metric-card styles for the prose sections. |
| `theme.js` | The header's theme toggle (Auto / Dark / Light; one `srbf_theme` localStorage entry, set only on an explicit choice). |
| `explorer_v2.js` | The explorer for benchmark releases from 2026-09 on (srbf 0.20 / flash-ansr 0.18, candidates ranked by the two-part code). Reads `window.RESULTS_V2` (schema 2) from `data/<release>/results.js`: the full metric registry (the metric floor of the site's first release plus SRR, MDL ratio, R², relative recovery, variable-set P/R, Pareto rank, ground-truth descriptors) and per method × catalog × rung the problems' counts, sums and sums of squares (a problem's value is the mean of its runs), so any catalog subset is averaged on the client over its catalogs by random effects (Paule–Mandel between-catalog variance, 95 % intervals over the selected catalogs (the weights held fixed), rates on the logit scale, ratios as geometric means; `scripts/site_random_effects.py` is the reference and the site suite checks the page against it). Six views: Curves (every plotted metric vs budget or reference-machine time), Table (every rung, or catalogs at one rung; TSV / CSV export), Catalogs (a shaded matrix of one metric), Distribution (one histogram per method, cumulative curves, a box per catalog, quartiles along the ladder; per-catalog rates for rate metrics), Ranks (average places from the pairwise chances to beat, averaged over catalogs; pairwise tests Holm-corrected, standings, head-to-head table, standings along the budgets or times), Paired Δ (each method against a baseline problem by problem: the mean difference over catalogs, tested on the difference for rates and on the per-problem superiority otherwise), Correlations (two metrics problem by problem). Predictions (the formulas themselves, typeset with KaTeX from their prefix form, page by page, next to the true formula). Every view but Curves reads every method at one position, a time per problem (the default) or a budget, anywhere on a slider or typed into its box; between two budgets a method ran, its numbers are interpolated in the logarithm of the position, and the arrows step to the release's time limits or to the budgets the shown methods ran. Ranks and Paired Δ compute their snapshot at the position in the browser (`pairstats.js` from `pv/`); their charts along the budgets and times read the exporter's cells at every slot (`ranks/<metric>.js`, `paired/<metric>.js`; `D.slots` places every method at every slot by the same rule). Histograms (`data/<release>/hist/<metric>.js`) and these files load on demand, only for the metrics on screen. Controls that only one view can use (the position, how a distribution is drawn) sit on that view. State in the URL (shareable) and localStorage; colours in one `srbf_colors` cookie, written only on an explicit change; one-sentence popovers on every metric and term. No library, plain SVG. Each view declares the controls it uses (`USES`), and the sidebar hides the rest, so a control on screen can always change what is on screen. Charts are built for the width of the container they land in (`chartWidth` + a `ResizeObserver`), so one SVG unit is one CSS pixel and a 12 px label is 12 px at any window size. |
| `data/<release>/results.js`, `hist/`, `ranks/`, `paired/` | One release, written by `scripts/site_export_v2.py` (see "Releases"). `ranks/<key>.js` and `paired/<key>.js` hold one metric's pairwise outcomes and paired contrasts per method pair × problem set × slot (every budget of the release and every time limit, each method placed by `results.js`'s `slots`), so a view loads only the metric it shows. |
| `data/<release>/summary.js` | The few kB the Progress page and the guide read (the protocol's texts, every method's progress and times per budget, and which methods are finished, in progress and scheduled), written next to `results.js` by the same exporter. |
| `data/<release>/pred/` | The Predictions view's files: every method's formula for every problem, as the judge read it (rounded to 4 significant digits), one file per method × problem set × budget × finished run × block of 500 problems (`pred/<method>/<set>/<budget>.<run>.<block>.js`), and the ground truth's (`pred/truth/<set>.<block>.js`). A file is written once its run is complete and never changes after. They come from the `predicted_expression` / `ground_truth_expression` columns of `srbf table`. |
| `data/<release>/pv/` | What the paired contrasts and the pairwise rank outcomes read of every problem, exactly (float64): per method × budget, over the problem sets it has finished there, a frame (`pv/<method>/<budget>/frame.js`: the problems and their runs) and one file per metric (`pv/<method>/<budget>/<key>.js`: each run's comparison score, and each problem's value or share for a paired metric), as `paired_cell_at` / `rank_pair_cell_at` in `scripts/site_export_v2.py` read them (`write_pair_values` documents the format; `results.js` indexes the files as `pv`). |
| `pairstats.js` | Computes the cells of `ranks/` and `paired/` in the browser from `pv/` at any position: at a budget a method ran, or between two of them (interpolated; the rule is in its header), bit for bit what the exporter computes at its slots. Also the one bracket rule (`bracketIn`, with a logarithm that gives the same double in Python and JavaScript). `tests/test_pairstats_parity.py` holds it to the exporter under node. |
| `data/<release>/sealed.js`, `data/<release>/sealed/` | The keyrings, the sealed units and their sealed files (see "Sharing private methods with the people they belong to"). Written by `tools/seal.mjs`, never by hand. |
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
    --public e2e,nesymres-100M,PySR,T8-3M,T8-20M,T8-120M,T8-120M-pysr,prior
```

The method names in `--tree`, `--subset` and `--suite` are the site's method keys (`METHODS` in the exporter).
`--pattern KEY=PATTERN` names one method's rung files when they are not `choices_<rung>.pkl`: a ladder that counts
iterations keeps its `niter_{rung:05d}.pkl` files on the timing subset too.
`--index-variables` is for the E2E and NeSymReS result files written before srbf renamed their variables at the
source. The catalog table reads `data/catalog_mu.json` (`scripts/catalog_mu.py`). A cell holds every run it has, grouped by problem;
it is complete once every problem has at least one run. The progress shown per method reads optional unit lists and completion marks in the root
(`units_<key>_d<draw>.txt`, `markers/<key>.txt`); without them it counts the finished cells.

The exporter writes `results.js` (registry + cells), `hist/<metric>.js`, `ranks/<key>.js`, `paired/<key>.js` and `pv/` next to it.

A new release is a new `data/<id>/results.js`, and the script tag in `index.html` that loads it.

## Local-only methods

Some methods are evaluated but **not published** (they are not ours to release). The site keeps
them out of every public channel by construction, not by convention:

- `scripts/site_export_v2.py --private <keys> --private-dir results-site/private/<release>`
  writes those methods to a separate directory as UNITS: `m/<method>/` per method (its overlay `unit.js`, histograms,
  its contrasts against the public methods, and its per-problem files `pp/`, `pred/`, `pv/`, indexed in the overlay like
  the release's; the ground truth's files are the release's own and are not copied there) and `p/<a>/<b>/` per pair
  of private methods (their contrasts against each other, nothing else). A unit names no other private method, not
  even through a budget. Each `unit.js` pushes its overlay onto `window.RESULTS_V2_PRIVATE_UNITS`. The public payload,
  the pages, `explorer_v2.js` and the repository carry no reference to them; the exporter refuses a key that is in
  both lists.
- `results-site/private/` and `results-site/explorer.local.html` are git-ignored. `build_local.sh`
  writes `explorer.local.html` (the public explorer page plus one script tag that loads every unit of the private
  directory) and refuses to run if either path is not ignored. Serve it locally:
  `cd results-site && python3 -m http.server 8765` then open `/explorer.local.html`. The explorer marks
  such methods "local only".
- `tests/public_guard.py` fails the deploy if a page mentions `private/` or a local page, if
  any method key in `data/*/results.js`, `data/*/hist/*.js`, `data/*/ranks/*.js` or `data/*/paired/*.js` is outside the public
  allowlist in that file, if a release payload lacks any metric of the floor, or if the private
  directory or the local page exists in the CI checkout. It runs before the Playwright suite.
- Screenshots, artifacts, PR descriptions and issues are made from the public page only.

## Sharing private methods with the people they belong to

Private methods can also travel with the site instead of staying on one machine, each opened only by the readers it
was given to. Who gets what is a table of grants: a reader holds one key, and the key opens a fixed set of methods
(one key may open methods 1-4, another methods 2, 3 and 5). `tools/seal.mjs` locks every unit under a secret of its
own and gives each key a keyring that holds the secrets of its methods; the page fetches `data/<release>/sealed.js`
only when someone enters a key under the method list, opens the key's keyring, then the units of its methods and of
every pair of them, runs each unit exactly as a `<script>` would, and from then on treats the methods like any other.
A key that opens no keyring is indistinguishable from a release that ships no sealed file. Keys add up: a reader with
two keys has both sets, and the pairs between them.

    SRBF_SEAL_MASTER="$(cat ~/.config/srbf/master-<release>.key)" SRBF_KEY_A="$(cat <key file A>)" SRBF_KEY_B="$(cat <key file B>)" \
      node tools/seal.mjs <release> --units private/<release> --master SRBF_SEAL_MASTER \
        --grant 'SRBF_KEY_A=*' --grant 'SRBF_KEY_B=gamma-*'

A grant lists method keys; one that ends in `*` takes every method whose key starts with what precedes it, and `*`
alone takes all. A grant that matches no method fails the run, and so do two grants with one key.

- **The locks.** A method's secret is the HMAC of `srbf-unit\0<release>\0<method>` under the master secret, which
  never leaves the machine that seals; a pair's is the HMAC of `srbf-pair\0<release>\0<a>\0<b>` under the two
  methods' secrets side by side (a < b), so only a reader of both makes it. From a unit's secret, HMACs give its id in
  `sealed.js` and three keys: one encrypts its `unit.js`, two seal its files (below). Nothing is random: an unchanged
  input reseals to the same bytes.
- **The keyrings.** Per grant, PBKDF2-HMAC-SHA256 of the key (salt `srbf-keyring/<release>`, 600k iterations, 512
  bits) gives an encryption key and an IV key; the keyring is the JSON list of its methods and their secrets, padded to
  4 kB, encrypted with AES-256-GCM. The keyrings are padded with look-alikes to eight (`--keyrings`) and written in an
  order that says nothing, so the file tells neither how many readers there are nor what each holds.
- **The file.** One line, `window.RESULTS_V2_SEALED["<release>"] = {v: 2, kdf, iter, salt, keyrings, units}`, each
  keyring and unit an `{iv, ct}` envelope.

**A unit's files are sealed one by one.** Its `pp/`, `pred/` and `pv/` files (the Correlations and Predictions views
read them, thousands per method) and its `ranks/`, `paired/` and `hist/` files (one per metric) do not go into
`sealed.js`: with them it would grow many times over and be re-encrypted whole on every change. Each goes into its own
file, `data/<release>/sealed/<name>.js`, which the page fetches only when a reader opens a view that needs it:

- A file's name is the first 32 hex digits of the HMAC of `name\0` and its path below the unit
  (e.g. `pp/<method>/feynman/16.1.js`) under the unit's name key: it gives away neither the unit nor the path. The page
  computes the name from the path it wants.
- The content is gzipped (with a fixed header). The IV is the first 12 bytes of the HMAC of `iv\0`, the path, `\0`
  and those gzipped bytes, so an IV repeats only for identical plaintext; the ciphertext is the gzipped bytes encrypted
  under the unit's file key with the path (`<release>/<path>`) as additional data, so a file only opens as itself.
  An unchanged file reseals to the same bytes and stays out of the next commit (a file that still opens to the same
  content is kept as it is, under its own IV, whatever gzip would write now); a changed one gets a new IV. A file no
  longer in any unit is deleted from `sealed/`. Every unit has its own `ranks/<key>.js`; their names differ, as their
  keys do.
- Each file is one line, `(window.RESULTS_V2_SEALED_FILES=window.RESULTS_V2_SEALED_FILES||{})["<name>"]={"iv":"…","ct":"…"};`.

Three things this rests on, none of them optional:

- **Keys and the master are generated, never chosen.** The sealed file is public, so guessing is offline and
  unlimited; the KDF is the only brake. Keep them outside the repository, out of commit messages and issues, and hand
  a key over through a channel that is not this site. The master never leaves the machine that seals.
- **Publishing a sealed file cannot be undone.** It enters git history, forks and archives. If the key
  leaks later, everything it opened is exposed retroactively, and taking a method out of a grant hides only what is
  sealed after. Seal only what may live in public in that form, and only with the agreement of whoever owns the
  results.
- **The guard checks it is sealed** (`tests/public_guard.py`): one line, the tool's fields and no others, KDF strength,
  keyrings of one size, unit ids that name nothing, and ciphertext that reads as ciphertext; and `data/<release>/sealed/` holds nothing but one-line envelopes named by
  their own file names, whose ciphertext reads neither as text nor as an unencrypted gzip stream. The checker is run
  against deliberately unsealed blobs on every invocation, so it cannot pass vacuously.

`tests/fixtures/` holds stand-in units and their sealed forms, so the Playwright suite exercises the whole path —
wrong key, each key its own methods, a key of two methods and their pair — without any real payload
(`tests/fixtures/README.md` has the commands that reseal them); `tests/seal_files.spec.mjs` seals stand-ins the same
way and opens them as a page does.

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
