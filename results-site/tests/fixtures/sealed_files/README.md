# Sealed per-problem files: test fixture

`source/` is a stand-in overlay for one method (`fixture-files`, "Fixture Files") with its per-problem files:
problem set feynman at budgets 16 and 32, Correlations files for runs 1 and 2 (`pp/`) and Predictions files for run 1
(`pred/`). They are the release's T8-20M files (`data/2026-09/pp/T8-20M/feynman/`, `data/2026-09/pred/T8-20M/feynman/`)
under the fixture's key. `sealed.js` and `sealed/` are what `tools/seal.mjs` makes of it for release 2026-09 with the
test key in `site_v2.spec.mjs`; a page test serves them as `data/2026-09/sealed.js` and `data/2026-09/sealed/`.
Regenerate them from `results-site/` with:

    SRBF_SEAL_KEY='fixture-key-for-the-tests' node tools/seal.mjs 2026-09 tests/fixtures/sealed_files/source tests/fixtures/sealed_files/sealed.js

A file in `sealed/` that still opens to the same content is kept as it is; to seal every file afresh (after a change
of the format), remove `tests/fixtures/sealed_files/sealed/` first.
