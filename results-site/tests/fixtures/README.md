# Sealed fixtures

Two sets of stand-in private units (the layout `scripts/site_export_v2.py --private-dir` writes: `m/<method>/` per
method, `p/<a>/<b>/` per pair of them) and what `tools/seal.mjs` makes of them for release 2026-09. The page tests serve
a set's `sealed.js` as `data/2026-09/sealed.js` and its `sealed/` as `data/2026-09/sealed/`; `seal_files.spec.mjs`
checks that each opens with its keys and is what a reseal of its units writes.

- `keyrings/`: two methods, `fixture-method` ("Fixture Method", one complete cell) and `fixture-method-two`, and their
  pair (a ranks file). Three keys: the first opens `fixture-method`, the second `fixture-method-two`, the third both and
  their pair.
- `sealed_files/`: one method, `fixture-files` ("Fixture Files"), with per-problem files: problem set feynman at budgets
  16 and 32, Correlations files for runs 1 and 2 (`pp/`) and Predictions files for run 1 (`pred/`), the release's T8-20M
  files (`data/2026-09/pp/T8-20M/feynman/`, `data/2026-09/pred/T8-20M/feynman/`) under another key. One key opens it.

The master secret and the keys are the tests' own (`seal_files.spec.mjs`, `site_v2.spec.mjs`). Regenerate both sets from
`results-site/` with:

    export SRBF_MASTER=fixture-master-secret-for-the-tests SRBF_KEY=fixture-key-for-the-tests \
      SRBF_KEY_TWO=a-second-fixture-key-for-the-tests SRBF_KEY_BOTH=a-third-fixture-key-for-the-tests
    node tools/seal.mjs 2026-09 --units tests/fixtures/keyrings/units --master SRBF_MASTER --grant SRBF_KEY=fixture-method \
      --grant SRBF_KEY_TWO=fixture-method-two --grant 'SRBF_KEY_BOTH=fixture-*' --out tests/fixtures/keyrings/sealed.js
    node tools/seal.mjs 2026-09 --units tests/fixtures/sealed_files/units --master SRBF_MASTER --grant SRBF_KEY=fixture-files \
      --out tests/fixtures/sealed_files/sealed.js

A file in a `sealed/` that still opens to the same content is kept as it is; to seal every file afresh (after a change of
the format), remove that `sealed/` first.
