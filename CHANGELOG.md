# Changelog

All notable changes to srbf are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.18.0] - 2026-09-12

### Fixed
- **`symbolic_recovery` judges both sides through the same simplify.** The ground truth's skeleton is
  simplified before the comparison (`pow x1 / <c> <c>` -> `pow x1 <c>`); a prediction whose canonical
  form is not strictly shorter was judged by its stored skeleton UNsimplified, so a prediction
  byte-identical to the ground truth (`pow x1 / 2 3`) was not exact. Every law with a rational
  exponent -- 396 of 6,660, 7 % of erbench-syneq -- was unjudgeable as exact, whatever the answer. The
  stored skeleton now goes through the same simplify on the fallback path. A second-stage metric over
  stored prefixes: re-derive, nothing needs re-running. Results judged under 0.17.x and 0.18.x are not
  comparable on those laws, which is what the minor bump marks.

### Added
- **Answer provenance in the flash_ansr adapter's rows:** `predicted_typed_frozen` (predicted typed
  literals kept verbatim), `predicted_typed_thaw` (a thawed duplicate: the typed token indices it
  re-fitted; None otherwise) and `predicted_spelling` (the constant ladder's re-spelling record) for
  the rank-0 answer, from flash-ansr >= 0.16.1 (whose ladder children carry the thaw mark). Pin
  `flash-ansr>=0.16.1,<0.17`.

## [0.17.0] - 2026-09-12

### Changed
- **Evaluates flash-ansr 0.16** (`flash-ansr>=0.16,<0.17`). flash-ansr 0.16.0 keeps the numbers a
  model predicts in a mixed emission (0.15.x post-processing deleted them from 72 % of valid
  candidates) and defaults to `refiner_typed_spans='freeze_then_free'`, so every flash-ansr arm
  evaluated under this srbf draws from a different candidate pool than under 0.16.x; results
  across the two are not comparable, which is what the minor bump marks. Nothing in the driver,
  the metrics or the judge changed.

### Added
- **The diffsym worker** (`worker: diffsym`, `configs/evaluation/baselines/diffsym_fastsrb.yaml`):
  Mara Eliana's discrete diffusion model (D3PM) conditioned on the support set, run out of process in
  its own interpreter (it pins simplipy 0.2.15 and its own torch). Per problem it samples `n_samples`
  token sequences, decodes/simplifies/validates them into prefix candidates, fits each candidate's
  constants with diffsym's own ConstantFitter and returns the best by R^2. Two translations sit
  between the two: the pre-0.12 vocabulary (`pow2`/`pow1_3`) is respelled with
  `respell_legacy_prefix`, and diffsym's positional `x1..xn` become the catalog's own variable names.
  A candidate naming a variable the problem does not have (the decoder's vocabulary is the
  checkpoint's `x1..x8`, not the problem's) is dropped and counted in `diffsym_n_out_of_range` rather
  than failing in the fitter. Smoked on the v4.0 checkpoint (119M parameters, step 1.7M): `srbf check`
  green on fastsrb, II.38.14 recovered at 32 draws.


## [0.16.1] - 2026-09-11

### Fixed
- **The judged skeleton is the canonical form the prediction was priced as.** flash-ansr < 0.15.2
  emitted a fitted candidate as its skeleton with the numbers filled in, so a factor the fit made
  cancel (`exp(-x^2) * tanh(x)^2 / tanh(x)^2`) or a constant the fit made fold stayed in the spelling
  while the certified price had collapsed it -- and `symbolic_recovery`, a masked-skeleton identity,
  missed the recovery. `derive_metrics` now judges the masked, simplified skeleton of the prediction's
  canonical form whenever that form is strictly shorter than the emitted one (the stored skeleton
  stays under `predicted_skeleton_prefix_as_emitted`), so files from before and after flash-ansr
  0.15.2 are judged alike. Measured on the T8-20M r = 0 readings (766 problems): symbolic recovery
  9.3 % as emitted -> 13.2 % canonical. Numeric metrics and the MDL price are unaffected.


## [0.16.0] - 2026-09-11

### Removed
- **The hybrid arm's method code.** srbf is the evaluation framework; a method does not live in it.
  `srbf/hybrid_adapter.py` (`model_adapter.type: flash_ansr_pysr`), `scripts/make_hybrid_config.py`,
  `scripts/run_hybrid_sweep.py`, `scripts/hybrid_rescore.py` and `scripts/hybrid_curve.py` are gone. The method -- Flash-ANSR
  generating by the clock until its share (1 - r) T, its top-K candidates seeding PySR, PySR on its own
  clock (its share minus the running means of its fixed cost and of the pricing that follows), its hall
  of fame re-fitted with flash-ansr's refiner, re-spelled by the constant ladder, priced and scored like
  Flash-ANSR's own candidates, Flash-ANSR's sorting picking rank 0 -- and its sweep tooling now live in
  the `flash-ansr-hybrid` package (0.1.0: `HybridRegressor`, `flash-ansr-hybrid-make-config`,
  `flash-ansr-hybrid-run-sweep`, `flash-ansr-hybrid-rescore`, `flash-ansr-hybrid-curve`). PySR runs in-process there, so the
  `pysr:` block no longer names an interpreter.

### Added
- **`model_adapter.type: flash_ansr_hybrid`** (`FlashANSRHybridAdapter`): the thin adapter for the
  flash-ansr-hybrid regressor -- `flash_ansr:` (the model block), `hybrid:` (the regressor's
  `HybridConfig` fields plus `snapshot_dir`), optionally `pysr:` (its `PySRSettings`). It hands each
  problem's arrays to the regressor and records its answer with the usual FVU columns; nothing of the
  method is in srbf. The package is the `hybrid` extra (`pip install "srbf[hybrid]"`), imported lazily.

## [0.15.1] - 2026-09-11

### Changed
- **In the hybrid arm, PySR adds candidates and Flash-ANSR's sorting picks.** After the GP stage,
  PySR's whole hall of fame joins the Flash-ANSR candidate pool, each entry priced the way Flash-ANSR
  prices its own (fit as FVU on the fitted target, MDL as the certified f64 default-canon price of the
  realized expression, the ranking's `score_row` with its MDL penalty), and Flash-ANSR's sorting picks
  rank 0 of the extended pool as the prediction. PySR's own choice (`model_selection`) no longer decides
  anything; it is kept for reference (`pysr_expression`). The record stores the ranked pool
  (`hybrid_candidates`) and the origin of rank 0 (`predicted_source`); a GP stage that failed leaves
  the Flash-ANSR candidates to rank alone (`pysr_error`). Snapshots now store the top-K pool
  (`candidates`); older snapshots contribute their rank 0, which the same sorting puts first either
  way. `scripts/hybrid_rescore.py` applies the rule offline to results already on disk (the hall of
  fame is stored in every row; the Flash-ANSR candidates come from the generation snapshots).

### Fixed
- **Predictions read from infix carried the raw reader tokens.** The out-of-process adapters (PySR, the
  hybrid arm's PySR stage), E2E and NeSymReS turned the method's infix string into a prefix with simplipy's
  raw `infix_to_prefix`, whose output spells a power as `**` and a negative literal as `neg` -- tokens the
  engine's `simplify` and `complexity` refuse as malformed. Numeric metrics were unaffected; exact symbolic
  recovery and the MDL price silently dropped every such row (243 of the hybrid arm's 562 recovered answers
  at r = 0.1). The adapters now use the engine's documented reader, `read_infix`, which returns the engine
  grammar (`pow`, signed literals); `derive_metrics` converts the stored prefixes of results already on disk
  through `engine.convert_expression` before judging or pricing them (`convert_fn`, the identity on prefixes
  already in the grammar). Decontamination reads its alternate renderings the same way.

## [0.15.0] - 2026-09-10

### Changed
- Requires `flash-ansr>=0.15,<0.16` (the constant ladder and the shared prefill in the sampler).

### Added
- **The hybrid arm: Flash-ANSR seeds + PySR at a fixed time budget** (`model_adapter.type:
  flash_ansr_pysr`, `srbf/hybrid_adapter.py`). One budget T per problem is split by a ratio r:
  Flash-ANSR gets (1 - r) T, PySR gets r T, both controlled by direct knobs from two measured time
  laws (`scripts/hybrid_scaling_laws.py`: seconds = a + b * choices, seconds = a + b * niterations),
  never by timeouts; the achieved seconds of both stages are recorded next to the targets. The
  top-K refined Flash-ANSR candidates enter PySR as initial `guesses` in Julia syntax (the worker
  reads per-problem `guesses` / `niterations` from the fit payload's `meta`; `SubprocessAdapter
  .evaluate_sample(extra_meta=...)` carries them). One chunked generation pass per problem is
  snapshotted at every ratio's candidate count and cached on disk, so the whole r-sweep pays the
  generation once. Because the catalog source re-draws the support points on every iteration
  (symbolic_data's `ProblemSource` is entropy-seeded; reproduction is a materialized source), the
  sweep runner FREEZES the stratified subset once per catalog (`<root>/hybrid_data/<catalog>.npz`,
  `meta.source_row_index` = the row in the source catalog) and every cell reads it through a
  derived `<config>.frozen.yaml`; each snapshot records a fingerprint of the arrays it was
  generated on (`hybrid_adapter.data_fingerprint`) and is regenerated, with a warning, on a
  mismatch. `scripts/make_hybrid_config.py`, `scripts/run_hybrid_sweep.py` (the calibrated
  protocol's stratified subset, one problem at a time) and `scripts/hybrid_curve.py` (the r-curve
  with intervals and the pre-registered paired test) complete the kit.
- **flash-ansr's constant ladder is configurable and its rows are stored.** A `constant_ladder`
  block under `model_adapter` (or `evaluation_config`) is passed through to `FlashANSR.load`
  (`true`, `false`, or a mapping; absent = flash-ansr's own default, which is ON with the surprise
  rule, over every candidate). The candidate store gains two columns for the
  re-spelled variant rows the ladder adds beside their parents: `spelling` (the re-spelling record,
  `''` for a fitted draw) and `parent` (the store row of the beam the variant came from, `-1`
  otherwise); the store's `run_meta` records the ladder settings so a reader knows whether such
  rows can exist. Skeleton, MDL and constants of a variant flow through the existing columns.

### Changed
- The shipped scaling ladder gains the 65,536 rung (powers of two to 16,384, then 65,536) for every
  arm; `flash-ansr-v25.0-T8-20M` and `flash-ansr-v25.0-T8-120M` join the generated model arms; the
  prior arm caps its attempts at 524,288 per problem (`max_tries`), so a one-column problem that
  starves of unique draws stops instead of spinning.

### Added
- `scripts/run_calibrated_ladder.py`: the calibrated protocol -- a scaling config's experiments x rungs run
  sequentially on one machine, the whole suite up to a rung (default 4,096) and a deterministic stratified
  subset above it (shard 0 of N per catalog, N by catalog size), `--refiner-workers` pinned into the recorded
  config, markers and resume.
- **MDL ratio.** The derived metrics carry `predicted_mdl`, `ground_truth_mdl` and `mdl_ratio`:
  the description length of the realized predicted expression over that of the ground truth, in
  the engine's own measure (`complexity`, milli-bits: certified, f64 parse, Default canon; the
  price the flash-ansr ranking modes score with). A prefix that still carries a `<constant>`
  placeholder is not priced, and "MDL ratio" is among the report's default metrics.
  `derive_metrics`/`compute_derived_metrics` take `mdl_fn` for a pricer other than the engine's.

## [0.14.0] - 2026-09-07

### Added
- **Out-of-process model adapters** (`model_adapter.type: subprocess`). `srbf` starts a
  standard-library-only worker (`srbf/worker/runner.py`) inside the interpreter you name and
  exchanges one problem at a time with it over a local socket (the worker's stdio is captured, never parsed), so a method runs against its own
  package versions while `srbf` stays on its pins. A worker is a Python file with
  `fit(x, y, *, x_val, variables, meta, options, state) -> {"expression": ...}` and optional
  `load`/`info`; `srbf` parses, evaluates and judges the returned expression with the run's
  engine, restarts a crashed or timed-out worker within `max_restarts`, and records every
  failure as an error row. Keys: `worker`, `python`, `options`, `simplipy_engine`, `timeout`,
  `startup_timeout`, `env`, `cwd`, `drop_unused_variables`, `max_restarts`, `worker_log`.
  `srbf_worker_helpers` (importable in the worker interpreter) respells the pre-0.12 simplipy
  vocabulary and renders prefix tokens as infix. The benchmark loop closes the worker however
  the run ends.

- **Sharded runs.** `srbf run --shard K/N` evaluates every N-th problem from K, writes
  `<output>.shard-K-of-N.<ext>` and resumes on its own; `srbf merge -o <output> <shards...>` puts
  the shards back into the unsharded file (one `count`, distinct indices, identical columns,
  disjoint `eval_row_index`, rows ordered by it; `--allow-partial` records a gap in `__meta__`).
  `CatalogSource` takes `shard=(index, count)`; `Benchmark.from_config` scales an explicit total by
  the shard's share and stamps `shard` into `__meta__`.

- **The contributor path is four commands.** `srbf new <name>` scaffolds an adapter (a worker
  with a placeholder `fit`, its whole-suite config, an environment recipe and a smoke test; `--repo`
  writes them into the checkout layout a pull request wants, `--python` names the interpreter);
  `srbf check -c <config>` walks the config end to end on a few real problems, one step at a time
  (provenance label, output directory, worker and interpreter, engine, catalog, adapter, fit,
  derived metrics), printing `ok`/`FAIL` with the fix beside each step and exiting non-zero on any
  failure; `srbf analyze -c <config> [--model NAME]` derives the report's runs from the config
  (experiments x sweep rungs whose outputs exist) instead of a hand-written manifest, and several
  `-c` render side by side. `srbf.testing.fit_once` runs one toy problem through a worker for its test.

- **`suite:` config shorthand.** One `run:` template plus `suite: srbf` (or a catalog list) expands
  into `experiments:` with one entry per catalog, `data_source.catalog` filled in and `{catalog}`
  substituted in every string, sweep values included; `srbf.suites.SRBF_CATALOGS` is the suite.

- The worker registry is a scan of `srbf/worker/models/*_worker.py`, so a merged
  `<name>_worker.py` is `worker: <name>` without a code change. `CONTRIBUTING.md` and a pull-request
  template carry the checklist.

### Fixed
- `srbf analyze` defaulted to the retired `dev_7-3` engine, which simplipy 0.14 refuses to load;
  the default is `acj-5-4-llm`.
- A worker log under a directory that did not exist yet failed the first run; the directory is
  created.

### Changed
- **PySR runs as a worker.** `type: pysr` keeps its keys and now launches the shipped
  `srbf/worker/models/pysr_worker.py` in a subprocess (`python:` names the environment that has pysr
  and Julia); the in-process `PySRAdapter` is gone.

## [0.13.0] - 2026-09-05

### Added
- **`r2_fit` / `r2_val`**, the coefficient of determination, and "R² (val)" among the default
  summary metrics. Defined as `1 - fvu` under FVU's own numerics (a recovered fit reads exactly
  1.0) and clipped to [0, 1], with a failed or non-finite prediction counting 0, which is the
  convention under which the literature reports a mean R² across problems. It separates methods
  far less than recovery or log FVU, and is reported because readers expect it.

### Changed
- **The shipped scaling ladder stops at 16,384 choices** (was 262,144), ten draws per expression
  on every rung: about two minutes per problem at the top rung, the budget the full-suite
  campaign runs at. The longer rungs are deferred.

### Fixed
- **`--sweep-filter` matches the rung it names.** A zipped `!sweep` axis was labelled by its
  first co-named sweep in document order, which in every shipped scaling config is the
  `problems_per_expression` column (`10` on all but the last rungs), so `--sweep-filter ladder=32`
  resolved zero runs. The label is now the value that tells the rungs apart (the `choices` /
  `samples` / `niterations` ladder); an axis without such a node keeps its first sweep's value.

## [0.12.0] - 2026-09-05

The release that pairs with flash-ansr 0.14: the third model generation, its reference checkpoint
`psaegert/flash-ansr-v25.0-T7-3M`, and the three candidate-ranking modes. Requires
`flash-ansr>=0.14,<0.15`, `simplipy>=0.14.6,<0.15` and `symbolic-data>=0.18,<0.19`.

### Added
- **A shipped evaluation of the reference checkpoint**:
  `configs/evaluation/scaling/flash-ansr-v25.0-T7-3M_fastsrb.yaml` sweeps the sampling budget on
  FastSRB with the application-mode settings (fittable emission and refine scope, MDL ranking at
  1e-2 per bit, 8 restarts, a decode cap of 160 tokens) and reads the model from the directory
  `flash_ansr install` writes to.
- **`refine_scope` on the flash_ansr adapter config** (adapter key, falling back to the
  evaluation config): which literals of a candidate the refiner may move. `'fittable'` (default)
  fits every `<constant>` slot plus every spelled literal a constant optimizer can move and keeps
  the typed literals, `pow` exponents and `rootn` indices, as the model spelled them;
  `'placeholders'` fits only the slots; `'all'` frees every literal. Rides on
  `FlashANSR.load(refiner_scope=)`.
- **Decontamination verification** (`python -m srbf decontamination -t <training catalog yaml>
  [-b name ...] [-o report.json]`; library: `srbf.decontamination.verify_decontamination`): probes
  every benchmark problem against the training catalog's registered holdout via `is_held_out`
  (the training-time sampler's own family quotient) and reports per-catalog coverage (total /
  held / missed / black-box / unparseable). Fail-closed: a problem whose tokens cannot be probed
  counts as unverified, never as covered; exit code 0 only when every probe-able benchmark problem
  is verified held out.

### Changed
- **`ranking:` is required on a flash_ansr adapter** (under `model_adapter`, else under its
  `evaluation_config`; an adapter-level block replaces the other outright), strict inside: `mode`
  (`mdl` | `weighted` | `pareto`) plus that mode's knobs (`mdl_strength`; `weights`; `metrics` +
  `tie_break`), validated through flash-ansr's own `resolve_ranking`. **Breaking:**
  `node_penalty`, `constants_penalty`, `likelihood_penalty`, `mdl_penalty` and the never-read
  `parsimony` / `length_penalty` now raise at every layer, naming the replacement. Every srbf
  flash_ansr run before this release ranked at an effective length penalty of 0.0 while its config
  said `parsimony: 0.05` (a key the adapter never read); the block exists so a run has to say how
  it ranks. The resolved values are written to `__meta__['ranking']` and to every row's `ranking`
  column (the three penalty columns are gone); rows also carry `predicted_mdl`,
  `predicted_n_nodes` and `predicted_pareto_rank`. The refiner baselines keep their own
  `node_penalty` / `constants_penalty` / `likelihood_penalty` and report them as a `weighted`
  ranking record.
- **The default `emission` is `'fittable'`** (was `'constants'`): the application mode, in which
  the model spells the typed literals and leaves every fittable constant as a placeholder that
  the refiner fits from random inits. Configs that want the unflagged format say
  `emission: constants` explicitly.
- **Provenance names both checkouts.** The srbf checkout is found by walking up to `.git` (the
  fixed `parents[3]` pointed above the repo, so every `__meta__['git']` written so far was
  empty); `git_flash_ansr` records the model code's checkout; `env` carries the flash-ansr,
  simplipy and symbolic-data versions; the model's `model.safetensors` is hashed.
- **The save-all candidate store carries the re-ranking substrate.** With `candidate_store_dir`
  set the adapter asks `infer(top_k='all')` and writes, per candidate, `n_nodes`, `n_constants`,
  `mdl`, `score`, `pareto_rank`, `rank` (from flash-ansr's ledger) and `fvu_val`,
  `recovery_fit`, `recovery_val` (from every candidate's predictions through
  `srbf.metrics.numeric`); the manifest's `run_meta` holds the run's ranking, the mu dialect and
  the simplipy version. A store written without the flag is unchanged.
- **Constants reach the evaluation record at float64.** The coercion between a
  `symbolic_data.Problem` and the eval record matches the generator's storage width, so a
  constant is benchmarked at the precision it was realized at. Fit-quality tolerances are
  unchanged: `is_perfect_fit` and the reference-FVU noise floor still use float32 epsilon, which
  is a bar on agreement, not a storage width.
- **The shipped baseline configs pin the `acj-5-4-llm` SimpliPy engine** (was `dev_7-3`, a
  generation-1 artifact that `simplipy>=0.14` refuses to load). The engine canonicalizes and
  simplifies every method's predictions for the recovery metrics, so it is part of the protocol:
  one engine for the reference checkpoint and the baselines alike. Results produced under
  `dev_7-3` are not comparable with new runs without re-running.
- **Dependencies.** `flash-ansr>=0.14,<0.15` (the ranking modes, `refiner_scope`, safetensors
  checkpoints), `simplipy>=0.14.6,<0.15`, `symbolic-data>=0.18,<0.19`.

### Removed
- **Retired evaluation configs cut from `configs/evaluation/`** (91 files; git history is the
  archive): every config referencing a retired catalog (`v23-val`, `lample-charton-v23`), pinning
  a flash-ansr v23 checkpoint (flash-ansr 0.14 refuses these at load), or requesting the retired
  `method: beam_search`. The shipped configs are the FastSRB arms: the reference checkpoint and
  the PySR, NeSymReS and E2E baselines. The schema gate
  (`tests/test_eval/test_scaling_configs.py`) accepts `fastsrb` as the only shipped catalog.

## [0.11.1] - 2026-07-10

FVU hardening parity with flash-ansr 0.11.0 (the 2026-07 reconciliation): the evaluation-side `fvu` adopts
the research-hardened implementation, baseline selection variance switches to `ddof=0`, and the flash-ansr
pin floor moves to `>=0.11` (corrected scoring semantics; 0.10-era selection scores are not comparable).

### Changed
- **`metrics.numeric.fvu` hardened** (finite paths byte-identical): squared-residual UNDERFLOW guard on
  tiny-magnitude targets (un-squared max-residual check distinguishes a genuine perfect fit from
  underflow; scale-invariant `_normalized_by_gt` fallback shared with the overflow guard), rescaled-sum
  underflow guard (FVU is never NaN), object-dtype/list inputs -> `inf` via coercion `try/except`,
  `ss_tot` finite check in the normalizer, and `@np.errstate` to silence the intentionally-probed numpy
  warnings. Keeps 0.11.0's empty-array guard and `bool()` cast.
- **Baseline selection variance `ddof=1 -> ddof=0`** (`baselines/_base.py`), matching the evaluation-side
  FVU definition and flash-ansr 0.11.0's selection variance, so selection-FVU == eval-FVU family-wide.
- **BREAKING (transitive): `flash-ansr>=0.11`.** Forces the corrected candidate scoring (scale-invariant
  `compute_fvu`, `score_from_fvu` ranking fix). Numbers produced against 0.10.x are not comparable by
  default; see the flash-ansr 0.11.0 changelog.

### Added
- `tests/fixtures/golden_baseline_results.json` re-recorded under 0.11 scoring semantics (the ddof=0 shift scales every selection FVU by n/(n-1); verified quantitatively before re-recording).
- `tests/test_fvu_correctness.py`: the cross-path FVU consistency suite (scale-invariance, tiny-magnitude
  regression, eval-never-NaN, non-finite-is-worst, input contract) ported from the research repo; asserts
  `srbf.metrics.numeric.fvu` and `flash_ansr.scoring.compute_fvu` agree on the reduced inputs.

## [0.11.0] - 2026-07-10

### Added
- **Reference-relative recovery (WP7 P0, real-data catalogs):** `CatalogSource` bridges
  symbolic-data 0.11.0's reference-law prediction arrays into per-row `y_ref`/`y_ref_val`
  columns (defaulting to the clean targets for synthetic problems), and
  `compute_derived_metrics` derives `reference_fvu_{fit,val}` (the accepted law's own FVU on
  the same target) plus `numeric_recovery_relative_{fit,val}` (candidate FVU ≤
  max(reference FVU, float32 eps); a RATE metric, failures = misses). On clean synthetic
  catalogs the reference FVU is exactly 0 and the relative criterion reduces elementwise to
  machine-precision `numeric_recovery` (regression-tested endpoint identity). The full R(ε)
  profile remains deferred; it is pure post-processing over these columns.
- **`gt_kind` bridged per row** (`exact` | `reference` | `none`; `exact` for older
  symbolic-data releases): metric regimes key on it — reference/black-box FVU is never pooled
  with exact-GT rows.

## [0.10.1] - 2026-07-07

### Added
- **PySR panel knobs + hall-of-fame persistence** (the WP3 selection-rule panel prerequisites):
  `model_selection` (default `'best'` = upstream), `parsimony` and the previously
  documented-but-unplumbed `maxsize`/`warmup` keys now flow from the `model_adapter` config
  block through `_build_pysr_adapter`; `maxsize`/`parsimony` are forwarded only when set
  (None = the installed PySR's own default, never a hardcoded number). Every successful fit
  persists the full Pareto hall of fame as `record['equations']`
  (complexity/loss/score/equation; ~3-15 KB per problem), making any selection rule an
  offline re-score against the stored raw arrays.
- **`configs/evaluation/panels/` carve-out** in the config gate: pre-registered
  side-experiment arms live there and must carry `config_provenance: harness_tuned`.

## [0.10.0] - 2026-07-05

### Added
- **Config provenance (WP3):** every `model_adapter` block declares who chose the configuration
  via `config_provenance: upstream_default | author_blessed | harness_tuned`, validated at config
  load (`srbf.config.coerce_config_provenance`) and embedded in every result pickle's `__meta__`
  next to the measured run provenance. All 76 shipped eval configs are labeled, test-gated to the
  policy assignment (third-party baselines `upstream_default`; Flash-ANSR `author_blessed`; an
  omitted key resolves to `harness_tuned` as the conservative reading).
- **docs/fairness.md:** the written fairness policy — one protocol for every method, the
  baselines-at-upstream-defaults policy (canonical home; models.md keeps the PySR worked example),
  the three provenance labels, the blessed-config submission path for method authors (one blessed
  configuration per method), and the rule that headline comparisons state every entrant's label.
  Cross-linked from the nav, README/index documentation tables, models.md, adapters.md (PR
  checklist), and the maxsize audit script.

### Fixed
- `ResultStore.extend` (and thus the constructor) now ignores the reserved `__meta__` key that
  `save` embeds, so a raw loaded pickle round-trips without the caller stripping it first.

## [0.9.0] - 2026-07-04

### Added
- **Rank leagues:** `srbf.reporting.rank_league` (tie-corrected Friedman omnibus + Nemenyi
  critical difference + indistinguishability cliques — the CD-diagram statistics, Demšar 2006)
  and `srbf.reporting.series_values_at_time` (the per-problem value vector of one series at
  exactly a wall-clock time t, exposing the at-time interpolation for cross-method
  constructions). Two block designs: `worst-rank` (missing methods rank strictly worst per
  problem — quality-to-GT axes over the full problem set) and `all-present` (the conditional
  all-methods-succeeded league for output-property metrics). Powers the results explorer's new
  **Ranks** display: a critical-difference league of the pre-declared roster at the marked
  budgets, with a declared property-based metric-eligibility rule and held-out log10 FVU as the
  single primary (quotable) league.

## [0.8.0] - 2026-07-03

### Added
- **Exact-t reports:** `srbf.reporting.paired_report_at_time` and
  `srbf.reporting.series_report_at_time` evaluate a pair (or one series) AT EXACTLY a wall-clock
  time t per problem, by per-problem linear interpolation in log10-time between the bracketing
  measured configurations (the same model as `paired_delta_curve`), with explicit, never-
  extrapolated boundaries: below-ladder → `None`; beyond-ladder carries the last measured value
  forward (`status='plateau'`, a lower bound under the monotone quality-in-compute assumption),
  and a verdict stands only if no plateau side could overturn it by improving (else `undecided`
  with `verdict_note='ladder-limited'`). These functions power the results site's standardized
  budget grid: cross-method verdicts and the Table view are now issued at exactly t, so a
  method's ladder phase (where its power-of-two configurations happen to land relative to the
  budget) no longer skews comparisons; same-knob pairs (ablations, size ladder, versions) keep
  the same measured configuration on both sides (one-factor principle).

### Changed
- **The results explorer's three tabs (Curves | Paired Δ | Matrix) generalize to a 2×2 view
  grid** — Display (Curves = the whole compute sweep | Table = one selected compute budget) ×
  Values (Absolute = each series' own value, marginal bootstrap CI | Paired = per-expression
  head-to-head differences with four-state verdicts) — adding the fourth quadrant, the
  **Table × Absolute view** (`?view=table`; the curves/paired/matrix deep links are unchanged):
  per benchmark, each series' best measured configuration within the selected budget (the same
  best-within-budget selection the matrix uses for its sides), with its marginal value + 95% CI,
  numerically identical to that configuration's point on the Curves view (enforced by an
  exporter self-check). The marginal-CI warning is built into the view: rows are never to be
  differenced — head-to-head questions belong to the Paired views.
- **Subtler confirmatory marking in the Matrix view:** confirmatory cells now carry only a small
  faint superscript C (the earlier accent outline + badge read as clutter); tapping/clicking a
  cell pins its full record — confirmatory status spelled out — below the table, so the detail
  stays reachable on mobile.

## [0.7.0] - 2026-07-03

The paired-statistics layer (improvement-plan WP1): compare models on the SAME expressions
instead of subtracting marginal averages.

### Added
- **`srbf.reporting.paired_report(snapshot_a, snapshot_b, metric_key, ...)`** — per-expression
  deltas joined on ids (never row order), one expression-bootstrap pass serving mean/median CIs,
  win/tie/loss, probability of superiority (+ CI), a Wilcoxon signed-rank companion with
  `zero_method='pratt'` (zeros counted), the two-sided bootstrap `p_value` on the mean delta,
  `mde_80` (minimum detectable effect), and a draws-vs-expressions variance decomposition.
  Optional `worst_rank=True` (union-of-ids rank statistics with sign-only sentinels for
  one-sided failures; degenerate at >= 50% imputed) and `hierarchical=True` (two-stage bootstrap
  for rank statistics).
- **Four-state verdicts vs pair-specific measurement-noise margins (MRD):**
  `self_noise` (split-half noise null of a series against itself, exact per-expression
  rescaling, bootstrap cross-check) + `pair_margin` (convolution of two nulls) →
  `better / equivalent / worse / undecided` with an `equivalence_attainable` diagnostic.
  Margins are derived from the data, never hand-picked (`scripts/derive_noise_margins.py`).
- **Pairing contract:** `pairing_fingerprint` / `PairingContractError` — benchmark-data
  provenance must match where available (`allow_unverified=True` for archived snapshots);
  id-set join diagnostics (`n_only_a/b`) disclosed in every report; strictly per-benchmark.
- **`paired_delta_curve`** — Δ over the compute axis: `x_policy='rung'` (exact configuration
  match) or `'time'` (per-expression linear interpolation in log wall-clock time between
  measured rungs; never extrapolates — out-of-range points are returned loudly; composition
  guard with per-point `n_pairs`; pointwise bands).
- **`bootstrap_band`** — shape-agnostic row bootstrap ((n,) or (n, k) profiles) whose scalar
  case reproduces `bootstrapped_metric_ci` exactly; `draw_values` / `paired_expression_deltas`
  primitives; `significant_round` / `rounded_triple` display rounding (precision limited to
  what the CI width justifies).
- **Docs:** a "Paired comparisons" page (statistical model, verdict semantics, contract,
  worked example). The interactive Paired Δ / Matrix views are live on the results explorer.
- `scipy>=1.9` is now a declared dependency.

## [0.6.2] - 2026-07-02

### Changed
- **PySR runs at its upstream default `maxsize` again** (the 0.6.1 override is removed). Policy
  decision: baselines run at their library defaults — a method's default hyperparameters are part
  of the method, and a default that limits it is a property of that method, not srbf's to correct.
  The 0.6.1 audit numbers stand as documentation (at PySR's `maxsize=20`, 23/120 FastSRB and
  743/1000 v23-val ground truths are not representable; `scripts/audit_pysr_maxsize.py`,
  docs/models.md). The optional `maxsize` key in the `model_adapter` block remains for side
  experiments only; it is no longer set by default. The warmup fit and the seedable bootstrap
  from 0.6.1 are unchanged.

## [0.6.1] - 2026-07-02

Fairness + reproducibility quick wins (improvement-plan WP0).

### Fixed
- **PySR gets an explicit complexity budget (`maxsize=45`).** PySR's own default (`maxsize=20`)
  makes 23/120 FastSRB and 743/1000 v23-val ground truths structurally inexpressible under the
  adapter vocabulary (largest ground truth = 40 nodes; v23-val median 25), so runs at the library
  default measured a representation handicap rather than search quality. The audit lives in
  `scripts/audit_pysr_maxsize.py`; PySR results produced before 0.6.1 should be treated as lower
  bounds. `maxsize` is overridable per `model_adapter` block.
- **PySR timing no longer carries the Julia precompile outlier.** `PySRAdapter.prepare()` runs a
  warmup fit on a throwaway model (`warmup: true` by default), so problem 0's `fit_time` starts
  warm. The other built-in adapters already load their models in `prepare()`.

### Added
- **Seedable bootstrap.** `bootstrapped_metric_ci` accepts `rng` (`np.random.Generator` | int
  seed | `None` for fresh entropy); `bootstrap_report` defaults to `rng=0`, making reports
  bit-reproducible by default (pass `rng=None` for the previous unseeded behaviour). Interpret
  results by the confidence interval either way.

## [0.6.0] - 2026-07-01

Post-release audit round (deferred tiers C + D): baseline de-duplication, fail-fast adapter
validation, and reporting/metric performance. Re-pinned to the coordinated `flash-ansr>=0.10` /
`symbolic-data>=0.10` release.

### Added
- **`srbf.derive_metrics(snapshot, *, engine=None, operator_arity=None, simplify_fn=None)`** -- the
  clean standardized second stage. A `Benchmark.run()` emits RAW results only; `derive_metrics`
  turns one raw snapshot into a NEW snapshot with the derived metric columns (FVU, numeric/symbolic
  recovery, F1, edit distances, ...) added, without mutating the input, and composes directly with
  `bootstrap_report` / `draw_distribution`. It hides the nested-dict wrapper that the lower-level
  `compute_derived_metrics` requires. Metrics are never computed inside the run (two-stage by design).
- `compute_derived_metrics` is also exported now (the in-place primitive `derive_metrics` wraps).
- **`srbf.analysis` -- the standardized results-page layer.** Turns raw run snapshots (tagged with
  model / benchmark / scaling coordinate) into the four standardized views: a model/baseline
  `leaderboard`, `scaling` curves, a `per_benchmark` breakdown, and per-expression `distribution`
  plots (all bootstrap-CI'd, honoring the unseeded-sources policy). `build_report(runs, out_dir, ...)`
  renders them to a Markdown page + PNG figures for the docs / github.io site. Figures live behind a
  new optional `[analysis]` extra (`pip install 'srbf[analysis]'`, matplotlib); tables/leaderboards
  need no extra.
- **`srbf.analysis.export_data(runs, path)`** + `RunResult.axis` / `RunResult.version`: export tidy
  aggregated records (per `series x benchmark x axis x x-value`, each with a metric's bootstrap
  median + CI and a provenance `version`) as JSON for the **interactive results page** -- a
  client-side Plotly explorer (pick x-axis / metric / benchmark / series, with CI bands). One dataset
  can carry several sweeps (compute / noise / n_support, ...) via the `axis` field.

### Changed
- **Baselines de-duplicated onto a shared `_RefiningBaselineModel` base.** `BruteForceModel` and
  `LampleChartonModel` now share the catalog handling, X/y coercion, per-candidate refine/score/build,
  and the fit loop (~250 duplicated lines removed); each subclass keeps only its skeleton source
  (exhaustive generator vs catalog sample) and extra `__init__` knobs. Public classes, constructors,
  and results are unchanged (golden-verified identical `_results` records + ordering).
- **`FlashANSRAdapter` validates the `complexity` mode at construction.** An unknown complexity string
  now raises `ValueError` immediately instead of failing lazily on the first problem (after the slow
  model load). Valid: `"none"`, `"ground_truth"`, or an int / float / list.
- **`NeSymReSAdapter` gained a `debug` flag (default `False`); its per-sample support/validation FVU
  print is now gated by it** (was printed unconditionally for every evaluated problem), matching the
  `E2EAdapter` convention.
- Re-pinned `flash-ansr>=0.10,<1.0` and `symbolic-data>=0.10` (coordinated family release).

### Performance
- **`bootstrapped_metric_ci` vectorizes** reducers that accept an `axis` kwarg (the default `np.nanmean`
  and friends), reducing all `n` resamples in one call instead of an `n`-iteration Python loop
  (`np.apply_along_axis`); falls back to the per-resample loop for metrics without `axis` support.
- **`bootstrap_report` shares a single valid-row scan** for both the distribution and its `n_rows`
  count (was scanned twice).
- **Variable-level F1 is derived from one precision + one recall computation** per row (was recomputing
  both a second time via `f1_score`); bit-identical to the previous values (same torch formula + NaN->0).

## [0.5.5] - 2026-07-01

### Changed
- Baselines (`LampleChartonModel`, `BruteForceModel`) read the fit attempts via the new public
  `Refiner.all_constants_values` property instead of the private `_all_constants_values`. Re-pinned
  `flash-ansr>=0.9.5` (which adds the property).

## [0.5.4] - 2026-07-01

### Fixed
- **Run provenance hashes the data catalog.** `provenance._resolve_inputs` now hashes a local catalog
  artifact (a saved catalog file or directory's `catalog.yaml`/`catalog.npz`), dropping the dead
  pre-0.5 `benchmark_path`/`dataset`/`skeleton_list` branches; a bare `name[@version]` / HF ref is
  captured verbatim by `config_sha`. Previously dataset provenance silently recorded nothing about the
  data source.

## [0.5.3] - 2026-07-01

Post-release audit cleanup + a robustness fix.

### Fixed
- **`LampleChartonModel._sample_skeletons` bounds its sampling loop** (`max(1000, samples*100)`
  attempts, then returns what it has with a warning) so it can no longer hang when the pool
  persistently fails to sample or its unique skeletons are exhausted.
- Baselines call the public `flash_ansr.scoring.compute_fvu` instead of reaching into a private
  method; `bootstrapped_metric_ci` return annotation + docstring corrected to the `(median, lower,
  upper)` tuple; clearer errors for missing/mistyped benchmark-config sections; dropped a dead
  `axis_of` parameter in `_substitute`; banned-term / stale-docstring cleanups.

## [0.5.2] - 2026-07-01

Post-release audit fixes (no API change).

### Fixed
- `BruteForceModel.fit` and `LampleChartonModel.fit` wrap their fit loop in `np.errstate(...)`, so the
  global numpy error state is restored even if the loop raises a non-`ConvergenceError` exception
  (previously a raise leaked `ignore` process-wide, silently suppressing later overflow/divide/invalid
  warnings).
- `fvu()` returns `inf` for empty/degenerate inputs (per its documented "invalid -> inf" contract)
  instead of raising `ValueError` on a zero-size reduction, and now accepts list / scalar `y_pred`.
- The `!sweep` YAML tag is registered on `import srbf`, so loading a sweep config (e.g. via the shared
  flash-ansr config loader) no longer raises `ConstructorError` without a prior `register_sweep_yaml()`.

## [0.5.1] - 2026-06-30

Adds the config-sweep + reporting layer (folded in from the planned 0.5.x scope) and finishes the
config/docs migration. Re-pinned `symbolic-data>=0.9`.

### Added
- **Inline `!sweep` config cross-products** (`srbf.Sweep` / `register_sweep_yaml` / `resolve_sweeps`):
  `!sweep [..]` = an anonymous grid axis; `!sweep {name: L, values: [..]}` = a named axis (co-named
  sweeps zip element-wise). `Benchmark.runs_from_config` expands an `experiments:` map and/or `!sweep`
  into per-run Benchmarks; `srbf run --sweep-filter AXIS=VALUE` selects runs.
- **Multi-draw reporting** `bootstrap_report` / `draw_distribution`: group per-problem metrics by
  `benchmark_eq_id` and bootstrap a CI (the no-seeding reproducibility story for sampling sources).

### Changed
- Migrated all `configs/evaluation/` configs to the catalog schema + `!sweep` (the 19-rung "choices
  ladder" collapses to one zipped axis); baselines resolve their catalog by name via
  `symbolic_data.build_catalog`; docs/README rewritten to the catalog/Benchmark/`!sweep` surface.
- `test_scaling_configs.py` upgraded from a key-presence check to a real catalog-schema gate.

## [0.5.0] - 2026-06-30

The data-layer redesign: srbf consumes `symbolic_data`'s catalog/`ProblemSource` API and
flash-ansr 0.9's public inference API, and the `eval/` engine layer collapses into a single
`Benchmark` driver. Breaking. Re-pinned `flash-ansr~=0.9`, `symbolic-data>=0.8`.

### Changed
- **Data source** is now always a `symbolic_data` catalog. The old `SkeletonDatasetSource` +
  `FastSRBSource` (and the `type: skeleton_dataset` / `type: fastsrb` config split) are replaced by
  one `CatalogSource` wrapping a `symbolic_data.ProblemSource`. The `data_source` config is
  `{catalog: <name/ref>, sampling: {n_support, n_validation, noise, problems_per_expression},
  target_size}`; the frozen sha-pinned `v23-val` catalog is the drift-safe validation set (the old
  val100 skeleton-pin machinery is gone). Non-flash_ansr adapters now require an explicit
  `model_adapter.simplipy_engine` (no dataset to borrow one from).
- **`FlashANSRAdapter`** is a thin mapper over `FlashANSR.infer()` -> `InferenceResult` (best
  candidate + the full classified `CandidateLedger`); no more `model._results` / `nth_best_beam` /
  generate-refine-phase scraping. The candidate-ledger JOIN lives in flash-ansr now; srbf only
  persists it (`CandidateStoreWriter`).
- **`Benchmark`** (`srbf.Benchmark`) replaces the `Evaluation*` driver surface. `Benchmark.run` is a
  plain serial loop (no cross-problem overlap; per-problem timing stays uncontended).
  `Benchmark.from_config` absorbs `build_evaluation_run` (resume/limit/completed math), building the
  model adapter last so a resumed sweep never reloads the model for a finished experiment.
- Baselines moved onto `symbolic_data.LampleChartonCatalog`: `SkeletonPoolModel` ->
  `LampleChartonModel` (adapter type `skeleton_pool` -> `lample_charton`; param `skeleton_pool` ->
  `catalog`).
- Relocated the package: dropped the `srbf/eval/` subpackage (modules moved to top-level `srbf/`;
  `result_store` -> `store`). `import srbf.eval.X` -> `import srbf.X`.

### Removed
- `EvaluationEngine` / `OverlappedEvaluationEngine` (the cross-problem overlap), `Evaluation`,
  `EvaluationRunPlan` / `build_evaluation_run` (-> `Benchmark.from_config`), `run_config.py`,
  `data_sources.SkeletonDatasetSource` / `FastSRBSource`, and srbf's duplicate
  `build_candidate_ledger` + `FIT_*` (imported from `flash_ansr.inference` now). The
  `srbf.benchmarks` re-export shim (its `symbolic_data.FastSRBBenchmark` source was removed upstream;
  FastSRB is the `fastsrb` catalog).

### Deferred (to 0.5.1)
- Inline `!sweep` config cross-products + multi-draw bootstrap reporting (the `experiments:` map
  still works), the columnar (Parquet) result store + typed `Result`/`ResultCollection` projection
  (the pickle `ResultStore` is unchanged behind its seam), and migrating the `configs/evaluation/`
  scaling configs to the new catalog schema (cheaper bundled with the `!sweep` collapse). Configs are
  not shipped in the wheel.

## [0.4.0] - 2026-06-29

### Changed
- Dropped srbf's local `FastSRBBenchmark` fork; `srbf.benchmarks` now re-exports
  `symbolic_data.FastSRBBenchmark` (single source of the FastSRB sampler), and the eval consumers
  import it directly from `symbolic_data`. Re-pinned `flash-ansr~=0.8`.

### Removed
- The `srbf.benchmarks.fastsrb` module (minor breaking for deep imports; the class remains importable
  as `srbf.benchmarks.FastSRBBenchmark` / `symbolic_data.FastSRBBenchmark`).

## [0.3.0] - 2026-06-28

### Changed
- `SkeletonDatasetSource` delegates per-skeleton dataset sampling to
  `symbolic_data.sample_from_skeleton` (behavior-identical, byte-verified), and a flaky
  skeleton-pin test was made deterministic.

## [0.2.0] - 2026-06-28

### Changed
- Imports updated to the carved `symbolic_data` / `simplipy` packages (flash-ansr 0.7 carve).

## [0.1.0] - 2026-06-26

Initial release: the Symbolic Regression Benchmark Framework carved out of
[flash-ansr](https://github.com/psaegert/flash-ansr) (flash-ansr 0.6 split).

### Added
- Evaluation engine (`srbf.eval.engine`) and model-agnostic protocols (`srbf.eval.core`:
  `EvaluationModelAdapter`, `EvaluationDataSource`, `EvaluationSample`, `EvaluationResult`).
- `srbf run` CLI: run an evaluation from a unified config.
- Built-in model adapters: `flash_ansr`, `pysr`, `nesymres`, `e2e`, `skeleton_pool`, `brute_force`.
- Benchmarks (FastSRB) and metrics (FVU, symbolic recovery, token/edit distance).
- Baseline provisioning recipes: `scripts/patch_{nesymres,symbolicregression,typing_io}.py`.
- Documentation under `docs/`: running, benchmarks, models, and the adapter contribution guide.

### Notes
- Depends one-way on `flash-ansr` (>=0.6) and `simplipy`; `flash-ansr` never imports `srbf`.
- Status: cleanly-carved eval, not yet a general framework. A plugin `register_adapter()` entry-point
  and raw `(X, y)` dataset ingestion are planned follow-ons.
