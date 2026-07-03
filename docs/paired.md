# Paired comparisons

Every model evaluated by srbf runs on the *same* benchmark expressions. The paired layer
(`srbf.reporting`) exploits that: instead of comparing two models' separate averages — whose
confidence intervals are inflated by shared expression difficulty (hard expressions are hard for
everyone) — it computes the difference **per expression** and asks whether those differences are
consistently one-sided. Formally, Var(Δ) = Var(A) + Var(B) − 2·Cov(A, B), and the covariance is
large precisely because difficulty is shared: pairing harvests it. Never subtract two marginal
confidence intervals; that is the anti-pattern this module exists to replace.

## Quickstart

```python
from srbf.reporting import paired_report, self_noise, pair_margin, draw_values

# snapshot_a / snapshot_b: two dict-of-lists result snapshots of the SAME benchmark
values_a = draw_values(snapshot_a, "numeric_recovery_val")
values_b = draw_values(snapshot_b, "numeric_recovery_val")
margin = pair_margin(self_noise(values_a), self_noise(values_b))

report = paired_report(
    snapshot_a, snapshot_b, "numeric_recovery_val",
    higher_is_better=True, margin=margin,
    allow_unverified=True,   # required for snapshots without embedded provenance
)
print(report["delta_mean"], (report["ci_lower"], report["ci_upper"]), report["verdict"])
```

`report` is a plain dict: `delta_mean` + CI (bootstrap over expressions), `delta_median` + CI,
`win_rate` / `prob_superiority` (+ CI), a Wilcoxon signed-rank companion with
`zero_method='pratt'` (zeros counted — on rate metrics ties are the majority and dropping them
silently changes the test), the two-sided bootstrap `p_value` on the mean delta (the primary
inference), `mde_80`, a `variance_decomposition`, pairing diagnostics, and — given a `margin` —
a four-state `verdict`.

## Verdicts and the measurement-noise margin

A verdict compares the CI against a **measurement-noise margin** (the minimum resolvable
difference): the largest aggregate
difference that could plausibly appear when comparing two *equally good* models with the two
series' own draw-to-draw noise. Margins are **derived, not hand-picked**
(`scripts/derive_noise_margins.py`): each series' noise null comes from comparing the series to
itself across random splits of its repeated draws (exact per-expression rescaling to the
full-draw scale; a centered draw-bootstrap cross-check must agree), and the margin for a pair
convolves the two nulls — margins are **pair-specific** by construction.

| verdict | meaning |
|---|---|
| `better` / `worse` | CI entirely above +margin (better) or entirely below −margin (worse): a difference larger than benchmark noise explains |
| `equivalent` | CI entirely inside ±margin: any residual difference is smaller than the benchmark can measure (**measurement**-equivalence, not proof of identity) |
| `undecided` | neither: not enough data — read `mde_80`, the smallest true effect the test would detect with 80% power at this sample size ((z₀.₉₇₅ + z₀.₈₀)·SE) |

`equivalence_attainable` reports whether `equivalent` was even reachable at this sample size;
when it is `False`, an `undecided` is resolution-limited, not evidence of parity. Validation:
the near-replicate pair v23.0-120M vs its KV-cache decode re-evaluation (identical weights,
different decode path) reads `equivalent` on both FastSRB cells; on the smaller v23-val
benchmark the same pair reads `undecided` with `equivalence_attainable=False` — v23-val's
paired sample is too small for `equivalent` to be reachable at all there, which is exactly what
the flag reports, and the point estimates sit well inside the margin, consistent with no real
difference.

## Missing data, honestly

Metric columns carry srbf's two-regime failure encoding, and pairing follows it: **rate**
metrics score failures as 0.0 and pair over (almost) all expressions; **diagnostic** metrics
drop failures, so their deltas pair over the both-models-succeeded intersection — a conditional
estimand that every report discloses via `n_only_a` / `n_only_b` (never silently).
`worst_rank=True` additionally reports the rank statistics over the *union* of expressions with
one-sided failures imputed as sign-only sentinels (worst-rank composite scoring, cf. Lachin
1999) — sound for ranks, never applied to the mean; degenerate (≥50% imputed) blocks
suppress themselves and point to the success rate instead.

## The pairing contract

`paired_report` refuses meaningless joins: expressions are joined on ids
(`benchmark_eq_id`, falling back to the ground-truth skeleton), never row order; snapshots with
embedded provenance are checked for identical benchmark-data hashes (`PairingContractError` on
mismatch); snapshots without provenance require an explicit `allow_unverified=True`. Pairing is
**strictly per benchmark** — there is deliberately no combined cross-benchmark number.

## Δ over the compute axis

On the results site, verdicts are issued at **standardized compute budgets** (≤1, 10, 100,
1000 s per expression, median; user-selectable): within the budget, each method's best measured
configuration, with same-method pairs additionally on the same configuration (one factor
varies). Correction families are per budget (`family_id` carries `t`), so the budget grid is
pre-declared, not cherry-picked. The **Table view** (Table × Absolute) runs on the same budget
grid and the same best-within-budget selection — per benchmark, each series' most expensive
usable configuration whose *median* fit time is ≤ t (the budget caps a configuration's median
cost, never per-expression equal budgets) — and shows that configuration's *marginal* value and
95% CI, numerically identical to its point on the Curves view (an exporter self-check enforces
this). Those rows are marginal: never difference two of them and never subtract their CIs —
that is exactly the anti-pattern from the top of this page; head-to-head questions belong to
the Paired views.

`paired_delta_curve(series_a, series_b, metric, x_policy=...)` compares two rung ladders:
`'rung'` matches identical configurations (same-method variants — ablation vs parent, size
ladder, versions); `'time'` interpolates per expression, linearly in log wall-clock time,
between measured rungs — never extrapolating (out-of-range points are returned loudly), with a
composition guard (an expression contributes at *t* only if valid at both bracketing rungs of
both series) and per-point `n_pairs`. Bands are pointwise; x positions are series-median costs
treated as fixed design points. The estimand of Δ(t) is "configurations whose median cost is
t", not per-expression equal budgets.

## Statistical model (why no further corrections)

The exchangeable unit is the **expression**; the ~10 draws inside an expression are
independently sampled problem instances, collapsed to a per-expression mean before resampling.
Bootstrapping expressions after collapsing is exactly the cluster bootstrap for the mean.
Nadeau–Bengio / Dietterich variance corrections do not apply: no data is reused across the
resampled units (they address overlapping training sets across CV folds). For rank statistics,
`hierarchical=True` propagates draw-level noise via a two-stage bootstrap. Multiple
comparisons are corrected at the *claim* layer — Holm within a declared confirmatory family,
Benjamini–Hochberg for exploratory matrices — and every corrected p-value is recorded together
with its `family_id`, family size, and method, so corrections remain auditable and recomputable
when the model roster changes.

The interactive rendering of all of this lives on the
[results explorer](https://psaegert.github.io/srbf/), whose views form a 2×2 grid — Display
(Curves | Table) × Values (Absolute | Paired). The Paired values render in both displays: Δ(t)
curves over the compute axis and the verdict matrix at a chosen budget; the Absolute Table
shares the budget machinery but shows marginal per-series values, never to be differenced. The
explorer's [Paired comparisons](https://psaegert.github.io/srbf/#paired) section explains the
same ideas for readers without a statistics background.
