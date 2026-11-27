"""Averages over benchmarks > subsets > problems (srbf.aggregate.combine_nested): three-level random effects.

The structure is fixed in advance (a problem's subset, else its catalog); the two variance components, between
benchmarks and between the subsets of one benchmark, are estimated for every average and only set the weights."""
import math
import time

import numpy as np
import pytest
from scipy import stats

from srbf import aggregate as ag


def _set(n, mean, var):
    """A subset of n problems with this mean and (sample) variance."""
    return ag.SetStat(n, n * mean, (n - 1) * var + n * mean * mean)


# ---- one subset per benchmark: the two-level average ------------------------------------------------------------------
@pytest.mark.parametrize("scale", ["normal", "logit"])
def test_benchmarks_of_one_subset_are_the_two_level_average(scale):
    rng = np.random.default_rng(3)
    for _ in range(20):
        sets = []
        for n in rng.choice([1, 2, 5, 40, 300], int(rng.choice([1, 2, 3, 8, 29]))):
            if scale == "logit":
                k = float(rng.binomial(n, rng.uniform()))
                sets.append(ag.SetStat(int(n), k, k))
            else:
                x = rng.normal(rng.normal(0, 2), 1.5, n)
                sets.append(ag.SetStat(int(n), float(x.sum()), float((x * x).sum())))
        a, b = ag.combine(sets, scale), ag.combine_nested([[s] for s in sets], scale)
        assert (a.mu, a.lo, a.hi, a.pi_lo, a.pi_hi, a.tau2, a.weights, a.p) == (b.mu, b.lo, b.hi, b.pi_lo, b.pi_hi, b.tau2, b.weights, b.p)
        assert b.tau2_within == 0.0 and b.subset_weights == [[w] for w in b.weights]


# ---- small cases by hand ----------------------------------------------------------------------------------------------
def test_means_by_hand():
    # benchmark A: two subsets of 10 problems with means 1 and 3 (variance 1: v = 0.1 each); benchmark B: one subset of
    # 5 problems with mean 5 (v = 0.2)
    groups = [[_set(10, 1.0, 1.0), _set(10, 3.0, 1.0)], [_set(5, 5.0, 1.0)]]
    c = ag.combine_nested(groups, "normal")
    # within A: Q(t) = 2 / (0.1 + t) = 1 subset's degree of freedom -> tau_S^2 = 1.9; A's estimate 2 with variance 1
    assert c.tau2_within == pytest.approx(1.9, rel=1e-12)
    # B: 5 with variance 0.2 + 1.9; two benchmarks: 9 / (1 + 2.1 + 2 tau_B^2) = 1 -> tau_B^2 = 2.95
    assert c.tau2 == pytest.approx(2.95, rel=1e-12)
    wa, wb = 1 / (1.0 + 2.95), 1 / (2.1 + 2.95)
    mu = (2 * wa + 5 * wb) / (wa + wb)
    se = math.sqrt(wa ** 2 * (0.25 * 0.1 + 0.25 * 0.1) + wb ** 2 * 0.2) / (wa + wb)   # only the problems' sampling
    t = stats.t.ppf(0.975, 9 + 9 + 4)
    assert (c.mu, c.lo, c.hi) == pytest.approx((mu, mu - t * se, mu + t * se), rel=1e-9)
    assert c.weights == pytest.approx([wa / (wa + wb), wb / (wa + wb)], rel=1e-12)
    assert sum(c.subset_weights, []) == pytest.approx([wa / (wa + wb) / 2] * 2 + [wb / (wa + wb)], rel=1e-12)
    assert c.pi_lo is None and c.S == 2


def test_rates_by_hand():
    # benchmark A: 2 of 10 and 8 of 10; benchmark B: 5 of 10. Within A, rate p = 0.5 whatever rho, and Pearson's
    # chi-square is c 10 (0.3^2 + 0.3^2) / 0.25 = 7.2 c with c = 1 / (1 + 9 rho): one degree of freedom -> c = 1 / 7.2
    groups = [[ag.SetStat(10, 2.0, 2.0), ag.SetStat(10, 8.0, 8.0)], [ag.SetStat(10, 5.0, 5.0)]]
    rho = ag.within_rho(groups)
    assert rho == pytest.approx(6.2 / 9, rel=1e-12)
    eff = ag.effective_sets(groups, rho)
    assert [x for e in eff for x in (e.n, e.s1)] == pytest.approx([20 / 7.2, 10 / 7.2, 10 / 7.2, 5 / 7.2], rel=1e-12)
    c = ag.combine_nested(groups, "logit")
    want = ag.combine_rates(eff, df=27, rho_within=rho)
    assert (c.mu, c.lo, c.hi) == (want.mu, want.lo, want.hi)
    assert c.mu == pytest.approx(0.5, abs=1e-12) and c.tau2_within == pytest.approx(6.2 / 2.8, rel=1e-12)
    assert sum(c.subset_weights, []) == pytest.approx([c.weights[0] / 2] * 2 + [c.weights[1]], rel=1e-12)


def test_a_benchmark_whose_subsets_are_all_at_zero_says_nothing_about_their_spread():
    assert ag.within_rho([[ag.SetStat(10, 0.0, 0.0), ag.SetStat(30, 0.0, 0.0)], [ag.SetStat(5, 1.0, 1.0)]]) == 0.0
    assert ag.within_rho([[ag.SetStat(10, 10.0, 10.0), ag.SetStat(3, 3.0, 3.0)]]) == 0.0
    # subsets at opposite ends: each counts as one problem (rho = 1)
    assert ag.within_rho([[ag.SetStat(50, 0.0, 0.0), ag.SetStat(50, 50.0, 50.0)]]) == 1.0


# ---- the weights: subsets that are alike collapse, subsets that differ count on their own ---------------------------
def test_alike_subsets_collapse_into_their_benchmark():
    rng = np.random.default_rng(11)
    vals = list(rng.normal(0.0, 1.0, 30))
    same = [ag.SetStat(30, float(sum(vals)), float(sum(v * v for v in vals)))] * 3       # three copies of one subset
    others = [[_set(30, mu, 1.0)] for mu in (-2.0, 1.0, 2.5, 4.0)]
    c = ag.combine_nested([same] + others, "normal")
    assert c.tau2_within == 0.0
    merged = ag.combine([ag.SetStat(90, 3 * same[0].s1, 3 * same[0].s2)] + [g[0] for g in others], "normal")
    assert c.mu == pytest.approx(merged.mu, abs=0.01) and c.weights[0] == pytest.approx(merged.weights[0], rel=0.02)
    assert c.weights[0] < 1.2 * max(c.weights[1:])     # one benchmark's worth, not three
    # rates: alike subsets (rho = 0) are their merged counts exactly; only t's degrees of freedom, sum (n - 1) over
    # the subsets, are two fewer than for one merged set
    rates = [[ag.SetStat(40, 12.0, 12.0), ag.SetStat(60, 18.0, 18.0), ag.SetStat(20, 6.0, 6.0)],
             [ag.SetStat(30, 3.0, 3.0)], [ag.SetStat(25, 20.0, 20.0)], [ag.SetStat(50, 40.0, 40.0)]]
    assert ag.within_rho(rates) == 0.0
    merged = [ag.SetStat(120, 36.0, 36.0)] + [g[0] for g in rates[1:]]
    r, m = ag.combine_nested(rates, "logit"), ag.combine_rates(merged, df=sum(s.n - 1 for g in rates for s in g))
    assert (r.mu, r.lo, r.hi, r.pi_lo, r.pi_hi, r.weights) == (m.mu, m.lo, m.hi, m.pi_lo, m.pi_hi, m.weights)
    assert (r.lo, r.hi) == pytest.approx((ag.combine(merged, "logit").lo, ag.combine(merged, "logit").hi), rel=1e-4)


def test_subsets_that_differ_count_on_their_own():
    # benchmark X: four subsets far apart; four benchmarks of one subset close together. Between benchmarks there is
    # little spread, within X a lot: X counts about as much as four benchmarks.
    x = [_set(50, mu, 1.0) for mu in (-6.0, -2.0, 2.0, 6.0)]
    others = [[_set(50, mu, 1.0)] for mu in (-0.2, 0.0, 0.1, 0.3)]
    c = ag.combine_nested([x] + others, "normal")
    assert c.tau2_within > 10 * c.tau2
    assert 3.0 < c.weights[0] / np.mean(c.weights[1:]) < 4.2
    # the same for rates: four subsets of X far apart (5 % to 95 %), four benchmarks close to 50 %
    rx = [ag.SetStat(60, k, k) for k in (3.0, 15.0, 45.0, 57.0)]
    ro = [[ag.SetStat(60, k, k)] for k in (27.0, 30.0, 31.0, 33.0)]
    r = ag.combine_nested([rx] + ro, "logit")
    assert r.tau2_within > 3 * r.tau2
    assert 2.5 < r.weights[0] / np.mean(r.weights[1:]) < 4.2
    # and four alike subsets of a benchmark among benchmarks that differ: about one benchmark's worth
    alike = [ag.SetStat(60, k, k) for k in (29.0, 30.0, 31.0, 30.0)]
    spread = [[ag.SetStat(60, k, k)] for k in (6.0, 24.0, 40.0, 54.0)]
    a = ag.combine_nested([alike] + spread, "logit")
    assert a.tau2_within == 0.0 and a.weights[0] / np.mean(a.weights[1:]) < 1.5


# ---- coverage by simulation -------------------------------------------------------------------------------------------
_SIZES = [[n] for n in (5301, 200, 150, 120, 100, 90, 33, 25, 20, 15, 12, 10, 8, 6, 2, 1)] + \
    [[40, 12], [200, 30, 8], [25, 25], [60, 6, 3, 100]]


def test_the_mean_interval_covers_these_subsets():
    """95 % of intervals hold the weighted average of these subsets' true means (the weights the interval used)."""
    rng = np.random.default_rng(20261127)
    hits = total = 0
    for tb, ts in ((1.0, 0.0), (1.0, 1.0), (0.2, 1.0)):
        truth = [[rng.normal(ub, ts) for _ in g] for g, ub in ((g, rng.normal(0, tb)) for g in _SIZES)]
        for _ in range(60):
            groups = [[ag.SetStat(n, float(x.sum()), float((x * x).sum())) for n, x in
                       ((n, rng.normal(mu, 2.0, n)) for n, mu in zip(g, mus))] for g, mus in zip(_SIZES, truth)]
            c = ag.combine_nested(groups, "normal")
            target = sum(w * mu for ws, mus in zip(c.subset_weights, truth) for w, mu in zip(ws, mus))
            hits += c.lo <= target <= c.hi
            total += 1
    assert hits / total >= 0.9


@pytest.mark.parametrize("m", [0.003, 0.3, 0.997])
def test_the_rate_interval_covers_these_subsets_near_0_and_100_percent_too(m):
    """The score-test interval against what the average estimates for these subsets: the beta-binomial fit to all the
    redraws' effective counts (the expected likelihood's maximum, White 1982)."""
    rng = np.random.default_rng(int(1000 * m) + 7)
    rates = []
    for g in _SIZES:
        pb = rng.beta(m * 3.0, (1 - m) * 3.0)
        rates.append([rng.beta(max(pb * 3.0, 1e-300), max((1 - pb) * 3.0, 1e-300)) for _ in g])
    fits, effs = [], []
    for _ in range(40):
        groups = [[ag.SetStat(n, float(k), float(k)) for n, k in zip(g, rng.binomial(g, ps))] for g, ps in zip(_SIZES, rates)]
        c = ag.combine_nested(groups, "logit")
        fits.append((c.lo, c.hi))
        effs += ag.effective_sets(groups, ag.within_rho(groups))
        assert 0.0 <= c.lo <= c.mu <= c.hi <= 1.0
    eta, _ = ag.rate_fit(effs)
    target = 0.0 if eta == -math.inf else 1.0 if eta == math.inf else 1 / (1 + math.exp(-eta))
    assert sum(lo <= target <= hi for lo, hi in fits) / len(fits) >= 0.85


# ---- speed at the release's size --------------------------------------------------------------------------------------
def test_an_average_over_the_releases_benchmarks_is_quick():
    """145 benchmarks of 255 subsets: a mean in well under a second, a rate (most of it the interval) in a few."""
    rng = np.random.default_rng(5)
    counts = [1] * 95 + [2] * 25 + [3] * 12 + [4] * 8 + [6] * 3 + [12] * 2
    sizes = [[int(round(10 ** rng.uniform(0, 3))) for _ in range(c)] for c in counts]
    assert (len(sizes), sum(map(len, sizes))) == (145, 255)
    means = [[_set(n, rng.normal(0, 1), 1.0) if n > 1 else ag.SetStat(1, 0.5, 0.25) for n in g] for g in sizes]
    rates = []
    for g in sizes:
        pb = rng.beta(0.6, 2.4)
        rates.append([ag.SetStat(n, float(k), float(k)) for n, k in zip(g, rng.binomial(g, rng.beta(pb * 3 + 1e-9, (1 - pb) * 3 + 1e-9, len(g))))])
    t0 = time.perf_counter()
    ag.combine_nested(means, "normal")
    t1 = time.perf_counter()
    c = ag.combine_nested(rates, "logit")
    t2 = time.perf_counter()
    assert c.lo < c.mu < c.hi
    assert t1 - t0 < 0.5 and t2 - t1 < 5.0
