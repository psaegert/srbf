"""The random-effects average the results site uses (owner 2026-09-27): problem sets combined with weights
1 / (tau^2 + noise), tau^2 by Paule-Mandel, and an interval over these problem sets (the weights held fixed, only the
problems' sampling within each set). The reference tau^2 and means come from statsmodels' combine_effects(method_re="pm")
on the same inputs (checked 2026-09-27). Rates by the beta-binomial model (owner 2026-11-17), checked against scipy's
beta-binomial likelihood and a generic optimiser."""
import importlib.util
import os
import sys

import numpy as np
import pytest
from scipy import optimize, special, stats

_SPEC = importlib.util.spec_from_file_location(
    "site_random_effects", os.path.join(os.path.dirname(__file__), "..", "scripts", "site_random_effects.py"))
re_ = importlib.util.module_from_spec(_SPEC)
sys.modules["site_random_effects"] = re_   # its dataclasses look themselves up there
_SPEC.loader.exec_module(re_)


@pytest.mark.parametrize("df", [1, 2, 5, 28, 200])
def test_the_t_distribution_is_students(df):
    assert re_.t_ppf(0.975, df) == pytest.approx(stats.t.ppf(0.975, df), abs=1e-9)
    for t in (-3.1, 0.2, 1.7):
        assert re_.t_cdf(t, df) == pytest.approx(stats.t.cdf(t, df), abs=1e-12)


@pytest.mark.parametrize("seed, tau2, mu", [
    (3, 0.064344, 0.268979),
    (7, 0.000000, 0.295731),     # the sets agree: tau^2 = 0, and the average is the precision-weighted one
    (11, 0.021327, 0.336120),
])
def test_paule_mandel_matches_statsmodels_and_the_interval_is_over_these_sets(seed, tau2, mu):
    rng = np.random.default_rng(seed)
    y, v = rng.normal(0.3, 0.2, 12), rng.uniform(0.001, 0.05, 12)
    c = re_.combine_scale(list(y), list(v), df=500)
    assert (c.tau2, c.mu) == pytest.approx((tau2, mu), abs=1e-5)
    assert sum(c.weights) == pytest.approx(1.0)
    w = 1 / (np.asarray(v) + c.tau2)
    se = np.sqrt((w ** 2 * v).sum()) / w.sum()               # the weights held fixed: only the sets' own noise
    assert (c.lo, c.hi) == pytest.approx((mu - stats.t.ppf(0.975, 500) * se, mu + stats.t.ppf(0.975, 500) * se), abs=1e-5)


def test_the_interval_covers_these_problem_sets_and_the_range_one_more():
    # three large sets that disagree: their average is known precisely, a fourth set could land far from it
    sets = [re_.SetStat(5000, 1000.0, 1000.0), re_.SetStat(4000, 2000.0, 2000.0), re_.SetStat(3000, 2400.0, 2400.0)]
    c = re_.combine(sets, "normal")
    assert c.hi - c.lo < 0.02
    assert c.pi_hi - c.pi_lo > 1.0


def test_a_large_set_counts_about_as_much_as_a_medium_one_and_a_tiny_set_less():
    # three sets that really differ, of 5,000, 200 and 1 problems
    sets = [re_.SetStat(5000, 2500.0, 2500.0), re_.SetStat(200, 30.0, 30.0), re_.SetStat(1, 1.0, 1.0)]
    c = re_.combine(sets, "logit")
    big, mid, one = c.weights
    assert big == pytest.approx(mid, rel=0.05)              # one set's worth each, whatever its size
    assert one < 0.5 * mid                                  # a single problem says little about its set
    assert 0.0 < c.lo < c.mu < c.hi < 1.0                   # a rate stays a rate
    assert c.pi_lo is not None and c.pi_lo < c.lo and c.pi_hi > c.hi


def wilson_ci(k, n):
    ci = stats.binomtest(int(k), int(n)).proportion_ci(0.95, "wilson")
    return ci.low, ci.high


def test_one_set_is_its_plain_rate_with_the_wilson_interval():
    c = re_.combine([re_.SetStat(20, 9.0, 9.0)], "logit")   # 9 of 20 problems recovered
    lo, hi = wilson_ci(9, 20)
    assert c.S == 1 and c.mu == pytest.approx(9 / 20) and (c.lo, c.hi) == pytest.approx((lo, hi), abs=1e-12)
    assert c.pi_lo is None


@pytest.mark.parametrize("k, n", [(8.0, 8), (0.0, 8), (1.0, 1), (0.0, 1)])
def test_a_set_at_0_or_100_percent_is_shown_as_it_is(k, n):
    # the logit of an observed rate needed (k + 0.5) / (n + 1), which showed 8 of 8 as 94.4 % and 1 of 1 as 75 %
    c = re_.combine([re_.SetStat(n, k, k)], "logit")
    assert c.mu == k / n
    lo, hi = wilson_ci(k, n)
    assert (c.lo, c.hi) == pytest.approx((lo, hi), abs=1e-12)


def test_sets_without_spread_of_their_own():
    # every problem recovered in every set: 100 %, with the Wilson interval over all their problems
    c = re_.combine([re_.SetStat(10, 10.0, 10.0), re_.SetStat(30, 30.0, 30.0), re_.SetStat(1, 1.0, 1.0)], "logit")
    assert c.mu == 1.0 and c.lo == pytest.approx(wilson_ci(41, 41)[0], abs=1e-12)
    # a continuous metric with one problem per set borrows the pooled spread of the others
    y, v = re_.set_estimates([re_.SetStat(1, 3.0, 9.0), re_.SetStat(3, 3.0, 5.0)], "normal")
    assert v[0] == pytest.approx(1.0) and v[1] == pytest.approx(1.0 / 3)


def test_sets_that_agree_give_the_pooled_rate():
    # no spread beyond the binomial: the mean rate is the pooled one, and inverting the estimating equation, which is then
    # the binomial score K - N m, gives the pooled rate plus or minus t sqrt(p (1 - p) / N)
    c = re_.combine([re_.SetStat(100, 30.0, 30.0)] * 3, "logit")
    assert c.mu == pytest.approx(0.3, abs=1e-12) and c.tau2 == 0.0
    half = re_.t_ppf(0.975, 297) * np.sqrt(0.3 * 0.7 / 300)   # (taken at s = e^15, the cap, not at s = inf)
    assert (c.lo, c.hi) == pytest.approx((0.3 - half, 0.3 + half), abs=1e-6)


def test_one_success_among_sets_at_zero_keeps_a_narrow_interval():
    # 28 sets at 0 % and one problem of 200 solved in one more: the linearised (delta-method) interval on the log-odds
    # reached 0 % to 88 % here; inverting the estimating equation keeps it under 1 %
    sizes = [5301, 200, 200, 150, 120, 120, 100, 90, 33, 25, 20, 15, 12, 12, 10, 8, 8, 8, 6, 6, 6, 2, 2, 2, 1, 1, 1, 1, 1]
    sets = [re_.SetStat(n, 1.0 if i == 1 else 0.0, 1.0 if i == 1 else 0.0) for i, n in enumerate(sizes)]
    c = re_.combine(sets, "logit")
    assert 0.0 <= c.lo <= c.mu < c.hi < 0.01


def test_two_sets_at_opposite_ends_say_half_and_little_more():
    c = re_.combine([re_.SetStat(1, 1.0, 1.0), re_.SetStat(8, 0.0, 0.0)], "logit")
    assert c.mu == pytest.approx(0.5, abs=1e-6) and c.lo < 0.01 and c.hi > 0.99


@pytest.mark.parametrize("x", [1e-9, 1e-4, 0.3, 1.0, 2.5, 9.99, 10.0, 123.4, 1e4, 3.3e6])
def test_digamma_and_trigamma(x):
    assert re_.digamma(x) == pytest.approx(special.digamma(x), rel=1e-12, abs=1e-12)
    assert re_.trigamma(x) == pytest.approx(special.polygamma(1, x), rel=1e-11)


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6])
def test_the_rate_is_the_beta_binomial_maximum_likelihood(seed):
    rng = np.random.default_rng(seed)
    S = int(rng.choice([3, 8, 29]))
    n = rng.choice([1, 2, 5, 20, 100, 1024], S)
    k = rng.binomial(n, 1 / (1 + np.exp(-rng.normal(rng.normal(0, 1.5), rng.choice([0.3, 1.0, 2.5]), S))))
    if k.sum() == 0 or (k == n).all():
        k[0], n[0] = 1, max(2, n[0])
    c = re_.combine([re_.SetStat(int(a), float(b), float(b)) for a, b in zip(n, k)], "logit")

    def nll(x):
        m, s = 1 / (1 + np.exp(-x[0])), np.exp(min(x[1], re_.LAM_MAX))
        return -stats.betabinom.logpmf(k, n, m * s, (1 - m) * s).sum()
    eta = np.log(c.mu / (1 - c.mu))
    lam = np.log(1 / c.tau2) if c.tau2 > 0 else re_.LAM_MAX
    best = min((optimize.minimize(nll, x0, method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000})
                for x0 in ([eta, lam], [0.0, 0.0], [eta, 5.0])), key=lambda r: r.fun)
    assert nll([eta, lam]) <= best.fun + 1e-7
    assert c.lo < c.mu < c.hi


def test_holm():
    assert re_.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_the_sites_parity_cases_are_current():
    """The explorer's own averaging is checked against these cases (site_v2.spec.mjs); they must be the reference's."""
    import json
    _FIX = importlib.util.spec_from_file_location(
        "site_random_effects_fixture", os.path.join(os.path.dirname(__file__), "..", "scripts", "site_random_effects_fixture.py"))
    fixture = importlib.util.module_from_spec(_FIX)
    _FIX.loader.exec_module(fixture)
    with open(fixture.OUT) as fh:
        assert json.load(fh) == json.loads(json.dumps(fixture.cases())), "regenerate: python scripts/site_random_effects_fixture.py"
