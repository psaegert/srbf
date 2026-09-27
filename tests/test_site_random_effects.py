"""The random-effects average the results site uses (owner 2026-09-27): problem sets combined with weights
1 / (tau^2 + noise), tau^2 by Paule-Mandel, rates on the logit scale, and an interval over these problem sets (the
weights held fixed, only the problems' sampling within each set). The reference tau^2 and means come from statsmodels'
combine_effects(method_re="pm") on the same inputs (checked 2026-09-27)."""
import importlib.util
import os
import sys

import numpy as np
import pytest
from scipy import stats

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


def test_one_set_is_its_own_mean_with_a_t_interval_over_its_problems():
    c = re_.combine([re_.SetStat(20, 9.0, 9.0)], "logit")   # 9 of 20 problems recovered
    assert c.S == 1 and c.mu == pytest.approx((9 + 0.5) / 21)
    assert c.lo < c.mu < c.hi and c.pi_lo is None


def test_sets_without_spread_of_their_own_borrow_one():
    # every problem recovered in every set: rates take their continuity-corrected binomial variance, never zero
    c = re_.combine([re_.SetStat(10, 10.0, 10.0), re_.SetStat(30, 30.0, 30.0), re_.SetStat(1, 1.0, 1.0)], "logit")
    assert 0.9 < c.mu < 1.0 and c.lo < c.mu
    # a continuous metric with one problem per set borrows the pooled spread of the others
    y, v = re_.set_estimates([re_.SetStat(1, 3.0, 9.0), re_.SetStat(3, 3.0, 5.0)], "normal")
    assert v[0] == pytest.approx(1.0) and v[1] == pytest.approx(1.0 / 3)


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
