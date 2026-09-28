"""Random-effects averaging over problem sets: the reference for the results site (owner 2026-09-27).

The results are organised as problem sets > problems > runs. A problem's value is the mean of its runs. A set is
summarised by the number of its problems with a value, their mean and their variance. The sets are then combined by a
random-effects model: every set has a true mean, the true means scatter around an overall mean mu with spread tau, and a
set's observed mean also carries its own sampling noise (its variance over its problems, divided by their number).
mu is estimated with weights 1 / (tau^2 + noise), tau^2 by Paule-Mandel. The interval covers these problem sets (owner
2026-09-27: a score is a claim about srbf's suite): the weights are held fixed and only the sampling of problems within
each set counts, Var(mu) = sum w^2 noise / (sum w)^2, with t on the problems' degrees of freedom, sum (n - 1). How far
single problem sets spread is the prediction interval for one more set, t with S - 2 (Higgins et al. 2009).
Rates are combined on the logit scale, with a continuity-corrected rate (x n + 0.5) / (n + 1).

results-site/explorer_v2.js implements the same arithmetic, step for step (the t distribution below is the same
continued fraction), so the two agree to rounding; tests/test_site_random_effects.py and the site suite check that.
The timing script (scripts/site_timing.py) averages seconds per problem with it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

PM_ITERATIONS = 80
EPS_VAR = 1e-12


# ---- the t distribution: regularised incomplete beta by its continued fraction ----------------------------------------
def _betacf(a: float, b: float, x: float) -> float:
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1.0 + aa / c
        c = c if abs(c) > 1e-300 else 1e-300
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1.0 + aa / c
        c = c if abs(c) > 1e-300 else 1e-300
        dl = d * c
        h *= dl
        if abs(dl - 1.0) < 1e-15:
            break
    return h


def betai(a: float, b: float, x: float) -> float:
    """The regularised incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: float) -> float:
    tail = 0.5 * betai(df / 2.0, 0.5, df / (df + t * t))
    return 1.0 - tail if t >= 0 else tail


def t_ppf(q: float, df: float) -> float:
    """The q-quantile of Student's t by bisection (q > 0.5)."""
    lo, hi = 0.0, 1.0
    while t_cdf(hi, df) < q:
        hi *= 2.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        lo, hi = (mid, hi) if t_cdf(mid, df) < q else (lo, mid)
    return (lo + hi) / 2.0


def t_two_sided_p(t: float, df: float) -> float:
    return min(1.0, 2.0 * (1.0 - t_cdf(abs(t), df)))


# ---- one problem set ---------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SetStat:
    """A problem set at one budget: n problems with a value, the sum and the sum of squares of their values."""
    n: int
    s1: float
    s2: float

    @property
    def mean(self) -> float:
        return self.s1 / self.n

    @property
    def var(self) -> float | None:
        if self.n < 2:
            return None
        return max(0.0, (self.s2 - self.n * self.mean ** 2) / (self.n - 1))


def logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def expit(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def set_estimates(sets: Sequence[SetStat], scale: str) -> tuple[list[float], list[float]]:
    """Each set's estimate on the scale the sets are combined on, and its squared standard error.

    A set whose problems all have the same value, or that has one problem, has no usable variance of its own: a rate then
    takes its continuity-corrected binomial variance p(1 - p), any other metric the variance pooled over the sets."""
    sets = [s for s in sets if s.n > 0]
    num = sum((s.n - 1) * s.var for s in sets if s.var is not None)
    den = sum(s.n - 1 for s in sets if s.var is not None)
    pooled = num / den if den > 0 else 0.0
    ys, vs = [], []
    for s in sets:
        own = s.var if s.var is not None and s.var > 0 else None
        if scale == "logit":
            p = (s.n * s.mean + 0.5) / (s.n + 1.0)
            v = own if own is not None else p * (1.0 - p)
            ys.append(logit(p))
            vs.append(max(EPS_VAR, v / s.n / (p * (1.0 - p)) ** 2))
        else:
            v = own if own is not None else pooled
            ys.append(s.mean)
            vs.append(max(EPS_VAR, v / s.n))
    return ys, vs


# ---- combining the sets ----------------------------------------------------------------------------------------------
def paule_mandel(y: Sequence[float], v: Sequence[float]) -> float:
    """tau^2 such that the weighted squared deviations equal their expectation S - 1 (0 when they fall short)."""
    S = len(y)

    def excess(t2: float) -> float:
        w = [1.0 / (vi + t2) for vi in v]
        mu = sum(wi * yi for wi, yi in zip(w, y)) / sum(w)
        return sum(wi * (yi - mu) ** 2 for wi, yi in zip(w, y)) - (S - 1)

    if S < 2 or excess(0.0) <= 0:
        return 0.0
    lo, hi = 0.0, 1.0
    while excess(hi) > 0:
        hi *= 2.0
    for _ in range(PM_ITERATIONS):
        mid = (lo + hi) / 2.0
        lo, hi = (mid, hi) if excess(mid) > 0 else (lo, mid)
    return (lo + hi) / 2.0


@dataclass
class Combined:
    """mu and its 95 % interval on the combining scale, the prediction interval for a new set (None below three sets),
    tau^2, the normalised weights, the number of sets and the two-sided p-value of mu = 0 (for differences)."""
    mu: float
    lo: float
    hi: float
    pi_lo: float | None
    pi_hi: float | None
    tau2: float
    weights: list[float]
    S: int
    p: float


def combine_scale(y: Sequence[float], v: Sequence[float], df: int = 1) -> Combined:
    """The random-effects mean of set estimates y with squared standard errors v, and its interval over these sets:
    Var(mu) = sum w^2 v / (sum w)^2 with the weights held fixed, t with ``df`` degrees of freedom (the problems' own,
    sum over the sets of n - 1). One set is its own mean, with the t interval on its problems."""
    S = len(y)
    tau2 = paule_mandel(y, v) if S > 1 else 0.0
    w = [1.0 / (vi + tau2) for vi in v]
    sw = sum(w)
    mu = sum(wi * yi for wi, yi in zip(w, y)) / sw
    se = math.sqrt(sum(wi * wi * vi for wi, vi in zip(w, v))) / sw
    df = max(1, df)
    tq = t_ppf(0.975, df)
    p = t_two_sided_p(mu / se, df) if se > 0 else (0.0 if mu != 0 else 1.0)
    pi_lo = pi_hi = None
    if S >= 3:
        half = t_ppf(0.975, S - 2) * math.sqrt(tau2 + 1.0 / sw)
        pi_lo, pi_hi = mu - half, mu + half
    return Combined(mu, mu - tq * se, mu + tq * se, pi_lo, pi_hi, tau2, [wi / sw for wi in w], S, p)


def combine(sets: Sequence[SetStat], scale: str = "normal") -> Combined | None:
    """Combine problem sets; a rate (scale "logit") comes back on its own 0-1 scale."""
    sets = [s for s in sets if s.n > 0]
    if not sets:
        return None
    y, v = set_estimates(sets, scale)
    c = combine_scale(y, v, sum(s.n - 1 for s in sets))
    if scale == "logit":
        f = expit
        return Combined(f(c.mu), f(c.lo), f(c.hi), None if c.pi_lo is None else f(c.pi_lo),
                        None if c.pi_hi is None else f(c.pi_hi), c.tau2, c.weights, c.S, c.p)
    return c


def holm(pvalues: Iterable[float]) -> list[float]:
    """Holm-adjusted p-values, in the input order."""
    ps = list(pvalues)
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        out[i] = running
    return out
