"""Averages over benchmarks: the averaging of srbf's results site and of ``srbf analyze``.

Results are organised as benchmarks > subsets > problems > runs. A benchmark is a problem set as its authors published
it; its subsets are the parts they published it in (a problem's subset), else the catalogs it is stored in (a catalog
that names no benchmark is a benchmark of its own). The structure is fixed by the data, never chosen by looking at
results (:func:`nest`). A problem's value is the mean of its runs, and a subset is summarised by the number of its
problems with a value, their mean and their variance. Subsets and benchmarks each have their own true values: a
subset's scatters around its benchmark's with variance tau_S^2, a benchmark's around the average with tau_B^2 (a
three-level random-effects model, :func:`combine_nested`). Means: a benchmark's subsets are weighted by
1 / (tau_S^2 + their sampling variance), the benchmarks by 1 / (tau_B^2 + the variance of their estimate), both
variances by Paule and Mandel's rule (tau_S^2 within benchmarks; DerSimonian and Laird 1986 for the random-effects
average). Rates (shares of problems solved): the correlation rho_S of a subset's problems within its benchmark by
Williams' (1982) rule, each benchmark as its effective counts (each subset's counts scaled by 1 / (1 + (n - 1) rho_S)),
and the benchmarks by the beta-binomial model (Skellam 1948): a benchmark's count is binomial with its rate, and the
rates follow a beta distribution whose mean is the average; its likelihood is closed-form, so benchmarks at 0 % or
100 % enter as they are. The variance components only set the weights: subsets that differ count more on their own,
subsets that are alike collapse into their benchmark. Ratios are averaged in log2 (geometric means), and medians weigh
each subset as the average does, with Woodruff's interval. With one subset per benchmark every result is the
two-level one, bit for bit.

Every interval covers these benchmarks and subsets: the weights (for rates, rho_S and the beta's concentration) are
held at their estimates and only the sampling of problems within each subset counts, with t on the problems' degrees
of freedom, sum (n - 1). Where one more benchmark would fall is the prediction range (Higgins, Thompson and
Spiegelhalter 2009), for a benchmark of one subset; for rates a beta's 2.5 % to 97.5 %. One benchmark of one subset is
its own mean with the t interval over its problems, a rate its plain rate with the Wilson interval.
docs/averaging.md has the formulas, the reasons and the sources.

results-site/explorer_v2.js implements the same arithmetic step for step (the t distribution below is the same
continued fraction); scripts/site_random_effects.py loads this file for the site's scripts, and tests and the site
suite check that the page and this module agree. :func:`average` is the entry point from per-problem run values;
:func:`combine_nested` works from per-subset sums grouped by benchmark, :func:`combine` from per-set sums.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence, TypeVar

import numpy as np
from scipy import special as _sp

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
    """A problem set at one budget: n problems with a value, the sum and the sum of squares of their values (a
    benchmark's effective counts, :func:`effective_sets`, may be fractional)."""
    n: float
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
    """Each set's mean and its squared standard error, for a metric that is not a rate (rates: combine_rates).

    A set whose problems all have the same value, or that has one problem, has no usable variance of its own: it takes
    the variance pooled over the sets."""
    assert scale == "normal", scale
    sets = [s for s in sets if s.n > 0]
    num = sum((s.n - 1) * s.var for s in sets if s.var is not None)
    den = sum(s.n - 1 for s in sets if s.var is not None)
    pooled = num / den if den > 0 else 0.0
    ys, vs = [], []
    for s in sets:
        own = s.var if s.var is not None and s.var > 0 else None
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
    tau^2, the normalised weights, the number of sets and the two-sided p-value of mu = 0 (for differences).

    Over benchmarks > subsets (:func:`combine_nested`) the sets are the benchmarks: ``tau2`` is the spread between
    benchmarks, ``tau2_within`` the spread between the subsets of one benchmark (a rate's 1 / nu_S, as ``tau2`` holds
    1 / nu_B), ``weights`` each benchmark's share and ``subset_weights`` each subset's (per benchmark, summing to its
    benchmark's share)."""
    mu: float
    lo: float
    hi: float
    pi_lo: float | None
    pi_hi: float | None
    tau2: float
    weights: list[float]
    S: int
    p: float | None
    tau2_within: float = 0.0
    subset_weights: list[list[float]] | None = None


def combine_scale(y: Sequence[float], v: Sequence[float], df: int = 1, *, sampling: Sequence[float] | None = None,
                  tau2_within: float = 0.0) -> Combined:
    """The random-effects mean of set estimates y with squared standard errors v, and its interval over these sets:
    Var(mu) = sum w^2 v / (sum w)^2 with the weights held fixed, t with ``df`` degrees of freedom (the problems' own,
    sum over the sets of n - 1). One set is its own mean, with the t interval on its problems.

    For benchmarks made of subsets (:func:`combine_scale_nested`), ``v`` is a benchmark's variance given its own true
    value (which counts the spread of its subsets, tau2_within), ``sampling`` the part of it that is the sampling of
    problems, the only part the interval counts, and ``tau2_within`` widens the range for one more benchmark."""
    S = len(y)
    tau2 = paule_mandel(y, v) if S > 1 else 0.0
    w = [1.0 / (vi + tau2) for vi in v]
    sw = sum(w)
    mu = sum(wi * yi for wi, yi in zip(w, y)) / sw
    se = math.sqrt(sum(wi * wi * si for wi, si in zip(w, v if sampling is None else sampling))) / sw
    df = max(1, df)
    tq = t_ppf(0.975, df)
    p = t_two_sided_p(mu / se, df) if se > 0 else (0.0 if mu != 0 else 1.0)
    pi_lo = pi_hi = None
    if S >= 3:
        half = t_ppf(0.975, S - 2) * math.sqrt(tau2 + tau2_within + 1.0 / sw)
        pi_lo, pi_hi = mu - half, mu + half
    return Combined(mu, mu - tq * se, mu + tq * se, pi_lo, pi_hi, tau2, [wi / sw for wi in w], S, p, tau2_within)


# ---- rates: the beta-binomial model ----------------------------------------------------------------------------------
# A set's sum k of n problem values (a value is the share of its runs that succeeded, so k may be fractional) is
# binomial with the set's rate p, and the sets' rates follow Beta(m s, (1 - m) s): m is the mean rate of a problem set,
# s how tightly the sets cluster around it (s -> inf: no spread beyond the binomial; s -> 0: every set at 0 % or 100 %).
# A set's likelihood is B(k + a, n - k + b) / B(a, b) with a = m s, b = (1 - m) s: closed-form, so 0 % and 100 % enter
# as they are. m and s by maximum likelihood, Newton in (eta, lam) = (logit m, log s) with the exact gradient and Hessian.
LAM_MIN, LAM_MAX = -20.0, 15.0   # s from 2e-9 to 3.3e6 (within 0.03 % of the binomial for sets of up to 1,024 problems)
MAX_STEP = 2.0                   # Newton moves at most this far in one step
Z975 = 1.959963984540054


def digamma(x: float) -> float:
    """psi(x), x > 0: the recurrence psi(x) = psi(x + 1) - 1/x up to x >= 10, then the asymptotic series."""
    r = 0.0
    while x < 10.0:
        r -= 1.0 / x
        x += 1.0
    f = 1.0 / (x * x)
    return r + math.log(x) - 0.5 / x - f * (1.0 / 12 - f * (1.0 / 120 - f * (1.0 / 252 - f * (1.0 / 240 - f * (1.0 / 132)))))


def trigamma(x: float) -> float:
    """psi'(x), x > 0: psi'(x) = psi'(x + 1) + 1/x^2 up to x >= 10, then the asymptotic series."""
    r = 0.0
    while x < 10.0:
        r += 1.0 / (x * x)
        x += 1.0
    f = 1.0 / (x * x)
    return r + 1.0 / x + 0.5 * f + (1.0 / x) * f * (1.0 / 6 - f * (1.0 / 30 - f * (1.0 / 42 - f * (1.0 / 30 - f * (5.0 / 66)))))


def _sig(x: float) -> float:
    """expit(x), without overflow."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _bb(sets: Sequence[SetStat], eta: float, lam: float) -> tuple[float, float, float, float, float, float]:
    """The log likelihood of all sets (without the binomial coefficients), its gradient (d/deta, d/dlam) and Hessian
    (eta eta, eta lam, lam lam): the derivatives in (a, b) are digamma and trigamma terms, carried over by the chain rule
    (da/deta = s m (1 - m) = -db/deta, da/dlam = a, db/dlam = b)."""
    m = 1.0 / (1.0 + math.exp(-eta))
    s = math.exp(lam)
    a, b, q = m * s, (1.0 - m) * s, m * (1.0 - m)
    ll = da = db = haa = hbb = hab = 0.0
    lg_s, dg_s, tg_s = math.lgamma(s), digamma(s), trigamma(s)
    lg_a, lg_b, dg_a, dg_b, tg_a, tg_b = math.lgamma(a), math.lgamma(b), digamma(a), digamma(b), trigamma(a), trigamma(b)
    for st in sets:
        n, k = st.n, st.s1
        ll += math.lgamma(k + a) + math.lgamma(n - k + b) - math.lgamma(n + s) - lg_a - lg_b + lg_s
        dns, tns = digamma(n + s), trigamma(n + s)
        da += digamma(k + a) - dns - dg_a + dg_s
        db += digamma(n - k + b) - dns - dg_b + dg_s
        haa += trigamma(k + a) - tns - tg_a + tg_s
        hbb += trigamma(n - k + b) - tns - tg_b + tg_s
        hab += -tns + tg_s
    ae, be, al, bl = s * q, -s * q, a, b
    return (ll, ae * da + be * db, al * da + bl * db,
            ae * ae * haa + 2 * ae * be * hab + be * be * hbb + (da - db) * s * q * (1.0 - 2.0 * m),
            ae * al * haa + (ae * bl + be * al) * hab + be * bl * hbb + (da - db) * s * q,
            al * al * haa + 2 * al * bl * hab + bl * bl * hbb + da * a + db * b)


def rate_fit(sets: Sequence[SetStat]) -> tuple[float, float]:
    """Maximum likelihood (eta, lam) = (logit m, log s) by Newton on |H|, from the pooled rate and s = 1.

    The step is V diag(1 / |ev|) V^T g for the Hessian's eigen-decomposition V diag(ev) V^T: an ascent step along every
    direction, long where the likelihood is flat. A step is at most MAX_STEP long and halved until the likelihood does
    not fall; lam stays within [LAM_MIN, LAM_MAX] (at a bound, eta moves alone). The pooled rate (s = inf, lam returned as
    inf) is kept when it is at least as likely. eta = -inf / +inf when every set is at 0 % / 100 %."""
    if all(s.s1 <= 0 for s in sets):
        return -math.inf, math.inf
    if all(s.s1 >= s.n for s in sets):
        return math.inf, math.inf
    p0 = sum(s.s1 for s in sets) / sum(s.n for s in sets)
    l0 = sum(s.s1 * math.log(p0) + (s.n - s.s1) * math.log(1.0 - p0) for s in sets)
    eta, lam = math.log(p0 / (1.0 - p0)), 0.0
    ll, ge, gl, hee, hel, hll = _bb(sets, eta, lam)
    for _ in range(100):
        if (lam >= LAM_MAX and gl > 0) or (lam <= LAM_MIN and gl < 0):
            de, dl = (-ge / hee if hee < 0 else ge), 0.0
        else:
            half_tr, rad = 0.5 * (hee + hll), math.sqrt((0.5 * (hee - hll)) ** 2 + hel * hel)
            ev1, ev2 = half_tr + rad, half_tr - rad
            if rad == 0.0:
                v1, v2 = (1.0, 0.0), (0.0, 1.0)
            elif abs(hel) > 0.0:
                nrm = math.hypot(ev1 - hll, hel)
                v1 = ((ev1 - hll) / nrm, hel / nrm)
                v2 = (-v1[1], v1[0])
            else:
                v1, v2 = ((1.0, 0.0), (0.0, 1.0)) if hee >= hll else ((0.0, 1.0), (1.0, 0.0))
            floor = 1e-10 * max(1.0, abs(ev1), abs(ev2))
            c1 = (v1[0] * ge + v1[1] * gl) / max(abs(ev1), floor)
            c2 = (v2[0] * ge + v2[1] * gl) / max(abs(ev2), floor)
            de, dl = c1 * v1[0] + c2 * v2[0], c1 * v1[1] + c2 * v2[1]
        big = max(abs(de), abs(dl))
        if big > MAX_STEP:
            de, dl = de * MAX_STEP / big, dl * MAX_STEP / big
        step = 1.0
        while True:
            eta_n, lam_n = eta + step * de, min(LAM_MAX, max(LAM_MIN, lam + step * dl))
            res = _bb(sets, eta_n, lam_n)
            if res[0] >= ll - 1e-12 * abs(ll) or step < 1e-12:
                break
            step *= 0.5
        moved = max(abs(eta_n - eta), abs(lam_n - lam))
        eta, lam = eta_n, lam_n
        ll, ge, gl, hee, hel, hll = res
        if moved <= 1e-10 or max(abs(ge), abs(gl)) <= 1e-9:
            break
    if l0 >= ll:
        return math.log(p0 / (1.0 - p0)), math.inf
    return eta, lam


def beta_ppf(q: float, a: float, b: float) -> float:
    """The q-quantile of Beta(a, b), by bisection on the regularised incomplete beta function."""
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if betai(a, b, mid) < q else (lo, mid)
    return 0.5 * (lo + hi)


def wilson(k: float, n: float) -> tuple[float, float]:
    """The Wilson score interval (95 %) of the rate k / n."""
    p, z2 = k / n, Z975 * Z975
    den = 1.0 + z2 / n
    c = (p + z2 / (2.0 * n)) / den
    h = Z975 * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


RATE_ETA_MAX = 30.0     # the interval is searched for eta in [-30, 30]: expit(+-30) is 1e-13 from 0 and 1
RATE_SCAN_STEP = 0.5    # ... on a grid of this step, outward from the estimate, then by bisection at the outermost edge
RATE_K_SPAN = 12.0      # a set's count is summed over its mean +- (12 sd + 4)


def _digamma_v(x: np.ndarray) -> np.ndarray:
    """digamma for an array (scipy's; it agrees with digamma() to rounding)."""
    return _sp.digamma(x)


def _trigamma_v(x: np.ndarray) -> np.ndarray:
    """trigamma for an array (scipy's; it agrees with trigamma() to rounding)."""
    return _sp.polygamma(1, x)


def _counts(sets: Sequence[SetStat]) -> tuple[np.ndarray, np.ndarray]:
    return np.array([st.n for st in sets], dtype=float), np.array([st.s1 for st in sets], dtype=float)


def _lam_slope_a(n: np.ndarray, k: np.ndarray, m: float, lam: float) -> tuple[float, float]:
    s = math.exp(lam)
    a, b = m * s, (1.0 - m) * s
    hit, miss = k > 0, n - k > 0
    g = -(_digamma_v(n + s) - digamma(s))
    h = -(_trigamma_v(n + s) - trigamma(s))
    g += np.where(hit, m * (_digamma_v(k + a) - digamma(a)), 0.0)
    h += np.where(hit, m * m * (_trigamma_v(k + a) - trigamma(a)), 0.0)
    g += np.where(miss, (1.0 - m) * (_digamma_v(n - k + b) - digamma(b)), 0.0)
    h += np.where(miss, (1.0 - m) * (1.0 - m) * (_trigamma_v(n - k + b) - trigamma(b)), 0.0)
    d1, d2 = float(g.sum()), float(h.sum())
    return s * d1, s * d1 + s * s * d2


def _lam_slope(sets: Sequence[SetStat], m: float, lam: float) -> tuple[float, float]:
    """d/dlam and d2/dlam2 of the log likelihood at mean m, lam = log s (a set at 0 % or 100 % has no k or n - k term)."""
    return _lam_slope_a(*_counts(sets), m, lam)


def _profile_lam_a(n: np.ndarray, k: np.ndarray, m: float, start: float) -> float:
    lo, hi, lo_known, hi_known = LAM_MIN, LAM_MAX, False, False
    x = min(LAM_MAX, max(LAM_MIN, start))
    for _ in range(200):
        d1, d2 = _lam_slope_a(n, k, m, x)
        if d1 > 0.0:
            if x >= LAM_MAX:
                return LAM_MAX
            lo, lo_known = x, True
        elif d1 < 0.0:
            if x <= LAM_MIN:
                return LAM_MIN
            hi, hi_known = x, True
        else:
            return x
        step = -d1 / d2 if d2 < 0.0 else math.copysign(MAX_STEP, d1)
        step = max(-MAX_STEP, min(MAX_STEP, step))
        xn = x + step
        if xn >= hi:
            xn = 0.5 * (lo + hi) if hi_known else LAM_MAX
        elif xn <= lo:
            xn = 0.5 * (lo + hi) if lo_known else LAM_MIN
        if abs(xn - x) <= 1e-12 * max(1.0, abs(x)) or (lo_known and hi_known and hi - lo <= 1e-12 * max(1.0, abs(x))):
            return xn
        x = xn
    return x


def _profile_lam(sets: Sequence[SetStat], m: float, start: float) -> float:
    """The lam = log s that maximises the likelihood at mean m, within [LAM_MIN, LAM_MAX]: Newton on the slope from
    ``start``, steps of at most MAX_STEP, bisection where Newton would leave the bracket of a sign change."""
    return _profile_lam_a(*_counts(sets), m, start)


def _rate_band_a(n: np.ndarray, k: np.ndarray, eta: float, lam: float, t2: float) -> bool:
    m = _sig(eta)
    s = math.exp(lam)
    a, b = m * s, (1.0 - m) * s
    dg_a, dg_b, tg_a, tg_b, tg_s = digamma(a), digamma(b), trigamma(a), trigamma(b), trigamma(s)
    hit, miss = k > 0, n - k > 0
    A = np.where(hit, _digamma_v(k + a) - dg_a, 0.0)
    Bt = np.where(miss, _digamma_v(n - k + b) - dg_b, 0.0)
    A1 = np.where(hit, _trigamma_v(k + a) - tg_a, 0.0)
    B1 = np.where(miss, _trigamma_v(n - k + b) - tg_b, 0.0)
    g = s * float((A - Bt).sum())
    if not math.isfinite(g):
        return False
    h_ms = float(((A - Bt) + s * (m * A1 - (1.0 - m) * B1)).sum())
    h_ss = float((m * m * A1 + (1.0 - m) * (1.0 - m) * B1 - (_trigamma_v(n + s) - tg_s)).sum())
    c = h_ms / h_ss if h_ss != 0.0 and math.isfinite(h_ms / h_ss) else 0.0
    ca, cb = s - c * m, s + c * (1.0 - m)        # U_m - c U_s = ca A - cb Bt + constant
    p = (k + a) / (n + s)
    keep = (p > 0.0) & (p < 1.0)
    if not keep.any():
        return g * g <= 0.0
    nk, pk = n[keep], p[keep]
    sd = np.sqrt(nk * pk * (1.0 - pk))
    lo = np.maximum(0.0, np.floor(nk * pk - RATE_K_SPAN * sd) - 4)
    hi = np.minimum(nk, np.ceil(nk * pk + RATE_K_SPAN * sd) + 4)
    lens = (hi - lo + 1).astype(int)
    starts = np.cumsum(lens) - lens
    seg = np.repeat(np.arange(len(lens)), lens)
    ks = lo[seg] + (np.arange(int(lens.sum())) - starts[seg])
    nn = nk[seg]
    x = ca * np.where(ks > 0, _digamma_v(ks + a) - dg_a, 0.0) - cb * np.where(nn - ks > 0, _digamma_v(nn - ks + b) - dg_b, 0.0)
    # the binomial weights relative to each set's first count: w(k + 1) / w(k) = (n - k) / (k + 1) * p / (1 - p)
    with np.errstate(divide="ignore"):
        step = np.where(nn - ks > 0, np.log(np.maximum(nn - ks, 1.0)) - np.log(ks + 1.0) + np.log(pk / (1.0 - pk))[seg], 0.0)
    cs = np.cumsum(step)
    logw = (cs - step) - (cs - step)[starts][seg]
    logw -= np.maximum.reduceat(logw, starts)[seg]
    w = np.exp(logw)
    tot = np.add.reduceat(w, starts)
    mean = np.add.reduceat(w * x, starts) / tot
    var = float((np.add.reduceat(w * (x - mean[seg]) ** 2, starts) / tot).sum())
    return g * g <= t2 * var


def _rate_band(sets: Sequence[SetStat], eta: float, lam: float, t2: float) -> bool:
    """Whether Rao's score test at mean m = expit(eta), with s refitted at m (lam = log s), keeps m.

    The statistic is the efficient score for m, sum_i (U_m,i - c U_s,i) with c = H_ms / H_ss (the nuisance s projected
    out; at s's own maximum sum_i U_s,i = 0, so it is the plain score sum_i U_m,i), against t^2 times its variance from
    the problems' sampling: every set's count k_i ~ Binomial(n_i, p_i) at its posterior rate p_i = (k_i + a)/(n_i + s),
    the variance summed exactly over k_i, not linearised. Computed over all sets at once (numpy)."""
    return _rate_band_a(*_counts(sets), eta, lam, t2)


def _rate_interval(sets: Sequence[SetStat], eta: float, lam: float, t2: float) -> tuple[float, float]:
    """The smallest interval holding every m the score test keeps (_rate_band). The test is run on a grid of eta from
    the estimate outward to +-RATE_ETA_MAX, refitting s along the way; the outermost kept point is refined by bisection
    against its outer neighbour. Where the kept points come in more than one piece (s refitted far from the estimate can
    be small enough that the test loses power), the interval spans them."""
    n, k = _counts(sets)
    lam0 = min(lam, LAM_MAX)
    eta = max(-RATE_ETA_MAX, min(RATE_ETA_MAX, eta))
    ends = []
    for direction in (-1.0, 1.0):
        points = [(eta, lam0, True)]
        x, warm = eta, lam0
        while abs(x) < RATE_ETA_MAX:
            x = x + direction * RATE_SCAN_STEP
            if abs(x) >= RATE_ETA_MAX:
                x = direction * RATE_ETA_MAX
            warm = _profile_lam_a(n, k, _sig(x), warm)
            points.append((x, warm, _rate_band_a(n, k, x, warm, t2)))
        last = max(i for i, pt in enumerate(points) if pt[2])
        if last == len(points) - 1:
            ends.append(direction * math.inf)
            continue
        inner, outer, warm = points[last][0], points[last + 1][0], points[last][1]
        for _ in range(45):
            mid = 0.5 * (inner + outer)
            lam_mid = _profile_lam_a(n, k, _sig(mid), warm)
            if _rate_band_a(n, k, mid, lam_mid, t2):
                inner, warm = mid, lam_mid
            else:
                outer = mid
        ends.append(inner)
    return (0.0 if ends[0] == -math.inf else _sig(ends[0])), (1.0 if ends[1] == math.inf else _sig(ends[1]))


def combine_rates(sets: Sequence[SetStat], df: float | None = None, rho_within: float = 0.0) -> Combined:
    """Rates over problem sets by the beta-binomial model: the mean rate m of a problem set, on the 0-1 scale.

    One set is its plain rate k / n with the Wilson interval; so is a group in which every set is at 0 % (or 100 %).
    The interval covers these problem sets, as for every other metric: the sets' rates fixed, only the sampling of
    problems within each set. It inverts Rao's score test for m with s as a nuisance (_rate_band): every m whose score,
    with s refitted at m and projected out, stays within t times its standard deviation, the variance at the tested m
    and summed exactly over the sets' binomial counts; t on sum (n - 1). Testing at each m, as Wilson's interval does
    for one rate, keeps the interval inside 0-100 % and sensible where many sets sit at 0 % or 100 %, where a
    delta-method interval on the log-odds spanned almost the whole range. The range for one more set is the 2.5 % to
    97.5 % quantiles of Beta(a, b). The weights are each set's share of the information about eta; tau2 carries 1 / s
    (0 for the pooled rate).

    For benchmarks made of subsets (:func:`combine_rates_nested`) the sets are the benchmarks' effective counts, ``df``
    the problems' degrees of freedom (sum (n - 1) over the subsets) and ``rho_within`` the correlation of the problems
    of one subset within its benchmark: one more benchmark of one subset spreads as a beta whose 1 / (nu + 1) is
    rho_B + (1 - rho_B) rho_within, rho_B = 1 / (s + 1)."""
    S = len(sets)
    N = sum(s.n for s in sets)
    K = sum(s.s1 for s in sets)
    eta, lam = rate_fit(sets) if S > 1 else (0.0, 0.0)
    within = rho_within / (1.0 - rho_within) if rho_within < 1.0 else math.inf
    if S == 1 or not math.isfinite(eta):
        lo, hi = wilson(K, N)
        return Combined(K / N, lo, hi, None, None, 0.0, [s.n / N for s in sets], S, None, within)
    pooled = not math.isfinite(lam)
    m = 1.0 / (1.0 + math.exp(-eta))
    s = math.exp(min(lam, LAM_MAX))
    a, b, q = m * s, (1.0 - m) * s, m * (1.0 - m)
    tg_a, tg_b = trigamma(a), trigamma(b)
    info = [-(s * q) ** 2 * (trigamma(st.s1 + a) - tg_a + trigamma(st.n - st.s1 + b) - tg_b) for st in sets]
    tq = t_ppf(0.975, max(1, sum(st.n - 1 for st in sets) if df is None else df))
    lo, hi = _rate_interval(sets, eta, lam, tq * tq)
    pi_lo = pi_hi = None
    if S >= 3:
        pa, pb = a, b
        if rho_within > 0.0:   # a benchmark of one subset: its spread between benchmarks and that of its subset
            rho_b = 1.0 / (s + 1.0)
            nu = max(math.exp(LAM_MIN), 1.0 / (rho_b + (1.0 - rho_b) * rho_within) - 1.0)
            pa, pb = m * nu, (1.0 - m) * nu
        pi_lo, pi_hi = beta_ppf(0.025, pa, pb), beta_ppf(0.975, pa, pb)
    tot = sum(info)
    return Combined(m, min(lo, m), max(hi, m), pi_lo, pi_hi, 0.0 if pooled else 1.0 / s, [x / tot for x in info], S, None,
                    within)


# ---- three levels: benchmarks > subsets > problems -------------------------------------------------------------------
# A benchmark may consist of subsets: the parts its authors published it in, or the catalogs it is stored in. The
# structure is read from the data (a problem's subset, else its catalog) and never chosen by looking at results.
# Problems sit within subsets within benchmarks, and each level has true values of its own: a subset's true value
# scatters around its benchmark's with variance tau_S^2, a benchmark's around the average with tau_B^2 (three-level
# random effects). Both are estimated for every average, and they only set the weights: subsets that really differ
# count more on their own, subsets that are alike collapse into their benchmark. A benchmark of one subset is a problem
# set as before, and when no benchmark has two subsets the average is the two-level one, bit for bit.

def within_tau2(ys: Sequence[Sequence[float]], vs: Sequence[Sequence[float]]) -> float:
    """tau_S^2, the spread of subsets around their benchmark, by Paule and Mandel's rule within the benchmarks: the
    subsets' weighted squared deviations from their benchmark's weighted mean (weights 1 / (v + tau_S^2)) equal their
    expectation, the sum over the benchmarks of (subsets - 1). 0 when they fall short of it, or no benchmark has two
    subsets."""
    groups = [(y, v) for y, v in zip(ys, vs) if len(y) > 1]
    df = sum(len(y) - 1 for y, _ in groups)

    def excess(t2: float) -> float:
        q = 0.0
        for y, v in groups:
            w = [1.0 / (vi + t2) for vi in v]
            mu = sum(wi * yi for wi, yi in zip(w, y)) / sum(w)
            q += sum(wi * (yi - mu) ** 2 for wi, yi in zip(w, y))
        return q - df

    if df == 0 or excess(0.0) <= 0:
        return 0.0
    lo, hi = 0.0, 1.0
    while excess(hi) > 0:
        hi *= 2.0
    for _ in range(PM_ITERATIONS):
        mid = (lo + hi) / 2.0
        lo, hi = (mid, hi) if excess(mid) > 0 else (lo, mid)
    return (lo + hi) / 2.0


def combine_scale_nested(ys: Sequence[Sequence[float]], vs: Sequence[Sequence[float]], df: int = 1) -> Combined:
    """The three-level random-effects mean of subset estimates ``ys[b][s]`` (benchmark b, subset s) with squared
    standard errors ``vs[b][s]``. Given tau_S^2 (:func:`within_tau2`), a benchmark's subsets combine with weights
    1 / (v + tau_S^2) into its estimate, whose variance given the benchmark's true value is 1 / their sum (one subset:
    v + tau_S^2); the benchmarks combine as sets do (:func:`combine_scale`), tau_B^2 by Paule and Mandel. This is the
    generalised least squares mean of the three-level model, done in two steps. The interval counts only the sampling
    of problems, sum over subsets of (weight)^2 v, the weights held fixed."""
    tau2_s = within_tau2(ys, vs)
    y_b: list[float] = []
    v_b: list[float] = []
    s_b: list[float] = []
    shares: list[list[float]] = []
    for y, v in zip(ys, vs):
        if len(y) == 1:
            y_b.append(y[0])
            v_b.append(v[0] + tau2_s)
            s_b.append(v[0])
            shares.append([1.0])
            continue
        w = [1.0 / (vi + tau2_s) for vi in v]
        W = sum(w)
        sh = [wi / W for wi in w]
        y_b.append(sum(wi * yi for wi, yi in zip(w, y)) / W)
        v_b.append(1.0 / W)
        s_b.append(sum(x * x * vi for x, vi in zip(sh, v)))
        shares.append(sh)
    c = combine_scale(y_b, v_b, df, sampling=s_b, tau2_within=tau2_s)
    c.subset_weights = [[wb * x for x in sh] for wb, sh in zip(c.weights, shares)]
    return c


def within_rho(groups: Sequence[Sequence[SetStat]]) -> float:
    """rho_S, the correlation between two problems of one subset that its benchmark's rate does not explain, by
    Williams' (1982) rule: a subset of n problems counts as c = 1 / (1 + (n - 1) rho) of them (its design effect), a
    benchmark's rate is p = sum c k / sum c n, and Pearson's chi-square of the subsets about their benchmark's rate,
    sum c n (k / n - p)^2 / (p (1 - p)), equals its degrees of freedom, the sum over benchmarks of (subsets - 1). A
    benchmark whose subsets are all at 0 % or all at 100 % says nothing about rho and is left out. 0 when the
    chi-square falls short of its degrees of freedom, or no benchmark has two subsets; 1 (every subset counts as one
    problem) when it exceeds them even there. The root is found by bisection."""
    use = [g for g in groups if len(g) > 1 and sum(st.s1 for st in g) > 0 and any(st.s1 < st.n for st in g)]
    df = sum(len(g) - 1 for g in use)

    def excess(rho: float) -> float:
        x2 = 0.0
        for g in use:
            c = [1.0 / (1.0 + (st.n - 1) * rho) for st in g]
            p = sum(ci * st.s1 for ci, st in zip(c, g)) / sum(ci * st.n for ci, st in zip(c, g))
            x2 += sum(ci * st.n * (st.s1 / st.n - p) ** 2 for ci, st in zip(c, g)) / (p * (1.0 - p))
        return x2 - df

    if df == 0 or excess(0.0) <= 0:
        return 0.0
    if excess(1.0) >= 0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(PM_ITERATIONS):
        mid = (lo + hi) / 2.0
        lo, hi = (mid, hi) if excess(mid) > 0 else (lo, mid)
    return (lo + hi) / 2.0


def effective_sets(groups: Sequence[Sequence[SetStat]], rho: float) -> list[SetStat]:
    """Each benchmark as one set of effective counts: its subsets' problems and sums, each subset's scaled by its
    c = 1 / (1 + (n - 1) rho) (Kish's effective sample size). rho = 0 merges the subsets: every problem counts once."""
    out = []
    for g in groups:
        if len(g) == 1 and rho == 0.0:
            out.append(g[0])
            continue
        c = [1.0 / (1.0 + (st.n - 1) * rho) for st in g]
        out.append(SetStat(sum(ci * st.n for ci, st in zip(c, g)), sum(ci * st.s1 for ci, st in zip(c, g)),
                           sum(ci * st.s2 for ci, st in zip(c, g))))
    return out


def combine_rates_nested(groups: Sequence[Sequence[SetStat]]) -> Combined:
    """Rates over benchmarks > subsets: rho_S within the benchmarks (:func:`within_rho`), each benchmark as its
    effective counts (:func:`effective_sets`), and the benchmarks by the beta-binomial model with the score-test
    interval (:func:`combine_rates`), t on the problems' degrees of freedom. Given its true rate, a benchmark's
    effective count has the mean and the variance of a binomial count of its effective size: the two-stage
    approximation of the model in which the subsets' rates follow a beta around their benchmark's (docs/averaging.md
    has its check against that model)."""
    rho = within_rho(groups)
    eff = effective_sets(groups, rho)
    c = combine_rates(eff, df=sum(st.n - 1 for g in groups for st in g), rho_within=rho)
    c.subset_weights = [[wb] if len(g) == 1 else [wb * st.n / (1.0 + (st.n - 1) * rho) / e.n for st in g]
                        for wb, g, e in zip(c.weights, groups, eff)]
    return c


def combine_nested(groups: Sequence[Sequence[SetStat]], scale: str = "normal") -> Combined | None:
    """Combine benchmarks made of subsets: ``groups[b]`` holds the subsets of benchmark b. A rate (scale "logit") comes
    back on its own 0-1 scale. A subset without problems is left out, and so is a benchmark without subsets."""
    groups = [g for g in ([s for s in g if s.n > 0] for g in groups) if g]
    if not groups:
        return None
    if scale == "logit":
        return combine_rates_nested(groups)
    flat = [s for g in groups for s in g]
    y, v = set_estimates(flat, scale)
    ys, vs, at = [], [], 0
    for g in groups:
        ys.append(y[at:at + len(g)])
        vs.append(v[at:at + len(g)])
        at += len(g)
    return combine_scale_nested(ys, vs, int(sum(s.n - 1 for s in flat)))


def combine(sets: Sequence[SetStat], scale: str = "normal") -> Combined | None:
    """Combine problem sets, each a benchmark of its own; a rate (scale "logit") comes back on its own 0-1 scale."""
    return combine_nested([[s] for s in sets], scale)


def holm(pvalues: Iterable[float]) -> list[float]:
    """Holm-adjusted p-values, in the input order."""
    ps = list(pvalues)
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        out[i] = running
    return out


# ---- benchmarks and their subsets: the problem sets -------------------------------------------------------------------
_T = TypeVar("_T")


def nest(problems: Mapping[str, Sequence[_T]], benchmarks: Mapping[str, str] | None = None,
         subsets: Mapping[str, Sequence[str | None]] | None = None) -> dict[str, dict[str, list[_T]]]:
    """The problems of each catalog placed in their benchmark and subset: {benchmark: {subset: problems}}.

    A catalog's benchmark is ``benchmarks[catalog]`` (a catalog it does not name is a benchmark of its own), and a
    problem's subset is ``subsets[catalog][i]`` for the catalog's i-th problem (the authors' part of the benchmark, or
    the part of it a catalog holds); a catalog ``subsets`` does not name, or a problem with an empty subset, is in the
    subset named after its catalog. Problems of one benchmark and one subset name are one subset, whichever catalog
    they are in. Keys in the order they first appear."""
    out: dict[str, dict[str, list[_T]]] = {}
    for catalog, values in problems.items():
        subs = out.setdefault((benchmarks or {}).get(catalog, catalog), {})
        labels = (subsets or {}).get(catalog)
        if labels is not None and len(labels) != len(values):
            raise ValueError(f"catalog {catalog!r}: {len(labels)} subsets for {len(values)} problems")
        for i, value in enumerate(values):
            label = labels[i] if labels is not None else None
            subs.setdefault(label if label else catalog, []).append(value)
    return out


def nest_sets(sets: Mapping[str, SetStat], benchmarks: Mapping[str, str] | None = None) -> list[list[SetStat]]:
    """Per-catalog summaries grouped by benchmark, each catalog a subset of it (for :func:`combine_nested`); a
    catalog ``benchmarks`` does not name is a benchmark of its own."""
    out: dict[str, list[SetStat]] = {}
    for catalog, st in sets.items():
        out.setdefault((benchmarks or {}).get(catalog, catalog), []).append(st)
    return list(out.values())


def by_benchmark(problems: Mapping[str, Sequence[_T]], benchmarks: Mapping[str, str]) -> dict[str, list[_T]]:
    """The problems of each catalog, merged into one list per benchmark (every problem in it once; a catalog
    ``benchmarks`` does not name is a benchmark of its own), keyed by benchmark in the order the benchmarks first
    appear: a benchmark's own value over all its problems. The averages nest instead (:func:`nest`)."""
    out: dict[str, list[_T]] = {}
    for catalog, values in problems.items():
        out.setdefault(benchmarks.get(catalog, catalog), []).extend(values)
    return out


def merge_sets(sets: Mapping[str, SetStat], benchmarks: Mapping[str, str]) -> dict[str, SetStat]:
    """Per-catalog summaries merged into one per benchmark, as :func:`by_benchmark` merges the problems: the counts,
    sums and sums of squares add up. A catalog ``benchmarks`` does not name is a benchmark of its own."""
    out: dict[str, SetStat] = {}
    for catalog, st in sets.items():
        name = benchmarks.get(catalog, catalog)
        prev = out.get(name)
        out[name] = st if prev is None else SetStat(prev.n + st.n, prev.s1 + st.s1, prev.s2 + st.s2)
    return out


# ---- averages from per-problem run values ----------------------------------------------------------------------------
KINDS = ("mean", "rate", "ratio", "median")


@dataclass(frozen=True)
class Average:
    """An average over benchmarks: the value and its 95 % interval over these benchmarks, the range for one more
    benchmark (None below three, and for a median), the problems, benchmarks and subsets behind it, each benchmark's
    share of the weight and each subset's ({benchmark: {subset: share}}, summing to its benchmark's)."""
    value: float
    lo: float
    hi: float
    pi_lo: float | None
    pi_hi: float | None
    n_problems: int
    n_sets: int
    weights: dict[str, float]
    n_subsets: int = 0
    subset_weights: dict[str, dict[str, float]] = field(default_factory=dict)


def _number(v: object) -> float | None:
    """A run's value as a float; None when it has none (None or NaN)."""
    if v is None:
        return None
    x = float(v)  # type: ignore[arg-type]
    return None if math.isnan(x) else x


def problem_value(runs: Sequence[object], kind: str = "mean") -> float | None:
    """A problem's value from its runs' values. A rate: the share of runs that hit (a run without a value misses). Any
    other kind: the mean of the runs' finite values, in log2 for a ratio (0 is -inf there); None when none is finite."""
    if kind == "rate":
        return sum(1.0 if (x := _number(v)) is not None and x != 0 else 0.0 for v in runs) / len(runs) if runs else None
    vals = []
    for v in runs:
        x = _number(v)
        if x is not None and kind == "ratio":
            x = math.log2(x) if x > 0 else (-math.inf if x == 0 else None)
        if x is not None and math.isfinite(x):
            vals.append(x)
    return sum(vals) / len(vals) if vals else None


def _set_stats(problems: Mapping[str, Sequence[Sequence[object]]], kind: str) -> tuple[list[str], list[SetStat]]:
    names, stats = [], []
    for name, runs_of in problems.items():
        vals = [v for v in (problem_value(runs, kind) for runs in runs_of if len(runs)) if v is not None]
        if vals:
            names.append(name)
            stats.append(SetStat(len(vals), float(sum(vals)), float(sum(v * v for v in vals))))
    return names, stats


def _nested_stats(nested: Mapping[str, Mapping[str, Sequence[Sequence[object]]]], kind: str
                  ) -> tuple[list[tuple[str, list[str]]], list[list[SetStat]]]:
    """Each benchmark's subsets with a value: ([(benchmark, [subset, ...]), ...], [[SetStat, ...], ...])."""
    names, groups = [], []
    for bench, subs in nested.items():
        sub_names, stats = _set_stats(subs, kind)
        if stats:
            names.append((bench, sub_names))
            groups.append(stats)
    return names, groups


def _weighted_quantile(points: Sequence[tuple[float, float]], q: float) -> float:
    """The smallest value whose cumulative weight reaches q of the total (points sorted by value)."""
    total = sum(w for _, w in points)
    cum = 0.0
    for x, w in points:
        cum += w
        if cum >= q * total - 1e-12 * total:
            return x
    return points[-1][0]


def nested_set_weights(ns: Sequence[Sequence[float]], n0_within: float, n0_between: float) -> list[list[float]]:
    """The weight of every subset of ``ns[b]`` (its numbers of problems, per benchmark) in a median or distribution,
    as the average weighs it. A subset of n problems counts n / (1 + n / n0_S) problems within its benchmark, so a
    benchmark counts N* = their sum, and the benchmark weighs N* / (N* + n0_B) (N* when n0_B is infinite: no spread
    between benchmarks), shared among its subsets by their counts. n0_S = sigma^2 / tau_S^2 and n0_B =
    sigma^2 / tau_B^2 from the fit of the mean (infinite when that spread is 0). A benchmark of one subset weighs
    n / (n + n0_B) when its subsets are alike, a set's weight in the two-level average."""
    out = []
    for g in ns:
        each = [n / (1.0 + n / n0_within) for n in g]
        nstar = sum(each)
        bench = nstar / (nstar + n0_between) if math.isfinite(n0_between) else nstar
        out.append([bench] if len(g) == 1 else [bench * e / nstar for e in each])
    return out


def _median(nested: Mapping[str, Mapping[str, Sequence[Sequence[object]]]]) -> Average | None:
    """The median over benchmarks > subsets: every problem weighs 1 within its subset, shared by its runs, and the
    subsets weigh as :func:`nested_set_weights` says, with n0_S and n0_B from the three-level fit of the mean (pooled
    within-subset variance over tau_S^2 and over tau_B^2): a benchmark counts about once whatever its size, a subset of
    a few problems less, and a subset that differs from its benchmark's others counts more on its own. Non-finite values
    count at their end of the scale. The interval is Woodruff's (1952): each subset's share of weight at or below the
    median is averaged over the benchmarks like any value, and that share's interval is read back through the weighted
    distribution."""
    benches: list[tuple[str, list[tuple[str, int, list[tuple[float, float]]]]]] = []
    for bench, subs in nested.items():
        units = []
        for sub, runs_of in subs.items():
            pts: list[tuple[float, float]] = []
            n = 0
            for runs in runs_of:
                vals = [x for x in (_number(v) for v in runs) if x is not None]
                if vals:
                    n += 1
                    pts += [(x, 1.0 / len(vals)) for x in vals]
            if n:
                units.append((sub, n, pts))
        if units:
            benches.append((bench, units))
    if not benches:
        return None
    _, groups = _nested_stats({b: {s: nested[b][s] for s, _, _ in units} for b, units in benches}, "mean")
    n0_s = n0_b = math.inf
    if sum(len(g) for g in groups) >= 2:
        own = [s for g in groups for s in g if s.var is not None]
        den = sum(s.n - 1 for s in own)
        pooled = sum((s.n - 1) * s.var for s in own if s.var is not None) / den if den > 0 else 0.0
        flat = [s for g in groups for s in g]
        y, v = set_estimates(flat, "normal")
        ys, vs, at = [], [], 0
        for g in groups:
            ys.append(y[at:at + len(g)])
            vs.append(v[at:at + len(g)])
            at += len(g)
        fit = combine_scale_nested(ys, vs, df=1)
        n0_b = pooled / fit.tau2 if fit.tau2 > 0 else math.inf
        n0_s = pooled / fit.tau2_within if fit.tau2_within > 0 else math.inf
    unit_w = nested_set_weights([[float(n) for _, n, _ in units] for _, units in benches], n0_s, n0_b)
    points = sorted((x, uw * w / n) for (_, units), ws in zip(benches, unit_w) for (_, n, pts), uw in zip(units, ws)
                    for x, w in pts)
    med = _weighted_quantile(points, 0.5)
    shares = []
    for _, units in benches:
        g = []
        for _, n, pts in units:
            tot = sum(w for _, w in pts)
            p = min(1.0, sum(w for x, w in pts if x <= med) / tot)
            k = max(1, round(tot))
            g.append(SetStat(k, p * k, p * k))
        shares.append(g)
    c = combine_nested(shares, "normal")
    half = max(0.0, (c.hi - c.lo) / 2) if c is not None else 0.0
    lo = _weighted_quantile(points, max(0.0, 0.5 - half))
    hi = _weighted_quantile(points, min(1.0, 0.5 + half))
    bench_w = [sum(ws) if len(ws) > 1 else ws[0] for ws in unit_w]
    total_w = sum(bench_w)
    return Average(med, lo, hi, None, None, sum(n for _, units in benches for _, n, _ in units), len(benches),
                   {b: w / total_w for (b, _), w in zip(benches, bench_w)}, sum(len(units) for _, units in benches),
                   {b: {s: w / total_w for (s, _, _), w in zip(units, ws)} for (b, units), ws in zip(benches, unit_w)})


def average(problems: Mapping[str, Sequence[Sequence[object]]], kind: str = "mean", *,
            benchmarks: Mapping[str, str] | None = None,
            subsets: Mapping[str, Sequence[str | None]] | None = None) -> Average | None:
    """The average over benchmarks of per-problem run values: ``problems[catalog]`` lists, for each problem of the
    catalog, the values of its runs. ``benchmarks`` ({catalog: benchmark}) names each catalog's benchmark (a catalog it
    does not name is a benchmark of its own), and ``subsets`` ({catalog: [subset of each problem]}) each problem's
    subset within it (by default its catalog): benchmarks > subsets > problems (:func:`nest`), averaged by the
    three-level model (:func:`combine_nested`). The weights are keyed by benchmark, the subset weights by benchmark and
    subset. ``kind``:

    * ``"mean"``: the random-effects average of the subsets' means (a problem's value is the mean of its runs' finite
      values; problems without one are left out);
    * ``"rate"``: the beta-binomial average (a run hits or misses; a problem's value is its share of hits);
    * ``"ratio"``: the mean in log2, shown back on the ratio's scale (a geometric mean);
    * ``"median"``: the median with each subset weighted as the average weighs it, and Woodruff's interval.

    None when nothing has a value. One benchmark of one subset is its own mean (a rate: its plain rate with the Wilson
    interval); without ``benchmarks`` and ``subsets`` every catalog is a benchmark of one subset, the two-level average."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, not {kind!r}")
    nested = nest(problems, benchmarks, subsets)
    if kind == "median":
        return _median(nested)
    names, groups = _nested_stats(nested, kind)
    if not groups:
        return None
    c = combine_nested(groups, "logit" if kind == "rate" else "normal")
    assert c is not None and c.subset_weights is not None
    back = (lambda x: 2.0 ** x) if kind == "ratio" else (lambda x: x)
    return Average(back(c.mu), back(c.lo), back(c.hi), None if c.pi_lo is None else back(c.pi_lo),
                   None if c.pi_hi is None else back(c.pi_hi), int(sum(s.n for g in groups for s in g)), len(groups),
                   {b: w for (b, _), w in zip(names, c.weights)}, sum(len(g) for g in groups),
                   {b: dict(zip(subs, ws)) for (b, subs), ws in zip(names, c.subset_weights)})
