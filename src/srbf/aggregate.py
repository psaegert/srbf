"""Averages over problem sets: the averaging of srbf's results site and of ``srbf analyze``.

Results are organised as problem sets (catalogs) > problems > runs. A problem's value is the mean of its runs, and a
set is summarised by the number of its problems with a value, their mean and their variance. The sets are combined by
a random-effects average (DerSimonian and Laird 1986): every set has its own true value, and the sets are weighted by
1 / (tau^2 + the set's own sampling variance), with the between-set variance tau^2 by Paule and Mandel's estimator.
Rates (shares of problems solved) are combined by the beta-binomial model (Skellam 1948): a set's sum of problem
values is binomial with the set's rate, and the rates follow a beta distribution whose mean is the average; its
likelihood is closed-form, so sets at 0 % or 100 % enter as they are. Ratios are averaged in log2 (geometric means),
and medians weigh each set as the average does, with Woodruff's interval.

Every interval covers these problem sets: the weights (for rates, the beta's concentration) are held at their
estimates and only the sampling of problems within each set counts, with t on the problems' degrees of freedom,
sum (n - 1). Where one more set would fall is the prediction range (Higgins, Thompson and Spiegelhalter 2009), for
rates the fitted beta's 2.5 % to 97.5 %. One set alone is its own mean with the t interval over its problems, a rate
its plain rate with the Wilson interval. docs/averaging.md has the formulas, the reasons and the sources.

results-site/explorer_v2.js implements the same arithmetic step for step (the t distribution below is the same
continued fraction); scripts/site_random_effects.py loads this file for the site's scripts, and tests and the site
suite check that the page and this module agree. :func:`average` is the entry point from per-problem run values;
:func:`combine` works from per-set sums.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

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
    tau^2, the normalised weights, the number of sets and the two-sided p-value of mu = 0 (for differences)."""
    mu: float
    lo: float
    hi: float
    pi_lo: float | None
    pi_hi: float | None
    tau2: float
    weights: list[float]
    S: int
    p: float | None


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


def _rate_score(sets: Sequence[SetStat], eta: float, s: float) -> float:
    """The score in eta of all sets, s held fixed: sum_i s q (psi(k_i + a) - psi(a) - psi(n_i - k_i + b) + psi(b)),
    q = m (1 - m). A difference psi(x + c) - psi(c) is exactly 0 for x = 0, where c may underflow."""
    m = 1.0 / (1.0 + math.exp(-eta))
    a, b = m * s, (1.0 - m) * s
    total = 0.0
    for st in sets:
        k, rest = st.s1, st.n - st.s1
        total += (digamma(k + a) - digamma(a) if k > 0 else 0.0) - (digamma(rest + b) - digamma(b) if rest > 0 else 0.0)
    return s * m * (1.0 - m) * total


def _rate_edge(sets: Sequence[SetStat], eta: float, s: float, bound: float, direction: float) -> float:
    """The last eta, going from the estimate in ``direction``, whose score stays within ``bound``: steps that grow by
    half each time until the score leaves it (or eta reaches +-RATE_ETA_MAX, an end of 0 or 1), then bisection."""
    def inside(e: float) -> bool:
        u = _rate_score(sets, e, s)
        return math.isfinite(u) and abs(u) <= bound
    last, step = eta, 0.25
    while True:
        e = last + direction * step
        if abs(e) >= RATE_ETA_MAX:
            e = direction * RATE_ETA_MAX
            if inside(e):
                return direction * math.inf
            break
        if not inside(e):
            break
        last, step = e, step * 1.5
    lo, hi = last, e
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if inside(mid):
            lo = mid
        else:
            hi = mid
    return lo


RATE_ETA_MAX = 30.0   # the interval's search ends at expit(+-30), 1e-13 from 0 and 1


def combine_rates(sets: Sequence[SetStat]) -> Combined:
    """Rates over problem sets by the beta-binomial model: the mean rate m of a problem set, on the 0-1 scale.

    One set is its plain rate k / n with the Wilson interval; so is a group in which every set is at 0 % (or 100 %).
    The interval covers these problem sets, as for every other metric: s held at its estimate, the sets' rates fixed,
    only the sampling of problems within each set. eta solves sum_i U_i(eta; k_i) = 0, U_i = s q (psi(k_i + a) - psi(a)
    - psi(n_i - k_i + b) + psi(b)) with q = m (1 - m), and the score's variance from the problems' sampling is
    V = sum_i (dU_i/dk_i)^2 Var(k_i), with dU_i/dk_i = s q (psi'(k_i + a) + psi'(n_i - k_i + b)) and Var(k_i) = n_i p_i
    (1 - p_i) at the set's posterior rate p_i = (k_i + a) / (n_i + s), all at the estimate. The interval is every eta
    whose score stays within t sqrt(V), t on sum (n - 1): the estimating equation inverted, not linearised around the
    estimate, so it stays sensible where many sets sit at 0 % or 100 % (there the linearised, delta-method interval on
    the log-odds reached 0 % to 100 % for one success in 6,660 problems). Where sets sit at both 0 % or 100 % and far from
    them, the fitted beta is U-shaped and its mean responds little to the data, so the interval can still be wide. The
    range for one more set is the 2.5 % to 97.5 % quantiles of Beta(a, b). The weights are each set's share of the information about eta; tau2 carries 1 / s
    (0 for the pooled rate)."""
    S = len(sets)
    N = sum(s.n for s in sets)
    K = sum(s.s1 for s in sets)
    eta, lam = rate_fit(sets) if S > 1 else (0.0, 0.0)
    if S == 1 or not math.isfinite(eta):
        lo, hi = wilson(K, N)
        return Combined(K / N, lo, hi, None, None, 0.0, [s.n / N for s in sets], S, None)
    pooled = not math.isfinite(lam)
    m = 1.0 / (1.0 + math.exp(-eta))
    s = math.exp(min(lam, LAM_MAX))
    a, b, q = m * s, (1.0 - m) * s, m * (1.0 - m)
    tg_a, tg_b = trigamma(a), trigamma(b)
    num = 0.0
    info = []
    for st in sets:
        n, k = st.n, st.s1
        tka, tnb = trigamma(k + a), trigamma(n - k + b)
        info.append(-(s * q) ** 2 * (tka - tg_a + tnb - tg_b))
        du_dk = s * q * (tka + tnb)
        p_i = (k + a) / (n + s)
        num += du_dk * du_dk * n * p_i * (1.0 - p_i)
    bound = t_ppf(0.975, max(1, sum(st.n - 1 for st in sets))) * math.sqrt(num)
    lo_eta, hi_eta = _rate_edge(sets, eta, s, bound, -1.0), _rate_edge(sets, eta, s, bound, 1.0)
    pi_lo = pi_hi = None
    if S >= 3:
        pi_lo, pi_hi = beta_ppf(0.025, a, b), beta_ppf(0.975, a, b)
    tot = sum(info)
    return Combined(m, 0.0 if lo_eta == -math.inf else _sig(lo_eta), 1.0 if hi_eta == math.inf else _sig(hi_eta),
                    pi_lo, pi_hi, 0.0 if pooled else 1.0 / s, [x / tot for x in info], S, None)


def combine(sets: Sequence[SetStat], scale: str = "normal") -> Combined | None:
    """Combine problem sets; a rate (scale "logit") comes back on its own 0-1 scale (combine_rates)."""
    sets = [s for s in sets if s.n > 0]
    if not sets:
        return None
    if scale == "logit":
        return combine_rates(sets)
    y, v = set_estimates(sets, scale)
    return combine_scale(y, v, sum(s.n - 1 for s in sets))


def holm(pvalues: Iterable[float]) -> list[float]:
    """Holm-adjusted p-values, in the input order."""
    ps = list(pvalues)
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        out[i] = running
    return out


# ---- averages from per-problem run values ----------------------------------------------------------------------------
KINDS = ("mean", "rate", "ratio", "median")


@dataclass(frozen=True)
class Average:
    """An average over problem sets: the value and its 95 % interval over these sets, the range for one more set (None
    below three sets, and for a median), the problems and sets behind it, and each set's share of the weight."""
    value: float
    lo: float
    hi: float
    pi_lo: float | None
    pi_hi: float | None
    n_problems: int
    n_sets: int
    weights: dict[str, float]


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


def _weighted_quantile(points: Sequence[tuple[float, float]], q: float) -> float:
    """The smallest value whose cumulative weight reaches q of the total (points sorted by value)."""
    total = sum(w for _, w in points)
    cum = 0.0
    for x, w in points:
        cum += w
        if cum >= q * total - 1e-12 * total:
            return x
    return points[-1][0]


def _median(problems: Mapping[str, Sequence[Sequence[object]]]) -> Average | None:
    """The median over problem sets: every problem weighs 1 within its set, shared by its runs, and a set of n problems
    weighs n / (n + n0), n0 = pooled within-set variance / tau^2 of the mean's random-effects fit (a set counts about once
    whatever its size, a set of a few problems less), as the average weighs it. Non-finite values count at their end of
    the scale. The interval is Woodruff's (1952): each set's share of weight at or below the median is averaged over the
    sets like any value, and that share's interval is read back through the weighted distribution."""
    sets: list[tuple[str, int, list[tuple[float, float]]]] = []
    for name, runs_of in problems.items():
        pts: list[tuple[float, float]] = []
        n = 0
        for runs in runs_of:
            vals = [x for x in (_number(v) for v in runs) if x is not None]
            if vals:
                n += 1
                pts += [(x, 1.0 / len(vals)) for x in vals]
        if n:
            sets.append((name, n, pts))
    if not sets:
        return None
    _, stats = _set_stats({name: problems[name] for name, _, _ in sets}, "mean")
    n0 = math.inf
    if len(stats) >= 2:
        own = [s for s in stats if s.var is not None]
        den = sum(s.n - 1 for s in own)
        pooled = sum((s.n - 1) * s.var for s in own if s.var is not None) / den if den > 0 else 0.0
        tau2 = combine_scale(*set_estimates(stats, "normal"), df=1).tau2
        n0 = pooled / tau2 if tau2 > 0 else math.inf
    set_w = {name: (n / (n + n0) if math.isfinite(n0) else float(n)) for name, n, _ in sets}
    points = sorted((x, set_w[name] * w / n) for name, n, pts in sets for x, w in pts)
    med = _weighted_quantile(points, 0.5)
    shares = []
    for name, n, pts in sets:
        tot = sum(w for _, w in pts)
        p = min(1.0, sum(w for x, w in pts if x <= med) / tot)
        k = max(1, round(tot))
        shares.append(SetStat(k, p * k, p * k))
    c = combine(shares, "normal")
    half = max(0.0, (c.hi - c.lo) / 2) if c is not None else 0.0
    lo = _weighted_quantile(points, max(0.0, 0.5 - half))
    hi = _weighted_quantile(points, min(1.0, 0.5 + half))
    total_w = sum(set_w.values())
    return Average(med, lo, hi, None, None, sum(n for _, n, _ in sets), len(sets),
                   {name: w / total_w for name, w in set_w.items()})


def average(problems: Mapping[str, Sequence[Sequence[object]]], kind: str = "mean") -> Average | None:
    """The average over problem sets of per-problem run values: ``problems[set]`` lists, for each problem of the set,
    the values of its runs. ``kind``:

    * ``"mean"``: the random-effects average of the sets' means (a problem's value is the mean of its runs' finite
      values; problems without one are left out);
    * ``"rate"``: the beta-binomial average (a run hits or misses; a problem's value is its share of hits);
    * ``"ratio"``: the mean in log2, shown back on the ratio's scale (a geometric mean);
    * ``"median"``: the median with each set weighted as the average weighs it, and Woodruff's interval.

    None when no set has a value. One set alone is its own mean (a rate: its plain rate with the Wilson interval)."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, not {kind!r}")
    if kind == "median":
        return _median(problems)
    names, stats = _set_stats(problems, kind)
    if not stats:
        return None
    c = combine(stats, "logit" if kind == "rate" else "normal")
    assert c is not None
    back = (lambda x: 2.0 ** x) if kind == "ratio" else (lambda x: x)
    return Average(back(c.mu), back(c.lo), back(c.hi), None if c.pi_lo is None else back(c.pi_lo),
                   None if c.pi_hi is None else back(c.pi_hi), sum(s.n for s in stats), len(stats),
                   dict(zip(names, c.weights)))
