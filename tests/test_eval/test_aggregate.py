"""srbf.aggregate.average: averages over problem sets from per-problem run values, for each kind of value."""
import math

import pytest

from srbf import aggregate as ag


def test_a_problems_value_is_the_mean_of_its_runs():
    assert ag.problem_value([1.0, 3.0]) == 2.0
    assert ag.problem_value([1.0, math.inf, None, float("nan")]) == 1.0      # only finite values count
    assert ag.problem_value([math.inf, None]) is None
    assert ag.problem_value([True, False, None], "rate") == pytest.approx(1 / 3)   # a run without a value misses
    assert ag.problem_value([4.0, 0.25], "ratio") == 0.0                      # log2: the geometric mean is 1


def test_a_mean_is_the_random_effects_average_of_the_sets_means():
    problems = {"a": [[1.0, 1.0], [3.0, math.inf]], "b": [[10.0], [12.0], [14.0]], "c": [[math.inf]]}
    got = ag.average(problems, "mean")
    want = ag.combine([ag.SetStat(2, 4.0, 10.0), ag.SetStat(3, 36.0, 440.0)], "normal")   # c has no finite value
    assert (got.value, got.lo, got.hi) == pytest.approx((want.mu, want.lo, want.hi))
    assert got.n_sets == 2 and got.n_problems == 5


def test_a_rate_is_the_beta_binomial_average():
    problems = {"a": [[1, 1]] * 8, "b": [[0, 0]] * 8, "c": [[1, 0], [1, 1], [0, 0], [1, 1]]}
    got = ag.average(problems, "rate")
    want = ag.combine([ag.SetStat(8, 8.0, 8.0), ag.SetStat(8, 0.0, 0.0), ag.SetStat(4, 2.5, 2.25)], "logit")
    assert (got.value, got.lo, got.hi, got.pi_lo, got.pi_hi) == pytest.approx((want.mu, want.lo, want.hi, want.pi_lo, want.pi_hi))
    one = ag.average({"a": [[1, 1]] * 8}, "rate")
    assert one.value == 1.0 and one.lo == pytest.approx(ag.wilson(8.0, 8.0)[0])     # one set: its plain rate


def test_a_ratio_is_a_geometric_mean():
    got = ag.average({"a": [[2.0]], "b": [[0.5]]}, "ratio")
    assert got.value == pytest.approx(1.0) and got.lo < 1.0 < got.hi


def test_a_median_weighs_each_set_as_the_average_does():
    # a set of 1,000 problems near 0 and two of 20 near 10 and 20: the large set is one third of the weight, not 96 %
    problems = {"big": [[0.001 * i] for i in range(1000)], "mid": [[10.0 + 0.01 * i] for i in range(20)],
                "far": [[20.0 + 0.01 * i] for i in range(20)]}
    got = ag.average(problems, "median")
    assert got.weights["big"] < 0.5
    assert 10.0 <= got.value <= 10.2                   # the middle of the three sets, not inside the large one
    assert got.lo <= got.value <= got.hi and got.pi_lo is None
    # a non-finite value counts at its end of the scale
    worst = ag.average({"a": [[1.0], [2.0], [-math.inf]]}, "median")
    assert worst.value == 1.0


def test_an_unknown_kind_is_refused_and_no_values_give_nothing():
    with pytest.raises(ValueError):
        ag.average({"a": [[1.0]]}, "mode")
    assert ag.average({"a": [[None]]}, "mean") is None


# ---- benchmarks: the catalogs of one benchmark are one problem set ------------------------------------------------
# A benchmark split over several catalogs for hosting or format reasons (a licence repository, fixed-point companions,
# its formulas and its data) is one problem set: its catalogs' problems merge, every problem in it once.
_SPLIT = {"feynman": [[1.0, 1.0], [3.0, 2.0], [0.5]], "feynman-restricted": [[7.0], [9.0, 8.0]],
          "feynman-points": [[2.0], [2.5]], "nguyen": [[10.0], [12.0], [14.0], [11.0]],
          "keijzer": [[0.25, 0.5], [4.0]]}
_RATES = {"feynman": [[1, 1], [0, 1], [0, 0]], "feynman-restricted": [[1], [1, 1]], "feynman-points": [[0], [0]],
          "nguyen": [[1], [0], [1], [1]], "keijzer": [[0, 0], [1]]}
_BENCHMARKS = {"feynman": "feynman", "feynman-restricted": "feynman", "feynman-points": "feynman"}


def _fields(a):
    return (a.value, a.lo, a.hi, a.pi_lo, a.pi_hi, a.n_problems, a.n_sets, a.weights)


@pytest.mark.parametrize("kind", ag.KINDS)
def test_the_catalogs_of_one_benchmark_are_one_problem_set(kind):
    problems = _RATES if kind == "rate" else _SPLIT
    got = ag.average(problems, kind, benchmarks=_BENCHMARKS)
    merged = {"feynman": problems["feynman"] + problems["feynman-restricted"] + problems["feynman-points"],
              "nguyen": problems["nguyen"], "keijzer": problems["keijzer"]}
    assert _fields(got) == _fields(ag.average(merged, kind))
    assert got.n_sets == 3 and set(got.weights) == {"feynman", "nguyen", "keijzer"}
    assert got.n_problems == 13     # every problem once, whichever catalog it is in


def test_a_catalog_without_a_benchmark_is_a_set_of_its_own():
    got = ag.average(_SPLIT, "mean", benchmarks={"feynman-points": "feynman"})
    assert set(got.weights) == {"feynman", "feynman-restricted", "nguyen", "keijzer"}
    assert ag.by_benchmark(_SPLIT, {"feynman-points": "feynman"})["feynman"] == _SPLIT["feynman"] + _SPLIT["feynman-points"]


def test_a_merged_benchmark_is_the_hand_merged_computation():
    """The sets combine() is given, summed by hand: a benchmark's count, sum and sum of squares are its catalogs'."""
    def stat(problems):
        vals = [sum(runs) / len(runs) for runs in problems]
        return ag.SetStat(len(vals), sum(vals), sum(v * v for v in vals))

    feynman = stat(_SPLIT["feynman"] + _SPLIT["feynman-restricted"] + _SPLIT["feynman-points"])
    assert feynman == ag.SetStat(7, 1.0 + 2.5 + 0.5 + 7.0 + 8.5 + 2.0 + 2.5, 1.0 + 6.25 + 0.25 + 49.0 + 72.25 + 4.0 + 6.25)
    want = ag.combine([feynman, stat(_SPLIT["nguyen"]), stat(_SPLIT["keijzer"])], "normal")
    got = ag.average(_SPLIT, "mean", benchmarks=_BENCHMARKS)
    assert (got.value, got.lo, got.hi, got.pi_lo, got.pi_hi) == pytest.approx((want.mu, want.lo, want.hi, want.pi_lo, want.pi_hi))
    merged = ag.merge_sets({c: stat(p) for c, p in _SPLIT.items()}, _BENCHMARKS)
    assert list(merged) == ["feynman", "nguyen", "keijzer"] and merged["feynman"].n == feynman.n
    assert (merged["feynman"].s1, merged["feynman"].s2) == pytest.approx((feynman.s1, feynman.s2))


@pytest.mark.parametrize("kind", ag.KINDS)
def test_every_catalog_its_own_benchmark_changes_nothing(kind):
    problems = _RATES if kind == "rate" else _SPLIT
    plain = ag.average(problems, kind)
    assert _fields(ag.average(problems, kind, benchmarks={c: c for c in problems})) == _fields(plain)
    assert _fields(ag.average(problems, kind, benchmarks={})) == _fields(plain)
    stats = {c: ag.SetStat(len(p), float(len(p)), float(len(p))) for c, p in problems.items()}
    assert ag.merge_sets(stats, {}) == stats
