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


# ---- benchmarks > subsets: the catalogs of one benchmark are its subsets --------------------------------------------
# A benchmark split over several catalogs for hosting or format reasons (a licence repository, fixed-point companions,
# its formulas and its data) has them as subsets: problems within subsets within benchmarks, averaged by the
# three-level model; every problem counts once, and the weights are keyed by benchmark.
_SPLIT = {"feynman": [[1.0, 1.0], [3.0, 2.0], [0.5]], "feynman-restricted": [[7.0], [9.0, 8.0]],
          "feynman-points": [[2.0], [2.5]], "nguyen": [[10.0], [12.0], [14.0], [11.0]],
          "keijzer": [[0.25, 0.5], [4.0]]}
_RATES = {"feynman": [[1, 1], [0, 1], [0, 0]], "feynman-restricted": [[1], [1, 1]], "feynman-points": [[0], [0]],
          "nguyen": [[1], [0], [1], [1]], "keijzer": [[0, 0], [1]]}
_BENCHMARKS = {"feynman": "feynman", "feynman-restricted": "feynman", "feynman-points": "feynman"}


def _fields(a):
    return (a.value, a.lo, a.hi, a.pi_lo, a.pi_hi, a.n_problems, a.n_sets, a.weights)


def _stat(problems, kind="mean"):
    vals = [ag.problem_value(runs, kind) for runs in problems]
    vals = [v for v in vals if v is not None]
    return ag.SetStat(len(vals), float(sum(vals)), float(sum(v * v for v in vals)))


@pytest.mark.parametrize("kind", ag.KINDS)
def test_the_catalogs_of_one_benchmark_are_its_subsets(kind):
    problems = _RATES if kind == "rate" else _SPLIT
    got = ag.average(problems, kind, benchmarks=_BENCHMARKS)
    assert got.n_sets == 3 and got.n_subsets == 5 and set(got.weights) == {"feynman", "nguyen", "keijzer"}
    assert set(got.subset_weights["feynman"]) == {"feynman", "feynman-restricted", "feynman-points"}
    assert got.n_problems == 13     # every problem once, whichever catalog it is in
    for bench, w in got.weights.items():
        assert sum(got.subset_weights[bench].values()) == pytest.approx(w)
    assert sum(got.weights.values()) == pytest.approx(1.0)


@pytest.mark.parametrize("kind", ["mean", "rate", "ratio"])
def test_a_split_benchmark_is_the_hand_nested_computation(kind):
    """The groups combine_nested() is given, built by hand: one group per benchmark, one SetStat per catalog."""
    problems = _RATES if kind == "rate" else _SPLIT
    groups = [[_stat(problems[c], kind) for c in ("feynman", "feynman-restricted", "feynman-points")],
              [_stat(problems["nguyen"], kind)], [_stat(problems["keijzer"], kind)]]
    want = ag.combine_nested(groups, "logit" if kind == "rate" else "normal")
    got = ag.average(problems, kind, benchmarks=_BENCHMARKS)
    back = (lambda x: 2.0 ** x) if kind == "ratio" else (lambda x: x)
    assert (got.value, got.lo, got.hi) == (back(want.mu), back(want.lo), back(want.hi))
    assert list(got.weights.values()) == want.weights
    assert ag.nest_sets({c: _stat(p, kind) for c, p in problems.items()}, _BENCHMARKS) == groups


def test_a_catalog_without_a_benchmark_is_a_benchmark_of_its_own():
    got = ag.average(_SPLIT, "mean", benchmarks={"feynman-points": "feynman"})
    assert set(got.weights) == {"feynman", "feynman-restricted", "nguyen", "keijzer"}
    assert set(got.subset_weights["feynman"]) == {"feynman", "feynman-points"}
    assert ag.by_benchmark(_SPLIT, {"feynman-points": "feynman"})["feynman"] == _SPLIT["feynman"] + _SPLIT["feynman-points"]
    assert list(ag.merge_sets({c: _stat(p) for c, p in _SPLIT.items()}, _BENCHMARKS)) == ["feynman", "nguyen", "keijzer"]


def test_problems_carry_their_own_subsets():
    """A problem's subset is named per problem (the authors' part of the benchmark); a catalog without names, or a
    problem with an empty one, is in the subset named after its catalog, and one subset name in two catalogs of a
    benchmark is one subset."""
    nested = ag.nest({"srsd": [[1.0], [2.0], [3.0], [4.0]], "srsd-points": [[5.0]], "other": [[6.0]]},
                     {"srsd": "srsd", "srsd-points": "srsd"},
                     {"srsd": ["easy", "hard", None, "easy"], "srsd-points": ["hard"]})
    assert nested == {"srsd": {"easy": [[1.0], [4.0]], "hard": [[2.0], [5.0]], "srsd": [[3.0]]}, "other": {"other": [[6.0]]}}
    with pytest.raises(ValueError, match="2 subsets for 3 problems"):
        ag.nest({"a": [[1.0], [2.0], [3.0]]}, None, {"a": ["x", "y"]})
    got = ag.average({"a": [[1.0], [2.0], [9.0], [10.0]]}, "mean", subsets={"a": ["lo", "lo", "hi", "hi"]})
    assert got.n_sets == 1 and got.n_subsets == 2 and set(got.subset_weights["a"]) == {"lo", "hi"}


def test_two_level_results_stay_bit_for_bit():
    """Every catalog a benchmark of one subset: the values the two-level code (before subsets) gave, to the last bit."""
    P = {"a": [[0.1 * i, 0.1 * i + 0.3] for i in range(12)], "b": [[2.0 + 0.5 * (i % 3)] for i in range(5)],
         "c": [[-1.0], [4.0], [math.inf]], "d": [[7.5, 7.0]]}
    R = {"a": [[1, 0], [1, 1], [0, 0], [1, 1], [0, 1]] * 3, "b": [[0]] * 9, "c": [[1], [1], [1], [0]], "d": [[1, 1]]}
    Q = {"a": [[2.0 ** (0.3 * i)] for i in range(10)], "b": [[0.5], [0.25], [1.0]], "c": [[4.0, 8.0]]}
    before = {  # srbf.aggregate.average at integration/0.22 (146f1b6d5), every catalog its own benchmark
        "mean": (3.022327559981948, 2.01428201433342, 4.030373105630476, -10.518457934050499, 16.563113054014394, 20, 4,
                 {"a": 0.2904007015373005, "b": 0.28949019034940066, "c": 0.1602644211746331, "d": 0.25984468693866575}),
        "rate": (0.466497567081599, 0.2607543949968242, 0.6770170627699348, 0.007083460624753098, 0.984474891658911, 29, 4,
                 {"a": 0.3697894102581526, "b": 0.16656657021370536, "c": 0.3094301897695006, "d": 0.1542138297586415}),
        "ratio": (1.8457904640856606, 1.0842752029935745, 3.142138110235606, 8.233573942242921e-08, 41378658.41988751, 14, 3,
                  {"a": 0.3707686469381501, "b": 0.33982207710717566, "c": 0.28940927595467425}),
        "median": (2.5, 1.0, 7.0, None, None, 21, 4,
                   {"a": 0.25905678731912213, "b": 0.2555388841061915, "c": 0.251633627845954, "d": 0.23377070072873238}),
    }
    for kind, probs in (("mean", P), ("rate", R), ("ratio", Q), ("median", P)):
        assert _fields(ag.average(probs, kind, benchmarks={c: c for c in probs})) == before[kind]
        assert _fields(ag.average(probs, kind)) == before[kind]


@pytest.mark.parametrize("kind", ag.KINDS)
def test_every_catalog_its_own_benchmark_changes_nothing(kind):
    problems = _RATES if kind == "rate" else _SPLIT
    plain = ag.average(problems, kind)
    assert _fields(ag.average(problems, kind, benchmarks={c: c for c in problems})) == _fields(plain)
    assert _fields(ag.average(problems, kind, benchmarks={})) == _fields(plain)
    stats = {c: ag.SetStat(len(p), float(len(p)), float(len(p))) for c, p in problems.items()}
    assert ag.merge_sets(stats, {}) == stats
