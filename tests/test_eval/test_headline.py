"""srbf.headline: the front area of a run, the time index over a method's budgets, and their averages."""
import inspect
import math

import numpy as np
import pytest

from srbf import aggregate as ag
from srbf import headline as hl

EPS = np.finfo(np.float64).eps


# ---- the box -----------------------------------------------------------------------------------------------------
def test_the_right_edge_is_26_bits_per_support_point():
    assert hl.BITS_PER_TARGET == 26.0 == 0.5 * math.log2(1 / EPS)
    assert hl.upper_edge_bits(512) == 13_312.0
    assert hl.upper_edge_bits(np.int64(10)) == 260.0
    for bad in (0, -3, 2.5):
        with pytest.raises(ValueError):
            hl.upper_edge_bits(bad)


def test_the_mean_is_written_as_the_shortest_round_trip_decimal():
    assert hl.mean_prefix([1.0, 2.0]) == ["1.5"]
    assert hl.mean_prefix([1.0, 2.0, 4.0]) == ["2.3333333333333335"]
    assert hl.mean_prefix(np.array([[-3.0], [-4.0]])) == ["-3.5"]
    assert hl.mean_prefix([1.0, np.nan, 2.0, np.inf]) == ["1.5"]    # the finite targets
    y = np.random.default_rng(0).normal(size=100)
    (token,) = hl.mean_prefix(y)
    assert float(token) == np.mean(y) and token == repr(float(np.mean(y)))
    with pytest.raises(ValueError):
        hl.mean_prefix([np.nan])
    with pytest.raises(ValueError):
        hl.mean_prefix([1e308, 1e308])         # the mean overflows


class _Engine:
    """Prices a one-token number at 1000 milli-bits per character."""

    def complexity(self, expression, certified=True, mode="f64", canon="default"):
        return 1000 * sum(len(token) for token in expression)


def test_the_left_edge_prices_the_mean_like_a_prediction():
    assert hl.lower_edge_bits([1.0, 2.0], _Engine()) == 3.0               # "1.5"
    assert hl.box_edges([1.0, 2.0, np.nan], _Engine()) == (3.0, 52.0)     # two finite support points


def test_the_box_through_the_engine():
    from simplipy import SimpliPyEngine

    from srbf.pricing import price_bits

    engine = SimpliPyEngine.load("acj-4-3", install=True)
    l_lo, l_hi = hl.box_edges(np.array([3.0, 4.0]), engine)
    assert l_lo == price_bits(engine, ["3.5"]) == pytest.approx(5.585) and l_hi == 52.0


# ---- the front area ----------------------------------------------------------------------------------------------
L_LO, L_HI = 8.0, 8.0 * 2 ** 10            # log2 from 3 to 13: u = (log2 L - 3) / 10


def _bits(u):
    return 2.0 ** (3.0 + 10.0 * np.asarray(u, dtype=float))


def _fvu(v):
    return 10.0 ** ((1.0 - np.asarray(v, dtype=float)) * np.log10(EPS))


def test_the_coordinates():
    u, v = hl.front_coordinates(_bits([0.3, 0.0, 1.0]), _fvu([0.5, 0.0, 1.0]), l_lo=L_LO, l_hi=L_HI)
    assert u == pytest.approx([0.3, 0.0, 1.0]) and v == pytest.approx([0.5, 0.0, 1.0])


def test_one_candidate_dominates_its_corner():
    area = hl.front_area(_bits([0.3]), _fvu([0.5]), l_lo=L_LO, l_hi=L_HI)
    assert area == pytest.approx((1 - 0.3) * (1 - 0.5))
    assert hl.dominated_area([0.2], [0.4]) == pytest.approx(0.8 * 0.6)


def test_two_candidates_dominate_the_union():
    assert hl.dominated_area([0.2, 0.5], [0.6, 0.1]) == pytest.approx(0.8 * 0.4 + 0.5 * 0.9 - 0.5 * 0.4)
    assert hl.dominated_area([0.5, 0.2], [0.1, 0.6]) == pytest.approx(0.57)     # the order does not matter


def test_a_dominated_candidate_adds_nothing():
    base = hl.dominated_area([0.2, 0.5], [0.6, 0.1])
    assert hl.dominated_area([0.2, 0.5, 0.6], [0.6, 0.1, 0.7]) == base
    assert hl.dominated_area([0.2, 0.5, 0.2, 0.5], [0.6, 0.1, 0.6, 0.1]) == base     # duplicates


def test_ties_in_length_keep_the_best_error():
    assert hl.dominated_area([0.2, 0.2, 0.2], [0.9, 0.3, 0.6]) == pytest.approx(0.8 * 0.7)


def _grid_area(u, v, k):
    """The area by counting cells of a k x k grid: correct for points on the grid's lines."""
    i, j = np.meshgrid(np.arange(k) / k, np.arange(k) / k, indexing="ij")
    covered = np.zeros((k, k), dtype=bool)
    for a, b in zip(u, v):
        covered |= (i >= a - 1e-12) & (j >= b - 1e-12)
    return covered.sum() / k ** 2


def test_the_area_agrees_with_counting_cells():
    rng = np.random.default_rng(1)
    for _ in range(50):
        m = int(rng.integers(1, 12))
        u, v = rng.integers(0, 101, m) / 100, rng.integers(0, 101, m) / 100
        assert hl.dominated_area(u, v) == pytest.approx(_grid_area(u, v, 100), abs=1e-12)


def test_adding_candidates_never_lowers_the_area():
    rng = np.random.default_rng(2)
    for _ in range(200):
        bits = 2.0 ** rng.uniform(2, 14, 8)
        fvu = 10.0 ** rng.uniform(-17, 1, 8)
        areas = [hl.front_area(bits[:k], fvu[:k], l_lo=L_LO, l_hi=L_HI) for k in range(9)]
        assert all(b >= a for a, b in zip(areas, areas[1:]))
        assert 0.0 <= areas[-1] <= 1.0


def test_candidates_are_clipped_to_the_box():
    whole = hl.front_area([1.0], [0.0], l_lo=L_LO, l_hi=L_HI)        # shorter than the mean and a perfect fit
    assert whole == 1.0
    assert hl.front_area([L_LO], [EPS / 1e10], l_lo=L_LO, l_hi=L_HI) == 1.0    # FVU floored at eps
    assert hl.front_area(_bits([0.3]), [1e-30], l_lo=L_LO, l_hi=L_HI) == pytest.approx(0.7)
    assert hl.front_area([4.0], _fvu([0.5]), l_lo=L_LO, l_hi=L_HI) == pytest.approx(0.5)   # shorter than L_lo: u = 0


def test_candidates_at_or_beyond_the_edges_dominate_nothing():
    for bits, fvu in [(L_HI, EPS), (L_HI * 4, EPS), (L_LO, 1.0), (L_LO, 3.0)]:
        assert hl.front_area([bits], [fvu], l_lo=L_LO, l_hi=L_HI) == 0.0
    inside = hl.front_area(_bits([0.3]), _fvu([0.5]), l_lo=L_LO, l_hi=L_HI)
    assert hl.front_area(np.append(_bits([0.3]), [L_HI, L_LO]), np.append(_fvu([0.5]), [EPS, 1.0]),
                         l_lo=L_LO, l_hi=L_HI) == inside


def test_candidates_without_a_finite_price_or_error_are_left_out():
    inside = hl.front_area(_bits([0.3]), _fvu([0.5]), l_lo=L_LO, l_hi=L_HI)
    bits = np.append(_bits([0.3]), [np.nan, 20.0, np.inf, 20.0])
    fvu = np.append(_fvu([0.5]), [EPS, np.nan, EPS, np.inf])
    assert hl.front_area(bits, fvu, l_lo=L_LO, l_hi=L_HI) == inside


def test_a_run_without_candidates_has_no_area():
    assert hl.front_area([], [], l_lo=L_LO, l_hi=L_HI) == 0.0
    assert hl.front_area([np.nan], [np.nan], l_lo=L_LO, l_hi=L_HI) == 0.0
    assert hl.dominated_area([], []) == 0.0


def test_a_box_without_width_has_no_area():
    assert hl.front_area([1.0], [EPS], l_lo=30.0, l_hi=26.0) == 0.0
    with pytest.raises(ValueError):
        hl.front_coordinates([1.0], [EPS], l_lo=30.0, l_hi=26.0)


def test_bad_inputs_raise():
    for l_lo, l_hi in [(0.0, 10.0), (np.nan, 10.0), (5.0, np.inf), (-1.0, 10.0)]:
        with pytest.raises(ValueError):
            hl.front_area([6.0], [0.1], l_lo=l_lo, l_hi=l_hi)
    with pytest.raises(ValueError):
        hl.front_area([6.0, 7.0], [0.1], l_lo=L_LO, l_hi=L_HI)


# ---- over time ---------------------------------------------------------------------------------------------------
def test_a_budgets_time_is_the_geometric_mean():
    assert hl.budget_time([0.1, 10.0]) == pytest.approx(1.0)
    assert hl.budget_time([2.0, 8.0]) == pytest.approx(4.0)
    assert hl.budget_time([3.0]) == pytest.approx(3.0)
    for bad in ([], [0.0, 1.0], [-1.0], [np.nan, 1.0], [np.inf]):
        with pytest.raises(ValueError):
            hl.budget_time(bad)


@pytest.mark.parametrize("t1", [1e-3, 0.05, 1.0, 37.0, 999.0])
def test_a_constant_curve_from_t1(t1):
    assert hl.time_index([1], [t1], [0.4]) == pytest.approx(0.4 * (3 - math.log10(t1)) / 6)
    assert hl.time_index([1, 2, 4], [t1, 2 * t1, 4 * t1], [0.4] * 3) == pytest.approx(0.4 * (3 - math.log10(t1)) / 6)


def test_a_linear_ramp_over_the_whole_range():
    assert hl.time_index([1, 2], [1e-3, 1e3], [0.0, 1.0]) == pytest.approx(0.5)


def _q(tau, taus, areas):
    """The front area over time, written out: 0 before the first budget, linear between, the last area after."""
    return np.where(tau < taus[0], 0.0, np.interp(tau, taus, areas))


def test_the_closed_form_matches_a_fine_numerical_integral():
    rng = np.random.default_rng(3)
    grid = np.linspace(-3.0, 3.0, 600_001)
    for _ in range(20):
        k = int(rng.integers(1, 7))
        times = np.sort(10.0 ** rng.uniform(-5, 5, k))
        areas = rng.uniform(0, 1, k)
        q = _q(grid, np.log10(times), areas)
        numeric = np.sum(np.diff(grid) * (q[:-1] + q[1:]) / 2) / 6.0
        assert hl.time_index(np.arange(1, k + 1), times, areas) == pytest.approx(numeric, abs=2e-5)


def test_budgets_are_cut_at_one_millisecond():
    # tau = -4 (0.2) and -2 (0.4): q(-3) = 0.3, then 0.4 to the end
    assert hl.time_index([1, 2], [1e-4, 1e-2], [0.2, 0.4]) == pytest.approx(((0.3 + 0.4) / 2 * 1 + 0.4 * 5) / 6)
    assert hl.time_index([1], [1e-6], [0.7]) == pytest.approx(0.7)


def test_budgets_are_cut_at_1000_seconds():
    # tau = 2 (0.2) and 4 (0.6): q(3) = 0.4
    assert hl.time_index([1, 2], [1e2, 1e4], [0.2, 0.6]) == pytest.approx((0.2 + 0.4) / 2 / 6)


def test_a_run_that_starts_after_1000_seconds_scores_0():
    assert hl.time_index([1, 2], [1.5e3, 1e4], [0.9, 1.0]) == 0.0
    assert hl.time_index([1], [1e3], [1.0]) == 0.0
    assert hl.time_index([], [], []) == 0.0


def test_the_budget_order_rule_reads_budgets_and_times_only():
    assert list(inspect.signature(hl.kept_budgets).parameters) == ["budgets", "times"]
    assert list(hl.kept_budgets([1, 2, 4, 8], [0.5, 0.3, 1.0, 2.0])) == [False, True, True, True]
    assert list(hl.kept_budgets([1, 2, 3], [0.5, 0.9, 0.4])) == [False, False, True]   # every larger budget counts
    assert list(hl.kept_budgets([1, 2], [1.0, 1.0])) == [False, True]                    # slower means longer
    assert list(hl.kept_budgets([8, 1, 4, 2], [2.0, 0.5, 1.0, 0.3])) == [True, False, True, True]   # input order
    assert list(hl.kept_budgets([], [])) == []


def test_an_out_of_order_budget_drops_out_whatever_its_area():
    without = hl.time_index([2, 4, 8], [0.3, 1.0, 2.0], [0.2, 0.3, 0.5])
    for area in (0.0, 0.25, 1.0):
        assert hl.time_index([1, 2, 4, 8], [0.5, 0.3, 1.0, 2.0], [area, 0.2, 0.3, 0.5]) == without


def test_bad_budgets_raise():
    for budgets, times, areas in [([1, 1], [1.0, 2.0], [0.1, 0.2]), ([1, 2], [0.0, 1.0], [0.1, 0.2]),
                                  ([1, 2], [1.0, np.nan], [0.1, 0.2]), ([1, 2], [1.0], [0.1, 0.2]),
                                  ([1, 2], [1.0, 2.0], [0.1]), ([1, 2], [1.0, 2.0], [0.1, np.nan])]:
        with pytest.raises(ValueError):
            hl.time_index(budgets, times, areas)
    with pytest.raises(ValueError):
        hl.time_index([1], [1.0], [0.5], t_min=1.0, t_max=1.0)


def test_the_time_range_is_a_protocol_constant():
    assert (hl.T_MIN, hl.T_MAX) == (1e-3, 1e3)
    # a wider range rescales a curve that starts inside both ranges by the ratio of their widths
    assert hl.time_index([1], [1.0], [0.6], t_min=1e-6, t_max=1e3) == pytest.approx(0.6 * 3 / 9)


# ---- over problems -----------------------------------------------------------------------------------------------
def test_the_average_is_the_random_effects_mean_over_problem_sets():
    values = {"a": [0.2, 0.5], "b": [0.1, 0.3, np.float64(0.7)]}      # one run per problem
    got = hl.average_over_problems(values)
    want = ag.combine([ag.SetStat(len(m), sum(m), sum(x * x for x in m)) for m in values.values()], "normal")
    assert (got.value, got.lo, got.hi) == pytest.approx((want.mu, want.lo, want.hi))
    assert got == ag.average({k: [[x] for x in v] for k, v in values.items()}, "mean")
    assert got.n_problems == 5 and got.n_sets == 2
    assert hl.average_over_problems({}) is None


def test_a_problem_with_several_runs_counts_with_their_mean():
    several = hl.average_over_problems({"a": [[0.1, 0.3], 0.5], "b": [np.array([0.0, 0.2]), 0.3, 0.7]})
    one = hl.average_over_problems({"a": [0.2, 0.5], "b": [0.1, 0.3, 0.7]})
    assert (several.value, several.lo, several.hi) == pytest.approx((one.value, one.lo, one.hi))
    assert several.n_problems == 5
