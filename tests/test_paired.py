"""Paired arm contrasts must report what MOVED, not just which percentage is bigger."""
import numpy as np
import pytest

from srbf.metrics import mcnemar_exact, paired_difference_ci, wilson_interval


class TestMcNemarExact:
    def test_reproduces_the_textbook_tail(self) -> None:
        # 13 vs 4 discordant -> exact two-sided binomial p (the shape of the T16/v23 headline).
        a = np.zeros(110, dtype=bool)
        b = np.zeros(110, dtype=bool)
        a[:4] = True
        b[20:33] = True
        r = mcnemar_exact(a, b)
        assert (r.n_a_only, r.n_b_only) == (4, 13)
        assert r.n_discordant == 17
        assert r.p_value == pytest.approx(0.04904, abs=1e-5)

    def test_identical_arms_cannot_be_distinguished(self) -> None:
        a = np.array([True, False, True, True])
        r = mcnemar_exact(a, a.copy())
        assert r.p_value == 1.0
        assert r.n_discordant == 0

    def test_a_one_problem_difference_is_never_significant(self) -> None:
        # The compaction claim's shape: a handful of problems moving each way.
        a = np.zeros(110, dtype=bool)
        b = np.zeros(110, dtype=bool)
        a[:4] = True
        b[50:53] = True
        r = mcnemar_exact(a, b)
        assert (r.n_a_only, r.n_b_only) == (4, 3)
        assert r.p_value == 1.0

    def test_a_total_sweep_is_significant(self) -> None:
        a = np.zeros(20, dtype=bool)
        b = np.ones(20, dtype=bool)
        assert mcnemar_exact(a, b).p_value < 1e-5

    def test_the_table_totals_the_problem_set(self) -> None:
        rng = np.random.default_rng(3)
        a, b = rng.random(64) < 0.3, rng.random(64) < 0.4
        r = mcnemar_exact(a, b)
        assert r.n_a_only + r.n_b_only + r.n_both + r.n_neither == 64

    def test_symmetry(self) -> None:
        rng = np.random.default_rng(5)
        a, b = rng.random(50) < 0.2, rng.random(50) < 0.5
        assert mcnemar_exact(a, b).p_value == mcnemar_exact(b, a).p_value

    def test_unaligned_arms_are_refused(self) -> None:
        with pytest.raises(ValueError, match="one outcome per problem"):
            mcnemar_exact(np.array([True, False]), np.array([True]))


class TestWilsonInterval:
    def test_brackets_the_point_estimate(self) -> None:
        lo, hi = wilson_interval(15, 110)
        assert lo < 15 / 110 < hi

    def test_never_leaves_the_unit_interval(self) -> None:
        assert wilson_interval(0, 20)[0] == 0.0
        assert wilson_interval(20, 20)[1] == 1.0

    def test_no_trials_is_no_information(self) -> None:
        assert wilson_interval(0, 0) == (0.0, 1.0)

    def test_tightens_with_sample_size(self) -> None:
        narrow = wilson_interval(150, 1100)
        wide = wilson_interval(15, 110)
        assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


class TestPairedDifferenceCI:
    def test_a_real_difference_excludes_zero(self) -> None:
        a = np.zeros(200, dtype=bool)
        b = np.zeros(200, dtype=bool)
        b[:60] = True
        point, lo, hi = paired_difference_ci(a, b)
        assert point == pytest.approx(0.3)
        assert lo > 0

    def test_a_noise_difference_includes_zero(self) -> None:
        a = np.zeros(110, dtype=bool)
        b = np.zeros(110, dtype=bool)
        a[:4] = True
        b[50:53] = True
        _, lo, hi = paired_difference_ci(a, b)
        assert lo < 0 < hi

    def test_seeded_by_default_so_a_published_interval_reproduces(self) -> None:
        rng = np.random.default_rng(1)
        a, b = rng.random(80) < 0.2, rng.random(80) < 0.35
        assert paired_difference_ci(a, b) == paired_difference_ci(a, b)

    def test_empty_input_is_a_zero_difference(self) -> None:
        assert paired_difference_ci(np.array([], dtype=bool), np.array([], dtype=bool)) == (0.0, 0.0, 0.0)

    def test_unaligned_arms_are_refused(self) -> None:
        with pytest.raises(ValueError):
            paired_difference_ci(np.array([True]), np.array([True, False]))
