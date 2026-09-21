"""R^2 is 1 - FVU under FVU's numerics, with no lower bound."""
import numpy as np

from srbf.metrics.numeric import fvu, r2


def test_perfect_fit_reads_exactly_one():
    y = np.array([1.0, 2.0, 3.5, -4.0])
    assert r2(y, y.copy()) == 1.0


def test_matches_one_minus_fvu_for_an_ordinary_fit():
    rng = np.random.default_rng(0)
    y = rng.normal(size=200)
    y_pred = y + 0.3 * rng.normal(size=200)
    assert np.isclose(r2(y, y_pred), 1.0 - fvu(y, y_pred))
    assert 0.0 < r2(y, y_pred) < 1.0


def test_the_mean_predictor_reads_zero():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r2(y, np.full(4, y.mean())) == 0.0


def test_an_answer_worse_than_the_mean_is_negative_without_a_floor():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    wild = np.array([40.0, -30.0, 20.0, -10.0])
    assert r2(y, wild) == 1.0 - fvu(y, wild) < -100.0
    assert r2(y, 1e6 * wild) < r2(y, wild)                        # and a wilder one is lower still


def test_a_non_finite_or_missing_prediction_has_no_finite_value():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r2(y, None) == -np.inf
    assert r2(y, np.array([1.0, np.nan, 3.0, 4.0])) == -np.inf
