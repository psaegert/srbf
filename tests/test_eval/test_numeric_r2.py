"""R^2 is 1 - FVU under FVU's numerics, clipped to [0, 1], 0 for a failed prediction."""
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


def test_worse_than_the_mean_and_failures_clip_to_zero():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r2(y, np.array([40.0, -30.0, 20.0, -10.0])) == 0.0   # FVU > 1
    assert r2(y, None) == 0.0                                    # no prediction
    assert r2(y, np.array([1.0, np.nan, 3.0, 4.0])) == 0.0       # non-finite prediction
