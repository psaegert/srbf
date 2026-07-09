"""Reference-relative recovery (WP7 P0): endpoint identity + real-data threshold semantics."""
import numpy as np

from srbf.result_processing import compute_derived_metrics


def _snapshot(y, y_pred, y_ref=None):
    n = len(y)
    r = {"y": y, "y_pred": y_pred, "y_val": y, "y_pred_val": y_pred,
         "skeleton": [None] * n, "predicted_skeleton_prefix": [None] * n,
         "prediction_success": [True] * n}
    if y_ref is not None:
        r["y_ref"] = y_ref
        r["y_ref_val"] = y_ref
    return {"m": {"results": {"t": {0: r}}}}


def _col(results, key):
    return results["m"]["results"]["t"][0][key]


def test_synthetic_endpoint_identity():
    """Without reference columns nothing changes; WITH y_ref == clean y (the synthetic
    default), reference_fvu is exactly 0 and relative recovery == machine-precision recovery
    elementwise, across exact fits, near-misses and clear misses."""
    y = [np.linspace(1, 2, 64).reshape(-1, 1).astype(np.float32) for _ in range(3)]
    y_pred = [y[0].copy(),                                   # exact fit
              y[1] * (1 + 2e-4),                             # near miss (fvu just above eps)
              y[2] + 1.0]                                    # clear miss
    results = _snapshot(y, y_pred, y_ref=[a.copy() for a in y])
    compute_derived_metrics(results, operator_arity={}, test_sets=["t"])
    r = results["m"]["results"]["t"][0]
    assert np.allclose(r["reference_fvu_fit"], 0.0)
    np.testing.assert_array_equal(r["numeric_recovery_relative_fit"], r["numeric_recovery_fit"])
    np.testing.assert_array_equal(r["numeric_recovery_relative_val"], r["numeric_recovery_val"])

    plain = _snapshot(y, y_pred)                             # no y_ref columns at all
    compute_derived_metrics(plain, operator_arity={}, test_sets=["t"])
    assert "reference_fvu_fit" not in plain["m"]["results"]["t"][0]


def test_real_data_reference_threshold():
    """A candidate at least as good as the reference law counts as relative recovery even far
    from machine precision; a worse-than-reference candidate does not."""
    rng = np.random.default_rng(0)
    x = np.linspace(0, 1, 128).reshape(-1, 1).astype(np.float32)
    y = (2 * x + rng.normal(0, 0.1, x.shape)).astype(np.float32)   # measured data
    reference = 2 * x                                              # the accepted law
    better = 2 * x + 0.01 * np.sin(9 * x)                          # closer to y than the law? keep simple:
    results = _snapshot([y, y], [reference.copy(), (y + 5.0)], y_ref=[reference, reference])
    compute_derived_metrics(results, operator_arity={}, test_sets=["t"])
    r = results["m"]["results"]["t"][0]
    assert r["reference_fvu_fit"][0] > np.finfo(np.float32).eps    # noise floor: law != data
    assert bool(r["numeric_recovery_relative_fit"][0]) is True     # candidate == the law itself
    assert bool(r["numeric_recovery_fit"][0]) is False             # but NOT machine-precision
    assert bool(r["numeric_recovery_relative_fit"][1]) is False    # far worse than the law
