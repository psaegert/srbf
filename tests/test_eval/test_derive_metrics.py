"""Tests for srbf.derive_metrics -- the clean standardized second stage (raw snapshot -> derived
metrics), the ergonomic wrapper over compute_derived_metrics."""
import numpy as np
import pytest

from srbf import derive_metrics, bootstrap_report


def _raw_snapshot():
    """A minimal RAW run snapshot: 4 problems in 2 groups; problems 0 & 2 are perfect fits."""
    n = 8
    y = np.linspace(-1.0, 1.0, n)
    ys, yps, yv, ypv = [], [], [], []
    for perfect in [True, False, True, False]:
        pred = y.copy() if perfect else y + 0.5
        ys.append(y.copy())
        yps.append(pred)
        yv.append(y.copy())
        ypv.append(pred.copy())
    return {
        'y': ys, 'y_pred': yps, 'y_val': yv, 'y_pred_val': ypv,
        'skeleton': [['add', 'x1', 'x1']] * 4,
        'predicted_skeleton_prefix': [['add', 'x1', 'x1'], ['mul', 'x1', 'x1'], ['add', 'x1', 'x1'], ['sin', 'x1']],
        'benchmark_eq_id': ['A', 'A', 'B', 'B'],
        'placeholder': [False] * 4,
    }


ARITY = {'add': 2, 'mul': 2, 'sin': 1}


def test_adds_derived_columns_without_mutating_input():
    snapshot = _raw_snapshot()
    scored = derive_metrics(snapshot, operator_arity=ARITY)

    # derived columns are present on the returned snapshot ...
    for key in ('fvu_val', 'numeric_recovery_val', 'symbolic_recovery', 'f1_score'):
        assert key in scored
    # ... and the caller's snapshot is NOT mutated
    assert 'numeric_recovery_val' not in snapshot
    # problems 0 & 2 recover symbolically (predicted == ground-truth skeleton), 1 & 3 do not
    assert list(scored['symbolic_recovery']) == [True, False, True, False]


def test_composes_with_bootstrap_report():
    scored = derive_metrics(_raw_snapshot(), operator_arity=ARITY)
    report = bootstrap_report(scored, 'numeric_recovery_val', n=200)
    assert report['metric'] == 'numeric_recovery_val'
    assert report['n_groups'] == 2 and report['n_rows'] == 4
    for bound in ('median', 'ci_lower', 'ci_upper'):
        assert 0.0 <= report[bound] <= 1.0


def test_requires_engine_or_operator_arity():
    with pytest.raises(ValueError, match="engine.*operator_arity|operator_arity"):
        derive_metrics(_raw_snapshot())
