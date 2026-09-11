"""Tests for srbf.derive_metrics -- the clean standardized second stage (raw snapshot -> derived
metrics), the ergonomic wrapper over compute_derived_metrics."""
import numpy as np
import pytest

from srbf import derive_metrics, bootstrap_report
from symbolic_data.token_ops import normalize_skeleton


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


def test_r2_is_one_minus_fvu_clipped():
    snapshot = _raw_snapshot()
    derived = derive_metrics(snapshot, operator_arity=ARITY)
    r2 = np.asarray(derived['r2_val'], dtype=float)
    fvu = np.asarray(derived['fvu_val'], dtype=float)
    assert r2.shape == fvu.shape
    assert np.allclose(r2, np.clip(1.0 - fvu, 0.0, 1.0))
    assert r2[0] == 1.0 and r2[2] == 1.0      # the perfect fits
    assert 0.0 <= r2[1] < 1.0                  # the offset fit explains some variance, not all


def _mdl_snapshot():
    snapshot = _raw_snapshot()
    snapshot['predicted_expression_prefix'] = [['add', 'x1', 'x1'], ['mul', '2.5', 'x1'], ['add', '<constant>', 'x1'], None]
    snapshot['ground_truth_prefix'] = [['add', 'x1', 'x1']] * 4
    return snapshot


def test_mdl_ratio_prices_realized_expressions_only():
    priced = []

    def mdl_fn(tokens):
        priced.append(list(tokens))
        return 1000.0 * len(tokens)

    scored = derive_metrics(_mdl_snapshot(), operator_arity=ARITY, mdl_fn=mdl_fn)
    assert list(scored['ground_truth_mdl']) == [3000.0] * 4
    assert list(scored['predicted_mdl']) == [3000.0, 3000.0, None, None]  # a <constant> placeholder and a missing prefix are not priced
    assert list(scored['mdl_ratio']) == [1.0, 1.0, None, None]
    assert ['<constant>' in t for t in priced] == [False] * len(priced)


def test_mdl_ratio_absent_without_a_pricer_and_inf_on_failure():
    scored = derive_metrics(_mdl_snapshot(), operator_arity=ARITY)
    assert 'mdl_ratio' not in scored

    def failing(tokens):
        raise RuntimeError("unpriceable")

    scored = derive_metrics(_mdl_snapshot(), operator_arity=ARITY, mdl_fn=failing)
    assert list(scored['mdl_ratio']) == [None] * 4


def test_mdl_ratio_through_the_engine():
    from simplipy import SimpliPyEngine

    engine = SimpliPyEngine.load("acj-4-3", install=True)
    snapshot = _mdl_snapshot()
    snapshot['skeleton'] = [['*', '<constant>', 'x1']] * 4  # the engine's vocabulary, not the fake arity map's
    snapshot['predicted_skeleton_prefix'] = [['+', 'x1', 'x1'], ['*', '<constant>', 'x1'], ['+', 'x1', 'x1'], ['+', 'x1', 'x1']]
    snapshot['predicted_expression_prefix'] = [['+', 'x1', 'x1'], ['*', '2.5', 'x1'], ['+', 'x1', 'x1'], ['+', 'x1', 'x1']]
    snapshot['ground_truth_prefix'] = [['*', '2', 'x1']] * 4
    scored = derive_metrics(snapshot, engine=engine)
    expected = float(engine.complexity(['*', '2', 'x1'], certified=True, mode='f64', canon='default'))
    assert list(scored['ground_truth_mdl']) == [expected] * 4
    assert scored['mdl_ratio'][1] > scored['mdl_ratio'][0] > 0  # a literal 2.5 costs more than the integer 2


def test_converts_legacy_reader_prefixes_before_judging():
    """Predictions stored through the infix path before 0.15.1 carry the raw reader tokens ('**' for a
    power); with a ``convert_fn`` they are judged and priced in the engine grammar, the input untouched."""
    snapshot = _raw_snapshot()
    snapshot['skeleton'] = [['pow', 'x1', '2']] * 4
    legacy = [['**', 'x1', '2'], ['**', 'x1', '3'], ['pow', 'x1', '2'], ['sin', 'x1']]
    snapshot['predicted_skeleton_prefix'] = [list(p) for p in legacy]
    snapshot['predicted_expression_prefix'] = [list(p) for p in legacy]
    priced: list[list[str]] = []

    def price(tokens):
        priced.append(list(tokens))
        return float(len(tokens))

    scored = derive_metrics(snapshot, operator_arity={**ARITY, 'pow': 2},
                            convert_fn=lambda toks: ['pow' if t == '**' else t for t in toks], mdl_fn=price)
    assert list(scored['symbolic_recovery']) == [True, False, True, False]
    assert list(scored['predicted_skeleton_prefix'][0]) == ['pow', 'x1', '2']
    assert priced and all('**' not in p for p in priced)                 # the MDL price sees the converted prefix
    assert list(snapshot['predicted_skeleton_prefix'][0]) == ['**', 'x1', '2']   # the caller's snapshot is not mutated


def test_judges_the_canonical_form_the_prediction_was_priced_as():
    """A fitted spelling whose factor cancels (`exp(-x^2) * tanh(x)^2 / tanh(x)^2`) is judged as `exp(-x^2)`:
    the exact recovery that flash-ansr < 0.15.2 hid behind the emitted spelling."""
    import numpy as np
    from simplipy import SimpliPyEngine
    engine = SimpliPyEngine.load("acj-4-3", install=True)
    x = np.linspace(-2, 2, 32).reshape(-1, 1)
    y = np.exp(-x[:, 0] ** 2)
    law = ["exp", "neg", "pow", "x1", "<constant>"]
    emitted = "/ * exp neg pow x1 2.0 pow tanh x1 2.0 pow tanh x1 2.0".split()
    snapshot = {
        "expression": [law], "skeleton": [law], "variables": [["x1"]],
        "x": [x], "y": [y], "y_pred": [y.reshape(-1, 1)], "x_val": [x], "y_val": [y], "y_pred_val": [y.reshape(-1, 1)],
        "predicted_expression_prefix": [emitted], "predicted_skeleton_prefix": [normalize_skeleton(emitted)],
        "prediction_success": [True], "fit_time": [1.0],
    }
    scored = derive_metrics(snapshot, engine=engine)
    assert scored["predicted_skeleton_prefix"][0] == engine.simplify(law)
    assert scored["predicted_skeleton_prefix_as_emitted"][0] == normalize_skeleton(emitted)
    assert bool(scored["symbolic_recovery"][0]) is True
