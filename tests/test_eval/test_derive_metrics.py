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


def test_r2_is_one_minus_fvu():
    snapshot = _raw_snapshot()
    snapshot['y_pred_val'][3] = snapshot['y_val'][3] * -40.0       # far worse than predicting the mean
    derived = derive_metrics(snapshot, operator_arity=ARITY)
    r2 = np.asarray(derived['r2_val'], dtype=float)
    fvu = np.asarray(derived['fvu_val'], dtype=float)
    assert r2.shape == fvu.shape
    assert np.allclose(r2, 1.0 - fvu, equal_nan=True)
    assert r2[0] == 1.0 and r2[2] == 1.0      # the perfect fits
    assert 0.0 <= r2[1] < 1.0                  # the offset fit explains some variance, not all
    assert r2[3] < -1000.0                     # no floor


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


def _fake_simplify(tokens):
    """A stand-in for the engine's simplify with the two moves the judge has to survive: a masked
    rational folds to one slot (``/ <constant> <constant>`` -> ``<constant>``) and a unit factor drops
    (``* 1.0 x1`` -> ``x1``); a REALIZED rational (``/ 2 3``) is kept, as the canon keeps it."""
    out = list(tokens)
    for i in range(len(out) - 2):
        if out[i:i + 3] == ['/', '<constant>', '<constant>']:
            return out[:i] + ['<constant>'] + out[i + 3:]
    if out[:2] == ['*', '1.0']:
        return out[2:]
    return out


def test_symbolic_recovery_judges_both_sides_through_the_same_simplify():
    """A prediction byte-identical to the ground truth must be exact. The ground truth's skeleton is
    simplified (``pow x1 / <c> <c>`` -> ``pow x1 <c>``), so a prediction whose canonical form is not
    shorter must have its stored skeleton simplified too, or no law with a rational exponent could be
    judged exact."""
    snapshot = {
        'skeleton': [['pow', 'x1', '/', '<constant>', '<constant>']] * 2 + [['x1']],
        'predicted_expression_prefix': [['pow', 'x1', '/', '2', '3'], ['rootn', 'x1', '3'], ['*', '1.0', 'x1']],
        'predicted_skeleton_prefix': [['pow', 'x1', '/', '<constant>', '<constant>'], ['rootn', 'x1', '<constant>'], ['*', '<constant>', 'x1']],
        'benchmark_eq_id': ['A', 'A', 'B'], 'placeholder': [False] * 3,
    }
    scored = derive_metrics(snapshot, operator_arity={'pow': 2, 'rootn': 2, '/': 2, '*': 2}, simplify_fn=_fake_simplify)
    # identical spelling: exact; a different structure (the rootn spelling of the same function is the
    # canon's business, not the judge's): not exact; the strictly-shorter canonical path is unchanged
    assert list(scored['symbolic_recovery']) == [True, False, True]
    assert scored['predicted_skeleton_prefix'][0] == ['pow', 'x1', '<constant>'] == scored['skeleton_simplified'][0]


def _minting_simplify(tokens):
    """The move the judge has to survive on top of `_fake_simplify`'s: simplifying a masked skeleton MINTS a
    number. ``* u u`` becomes ``pow u 2`` and ``/ + u v u`` becomes ``+ / v u 1``, as the engine rewrites them."""
    out = list(tokens)
    if len(out) == 3 and out[0] == '*' and out[1] == out[2]:
        return ['pow', out[1], '2']
    if out == ['/', '+', 'x3', 'sin', 'x1', 'x3']:
        return ['+', '/', 'sin', 'x1', 'x3', '1']
    return out


def test_a_number_minted_by_simplification_is_a_constant_on_both_sides():
    """``x1 * x1`` and ``x1 ** 2`` are one skeleton. Simplify rewrites the product into ``pow x1 2``; left as it
    is, that ``2`` never equals the ``<constant>`` on the other side, in either direction: a law written as a
    product could not be recovered by a power, nor a law written as a power by a product."""
    snapshot = {
        'skeleton': [['pow', 'x1', '<constant>'], ['*', 'x1', 'x1'], ['/', '+', 'x3', 'sin', 'x1', 'x3'], ['pow', 'x1', '<constant>']],
        'predicted_expression_prefix': [['*', 'x1', 'x1'], ['pow', 'x1', '2.0'], ['+', '/', 'sin', 'x1', 'x3', '1.0'], ['sin', 'x1']],
        'predicted_skeleton_prefix': [['*', 'x1', 'x1'], ['pow', 'x1', '<constant>'], ['+', '/', 'sin', 'x1', 'x3', '<constant>'], ['sin', 'x1']],
        'benchmark_eq_id': ['A', 'B', 'C', 'D'], 'placeholder': [False] * 4,
    }
    scored = derive_metrics(snapshot, operator_arity={'pow': 2, '*': 2, '/': 2, '+': 2, 'sin': 1}, simplify_fn=_minting_simplify)
    assert list(scored['symbolic_recovery']) == [True, True, True, False]
    assert scored['skeleton_simplified'][1] == ['pow', 'x1', '<constant>'] == scored['predicted_skeleton_prefix'][0]
    assert scored['skeleton_simplified'][2] == ['+', '/', 'sin', 'x1', 'x3', '<constant>']
    assert scored['n_constants'][1] == 1          # the minted exponent is a constant of the judged skeleton


def test_the_engine_judges_a_product_and_a_power_alike():
    """The same with the engine the srbf catalogs are judged with."""
    from simplipy import SimpliPyEngine
    engine = SimpliPyEngine.load('acj-5-4-llm', install=True)
    snapshot = {
        'skeleton': [['pow', 'x1', '<constant>'], ['*', 'x1', 'x1'], ['+', 'x1', 'x1']],
        'predicted_expression_prefix': [['*', 'x1', 'x1'], ['pow', 'x1', '2'], ['*', '2.0', 'x1']],
        'predicted_skeleton_prefix': [['*', 'x1', 'x1'], ['pow', 'x1', '<constant>'], ['*', '<constant>', 'x1']],
        'benchmark_eq_id': ['A', 'B', 'C'], 'placeholder': [False] * 3,
    }
    scored = derive_metrics(snapshot, engine=engine)
    assert list(scored['symbolic_recovery']) == [True, True, True]


def test_law_and_prediction_are_judged_by_one_function_of_their_canonical_forms():
    """An expression shows what it is while its numbers are numbers: ``x2 * x4 / (c * x4 ** 3)`` cancels to
    ``x2 / (c * x4 ** 2)``, ``pow(u, 0.5)`` is ``rootn(u, 2)``, and ``pi`` is a number. Law and prediction are
    canonicalized WITH their values, then masked, so the side that wrote the longer spelling does not lose."""
    from simplipy import SimpliPyEngine
    engine = SimpliPyEngine.load('acj-5-4-llm', install=True)
    coulomb = ['/', '*', 'x1', 'x2', '*', '*', '*', '4', '3.1415926535897', '8.854e-12', 'pow', 'x2', '3']
    laws = [coulomb, ['rootn', '/', '*', 'x1', 'x2', 'x3', '2'], ['*', '*', '4', '3.141592653589793', 'x1'], coulomb]
    predictions = [['/', '/', '8.987742e9', 'x2', '/', 'x2', 'x1'], ['pow', '/', '*', 'x2', 'x1', 'x3', '0.5'],
                   ['*', '4', '*', 'np.pi', 'x1'], ['/', '8.987742e9', 'pow', 'x2', '3']]
    from symbolic_data.token_ops import normalize_skeleton
    snapshot = {
        'skeleton': [normalize_skeleton(law) for law in laws], 'ground_truth_prefix': laws,
        'predicted_expression_prefix': predictions, 'predicted_skeleton_prefix': [normalize_skeleton(p) for p in predictions],
        'benchmark_eq_id': ['A', 'B', 'C', 'D'], 'placeholder': [False] * 4,
    }
    scored = derive_metrics(snapshot, engine=engine)
    # the last answer dropped a variable the law keeps: a different function, whatever the masking
    assert list(scored['symbolic_recovery']) == [True, True, True, False]
    assert scored['skeleton_simplified'][0] == scored['predicted_skeleton_prefix'][0]


def test_a_match_in_either_form_is_a_match():
    """No simplifier is complete. ``x1 ** 1.5 * (x1 ** 1.5 + x1)`` and ``x1 * x1 * (x1 ** 0.5 + x1)`` are one
    function whose canonical forms differ, while their skeletons as written agree; a product and a power
    agree in canonical form only. Either agreement is a sound witness, and the judge takes both."""
    from simplipy import SimpliPyEngine
    from symbolic_data.token_ops import normalize_skeleton
    engine = SimpliPyEngine.load('acj-5-4-llm', install=True)
    laws = [['*', 'pow', 'x1', '/', '3', '2', '+', 'pow', 'x1', '/', '3', '2', 'x1'], ['pow', 'x1', '2']]
    predictions = [['*', '*', 'x1', 'x1', '+', 'pow', 'x1', '0.5', 'x1'], ['*', 'x1', 'x1']]
    snapshot = {
        'skeleton': [['*', 'pow', 'x1', '<constant>', '+', 'pow', 'x1', '<constant>', 'x1'], ['pow', 'x1', '<constant>']],
        'ground_truth_prefix': laws, 'predicted_expression_prefix': predictions,
        'predicted_skeleton_prefix': [normalize_skeleton(p) for p in predictions],
        'benchmark_eq_id': ['A', 'B'], 'placeholder': [False] * 2,
    }
    scored = derive_metrics(snapshot, engine=engine)
    assert list(scored['symbolic_recovery']) == [True, True]
    assert scored['skeleton_simplified'][0] == scored['predicted_skeleton_prefix'][0]      # the pair that agreed
    assert list(scored['edit_distance']) == [0, 0]


def test_a_malformed_answer_is_judged_as_stored_and_costs_no_other_row():
    """A stored prefix the engine cannot parse (trailing tokens) is masked token by token and compared as it
    stands. It must not raise: one bad row would take the whole result file with it."""
    from simplipy import SimpliPyEngine
    engine = SimpliPyEngine.load('acj-5-4-llm', install=True)
    snapshot = {
        'skeleton': [['pow', 'x1', '<constant>'], ['sin', 'x1']], 'ground_truth_prefix': [['pow', 'x1', '2'], ['sin', 'x1']],
        'predicted_expression_prefix': [['*', 'x1', 'x1'], ['sin', 'x1', 'x2', '3.0']],
        'predicted_skeleton_prefix': [['*', 'x1', 'x1'], ['sin', 'x1', 'x2', '<constant>']],
        'benchmark_eq_id': ['A', 'B'], 'placeholder': [False] * 2,
    }
    scored = derive_metrics(snapshot, engine=engine)
    assert list(scored['symbolic_recovery']) == [True, False]


def test_a_failed_prediction_is_a_miss_on_every_recovery_rate() -> None:
    """Rates are defined for every problem: a method that returns nothing has missed, on symbolic recovery as on
    numeric recovery. A rate conditioned on success would rise when a method fails on the hard laws."""
    import numpy as np
    from srbf import bootstrap_report, derive_metrics
    x = np.linspace(0.0, 1.0, 8).reshape(-1, 1)
    law = {"skeleton": ["+", "x1", "<constant>"], "expression": ["+", "x1", "1.5"], "y": x + 1.5}
    snapshot = {
        "skeleton": [law["skeleton"], law["skeleton"]], "expression": [law["expression"], law["expression"]],
        "x": [x, x], "y": [law["y"], law["y"]], "x_val": [x, x], "y_val": [law["y"], law["y"]],
        "y_pred": [law["y"], None], "y_pred_val": [law["y"], None],
        "predicted_skeleton_prefix": [["+", "x1", "<constant>"], None], "predicted_expression_prefix": [["+", "x1", "1.5"], None],
        "prediction_success": [True, False], "placeholder": [False, False], "benchmark_eq_id": ["a", "b"],
    }
    scored = derive_metrics(snapshot, operator_arity={"+": 2})
    assert list(scored["symbolic_recovery"]) == [True, False]
    assert list(scored["numeric_recovery_val"]) == [True, False]
    assert bootstrap_report(scored, "symbolic_recovery")["median"] == 0.5     # one of two laws, not one of one


def test_a_failed_prediction_has_no_fit_quality() -> None:
    """How good an answer is can only be said of an answer. A problem without one has no FVU and no R^2, not the
    worst one (none would be the worst: R^2 has no lower bound), and summaries of these columns describe the answers
    that were made. Whether the problem was solved is the rates' business."""
    import numpy as np
    from srbf import bootstrap_report, derive_metrics
    x = np.linspace(0.0, 1.0, 8).reshape(-1, 1)
    y = x + 1.5
    snapshot = {
        "skeleton": [["+", "x1", "<constant>"]] * 3, "expression": [["+", "x1", "1.5"]] * 3,
        "x": [x] * 3, "y": [y] * 3, "x_val": [x] * 3, "y_val": [y] * 3,
        "y_pred": [y, y + 0.1, None], "y_pred_val": [y, y + 0.1, None],
        "predicted_skeleton_prefix": [["+", "x1", "<constant>"]] * 2 + [None],
        "predicted_expression_prefix": [["+", "x1", "1.5"], ["+", "x1", "1.6"], None],
        "prediction_success": [True, True, False], "placeholder": [False] * 3, "benchmark_eq_id": ["a", "b", "c"],
    }
    scored = derive_metrics(snapshot, operator_arity={"+": 2})
    for column in ("fvu_val", "log10_fvu_val", "r2_val", "fvu_fit", "r2_fit"):
        values = np.asarray(scored[column], dtype=float)
        assert np.isfinite(values[1]) and np.isnan(values[2]), column
    assert list(scored["numeric_recovery_val"]) == [True, False, False]            # the rate still counts the miss
    assert bootstrap_report(scored, "r2_val")["n_groups"] == 2                      # the two answers, not three problems


def _three_problems_one_failed(failed_skeleton):
    import numpy as np
    x = np.linspace(0.0, 1.0, 8).reshape(-1, 1)
    y = x + 1.5
    return {
        "skeleton": [["+", "x1", "<constant>"]] * 3, "expression": [["+", "x1", "1.5"]] * 3,
        "x": [x] * 3, "y": [y] * 3, "x_val": [x] * 3, "y_val": [y] * 3,
        "y_pred": [y, y + 0.1, None], "y_pred_val": [y, y + 0.1, None],
        "predicted_skeleton_prefix": [["+", "x1", "<constant>"], ["sin", "x2"], failed_skeleton],
        "predicted_expression_prefix": [["+", "x1", "1.5"], ["sin", "x2"], None],
        "prediction_success": [True, True, False], "placeholder": [False] * 3, "benchmark_eq_id": ["a", "b", "c"],
    }


@pytest.mark.parametrize("failed_skeleton", [None, ["+", "x1", "<constant>"]], ids=["nothing returned", "text left behind"])
def test_a_failed_problem_takes_the_worst_value_where_the_range_has_one(failed_skeleton) -> None:
    """An overlap is a share in [0, 1]: the range ends, and a problem the method failed sits at the bad end.
    Leaving it out would let a method raise its mean overlap by failing on the hard laws. A failure is a failure
    whatever text it left behind."""
    from srbf.result_processing import WORST_VALUE
    scored = derive_metrics(_three_problems_one_failed(failed_skeleton), operator_arity={"+": 2, "sin": 1})
    assert WORST_VALUE == {
        "f1_score": 0.0, "precision_score": 0.0, "recall_score": 0.0,
        "f1_score_unique_variables": 0.0, "precision_unique_variables": 0.0, "recall_unique_variables": 0.0}
    for column, worst in WORST_VALUE.items():
        values = [float(v) for v in scored[column]]
        assert values[2] == worst, column
        assert values[0] == 1.0 - worst, column                  # the exact answer sits at the other end
    assert bootstrap_report(scored, "f1_score")["n_groups"] == 3  # three problems, not two answers
    assert list(scored["symbolic_recovery"]) == [True, False, False]


@pytest.mark.parametrize("failed_skeleton", [None, ["+", "x1", "<constant>"]], ids=["nothing returned", "text left behind"])
def test_a_failed_problem_has_no_value_where_the_range_has_no_end(failed_skeleton) -> None:
    """A length, a ratio of lengths, a raw distance, R^2: each can be arbitrarily bad, so none has a worst value
    to take, and the column describes the answers that were made."""
    scored = derive_metrics(_three_problems_one_failed(failed_skeleton), operator_arity={"+": 2, "sin": 1})
    for column in ("predicted_skeleton_prefix_length", "skeleton_length_ratio", "predicted_n_constants", "n_constants_delta",
                   "edit_distance", "edit_distance_norm", "zss_edit_distance", "predicted_total_nestedness"):
        assert scored[column][1] is not None and scored[column][2] is None, column
    assert np.isnan(float(scored["r2_val"][2])) and np.isnan(float(scored["log10_fvu_val"][2]))


def test_a_placeholder_row_is_not_a_failed_problem() -> None:
    snapshot = _three_problems_one_failed(None)
    snapshot["placeholder"] = [False, False, True]
    scored = derive_metrics(snapshot, operator_arity={"+": 2, "sin": 1})
    assert scored["f1_score"][2] is None and scored["recall_score"][2] is None


def test_the_normalized_edit_distance_is_a_share_of_the_longer_sequence() -> None:
    scored = derive_metrics(_three_problems_one_failed(None), operator_arity={"+": 2, "sin": 1})
    assert float(scored["edit_distance_norm"][0]) == 0.0
    assert float(scored["edit_distance_norm"][1]) == 1.0          # sin x2 against + x1 <constant>: all three differ
    assert float(scored["precision_score"][1]) == 0.0 and float(scored["recall_score"][1]) == 0.0


def test_failed_problems_can_be_left_out_instead() -> None:
    """The other reading of the same columns: every analysis metric describes the answers that were made."""
    from srbf.result_processing import WORST_VALUE
    snapshot = _three_problems_one_failed(["+", "x1", "<constant>"])
    counted = derive_metrics(snapshot, operator_arity={"+": 2, "sin": 1})
    left_out = derive_metrics(snapshot, operator_arity={"+": 2, "sin": 1}, impute_failed=False)
    for column in WORST_VALUE:
        assert left_out[column][2] is None, column
        assert [float(v) for v in left_out[column][:2]] == [float(v) for v in counted[column][:2]], column
    assert bootstrap_report(left_out, "f1_score")["n_groups"] == 2 and bootstrap_report(counted, "f1_score")["n_groups"] == 3
    assert list(left_out["symbolic_recovery"]) == [True, False, False]     # a rate counts the failure either way


def test_an_exception_stored_as_a_placeholder_is_read_as_the_failed_prediction_it_is() -> None:
    """Result files written before an escaping exception became a failed prediction hold it as a placeholder with
    the exception in `error`. The problem was posed and the method failed it: it is a miss and takes the worst
    values, while a placeholder of the catalog (a problem that could not be drawn) stays out of every summary."""
    snapshot = _three_problems_one_failed(None)
    for key, values in (("skeleton", ["+", "x1", "<constant>"]), ("expression", ["+", "x1", "1.5"])):
        snapshot[key] = snapshot[key] + [values]
    for key in ("x", "y", "x_val", "y_val"):
        snapshot[key] = snapshot[key] + [snapshot[key][0]]
    for key in ("y_pred", "y_pred_val", "predicted_skeleton_prefix", "predicted_expression_prefix"):
        snapshot[key] = snapshot[key] + [None]
    snapshot["prediction_success"] = [True, True, False, False]
    snapshot["benchmark_eq_id"] = ["a", "b", "c", "d"]
    snapshot["placeholder"] = [False, False, True, True]
    snapshot["placeholder_reason"] = [None, None, None, "max_trials_exhausted"]
    snapshot["error"] = [None, None, "'__builtins__'", "max_trials_exhausted"]
    before = list(snapshot["placeholder"])
    scored = derive_metrics(snapshot, operator_arity={"+": 2, "sin": 1})
    assert list(scored["placeholder"]) == [False, False, False, True] and snapshot["placeholder"] == before
    assert float(scored["f1_score"][2]) == 0.0 and float(scored["recall_score"][2]) == 0.0
    assert scored["f1_score"][3] is None
    report = bootstrap_report(scored, "numeric_recovery_val")
    assert report["n_groups"] == 3 and report["n_rows"] == 3             # the exception counts, the undrawn problem does not


def _levels_snapshot():
    """One law, x1 ** 2 + 1.5 * x2, and five answers on its own points."""
    from symbolic_data.token_ops import normalize_skeleton
    rng = np.random.default_rng(0)
    x = rng.uniform(0.5, 2.0, size=(64, 2))

    def values(power, factor):
        return (x[:, 0] ** power + factor * x[:, 1]).reshape(-1, 1)

    law = ['+', 'pow', 'x1', '2', '*', '1.5', 'x2']
    answers = [
        (['+', '*', 'x1', 'x1', '*', '1.5', 'x2'], values(2, 1.5), True),              # the law, spelled with a product
        (['+', 'pow', 'x1', '2', '*', '1.4', 'x2'], values(2, 1.4), True),             # its family, a constant off
        (['+', 'pow', 'x1', '3', '*', '1.5', 'x2'], values(3, 1.5), True),             # another exponent
        (['+', 'pow', 'x1', '2.0000001', '*', '1.5', 'x2'], values(2.0000001, 1.5), True),   # an exponent left unsnapped
        (['+', 'pow', 'x1', '2', '*', '1.5', 'x2'], values(2, 1.5), False),            # the law itself from a method that failed
    ]
    n = len(answers)
    y = values(2, 1.5)
    return {
        'skeleton': [normalize_skeleton(law)] * n, 'ground_truth_prefix': [law] * n,
        'x': [x] * n, 'y': [y] * n, 'x_val': [x] * n, 'y_val': [y] * n,
        'y_pred': [a[1] for a in answers], 'y_pred_val': [a[1] for a in answers],
        'predicted_expression_prefix': [a[0] for a in answers],
        'predicted_skeleton_prefix': [normalize_skeleton(a[0]) for a in answers],
        'prediction_success': [a[2] for a in answers],
        'benchmark_eq_id': list('ABCDE'), 'placeholder': [False] * n,
    }


def test_symbolic_recovery_is_asked_at_three_levels_of_masking():
    """Every number masked: the structure is right. Only the fittable constants masked: the exponents are right
    as well. Nothing masked: the constants are right too, to the precision numeric recovery asks for. Each level
    implies the one before it, and a failed prediction is a miss at all three."""
    from simplipy import SimpliPyEngine
    scored = derive_metrics(_levels_snapshot(), engine=SimpliPyEngine.load('acj-5-4-llm', install=True))
    assert list(scored['symbolic_recovery']) == [True, True, True, True, False]
    assert list(scored['symbolic_recovery_mask_fittable']) == [True, True, False, False, False]
    assert list(scored['symbolic_recovery_mask_none']) == [True, False, False, False, False]
    assert list(scored['numeric_recovery_val']) == [True, False, False, True, False]    # the unsnapped exponent fits, and is not the law's


def test_agreement_at_the_stricter_level_is_a_witness_at_the_looser_one():
    """The levels are nested by construction, whatever the simplifier finds at either of them."""
    from srbf.result_processing import _judged_pair

    def refuses(tokens):
        raise ValueError("no canonical form")

    same = ['+', 'pow', 'x1', '2', '<constant>']
    law, prediction = _judged_pair(refuses, lambda t: None, None, ['sin', 'x1'], None, ['cos', 'x1'], fittable=(same, list(same)))
    assert law == prediction and law is not None
    law, prediction = _judged_pair(refuses, lambda t: None, None, ['sin', 'x1'], None, ['cos', 'x1'])
    assert law != prediction


def test_the_stricter_levels_need_an_engine():
    scored = derive_metrics(_levels_snapshot(), operator_arity={'+': 2, '*': 2, 'pow': 2})
    assert 'symbolic_recovery' in scored and 'symbolic_recovery_mask_fittable' not in scored and 'symbolic_recovery_mask_none' not in scored


def test_a_subtree_of_free_constants_is_one_constant():
    """A function of free constants is a free constant: after masking, a subtree without a variable that holds a
    ``<constant>`` becomes one ``<constant>``. A subtree of numbers alone is a fixed number and stays as written."""
    from simplipy import SimpliPyEngine
    from srbf.result_processing import _fold_constant_subtrees
    arity = SimpliPyEngine.load('acj-5-4-llm', install=True).operator_arity
    fold = lambda tokens: _fold_constant_subtrees(tokens, arity)  # noqa: E731
    assert fold(['*', '<constant>', '*', 'x1', 'rootn', '<constant>', '<constant>']) == ['*', '<constant>', '*', 'x1', '<constant>']
    assert fold(['/', 'exp', 'x1', 'inv', 'pow', '<constant>', '<constant>']) == ['/', 'exp', 'x1', '<constant>']
    assert fold(['*', '<constant>', 'np.pi']) == ['<constant>']
    assert fold(['pow', 'x1', '/', '1', '2']) == ['pow', 'x1', '/', '1', '2']            # numbers alone: a fixed number
    assert fold(['pow', 'x1', '<constant>']) == ['pow', 'x1', '<constant>']              # a variable below: no fold
    assert fold(['+', 'x1']) == ['+', 'x1']                                               # does not parse: unchanged


def test_one_constant_factor_in_two_spellings_is_one_structure():
    """The exact law, refitted: its constant factor written as ``rootn(2 pi, 2)`` in the ground truth and as
    ``1 / pow(2 pi, 1/2)`` or a folded value in the prediction. Before the fold the masked forms kept the two
    constant subtrees apart (found by the oracle: 4 of 176 exact fits judged as another structure)."""
    from simplipy import SimpliPyEngine
    from symbolic_data.token_ops import normalize_skeleton
    engine = SimpliPyEngine.load('acj-5-4-llm', install=True)
    laws = [['/', 'exp', '/', 'neg', 'pow', 'x1', '2', '2', 'rootn', '*', '2', '3.141592653589793', '2'],
            ['/', 'exp', '/', 'neg', 'pow', '/', 'x2', 'x1', '2', '2', '*', 'rootn', '*', '2', '3.141592653589793', '2', 'x1'],
            ['/', '*', '*', '25', 'rootn', '2', '2', 'x1', '204'],
            ['/', 'exp', '/', 'neg', 'pow', 'x1', '2', '2', 'rootn', '*', '2', '3.141592653589793', '2']]
    predictions = [['/', 'exp', '/', 'neg', 'pow', 'x1', '2', '2', 'pow', '*', '2', 'np.pi', '/', '1', '2'],
                   ['/', '*', 'exp', '/', 'neg', '*', 'pow', 'x2', '2', 'inv', 'pow', 'x1', '2', '2', 'inv', 'pow', '*', '2',
                    'np.pi', '/', '1', '2', 'x1'],
                   ['*', '0.1733104855849381', 'x1'],
                   ['/', 'exp', '/', 'neg', 'pow', 'x1', '3', '2', 'pow', '*', '2', 'np.pi', '/', '1', '2']]   # x1^3: another law
    snapshot = {
        'skeleton': [normalize_skeleton(law) for law in laws], 'ground_truth_prefix': laws,
        'predicted_expression_prefix': predictions, 'predicted_skeleton_prefix': [normalize_skeleton(p) for p in predictions],
        'benchmark_eq_id': ['A', 'B', 'C', 'D'], 'placeholder': [False] * 4,
    }
    scored = derive_metrics(snapshot, engine=engine)
    assert list(scored['symbolic_recovery']) == [True, True, True, True]           # the exponent is a number: masked
    assert list(scored['symbolic_recovery_mask_fittable']) == [True, True, True, False]  # ... but it counts here
