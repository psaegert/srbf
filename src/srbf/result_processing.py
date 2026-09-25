"""Helpers for loading, cleaning, and deriving metrics from evaluation results."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import torch
from editdistance import eval as edit_distance

from srbf.metrics.numeric import (
    fvu,
    is_perfect_fit,
    log10_fvu,
    r2,
    safe_divide,
)
from srbf.metrics.symbolic import total_nestedness
from srbf.metrics.token_prediction import f1_score, precision, recall
from srbf.metrics.zss import zss_tree_edit_distance
from symbolic_data.token_ops import normalize_skeleton


# ── Default placeholder values for missing / failed predictions ───────────

DEFAULT_NEGATIVES: dict[str, Any] = {
    'constants': [],
    'error': None,
    'skeleton': [],
    'skeleton_simplified': [],
    'expression': [],
    'variables': [],
    'variable_names': [],
    'complexity': np.nan,
    'y_ref': np.nan,
    'y_ref_val': np.nan,
    'reference_fvu_fit': np.inf,
    'reference_fvu_val': np.inf,
    'numeric_recovery_relative_fit': 0.0,   # RATE metric: failures count as misses
    'numeric_recovery_relative_val': 0.0,
    'placeholder_reason': None,
    'benchmark_metadata': {},
    'ground_truth_infix': [],
    'ground_truth_prefix': [],
    'fit_time': np.inf,
    'input_ids': [],
    'labels': [],
    'labels_decoded': [],
    'n_support': np.nan,
    'predicted_expression': [],
    'predicted_expression_prefix': [],
    'predicted_skeleton_prefix': [],
    'predicted_skeleton_prefix_raw': [],
    'predicted_constants': [],
    'predicted_score': -np.inf,
    'predicted_log_prob': -np.inf,
    'prediction_success': False,
    'skeleton_hash': [],
    'x': np.nan,
    'x_val': np.nan,
    'y': np.nan,
    'y_noisy': np.nan,
    'y_noisy_val': np.nan,
    'y_pred': np.nan,
    'y_pred_val': np.nan,
    'y_val': np.nan,

    'f1_score': 0.0,
    'skeleton_length': np.inf,
    'expression_length': np.inf,
    'predicted_skeleton_prefix_length': np.inf,
    'fvu_fit': np.inf,
    'fvu_val': np.inf,
    'r2_fit': 0.0,
    'r2_val': 0.0,
    'log10_fvu_fit': np.inf,
    'log10_fvu_val': np.inf,
    'only_approx_fvu_fit': np.inf,
    'only_approx_fvu_val': np.inf,
    'only_approx_log10_fvu_fit': np.inf,
    'only_approx_log10_fvu_val': np.inf,
    'numeric_recovery_fit': 0.0,
    'numeric_recovery_val': 0.0,
    'n_variables': np.nan,
    'n_constants': np.inf,
    'predicted_n_constants': np.inf,
    'n_constants_delta': np.inf,
    'symbolic_recovery': 0.0,
    'skeleton_length_ratio': np.inf,
    'predicted_mdl': np.inf,
    'mdl_ratio': np.inf,
    'edit_distance': np.inf,
    'zss_edit_distance': np.inf,
    'unique_variables': np.nan,
    'predicted_unique_variables': 0,
    'f1_score_unique_variables': 0.0,
    'precision_unique_variables': 0.0,
    'recall_unique_variables': 0.0,
    'total_nestedness': np.nan,
    'predicted_total_nestedness': np.inf,
}


# ── Utility ───────────────────────────────────────────────────────────────

def parse_p_notation(value: Any) -> float:
    """Convert strings like ``'0p001'`` to floats (e.g. ``0.001``).

    Passes through numeric types unchanged.  Returns ``np.nan`` on failure.
    """
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.floating)):
        return float(value)
    if not isinstance(value, str):
        raise TypeError(f"Unsupported type for parse_p_notation: {type(value)}")

    cleaned = value.strip().replace('p', '.')
    try:
        return float(cleaned)
    except ValueError:
        return np.nan


def _extract_var_index(variable_name: str) -> int:
    """Extract the numeric index from a variable name like ``x1`` or ``x_1``."""
    if variable_name.startswith('x_'):
        return int(variable_name[2:])
    if variable_name.startswith('x'):
        return int(variable_name[1:])
    raise ValueError(f"Unexpected variable name format: {variable_name}")


# ── None → default replacement ────────────────────────────────────────────

def fill_none_with_defaults(
    results: dict[str, Any],
    test_sets: Sequence[str] | None = None,
    defaults: dict[str, Any] | None = None,
) -> None:
    """Replace ``None`` entries in result arrays with default placeholder values.

    Operates **in-place** on ``results``.

    Parameters
    ----------
    results : dict
        Nested results dict: ``results[model]['results'][test_set][scaling_value][metric]``.
    test_sets : Sequence[str], optional
        Restrict processing to these test sets.  ``None`` means all.
    defaults : dict, optional
        Override :data:`DEFAULT_NEGATIVES` with a custom mapping.
    """
    if defaults is None:
        defaults = DEFAULT_NEGATIVES

    for model in results:
        for test_set in results[model].get('results', {}):
            if test_sets is not None and test_set not in test_sets:
                continue
            for scaling_value in results[model]['results'][test_set]:
                for metric in results[model]['results'][test_set][scaling_value]:
                    data = results[model]['results'][test_set][scaling_value][metric]
                    if not hasattr(data, '__len__') or len(data) == 0:
                        continue
                    try:
                        replaced: Any = [
                            (defaults.get(metric, np.nan) if r is None else r)
                            for r in data
                        ]
                        if not isinstance(defaults.get(metric, np.nan), list):
                            replaced = np.array(replaced)
                        results[model]['results'][test_set][scaling_value][metric] = replaced
                    except ValueError:
                        pass  # setting an array element with a sequence


# ── Derived metric computation ────────────────────────────────────────────

def _price_mdl(mdl_fn: Callable[[list[str]], float], tokens: Any) -> float | None:
    """``mdl_fn`` on a realized prefix; None for a missing prefix, a placeholder-carrying one, or a pricing failure."""
    if tokens is None:
        return None
    tokens = list(tokens)
    if not tokens or any(tok == '<constant>' for tok in tokens):
        return None
    try:
        value = float(mdl_fn(tokens))
    except Exception:  # noqa: BLE001 - an unpriceable expression has no price, not a wrong one
        return None
    return value if np.isfinite(value) else None


def _convert_prefix(convert_fn: Callable[[list[str]], list[str]], tokens: Any) -> list[str] | None:
    """``convert_fn`` on a stored prefix; None stays None, an unconvertible prefix is judged as stored."""
    if tokens is None:
        return None
    try:
        return list(convert_fn(list(tokens)))
    except Exception:  # noqa: BLE001 - conversion failures vary by engine
        return list(tokens)


#: Rounds of mask -> simplify before the judged skeleton is taken as it stands. Two settle every ground truth of the
#: srbf suite; the bound only keeps a pathological input from looping.
_JUDGE_ROUNDS = 4

MaskFn = Callable[[list[str]], list[str] | None]


def _mask_every_number(tokens: list[str]) -> list[str] | None:
    """Every numeric token becomes ``<constant>``: the mask of a judge without an engine."""
    return normalize_skeleton(list(tokens))


_NAMED_CONSTANTS = frozenset({"np.pi", "np.e", "pi", "e"})


def _constant_leaf(token: str) -> bool:
    """A leaf that names a number: a masked constant, a numeral, or pi / e."""
    if token == "<constant>" or token in _NAMED_CONSTANTS:
        return True
    try:
        float(token)
    except ValueError:
        return False
    return True


def _fold_constant_subtrees(tokens: list[str], arity: Mapping[str, int]) -> list[str]:
    """Every subtree without a variable that holds a free ``<constant>`` becomes one ``<constant>``.

    A function of free constants is a free constant, so the family of functions the skeleton stands for is unchanged
    (the fold is sound: it can make two skeletons meet only if they meet as families), and two spellings of one
    constant factor -- ``rootn(<constant>, <constant>)`` and ``1 / pow(<constant>, <constant>)`` -- become the same
    ``<constant>``. A subtree of numbers alone is a fixed number and stays as written (the canonical form evaluates
    it). A prefix that does not parse is returned unchanged."""
    out: list[str] = []

    def walk(i: int) -> tuple[int, bool, bool, int]:
        """Parse the subtree at ``i``; returns (end, has_variable, has_free_constant, start in `out`)."""
        token = tokens[i]
        start = len(out)
        out.append(token)
        n = int(arity.get(token, 0))
        if n == 0:
            leaf_constant = _constant_leaf(token)
            return i + 1, not leaf_constant, token == "<constant>", start
        j, has_variable, has_free = i + 1, False, False
        for _ in range(n):
            j, child_variable, child_free, _start = walk(j)
            has_variable, has_free = has_variable or child_variable, has_free or child_free
        if not has_variable and has_free:
            del out[start:]
            out.append("<constant>")
        return j, has_variable, has_free, start

    try:
        end = walk(0)[0] if tokens else 0
    except (IndexError, RecursionError):
        return list(tokens)
    return out if end == len(tokens) else list(tokens)


def _settled(simplify_fn: Callable[[list[str]], list[str] | None], mask_fn: MaskFn, tokens: list[str]) -> list[str]:
    """Mask, simplify, mask again, until the skeleton stands still.

    Simplifying a masked skeleton can WRITE a number: ``x1 * x1`` becomes ``pow x1 2`` and ``(x3 + sin(x1)) / x3``
    becomes ``sin(x1) / x3 + 1``. That number is a constant like any other at this level of comparison, so it is
    masked too (``pow x1 <constant>``), and the masked form is simplified again, because masking can open a
    further rewrite. Without the second mask ``x1 * x1`` and ``x1 ** 2`` are different skeletons."""
    def mask(sequence: list[str]) -> list[str] | None:
        try:
            return mask_fn(list(sequence))
        except Exception:  # noqa: BLE001 - a token list the engine cannot parse is masked token by token
            return _mask_every_number(list(sequence))

    current = mask(list(tokens)) or list(tokens)
    for _ in range(_JUDGE_ROUNDS):
        try:
            simplified = simplify_fn(list(current))
        except Exception:  # noqa: BLE001 - a skeleton the engine refuses is judged as it stands
            break
        masked = mask(list(simplified)) if simplified is not None else None
        if masked is None or masked == current:
            break
        current = masked
    return list(current)


#: Judged forms by (engine, mask, kind, tokens). One ground truth is judged once per process, not once per result file, and a
#: method that predicts a problem the same way at several budgets is judged once. Bounded; cleared when full.
_JUDGE_CACHE: dict[tuple[Any, ...], list[str] | None] = {}
_JUDGE_CACHE_MAX = 200_000


def _judged_form(simplify_fn: Callable[[list[str]], list[str] | None], mask_fn: MaskFn, tokens: Any, *,
                 canonical: bool, level: str = 'all') -> list[str] | None:
    """One of the two forms an expression is judged in, the same function for the ground truth and for the prediction.

    ``canonical=True``: the expression WITH its numbers is brought into canonical form first, because that is
    where it shows what it is (``x2 * x4 / (c * x4 ** 3)`` cancels to ``x2 / (c * x4 ** 2)`` only while the ``3``
    is a number, ``pow(u, 0.5)`` becomes ``rootn(u, 2)``); then its numbers are masked and the skeleton settles.
    ``canonical=False``: the skeleton as it was written settles. ``None`` when the engine refuses the input.
    ``level`` names what ``mask_fn`` masks (``'all'`` or ``'fittable'``), so the two levels do not share a cache."""
    if tokens is None:
        return None
    key = (id(getattr(simplify_fn, "__self__", simplify_fn)), mask_fn is _mask_every_number, level, canonical, tuple(map(str, tokens)))
    if key in _JUDGE_CACHE:
        cached = _JUDGE_CACHE[key]
        return list(cached) if cached is not None else None
    base: list[str] | None = list(tokens)
    if canonical:
        try:
            base = simplify_fn(list(tokens))
        except Exception:  # noqa: BLE001 - a prefix the engine refuses has no canonical form
            base = None
    form = _settled(simplify_fn, mask_fn, list(base)) if base is not None else None
    if len(_JUDGE_CACHE) >= _JUDGE_CACHE_MAX:
        _JUDGE_CACHE.clear()
    _JUDGE_CACHE[key] = form
    return list(form) if form is not None else None


def _judged_pair(simplify_fn: Callable[[list[str]], list[str] | None], mask_fn: MaskFn, law: Any, law_skeleton: Any,
                 prediction: Any, prediction_skeleton: Any,
                 fittable: tuple[list[str] | None, list[str] | None] = (None, None)) -> tuple[list[str] | None, list[str] | None]:
    """The judged skeletons of a ground truth and of its prediction, ``(ground truth, prediction)``.

    Two expressions are the same up to their constants when they arrive at one skeleton, and the judge looks
    for that in two places: in their canonical forms (:func:`_judged_form`, ``canonical=True``) and in their
    skeletons as written. Each is a sound witness, since simplification never changes the function and masking
    only forgets numbers; neither is complete, because no simplifier is. The canonical pair is returned unless
    only the written pair agrees. ``fittable`` holds the two forms at the stricter level, where only the fittable
    constants are masked: forms that agree there agree here too once their remaining numbers are masked, which
    keeps the levels nested whatever the simplifier finds."""
    law_canonical = _judged_form(simplify_fn, mask_fn, law, canonical=True)
    if law_canonical is None:
        law_canonical = _judged_form(simplify_fn, mask_fn, law_skeleton, canonical=False)
    if prediction_skeleton is None and prediction is None:
        return law_canonical, None
    prediction_canonical = _judged_form(simplify_fn, mask_fn, prediction, canonical=True)
    if prediction_canonical is None:
        prediction_canonical = _judged_form(simplify_fn, mask_fn, prediction_skeleton, canonical=False)
    if prediction_canonical is not None and prediction_canonical == law_canonical:
        return law_canonical, prediction_canonical
    law_written = _judged_form(simplify_fn, mask_fn, law_skeleton, canonical=False)
    prediction_written = _judged_form(simplify_fn, mask_fn, prediction_skeleton, canonical=False)
    if prediction_written is not None and prediction_written == law_written:
        return law_written, prediction_written
    if fittable[0] is not None and fittable[0] == fittable[1]:
        shared = _judged_form(simplify_fn, mask_fn, fittable[0], canonical=False)
        return shared, (list(shared) if shared is not None else None)
    return law_canonical, prediction_canonical


def _fittable_form(simplify_fn: Callable[[list[str]], list[str] | None], fittable_fn: MaskFn, tokens: Any) -> list[str] | None:
    """The form an expression is judged in when the numbers of its STRUCTURE count: canonical with its numbers,
    then only the fittable constants masked (coefficients, offsets), and settled like the skeleton. An exponent
    or the index of a root stays a number, so ``x1 ** 2`` and ``x1 ** 3`` differ here, and ``x1 * x1`` is
    ``pow x1 2`` like ``x1 ** 2``."""
    return _judged_form(simplify_fn, fittable_fn, tokens, canonical=True, level='fittable')


def _answered(row_columns: Mapping[str, Any], prediction_key: str) -> np.ndarray:
    """Which problems have a prediction: the method reported success (where it reports at all) and predicted values
    exist."""
    predictions = row_columns[prediction_key]
    has_values = np.array([yp is not None for yp in predictions], dtype=bool)
    success = row_columns.get('prediction_success')
    if success is None:
        return has_values
    return has_values & np.array([bool(ok) if ok is not None else True for ok in success], dtype=bool)


#: Analysis metrics whose range has a worst value, and that value. A problem the method failed takes it, so a method
#: cannot raise its mean similarity by failing on the hard problems. The value is the end of the metric's RANGE, not the
#: score of some stand-in prediction: an overlap is a share in [0, 1] and nothing is below 0. These are the overlaps and
#: nothing else. R^2, FVU, a length, a ratio of lengths and a raw distance have no bound on the bad side, and the
#: edit distances, normalized or not, describe the predictions that were made.
WORST_VALUE: dict[str, float] = {
    'f1_score': 0.0, 'precision_score': 0.0, 'recall_score': 0.0,
    'f1_score_unique_variables': 0.0, 'precision_unique_variables': 0.0, 'recall_unique_variables': 0.0,
}


def _normalized_edit_distance(a: Sequence[str] | None, b: Sequence[str] | None) -> float | None:
    """The edit distance over the length of the longer sequence, so in [0, 1]: 0 for the same sequence."""
    if a is None or b is None:
        return None
    longest = max(len(a), len(b))
    return float(edit_distance(a, b)) / longest if longest else None


def _raised(row_columns: Mapping[str, Any], i: int) -> bool:
    """Whether row ``i`` is a method's exception that an earlier srbf stored as a placeholder.

    An exception escaping an adapter used to be recorded as a placeholder row: ``placeholder`` true, the exception
    in ``error``, and no reason from the catalog (``placeholder_reason`` empty or ``adapter_exception``). The
    problem was posed and the method failed it, so such a row is read as the failed prediction it is. A placeholder
    of the catalog, a problem that could not be drawn, carries the catalog's reason or no error at all."""
    def at(key: str) -> Any:
        column = row_columns.get(key)
        return column[i] if column is not None and i < len(column) else None
    return bool(at('placeholder')) and at('error') is not None and at('placeholder_reason') in (None, 'adapter_exception')


def _failed(row_columns: Mapping[str, Any], n_rows: int) -> list[bool]:
    """Which problems the method failed: it reported no success or, where it does not report, returned no
    expression. A placeholder row is not a problem the method was given, so it is never a failure."""
    success = row_columns.get('prediction_success')
    skeletons = row_columns.get('predicted_skeleton_prefix')
    placeholder = row_columns.get('placeholder')
    failed = []
    for i in range(n_rows):
        if placeholder is not None and i < len(placeholder) and placeholder[i]:
            failed.append(False)
        elif success is not None and i < len(success) and success[i] is not None:
            failed.append(not bool(success[i]))
        else:
            failed.append(skeletons is None or i >= len(skeletons) or skeletons[i] is None)
    return failed


def compute_derived_metrics(
    results: dict[str, Any],
    test_sets: Sequence[str],
    operator_arity: Mapping[str, int],
    simplify_fn: Callable[[list[str]], list[str] | None] | None = None,
    mdl_fn: Callable[[list[str]], float] | None = None,
    convert_fn: Callable[[list[str]], list[str]] | None = None,
    mask_fn: MaskFn | None = None,
    impute_failed: bool = True,
    fittable_mask_fn: MaskFn | None = None,
) -> None:
    """Compute derived evaluation metrics in-place on *results*.

    This adds the following keys to each ``results[model]['results'][test_set][scaling_value]``
    dict:

    - ``fvu_fit``, ``fvu_val``, ``log10_fvu_fit``, ``log10_fvu_val``, ``r2_fit``, ``r2_val``
    - ``numeric_recovery_fit``, ``numeric_recovery_val``
    - ``only_approx_fvu_*``, ``only_approx_log10_fvu_*``
    - ``skeleton_simplified``, ``f1_score``, ``precision_score``, ``recall_score``,
      ``skeleton_length``, ``predicted_skeleton_prefix_length``
    - ``n_variables``, ``n_constants``, ``predicted_n_constants``, ``n_constants_delta``
    - ``symbolic_recovery``, ``skeleton_length_ratio``
    - ``symbolic_recovery_mask_fittable``, ``symbolic_recovery_mask_none`` (when ``fittable_mask_fn`` is given)
    - ``predicted_mdl``, ``ground_truth_mdl``, ``mdl_ratio`` (when ``mdl_fn`` is given)
    - ``edit_distance``, ``edit_distance_norm``, ``zss_edit_distance``
    - ``unique_variables``, ``predicted_unique_variables``
    - ``f1_score_unique_variables``, ``precision_unique_variables``,
      ``recall_unique_variables``
    - ``total_nestedness``, ``predicted_total_nestedness``

    A problem the method failed is a miss on every rate, takes the worst value of the metrics that
    have one (:data:`WORST_VALUE`) and has no value on the others. With ``impute_failed=False`` it has
    no value on any analysis metric, and every one of them describes the predictions that were made.

    Parameters
    ----------
    results : dict
        Nested results dict.
    test_sets : Sequence[str]
        Test sets to process.
    operator_arity : Mapping[str, int]
        Map from operator name to arity (needed for nestedness &
        tree-edit-distance).
    simplify_fn : callable, optional
        Function to simplify a skeleton token list (e.g.
        ``engine.simplify``).  If ``None``, simplified skeletons are
        set to the raw skeletons.
    convert_fn : callable, optional
        Converts a stored predicted prefix into the engine grammar before anything judges
        or prices it (``engine.convert_expression``). Predictions that came back through an
        infix string (the out-of-process adapters, E2E, NeSymReS) may hold the raw reader output
        -- ``**`` for a power, ``neg`` on a literal -- which the engine's simplify and complexity
        refuse.
    mask_fn : callable, optional
        Masks the numbers of a token list; defaults to masking every numeric token.
        :func:`derive_metrics` passes the engine's own mask, which covers ``pi`` and ``e`` as well.
    impute_failed : bool, optional
        Count a failed problem at the worst value of the metrics that have one (the default). ``False``
        leaves it without a value there too.
    fittable_mask_fn : callable, optional
        Masks only the fittable constants of a token list and keeps the numbers of the structure (exponents,
        root indices). With it, symbolic recovery is also judged at the two stricter levels.
        :func:`derive_metrics` passes the engine's ``fittable`` mask.
    """
    if mask_fn is None:
        mask_fn = _mask_every_number
    for model in results:
        for test_set in test_sets:
            if test_set not in results[model].get('results', {}):
                continue
            for scaling_value in results[model]['results'][test_set]:
                r = results[model]['results'][test_set][scaling_value]

                # ── A method's exception is a failed prediction, in files of every age (`_raised`) ──
                if 'placeholder' in r:
                    r['placeholder'] = [bool(flag) and not _raised(r, i) for i, flag in enumerate(r['placeholder'])]

                # ── Stored prefixes into the engine grammar ────────
                if convert_fn is not None:
                    for key in ('predicted_expression_prefix', 'predicted_skeleton_prefix'):
                        if key in r:
                            r[key] = [_convert_prefix(convert_fn, p) for p in r[key]]

                # ── The judged skeletons of ground truth and prediction (`_judged_pair`) ──
                # A factor the fit made cancel (`tanh(x)^2 / tanh(x)^2`) or a constant it made fold is judged as
                # what it is, not as it was spelled; the ground truth goes through the same function. The stored spelling of
                # the prediction stays under `_as_emitted`.
                n_rows = len(r['skeleton']) if 'skeleton' in r else 0
                fittable_pairs: list[tuple[list[str] | None, list[str] | None]] = []
                if simplify_fn is not None and 'skeleton' in r:
                    laws = r.get('ground_truth_prefix') or r.get('expression') or [None] * n_rows
                    predictions = r.get('predicted_expression_prefix') or [None] * n_rows
                    emitted = list(r['predicted_skeleton_prefix']) if 'predicted_skeleton_prefix' in r else [None] * n_rows
                    fittable_pairs = [
                        (_fittable_form(simplify_fn, fittable_mask_fn, law), _fittable_form(simplify_fn, fittable_mask_fn, pe))
                        if fittable_mask_fn is not None and sk is not None else (None, None)
                        for law, sk, pe in zip(laws, r['skeleton'], predictions)
                    ]
                    pairs = [
                        _judged_pair(simplify_fn, mask_fn, law, sk, pe, ps, fittable=fp) if sk is not None else (None, None)
                        for law, sk, pe, ps, fp in zip(laws, r['skeleton'], predictions, emitted, fittable_pairs)
                    ]
                    r['skeleton_simplified'] = [pair[0] for pair in pairs]
                    if 'predicted_skeleton_prefix' in r:
                        r['predicted_skeleton_prefix'] = [pair[1] if ps is not None else None for pair, ps in zip(pairs, emitted)]
                        if any(a != b for a, b in zip(r['predicted_skeleton_prefix'], emitted)):
                            r['predicted_skeleton_prefix_as_emitted'] = emitted

                # ── FVU / NRR for fit and val splits ──────────────
                for split, saved_split_name in [('fit', ''), ('val', '_val')]:
                    y_key = f'y{saved_split_name}'
                    yp_key = f'y_pred{saved_split_name}'
                    if y_key not in r or yp_key not in r:
                        continue
                    r[f'fvu_{split}'] = np.array([
                        fvu(yt, yp) for yt, yp in zip(r[y_key], r[yp_key])
                    ])
                    r[f'log10_fvu_{split}'] = np.array([
                        log10_fvu(yt, yp) for yt, yp in zip(r[y_key], r[yp_key])
                    ])
                    r[f'r2_{split}'] = np.array([
                        r2(yt, yp) for yt, yp in zip(r[y_key], r[yp_key])
                    ])
                    r[f'numeric_recovery_{split}'] = np.array([
                        is_perfect_fit(yt, yp) for yt, yp in zip(r[y_key], r[yp_key])
                    ])
                    r[f'only_approx_fvu_{split}'] = np.where(
                        r[f'numeric_recovery_{split}'], -np.inf, r[f'fvu_{split}'],
                    )
                    r[f'only_approx_log10_fvu_{split}'] = np.where(
                        r[f'numeric_recovery_{split}'], -np.inf, r[f'log10_fvu_{split}'],
                    )

                    # ── Reference-relative recovery (real-data catalogs) ──
                    # reference_fvu = the accepted ground truth's own FVU on the same target; recovery
                    # relative to it: candidate at least as good as the reference (with the
                    # float32-eps floor). On clean synthetic data reference_fvu == 0, the
                    # threshold is eps, and the relative criterion reduces EXACTLY to
                    # numeric_recovery (regression-tested identity).
                    yref_key = f'y_ref{saved_split_name}'
                    if yref_key in r:
                        r[f'reference_fvu_{split}'] = np.array([
                            fvu(yt, yr) for yt, yr in zip(r[y_key], r[yref_key])
                        ])
                        eps = float(np.finfo(np.float32).eps)
                        ref = r[f'reference_fvu_{split}']
                        thr = np.where(np.isfinite(ref), np.maximum(ref, eps), eps)
                        r[f'numeric_recovery_relative_{split}'] = r[f'fvu_{split}'] <= thr

                    # ── Analysis metrics describe the predictions that were made ──
                    # Whether a problem was solved is what the recovery rates say, and a problem without a prediction
                    # is a miss there. How good a prediction is can only be said of a prediction: a problem without one
                    # has no fit quality, not the worst one (no value would be the worst: R^2 has no lower bound).
                    answered = _answered(r, yp_key)
                    for column in (f'fvu_{split}', f'log10_fvu_{split}', f'r2_{split}',
                                   f'only_approx_fvu_{split}', f'only_approx_log10_fvu_{split}'):
                        r[column] = np.where(answered, np.asarray(r[column], dtype=float), np.nan)
                    # ... and a failed prediction has recovered nothing, whatever values it left behind
                    for column in (f'numeric_recovery_{split}', f'numeric_recovery_relative_{split}'):
                        if column in r:
                            r[column] = np.asarray(r[column], dtype=bool) & answered

                if 'skeleton_simplified' not in r:
                    r['skeleton_simplified'] = list(r['skeleton'])

                skel_sim = r['skeleton_simplified']
                # The columns below describe predictions. A failed prediction is none, whatever text it left behind.
                failed = _failed(r, len(skel_sim))
                pred_skel = [None if miss else ps for ps, miss in zip(r['predicted_skeleton_prefix'], failed)]

                # ── Token-level F1 ────────────────────────────────
                r['f1_score'] = np.array([
                    f1_score(np.array([ps]), np.array([sk])) if ps is not None else None
                    for ps, sk in zip(pred_skel, skel_sim)
                ])

                r['precision_score'] = np.array([
                    float(precision([ps], [sk])) if ps is not None and sk is not None else None
                    for ps, sk in zip(pred_skel, skel_sim)
                ], dtype=object)
                r['recall_score'] = np.array([
                    float(recall([ps], [sk])) if ps is not None and sk is not None else None
                    for ps, sk in zip(pred_skel, skel_sim)
                ], dtype=object)

                # ── Lengths ───────────────────────────────────────
                r['skeleton_length'] = np.array([
                    len(sk) if sk is not None else None for sk in skel_sim
                ])
                r['predicted_skeleton_prefix_length'] = np.array([
                    len(ps) if ps is not None else None for ps in pred_skel
                ])

                # ── Variable / constant counts ────────────────────
                r['n_variables'] = np.array([
                    len(set(t for t in sk if t.startswith('x'))) if sk is not None else None
                    for sk in skel_sim
                ])
                r['n_constants'] = np.array([
                    sk.count('<constant>') if sk is not None else None
                    for sk in skel_sim
                ])
                r['predicted_n_constants'] = np.array([
                    ps.count('<constant>') if ps is not None else None
                    for ps in pred_skel
                ])
                r['n_constants_delta'] = np.array([
                    pnc - nc if pnc is not None and nc is not None else None
                    for pnc, nc in zip(r['predicted_n_constants'], r['n_constants'])
                ])

                # ── Symbolic recovery ─────────────────────────────
                # A RATE: defined for every problem. A problem without a prediction is a miss, as it is
                # for numeric recovery; conditioning on success would reward failing on the hard problems.
                r['symbolic_recovery'] = np.array([
                    ps is not None and ps == sk
                    for ps, sk in zip(pred_skel, skel_sim)
                ])

                # ── The same question with fewer numbers masked ───
                # `symbolic_recovery` masks every number: the structure is right. With only the fittable constants
                # masked, the numbers of the structure (exponents, root indices) have to be right as well. With
                # nothing masked, the constants have to be right too, and what is right for a fitted constant is what
                # numeric recovery measures: the ground truth is reproduced to float32 precision on the validation points.
                # (A token-by-token comparison of numbers is not available: the canonical form spreads a rational
                # through the tree, `1.5 * x2` is `(3 * x2) / 2`.) Each level implies the one before it.
                if fittable_mask_fn is not None and fittable_pairs:
                    r['symbolic_recovery_mask_fittable'] = np.array([
                        bool(agrees) and not miss and fp[0] is not None and fp[0] == fp[1]
                        for agrees, miss, fp in zip(r['symbolic_recovery'], failed, fittable_pairs)
                    ])
                    if 'numeric_recovery_val' in r:
                        r['symbolic_recovery_mask_none'] = np.array([
                            bool(typed) and bool(numeric)
                            for typed, numeric in zip(r['symbolic_recovery_mask_fittable'], r['numeric_recovery_val'])
                        ])

                # ── Length ratio ──────────────────────────────────
                r['skeleton_length_ratio'] = np.array([
                    safe_divide(pl, tl) if pl is not None and tl is not None else None
                    for pl, tl in zip(
                        r['predicted_skeleton_prefix_length'], r['skeleton_length'],
                    )
                ])

                # ── MDL: the description length of the REALIZED expressions ──
                # ``mdl_fn`` is the engine's complexity (milli-bits, simplipy's native unit; the
                # same pricing the flash-ansr ranking modes score with). Priced on the realized
                # expressions, numeric constants inlined: a prefix that still carries a
                # ``<constant>`` placeholder is not priced (a skeleton price would masquerade as a
                # realized one), and an unpriceable expression is None, never a made-up number.
                if mdl_fn is not None:
                    r['predicted_mdl'] = np.array([
                        _price_mdl(mdl_fn, pe) for pe in r.get('predicted_expression_prefix', [None] * len(pred_skel))
                    ], dtype=object)
                    r['ground_truth_mdl'] = np.array([
                        _price_mdl(mdl_fn, ge) for ge in r.get('ground_truth_prefix', [None] * len(pred_skel))
                    ], dtype=object)
                    r['mdl_ratio'] = np.array([
                        safe_divide(pm, gm) if pm is not None and gm is not None else None
                        for pm, gm in zip(r['predicted_mdl'], r['ground_truth_mdl'])
                    ])

                # ── Edit distances ────────────────────────────────
                r['edit_distance'] = np.array([
                    edit_distance(ps, sk)
                    if ps is not None and sk is not None else None
                    for ps, sk in zip(pred_skel, skel_sim)
                ])
                normalized: list[Any] = [_normalized_edit_distance(ps, sk) for ps, sk in zip(pred_skel, skel_sim)]
                r['edit_distance_norm'] = np.array(normalized, dtype=object)
                r['zss_edit_distance'] = np.array([
                    zss_tree_edit_distance(ps, sk, operator_arity)
                    if ps is not None and sk is not None else None
                    for ps, sk in zip(pred_skel, skel_sim)
                ])

                # ── Unique variables ──────────────────────────────
                r['unique_variables'] = [
                    sorted(
                        list(set(t for t in sk if t.startswith('x'))),
                        key=_extract_var_index,
                    ) if sk is not None else None
                    for sk in skel_sim
                ]
                r['predicted_unique_variables'] = [
                    sorted(
                        list(set(t for t in ps if t.startswith('x'))),
                        key=_extract_var_index,
                    ) if ps is not None else None
                    for ps in pred_skel
                ]

                # ── Variable-level F1 / precision / recall ────────
                # Compute precision & recall ONCE per row and derive F1 from them with the SAME
                # torch formula (and float32 dtype + NaN->0) that f1_score uses internally, instead
                # of also calling f1_score (which would recompute precision and recall a second time).
                f1_uv: list[Any] = []
                prec_uv: list[Any] = []
                rec_uv: list[Any] = []
                for puv, uv in zip(r['predicted_unique_variables'], r['unique_variables']):
                    if puv is not None and uv is not None:
                        p = precision([puv], [uv])
                        rc = recall([puv], [uv])
                        f1_uv.append(torch.nan_to_num(2 * (p * rc) / (p + rc), nan=0.0))
                        prec_uv.append(p)
                        rec_uv.append(rc)
                    else:
                        f1_uv.append(None)
                        prec_uv.append(None)
                        rec_uv.append(None)
                r['f1_score_unique_variables'] = np.array(f1_uv)
                r['precision_unique_variables'] = np.array(prec_uv)
                r['recall_unique_variables'] = np.array(rec_uv)

                # ── Nestedness ────────────────────────────────────
                r['total_nestedness'] = np.array([
                    total_nestedness(sk, operator_arity) if sk is not None else None
                    for sk in skel_sim
                ])
                r['predicted_total_nestedness'] = np.array([
                    total_nestedness(ps, operator_arity) if ps is not None else None
                    for ps in pred_skel
                ])

                # ── A failed problem takes the worst value where the range has one ──
                for column, worst in (WORST_VALUE if impute_failed else {}).items():
                    values = np.array(list(r[column]), dtype=object)
                    for i, miss in enumerate(failed):
                        if miss:
                            values[i] = worst
                    r[column] = values


def derive_metrics(
    snapshot: Mapping[str, Sequence[Any]],
    *,
    engine: Any = None,
    operator_arity: Mapping[str, int] | None = None,
    simplify_fn: Callable[[list[str]], list[str] | None] | None = None,
    mdl_fn: Callable[[list[str]], float] | None = None,
    convert_fn: Callable[[list[str]], list[str]] | None = None,
    impute_failed: bool = True,
) -> dict[str, Any]:
    """Compute the standardized derived metrics for one raw ``Benchmark.run()`` snapshot.

    A benchmark run emits RAW results only; this is the standardized second stage. It returns a NEW
    snapshot -- the raw columns PLUS the derived metric columns (``fvu_fit`` / ``fvu_val``,
    ``log10_fvu_*``, ``numeric_recovery_*``, ``symbolic_recovery``, ``f1_score``, skeleton lengths,
    edit distances, unique-variable precision/recall, nestedness, ...) -- WITHOUT mutating the input.
    Compose the result with :func:`bootstrap_report` / :func:`draw_distribution`, or compute your own
    metrics over the raw columns instead.

    This is the ergonomic entry point over :func:`compute_derived_metrics` (which mutates a 4-level
    nested ``results[model]['results'][test_set][scaling_value]`` dict in place): it lifts the flat
    snapshot into that shape, derives the metrics, and returns the augmented leaf.

    Parameters
    ----------
    snapshot : Mapping[str, Sequence]
        A raw run snapshot (the dict-of-lists a ``Benchmark.run()`` returns).
    engine : SimpliPyEngine, optional
        If given, its ``operator_arity`` and ``simplify`` are used unless overridden below. Provide
        either ``engine`` or ``operator_arity`` (e.g. ``engine=adapter.get_simplipy_engine()``).
    operator_arity : Mapping[str, int], optional
        Operator-token -> arity map (needed for tree-edit-distance and nestedness). Used instead of
        the engine's when given.
    simplify_fn : callable, optional
        Skeleton simplifier; defaults to the engine's ``simplify`` when an ``engine`` is given, else
        ``None`` (simplified skeletons then fall back to the raw skeletons).
    convert_fn : callable, optional
        Converts the stored predicted prefixes into the engine grammar before they are judged or
        priced; defaults to the engine's ``convert_expression`` when an ``engine`` is given (the
        identity on prefixes already in that grammar), else ``None``.
    impute_failed : bool, optional
        What a failed problem counts in the analysis metrics whose range has a worst value
        (:data:`WORST_VALUE`): that value (the default), so the column is read over every problem and a
        method cannot gain by failing on the hard ones; or, with ``False``, nothing, so the column
        describes the predictions that were made, like the metrics without a worst value.

    Returns
    -------
    dict
        A NEW snapshot: the raw columns plus the derived metric columns. The input is not modified.
    """
    if operator_arity is None:
        if engine is None:
            raise ValueError("derive_metrics needs either `engine` or `operator_arity`.")
        operator_arity = engine.operator_arity
    if simplify_fn is None and engine is not None:
        simplify_fn = engine.simplify
    if mdl_fn is None and engine is not None:
        # The price the flash-ansr ranking modes use: certified, f64 parse, the public Default canon.
        def mdl_fn(tokens: list[str], _engine: Any = engine) -> float:
            return float(_engine.complexity(list(tokens), certified=True, mode='f64', canon='default'))

    if convert_fn is None and engine is not None:
        convert_fn = engine.convert_expression

    # The engine's own mask: every number, the named constants pi and e included, becomes <constant>; then every
    # subtree of free constants is one free constant (`_fold_constant_subtrees`), on both sides alike.
    arity = dict(operator_arity)

    def engine_mask(tokens: list[str], _engine: Any = engine) -> list[str] | None:
        return _fold_constant_subtrees(list(_engine.mask(list(tokens), 'all', collect=False)), arity)

    def engine_mask_fittable(tokens: list[str], _engine: Any = engine) -> list[str] | None:
        return _fold_constant_subtrees(list(_engine.mask(list(tokens), 'fittable', collect=False)), arity)

    mask_fn: MaskFn | None = engine_mask if engine is not None and hasattr(engine, "mask") else None
    fittable_mask_fn: MaskFn | None = engine_mask_fittable if engine is not None and hasattr(engine, "mask") else None

    # Shallow-copy the snapshot as the nested leaf: compute_derived_metrics only ADDS derived keys to
    # the leaf (and rebinds the converted prefix columns), so the derived columns land in this copy
    # and the caller's snapshot stays untouched.
    leaf = dict(snapshot)
    results = {"model": {"results": {"test": {0: leaf}}}}
    compute_derived_metrics(results, test_sets=["test"], operator_arity=operator_arity, simplify_fn=simplify_fn, mdl_fn=mdl_fn,
                            convert_fn=convert_fn, mask_fn=mask_fn, impute_failed=impute_failed,
                            fittable_mask_fn=fittable_mask_fn)
    return results["model"]["results"]["test"][0]
