"""Flash-ANSR seeding + PySR at a fixed time budget: the hybrid arm.

One budget T per problem is split by a ratio r: Flash-ANSR gets (1 - r) T, PySR gets r T. The budget
is controlled by DIRECT knobs, never by timeouts: two measured time laws turn a share of T into a
number of Flash-ANSR candidates (``choices``) and a number of PySR iterations (``niterations``); the
achieved wall time of both stages is recorded next to the target.

r = 0: Flash-ANSR alone, its rank-0 answer. r = 1: PySR alone, cold. In between: the top-K refined
Flash-ANSR candidates enter PySR as initial ``guesses``. PySR adds candidates: its whole hall of fame
joins the Flash-ANSR candidate pool, priced the way Flash-ANSR prices its own (fit, MDL, the
ranking's score), and Flash-ANSR's sorting picks the prediction from the extended pool.

The snapshot design: iid draws compose, so ONE chunked generation pass per problem serves every
ratio. The pass generates up to the largest target in chunks whose cumulative sizes are the
targets c((1 - r) T) of all ratios; after each chunk the merged pool's rank 0 and its top-K seeds
are snapshotted together with the cumulative wall time -- exactly what a run with that many
candidates would have produced and cost. Snapshots are cached on disk per problem, so the cells
for the other ratios reuse them.

A snapshot is only valid for the exact (X, y) it was generated on. srbf's data source re-draws the
support points on every iteration (symbolic_data's ProblemSource is entropy-seeded by design;
reproduction comes from a MATERIALIZED source), so every cell of a sweep must read the same frozen
subset -- `scripts/run_hybrid_sweep.py` freezes it once per catalog. The snapshot records a
fingerprint of the data it saw; a mismatch regenerates (and warns) instead of returning another
instance's predictions.
"""
from __future__ import annotations

import hashlib
import math
import pickle
import time
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from flash_ansr.flash_ansr import FlashANSR, _price_realized, _respell_result
from flash_ansr.refine import DEFAULT_REFINE_SCOPE, Refiner, refinement_slots
from flash_ansr.scoring import RankingConfig, count_constants, score_row
from flash_ansr.spelling import ConstantLadderConfig
from simplipy.engine import Mode
from symbolic_data.token_ops import normalize_expression, normalize_skeleton

from srbf.core import EvaluationModelAdapter, EvaluationResult, EvaluationSample
from srbf.metrics.numeric import fvu as fvu_array
from srbf.model_adapters import FlashANSRAdapter
from srbf.subprocess_adapter import SubprocessAdapter, _compute_fvu_from_predictions, evaluate_prefix

__all__ = ["FlashANSRPySRAdapter", "TimeLaw", "prefix_to_julia", "data_fingerprint", "PYSR_OPERATORS",
           "pysr_candidates", "rank_candidates", "pick_prediction", "refine_settings", "REFINE_DEFAULTS"]

#: the PySR worker's vocabulary (srbf/worker/models/pysr_worker.py): a seed using anything else is dropped
PYSR_UNARY = {"neg", "abs", "inv", "sin", "cos", "tan", "asin", "acos", "atan",
              "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "exp", "log"}
PYSR_BINARY = {"+", "-", "*", "/", "pow", "rootn"}
PYSR_OPERATORS = PYSR_UNARY | PYSR_BINARY
SPECIAL_LITERALS = {"np.pi": math.pi, "np.e": math.e, "pi": math.pi, "e": math.e}


class TimeLaw:
    """``t = a + b * x`` measured on the target machine; ``units_for(t)`` inverts it, floored at
    ``minimum`` when the share of the budget is positive and 0 when it is not worth one unit."""

    def __init__(self, a: float, b: float, minimum: int = 1):
        self.a, self.b, self.minimum = float(a), float(b), int(minimum)
        if self.b <= 0:
            raise ValueError("a time law needs a positive slope")

    def units_for(self, seconds: float) -> int:
        if seconds <= 0:
            return 0
        raw = (seconds - self.a) / self.b
        if raw < 0.5:
            return self.minimum if seconds > 0 else 0
        return max(self.minimum, int(round(raw)))

    def seconds_for(self, units: int) -> float:
        return 0.0 if units <= 0 else self.a + self.b * units

    def as_dict(self) -> dict[str, float]:
        return {"a": self.a, "b": self.b, "minimum": self.minimum}


def _literal(token: str) -> str | None:
    if token in SPECIAL_LITERALS:
        return repr(float(SPECIAL_LITERALS[token]))
    try:
        value = float(token)
    except ValueError:
        if "/" in token:
            num, _, den = token.partition("/")
            try:
                value = float(num) / float(den)
            except (ValueError, ZeroDivisionError):
                return None
        else:
            return None
    if not math.isfinite(value):
        return None
    text = repr(value)
    return f"({text})" if value < 0 else text


def prefix_to_julia(tokens: Sequence[str], arity: Mapping[str, int], variables: Sequence[str]) -> str | None:
    """A realized prefix expression (constants inlined, ``x1..xn`` variables) as the Julia-syntax
    infix string PySR's ``guesses`` parser reads: unaries as calls, ``pow`` as ``^``, ``rootn`` as
    a call, negative literals in parentheses, ``x<i>`` mapped onto the fit's variable names.
    ``None`` when a token is outside PySR's vocabulary (the seed is then not offered)."""
    tokens = list(tokens)

    def render(i: int) -> tuple[str, int]:
        tok = tokens[i]
        n = int(arity.get(tok, 0))
        if n == 0:
            if len(tok) > 1 and tok[0] == "x" and tok[1:].isdigit():
                k = int(tok[1:]) - 1
                if k < 0 or k >= len(variables):
                    raise ValueError("variable outside the problem")
                return str(variables[k]), i + 1
            lit = _literal(tok)
            if lit is None:
                raise ValueError(f"unknown leaf {tok!r}")
            return lit, i + 1
        if tok not in PYSR_OPERATORS:
            raise ValueError(f"operator {tok!r} is not in PySR's vocabulary")
        args = []
        j = i + 1
        for _ in range(n):
            s, j = render(j)
            args.append(s)
        if tok in ("+", "-", "*", "/"):
            return f"({args[0]} {tok} {args[1]})", j
        if tok == "pow":
            return f"({args[0]} ^ {args[1]})", j
        return f"{tok}({', '.join(args)})", j

    try:
        text, end = render(0)
    except (ValueError, IndexError):
        return None
    return text if end == len(tokens) else None


def _score_key(candidate: Mapping[str, Any]) -> tuple[float, tuple[str, ...]]:
    score = candidate.get("score")
    try:
        value = float(score)
    except (TypeError, ValueError):
        value = float("inf")
    if not math.isfinite(value):
        value = float("inf")
    return (value, tuple(map(str, candidate.get("expression") or ())))


def data_fingerprint(sample: EvaluationSample) -> str:
    """A digest of the arrays a snapshot's candidates were fitted on and predicted for: support X,
    the fitted y (noisy when the sample carries one), validation X."""
    y_fit = sample.y_support_noisy if sample.y_support_noisy is not None else sample.y_support
    h = hashlib.sha1()
    for arr in (sample.x_support, y_fit, sample.x_validation):
        a = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _sort_key(entry: Mapping[str, Any]) -> tuple[float, tuple[str, ...]]:
    return _score_key({"score": entry.get("score"), "expression": entry.get("expression_prefix")})


def _flash_candidate(cand: Mapping[str, Any]) -> dict[str, Any]:
    """A Flash-ANSR candidate as a snapshot stores it (``best`` or a ``candidates`` row), as a pool
    entry: its stored score is the ranking's own, computed by the refine worker on the same criterion."""
    return {
        "source": "flash-ansr", "hof_index": -1,
        "expression_infix": str(cand.get("expression_infix")),
        "expression_prefix": list(cand["expression_prefix"]), "skeleton_prefix": list(cand["skeleton_prefix"]),
        "constants": cand.get("constants"), "fvu": cand.get("fvu"), "mdl": cand.get("mdl"), "score": cand.get("score"),
        "n_nodes": cand.get("n_nodes"), "log_prob": cand.get("log_prob"), "pareto_rank": cand.get("pareto_rank", -1),
        "y_pred": cand.get("y_pred"), "y_pred_val": cand.get("y_pred_val"),
    }


#: the refinement settings a PySR candidate is fitted and re-spelled with when the caller gives none:
#: the doctrine arm's (8 restarts, LM, normal p0 noise of scale 5, the fittable scope, the ladder on)
REFINE_DEFAULTS: dict[str, Any] = {
    "n_restarts": 8, "method": "curve_fit_lm", "p0_noise": "normal", "p0_noise_kwargs": {"loc": 0.0, "scale": 5},
    "refine_scope": DEFAULT_REFINE_SCOPE, "constant_ladder": True,
}


def refine_settings(model: Any) -> dict[str, Any]:
    """The refinement settings of a loaded ``FlashANSR`` (what its own candidates are fitted and
    re-spelled with), for the PySR candidates that join its pool."""
    return {
        "n_restarts": int(getattr(model, "n_restarts", REFINE_DEFAULTS["n_restarts"])),
        "method": getattr(model, "refiner_method", REFINE_DEFAULTS["method"]),
        "p0_noise": getattr(model, "refiner_p0_noise", REFINE_DEFAULTS["p0_noise"]),
        "p0_noise_kwargs": getattr(model, "refiner_p0_noise_kwargs", REFINE_DEFAULTS["p0_noise_kwargs"]),
        "refine_scope": getattr(model, "refine_scope", REFINE_DEFAULTS["refine_scope"]),
        "constant_ladder": getattr(model, "constant_ladder", REFINE_DEFAULTS["constant_ladder"]),
    }


def _literal_value(token: str) -> float | None:
    if token in SPECIAL_LITERALS:
        return float(SPECIAL_LITERALS[token])
    try:
        return float(token)
    except ValueError:
        return None


def _fit_pysr_candidate(prefix: list[str], X: np.ndarray, y: np.ndarray, *, engine: Any,
                        fit: Mapping[str, Any]) -> tuple[Any, list[str]] | None:
    """PySR's expression through flash-ansr's Refiner: its fittable literals become the slots
    (the same scope policy Flash-ANSR's own candidates get), warm-started at PySR's values with
    one restart, a cold fit at the doctrine's restarts when the warm start does not converge.
    ``None`` when the expression has no fittable literal or no fit converged."""
    slots = refinement_slots(list(prefix), engine, fit["refine_scope"])
    if not slots:
        return None
    p0_values = [_literal_value(prefix[i]) for i in slots]
    p0 = np.asarray(p0_values, dtype=float) if all(v is not None for v in p0_values) else None
    refiner = Refiner(simplipy_engine=engine, n_variables=int(X.shape[1]))
    try:
        if p0 is not None:
            refiner.fit(list(prefix), X, y, p0=p0, p0_noise=None, p0_noise_kwargs=None, n_restarts=1,
                        method=fit["method"], converge_error="ignore", refine_scope=fit["refine_scope"])
        if p0 is None or not refiner.valid_fit:
            refiner = Refiner(simplipy_engine=engine, n_variables=int(X.shape[1]))
            refiner.fit(list(prefix), X, y, p0=None, p0_noise=fit["p0_noise"], p0_noise_kwargs=fit["p0_noise_kwargs"],
                        n_restarts=int(fit["n_restarts"]), method=fit["method"], converge_error="ignore",
                        refine_scope=fit["refine_scope"])
    except Exception:  # noqa: BLE001 - a candidate the refiner cannot fit is priced as spelled
        return None
    if not refiner.valid_fit or not refiner.all_constants_values:
        return None
    abstracted = ["<constant>" if i in set(slots) else tok for i, tok in enumerate(prefix)]
    return refiner, abstracted


def _row_from_prefix(prefix: list[str], *, engine: Any, names: Sequence[str], X: np.ndarray, Xv: np.ndarray,
                     fvu: float, mdl: float | None, score: float, spelling: str | None = None) -> dict[str, Any] | None:
    """A pool entry for a realized prefix (its curves from the engine's realizations)."""
    try:
        cols = evaluate_prefix(engine, list(prefix), list(names), X, Xv)
    except Exception:  # noqa: BLE001 - unevaluable: not a candidate
        return None
    y_pred = np.asarray(cols[0], dtype=float).reshape(-1)
    y_pred_val = np.asarray(cols[1], dtype=float).reshape(-1) if Xv.shape[0] else np.empty(0)
    constants = [v for v in (_literal_value(tok) for tok in prefix) if v is not None]
    return {
        "expression_prefix": list(prefix), "skeleton_prefix": list(normalize_skeleton(prefix) or []),
        "constants": constants, "fvu": float(fvu), "mdl": mdl, "score": float(score), "n_nodes": len(prefix),
        "log_prob": None, "pareto_rank": -1, "y_pred": y_pred, "y_pred_val": y_pred_val, "spelling": spelling,
    }


def _fitted_rows(refiner: Any, abstracted: list[str], *, engine: Any, weights: Mapping[str, float], names: Sequence[str],
                 X: np.ndarray, Xv: np.ndarray, y: np.ndarray, y_variance: float, fit: Mapping[str, Any],
                 ladder: ConstantLadderConfig | None) -> list[dict[str, Any]]:
    """The fitted PySR candidate as a pool entry and, when the constant ladder finds a better
    spelling, its re-spelled variant -- through flash-ansr's own ladder pass (``_respell_result``):
    a tie replaces the parent, a strict improvement stands beside it, as in Flash-ANSR's pool."""
    n = int(y.shape[0])
    fvu = FlashANSR._compute_fvu(float(refiner.loss), n, y_variance)
    mdl = _price_realized(engine, refiner, abstracted)
    constant_count = len(refiner.slot_indices)
    score = score_row({"fvu": fvu, "expression": abstracted, "constant_count": constant_count, "log_prob": None, "mdl": mdl}, weights)
    parent = {"fvu": float(fvu), "mdl": mdl, "score": float(score), "expression": list(abstracted), "constant_count": constant_count,
              "fits": list(refiner.all_constants_values), "valid_fit": True, "log_prob": None, "spelling": None, "respelled": None}
    realized = list(refiner.transform(list(abstracted), return_prefix=True))
    parent_row = _row_from_prefix(list(normalize_expression(realized) or realized), engine=engine, names=names, X=X, Xv=Xv,
                                  fvu=fvu, mdl=mdl, score=float(score))
    rows = [parent_row] if parent_row is not None else []
    if ladder is None or mdl is None:
        return rows
    payload = {"constant_ladder": ladder, "ranking_weights": dict(weights), "expression": list(abstracted), "log_prob": None,
               "y_variance": y_variance, "n_variables": int(X.shape[1]), "method": fit["method"],
               "n_restarts": int(fit["n_restarts"]), "p0_noise": fit["p0_noise"], "p0_noise_kwargs": fit["p0_noise_kwargs"]}
    try:
        child = _respell_result(payload, engine, refiner, X, y, parent)
    except Exception:  # noqa: BLE001 - the ladder is best-effort, the fitted candidate stands
        child = None
    if child is None:
        return rows
    try:
        child_refiner = Refiner.from_serialized(simplipy_engine=engine, n_variables=int(X.shape[1]), expression=list(child["expression"]),
                                                n_inputs=int(X.shape[1]), fits=list(child["fits"]), refine_scope="placeholders")
        child_realized = list(child_refiner.transform(list(child["expression"]), return_prefix=True))
    except Exception:  # noqa: BLE001
        return rows
    child_row = _row_from_prefix(list(normalize_expression(child_realized) or child_realized), engine=engine, names=names, X=X, Xv=Xv,
                                 fvu=float(child["fvu"]), mdl=child["mdl"], score=float(child["score"]), spelling=child.get("spelling"))
    if child_row is None:
        return rows
    return [child_row] if child.get("replaces_parent") else rows + [child_row]


def pysr_candidates(
    equations: Sequence[Mapping[str, Any] | str] | None,
    *,
    engine: Any,
    weights: Mapping[str, float],
    x_support: np.ndarray,
    y_fit: np.ndarray,
    x_val: np.ndarray | None,
    variables: Sequence[str],
    refine: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """PySR's hall of fame as Flash-ANSR candidates.

    Each equation (PySR's infix in the fit's variable names) goes through the engine's reader into the
    engine grammar with ``x1..xn`` variables (by column position) and then through what Flash-ANSR's
    own candidates go through: its fittable literals are re-fitted with flash-ansr's Refiner (warm at
    PySR's values), the constant ladder re-spells them, the realized expression is priced (fit as
    flash-ansr's FVU on the fitted target, MDL as the certified f64 default-canon price) and scored
    with the ranking's own ``score_row`` under ``weights`` (``RankingConfig.effective_weights``).
    ``refine`` carries the model's refinement settings (:func:`refine_settings`; the doctrine's when
    None). An expression without a fittable literal, or one the refiner cannot fit, is priced as
    spelled; one the engine cannot read or evaluate is dropped; an unpriceable one scores +inf under
    a live MDL weight and sorts last."""
    X = np.asarray(x_support, dtype=float)
    Xv = np.asarray(x_val, dtype=float) if x_val is not None and np.size(x_val) else np.empty((0, X.shape[1]))
    y = np.asarray(y_fit, dtype=float).reshape(-1)
    finite = np.isfinite(y)
    # ddof=0 over the finite targets: flash-ansr's own definition, so the selection FVU equals the evaluation FVU
    y_variance = float(np.var(y[finite])) if int(finite.sum()) > 1 else float("nan")
    names = [f"x{i + 1}" for i in range(X.shape[1])]
    # `variables` names the columns of X in order; when the worker saw a reduced column set the
    # names it used still map by their position in the FULL list. A list of another length cannot be
    # placed and the reader's own canonical names (`v3` -> `x3`) are trusted instead.
    rename = {str(v): names[i] for i, v in enumerate(variables)} if len(variables) == X.shape[1] else {}
    fit = dict(REFINE_DEFAULTS, **dict(refine or {}))
    ladder_setting = fit.get("constant_ladder")
    ladder = ladder_setting if isinstance(ladder_setting, ConstantLadderConfig) else ConstantLadderConfig.from_mapping(ladder_setting)
    out: list[dict[str, Any]] = []
    for k, entry in enumerate(equations or []):
        text = entry.get("equation") if isinstance(entry, Mapping) else entry
        if not text:
            continue
        try:
            raw = [rename.get(str(t), str(t)) for t in engine.read_infix(str(text))]
            prefix = list(normalize_expression(raw) or [])
        except Exception:  # noqa: BLE001 - an unreadable equation is not a candidate
            continue
        fitted = _fit_pysr_candidate(prefix, X, y, engine=engine, fit=fit)
        if fitted is not None:
            rows = _fitted_rows(fitted[0], fitted[1], engine=engine, weights=weights, names=names, X=X, Xv=Xv, y=y,
                                y_variance=y_variance, fit=fit, ladder=ladder)
        else:
            # no fittable literal (or no converged fit): the expression as PySR spelled it
            try:
                mdl: float | None = float(engine.complexity(prefix, certified=True, mode=Mode.f64, canon="default"))
            except Exception:  # noqa: BLE001 - unpriceable: the scorer decides (+inf under a live mdl weight)
                mdl = None
            if mdl is not None and not np.isfinite(mdl):
                mdl = None
            try:
                cols = evaluate_prefix(engine, prefix, names, X, Xv)
            except Exception:  # noqa: BLE001
                continue
            y_pred = np.asarray(cols[0], dtype=float).reshape(-1)
            if y_pred.shape[0] != y.shape[0]:
                continue
            fvu = float(fvu_array(y, y_pred))
            score = score_row({"fvu": fvu, "expression": prefix, "constant_count": count_constants(prefix), "log_prob": None, "mdl": mdl}, weights)
            row = _row_from_prefix(prefix, engine=engine, names=names, X=X, Xv=Xv, fvu=fvu, mdl=mdl, score=float(score))
            rows = [row] if row is not None else []
        for row in rows:
            row.update({"source": "pysr", "hof_index": k, "expression_infix": str(text),
                        "pysr_complexity": entry.get("complexity") if isinstance(entry, Mapping) else None,
                        "pysr_loss": entry.get("loss") if isinstance(entry, Mapping) else None})
            out.append(row)
    return out


def rank_candidates(flash: Sequence[Mapping[str, Any]], pysr: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The extended candidate pool -- Flash-ANSR's candidates plus PySR's -- in Flash-ANSR's ranking
    order: by score, ties broken on the expression tokens (flash-ansr's own deterministic tie-break).
    Rank 0 is the prediction."""
    entries: list[dict[str, Any]] = [_flash_candidate(c) for c in flash]
    entries.extend(dict(p) for p in pysr)
    return sorted(entries, key=_sort_key)


def pick_prediction(
    values: Any,
    *,
    flash: Sequence[Mapping[str, Any]],
    equations: Sequence[Mapping[str, Any] | str] | None,
    engine: Any,
    weights: Mapping[str, float],
    x_support: np.ndarray,
    y_fit: np.ndarray,
    x_val: np.ndarray | None,
    y_val: np.ndarray | None,
    variables: Sequence[str],
    refine: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Flash-ANSR's sorting picks the prediction from the extended pool (in place); returns rank 0.

    PySR's own choice stays under ``pysr_expression`` / ``pysr_expression_prefix`` for reference; the
    ranked pool (source, index, score, fvu, mdl, n_nodes) is stored as ``hybrid_candidates`` and
    ``predicted_source`` names the origin of rank 0. When nothing could be priced the record keeps
    PySR's answer and ``hybrid_ranking_error`` says so; a failed GP stage (no hall of fame) leaves the
    Flash-ANSR candidates to rank alone, PySR's error kept under ``pysr_error``."""
    values["pysr_expression"] = values.get("predicted_expression")
    values["pysr_expression_prefix"] = values.get("predicted_expression_prefix")
    added = pysr_candidates(equations, engine=engine, weights=weights, x_support=x_support, y_fit=y_fit,
                            x_val=x_val, variables=variables, refine=refine)
    ranked = rank_candidates(flash, added)
    values["hybrid_candidates"] = [{k: e.get(k) for k in ("source", "hof_index", "score", "fvu", "mdl", "n_nodes")} for e in ranked]
    if not ranked:
        values["predicted_source"] = "pysr"
        values["hybrid_ranking_error"] = "no candidate could be priced"
        return None
    best = ranked[0]
    if best.get("y_pred") is None or not np.size(best.get("y_pred")):
        # a Flash-ANSR candidate stored without its predictions (a snapshot's `candidates` row)
        names = [f"x{i + 1}" for i in range(np.asarray(x_support).shape[1])]
        try:
            cols = evaluate_prefix(engine, list(best["expression_prefix"]), names, np.asarray(x_support, dtype=float),
                                   np.asarray(x_val, dtype=float) if x_val is not None and np.size(x_val) else np.empty((0, len(names))))
            best["y_pred"] = np.asarray(cols[0], dtype=float).reshape(-1)
            best["y_pred_val"] = np.asarray(cols[1], dtype=float).reshape(-1)
        except Exception:  # noqa: BLE001 - the prediction stays, its curve is missing
            pass
    values["predicted_source"] = best["source"]
    values["predicted_hof_index"] = best["hof_index"]
    values["predicted_expression"] = best["expression_infix"]
    values["predicted_expression_prefix"] = list(best["expression_prefix"])
    values["predicted_skeleton_prefix"] = list(best["skeleton_prefix"])
    values["predicted_constants"] = best["constants"]
    values["predicted_score"] = best["score"]
    values["predicted_mdl"] = best["mdl"]
    values["predicted_n_nodes"] = best["n_nodes"]
    values["predicted_log_prob"] = best["log_prob"]
    values["predicted_pareto_rank"] = best["pareto_rank"]
    n_fit = int(np.asarray(y_fit).reshape(-1).shape[0])
    y_pred = best.get("y_pred")
    values["y_pred"] = (np.asarray(y_pred, dtype=float).reshape(-1, 1) if y_pred is not None and np.size(y_pred)
                        else np.full((n_fit, 1), np.nan))
    y_pred_val = best.get("y_pred_val")
    values["y_pred_val"] = (np.asarray(y_pred_val, dtype=float).reshape(-1, 1)
                            if y_pred_val is not None and np.size(y_pred_val) else np.empty((0, 1)))
    if values.get("error") and not values.get("prediction_success"):
        values["pysr_error"] = values.get("error")
        values["error"] = None
    values["prediction_success"] = True
    y_sup = np.asarray(y_fit, dtype=float).reshape(-1, 1)
    if values["y_pred"].shape[0] == y_sup.shape[0]:
        values["support_fvu"] = _compute_fvu_from_predictions(y_sup, values["y_pred"])
    if y_val is not None and np.size(y_val):
        y_v = np.asarray(y_val, dtype=float).reshape(-1, 1)
        if values["y_pred_val"].shape[0] == y_v.shape[0]:
            values["validation_fvu"] = _compute_fvu_from_predictions(y_v, values["y_pred_val"])
    return best


class FlashANSRPySRAdapter(EvaluationModelAdapter):
    """Flash-ANSR (in-process) seeding PySR (a worker in its own environment) at a time budget."""

    def __init__(
        self,
        flash: FlashANSRAdapter,
        pysr: SubprocessAdapter,
        *,
        budget_s: float,
        ratio: float,
        ratios: Sequence[float],
        choices_law: TimeLaw,
        niterations_law: TimeLaw,
        snapshot_dir: str,
        k_seeds: int = 100,
        max_seed_complexity: int | None = None,
    ) -> None:
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("ratio must lie in [0, 1]")
        self.flash = flash
        self.pysr = pysr
        self.budget_s = float(budget_s)
        self.ratio = float(ratio)
        self.ratios = sorted(set(float(r) for r in ratios) | {self.ratio})
        self.choices_law = choices_law
        self.niterations_law = niterations_law
        self.snapshot_dir = Path(snapshot_dir)
        self.k_seeds = int(k_seeds)
        self.max_seed_complexity = max_seed_complexity

    # -- protocol -------------------------------------------------------------------------------
    def get_simplipy_engine(self) -> Any:
        return self.flash.get_simplipy_engine()

    def ranking_config(self) -> dict[str, Any] | None:
        return self.flash.ranking_config()

    def prepare(self, *, data_source: Any | None = None) -> None:  # type: ignore[override]
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.flash.prepare(data_source=data_source)
        if self.niterations_for(self.ratio) > 0:
            self.pysr.prepare(data_source=data_source)

    def close(self) -> None:
        close = getattr(self.pysr, "close", None)
        if callable(close):
            close()

    # -- the budget split ------------------------------------------------------------------------
    def choices_for(self, ratio: float) -> int:
        return self.choices_law.units_for((1.0 - ratio) * self.budget_s)

    def niterations_for(self, ratio: float) -> int:
        return self.niterations_law.units_for(ratio * self.budget_s)

    def choice_targets(self) -> list[int]:
        return sorted({self.choices_for(r) for r in self.ratios} - {0})

    # -- snapshots -------------------------------------------------------------------------------
    def _snapshot_path(self, record: Mapping[str, Any]) -> Path:
        pid = record.get("eval_row_index")
        if pid is None:
            raise RuntimeError("the hybrid adapter needs 'eval_row_index' on every sample (the data source stamps it)")
        return self.snapshot_dir / f"problem_{int(pid):06d}.pkl"

    def _snapshot(self, sample: EvaluationSample, record: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
        path = self._snapshot_path(record)
        fingerprint = data_fingerprint(sample)
        if path.exists():
            with path.open("rb") as fh:
                stored = pickle.load(fh)
            if isinstance(stored, Mapping) and "targets" in stored:
                if stored.get("fingerprint") != fingerprint:
                    warnings.warn(
                        f"hybrid snapshot {path.name} was generated on different data than this sample "
                        "(the data source re-drew the problem); regenerating. Every cell must read the same "
                        "frozen subset -- scripts/run_hybrid_sweep.py materializes it.")
                elif set(self.choice_targets()) <= set(stored["targets"]):
                    return stored["targets"]
        targets = self._generate_snapshots(sample, record)
        tmp = path.with_suffix(".pkl.tmp")
        with tmp.open("wb") as fh:
            pickle.dump({"fingerprint": fingerprint, "targets": targets}, fh)
        tmp.replace(path)
        return targets

    def _generate_snapshots(self, sample: EvaluationSample, record: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
        model = self.flash.model
        engine = model.simplipy_engine
        y_fit = sample.y_support_noisy if sample.y_support_noisy is not None else sample.y_support
        variable_names = record.get("variable_names")
        x_val = sample.x_validation if sample.x_validation.shape[0] > 0 else None
        complexity_value = self.flash._resolve_complexity(dict(record))
        numpy_errors = getattr(model, "numpy_errors", None)
        pysr_variables = self._pysr_variable_names(record, sample)
        arity = dict(getattr(engine, "operator_arity", {}) or {})

        targets = self.choice_targets()
        config = model.generation_config
        original_choices = getattr(config, "choices", None)
        pool: dict[tuple[int, ...], dict[str, Any]] = {}
        snap: dict[int, dict[str, Any]] = {}
        cum_wall = cum_gen = cum_ref = 0.0
        previous = 0
        try:
            for target in targets:
                delta = target - previous
                previous = target
                if delta <= 0:
                    continue
                config.choices = int(delta)
                t0 = time.time()
                try:
                    if numpy_errors is not None:
                        with np.errstate(all=numpy_errors):
                            result = model.infer(
                                sample.x_support, y_fit,
                                variable_names=variable_names if variable_names is not None else "auto",
                                X_val=x_val, complexity=complexity_value, emission=self.flash.emission,
                                predict_val=True, top_k=max(1, self.k_seeds))
                    else:
                        result = model.infer(
                            sample.x_support, y_fit,
                            variable_names=variable_names if variable_names is not None else "auto",
                            X_val=x_val, complexity=complexity_value, emission=self.flash.emission,
                            predict_val=True, top_k=max(1, self.k_seeds))
                except Exception as exc:  # noqa: BLE001 - a failed chunk leaves the pool as it was
                    warnings.warn(f"hybrid generation chunk of {delta} failed: {exc}")
                    result = None
                cum_wall += time.time() - t0
                if result is not None:
                    cum_gen += float(getattr(result, "generation_time", 0.0) or 0.0)
                    cum_ref += float(getattr(result, "refinement_time", 0.0) or 0.0)
                    for cand in result.candidates:
                        key = tuple(int(t) for t in cand.raw_beam) + ((cand.spelling,) if getattr(cand, "spelling", None) else ())
                        entry = self._candidate_dict(cand)
                        old = pool.get(key)
                        if old is None or _score_key(entry) < _score_key(old):
                            pool[key] = entry
                ranked = sorted(pool.values(), key=_score_key)
                seeds = self._seeds(ranked, arity, pysr_variables, sample, engine)
                snap[target] = {
                    "choices": target, "cum_wall": cum_wall, "cum_generation": cum_gen, "cum_refinement": cum_ref,
                    "pool_size": len(pool), "best": ranked[0] if ranked else None, "seeds": seeds,
                    # the top-K of the ranked pool without their prediction curves: the candidates PySR's join
                    "candidates": [ranked[0]] + [{k: v for k, v in c.items() if k not in ("y_pred", "y_pred_val")}
                                                 for c in ranked[1:max(1, self.k_seeds)]] if ranked else [],
                }
        finally:
            if original_choices is not None:
                config.choices = original_choices
        return snap

    @staticmethod
    def _candidate_dict(cand: Any) -> dict[str, Any]:
        return {
            "raw_beam": [int(t) for t in cand.raw_beam],
            "expression": list(cand.expression),
            "expression_prefix": list(cand.expression_prefix),
            "expression_infix": str(cand.expression_infix),
            "skeleton_prefix": list(cand.skeleton_prefix),
            "constants": list(cand.constants) if cand.constants is not None else None,
            "score": cand.score, "fvu": cand.fvu, "mdl": cand.mdl, "n_nodes": cand.n_nodes,
            "log_prob": cand.log_prob, "pareto_rank": cand.pareto_rank, "spelling": getattr(cand, "spelling", None),
            "y_pred": None if cand.y_pred is None else np.asarray(cand.y_pred, dtype=float).reshape(-1),
            "y_pred_val": None if cand.y_pred_val is None else np.asarray(cand.y_pred_val, dtype=float).reshape(-1),
        }

    @staticmethod
    def _pysr_variable_names(record: Mapping[str, Any], sample: EvaluationSample) -> list[str]:
        variables = list(record.get("variables") or record.get("variable_names") or [])
        n = int(np.asarray(sample.x_support).shape[1])
        if len(variables) != n:
            variables = [f"x{i + 1}" for i in range(n)]
        return variables

    def _seeds(self, ranked: Sequence[Mapping[str, Any]], arity: Mapping[str, int], variables: Sequence[str],
               sample: EvaluationSample, engine: Any) -> list[str]:
        """Top-K candidates as Julia-syntax guesses: finite over the support, inside PySR's
        vocabulary, under the complexity cap when one is set. Duplicates collapse."""
        out: list[str] = []
        seen: set[str] = set()
        X = np.asarray(sample.x_support, dtype=float)
        for cand in ranked:
            if len(out) >= self.k_seeds:
                break
            prefix = cand.get("expression_prefix") or []
            if not prefix or any(t == "<constant>" for t in prefix):
                continue
            if self.max_seed_complexity is not None and len(prefix) > self.max_seed_complexity:
                continue
            y_pred = cand.get("y_pred")
            if y_pred is None:
                try:
                    (y_pred,) = evaluate_prefix(engine, list(prefix), [f"x{i + 1}" for i in range(X.shape[1])], X)
                except Exception:  # noqa: BLE001 - an unevaluable candidate is not a seed
                    continue
            if not np.all(np.isfinite(np.asarray(y_pred, dtype=float))):
                continue
            text = prefix_to_julia(prefix, arity, variables)
            if text is None or text in seen:
                continue
            seen.add(text)
            out.append(text)
        return out

    # -- evaluation ------------------------------------------------------------------------------
    def evaluate_sample(self, sample: EvaluationSample) -> EvaluationResult:
        record = sample.clone_metadata()
        record["ranking"] = self.ranking_config()
        choices = self.choices_for(self.ratio)
        niterations = self.niterations_for(self.ratio)
        hybrid = {
            "hybrid_ratio": self.ratio, "hybrid_budget_s": self.budget_s, "hybrid_choices": choices,
            "hybrid_niterations": niterations, "hybrid_k_seeds": self.k_seeds,
            "hybrid_target_generation_s": (1.0 - self.ratio) * self.budget_s,
            "hybrid_target_gp_s": self.ratio * self.budget_s,
        }
        seeds: list[str] = []
        gen_wall = 0.0
        best: Mapping[str, Any] | None = None
        flash_candidates: list[Mapping[str, Any]] = []
        if choices > 0:
            try:
                snap = self._snapshot(sample, record)[choices]
            except Exception as exc:  # noqa: BLE001
                record.update(hybrid)
                record["error"] = f"hybrid generation failed: {exc}"
                record["prediction_success"] = False
                return EvaluationResult(record)
            gen_wall = float(snap["cum_wall"])
            seeds = list(snap["seeds"])[: self.k_seeds]
            best = snap["best"]
            # the Flash-ANSR candidates PySR's join: the snapshot's top-K pool (rank 0 carries its
            # predictions); a snapshot from before the pool was stored holds rank 0 alone, which the
            # same sorting puts first either way
            flash_candidates = list(snap.get("candidates") or ([best] if best is not None else []))
            hybrid.update(hybrid_generation_s=gen_wall, hybrid_pool_size=snap["pool_size"],
                          hybrid_generation_time=snap["cum_generation"], hybrid_refinement_time=snap["cum_refinement"],
                          hybrid_n_seeds=len(seeds))
            if best is not None:
                hybrid["hybrid_seed_best_expression"] = best.get("expression_infix")

        if niterations <= 0:
            # Flash-ANSR alone: the merged pool's rank 0 at this budget
            record.update(hybrid)
            if best is None:
                record["error"] = "hybrid: no Flash-ANSR candidate at this budget"
                record["prediction_success"] = False
                return EvaluationResult(record)
            record["fit_time"] = gen_wall
            record["generation_time"] = hybrid.get("hybrid_generation_time")
            record["refinement_time"] = hybrid.get("hybrid_refinement_time")
            record["predicted_source"] = "flash-ansr"
            record["prediction_success"] = True
            record["predicted_expression"] = best["expression_infix"]
            record["predicted_expression_prefix"] = normalize_expression(list(best["expression_prefix"]))
            record["predicted_skeleton_prefix"] = list(best["skeleton_prefix"])
            record["predicted_constants"] = best["constants"]
            record["predicted_score"] = best["score"]
            record["predicted_log_prob"] = best["log_prob"]
            record["predicted_mdl"] = best["mdl"]
            record["predicted_n_nodes"] = best["n_nodes"]
            record["predicted_pareto_rank"] = best["pareto_rank"]
            y_pred = best["y_pred"]
            record["y_pred"] = (np.asarray(y_pred, dtype=float).reshape(-1, 1) if y_pred is not None
                                else np.full((np.asarray(sample.y_support).shape[0], 1), np.nan))
            y_pred_val = best["y_pred_val"]
            record["y_pred_val"] = (np.asarray(y_pred_val, dtype=float).reshape(-1, 1) if y_pred_val is not None
                                    else np.empty((0, 1)))
            # the same FVU columns the worker path records
            from srbf.subprocess_adapter import _compute_fvu_from_predictions
            y_sup = np.asarray(sample.y_support_noisy if sample.y_support_noisy is not None else sample.y_support, dtype=float).reshape(-1, 1)
            if record["y_pred"].shape[0] == y_sup.shape[0]:
                record["support_fvu"] = _compute_fvu_from_predictions(y_sup, record["y_pred"])
            y_val = np.asarray(sample.y_validation_noisy if sample.y_validation_noisy is not None else sample.y_validation, dtype=float).reshape(-1, 1)
            if y_val.size and record["y_pred_val"].shape[0] == y_val.shape[0]:
                record["validation_fvu"] = _compute_fvu_from_predictions(y_val, record["y_pred_val"])
            return EvaluationResult(record)

        # the GP stage, seeded (or cold at r = 1)
        t0 = time.time()
        result = self.pysr.evaluate_sample(sample, extra_meta={"guesses": seeds, "niterations": int(niterations)})
        gp_wall = time.time() - t0
        values = result.to_mapping()
        values["ranking"] = record["ranking"]
        values.update(hybrid)
        values["hybrid_gp_s"] = gp_wall
        values["gp_fit_time"] = values.get("fit_time")
        values["fit_time"] = gen_wall + float(values.get("fit_time") or gp_wall)
        self._pick(values, flash_candidates, sample)
        return result

    def _pick(self, values: Any, flash: Sequence[Mapping[str, Any]], sample: EvaluationSample) -> None:
        """PySR's hall of fame joins the Flash-ANSR candidates and Flash-ANSR's sorting picks the
        prediction; the pricing time of the added candidates is part of the recorded fit time."""
        ranking = self.ranking_config()
        if not ranking:
            values["predicted_source"] = "pysr"
            values["hybrid_ranking_error"] = "no ranking config to price the candidates with"
            return
        t0 = time.time()
        y_fit = sample.y_support_noisy if sample.y_support_noisy is not None else sample.y_support
        y_val = sample.y_validation_noisy if sample.y_validation_noisy is not None else sample.y_validation
        pick_prediction(
            values, flash=flash, equations=values.get("equations"), engine=self.get_simplipy_engine(),
            weights=RankingConfig.from_dict(dict(ranking)).effective_weights,
            x_support=sample.x_support, y_fit=y_fit, x_val=sample.x_validation, y_val=y_val,
            # the FULL variable list, by column of x_support: the PySR stage may have dropped unused
            # columns (`variable_names` is then the reduced list); the candidates are priced on the full X
            variables=list(values.get("variables") or values.get("variable_names") or self._pysr_variable_names(values, sample)),
            refine=refine_settings(self.flash.model))
        values["hybrid_ranking_s"] = time.time() - t0
        values["fit_time"] = float(values.get("fit_time") or 0.0) + values["hybrid_ranking_s"]
