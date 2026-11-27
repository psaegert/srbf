"""The oracle: a problem's true formula with its constants refitted.

The oracle is a reference, not a method. It is handed what no method is handed, the problem's ground truth (its formula
over the columns ``x1 .. xN``, with its literals spelled), and only has to fit the constants. Every literal a fit may move
(SimpliPy's ``mask_fittable`` policy: a coefficient, an offset) becomes a free constant; a structural literal (a pow
exponent, a root index) stays as written. The constants are then fitted, priced and ranked by flash-ansr's own candidate
path, the one each Flash-ANSR candidate takes: ``n_restarts`` fits from random starting values, the fit in its canonical
form, the constant ladder (integers, fractions, multiples of pi and e), the duplicate with the typed literals free, and the
ranking that picks the answer among them. The number of restarts is the oracle's budget.

No checkpoint, tokenizer or vocabulary takes part: a formula over any number of variables, with any operator the SimpliPy
engine knows, is fitted as it is written.

:func:`refit_formula` fits one problem; :class:`srbf.model_adapters.OracleAdapter` runs it in a benchmark
(``model_adapter.type: oracle``).
"""
from __future__ import annotations

import dataclasses
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

#: The token of a free constant, as the refiner reads it.
CONSTANT = "<constant>"


def oracle_candidate(expression: Sequence[str], engine: Any) -> list[str] | None:
    """The ground truth as the refiner's candidate: each fittable literal (``mask_fittable``) a ``<constant>``, every
    other token as written. ``None`` when the engine cannot read the formula."""
    from flash_ansr.refine import refinement_slots

    tokens = [str(token) for token in expression]
    try:
        if not tokens or not engine.is_valid(tokens):
            return None
        slots = set(refinement_slots(tokens, engine, "fittable"))
    except Exception:  # noqa: BLE001 - a formula the engine cannot walk has no candidate
        return None
    candidate = [CONSTANT if index in slots else token for index, token in enumerate(tokens)]
    return candidate if engine.is_valid(candidate) else None


@dataclass(frozen=True)
class OracleCandidate:
    """One fitted spelling of the formula: ``expression`` with its ``<constant>`` slots, ``prefix`` and ``infix`` with
    the fitted constants at full precision (the prefix in ``x1 .. xN``, the infix in the problem's variable names),
    ``mdl`` in bits."""

    expression: list[str]
    prefix: list[str]
    infix: str
    skeleton: list[str]
    constants: list[float]
    fvu: float
    mdl: float | None
    score: float
    n_nodes: int
    pareto_rank: int
    spelling: str | None
    typed_frozen: int
    typed_thaw: str | None
    refiner: Any = field(repr=False, compare=False)

    def predict(self, X: Any) -> np.ndarray:
        """The candidate on ``X`` (``(n_points, n_columns)``) -> ``(n_points,)``."""
        return np.asarray(self.refiner.predict(np.asarray(X, dtype=float)), dtype=float).reshape(-1)


@dataclass(frozen=True)
class OracleFit:
    """The oracle on one problem: its candidates, best first (empty when nothing could be fitted, ``reason`` says
    why), and the seconds the fit took."""

    candidates: list[OracleCandidate]
    refinement_time: float
    reason: str | None = None

    @property
    def best(self) -> OracleCandidate | None:
        """The answer: the candidate the ranking puts first."""
        return self.candidates[0] if self.candidates else None


#: Why the oracle has no candidate when the engine cannot read the ground truth.
UNREADABLE = "the SimpliPy engine cannot read the ground truth"


def _settings(refine: Any) -> Any:
    return refine if hasattr(refine, "n_restarts") else refine_config(refine)


def _arrays(X: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    """``X`` as ``(n_points, n_columns)`` and ``y`` as ``(n_points, 1)`` floats; ``ValueError`` on a non-finite input."""
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    if not np.all(np.isfinite(X_arr)):
        raise ValueError("X contains non-finite values; the variables range over the reals")
    return X_arr, np.asarray(y, dtype=float).reshape(-1, 1)


def _prepared_candidate(expression: Sequence[str], engine: Any, typed_spans: str) -> tuple[list[str], int] | None:
    """The candidate as the refiner fits it first and the number of typed literals it keeps as written (Flash-ANSR's
    ``typed_frozen``): under ``typed_spans: refine`` the typed literals are freed before the first fit."""
    from flash_ansr.refine import thaw_typed_literals, typed_literal_sites

    candidate = oracle_candidate(expression, engine)
    if candidate is None:
        return None
    try:
        typed_frozen = len(typed_literal_sites(candidate, engine))
    except Exception:  # noqa: BLE001 - an untypeable formula keeps every literal as written
        typed_frozen = 0
    if typed_spans == "refine" and typed_frozen:
        thawed, _values = thaw_typed_literals(candidate, engine)
        return list(thawed), 0
    return candidate, typed_frozen


def _candidate_seed(seed: int | Sequence[int] | None) -> int:
    """The seed of the candidate's fit, from which flash-ansr also derives its duplicate's and its ladder's: ``seed``
    spread by numpy's ``SeedSequence``, fresh entropy when None."""
    return int(np.random.SeedSequence(seed).generate_state(1, dtype=np.uint32)[0])


def refine_config(settings: Mapping[str, Any] | Any | None = None) -> Any:
    """flash-ansr's ``RefineConfig`` (the refinement settings of a Flash-ANSR model) from a mapping of its keys, with
    flash-ansr's defaults for the keys the mapping leaves out: 8 restarts of Levenberg-Marquardt from N(0, 5^2)
    starting values, the fittable scope, the typed-span duplicate, the constant ladder."""
    from flash_ansr.estimator_config import RefineConfig

    return RefineConfig.from_mapping(settings)


def _fitted_rows(result: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """The fitted candidate, its typed-span duplicates and each one's re-spelled ladder variant, as Flash-ANSR's pool
    takes them: the parent first; a variant that only tied its parent replaces it, a strict improvement stands beside
    it; a duplicate never replaces anything."""
    if result is None:
        return []
    rows: list[dict[str, Any]] = []
    for parent in [result, *(result.get("thawed") or [])]:
        child = parent.get("respelled")
        if child is not None and child.get("replaces_parent"):
            rows.append(child)
        else:
            rows.extend(row for row in (parent, child) if row is not None)
    return [row for row in rows if row.get("fits") and row.get("valid_fit", False)]


def refit_formula(expression: Sequence[str], X: Any, y: Any, *, engine: Any, refine: Any = None, ranking: Any = None,
                  seed: int | Sequence[int] | None = None, variable_names: Sequence[str] | None = None) -> OracleFit:
    """Fit the constants of ``expression`` (the ground truth's prefix tokens over ``x1 .. xN``, literals spelled) to
    ``(X, y)``.

    ``refine`` holds the refinement settings (a ``RefineConfig`` or a mapping of its keys, :func:`refine_config`),
    ``ranking`` the ranking that orders the candidates (a flash-ansr ``RankingConfig``; the two-part code when None).
    ``seed`` (an int or a sequence of ints) fixes every random starting value, so that the same problem gives the same
    answer; None draws fresh entropy. ``variable_names`` names the columns in the infix spellings. Raises ``ValueError``
    on a non-finite input.
    """
    from flash_ansr.flash_ansr import FlashANSR, _refine_candidate_worker
    from flash_ansr.refine import Refiner
    from flash_ansr.scoring import count_constants, order_rows, resolve_ranking
    from symbolic_data.token_ops import normalize_skeleton

    from srbf.candidate_store import milli_bits_to_bits

    settings = _settings(refine)
    ranking = resolve_ranking("mdl") if ranking is None else ranking
    X_arr, y_arr = _arrays(X, y)
    finite = np.isfinite(y_arr[:, 0])
    n_points = int(finite.sum())
    # ddof=0 over the finite targets: the variance flash-ansr's FVU divides by, so the ranking's FVU is the judge's
    y_variance = float(np.var(y_arr[finite, 0])) if n_points > 1 else float("nan")
    n_inputs = int(X_arr.shape[1])

    prepared = _prepared_candidate(expression, engine, settings.typed_spans)
    if prepared is None:
        return OracleFit([], 0.0, reason=UNREADABLE)
    candidate, typed_frozen = prepared
    typed_spans = settings.typed_spans
    job_seed = _candidate_seed(seed)
    job = {
        "simplipy_engine": engine, "X": X_arr, "y": y_arr,
        "raw_beam": list(candidate), "beam": list(candidate), "raw_beam_decoded": list(candidate),
        "expression": list(candidate), "log_prob": float("nan"), "constant_count": count_constants(candidate),
        "p0": None, "pruned_variant": False, "typed_spans": typed_spans, "typed_frozen": typed_frozen,
        "n_variables": n_inputs, "n_restarts": settings.n_restarts, "method": settings.method,
        "p0_noise": settings.p0_noise,
        "p0_noise_kwargs": None if settings.p0_noise_kwargs is None else dict(settings.p0_noise_kwargs),
        "refine_scope": settings.scope, "constant_ladder": settings.ladder, "converge_error": "ignore",
        "numpy_errors": settings.numpy_errors, "y_variance": y_variance,
        "ranking_weights": ranking.weights_for(n_points), "complexity": None, "seed": job_seed,
    }
    started = time.perf_counter()
    with np.errstate(all=settings.numpy_errors) if settings.numpy_errors is not None else nullcontext():
        result, _warning = _refine_candidate_worker(job)
        rows = order_rows(FlashANSR._dedup_respelled(_fitted_rows(result)), ranking, n_points=n_points)
    refinement_time = time.perf_counter() - started

    mapping = ({f"x{i + 1}": str(name) for i, name in enumerate(variable_names)}
               if variable_names is not None and len(variable_names) == n_inputs else None)
    candidates: list[OracleCandidate] = []
    for row in rows:
        refiner = Refiner.from_serialized(simplipy_engine=engine, n_variables=n_inputs, expression=list(row["expression"]),
                                          n_inputs=n_inputs, fits=row["fits"],
                                          refine_scope=row.get("refine_scope", settings.scope))
        if not refiner.valid_fit or not refiner.all_constants_values:
            continue
        candidates.append(OracleCandidate(
            expression=list(row["expression"]),
            prefix=list(refiner.transform(list(row["expression"]), return_prefix=True)),
            infix=str(refiner.transform(list(row["expression"]), return_prefix=False, variable_mapping=mapping)),
            skeleton=list(normalize_skeleton(list(row["expression"])) or []),
            constants=np.asarray(refiner.all_constants_values[0][0], dtype=float).ravel().tolist(),
            fvu=float(row.get("fvu", float("nan"))),
            mdl=milli_bits_to_bits(row.get("mdl")),          # flash-ansr prices in thousandths of a bit
            score=float(row.get("score", float("nan"))),
            n_nodes=int(row.get("complexity", len(row["expression"]))),
            pareto_rank=int(row.get("pareto_rank", -1)),
            spelling=row.get("spelling"),
            typed_frozen=int(row.get("typed_frozen", 0) or 0),
            typed_thaw=row.get("typed_thaw"),
            refiner=refiner,
        ))
    return OracleFit(candidates, refinement_time, reason=None if candidates else "no restart converged")


@dataclass(frozen=True)
class Recovery:
    """The oracle stopped at its first numeric recovery (:func:`first_recovery`): ``fit`` is the oracle at the restart
    that recovered (None when none did), ``restarts`` the restarts spent, ``answered`` whether any restart converged
    (a restart that diverges everywhere leaves no answer)."""

    fit: OracleFit | None
    restarts: int
    answered: bool

    @property
    def recovered(self) -> bool:
        return self.fit is not None


def first_recovery(expression: Sequence[str], X: Any, y: Any, X_val: Any, y_val: Any, *, engine: Any, refine: Any = None,
                   ranking: Any = None, seed: int = 0, variable_names: Sequence[str] | None = None) -> Recovery:
    """The oracle stopped at its first numeric recovery: whether the formula can be recovered from ``(X, y)`` at all.

    Restart ``r`` (``0 .. n_restarts - 1`` of ``refine``) is the oracle at one restart from the seed ``(seed, r)``. Its
    first stage, the fit of the formula as written from that restart's starting values, is the same as the full
    oracle's; only a fit that reproduces the held-out targets ``y_val`` (``is_perfect_fit``) goes on to the rest (the
    canonical form, the constant ladder, the duplicate and the ranking), and the first restart whose answer reproduces
    them ends the search. A fit that misses the held-out targets is not passed on: the stages after it re-spell a fit
    and do not search for another. ``(X_val, y_val)`` are points the oracle is never fitted on.
    """
    from flash_ansr.refine import Refiner

    from srbf.metrics.numeric import is_perfect_fit

    settings = _settings(refine)
    X_arr, y_arr = _arrays(X, y)
    X_val_arr = np.asarray(X_val, dtype=float).reshape(-1, X_arr.shape[1])
    y_val_arr = np.asarray(y_val, dtype=float).reshape(-1)
    prepared = _prepared_candidate(expression, engine, settings.typed_spans)
    if prepared is None:
        return Recovery(None, 0, False)
    candidate, _typed_frozen = prepared
    one = dataclasses.replace(settings, n_restarts=1)
    answered = False
    for restart in range(settings.n_restarts):
        job_seed = _candidate_seed((int(seed), restart))
        state = np.random.get_state()
        np.random.seed(job_seed)             # the starting values the oracle's own first fit draws from this seed
        converged = perfect = False
        try:
            with np.errstate(all=settings.numpy_errors) if settings.numpy_errors is not None else nullcontext():
                first = Refiner(simplipy_engine=engine, n_variables=int(X_arr.shape[1])).fit(
                    expression=list(candidate), X=X_arr, y=y_arr, n_restarts=1, method=settings.method, p0=None,
                    p0_noise=settings.p0_noise, p0_noise_kwargs=(None if settings.p0_noise_kwargs is None
                                                                 else dict(settings.p0_noise_kwargs)),
                    converge_error="ignore", refine_scope=settings.scope)
                converged = bool(first.valid_fit)
                perfect = converged and is_perfect_fit(y_val_arr, np.asarray(first.predict(X_val_arr)).reshape(-1))
        except (NameError, KeyError, ValueError, OverflowError):   # unrealizable: as the full oracle, no candidate
            pass
        finally:
            np.random.set_state(state)
        answered = answered or converged
        if not perfect:
            continue
        fit = refit_formula(expression, X_arr, y_arr, engine=engine, refine=one, ranking=ranking, seed=(int(seed), restart),
                            variable_names=variable_names)
        if fit.best is not None and is_perfect_fit(y_val_arr, fit.best.predict(X_val_arr)):
            return Recovery(fit, restart + 1, True)
    return Recovery(None, settings.n_restarts, answered)


__all__ = ["CONSTANT", "OracleCandidate", "OracleFit", "Recovery", "UNREADABLE", "first_recovery", "oracle_candidate",
           "refine_config", "refit_formula"]
