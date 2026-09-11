"""Model adapter implementations for the evaluation engine."""
from __future__ import annotations

from pathlib import Path

import time
import warnings
import functools
import re
from contextlib import nullcontext
from typing import Any, Callable, Iterable, Mapping, TYPE_CHECKING

import numpy as np
from srbf.baselines import BruteForceModel, LampleChartonModel
from symbolic_data.token_ops import normalize_expression, normalize_skeleton
# sympy is imported lazily inside the two baseline adapters that use it (E2E, NeSymReS);
# it is an optional `[baselines]` extra, not a core runtime dependency.

from srbf.core import EvaluationModelAdapter, EvaluationResult, EvaluationSample
from srbf.candidate_store import CandidateStoreWriter
from flash_ansr.flash_ansr import FlashANSR
from flash_ansr.refine import ConvergenceError
from flash_ansr.scoring import compute_fvu

PySRRegressor: type[Any] | None  # pragma: no cover - assigned lazily
PySRRegressor = None

E2ERegressor: type[Any] | None  # pragma: no cover - assigned lazily
E2ERegressor = None

_torch_module: Any | None  # pragma: no cover - assigned lazily
_torch_module = None

try:  # pragma: no cover - optional dependency
    from nesymres.architectures.model import Model as _RuntimeNeSymResModel  # type: ignore
    _HAVE_NESYMRES = True
except Exception:  # pragma: no cover - optional dependency missing
    _RuntimeNeSymResModel = Any  # type: ignore
    _HAVE_NESYMRES = False

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from nesymres.architectures.model import Model as NesymresModel  # type: ignore
else:
    NesymresModel = Any


def _baseline_ranking_config(model: Any) -> dict[str, Any] | None:
    """The refiner baselines rank with the three loose penalties; spell that as the weighted mode so
    every row's `ranking` column reads the same way. None when the model has none of them."""
    names = (("node_penalty", "n_nodes"), ("constants_penalty", "n_constants"),
             ("likelihood_penalty", "neg_log_prob"))
    if not any(getattr(model, attr, None) is not None for attr, _ in names):
        return None
    weights = {metric: float(getattr(model, attr)) for attr, metric in names
               if getattr(model, attr, None) is not None}
    return {"mode": "weighted", "weights": {k: v for k, v in weights.items() if v != 0.0}}


class FlashANSRAdapter(EvaluationModelAdapter):
    """Wrap the `FlashANSR` model with the evaluation adapter protocol."""

    def __init__(
        self,
        model: FlashANSR,
        *,
        device: str = "cpu",
        complexity: str | list[int | float] | int | float = "none",
        emission: str = "fittable",
        refiner_workers: int | None = None,
        candidate_store_dir: str | None = None,
    ) -> None:
        self.model = model
        self.device = device
        # The promptable emission FORMAT the model is directed to use. 'fittable' (default,
        # the application mode -- owner ruling 2026-09-02) sends <mask_fittable>: the model
        # spells the typed literals and leaves every fittable constant as a placeholder for the
        # refiner; 'skeleton' sends <mask_all> so the refiner fits every slot from p0 noise;
        # 'constants' is the unflagged training format (the model spells everything). Validated
        # here for the same
        # reason as `complexity`: a bad string must not wait for the first problem.
        if emission not in ("constants", "skeleton", "fittable"):
            raise ValueError(
                f"emission must be 'constants', 'skeleton' or 'fittable'; got {emission!r}")
        self.emission = emission
        # Fail fast on an unknown mode: a bad string would otherwise only surface on the first problem,
        # AFTER the (slow) model load. Keep the accepted strings in sync with `_resolve_complexity`.
        if isinstance(complexity, str) and complexity not in ("none", "ground_truth"):
            raise ValueError(
                f"complexity string must be 'none' or 'ground_truth' (or pass an int/float/list); "
                f"got {complexity!r}"
            )
        self.complexity = complexity
        self.refiner_workers = refiner_workers
        # Save-all-candidates (thorough-tier quality shards only): when set, every problem's FULL
        # candidate ledger is streamed to a compact columnar store. Off (None) -> zero overhead, so
        # timing runs are untouched. The writer is created lazily on first capture. See STANDARD_EVAL.md
        # Section 7 + item 5.
        self.candidate_store_dir = candidate_store_dir
        self._candidate_store: Any | None = None

    def get_simplipy_engine(self) -> Any:  # pragma: no cover - trivial accessor
        """Return the SimpliPy engine backing this adapter's model."""
        return self.model.simplipy_engine

    def ranking_config(self) -> dict[str, Any] | None:
        """The resolved candidate ranking in force (flash-ansr's ``RankingConfig.as_dict()``): what
        goes into ``__meta__`` and every per-sample row."""
        fn = getattr(self.model, "ranking_config", None)
        return fn() if callable(fn) else None

    def prepare(self, *, data_source: Any | None = None) -> None:  # type: ignore[override]
        self.model.to(self.device).eval()
        if self.refiner_workers is not None:
            self.model.refiner_workers = self.refiner_workers

        # Fail HERE, once, before the campaign starts -- not per problem inside _capture_ledger,
        # whose `except Exception: warnings.warn(...)` would turn a store that cannot represent
        # this vocabulary into an EMPTY candidate store for the whole run while every eval row
        # still reports success. Measured 2026-08-27: the byte-alphabet vocabulary (95 -> 335)
        # crosses the writer's old uint8 bound and did exactly that.
        if self.candidate_store_dir is not None:
            from srbf.candidate_store import CandidateStoreWriter
            vocab_size = len(self.model.tokenizer)
            probe = Path(self.candidate_store_dir)
            probe.mkdir(parents=True, exist_ok=True)
            # The probe's manifest is the FIRST one on disk; give it the run meta too, or a run
            # killed before the writer's first periodic manifest leaves a manifest without it.
            CandidateStoreWriter(probe, vocab_size=vocab_size, run_meta=self._store_run_meta()).close()

    def evaluate_sample(self, sample: EvaluationSample) -> EvaluationResult:
        """Serial fit + evaluate via the model's own public inference API.

        ``FlashANSR.infer`` runs generation + constant refinement on one problem and returns an
        ``InferenceResult`` (the best candidate + the full classified candidate ledger), so this
        adapter is a THIN mapper -- no reaching into ``model._results`` / ``predict(nth_best_beam=...)``
        / ``get_expression`` / a generate-refine phase split. ``np.errstate`` restores the model's
        ``numpy_errors`` policy around the call (single-threaded; benign)."""
        record = sample.clone_metadata()
        record["ranking"] = self.ranking_config()

        y_fit = sample.y_support_noisy if sample.y_support_noisy is not None else sample.y_support
        complexity_value = self._resolve_complexity(record)
        variable_names = record.get("variable_names")
        x_val = sample.x_validation if sample.x_validation.shape[0] > 0 else None

        numpy_errors = getattr(self.model, "numpy_errors", None)
        fit_t0 = time.time()
        try:
            with np.errstate(all=numpy_errors) if numpy_errors is not None else nullcontext():
                result = self.model.infer(
                    sample.x_support, y_fit,
                    variable_names=variable_names if variable_names is not None else "auto",
                    X_val=x_val,
                    complexity=complexity_value,
                    emission=self.emission,
                    predict_val=True,
                    # The save-all tier needs every candidate's predictions (per-candidate validation
                    # FVU and recovery go into the store); timing runs keep the best-only path.
                    top_k='all' if self.candidate_store_dir is not None else 1,
                )
        except (ConvergenceError, OverflowError, TypeError, ValueError) as exc:
            record["error"] = str(exc)
            record["prediction_success"] = False
            return EvaluationResult(record)

        record["fit_time"] = time.time() - fit_t0
        record["generation_time"] = result.generation_time
        record["refinement_time"] = result.refinement_time

        # Full candidate ledger -> compact columnar store (save-all tier); off (None) => zero overhead.
        if self.candidate_store_dir is not None:
            self._capture_ledger(record, result, sample)

        best = result.best
        if best is None:
            warnings.warn("Model produced no results. Filling nan.")
            record["error"] = "Model produced no results."
            record["prediction_success"] = False
            return EvaluationResult(record)

        record["prediction_success"] = True
        record["predicted_expression"] = best.expression_infix
        # Candidate.expression_prefix is the RAW substituted prefix; normalize it to match the stored
        # form (best.skeleton_prefix is already normalized by infer()).
        record["predicted_expression_prefix"] = normalize_expression(list(best.expression_prefix))
        record["predicted_skeleton_prefix"] = list(best.skeleton_prefix)
        record["predicted_constants"] = list(best.constants) if best.constants is not None else None
        record["predicted_score"] = best.score
        record["predicted_log_prob"] = best.log_prob
        record["predicted_mdl"] = best.mdl
        record["predicted_n_nodes"] = best.n_nodes
        record["predicted_pareto_rank"] = best.pareto_rank

        y_pred = best.y_pred
        y_pred_val = best.y_pred_val if best.y_pred_val is not None else np.empty_like(sample.y_validation)
        record["y_pred"] = np.asarray(y_pred).copy() if y_pred is not None else np.empty_like(sample.y_support)
        record["y_pred_val"] = np.asarray(y_pred_val).copy()

        return EvaluationResult(record)

    # ------------------------------------------------------------------
    def _capture_ledger(self, record: dict[str, Any], result: Any, sample: EvaluationSample | None = None) -> None:
        """Stream this problem's FULL candidate ledger (from infer()) to the compact columnar store.

        The ledger is built by ``FlashANSR.infer`` (``result.ledger``: the generation pool U refined
        survivors, classified FIT_OK/FAILED/INVALID, with the ranking columns copied from the refined
        rows). Per-candidate validation FVU and recovery are added HERE from each candidate's
        ``y_pred`` / ``y_pred_val`` (``top_k='all'``) with the shared ``srbf.metrics.numeric``
        definitions -- never a hand-rolled copy. Best-effort, keyed by the resume-stable
        ``eval_row_index`` the data source stamped on the sample; failures warn and are swallowed --
        candidate capture must never abort an eval row."""
        try:
            problem_id = record.get("eval_row_index")
            if problem_id is None:
                warnings.warn(
                    "candidate_store_dir is set but the sample carries no 'eval_row_index'; skipping "
                    "candidate capture for this problem (the data source must stamp it).",
                    RuntimeWarning,
                )
                return
            if self._candidate_store is None:
                assert self.candidate_store_dir is not None  # _capture_ledger only runs when set
                self._candidate_store = CandidateStoreWriter(
                    self.candidate_store_dir, vocab_size=len(self.model.tokenizer),
                    run_meta=self._store_run_meta(),
                )
            if self._candidate_store.has_problem(int(problem_id)):
                return  # already written (resume)
            led = result.ledger
            extra: dict[str, Any] = {}
            for col in ("n_nodes", "n_constants", "mdl", "score", "pareto_rank", "rank", "spelling", "parent"):
                values = getattr(led, col, None)
                if values is not None and len(values) == len(led):
                    extra[col] = values
            result_index = getattr(led, "result_index", None)
            candidates = getattr(result, "candidates", None)
            if sample is not None and result_index is not None and candidates is not None and len(result_index) == len(led):
                from srbf.metrics.numeric import fvu as _fvu, is_perfect_fit as _perfect
                fvu_val = [float("nan")] * len(led)
                rec_fit = [0] * len(led)
                rec_val = [0] * len(led)
                for i, ri in enumerate(result_index):
                    if ri < 0:
                        continue
                    cand = candidates[ri]
                    if cand.y_pred is not None:
                        rec_fit[i] = int(bool(_perfect(sample.y_support, np.asarray(cand.y_pred))))
                    if cand.y_pred_val is not None and sample.y_validation.size:
                        yv = np.asarray(cand.y_pred_val)
                        fvu_val[i] = float(_fvu(sample.y_validation, yv))
                        rec_val[i] = int(bool(_perfect(sample.y_validation, yv)))
                extra.update(fvu_val=fvu_val, recovery_fit=rec_fit, recovery_val=rec_val)
            self._candidate_store.write_problem(
                int(problem_id), led.token_lists, led.fvu, led.log_prob,
                valid=led.valid, fit_status=led.fit_status, constants=led.constants, **extra,
            )
        except Exception as exc:  # noqa: BLE001 - capture is auxiliary; never break the eval row
            warnings.warn(f"Candidate-ledger capture failed for this problem: {exc}", RuntimeWarning)

    def _constant_ladder_config(self) -> dict[str, Any] | None:
        """The constant re-spelling settings the model refines under (flash-ansr ``constant_ladder``),
        or None when off -- a store reader must know whether ``spelling``/``parent`` rows can exist."""
        ladder = getattr(self.model, "constant_ladder", None)
        if ladder is None:
            return None
        to_dict = getattr(ladder, "to_dict", None)
        return dict(to_dict()) if callable(to_dict) else dict(ladder)

    def _store_run_meta(self) -> dict[str, Any]:
        """What a later reader of the candidate store needs once per run: the ranking that produced
        `score`/`rank`, and the dialect `mdl` was priced in (mu moves up to 1.8x between dialects, so a
        bare number is unusable later)."""
        meta: dict[str, Any] = {"ranking": self.ranking_config(),
                                "constant_ladder": self._constant_ladder_config(),
                                "mdl_dialect": {"certified": True, "mode": "f64", "canon": "default", "unit": "milli-bits"}}
        try:
            import simplipy
            meta["simplipy"] = simplipy.__version__
        except Exception:  # pragma: no cover - version is advisory
            pass
        engine = getattr(self.model, "simplipy_engine", None)
        for attr in ("name", "artifact", "artifact_name"):
            if getattr(engine, attr, None):
                meta["simplipy_engine"] = str(getattr(engine, attr))
                break
        return meta

    # ------------------------------------------------------------------

    def _resolve_complexity(self, metadata: dict[str, Any]) -> int | float | None:
        mode = self.complexity
        if isinstance(mode, (int, float)):
            return mode
        if isinstance(mode, list):
            return mode[0] if mode else None
        if mode == "none":
            return None
        if mode == "ground_truth":
            return metadata.get("complexity")
        raise NotImplementedError(f"Unsupported complexity configuration: {mode}")


class FlashANSRHybridAdapter(EvaluationModelAdapter):
    """Flash-ANSR seeding PySR at a time budget. The method is ``flash_ansr_hybrid.HybridRegressor``
    (Flash-ANSR generates by the clock, PySR searches from its seeds by the clock, PySR's hall of fame
    joins the Flash-ANSR candidates and Flash-ANSR's ranking picks); this adapter hands it each
    problem's arrays and records its answer the way every adapter does."""

    def __init__(self, flash: FlashANSRAdapter, regressor: Any) -> None:
        self.flash = flash
        self.regressor = regressor

    def get_simplipy_engine(self) -> Any:
        return self.flash.get_simplipy_engine()

    def ranking_config(self) -> dict[str, Any] | None:
        return self.flash.ranking_config()

    def prepare(self, *, data_source: Any | None = None) -> None:  # type: ignore[override]
        self.flash.prepare(data_source=data_source)
        self.regressor.prepare()

    def evaluate_sample(self, sample: EvaluationSample) -> EvaluationResult:
        record = sample.clone_metadata()
        record["ranking"] = self.ranking_config()
        y_fit = sample.y_support_noisy if sample.y_support_noisy is not None else sample.y_support
        y_val = sample.y_validation_noisy if sample.y_validation_noisy is not None else sample.y_validation
        variables = list(record.get("variables") or record.get("variable_names") or [])
        problem_id = record.get("eval_row_index")
        try:
            result = self.regressor.fit(
                sample.x_support, y_fit, X_val=sample.x_validation, variables=variables or None,
                problem_id=None if problem_id is None else int(problem_id),
                complexity=self.flash._resolve_complexity(dict(record)))
        except Exception as exc:  # noqa: BLE001 - the method's failure is the row's error, never the run's
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["prediction_success"] = False
            return EvaluationResult(record)
        record.update(result)
        # the FVU columns every adapter records, from the method's own curves
        y_pred = record.get("y_pred")
        y_sup = np.asarray(y_fit, dtype=float).reshape(-1, 1)
        if y_pred is not None and np.asarray(y_pred).shape[0] == y_sup.shape[0]:
            record["support_fvu"] = _compute_fvu_from_predictions(y_sup, np.asarray(y_pred, dtype=float))
        y_pred_val = record.get("y_pred_val")
        y_v = np.asarray(y_val, dtype=float).reshape(-1, 1) if y_val is not None else np.empty((0, 1))
        if y_v.size and y_pred_val is not None and np.asarray(y_pred_val).shape[0] == y_v.shape[0]:
            record["validation_fvu"] = _compute_fvu_from_predictions(y_v, np.asarray(y_pred_val, dtype=float))
        return EvaluationResult(record)


def _evaluate_refiner_baseline(model: Any, sample: EvaluationSample) -> EvaluationResult:
    """Shared evaluation logic for refiner-backed baseline models."""

    record = sample.clone_metadata()
    record["ranking"] = _baseline_ranking_config(model)

    y_fit = sample.y_support_noisy if sample.y_support_noisy is not None else sample.y_support

    fit_time_start = time.time()
    try:
        model.fit(sample.x_support, y_fit)
        record["fit_time"] = time.time() - fit_time_start
        record["prediction_success"] = True
    except Exception as exc:  # pragma: no cover - baseline errors vary
        record["error"] = str(exc)
        record["prediction_success"] = False
        return EvaluationResult(record)

    if not getattr(model, "_results", None):
        record["error"] = "Model produced no results."
        record["prediction_success"] = False
        return EvaluationResult(record)

    try:
        y_pred = model.predict(sample.x_support, nth_best=0)
        if sample.x_validation.size:
            y_pred_val = model.predict(sample.x_validation, nth_best=0)
        else:
            y_pred_val = np.empty_like(sample.y_validation)
        record["y_pred"] = np.asarray(y_pred).copy()
        record["y_pred_val"] = np.asarray(y_pred_val).copy()
    except Exception as exc:  # pragma: no cover - baseline errors vary
        record["error"] = str(exc)
        record["prediction_success"] = False
        return EvaluationResult(record)

    try:
        predicted_expression = model.get_expression(nth_best=0, return_prefix=False)
        predicted_prefix = model.get_expression(nth_best=0, return_prefix=True)
        record["predicted_expression"] = predicted_expression
        record["predicted_expression_prefix"] = normalize_expression(predicted_prefix)
        record["predicted_skeleton_prefix"] = normalize_skeleton(predicted_prefix)
    except Exception as exc:  # pragma: no cover - parse errors vary
        record["error"] = f"Failed to extract expression: {exc}"
        record["prediction_success"] = False
        return EvaluationResult(record)

    best_result = model._results[0]
    fits = best_result.get("fits")
    if fits:
        constants = np.asarray(fits[0][0]).tolist() if len(fits[0]) > 0 else None
        record["predicted_constants"] = constants
    record["predicted_score"] = best_result.get("score")
    record["predicted_log_prob"] = best_result.get("log_prob")

    return EvaluationResult(record)


class LampleChartonAdapter(EvaluationModelAdapter):
    """Adapter for the sampling-only `LampleChartonModel` baseline."""

    def __init__(self, model: LampleChartonModel) -> None:
        self.model = model

    def get_simplipy_engine(self) -> Any:  # pragma: no cover - trivial accessor
        """Return the SimpliPy engine backing this adapter's model."""
        return self.model.simplipy_engine

    def prepare(self, *, data_source: Any | None = None) -> None:  # noqa: ARG002
        return None

    def evaluate_sample(self, sample: EvaluationSample) -> EvaluationResult:
        return _evaluate_refiner_baseline(self.model, sample)


class BruteForceAdapter(EvaluationModelAdapter):
    """Adapter for the exhaustive `BruteForceModel` baseline."""

    def __init__(self, model: BruteForceModel) -> None:
        self.model = model

    def get_simplipy_engine(self) -> Any:  # pragma: no cover - trivial accessor
        """Return the SimpliPy engine backing this adapter's model."""
        return self.model.simplipy_engine

    def prepare(self, *, data_source: Any | None = None) -> None:  # noqa: ARG002
        return None

    def evaluate_sample(self, sample: EvaluationSample) -> EvaluationResult:
        return _evaluate_refiner_baseline(self.model, sample)


__all__ = [
    "FlashANSRAdapter",
    "E2EAdapter",
    "NeSymReSAdapter",
    "LampleChartonAdapter",
    "BruteForceAdapter",
]


class E2EAdapter(EvaluationModelAdapter):
    """Adapter for the End-to-end symbolic regression (E2E) baseline."""

    def __init__(
        self,
        *,
        model_path: str,
        simplipy_engine: Any,
        device: str = "cpu",
        candidates_per_bag: int = 1,
        max_input_points: int = 200,
        max_number_bags: int = 10,
        n_trees_to_refine: int = 10,
        rescale: bool = True,
        max_generated_output_len: int = 200,
        debug: bool = False,
    ) -> None:
        self.model_path = model_path
        self.simplipy_engine = simplipy_engine
        self.device = device
        self.candidates_per_bag = candidates_per_bag
        self.max_input_points = max_input_points
        self.max_number_bags = max_number_bags
        self.n_trees_to_refine = n_trees_to_refine
        self.rescale = rescale
        self.max_generated_output_len = max_generated_output_len
        self.debug = debug

        self._estimator: Any | None = None

    def get_simplipy_engine(self) -> Any:  # pragma: no cover - trivial accessor
        """Return this adapter's SimpliPy engine."""
        return self.simplipy_engine

    def prepare(self, *, data_source: Any | None = None) -> None:  # type: ignore[override]
        torch_mod = _require_torch()
        Estimator = _require_e2e_regressor()

        # Allowlist E2E classes for safe deserialization on torch>=2.6.
        add_safe_globals = getattr(torch_mod.serialization, "add_safe_globals", None)
        if add_safe_globals is not None:
            try:  # pragma: no cover - depends on optional dependency
                from symbolicregression.model.model_wrapper import ModelWrapper  # type: ignore
                from symbolicregression.model.embedders import LinearPointEmbedder  # type: ignore

                add_safe_globals([ModelWrapper, LinearPointEmbedder])
            except Exception:
                pass

        try:
            model = torch_mod.load(self.model_path, map_location=torch_mod.device(self.device))
        except Exception as exc:  # pragma: no cover - defensive retry
            if "Weights only load failed" not in str(exc):
                raise
            model = torch_mod.load(
                self.model_path,
                map_location=torch_mod.device(self.device),
                weights_only=False,
            )
        try:
            model.to(self.device)
        except Exception:  # pragma: no cover - defensive guard
            pass

        if hasattr(model, "beam_size"):
            model.beam_size = self.candidates_per_bag
        elif hasattr(model, "module") and hasattr(model.module, "beam_size"):
            model.module.beam_size = self.candidates_per_bag

        # Allow overriding generation length to keep chunking stable for large beam sizes.
        if hasattr(model, "max_generated_output_len"):
            model.max_generated_output_len = self.max_generated_output_len
        elif hasattr(model, "module") and hasattr(model.module, "max_generated_output_len"):
            model.module.max_generated_output_len = self.max_generated_output_len

        self._estimator = Estimator(
            model=model,
            max_input_points=self.max_input_points,
            max_number_bags=self.max_number_bags,
            n_trees_to_refine=self.n_trees_to_refine,
            rescale=self.rescale,
        )

    def evaluate_sample(self, sample: EvaluationSample) -> EvaluationResult:
        if self._estimator is None:
            raise RuntimeError("E2EAdapter.prepare must be called before evaluation")

        record = sample.clone_metadata()
        record["ranking"] = _baseline_ranking_config(self._estimator)

        X_support = sample.x_support.copy()
        X_val = sample.x_validation.copy()
        y_support = (sample.y_support_noisy if sample.y_support_noisy is not None else sample.y_support).copy()

        mask, used_variables = _compute_variable_mask(record.get("variables"), record.get("skeleton"))
        if mask is not None:
            X_support = X_support[:, mask]
            X_val = X_val[:, mask] if X_val.size else X_val
            if used_variables:
                record["variable_names"] = used_variables

        fit_time_start = time.time()
        try:
            self._estimator.fit(X_support, y_support, verbose=False)
            record["fit_time"] = time.time() - fit_time_start
            record["prediction_success"] = True
        except Exception as exc:  # pragma: no cover - upstream exceptions vary
            record["error"] = str(exc)
            record["prediction_success"] = False
            return EvaluationResult(record)

        try:
            y_pred = self._estimator.predict(X_support)
            y_pred_val = self._estimator.predict(X_val) if X_val.size else np.empty_like(sample.y_validation)
        except Exception as exc:  # pragma: no cover - defensive guard
            record["error"] = str(exc)
            record["prediction_success"] = False
            return EvaluationResult(record)

        if y_pred is None:
            record["error"] = "E2E returned no predictions"
            record["prediction_success"] = False
            return EvaluationResult(record)

        if y_pred_val is None:
            y_pred_val = np.empty_like(sample.y_validation)

        record["y_pred"] = np.asarray(y_pred).reshape(-1, 1)
        record["y_pred_val"] = np.asarray(y_pred_val).reshape(-1, 1)

        try:
            tree_info = self._estimator.retrieve_tree(with_infos=True)
            if isinstance(tree_info, list):
                tree_info = tree_info[0] if tree_info else None
            predicted_tree = None
            if isinstance(tree_info, Mapping):
                predicted_tree = tree_info.get("relabed_predicted_tree") or tree_info.get("predicted_tree")
            if predicted_tree is None:
                raise ValueError("E2E returned no tree")

            predicted_expression_raw = str(predicted_tree.infix())
            canonical_infix = _canonicalize_e2e_infix(predicted_expression_raw)
            try:
                import sympy as sp  # lazy: only the E2E baseline adapter needs sympy
                sympy_expr = sp.parse_expr(canonical_infix)
                predicted_expression = str(sympy_expr)
            except Exception:
                predicted_expression = canonical_infix

            # Normalize function casing for simplipy compatibility.
            predicted_expression = re.sub(r"\bAbs\b", "abs", predicted_expression)

            if self.debug:
                print("[E2EAdapter][debug] raw infix:", predicted_expression_raw, flush=True)
                print("[E2EAdapter][debug] canonical infix:", canonical_infix, flush=True)
                print("[E2EAdapter][debug] sympy infix:", predicted_expression, flush=True)

            record["predicted_expression"] = predicted_expression
            predicted_prefix = self.simplipy_engine.read_infix(predicted_expression)  # engine grammar, not the raw reader tokens
            record["predicted_expression_prefix"] = normalize_expression(predicted_prefix)
            record["predicted_skeleton_prefix"] = normalize_skeleton(predicted_prefix)

            if self.debug:
                print("[E2EAdapter][debug] prefix:", predicted_prefix, flush=True)
                print("[E2EAdapter][debug] skeleton:", record["predicted_skeleton_prefix"], flush=True)
        except Exception as exc:  # pragma: no cover - parse errors vary
            record["error"] = f"Failed to parse E2E expression: {exc}"
            record["prediction_success"] = False
            return EvaluationResult(record)

        return EvaluationResult(record)


class NeSymReSAdapter(EvaluationModelAdapter):
    """Adapter for NeSymReS models using the generic evaluation engine."""

    def __init__(
        self,
        model: NesymresModel,
        fitfunc: Callable[[np.ndarray, np.ndarray], dict[str, Any]],
        simplipy_engine: Any,
        *,
        device: str = "cpu",
        beam_width: int | None = None,
        remove_padding: bool = True,
        debug: bool = False,
    ) -> None:
        if not _HAVE_NESYMRES:  # pragma: no cover - defensive guard
            raise ImportError("The 'nesymres' package is required for NeSymReSAdapter")
        self.model = model
        self.fitfunc = fitfunc
        self.simplipy_engine = simplipy_engine
        self.device = device
        self.beam_width = beam_width
        self.remove_padding = remove_padding
        self.debug = debug
        self._fit_cfg_params: Any | None = None
        self._max_variables: int | None = None
        self._warned_feature_mismatch = False

    def get_simplipy_engine(self) -> Any:  # pragma: no cover - trivial accessor
        """Return this adapter's SimpliPy engine."""
        return self.simplipy_engine

    def prepare(self, *, data_source: Any | None = None) -> None:  # type: ignore[override]
        self.model.to(self.device).eval()
        cfg_params = _extract_cfg_params(self.fitfunc)
        self._fit_cfg_params = cfg_params
        if cfg_params is not None:
            total_vars = getattr(cfg_params, "total_variables", None)
            if isinstance(total_vars, Iterable):
                try:
                    self._max_variables = len(list(total_vars))
                except TypeError:  # pragma: no cover - defensive
                    self._max_variables = None
            if self.beam_width is not None and hasattr(cfg_params, "beam_size"):
                cfg_params.beam_size = self.beam_width

    def evaluate_sample(self, sample: EvaluationSample) -> EvaluationResult:
        record = sample.clone_metadata()
        record["ranking"] = _baseline_ranking_config(self.model)

        X_support = sample.x_support.copy()
        X_validation = sample.x_validation.copy()

        if self.remove_padding:
            variables = record.get("variables") or record.get("variable_names")
            mask, used_variables = _compute_variable_mask(variables, record.get("skeleton"))
            if mask is not None:
                X_support = X_support[:, mask]
                X_validation = X_validation[:, mask]
                if used_variables:
                    record["variable_names"] = used_variables

        X_support = self._prepare_inputs(X_support)
        X_validation = self._prepare_inputs(X_validation)
        y_fit = (sample.y_support_noisy if sample.y_support_noisy is not None else sample.y_support).reshape(-1)

        fit_time_start = time.time()
        try:
            nesymres_output = self.fitfunc(X_support, y_fit)
            record["fit_time"] = time.time() - fit_time_start
            record["prediction_success"] = True
        except Exception as exc:  # pragma: no cover - upstream exceptions vary
            record["error"] = str(exc)
            record["prediction_success"] = False
            return EvaluationResult(record)

        predicted_expr = _extract_first_prediction(
            nesymres_output,
            preferred_key="best_bfgs_preds",
            fallback_key="best_preds",
        )
        if predicted_expr is None:
            record["error"] = "NeSymReS returned no expression"
            record["prediction_success"] = False
            return EvaluationResult(record)

        try:
            predicted_expression = str(predicted_expr)
            record["predicted_expression"] = predicted_expression
            predicted_prefix = self.simplipy_engine.read_infix(predicted_expression)  # engine grammar, not the raw reader tokens
            record["predicted_expression_prefix"] = normalize_expression(predicted_prefix)
            record["predicted_skeleton_prefix"] = normalize_skeleton(predicted_prefix)
        except Exception as exc:  # pragma: no cover - parse errors
            record["error"] = f"Failed to parse NeSymReS expression: {exc}"
            record["prediction_success"] = False
            return EvaluationResult(record)

        predicted_constants = _extract_first_prediction(
            nesymres_output,
            preferred_key="best_bfgs_consts",
            fallback_key="best_consts",
        )
        if predicted_constants is not None:
            record["predicted_constants"] = _convert_constants(predicted_constants)

        try:
            y_pred, y_pred_val = _evaluate_symbolic_expression(
                predicted_expr,
                X_support,
                X_validation,
            )
            record["y_pred"] = y_pred
            record["y_pred_val"] = y_pred_val

            support_targets = sample.y_support_noisy if sample.y_support_noisy is not None else sample.y_support
            support_fvu = _compute_fvu_from_predictions(support_targets, y_pred)
            record["support_fvu"] = support_fvu

            validation_targets = (
                sample.y_validation_noisy if sample.y_validation_noisy is not None else sample.y_validation
            )
            validation_fvu: float | None = None
            if validation_targets.size and y_pred_val.size:
                validation_fvu = _compute_fvu_from_predictions(validation_targets, y_pred_val)
                record["validation_fvu"] = validation_fvu

            if self.debug:
                _print_fvu_summary(support_fvu, validation_fvu)
        except Exception as exc:  # pragma: no cover - evaluation errors
            record["error"] = f"Failed to evaluate NeSymReS expression: {exc}"
            record["prediction_success"] = False

        return EvaluationResult(record)

    def _prepare_inputs(self, array: np.ndarray) -> np.ndarray:
        if not isinstance(array, np.ndarray) or self._max_variables is None:
            return array
        n_features = array.shape[1]
        if n_features == self._max_variables:
            return array
        if n_features > self._max_variables:
            if not self._warned_feature_mismatch:
                warnings.warn(
                    (
                        "NeSymReS checkpoint supports only %d variables; "
                        "truncating inputs from %d features."
                    )
                    % (self._max_variables, n_features),
                    RuntimeWarning,
                )
                self._warned_feature_mismatch = True
            return array[:, : self._max_variables].copy()
        pad_width = self._max_variables - n_features
        pad = np.zeros((array.shape[0], pad_width), dtype=array.dtype)
        return np.concatenate([array, pad], axis=1)


# ---------------------------------------------------------------------------
# Helper utilities

def _compute_variable_mask(
    variables: Iterable[str] | None,
    skeleton_tokens: Iterable[str] | None,
) -> tuple[np.ndarray | None, list[str] | None]:
    if not variables or not skeleton_tokens:
        return None, None
    skeleton_set = set(skeleton_tokens)
    mask = []
    kept = []
    for var in variables:
        keep = var in skeleton_set
        mask.append(keep)
        if keep:
            kept.append(var)
    if not any(mask):
        return None, None
    return np.array(mask, dtype=bool), kept


def _extract_cfg_params(fitfunc: Any) -> Any:
    if hasattr(fitfunc, "cfg_params"):
        return fitfunc.cfg_params
    if isinstance(fitfunc, functools.partial):  # type: ignore[name-defined]
        keywords = fitfunc.keywords or {}
        return keywords.get("cfg_params")
    return None


def _convert_constants(constants: Any) -> list[float] | Any:
    if isinstance(constants, np.ndarray):
        return constants.tolist()
    if isinstance(constants, (list, tuple)):
        return list(constants)
    return constants


def _evaluate_symbolic_expression(predicted_expr: Any, X_support: np.ndarray, X_val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from sympy import lambdify  # lazy: only the NeSymReS baseline adapter needs sympy
    var_symbols = [f"x_{idx + 1}" for idx in range(X_support.shape[1])]
    evaluate_expression = lambdify(var_symbols, predicted_expr, "numpy")
    y_pred = np.asarray(evaluate_expression(*X_support.T), dtype=float).reshape(-1, 1)
    if X_val.size > 0:
        y_pred_val = np.asarray(evaluate_expression(*X_val.T), dtype=float).reshape(-1, 1)
    else:
        y_pred_val = np.empty((0, 1), dtype=float)
    return y_pred, y_pred_val


def _extract_first_prediction(
    output: Mapping[str, Any] | None,
    *,
    preferred_key: str,
    fallback_key: str | None = None,
) -> Any:
    if not isinstance(output, Mapping):
        return None
    candidate = _first_non_none(output.get(preferred_key))
    if candidate is not None:
        return candidate
    if fallback_key is not None:
        return _first_non_none(output.get(fallback_key))
    return None


def _first_non_none(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            if item is not None:
                return item
        return None
    return value


def _require_torch() -> Any:
    global _torch_module
    if _torch_module is not None:
        return _torch_module
    try:  # pragma: no cover - optional dependency
        import torch as _torch  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise ImportError("PyTorch is required for the E2E adapter") from exc
    _torch_module = _torch
    return _torch_module


def _require_e2e_regressor() -> type[Any]:
    global E2ERegressor
    if E2ERegressor is not None:
        return E2ERegressor
    try:  # pragma: no cover - optional dependency
        from symbolicregression.model.sklearn_wrapper import SymbolicTransformerRegressor as _E2ERegressor  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise ImportError(
            "symbolicregression is not installed; install the E2E dependencies to use the E2E adapter",
        ) from exc
    E2ERegressor = _E2ERegressor
    return E2ERegressor


def _compute_fvu_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_arr = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred_arr = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true_arr.size == 0 or y_pred_arr.size == 0:
        return float("nan")
    loss = float(np.mean((y_true_arr - y_pred_arr) ** 2))
    variance = float(np.var(y_true_arr))
    return compute_fvu(loss, y_true_arr.size, variance)


def _print_fvu_summary(support_fvu: float, validation_fvu: float | None) -> None:
    support_str = _format_fvu_value(support_fvu)
    message = f"[NeSymReSAdapter] support FVU={support_str}"
    if validation_fvu is not None:
        message = f"{message} | validation FVU={_format_fvu_value(validation_fvu)}"
    print(message, flush=True)


def _format_fvu_value(value: float) -> str:
    if np.isnan(value):
        return "nan"
    if np.isposinf(value):  # pragma: no cover - defensive
        return "+inf"
    if np.isneginf(value):  # pragma: no cover - defensive
        return "-inf"
    return f"{value:.6g}"


def _canonicalize_e2e_infix(expr: str) -> str:
    """Map E2E operator tokens to standard infix symbols before SymPy parsing."""
    replacements = {
        "add": "+",
        "sub": "-",
        "mul": "*",
        "pow": "**",
        "inv": "1/",
    }
    # Tokenize on whitespace and parentheses to avoid touching identifiers like "mulx_0".
    spaced = re.sub(r"([()])", r" \1 ", expr)
    tokens = spaced.split()
    mapped = [replacements.get(tok, tok) for tok in tokens]
    return " ".join(mapped)
