"""Flash-ANSR seeding + PySR at a fixed time budget: the hybrid arm.

One budget T per problem is split by a ratio r: Flash-ANSR gets (1 - r) T, PySR gets r T. The budget
is controlled by DIRECT knobs, never by timeouts: two measured time laws turn a share of T into a
number of Flash-ANSR candidates (``choices``) and a number of PySR iterations (``niterations``); the
achieved wall time of both stages is recorded next to the target.

r = 0: Flash-ANSR alone, its rank-0 answer. r = 1: PySR alone, cold. In between: the top-K refined
Flash-ANSR candidates enter PySR as initial ``guesses`` and PySR's own answer is returned.

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
from symbolic_data.token_ops import normalize_expression

from srbf.core import EvaluationModelAdapter, EvaluationResult, EvaluationSample
from srbf.model_adapters import FlashANSRAdapter
from srbf.subprocess_adapter import SubprocessAdapter, evaluate_prefix

__all__ = ["FlashANSRPySRAdapter", "TimeLaw", "prefix_to_julia", "data_fingerprint", "PYSR_OPERATORS"]

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
        return result
