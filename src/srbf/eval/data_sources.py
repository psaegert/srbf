"""Evaluation data source: bridge a symbolic_data ProblemSource into srbf EvaluationSamples.

`CatalogSource` wraps a `symbolic_data.ProblemSource` (built from a catalog reference + sampling
policy -- e.g. ``v23-val`` / ``fastsrb`` resolved by name, or a generative config) and bridges each
`sd.Problem` into the `EvaluationSample` the benchmark driver consumes, adding the eval-only metadata
srbf records: the shared `build_base_metadata` block + tokenizer ``input_ids``/``labels`` + the
ground-truth expression + a resume-stable ``eval_row_index``.

This replaces the old `SkeletonDatasetSource` + `FastSRBSource`. The data layer (symbolic_data) now
owns ALL sampling -- on-the-fly generation, fixed-set iteration, noise, decontamination, and the
per-draw placeholder protocol -- so srbf no longer reaches into a skeleton pool, reproduces the
support sampler, or carries a pinned-skeleton-list drift guard: the frozen ``v23-val`` catalog
(deterministic, sha256-pinned on HF) IS the pinned, drift-safe evaluation set.
"""
from __future__ import annotations

from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from simplipy import normalize_skeleton

from srbf.eval.core import EvaluationDataSource, EvaluationSample
from srbf.eval.sample_metadata import build_base_metadata


class CatalogSource(EvaluationDataSource):
    """Stream srbf ``EvaluationSample``s from a ``symbolic_data.ProblemSource``."""

    def __init__(
        self,
        problem_source: Any,
        *,
        tokenizer: Any | None = None,
        target_size: int | None = None,
        skip: int = 0,
        tokenizer_oov: str = "unk",
        resume_state: Mapping[str, Any] | None = None,
    ) -> None:
        self.problem_source = problem_source
        self.tokenizer = tokenizer
        self.tokenizer_oov = tokenizer_oov
        self._target_size = None if target_size is None else max(0, int(target_size))
        self._skip = max(0, int(skip))
        self._engine: Any | None = None
        self._produced = 0
        if resume_state is not None:
            self.load_state_dict(resume_state)

    @classmethod
    def from_catalog(
        cls,
        catalog: str | Mapping[str, Any] | Any,
        *,
        sampling: Mapping[str, Any] | None = None,
        holdouts: Sequence[Mapping[str, Any]] | None = None,
        simplipy_engine: Any | None = None,
        tokenizer: Any | None = None,
        **kwargs: Any,
    ) -> "CatalogSource":
        """Build a `ProblemSource` from a catalog ref/config (+ usage policy) and wrap it."""
        from symbolic_data import ProblemSource

        config: dict[str, Any] = {"catalog": catalog}
        if sampling is not None:
            config["sampling"] = dict(sampling)
        if holdouts is not None:
            config["holdouts"] = list(holdouts)
        source = ProblemSource(config, simplipy_engine=simplipy_engine)
        return cls(source, tokenizer=tokenizer, **kwargs)

    # ----------------------------------------------------------------- driver hooks
    def size_hint(self) -> int | None:
        hint = self.problem_source.size_hint()
        if hint is None:
            return self._target_size  # unbounded generative source; only `target_size` bounds it
        available = max(0, int(hint) - self._skip)
        return available if self._target_size is None else min(self._target_size, available)

    def prepare(self, *, adapter: Any | None = None) -> None:  # type: ignore[override]
        """Share the adapter's simplipy engine with the source (so GT normalization agrees)."""
        engine = None
        if adapter is not None:
            get_engine = getattr(adapter, "get_simplipy_engine", None)
            engine = get_engine() if callable(get_engine) else getattr(adapter, "simplipy_engine", None)
        if engine is not None:
            self._engine = engine
            source_prepare = getattr(self.problem_source, "prepare", None)
            if callable(source_prepare):
                source_prepare(simplipy_engine=engine)

    def __iter__(self) -> Iterator[EvaluationSample]:
        target = self.size_hint()
        self._produced = 0
        for index, problem in enumerate(self.problem_source):
            if index < self._skip:
                continue
            if target is not None and self._produced >= target:
                break
            row_index = index  # absolute, resume-stable (skip-aware)
            if getattr(problem, "is_placeholder", False):
                sample = self._bridge_placeholder(problem, row_index)
            else:
                sample = self._bridge(problem, row_index)
            self._produced += 1
            yield sample

    # ----------------------------------------------------------------- resume
    def state_dict(self) -> Mapping[str, Any]:
        return {"type": "catalog_source", "state": {"row_index": self._skip + self._produced}}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._skip = max(self._skip, int(state.get("row_index", 0)))

    # ----------------------------------------------------------------- bridge
    def _bridge(self, problem: Any, row_index: int) -> EvaluationSample:
        skeleton = list(normalize_skeleton(list(problem.skeleton))) if problem.skeleton else None
        input_ids = self._encode_input_ids(skeleton) if (self.tokenizer is not None and skeleton) else None
        y_support_noisy = problem.y_support_noisy if problem.y_support_noisy is not None else problem.y_support
        y_validation_noisy = problem.y_validation_noisy if problem.y_validation_noisy is not None else problem.y_validation

        metadata = build_base_metadata(
            skeleton=skeleton,
            expression=problem.expression,
            variables=list(problem.variables) if problem.variables else None,
            x_support=problem.x_support,
            y_support=problem.y_support,
            x_validation=problem.x_validation,
            y_validation=problem.y_validation,
            y_support_noisy=y_support_noisy,
            y_validation_noisy=y_validation_noisy,
            noise_level=float(problem.noise) if isinstance(problem.noise, (int, float)) else 0.0,
            skeleton_hash=skeleton,
            labels_decoded=(self.tokenizer.decode(input_ids, special_tokens="<constant>")
                            if (input_ids is not None and self.tokenizer is not None) else None),
            complexity=problem.complexity,
        )
        metadata.update(self._eval_fields(problem, skeleton, input_ids, row_index))
        return EvaluationSample(
            x_support=problem.x_support,
            y_support=problem.y_support,
            x_validation=problem.x_validation,
            y_validation=problem.y_validation,
            y_support_noisy=problem.y_support_noisy,
            y_validation_noisy=problem.y_validation_noisy,
            metadata=metadata,
        )

    def _bridge_placeholder(self, problem: Any, row_index: int) -> EvaluationSample:
        reason = getattr(problem, "placeholder_reason", None) or "source_exhausted"
        metadata = build_base_metadata(
            skeleton=None, expression=None, variables=list(problem.variables) if problem.variables else None,
            x_support=problem.x_support, y_support=problem.y_support,
            x_validation=problem.x_validation, y_validation=problem.y_validation,
            y_support_noisy=problem.y_support if problem.y_support_noisy is None else problem.y_support_noisy,
            y_validation_noisy=problem.y_validation if problem.y_validation_noisy is None else problem.y_validation_noisy,
            noise_level=float(problem.noise) if isinstance(problem.noise, (int, float)) else 0.0,
            skeleton_hash=None, labels_decoded=None, complexity=None,
        )
        metadata.update(self._eval_fields(problem, None, None, row_index))
        metadata.update({
            "placeholder": True,
            "placeholder_reason": reason,
            "error": reason,
            "prediction_success": False,
        })
        return EvaluationSample(
            x_support=problem.x_support, y_support=problem.y_support,
            x_validation=problem.x_validation, y_validation=problem.y_validation,
            y_support_noisy=problem.y_support_noisy, y_validation_noisy=problem.y_validation_noisy,
            metadata=metadata, is_placeholder=True, placeholder_reason=reason,
        )

    def _eval_fields(self, problem: Any, skeleton: list[str] | None, input_ids: list[int] | None,
                     row_index: int) -> dict[str, Any]:
        """The eval-only metadata srbf adds on top of build_base_metadata."""
        gt_prefix = list(problem.expression) if problem.expression else None
        gt_infix = None
        if gt_prefix is not None and self._engine is not None:
            try:
                gt_infix = self._engine.prefix_to_infix(gt_prefix, realization=False)
            except Exception:  # noqa: BLE001 - infix rendering is best-effort metadata
                gt_infix = None
        return {
            "input_ids": np.asarray(input_ids, dtype=np.int64) if input_ids is not None else None,
            "labels": np.asarray(input_ids[1:], dtype=np.int64) if input_ids is not None else None,
            "constants": [np.asarray(problem.constants, dtype=np.float32)] if problem.constants else [],
            "benchmark_eq_id": problem.eq_id,
            "variable_names": list(problem.variables) if problem.variables else None,
            "ground_truth_prefix": gt_prefix,
            "ground_truth_infix": gt_infix,
            "eval_row_index": row_index,
        }

    def _encode_input_ids(self, skeleton_tokens: list[str]) -> list[int]:
        tokenizer = self.tokenizer
        body_tokens = skeleton_tokens
        if "<expression>" in tokenizer and "</expression>" in tokenizer:
            body_tokens = ["<expression>", *skeleton_tokens, "</expression>"]
        body_ids = tokenizer.encode(body_tokens, oov=self.tokenizer_oov)
        return [int(tokenizer["<bos>"]), *map(int, body_ids), int(tokenizer["<eos>"])]


__all__ = ["CatalogSource"]
