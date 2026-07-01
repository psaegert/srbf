import math
import random
import warnings
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from simplipy import SimpliPyEngine

from symbolic_data import LampleChartonCatalog, NoValidSampleFoundError
from flash_ansr.results import (
    RESULTS_FORMAT_VERSION,
    deserialize_results_payload,
    load_results_payload,
    save_results_payload,
    serialize_results_payload,
)

from ._base import _RefiningBaselineModel, RefinerMethod


class LampleChartonModel(_RefiningBaselineModel):
    """Baseline model that samples skeletons from a catalog and fits constants.

    This model is intended for research/ablation baselines that do **not** use
    the Flash-ANSR transformer. It samples expression skeletons from a provided
    ``LampleChartonCatalog`` and refines their constants against user-provided data
    using the same `Refiner` used by `flash_ansr.flash_ansr.FlashANSR`.
    """

    def __init__(
        self,
        *,
        simplipy_engine: SimpliPyEngine,
        catalog: str | dict[str, Any] | LampleChartonCatalog,
        samples: int = 32,
        unique: bool = True,
        ignore_holdouts: bool = True,
        seed: int | None = None,
        n_restarts: int = 8,
        refiner_method: RefinerMethod = 'curve_fit_lm',
        refiner_p0_noise: Literal['uniform', 'normal'] | None = 'normal',
        refiner_p0_noise_kwargs: dict | Literal['default'] | None = 'default',
        numpy_errors: Literal['ignore', 'warn', 'raise', 'call', 'print', 'log'] | None = 'ignore',
        length_penalty: float = 0.05,
        constants_penalty: float = 0.0,
        likelihood_penalty: float = 0.0,
    ) -> None:
        self.samples = samples
        self.unique = unique
        self.seed = seed
        self._init_refiner_common(
            simplipy_engine=simplipy_engine,
            catalog=catalog,
            ignore_holdouts=ignore_holdouts,
            n_restarts=n_restarts,
            refiner_method=refiner_method,
            refiner_p0_noise=refiner_p0_noise,
            refiner_p0_noise_kwargs=refiner_p0_noise_kwargs,
            numpy_errors=numpy_errors,
            length_penalty=length_penalty,
            constants_penalty=constants_penalty,
            likelihood_penalty=likelihood_penalty,
        )

    def _sample_skeletons(self) -> list[tuple[str, ...]]:
        if self.samples <= 0:
            return []

        rng = random.Random()
        if self.seed is not None:
            rng.seed(int(self.seed))

        cached_skeletons = list(self._pool.skeletons)
        if cached_skeletons:
            unique_sampling = self.unique and self.samples <= len(cached_skeletons)
            if self.unique and not unique_sampling:
                warnings.warn(
                    "Requested unique samples exceeds pool size; sampling with replacement instead.",
                    RuntimeWarning,
                    stacklevel=2,
                )

            if unique_sampling:
                return rng.sample(cached_skeletons, k=self.samples)
            return [rng.choice(cached_skeletons) for _ in range(self.samples)]

        selected: list[tuple[str, ...]] = []
        seen: set[tuple[str, ...]] = set()
        # Bound the draws: a pool that persistently fails to sample, or (with unique=True) that has
        # fewer distinct valid skeletons than requested, would otherwise loop here forever.
        max_attempts = max(1000, self.samples * 100)
        for _ in range(max_attempts):
            if len(selected) >= self.samples:
                break
            try:
                skeleton, _code, _constants = self._pool.sample_skeleton(new=True, decontaminate=not self.ignore_holdouts)
            except NoValidSampleFoundError as exc:
                if not selected:
                    raise RuntimeError("Unable to sample skeletons from the pool configuration.") from exc
                continue

            skeleton_tuple = tuple(skeleton)
            if self.unique and skeleton_tuple in seen:
                continue
            seen.add(skeleton_tuple)
            selected.append(skeleton_tuple)

        if len(selected) < self.samples:
            warnings.warn(
                f"Collected only {len(selected)} of {self.samples} requested skeletons within "
                f"{max_attempts} attempts (pool sampling failed or unique skeletons were exhausted).",
                RuntimeWarning,
                stacklevel=2,
            )
        return selected

    def fit(self, X: np.ndarray | torch.Tensor | pd.DataFrame, y: np.ndarray | torch.Tensor | pd.DataFrame | Any, *, verbose: bool = False) -> "LampleChartonModel":
        """Sample skeletons from the catalog, refine each against ``(X, y)``, and store the results.

        Candidates come from :meth:`_sample_skeletons` (``samples`` draws, optionally ``unique``) and
        are sorted by ascending score.

        Parameters
        ----------
        X : np.ndarray or torch.Tensor or pd.DataFrame
            Support inputs; truncated or zero-padded to ``n_variables``.
        y : np.ndarray or torch.Tensor or pd.DataFrame
            Support targets (single output dimension).
        verbose : bool, default False
            Accepted but currently unused.

        Returns
        -------
        LampleChartonModel
            ``self``, with the sorted candidate records available on ``results``.
        """
        # An empty skeleton sample yields an empty (but well-formed) result set via _run_fit.
        self._run_fit(self._sample_skeletons(), X, y)
        return self

    def save_results(self, path: str) -> None:
        """Persist fitted results (without lambdas) for reuse."""

        if not self._results:
            raise ValueError("No results available to save. Run `fit` first.")

        input_dim = self._input_dim if self._input_dim is not None else self.n_variables
        metadata = {
            "format_version": RESULTS_FORMAT_VERSION,
            "length_penalty": self.length_penalty,
            "constants_penalty": self.constants_penalty,
            "likelihood_penalty": self.likelihood_penalty,
            "n_variables": self.n_variables,
            "input_dim": input_dim,
            "variable_mapping": None,
        }

        payload = serialize_results_payload(self._results, metadata=metadata)
        save_results_payload(payload, path)

    def load_results(self, path: str, *, rebuild_refiners: bool = True) -> None:
        """Load previously saved results and rebuild refiners if requested."""

        payload = load_results_payload(path)
        metadata = payload.get("metadata", {})

        version = int(payload.get("version", 0))
        if version != RESULTS_FORMAT_VERSION:
            warnings.warn(
                f"Results payload version {version} does not match expected {RESULTS_FORMAT_VERSION}; attempting to proceed anyway."
            )

        self.length_penalty = float(metadata.get("length_penalty", self.length_penalty))
        self.constants_penalty = float(metadata.get("constants_penalty", self.constants_penalty))
        self.likelihood_penalty = float(metadata.get("likelihood_penalty", self.likelihood_penalty))
        n_variables = int(metadata.get("n_variables", self.n_variables))
        input_dim = int(metadata.get("input_dim", n_variables))

        self._input_dim = input_dim

        restored = deserialize_results_payload(
            payload,
            simplipy_engine=self.simplipy_engine,
            n_variables=n_variables,
            input_dim=input_dim,
            rebuild_refiners=rebuild_refiners,
        )

        self._results = sorted(
            restored,
            key=lambda item: item.get("score", float("inf")) if not math.isnan(item.get("score", float("nan"))) else float("inf"),
        )
        self.results = pd.DataFrame(self._results)
