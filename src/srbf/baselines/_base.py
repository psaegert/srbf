import copy
from typing import Any, Iterable, Literal, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator
from simplipy import SimpliPyEngine

from symbolic_data import LampleChartonCatalog, build_catalog
from flash_ansr.refine import Refiner, ConvergenceError
from flash_ansr.scoring import compute_fvu, count_constants, is_constant_token, normalize_variance, score_from_fvu
from flash_ansr.utils.paths import substitute_root_path

# The optimizer backends the shared Refiner accepts; single source for both baselines' signatures.
RefinerMethod = Literal[
    'curve_fit_lm',
    'minimize_bfgs',
    'minimize_lbfgsb',
    'minimize_neldermead',
    'minimize_powell',
    'least_squares_trf',
    'least_squares_dogbox',
]


class _RefiningBaselineModel(BaseEstimator):
    """Shared machinery for the constant-refining baselines.

    Both baselines fit constants of candidate skeletons with the same ``Refiner`` and build the same
    per-candidate result records; they differ only in *where the skeletons come from* (an exhaustive
    generator for :class:`~srbf.baselines.BruteForceModel`, a catalog sample for
    :class:`~srbf.baselines.LampleChartonModel`) and in a few extra ``__init__`` knobs. This base
    owns the catalog handling, the X/y coercion, the per-candidate refine/score/build, the fit loop
    (sort by score), and ``predict`` / ``get_expression``.

    Subclasses keep their own explicit ``__init__`` signature (sklearn ``get_params`` introspects the
    concrete class), call :meth:`_init_refiner_common` to store the shared attributes, and implement
    ``fit`` as a thin call to :meth:`_run_fit` over their skeleton source.
    """

    FLOAT64_EPS: float = float(np.finfo(np.float64).eps)

    def _init_refiner_common(
        self,
        *,
        simplipy_engine: SimpliPyEngine,
        catalog: str | dict[str, Any] | LampleChartonCatalog,
        ignore_holdouts: bool,
        n_restarts: int,
        refiner_method: RefinerMethod,
        refiner_p0_noise: Literal['uniform', 'normal'] | None,
        refiner_p0_noise_kwargs: dict | Literal['default'] | None,
        numpy_errors: Literal['ignore', 'warn', 'raise', 'call', 'print', 'log'] | None,
        length_penalty: float,
        constants_penalty: float,
        likelihood_penalty: float,
    ) -> None:
        '''Store the attributes common to both baselines. Called from each subclass ``__init__``.'''
        self.simplipy_engine = simplipy_engine
        self.ignore_holdouts = ignore_holdouts
        self.n_restarts = n_restarts
        self.refiner_method = refiner_method
        self.refiner_p0_noise = refiner_p0_noise
        if refiner_p0_noise_kwargs == 'default':
            refiner_p0_noise_kwargs = {'loc': 0.0, 'scale': 5.0}
        self.refiner_p0_noise_kwargs = copy.deepcopy(refiner_p0_noise_kwargs) if refiner_p0_noise_kwargs is not None else None
        self.numpy_errors = numpy_errors
        self.length_penalty = float(length_penalty)
        self.constants_penalty = float(constants_penalty)
        self.likelihood_penalty = float(likelihood_penalty)

        self._pool = self._ensure_pool(catalog)
        self._results: list[dict[str, Any]] = []
        self.results: pd.DataFrame = pd.DataFrame()
        self._input_dim: int | None = None

    @property
    def n_variables(self) -> int:
        """Number of input variables of the skeleton pool (the model's fixed input dimension)."""
        return self._pool.n_variables

    def _ensure_pool(self, catalog_ref: str | dict[str, Any] | LampleChartonCatalog) -> LampleChartonCatalog:
        # build_catalog resolves a name[@version] (HF), a local path/file, an inline {type: ...} dict, or
        # an existing Catalog instance -- uniform with the data_source catalog resolution.
        ref = substitute_root_path(catalog_ref) if isinstance(catalog_ref, str) else catalog_ref
        pool = build_catalog(ref)
        if not isinstance(pool, LampleChartonCatalog):
            raise TypeError(
                f"`catalog` must resolve to a generative LampleChartonCatalog (to sample skeletons); "
                f"got {type(pool).__name__}"
            )
        if self.ignore_holdouts:
            pool.clear_holdouts()
        return pool

    def _truncate_input(self, X: np.ndarray) -> np.ndarray:
        n_features = X.shape[-1]
        if n_features == self.n_variables:
            return X
        if n_features < self.n_variables:
            pad_width = self.n_variables - n_features
            pad = np.zeros((*X.shape[:-1], pad_width), dtype=X.dtype)
            return np.concatenate([X, pad], axis=-1)

        return X[..., : self.n_variables]

    # --- thin wrappers over flash_ansr.scoring (kept for the models' internal API surface) ---
    @staticmethod
    def _normalize_variance(variance: float) -> float:
        return normalize_variance(variance)

    @staticmethod
    def _compute_fvu(loss: float, sample_count: int, variance: float) -> float:
        return compute_fvu(loss, sample_count, variance)

    @staticmethod
    def _is_constant_token(token: str) -> bool:
        return is_constant_token(token)

    @staticmethod
    def _count_constants(expression: Sequence[str]) -> int:
        return count_constants(expression)

    @staticmethod
    def _score_from_fvu(
            fvu: float,
            complexity: int,
            constant_count: int,
            log_prob: float | None,
            length_penalty: float,
            constants_penalty: float,
            likelihood_penalty: float) -> float:
        return score_from_fvu(
            fvu, complexity, constant_count, log_prob,
            length_penalty, constants_penalty, likelihood_penalty)

    def _coerce_xy(self, X: np.ndarray | torch.Tensor | pd.DataFrame, y: np.ndarray | torch.Tensor | pd.DataFrame | Sequence[float]) -> tuple[np.ndarray, np.ndarray, int, float]:
        '''Coerce ``X``/``y`` to ``float`` numpy (truncate/pad X to ``n_variables``, single-column y),
        set ``self._input_dim``, and return ``(X_np, y_np, sample_count, y_variance)``.'''
        if len(np.shape(y)) == 1:
            y = np.reshape(y, (-1, 1))

        if isinstance(X, torch.Tensor):
            X_np = X.detach().cpu().numpy()
        elif isinstance(X, pd.DataFrame):
            X_np = X.values
        else:
            X_np = np.asarray(X)

        if isinstance(y, torch.Tensor):
            y_np = y.detach().cpu().numpy()
        elif isinstance(y, (pd.DataFrame, pd.Series)):
            y_np = y.values
        else:
            y_np = np.asarray(y)

        if y_np.ndim == 1:
            y_np = y_np.reshape(-1, 1)
        elif y_np.shape[-1] != 1:
            raise ValueError("The target data must have a single output dimension.")

        X_np = self._truncate_input(np.asarray(X_np))
        self._input_dim = X_np.shape[1]

        sample_count = y_np.shape[0]
        if sample_count <= 1:
            y_variance = float('nan')
        else:
            y_variance = float(np.var(y_np, axis=0, ddof=0).item())  # ddof=0: match the evaluation-side FVU definition (flash-ansr 0.11.0)

        return X_np, y_np, sample_count, y_variance

    def _refine_one(self, expression_tokens: list[str], X_np: np.ndarray, y_np: np.ndarray, sample_count: int, y_variance: float) -> dict[str, Any] | None:
        '''Refine one skeleton's constants and build its result record, or ``None`` to skip it
        (non-convergence, no fit, invalid fit, or non-finite loss/FVU).'''
        try:
            refiner = Refiner(self.simplipy_engine, n_variables=self.n_variables).fit(
                expression=expression_tokens,
                X=X_np,
                y=y_np,
                n_restarts=self.n_restarts,
                method=self.refiner_method,
                p0=None,
                p0_noise=self.refiner_p0_noise,
                p0_noise_kwargs=copy.deepcopy(self.refiner_p0_noise_kwargs) if self.refiner_p0_noise_kwargs is not None else None,
                converge_error='ignore',
            )
        except ConvergenceError:
            return None

        # Accept constant-free expressions even though Refiner.valid_fit stays False in that case.
        if len(refiner.all_constants_values) == 0:
            return None

        has_constants = len(refiner.constants_symbols) > 0
        valid_fit = refiner.valid_fit or not has_constants
        if not valid_fit:
            return None

        loss = float(refiner.all_constants_values[0][-1])
        if not np.isfinite(loss):
            return None

        fvu = self._compute_fvu(loss, sample_count, y_variance)
        if not np.isfinite(fvu):
            return None

        constant_count = self._count_constants(expression_tokens)
        score = self._score_from_fvu(
            fvu,
            len(expression_tokens),
            constant_count,
            None,
            self.length_penalty,
            self.constants_penalty,
            self.likelihood_penalty,
        )

        return {
            'log_prob': float('nan'),
            'fvu': fvu,
            'score': score,
            'expression': expression_tokens,
            'constant_count': constant_count,
            'complexity': len(expression_tokens),
            'requested_complexity': None,
            'raw_beam': expression_tokens,
            'beam': expression_tokens,
            'raw_beam_decoded': ' '.join(expression_tokens),
            'function': refiner.expression_lambda,
            'refiner': refiner,
            'fits': copy.deepcopy(refiner.all_constants_values),
            'prompt_metadata': None,
        }

    def _run_fit(self, skeletons: Iterable[Sequence[str]], X: np.ndarray | torch.Tensor | pd.DataFrame, y: np.ndarray | torch.Tensor | pd.DataFrame | Sequence[float], *, max_results: int | None = None) -> "_RefiningBaselineModel":
        '''Refine every skeleton in ``skeletons`` (stopping once ``max_results`` records are built),
        sort the records by ascending score, and store them on the model.'''
        X_np, y_np, sample_count, y_variance = self._coerce_xy(X, y)

        results: list[dict[str, Any]] = []
        # np.errstate restores the global error state on exit even if the loop raises a
        # non-ConvergenceError (is_valid / construct_expressions / Refiner), so 'ignore' never
        # leaks process-wide and silently suppresses downstream overflow/divide/invalid warnings.
        with np.errstate(all=self.numpy_errors):
            for skeleton in skeletons:
                expression_tokens = list(skeleton)
                entry = self._refine_one(expression_tokens, X_np, y_np, sample_count, y_variance)
                if entry is None:
                    continue
                results.append(entry)
                if max_results is not None and len(results) >= max_results:
                    break

        # Lower scores are better because they combine log-scaled FVU with penalties.
        results.sort(key=lambda item: item['score'])

        self._results = results
        self.results = pd.DataFrame(results)
        return self

    def predict(self, X: np.ndarray | torch.Tensor | pd.DataFrame, nth_best: int = 0) -> np.ndarray:
        """Evaluate the ``nth_best`` fitted candidate's refined expression on ``X``.

        Parameters
        ----------
        X : np.ndarray or torch.Tensor or pd.DataFrame
            Input points; truncated or zero-padded to ``n_variables`` like at fit time.
        nth_best : int, default 0
            Rank (by ascending score) of the fitted candidate to evaluate; 0 is the best.

        Returns
        -------
        np.ndarray
            Model predictions for ``X``.

        Raises
        ------
        ValueError
            If the model has not been fitted (``fit`` produced no results).
        IndexError
            If ``nth_best`` exceeds the number of fitted candidates.
        """
        if not self._results:
            raise ValueError("The model has not been fitted yet. Please call `fit` first.")

        if nth_best >= len(self._results):
            raise IndexError(f"nth_best={nth_best} is out of range for {len(self._results)} results.")

        refiner = self._results[nth_best]['refiner']

        if isinstance(X, torch.Tensor):
            X_np = X.detach().cpu().numpy()
        elif isinstance(X, pd.DataFrame):
            X_np = X.values
        else:
            X_np = np.asarray(X)

        X_np = self._truncate_input(np.asarray(X_np))
        return refiner.predict(X_np)

    def get_expression(self, nth_best: int = 0, *, return_prefix: bool = False, precision: int = 2) -> list[str] | str:
        """Return the ``nth_best`` fitted candidate's expression with its refined constants substituted.

        Parameters
        ----------
        nth_best : int, default 0
            Rank (by ascending score) of the fitted candidate to return; 0 is the best.
        return_prefix : bool, default False
            If True, return the prefix token list; otherwise return the infix string.
        precision : int, default 2
            Number of decimal places used when formatting the substituted constants.

        Returns
        -------
        list of str or str
            The prefix token list (``return_prefix=True``) or the infix expression string.

        Raises
        ------
        ValueError
            If the model has not been fitted (``fit`` produced no results).
        IndexError
            If ``nth_best`` exceeds the number of fitted candidates.
        """
        if not self._results:
            raise ValueError("The model has not been fitted yet. Please call `fit` first.")

        if nth_best >= len(self._results):
            raise IndexError(f"nth_best={nth_best} is out of range for {len(self._results)} results.")

        refiner = self._results[nth_best]['refiner']
        return refiner.transform(
            self._results[nth_best]['expression'],
            nth_best_constants=0,
            return_prefix=return_prefix,
            precision=precision,
        )
