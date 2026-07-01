from collections import defaultdict
from typing import Any, Generator, Literal

import numpy as np
import pandas as pd
import torch
from simplipy import SimpliPyEngine
from simplipy.utils import construct_expressions

from symbolic_data import LampleChartonCatalog

from ._base import _RefiningBaselineModel, RefinerMethod


class BruteForceModel(_RefiningBaselineModel):
    """Exhaustive baseline that enumerates expressions in increasing length.

    Expressions are generated shortest-first using ``simplipy.utils.construct_expressions``
    over the operator and variable vocabulary defined by the provided
    ``LampleChartonCatalog``. Each candidate is refined with the shared ``Refiner`` to
    fit constants against user-supplied data.
    """

    def __init__(
        self,
        *,
        simplipy_engine: SimpliPyEngine,
        catalog: str | dict[str, Any] | LampleChartonCatalog,
        max_expressions: int = 10_000,
        max_length: int | None = None,
        include_constant_token: bool = True,
        ignore_holdouts: bool = True,
        n_restarts: int = 8,
        refiner_method: RefinerMethod = 'curve_fit_lm',
        refiner_p0_noise: Literal['uniform', 'normal'] | None = 'normal',
        refiner_p0_noise_kwargs: dict | Literal['default'] | None = 'default',
        numpy_errors: Literal['ignore', 'warn', 'raise', 'call', 'print', 'log'] | None = 'ignore',
        length_penalty: float = 0.05,
        constants_penalty: float = 0.0,
        likelihood_penalty: float = 0.0,
    ) -> None:
        self.max_expressions = int(max_expressions)
        self.max_length = max_length
        self.include_constant_token = include_constant_token
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

    def _leaf_nodes(self) -> list[str]:
        leaves = list(self._pool.variables)
        if self.include_constant_token:
            leaves.append('<constant>')
        return leaves

    def _non_leaf_nodes(self) -> dict[str, int]:
        operator_weights = self._pool.operator_weights or {}
        return {op: arity for op, arity in self.simplipy_engine.operator_arity.items() if operator_weights.get(op, 0) > 0}

    def _expression_generator(self) -> Generator[tuple[str, ...], None, None]:
        hashes_by_size: defaultdict[int, set[tuple[str, ...]]] = defaultdict(set)
        seen: set[tuple[str, ...]] = set()

        for leaf in self._leaf_nodes():
            expr: tuple[str, ...] = (leaf,)
            hashes_by_size[1].add(expr)
            seen.add(expr)
            yield expr
            if len(seen) >= self.max_expressions:
                return

        target_length = 2
        while len(seen) < self.max_expressions:
            new_expressions: list[tuple[str, ...]] = []
            for expr in construct_expressions(hashes_by_size, self._non_leaf_nodes(), must_have_sizes=None):
                expr_len = len(expr)
                if self.max_length is not None and expr_len > self.max_length:
                    continue
                if expr_len != target_length:
                    continue
                if expr in seen:
                    continue
                if not self.simplipy_engine.is_valid(list(expr)):
                    continue

                seen.add(expr)
                new_expressions.append(expr)
                yield expr
                if len(seen) >= self.max_expressions:
                    break

            if not new_expressions:
                break

            hashes_by_size[target_length].update(new_expressions)
            target_length += 1

    def fit(self, X: np.ndarray | torch.Tensor | pd.DataFrame, y: np.ndarray | torch.Tensor | pd.DataFrame | Any, *, verbose: bool = False) -> "BruteForceModel":
        """Enumerate expressions shortest-first, refine each against ``(X, y)``, and store the results.

        Candidates are drawn from the exhaustive generator (bounded by ``max_expressions`` /
        ``max_length``) and sorted by ascending score.

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
        BruteForceModel
            ``self``, with the sorted candidate records available on ``results``.
        """
        self._run_fit(self._expression_generator(), X, y, max_results=self.max_expressions)
        return self
