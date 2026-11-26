"""The polynomial family: least-squares polynomials of total degree 0 to 10, the answer picked by cross-validation."""
from __future__ import annotations

import itertools
import math
import time
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from threadpoolctl import threadpool_limits


def n_monomials(n_variables: int, degree: int) -> int:
    """The number of monomials of total degree at most ``degree`` in ``n_variables`` variables, C(d + k, k)."""
    return math.comb(n_variables + degree, degree)


def monomial_exponents(n_variables: int, max_degree: int) -> np.ndarray:
    """The exponents of every monomial of total degree at most ``max_degree``, one row per monomial: degree 0 first,
    then degree 1, 2, ..., each degree in lexicographic order (``x1**2, x1*x2, x2**2``). The first
    ``n_monomials(n_variables, k)`` rows are the monomials of the polynomial of degree ``k``."""
    rows: list[list[int]] = []
    for degree in range(max_degree + 1):
        for factors in itertools.combinations_with_replacement(range(n_variables), degree):
            exponents = [0] * n_variables
            for j in factors:
                exponents[j] += 1
            rows.append(exponents)
    return np.asarray(rows, dtype=np.int64).reshape(len(rows), n_variables)


def _solve(design: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The least-squares weights of ``design``'s columns for ``y``: each column scaled by the power of two nearest its
    root-mean-square, solved by LAPACK's SVD-based solver, and the scales taken back out (exactly: powers of two)."""
    rms = np.sqrt(np.mean(design * design, axis=0))
    _, exponent = np.frexp(rms)
    shift = np.where(rms > 0, -exponent, 0)
    weights = np.linalg.lstsq(np.ldexp(design, shift), y, rcond=None)[0]
    return np.ldexp(weights, shift)


class PolynomialModel(BaseEstimator):
    """Polynomials of total degree 0, 1, ..., ``max_degree`` in the problem's variables, each fitted by least squares
    on the fitting points; the answer is picked by ``n_folds``-fold cross-validation with the one-standard-error rule.

    **The family.** The polynomial of degree ``k`` has every monomial ``x1**a1 * ... * xd**ad`` of total degree
    ``a1 + ... + ad <= k``, C(d + k, k) of them for d variables. A degree is fitted only while it has at most as many
    monomials as there are fitting points; the higher degrees are skipped, so a problem with many variables stops early.

    **The answer.** The fitting points are shuffled (``seed``) and split into ``n_folds`` folds whose sizes differ by
    at most one (as many folds as points when there are fewer). Each fitted degree is solved as below on all folds but
    one and its mean squared error taken on the fold left out; its CV error is the mean over the folds and its standard
    error the folds' standard deviation over the square root of their number. The answer is the lowest degree whose CV
    error is at most the smallest CV error plus the standard error of the degree that attains it. Without a finite CV
    error (a single point) the answer is degree 0.

    A fold's error below one rounding unit of the targets' variance (float64 epsilon times the variance of the fitting
    targets, the floor srbf puts under log10 FVU) is rounding noise, and counts as that floor. On noiseless data every
    degree from the law's own up fits to rounding noise, and the higher ones fit the targets' rounding a little more
    closely; on the floor these degrees tie, and the rule picks the lowest of them.

    **The fit.** On inputs of wide range the monomials of high degree are nearly collinear and of very different
    magnitudes, so the normal equations are never formed. Each variable is divided by a power of two,
    ``2**e_j`` with ``max|x_j| < 2**e_j <= 2 max|x_j|``, so that every scaled variable lies in (-1, 1); each monomial
    column of the scaled variables is scaled by the power of two nearest its root-mean-square; and the least-squares
    problem in these columns is solved by ``numpy.linalg.lstsq`` (LAPACK ``gelsd``, a singular value decomposition),
    which discards the directions whose singular value is below ``eps * max(n, M)`` times the largest: where columns
    are numerically dependent, it returns the minimum-norm solution. Every scale is a power of two, so the coefficient
    of ``x**a`` in the original variables is the solved weight times a power of two: the conversion back is exact, as
    long as the coefficient stays within float64's normal range. The fit is not centred: a centred polynomial would
    have to be expanded into the original variables, and its expansion rounded, whereas here the solved coefficients
    are the written ones. The expression is the fitted polynomial, coefficient for coefficient, and computes its
    predictions up to the rounding of its terms; where the inputs lie far from zero the terms of high degree cancel,
    and any evaluation of the expanded polynomial, the fit's own included, is correspondingly less precise.

    After :meth:`fit`:

    * ``degrees_``: the fitted degrees, ascending; ``n_monomials_``: the number of monomials of each;
    * ``coefficients_``: for each fitted degree, the coefficients of its monomials in the original variables
      (``exponents_[:n_monomials]`` order);
    * ``cv_mse_``, ``cv_se_``: each fitted degree's CV error and its standard error (NaN without folds);
    * ``degree_``: the degree the one-standard-error rule picks, the answer;
    * ``timing_``: seconds spent on the fits (``fit``) and on the cross-validation (``cross_validation``).
    """

    def __init__(self, *, max_degree: int = 10, n_folds: int = 5, seed: int = 0) -> None:
        self.max_degree = max_degree
        self.n_folds = n_folds
        self.seed = seed

    # ---- fitting -----------------------------------------------------------------------------------------------
    def fit(self, X: Any, y: Any) -> "PolynomialModel":
        """Fit every degree the fitting points allow on ``(X, y)`` and pick the answer by cross-validation.

        Parameters
        ----------
        X : array-like of shape (n, d)
            The fitting points; a one-dimensional array is one variable.
        y : array-like of shape (n,) or (n, 1)
            The targets.

        Returns
        -------
        PolynomialModel
            ``self``, fitted.

        Raises
        ------
        ValueError
            If ``max_degree`` or ``n_folds`` is out of range, the shapes disagree, there are no points, or a value is
            not finite.
        """
        if int(self.max_degree) < 0:
            raise ValueError(f"max_degree must be at least 0; got {self.max_degree}")
        if int(self.n_folds) < 2:
            raise ValueError(f"n_folds must be at least 2; got {self.n_folds}")
        X_np = np.asarray(X, dtype=np.float64)
        if X_np.ndim == 1:
            X_np = X_np.reshape(-1, 1)
        y_np = np.asarray(y, dtype=np.float64)
        if y_np.ndim == 2 and y_np.shape[1] == 1:
            y_np = y_np[:, 0]
        if X_np.ndim != 2 or y_np.ndim != 1 or X_np.shape[0] != y_np.shape[0]:
            raise ValueError(f"X of shape {np.shape(X)} and y of shape {np.shape(y)} do not describe the same points")
        n, d = X_np.shape
        if n == 0:
            raise ValueError("no fitting points")
        if not (np.isfinite(X_np).all() and np.isfinite(y_np).all()):
            raise ValueError("the fitting points or targets are not all finite")

        top = 0 if d == 0 else int(self.max_degree)       # without variables every degree is the constant
        degrees = [k for k in range(top + 1) if n_monomials(d, k) <= n]
        self.n_variables_ = d
        self.exponents_ = monomial_exponents(d, degrees[-1])
        self._scale_exponents = np.frexp(np.max(np.abs(X_np), axis=0, initial=0.0))[1].astype(np.int64)
        design = self._design(X_np, len(self.exponents_))

        # The solves are small (at most as many columns as points); a BLAS thread pool costs more than it saves on
        # them, and far more on a busy machine, so they run on one BLAS thread.
        with threadpool_limits(limits=1, user_api="blas"):
            started = time.perf_counter()
            weights: dict[int, np.ndarray] = {}
            for k in degrees:
                weights[k] = _solve(design[:, :n_monomials(d, k)], y_np)
            fitted = time.perf_counter()
            cv_mse, cv_se = self._cross_validate(design, y_np, degrees)
            self.timing_ = {"fit": fitted - started, "cross_validation": time.perf_counter() - fitted}

        self.degrees_ = degrees
        self.n_monomials_ = [n_monomials(d, k) for k in degrees]
        self._weights = weights
        shifts = self.exponents_ @ self._scale_exponents
        self.coefficients_ = {k: np.ldexp(weights[k], -shifts[:n_monomials(d, k)]) for k in degrees}
        self.cv_mse_ = cv_mse
        self.cv_se_ = cv_se
        self.degree_ = self._one_standard_error_pick(degrees, cv_mse, cv_se)
        return self

    def _design(self, X: np.ndarray, n_columns: int) -> np.ndarray:
        """The first ``n_columns`` monomials of the scaled variables at the rows of ``X``."""
        exponents = self.exponents_[:n_columns]
        scaled = np.ldexp(X, -self._scale_exponents)
        design = np.ones((X.shape[0], n_columns), dtype=np.float64)
        top = int(exponents.max(initial=0))
        for j in range(X.shape[1]):
            powers = np.cumprod(np.concatenate([np.ones((X.shape[0], 1)), np.repeat(scaled[:, j:j + 1], top, axis=1)],
                                               axis=1), axis=1)
            design *= powers[:, exponents[:, j]]
        return design

    def _cross_validate(self, design: np.ndarray, y: np.ndarray, degrees: list[int]) -> tuple[np.ndarray, np.ndarray]:
        n = y.shape[0]
        n_folds = min(int(self.n_folds), n)
        cv_mse = np.full(len(degrees), np.nan)
        cv_se = np.full(len(degrees), np.nan)
        if n_folds < 2:
            return cv_mse, cv_se
        folds = np.array_split(np.random.default_rng(int(self.seed)).permutation(n), n_folds)
        floor = float(np.finfo(np.float64).eps) * float(np.var(y))
        for i, k in enumerate(degrees):
            columns = n_monomials(self.n_variables_, k)
            errors = np.empty(n_folds)
            for f, held_out in enumerate(folds):
                train = np.ones(n, dtype=bool)
                train[held_out] = False
                with np.errstate(all="ignore"):
                    weights = _solve(design[train, :columns], y[train])
                    residual = design[held_out, :columns] @ weights - y[held_out]
                    errors[f] = np.mean(residual * residual)
            errors = np.maximum(errors, floor)
            if np.isfinite(errors).all():
                cv_mse[i] = float(np.mean(errors))
                cv_se[i] = float(np.std(errors, ddof=1) / math.sqrt(n_folds))
            else:
                cv_mse[i] = np.inf
        return cv_mse, cv_se

    @staticmethod
    def _one_standard_error_pick(degrees: list[int], cv_mse: np.ndarray, cv_se: np.ndarray) -> int:
        finite = np.isfinite(cv_mse)
        if not finite.any():
            return degrees[0]
        best = int(np.argmin(np.where(finite, cv_mse, np.inf)))
        threshold = cv_mse[best] + cv_se[best]
        return next(k for k, error in zip(degrees, cv_mse) if error <= threshold)

    # ---- using the fit -----------------------------------------------------------------------------------------
    def _degree(self, degree: int | None) -> int:
        if not hasattr(self, "degree_"):
            raise ValueError("the model has not been fitted yet; call `fit` first")
        if degree is None:
            return int(self.degree_)
        if degree not in self.degrees_:
            raise ValueError(f"degree {degree} was not fitted; the fitted degrees are {self.degrees_}")
        return int(degree)

    def predict(self, X: Any, degree: int | None = None) -> np.ndarray:
        """The predictions of the polynomial of ``degree`` (default: the answer, ``degree_``) at ``X``."""
        k = self._degree(degree)
        X_np = np.asarray(X, dtype=np.float64)
        if X_np.ndim == 1:
            X_np = X_np.reshape(-1, 1)
        if X_np.shape[0] == 0:
            return np.empty(0)
        return self._design(X_np, n_monomials(self.n_variables_, k)) @ self._weights[k]

    def get_expression(self, degree: int | None = None, *, return_prefix: bool = False) -> str | list[str]:
        """The polynomial of ``degree`` (default: the answer) written out: the sum of its coefficients times their
        monomials in ``x1, x2, ...`` (by column position), every coefficient at full float64 precision (the shortest
        string that reads back as the same number), a negative one after the first term as a subtraction, a zero one
        left out. ``return_prefix`` gives the prefix tokens (``pow`` for a power) instead of the infix string."""
        k = self._degree(degree)
        terms = [(float(c), a) for c, a in zip(self.coefficients_[k], self.exponents_) if c != 0.0]
        if not terms:
            return ["0.0"] if return_prefix else "0.0"
        infix: list[str] = []
        operators: list[str] = []
        operands: list[list[str]] = []
        for i, (coefficient, exponents) in enumerate(terms):
            sign = "-" if i > 0 and coefficient < 0 else "+"
            written = repr(abs(coefficient) if sign == "-" else coefficient)
            factors = [(f"x{j + 1}", int(p)) for j, p in enumerate(exponents) if p > 0]
            infix.append((f" {sign} " if i > 0 else "") + "*".join(
                [written] + [name if p == 1 else f"{name}**{p}" for name, p in factors]))
            operand = ["*"] * len(factors) + [written]
            for name, p in factors:
                operand += [name] if p == 1 else ["pow", name, str(p)]
            operands.append(operand)
            if i > 0:
                operators.append(sign)
        if not return_prefix:
            return "".join(infix)
        prefix = operators[::-1]
        for operand in operands:
            prefix += operand
        return prefix


__all__ = ["PolynomialModel", "monomial_exponents", "n_monomials"]
