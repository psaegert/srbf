"""The polynomial family: least-squares polynomials of total degree 0 to 10, every degree a candidate, the answer by
5-fold cross-validation with the one-standard-error rule. It needs NumPy only, not flash-ansr."""
import math

import numpy as np
import pytest
from simplipy import SimpliPyEngine

from srbf.baselines import PolynomialModel
from srbf.baselines.polynomial_model import monomial_exponents, n_monomials
from srbf.candidates import evaluate_prefix


@pytest.fixture(scope="module")
def engine() -> SimpliPyEngine:
    return SimpliPyEngine.load("acj-4-3", install=True)


def _points(seed: int, n: int, d: int, low: float, high: float) -> np.ndarray:
    return np.random.default_rng(seed).uniform(low, high, size=(n, d))


LAWS = [  # (variables, law, its degree)
    (1, lambda X: 1.5 * X[:, 0] ** 3 - 2.0 * X[:, 0] + 0.5, 3),
    (1, lambda X: 0.1 * X[:, 0] ** 5 - X[:, 0] ** 2, 5),
    (2, lambda X: X[:, 0] ** 2 * X[:, 1] - 3.0 * X[:, 1] + 1.0, 3),
    (3, lambda X: X[:, 0] * X[:, 1] + X[:, 2] ** 2 - 1.0, 2),
    (2, lambda X: 2.0 * X[:, 0] - 0.5 * X[:, 1] + 4.0, 1),
]
RANGES = [(-2.0, 2.0), (1.0, 5.0), (0.1, 10.0), (-100.0, 100.0), (100.0, 1000.0)]


# ---- the family ---------------------------------------------------------------------------------------------------
def test_the_monomials_of_degree_k_come_first_and_number_c_d_plus_k_choose_k():
    assert monomial_exponents(2, 2).tolist() == [[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2]]
    for d in range(1, 5):
        exponents = monomial_exponents(d, 6)
        for k in range(7):
            assert n_monomials(d, k) == math.comb(d + k, k)
            assert (exponents[:n_monomials(d, k)].sum(axis=1) <= k).all()
        assert len({tuple(row) for row in exponents}) == len(exponents) == n_monomials(d, 6)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("low,high", RANGES)
@pytest.mark.parametrize("d,law,degree", LAWS)
def test_a_polynomial_is_recovered_at_its_own_degree_on_noiseless_data(d, law, degree, low, high, seed):
    X = _points(seed, 512, d, low, high)
    model = PolynomialModel().fit(X, law(X))
    assert model.degrees_ == list(range(11))
    assert model.degree_ == degree
    np.testing.assert_allclose(model.predict(X), law(X), rtol=1e-9, atol=1e-9 * np.abs(law(X)).max())


def test_the_true_degree_s_coefficients_are_found():
    X = _points(0, 200, 2, -2.0, 2.0)
    model = PolynomialModel().fit(X, 3.0 - 0.25 * X[:, 0] + 1.5 * X[:, 0] ** 2 * X[:, 1])
    found = dict(zip(map(tuple, model.exponents_), model.coefficients_[3]))
    expected = {(0, 0): 3.0, (1, 0): -0.25, (2, 1): 1.5}
    for exponents, value in found.items():
        assert value == pytest.approx(expected.get(exponents, 0.0), abs=1e-12)


def test_a_degree_is_fitted_only_while_it_has_at_most_as_many_monomials_as_points():
    X = _points(0, 10, 2, -1.0, 1.0)
    model = PolynomialModel().fit(X, np.sin(X[:, 0]) + X[:, 1])
    assert model.degrees_ == [0, 1, 2, 3]             # 1, 3, 6, 10 monomials; degree 4 would need 15
    assert model.n_monomials_ == [1, 3, 6, 10]
    assert sorted(model.coefficients_) == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="not fitted"):
        model.get_expression(4)
    model = PolynomialModel().fit(_points(1, 6, 1, -1.0, 1.0), np.arange(6.0))
    assert model.degrees_ == [0, 1, 2, 3, 4, 5]       # one variable: degree k has k + 1 monomials
    model = PolynomialModel(max_degree=3).fit(_points(2, 512, 1, -1.0, 1.0), np.arange(512.0))
    assert model.degrees_ == [0, 1, 2, 3]


@pytest.mark.parametrize("d,n,degrees", [(40, 30, [0]), (20, 30, [0, 1]), (31, 512, [0, 1]), (10, 512, [0, 1, 2, 3])])
def test_many_variables_stop_early(d, n, degrees):
    X = _points(3, n, d, -1.0, 1.0)
    model = PolynomialModel().fit(X, X.sum(axis=1) ** 2)
    assert model.degrees_ == degrees
    assert model.degree_ in degrees


@pytest.mark.parametrize("value", [3.0, 0.1, -7.3, 1e6, 0.0])
@pytest.mark.parametrize("seed", range(3))
def test_a_constant_target_is_degree_zero(value, seed):
    X = _points(seed, 100, 2, -2.0, 2.0)
    model = PolynomialModel().fit(X, np.full(100, value))
    assert model.degree_ == 0
    assert model.coefficients_[0][0] == pytest.approx(value, rel=4 * np.finfo(float).eps, abs=0.0)


# ---- the answer ---------------------------------------------------------------------------------------------------
def test_the_answer_is_the_lowest_degree_within_one_standard_error_of_the_best():
    rng = np.random.default_rng(0)
    X = rng.uniform(-2.0, 2.0, (100, 1))
    y = X[:, 0] ** 3 - 2.0 * X[:, 0] + rng.normal(0.0, 1.0, 100)
    model = PolynomialModel().fit(X, y)
    best = int(np.argmin(model.cv_mse_))
    threshold = model.cv_mse_[best] + model.cv_se_[best]
    assert model.degree_ == min(k for k, e in zip(model.degrees_, model.cv_mse_) if e <= threshold)
    assert model.degree_ == 3 < model.degrees_[best]  # here the best CV error is a higher degree's
    assert (model.cv_se_ > 0).all()


def test_the_one_standard_error_rule_on_given_errors():
    pick = PolynomialModel._one_standard_error_pick
    degrees = [0, 1, 2, 3]
    assert pick(degrees, np.array([9.0, 2.0, 1.5, 1.0]), np.array([1.0, 0.5, 0.5, 0.6])) == 2
    assert pick(degrees, np.array([9.0, 2.0, 1.5, 1.0]), np.array([1.0, 0.5, 0.5, 1.0])) == 1
    assert pick(degrees, np.array([9.0, np.inf, 1.5, 1.0]), np.array([1.0, np.nan, 0.5, 0.1])) == 3
    assert pick(degrees, np.full(4, np.nan), np.full(4, np.nan)) == 0


def test_the_folds_are_seeded_and_a_single_point_is_degree_zero():
    X = _points(4, 60, 1, -1.0, 1.0)
    y = np.cos(3 * X[:, 0])
    a, b = PolynomialModel().fit(X, y), PolynomialModel().fit(X, y)
    np.testing.assert_array_equal(a.cv_mse_, b.cv_mse_)
    assert not np.array_equal(a.cv_mse_, PolynomialModel(seed=1).fit(X, y).cv_mse_)
    one = PolynomialModel().fit([[0.5]], [2.0])
    assert one.degrees_ == [0] and one.degree_ == 0 and np.isnan(one.cv_mse_).all()
    assert one.get_expression() == "2.0"


def test_bad_input_is_refused():
    with pytest.raises(ValueError, match="finite"):
        PolynomialModel().fit([[0.0], [np.nan]], [1.0, 2.0])
    with pytest.raises(ValueError, match="same points"):
        PolynomialModel().fit([[0.0], [1.0]], [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="n_folds"):
        PolynomialModel(n_folds=1).fit([[0.0], [1.0]], [1.0, 2.0])
    with pytest.raises(ValueError, match="fitted yet"):
        PolynomialModel().predict([[0.0]])


# ---- the candidates -----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("low,high", [(-2.0, 2.0), (0.1, 10.0), (1000.0, 3000.0)])
@pytest.mark.parametrize("d", [1, 2, 3])
def test_every_candidate_written_out_computes_the_fitted_predictions(engine, d, low, high):
    X = _points(5, 400, d, low, high)
    X_val = _points(6, 100, d, low, high)
    y = np.exp(np.sin(X).sum(axis=1)) * (1.0 + 0.3 * np.random.default_rng(7).standard_t(2, size=len(X)))
    model = PolynomialModel().fit(X, y)
    names = [f"x{j + 1}" for j in range(d)]
    eps = np.finfo(np.float64).eps
    for k in model.degrees_:
        prefix = model.get_expression(k, return_prefix=True)
        written, written_val = evaluate_prefix(engine, prefix, names, X, X_val)
        for points, by_string in ((X, written[:, 0]), (X_val, written_val[:, 0])):
            # The string is the fitted polynomial, coefficient for coefficient; two evaluations of one polynomial (powers
            # and sums in another order) differ by the rounding of its terms, at most about eps times the sum of the
            # terms' magnitudes for every term and power multiplied in: large where the terms cancel (far from zero).
            m = n_monomials(d, k)
            powers = np.abs(points)[:, None, :] ** model.exponents_[None, :m]
            terms = np.abs(model.coefficients_[k]) * np.prod(powers, axis=2)
            bound = 2 * (m + k + 2) * eps * terms.sum(axis=1)
            assert (np.abs(by_string - model.predict(points, degree=k)) <= bound).all()
        # the infix reads back as the same tree
        assert list(engine.read_infix(model.get_expression(k))) == prefix
    assert all(np.isfinite(c).all() for c in model.coefficients_.values())


def test_coefficients_are_written_at_full_precision():
    X = _points(8, 50, 1, 0.0, 1.0)
    model = PolynomialModel().fit(X, 1.0 / 3.0 + np.pi * X[:, 0])
    prefix = model.get_expression(1, return_prefix=True)
    assert prefix[0] == "+"
    c0, c1 = model.coefficients_[1]
    assert float(prefix[1]) == c0 and float(prefix[3]) == c1      # the same float64, bit for bit
    assert prefix[1] == repr(float(c0)) and len(prefix[1]) > 15   # the shortest round-trip spelling
    assert model.get_expression(1) == f"{float(c0)!r} + {float(c1)!r}*x1"


def test_signs_powers_and_zero_terms_are_written_as_the_engine_reads_them(engine):
    model = PolynomialModel()
    model.n_variables_ = 2
    model.exponents_ = monomial_exponents(2, 2)
    model.degrees_ = [0, 1, 2]
    model.degree_ = 2
    model.coefficients_ = {2: np.array([-0.5, 2.0, 0.0, 0.0, -1.25, 3e-20])}
    assert model.get_expression() == "-0.5 + 2.0*x1 - 1.25*x1*x2 + 3e-20*x2**2"
    assert model.get_expression(return_prefix=True) == [
        "+", "-", "+", "-0.5", "*", "2.0", "x1", "*", "*", "1.25", "x1", "x2", "*", "3e-20", "pow", "x2", "2"]
    assert list(engine.read_infix(model.get_expression())) == model.get_expression(return_prefix=True)
    model.coefficients_ = {2: np.zeros(6)}
    assert model.get_expression() == "0.0" and model.get_expression(return_prefix=True) == ["0.0"]


def test_the_fit_reports_its_timing():
    model = PolynomialModel().fit(_points(9, 200, 2, -1.0, 1.0), np.arange(200.0))
    assert set(model.timing_) == {"fit", "cross_validation"} and all(t >= 0 for t in model.timing_.values())
