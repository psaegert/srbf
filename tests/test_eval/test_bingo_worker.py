"""The Bingo worker (srbf/worker/models/bingo_worker.py) without Bingo: a stand-in for the parts of bingo-nasa 0.5.7
it uses.

The stand-in mirrors the real attributes: ``SymbolicRegressor(**params)`` with ``fit(X, y)``,
``get_best_individual()``, ``get_pareto_front()`` and ``archipelago`` (``get_fitness_evaluation_count()``,
``generational_age``); the returned ``EquationRegressor`` with ``equation``, ``fitness``, ``complexity`` and
``__str__``; its ``AGraph`` with ``_simplified_command_array`` (rows ``node, param1, param2``), ``constants``,
``get_complexity()``, ``evaluate_equation_at(X)`` (an ``(n, 1)`` array) and ``fitness``. The stand-in evaluates a
command stack with Bingo's own semantics (bingo/symbolic_regression/agraph/evaluation_backend/operator_eval.py), so
that the written string is checked against what Bingo computes, not against the worker's own printer.
"""
import os

import numpy as np
import pytest

from srbf.worker.models import bingo_worker as w

# Bingo's forward evaluation per operator code (operator_eval.py; the C++ backend computes the same)
FORWARD = {
    2: np.add, 3: np.subtract, 4: np.multiply, 5: np.divide, 6: np.sin, 7: np.cos, 8: np.exp,
    9: lambda a: np.log(np.abs(a)), 10: np.power, 11: np.abs, 12: lambda a: np.sqrt(np.abs(a)),
    13: lambda a, b: np.power(np.abs(a), b), 14: np.sinh, 15: np.cosh, 16: np.tan, 17: np.arcsin, 18: np.arccos,
    19: np.arctan, 20: np.tanh, 21: lambda a: a ** 2, 22: lambda a: a ** 3,
}


def bingo_evaluate(commands, constants, X):
    values = []
    with np.errstate(all="ignore"):
        for node, p1, p2 in commands:
            if node == w.VARIABLE:
                values.append(X[:, p1])
            elif node == w.CONSTANT:
                values.append(np.full(X.shape[0], float(constants[p1])))
            elif node == w.INTEGER:
                values.append(np.full(X.shape[0], float(p1)))
            elif node in w.BINARY:
                values.append(FORWARD[node](values[p1], values[p2]))
            else:
                values.append(FORWARD[node](values[p1]))
    return values[-1].reshape(-1, 1)


class _AGraph:
    def __init__(self, commands, constants, fitness=1e-3):
        self._simplified_command_array = np.array(commands, dtype=int).reshape(-1, 3)
        self._constants = tuple(constants)
        self.fitness = fitness
        self.brought_up_to_date = False

    @property
    def constants(self):
        return self._constants

    def get_complexity(self):
        self.brought_up_to_date = True
        return self._simplified_command_array.shape[0]

    def evaluate_equation_at(self, X):
        return bingo_evaluate(self._simplified_command_array.tolist(), self._constants, X)


class _EquationRegressor:
    def __init__(self, equation):
        self.equation = equation

    @property
    def fitness(self):
        return self.equation.fitness

    @property
    def complexity(self):
        return self.equation.get_complexity()

    def __str__(self):
        return "bingo's console string"


class _Island:
    def __init__(self, evaluations, generations):
        self._evaluations, self.generational_age = evaluations, generations

    def get_fitness_evaluation_count(self):
        return self._evaluations


# 2.5 * log(|X_0|) / sqrt(|X_1|) + (-0.125), a constant as Bingo stores it, and a row the result does not use
BEST = ([[0, 0, 0], [0, 1, 1], [1, 0, 0], [9, 0, 0], [4, 2, 3], [12, 1, 1], [5, 4, 5], [1, 1, 1], [8, 0, 0],
         [2, 6, 7]], (2.5000000000000004, -0.125))
SIMPLE = ([[0, 0, 0]], ())


class _Regressor:
    instances: list = []

    def __init__(self, **params):
        self.params = params
        _Regressor.instances.append(self)

    def fit(self, X, y):
        self.omp_during_fit = os.environ.get("OMP_NUM_THREADS")
        self.archipelago = _Island(evaluations=41_234, generations=40)
        self.best = _EquationRegressor(_AGraph(*BEST, fitness=2e-17))
        self.front = [_EquationRegressor(_AGraph(*SIMPLE, fitness=0.5)), self.best]
        return self

    def get_best_individual(self):
        return self.best

    def get_pareto_front(self):
        return list(self.front)


@pytest.fixture
def fake_bingo(monkeypatch):
    _Regressor.instances = []
    monkeypatch.setattr(w, "_require_bingo", lambda: _Regressor)
    return _Regressor


def _data():
    rng = np.random.default_rng(0)
    X = rng.uniform(1, 2, size=(64, 2))
    return X, 2.5 * np.log(X[:, 0]) / np.sqrt(X[:, 1]) - 0.125


def _fit(state):
    X, y = _data()
    return w.fit(X.tolist(), y.tolist(), x_val=[], variables=["v1", "v2"], meta={}, options={}, state=state)


def test_the_author_configuration_the_operators_and_the_budget_reach_the_regressor(fake_bingo):
    _fit(w.load({"max_evals": 65536}))
    params = fake_bingo.instances[-1].params
    assert params["population_size"] == 500 and params["stack_size"] == 24 and params["use_simplification"] is True
    assert params["crossover_prob"] == 0.3 and params["mutation_prob"] == 0.45 and params["metric"] == "mse"
    assert params["clo_alg"] == "lm" and params["clo_threshold"] == 1e-5 and params["fitness_threshold"] == 1e-16
    assert params["evolutionary_algorithm"] == "AgeFitnessEA" and params["max_time"] == 3500
    assert params["generations"] >= 10 ** 18 and params["scale_max_evals"] is False
    assert params["max_evals"] == 65536
    assert set(params["operators"]) == {"+", "-", "*", "/", "sin", "cos", "sinh", "cosh", "exp", "log", "pow", "|",
                                        "sqrt"}


def test_bingo_evaluates_in_the_worker_s_own_process(fake_bingo, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    _fit(w.load({}))
    assert fake_bingo.instances[-1].omp_during_fit == "0"      # Bingo's process count: 0 = serial evaluation
    assert os.environ["OMP_NUM_THREADS"] == "1"
    monkeypatch.delenv("OMP_NUM_THREADS")
    _fit(w.load({}))
    assert fake_bingo.instances[-1].omp_during_fit == "0" and "OMP_NUM_THREADS" not in os.environ


def test_the_answer_is_bingo_s_equation_at_full_precision_in_srbf_s_names(fake_bingo):
    out = _fit(w.load({"max_evals": 50_000}))
    assert out["expression"] == "(((2.5000000000000004*log(abs(v1)))/sqrt(abs(v2))) + (-0.125))"
    assert "y_pred" not in out and out["fit_time"] >= 0
    extra = out["extra"]
    assert extra["string_deviation"] == 0.0
    assert extra["evaluations"] == 41_234 and extra["generations"] == 40 and extra["max_evals"] == 50_000
    assert extra["converged"] and extra["fitness"] == 2e-17 and extra["model_complexity"] == 10
    assert extra["bingo_string"] == "bingo's console string" and extra["fitness_predictor_island"] is False
    assert [m["expression"] for m in extra["front"]] == ["v1", out["expression"]]


def test_every_operator_is_written_as_what_bingo_computes():
    rng = np.random.default_rng(1)
    X = np.concatenate([rng.uniform(-3, 3, size=(200, 2)), [[0.0, 0.0], [-2.0, 0.5], [2.0, -1.0]]])
    for node in w.SPELLING:
        if node in w.BINARY:
            commands = [[0, 0, 0], [0, 1, 1], [node, 0, 1]]
        else:
            commands = [[0, 0, 0], [node, 0, 0]]
        if node in (17, 18):                                   # asin, acos: keep inside their domain
            commands = [[0, 0, 0], [1, 0, 0], [4, 0, 1], [node, 2, 2]]
        expression = w.infix(commands, (0.3,), ["a", "b"])
        reference = bingo_evaluate(commands, (0.3,), X)
        assert w.string_deviation(expression, ["a", "b"], X, reference) == 0.0, (node, expression)


def test_integers_negative_numbers_and_unused_rows():
    commands = [[0, 0, 0], [-1, -1, -1], [10, 0, 1], [1, 0, 0], [8, 0, 0], [4, 3, 2]]
    assert w.infix(commands, (-1.5e-07,), ["x"]) == "((-1.5e-07)*(x)**((-1)))"
    assert w.infix([[-1, 2, 2]], (), []) == "2"
    X = np.array([[-2.0], [0.5], [3.0]])
    ref = bingo_evaluate(commands, (-1.5e-07,), X)
    assert w.string_deviation(w.infix(commands, (-1.5e-07,), ["x"]), ["x"], X, ref) == 0.0


def test_a_power_of_two_numbers_follows_numpy_not_python():
    commands = [[-1, -2, -2], [1, 0, 0], [10, 0, 1], [0, 0, 0], [2, 2, 3]]   # (-2)**0.5 + x: nan, as in Bingo
    expression = w.infix(commands, (0.5,), ["x"])
    X = np.array([[1.0], [2.0]])
    assert np.all(np.isnan(w.evaluate(expression, ["x"], X)))
    assert w.string_deviation(expression, ["x"], X, bingo_evaluate(commands, (0.5,), X)) == 0.0


def test_a_wrong_spelling_shows_in_the_deviation():
    X = np.array([[-2.0], [1.0], [2.5]])
    reference = bingo_evaluate([[0, 0, 0], [9, 0, 0]], (), X)            # Bingo's log(|x|)
    assert w.string_deviation("log(x)", ["x"], X, reference) == float("inf")
    assert w.string_deviation("log(abs(x)) + 1e-3", ["x"], X, reference) == pytest.approx(1e-3)
    assert w.string_deviation("foo(x)", ["x"], X, reference) is None


def test_what_cannot_be_written_is_an_error():
    with pytest.raises(ValueError, match="non-finite constant"):
        w.infix([[1, 0, 0]], (float("nan"),), ["x"])
    with pytest.raises(ValueError, match="no spelling"):
        w.infix([[0, 0, 0], [99, 0, 0]], (), ["x"])
    with pytest.raises(ValueError, match="X_1"):
        w.infix([[0, 1, 1]], (), ["x"])


def test_a_problem_gets_one_seed_from_its_data_and_the_configured_seed(fake_bingo):
    X, y = _data()
    assert w.run_seed(X, y, 0) == w.run_seed(X.copy(), y.copy(), 0)
    assert w.run_seed(X, y, 0) != w.run_seed(X, y + 1e-9, 0)       # a fresh draw of the law: a different run
    assert w.run_seed(X, y, 0) != w.run_seed(X, y, 1)
    _fit(w.load({"seed": 1}))
    assert fake_bingo.instances[-1].params["random_state"] == w.run_seed(X, y, 1)


def test_a_side_experiment_replaces_settings_and_the_default_leaves_them(fake_bingo):
    authors = ["+", "-", "*", "/", "sin", "cos", "exp", "log", "sqrt"]
    _fit(w.load({"config": {"operators": authors, "max_time": 28800}}))
    params = fake_bingo.instances[-1].params
    assert params["operators"] == authors and params["max_time"] == 28800 and params["stack_size"] == 24
    _fit(w.load({}))
    assert "pow" in fake_bingo.instances[-1].params["operators"]
    with pytest.raises(ValueError, match="not max_evals"):
        w.load({"config": {"max_evals": 5}})
    with pytest.raises(ValueError, match="not parallel"):
        w.load({"config": {"parallel": True}})


def test_the_worker_matches_the_installed_bingo():
    """Run where bingo-nasa 0.5.7 is importable (its own environment): operator codes, arguments, author settings."""
    pytest.importorskip("bingo")
    import inspect

    from bingo.symbolic_regression.agraph import operator_definitions as od
    from bingo.symbolic_regression.srbench_interface import est
    from bingo.symbolic_regression.symbolic_regressor import SymbolicRegressor

    assert (od.INTEGER, od.VARIABLE, od.CONSTANT) == (w.INTEGER, w.VARIABLE, w.CONSTANT)
    names = {2: "ADDITION", 3: "SUBTRACTION", 4: "MULTIPLICATION", 5: "DIVISION", 6: "SIN", 7: "COS", 8: "EXPONENTIAL",
             9: "LOGARITHM", 10: "POWER", 11: "ABS", 12: "SQRT", 13: "SAFE_POWER", 14: "SINH", 15: "COSH", 16: "TAN",
             17: "ARCSIN", 18: "ARCCOS", 19: "ARCTAN", 20: "TANH", 21: "SQUARE", 22: "CUBE"}
    assert {code: getattr(od, name) for code, name in names.items()} == {code: code for code in names}
    assert {code for code, binary in od.IS_ARITY_2_MAP.items() if binary} == w.BINARY
    accepted = set(inspect.signature(SymbolicRegressor.__init__).parameters)
    assert set(w.AUTHOR_CONFIG) | set(w.SET_PER_FIT) <= accepted
    for operator in w.OPERATORS:
        assert any(operator in aliases for aliases in od.OPERATOR_NAMES.values()), operator
    shipped = est.get_params()
    for key, value in w.AUTHOR_CONFIG.items():
        if key == "evolutionary_algorithm":
            assert shipped[key].__name__ == value
        elif key == "max_time":
            assert shipped[key] == pytest.approx(350 * 0.97)          # the interface's own (shorter) limit
        elif key != "operators":
            assert shipped[key] == value, key
