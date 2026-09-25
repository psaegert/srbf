"""The DSO worker (srbf/worker/models/dso_worker.py) without DSO: a stand-in for the parts of DSO v3.0.0 it uses.

The stand-in mirrors the real attributes: ``DeepSymbolicOptimizer(config)`` with ``setup()``, ``train_one_step()``,
``trainer`` (``done``, ``nevals``, ``iteration``, ``p_r_best``), ``gp_controller.algorithm.logbook`` (records with
``nevals`` and ``uncached_size``) and ``sess``; ``Program.cache`` and ``Program.clear_cache()``; programs with
``traversal``, ``r``, ``complexity``, ``invalid``, ``evaluate`` and ``execute(X)``; tokens with ``name``, ``arity``,
``input_var`` and, for constants, ``value`` (``poly``: ``exponents`` and ``coef``); ``gp.utils.Individual`` and
``task.regression.polyfit.inverse_function_map``.
"""
import copy
import types

import numpy as np
import pytest

from srbf.worker.models import dso_worker as w


class Token:
    def __init__(self, name, arity=0, input_var=None, value=None):
        self.name, self.arity, self.input_var = name, arity, input_var
        self.value = None if value is None else np.atleast_1d(value)


class Poly(Token):
    def __init__(self, exponents, coef):
        super().__init__("poly")
        self.exponents, self.coef = exponents, np.asarray(coef, dtype=np.float64)


def var(i):
    return Token("x%d" % (i + 1), input_var=i)


def op(name):
    return Token(name, arity=2 if name in ("add", "sub", "mul", "div") else 1)


# DSO's own functions (dso/functions.py), for an evaluation independent of the worker's printer
FUNCTIONS = {"add": np.add, "sub": np.subtract, "mul": np.multiply, "div": np.divide, "neg": np.negative,
             "abs": np.abs, "inv": np.reciprocal, "sin": np.sin, "cos": np.cos, "tan": np.tan, "tanh": np.tanh,
             "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "n2": np.square, "n3": lambda a: np.power(a, 3),
             "n4": lambda a: np.power(a, 4)}


def execute(traversal, X):
    stack = []
    for t in reversed(traversal):
        if t.input_var is not None:
            stack.append(X[:, t.input_var])
        elif t.name == "poly":
            monomials = np.stack([np.prod(X ** np.asarray(e), axis=1) for e in t.exponents], axis=1)
            stack.append(monomials @ t.coef)
        elif t.arity == 0:
            stack.append(np.full(X.shape[0], float(t.value[0])))
        else:
            args = [stack.pop() for _ in range(t.arity)]
            stack.append(FUNCTIONS[t.name](*args))
    return stack[0]


class Program:
    cache: dict = {}

    def __init__(self, traversal, r, complexity, invalid=False, success=False):
        self.traversal, self.r, self.complexity, self.invalid = traversal, r, complexity, invalid
        self.evaluate = {"nmse_test": 1e-3, "success": success}

    def execute(self, X):
        return execute(self.traversal, X)

    @classmethod
    def clear_cache(cls):
        cls.cache = {}


class Session:
    closed = False

    def close(self):
        self.closed = True


class Optimizer:
    """DeepSymbolicOptimizer: a run of iterations, each adding its expressions to Program.cache."""
    instances: list = []
    programs: list = []          # what the iterations find, in order
    success_at = None            # the iteration whose program fits exactly (early stop)

    def __init__(self, config):
        self.config = config
        Optimizer.instances.append(self)

    def setup(self):
        Program.clear_cache()
        self.sess = Session()
        gp = self.config["gp_meld"].get("run_gp_meld")
        self.gp_controller = types.SimpleNamespace(algorithm=types.SimpleNamespace(logbook=[])) if gp else None
        self.trainer = types.SimpleNamespace(done=False, nevals=0, iteration=0, p_r_best=None, r_best=-np.inf)
        self.batch = self.config["training"]["batch_size"]

    def train_one_step(self):
        trainer = self.trainer
        p = Optimizer.programs[trainer.iteration % len(Optimizer.programs)]
        p.evaluate["success"] = trainer.iteration == Optimizer.success_at
        Program.cache[id(p)] = p
        trainer.nevals += self.batch
        if self.gp_controller is not None:
            for _ in range(25):
                self.gp_controller.algorithm.logbook.append({"nevals": 500, "uncached_size": 180})
            trainer.nevals += 25 * 500
        if p.r > trainer.r_best:
            trainer.r_best, trainer.p_r_best = p.r, p
        if p.evaluate["success"] or trainer.nevals >= self.config["training"]["n_samples"]:
            trainer.done = True
        trainer.iteration += 1


def _individual_class():
    class Individual:
        """dso.gp.utils.Individual in v3.0.0: a DEAP tree whose tokenized_repr is the array it was created from,
        and whose deepcopy rebuilds the tree from that array and deep-copies every attribute, the shared ``pset``
        included."""

        def __init__(self, names, created_from, pset):
            self.nodes = [types.SimpleNamespace(name=n) for n in names]
            self.work_repr = np.asarray(created_from, dtype=np.int32)
            self.pset = pset

        def __iter__(self):
            return iter(self.nodes)

        @property
        def tokenized_repr(self):
            return self.work_repr.copy()

        def __deepcopy__(self, memo):
            new = type(self)(list(self.tokenized_repr), self.work_repr, self.pset)
            for key, value in self.__dict__.items():
                if key != "nodes":
                    setattr(new, key, copy.deepcopy(value, memo))
            return new

    return Individual


def _modules():
    return {"dso": types.SimpleNamespace(__file__="/nonexistent/dso/dso/__init__.py"),
            "DeepSymbolicOptimizer": Optimizer, "Program": Program,
            "gp_utils": types.SimpleNamespace(Individual=_individual_class()),
            "polyfit": types.SimpleNamespace(inverse_function_map={"exp": np.log, "n2": np.sqrt})}


@pytest.fixture
def fake_dso(monkeypatch):
    Optimizer.instances, Optimizer.programs, Optimizer.success_at = [], [P1, P2], None
    Program.clear_cache()
    modules = _modules()
    monkeypatch.setattr(w, "_require_dso", lambda: modules)
    return modules


def _data(n=64, d=2):
    rng = np.random.default_rng(0)
    X = rng.uniform(1, 2, size=(n, d))
    return X, X[:, 0] * X[:, 1]


# a program per iteration, improving: const * x1, then sin(x2) + poly(x1, x2)
P1 = Program([op("mul"), Token("const", value=0.12345678900000578), var(0)], r=0.5, complexity=3)
P2 = Program([op("add"), op("sin"), var(1), Poly([(0, 0), (1, 1), (2, 0)], [-1.5, 0.30000000000000004, 2.0])],
             r=0.9, complexity=8)


def test_the_arms_run_the_published_configurations(fake_dso):
    X, y = _data()
    dsr = w.build_config("dsr", X, y, n_samples=1000, seed=3)
    assert dsr["training"]["batch_size"] == 1000 and dsr["training"]["epsilon"] == 0.05
    assert dsr["policy_optimizer"] == {"policy_optimizer_type": "pg", "learning_rate": 0.0005, "optimizer": "adam",
                                       "entropy_weight": 0.03, "entropy_gamma": 0.7}
    assert dsr["gp_meld"] == {"run_gp_meld": False}
    assert dsr["task"]["function_set"] == w.OPERATORS + ["const"]
    udsr = w.build_config("udsr", X, y, n_samples=13000, seed=3)
    po = udsr["policy_optimizer"]
    assert po["policy_optimizer_type"] == "pqt" and po["learning_rate"] == 0.0025 and po["pqt_k"] == 10
    assert po["entropy_weight"] == 0.03 and po["entropy_gamma"] == 0.7
    assert udsr["training"]["batch_size"] == 500 and udsr["training"]["epsilon"] == 0.02
    gp = udsr["gp_meld"]
    assert gp["run_gp_meld"] and gp["population_size"] == 500 and gp["generations"] == 25 and not gp["parallel_eval"]
    poly = udsr["task"]["poly_optimizer_params"]
    assert poly["degree"] == 3 and poly["regressor_params"]["n_max_terms"] == 10
    assert udsr["task"]["function_set"] == w.OPERATORS + ["1.0", "const", "poly"]
    for config in (dsr, udsr):
        assert config["task"]["protected"] is False and config["task"]["threshold"] == 1e-12
        assert config["training"]["n_cores_batch"] == 1 and config["experiment"] == {"logdir": None, "seed": 3}
        assert config["prior"]["length"] == {"min_": 4, "max_": 64, "on": True}
        assert config["prior"]["soft_length"]["on"] and not config["prior"]["domain_range"]["on"]
        assert isinstance(config["task"]["dataset"], tuple)   # a tuple: DSO's seed shift is fixed
    assert "controller" not in udsr                           # a key DSO v3.0.0 ignores
    with pytest.raises(ValueError, match="arm"):
        w.build_config("gp", X, y, n_samples=1, seed=0)


def test_the_operators_are_the_benchmark_ones_dso_has():
    assert set(w.OPERATORS) == {"add", "sub", "mul", "div", "neg", "abs", "inv", "sin", "cos", "tan", "tanh", "exp",
                                "log", "sqrt", "n2", "n3", "n4"}
    X = np.random.default_rng(1).uniform(0.5, 1.5, size=(16, 1))
    for name in w.OPERATORS:           # each prints in the benchmark's notation and means what DSO computes
        arity = 2 if name in ("add", "sub", "mul", "div") else 1
        traversal = [op(name)] + [var(0)] * arity
        expression = w.infix(traversal, ["v1"])
        assert w.string_deviation(expression, ["v1"], X, execute(traversal, X)) < 1e-15, name


def test_the_printer_writes_every_constant_at_full_precision_in_the_variable_names():
    names = ["a%d" % i for i in range(11)]
    poly = Poly([(0,) * 11, (1,) + (0,) * 9 + (2,)], [-0.12345678900000578, 3.0000000000000004])
    traversal = [op("div"), op("neg"), Token("const", value=-2.5e-07), op("add"), Token("1.0", value=np.float32(1.0)),
                 op("sub"), poly, op("n3"), var(10)]
    expression = w.infix(traversal, names)
    assert expression == ("(neg((-2.5e-07)) / (1.0 + (((-0.12345678900000578) + 3.0000000000000004 * a0 * a10**2) "
                          "- (a10)**3)))")
    X = np.random.default_rng(2).uniform(1, 2, size=(32, 11))
    assert w.string_deviation(expression, names, X, execute(traversal, X)) < 1e-15
    assert w.token_sequence(traversal, names).startswith("div,neg,(-2.5e-07),add,1.0,sub,((-0.12345678900000578)")
    assert w.infix([Poly([(0,)], [])], ["v1"]) == "0.0"            # a LINEAR fitted to all zeros
    with pytest.raises(ValueError, match="no spelling"):
        w.infix([op("max"), var(0)], ["v1"])


def test_a_fit_runs_to_the_budget_and_returns_the_best_expression_with_its_record(fake_dso):
    Optimizer.programs = [P1, P2]
    X, y = _data()
    state = w.load({"arm": "udsr", "n_samples": 26000, "seed": 1})
    out = w.fit(X, y, x_val=[], variables=["v1", "v2"], meta={}, options={}, state=state)
    assert out["expression"] == "(sin(v2) + ((-1.5) + 0.30000000000000004 * v1 * v2 + 2.0 * v1**2))"
    assert "y_pred" not in out and out["fit_time"] >= 0      # srbf evaluates the string
    extra = out["extra"]
    assert extra["nevals"] == 26000 and extra["iterations"] == 2 and extra["n_samples"] == 26000
    assert extra["reward"] == 0.9 and extra["arm"] == "udsr" and extra["seed"] == w.run_seed(X, y, 1)
    assert not extra["early_stop"] and not extra["guard_hit"] and extra["string_deviation"] < 1e-15
    assert extra["gp_offspring"] == 2 * 25 * 500 and extra["gp_new_expressions"] == 2 * 25 * 180
    assert extra["linear_underdetermined"] is False and extra["unique_expressions"] == 2
    assert extra["model_complexity"] == 8.0 and extra["model_length"] == 4 and "complexity" not in extra
    assert [e["complexity"] for e in extra["front"]] == [3.0, 8.0]
    assert extra["front"][0]["expression"] == "(0.12345678900000578 * v1)"
    assert Program.cache == {} and Optimizer.instances[-1].sess.closed   # nothing left behind for the next fit


def test_an_exact_fit_stops_early_and_the_guard_returns_the_best_so_far(fake_dso):
    Optimizer.programs = [P1, P2]
    X, y = _data()
    Optimizer.success_at = 0
    out = w.fit(X, y, x_val=[], variables=["v1", "v2"], meta={}, options={},
                state=w.load({"arm": "dsr", "n_samples": 10**6}))
    assert out["extra"]["early_stop"] and out["extra"]["iterations"] == 1 and "gp_offspring" not in out["extra"]
    assert "linear_underdetermined" not in out["extra"]
    Optimizer.success_at = None
    out = w.fit(X, y, x_val=[], variables=["v1", "v2"], meta={}, options={},
                state=w.load({"arm": "dsr", "n_samples": 10**6, "max_seconds": 0}))
    assert out["extra"]["guard_hit"] and out["extra"]["iterations"] == 1       # one iteration always runs
    assert out["expression"] == "(0.12345678900000578 * v1)"


def test_linear_is_flagged_when_its_basis_outgrows_the_data(fake_dso):
    Optimizer.programs = [P1]
    X, y = _data(n=16, d=2)                       # C(2 + 3, 3) = 10 monomials <= 16 rows
    state = w.load({"arm": "udsr", "n_samples": 1})
    assert not w.fit(X, y, x_val=[], variables=["v1", "v2"], meta={}, options={}, state=state)["extra"]["linear_underdetermined"]
    X, y = _data(n=16, d=3)                       # C(3 + 3, 3) = 20 > 16
    extra = w.fit(X, y, x_val=[], variables=["v1", "v2", "v3"], meta={}, options={}, state=state)["extra"]
    assert extra["linear_underdetermined"]


def test_the_patches_restore_gp_meld_and_complete_linears_inverse_table(fake_dso):
    pset = {"primitives": list(range(50))}
    individual = fake_dso["gp_utils"].Individual(names=[3, 0, 1], created_from=[3, 0, 0, 7], pset=pset)  # varied
    assert list(individual.tokenized_repr) == [3, 0, 0, 7]
    assert list(copy.deepcopy(individual)) and copy.deepcopy(individual).pset is not pset
    state = w.load({"arm": "udsr", "warmup": False})
    assert state["patches"] == list(w.PATCHES) and not Optimizer.instances
    assert list(individual.tokenized_repr) == [3, 0, 1] and individual.tokenized_repr.dtype == np.int32
    clone = copy.deepcopy(individual)
    assert clone.pset is pset and [n.name for n in clone] == [3, 0, 1] and clone is not individual
    w.load({"arm": "udsr", "warmup": False})                          # a second load does not wrap twice
    assert copy.deepcopy(individual).pset is pset
    inverses = fake_dso["polyfit"].inverse_function_map
    assert inverses["neg"](np.array([2.0])) == -2.0 and inverses["n4"](np.array([16.0])) == 2.0
    assert inverses["n2"] is np.sqrt                                    # DSO's own entries stay
    info = w.info(state)
    assert info["worker"] == "dso" and info["arm"] == "udsr" and info["patches"] == list(w.PATCHES)
    assert "dso_commit" not in info                                     # not an editable checkout


def test_a_problem_gets_one_seed_from_its_data_and_the_configured_seed(fake_dso):
    Optimizer.programs = [P1]
    X, y = _data()
    assert w.run_seed(X, y, 0) == w.run_seed(X.copy(), y.copy(), 0)
    assert w.run_seed(X, y, 0) != w.run_seed(X, y + 1e-9, 0)       # a fresh draw of the law: a different run
    assert w.run_seed(X, y, 0) != w.run_seed(X, y, 1)
    w.fit(X, y, x_val=[], variables=["v1", "v2"], meta={}, options={}, state=w.load({"arm": "dsr", "seed": 1}))
    assert Optimizer.instances[-1].config["experiment"]["seed"] == w.run_seed(X, y, 1)


def test_a_side_experiment_replaces_settings_and_the_default_leaves_them(fake_dso):
    Optimizer.programs = [P1]
    X, y = _data()
    nggp = {"task": {"function_set": ["add", "sub", "mul", "div", "sin", "cos", "exp", "log"]},
            "prior": {"length": {"max_": 30}, "soft_length": {"on": False}},
            "policy_optimizer": {"entropy_weight": 0.005, "entropy_gamma": 1.0}}
    w.fit(X, y, x_val=[], variables=["v1", "v2"], meta={}, options={}, state=w.load({"arm": "udsr", "config": nggp}))
    config = Optimizer.instances[-1].config
    assert config["task"]["function_set"] == nggp["task"]["function_set"]
    assert config["prior"]["length"] == {"min_": 4, "max_": 30, "on": True} and not config["prior"]["soft_length"]["on"]
    assert config["policy_optimizer"]["entropy_weight"] == 0.005 and config["policy_optimizer"]["pqt_k"] == 10
    w.fit(X, y, x_val=[], variables=["v1", "v2"], meta={}, options={}, state=w.load({"arm": "udsr"}))
    assert Optimizer.instances[-1].config["policy_optimizer"]["entropy_weight"] == 0.03
    assert w.ARM_CONFIGS["udsr"]["prior"]["length"]["max_"] == 64                 # the defaults are not mutated


def test_the_arm_is_required(fake_dso):
    with pytest.raises(ValueError, match="arm"):
        w.load({})


def test_load_warms_up_with_a_small_run_of_the_arm(fake_dso):
    w.load({"arm": "udsr", "n_samples": 10**6})
    (warmup,) = Optimizer.instances
    assert warmup.config["training"]["n_samples"] == 1 and warmup.config["training"]["batch_size"] == 20
    assert warmup.config["gp_meld"]["population_size"] == 20 and warmup.config["policy_optimizer"]["pqt_k"] == 10


def test_the_front_keeps_the_non_dominated_valid_expressions():
    progs = [Program([var(0)], r=0.2, complexity=1), Program([var(1)], r=0.1, complexity=1),
             Program([op("sin"), var(0)], r=0.6, complexity=4), Program([op("exp"), var(0)], r=0.5, complexity=4),
             Program([op("log"), var(0)], r=0.95, complexity=5, invalid=True),
             Program([op("add"), var(0), var(1)], r=0.9, complexity=3)]
    front = w.pareto_front(progs, ["v1", "v2"])
    assert [(e["expression"], e["reward"]) for e in front] == [("v1", 0.2), ("(v1 + v2)", 0.9)]
    assert list(w.pareto_mask(np.array([[1, 0], [0, 1], [1, 1], [0, 1]]))) == [True, True, False, True]
