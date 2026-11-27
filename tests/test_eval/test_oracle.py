"""The oracle (``srbf.oracle``, ``model_adapter.type: oracle``): each problem's ground truth with its constants refitted
by flash-ansr's candidate path, without a checkpoint, a tokenizer or a vocabulary."""
import numpy as np
import pytest

pytest.importorskip("flash_ansr", reason="the oracle fits with flash-ansr's refiner (pip install 'srbf[flash-ansr]')")
from flash_ansr.scoring import resolve_ranking  # noqa: E402
from simplipy import SimpliPyEngine  # noqa: E402

from srbf import config as run_config  # noqa: E402
from srbf.core import EvaluationSample  # noqa: E402
from srbf.metrics.numeric import is_perfect_fit  # noqa: E402
from srbf.model_adapters import FlashANSRAdapter, OracleAdapter  # noqa: E402
from srbf.oracle import first_recovery, oracle_candidate, refine_config, refit_formula  # noqa: E402
from srbf.warmup import warmup_sample  # noqa: E402

ENGINE = "acj-5-4-llm"
#: 2.5 * x1^2 + sin(x2): one fittable constant, one structural exponent
LAW = ["+", "*", "2.5", "pow", "x1", "2", "sin", "x2"]
#: 0.7 * x18 + 1.3 / (x19 * x20) over 20 columns: variables beyond any Flash-ANSR checkpoint's vocabulary
WIDE_LAW = ["+", "*", "0.7", "x18", "/", "1.3", "*", "x19", "x20"]


@pytest.fixture(scope="module")
def engine() -> SimpliPyEngine:
    return SimpliPyEngine.load(ENGINE, install=True)


def _law(X: np.ndarray) -> np.ndarray:
    return 2.5 * X[:, 0] ** 2 + np.sin(X[:, 1])


def _wide_law(X: np.ndarray) -> np.ndarray:
    return 0.7 * X[:, 17] + 1.3 / (X[:, 18] * X[:, 19])


def _sample(expression, f=_law, n_columns=2, n=96, seed=0) -> EvaluationSample:
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.5, 3.0, size=(2 * n, n_columns))
    y = f(X)
    names = [f"x{i + 1}" for i in range(n_columns)]
    return EvaluationSample(x_support=X[:n], y_support=y[:n], x_validation=X[n:], y_validation=y[n:],
                            metadata={"expression": expression, "variable_names": names, "eval_row_index": 0})


def test_the_candidate_frees_each_fittable_literal_and_keeps_the_exponent(engine):
    assert oracle_candidate(LAW, engine) == ["+", "*", "<constant>", "pow", "x1", "2", "sin", "x2"]
    assert oracle_candidate(WIDE_LAW, engine) == ["+", "*", "<constant>", "x18", "/", "<constant>", "*", "x19", "x20"]


def test_a_formula_the_engine_cannot_read_has_no_candidate(engine):
    sample = _sample(["+", "x1"])
    assert oracle_candidate(["+", "x1"], engine) is None
    fit = refit_formula(["+", "x1"], sample.x_support, sample.y_support, engine=engine)
    assert fit.best is None and fit.candidates == [] and "cannot read" in fit.reason


def test_the_oracle_recovers_the_law_and_its_answer_is_its_prediction(engine):
    sample = _sample(LAW)
    fit = refit_formula(LAW, sample.x_support, sample.y_support, engine=engine, refine={"n_restarts": 8}, seed=0)
    best = fit.best
    assert best is not None and fit.reason is None
    assert np.allclose(best.constants, [2.5])
    assert is_perfect_fit(sample.y_validation, best.predict(sample.x_validation))
    # the answer's prefix (constants at full precision) evaluates to what the answer predicts
    from flash_ansr.inference import evaluate_prefix
    assert np.allclose(np.asarray(evaluate_prefix(engine, best.prefix, sample.x_validation)).reshape(-1),
                       best.predict(sample.x_validation), rtol=1e-12, atol=0)
    # best first, under the ranking: every other candidate scores no better
    assert all(best.score <= c.score for c in fit.candidates)
    assert best.mdl is not None and best.mdl > 0


def test_a_seed_fixes_every_starting_value(engine):
    sample = _sample(["+", "*", "1.7", "sin", "*", "0.9", "x1", "*", "0.3", "x2"],
                     f=lambda X: 1.7 * np.sin(0.9 * X[:, 0]) + 0.3 * X[:, 1])
    fits = [refit_formula(sample.metadata["expression"], sample.x_support, sample.y_support, engine=engine,
                          refine={"n_restarts": 2}, seed=7) for _ in range(2)]
    assert [c.prefix for c in fits[0].candidates] == [c.prefix for c in fits[1].candidates]
    assert fits[0].best.constants == fits[1].best.constants


def test_twenty_variables_need_no_vocabulary(engine):
    sample = _sample(WIDE_LAW, f=_wide_law, n_columns=20)
    fit = refit_formula(WIDE_LAW, sample.x_support, sample.y_support, engine=engine, refine={"n_restarts": 8}, seed=0,
                        variable_names=sample.metadata["variable_names"])
    best = fit.best
    assert best is not None
    assert {"x18", "x19", "x20"} <= set(best.prefix) and "x20" in best.infix
    assert is_perfect_fit(sample.y_validation, best.predict(sample.x_validation))


def test_the_first_recovery_stops_at_the_first_restart_whose_answer_reproduces_the_held_out_points(engine):
    """A frequency far from the starting values: most restarts end in another minimum. The early stop is the oracle at
    one restart per seed (seed, r), so it stops exactly where the oracle at that seed first recovers."""
    law = ["*", "1.3", "sin", "*", "7.3", "x1"]
    sample = _sample(law, f=lambda X: 1.3 * np.sin(7.3 * X[:, 0]), n_columns=1)
    refine = {"n_restarts": 64, "constant_ladder": False}
    recovery = first_recovery(law, sample.x_support, sample.y_support, sample.x_validation, sample.y_validation,
                              engine=engine, refine=refine, seed=4)
    assert recovery.recovered and recovery.answered
    assert is_perfect_fit(sample.y_validation, recovery.fit.best.predict(sample.x_validation))
    first = next(r for r in range(64) if is_perfect_fit(sample.y_validation, refit_formula(
        law, sample.x_support, sample.y_support, engine=engine, refine={**refine, "n_restarts": 1},
        seed=(4, r)).best.predict(sample.x_validation)))
    assert recovery.restarts == first + 1 > 1
    again = refit_formula(law, sample.x_support, sample.y_support, engine=engine, refine={**refine, "n_restarts": 1},
                          seed=(4, first))
    assert again.best.prefix == recovery.fit.best.prefix


def test_the_first_recovery_counts_what_the_constant_ladder_recovers(engine):
    """A product of constants of very different sizes: the plain fit from N(0, 5^2) starting values cannot move the tiny
    one, the constant ladder re-spells one constant and refits the rest, and the answer recovers on the first restart."""
    law = ["/", "x1", "*", "*", "*", "4", "3.1415926535897", "8.854e-12", "x2"]
    sample = _sample(law, f=lambda X: X[:, 0] / (4 * 3.1415926535897 * 8.854e-12 * X[:, 1]), n_columns=2)
    recovery = first_recovery(law, sample.x_support, sample.y_support, sample.x_validation, sample.y_validation,
                              engine=engine, refine={"n_restarts": 8}, seed=0)
    assert recovery.recovered and is_perfect_fit(sample.y_validation, recovery.fit.best.predict(sample.x_validation))
    assert recovery.fit.best.spelling is not None                  # the answer is the ladder's


def test_a_formula_that_does_not_fit_the_data_is_not_recovered(engine):
    sample = _sample(LAW)            # the data of LAW, the formula of another law
    recovery = first_recovery(["*", "2.0", "exp", "x1"], sample.x_support, sample.y_support, sample.x_validation,
                              sample.y_validation, engine=engine, refine={"n_restarts": 4}, seed=0)
    assert not recovery.recovered and recovery.fit is None and recovery.restarts == 4 and recovery.answered
    unreadable = first_recovery(["+", "x1"], sample.x_support, sample.y_support, sample.x_validation,
                                sample.y_validation, engine=engine)
    assert not unreadable.recovered and unreadable.restarts == 0 and not unreadable.answered


def test_the_adapter_answers_from_the_ground_truth_and_hands_over_its_candidates(engine):
    adapter = OracleAdapter(engine, refine=refine_config({"n_restarts": 4}), ranking=resolve_ranking("mdl"), seed=0)
    sample = _sample(WIDE_LAW, f=_wide_law, n_columns=20)
    record = adapter.evaluate_sample(sample).to_mapping()
    assert record["prediction_success"] is True and record["fit_time"] > 0
    assert record["ranking"] == {"mode": "mdl", "mdl_strength": None}
    assert record["candidate_notation"] == "prefix" and record["candidate_prediction"] == 0
    assert record["candidate_variables"] == [f"x{i + 1}" for i in range(20)]
    assert record["candidate_expressions"][0].split() == record["predicted_expression_prefix"]
    assert is_perfect_fit(sample.y_validation, record["y_pred_val"])
    assert is_perfect_fit(sample.y_support, record["y_pred"])
    # the same seed, the same answer
    again = adapter.evaluate_sample(sample).to_mapping()
    assert again["predicted_expression_prefix"] == record["predicted_expression_prefix"]


def test_the_adapter_needs_a_ground_truth(engine):
    adapter = OracleAdapter(engine, refine=refine_config(None), ranking=resolve_ranking("mdl"))
    record = adapter.evaluate_sample(_sample(None)).to_mapping()
    assert record["prediction_success"] is False and "ground truth" in record["error"]


def test_the_adapter_answers_the_warm_up_problem(engine):
    adapter = OracleAdapter(engine, refine=refine_config({"n_restarts": 4}), ranking=resolve_ranking("mdl"), seed=0)
    sample = warmup_sample()
    record = adapter.evaluate_sample(sample).to_mapping()
    assert record["prediction_success"] is True
    assert is_perfect_fit(sample.y_validation, record["y_pred_val"])


def test_the_config_builds_the_oracle_from_an_engine_alone():
    adapter = run_config.build_model_adapter({
        "type": "oracle", "config_provenance": "author_blessed", "simplipy_engine": ENGINE, "n_restarts": 16,
        "refiner_p0_noise_kwargs": {"loc": 0.0, "scale": 5}, "ranking": {"mode": "mdl"}, "seed": 3})
    assert isinstance(adapter, OracleAdapter)
    assert adapter.refine.n_restarts == 16 and adapter.refine.method == "curve_fit_lm" and adapter.seed == 3
    assert adapter.refine.scope == "fittable" and adapter.refine.ladder is not None
    assert adapter.ranking_config()["mode"] == "mdl"


@pytest.mark.parametrize("extra, message", [
    ({"model_path": "x"}, "unknown keys"),
    ({"ranking": None}, "ranking"),
    ({"refiner_method": "newton"}, "method"),
    ({"node_penalty": 0.1}, "node_penalty"),
])
def test_the_config_refuses_what_the_oracle_does_not_read(extra, message):
    block = {"type": "oracle", "simplipy_engine": ENGINE, "ranking": {"mode": "mdl"}, **extra}
    block = {k: v for k, v in block.items() if v is not None}
    with pytest.raises(ValueError, match=message):
        run_config.build_model_adapter(block)


def test_the_flash_ansr_adapter_has_no_oracle_mode():
    block = {"type": "flash_ansr", "model_path": "x", "evaluation_config": {
        "n_restarts": 8, "refiner_p0_noise": "normal", "ranking": {"mode": "mdl"},
        "generation_config": {"method": "oracle"}}}
    with pytest.raises(ValueError, match="type: oracle"):
        run_config.flash_ansr_settings(block)


def test_a_decoder_never_sees_the_ground_truth():
    """The Flash-ANSR adapter hands a model nothing of the problem's ground truth: its generation config is untouched."""
    import flash_ansr.utils.generation as generation

    class _Model:
        def __init__(self, generation_config):
            self.generation_config = generation_config
            self.numpy_errors = None
            self.seen = []

        def fit(self, X, y, **kwargs):
            self.seen.append(self.generation_config)
            raise ValueError("stop after generation config")

    config = generation.SoftmaxSamplingConfig(draws=4)
    model = _Model(config)
    FlashANSRAdapter(model).evaluate_sample(_sample(LAW))
    assert model.seen == [config] and model.seen[0] is config
    assert "expression" not in config.to_kwargs()
