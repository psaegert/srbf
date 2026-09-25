"""The Flash-ANSR oracle (``generation_config.method: oracle``): the adapter hands the problem's ground truth
to the model, and only in oracle mode. Every other method sees exactly what it saw before."""
import numpy as np
import pytest

import flash_ansr.utils.generation as generation
from srbf.core import EvaluationSample
from srbf.model_adapters import FlashANSRAdapter

LAW = ["+", "*", "2.5", "pow", "x1", "2", "sin", "x2"]


class _Model:
    """The two attributes of FlashANSR the adapter reads before `fit`; `fit` records the generation
    config in force and stops the evaluation (the adapter reports a ValueError as a failed prediction)."""

    def __init__(self, generation_config):
        self.generation_config = generation_config
        self.numpy_errors = None
        self.seen = []

    def fit(self, X, y, **kwargs):
        self.seen.append(self.generation_config)
        raise ValueError("stop after generation config")


def _sample(expression):
    rng = np.random.default_rng(0)
    x = rng.uniform(-2, 2, size=(16, 2))
    y = 2.5 * x[:, 0] ** 2 + np.sin(x[:, 1])
    return EvaluationSample(x_support=x, y_support=y, x_validation=x[:4], y_validation=y[:4],
                            metadata={"expression": expression, "variable_names": ["x1", "x2"]})


def test_a_decoder_never_sees_the_ground_truth():
    config = generation.SoftmaxSamplingConfig(draws=4)
    model = _Model(config)
    FlashANSRAdapter(model).evaluate_sample(_sample(LAW))
    assert model.seen == [config] and model.seen[0] is config        # the same config object, untouched
    assert "expression" not in config.to_kwargs()


@pytest.mark.skipif(not hasattr(generation, "OracleConfig"), reason="flash-ansr without the oracle")
def test_the_oracle_gets_this_problems_ground_truth_and_nothing_else():
    model = _Model(generation.OracleConfig())
    adapter = FlashANSRAdapter(model)
    adapter.evaluate_sample(_sample(LAW))
    adapter.evaluate_sample(_sample(["*", "x1", "3.0"]))
    assert [c.expression for c in model.seen] == [tuple(LAW), ("*", "x1", "3.0")]
    assert all(isinstance(c, generation.OracleConfig) for c in model.seen)


@pytest.mark.skipif(not hasattr(generation, "OracleConfig"), reason="flash-ansr without the oracle")
def test_the_oracle_does_not_run_without_a_ground_truth():
    model = _Model(generation.OracleConfig())
    record = FlashANSRAdapter(model).evaluate_sample(_sample(None)).to_mapping()
    assert model.seen == []                                             # fit never ran
    assert record["prediction_success"] is False and "ground truth" in record["error"]
