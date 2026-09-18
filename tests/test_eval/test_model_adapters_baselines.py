import time

import numpy as np
import pytest
from simplipy import SimpliPyEngine

from symbolic_data import LampleChartonCatalog
from srbf.baselines import BruteForceModel, LampleChartonModel
from srbf.core import EvaluationSample
from srbf.model_adapters import (BruteForceAdapter, E2EAdapter, FlashANSRAdapter, LampleChartonAdapter,
                                 NeSymReSAdapter, _evaluate_refiner_baseline)


def test_flash_ansr_adapter_rejects_unknown_complexity_string() -> None:
    # An invalid complexity mode must fail at construction, not lazily on the first problem.
    with pytest.raises(ValueError, match="complexity"):
        FlashANSRAdapter(model=object(), complexity="not-a-mode")


@pytest.mark.parametrize("complexity", ["none", "ground_truth", 3, 4.0, [7]])
def test_flash_ansr_adapter_accepts_valid_complexity(complexity) -> None:
    # The documented modes / numeric / list forms construct without error.
    FlashANSRAdapter(model=object(), complexity=complexity)


@pytest.fixture(scope="module")
def simplipy_engine() -> SimpliPyEngine:
    return SimpliPyEngine.load("acj-4-3", install=True)


def _build_toy_pool(engine: SimpliPyEngine) -> LampleChartonCatalog:
    sample_strategy = {
        "n_operator_distribution": "equiprobable_lengths",
        "min_operators": 0,
        "max_operators": 0,
        "power": 1,
        "max_length": 4,
        "max_tries": 1,
        "independent_dimensions": True,
    }

    support_sampler_config = {
        "support_prior": {
            "name": "uniform",
            "kwargs": {"low": -1, "high": 1, "min_value": -1, "max_value": 1},
        },
        "n_support_prior": {
            "name": "uniform",
            "kwargs": {"low": 6, "high": 6, "min_value": 6, "max_value": 6},
        },
    }

    pool = LampleChartonCatalog.from_dict(
        skeletons={("x1",)},
        simplipy_engine=engine,
        sample_strategy=sample_strategy,
        literal_prior={"name": "normal", "kwargs": {"loc": 0, "scale": 1}},
        variables=["x1"],
        support_sampler_config=support_sampler_config,
    )

    pool.skeletons = {("x1",)}
    return pool


def _build_sample() -> EvaluationSample:
    x_support = np.linspace(-1.0, 1.0, 6).reshape(-1, 1)
    y_support = x_support.copy()
    x_validation = np.linspace(-0.5, 0.5, 2).reshape(-1, 1)
    y_validation = x_validation.copy()

    metadata = {
        "skeleton": ["x1"],
        "variables": ["x1"],
        "variable_names": ["x1"],
        "complexity": 1,
    }

    return EvaluationSample(
        x_support=x_support,
        y_support=y_support,
        x_validation=x_validation,
        y_validation=y_validation,
        metadata=metadata,
    )


def test_lample_charton_adapter_identity(simplipy_engine: SimpliPyEngine) -> None:
    pool = _build_toy_pool(simplipy_engine)
    model = LampleChartonModel(
        simplipy_engine=simplipy_engine,
        catalog=pool,
        samples=1,
        unique=True,
        ignore_holdouts=True,
        seed=0,
        n_restarts=1,
        refiner_p0_noise=None,
    )

    adapter = LampleChartonAdapter(model)
    adapter.prepare()
    sample = _build_sample()

    result = adapter.evaluate_sample(sample)
    values = result.to_mapping()

    assert values["prediction_success"] is True
    np.testing.assert_allclose(values["y_pred"].squeeze(), sample.y_support.squeeze(), atol=1e-3)
    np.testing.assert_allclose(values["y_pred_val"].squeeze(), sample.y_validation.squeeze(), atol=1e-3)
    assert values["predicted_expression"]
    assert values["predicted_skeleton_prefix"] is not None


def test_brute_force_adapter_identity(simplipy_engine: SimpliPyEngine) -> None:
    pool = _build_toy_pool(simplipy_engine)
    model = BruteForceModel(
        simplipy_engine=simplipy_engine,
        catalog=pool,
        max_expressions=16,
        max_length=2,
        include_constant_token=False,
        ignore_holdouts=True,
        n_restarts=1,
        refiner_p0_noise=None,
    )

    adapter = BruteForceAdapter(model)
    adapter.prepare()
    sample = _build_sample()

    result = adapter.evaluate_sample(sample)
    values = result.to_mapping()

    assert values["prediction_success"] is True
    np.testing.assert_allclose(values["y_pred"].squeeze(), sample.y_support.squeeze(), atol=1e-3)
    np.testing.assert_allclose(values["y_pred_val"].squeeze(), sample.y_validation.squeeze(), atol=1e-3)
    assert values["predicted_expression"]
    assert values["predicted_skeleton_prefix"] is not None

# ---- a failed fit is NOT a timed fit -----------------------------------------------------------------------------
# Owner 2026-09-18: "Failures do not count towards the time." The compute axis is the cost of the problems a method
# actually answers; a row that raised carries no fit_time, so the read-out drops it and the published seconds are the
# mean over answered problems. This is a deliberate choice, not an oversight -- these tests pin it, so that "a failed
# fit still spent the time" is not silently reintroduced as a bug fix. (Method failures are already counted where
# they belong: on the y axis, where every error is a miss.) The upstream defects that produce those failures stay
# unpatched by the same ruling: baselines are benchmarked as they ship.
def _toy_sample() -> EvaluationSample:
    x = np.linspace(-1.0, 1.0, 8).reshape(-1, 1)
    return EvaluationSample(x_support=x, y_support=(2.0 * x).reshape(-1, 1),
                            x_validation=np.empty((0, 1)), y_validation=np.empty((0, 1)),
                            metadata={"benchmark_eq_id": "toy"})


class _RaisingModel:
    """A model whose fit burns a measurable slice of time and then raises, like NeSymReS on a wide support box."""

    def fit(self, *args, **kwargs):
        time.sleep(0.01)
        raise ValueError("upstream blew up")

    def __call__(self, *args, **kwargs):     # the NeSymReS adapter calls a fitfunc, not a .fit
        return self.fit(*args, **kwargs)


def _failed(values) -> None:
    assert values["prediction_success"] is False and "upstream blew up" in values["error"]
    assert values.get("fit_time") is None, "a failed fit must carry no time (owner 2026-09-18)"


def test_flash_ansr_adapter_does_not_time_a_failed_fit() -> None:
    _failed(FlashANSRAdapter(model=_RaisingModel()).evaluate_sample(_toy_sample()).values)


def test_refiner_baseline_does_not_time_a_failed_fit() -> None:
    _failed(_evaluate_refiner_baseline(_RaisingModel(), _toy_sample()).values)


def test_e2e_adapter_does_not_time_a_failed_fit() -> None:
    # The constructor imports torch and the upstream package; only the fit path is under test, so build the shell.
    adapter = E2EAdapter.__new__(E2EAdapter)
    adapter._estimator = _RaisingModel()
    adapter.debug = False
    _failed(adapter.evaluate_sample(_toy_sample()).values)


def test_nesymres_adapter_does_not_time_a_failed_fit() -> None:
    adapter = NeSymReSAdapter.__new__(NeSymReSAdapter)
    adapter.model = _RaisingModel()
    adapter.fitfunc = _RaisingModel()
    adapter.remove_padding = False
    adapter.debug = False
    adapter._max_variables = None
    adapter._warned_feature_mismatch = False
    _failed(adapter.evaluate_sample(_toy_sample()).values)
