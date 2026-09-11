"""The hybrid arm's adapter is thin: it hands each problem to flash-ansr-hybrid's regressor and records
the answer like every adapter (the method itself is tested in the flash-ansr-hybrid package)."""
from types import SimpleNamespace

import numpy as np
import pytest

from srbf.model_adapters import FlashANSRHybridAdapter
from srbf.testing import toy_sample


class _FakeFlash:
    emission = "fittable"

    def __init__(self):
        self.model = SimpleNamespace(simplipy_engine="engine")
        self.prepared = False

    def get_simplipy_engine(self):
        return self.model.simplipy_engine

    def ranking_config(self):
        return {"mode": "mdl", "mdl_strength": 1e-2}

    def prepare(self, *, data_source=None):
        self.prepared = True

    def _resolve_complexity(self, metadata):
        return metadata.get("complexity")


class _FakeRegressor:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.calls, self.prepared = result, error, [], False

    def prepare(self):
        self.prepared = True

    def fit(self, X, y, *, X_val=None, variables=None, problem_id=None, complexity=None):
        self.calls.append({"X": X, "y": y, "X_val": X_val, "variables": variables, "problem_id": problem_id, "complexity": complexity})
        if self.error:
            raise self.error
        return dict(self.result)


def test_records_the_regressor_answer_and_the_fvu_columns():
    sample = toy_sample(3)
    y = sample.y_support
    result = {"prediction_success": True, "error": None, "predicted_expression": "1.5 * v1", "predicted_source": "pysr",
              "predicted_expression_prefix": ["*", "1.5", "x1"], "fit_time": 1.0, "hybrid_ratio": 0.5,
              "y_pred": y.reshape(-1, 1), "y_pred_val": sample.y_validation.reshape(-1, 1)}
    flash, regressor = _FakeFlash(), _FakeRegressor(result)
    adapter = FlashANSRHybridAdapter(flash, regressor)
    adapter.prepare()
    assert flash.prepared and regressor.prepared and adapter.get_simplipy_engine() == "engine"
    values = adapter.evaluate_sample(sample).to_mapping()
    call = regressor.calls[0]
    assert call["problem_id"] == 3 and call["variables"] == list(sample.metadata["variables"])   # the full list, by column
    np.testing.assert_array_equal(call["X"], sample.x_support)
    assert values["predicted_expression"] == "1.5 * v1" and values["predicted_source"] == "pysr" and values["hybrid_ratio"] == 0.5
    assert values["ranking"] == {"mode": "mdl", "mdl_strength": 1e-2}
    assert values["support_fvu"] == 0.0 and values["validation_fvu"] == 0.0   # the curves are the truth here
    assert values["prediction_success"] is True and values["expression"] == sample.metadata["expression"]   # the sample's metadata stays


def test_a_method_failure_is_the_rows_error():
    sample = toy_sample(0)
    adapter = FlashANSRHybridAdapter(_FakeFlash(), _FakeRegressor(error=RuntimeError("Julia died")))
    values = adapter.evaluate_sample(sample).to_mapping()
    assert values["prediction_success"] is False and values["error"] == "RuntimeError: Julia died"
    assert "support_fvu" not in values


def test_builder_validates_blocks_before_importing_the_method():
    from srbf.config import build_model_adapter
    with pytest.raises(ValueError, match="flash_ansr_hybrid needs"):
        build_model_adapter({"type": "flash_ansr_hybrid", "hybrid": {"ratio": 0.5}})
    with pytest.raises(ValueError):
        build_model_adapter({"type": "flash_ansr_pysr"})                     # the old type is gone
