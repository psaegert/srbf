import numpy as np
import pytest

from srbf.core import EvaluationResult, EvaluationSample
from srbf import model_adapters


class _DummyModel:
    def to(self, device: str):  # noqa: D401 - simple stub
        return self

    def eval(self):  # noqa: D401 - simple stub
        return self


class _DummyEngine:
    def infix_to_prefix(self, expression):  # noqa: D401 - simple stub
        return [expression]


@pytest.mark.parametrize(
    "pred_key",
    ["best_bfgs_preds", "best_preds"],
)
def test_extract_first_prediction_handles_empty_sequences(pred_key: str):
    output = {pred_key: [None, None]}
    assert model_adapters._extract_first_prediction(
        output,
        preferred_key=pred_key,
        fallback_key=None,
    ) is None


def test_nesymres_adapter_handles_missing_predictions(monkeypatch):
    monkeypatch.setattr(model_adapters, "_HAVE_NESYMRES", True)

    def failing_fitfunc(X_support, y_fit):  # noqa: D401 - simple stub
        return {"best_bfgs_preds": [], "best_bfgs_consts": []}

    adapter = model_adapters.NeSymReSAdapter(
        model=_DummyModel(),
        fitfunc=failing_fitfunc,
        simplipy_engine=_DummyEngine(),
    )

    sample = EvaluationSample(
        x_support=np.zeros((1, 1), dtype=float),
        y_support=np.zeros((1, 1), dtype=float),
        x_validation=np.zeros((0, 1), dtype=float),
        y_validation=np.zeros((0, 1), dtype=float),
    )

    adapter.prepare()
    result: EvaluationResult = adapter.evaluate_sample(sample)
    mapping = result.to_mapping()

    assert mapping["prediction_success"] is False
    assert "error" in mapping
    assert "no expression" in mapping["error"].lower()


def test_nesymres_adapter_removes_padding_before_fit(monkeypatch):
    monkeypatch.setattr(model_adapters, "_HAVE_NESYMRES", True)

    captured: dict[str, np.ndarray] = {}

    def capture_fitfunc(X_support, y_fit):  # noqa: D401 - simple stub
        captured["support"] = X_support.copy()
        raise RuntimeError("boom")

    adapter = model_adapters.NeSymReSAdapter(
        model=_DummyModel(),
        fitfunc=capture_fitfunc,
        simplipy_engine=_DummyEngine(),
        remove_padding=True,
    )
    adapter._max_variables = 3  # emulate configured constraint

    sample = EvaluationSample(
        x_support=np.arange(8, dtype=float).reshape(2, 4),
        y_support=np.zeros((2, 1), dtype=float),
        x_validation=np.zeros((0, 4), dtype=float),
        y_validation=np.zeros((0, 1), dtype=float),
        metadata={
            "variables": ["x1", "x2", "x3", "x4"],
            "skeleton": ["x3", "x4"],
        },
    )

    adapter.evaluate_sample(sample)

    assert "support" in captured
    np.testing.assert_array_equal(captured["support"].shape, (2, 3))
    np.testing.assert_array_equal(captured["support"][:, :2], sample.x_support[:, 2:4])
    assert np.allclose(captured["support"][:, 2], 0.0)


def test_nesymres_adapter_padding_can_be_disabled(monkeypatch):
    monkeypatch.setattr(model_adapters, "_HAVE_NESYMRES", True)

    captured: dict[str, np.ndarray] = {}

    def capture_fitfunc(X_support, y_fit):  # noqa: D401 - simple stub
        captured["support"] = X_support.copy()
        raise RuntimeError("boom")

    adapter = model_adapters.NeSymReSAdapter(
        model=_DummyModel(),
        fitfunc=capture_fitfunc,
        simplipy_engine=_DummyEngine(),
        remove_padding=False,
    )
    adapter._max_variables = 3
    adapter._warned_feature_mismatch = True

    sample = EvaluationSample(
        x_support=np.arange(8, dtype=float).reshape(2, 4),
        y_support=np.zeros((2, 1), dtype=float),
        x_validation=np.zeros((0, 4), dtype=float),
        y_validation=np.zeros((0, 1), dtype=float),
        metadata={
            "variables": ["x1", "x2", "x3", "x4"],
            "skeleton": ["x3", "x4"],
        },
    )

    adapter.evaluate_sample(sample)

    assert "support" in captured
    np.testing.assert_array_equal(captured["support"], sample.x_support[:, :3])


class _FakePySRModel:
    def __init__(self, niterations: int, maxsize: int) -> None:
        self.niterations = niterations
        self.maxsize = maxsize
        self.n_fits = 0

    def fit(self, X, y, variable_names=None):  # noqa: D401 - simple stub
        self.n_fits += 1


def _patch_pysr_factory(monkeypatch):
    created: list[_FakePySRModel] = []

    def fake_create(*, timeout_in_seconds, niterations, use_mult_div_operators, maxsize=None,
                    model_selection="best", parsimony=None):
        model = _FakePySRModel(niterations, maxsize)
        created.append(model)
        return model

    monkeypatch.setattr(model_adapters, "_require_pysr", lambda: object)
    monkeypatch.setattr(model_adapters, "_create_pysr_model", fake_create)
    return created


def test_pysr_adapter_prepare_runs_a_warmup_fit_by_default(monkeypatch):
    # Julia precompile makes the first fit an order-of-magnitude timing outlier; prepare()
    # must pay it on a THROWAWAY model so problem 0's fit_time starts warm.
    created = _patch_pysr_factory(monkeypatch)
    adapter = model_adapters.PySRAdapter(
        timeout_in_seconds=10, niterations=5, use_mult_div_operators=False,
        padding=True, simplipy_engine=_DummyEngine())
    adapter.prepare()

    assert len(created) == 2                      # the timed model + the warmup model
    timed, warmup = created
    assert timed.niterations == 5 and timed.n_fits == 0   # timed model untouched
    assert warmup.niterations == 1 and warmup.n_fits == 1  # warmup fit happened
    assert adapter._model is timed
    # Benchmark policy: baselines run at their upstream defaults -- maxsize is NOT overridden.
    assert timed.maxsize is None
    assert warmup.maxsize is None


def test_pysr_adapter_warmup_can_be_disabled(monkeypatch):
    created = _patch_pysr_factory(monkeypatch)
    adapter = model_adapters.PySRAdapter(
        timeout_in_seconds=10, niterations=5, use_mult_div_operators=False,
        padding=True, simplipy_engine=_DummyEngine(), warmup=False)
    adapter.prepare()

    assert len(created) == 1
    assert created[0].n_fits == 0
