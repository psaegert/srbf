import numpy as np
import pytest

from srbf.core import EvaluationResult, EvaluationSample
from srbf import model_adapters
from srbf import variable_renaming


class _DummyModel:
    def to(self, device: str):  # noqa: D401 - simple stub
        return self

    def eval(self):  # noqa: D401 - simple stub
        return self


class _DummyEngine:
    def infix_to_prefix(self, expression):  # noqa: D401 - simple stub
        return [expression]

    def read_infix(self, expression, convert_expression=True):  # noqa: D401 - simple stub
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
    from srbf.worker.models import pysr_worker
    created: list[_FakePySRModel] = []

    def fake_create(*, timeout_in_seconds, niterations, maxsize=None,
                    model_selection="best", parsimony=None):
        model = _FakePySRModel(niterations, maxsize)
        created.append(model)
        return model

    monkeypatch.setattr(pysr_worker, "create_model", fake_create)
    return created


def test_pysr_worker_load_runs_a_warmup_fit_by_default(monkeypatch):
    # Julia precompile makes the first fit an order-of-magnitude timing outlier; load()
    # must pay it on a THROWAWAY model so problem 0's fit_time starts warm.
    from srbf.worker.models import pysr_worker
    created = _patch_pysr_factory(monkeypatch)
    state = pysr_worker.load({"timeout_in_seconds": 10, "niterations": 5})

    assert len(created) == 2                      # the timed model + the warmup model
    timed, warmup = created
    assert timed.niterations == 5 and timed.n_fits == 0   # timed model untouched
    assert warmup.niterations == 1 and warmup.n_fits == 1  # warmup fit happened
    assert state["model"] is timed
    # Benchmark policy: baselines run at their upstream defaults -- maxsize is NOT overridden.
    assert timed.maxsize is None
    assert warmup.maxsize is None


def test_pysr_worker_warmup_can_be_disabled(monkeypatch):
    from srbf.worker.models import pysr_worker
    created = _patch_pysr_factory(monkeypatch)
    pysr_worker.load({"timeout_in_seconds": 10, "niterations": 5, "warmup": False})

    assert len(created) == 1
    assert created[0].n_fits == 0


def test_pysr_config_builds_a_worker_backed_adapter(monkeypatch):
    # type: pysr keeps its historical keys and now runs the shipped worker in a subprocess.
    from srbf import config as run_config
    from srbf.subprocess_adapter import SubprocessAdapter
    monkeypatch.setattr(run_config, "resolve_simplipy_engine", lambda cfg, adapter_name: _DummyEngine())
    adapter = run_config.build_model_adapter({
        "type": "pysr", "timeout_in_seconds": 30, "niterations": 3, "padding": False,
        "simplipy_engine": "unused", "python": "/opt/pysr-venv/bin/python", "timeout": 120})
    assert isinstance(adapter, SubprocessAdapter)
    assert adapter.worker.name == "pysr_worker.py"
    assert adapter.python == "/opt/pysr-venv/bin/python"
    assert adapter.timeout == 120.0
    assert adapter.drop_unused_variables is True          # padding: false
    assert adapter.options["timeout_in_seconds"] == 30 and adapter.options["niterations"] == 3
    assert adapter.options["maxsize"] is None and adapter.options["warmup"] is True


class TestEmissionConfig:
    """The promptable emission format is a sampling POLICY (flash-ansr 0.17): the builder puts the
    adapter config's `emission` into the softmax generation config, 'fittable' by default."""

    @staticmethod
    def _generation_kwargs(monkeypatch, config_extra: dict) -> dict:
        import srbf.config as cfg_mod
        captured: dict = {}

        class FakeModel:
            simplipy_engine = object()

        monkeypatch.setattr(cfg_mod.FlashANSR, "load", staticmethod(lambda **kwargs: FakeModel()))

        def fake_create(method, **kwargs):
            captured.update(method=method, **kwargs)
            return object()

        monkeypatch.setattr(cfg_mod, "create_generation_config", fake_create)
        config = {"model_path": "/nowhere", "evaluation_config": {
            "n_restarts": 2, "refiner_p0_noise": "normal", "ranking": {"mode": "mdl"},
            "generation_config": {"method": "softmax_sampling", "kwargs": {"draws": 4}}}}
        config.update(config_extra)
        cfg_mod._build_flash_ansr_adapter(config)
        return captured

    def test_emission_default_is_the_application_mode(self, monkeypatch) -> None:
        # Owner ruling 2026-09-02: the model predicts the typed literals, the refiner fits
        # the placeholders -- <mask_fittable> is the default emission.
        assert self._generation_kwargs(monkeypatch, {})["emission"] == "fittable"

    def test_emission_constants_is_still_selectable(self, monkeypatch) -> None:
        assert self._generation_kwargs(monkeypatch, {"emission": "constants"})["emission"] == "constants"

    def test_emission_skeleton_is_stored(self, monkeypatch) -> None:
        assert self._generation_kwargs(monkeypatch, {"emission": "skeleton"})["emission"] == "skeleton"

    def test_emission_rejects_unknown_values_before_model_load(self) -> None:
        import pytest
        from flash_ansr import SoftmaxSamplingConfig
        with pytest.raises(ValueError, match="emission"):
            SoftmaxSamplingConfig(emission="masked")


class TestRefineScopeConfig:
    """`refine_scope` is a MODEL knob (which literals the refiner may move) and rides on
    FlashANSR.load; the default is the predict-vs-refine doctrine ('fittable')."""

    @staticmethod
    def _build(monkeypatch, config_extra: dict, eval_extra: dict | None = None) -> dict:
        import srbf.config as cfg_mod
        captured: dict = {}

        class FakeModel:
            simplipy_engine = object()

        def fake_load(**kwargs):
            captured.update(kwargs)
            return FakeModel()

        monkeypatch.setattr(cfg_mod.FlashANSR, "load", staticmethod(fake_load))
        monkeypatch.setattr(cfg_mod, "create_generation_config", lambda method, **kw: object())
        eval_cfg = {"n_restarts": 2, "refiner_p0_noise": "normal", "ranking": {"mode": "mdl"},
                    "generation_config": {"method": "softmax", "kwargs": {}}}
        eval_cfg.update(eval_extra or {})
        config = {"model_path": "/nowhere", "evaluation_config": eval_cfg}
        config.update(config_extra)
        cfg_mod._build_flash_ansr_adapter(config)
        return captured

    def test_default_scope_is_fittable(self, monkeypatch) -> None:
        assert self._build(monkeypatch, {})["refine"]["scope"] == "fittable"

    def test_adapter_config_overrides_evaluation_config(self, monkeypatch) -> None:
        captured = self._build(monkeypatch, {"refine_scope": "placeholders"}, {"refine_scope": "all"})
        assert captured["refine"]["scope"] == "placeholders"

    def test_evaluation_config_sets_the_scope(self, monkeypatch) -> None:
        assert self._build(monkeypatch, {}, {"refine_scope": "all"})["refine"]["scope"] == "all"


class TestAnswerProvenanceColumns:
    """The flash_ansr adapter records where its rank-0 answer came from: how many predicted typed
    literals it kept frozen, whether it is a thawed duplicate (which typed token indices it re-fitted)
    and the constant ladder's re-spelling record. Without them a campaign cannot attribute its own
    answers (2026-09-12: the thaw lineage held 26 of 60 erbench-syneq rank-0 answers, invisible)."""

    def test_rank0_provenance_lands_in_the_row(self, monkeypatch) -> None:
        import numpy as np
        from flash_ansr.inference import Candidate, FitResult
        from flash_ansr.scoring import resolve_ranking
        from srbf.core import EvaluationSample
        from srbf.model_adapters import FlashANSRAdapter

        best = Candidate(raw_beam=[7, 8, 9], expression=["pow", "x1", "<constant>"], slots=[2], expression_prefix=["pow", "x1", "2.31"],
                         expression_infix="pow(x1, 2.31)", skeleton_prefix=["pow", "x1", "<constant>"], constants=[2.31],
                         constants_emitted=[2.0], log_prob=-1.0, score=-15.0, fvu=0.0, n_nodes=3, mu=None, mdl=19443.0,
                         constant_count=1, pruned_variant=False, pareto_rank=-1, rank=0,
                         spelling="c0=/ 231 100", typed_frozen=0, typed_thaw="2")

        class FakeModel:
            numpy_errors = "ignore"

            def fit(self, X, y, **kwargs):
                return FitResult(candidates=[best], ledger=None, generation_time=0.1, refinement_time=0.2,
                                 ranking=resolve_ranking("mdl"), n_variables=1)

        monkeypatch.setattr(FitResult, "predict", lambda self, X, rank=0: np.zeros((np.asarray(X).shape[0], 1)))
        adapter = FlashANSRAdapter(FakeModel(), device="cpu", complexity="none")
        monkeypatch.setattr(adapter, "ranking_config", lambda: {"mode": "mdl"})
        x = np.linspace(1.0, 5.0, 16).reshape(-1, 1)
        sample = EvaluationSample(x_support=x, y_support=x[:, 0] ** 2.31, x_validation=np.empty((0, 1)), y_validation=np.empty((0,)),
                                  metadata={"variable_names": ["x1"]})
        record = adapter.evaluate_sample(sample).to_mapping() if hasattr(adapter.evaluate_sample(sample), "to_mapping") else adapter.evaluate_sample(sample).record
        assert record["predicted_expression_prefix"] == ["pow", "x1", "2.31"]
        assert record["predicted_typed_frozen"] == 0
        assert record["predicted_typed_thaw"] == "2"
        assert record["predicted_spelling"] == "c0=/ 231 100"


class _SplitEngine:
    """An engine stub whose reader is just whitespace tokenisation (prefix in, prefix out)."""

    def infix_to_prefix(self, expression):  # noqa: D401 - simple stub
        return str(expression).split()

    def read_infix(self, expression, convert_expression=True):  # noqa: D401 - simple stub
        return str(expression).split()


class TestBaselineVariableDialect:
    """E2E and NeSymReS name the columns they are handed in their own dialect; the ground-truth
    skeleton names the same columns x1, x2, ... positionally. Unmapped, a prediction that IS the law
    shares no variable with it and every symbolic comparison reads zero."""

    def test_columns_are_named_by_position_unless_already_x_names(self) -> None:
        assert variable_renaming.skeleton_variable_names(["v1", "v2", "v3"]) == ["x1", "x2", "x3"]
        assert variable_renaming.skeleton_variable_names(["x3", "x4"]) == ["x3", "x4"]   # unused columns already dropped
        assert variable_renaming.skeleton_variable_names([]) == []
        assert variable_renaming.skeleton_variable_names(None) == []

    def test_e2e_tokens_are_zero_indexed_nesymres_tokens_are_one_indexed(self) -> None:
        names = ["x1", "x2"]
        assert variable_renaming.rename_variable_tokens(
            ["+", "x_0", "*", "<constant>", "x_1"], names, first_index=variable_renaming.E2E_FIRST_INDEX
        ) == ["+", "x1", "*", "<constant>", "x2"]
        assert variable_renaming.rename_variable_tokens(
            ["+", "x_1", "*", "<constant>", "x_2"], names, first_index=variable_renaming.NESYMRES_FIRST_INDEX
        ) == ["+", "x1", "*", "<constant>", "x2"]

    def test_a_column_the_model_was_never_given_keeps_its_own_name(self) -> None:
        # NeSymReS pads the column block with zeros; a prediction that reads a padding column is a
        # miss, and inventing a variable name for it would hide that.
        assert variable_renaming.rename_variable_tokens(["x_9"], ["x1"], first_index=1) == ["x_9"]
        assert variable_renaming.rename_variable_tokens(None, ["x1"], first_index=1) is None

    def test_the_infix_map_leaves_identifiers_that_merely_end_in_a_variable_alone(self) -> None:
        assert variable_renaming.rename_variables_in_infix("mulx_0 + x_0", ["x1"], first_index=0) == "mulx_0 + x1"

    def test_nesymres_adapter_maps_its_prediction_onto_the_ground_truth_names(self, monkeypatch) -> None:
        monkeypatch.setattr(model_adapters, "_HAVE_NESYMRES", True)

        def fitfunc(X_support, y_fit):  # noqa: D401 - simple stub
            return {"best_bfgs_preds": ["+ x_1 x_2"], "best_bfgs_consts": [[]]}

        adapter = model_adapters.NeSymReSAdapter(
            model=_DummyModel(), fitfunc=fitfunc, simplipy_engine=_SplitEngine(), remove_padding=False)
        sample = EvaluationSample(
            x_support=np.ones((2, 2), dtype=float),
            y_support=np.zeros((2, 1), dtype=float),
            x_validation=np.zeros((0, 2), dtype=float),
            y_validation=np.zeros((0, 1), dtype=float),
            metadata={"variables": ["v1", "v2"], "skeleton": ["+", "x1", "x2"]},
        )

        mapping = adapter.evaluate_sample(sample).to_mapping()

        assert mapping["predicted_skeleton_prefix"] == ["+", "x1", "x2"]
        assert mapping["predicted_expression"] == "+ x1 x2"

    def test_e2e_adapter_maps_its_prediction_onto_the_ground_truth_names(self) -> None:
        class _Tree:
            def infix(self):  # noqa: D401 - simple stub
                return "x_0 + x_1"

        class _Estimator:
            def fit(self, X, y, verbose=False):  # noqa: D401 - simple stub
                return None

            def predict(self, X):  # noqa: D401 - simple stub
                return np.zeros(X.shape[0])

            def retrieve_tree(self, with_infos=True):  # noqa: D401 - simple stub
                return {"predicted_tree": _Tree()}

        adapter = model_adapters.E2EAdapter(model_path="unused", simplipy_engine=_SplitEngine())
        adapter._estimator = _Estimator()
        sample = EvaluationSample(
            x_support=np.ones((2, 2), dtype=float),
            y_support=np.zeros((2, 1), dtype=float),
            x_validation=np.zeros((0, 2), dtype=float),
            y_validation=np.zeros((0, 1), dtype=float),
            metadata={"variables": ["v1", "v2"], "skeleton": ["+", "x1", "x2"]},
        )

        mapping = adapter.evaluate_sample(sample).to_mapping()

        # the engine stub tokenises without reordering, so this is E2E's own infix, mapped
        assert mapping["predicted_skeleton_prefix"] == ["x1", "+", "x2"]


class TestWorkerVariableDialect:
    """An out-of-process worker is told the problem's variable names and answers in them: PySR says
    v1 on a catalog whose columns are called that, while the ground truth says x1."""

    def test_the_handed_names_map_onto_the_skeletons(self) -> None:
        assert variable_renaming.rename_named_variables(["+", "v1", "*", "<constant>", "v2"], ["v1", "v2"]) == [
            "+", "x1", "*", "<constant>", "x2"]
        assert variable_renaming.rename_named_variables_in_infix("v1 + 2*v2", ["v1", "v2"]) == "x1 + 2*x2"

    def test_a_worker_that_ignores_the_names_is_left_alone(self) -> None:
        # diffsym answers in x1, x2 whatever it is handed; those tokens are not the handed names
        assert variable_renaming.rename_named_variables(["+", "x1", "x2"], ["v1", "v2"]) == ["+", "x1", "x2"]
        assert variable_renaming.rename_named_variables(["+", "x3", "x4"], ["x3", "x4"]) == ["+", "x3", "x4"]

    def test_a_longer_name_is_not_eaten_by_a_shorter_one(self) -> None:
        assert variable_renaming.rename_named_variables_in_infix("v1 + v11", ["v1", "v11"]) == "x1 + x2"

    def test_the_map_is_always_a_bijection(self) -> None:
        # a mixed list (one x-name among catalog names) must not send two columns to the same name
        names = variable_renaming.skeleton_variable_names(["v1", "x1"])
        assert names == ["x1", "x2"] and len(set(names)) == 2
