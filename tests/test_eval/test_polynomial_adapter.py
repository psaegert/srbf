"""The polynomial baseline in a run: its config block, its record, its candidates in the run's store, and its result
file read by `srbf table` at its one rung."""
import pickle

import numpy as np
import pytest
from simplipy import SimpliPyEngine

import srbf.config
from srbf.benchmark import Benchmark
from srbf.candidates import load_candidates
from srbf.config import build_model_adapter
from srbf.core import EvaluationSample
from srbf.metrics.numeric import fvu
from srbf.model_adapters import PolynomialAdapter
from srbf.rungs import RESULT_FILE
from srbf.sample_metadata import build_base_metadata
from srbf.testing import toy_sample


@pytest.fixture(scope="module")
def engine() -> SimpliPyEngine:
    return SimpliPyEngine.load("acj-4-3", install=True)


class _ListSource:
    def __init__(self, problems):
        self.problems = problems

    def __iter__(self):
        yield from self.problems

    def size_hint(self):
        return len(self.problems)


def _adapter(engine, monkeypatch, **keys) -> PolynomialAdapter:
    monkeypatch.setattr(srbf.config, "resolve_simplipy_engine", lambda cfg, adapter_name: engine)
    return build_model_adapter({"type": "polynomial", "config_provenance": "harness_tuned",
                                "simplipy_engine": "acj-4-3", **keys})


CUBIC = ["+", "-", "pow", "x1", "3", "*", "2", "x1", "0.5"]     # x1**3 - 2*x1 + 0.5


def _cubic(seed: int, *, noise: float = 0.0) -> EvaluationSample:
    """A one-variable cubic in a catalog's column name (v1), 64 fitting and 32 held-out points."""
    rng = np.random.default_rng(seed)
    x, x_val = rng.uniform(-2.0, 2.0, (64, 1)), rng.uniform(-2.0, 2.0, (32, 1))
    y, y_val = x ** 3 - 2.0 * x + 0.5, x_val ** 3 - 2.0 * x_val + 0.5
    noisy = y + noise * rng.normal(size=y.shape) if noise else y
    meta = build_base_metadata(skeleton=CUBIC, expression=CUBIC, variables=["v1"], x_support=x, y_support=y,
                               x_validation=x_val, y_validation=y_val, y_support_noisy=noisy, y_validation_noisy=y_val,
                               noise_level=noise,
                               extra_fields={"benchmark_eq_id": f"cubic-{seed}", "eval_row_index": seed, "gt_kind": "exact"})
    return EvaluationSample(x_support=x, y_support=y, x_validation=x_val, y_validation=y_val,
                            y_support_noisy=noisy if noise else None, metadata=meta)


def test_the_config_block_builds_the_adapter_and_refuses_unknown_keys(engine, monkeypatch):
    adapter = _adapter(engine, monkeypatch)
    assert isinstance(adapter, PolynomialAdapter) and adapter.get_simplipy_engine() is engine
    assert (adapter.model.max_degree, adapter.model.n_folds, adapter.model.seed) == (10, 5, 0)
    assert adapter.ranking_config() == {"mode": "cross_validation", "folds": 5, "rule": "one_standard_error", "seed": 0}
    assert _adapter(engine, monkeypatch, max_degree=4, n_folds=10, seed=3).model.get_params() == {
        "max_degree": 4, "n_folds": 10, "seed": 3}
    with pytest.raises(ValueError, match="unknown keys"):
        _adapter(engine, monkeypatch, max_degre=4)
    with pytest.raises(ValueError, match="n_folds"):
        _adapter(engine, monkeypatch, n_folds=1)


def test_the_record_carries_the_answer_its_degree_and_every_degree_as_a_candidate(engine, monkeypatch):
    record = _adapter(engine, monkeypatch).evaluate_sample(_cubic(0)).to_mapping()
    assert record["prediction_success"] is True and record["fit_time"] > 0
    assert record["polynomial_degrees"] == list(range(11)) and record["polynomial_degree"] == 3
    assert len(record["polynomial_cv_mse"]) == len(record["polynomial_cv_se"]) == 11
    assert set(record["polynomial_timing"]) == {"fit", "cross_validation"}
    assert record["ranking"]["mode"] == "cross_validation"
    assert record["candidate_notation"] == "prefix" and record["candidate_variables"] == ["x1"]
    assert len(record["candidate_expressions"]) == 11 and record["candidate_prediction"] == 3
    answer = record["candidate_expressions"][3].split()
    assert record["predicted_expression_prefix"] == answer
    assert list(engine.read_infix(record["predicted_expression"])) == answer
    assert "<constant>" in record["predicted_skeleton_prefix"]
    # constants at full precision: the record's constants are the written ones, bit for bit
    assert all(repr(c) in answer or repr(-c) in answer for c in record["predicted_constants"])
    assert len(record["predicted_constants"]) == 4
    assert fvu(_cubic(0).y_validation, record["y_pred_val"]) < 1e-20


def test_a_noisy_problem_is_fitted_to_its_noisy_targets(engine, monkeypatch):
    sample = _cubic(1, noise=0.5)
    record = _adapter(engine, monkeypatch).evaluate_sample(sample).to_mapping()
    assert record["prediction_success"] is True
    assert fvu(sample.y_support, record["y_pred"]) > 1e-6     # it fits the noisy targets, not the clean ones


def test_a_failed_fit_is_a_failed_prediction(engine, monkeypatch):
    sample = _cubic(2)
    sample.y_support[0, 0] = np.nan
    record = _adapter(engine, monkeypatch).evaluate_sample(sample).to_mapping()
    assert record["prediction_success"] is False and "finite" in record["error"]
    assert "candidate_expressions" not in record


def test_a_run_stores_every_degree_judged_with_the_answer_marked(engine, monkeypatch, tmp_path):
    out = tmp_path / "toy" / "degree_10.pkl"
    problems = [_cubic(0), _cubic(1, noise=0.5), toy_sample(2)]
    Benchmark(_ListSource(problems), _adapter(engine, monkeypatch), engine_name="acj-4-3").run(
        output_path=str(out), progress=False, verbose=False, summary_interval=None)
    with open(out, "rb") as fh:
        snapshot = pickle.load(fh)
    assert snapshot["n_candidates"] == [11, 11, 9]           # two variables, 48 points: degree 9 would need 55
    stored = load_candidates(out)
    assert list(stored) == [0, 1, 2]
    for i, (row, c) in enumerate(stored.items()):
        assert c.valid.all() and c.notation == "prefix"
        assert c.predicted == snapshot["polynomial_degrees"][i].index(snapshot["polynomial_degree"][i])
        assert c.expressions[c.predicted].split() == snapshot["predicted_expression_prefix"][i]
        # the judge's FVU of the answer is the run's, from the same written expression
        assert c.fvu_fit[c.predicted] == pytest.approx(fvu(problems[i].y_support, snapshot["y_pred"][i]), rel=1e-6, abs=1e-30)
        assert np.all(np.diff(c.bits) > 0)                    # each degree adds terms, and costs more bits
    assert stored[2].meta["prediction_success"] and snapshot["polynomial_degree"][2] == 1
    assert stored[2].fvu_val[stored[2].predicted] < 1e-20


def test_the_result_file_is_read_at_its_one_rung(engine, monkeypatch, tmp_path):
    from srbf.table import COLUMNS, judge_result_file

    out = tmp_path / "toy" / "degree_10.pkl"
    assert RESULT_FILE.search(out.name).group(1) == "10"
    Benchmark(_ListSource([toy_sample(0), toy_sample(1)]), _adapter(engine, monkeypatch), engine_name="acj-4-3").run(
        output_path=str(out), progress=False, verbose=False, summary_interval=None)
    rows = judge_result_file(str(out), method="polynomial", engine=engine)
    assert len(rows) == 2
    for row in rows:
        r = dict(zip(COLUMNS, row))
        assert r["rung"] == 10 and r["success"] == 1 and r["numeric_recovery_val"] == 1


def test_an_answer_too_long_for_the_engine_is_still_judged_from_its_predictions(engine, monkeypatch, tmp_path):
    from srbf.baselines.polynomial_model import monomial_exponents
    from srbf.candidates import STATUS_JUDGED, STATUS_UNEVALUABLE

    rng = np.random.default_rng(0)
    exponents = monomial_exponents(4, 8)                      # 495 monomials: a degree-8 law in four variables
    weights = rng.normal(size=len(exponents))

    def law(X):
        return (np.prod(X[:, None, :] ** exponents[None], axis=2) @ weights).reshape(-1, 1)

    x, x_val = rng.uniform(-1.0, 1.0, (512, 4)), rng.uniform(-1.0, 1.0, (64, 4))
    meta = build_base_metadata(skeleton=["x1"], expression=["x1"], variables=["x1", "x2", "x3", "x4"], x_support=x,
                               y_support=law(x), x_validation=x_val, y_validation=law(x_val), y_support_noisy=law(x),
                               y_validation_noisy=law(x_val), noise_level=0.0,
                               extra_fields={"benchmark_eq_id": "long", "eval_row_index": 0, "gt_kind": "exact"})
    sample = EvaluationSample(x_support=x, y_support=law(x), x_validation=x_val, y_validation=law(x_val), metadata=meta)
    out = tmp_path / "toy" / "degree_10.pkl"
    snapshot = Benchmark(_ListSource([sample]), _adapter(engine, monkeypatch), engine_name="acj-4-3").run(
        output_path=str(out), progress=False, verbose=False, summary_interval=None)
    assert snapshot["prediction_success"] == [True] and snapshot["polynomial_degree"] == [8]
    assert len(snapshot["predicted_expression_prefix"][0]) > 4096
    assert fvu(law(x_val), snapshot["y_pred_val"][0]) < 1e-20
    stored = load_candidates(out)[0]
    assert stored.valid[:8].all() and stored.predicted == 8
    assert stored.status[8] in (STATUS_JUDGED, STATUS_UNEVALUABLE)    # the engine evaluates at most 4096 tokens
