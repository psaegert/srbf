"""Accepting frozen draws (srbf.draw_acceptance): each problem keeps the first seeded attempt on which the oracle and the
judge recover its own formula; attempt 1 and a formula-inherent flag when none does."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import yaml

from srbf.draw_acceptance import ACCEPTED, RULE, OracleCheck, accept_draws, oracle_adapter_config

ROOT = Path(__file__).resolve().parents[2]
ORACLE_CONFIG = ROOT / "configs/evaluation/scaling/flash-ansr-v25.0-T8-oracle_srbf.yaml"


@pytest.fixture(scope="module")
def engine():
    from simplipy import SimpliPyEngine
    return SimpliPyEngine.load("acj-4", install=True)


def _problem(y_val_constant: bool = False, finite: bool = True):
    from symbolic_data import Problem
    rng = np.random.default_rng(0)
    x, xv = rng.uniform(1, 2, (32, 1)), rng.uniform(1, 2, (16, 1))
    y, yv = 2 * x, (np.full((16, 1), 3.0) if y_val_constant else 2 * xv)
    if not finite:
        y = y.copy()
        y[0, 0] = np.inf
    return Problem(x_support=x, y_support=y, y_support_noisy=y.copy(), x_validation=xv, y_validation=yv,
                   y_validation_noisy=yv.copy(), skeleton=("*", "<constant>", "x1"), expression=["*", "2", "x1"],
                   constants=[2.0], variables=["x1"], complexity=3, eq_id="e")


class _Adapter:
    """Answers each problem as an adapter does (`evaluate_sample` -> the run's record), with a chosen prediction."""

    def __init__(self, mode: str) -> None:
        self.mode = mode

    def evaluate_sample(self, sample):
        from srbf.core import EvaluationResult
        record = sample.clone_metadata()
        if self.mode == "fail":
            return EvaluationResult({**record, "prediction_success": False, "error": "no candidate"})
        prefix = ["abs", "*", "2", "x1"] if self.mode == "other_structure" else ["*", "2", "x1"]
        skeleton = ["abs", "*", "<constant>", "x1"] if self.mode == "other_structure" else ["*", "<constant>", "x1"]
        y_pred_val = sample.y_validation * (1.001 if self.mode == "off" else 1.0)
        record.update(prediction_success=True, predicted_expression=" ".join(prefix), predicted_expression_prefix=prefix,
                      predicted_skeleton_prefix=skeleton, y_pred=sample.y_support.copy(), y_pred_val=y_pred_val)
        return EvaluationResult(record)


@pytest.mark.parametrize("mode, problem, verdict", [
    ("exact", {}, ACCEPTED),
    ("off", {}, "numeric"),
    ("other_structure", {}, "symbolic"),
    ("fail", {}, "no_prediction"),
    ("exact", {"y_val_constant": True}, "fvu_undefined"),
    ("exact", {"finite": False}, "not_finite"),
])
def test_an_attempt_is_judged_by_the_judge(engine, mode, problem, verdict):
    assert OracleCheck(_Adapter(mode), engine)(_problem(**problem)) == verdict


def test_a_placeholder_is_never_accepted(engine):
    from symbolic_data import Problem
    assert OracleCheck(_Adapter("exact"), engine)(Problem.placeholder(variables=["x1"], reason="max_trials_exhausted")) == "placeholder"


class _ScriptedCheck:
    """Accepts problem `a` on its third attempt, problem `c` at once, and never problem `b`."""

    def __init__(self) -> None:
        self.calls: Counter[str] = Counter()

    def __call__(self, problem) -> str:
        self.calls[problem.eq_id] += 1
        if problem.eq_id == "c" or (problem.eq_id == "a" and self.calls["a"] == 3):
            return ACCEPTED
        return "symbolic" if problem.eq_id == "b" else "numeric"


def test_the_first_accepted_attempt_is_kept_and_none_is_flagged(tmp_path):
    draws = pytest.importorskip("symbolic_data.draws")
    unit = {"sample_range": [1.0, 2.0], "sample_type": ["uni", "pos"]}
    catalog = tmp_path / "toy.yaml"
    catalog.write_text(yaml.safe_dump({"metadata": {"name": "toy", "version": 1}, "expressions": {
        "a": {"prepared": "v1 * v2 + 2.5", "vars": {"v1": dict(unit), "v2": dict(unit)}},
        "b": {"prepared": "sin(v1)", "vars": {"v1": dict(unit)}},
        "c": {"prepared": "1.5 * v1", "vars": {"v1": dict(unit)}, "per_run": {"params": {"c1": dict(unit)}, "prepared": "c1 * v1"}},
        "k": {"prepared": "2.5", "vars": {"v1": dict(unit)}}}}))
    settings = draws.DrawsSettings(n_support=16, n_validation=8, max_trials=4, simplipy_engine="acj-4")
    out = tmp_path / "attempts.json"
    summary = accept_draws([str(catalog)], out, check_factory=_ScriptedCheck, rule=RULE.format(n_restarts=8),
                           settings=settings, log=lambda _: None)
    doc = json.loads(out.read_text())
    chosen = doc["catalogs"]["toy"]["attempts"]["1"]
    assert list(chosen) == ["a", "b", "c"]                                       # the constant formula is not checked
    assert chosen["a"] == {"attempt": 3, "formula_inherent": False, "failed": {"numeric": 2}}
    assert chosen["b"] == {"attempt": 1, "formula_inherent": True, "failed": {"symbolic": 4}}   # max_trials attempts
    assert chosen["c"] == {"attempt": 1, "formula_inherent": False, "failed": {}}
    assert summary["checked"] == 3 and summary["later_attempt"] == 1 and summary["formula_inherent"] == 1
    assert set(summary["seconds"]["toy"]) == {"a", "b", "c", "k"}
    # symbolic-data renders the attempts it was handed
    manifest = draws.render([str(catalog)], tmp_path / "draws", settings, attempts=str(out), log=lambda _: None)
    records = draws.read_meta(tmp_path / "draws/toy/repeat-1.npz")["problems"]
    assert [(r["attempt"], r["checked"], r["formula_inherent"]) for r in records] == [(3, True, False), (1, True, True),
                                                                                      (1, True, False), (1, False, False)]
    assert manifest["catalogs"]["toy"]["acceptance"]["1"]["formula_inherent"] == ["b"]
    third = draws.CatalogDraws(str(catalog), settings).draw(0, 1, 3).problem
    with np.load(tmp_path / "draws/toy/repeat-1.npz") as data:
        np.testing.assert_array_equal(data["p0__x_support"], third.x_support)


def test_the_oracle_block_comes_from_the_oracle_config():
    block = oracle_adapter_config(ORACLE_CONFIG, model_path="/models/oracle", n_restarts=1024, seed=7)
    assert block["type"] == "flash_ansr" and block["model_path"] == "/models/oracle" and block["seed"] == 7
    assert block["evaluation_config"]["n_restarts"] == 1024 and block["evaluation_config"]["generation_config"]["method"] == "oracle"
    assert "evaluation_overrides" not in block and block["device"] == "cpu" and block["refiner_workers"] == 1
    with pytest.raises(ValueError, match="not an oracle config"):
        oracle_adapter_config(ROOT / "configs/evaluation/scaling/flash-ansr-v25.0-T8-3M_srbf.yaml",
                              model_path="/m", n_restarts=8, seed=0)


def test_the_flash_ansr_adapter_forwards_its_seed():
    from srbf.core import EvaluationSample
    from srbf.model_adapters import FlashANSRAdapter

    class _Model:
        generation_config = None
        numpy_errors = None

        def __init__(self):
            self.kwargs = []

        def fit(self, X, y, **kwargs):
            self.kwargs.append(kwargs)
            raise ValueError("stop")

    pytest.importorskip("flash_ansr")
    sample = EvaluationSample(x_support=np.ones((4, 1)), y_support=np.ones((4, 1)), x_validation=np.ones((2, 1)),
                              y_validation=np.ones((2, 1)), metadata={"variable_names": ["x1"]})
    seeded, unseeded = _Model(), _Model()
    FlashANSRAdapter(seeded, seed=11).evaluate_sample(sample)
    FlashANSRAdapter(unseeded).evaluate_sample(sample)
    assert seeded.kwargs[0]["seed"] == 11 and "seed" not in unseeded.kwargs[0]


class _NeverCalled:
    def __call__(self, problem) -> str:
        raise AssertionError(f"{problem.eq_id} was judged again")


def test_a_run_resumes_from_its_journal(tmp_path):
    draws = pytest.importorskip("symbolic_data.draws")
    unit = {"sample_range": [1.0, 2.0], "sample_type": ["uni", "pos"]}
    catalog = tmp_path / "toy.yaml"
    catalog.write_text(yaml.safe_dump({"metadata": {"name": "toy", "version": 1}, "expressions": {
        "a": {"prepared": "v1 * v2 + 2.5", "vars": {"v1": dict(unit), "v2": dict(unit)}},
        "b": {"prepared": "sin(v1)", "vars": {"v1": dict(unit)}},
        "c": {"prepared": "1.5 * v1", "vars": {"v1": dict(unit)}}}}))
    settings = draws.DrawsSettings(n_support=16, n_validation=8, max_trials=4, simplipy_engine="acj-4")
    journal, out = tmp_path / "journal.jsonl", tmp_path / "attempts.json"
    kwargs = dict(rule="r", settings=settings, journal=journal, log=lambda _: None)
    accept_draws([str(catalog)], out, check_factory=_ScriptedCheck, **kwargs)
    first = json.loads(out.read_text())
    accept_draws([str(catalog)], tmp_path / "again.json", check_factory=_NeverCalled, **kwargs)   # everything is journaled
    assert json.loads((tmp_path / "again.json").read_text()) == first
    lines = journal.read_text().splitlines()
    journal.write_text("\n".join(lines[:2]) + "\n" + lines[2][:10])                       # stopped after one problem
    accept_draws([str(catalog)], tmp_path / "resumed.json", check_factory=_ScriptedCheck, **kwargs)
    assert json.loads((tmp_path / "resumed.json").read_text()) == first
    with pytest.raises(ValueError, match="another rule"):
        accept_draws([str(catalog)], out, check_factory=_NeverCalled, **{**kwargs, "rule": "another"})
