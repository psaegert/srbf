"""Accepting frozen draws (srbf.draw_acceptance): each problem keeps the first seeded attempt on which the oracle and the
judge recover its own formula; attempt 1 and a formula-inherent flag when none does."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import yaml

from srbf.draw_acceptance import (ACCEPTED, RULE, OracleCheck, accept_draws, merge_journals, oracle_adapter_config,
                                  oracle_check)

ROOT = Path(__file__).resolve().parents[2]
ORACLE_CONFIG = ROOT / "configs/evaluation/scaling/oracle_srbf.yaml"


@pytest.fixture(scope="module")
def engine():
    from simplipy import SimpliPyEngine
    return SimpliPyEngine.load("acj-4", install=True)


def _problem(y_val_constant: bool = False, finite: bool = True, expression=("*", "2", "x1"), skeleton=("*", "<constant>", "x1"),
             f=lambda x: 2 * x, low: float = 1.0, high: float = 2.0):
    from symbolic_data import Problem
    rng = np.random.default_rng(0)
    x, xv = rng.uniform(low, high, (32, 1)), rng.uniform(low, high, (16, 1))
    y, yv = f(x), (np.full((16, 1), 3.0) if y_val_constant else f(xv))
    if not finite:
        y = y.copy()
        y[0, 0] = np.inf
    return Problem(x_support=x, y_support=y, y_support_noisy=y.copy(), x_validation=xv, y_validation=yv,
                   y_validation_noisy=yv.copy(), skeleton=tuple(skeleton), expression=list(expression),
                   constants=[2.0], variables=["x1"], complexity=len(skeleton), eq_id="e")


def _oracle(engine, n_restarts: int = 4):
    pytest.importorskip("flash_ansr", reason="the oracle fits with flash-ansr's refiner")
    from flash_ansr.scoring import resolve_ranking

    from srbf.model_adapters import OracleAdapter
    from srbf.oracle import refine_config
    return OracleAdapter(engine, refine=refine_config({"n_restarts": n_restarts}), ranking=resolve_ranking("mdl"), seed=0)


@pytest.mark.parametrize("problem, verdict", [
    ({}, ACCEPTED),
    ({"f": lambda x: 2 * x + 0.3 * np.sin(5 * x)}, "numeric"),                      # the data of another law
    ({"expression": ("+", "log", "x1", "2"), "low": -2.0, "high": -1.0}, "no_prediction"),   # no restart converges
    ({"y_val_constant": True}, "fvu_undefined"),
    ({"finite": False}, "not_finite"),
])
def test_an_attempt_is_judged_by_the_oracle_and_the_judge(engine, problem, verdict):
    check = OracleCheck(_oracle(engine), engine)
    assert check(_problem(**problem)) == verdict
    assert check.spent == {ACCEPTED: 1, "numeric": 4, "no_prediction": 4}.get(verdict, 0)


def test_a_recovery_the_judge_does_not_recognize_fails_the_attempt(engine, monkeypatch):
    """The judge reads the stopping restart's answer: where it calls the refitted law another structure (the judge is
    strict about a constant factor spelled differently, say), the attempt fails on (c)."""
    import srbf.table

    judge = srbf.table.judge_snapshot

    def strict(snap, *, engine, first_index=None):
        judged = judge(snap, engine=engine, first_index=first_index)
        if snap.get("prediction_success", [False])[0]:
            judged["symbolic_recovery"] = [False]
        return judged

    monkeypatch.setattr(srbf.table, "judge_snapshot", strict)
    check = OracleCheck(_oracle(engine), engine)
    assert check(_problem()) == "symbolic" and check.spent == 1


def test_a_placeholder_is_never_accepted(engine):
    from symbolic_data import Problem
    check = OracleCheck(_oracle(engine), engine)
    assert check(Problem.placeholder(variables=["x1"], reason="max_trials_exhausted")) == "placeholder"


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
    block = oracle_adapter_config(ORACLE_CONFIG, n_restarts=1024, seed=7)
    assert block["type"] == "oracle" and block["seed"] == 7 and block["n_restarts"] == 1024
    assert block["simplipy_engine"] == "acj-5-4-llm" and "model_path" not in block
    with pytest.raises(ValueError, match="not an oracle config"):
        oracle_adapter_config(ROOT / "configs/evaluation/scaling/flash-ansr-v25.0-T8-3M_srbf.yaml", n_restarts=8, seed=0)


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


def _toy(tmp_path):
    unit = {"sample_range": [1.0, 2.0], "sample_type": ["uni", "pos"]}
    catalog = tmp_path / "toy.yaml"
    catalog.write_text(yaml.safe_dump({"metadata": {"name": "toy", "version": 1}, "expressions": {
        "a": {"prepared": "v1 * v2 + 2.5", "vars": {"v1": dict(unit), "v2": dict(unit)}},
        "b": {"prepared": "sin(v1)", "vars": {"v1": dict(unit)}},
        "c": {"prepared": "1.5 * v1", "vars": {"v1": dict(unit)}},
        "d": {"prepared": "v1 / 2.5", "vars": {"v1": dict(unit)}}}}))
    return catalog


def test_shards_are_independent_runs_whose_journals_merge_into_the_whole_document(tmp_path):
    draws = pytest.importorskip("symbolic_data.draws")
    catalog = _toy(tmp_path)
    settings = draws.DrawsSettings(n_support=16, n_validation=8, max_trials=4, simplipy_engine="acj-4")
    kwargs = dict(rule="r", settings=settings, log=lambda _: None)
    accept_draws([str(catalog)], tmp_path / "whole.json", check_factory=_ScriptedCheck, **kwargs)
    for k in range(3):
        accept_draws([str(catalog)], tmp_path / f"shard{k}.json", check_factory=_ScriptedCheck,
                     journal=tmp_path / f"j{k}.jsonl", shard=(k, 3), **kwargs)
    shard0 = json.loads((tmp_path / "shard0.json").read_text())["catalogs"]["toy"]["attempts"]["1"]
    assert list(shard0) == ["a", "d"]                                  # tasks 0 and 3 of the plan
    journals = [tmp_path / f"j{k}.jsonl" for k in range(3)]
    summary = merge_journals([str(catalog)], tmp_path / "merged.json", journals, **{k: v for k, v in kwargs.items() if k != "log"})
    assert json.loads((tmp_path / "merged.json").read_text()) == json.loads((tmp_path / "whole.json").read_text())
    assert summary["problems"] == 4 and summary["checked"] == 4
    with pytest.raises(ValueError, match="1 of 4 problems are in no journal"):
        merge_journals([str(catalog)], tmp_path / "x.json", journals[:2], rule="r", settings=settings)
    with pytest.raises(ValueError, match="another rule"):
        merge_journals([str(catalog)], tmp_path / "x.json", journals, rule="another", settings=settings)
    with pytest.raises(ValueError, match="another shard"):          # a shard's journal resumes only that shard
        accept_draws([str(catalog)], tmp_path / "x.json", check_factory=_NeverCalled, journal=journals[0], shard=(1, 3),
                     **kwargs)


def test_a_journal_cut_mid_line_is_repaired_before_it_grows(tmp_path):
    draws = pytest.importorskip("symbolic_data.draws")
    catalog = _toy(tmp_path)
    settings = draws.DrawsSettings(n_support=16, n_validation=8, max_trials=4, simplipy_engine="acj-4")
    journal = tmp_path / "journal.jsonl"
    kwargs = dict(rule="r", settings=settings, journal=journal, log=lambda _: None)
    accept_draws([str(catalog)], tmp_path / "first.json", check_factory=_ScriptedCheck, **kwargs)
    lines = journal.read_text().splitlines()
    journal.write_text("\n".join(lines[:3]) + "\n" + lines[3][:12])          # stopped while writing the third problem
    accept_draws([str(catalog)], tmp_path / "again.json", check_factory=_ScriptedCheck, **kwargs)
    again = journal.read_text().splitlines()
    assert len(again) == 5 and all(json.loads(line) for line in again)      # header + four problems, every line whole
    assert json.loads((tmp_path / "again.json").read_text()) == json.loads((tmp_path / "first.json").read_text())


def test_the_oracle_check_runs_from_the_oracle_config_and_counts_its_restarts(tmp_path):
    draws = pytest.importorskip("symbolic_data.draws")
    pytest.importorskip("flash_ansr", reason="the oracle fits with flash-ansr's refiner")
    import functools

    config = tmp_path / "oracle.yaml"
    config.write_text(yaml.safe_dump({"experiments": {"toy": {
        "data_source": {"catalog": "toy", "sampling": {"n_support": 16, "n_validation": 8, "problems_per_expression": 1}},
        "model_adapter": {"type": "oracle", "config_provenance": "author_blessed", "simplipy_engine": "acj-4",
                          "ranking": {"mode": "mdl"}, "n_restarts": 8},
        "runner": {"output": str(tmp_path / "out.pkl")}}}}))
    catalog = _toy(tmp_path)
    settings = draws.DrawsSettings(n_support=32, n_validation=16, max_trials=2, simplipy_engine="acj-4")
    factory = functools.partial(oracle_check, str(config), 16, 0, "acj-4")
    summary = accept_draws([str(catalog)], tmp_path / "attempts.json", check_factory=factory, rule=RULE.format(n_restarts=16),
                           settings=settings, log=lambda _: None)
    chosen = json.loads((tmp_path / "attempts.json").read_text())["catalogs"]["toy"]["attempts"]["1"]
    assert {eq_id: c["attempt"] for eq_id, c in chosen.items()} == {"a": 1, "b": 1, "c": 1, "d": 1}
    assert all(not c["formula_inherent"] for c in chosen.values())
    assert all(1 <= summary["seconds"]["toy"][eq_id]["restarts"] <= 16 for eq_id in chosen)
