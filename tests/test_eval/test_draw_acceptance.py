"""Accepting frozen draws (srbf.draw_acceptance): each problem keeps the first seeded attempt on which the oracle and the
judge recover its own formula; attempt 1 and a formula-inherent flag when none does."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from srbf.draw_acceptance import (ACCEPTED, RULE, TIME, OracleCheck, accept_draws, cpu_time_cap, merge_journals,
                                  oracle_adapter_config, oracle_check)

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
    """The judge reads every restart's answer that recovers numerically: where it calls each refitted law another
    structure (the judge is strict about a constant factor spelled differently, say), the attempt fails on (c)."""
    import srbf.table

    judge = srbf.table.judge_snapshot

    def strict(snap, *, engine, first_index=None):
        judged = judge(snap, engine=engine, first_index=first_index)
        if snap.get("prediction_success", [False])[0]:
            judged["symbolic_recovery"] = [False]
        return judged

    monkeypatch.setattr(srbf.table, "judge_snapshot", strict)
    check = OracleCheck(_oracle(engine), engine)
    assert check(_problem()) == "symbolic" and check.spent == 4          # every restart recovers, none is recognized


def test_an_attempt_passes_when_any_restart_is_also_recognized(engine, monkeypatch):
    """The judge rejects the first two restarts that recover numerically and recognizes the third: the attempt passes on
    the third, and fails only when no restart among the full count passes both."""
    import srbf.table

    judge = srbf.table.judge_snapshot
    calls = []

    def third(snap, *, engine, first_index=None):
        judged = judge(snap, engine=engine, first_index=first_index)
        if snap.get("prediction_success", [False])[0]:
            calls.append(1)
            judged["symbolic_recovery"] = [len(calls) >= 3]
        return judged

    monkeypatch.setattr(srbf.table, "judge_snapshot", third)
    check = OracleCheck(_oracle(engine), engine)
    assert check(_problem()) == ACCEPTED and check.spent == 3 and len(calls) == 3


def test_a_placeholder_is_never_accepted(engine):
    from symbolic_data import Problem
    check = OracleCheck(_oracle(engine), engine)
    assert check(Problem.placeholder(variables=["x1"], reason="max_trials_exhausted")) == "placeholder"


class _ScriptedCheck:
    """Accepts problem `a` on its third attempt, problem `c` at once, and never problem `b` (nor `d`). It tells the
    attempt from the problem's points, so that a run resumed in the middle of a problem follows the same script, and
    records every attempt it judges in ``judged``."""

    judged: list[tuple[str, int]] = []

    def __init__(self, catalog, settings) -> None:
        from symbolic_data.draws import CatalogDraws
        self.draws = CatalogDraws(str(catalog), settings)
        self.max_trials = int(settings.max_trials)

    def attempt(self, problem) -> int:
        row = self.draws.ids.index(problem.eq_id)
        return next(k for k in range(1, self.max_trials + 1)
                    if np.array_equal(self.draws.draw(row, 1, k).problem.x_support, problem.x_support))

    def __call__(self, problem) -> str:
        attempt = self.attempt(problem)
        _ScriptedCheck.judged.append((problem.eq_id, attempt))
        if problem.eq_id == "c" or (problem.eq_id == "a" and attempt == 3):
            return ACCEPTED
        return "symbolic" if problem.eq_id == "b" else "numeric"


def _script(catalog, settings):
    import functools
    _ScriptedCheck.judged = []
    return functools.partial(_ScriptedCheck, str(catalog), settings)


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
    summary = accept_draws([str(catalog)], out, check_factory=_script(catalog, settings), rule=RULE.format(n_restarts=8, time_cap=1000),
                           settings=settings, log=lambda _: None)
    doc = json.loads(out.read_text())
    chosen = doc["catalogs"]["toy"]["attempts"]["1"]
    assert list(chosen) == ["a", "b", "c"]                                       # the constant formula is not checked
    assert chosen["a"] == {"attempt": 3, "formula_inherent": False, "failures": ["numeric", "numeric"]}
    assert chosen["b"] == {"attempt": 1, "formula_inherent": True, "failures": ["symbolic"] * 4}   # max_trials attempts
    assert chosen["c"] == {"attempt": 1, "formula_inherent": False, "failures": []}
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


def test_a_run_resumes_from_its_journal_in_the_middle_of_a_problem(tmp_path):
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
    accept_draws([str(catalog)], out, check_factory=_script(catalog, settings), **kwargs)
    first = json.loads(out.read_text())
    lines = journal.read_text().splitlines()
    assert len(lines) == 1 + 3 + 4 + 1                                       # one line per attempt: a 3, b 4, c 1
    assert [(e["eq_id"], e["attempt"], e["verdict"]) for e in map(json.loads, lines[1:4])] == [
        ("a", 1, "numeric"), ("a", 2, "numeric"), ("a", 3, ACCEPTED)]
    accept_draws([str(catalog)], tmp_path / "again.json", check_factory=_NeverCalled, **kwargs)   # everything is journaled
    assert json.loads((tmp_path / "again.json").read_text()) == first
    journal.write_text("\n".join(lines[:3]) + "\n" + lines[3][:10])      # stopped while judging a's third attempt
    accept_draws([str(catalog)], tmp_path / "resumed.json", check_factory=_script(catalog, settings), **kwargs)
    assert json.loads((tmp_path / "resumed.json").read_text()) == first
    assert _ScriptedCheck.judged[0] == ("a", 3)                              # a resumes at its third attempt
    assert sorted(_ScriptedCheck.judged) == [("a", 3), ("b", 1), ("b", 2), ("b", 3), ("b", 4), ("c", 1)]
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
    accept_draws([str(catalog)], tmp_path / "whole.json", check_factory=_script(catalog, settings), **kwargs)
    for k in range(3):
        accept_draws([str(catalog)], tmp_path / f"shard{k}.json", check_factory=_script(catalog, settings),
                     journal=tmp_path / f"j{k}.jsonl", shard=(k, 3), **kwargs)
    shard0 = json.loads((tmp_path / "shard0.json").read_text())["catalogs"]["toy"]["attempts"]["1"]
    assert list(shard0) == ["a", "d"]                                  # tasks 0 and 3 of the plan
    journals = [tmp_path / f"j{k}.jsonl" for k in range(3)]
    summary = merge_journals([str(catalog)], tmp_path / "merged.json", journals, **{k: v for k, v in kwargs.items() if k != "log"})
    assert json.loads((tmp_path / "merged.json").read_text()) == json.loads((tmp_path / "whole.json").read_text())
    assert summary["problems"] == 4 and summary["checked"] == 4
    with pytest.raises(ValueError, match="1 of 4 problems are not finished in any journal"):
        merge_journals([str(catalog)], tmp_path / "x.json", journals[:2], rule="r", settings=settings)
    with pytest.raises(ValueError, match="another rule"):
        merge_journals([str(catalog)], tmp_path / "x.json", journals, rule="another", settings=settings)
    with pytest.raises(ValueError, match="another shard"):          # a shard's journal resumes only that shard
        accept_draws([str(catalog)], tmp_path / "x.json", check_factory=_NeverCalled, journal=journals[0], shard=(1, 3),
                     **kwargs)


def test_worker_processes_judge_attempt_by_attempt_and_leave_nothing_behind(tmp_path):
    """With workers, each attempt is a task and a problem's next attempt is queued when its last one comes back: the
    document and the journal hold what a serial run writes, and the workers are gone afterwards."""
    draws = pytest.importorskip("symbolic_data.draws")
    catalog = _toy(tmp_path)
    settings = draws.DrawsSettings(n_support=16, n_validation=8, max_trials=4, simplipy_engine="acj-4")
    kwargs = dict(rule="r", settings=settings, log=lambda _: None)
    accept_draws([str(catalog)], tmp_path / "serial.json", check_factory=_script(catalog, settings), **kwargs)
    before = _children()
    accept_draws([str(catalog)], tmp_path / "pool.json", check_factory=_script(catalog, settings), jobs=2,
                 journal=tmp_path / "pool.jsonl", **kwargs)
    assert json.loads((tmp_path / "pool.json").read_text()) == json.loads((tmp_path / "serial.json").read_text())
    attempts = [json.loads(line) for line in (tmp_path / "pool.jsonl").read_text().splitlines()[1:]]
    assert sorted((e["eq_id"], e["attempt"]) for e in attempts) == sorted(
        [("a", k) for k in (1, 2, 3)] + [("b", k) for k in range(1, 5)] + [("c", 1)] + [("d", k) for k in range(1, 5)])
    assert _children() == before


def test_a_journal_cut_mid_line_is_repaired_before_it_grows(tmp_path):
    draws = pytest.importorskip("symbolic_data.draws")
    catalog = _toy(tmp_path)
    settings = draws.DrawsSettings(n_support=16, n_validation=8, max_trials=4, simplipy_engine="acj-4")
    journal = tmp_path / "journal.jsonl"
    kwargs = dict(rule="r", settings=settings, journal=journal, log=lambda _: None)
    accept_draws([str(catalog)], tmp_path / "first.json", check_factory=_script(catalog, settings), **kwargs)
    lines = journal.read_text().splitlines()
    journal.write_text("\n".join(lines[:3]) + "\n" + lines[3][:12])          # stopped while writing the third attempt
    accept_draws([str(catalog)], tmp_path / "again.json", check_factory=_script(catalog, settings), **kwargs)
    again = journal.read_text().splitlines()
    assert len(again) == len(lines) and all(json.loads(line) for line in again)   # every attempt once, every line whole
    assert json.loads((tmp_path / "again.json").read_text()) == json.loads((tmp_path / "first.json").read_text())


def _profiling_clock() -> tuple[int, float] | None:
    """The calling process's profiling CPU clock, the clock ITIMER_PROF counts (user + system time, sampled at every
    tick; Linux's clock id for a process's CPUCLOCK_PROF, pid 0 meaning the caller), and its resolution, one tick.
    None where it cannot be read."""
    clock = (~0 << 3) | 0
    try:
        time.clock_gettime(clock)
        return clock, time.clock_getres(clock)
    except (OSError, AttributeError):
        return None


#: Where the profiling clock cannot be read, how far below the cap time.process_time may stop: 20 ms. On a loaded
#: 1000 Hz machine the profiling clock ran up to 14 ms ahead of process_time at a 1 s cap (2026-11-27), while it read
#: the cap plus one tick every time.
PROCESS_TIME_MARGIN = 0.02


def _cap_clock() -> tuple[callable, float]:
    """The clock to read a cap's floor on, and how far below the cap it may stop: the profiling clock with one tick of
    tolerance, else process_time with PROCESS_TIME_MARGIN."""
    prof = _profiling_clock()
    if prof is None:
        return time.process_time, PROCESS_TIME_MARGIN
    return (lambda: time.clock_gettime(prof[0])), prof[1]


class _SlowFirstAttempt(OracleCheck):
    """An oracle check whose first attempt would burn CPU far past its cap and whose later attempts pass at once.
    ``capped`` receives, for every attempt that reaches the cap, the CPU time the whole capped call ran on the clock
    the cap counts (:func:`_cap_clock`); the cap covers all of ``_check``, so a first call that loads the catalog may
    reach it before the loop below starts."""

    def __init__(self, catalog, settings, time_cap, capped: list[float]) -> None:
        super().__init__(adapter=None, judge_engine=None, time_cap=time_cap)
        self.script = _ScriptedCheck(catalog, settings)
        self.capped = capped

    def __call__(self, problem) -> str:
        clock, _ = _cap_clock()
        started = clock()
        verdict = super().__call__(problem)
        if verdict == TIME:
            self.capped.append(clock() - started)
        return verdict

    def _check(self, problem) -> str:
        if self.script.attempt(problem) == 1:
            stop = time.process_time() + 60
            while time.process_time() < stop:       # pure Python: the cap's signal lands between two bytecodes
                pass
            raise AssertionError("the cap did not fire")
        return ACCEPTED


def _children() -> tuple[int, int]:
    """Python threads and child processes of this process."""
    import multiprocessing
    import threading
    return threading.active_count(), len(multiprocessing.active_children())


def test_an_attempt_that_reaches_the_cap_fails_with_time_and_the_next_one_runs(tmp_path):
    import functools
    import signal

    draws = pytest.importorskip("symbolic_data.draws")
    catalog = _toy(tmp_path)
    settings = draws.DrawsSettings(n_support=16, n_validation=8, max_trials=4, simplipy_engine="acj-4")
    before, handler = _children(), signal.getsignal(signal.SIGPROF)
    journal = tmp_path / "journal.jsonl"
    capped: list[float] = []
    summary = accept_draws([str(catalog)], tmp_path / "attempts.json", rule="r", settings=settings, journal=journal,
                           check_factory=functools.partial(_SlowFirstAttempt, str(catalog), settings, 0.3, capped),
                           log=lambda _: None)
    chosen = json.loads((tmp_path / "attempts.json").read_text())["catalogs"]["toy"]["attempts"]["1"]
    assert all(c == {"attempt": 2, "formula_inherent": False, "failures": [TIME]} for c in chosen.values())
    assert summary["time"] == 4
    # the cap stops an attempt once the profiling clock reaches it; that clock samples whole ticks and runs ahead of
    # process_time under load, so the floor is read on it, with one tick of tolerance (its resolution)
    _, slack = _cap_clock()
    assert len(capped) == 4 and all(0.3 - slack <= c < 5 for c in capped)
    # the journal records process_time, which may stop up to PROCESS_TIME_MARGIN short of the cap
    first = [json.loads(line) for line in journal.read_text().splitlines()[1:] if json.loads(line)["attempt"] == 1]
    assert all(e["verdict"] == TIME and 0.3 - PROCESS_TIME_MARGIN <= e["timing"]["cpu_seconds"] < 5 for e in first)
    # nothing left behind: no thread, no process, the profiling timer off and the previous handler back
    assert _children() == before
    assert signal.getitimer(signal.ITIMER_PROF) == (0.0, 0.0) and signal.getsignal(signal.SIGPROF) == handler


def test_the_cap_abandons_the_oracle_in_the_middle_of_its_restarts(engine):
    import signal

    check = OracleCheck(_oracle(engine, n_restarts=10 ** 6), engine, time_cap=1.0)
    unrecoverable = _problem(f=lambda x: 2 * x + 0.3 * np.sin(5 * x))       # would run a million restarts
    assert check(unrecoverable) == TIME                                      # warm: the engine's own pools exist now
    before, tasks = _children(), len(os.listdir("/proc/self/task"))
    clock, slack = _cap_clock()
    started, cap_started = time.process_time(), clock()
    verdict = check(unrecoverable)
    spent, capped = time.process_time() - started, clock() - cap_started
    # the cap stops the attempt once the profiling clock reaches it; that clock samples whole ticks and runs ahead of
    # process_time under load, so the floor is read on it, with one tick of tolerance (its resolution)
    assert verdict == TIME and 1.0 - slack <= capped and spent < 6 and 0 < check.spent < 10 ** 6
    assert _children() == before and len(os.listdir("/proc/self/task")) == tasks
    assert signal.getitimer(signal.ITIMER_PROF) == (0.0, 0.0)
    assert check(_problem()) == ACCEPTED                                     # the next attempt runs as usual


def test_the_cap_needs_the_main_thread():
    import threading

    errors = []

    def run():
        try:
            with cpu_time_cap(1.0):
                pass
        except RuntimeError as exc:
            errors.append(str(exc))

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    assert errors and "main thread" in errors[0]


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
    summary = accept_draws([str(catalog)], tmp_path / "attempts.json", check_factory=factory, rule=RULE.format(n_restarts=16, time_cap=1000),
                           settings=settings, log=lambda _: None)
    chosen = json.loads((tmp_path / "attempts.json").read_text())["catalogs"]["toy"]["attempts"]["1"]
    assert {eq_id: c["attempt"] for eq_id, c in chosen.items()} == {"a": 1, "b": 1, "c": 1, "d": 1}
    assert all(not c["formula_inherent"] for c in chosen.values())
    assert all(1 <= summary["seconds"]["toy"][eq_id]["restarts"] <= 16 for eq_id in chosen)
