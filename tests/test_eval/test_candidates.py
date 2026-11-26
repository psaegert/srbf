"""Every candidate of every method, stored next to the run's results and judged by srbf itself (srbf.candidates): the
judgement, the store, the run's side (Benchmark + CandidateRecorder), the worker protocol and the reading API."""
import json
import pickle
import textwrap
from pathlib import Path

import numpy as np
import pytest
from simplipy import SimpliPyEngine

from srbf import candidates as cand
from srbf.benchmark import Benchmark
from srbf.candidates import (
    STATUS_JUDGED, STATUS_UNEVALUABLE, STATUS_UNPRICED, STATUS_UNREADABLE, CandidateLog, CandidateRecorder,
    candidate_store_path, iter_candidates, judge_candidates, load_candidates, read_manifests, rejudge_candidates)
from srbf.core import EvaluationResult, EvaluationSample
from srbf.metrics.numeric import fvu
from srbf.store import ResultStore
from srbf.subprocess_adapter import SubprocessAdapter
from srbf.table import mdl_bits
from srbf.testing import toy_sample
from srbf.worker import runner


@pytest.fixture(scope="module")
def engine() -> SimpliPyEngine:
    return SimpliPyEngine.load("acj-4-3", install=True)


EXACT = "2.0*v1 - 0.5*v2 + 1.0"     # toy_sample's law in the catalog's column names


def _toy(seed: int = 0, **kwargs) -> EvaluationSample:
    """toy_sample with catalog column names (v1, v2), as the srbf catalogs hand them over."""
    sample = toy_sample(seed, **kwargs)
    meta = dict(sample.metadata, variables=["v1", "v2"], variable_names=["v1", "v2"])
    return EvaluationSample(x_support=sample.x_support, y_support=sample.y_support, x_validation=sample.x_validation,
                            y_validation=sample.y_validation, metadata=meta)


def _judge(engine, expressions, sample=None, notation="infix", variables=("v1", "v2")):
    sample = sample or _toy()
    return judge_candidates(expressions, engine=engine, variables=list(variables), x_fit=sample.x_support,
                            y_fit=sample.y_support, x_val=sample.x_validation, y_val=sample.y_validation,
                            notation=notation)


# ---- the judgement -------------------------------------------------------------------------------------------------
class TestJudgement:
    def test_a_candidate_is_priced_and_its_fvu_taken_on_both_splits(self, engine):
        sample = _toy()
        j = _judge(engine, [EXACT, "2.0*v1"], sample)
        assert list(j.status) == [STATUS_JUDGED, STATUS_JUDGED] and j.reasons == ["", ""]
        assert j.fvu_fit[0] == 0.0 and j.fvu_val[0] == 0.0
        x, xv = sample.x_support, sample.x_validation
        assert j.fvu_fit[1] == fvu(sample.y_support, 2.0 * x[:, 0]) and j.fvu_val[1] == fvu(sample.y_validation, 2.0 * xv[:, 0])
        # the price is the table's: mdl_bits of the prefix the judge read, in the ground truth's names
        assert j.prefixes[1] == "* 2.0 x1"
        assert j.bits[1] == mdl_bits(engine)(["*", "2.0", "x1"]) and j.bits[0] > j.bits[1] > 0

    def test_what_the_judge_cannot_read_evaluate_or_price_says_so(self, engine, monkeypatch):
        j = _judge(engine, ["erf(v1)", "", "x9 + 1.0", "v1 + v2"], notation="infix")
        assert list(j.status[:2]) == [STATUS_UNREADABLE, STATUS_UNREADABLE]
        assert "erf" in j.reasons[0] and np.isnan(j.bits[0]) and np.isnan(j.fvu_fit[0]) and j.prefixes[0] == ""
        # a variable the problem does not have is read, priced, and cannot be evaluated
        assert j.status[2] == STATUS_UNEVALUABLE and np.isfinite(j.bits[2]) and np.isnan(j.fvu_val[2])
        assert j.status[3] == STATUS_JUDGED
        real = cand.mdl_bits
        monkeypatch.setattr(cand, "mdl_bits", lambda e: (lambda prefix: float("nan") if "x2" in prefix else real(e)(prefix)))
        j = _judge(engine, ["v1 + v2", "v1"])
        assert list(j.status) == [STATUS_UNPRICED, STATUS_JUDGED] and np.isfinite(j.fvu_fit[0])

    def test_a_placeholder_is_unreadable_candidates_carry_their_constants(self, engine):
        j = _judge(engine, ["* <constant> x1"], notation="prefix")
        assert j.status[0] == STATUS_UNREADABLE and "placeholder" in j.reasons[0]

    def test_a_non_finite_value_is_judged_at_an_infinite_fvu(self, engine):
        sample = _toy()
        j = _judge(engine, ["log(v1)"], sample)      # half the toy points are negative
        assert j.status[0] == STATUS_JUDGED and j.fvu_fit[0] == np.inf and np.isfinite(j.bits[0])

    def test_a_problem_without_held_out_points_has_no_held_out_fvu(self, engine):
        sample = _toy(n_validation=0)
        j = _judge(engine, [EXACT], sample)
        assert j.fvu_fit[0] == 0.0 and np.isnan(j.fvu_val[0]) and j.status[0] == STATUS_JUDGED

    def test_prefix_and_infix_notation_are_judged_alike(self, engine):
        a = _judge(engine, ["2.0*v1 + sin(v2)", "sqrt(v1**2) + Abs(v2)"])
        b = _judge(engine, ["+ * 2.0 x1 sin x2", "+ sqrt pow x1 2 Abs x2"], notation="prefix")
        assert a.prefixes == b.prefixes
        np.testing.assert_array_equal(a.bits, b.bits)
        np.testing.assert_array_equal(a.fvu_fit, b.fvu_fit)
        np.testing.assert_array_equal(a.status, b.status)

    def test_the_prediction_is_judged_as_the_table_judges_it(self, engine, tmp_path):
        """The worker's answer among its candidates gets the price and the FVU the table gives the record."""
        worker = tmp_path / "w.py"
        worker.write_text("def fit(x, y, *, x_val, variables, meta, options, state):\n"
                          "    return {'expression': 'sqrt(v1**2)*2.0 - 0.5*v2 + 1.25'}\n")
        adapter = SubprocessAdapter(worker=str(worker), simplipy_engine=engine)
        adapter.prepare()
        try:
            sample = _toy()
            record = dict(adapter.evaluate_sample(sample).to_mapping())
        finally:
            adapter.close()
        j = _judge(engine, record["candidate_expressions"], sample)
        assert j.prefixes[0].split() == list(engine.convert_expression(record["predicted_expression_prefix"]))
        assert j.bits[0] == mdl_bits(engine)(record["predicted_expression_prefix"])
        assert j.fvu_fit[0] == fvu(sample.y_support, record["y_pred"])
        assert j.fvu_val[0] == fvu(sample.y_validation, record["y_pred_val"])


# ---- the store ----------------------------------------------------------------------------------------------------
class TestStore:
    def test_the_store_of_a_result_file_and_of_its_shards_is_one_directory(self, tmp_path):
        assert candidate_store_path(tmp_path / "choices_000064.pkl") == tmp_path / "choices_000064.candidates"
        assert candidate_store_path(tmp_path / "choices_000064.shard-3-of-8.pkl") == tmp_path / "choices_000064.candidates"
        assert candidate_store_path(tmp_path / "x.candidates") == tmp_path / "x.candidates"
        assert candidate_store_path(tmp_path) == tmp_path

    def test_a_problem_round_trips_with_its_strings_its_judgement_and_its_pick(self, engine, tmp_path):
        expressions = ["2.0*v1", EXACT, "erf(v1)", "v1*π"]
        j = _judge(engine, expressions)
        j.fvu_fit[0] = 1e-50                       # float64 columns: a tiny FVU must not read as a perfect fit
        log = CandidateLog(tmp_path / "s.candidates", "choices_000001", judge={"fingerprint": "abc"}, run_meta={"ladder": 1})
        log.append(7, expressions=expressions, judgement=j, notation="infix", variables=["v1", "v2"], prediction=1,
                   meta={"benchmark_eq_id": "toy-7"})
        log.close()
        manifest = read_manifests(tmp_path / "s.candidates")["choices_000001"]
        assert manifest["kind"] == "candidates" and manifest["judge"] == {"fingerprint": "abc"}
        assert manifest["problems"] == 1 and manifest["run_meta"] == {"ladder": 1} and manifest["log"] == "choices_000001.log"
        (c,) = list(iter_candidates(tmp_path / "s.candidates"))
        assert c.row == 7 and len(c) == 4 and c.expressions == expressions
        assert c.prefixes[1] == j.prefixes[1].split() and c.prefixes[2] == []
        assert c.fvu_fit[0] == 1e-50 and c.fvu_fit.dtype == np.float64
        np.testing.assert_array_equal(c.bits, j.bits)
        np.testing.assert_array_equal(c.fvu_val, j.fvu_val)
        np.testing.assert_array_equal(c.status, j.status)
        assert list(c.prediction) == [False, True, False, False] and c.predicted == 1
        assert list(c.valid) == [True, True, False, False] and "erf" in c.reasons[2]
        assert c.meta["benchmark_eq_id"] == "toy-7" and c.variables == ["v1", "v2"] and c.notation == "infix"
        assert not c.current                       # judged by a judge named "abc", not this one

    def test_the_newest_record_of_a_problem_counts_and_rows_select_problems(self, engine, tmp_path):
        log = CandidateLog(tmp_path / "s.candidates", "run")
        for row, exprs in ((3, ["v1"]), (1, ["v2", "v1"]), (3, ["v1 + v2"])):
            log.append(row, expressions=exprs, judgement=_judge(engine, exprs), notation="infix", variables=["v1", "v2"])
        log.close()
        everything = load_candidates(tmp_path / "s.candidates")
        assert list(everything) == [1, 3] and everything[3].expressions == ["v1 + v2"]
        assert everything[1].predicted is None and not everything[1].prediction.any()
        assert list(load_candidates(tmp_path / "s.candidates", rows=[3])) == [3]
        assert [c.row for c in iter_candidates(tmp_path / "s.candidates")] == [1, 3]   # superseded records are skipped

    def test_a_record_a_killed_run_left_unfinished_ends_the_log_and_the_next_writer_cuts_it(self, engine, tmp_path):
        store = tmp_path / "s.candidates"
        log = CandidateLog(store, "run")
        log.append(0, expressions=["v1"], judgement=_judge(engine, ["v1"]), notation="infix", variables=["v1", "v2"])
        log.append(1, expressions=["v2"], judgement=_judge(engine, ["v2"]), notation="infix", variables=["v1", "v2"])
        log.close()
        path = store / "run.log"
        whole = path.read_bytes()
        path.write_bytes(whole[:-7])               # the last record's write stopped short
        assert list(load_candidates(store)) == [0]
        with pytest.warns(RuntimeWarning, match="unfinished record"):
            log = CandidateLog(store, "run")
        log.append(2, expressions=["v1 + v2"], judgement=_judge(engine, ["v1 + v2"]), notation="infix",
                   variables=["v1", "v2"])
        log.close()
        assert list(load_candidates(store)) == [0, 2]
        flipped = bytearray(path.read_bytes())
        flipped[-3] ^= 0xFF                        # a damaged body fails its check
        path.write_bytes(bytes(flipped))
        assert list(load_candidates(store)) == [0]

    def test_a_bad_pick_a_length_mismatch_or_an_unknown_notation_raises(self, engine, tmp_path):
        log = CandidateLog(tmp_path / "s.candidates", "run")
        j = _judge(engine, ["v1"])
        with pytest.raises(ValueError, match="prediction index"):
            log.append(0, expressions=["v1"], judgement=j, notation="infix", variables=["v1", "v2"], prediction=1)
        with pytest.raises(ValueError, match="judgements"):
            log.append(0, expressions=["v1", "v2"], judgement=j, notation="infix", variables=["v1", "v2"])
        with pytest.raises(ValueError, match="notation"):
            log.append(0, expressions=["v1"], judgement=j, notation="latex", variables=["v1", "v2"])
        log.close()


# ---- the worker protocol -------------------------------------------------------------------------------------------
class TestProtocol:
    def test_the_runner_and_srbf_hold_the_same_limits(self):
        assert runner.MAX_CANDIDATES == cand.MAX_CANDIDATES and runner.MAX_CANDIDATE_CHARS == cand.MAX_CANDIDATE_CHARS
        long = "v1+" * cand.MAX_CANDIDATE_CHARS
        sample = ["v1", None, long, 3.5]
        assert runner._limited(sample) == cand.limit_candidates(sample) == (["v1", "3.5"], 2)

    def test_the_first_candidates_are_kept_and_the_rest_counted(self, monkeypatch):
        monkeypatch.setattr(cand, "MAX_CANDIDATES", 3)
        assert cand.limit_candidates(["a", "b", "c", "d", "e"]) == (["a", "b", "c"], 2)

    def _adapter(self, tmp_path, engine, body, **kwargs):
        worker = tmp_path / "w.py"
        worker.write_text(textwrap.dedent(body))
        adapter = SubprocessAdapter(worker=str(worker), simplipy_engine=engine, **kwargs)
        adapter.prepare()
        return adapter

    def _record(self, adapter, sample=None):
        try:
            return dict(adapter.evaluate_sample(sample or _toy()).to_mapping())
        finally:
            adapter.close()

    def test_an_expression_and_its_front_arrive_with_the_expression_as_the_pick(self, tmp_path, engine):
        adapter = self._adapter(tmp_path, engine, f'''
            def fit(x, y, *, x_val, variables, meta, options, state):
                return {{"expression": "{EXACT}", "candidates": ["2.0*v1", "{EXACT}", None, "v1" * 6000]}}
        ''')
        record = self._record(adapter)
        assert record["prediction_success"] and record["predicted_expression"] == "2.0*x1 - 0.5*x2 + 1.0"
        assert record["candidate_expressions"] == ["2.0*v1", EXACT] and record["candidate_prediction"] == 1
        assert record["candidate_notation"] == "infix" and record["candidate_variables"] == ["v1", "v2"]
        assert record["candidates_dropped"] == 2       # a missing entry and an over-long one, dropped by the runner

    def test_an_expression_the_worker_did_not_list_is_added_to_its_candidates(self, tmp_path, engine):
        adapter = self._adapter(tmp_path, engine, '''
            def fit(x, y, *, x_val, variables, meta, options, state):
                return {"expression": "2.0*v1", "candidates": ["v1", "v2"]}
        ''')
        record = self._record(adapter)
        assert record["candidate_expressions"] == ["v1", "v2", "2.0*v1"] and record["candidate_prediction"] == 2

    def test_a_single_model_is_its_only_candidate(self, tmp_path, engine):
        adapter = SubprocessAdapter(worker="example", simplipy_engine=engine)
        adapter.prepare()
        record = self._record(adapter)
        assert record["candidate_expressions"] == [record["predicted_expression"].replace("x1", "v1").replace("x2", "v2")]
        assert record["candidate_prediction"] == 0 and record["candidates_dropped"] == 0

    def test_the_picked_candidate_is_the_pick_and_without_a_pick_the_candidates_stay(self, tmp_path, engine):
        body = f'''
            def fit(x, y, *, x_val, variables, meta, options, state):
                return {{"candidates": ["2.0*v1", "{EXACT}", "erf(v1)"]}}
        '''
        record = self._record(self._adapter(tmp_path, engine, body, selection={"mode": "mdl"}))
        assert record["prediction_success"] and record["selection"]["chosen"] == 1
        assert record["candidate_expressions"] == ["2.0*v1", EXACT, "erf(v1)"] and record["candidate_prediction"] == 1
        record = self._record(self._adapter(tmp_path, engine, body))
        assert not record["prediction_success"] and "no single answer" in record["error"]
        assert record["candidate_expressions"] == ["2.0*v1", EXACT, "erf(v1)"] and record["candidate_prediction"] is None

    def test_an_unreadable_answer_fails_the_record_and_stays_the_pick(self, tmp_path, engine):
        adapter = self._adapter(tmp_path, engine, '''
            def fit(x, y, *, x_val, variables, meta, options, state):
                return {"expression": "erf(v1)", "candidates": ["v1"]}
        ''')
        record = self._record(adapter)
        assert not record["prediction_success"] and record["candidate_expressions"] == ["v1", "erf(v1)"]
        assert record["candidate_prediction"] == 1

    def test_a_failed_fit_hands_nothing_over(self, tmp_path, engine):
        adapter = self._adapter(tmp_path, engine, '''
            def fit(x, y, *, x_val, variables, meta, options, state):
                return {"error": "no luck", "candidates": ["v1"]}
        ''')
        record = self._record(adapter)
        assert not record["prediction_success"] and "candidate_expressions" not in record

    def test_the_example_worker_returns_a_nested_front_with_the_full_fit_as_the_pick(self, engine):
        adapter = SubprocessAdapter(worker="example", simplipy_engine=engine, options={"nested": True})
        adapter.prepare()
        record = self._record(adapter)
        exprs = record["candidate_expressions"]
        assert len(exprs) == 3 and record["candidate_prediction"] == 2 and "v2" not in exprs[1] and "v" not in exprs[0]
        j = _judge(engine, exprs)
        assert list(j.status) == [STATUS_JUDGED] * 3 and j.fvu_fit[2] < 1e-12 < j.fvu_fit[1] <= j.fvu_fit[0]


# ---- the run's side ------------------------------------------------------------------------------------------------
class _ListSource:
    def __init__(self, problems):
        self.problems = problems

    def __iter__(self):
        yield from self.problems

    def size_hint(self):
        return len(self.problems)


class _HandingAdapter:
    """An in-process adapter that hands its candidates over in the record, with a fixed fit time."""

    def __init__(self, engine, expressions=("2.0*v1", EXACT), prediction=1, notation="infix", fields=True):
        self.engine, self.expressions, self.prediction, self.notation, self.fields = engine, list(expressions), prediction, notation, fields

    def get_simplipy_engine(self):
        return self.engine

    def evaluate_sample(self, sample):
        record = sample.clone_metadata()
        record.update(prediction_success=True, fit_time=1.0, predicted_expression_prefix=["+", "x1", "x2"])
        if self.fields:
            record.update(candidate_expressions=list(self.expressions), candidate_notation=self.notation,
                          candidate_prediction=self.prediction)
        return EvaluationResult(record)


class TestRun:
    def test_the_run_judges_and_stores_every_problem_outside_the_fit_time(self, engine, tmp_path):
        out = tmp_path / "fastsrb" / "choices_000004.pkl"
        bench = Benchmark(_ListSource([_toy(0), _toy(1)]), _HandingAdapter(engine), result_store=ResultStore(),
                          engine_name="acj-4-3", label={"ladder": 4})
        bench.run(output_path=str(out), progress=False, verbose=False, summary_interval=None)
        with open(out, "rb") as fh:
            snapshot = pickle.load(fh)
        assert not {"candidate_expressions", "candidate_prediction", "candidate_notation"} & set(snapshot)
        assert snapshot["fit_time"] == [1.0, 1.0] and snapshot["n_candidates"] == [2, 2]
        assert all(t > 0 for t in snapshot["candidates_judge_time"])
        store = candidate_store_path(out)
        assert store == tmp_path / "fastsrb" / "choices_000004.candidates"
        assert sorted(p.name for p in store.iterdir()) == ["choices_000004.json", "choices_000004.log"]
        problems = load_candidates(out)
        assert list(problems) == [0, 1]
        assert problems[1].predicted == 1 and problems[1].fvu_val[1] == 0.0 and problems[1].current
        manifest = read_manifests(out)["choices_000004"]
        assert manifest["judge"]["engine"] == "acj-4-3" and manifest["judge"]["fingerprint"] == cand.candidate_judge_fingerprint()
        assert manifest["run_meta"]["ladder"] == 4 and manifest["problems"] == 2

    def test_an_adapter_that_hands_nothing_over_has_its_prediction_as_its_only_candidate(self, engine, tmp_path):
        out = tmp_path / "choices_000001.pkl"
        Benchmark(_ListSource([_toy(0)]), _HandingAdapter(engine, fields=False)).run(
            output_path=str(out), progress=False, verbose=False, summary_interval=None)
        (c,) = list(iter_candidates(out))
        assert c.expressions == ["+ x1 x2"] and c.notation == "prefix" and c.predicted == 0 and c.valid.all()

    def test_switched_off_or_without_an_output_the_candidates_leave_the_record(self, engine, tmp_path):
        out = tmp_path / "choices_000001.pkl"
        snap = Benchmark(_ListSource([_toy(0)]), _HandingAdapter(engine), candidates=False).run(
            output_path=str(out), progress=False, verbose=False, summary_interval=None)
        assert not candidate_store_path(out).exists() and "candidate_expressions" not in snap and "n_candidates" not in snap
        snap = Benchmark(_ListSource([_toy(0)]), _HandingAdapter(engine)).run(progress=False, verbose=False,
                                                                              summary_interval=None)
        assert "candidate_expressions" not in snap and "n_candidates" not in snap

    def test_a_directory_can_be_named_and_shards_share_the_store(self, engine, tmp_path):
        named = tmp_path / "elsewhere"
        Benchmark(_ListSource([_toy(0)]), _HandingAdapter(engine), candidates=str(named)).run(
            output_path=str(tmp_path / "a.pkl"), progress=False, verbose=False, summary_interval=None)
        assert list(load_candidates(named)) == [0]
        for index, seed in ((0, 2), (1, 5)):
            Benchmark(_ListSource([_toy(seed)]), _HandingAdapter(engine), shard=(index, 2)).run(
                output_path=str(tmp_path / f"choices_000008.shard-{index}-of-2.pkl"), progress=False, verbose=False,
                summary_interval=None)
        assert list(load_candidates(tmp_path / "choices_000008.pkl")) == [2, 5]
        tags = sorted(read_manifests(tmp_path / "choices_000008.pkl"))
        assert tags == ["choices_000008.shard-0-of-2", "choices_000008.shard-1-of-2"]

    def test_a_problem_fitted_again_on_resume_replaces_its_candidates(self, engine, tmp_path):
        out = tmp_path / "choices_000001.pkl"
        Benchmark(_ListSource([_toy(0)]), _HandingAdapter(engine, expressions=["v1"], prediction=0)).run(
            output_path=str(out), progress=False, verbose=False, summary_interval=None)
        Benchmark(_ListSource([_toy(0)]), _HandingAdapter(engine, expressions=["v2", "v1 + v2"], prediction=1)).run(
            output_path=str(out), progress=False, verbose=False, summary_interval=None)
        (c,) = load_candidates(out).values()
        assert c.expressions == ["v2", "v1 + v2"] and c.predicted == 1

    def test_a_store_that_fails_keeps_the_row_and_says_why(self, engine, tmp_path, monkeypatch):
        def broken(self, sample, record):
            raise OSError("disk full")
        monkeypatch.setattr(CandidateRecorder, "record", broken)
        with pytest.warns(RuntimeWarning, match="not stored"):
            snap = Benchmark(_ListSource([_toy(0)]), _HandingAdapter(engine)).run(
                output_path=str(tmp_path / "c.pkl"), progress=False, verbose=False, summary_interval=None)
        assert snap["prediction_success"] == [True] and snap["candidates_error"] == ["OSError: disk full"]
        assert "candidate_expressions" not in snap

    def test_a_record_without_a_row_index_is_not_stored(self, engine, tmp_path):
        recorder = CandidateRecorder(tmp_path / "s.candidates", engine=engine)
        record = {"candidate_expressions": ["v1"], "variables": ["v1", "v2"]}
        with pytest.warns(RuntimeWarning, match="eval_row_index"):
            assert recorder.record(_toy(), record) == 0
        recorder.close()
        assert "candidate_expressions" not in record and not load_candidates(tmp_path / "s.candidates")

    def test_the_config_key_is_true_false_or_a_directory(self):
        from srbf.benchmark import check_candidates_setting
        assert check_candidates_setting(True) is True and check_candidates_setting("d") == "d"
        with pytest.raises(ValueError, match="runner.candidates"):
            check_candidates_setting(3)

    def test_an_example_worker_run_end_to_end(self, engine, tmp_path):
        adapter = SubprocessAdapter(worker="example", simplipy_engine=engine, options={"nested": True})
        out = tmp_path / "fastsrb" / "choices_000001.pkl"
        Benchmark(_ListSource([_toy(0), _toy(1), _toy(2)]), adapter).run(
            output_path=str(out), progress=False, verbose=False, summary_interval=None)
        problems = load_candidates(out)
        assert list(problems) == [0, 1, 2]
        for c in problems.values():
            assert len(c) == 3 and c.predicted == 2 and c.valid.all() and c.fvu_val[2] < 1e-12
            assert c.bits[0] < c.bits[1] < c.bits[2]

    def test_a_stored_run_is_judged_again_with_another_price(self, engine, tmp_path, monkeypatch):
        out = tmp_path / "choices_000002.pkl"
        Benchmark(_ListSource([_toy(0), _toy(1)]), _HandingAdapter(engine)).run(
            output_path=str(out), progress=False, verbose=False, summary_interval=None)
        before = load_candidates(out)
        monkeypatch.setattr(cand, "mdl_bits", lambda e: (lambda prefix: 42.0))
        monkeypatch.setattr(cand, "_FINGERPRINT", "another-judge")
        assert rejudge_candidates(out, engine=engine, engine_name="acj-4-3") == 2
        after = load_candidates(out)
        for row in (0, 1):
            assert (after[row].bits == 42.0).all() and after[row].expressions == before[row].expressions
            np.testing.assert_array_equal(after[row].fvu_val, before[row].fvu_val)
            assert after[row].predicted == before[row].predicted and after[row].header["judge"] == "another-judge"
            assert after[row].current and after[row].meta["benchmark_eq_id"] == before[row].meta["benchmark_eq_id"]
        assert read_manifests(out)["choices_000002"]["judge"]["fingerprint"] == "another-judge"
        assert len(list(cand._scan(candidate_store_path(out) / "choices_000002.log"))) == 2   # rewritten, not appended

    def test_judging_one_shard_again_leaves_the_other_shard_alone(self, engine, tmp_path, monkeypatch):
        for index, seed in ((0, 2), (1, 5)):
            Benchmark(_ListSource([_toy(seed)]), _HandingAdapter(engine), shard=(index, 2)).run(
                output_path=str(tmp_path / f"choices_000008.shard-{index}-of-2.pkl"), progress=False, verbose=False,
                summary_interval=None)
        monkeypatch.setattr(cand, "mdl_bits", lambda e: (lambda prefix: 42.0))
        assert rejudge_candidates(tmp_path / "choices_000008.shard-1-of-2.pkl", engine=engine) == 1
        after = load_candidates(tmp_path / "choices_000008.pkl")
        assert (after[5].bits == 42.0).all() and not (after[2].bits == 42.0).any()


# ---- the Flash-ANSR adapter ----------------------------------------------------------------------------------------
def test_flash_ansr_hands_over_every_fitted_candidate(engine, monkeypatch):
    pytest.importorskip("flash_ansr", reason="the Flash-ANSR adapter needs flash-ansr")
    from flash_ansr.inference import Candidate, FitResult
    from flash_ansr.scoring import resolve_ranking
    from flash_ansr.utils.generation import SoftmaxSamplingConfig
    from srbf.model_adapters import FlashANSRAdapter

    def candidate(prefix, rank):
        return Candidate(raw_beam=[rank], expression=list(prefix), slots=[], expression_prefix=list(prefix),
                         expression_infix=" ".join(prefix), skeleton_prefix=list(prefix), constants=[],
                         constants_emitted=None, log_prob=-1.0, score=float(rank), fvu=0.0, n_nodes=len(prefix), mu=None,
                         mdl=None, constant_count=0, pruned_variant=False, pareto_rank=-1, rank=rank)

    fitted = [candidate(["+", "-", "*", "2.0", "x1", "*", "0.5", "x2", "1.0"], 0), candidate(["*", "2.0", "x1"], 1)]

    class FakeModel:
        numpy_errors = "ignore"
        generation_config = SoftmaxSamplingConfig()
        simplipy_engine = engine

        def fit(self, X, y, **kwargs):
            return FitResult(candidates=list(fitted), ledger=None, generation_time=0.1, refinement_time=0.2,
                             ranking=resolve_ranking("mdl"), n_variables=2)

    monkeypatch.setattr(FitResult, "predict", lambda self, X, rank=0: np.zeros((np.asarray(X).shape[0], 1)))
    adapter = FlashANSRAdapter(FakeModel(), device="cpu", complexity="none")
    monkeypatch.setattr(adapter, "ranking_config", lambda: {"mode": "mdl"})
    sample = _toy()
    record = dict(adapter.evaluate_sample(sample).to_mapping())
    assert record["candidate_expressions"] == [" ".join(c.expression_prefix) for c in fitted]
    assert record["candidate_notation"] == "prefix" and record["candidate_variables"] == ["x1", "x2"]
    assert record["candidate_prediction"] == 0
    j = judge_candidates(record["candidate_expressions"], engine=engine, variables=record["candidate_variables"],
                         x_fit=sample.x_support, y_fit=sample.y_support, x_val=sample.x_validation,
                         y_val=sample.y_validation, notation="prefix")
    assert j.fvu_val[0] == 0.0 and j.fvu_val[1] > 0 and list(j.status) == [STATUS_JUDGED] * 2


def test_the_judge_fingerprint_names_the_candidate_judge_and_not_the_table():
    """Changing how candidates are judged does not re-judge the table, and the table's judge does not include them."""
    import srbf
    from srbf.table import judge_dependencies
    root = Path(srbf.__file__).resolve().parent
    table_sources = {p.name for p in judge_dependencies(root)[0]}
    candidate_sources = {p.name for p in judge_dependencies(root, entry="srbf.candidates")[0]}
    assert "candidates.py" not in table_sources and {"candidates.py", "table.py", "numeric.py"} <= candidate_sources
    assert len(cand.candidate_judge_fingerprint()) == 16
    json.dumps(cand.judge_identity("acj-4-3"))
