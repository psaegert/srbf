"""Serial Benchmark.run: placeholder-on-exception, save_every + resume, progress/remaining accounting.

Ported from the old test_engine.py (the serial EvaluationEngine.run behaviour the overlap engine also
had to match); the overlap engine is removed in 0.5.0, so this is the single driver under test.
"""
import pickle

import numpy as np

from srbf.benchmark import Benchmark
from srbf.eval.core import EvaluationSample, EvaluationResult
from srbf.eval.result_store import ResultStore


class _ListSource:
    def __init__(self, problems, *, skip: int = 0):
        self._problems = problems
        self._skip = max(0, skip)

    def __iter__(self):
        yield from self._problems[self._skip:]

    def size_hint(self):
        return max(0, len(self._problems) - self._skip)


class _FlakyAdapter:
    def __init__(self):
        self.calls = 0

    def evaluate_sample(self, problem):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("boom")
        record = problem.clone_metadata()
        record["prediction_success"] = True
        return EvaluationResult(record)


class _PassthroughAdapter:
    def evaluate_sample(self, problem):
        record = problem.clone_metadata()
        record["prediction_success"] = True
        return EvaluationResult(record)


def _dummy(idx: int) -> EvaluationSample:
    data = np.zeros((2, 1), dtype=np.float32)
    return EvaluationSample(x_support=data, y_support=data, x_validation=data, y_validation=data,
                            metadata={"sample_id": idx})


def test_benchmark_records_placeholder_on_adapter_exception(capsys):
    source = _ListSource([_dummy(0), _dummy(1)])
    bench = Benchmark(source=source, model_adapter=_FlakyAdapter(), result_store=ResultStore())
    snapshot = bench.run(progress=False, verbose=False, summary_interval=1)
    output = capsys.readouterr().out

    assert len(snapshot["placeholder"]) == 2
    assert snapshot["placeholder"][0] is True
    assert snapshot["prediction_success"][0] is False
    assert snapshot["placeholder_reason"][0] == "adapter_exception"
    assert snapshot["sample_id"] == [0, 1]
    assert snapshot["prediction_success"][1] is True
    assert "Placeholder #1 recorded" in output
    assert "Final summary" in output


def test_benchmark_can_resume_from_saved_results(tmp_path):
    problems = [_dummy(idx) for idx in range(3)]
    output_path = tmp_path / "results.pkl"

    first_store = ResultStore()
    first = Benchmark(source=_ListSource(problems), model_adapter=_FlakyAdapter(), result_store=first_store)
    snapshot = first.run(limit=2, save_every=1, output_path=str(output_path),
                         progress=False, verbose=False, summary_interval=1)

    assert output_path.exists()
    assert snapshot["sample_id"] == [0, 1]
    assert snapshot["placeholder"] == [True, False]
    assert snapshot["placeholder_reason"][0] == "adapter_exception"

    with output_path.open("rb") as handle:
        payload = pickle.load(handle)
    resumed_store = ResultStore(payload)
    assert resumed_store.size == 2

    resumed_adapter = _FlakyAdapter()
    resumed_adapter.calls = 1  # skip the initial failure on resume
    resumed = Benchmark(source=_ListSource(problems, skip=resumed_store.size),
                        model_adapter=resumed_adapter, result_store=resumed_store)
    final_snapshot = resumed.run(progress=False, verbose=False, summary_interval=1)

    assert final_snapshot["sample_id"] == [0, 1, 2]
    assert final_snapshot["placeholder"] == [True, False, False]
    assert final_snapshot["prediction_success"] == [False, True, True]
    assert final_snapshot["placeholder_reason"][0] == "adapter_exception"


def test_progress_tracker_reports_remaining_on_resume(capsys):
    store = ResultStore({
        "sample_id": [0, 1, 2],
        "placeholder": [False, False, False],
        "placeholder_reason": [None, None, None],
        "prediction_success": [True, True, True],
    })
    source = _ListSource([_dummy(idx) for idx in range(3, 5)])
    bench = Benchmark(source=source, model_adapter=_PassthroughAdapter(), result_store=store)
    bench.run(limit=2, progress=False, verbose=False, summary_interval=1)

    output = capsys.readouterr().out
    assert "[Benchmark] Starting state: total=3; valid=3; placeholders=0; remaining=2" in output
    assert "Final summary" in output


def test_benchmark_preserves_problem_order(capsys):
    # Serial commit order == source order (the property the overlap engine had to preserve; now trivial).
    source = _ListSource([_dummy(idx) for idx in range(5)])
    bench = Benchmark(source=source, model_adapter=_PassthroughAdapter(), result_store=ResultStore())
    snapshot = bench.run(progress=False, verbose=False, summary_interval=0)
    assert snapshot["sample_id"] == [0, 1, 2, 3, 4]
