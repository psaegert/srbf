"""CatalogSource: bridge a symbolic_data ProblemSource (sd.Problem) -> srbf EvaluationSample.

The data layer (symbolic_data) owns all sampling; srbf's source just bridges each Problem to an
EvaluationSample with the eval metadata + a resume-stable eval_row_index. Tests are hermetic: a fake
source of hand-built sd.Problems for the bridge contract, plus a real ProblemSource fixed-mode
integration (no HF / no model).
"""
import numpy as np

from symbolic_data import Problem, ProblemSource
from srbf.core import EvaluationSample
from srbf.data_sources import CatalogSource


def _problem(skeleton, expression, variables, complexity, *, eq_id=None, n_support=8, n_val=4,
             is_placeholder=False):
    nv = max(1, len(variables))
    xs = np.arange(n_support * nv, dtype=np.float32).reshape(n_support, nv)
    ys = np.arange(n_support, dtype=np.float32).reshape(n_support, 1)
    xv = np.zeros((n_val, nv), dtype=np.float32)
    yv = np.zeros((n_val, 1), dtype=np.float32)
    return Problem(
        x_support=xs, y_support=ys, y_support_noisy=ys.copy(),
        x_validation=xv, y_validation=yv, y_validation_noisy=yv.copy(),
        skeleton=skeleton, expression=expression, constants=[], variables=variables,
        complexity=complexity, eq_id=eq_id, is_placeholder=is_placeholder,
        placeholder_reason=("forced" if is_placeholder else None),
    )


class _FakeSource:
    def __init__(self, problems):
        self._problems = list(problems)

    def __iter__(self):
        yield from self._problems

    def size_hint(self):
        return len(self._problems)


class _Tok:
    """Minimal tokenizer: no <expression> wrapping, bos/eos sentinels, id = 10 + position."""
    def __contains__(self, key):
        return key in ("<bos>", "<eos>")

    def __getitem__(self, key):
        return {"<bos>": 1, "<eos>": 2}[key]

    def encode(self, tokens, oov="unk"):
        return [10 + i for i in range(len(tokens))]

    def decode(self, ids, special_tokens=None):
        return [f"t{i}" for i in ids]


def test_bridges_problem_fields_and_stamps_row_index():
    problems = [
        _problem(["sin", "x1"], ["sin", "x1"], ["x1", "x2"], 2, eq_id="E1"),
        _problem(["+", "x1", "x2"], ["+", "x1", "x2"], ["x1", "x2"], 3, eq_id="E2"),
    ]
    src = CatalogSource(_FakeSource(problems), tokenizer=_Tok())
    samples = list(src)
    assert len(samples) == 2
    for i, (s, p) in enumerate(zip(samples, problems)):
        assert isinstance(s, EvaluationSample) and not s.is_placeholder
        assert s.n_support == 8
        assert s.metadata["eval_row_index"] == i
        assert s.metadata["skeleton"] == list(p.skeleton)
        assert s.metadata["expression"] == list(p.expression)
        assert s.metadata["variables"] == list(p.variables)
        assert s.metadata["variable_names"] == list(p.variables)
        assert s.metadata["complexity"] == p.complexity
        assert s.metadata["benchmark_eq_id"] == p.eq_id
        assert s.metadata["ground_truth_prefix"] == list(p.expression)
        # tokenizer encoding: bos + body + eos; labels = input_ids[1:]
        assert s.metadata["input_ids"][0] == 1 and s.metadata["input_ids"][-1] == 2
        np.testing.assert_array_equal(s.metadata["labels"], s.metadata["input_ids"][1:])
        np.testing.assert_array_equal(s.x_support, p.x_support)


def test_no_tokenizer_leaves_input_ids_none():
    src = CatalogSource(_FakeSource([_problem(["sin", "x1"], ["sin", "x1"], ["x1"], 2)]))
    sample = next(iter(src))
    assert sample.metadata["input_ids"] is None and sample.metadata["labels"] is None


def test_skip_and_target_size_bound_the_stream():
    problems = [_problem(["sin", "x1"], ["sin", "x1"], ["x1"], 2, eq_id=f"E{i}") for i in range(5)]
    src = CatalogSource(_FakeSource(problems), skip=2, target_size=2)
    samples = list(src)
    assert [s.metadata["eval_row_index"] for s in samples] == [2, 3]   # skip-aware, bounded
    assert src.size_hint() == 2


def test_placeholder_problem_bridges_to_placeholder_sample():
    problems = [_problem([], None, ["x1"], None, is_placeholder=True)]
    sample = next(iter(CatalogSource(_FakeSource(problems))))
    assert sample.is_placeholder is True
    assert sample.metadata["placeholder"] is True
    assert sample.metadata["prediction_success"] is False
    assert sample.metadata["eval_row_index"] == 0


def test_resume_state_dict_round_trip():
    problems = [_problem(["sin", "x1"], ["sin", "x1"], ["x1"], 2) for _ in range(4)]
    src = CatalogSource(_FakeSource(problems))
    list(src)  # produce all 4
    state = src.state_dict()["state"]
    assert state["row_index"] == 4
    resumed = CatalogSource(_FakeSource(problems), resume_state=state)
    assert resumed._skip == 4 and list(resumed) == []   # nothing left after resume


def test_integration_real_problemsource_fixed_mode():
    # End-to-end with a REAL sd.ProblemSource (fixed/inline mode -- no HF, no model): the bridge must
    # consume what ProblemSource actually yields.
    problems = [_problem(["sin", "x1"], ["sin", "x1"], ["x1"], 2, eq_id="E1")]
    ps = ProblemSource({"problems": [p.to_dict() for p in problems]})
    src = CatalogSource(ps)
    out = list(src)
    assert len(out) == 1
    assert out[0].metadata["expression"] == ["sin", "x1"]
    assert out[0].metadata["eval_row_index"] == 0
