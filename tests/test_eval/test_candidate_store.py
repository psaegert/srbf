"""Tests for the save-all candidate store + ledger join (STANDARD_EVAL.md item 5)."""
import json
import types
import warnings as _warnings

import numpy as np
import pytest

from srbf.eval.candidate_store import CandidateStoreReader, CandidateStoreWriter
from srbf.eval.model_adapters import FlashANSRAdapter
# The candidate-ledger JOIN (gen pool U refined, classified) now lives in flash-ansr (infer() builds
# result.ledger); srbf only persists it. FIT_* + CandidateLedger come from there.
from flash_ansr.inference import CandidateLedger, FIT_FAILED, FIT_OK, INVALID


def test_ledger_round_trips_through_writer_reader(tmp_path):
    # A fitted (FIT_OK) candidate + a valid-but-unfit skeleton (FIT_FAILED, NaN fvu): persist the full
    # set and read it back. (The JOIN that classifies candidates is flash-ansr's job + tested there;
    # srbf only round-trips the resulting columns through the store.)
    A, B = [1, 2, 3], [4, 5]
    w = CandidateStoreWriter(tmp_path, vocab_size=83)
    w.write_problem(
        0, [A, B], [0.2, float("nan")], [-1.0, -2.0],
        valid=[1, 1], fit_status=[FIT_OK, FIT_FAILED], constants=[[7.0], []],
    )
    w.close()
    block = next(iter(CandidateStoreReader(tmp_path)))
    np.testing.assert_array_equal(CandidateStoreReader.candidate_tokens(block, 0), np.array(A, np.uint8))
    np.testing.assert_array_equal(block["fit_status"], [FIT_OK, FIT_FAILED])
    assert np.isnan(block["fvu"][1])


def test_writer_resume_rebuilds_index_and_skips_done(tmp_path):
    w1 = CandidateStoreWriter(tmp_path, vocab_size=83)
    w1.write_problem(0, [[1, 2]], [0.1], [-1.0])
    w1.write_problem(1, [[3, 4]], [0.2], [-2.0])
    w1.close()

    # fresh writer over the same dir (a resume): index rebuilt from disk, done problems skippable
    w2 = CandidateStoreWriter(tmp_path, vocab_size=83)
    assert w2.has_problem(0) and w2.has_problem(1) and not w2.has_problem(2)
    w2.write_problem(2, [[5, 6]], [0.3], [-3.0])
    man = w2.close()
    assert man["n_problems"] == 3                      # manifest covers ALL, not just the resume slice
    assert {p["problem_id"] for p in man["problems"]} == {0, 1, 2}


def test_reader_globs_authoritatively_over_stale_manifest(tmp_path):
    w = CandidateStoreWriter(tmp_path, vocab_size=83)
    for i in range(3):
        w.write_problem(i, [[i, i + 1]], [0.1 * i], [-1.0 * i])
    w.close()
    # corrupt the manifest to claim only problem 0 -- the reader must still find all three on disk
    (tmp_path / "manifest.json").write_text(json.dumps({"problems": [{"problem_id": 0}]}))
    assert CandidateStoreReader(tmp_path).problem_ids() == [0, 1, 2]
    assert len(list(CandidateStoreReader(tmp_path))) == 3


# --- adapter capture (mock model; no GPU) -------------------------------------------------
class _Tok:
    # __len__ must live on the type (dunder lookup skips instances), so a class not SimpleNamespace
    def extract_expression_from_beam(self, beam):
        return (list(beam), [], [])

    def decode(self, ids, special_tokens=None):
        return [str(i) for i in ids]

    def __len__(self):
        return 83


def _mock_model():
    engine = types.SimpleNamespace(is_valid=lambda toks: toks != ["3", "4"])  # [3,4] is "invalid"
    return types.SimpleNamespace(tokenizer=_Tok(), simplipy_engine=engine)


def test_adapter_capture_writes_ledger_keyed_by_row_index(tmp_path):
    # The join is now done by FlashANSR.infer() (result.ledger); the adapter just streams it to the
    # store keyed by eval_row_index. (The join classification itself is tested above + in flash-ansr.)
    adapter = FlashANSRAdapter(_mock_model(), candidate_store_dir=str(tmp_path))
    result = types.SimpleNamespace(ledger=CandidateLedger(
        token_lists=[[1, 2], [3, 4]], fvu=[0.1, float("nan")], log_prob=[-1.0, -2.0],
        valid=[1, 0], fit_status=[FIT_OK, INVALID], constants=[[2.0], []]))
    adapter._capture_ledger({"eval_row_index": 5}, result)

    assert (tmp_path / "problem_000005.npz").exists()
    block = next(iter(CandidateStoreReader(tmp_path)))
    np.testing.assert_array_equal(block["fit_status"], [FIT_OK, INVALID])
    np.testing.assert_array_equal(block["valid"], [1, 0])

    # resume: a second capture of the same row is a no-op (file already present)
    adapter._capture_ledger({"eval_row_index": 5}, result)
    assert CandidateStoreReader(tmp_path).problem_ids() == [5]


def test_adapter_capture_skips_without_row_index(tmp_path):
    adapter = FlashANSRAdapter(_mock_model(), candidate_store_dir=str(tmp_path))
    result = types.SimpleNamespace(ledger=CandidateLedger(
        token_lists=[[1, 2]], fvu=[0.1], log_prob=[-1.0], valid=[1], fit_status=[FIT_OK], constants=[[]]))
    with pytest.warns(RuntimeWarning, match="eval_row_index"):
        adapter._capture_ledger({}, result)
    assert not list(tmp_path.glob("problem_*.npz"))


def test_adapter_capture_is_best_effort_on_error(tmp_path):
    adapter = FlashANSRAdapter(_mock_model(), candidate_store_dir=str(tmp_path))
    broken = types.SimpleNamespace(ledger=types.SimpleNamespace())  # .token_lists missing -> caught
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        adapter._capture_ledger({"eval_row_index": 0}, broken)
    # no crash; nothing written
    assert not list(tmp_path.glob("problem_*.npz"))
