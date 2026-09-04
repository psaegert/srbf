"""Tests for the save-all candidate store + ledger join (STANDARD_EVAL.md item 5)."""
import json
import types
import warnings as _warnings

import numpy as np
import pytest

from srbf.candidate_store import CandidateStoreReader, CandidateStoreWriter
from srbf.model_adapters import FlashANSRAdapter
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
    np.testing.assert_array_equal(CandidateStoreReader.candidate_tokens(block, 0), np.array(A, np.uint16))
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


def test_byte_alphabet_vocabulary_is_representable(tmp_path):
    """The f64 + byte-token migration takes the vocabulary 95 -> 335.

    The writer used to bound vocab_size at 256 (uint8 tokens) and raise. That raise is caught by
    FlashANSRAdapter._capture_ledger's `except Exception: warnings.warn(...)`, so crossing the
    bound produced an EMPTY candidate store for a whole campaign while every eval row reported
    success. Token ids above 255 must round-trip.
    """
    w = CandidateStoreWriter(tmp_path, vocab_size=335)
    high = [334, 256, 255, 0]
    w.write_problem(0, [high], [0.5], [-1.0])
    w.close()
    block = next(iter(CandidateStoreReader(tmp_path)))
    got = CandidateStoreReader.candidate_tokens(block, 0)
    assert got.dtype == np.uint16
    np.testing.assert_array_equal(got, np.array(high, np.uint16))


def test_tiny_fvu_survives_the_store(tmp_path):
    """float32 flushed a genuine FVU of ~1e-50 to 0.0, which reads back as a PERFECT recovery."""
    tiny = 1e-50
    w = CandidateStoreWriter(tmp_path, vocab_size=335)
    w.write_problem(0, [[1, 2]], [tiny], [-1.0], constants=[[1.18885916993963e-52]])
    w.close()
    block = next(iter(CandidateStoreReader(tmp_path)))
    assert block["fvu"].dtype == np.float64
    assert block["fvu"][0] == tiny and block["fvu"][0] != 0.0
    assert block["const_vals"][0] == 1.18885916993963e-52


def test_ranking_and_validation_columns_round_trip(tmp_path):
    """The offline re-sort substrate: per-candidate ranking columns from flash-ansr's ledger plus the
    adapter-computed validation FVU / recovery flags, and the run's ranking in the manifest."""
    w = CandidateStoreWriter(tmp_path, vocab_size=335, run_meta={"ranking": {"mode": "mdl", "mdl_strength": 4.5e-3}})
    w.write_problem(
        3, [[1, 2, 3], [4, 5]], [0.2, float("nan")], [-1.0, -2.0],
        valid=[1, 1], fit_status=[FIT_OK, FIT_FAILED], constants=[[7.0], []],
        n_nodes=[3, -1], n_constants=[1, -1], mdl=[123456.0, float("nan")], score=[-0.7, float("nan")],
        pareto_rank=[-1, -1], rank=[0, -1], fvu_val=[0.25, float("nan")], recovery_fit=[0, 0], recovery_val=[1, 0],
    )
    manifest = w.close()
    assert manifest["run_meta"]["ranking"]["mode"] == "mdl"
    block = next(iter(CandidateStoreReader(tmp_path)))
    np.testing.assert_array_equal(block["n_nodes"], [3, -1])
    np.testing.assert_array_equal(block["rank"], [0, -1])
    assert block["mdl"].dtype == np.float64 and block["mdl"][0] == 123456.0 and np.isnan(block["mdl"][1])
    assert block["fvu_val"][0] == 0.25 and np.isnan(block["fvu_val"][1])
    np.testing.assert_array_equal(block["recovery_val"], [1, 0])
    with pytest.raises(ValueError, match="rank length"):
        w.write_problem(4, [[1]], [0.1], [-1.0], rank=[0, 1])


def test_adapter_capture_computes_validation_metrics_from_every_candidate(tmp_path):
    """top_k='all' gives every FIT_OK candidate its predictions; the adapter turns them into per-candidate
    validation FVU and recovery with the shared srbf metrics, aligned through result_index."""
    from srbf.metrics.numeric import fvu, is_perfect_fit
    adapter = FlashANSRAdapter(_mock_model(), candidate_store_dir=str(tmp_path))
    y_sup = np.array([1.0, 2.0, 3.0]); y_val = np.array([4.0, 5.0])
    exact = types.SimpleNamespace(y_pred=y_sup.copy(), y_pred_val=y_val.copy())
    off = types.SimpleNamespace(y_pred=y_sup + 0.5, y_pred_val=y_val + 1.0)
    ledger = CandidateLedger(
        token_lists=[[1, 2], [3, 4], [5]], fvu=[0.0, 0.1, float("nan")], log_prob=[-1.0, -2.0, -3.0],
        valid=[1, 1, 0], fit_status=[FIT_OK, FIT_OK, INVALID], constants=[[], [], []],
        n_nodes=[2, 2, -1], n_constants=[0, 0, -1], mdl=[1.0, 2.0, float("nan")], score=[-9.0, -1.0, float("nan")],
        pareto_rank=[-1, -1, -1], rank=[0, 1, -1], result_index=[0, 1, -1])
    result = types.SimpleNamespace(ledger=ledger, candidates=[exact, off])
    sample = types.SimpleNamespace(y_support=y_sup, y_validation=y_val)
    adapter._capture_ledger({"eval_row_index": 9}, result, sample)
    block = next(iter(CandidateStoreReader(tmp_path)))
    np.testing.assert_array_equal(block["recovery_fit"], [1, 0, 0])
    np.testing.assert_array_equal(block["recovery_val"], [1, 0, 0])
    assert block["fvu_val"][0] == fvu(y_val, y_val) == 0.0
    assert block["fvu_val"][1] == fvu(y_val, y_val + 1.0)
    assert np.isnan(block["fvu_val"][2])
    assert is_perfect_fit(y_val, y_val)
    manifest = json.loads((tmp_path / "manifest.json").read_text()) if (tmp_path / "manifest.json").exists() else adapter._candidate_store.close()
    assert "mdl_dialect" in manifest["run_meta"]


def test_manifest_carries_run_meta_from_the_first_problem(tmp_path):
    """A run killed before the 64th problem must still leave the ranking / dialect on disk."""
    w = CandidateStoreWriter(tmp_path, vocab_size=335, run_meta={"ranking": {"mode": "mdl"}})
    w.write_problem(0, [[1, 2]], [0.1], [-1.0])
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["run_meta"] == {"ranking": {"mode": "mdl"}}
    assert manifest["n_problems"] == 1
