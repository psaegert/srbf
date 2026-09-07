"""Sharding a run (``srbf run --shard K/N`` + ``srbf merge``): the catalog source's interleaved slice,
resume inside a shard, the shard's output name and meta stamp, and the merge back into one file."""
import pickle

import numpy as np
import pytest
from symbolic_data import Problem

from srbf.benchmark import Benchmark
from srbf.core import EvaluationResult, EvaluationSample
from srbf.data_sources import CatalogSource
from srbf.shards import merge_shards, parse_shard, shard_output_path, shard_share
from srbf.store import ResultStore


def _problem(i: int) -> Problem:
    xs = np.full((4, 1), float(i))
    ys = np.full((4, 1), float(i))
    return Problem(x_support=xs, y_support=ys, y_support_noisy=ys.copy(), x_validation=xs[:2], y_validation=ys[:2],
                   y_validation_noisy=ys[:2].copy(), skeleton=["x1"], expression=["x1"], constants=[], variables=["x1"],
                   complexity=1, eq_id=f"E{i}", is_placeholder=False, placeholder_reason=None)


class _FakeSource:
    def __init__(self, n: int) -> None:
        self._problems = [_problem(i) for i in range(n)]

    def __iter__(self):
        yield from self._problems

    def size_hint(self):
        return len(self._problems)


def _rows(source: CatalogSource) -> list[int]:
    return [int(sample.metadata["eval_row_index"]) for sample in source]


class TestShardArithmetic:
    def test_shares_partition_the_total(self) -> None:
        for total in (0, 1, 7, 100, 5301):
            for count in (1, 2, 3, 8, 16):
                assert sum(shard_share(total, k, count) for k in range(count)) == total
        assert shard_share(10, 0, 3) == 4 and shard_share(10, 2, 3) == 3

    def test_parse_shard(self) -> None:
        assert parse_shard("0/4") == (0, 4) and parse_shard("3/4") == (3, 4)
        for bad in ("4/4", "a/b", "1", "-1/2", "1/0"):
            with pytest.raises(ValueError):
                parse_shard(bad)

    def test_output_name(self) -> None:
        assert shard_output_path("r/fastsrb/choices_016384.pkl", 2, 8) == "r/fastsrb/choices_016384.shard-2-of-8.pkl"


class TestShardedSource:
    def test_shards_partition_the_catalog(self) -> None:
        seen = []
        for k in range(3):
            source = CatalogSource(_FakeSource(10), shard=(k, 3))
            rows = _rows(source)
            assert rows == list(range(k, 10, 3))
            assert source.size_hint() == len(rows)
            seen.extend(rows)
        assert sorted(seen) == list(range(10))

    def test_skip_and_target_count_the_shards_own_rows(self) -> None:
        # shard 1 of 3 owns rows 1, 4, 7; resume past one of them, then cap at one more.
        source = CatalogSource(_FakeSource(10), shard=(1, 3), skip=1)
        assert source.size_hint() == 2 and _rows(source) == [4, 7]
        source = CatalogSource(_FakeSource(10), shard=(1, 3), skip=1, target_size=1)
        assert _rows(source) == [4]

    def test_unsharded_behaviour_is_unchanged(self) -> None:
        source = CatalogSource(_FakeSource(5), skip=2)
        assert _rows(source) == [2, 3, 4] and source.size_hint() == 3

    def test_rejects_bad_shards(self) -> None:
        with pytest.raises(ValueError):
            CatalogSource(_FakeSource(3), shard=(3, 3))


class _ListSource:
    def __init__(self, samples):
        self._samples = list(samples)

    def __iter__(self):
        yield from self._samples

    def size_hint(self):
        return len(self._samples)


class _Adapter:
    def evaluate_sample(self, sample):
        record = sample.clone_metadata()
        record["prediction_success"] = True
        record["value"] = float(sample.metadata["eval_row_index"]) * 10
        return EvaluationResult(record)


def _sample(i: int) -> EvaluationSample:
    x = np.zeros((2, 1))
    y = np.zeros((2, 1))
    return EvaluationSample(x_support=x, y_support=y, x_validation=x, y_validation=y, metadata={"eval_row_index": i, "eq_id": f"E{i}"})


def _write_shard(path, rows, index, count, extra_meta=None):
    store = ResultStore({"eval_row_index": rows, "value": [r * 10.0 for r in rows], "prediction_success": [True] * len(rows)})
    store.save(path, meta={"config_provenance": "harness_tuned", "shard": {"index": index, "count": count}, **(extra_meta or {})})


class TestMetaAndMerge:
    def test_run_stamps_the_shard_into_meta(self, tmp_path) -> None:
        out = tmp_path / "choices_000128.shard-1-of-2.pkl"
        bench = Benchmark(_ListSource([_sample(1), _sample(3)]), _Adapter(), shard=(1, 2))
        bench.run(output_path=str(out), verbose=False, progress=False, meta={"config_provenance": "harness_tuned"})
        with out.open("rb") as f:
            payload = pickle.load(f)
        assert payload["__meta__"]["shard"] == {"index": 1, "count": 2}
        assert payload["eval_row_index"] == [1, 3]

    def test_merge_orders_rows_and_records_provenance(self, tmp_path) -> None:
        a = tmp_path / "choices_000128.shard-0-of-2.pkl"
        b = tmp_path / "choices_000128.shard-1-of-2.pkl"
        _write_shard(a, [0, 2, 4], 0, 2)
        _write_shard(b, [1, 3], 1, 2)
        out = tmp_path / "choices_000128.pkl"
        summary = merge_shards([str(b), str(a)], str(out))
        assert summary["rows"] == 5 and summary["shards"] == [0, 1] and summary["missing"] == []
        with out.open("rb") as f:
            payload = pickle.load(f)
        assert payload["eval_row_index"] == [0, 1, 2, 3, 4]
        assert payload["value"] == [0.0, 10.0, 20.0, 30.0, 40.0]
        assert "shard" not in payload["__meta__"]
        assert payload["__meta__"]["shards"] == {"count": 2, "merged": [0, 1], "partial": False, "sources": [a.name, b.name]}
        assert payload["__meta__"]["config_provenance"] == "harness_tuned"

    def test_merge_refuses_overlap_mismatch_and_silent_gaps(self, tmp_path) -> None:
        a, b, c, d = (tmp_path / f"s{i}.pkl" for i in range(4))
        _write_shard(a, [0, 2], 0, 2)
        _write_shard(b, [2, 3], 1, 2)
        _write_shard(c, [1], 1, 3)
        _write_shard(d, [5], 0, 2)
        with pytest.raises(ValueError, match="overlap"):
            merge_shards([str(a), str(b)], str(tmp_path / "o1.pkl"))
        with pytest.raises(ValueError, match="count"):
            merge_shards([str(a), str(c)], str(tmp_path / "o2.pkl"))
        with pytest.raises(ValueError, match="missing shards"):
            merge_shards([str(a)], str(tmp_path / "o3.pkl"))
        summary = merge_shards([str(a)], str(tmp_path / "o4.pkl"), allow_partial=True)
        assert summary["missing"] == [1]
        with (tmp_path / "o4.pkl").open("rb") as f:
            assert pickle.load(f)["__meta__"]["shards"]["partial"] is True
        with pytest.raises(ValueError, match="duplicate"):
            merge_shards([str(a), str(d)], str(tmp_path / "o5.pkl"))

    def test_merge_needs_shard_meta(self, tmp_path) -> None:
        plain = tmp_path / "plain.pkl"
        ResultStore({"eval_row_index": [0]}).save(plain, meta={"config_provenance": "harness_tuned"})
        with pytest.raises(ValueError, match="shard"):
            merge_shards([str(plain)], str(tmp_path / "o.pkl"))
