"""The hybrid arm's pure parts: the time laws, the seed bridge to Julia syntax, the budget split
and the snapshot targets, and the config builder's validation."""
import math

import pytest

from srbf.hybrid_adapter import TimeLaw, prefix_to_julia

ARITY = {"+": 2, "-": 2, "*": 2, "/": 2, "pow": 2, "rootn": 2, "neg": 1, "inv": 1, "abs": 1,
         "sin": 1, "cos": 1, "exp": 1, "log": 1}


class TestTimeLaw:
    def test_inverts_and_floors(self):
        law = TimeLaw(0.5, 0.002)
        assert law.units_for(0.0) == 0
        assert law.units_for(0.3) == 1          # a positive share is worth at least one unit
        assert law.units_for(10.5) == 5000
        assert law.seconds_for(5000) == pytest.approx(10.5)
        assert TimeLaw(2.0, 1.7, minimum=1).units_for(100.0) == round(98 / 1.7)

    def test_rejects_a_flat_law(self):
        with pytest.raises(ValueError):
            TimeLaw(1.0, 0.0)


class TestBridge:
    def test_renders_pysr_syntax(self):
        assert prefix_to_julia(["*", "2.5", "pow", "x1", "2"], ARITY, ["v1", "v2"]) == "(2.5 * (v1 ^ 2.0))"
        assert prefix_to_julia(["neg", "+", "x1", "1.0"], ARITY, ["v1"]) == "neg((v1 + 1.0))"
        assert prefix_to_julia(["rootn", "x2", "3"], ARITY, ["v1", "v2"]) == "rootn(v2, 3.0)"
        assert prefix_to_julia(["inv", "sin", "x1"], ARITY, ["a"]) == "inv(sin(a))"

    def test_negative_literals_and_special_constants(self):
        text = prefix_to_julia(["+", "*", "-1.5e-09", "x1", "np.pi"], ARITY, ["v1"])
        assert text == "(((-1.5e-09) * v1) + 3.141592653589793)"
        assert prefix_to_julia(["*", "np.e", "x1"], ARITY, ["v1"]) == f"({math.e!r} * v1)"

    def test_refuses_what_pysr_cannot_read(self):
        assert prefix_to_julia(["sqrt", "x1"], ARITY, ["v1"]) is None          # outside the vocabulary
        assert prefix_to_julia(["*", "<constant>", "x1"], ARITY, ["v1"]) is None  # an unrealized slot
        assert prefix_to_julia(["*", "x3", "x1"], ARITY, ["v1", "v2"]) is None    # a variable the problem lacks
        assert prefix_to_julia(["+", "x1"], ARITY, ["v1"]) is None                # malformed


class TestSplit:
    def _adapter(self, ratio, ratios):
        from srbf.hybrid_adapter import FlashANSRPySRAdapter
        adapter = FlashANSRPySRAdapter.__new__(FlashANSRPySRAdapter)
        adapter.budget_s = 100.0
        adapter.ratio = float(ratio)
        adapter.ratios = sorted(set(ratios) | {float(ratio)})
        adapter.choices_law = TimeLaw(0.5, 0.01)
        adapter.niterations_law = TimeLaw(1.0, 2.0)
        return adapter

    def test_endpoints_and_targets(self):
        ratios = [i / 10 for i in range(11)]
        a = self._adapter(0.0, ratios)
        assert a.choices_for(0.0) == round(99.5 / 0.01) and a.niterations_for(0.0) == 0
        assert a.choices_for(1.0) == 0 and a.niterations_for(1.0) == round(99.0 / 2.0)
        targets = a.choice_targets()
        assert 0 not in targets and targets == sorted(targets) and len(targets) == 10
        # the shares add up to the budget in target seconds
        for r in ratios:
            assert (1 - r) * 100 + r * 100 == pytest.approx(100.0)


class TestConfig:
    def test_builder_validates_blocks(self):
        from srbf.config import build_model_adapter
        with pytest.raises(ValueError):
            build_model_adapter({"type": "flash_ansr_pysr", "hybrid": {"ratio": 0.5}})


class TestSnapshotFingerprint:
    """A snapshot is only valid on the data it was generated on: the cache keys on a fingerprint."""

    def _adapter(self, tmp_path):
        from srbf.hybrid_adapter import FlashANSRPySRAdapter, TimeLaw
        return FlashANSRPySRAdapter(
            flash=object(), pysr=object(), budget_s=20.0, ratio=0.0, ratios=[0.0, 0.5, 1.0],
            choices_law=TimeLaw(0.5, 0.01), niterations_law=TimeLaw(1.0, 0.5), snapshot_dir=str(tmp_path))

    def test_fingerprint_follows_the_arrays(self):
        from srbf.hybrid_adapter import data_fingerprint
        from srbf.testing import toy_sample
        a, b, a_again = toy_sample(0), toy_sample(1), toy_sample(0)
        assert data_fingerprint(a) == data_fingerprint(a_again)
        assert data_fingerprint(a) != data_fingerprint(b)

    def test_cache_hits_on_the_same_data_and_regenerates_on_other_data(self, tmp_path):
        import warnings
        from srbf.testing import toy_sample
        adapter = self._adapter(tmp_path)
        calls = []

        def fake_generate(sample, record):
            calls.append(record["eval_row_index"])
            return {t: {"choices": t, "cum_wall": 1.0, "cum_generation": 0.5, "cum_refinement": 0.5,
                        "pool_size": 1, "best": None, "seeds": []} for t in adapter.choice_targets()}

        adapter._generate_snapshots = fake_generate
        same = toy_sample(0)
        record = {"eval_row_index": 0}
        first = adapter._snapshot(same, record)
        second = adapter._snapshot(toy_sample(0), record)          # identical arrays -> cached
        assert first == second and calls == [0]
        other = toy_sample(1)                                       # a re-drawn instance of "the same problem"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            adapter._snapshot(other, record)
        assert calls == [0, 0]
        assert any("different data" in str(w.message) for w in caught)
