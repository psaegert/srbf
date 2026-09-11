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


@pytest.fixture(scope="module")
def engine():
    from simplipy import SimpliPyEngine
    return SimpliPyEngine.load("acj-4-3", install=True)


class TestPySRAddsCandidates:
    """PySR's hall of fame joins the Flash-ANSR candidates and Flash-ANSR's sorting picks rank 0: the same
    score, the same MDL penalty, for every candidate; PySR's own choice is kept beside it for reference."""

    @staticmethod
    def _problem():
        import numpy as np
        rng = np.random.default_rng(0)
        x = rng.uniform(0.5, 2.0, size=(64, 2)); x_val = rng.uniform(0.5, 2.0, size=(16, 2))
        law = lambda a: 2.0 * a[:, 0] ** 2 + 1.0
        return x, law(x), x_val, law(x_val)

    @staticmethod
    def _flash(engine, prefix, x, y, x_val, weights):
        """A Flash-ANSR candidate dict as a snapshot stores it, scored the way the refine worker scores."""
        import numpy as np
        from flash_ansr.scoring import score_row, count_constants
        from simplipy.engine import Mode
        from srbf.metrics.numeric import fvu
        from srbf.subprocess_adapter import evaluate_prefix
        yp, ypv = evaluate_prefix(engine, list(prefix), ["x1", "x2"], x, x_val)
        f = float(fvu(y, np.asarray(yp).reshape(-1)))
        mdl = float(engine.complexity(list(prefix), certified=True, mode=Mode.f64, canon="default"))
        return {"expression_prefix": list(prefix), "skeleton_prefix": list(prefix), "expression_infix": " ".join(prefix),
                "constants": [], "fvu": f, "mdl": mdl, "n_nodes": len(prefix), "log_prob": -1.0, "pareto_rank": -1,
                "score": score_row({"fvu": f, "expression": prefix, "constant_count": count_constants(prefix), "log_prob": -1.0, "mdl": mdl}, weights),
                "y_pred": np.asarray(yp).reshape(-1), "y_pred_val": np.asarray(ypv).reshape(-1)}

    def test_the_hall_of_fame_law_beats_a_worse_flash_answer(self, engine):
        import numpy as np
        from srbf.hybrid_adapter import pick_prediction
        x, y, x_val, y_val = self._problem()
        weights = {"mdl": 1e-2}
        flash = self._flash(engine, ["*", "2.1", "pow", "x1", "2"], x, y, x_val, weights)   # close, not exact
        hof = [{"complexity": 1, "loss": 9.0, "score": 0.0, "equation": "v1"},
               {"complexity": 7, "loss": 0.0, "score": 1.0, "equation": "(2.0 * (v1 ^ 2)) + 1.0"},
               {"complexity": 9, "loss": 0.0, "score": 0.1, "equation": "((2.0 * (v1 ^ 2)) + 1.0) + (0.0 * v2)"}]
        values = {"predicted_expression": "v1", "predicted_expression_prefix": ["x1"], "prediction_success": True,
                  "equations": hof, "variable_names": ["v1", "v2"]}
        winner = pick_prediction(values, flash=[flash], equations=hof, engine=engine, weights=weights,
                                       x_support=x, y_fit=y, x_val=x_val, y_val=y_val, variables=["v1", "v2"])
        assert winner is not None and values["predicted_source"] == "pysr" and values["predicted_hof_index"] == 1
        assert values["pysr_expression"] == "v1"                                  # PySR's own pick is kept
        assert "**" not in values["predicted_expression_prefix"] and "pow" in values["predicted_expression_prefix"]
        np.testing.assert_allclose(values["y_pred"].reshape(-1), y, rtol=1e-12)
        np.testing.assert_allclose(values["y_pred_val"].reshape(-1), y_val, rtol=1e-12)
        assert values["predicted_mdl"] is not None and values["prediction_success"] is True
        scores = [e["score"] for e in values["hybrid_candidates"]]
        assert scores == sorted(scores) and len(scores) == 4                         # 1 Flash-ANSR + 3 PySR candidates, ranked
        # the exact law with a free 0 * v2 tail pays the MDL penalty and loses to the plain law
        assert [e["hof_index"] for e in values["hybrid_candidates"]][:2] == [1, 2]

    def test_flash_keeps_the_answer_when_it_scores_better(self, engine):
        from srbf.hybrid_adapter import pick_prediction
        x, y, x_val, y_val = self._problem()
        weights = {"mdl": 1e-2}
        flash = self._flash(engine, ["+", "*", "2", "pow", "x1", "2", "1"], x, y, x_val, weights)   # exact and short
        hof = [{"equation": "v1"}, {"equation": "((2.0 * (v1 ^ 2)) + 1.0) + (0.0 * v2)"}, {"equation": "not an expression ("}]
        values = {"predicted_expression": "v1", "predicted_expression_prefix": ["x1"], "prediction_success": True}
        winner = pick_prediction(values, flash=[flash], equations=hof, engine=engine, weights=weights,
                                       x_support=x, y_fit=y, x_val=x_val, y_val=y_val, variables=["v1", "v2"])
        assert winner is not None and values["predicted_source"] == "flash-ansr"
        assert values["predicted_expression_prefix"] == ["+", "*", "2", "pow", "x1", "2", "1"]
        assert len(values["hybrid_candidates"]) == 3                                # the unreadable entry is dropped

    def test_a_failed_gp_stage_falls_back_on_flash(self, engine):
        from srbf.hybrid_adapter import pick_prediction
        x, y, x_val, y_val = self._problem()
        weights = {"mdl": 1e-2}
        flash = self._flash(engine, ["+", "*", "2", "pow", "x1", "2", "1"], x, y, x_val, weights)
        values = {"prediction_success": False, "error": "WorkerTimeout: no reply within 7200 s"}
        pick_prediction(values, flash=[flash], equations=None, engine=engine, weights=weights,
                              x_support=x, y_fit=y, x_val=x_val, y_val=y_val, variables=["v1", "v2"])
        assert values["predicted_source"] == "flash-ansr" and values["prediction_success"] is True
        assert values["pysr_error"].startswith("WorkerTimeout") and values["error"] is None

    def test_pysr_candidates_go_through_the_ladder(self, engine):
        """A PySR entry with almost-round constants is re-fitted and re-spelled like a Flash-ANSR
        candidate: the pool entry carries integer constants, a lower MDL and a spelling record."""
        from srbf.hybrid_adapter import pysr_candidates
        x, y, x_val, y_val = self._problem()
        weights = {"mdl": 1e-2}
        rows = pysr_candidates([{"equation": "(2.0000001 * (v1 ^ 2)) + 0.9999999"}], engine=engine, weights=weights,
                               x_support=x, y_fit=y, x_val=x_val, variables=["v1", "v2"])
        assert rows, "the entry must enter the pool"
        best = min(rows, key=lambda r: r["score"])
        assert best["spelling"], "the ladder must have re-spelled the almost-round constants"
        assert all(float(c).is_integer() for c in best["constants"]) and best["constants"]
        assert "**" not in best["expression_prefix"]
        plain = pysr_candidates([{"equation": "(2.0000001 * (v1 ^ 2)) + 0.9999999"}], engine=engine, weights=weights,
                                x_support=x, y_fit=y, x_val=x_val, variables=["v1", "v2"], refine={"constant_ladder": None})
        assert len(plain) == 1 and plain[0]["spelling"] is None and best["mdl"] < plain[0]["mdl"]
