"""The hybrid arm's pure parts: the seed bridge to Julia syntax, the budget split by the clock and the
snapshot targets, the clocked generation pass, the config builder's validation, PySR's candidates."""
import math

import pytest

from srbf.hybrid_adapter import prefix_to_julia

ARITY = {"+": 2, "-": 2, "*": 2, "/": 2, "pow": 2, "rootn": 2, "neg": 1, "inv": 1, "abs": 1,
         "sin": 1, "cos": 1, "exp": 1, "log": 1}


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
    def _adapter(self, ratio, ratios, **kw):
        from srbf.hybrid_adapter import FlashANSRPySRAdapter
        snapshot_dir = kw.pop("snapshot_dir", "/nonexistent/unused")     # no clock_state.json to inherit
        return FlashANSRPySRAdapter(flash=object(), pysr=object(), budget_s=100.0, ratio=ratio, ratios=ratios,
                                    snapshot_dir=snapshot_dir, **kw)

    def test_shares_and_targets(self):
        ratios = [i / 10 for i in range(11)]
        a = self._adapter(0.0, ratios)
        assert a.generation_seconds(0.0) == 100.0 and a.pysr_seconds(0.0) == 0.0
        assert a.generation_seconds(1.0) == 0.0 and a.pysr_seconds(1.0) == 100.0
        assert a.time_targets() == [10.0 * k for k in range(1, 11)]        # one snapshot per share, 0 excluded
        for r in ratios:
            assert a.generation_seconds(r) + a.pysr_seconds(r) == pytest.approx(100.0)

    def test_pysr_runs_on_its_own_clock_minus_the_running_costs(self):
        a = self._adapter(0.1, [0.1], pysr_overhead_s=2.5, pricing_reserve_s=0.5, snapshot_dir="/nonexistent/unused")
        assert a.pysr_search_seconds(0.1) == pytest.approx(7.0)             # 10 s share - 2.5 - 0.5
        a._pysr_overhead.observe(3.5)                                      # the configured value counts as one observation
        a._pricing_reserve.observe(1.5)
        assert a.pysr_search_seconds(0.1) == pytest.approx(10.0 - 3.0 - 1.0)
        a._pysr_overhead.observe(float("nan"))                            # ignored
        assert a.pysr_search_seconds(0.1) == pytest.approx(6.0)
        assert a.pysr_search_seconds(0.01) == 0.0                          # a share the fixed costs eat: no search

    def test_running_costs_persist_across_cells(self, tmp_path):
        from srbf.hybrid_adapter import FlashANSRPySRAdapter
        first = FlashANSRPySRAdapter(flash=object(), pysr=object(), budget_s=100.0, ratio=0.5, ratios=[0.5],
                                     snapshot_dir=str(tmp_path / "snapshots" / "fastsrb"), pysr_overhead_s=4.0, pricing_reserve_s=0.2)
        first._pysr_overhead.observe(6.0); first._pricing_reserve.observe(0.6); first.clock.save()
        assert (tmp_path / "snapshots" / "clock_state.json").exists()
        second = FlashANSRPySRAdapter(flash=object(), pysr=object(), budget_s=100.0, ratio=0.3, ratios=[0.3],
                                      snapshot_dir=str(tmp_path / "snapshots" / "feynman"), pysr_overhead_s=4.0, pricing_reserve_s=0.2)
        assert second._pysr_overhead.value == pytest.approx(5.0) and second._pysr_overhead.count == 2   # the next cell starts where the last one left off
        assert second.pysr_search_seconds(0.3) == pytest.approx(30.0 - 5.0 - 0.4)

    def test_rejects_a_bad_tolerance(self):
        with pytest.raises(ValueError, match="landing_tolerance"):
            self._adapter(0.5, [0.5], landing_tolerance=0.0)


class TestConfig:
    def test_builder_validates_blocks(self):
        from srbf.config import build_model_adapter
        with pytest.raises(ValueError):
            build_model_adapter({"type": "flash_ansr_pysr", "hybrid": {"ratio": 0.5}})


class TestSnapshotFingerprint:
    """A snapshot is only valid on the data it was generated on: the cache keys on a fingerprint."""

    def _adapter(self, tmp_path):
        from srbf.hybrid_adapter import FlashANSRPySRAdapter
        return FlashANSRPySRAdapter(
            flash=object(), pysr=object(), budget_s=20.0, ratio=0.0, ratios=[0.0, 0.5, 1.0], snapshot_dir=str(tmp_path))

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
            return {t: {"seconds": t, "choices": 10, "cum_wall": t, "cum_generation": 0.5, "cum_refinement": 0.5,
                        "pool_size": 1, "best": None, "seeds": []} for t in adapter.time_targets()}

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


class TestClockedGeneration:
    """The generation pass runs by the clock: chunks until the wall time reaches each share, landing
    within the tolerance; the candidate count is what the clock allowed."""

    @staticmethod
    def _fake_model(seconds_per_candidate: float, call_overhead: float):
        import time as _time
        from types import SimpleNamespace
        import numpy as np

        def infer(X, y, **kw):
            n = int(model.generation_config.choices)
            _time.sleep(call_overhead + seconds_per_candidate * n)
            cands = []
            for i in range(n):
                cands.append(SimpleNamespace(
                    raw_beam=[infer.counter + i], expression=["x1"], expression_prefix=["*", str(1.0 + 1e-3 * (infer.counter + i)), "x1"],
                    expression_infix="c * x1", skeleton_prefix=["*", "<constant>", "x1"], constants=[1.0],
                    score=-1.0 - 1e-6 * (infer.counter + i), fvu=1e-3, mdl=1000.0, n_nodes=3, log_prob=-1.0, pareto_rank=-1,
                    spelling=None, y_pred=np.ones(X.shape[0]), y_pred_val=np.ones(0)))
            infer.counter += n
            return SimpleNamespace(candidates=cands, generation_time=seconds_per_candidate * n, refinement_time=0.0)

        infer.counter = 0
        model = SimpleNamespace(infer=infer, generation_config=SimpleNamespace(choices=1024),
                                simplipy_engine=SimpleNamespace(operator_arity={"*": 2, "+": 2}))
        return model

    def test_lands_each_share_within_the_tolerance(self, tmp_path):
        from types import SimpleNamespace
        from srbf.hybrid_adapter import FlashANSRPySRAdapter
        from srbf.testing import toy_sample
        model = self._fake_model(seconds_per_candidate=0.0005, call_overhead=0.01)
        flash = SimpleNamespace(model=model, emission="fittable", _resolve_complexity=lambda record: None)
        budget = 1.2
        adapter = FlashANSRPySRAdapter(flash=flash, pysr=object(), budget_s=budget, ratio=0.5, ratios=[0.0, 0.5, 1.0],
                                       snapshot_dir=str(tmp_path), k_seeds=5, landing_tolerance=0.02, first_chunk=32, min_chunk=8)
        sample = toy_sample(0)
        record = {"eval_row_index": 0, "variables": ["v1", "v2"]}
        targets = adapter._generate_snapshots(sample, record)
        assert sorted(targets) == [0.6, 1.2]
        tol = 0.02 * budget
        for t, snap in targets.items():
            assert snap["seconds"] == t
            assert t - tol <= snap["cum_wall"] <= t + 0.15, (t, snap["cum_wall"])   # short by at most the tolerance, over by at most a chunk's slip
            assert snap["choices"] == model.infer.counter or snap["choices"] < model.infer.counter
            assert snap["best"] is not None and len(snap["seeds"]) == 5 and len(snap["candidates"]) == 5
        assert targets[1.2]["choices"] > targets[0.6]["choices"] > 0
        assert targets[1.2]["calls"] <= 8                                   # a big chunk and a landing chunk per share, not a dribble
