"""Benchmark.from_config: config -> components, resume math, completed no-op, adapter builders.

Hermetic -- the source/adapter builders are monkeypatched (or fed fakes), so no HF download, no
model load. This is the 0.5.0 replacement for the old run_config.build_evaluation_run tests; the
old `type: skeleton_dataset`/`type: fastsrb`/skeleton-pin behaviours are gone (the data source is a
`symbolic_data` catalog now), so only the still-relevant behaviours are ported.
"""
import pickle
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from srbf.benchmark import Benchmark
from srbf import config as run_config


def _write_results(path, length: int) -> None:
    payload = {"expression": list(range(length)), "y_pred": list(range(length))}
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


# --- resume / completed / remaining math -------------------------------------------------

def test_from_config_short_circuits_when_limit_completed(tmp_path, monkeypatch):
    results_path = tmp_path / "existing.pkl"
    _write_results(results_path, length=3)

    flags = {"data": False, "adapter": False}

    def fake_build_catalog_source(*args, **kwargs):  # pragma: no cover - must NOT run when complete
        flags["data"] = True
        return object()

    def fake_build_model_adapter(*args, **kwargs):  # pragma: no cover - must NOT run when complete
        flags["adapter"] = True
        return object()

    monkeypatch.setattr(run_config, "build_catalog_source", fake_build_catalog_source)
    monkeypatch.setattr(run_config, "build_model_adapter", fake_build_model_adapter)

    bench = Benchmark.from_config(
        {
            "run": {
                "data_source": {"catalog": "v23-val"},
                "model_adapter": {"type": "flash_ansr"},
                "runner": {"limit": 3, "output": str(results_path), "resume": True},
            }
        }
    )

    assert bench.completed is True
    assert bench.source is None and bench.model_adapter is None
    assert bench.existing_results == 3
    assert flags == {"data": False, "adapter": False}  # neither builder ran (no model load on a done run)


def test_from_config_builds_with_remaining(tmp_path, monkeypatch):
    results_path = tmp_path / "partial.pkl"
    _write_results(results_path, length=2)

    fake_source = object()
    fake_adapter = object()
    captured: dict[str, Any] = {}

    def fake_build_catalog_source(config, *, target_size, skip):
        captured["data_cfg"] = config
        captured["target_size"] = target_size
        captured["skip"] = skip
        return fake_source

    def fake_build_model_adapter(config):
        captured["adapter_cfg"] = config
        return fake_adapter

    monkeypatch.setattr(run_config, "build_catalog_source", fake_build_catalog_source)
    monkeypatch.setattr(run_config, "build_model_adapter", fake_build_model_adapter)

    bench = Benchmark.from_config(
        {
            "run": {
                "data_source": {"catalog": "v23-val", "target_size": 99},
                "model_adapter": {"type": "flash_ansr", "extra": True},
                "runner": {"limit": 5, "save_every": 2, "output": str(results_path), "resume": True},
            }
        }
    )

    assert bench.completed is False
    assert bench.limit == 3                      # runner.limit (5) wins over data_source.target_size; 5 - 2 existing
    assert bench.save_every == 2
    assert bench.total_limit == 5
    assert bench.source is fake_source and bench.model_adapter is fake_adapter
    assert bench.result_store.size == 2
    assert captured["target_size"] == 3 and captured["skip"] == 2
    assert captured["adapter_cfg"]["extra"] is True


def test_from_config_infers_remaining_from_size_hint(tmp_path, monkeypatch):
    results_path = tmp_path / "partial.pkl"
    _write_results(results_path, length=4)

    class DummySource:
        def __init__(self, pending):
            self._pending = pending

        def size_hint(self):
            return self._pending

    captured: dict[str, Any] = {}

    def fake_build_catalog_source(config, *, target_size, skip):
        captured["target_size"] = target_size
        captured["skip"] = skip
        return DummySource(2)   # 2 problems remain in the (frozen) catalog after skip

    monkeypatch.setattr(run_config, "build_catalog_source", fake_build_catalog_source)
    monkeypatch.setattr(run_config, "build_model_adapter", lambda config: object())

    bench = Benchmark.from_config(
        {
            "run": {
                "data_source": {"catalog": "v23-val"},     # no target_size; no runner.limit
                "model_adapter": {"type": "flash_ansr"},
                "runner": {"output": str(results_path), "resume": True},
            }
        }
    )

    assert bench.completed is False
    assert bench.total_limit == 6                # existing 4 + pending 2
    assert bench.limit == 2                       # remaining inferred from size_hint
    assert captured["target_size"] is None and captured["skip"] == 4


def test_from_config_completed_when_size_hint_exhausted(tmp_path, monkeypatch):
    results_path = tmp_path / "existing.pkl"
    _write_results(results_path, length=6)

    adapter_built = {"flag": False}

    def fake_build_catalog_source(config, *, target_size, skip):
        return SimpleNamespace(size_hint=lambda: 0)  # nothing left after skip

    def fake_build_model_adapter(config):  # pragma: no cover - must NOT run
        adapter_built["flag"] = True
        return object()

    monkeypatch.setattr(run_config, "build_catalog_source", fake_build_catalog_source)
    monkeypatch.setattr(run_config, "build_model_adapter", fake_build_model_adapter)

    bench = Benchmark.from_config(
        {
            "run": {
                "data_source": {"catalog": "v23-val"},
                "model_adapter": {"type": "flash_ansr"},
                "runner": {"output": str(results_path), "resume": True},
            }
        }
    )

    assert bench.completed is True
    assert adapter_built["flag"] is False        # model not loaded once the source is exhausted


def test_completed_benchmark_run_is_a_noop(tmp_path, monkeypatch):
    results_path = tmp_path / "existing.pkl"
    _write_results(results_path, length=3)
    monkeypatch.setattr(run_config, "build_catalog_source", lambda *a, **k: object())
    monkeypatch.setattr(run_config, "build_model_adapter", lambda *a, **k: object())

    bench = Benchmark.from_config(
        {"run": {"data_source": {"catalog": "v23-val"}, "model_adapter": {"type": "flash_ansr"},
                 "runner": {"limit": 3, "output": str(results_path), "resume": True}}}
    )
    snapshot = bench.run(verbose=False, progress=False)
    assert snapshot["expression"] == list(range(3))  # the loaded results, unchanged


# --- experiment selection ----------------------------------------------------------------

def test_from_config_selects_named_experiment(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_build_catalog_source(config, *, target_size, skip):
        captured["data_cfg"] = config
        return object()

    def fake_build_model_adapter(config):
        captured["adapter_cfg"] = config
        return object()

    monkeypatch.setattr(run_config, "build_catalog_source", fake_build_catalog_source)
    monkeypatch.setattr(run_config, "build_model_adapter", fake_build_model_adapter)

    bench = Benchmark.from_config(
        {
            "default_experiment": "fast",
            "experiments": {
                "fast": {"run": {"data_source": {"catalog": "cat_a", "target_size": 5},
                                 "model_adapter": {"type": "flash_ansr", "alpha": 1},
                                 "runner": {"limit": 2}}},
                "slow": {"run": {"data_source": {"catalog": "cat_b"}, "model_adapter": {"type": "flash_ansr"}}},
            },
        },
        experiment="fast",
    )

    assert bench.source is not None
    assert captured["data_cfg"]["catalog"] == "cat_a"
    assert captured["adapter_cfg"]["alpha"] == 1


def test_from_config_requires_experiment_name():
    with pytest.raises(ValueError):
        Benchmark.from_config(
            {"experiments": {"only": {"run": {"data_source": {"catalog": "cat_a"},
                                              "model_adapter": {"type": "flash_ansr"}}}}}
        )


# --- adapter builders (config.py) --------------------------------------------------------

def _patch_flash_ansr(monkeypatch, captured):
    def fake_create_generation_config(method, **kwargs):
        captured["method"] = method
        captured["kwargs"] = kwargs
        return {"method": method, "kwargs": kwargs}

    class FakeFlashANSR:
        @staticmethod
        def load(*, directory, generation_config, **kwargs):
            captured["flash_ansr_gen"] = generation_config
            captured["flash_ansr_dir"] = directory
            captured["flash_ansr_kwargs"] = kwargs
            return SimpleNamespace()

    class DummyAdapter:
        def __init__(self, model, device, complexity, refiner_workers, candidate_store_dir=None):
            self.model = model

    monkeypatch.setattr(run_config, "create_generation_config", fake_create_generation_config)
    monkeypatch.setattr(run_config, "FlashANSR", FakeFlashANSR)
    monkeypatch.setattr(run_config, "FlashANSRAdapter", DummyAdapter)
    return DummyAdapter


def test_build_flash_ansr_adapter_generation_overrides(tmp_path, monkeypatch):
    eval_cfg = {"evaluation": {
        "n_restarts": 1, "refiner_method": "curve_fit_lm", "refiner_p0_noise": "normal",
        "refiner_p0_noise_kwargs": {"loc": 0.0, "scale": 1.0}, "length_penalty": 0.2,
        "constants_penalty": 0.01, "likelihood_penalty": 0.0, "device": "cuda", "refiner_workers": None,
        "generation_config": {"method": "softmax_sampling", "kwargs": {"choices": 8, "max_len": 16}}}}
    eval_path = tmp_path / "evaluation.yaml"
    with eval_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(eval_cfg, handle)

    captured: dict[str, Any] = {}
    DummyAdapter = _patch_flash_ansr(monkeypatch, captured)

    adapter = run_config.build_model_adapter(
        {"type": "flash_ansr", "model_path": str(tmp_path), "evaluation_config": str(eval_path),
         "generation_overrides": {"kwargs": {"choices": 2}}}
    )

    assert isinstance(adapter, DummyAdapter)
    assert captured["method"] == "softmax_sampling"
    assert captured["kwargs"]["choices"] == 2
    assert captured["flash_ansr_gen"]["kwargs"]["choices"] == 2
    assert captured["flash_ansr_kwargs"]["length_penalty"] == 0.2


def test_build_flash_ansr_adapter_inline_evaluation_config(monkeypatch):
    captured: dict[str, Any] = {}
    DummyAdapter = _patch_flash_ansr(monkeypatch, captured)

    inline_cfg = {
        "n_restarts": 2, "refiner_method": "curve_fit_lm", "refiner_p0_noise": "normal",
        "refiner_p0_noise_kwargs": {"loc": 0.0, "scale": 1.0}, "length_penalty": 0.15,
        "constants_penalty": 0.0, "likelihood_penalty": 0.0, "device": "cuda",
        "generation_config": {"method": "softmax_sampling", "kwargs": {"choices": 4, "max_len": 16}}}

    adapter = run_config.build_model_adapter(
        {"type": "flash_ansr", "model_path": "./models/v23", "evaluation_config": inline_cfg,
         "complexity": "none", "device": "cuda"}
    )

    assert isinstance(adapter, DummyAdapter)
    assert captured["kwargs"]["choices"] == 4
    assert captured["flash_ansr_kwargs"]["n_restarts"] == 2
    assert captured["flash_ansr_kwargs"]["length_penalty"] == 0.15


def test_build_pysr_adapter_requires_explicit_engine(monkeypatch):
    class DummyEngineLoader:
        @staticmethod
        def load(path, install=True):
            return SimpleNamespace(name="engine", path=path)

    class DummyAdapter:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(run_config, "SimpliPyEngine", DummyEngineLoader)
    monkeypatch.setattr(run_config, "PySRAdapter", DummyAdapter)

    adapter = run_config.build_model_adapter({"type": "pysr", "niterations": 2, "simplipy_engine": "dev_7-3"})
    assert isinstance(adapter, DummyAdapter)
    assert adapter.kwargs["simplipy_engine"].path == "dev_7-3"

    with pytest.raises(ValueError):  # no dataset to borrow an engine from anymore -> must be explicit
        run_config.build_model_adapter({"type": "pysr", "niterations": 1})


# --- end-to-end wiring smoke -------------------------------------------------------------

def test_from_config_end_to_end_through_catalog_source(monkeypatch):
    """The one smoke exercising the REAL chain build_catalog_source -> CatalogSource.from_catalog ->
    ProblemSource -> Benchmark.run -> store (build_catalog_source NOT mocked; only the model adapter
    is faked, since the real flash_ansr one needs a GPU model). Catches any schema mismatch in the
    `{catalog, sampling, holdouts}` wrapping that the mocked from_config tests structurally cannot.

    Hermetic via a FROZEN in-memory catalog (no HF, no skeleton realization): freeze 3 inline problems
    into a ProblemCatalog and feed it as the data_source `catalog:`.
    """
    import numpy as np
    from symbolic_data import Problem, ProblemSource

    def _p(i: int) -> Problem:
        x = np.arange(8, dtype=np.float32).reshape(8, 1)
        y = (x * i).astype(np.float32)
        return Problem(
            x_support=x, y_support=y, y_support_noisy=y.copy(),
            x_validation=x[:2], y_validation=y[:2], y_validation_noisy=y[:2].copy(),
            skeleton=["*", "x1", "c"], expression=["*", "x1", str(i)], constants=[float(i)],
            variables=["x1"], complexity=3, eq_id=f"E{i}",
        )

    catalog = ProblemSource({"problems": [_p(i).to_dict() for i in (1, 2, 3)]}).to_catalog(name="smoke")

    class _FakeAdapter:
        def prepare(self, *, data_source=None):
            return None

        def evaluate_sample(self, sample):
            return {**sample.clone_metadata(), "prediction_success": True}

    monkeypatch.setattr(run_config, "build_model_adapter", lambda cfg: _FakeAdapter())

    bench = Benchmark.from_config({
        "run": {
            "data_source": {"catalog": catalog},
            "model_adapter": {"type": "flash_ansr"},
            "runner": {},
        }
    })
    snapshot = bench.run(verbose=False, progress=False)
    assert bench.result_store.size == 3
    assert snapshot["prediction_success"] == [True, True, True]
    assert sorted(snapshot["benchmark_eq_id"]) == ["E1", "E2", "E3"]


# --- runs_from_config: experiments map + inline !sweep -----------------------------------

def test_runs_from_config_expands_named_sweep(monkeypatch):
    from srbf.sweep import Sweep
    captured: list[dict[str, Any]] = []

    def fake_src(cfg, *, target_size, skip):
        captured.append(cfg)
        return SimpleNamespace(size_hint=lambda: 4)

    monkeypatch.setattr(run_config, "build_catalog_source", fake_src)
    monkeypatch.setattr(run_config, "build_model_adapter", lambda cfg: object())

    cfg = {"run": {
        "data_source": {"catalog": "v23-val", "problems_per_expression": Sweep([10, 5], name="L")},
        "model_adapter": {"type": "flash_ansr", "choices": Sweep([16, 256], name="L")},
        "runner": {},
    }}
    benches = Benchmark.runs_from_config(cfg)
    assert len(benches) == 2                                  # zipped on "L", not a 2x2 grid
    assert [c["problems_per_expression"] for c in captured] == [10, 5]   # matched to the rung

    # --sweep-filter selects one rung (label value = first-declared member of the axis)
    one = Benchmark.runs_from_config(cfg, sweep_filter={"L": 5})
    assert len(one) == 1 and one[0].label == {"L": 5}


def test_runs_from_config_expands_experiments(monkeypatch):
    monkeypatch.setattr(run_config, "build_catalog_source",
                        lambda cfg, *, target_size, skip: SimpleNamespace(size_hint=lambda: 3))
    monkeypatch.setattr(run_config, "build_model_adapter", lambda cfg: object())

    cfg = {"experiments": {
        "a": {"run": {"data_source": {"catalog": "v23-val"}, "model_adapter": {"type": "flash_ansr"}, "runner": {}}},
        "b": {"run": {"data_source": {"catalog": "fastsrb"}, "model_adapter": {"type": "flash_ansr"}, "runner": {}}},
    }}
    benches = Benchmark.runs_from_config(cfg)
    assert len(benches) == 2
    assert {b.label["experiment"] for b in benches} == {"a", "b"}

    only = Benchmark.runs_from_config(cfg, experiment="b")
    assert len(only) == 1 and only[0].label["experiment"] == "b"
