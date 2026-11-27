"""flash-ansr 0.20's inference precision: srbf forwards a configured `precision` to FlashANSR's compute settings
(and nothing when the config is silent, so a flash-ansr without the key keeps loading), and records the precision in
force with every result."""
from types import SimpleNamespace

import pytest

from srbf.model_adapters import FlashANSRAdapter

flash_ansr = pytest.importorskip("flash_ansr")


def _build(monkeypatch, config_extra: dict, eval_extra: dict | None = None) -> dict:  # type: ignore[no-untyped-def]
    import flash_ansr.flash_ansr
    import flash_ansr.utils.generation
    import srbf.config as cfg_mod
    captured: dict = {}

    class FakeModel:
        simplipy_engine = object()

    def fake_load(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return FakeModel()

    monkeypatch.setattr(flash_ansr.flash_ansr.FlashANSR, "load", staticmethod(fake_load))
    monkeypatch.setattr(flash_ansr.utils.generation, "create_generation_config", lambda method, **kw: object())
    eval_cfg = {"n_restarts": 2, "refiner_p0_noise": "normal", "ranking": {"mode": "mdl"},
                "generation_config": {"method": "softmax", "kwargs": {}}}
    eval_cfg.update(eval_extra or {})
    config = {"model_path": "/nowhere", "evaluation_config": eval_cfg}
    config.update(config_extra)
    cfg_mod._build_flash_ansr_adapter(config)
    return captured


def test_silent_config_forwards_no_precision(monkeypatch):
    assert "precision" not in _build(monkeypatch, {})["compute"]


def test_adapter_and_evaluation_config_set_the_precision(monkeypatch):
    assert _build(monkeypatch, {"precision": "fp32"})["compute"]["precision"] == "fp32"
    assert _build(monkeypatch, {}, {"precision": "bf16"})["compute"]["precision"] == "bf16"
    assert _build(monkeypatch, {"precision": "fp32"}, {"precision": "bf16"})["compute"]["precision"] == "fp32"


def test_adapter_reports_the_models_precision():
    adapter = FlashANSRAdapter.__new__(FlashANSRAdapter)
    adapter.model = SimpleNamespace(precision="bf16")
    assert adapter.inference_precision() == "bf16"
    adapter.model = SimpleNamespace()          # flash-ansr before 0.20: no attribute, ran fp32/TF32
    assert adapter.inference_precision() is None
