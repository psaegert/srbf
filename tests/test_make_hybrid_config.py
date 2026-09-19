"""The hybrid config writer must emit configs the installed srbf and flash-ansr-hybrid can run.

On 2026-09-19 every PySR-alone cell of the solomon timing queue died at config load
(``Unsupported model adapter type: flash_ansr_pysr``): srbf 0.16.0 had renamed the adapter to
``flash_ansr_hybrid`` and moved PySR in-process, but this writer still emitted the old name, the old
worker-protocol ``pysr:`` block, the pre-0.17 ``choices`` key and the pre-ruling S0 ranking.
"""
import subprocess
import sys
from pathlib import Path

import yaml

import srbf  # noqa: F401  (registers the !sweep tag)
from srbf.config import _ADAPTER_REGISTRY

WRITER = Path(__file__).parents[1] / "scripts" / "make_hybrid_config.py"
PYSR_SETTINGS_KEYS = {"maxsize", "parsimony", "model_selection", "warmup"}   # flash_ansr_hybrid.PySRSettings


def _written(tmp_path: Path) -> dict:
    out = tmp_path / "hybrid.yaml"
    subprocess.run([sys.executable, str(WRITER), "--model-path", "/models/m", "--model-name", "m", "--budget", "10",
                    "--ratios", "1", "--snapshot-dir", str(tmp_path / "snap"), "--out", str(out),
                    "--catalogs", "fastsrb,feynman"], check=True)
    return yaml.load(out.read_text(), Loader=yaml.FullLoader)["experiments"]


def test_every_adapter_the_writer_emits_is_one_srbf_can_build(tmp_path: Path) -> None:
    for cat, exp in _written(tmp_path).items():
        adapter = exp["model_adapter"]
        assert adapter["type"] in _ADAPTER_REGISTRY, f"{cat}: {adapter['type']!r} is not an srbf adapter type"
        assert adapter["flash_ansr"]["type"] in _ADAPTER_REGISTRY


def test_the_model_block_speaks_the_current_flash_ansr_api(tmp_path: Path) -> None:
    for exp in _written(tmp_path).values():
        evaluation = exp["model_adapter"]["flash_ansr"]["evaluation_config"]
        kwargs = evaluation["generation_config"]["kwargs"]
        assert "draws" in kwargs and "choices" not in kwargs, "flash-ansr >= 0.17 counts draws"
        assert evaluation["ranking"] == {"mode": "mdl"}, "S1, the ruled default (owner 2026-09-16), not S0's mdl_strength"


def test_the_pysr_block_holds_only_pysr_settings(tmp_path: Path) -> None:
    # Any other key is forwarded to PySRRegressor as a kwarg: the old worker-protocol keys (python,
    # timeout_in_seconds, niterations, ...) would collide with the clock's own or be rejected outright.
    for exp in _written(tmp_path).values():
        assert set(exp["model_adapter"]["pysr"]) <= PYSR_SETTINGS_KEYS
