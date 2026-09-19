"""PySR's full-suite evaluation on the reference machine (owner 2026-09-19): the config generator copies the
Flash-ANSR data exactly, and the ladder runner's --full-suite mode leaves every catalog whole."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import srbf  # noqa: F401  (registers the !sweep tag)
from srbf.config import load_config, select_experiment

ROOT = Path(__file__).parents[1]
T8 = ROOT / "configs" / "evaluation" / "scaling" / "flash-ansr-v25.0-T8-20M_srbf.yaml"


def _module(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _config(tmp_path: Path) -> dict:
    out = tmp_path / "pysr_suite.yaml"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "make_pysr_suite_config.py"), "--from", str(T8),
                    "--python", "/venvs/pysr/bin/python", "--out", str(out), "--ladder", "1,2,4"], check=True)
    return load_config(str(out))


def test_pysr_sees_exactly_the_data_flash_ansr_sees(tmp_path: Path) -> None:
    cfg, t8 = _config(tmp_path), load_config(str(T8))
    assert list(cfg["experiments"]) == list(t8["experiments"])
    for cat in cfg["experiments"]:
        assert select_experiment(cfg, cat)["data_source"] == select_experiment(t8, cat)["data_source"], cat


def test_pysr_runs_as_it_ships_with_iterations_as_the_ladder(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    cat = next(iter(cfg["experiments"]))
    adapter = select_experiment(cfg, cat)["model_adapter"]
    assert adapter["type"] == "pysr" and adapter["config_provenance"] == "upstream_default"
    assert "maxsize" not in adapter and "parsimony" not in adapter          # upstream defaults: never set on a headline baseline
    assert _module("run_timing_ladder").ladder_of(cfg, cat) == [1, 2, 4]   # the runner reads the iterations, not the guard
    # the owner's hang policy (2026-09-19): a minute idle is a hang, retried once, logged apart; crashes restart
    assert adapter["hang_after_idle_s"] == 60 and adapter["hang_log"].endswith("/suite/pysr/hangs.jsonl")
    assert adapter["max_restarts"] >= 100 and adapter["worker_log"].endswith(f"/worker_logs/{cat}.log")


def test_full_suite_mode_leaves_each_catalog_whole(tmp_path: Path) -> None:
    rtl = _module("run_timing_ladder")
    cfg = _config(tmp_path)
    cats = list(cfg["experiments"])[:2]
    derived = rtl.timing_config(cfg, cats, None, "pysr", None, None)
    for cat in cats:
        block = derived["experiments"][cat]
        assert block["data_source"]["catalog"] == select_experiment(cfg, cat)["data_source"]["catalog"]
        assert all("/results/evaluation/suite/pysr/" in v for v in block["runner"]["output"].values)


def test_up_to_leaves_the_rungs_above_for_a_later_call(tmp_path: Path) -> None:
    """The queue raises several model rows together, one rung at a time: a call stops after --up-to."""
    _config(tmp_path)
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_timing_ladder.py"), "-c", str(tmp_path / "pysr_suite.yaml"),
                          "--full-suite", "--model-name", "pysr", "--root", str(tmp_path), "--experiments", "nguyen",
                          "--up-to", "2", "--dry-run"], check=True, capture_output=True, text=True).stdout
    assert "ladder=1" in out and "ladder=2" in out and "ladder=4" not in out
