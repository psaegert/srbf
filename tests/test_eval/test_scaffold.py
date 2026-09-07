"""``srbf new``: the scaffold writes a worker, a suite config, requirements and a test, in the bench
layout under the asset root or in the repo layout of a pull request, and the worker it writes runs."""
import sys

import pytest

from srbf.config import load_run_config, select_experiment
from srbf.scaffold import scaffold_adapter
from srbf.suites import SRBF_CATALOGS


def test_bench_layout_under_the_asset_root(tmp_path, monkeypatch):
    monkeypatch.setenv("FLASH_ANSR_ROOT", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    scaffold = scaffold_adapter("mymethod")
    adapter_dir = tmp_path / "adapters" / "mymethod"
    assert [p.name for p in scaffold.files] == ["worker.py", "config.yaml", "requirements.txt", "test_worker.py"]
    assert all(p.parent == adapter_dir for p in scaffold.files)
    raw = load_run_config(str(scaffold.config))
    assert list(raw["experiments"]) == list(SRBF_CATALOGS)
    model_cfg = select_experiment(raw, "fastsrb")["model_adapter"]
    assert model_cfg["type"] == "subprocess"
    assert model_cfg["worker"] == "{{ROOT}}/adapters/mymethod/worker.py"
    assert model_cfg["python"] == "{{ROOT}}/envs/mymethod/bin/python"
    assert model_cfg["simplipy_engine"] == "acj-5-4-llm"
    assert select_experiment(raw, "feynman")["runner"]["output"] == "{{ROOT}}/results/evaluation/mymethod/feynman.pkl"
    assert "python -m venv" in " ".join(scaffold.next_steps)
    assert (adapter_dir / "test_worker.py").read_text().count('"config.yaml"') == 1
    with pytest.raises(FileExistsError):
        scaffold_adapter("mymethod")
    scaffold_adapter("mymethod", force=True)


def test_explicit_dir_and_python(tmp_path, monkeypatch):
    monkeypatch.setenv("FLASH_ANSR_ROOT", str(tmp_path / "root"))
    scaffold = scaffold_adapter("other", directory=str(tmp_path / "elsewhere"), python="/opt/venv/bin/python")
    model_cfg = select_experiment(load_run_config(str(scaffold.config)), "fastsrb")["model_adapter"]
    assert model_cfg["worker"] == str(tmp_path / "elsewhere" / "other" / "worker.py")  # outside the root: absolute
    assert model_cfg["python"] == "/opt/venv/bin/python"
    assert not any("python -m venv" in step for step in scaffold.next_steps)


def test_repo_layout(tmp_path, monkeypatch):
    repo = tmp_path / "srbf"
    (repo / "src" / "srbf" / "worker" / "models").mkdir(parents=True)
    (repo / "pyproject.toml").write_text('[project]\nname = "srbf"\n')
    monkeypatch.chdir(repo / "src")
    scaffold = scaffold_adapter("newmethod", repo=True)
    assert scaffold.files == [
        repo / "src/srbf/worker/models/newmethod_worker.py",
        repo / "configs/evaluation/newmethod_srbf.yaml",
        repo / "envs/newmethod/requirements.txt",
        repo / "tests/test_workers/test_newmethod_worker.py",
    ]
    model_cfg = select_experiment(load_run_config(str(scaffold.config)), "fastsrb")["model_adapter"]
    assert model_cfg["worker"] == "newmethod"  # a built-in name once merged
    assert "newmethod_srbf.yaml" in (repo / "tests/test_workers/test_newmethod_worker.py").read_text()
    assert any("docs/models.md" in step for step in scaffold.next_steps)


def test_bad_names_and_missing_checkout(tmp_path, monkeypatch):
    for bad in ("My-Method", "1abc", "", "a b"):
        with pytest.raises(ValueError):
            scaffold_adapter(bad, directory=str(tmp_path))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="srbf checkout"):
        scaffold_adapter("fine", repo=True)


def test_the_scaffolded_worker_fits_the_toy_problem(tmp_path, monkeypatch):
    from srbf.testing import fit_once

    monkeypatch.setenv("FLASH_ANSR_ROOT", str(tmp_path))
    scaffold = scaffold_adapter("linear", python=sys.executable)
    record = fit_once(config=scaffold.config)
    assert record["prediction_success"], record.get("error")
    assert "x1" in record["predicted_expression"] and "x2" in record["predicted_expression"]
    assert len(record["y_pred_val"]) == 12
