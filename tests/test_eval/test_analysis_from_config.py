"""``srbf analyze -c``: the run manifest derived from a run config (experiments x sweep rungs whose
outputs exist), with the model name from the adapter block or the caller."""
import pickle

from srbf.analysis import runs_from_config


def _snapshot(path, n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump({"eval_row_index": list(range(n)), "prediction_success": [True] * n, "__meta__": {"config_provenance": "harness_tuned"}}, handle)


def test_runs_follow_the_experiments_and_rungs_that_have_outputs(tmp_path):
    config = tmp_path / "mymethod.yaml"
    config.write_text(
        "suite: [fastsrb, feynman]\n"
        "run:\n"
        "  data_source: {sampling: {n_support: 8}}\n"
        "  model_adapter:\n"
        "    type: subprocess\n"
        "    worker: /somewhere/mymethod_worker.py\n"
        "    options: {choices: !sweep {name: ladder, values: [1, 2]}}\n"
        "  runner:\n"
        f"    output: !sweep {{name: ladder, values: ['{tmp_path}/{{catalog}}_1.pkl', '{tmp_path}/{{catalog}}_2.pkl']}}\n"
    )
    _snapshot(tmp_path / "fastsrb_1.pkl", 3)
    _snapshot(tmp_path / "fastsrb_2.pkl", 3)
    _snapshot(tmp_path / "feynman_2.pkl", 5)
    skipped = []
    runs = runs_from_config(str(config), warn=skipped.append)
    assert [(r.model, r.benchmark, r.scaling, r.axis) for r in runs] == [
        ("mymethod", "fastsrb", 1.0, "ladder"), ("mymethod", "fastsrb", 2.0, "ladder"), ("mymethod", "feynman", 2.0, "ladder")]
    assert len(runs[2].snapshot["eval_row_index"]) == 5
    assert skipped == [f"no results yet: {tmp_path}/feynman_1.pkl"]
    assert [r.model for r in runs_from_config(str(config), model="Ours")] == ["Ours"] * 3


def test_single_run_configs_and_flash_ansr_names(tmp_path):
    config = tmp_path / "single.yaml"
    config.write_text(
        "run:\n"
        "  data_source: {catalog: nguyen}\n"
        "  model_adapter: {type: flash_ansr, model_path: '{{ROOT}}/models/psaegert/flash-ansr-v25.0-T7-3M'}\n"
        f"  runner: {{output: '{tmp_path}/nguyen.pkl'}}\n"
    )
    _snapshot(tmp_path / "nguyen.pkl", 2)
    (run,) = runs_from_config(str(config))
    assert (run.model, run.benchmark, run.scaling, run.axis) == ("flash-ansr-v25.0-T7-3M", "nguyen", None, "scaling")


def test_model_names_from_adapter_blocks():
    from srbf.analysis import _model_name

    assert _model_name({"type": "subprocess", "worker": "{{ROOT}}/adapters/smoke/worker.py"}) == "smoke"
    assert _model_name({"type": "subprocess", "worker": "/x/mymethod_worker.py"}) == "mymethod"
    assert _model_name({"type": "subprocess", "worker": "pysr"}) == "pysr"
    assert _model_name({"type": "pysr"}) == "pysr"
    assert _model_name({"type": "flash_ansr", "model_path": "{{ROOT}}/models/psaegert/flash-ansr-v25.0-T7-3M/"}) == "flash-ansr-v25.0-T7-3M"
