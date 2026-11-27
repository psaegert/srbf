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
    runs = runs_from_config(str(config), warn=skipped.append, benchmarks={})   # every catalog its own benchmark
    assert [(r.model, r.benchmark, r.scaling, r.axis) for r in runs] == [
        ("mymethod", "fastsrb", 1.0, "ladder"), ("mymethod", "fastsrb", 2.0, "ladder"), ("mymethod", "feynman", 2.0, "ladder")]
    assert len(runs[2].snapshot["eval_row_index"]) == 5
    assert skipped == [f"no results yet: {tmp_path}/feynman_1.pkl"]
    assert [r.model for r in runs_from_config(str(config), model="Ours", benchmarks={})] == ["Ours"] * 3


def test_single_run_configs_and_flash_ansr_names(tmp_path):
    config = tmp_path / "single.yaml"
    config.write_text(
        "run:\n"
        "  data_source: {catalog: nguyen}\n"
        "  model_adapter: {type: flash_ansr, model_path: '{{ROOT}}/models/psaegert/flash-ansr-v25.0-T8-3M'}\n"
        f"  runner: {{output: '{tmp_path}/nguyen.pkl'}}\n"
    )
    _snapshot(tmp_path / "nguyen.pkl", 2)
    (run,) = runs_from_config(str(config), benchmarks={})
    assert (run.model, run.benchmark, run.scaling, run.axis) == ("flash-ansr-v25.0-T8-3M", "nguyen", None, "scaling")


def test_model_names_from_adapter_blocks():
    from srbf.analysis import _model_name

    assert _model_name({"type": "subprocess", "worker": "{{ROOT}}/adapters/smoke/worker.py"}) == "smoke"
    assert _model_name({"type": "subprocess", "worker": "/x/mymethod_worker.py"}) == "mymethod"
    assert _model_name({"type": "subprocess", "worker": "pysr"}) == "pysr"
    assert _model_name({"type": "pysr"}) == "pysr"
    assert _model_name({"type": "flash_ansr", "model_path": "{{ROOT}}/models/psaegert/flash-ansr-v25.0-T8-3M/"}) == "flash-ansr-v25.0-T8-3M"


def test_a_runs_benchmark_comes_from_its_catalogs_metadata(tmp_path):
    """The catalogs of one benchmark (here its formulas and its fixed-point companion) name it in their metadata; a
    catalog that names none, or that cannot be read (reported), is a benchmark of its own."""
    import yaml

    for name, meta in (("feynman", {"benchmark": "feynman"}), ("feynman-points", {"benchmark": "feynman"}), ("nguyen", {})):
        (tmp_path / f"{name}.yaml").write_text(yaml.safe_dump({"metadata": {"name": name, **meta}, "expressions": {}}))
    experiments = {name: {"data_source": {"catalog": str(tmp_path / f"{name}.yaml")},
                          "model_adapter": {"type": "subprocess", "worker": "/x/mymethod_worker.py"},
                          "runner": {"output": str(tmp_path / f"{name}.pkl")}}
                   for name in ("feynman", "feynman-points", "nguyen", "gone")}
    experiments["gone"]["data_source"]["catalog"] = str(tmp_path / "gone.yaml")
    config = tmp_path / "split.yaml"
    config.write_text(yaml.safe_dump({"experiments": experiments}, sort_keys=False))
    for name in experiments:
        _snapshot(tmp_path / f"{name}.pkl", 2)
    said = []
    runs = runs_from_config(str(config), warn=said.append)
    assert [(r.catalog, r.benchmark) for r in runs] == [
        ("feynman", "feynman"), ("feynman-points", "feynman"), ("nguyen", "nguyen"), ("gone", "gone")]
    assert len(said) == 1 and "'gone'" in said[0]
    assert [r.benchmark for r in runs_from_config(str(config), benchmarks={"nguyen": "classic"})] == [
        "feynman", "feynman-points", "classic", "gone"]   # a given mapping replaces the metadata
