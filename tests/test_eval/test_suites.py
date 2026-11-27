"""The ``suite:`` shorthand: one run template expands to one experiment per catalog, with the
catalog filled in and ``{catalog}`` substituted everywhere (sweep values included)."""
from pathlib import Path

import pytest

from srbf.config import load_run_config, select_experiment
from srbf.suites import SRBF_CATALOGS, expand_suite, suite_catalogs
from srbf.sweep import Sweep, resolve_sweeps

REFERENCE = Path(__file__).resolve().parents[2] / "configs" / "evaluation" / "scaling" / "flash-ansr-v25.0-T8-3M_srbf.yaml"


def _template() -> dict:
    return {
        "data_source": {"sampling": {"n_support": 512}},
        "model_adapter": {"type": "subprocess", "worker": "w.py", "options": {"tag": "{catalog}", "n": 3}},
        "runner": {"output": Sweep(["{{ROOT}}/r/{catalog}/c1.pkl", "{{ROOT}}/r/{catalog}/c2.pkl"], name="ladder")},
    }


def test_the_srbf_suite_is_the_reference_config_in_order():
    assert list(load_run_config(str(REFERENCE))["experiments"]) == list(SRBF_CATALOGS)
    assert len(SRBF_CATALOGS) == 29


def test_expansion_fills_the_catalog_and_substitutes_the_placeholder():
    expanded = expand_suite({"suite": ["fastsrb", "feynman"], "run": _template(), "default_experiment": "fastsrb"})
    assert list(expanded["experiments"]) == ["fastsrb", "feynman"]
    assert "suite" not in expanded and "run" not in expanded and expanded["default_experiment"] == "fastsrb"
    feynman = select_experiment(expanded, "feynman")
    assert feynman["data_source"] == {"sampling": {"n_support": 512}, "catalog": "feynman"}
    assert feynman["model_adapter"]["options"] == {"tag": "feynman", "n": 3}
    output = feynman["runner"]["output"]
    assert isinstance(output, Sweep) and output.name == "ladder"
    assert output.values == ["{{ROOT}}/r/feynman/c1.pkl", "{{ROOT}}/r/feynman/c2.pkl"]
    # the experiments are independent copies
    feynman["model_adapter"]["options"]["n"] = 4
    assert select_experiment(expanded, "fastsrb")["model_adapter"]["options"]["n"] == 3


def test_a_preset_catalog_in_the_template_is_kept():
    expanded = expand_suite({"suite": ["fastsrb"], "run": {"data_source": {"catalog": "my/repo:fastsrb@1"}, "model_adapter": {}}})
    assert expanded["experiments"]["fastsrb"]["data_source"]["catalog"] == "my/repo:fastsrb@1"


def test_list_form_and_errors():
    assert suite_catalogs(["a", "b", "a"]) == ["a", "b"]
    assert suite_catalogs("srbf") == list(SRBF_CATALOGS)
    with pytest.raises(ValueError, match="unknown suite"):
        suite_catalogs("nope")
    with pytest.raises(ValueError):
        suite_catalogs([])
    with pytest.raises(ValueError, match="not both"):
        expand_suite({"suite": "srbf", "run": {}, "experiments": {}})
    with pytest.raises(ValueError, match="run: template"):
        expand_suite({"suite": "srbf"})
    plain = {"run": {"data_source": {"catalog": "fastsrb"}}}
    assert expand_suite(plain) == plain


def test_loader_expands_a_file_with_sweeps(tmp_path):
    path = tmp_path / "suite.yaml"
    path.write_text(
        "suite: [fastsrb, feynman]\n"
        "run:\n"
        "  data_source: {sampling: {n_support: 8}}\n"
        "  model_adapter:\n"
        "    type: subprocess\n"
        "    worker: w.py\n"
        "    options: {choices: !sweep {name: ladder, values: [1, 2]}}\n"
        "  runner:\n"
        "    output: !sweep {name: ladder, values: ['{{ROOT}}/{catalog}_1.pkl', '{{ROOT}}/{catalog}_2.pkl']}\n"
    )
    raw = load_run_config(str(path))
    assert list(raw["experiments"]) == ["fastsrb", "feynman"]
    rungs = resolve_sweeps(select_experiment(raw, "feynman"))
    assert [labels for _, labels in rungs] == [{"ladder": 1}, {"ladder": 2}]
    assert [cfg["runner"]["output"] for cfg, _ in rungs] == ["{{ROOT}}/feynman_1.pkl", "{{ROOT}}/feynman_2.pkl"]
    assert rungs[1][0]["model_adapter"]["options"]["choices"] == 2


# ---- the benchmark a catalog belongs to: its metadata's `benchmark` -----------------------------------------------
def _catalog(tmp_path, name, **meta):
    import yaml

    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump({"metadata": {"name": name, **meta}, "expressions": {}}))
    return str(path)


def test_a_catalog_names_its_benchmark_in_its_metadata(tmp_path):
    from srbf.suites import catalog_benchmark

    assert catalog_benchmark(_catalog(tmp_path, "feynman-points", benchmark="feynman")) == "feynman"
    assert catalog_benchmark(_catalog(tmp_path, "nguyen")) is None                 # no field: no benchmark named
    assert catalog_benchmark({"type": "generative"}) is None                        # a generative catalog's config
    for bad in ("", "  ", 3, ["feynman"]):
        with pytest.raises(ValueError, match="non-empty string"):
            catalog_benchmark(_catalog(tmp_path, f"bad{bad!r}".replace(" ", "_"), benchmark=bad))


def test_catalogs_map_to_their_benchmarks_or_to_themselves(tmp_path, monkeypatch):
    from srbf.suites import catalog_benchmarks

    refs = {"feynman": _catalog(tmp_path, "feynman", benchmark="feynman"),
            "feynman-restricted": _catalog(tmp_path, "feynman-restricted", benchmark="feynman"),
            "nguyen": _catalog(tmp_path, "nguyen"),
            "gone": str(tmp_path / "gone.yaml")}
    said = []
    assert catalog_benchmarks(refs, warn=said.append) == {
        "feynman": "feynman", "feynman-restricted": "feynman", "nguyen": "nguyen", "gone": "gone"}
    assert len(said) == 1 and "'gone'" in said[0] and "a benchmark of its own" in said[0]
    # names alone are references too, and {{ROOT}} is expanded as a run expands it
    import srbf.paths as paths
    assert catalog_benchmarks([refs["nguyen"]]) == {refs["nguyen"]: refs["nguyen"]}
    monkeypatch.setattr(paths, "substitute_root_path", lambda p: p.replace("{{ROOT}}", str(tmp_path)))
    assert catalog_benchmarks({"f": "{{ROOT}}/feynman-restricted.yaml"}) == {"f": "feynman"}
