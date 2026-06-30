"""!sweep: named-zip axes + anonymous grid product, draw-axis orthogonality, YAML tag."""
import pytest
import yaml

from srbf.sweep import Sweep, register_sweep_yaml, resolve_sweeps


def test_no_sweep_yields_single_run():
    cfg = {"a": 1, "b": {"c": [2, 3]}}
    runs = resolve_sweeps(cfg)
    assert len(runs) == 1
    resolved, labels = runs[0]
    assert resolved == cfg and labels == {}
    assert resolved is not cfg  # a copy, not the original


def test_anonymous_sweeps_form_a_grid_product():
    cfg = {"x": Sweep([1, 2, 3]), "y": Sweep(["a", "b"])}
    runs = resolve_sweeps(cfg)
    assert len(runs) == 6  # 3 x 2 cross-product
    combos = {(r["x"], r["y"]) for r, _ in runs}
    assert combos == {(1, "a"), (1, "b"), (2, "a"), (2, "b"), (3, "a"), (3, "b")}
    # anonymous axes carry no stable name -> no labels
    assert all(labels == {} for _, labels in runs)


def test_named_sweeps_zip_elementwise():
    # the scaling-ladder shape: choices + draws + output all zip on one axis -> matched tuples, NOT a grid
    cfg = {
        "run": {
            "model_adapter": {"choices": Sweep([1, 16, 256], name="ladder")},
            "data_source": {"problems_per_expression": Sweep([10, 10, 5], name="ladder")},
            "runner": {"output": Sweep(["c1.pkl", "c16.pkl", "c256.pkl"], name="ladder")},
        }
    }
    runs = resolve_sweeps(cfg)
    assert len(runs) == 3  # zipped, not 3*3*3
    triples = [
        (r["run"]["model_adapter"]["choices"],
         r["run"]["data_source"]["problems_per_expression"],
         r["run"]["runner"]["output"],
         labels["ladder"])
        for r, labels in runs
    ]
    assert triples == [(1, 10, "c1.pkl", 1), (16, 10, "c16.pkl", 16), (256, 5, "c256.pkl", 256)]


def test_named_and_anonymous_compose():
    # one named axis (len 2) x one anonymous axis (len 2) -> 4 runs
    cfg = {"a": Sweep([1, 2], name="L"), "b": Sweep([1, 2], name="L"), "c": Sweep(["x", "y"])}
    runs = resolve_sweeps(cfg)
    assert len(runs) == 4
    # within the named axis a and b stay matched; c varies independently
    for r, labels in runs:
        assert r["a"] == r["b"]                 # zipped on L
        assert labels == {"L": r["a"]}          # only the named axis is labelled


def test_named_axis_length_mismatch_raises():
    cfg = {"a": Sweep([1, 2, 3], name="L"), "b": Sweep([1, 2], name="L")}
    with pytest.raises(ValueError, match="inconsistent lengths"):
        resolve_sweeps(cfg)


def test_empty_sweep_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        Sweep([])


def test_yaml_tag_round_trips_both_forms():
    register_sweep_yaml()
    cfg = yaml.safe_load(
        "anon: !sweep [1, 2, 4]\n"
        "named: !sweep {name: ladder, values: [a, b]}\n"
    )
    assert isinstance(cfg["anon"], Sweep) and cfg["anon"].name is None and cfg["anon"].values == [1, 2, 4]
    assert isinstance(cfg["named"], Sweep) and cfg["named"].name == "ladder" and cfg["named"].values == ["a", "b"]
    # and they resolve: 3 (anon) x 2 (named) = 6
    assert len(resolve_sweeps(cfg)) == 6


def test_yaml_sweep_mapping_requires_values():
    register_sweep_yaml()
    with pytest.raises(Exception):  # noqa: B017 - yaml wraps the ValueError from the constructor
        yaml.safe_load("bad: !sweep {name: x}\n")
