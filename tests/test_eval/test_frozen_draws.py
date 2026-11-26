"""Frozen draws: every method, budget, shard and machine reads one render of each problem's data and never draws.

The renders here are written by `_render`, which follows the format symbolic-data writes (``symbolic-data/frozen-draws``
version 1), so that these tests run against any symbolic-data release; `test_reads_a_render_written_by_symbolic_data`
reads one written by symbolic-data itself where it is installed.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys

import numpy as np
import pytest
import yaml

from srbf import Benchmark
from srbf import config as run_config
from srbf.data_sources import CatalogSource
from srbf.frozen_draws import FrozenDrawsError, check_resume, noised
from srbf.shards import merge_shards

SETTINGS = {"n_support": 6, "n_validation": 4, "layout": "random", "max_trials": 100, "seed_root": "test",
            "simplipy_engine": "acj-4"}


def _sha(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _toy_problems(seed: int) -> list[tuple[dict, dict]]:
    """(record, arrays) per problem: two formula problems, a data problem with stored noisy targets, a placeholder."""
    rng = np.random.default_rng(seed)
    x0, xv0 = rng.uniform(1, 2, (6, 1)), rng.uniform(1, 2, (4, 1))
    x1, xv1 = rng.uniform(1, 2, (6, 2)), rng.uniform(1, 2, (4, 2))
    xd = rng.uniform(0, 1, (6, 1))

    def formula(eq_id, row, x, xv, f, expression, meta=None):
        record = {"row": row, "eq_id": eq_id, "kind": "formula", "skeleton": ["*", "<constant>", "x1"], "expression": expression,
                  "constants": [2.0], "variables": [f"x{i + 1}" for i in range(x.shape[1])], "complexity": len(expression),
                  "gt_kind": "exact", "noise": 0.0, "meta": meta or {}, "is_placeholder": False, "placeholder_reason": None}
        arrays = {"x_support": x, "y_support": f(x), "x_validation": xv, "y_validation": f(xv),
                  "noise_support": rng.standard_normal((6, 1)), "noise_validation": rng.standard_normal((4, 1))}
        return record, arrays

    data_record = {"row": 2, "eq_id": "d", "kind": "data", "skeleton": None, "expression": None, "constants": [],
                   "variables": ["x1"], "complexity": None, "gt_kind": "none", "noise": 0.0,
                   "meta": {"test_split": "none", "n_fit_pool": 40, "n_test_pool": 0}, "is_placeholder": False,
                   "placeholder_reason": None}
    data_arrays = {"x_support": xd, "y_support": 3 * xd, "x_validation": np.empty((0, 1)), "y_validation": np.empty((0, 1)),
                   "y_support_noisy": 3 * xd + 0.25}
    placeholder = {"row": 3, "eq_id": "p", "kind": "formula", "skeleton": None, "expression": None, "constants": [],
                   "variables": ["x1"], "complexity": None, "gt_kind": "exact", "noise": 0.0, "meta": {},
                   "is_placeholder": True, "placeholder_reason": "max_trials_exhausted"}
    c = 1.0 + seed / 10
    return [formula("a", 0, x0, xv0, lambda x: 2 * x, ["*", "2", "x1"]),
            formula("b", 1, x1, xv1, lambda x: c * x[:, :1] + x[:, 1:], ["+", "*", repr(c), "x1", "x2"],
                    meta={"per_run_values": {"c": c}}),
            (data_record, data_arrays), (placeholder, {})]


def _render(root, catalog_path, *, name="toy", repeats=(1,), seed=0) -> dict:
    """A render as symbolic-data writes it: <root>/<name>/repeat-<r>.npz and <root>/manifest.json."""
    identity = {"name": name, "version": 1, "source": catalog_path.name, "source_sha256": _sha(catalog_path)}
    (root / name).mkdir(parents=True, exist_ok=True)
    files = {}
    for repeat in repeats:
        problems = _toy_problems(seed=repeat + seed)
        arrays = {f"p{i}__{key}": value for i, (_, a) in enumerate(problems) for key, value in a.items()}
        meta = {"format": "symbolic-data/frozen-draws", "format_version": 1, "catalog": identity, "repeat": repeat,
                "settings": SETTINGS, "problems": [record for record, _ in problems]}
        path = root / name / f"repeat-{repeat}.npz"
        np.savez(path, _meta=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8), **arrays)
        files[str(repeat)] = {"path": f"{name}/{path.name}", "bytes": path.stat().st_size, "sha256": _sha(path)}
    manifest = {"format": "symbolic-data/frozen-draws", "format_version": 1, "settings": SETTINGS, "repeats": list(repeats),
                "rendered_with": {}, "catalogs": {name: {**identity, "problems": 4, "kinds": {}, "ids": None,
                                                         "placeholders": {}, "files": files}}}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return manifest


@pytest.fixture
def catalog(tmp_path):
    path = tmp_path / "toy.yaml"
    path.write_text(yaml.safe_dump({"metadata": {"name": "toy", "version": 1},
                                    "expressions": {"a": {"prepared": "2 * v1", "vars": {"v1": {"sample_range": [1, 2], "sample_type": ["uni", "pos"]}}}}}))
    return path


@pytest.fixture
def draws(tmp_path, catalog):
    _render(tmp_path / "draws", catalog)
    return tmp_path / "draws"


def _source(catalog, draws, **kwargs):
    sampling = kwargs.pop("sampling", {"n_support": 6, "n_validation": 4, "noise": 0.0})
    return CatalogSource.from_catalog(str(catalog), sampling=sampling, draws={"root": str(draws), **kwargs.pop("spec", {})}, **kwargs)


def test_reads_every_problem_as_rendered_and_records_the_draws(catalog, draws):
    source = _source(catalog, draws)
    assert source.size_hint() == 4
    samples = list(source)
    rendered = _toy_problems(seed=1)
    sha = _sha(draws / "toy" / "repeat-1.npz")
    for index, (sample, (record, arrays)) in enumerate(zip(samples, rendered)):
        meta = sample.metadata
        assert meta["eval_row_index"] == index and meta["benchmark_eq_id"] == record["eq_id"]
        assert meta["draws_sha256"] == sha and meta["draws_repeat"] == 1
        if record["is_placeholder"]:
            assert sample.is_placeholder and meta["placeholder_reason"] == "max_trials_exhausted"
            continue
        np.testing.assert_array_equal(sample.x_support, arrays["x_support"])
        np.testing.assert_array_equal(sample.y_validation, arrays["y_validation"])
        assert meta["ground_truth_prefix"] == record["expression"] and meta["gt_kind"] == record["gt_kind"]
    np.testing.assert_array_equal(samples[0].y_support_noisy, rendered[0][1]["y_support"])     # noise 0: clean
    np.testing.assert_array_equal(samples[2].y_support_noisy, rendered[2][1]["y_support_noisy"])  # stored noisy targets
    assert samples[1].metadata["constants"] == [np.asarray([2.0])]
    identity = source.draws_identity()
    assert identity["sha256"] == sha and identity["repeat"] == 1 and identity["catalog"] == "toy"
    assert identity["manifest_sha256"] == _sha(draws / "manifest.json")


def test_a_noise_level_scales_the_stored_vector_on_formula_problems_only(catalog, draws):
    rendered = _toy_problems(seed=1)
    for level in (0.01, 0.1):
        samples = list(_source(catalog, draws, sampling={"noise": level}))
        for i in (0, 1):
            arrays = rendered[i][1]
            for side, noisy, vector in (("y_support", samples[i].y_support_noisy, "noise_support"),
                                        ("y_validation", samples[i].y_validation_noisy, "noise_validation")):
                expected = arrays[side] + level * float(np.std(arrays[side])) * arrays[vector]
                np.testing.assert_array_equal(noisy, expected)
            np.testing.assert_array_equal(samples[i].y_support, arrays["y_support"])              # clean targets stay
            assert samples[i].metadata["noise_level"] == level
        np.testing.assert_array_equal(samples[2].y_support_noisy, rendered[2][1]["y_support_noisy"])  # data: as rendered
        assert samples[2].metadata["noise_level"] == 0.0


def test_the_noise_is_symbolic_datas_scalar_noise_with_the_vector_stored():
    from symbolic_data.source import _inject_noise
    y = np.linspace(-2.0, 5.0, 24).reshape(-1, 1)
    for level in (0.0, 0.05, 1.0):
        vector = np.random.default_rng(3).standard_normal(y.shape)
        np.testing.assert_array_equal(noised(y, level, vector), _inject_noise(y, level, np.random.default_rng(3)))
    np.testing.assert_array_equal(noised(np.ones((4, 1)), 0.1, np.ones((4, 1))), np.ones((4, 1)))   # a constant side
    assert noised(np.empty((0, 1)), 0.1, np.empty((0, 1))).shape == (0, 1)


def test_shards_read_the_rows_of_the_whole(catalog, draws):
    whole = {s.metadata["eval_row_index"]: s for s in _source(catalog, draws)}
    for index in range(2):
        for sample in _source(catalog, draws, shard=(index, 2)):
            row = sample.metadata["eval_row_index"]
            assert row % 2 == index
            np.testing.assert_array_equal(sample.x_support, whole[row].x_support)
            assert sample.metadata["draws_sha256"] == whole[row].metadata["draws_sha256"]


def _config(tmp_path, catalog, draws, output, **data_source):
    config = {"run": {"data_source": {"catalog": str(catalog), "sampling": {"n_support": 6, "n_validation": 4, "noise": 0.0},
                                      "draws": {"root": str(draws), "repeat": 1}, **data_source},
                      "model_adapter": {"type": "subprocess", "worker": "example", "python": sys.executable,
                                        "simplipy_engine": "acj-4", "config_provenance": "harness_tuned", "timeout": 120},
                      "runner": {"output": str(output), "save_every": 2}}}
    path = tmp_path / f"{output.stem}.yaml"
    path.write_text(yaml.safe_dump(config))
    return str(path)


def _load(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def test_two_runs_read_identical_points_and_record_the_draws(tmp_path, catalog, draws):
    outputs = [tmp_path / "out" / "first.pkl", tmp_path / "out" / "second.pkl"]
    for output, options in zip(outputs, ({}, {"ridge": 0.5})):
        config = _config(tmp_path, catalog, draws, output)
        raw = yaml.safe_load(open(config))
        raw["run"]["model_adapter"]["options"] = options       # another method configuration, the same data
        open(config, "w").write(yaml.safe_dump(raw))
        (benchmark,) = Benchmark.runs_from_config(config)
        benchmark.run(verbose=False, progress=False)
    first, second = (_load(p) for p in outputs)
    for key in ("x", "y", "y_noisy", "x_val", "y_val", "y_noisy_val"):
        for a, b in zip(first[key], second[key]):
            np.testing.assert_array_equal(a, b)
    sha = _sha(draws / "toy" / "repeat-1.npz")
    assert first["draws_sha256"] == [sha] * 4 and first["draws_repeat"] == [1] * 4
    meta = first["__meta__"]
    assert meta["draws"]["sha256"] == sha and meta["draws"]["repeat"] == 1
    assert meta["inputs"]["catalog_draws"]["sha256"] == sha
    from srbf.reporting import pairing_fingerprint
    assert pairing_fingerprint(first)["data_inputs"]["catalog_draws"] == sha


def test_a_result_file_from_other_draws_is_not_continued(tmp_path, catalog, draws):
    output = tmp_path / "out" / "run.pkl"
    config = _config(tmp_path, catalog, draws, output)
    (benchmark,) = Benchmark.runs_from_config(config, limit_override=2)
    benchmark.run(verbose=False, progress=False)
    (again,) = Benchmark.runs_from_config(config)                  # the same draws: resumes
    assert again.existing_results == 2
    other = tmp_path / "other"
    _render(other, catalog, seed=5)                                 # another render of the same catalog
    with pytest.raises(FrozenDrawsError, match="this run reads"):
        Benchmark.runs_from_config(_config(tmp_path, catalog, other, output))
    with pytest.raises(FrozenDrawsError, match="drawn afresh"):     # drawn afresh into a file of frozen rows
        check_resume(_load(output), None)


def test_check_resume():
    check_resume({}, {"sha256": "a"})
    check_resume({"x": [1, 2]}, None)                                # no draws on either side
    check_resume({"draws_sha256": ["a", "a"]}, {"sha256": "a"})
    with pytest.raises(FrozenDrawsError):
        check_resume({"draws_sha256": ["a", "b"]}, {"sha256": "a"})
    with pytest.raises(FrozenDrawsError, match="drawn afresh"):
        check_resume({"x": [1]}, {"sha256": "a"})


def test_merge_refuses_shards_of_different_draws(tmp_path):
    from srbf.store import ResultStore
    paths = []
    for index, sha in enumerate(("a", "b")):
        path = tmp_path / f"run.shard-{index}-of-2.pkl"
        ResultStore({"eval_row_index": [index], "draws_sha256": [sha]}).save(path, meta={"shard": {"index": index, "count": 2}})
        paths.append(str(path))
    with pytest.raises(ValueError, match="different frozen draws"):
        merge_shards(paths, str(tmp_path / "run.pkl"))


@pytest.mark.parametrize("change, message", [
    (lambda root, catalog: (root / "manifest.json").unlink(), "missing"),
    (lambda root, catalog: (root / "toy" / "repeat-1.npz").unlink(), "missing"),
    (lambda root, catalog: catalog.write_text(catalog.read_text() + "\n# another version\n"), "holds no draws of catalog"),
    (lambda root, catalog: (root / "toy" / "repeat-1.npz").write_bytes(
        (root / "toy" / "repeat-1.npz").read_bytes().replace(b"x_support", b"x_suppore", 1)), "differs from the manifest"),
])
def test_absent_or_altered_draws_are_refused(catalog, draws, change, message):
    change(draws, catalog)
    with pytest.raises(FrozenDrawsError, match=message):
        _source(catalog, draws)


def test_a_file_that_names_another_repeat_is_refused(tmp_path, catalog):
    root = tmp_path / "two"
    _render(root, catalog, repeats=(1, 2))
    (root / "toy" / "repeat-1.npz").write_bytes((root / "toy" / "repeat-2.npz").read_bytes())
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["catalogs"]["toy"]["files"]["1"]["sha256"] = _sha(root / "toy" / "repeat-1.npz")
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(FrozenDrawsError, match="another catalog, repeat or settings"):
        _source(catalog, root)
    assert _source(catalog, root, spec={"repeat": 2}).draws_identity()["repeat"] == 2


@pytest.mark.parametrize("kwargs, message", [
    ({"spec": {"repeat": 2}}, "not 2"),
    ({"spec": {"repeat": 0}}, "repeat must be an integer"),
    ({"spec": {"manifest_sha256": "0" * 64}}, "pinned"),
    ({"spec": {"seed": 3}}, "unknown keys"),
    ({"sampling": {"n_support": 512}}, "n_support is 512"),
    ({"sampling": {"problems_per_expression": 10}}, "one problem per expression"),
    ({"sampling": {"method": "random_without_replacement"}}, "render's order"),
    ({"sampling": {"noise": {"p_clean": 0.3}}}, "number >= 0"),
    ({"sampling": {"noise": -0.1}}, "number >= 0"),
    ({"sampling": {"size": 10}}, "do not apply"),
    ({"holdouts": [{"filter": {"finite": True}}]}, "holdouts"),
])
def test_a_run_that_asks_for_other_data_than_the_render_is_refused(catalog, draws, kwargs, message):
    with pytest.raises(FrozenDrawsError, match=message):
        _source(catalog, draws, **kwargs)


def test_the_manifest_can_be_pinned_and_an_inline_catalog_is_refused(catalog, draws):
    assert _source(catalog, draws, spec={"manifest_sha256": _sha(draws / "manifest.json")}).size_hint() == 4
    with pytest.raises(FrozenDrawsError, match="inline catalog"):
        CatalogSource.from_catalog({"type": "lample_charton"}, draws={"root": str(draws)})


def test_the_config_expands_the_root(tmp_path, monkeypatch):
    monkeypatch.setenv("SRBF_ROOT", str(tmp_path))
    seen = {}
    monkeypatch.setattr(run_config.CatalogSource, "from_catalog", classmethod(lambda cls, **kwargs: seen.update(kwargs)))
    run_config.build_catalog_source({"catalog": "fastsrb", "draws": {"root": "{{ROOT}}/data/draws", "repeat": 1}},
                                    target_size=None, skip=0)
    assert seen["draws"] == {"root": f"{tmp_path}/data/draws", "repeat": 1}
    run_config.build_catalog_source({"catalog": "fastsrb"}, target_size=None, skip=0)
    assert seen["draws"] is None


def test_reads_a_render_written_by_symbolic_data(tmp_path):
    draws_module = pytest.importorskip("symbolic_data.draws")
    path = tmp_path / "toy.yaml"
    unit = {"sample_range": [1.0, 2.0], "sample_type": ["uni", "pos"]}
    path.write_text(yaml.safe_dump({"metadata": {"name": "toy", "version": 1}, "expressions": {
        "a": {"prepared": "v1 * v2 + 2.5", "vars": {"v1": dict(unit), "v2": dict(unit)}},
        "c": {"prepared": "1.5 * v1", "vars": {"v1": dict(unit)}, "per_run": {"params": {"c1": dict(unit)}, "prepared": "c1 * v1"}}}}))
    settings = draws_module.DrawsSettings(n_support=16, n_validation=8, simplipy_engine="acj-4")
    draws_module.render([str(path)], tmp_path / "draws", settings, log=lambda _: None)
    samples = list(CatalogSource.from_catalog(str(path), sampling={"n_support": 16, "noise": 0.1}, draws={"root": str(tmp_path / "draws")}))
    meta = draws_module.read_meta(tmp_path / "draws" / "toy" / "repeat-1.npz")
    with np.load(tmp_path / "draws" / "toy" / "repeat-1.npz") as data:
        for index, sample in enumerate(samples):
            np.testing.assert_array_equal(sample.x_support, data[f"p{index}__x_support"])
            np.testing.assert_array_equal(sample.y_support_noisy, noised(data[f"p{index}__y_support"], 0.1, data[f"p{index}__noise_support"]))
            assert sample.metadata["ground_truth_prefix"] == meta["problems"][index]["expression"]
    c1 = meta["problems"][1]["meta"]["per_run_values"]["c1"]
    np.testing.assert_allclose(samples[1].y_validation.ravel(), c1 * samples[1].x_validation[:, 0])
