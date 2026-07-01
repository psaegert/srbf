"""A3: run provenance hashes the DATA CATALOG.

Before 0.5.4, `_resolve_inputs` branched on the removed pre-0.5 `benchmark_path`/`dataset`/
`skeleton_list` keys, so the catalog was never hashed and dataset provenance silently captured
nothing. A local catalog artifact must now be hashed; a bare NAME/HF ref is captured verbatim by
`config_sha` (no dead keys).
"""
import yaml

from srbf.provenance import _resolve_inputs


def _write_cfg(tmp_path, data_source):
    cfg = {"run": {"data_source": data_source, "model_adapter": {}, "runner": {}}}
    path = tmp_path / "run.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle)
    return str(path)


def test_local_catalog_file_is_hashed(tmp_path):
    catf = tmp_path / "cat.yaml"
    catf.write_text("metadata:\n  name: t\n  version: 1\nexpressions: {}\n", encoding="utf-8")
    files = _resolve_inputs(_write_cfg(tmp_path, {"catalog": str(catf)}), None)
    assert files.get("catalog") == str(catf)


def test_local_catalog_dir_hashes_its_index(tmp_path):
    catdir = tmp_path / "saved"
    catdir.mkdir()
    (catdir / "catalog.yaml").write_text("metadata: {name: t, version: 1}\n", encoding="utf-8")
    files = _resolve_inputs(_write_cfg(tmp_path, {"catalog": str(catdir)}), None)
    assert files.get("catalog/catalog.yaml") == str(catdir / "catalog.yaml")


def test_name_catalog_adds_no_file_and_no_dead_keys(tmp_path):
    # A bare NAME (HF ref) has no local artifact -> captured by config_sha; no old-schema keys appear.
    files = _resolve_inputs(_write_cfg(tmp_path, {"catalog": "v23-val"}), None)
    assert "catalog" not in files
    assert not any(k.split("/")[0] in {"benchmark", "dataset_cfg", "skeleton_list_pin"} for k in files)
