"""Schema gate for the shipped eval configs: catalog data_source + valid adapter + resolvable !sweep.

Replaces the old key-presence check (which asserted the REMOVED `data_source.type`/`model_adapter.type`
schema and so silently stayed green over un-migrated configs). This validates the 0.5.0 catalog schema
and that every config's ``!sweep`` ladder resolves -- and that the banned term / removed schema is gone.
"""
from pathlib import Path

import pytest
import yaml

from srbf.sweep import Sweep, register_sweep_yaml, resolve_sweeps

EVAL_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "evaluation"
VALID_ADAPTERS = {"flash_ansr", "pysr", "nesymres", "e2e", "lample_charton", "brute_force"}
VALID_CATALOGS = {"v23-val", "fastsrb"}
BANNED = ["skeleton_pool", "skeleton dataset", "skeleton_dataset", "type: fastsrb",
          "benchmark_path", "datasets_per_expression", "noise_level", "support_points"]

register_sweep_yaml()


def _has_sweep(node) -> bool:
    if isinstance(node, Sweep):
        return True
    if isinstance(node, dict):
        return any(_has_sweep(v) for v in node.values())
    if isinstance(node, list):
        return any(_has_sweep(v) for v in node)
    return False


@pytest.mark.parametrize("config_path", sorted(EVAL_CONFIG_DIR.glob("**/*.yaml")), ids=lambda p: str(p.name))
def test_eval_config_uses_catalog_schema_and_resolves(config_path):
    text = config_path.read_text(encoding="utf-8")
    for token in BANNED:
        assert token not in text, f"{config_path}: removed schema / banned term {token!r} present"

    config = yaml.safe_load(text) or {}
    runs = resolve_sweeps(config)
    assert runs, f"{config_path}: no runs"

    for resolved, _labels in runs:
        run = resolved.get("run", resolved)
        ds = run["data_source"]
        ma = run["model_adapter"]
        assert ds.get("catalog") in VALID_CATALOGS, f"{config_path}: bad catalog {ds.get('catalog')!r}"
        assert ma.get("type") in VALID_ADAPTERS, f"{config_path}: bad adapter {ma.get('type')!r}"
        sampling = ds.get("sampling", {})
        assert {"n_support", "n_validation", "problems_per_expression"} <= set(sampling), \
            f"{config_path}: sampling missing keys ({sorted(sampling)})"
        assert not _has_sweep(resolved), f"{config_path}: unresolved !sweep after expansion"
        output = run.get("runner", {}).get("output")
        assert isinstance(output, str) and output.endswith(".pkl"), f"{config_path}: bad output {output!r}"
