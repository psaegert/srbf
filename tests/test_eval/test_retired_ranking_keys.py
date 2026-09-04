"""A retired ranking key must fail loudly, never fall through to a default.

`parsimony` was never read by the flash_ansr adapter (it is a PySR key) and `length_penalty` was
renamed to `node_penalty` on 2026-09-04. Either one left in a config would be ignored and the run
would rank at whatever the default happens to be -- which is how every srbf run to date came to rank
at penalty 0.0 while its config said 0.05. These tests pin the loud failure.
"""
from __future__ import annotations

import pytest

from srbf.config import reject_retired_ranking_keys


@pytest.mark.parametrize("dead", ["length_penalty", "parsimony"])
def test_retired_key_raises_and_names_the_replacement(dead: str) -> None:
    with pytest.raises(ValueError) as excinfo:
        reject_retired_ranking_keys({dead: 0.05}, where="unit test")
    message = str(excinfo.value)
    assert dead in message
    assert "node_penalty" in message, "the error must say what to rename it to"
    assert "unit test" in message, "the error must say which layer carried the key"


def test_a_clean_config_passes() -> None:
    reject_retired_ranking_keys({"node_penalty": 0.05, "constants_penalty": 0.0}, where="unit test")


def test_a_zero_value_still_raises() -> None:
    """Presence is what matters, not the value: `length_penalty: 0.0` is still a silent no-op."""
    with pytest.raises(ValueError):
        reject_retired_ranking_keys({"length_penalty": 0.0}, where="unit test")


def test_the_flash_ansr_builder_checks_both_layers() -> None:
    """The adapter config and the nested evaluation config are both checked.

    The historical defect lived in the NESTED layer: the doctrine configs set `parsimony: 0.05`
    inside the evaluation config while the adapter read `length_penalty` from it.
    """
    import inspect

    from srbf import config as config_module

    source = inspect.getsource(config_module._build_flash_ansr_adapter)
    assert source.count("reject_retired_ranking_keys") == 2, (
        "both the adapter config and the resolved evaluation config must be checked"
    )
    assert "eval_cfg" in source.split("reject_retired_ranking_keys")[2], (
        "the nested evaluation config is the layer the historical defect lived in"
    )
