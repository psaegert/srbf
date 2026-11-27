"""Global pytest fixtures and dependency shims for the srbf test suite."""
from __future__ import annotations

import shutil
import sys
import types
from pathlib import Path

import pytest


def _ensure_module(name: str) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        sys.modules[name] = module
    return module


def _install_nesymres_stub() -> None:
    """Provide a lightweight stand-in for the optional nesymres dependency."""
    root = _ensure_module("nesymres")
    architectures = _ensure_module("nesymres.architectures")
    architectures.__dict__.setdefault("__path__", [])

    model_module = _ensure_module("nesymres.architectures.model")

    if not hasattr(model_module, "Model"):
        class _DummyModel:  # pragma: no cover - trivial shim
            pass

        model_module.Model = _DummyModel  # type: ignore[attr-defined]

    architectures.model = model_module  # type: ignore[attr-defined]
    root.architectures = architectures  # type: ignore[attr-defined]


_install_nesymres_stub()


# ---- engines with and without defined functions ---------------------------------------------------------------------
#: The config section that defines the square root and the reciprocal trigonometric functions, as the released engine
#: configs declare it. The small test engine's config may or may not carry it; the two fixtures below set it either way.
DEFINED_FUNCTIONS = {
    "sqrt": {"arity": 1, "definition": "rootn(_0, 2)"},
    "cot": {"arity": 1, "definition": "inv(tan(_0))"},
    "acot": {"arity": 1, "definition": "atan(inv(_0))"},
    "sec": {"arity": 1, "definition": "inv(cos(_0))"},
    "csc": {"arity": 1, "definition": "inv(sin(_0))"},
    "asec": {"arity": 1, "definition": "acos(inv(_0))"},
    "acsc": {"arity": 1, "definition": "asin(inv(_0))"},
}


def _small_engine_with(tmp_path_factory, defined: bool):
    """The small test engine (acj2-4-llm, its rules unchanged), its config with or without ``defined_functions:``."""
    import yaml
    from simplipy import SimpliPyEngine, asset_manager

    source = Path(asset_manager.get_path("acj2-4-llm", install=True)).parent
    target = tmp_path_factory.mktemp("defined" if defined else "undefined") / "engine"
    shutil.copytree(source, target)
    config = yaml.safe_load((target / "config.yaml").read_text())
    config.pop("defined_functions", None)
    if defined:
        config["defined_functions"] = DEFINED_FUNCTIONS
    (target / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    return SimpliPyEngine.from_config(str(target / "config.yaml"))


@pytest.fixture(scope="session")
def engine_defining_sqrt(tmp_path_factory):
    """An engine whose config defines ``sqrt`` (and ``cot``, ``sec``, ...): it reads, prices and evaluates them as
    written."""
    return _small_engine_with(tmp_path_factory, defined=True)


@pytest.fixture(scope="session")
def engine_without_definitions(tmp_path_factory):
    """An engine whose config defines no function: ``sqrt`` is no head of it, and srbf writes ``rootn(u, 2)``."""
    return _small_engine_with(tmp_path_factory, defined=False)
