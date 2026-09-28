"""srbf's root directory and its config files.

srbf keeps models, results, adapters and environments under one directory, its root: the ``SRBF_ROOT``
environment variable, or the current directory when that is unset. In a config, ``{{ROOT}}`` stands for it.
"""
from __future__ import annotations

import os
from typing import Any, Callable

import yaml

#: The environment variable that names srbf's root.
ROOT_ENV_VAR = "SRBF_ROOT"

#: The token a config uses for the root.
ROOT_TOKEN = "{{ROOT}}"


def get_root() -> str:
    """The root ``{{ROOT}}`` stands for: ``$SRBF_ROOT`` when set to a non-empty value, else the current directory."""
    override = os.environ.get(ROOT_ENV_VAR)
    return os.path.abspath(override) if override else os.getcwd()


def substitute_root_path(path: str) -> str:
    """``path`` with every ``{{ROOT}}`` replaced by the root."""
    return path.replace(ROOT_TOKEN, get_root())


def normalize_path_preserve_leading_dot(path: str) -> str:
    """``os.path.normpath``, keeping a leading ``./`` (it marks a path relative to the config it appears in)."""
    starts_with_dot_sep = path.startswith(f".{os.sep}")
    normalized = os.path.normpath(path)
    if starts_with_dot_sep and not os.path.isabs(normalized) and not normalized.startswith("..") and normalized != ".":
        return f".{os.sep}{normalized}"
    return normalized


def _apply_on_nested(structure: Any, func: Callable[[Any], Any]) -> Any:
    """``func`` on every leaf of nested dicts and lists, in place (the containers keep their identity and type)."""
    items = enumerate(structure) if isinstance(structure, list) else structure.items() if isinstance(structure, dict) else ()
    for key, value in list(items):
        structure[key] = _apply_on_nested(value, func) if isinstance(value, (dict, list)) else func(value)
    return structure


def load_config(config: dict[str, Any] | str, resolve_paths: bool = True) -> Any:
    """A config: a mapping is returned as it is; a path (``{{ROOT}}`` replaced) is read as YAML.

    With ``resolve_paths``, a value that names another config relative to this one (a string starting with ``.``
    and ending in ``.yaml`` or ``.json``) is rewritten relative to this file's directory.
    """
    if not isinstance(config, str):
        return config
    config_path = substitute_root_path(config)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} not found.")
    if not os.path.isfile(config_path):
        raise ValueError(f"Config file {config_path} is not a valid file.")
    with open(config_path, "r") as fh:
        loaded = yaml.safe_load(fh)
    if not resolve_paths:
        return loaded
    base = os.path.dirname(config_path)

    def resolve(value: Any) -> Any:
        if isinstance(value, str) and value.startswith(".") and (value.endswith(".yaml") or value.endswith(".json")):
            return normalize_path_preserve_leading_dot(os.path.join(base, value))
        return value

    return _apply_on_nested(loaded, resolve)
