"""Inline ``!sweep`` config cross-products (the named-zip-axes design).

A ``!sweep`` YAML tag marks a value that varies across runs. Two forms:

- ``!sweep [v1, v2, ...]`` -- an ANONYMOUS axis: it forms its own dimension of the cross-product
  (grid) with every other anonymous sweep.
- ``!sweep {name: <axis>, values: [v1, v2, ...]}`` -- a NAMED axis: every ``!sweep`` sharing the same
  ``name`` advances together (element-wise ZIP), so they must be equal length. Different names form
  separate cross-product dimensions.

So the default is a grid product; sharing an axis name collapses those sweeps into one zipped
dimension. The scaling "choices ladder" is one named axis ``ladder`` carried by ``choices``,
``datasets_per_expression`` (a.k.a. ``problems_per_expression``), and the per-rung ``output`` path --
all zipped -- giving N runs with matched (choices, draws, output) tuples and no spurious product.

``resolve_sweeps(config)`` expands a config into ``[(resolved_config, axis_labels), ...]``. The draw
axis (``problems_per_expression``) is orthogonal: a sweep over hyperparameters never multiplies the
number of draws per expression (that is the source's own usage policy, applied per resolved run).
"""
from __future__ import annotations

import copy
from itertools import product
from typing import Any, Mapping

import yaml


class Sweep:
    """A swept value: ``values`` enumerated across runs; ``name`` (optional) zips co-named sweeps."""

    __slots__ = ("values", "name")

    def __init__(self, values: list[Any], name: str | None = None) -> None:
        if not isinstance(values, list) or len(values) == 0:
            raise ValueError("!sweep requires a non-empty list of values")
        self.values = values
        self.name = name

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"Sweep(name={self.name!r}, values={self.values!r})"


def _construct_sweep(loader: yaml.Loader, node: yaml.Node) -> Sweep:
    if isinstance(node, yaml.SequenceNode):
        return Sweep(loader.construct_sequence(node, deep=True))
    if isinstance(node, yaml.MappingNode):
        mapping = loader.construct_mapping(node, deep=True)
        if "values" not in mapping:
            raise ValueError("!sweep mapping form requires a 'values' key (e.g. {name: ladder, values: [...]})")
        name = mapping.get("name", mapping.get("axis"))
        return Sweep(mapping["values"], name=None if name is None else str(name))
    raise ValueError("!sweep must tag a sequence (anonymous axis) or a mapping {name?, values} (named axis)")


def register_sweep_yaml() -> None:
    """Register the ``!sweep`` tag on PyYAML's safe loaders (idempotent).

    ``flash_ansr.utils.config_io.load_config`` parses with ``yaml.safe_load`` (the default
    ``SafeLoader``), so registering here lets the shared loader build ``Sweep`` markers; configs
    without ``!sweep`` are unaffected.
    """
    yaml.SafeLoader.add_constructor("!sweep", _construct_sweep)
    yaml.FullLoader.add_constructor("!sweep", _construct_sweep)


def _collect(node: Any, out: list[Sweep]) -> None:
    if isinstance(node, Sweep):
        out.append(node)
    elif isinstance(node, Mapping):
        for v in node.values():
            _collect(v, out)
    elif isinstance(node, list):
        for v in node:
            _collect(v, out)


def _axis_key(sweep: Sweep, anon_index: int) -> str:
    return sweep.name if sweep.name is not None else f"\x00anon{anon_index}"


def _substitute(node: Any, picks: Mapping[int, int]) -> Any:
    """Deep-copy ``node`` replacing each Sweep with its value at the picked index for its axis."""
    if isinstance(node, Sweep):
        return copy.deepcopy(node.values[picks[id(node)]])
    if isinstance(node, Mapping):
        return {k: _substitute(v, picks) for k, v in node.items()}
    if isinstance(node, list):
        return [_substitute(v, picks) for v in node]
    return copy.deepcopy(node)


def resolve_sweeps(config: Mapping[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Expand ``!sweep`` markers into ``[(resolved_config, axis_labels), ...]``.

    Named axes zip (equal length required); anonymous axes form the cross-product. ``axis_labels``
    maps each axis name to the value selected for that run (anonymous axes are omitted from labels,
    since they have no stable name). A config with no sweeps yields a single ``(config, {})``.
    """
    sweeps: list[Sweep] = []
    _collect(config, sweeps)
    if not sweeps:
        return [(copy.deepcopy(dict(config)), {})]

    # Assign each Sweep an axis key; named axes are shared (and length-checked), anonymous are unique.
    axis_of: dict[int, str] = {}
    axis_len: dict[str, int] = {}
    axis_repr: dict[str, Sweep] = {}
    anon = 0
    for sweep in sweeps:
        if sweep.name is None:
            key = _axis_key(sweep, anon)
            anon += 1
        else:
            key = sweep.name
        axis_of[id(sweep)] = key
        if key in axis_len:
            if axis_len[key] != len(sweep.values):
                raise ValueError(
                    f"!sweep axis '{key}' has inconsistent lengths ({axis_len[key]} vs {len(sweep.values)}); "
                    f"co-named sweeps zip element-wise and must be equal length"
                )
        else:
            axis_len[key] = len(sweep.values)
            axis_repr[key] = sweep

    axis_keys = list(axis_len.keys())
    runs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for combo in product(*(range(axis_len[k]) for k in axis_keys)):
        index_by_axis = dict(zip(axis_keys, combo))
        picks = {sid: index_by_axis[axis_of[sid]] for sid in axis_of}
        resolved = _substitute(config, picks)
        labels = {
            key: axis_repr[key].values[index_by_axis[key]]
            for key in axis_keys
            if not key.startswith("\x00anon")
        }
        runs.append((resolved, labels))
    return runs


__all__ = ["Sweep", "register_sweep_yaml", "resolve_sweeps"]
