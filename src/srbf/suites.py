"""Catalog suites and the ``suite:`` config shorthand.

A whole-suite config names a suite (or lists catalogs) and gives ONE ``run:`` template::

    suite: srbf                    # every srbf catalog; or a list: [fastsrb, feynman]
    run:
      data_source:
        sampling: {n_support: 512, n_validation: 512, noise: 0.0}
      model_adapter: {type: subprocess, worker: ..., python: ..., simplipy_engine: acj-5-4-llm}
      runner:
        output: '{{ROOT}}/results/evaluation/mymethod/{catalog}.pkl'

The loader expands it into ``experiments:`` with one entry per catalog: ``data_source.catalog`` is
filled in and every ``{catalog}`` in a string (outputs, logs, sweep values) is substituted. So
``srbf run -c`` runs the suite, ``--experiment <catalog>`` selects one, and ``!sweep`` inside the
template applies to every experiment. The reference configs under ``configs/evaluation/scaling``
are the expanded form of the same thing.

A catalog belongs to a benchmark, the unit results are averaged over (docs/averaging.md): one benchmark can be split
over several catalogs for hosting or format reasons, and each of them names it in the ``benchmark`` field of its
metadata. :func:`catalog_benchmark` reads that field, and :func:`catalog_benchmarks` maps catalogs to their benchmarks,
a catalog without the field being a benchmark of its own.
"""
from __future__ import annotations

import copy
import functools
import os
import types
from typing import Any, Callable, Iterable, Mapping

SRBF_CATALOGS: tuple[str, ...] = (
    'fastsrb', 'feynman', 'feynman-bonus', 'srsd-dummy',
    'erbench-syneq', 'erbench-densities', 'erbench-phybench',
    'soose-fc', 'soose-nc', 'soose-wc',
    'physo-astro', 'physo-class',
    'nguyen', 'keijzer', 'korns', 'koza', 'livermore', 'livermore2', 'vladislavleva', 'jin', 'neat',
    'pagie', 'poly', 'nonic', 'sine', 'meier', 'r-rationals', 'constant', 'grammarvae',
)
"""The srbf suite: the catalogs every published number covers, in the reference configs' order."""

SUITES: dict[str, tuple[str, ...]] = {"srbf": SRBF_CATALOGS}


def suite_catalogs(suite: Any) -> list[str]:
    """``suite: srbf`` -> the suite's catalogs; ``suite: [a, b]`` -> that list (order kept, no repeats)."""
    if isinstance(suite, str):
        if suite not in SUITES:
            raise ValueError(f"unknown suite {suite!r}; known suites: {', '.join(sorted(SUITES))}, or list catalogs")
        return list(SUITES[suite])
    if isinstance(suite, (list, tuple)) and suite and all(isinstance(item, str) for item in suite):
        return list(dict.fromkeys(suite))
    raise ValueError("suite: must be a suite name or a non-empty list of catalog names")


def _fill(node: Any, catalog: str) -> Any:
    from srbf.sweep import Sweep

    if isinstance(node, Sweep):
        return Sweep([_fill(value, catalog) for value in node.values], name=node.name)
    if isinstance(node, Mapping):
        return {key: _fill(value, catalog) for key, value in node.items()}
    if isinstance(node, list):
        return [_fill(value, catalog) for value in node]
    if isinstance(node, str):
        return node.replace("{catalog}", catalog)
    return copy.deepcopy(node)


def expand_suite(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Expand the ``suite:`` shorthand into ``experiments:``; a config without it is returned as is."""
    if "suite" not in raw:
        return dict(raw)
    if "experiments" in raw:
        raise ValueError("a config carries either suite: or experiments:, not both")
    template = raw.get("run")
    if not isinstance(template, Mapping):
        raise ValueError("suite: needs one run: template (data_source / model_adapter / runner)")
    experiments: dict[str, Any] = {}
    for catalog in suite_catalogs(raw["suite"]):
        experiment = _fill(template, catalog)
        data_source = dict(experiment.get("data_source") or {})
        data_source.setdefault("catalog", catalog)
        experiment["data_source"] = data_source
        experiments[catalog] = experiment
    expanded = {key: value for key, value in raw.items() if key not in ("suite", "run")}
    expanded["experiments"] = experiments
    return expanded


#: The catalog metadata field that names the benchmark a catalog belongs to.
BENCHMARK_FIELD = "benchmark"


@functools.lru_cache(maxsize=None)
def _read_metadata(ref: str) -> Mapping[str, Any]:
    from symbolic_data import load_catalog

    from srbf.paths import substitute_root_path

    return types.MappingProxyType(dict(load_catalog(os.path.expanduser(substitute_root_path(ref))).meta or {}))


def _metadata(catalog: Any) -> Mapping[str, Any]:
    """A catalog's metadata block: a reference is read with symbolic-data (``{{ROOT}}`` and ``~`` expanded, as a run
    expands them; once per process), a loaded catalog has its own, and a mapping (a generative catalog's config) has
    none."""
    if not isinstance(catalog, str):
        meta = getattr(catalog, "meta", None)
        return meta if isinstance(meta, Mapping) else {}
    return _read_metadata(catalog)


def _named_benchmark(meta: Mapping[str, Any], catalog: Any) -> str | None:
    value = meta.get(BENCHMARK_FIELD)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"catalog {catalog!r}: its metadata's {BENCHMARK_FIELD!r} must be a non-empty string, not {value!r}")
    return value.strip()


def catalog_benchmark(catalog: Any) -> str | None:
    """The benchmark ``catalog`` belongs to: the ``benchmark`` field of its metadata, or None when it names none.

    ``catalog`` is what a run's ``data_source.catalog`` holds: a reference (``name``, ``name@version``,
    ``repo_id:name``, or a path), read with symbolic-data, a loaded catalog, or a mapping (a generative catalog, which
    has no such metadata). Raises when the catalog cannot be read, or when the field is not a non-empty string."""
    return _named_benchmark(_metadata(catalog), catalog)


def catalog_benchmarks(catalogs: Mapping[str, Any] | Iterable[str], *,
                       warn: Callable[[str], None] | None = None) -> dict[str, str]:
    """{catalog: benchmark} for ``catalogs`` (names, or {name: reference} where the reference is not the name itself):
    each catalog's benchmark from its metadata (:func:`catalog_benchmark`), else the catalog's own name. A catalog
    whose metadata cannot be read is a benchmark of its own too, and is reported through ``warn``; a field that is not
    a non-empty string raises."""
    refs = dict(catalogs) if isinstance(catalogs, Mapping) else {name: name for name in catalogs}
    out: dict[str, str] = {}
    for name, ref in refs.items():
        try:
            meta = _metadata(ref)
        except Exception as exc:  # noqa: BLE001 - an unreadable catalog keeps its own name, and says so
            if warn is not None:
                warn(f"cannot read the metadata of catalog {name!r} ({type(exc).__name__}: {exc}); it is a benchmark of its own")
            meta = {}
        out[name] = _named_benchmark(meta, ref) or name
    return out


__all__ = ["BENCHMARK_FIELD", "SRBF_CATALOGS", "SUITES", "catalog_benchmark", "catalog_benchmarks", "expand_suite",
           "suite_catalogs"]
