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
a catalog without the field being a benchmark of its own. Within its benchmark each problem belongs to a subset, the
``subset`` field of the problem (the authors' part of the benchmark, or the part one catalog holds); a problem without
it is in the subset named after its catalog (:func:`catalog_subset_rows`, :func:`catalog_subsets`).
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
#: The field of a catalog's problem (a yaml entry, a frozen problem's meta) that names the subset of its benchmark the
#: problem belongs to: the part of the benchmark its authors published it in, or the part one catalog holds.
SUBSET_FIELD = "subset"


@functools.lru_cache(maxsize=None)
def _read_catalog(ref: str) -> tuple[Mapping[str, Any], tuple[Any, ...]]:
    """A catalog's metadata block and the subset field of each of its problems, in row order; read once per process."""
    from symbolic_data import load_catalog

    from srbf.paths import substitute_root_path

    catalog = load_catalog(os.path.expanduser(substitute_root_path(ref)))
    return types.MappingProxyType(dict(catalog.meta or {})), tuple(_problem_subsets(catalog))


def _read_metadata(ref: str) -> Mapping[str, Any]:
    return _read_catalog(ref)[0]


def _problem_subsets(catalog: Any) -> list[Any]:
    """The subset field of each problem of a loaded catalog, in row order: a frozen catalog's problems' meta, a
    declarative catalog's entries (None where a problem names none)."""
    if getattr(catalog, "frozen", False):
        return [(getattr(p, "meta", None) or {}).get(SUBSET_FIELD) for p in getattr(catalog, "problems", None) or []]
    entries = getattr(catalog, "entries", None)
    return [(getattr(e, "meta", None) or {}).get(SUBSET_FIELD) for e in entries.values()] if isinstance(entries, Mapping) else []


def _metadata(catalog: Any) -> Mapping[str, Any]:
    """A catalog's metadata block: a reference is read with symbolic-data (``{{ROOT}}`` and ``~`` expanded, as a run
    expands them; once per process), a loaded catalog has its own, and a mapping (a generative catalog's config) has
    none."""
    if not isinstance(catalog, str):
        meta = getattr(catalog, "meta", None)
        return meta if isinstance(meta, Mapping) else {}
    return _read_metadata(catalog)


def _subset_fields(catalog: Any) -> tuple[Any, ...]:
    """The subset field of each problem of ``catalog`` (a reference, a loaded catalog, or a generative config)."""
    if isinstance(catalog, str):
        return _read_catalog(catalog)[1]
    return tuple(_problem_subsets(catalog))


def _named_benchmark(meta: Mapping[str, Any], catalog: Any) -> str | None:
    value = meta.get(BENCHMARK_FIELD)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"catalog {catalog!r}: its metadata's {BENCHMARK_FIELD!r} must be a non-empty string, not {value!r}")
    return value.strip()


def _named_subset(value: Any, catalog: Any, name: str) -> str:
    """A problem's subset: its field, else the catalog's name. A subset is a non-empty string without "/" (the results
    site keys a catalog's cells per subset as "<catalog>/<subset>")."""
    if value is None:
        return name
    if not isinstance(value, str) or not value.strip() or "/" in value:
        raise ValueError(f"catalog {catalog!r}: a problem's {SUBSET_FIELD!r} must be a non-empty string without '/', "
                         f"not {value!r}")
    return value.strip()


def catalog_benchmark(catalog: Any) -> str | None:
    """The benchmark ``catalog`` belongs to: the ``benchmark`` field of its metadata, or None when it names none.

    ``catalog`` is what a run's ``data_source.catalog`` holds: a reference (``name``, ``name@version``,
    ``repo_id:name``, or a path), read with symbolic-data, a loaded catalog, or a mapping (a generative catalog, which
    has no such metadata). Raises when the catalog cannot be read, or when the field is not a non-empty string."""
    return _named_benchmark(_metadata(catalog), catalog)


def catalog_subset_rows(catalog: Any, name: str | None = None) -> list[str]:
    """The subset of each problem of ``catalog``, in row order (a run's ``eval_row_index``): the problem's ``subset``
    field, else the catalog's name (``name``, by default the reference itself). A generative catalog lists no problems
    (every row is then in the subset named after it). Raises when the catalog cannot be read, or when a field is not a
    non-empty string without "/"."""
    label = name if name is not None else str(catalog)
    return [_named_subset(v, catalog, label) for v in _subset_fields(catalog)]


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


class UnreadableCatalogs(ValueError):
    """Catalogs whose benchmark and subsets cannot be read, where falling back to their names is not allowed."""


def catalog_structure(catalogs: Mapping[str, Any] | Iterable[str], *, warn: Callable[[str], None] | None = None,
                      strict: bool = False) -> tuple[dict[str, str], dict[str, list[str] | None]]:
    """({catalog: benchmark}, {catalog: the subset of each of its problems}) for ``catalogs``, each catalog read once:
    :func:`catalog_benchmarks` and :func:`catalog_subsets` together. A catalog that cannot be read is reported once and
    is a benchmark of its own, one subset (None); with ``strict``, :class:`UnreadableCatalogs` names every such catalog
    instead (what publishes must not guess)."""
    refs = dict(catalogs) if isinstance(catalogs, Mapping) else {name: name for name in catalogs}
    benchmarks: dict[str, str] = {}
    subsets: dict[str, list[str] | None] = {}
    unreadable: list[str] = []
    for name, ref in refs.items():
        try:
            meta, fields = _metadata(ref), _subset_fields(ref)
        except Exception as exc:  # noqa: BLE001 - an unreadable catalog keeps its own name, and says so
            unreadable.append(f"{name!r} ({type(exc).__name__}: {exc})")
            if warn is not None and not strict:
                warn(f"cannot read catalog {name!r} ({type(exc).__name__}: {exc}); it is a benchmark of its own, one subset")
            benchmarks[name], subsets[name] = name, None
            continue
        benchmarks[name] = _named_benchmark(meta, ref) or name
        subsets[name] = [_named_subset(v, ref, name) for v in fields]
    if strict and unreadable:
        raise UnreadableCatalogs(f"cannot read the benchmark and subsets of {len(unreadable)} catalog(s): "
                                 + "; ".join(unreadable))
    return benchmarks, subsets


def catalog_subsets(catalogs: Mapping[str, Any] | Iterable[str], *,
                    warn: Callable[[str], None] | None = None) -> dict[str, list[str] | None]:
    """{catalog: the subset of each of its problems} for ``catalogs`` (as :func:`catalog_benchmarks` takes them), from
    each problem's ``subset`` field, else the catalog's name (:func:`catalog_subset_rows`). A catalog that cannot be read
    maps to None (each of its problems is then in the subset named after it), reported through ``warn``; a field that is
    not a non-empty string without "/" raises."""
    refs = dict(catalogs) if isinstance(catalogs, Mapping) else {name: name for name in catalogs}
    out: dict[str, list[str] | None] = {}
    for name, ref in refs.items():
        try:
            fields = _subset_fields(ref)
        except Exception as exc:  # noqa: BLE001 - an unreadable catalog is one subset, and says so
            if warn is not None:
                warn(f"cannot read the problems of catalog {name!r} ({type(exc).__name__}: {exc}); it is one subset")
            out[name] = None
            continue
        out[name] = [_named_subset(v, ref, name) for v in fields]
    return out


__all__ = ["BENCHMARK_FIELD", "SRBF_CATALOGS", "SUBSET_FIELD", "SUITES", "catalog_benchmark", "catalog_benchmarks",
           "catalog_structure", "catalog_subset_rows", "catalog_subsets", "expand_suite", "suite_catalogs",
           "UnreadableCatalogs"]
