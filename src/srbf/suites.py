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
"""
from __future__ import annotations

import copy
from typing import Any, Mapping

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


__all__ = ["SRBF_CATALOGS", "SUITES", "expand_suite", "suite_catalogs"]
