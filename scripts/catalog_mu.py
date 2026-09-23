"""Describe the suite's catalogs for the results site: every problem's ground-truth description length.

For every catalog of the srbf suite, the description length (certified, f64 parse, default canon, in bits) of each
problem's ground-truth expression, one problem per expression, in catalog order; a placeholder (an expression the
catalog cannot realize) is null. The results explorer shows each catalog's size and the quartiles of these values
(scripts/site_export_v2.py --sizes).

  catalog_mu.py --out catalog_mu.json [--engine acj-5-4-llm] [--catalogs a,b,...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable

import numpy as np


def describe(catalogs: list[str], engine_name: str, log: Callable[[str], None] = print) -> dict:
    from simplipy import SimpliPyEngine
    from simplipy.engine import Mode
    from symbolic_data import ProblemSource

    engine = SimpliPyEngine.load(engine_name, install=True)

    def mu(prefix: list[str]) -> float | None:
        try:
            return float(engine.complexity(list(prefix), certified=True, mode=Mode.f64, canon="default")) / 1000.0
        except Exception:  # noqa: BLE001 - an expression the engine cannot price has no description length
            return None

    per_catalog: dict[str, list[float | None]] = {}
    t0 = time.time()
    for catalog in catalogs:
        sampling = {"n_support": 8, "n_validation": 0, "noise": 0.0, "problems_per_expression": 1}
        source = ProblemSource({"catalog": catalog, "sampling": sampling}, simplipy_engine=engine)
        values = [None if getattr(p, "is_placeholder", False) or not p.expression else mu(p.expression) for p in source]
        per_catalog[catalog] = values
        priced = [v for v in values if v is not None]
        log(f"{catalog:22s} n={len(values):5d} priced={len(priced):5d} median={np.median(priced) if priced else float('nan'):7.1f} t={time.time() - t0:5.0f}s")
    pooled = np.array([v for vs in per_catalog.values() for v in vs if v is not None])
    return {"per_catalog": per_catalog,
            "pooled_quantiles": {f"q{int(100 * a)}": float(np.quantile(pooled, a)) for a in (0.1, 0.25, 0.5, 0.75, 0.9)},
            "n_pooled": int(pooled.size),
            "catalog_medians": {c: float(np.median([v for v in vs if v is not None])) for c, vs in per_catalog.items()
                                if any(v is not None for v in vs)},
            "pricing": f"certified f64 default-canon description length in bits, engine {engine_name}"}


def main() -> int:
    from srbf.suites import SRBF_CATALOGS

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--engine", default="acj-5-4-llm")
    ap.add_argument("--catalogs", default=",".join(SRBF_CATALOGS))
    a = ap.parse_args()
    out = describe([c for c in a.catalogs.split(",") if c], a.engine)
    tmp = a.out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh)
    os.replace(tmp, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
