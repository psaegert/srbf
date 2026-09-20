"""How many benchmark laws fit into PySR's default complexity budget?

PySR's ``maxsize`` counts the nodes of an expression, one per operator, variable and constant, and bounds what
the search can represent at all. A law that needs more nodes than ``maxsize`` in the operator vocabulary of the
PySR adapter cannot be recovered symbolically, whatever the search does. Baselines run at their upstream
defaults, so this script documents what the default implies on a catalog; it changes nothing.

    python scripts/audit_pysr_maxsize.py                       # fastsrb
    python scripts/audit_pysr_maxsize.py --catalogs feynman nguyen
    python scripts/audit_pysr_maxsize.py --suite               # every srbf catalog

A law is counted as its catalog writes it, read by the engine the catalogs are judged with. The default
``maxsize`` is read from the installed ``pysr``; ``--maxsize`` sets it where ``pysr`` is not installed.
"""
from __future__ import annotations

import argparse
import sys
from typing import Any

ENGINE = "acj-5-4-llm"


def installed_default_maxsize() -> int | None:
    try:
        import inspect

        import pysr
        return int(inspect.signature(pysr.PySRRegressor.__init__).parameters["maxsize"].default)
    except Exception:
        return None


def node_count(prefix: list[str]) -> int:
    """One node per token: the engine's operators are the adapter's operators, and every leaf is one node."""
    return len(prefix)


def audit(catalog_name: str, engine: Any, maxsize: int) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    import symbolic_data as sd
    catalog = sd.load_catalog(catalog_name)
    rows = [(str(key), node_count(engine.read_infix(entry.prepared))) for key, entry in catalog.entries.items()]
    return rows, sorted((row for row in rows if row[1] > maxsize), key=lambda row: -row[1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--catalogs", nargs="+", default=["fastsrb"], help="catalog names (default: fastsrb)")
    parser.add_argument("--suite", action="store_true", help="every catalog of the srbf suite")
    parser.add_argument("--maxsize", type=int, help="the budget to audit against (default: the installed pysr's)")
    args = parser.parse_args()

    maxsize = args.maxsize if args.maxsize is not None else installed_default_maxsize()
    if maxsize is None:
        sys.exit("pysr is not installed here: pass --maxsize (30 in pysr 1.5)")

    from simplipy import SimpliPyEngine
    engine = SimpliPyEngine.load(ENGINE, install=True)
    if args.suite:
        from srbf.suites import SRBF_CATALOGS
        catalogs = list(SRBF_CATALOGS)
    else:
        catalogs = args.catalogs

    print(f"maxsize {maxsize}; engine {ENGINE}")
    for name in catalogs:
        rows, over = audit(name, engine, maxsize)
        sizes = sorted(size for _, size in rows)
        print(f"{name}: {len(over)} of {len(rows)} laws ({100 * len(over) / len(rows):.1f} %) need more than {maxsize} nodes; "
              f"median {sizes[len(sizes) // 2]}, largest {sizes[-1]}")
        for key, size in over[:10]:
            print(f"    {size:3d}  {key}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
