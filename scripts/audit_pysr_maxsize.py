"""Audit: can PySR express every benchmark ground truth within its default ``maxsize``?

PySR's complexity budget (``maxsize``, a node count where every operator/variable/constant
node costs 1) caps the size of equations the search can represent at all. If a benchmark
ground truth NEEDS more nodes than ``maxsize`` under the adapter's operator vocabulary,
PySR is structurally unable to recover that expression.

Benchmark policy: baselines run at their upstream defaults, so this audit DOCUMENTS what
PySR's default budget implies on these benchmarks (the numbers belong next to PySR's
results); it does not justify overriding the default.

Node counting mirrors ``srbf.model_adapters._create_pysr_model``'s vocabulary: the
powers/roots ``pow2..pow5`` / ``pow1_2..pow1_5`` are single-node unary operators; the
compound ``mult2..mult5`` / ``div2..div5`` are single-node only when
``use_mult_div_operators=True`` (the shipped evaluation configs use ``False``, so they
expand to two nodes, e.g. ``mult3 x -> * 3 x``); every other token (variables,
``<constant>``, numeric literals) is one node.

Ground truths audited:
- ``fastsrb`` (declarative catalog, 120 expressions): prepared infix -> prefix via the
  ``dev_7-3`` simplipy engine.
- ``v23-val`` (frozen generative catalog): all pinned prefix skeletons.

Run: ``python scripts/audit_pysr_maxsize.py`` (needs symbolic-data, simplipy + assets).
"""
from __future__ import annotations

import sys
from collections import Counter

# The single-node operator vocabulary of srbf.model_adapters._create_pysr_model.
PYSR_UNARY = {
    "neg", "abs", "inv",
    "sin", "cos", "tan", "asin", "acos", "atan",
    "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
    "exp", "log",
    "pow2", "pow3", "pow4", "pow5",
    "pow1_2", "pow1_3", "pow1_4", "pow1_5",
}
PYSR_BINARY = {"+", "-", "*", "/", "^", "pow", "**"}
COMPOUND_MULT_DIV = {"mult2", "mult3", "mult4", "mult5", "div2", "div3", "div4", "div5"}
PYSR_LIBRARY_DEFAULT_MAXSIZE = 20  # pysr.PySRRegressor()'s own default (runs before srbf 0.6.1)


def node_count(tokens: list[str] | tuple[str, ...], operator_arity: dict[str, int],
               *, use_mult_div_operators: bool = False) -> int:
    """PySR node count of a prefix expression under the adapter vocabulary.

    One node per token, except compound mult/div tokens which cost two nodes when
    ``use_mult_div_operators`` is False (binary op + literal). Any OPERATOR token
    (known to the engine) that the PySR vocabulary lacks raises, so unmapped
    operators can never be silently under-counted.
    """
    total = 0
    for token in tokens:
        if token in COMPOUND_MULT_DIV:
            total += 1 if use_mult_div_operators else 2
        elif token in PYSR_UNARY or token in PYSR_BINARY:
            total += 1
        elif token in operator_arity:
            raise ValueError(f"operator {token!r} is not expressible in the PySR vocabulary")
        else:
            total += 1  # leaf: variable / <constant> / numeric literal / named constant
    return total


def main() -> int:
    import symbolic_data as sd
    from simplipy import SimpliPyEngine

    engine = SimpliPyEngine.load("dev_7-3", install=True)
    arity = dict(engine.operator_arity)

    audits: dict[str, list[tuple[str, int]]] = {}

    fastsrb = sd.load_catalog("fastsrb")
    audits["fastsrb"] = [
        (str(key), node_count(engine.infix_to_prefix(entry.prepared), arity))
        for key, entry in fastsrb.entries.items()
    ]

    val = sd.build_catalog("v23-val")
    audits["v23-val"] = [
        (" ".join(skeleton)[:60], node_count(skeleton, arity))
        for skeleton in sorted(val.skeletons)
    ]

    for name, rows in audits.items():
        sizes = sorted(size for _, size in rows)
        n = len(sizes)
        over = [(gt, size) for gt, size in rows if size > PYSR_LIBRARY_DEFAULT_MAXSIZE]
        p95 = sizes[int(0.95 * (n - 1))]
        print(f"== {name}: n={n}  min={sizes[0]}  median={sizes[n // 2]}  p95={p95}  max={sizes[-1]}")
        hist = Counter(size for size in sizes)
        print("   sizes:", dict(sorted(hist.items())))
        print(f"   > PySR default maxsize ({PYSR_LIBRARY_DEFAULT_MAXSIZE}): "
              f"{len(over)}/{n} ({100 * len(over) / n:.1f}%) not representable")
        for gt, size in sorted(over, key=lambda t: -t[1])[:10]:
            print(f"     {size:3d}  {gt}")

    print()
    print("NOTE: benchmark policy is to run baselines at their upstream defaults, so these numbers "
          "are DOCUMENTATION of what PySR's default complexity budget implies on these benchmarks, "
          "not a defect to patch. See docs/models.md (PySR section).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
