"""srbf.pricing: every description length in srbf goes through one function, in bits."""
import math
import re
from pathlib import Path

import pytest

from srbf.pricing import price_bits, pricer

ROOT = Path(__file__).resolve().parents[2]


class _Engine:
    """Prices a prefix at 1000 milli-bits per token and refuses an expression with an unknown token."""

    def __init__(self):
        self.calls = []

    def complexity(self, expression, certified=True, mode="f64", canon="default"):
        self.calls.append((list(expression), certified, str(getattr(mode, "value", mode)), canon))
        if "?" in expression:
            raise ValueError("unknown token")
        return 1000 * len(expression)


def test_the_price_is_the_engines_certified_f64_complexity_in_bits():
    engine = _Engine()
    assert price_bits(engine, ("+", "x1", "1.5")) == 3.0
    assert engine.calls == [(["+", "x1", "1.5"], True, "f64", "default")]


def test_an_expression_the_engine_cannot_price_has_no_price():
    assert math.isnan(price_bits(_Engine(), ["+", "x1", "?"]))


def test_the_table_the_selection_rule_and_derive_metrics_price_the_same_way():
    from srbf.result_processing import derive_metrics
    from srbf.table import mdl_bits

    engine = _Engine()
    assert pricer(engine)(["sin", "x1"]) == mdl_bits(engine)(["sin", "x1"]) == price_bits(engine, ["sin", "x1"]) == 2.0
    engine.operator_arity = {"+": 2, "sin": 1}
    engine.simplify = lambda tokens: list(tokens)
    engine.convert_expression = lambda tokens: list(tokens)
    snapshot = {"skeleton": [["sin", "x1"]], "expression": [["sin", "x1"]], "variables": [["x1"]],
                "predicted_expression_prefix": [["+", "x1", "1.5"]],
                "predicted_skeleton_prefix": [["+", "x1", "<constant>"]],
                "ground_truth_prefix": [["sin", "x1"]], "prediction_success": [True]}
    scored = derive_metrics(snapshot, engine=engine)
    assert list(scored["predicted_mdl"]) == [3.0] and list(scored["ground_truth_mdl"]) == [2.0]   # bits


def test_no_other_module_prices_an_expression_itself():
    """The engine's complexity is called in srbf.pricing and nowhere else in the package or its scripts (the workers
    run in the methods' own environments and are not judged by srbf's price)."""
    callers = []
    for path in list((ROOT / "src" / "srbf").rglob("*.py")) + list((ROOT / "scripts").rglob("*.py")):
        if "worker" in path.relative_to(ROOT).parts or path.name == "pricing.py":
            continue
        if re.search(r"\.complexity\(", path.read_text(encoding="utf-8")):
            callers.append(str(path.relative_to(ROOT)))
    assert not callers, f"price through srbf.pricing.price_bits, not engine.complexity: {callers}"


@pytest.mark.parametrize("prefix, bits", [(["x1"], 6.0), (["3.5"], 5.585), (["-3.5"], 6.585)])
def test_prices_through_the_engine(prefix, bits):
    from simplipy import SimpliPyEngine

    engine = SimpliPyEngine.load("acj-4-3", install=True)
    assert price_bits(engine, prefix) == pytest.approx(bits)
