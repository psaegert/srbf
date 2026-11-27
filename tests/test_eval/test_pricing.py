"""srbf.pricing: every description length in srbf goes through one function, in bits, on the form as written."""
import math
import re
from pathlib import Path

import numpy as np
import pytest
import yaml

from srbf.pricing import price_bits, pricer

ROOT = Path(__file__).resolve().parents[2]


class _Engine:
    """Prices a prefix as written at 1000 milli-bits per token and refuses an expression with an unknown token, as
    ``SimpliPyEngine.written_complexity`` refuses a head the engine neither declares nor defines."""

    def __init__(self):
        self.calls = []

    def written_complexity(self, expression):
        self.calls.append(list(expression))
        if "?" in expression:
            raise ValueError("unknown function '?'\nsecond line")
        return 1000 * len(expression)


@pytest.fixture(scope="module")
def engine():
    from simplipy import SimpliPyEngine

    return SimpliPyEngine.load("acj2-4-llm", install=True)


def test_the_price_is_the_engines_written_complexity_in_bits():
    engine = _Engine()
    assert price_bits(engine, ("+", "x1", "1.5")) == 3.0
    assert engine.calls == [["+", "x1", "1.5"]]


def test_an_expression_the_engine_cannot_price_has_no_price_and_says_why():
    assert math.isnan(price_bits(_Engine(), ["+", "x1", "?"]))
    refusals: list[str] = []
    assert math.isnan(price_bits(_Engine(), ["+", "x1", "?"], refusals=refusals))
    assert refusals == ["ValueError: unknown function '?'"]          # the first line, with the error's type
    assert price_bits(_Engine(), ["x1"], refusals=refusals) == 1.0 and len(refusals) == 1


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
    """The engine's prices are called in srbf.pricing and nowhere else in the package or its scripts (the workers
    run in the methods' own environments and are not judged by srbf's price)."""
    callers = []
    for path in list((ROOT / "src" / "srbf").rglob("*.py")) + list((ROOT / "scripts").rglob("*.py")):
        if "worker" in path.relative_to(ROOT).parts or path.name == "pricing.py":
            continue
        if re.search(r"\.(?:written_)?complexity\(", path.read_text(encoding="utf-8")):
            callers.append(str(path.relative_to(ROOT)))
    assert not callers, f"price through srbf.pricing.price_bits, not the engine's price: {callers}"


@pytest.mark.parametrize("prefix, bits", [(["x1"], 6.0), (["3.5"], 5.585), (["-3.5"], 6.585)])
def test_prices_through_the_engine(engine, prefix, bits):
    assert price_bits(engine, prefix) == pytest.approx(bits)


# ---- the written form, not the canonical one -------------------------------------------------------------------------
@pytest.mark.parametrize("shorter, longer, short_bits, long_bits, canonical_bits", [
    ("-(x1 + x2)", "-x1 - x2", 16.0, 17.0, 16.0),     # one sign on the sum, or one on each term
    ("x1**2", "x1*x1", 11.585, 15.0, 11.585),
    ("x1", "x1 + 0", 6.0, 11.0, 6.0),
    ("5*x1", "2*x1 + 3*x1", 12.585, 26.585, 12.585),
])
def test_two_spellings_of_one_function_are_priced_as_written(engine, shorter, longer, short_bits, long_bits,
                                                             canonical_bits):
    """Both spellings have one canonical form and one canonical price; the written price charges what was written."""
    a, b = engine.read_infix(shorter), engine.read_infix(longer)
    assert engine.simplify(a) == engine.simplify(b)
    assert engine.complexity(a) / 1000 == pytest.approx(canonical_bits) == engine.complexity(b) / 1000
    assert price_bits(engine, a) == pytest.approx(short_bits) and price_bits(engine, b) == pytest.approx(long_bits)


def test_the_canonical_form_is_not_always_the_shorter_one(engine):
    """The canonical state spreads the reciprocal of a product over its factors: as written it is one reciprocal."""
    written = engine.read_infix("x1*x2/(4*pi*x3)")
    assert price_bits(engine, written) == pytest.approx(34.322)
    assert engine.complexity(written) / 1000 == pytest.approx(35.322)


def test_a_head_the_engine_does_not_read_is_refused_with_its_reason(engine):
    """The tolerant reader keeps an undeclared call as a bare token; the written price refuses it and says why."""
    prefix = engine.read_infix("zeta(x1)")
    assert prefix == ["zeta", "x1"]
    refusals: list[str] = []
    assert math.isnan(price_bits(engine, prefix, refusals=refusals))
    assert len(refusals) == 1 and refusals[0].startswith("ValueError: ")
    with pytest.raises(ValueError, match="zeta"):        # the strict reading of the string names the function
        engine.written_complexity("zeta(x1)")


def test_a_methods_string_is_priced_as_written(engine, tmp_path):
    """A worker returns -v1 - v2: the record stores the reader's prefix of that string, and the table, the candidate
    judge and the selection rule price it at 17 bits, not at the 16 of its canonical form -(x1 + x2)."""
    from srbf.candidates import judge_candidates
    from srbf.result_processing import derive_metrics
    from srbf.subprocess_adapter import SubprocessAdapter
    from srbf.testing import TOY_EXPRESSION, toy_sample

    worker = tmp_path / "w.py"
    worker.write_text("def fit(x, y, *, x_val, variables, meta, options, state):\n"
                      "    return {'expression': '-v1 - v2'}\n")
    adapter = SubprocessAdapter(worker=str(worker), simplipy_engine=engine)
    adapter.prepare()
    try:
        sample = toy_sample(0)
        meta = dict(sample.metadata, variables=["v1", "v2"], variable_names=["v1", "v2"])
        sample = type(sample)(x_support=sample.x_support, y_support=sample.y_support, x_validation=sample.x_validation,
                              y_validation=sample.y_validation, metadata=meta)
        record = dict(adapter.evaluate_sample(sample).to_mapping())
    finally:
        adapter.close()
    assert record["predicted_expression_prefix"] == ["-", "neg", "x1", "x2"]
    snapshot = {key: [record[key]] for key in ("skeleton", "predicted_expression_prefix", "predicted_skeleton_prefix",
                                               "prediction_success")}
    snapshot["ground_truth_prefix"] = [TOY_EXPRESSION]
    scored = derive_metrics(snapshot, engine=engine)
    assert list(scored["predicted_mdl"]) == [17.0]
    assert engine.complexity(record["predicted_expression_prefix"]) == 16000          # the canonical price differs
    judged = judge_candidates(record["candidate_expressions"], engine=engine, variables=["v1", "v2"],
                              x_fit=sample.x_support, y_fit=sample.y_support)
    assert judged.bits[record["candidate_prediction"]] == 17.0


def test_a_ground_truth_is_priced_as_its_catalog_writes_it(engine, tmp_path):
    """A catalog writes its law as x1*x1 - (-x2): the ground truth is priced as written (26 bits; 20.585 canonical),
    while symbolic recovery reads canonical forms and finds the prediction x1**2 + x2 to be that law."""
    from symbolic_data.token_ops import normalize_skeleton

    from srbf.data_sources import CatalogSource
    from srbf.result_processing import derive_metrics

    unit = {"sample_range": [1.0, 2.0], "sample_type": ["uni", "pos"]}
    path = tmp_path / "toy.yaml"
    path.write_text(yaml.safe_dump({"metadata": {"name": "toy", "version": 1}, "expressions": {
        "a": {"prepared": "v1*v1 - (-v2)", "vars": {"v1": dict(unit), "v2": dict(unit)}}}}))
    (sample,) = list(CatalogSource.from_catalog(str(path), sampling={"n_support": 16, "n_validation": 8, "noise": 0.0},
                                                simplipy_engine=engine))
    truth = sample.metadata["ground_truth_prefix"]
    assert truth == ["-", "*", "x1", "x1", "neg", "x2"]                       # the catalog's spelling, read
    prediction = ["+", "pow", "x1", "2", "x2"]
    snapshot = {"skeleton": [sample.metadata["skeleton"]], "ground_truth_prefix": [truth],
                "predicted_expression_prefix": [prediction], "predicted_skeleton_prefix": [normalize_skeleton(prediction)],
                "prediction_success": [True]}
    scored = derive_metrics(snapshot, engine=engine)
    assert list(scored["ground_truth_mdl"]) == [price_bits(engine, truth)] == [26.0]
    assert engine.complexity(truth) / 1000 == pytest.approx(20.585)
    assert list(scored["predicted_mdl"]) == [pytest.approx(20.585)]
    assert list(scored["mdl_ratio"]) == [pytest.approx(20.585 / 26.0)]
    assert list(scored["symbolic_recovery"]) == [True]


def test_the_headline_box_starts_at_the_written_price_of_the_mean(engine):
    """The left edge prices the mean as the decimal it is written as: every digit of it."""
    from srbf.headline import lower_edge_bits

    assert lower_edge_bits(np.array([3.0, 4.0]), engine) == price_bits(engine, ["3.5"]) == pytest.approx(5.585)
    assert lower_edge_bits(np.array([2.0, 2.0, 3.0]), engine) == price_bits(engine, ["2.3333333333333335"])
    assert lower_edge_bits(np.array([2.0, 2.0, 3.0]), engine) == pytest.approx(59.460)
