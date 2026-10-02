"""The Predictions view shows each ground truth as the benchmark states it and in its canonical form: the SimpliPy
engine's simplification with the numbers kept, the judge's first step. The forms are cached per simplipy version and
engine, so a refresh simplifies only new ground truths."""
import importlib.util
import json
import os

_SPEC = importlib.util.spec_from_file_location(
    "site_export_v2", os.path.join(os.path.dirname(__file__), "..", "scripts", "site_export_v2.py"))
export = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(export)


class _Engine:
    """Stands in for SimpliPyEngine.simplify: a prefix token list in, a prefix token list out."""

    def __init__(self):
        self.calls = []

    def simplify(self, expression):
        self.calls.append(list(expression))
        if expression == ["*", "x1", "x1"]:
            return ["pow", "x1", "2"]
        if expression == ["bad"]:
            raise ValueError("unparsable")
        return list(expression)


def test_canonical_forms_come_from_the_engine_and_are_cached(tmp_path):
    truth = {"cat": {0: "* x1 x1", 1: "+ x1 0.123456789", 2: "bad"}}
    cache = str(tmp_path / "canonical.json")
    engine = _Engine()
    forms = export.canonical_truths(truth, "some-engine", cache, engine=engine)
    assert forms["* x1 x1"] == "pow x1 2"
    assert forms["+ x1 0.123456789"] == "+ x1 0.123456789"
    assert forms["bad"] is None                                   # refused: shown as stated
    assert len(engine.calls) == 3
    again = _Engine()
    assert export.canonical_truths(truth, "some-engine", cache, engine=again) == forms
    assert again.calls == []                                       # all from the cache
    other = _Engine()
    export.canonical_truths(truth, "another-engine", cache, engine=other)
    assert len(other.calls) == 3                                   # another engine: simplified afresh
    assert json.load(open(cache))["tag"].endswith("|another-engine")


def test_truth_files_hold_the_stated_and_the_canonical_form_rounded(tmp_path):
    truth = {"cat": {0: "* x1 0.123456789", 1: "bad"}}
    canonical = {"* x1 0.123456789": "* 0.123456789 x1", "bad": None}
    export.write_predictions(str(tmp_path), "r", {}, truth, {"cat": 2}, canonical)
    text = (tmp_path / "pred" / "truth" / "cat.0.js").read_text()
    body = json.loads(text[text.index('"truth|cat|0"]=') + len('"truth|cat|0"]='):text.rindex(";})();")])
    assert body == {"0": ["* x1 0.1235", "* 0.1235 x1"], "1": ["bad", None]}
