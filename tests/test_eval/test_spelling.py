"""srbf.spelling: what a method printed, in the spelling the judge's engine reads, and nothing else rewritten."""
import pytest

from srbf.candidates import read_spelled
from srbf.spelling import engine_spelling, spell_square_roots, written_arity


def test_the_written_arity_holds_the_operators_and_the_defined_functions(engine_defining_sqrt, engine_without_definitions):
    defined = written_arity(engine_defining_sqrt)
    assert {name: defined[name] for name in ("sqrt", "cot", "acsc")} == {"sqrt": 1, "cot": 1, "acsc": 1}
    assert defined.items() >= engine_defining_sqrt.operator_arity.items()
    assert written_arity(engine_without_definitions) == dict(engine_without_definitions.operator_arity)


def test_a_square_root_stays_where_the_engine_defines_it(engine_defining_sqrt, engine_without_definitions):
    prefix = ["+", "sqrt", "x1", "cot", "x2"]
    assert spell_square_roots(prefix, written_arity(engine_defining_sqrt)) == prefix
    assert spell_square_roots(["*", "sqrt", "x1", "x2"], written_arity(engine_without_definitions)) == [
        "*", "rootn", "x1", "2", "x2"]
    with pytest.raises(ValueError):                  # cot is no head of this engine: the walk cannot place x2
        spell_square_roots(prefix, written_arity(engine_without_definitions))


def test_sympys_spellings_are_the_engines_and_an_unwalkable_prediction_stays_as_written(
        engine_defining_sqrt, engine_without_definitions):
    assert engine_spelling(["Abs", "sqrt", "x1"], written_arity(engine_defining_sqrt)) == ["abs", "sqrt", "x1"]
    assert engine_spelling(["Abs", "sqrt", "x1"], written_arity(engine_without_definitions)) == ["abs", "rootn", "x1", "2"]
    assert engine_spelling(["sqrt", "conjugate", "x1"], written_arity(engine_without_definitions)) == ["sqrt", "conjugate", "x1"]


def test_a_methods_string_is_read_as_written(engine_defining_sqrt, engine_without_definitions):
    """The reader's prefix in the ground truth's names; the square root is the one rewrite, for an engine without it."""
    assert read_spelled(engine_defining_sqrt, "-v1 - sqrt(v2)", ["v1", "v2"]) == ["-", "neg", "x1", "sqrt", "x2"]
    assert read_spelled(engine_without_definitions, "-v1 - sqrt(v2)", ["v1", "v2"]) == ["-", "neg", "x1", "rootn", "x2", "2"]
    assert read_spelled(engine_defining_sqrt, "cot(v1)", ["v1"]) == ["cot", "x1"]
    with pytest.raises(ValueError, match=r"does not read zeta; .* cot .* sqrt"):
        read_spelled(engine_defining_sqrt, "zeta(v1)", ["v1"])
    with pytest.raises(ValueError, match=r"does not read cot; .* sqrt"):
        read_spelled(engine_without_definitions, "cot(v1)", ["v1"])
