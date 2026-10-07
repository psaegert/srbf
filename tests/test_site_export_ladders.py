"""Budgets the release publishes: every budget a method was run at, in the method's own unit. A ladder need not be a
power of two (DSO samples in batches of 1,000 expressions, so its ladder doubles from 1,000) nor stop at a fixed grid:
a budget left out here shows on the site as a method with no results at all."""
import importlib.util
import os

_SPEC = importlib.util.spec_from_file_location(
    "site_export_v2", os.path.join(os.path.dirname(__file__), "..", "scripts", "site_export_v2.py"))
export = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(export)


def test_any_budget_a_method_ran_at_is_published() -> None:
    for key, r in (("dsr", 1000), ("dsr", 2000), ("udsr", 13000), ("operon", 32768), ("gpgomea", 131072), ("T8-20M", 1024)):
        assert export.usable(key, r)
    assert not export.usable("dsr", 0)
    assert not export.usable("e2e", 512)          # E2E only up to its default
    assert export.usable("e2e", 256)


def test_the_explorer_steps_through_every_budget_with_data_in_order() -> None:
    assert export.published_rungs(["2000", 1024, "1000", 13000, 1024, "131072"]) == [1000, 1024, 2000, 13000, 131072]


def test_a_budget_that_ran_out_of_memory_ends_the_ladder_below_it() -> None:
    """owner 2026-10-07: out of memory on the reference machine = DNF; the last budget below it is the last achievable."""
    export.DNF.clear()
    try:
        assert export.usable("udsr", 416000) and export.ladder_end("udsr", {})[0] is None
        export.DNF["udsr"] = 416000
        assert export.usable("udsr", 208000)
        assert not export.usable("udsr", 416000) and not export.usable("udsr", 832000)
        how, why = export.ladder_end("udsr", {"udsr": {"104000": 45.2}})
        assert how == "declared" and "out of memory" in why and "416,000" in why and why.startswith("uDSR")
        assert export.usable("dsr", 64000)                     # another method is not affected
    finally:
        export.DNF.clear()
