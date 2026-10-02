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
