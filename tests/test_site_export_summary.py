"""The progress summary: which methods are finished, in progress and scheduled. The Results page's progress line and the
Progress page both show it, so the rule lives here once. Finished means every planned run, every budget's time,
and a ladder that has reached its end."""
import importlib.util
import os

_SPEC = importlib.util.spec_from_file_location(
    "site_export_v2", os.path.join(os.path.dirname(__file__), "..", "scripts", "site_export_v2.py"))
export = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(export)


def _method(key, budgets):
    return {"key": key, "label": key.upper(), "color": "#000", "budgets": budgets}


def test_finished_needs_every_run_and_every_budget_timed():
    methods = [_method("done", [1, 2]), _method("untimed", [1, 2]), _method("running", [1, 2]), _method("unplanned", None)]
    status = {"done": [4, 4], "untimed": [4, 4], "running": [3, 4], "unplanned": [2, None]}
    timing = {"done": {"1": 40.0, "2": 90.0}, "untimed": {"1": 90.0}, "running": {"1": 40.0, "2": 90.0}}
    s = export.progress_summary(methods, status, timing)
    assert s["finished"] == ["done"]
    assert s["in_progress"] == ["untimed", "running", "unplanned"]       # no plan: never finished, never a guess


def test_finished_needs_the_ladder_to_have_reached_its_end():
    """Every planned run and time in is not enough: the budgets double up to about 100 s per problem, and the ladder
    ends at the doubling nearest that on a log scale -- a measured time of at least 100 / sqrt(2) s."""
    top = export.LADDER_TOP_S / 2 ** 0.5
    methods = [_method("short", [1, 2]), _method("at_end", [1, 2]), _method("just_below", [1, 2])]
    status = {k: [4, 4] for k in ("short", "at_end", "just_below")}
    timing = {"short": {"1": 0.1, "2": 2.0}, "at_end": {"1": 40.0, "2": top}, "just_below": {"1": 40.0, "2": top - 0.1}}
    s = export.progress_summary(methods, status, timing)
    assert s["finished"] == ["at_end"]
    assert s["in_progress"] == ["short", "just_below"]
    assert s["more_budgets"] == ["short", "just_below"]                 # all in so far, larger budgets to come
    assert s["ladder"]["at_end"] == {"end": "reached"}
    assert "short" not in s["ladder"]


def test_a_method_that_cannot_run_a_larger_budget_ends_early_with_its_reason():
    s = export.progress_summary([_method("e2e", [1, 256])], {"e2e": [4, 4]}, {"e2e": {"1": 0.5, "256": 9.7}})
    assert s["finished"] == ["e2e"] and s["more_budgets"] == []
    assert s["ladder"]["e2e"]["end"] == "declared" and s["ladder"]["e2e"]["reason"] == export.LADDER_END["e2e"]
    running = export.progress_summary([_method("e2e", [1, 256])], {"e2e": [3, 4]}, {"e2e": {"1": 0.5, "256": 9.7}})
    assert running["in_progress"] == ["e2e"]                             # a declared end still needs every planned run


def test_dropped_budgets_are_not_planned():
    """The T8 series' 65,536-draw runs were stopped once its ladder had passed 100 s; they leave the plan."""
    assert all(65536 in rungs for rungs in export.PLAN_DROPPED.values())
    assert set(export.PLAN_DROPPED) == {"T8-3M", "T8-20M", "T8-120M"}


def test_a_scheduled_method_leaves_the_line_once_the_release_carries_it():
    first_key, first_label, _ = export.SCHEDULED[0]
    without = export.progress_summary([_method("e2e", [1])], {"e2e": [0, 2]}, {})
    assert [x["label"] for x in without["scheduled"]] == [label for _, label, _ in export.SCHEDULED]
    assert all(set(x) == {"label", "note"} for x in without["scheduled"])   # the future key is never published
    carried = export.progress_summary([_method(first_key, [1])], {first_key: [0, 2]}, {})
    assert first_label not in [x["label"] for x in carried["scheduled"]]
    assert carried["in_progress"] == [first_key]


def test_the_scheduled_keys_are_not_published_method_keys_yet():
    published = {m[0] for m in export.METHODS}
    # a scheduled method with a METHODS entry must be the same method under the same key (the oracle is one)
    for key, label, _ in export.SCHEDULED:
        if key in published:
            assert next(m[1] for m in export.METHODS if m[0] == key) == label


def test_the_summary_payload_is_small_and_complete():
    payload = {"release": {"id": "r", "updated": "u", "generated": "g", "scoring": "s", "judge": "j", "versions": "v", "notes": ""},
               "timing_note": "t", "catalogs": [{"key": "a", "laws": 3}, {"key": "b", "laws": 4}],
               "methods": [{"key": "m", "label": "M", "color": "#123", "budgets": [1], "param": "draws", "selection": "long text"}],
               "status": {"m": [1, 2]}, "progress": {"m": {"1": [1, 2]}}, "timing": {"m": {"1": 0.5}},
               "summary": {"finished": [], "in_progress": ["m"], "scheduled": []}, "cells": {"m": {"a": {}}}}
    out = export.summary_payload(payload)
    assert out["problem_sets"] == 2 and out["problems"] == 7
    assert out["methods"] == [{"key": "m", "label": "M", "color": "#123", "budgets": [1]}]
    assert "cells" not in out and out["release"]["scoring"] == "s" and out["summary"]["in_progress"] == ["m"]
