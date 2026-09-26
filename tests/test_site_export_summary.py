"""The progress summary: which methods are finished, in progress and scheduled. The Results page's progress line and the
Progress page both show it, so the rule lives here once. Finished means every planned run and every budget's time."""
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
    timing = {"done": {"1": 0.1, "2": 0.2}, "untimed": {"1": 0.1}, "running": {"1": 0.1, "2": 0.2}}
    s = export.progress_summary(methods, status, timing)
    assert s["finished"] == ["done"]
    assert s["in_progress"] == ["untimed", "running", "unplanned"]       # no plan: never finished, never a guess


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
