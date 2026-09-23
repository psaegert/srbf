"""The names the results site shows for its metrics (scripts/site_export_v2.py): one plain name per metric, no
parenthesized qualifier, the abbreviation in the short name, and every name distinct."""
import importlib.util
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def _exporter() -> Any:
    spec = importlib.util.spec_from_file_location("site_export_v2", ROOT / "scripts" / "site_export_v2.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_metric_names_are_plain_and_distinct() -> None:
    rows = _exporter().METRICS
    labels = [row[1] for row in rows]
    shorts = [row[2] for row in rows]
    assert labels, "no metrics"
    for label in labels:
        assert not re.search(r"[()]", label), f"a name carries a parenthesized qualifier: {label!r}"
        assert label == label.strip() and "  " not in label, label
    assert len(set(labels)) == len(labels), "two metrics share a name"
    assert len(set(shorts)) == len(shorts), "two metrics share a short name"
