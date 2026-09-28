"""The site's time axis averages each catalog's mean fit time over the catalogs like every other number on the site
(owner 2026-09-27), so a large catalog cannot decide the time on its own."""
import importlib.util
import os
import sys

import numpy as np
import pytest

_HERE = os.path.join(os.path.dirname(__file__), "..", "scripts")
sys.path.insert(0, _HERE)
_SPEC = importlib.util.spec_from_file_location("site_timing", os.path.join(_HERE, "site_timing.py"))
site_timing = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(site_timing)


def test_one_catalog_is_its_mean():
    assert site_timing.over_sets([np.array([1.0, 2.0, 6.0])]) == pytest.approx(3.0)


def test_a_large_catalog_does_not_decide_the_time():
    rng = np.random.default_rng(0)
    big = rng.normal(10.0, 1.0, 200)                                  # one slow catalog with many problems
    small = [rng.normal(1.0, 0.1, 9) for _ in range(5)]               # five fast ones with a few each
    pooled = np.concatenate([big] + small).mean()
    t = site_timing.over_sets([big] + small)
    assert pooled > 8.0 and t < 4.0                                   # the plain pooled mean follows the big catalog


def test_no_timed_problem_is_no_time():
    assert site_timing.over_sets([np.array([])]) is None
