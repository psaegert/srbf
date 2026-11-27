"""The site's time axis averages each benchmark's mean fit time over the benchmarks like every other number on the site
(owner 2026-09-27), so a large one cannot decide the time on its own; the catalogs of a benchmark split over several are
its subsets (benchmarks > subsets > problems)."""
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
    assert site_timing.over_sets({"a": np.array([1.0, 2.0, 6.0])}) == pytest.approx(3.0)


def test_a_large_catalog_does_not_decide_the_time():
    rng = np.random.default_rng(0)
    big = rng.normal(10.0, 1.0, 200)                                  # one slow catalog with many problems
    small = [rng.normal(1.0, 0.1, 9) for _ in range(5)]               # five fast ones with a few each
    pooled = np.concatenate([big] + small).mean()
    t = site_timing.over_sets({"big": big, **{f"small{i}": x for i, x in enumerate(small)}})
    assert pooled > 8.0 and t < 4.0                                   # the plain pooled mean follows the big catalog


def test_no_timed_problem_is_no_time():
    assert site_timing.over_sets({"a": np.array([])}) is None


def test_the_catalogs_of_one_benchmark_are_its_subsets():
    from site_random_effects import SetStat, combine, combine_nested

    times = {"feynman": np.array([1.0, 2.0]), "feynman-points": np.array([3.0, 4.0, 5.0]), "nguyen": np.array([0.5, 0.7])}
    manifest = {"feynman": {"size": 2, "benchmark": "feynman"}, "feynman-points": {"size": 3, "benchmark": "feynman"},
                "nguyen": {"size": 2}}
    benchmarks = site_timing.benchmarks_of(manifest)
    assert benchmarks == {"feynman": "feynman", "feynman-points": "feynman", "nguyen": "nguyen"}
    want = combine_nested([[SetStat(2, 3.0, 5.0), SetStat(3, 12.0, 50.0)], [SetStat(2, 1.2, 0.74)]], "normal").mu   # nested by hand
    assert site_timing.over_sets(times, benchmarks) == pytest.approx(want)
    apart = site_timing.over_sets(times)                                                # each catalog its own benchmark
    assert apart == pytest.approx(combine([SetStat(2, 3.0, 5.0), SetStat(3, 12.0, 50.0), SetStat(2, 1.2, 0.74)], "normal").mu)
    assert apart != pytest.approx(want, abs=1e-6)     # feynman's two catalogs differ: close to apart, not equal
    assert site_timing.over_sets(times, {c: c for c in times}) == apart
