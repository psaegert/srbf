"""Golden characterization of the two refining baselines' ``_results``.

Guards the ``BruteForceModel`` / ``LampleChartonModel`` dedup (shared ``_RefiningBaselineModel``
base): the refactor moves the per-candidate refine/score/build and the fit loop into shared helpers,
so it must not change the produced results at all. Fitting is deterministic here
(``refiner_p0_noise=None`` -> no random init; ``LampleChartonModel`` is seeded), so we assert an
exact value + schema + order match against ``tests/fixtures/golden_baseline_results.json`` (captured
on the pre-dedup code).
"""
import json
import math
import os

import numpy as np
import pytest
from simplipy import SimpliPyEngine

from symbolic_data import LampleChartonCatalog
from srbf.baselines import BruteForceModel, LampleChartonModel

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "fixtures", "golden_baseline_results.json")

SAMPLE_STRATEGY = {
    "n_operator_distribution": "equiprobable_lengths",
    "min_operators": 0, "max_operators": 2, "power": 1,
    "max_length": 6, "max_tries": 1, "independent_dimensions": True,
}
SUPPORT_CFG = {
    "support_prior": {"name": "uniform", "kwargs": {"low": -1, "high": 1, "min_value": -1, "max_value": 1}},
    "n_support_prior": {"name": "uniform", "kwargs": {"low": 4, "high": 4, "min_value": 4, "max_value": 4}},
}


@pytest.fixture(scope="module")
def simplipy_engine() -> SimpliPyEngine:
    return SimpliPyEngine.load("dev_7-3", install=True)


@pytest.fixture(scope="module")
def golden() -> dict:
    with open(FIXTURE) as handle:
        return json.load(handle)


def _build_pool(engine, skeletons, variables=("x1",)):
    pool = LampleChartonCatalog.from_dict(
        skeletons=set(skeletons), simplipy_engine=engine,
        sample_strategy=SAMPLE_STRATEGY,
        literal_prior={"name": "normal", "kwargs": {"loc": 0, "scale": 1}},
        variables=list(variables), support_sampler_config=SUPPORT_CFG,
    )
    pool.skeletons = set(skeletons)
    pool.skeleton_codes = pool.compile_codes(verbose=False)
    return pool


def _canon(model) -> list[dict]:
    rows = []
    for r in model._results:
        rows.append({
            "keys": sorted(r.keys()),
            "expression": list(r["expression"]),
            "raw_beam_decoded": r["raw_beam_decoded"],
            "complexity": int(r["complexity"]),
            "constant_count": int(r["constant_count"]),
            "score": (None if (isinstance(r["score"], float) and math.isnan(r["score"])) else round(float(r["score"]), 6)),
            "fvu": (None if (isinstance(r["fvu"], float) and math.isnan(r["fvu"])) else round(float(r["fvu"]), 6)),
            "n_fits": len(r["fits"]),
            "function_callable": callable(r["function"]),
            "has_refiner": r["refiner"] is not None,
        })
    return rows


def _total_key(row: dict) -> tuple:
    """A hash-independent total order. The models sort by score only; among *tied* scores the order
    is set-iteration (``PYTHONHASHSEED``) dependent and thus arbitrary, so we canonicalize it here
    before comparing to the golden (which cannot pin an arbitrary tie order)."""
    inf = float("inf")
    return (
        inf if row["score"] is None else row["score"],
        tuple(row["expression"]),
        inf if row["fvu"] is None else row["fvu"],
        row["raw_beam_decoded"],
    )


def _assert_matches_golden(model, golden_rows: list[dict]) -> None:
    rows = _canon(model)
    # Records are identical up to tie order (the only nondeterministic axis).
    assert sorted(rows, key=_total_key) == sorted(golden_rows, key=_total_key)
    # The model's own primary ordering (ascending score) still holds on the live output.
    scores = [r["score"] for r in rows if r["score"] is not None]
    assert scores == sorted(scores)


def test_bruteforce_results_match_golden(simplipy_engine, golden) -> None:
    pool = _build_pool(simplipy_engine, [("x1",)], variables=("x1",))
    # max_expressions huge + short max_length => the generator enumerates the COMPLETE length<=2 space
    # (no hash-order-dependent cutoff), so the found set is deterministic across processes/hash seeds.
    model = BruteForceModel(
        simplipy_engine=simplipy_engine, catalog=pool, max_expressions=100_000, max_length=2,
        include_constant_token=True, ignore_holdouts=True, n_restarts=1, refiner_p0_noise=None,
    )
    X = np.linspace(-1.5, 1.5, 12).reshape(-1, 1)
    y = (X ** 2).copy()
    model.fit(X, y)
    _assert_matches_golden(model, golden["bruteforce"])


def test_lample_results_match_golden(simplipy_engine, golden) -> None:
    pool = _build_pool(
        simplipy_engine, [("x1",), ("sin", "x1"), ("*", "x1", "x1"), ("+", "x1", "<constant>")], variables=("x1",),
    )
    model = LampleChartonModel(
        simplipy_engine=simplipy_engine, catalog=pool, samples=4, unique=True, seed=123,
        ignore_holdouts=True, n_restarts=1, refiner_p0_noise=None,
    )
    X = np.linspace(-1.5, 1.5, 12).reshape(-1, 1)
    y = (X ** 2).copy()
    model.fit(X, y)
    _assert_matches_golden(model, golden["lample"])
