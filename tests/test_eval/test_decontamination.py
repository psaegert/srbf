"""Decontamination verification: hermetic toy catalogs (no HF pulls, no 29-pool registration).

The training side is a toy LampleChartonCatalog with ONE registered holdout pool; the benchmark
side is a tiny frozen ProblemCatalog with one held-out, one deliberately-missed and one black-box
problem, plus a declarative catalog whose entry cannot be probed (fail-closed: UNVERIFIED, never
covered). CLI wiring (exit code + JSON report) is tested against a stubbed verifier.
"""
import json

import numpy as np
import pytest
from simplipy import SimpliPyEngine

from symbolic_data import LampleChartonCatalog, Problem
from symbolic_data.catalog import CatalogEntry, ProblemCatalog

import srbf.decontamination as decon
from srbf.__main__ import main
from srbf.decontamination import CatalogCoverage, DecontaminationReport, ProblemFinding, verify_decontamination


@pytest.fixture(scope="module")
def simplipy_engine() -> SimpliPyEngine:
    return SimpliPyEngine.load("acj-4-3", install=True)


def _problem(skeleton, variables, *, eq_id, gt_kind=None):
    nv = max(1, len(variables))
    xs = np.zeros((4, nv), dtype=np.float64)
    ys = np.zeros((4, 1), dtype=np.float64)
    return Problem(
        x_support=xs, y_support=ys, y_support_noisy=ys.copy(),
        x_validation=xs.copy(), y_validation=ys.copy(), y_validation_noisy=ys.copy(),
        skeleton=skeleton, expression=list(skeleton) if skeleton else None,
        constants=[], variables=variables, complexity=len(skeleton) if skeleton else None,
        eq_id=eq_id, gt_kind=gt_kind,
    )


def _toy_training_catalog(engine, holdout_pools):
    # Deterministic single-skeleton recipe (mirrors the toy pools in test_baselines); the
    # holdout pools are the only part the decontamination check exercises.
    sample_strategy = {
        "n_operator_distribution": "equiprobable_lengths",
        "min_operators": 0,
        "max_operators": 0,
        "power": 1,
        "max_length": 4,
        "max_tries": 1,
        "independent_dimensions": True,
    }
    support_sampler_config = {
        "support_prior": {"name": "uniform", "kwargs": {"low": -1, "high": 1, "min_value": -1, "max_value": 1}},
        "n_support_prior": {"name": "uniform", "kwargs": {"low": 4, "high": 4, "min_value": 4, "max_value": 4}},
    }
    return LampleChartonCatalog.from_dict(
        skeletons={("x1",)},
        simplipy_engine=engine,
        sample_strategy=sample_strategy,
        literal_prior={"name": "normal", "kwargs": {"loc": 0, "scale": 1}},
        variables=["x1"],
        support_sampler_config=support_sampler_config,
        holdout_pools=holdout_pools,
    )


def test_held_missed_and_black_box_are_separated(simplipy_engine):
    pool = ProblemCatalog.from_problems(
        [_problem(("sin", "x1"), ["x1"], eq_id="HELD-1")], name="toypool")
    training = _toy_training_catalog(simplipy_engine, [pool])
    bench = ProblemCatalog.from_problems([
        _problem(("sin", "x1"), ["x1"], eq_id="HELD-1"),
        _problem(("exp", "x1"), ["x1"], eq_id="MISS-1"),      # deliberately NOT registered
        _problem(None, ["x1"], eq_id="BB-1", gt_kind="none"),  # black-box: nothing to hold out
    ], name="toybench")

    report = verify_decontamination(training, benchmarks=[bench])

    (coverage,) = report.catalogs
    assert coverage.name == "toybench"
    assert coverage.total == 3
    assert coverage.held == 1
    assert coverage.black_box == 1
    assert coverage.probeable == 2
    assert [finding.eq_id for finding in coverage.missed] == ["MISS-1"]
    assert coverage.missed[0].tokens == ["exp", "x1"]
    assert not coverage.unparseable
    assert report.ok is False
    assert report.coverage == pytest.approx(0.5)   # 1 of 2 probe-able

    payload = report.to_dict()
    assert payload["overall"]["ok"] is False
    assert payload["overall"]["missed"] == 1
    assert payload["catalogs"]["toybench"]["missed"][0]["eq_id"] == "MISS-1"


def test_family_equivalent_renderings_count_as_held(simplipy_engine):
    # The holdout quotient generalizes over variable names and affine constants; the probe
    # must inherit that (a registered sin(x1) covers 2*sin(v1)).
    pool = ProblemCatalog.from_problems(
        [_problem(("sin", "x1"), ["x1"], eq_id="HELD-1")], name="toypool")
    training = _toy_training_catalog(simplipy_engine, [pool])
    bench = ProblemCatalog.from_problems(
        [_problem(("*", "2", "sin", "v1"), ["v1"], eq_id="HELD-2")], name="toybench")

    report = verify_decontamination(training, benchmarks=[bench])

    assert report.ok is True
    assert report.catalogs[0].held == 1
    assert report.coverage == 1.0


def test_unprobeable_problem_is_unverified_never_covered(simplipy_engine):
    # Fail-closed: tokens the engine cannot read land in `unparseable` (and fail the check),
    # even though the raw `is_held_out` seam would report them held out.
    pool = ProblemCatalog.from_problems(
        [_problem(("sin", "x1"), ["x1"], eq_id="HELD-1")], name="toypool")
    training = _toy_training_catalog(simplipy_engine, [pool])
    bench = ProblemCatalog.from_problems(
        [_problem(("quxop", "x1"), ["x1"], eq_id="BAD-1")], name="toybad")

    report = verify_decontamination(training, benchmarks=[bench])

    (coverage,) = report.catalogs
    assert coverage.held == 0 and not coverage.missed
    assert [finding.eq_id for finding in coverage.unparseable] == ["BAD-1"]
    assert report.ok is False
    assert report.coverage == 0.0


def test_declarative_benchmark_entries_probe_via_infix(simplipy_engine):
    # Non-frozen catalogs (the shape `build_catalog` resolves the curated suites to) probe
    # each entry's prepared/raw infix; a malformed entry is UNVERIFIED.
    pool = ProblemCatalog.from_problems(
        [_problem(("sin", "x1"), ["x1"], eq_id="HELD-1")], name="toypool")
    training = _toy_training_catalog(simplipy_engine, [pool])
    bench = ProblemCatalog(
        name="toydecl", version=1,
        entries={
            "OK-1": CatalogEntry(id="OK-1", prepared="sin(v1)"),
            "BAD-1": CatalogEntry(id="BAD-1", prepared="v1 +* ("),
        })

    report = verify_decontamination(training, benchmarks=[bench])

    (coverage,) = report.catalogs
    assert coverage.total == 2
    assert coverage.held == 1
    assert [finding.eq_id for finding in coverage.unparseable] == ["BAD-1"]
    assert report.ok is False


def _stub_report(ok: bool) -> DecontaminationReport:
    coverage = CatalogCoverage(name="stub", total=2, held=2 if ok else 1, black_box=0)
    if not ok:
        coverage.held = 1
        coverage.missed.append(ProblemFinding("MISS-1", "stub", ["exp", "x1"], "rendering is NOT held out by the training catalog"))
    return DecontaminationReport(training_catalog="stub.yaml", catalogs=[coverage])


def test_cli_exit_zero_only_on_full_coverage(monkeypatch, tmp_path, capsys):
    report_path = tmp_path / "report.json"
    monkeypatch.setattr(decon, "verify_decontamination", lambda *a, **k: _stub_report(ok=True))
    main(["decontamination", "-t", "stub.yaml", "-o", str(report_path)])  # must not raise
    payload = json.loads(report_path.read_text())
    assert payload["overall"]["ok"] is True

    monkeypatch.setattr(decon, "verify_decontamination", lambda *a, **k: _stub_report(ok=False))
    with pytest.raises(SystemExit) as excinfo:
        main(["decontamination", "-t", "stub.yaml"])
    assert excinfo.value.code == 1
    assert "MISS stub/MISS-1" in capsys.readouterr().out
