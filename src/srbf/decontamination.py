"""Decontamination verification: prove the training-time holdout covers the benchmark set.

srbf owns evaluation end-to-end, and the fairness contract requires that a model was never
trained on the problems it is benchmarked on. This module VERIFIES that claim instead of
assuming it: every problem of every benchmark catalog is probed against the training
catalog's registered holdout via ``is_held_out`` -- the exact family quotient the
training-time sampler rejects candidate draws with -- and the result is a per-catalog
coverage report (`verify_decontamination` / ``python -m srbf decontamination``).

Fail-closed semantics: a benchmark problem whose tokens cannot be probed (unreadable
tokens, a non-canonicalizable prototype, an unrealizable probe, an overflowing holdout
evaluation) counts as UNVERIFIED and is listed, never as covered. ``is_held_out`` itself
fails closed toward *rejection* -- correct for the sampler, which may over-reject -- so the
probe pre-realizes the compiled probe code itself and records the seam's fail-closed
warnings, classifying those paths as UNVERIFIED rather than as verified coverage.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from simplipy.utils import codify, explicit_constant_placeholders
from symbolic_data.catalog import Catalog
from symbolic_data.config_io import load_config
from symbolic_data.generative import GenerativeCatalog, LampleChartonCatalog, build_catalog
from symbolic_data.token_ops import desugar_sqrt


@dataclass
class ProblemFinding:
    """One benchmark problem the check could not verify as held out (a miss, or unverified)."""

    eq_id: str
    catalog: str
    tokens: list[str]     # the offending rendering (prefix tokens), [] when none could be derived
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"eq_id": self.eq_id, "catalog": self.catalog, "tokens": list(self.tokens), "reason": self.reason}


@dataclass
class CatalogCoverage:
    """Holdout coverage of one benchmark catalog by the training catalog."""

    name: str
    total: int = 0
    held: int = 0
    black_box: int = 0                                        # gt_kind == "none": nothing to hold out
    missed: list[ProblemFinding] = field(default_factory=list)        # verified NOT held out
    unparseable: list[ProblemFinding] = field(default_factory=list)   # UNVERIFIED (fail-closed)

    @property
    def probeable(self) -> int:
        """Problems with a ground truth to hold out (everything but black-box)."""
        return self.total - self.black_box

    @property
    def ok(self) -> bool:
        """Every probe-able problem verified held out (no misses, nothing unverified)."""
        return not self.missed and not self.unparseable

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "held": self.held,
            "black_box": self.black_box,
            "missed": [finding.to_dict() for finding in self.missed],
            "unparseable": [finding.to_dict() for finding in self.unparseable],
        }


@dataclass
class DecontaminationReport:
    """Coverage of the full benchmark set by one training catalog's holdout."""

    training_catalog: str
    catalogs: list[CatalogCoverage]

    @property
    def total(self) -> int:
        return sum(coverage.total for coverage in self.catalogs)

    @property
    def held(self) -> int:
        return sum(coverage.held for coverage in self.catalogs)

    @property
    def black_box(self) -> int:
        return sum(coverage.black_box for coverage in self.catalogs)

    @property
    def probeable(self) -> int:
        return sum(coverage.probeable for coverage in self.catalogs)

    @property
    def missed(self) -> list[ProblemFinding]:
        return [finding for coverage in self.catalogs for finding in coverage.missed]

    @property
    def unparseable(self) -> list[ProblemFinding]:
        return [finding for coverage in self.catalogs for finding in coverage.unparseable]

    @property
    def coverage(self) -> float:
        """held / probe-able. UNVERIFIED problems count against coverage (fail-closed)."""
        probeable = self.probeable
        return 1.0 if probeable == 0 else self.held / probeable

    @property
    def ok(self) -> bool:
        return all(coverage.ok for coverage in self.catalogs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "training_catalog": self.training_catalog,
            "catalogs": {coverage.name: coverage.to_dict() for coverage in self.catalogs},
            "overall": {
                "total": self.total,
                "held": self.held,
                "black_box": self.black_box,
                "probeable": self.probeable,
                "missed": len(self.missed),
                "unparseable": len(self.unparseable),
                "coverage": self.coverage,
                "ok": self.ok,
            },
        }


def _probe_rendering(training: LampleChartonCatalog, tokens: Sequence[str]) -> tuple[bool | None, str | None]:
    """Probe ONE rendering (prefix tokens) against the training catalog's holdout.

    Returns ``(verdict, reason)``: ``(True, None)`` verified held out; ``(False, None)``
    verified NOT held out (contamination); ``(None, reason)`` UNVERIFIED. Mirrors
    ``register_holdout_pool``'s probe-side pipeline exactly (desugar -> family prototype ->
    realize -> ``is_held_out``), but pre-realizes the compiled probe code so ``is_held_out``'s
    internal fail-closed ``return True`` paths (fine for the sampler, which may over-reject)
    can never masquerade as verified coverage here.
    """
    engine = training.simplipy_engine
    try:
        desugared = desugar_sqrt([str(token) for token in tokens], engine.operator_arity_compat)
    except Exception as exc:  # noqa: BLE001 - any unreadable stream is UNVERIFIED, fail-closed
        return None, f"cannot desugar tokens ({type(exc).__name__}: {exc})"

    prototype = training.holdout_family_prototype(desugared)
    if prototype is None:
        return None, "not canonicalizable (no holdout family prototype)"

    try:
        realized = engine.operators_to_realizations(prototype)
        with_placeholders, constants = explicit_constant_placeholders(
            realized, inplace=True, convert_numbers_to_constant=False)
        code_string = engine.prefix_to_infix(with_placeholders, realization=True)
        code = codify(code_string, list(training.variables) + constants)
    except Exception as exc:  # noqa: BLE001 - an unrealizable probe is UNVERIFIED, fail-closed
        return None, f"probe not realizable ({type(exc).__name__}: {exc})"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        held = training.is_held_out(desugared, constants, code=code, n_variables=training.n_variables)
    assumed = [w for w in caught if "assuming held out" in str(w.message)]
    if assumed:
        return None, f"holdout evaluation failed closed ({assumed[0].message})"
    return bool(held), None


def _iter_problem_candidates(
    training: LampleChartonCatalog,
    benchmark: Catalog,
) -> Iterator[tuple[str, bool, list[list[str]], list[str]]]:
    """Yield ``(eq_id, is_black_box, candidate_renderings, extraction_failures)`` per problem.

    Mirrors ``register_holdout_pool``'s token extraction: a FROZEN catalog contributes each
    problem's ``skeleton`` (or ``expression``) plus any ``meta["alternate_renderings"]``
    (v-infix) parsed in the training catalog's space; a declarative catalog contributes each
    entry's ``prepared`` (or ``raw``) infix the same way. ``gt_kind == "none"`` (black-box)
    problems have nothing to hold out and are flagged instead.
    """
    engine = training.simplipy_engine
    if isinstance(benchmark, GenerativeCatalog):
        raise ValueError(
            f"benchmark {getattr(benchmark, 'name', benchmark)!r} is a generative catalog; "
            "decontamination verification needs a fixed benchmark problem set")

    if getattr(benchmark, "frozen", False) or getattr(benchmark, "problems", None) is not None:
        for index, problem in enumerate(getattr(benchmark, "problems", None) or []):
            eq_id = str(problem.eq_id) if problem.eq_id is not None else f"#{index}"
            if getattr(problem, "gt_kind", None) == "none":
                yield eq_id, True, [], []
                continue
            candidates: list[list[str]] = []
            failures: list[str] = []
            tokens = problem.skeleton or problem.expression
            if tokens:
                candidates.append([str(token) for token in tokens])
            for alternate in (problem.meta or {}).get("alternate_renderings", []):
                try:
                    candidates.append(engine.infix_to_prefix(alternate))
                except Exception as exc:  # noqa: BLE001 - an unreadable rendering is UNVERIFIED
                    failures.append(f"alternate rendering does not parse ({type(exc).__name__}): {alternate!r}")
            if not candidates and not failures:
                failures.append("no usable ground-truth tokens")
            yield eq_id, False, candidates, failures
    else:
        for entry in benchmark.iter_entries(np.random.default_rng()):
            eq_id = str(getattr(entry, "id", None) or getattr(entry, "eq_id", None) or "?")
            expression = getattr(entry, "prepared", None) or getattr(entry, "raw", None)
            if expression is None:
                yield eq_id, False, [], ["entry has no expression"]
                continue
            try:
                prefix = engine.infix_to_prefix(expression)
            except Exception as exc:  # noqa: BLE001 - an unreadable entry is UNVERIFIED
                yield eq_id, False, [], [f"infix does not parse ({type(exc).__name__}): {expression!r}"]
                continue
            yield eq_id, False, [prefix], []


def verify_decontamination(
    training: LampleChartonCatalog | str | Mapping[str, Any],
    benchmarks: Sequence[str | Catalog] | None = None,
    verbose: bool = False,
) -> DecontaminationReport:
    """Verify that ``training``'s registered holdout covers every benchmark problem.

    Parameters
    ----------
    training : LampleChartonCatalog | str | Mapping
        The training catalog: a built catalog, the path to its yaml config, or the parsed
        config mapping. Building from a config registers its ``holdout_pools`` (the slow
        step; probing itself is milliseconds per problem).
    benchmarks : sequence of str | Catalog, optional
        Benchmark catalog refs (``build_catalog`` names/paths) or built catalogs to verify.
        Defaults to the training config's ``holdout_pools`` (for a pre-built catalog: its
        registered pool refs) -- the exact evaluation suite the catalog commits to.
    verbose : bool, optional
        Print a per-catalog progress line.

    Returns
    -------
    DecontaminationReport
        Per-catalog ``total`` / ``held`` / ``black_box`` counts plus ``missed`` (verified
        contamination holes) and ``unparseable`` (UNVERIFIED, fail-closed) findings.
        ``report.ok`` is True only when every probe-able problem is verified held out.
    """
    config: Mapping[str, Any] | None = None
    if isinstance(training, str):
        config = load_config(training)
        training_label = training
        training_catalog = LampleChartonCatalog.from_config(training)
    elif isinstance(training, Mapping):
        config = training
        training_label = str(training.get("name", "lample_charton"))
        training_catalog = LampleChartonCatalog.from_config(dict(training))
    else:
        training_catalog = training
        training_label = getattr(training, "name", type(training).__name__)

    resolved_benchmarks: list[str | Catalog]
    if benchmarks is None:
        if config is not None:
            resolved_benchmarks = list(config.get("holdout_pools", []) or [])
        else:
            resolved_benchmarks = [pool for pool in training_catalog.holdout_pools if isinstance(pool, str)]
    else:
        resolved_benchmarks = list(benchmarks)
    if not resolved_benchmarks:
        raise ValueError(
            "no benchmark catalogs to verify: the training catalog declares no holdout_pools "
            "and none were passed")

    coverages: list[CatalogCoverage] = []
    for ref in resolved_benchmarks:
        benchmark = build_catalog(ref) if isinstance(ref, str) else ref
        name = ref if isinstance(ref, str) else str(getattr(benchmark, "name", type(benchmark).__name__))
        coverage = CatalogCoverage(name=name)
        for eq_id, is_black_box, candidates, failures in _iter_problem_candidates(training_catalog, benchmark):
            coverage.total += 1
            if is_black_box:
                coverage.black_box += 1
                continue
            verdicts = [_probe_rendering(training_catalog, candidate) for candidate in candidates]
            confirmed = [(candidate, verdict) for candidate, verdict in zip(candidates, verdicts)
                         if verdict[0] is False]
            unverified = list(failures) + [str(reason) for held, reason in verdicts if held is None]
            if confirmed:
                # A verified miss on ANY rendering is a contamination hole, the strongest finding.
                tokens, _ = confirmed[0]
                coverage.missed.append(ProblemFinding(
                    eq_id, name, [str(token) for token in tokens],
                    "rendering is NOT held out by the training catalog"))
            elif unverified:
                # Fail-closed: a problem that cannot be FULLY probed is UNVERIFIED, never covered.
                tokens = [str(token) for token in candidates[0]] if candidates else []
                coverage.unparseable.append(ProblemFinding(eq_id, name, tokens, "; ".join(unverified)))
            else:
                coverage.held += 1
        coverages.append(coverage)
        if verbose:
            status = "OK" if coverage.ok else "FAIL"
            print(f"[{status}] {coverage.name}: {coverage.held}/{coverage.probeable} held out "
                  f"({coverage.black_box} black-box, {len(coverage.missed)} missed, "
                  f"{len(coverage.unparseable)} unverified)", flush=True)

    return DecontaminationReport(training_catalog=training_label, catalogs=coverages)
