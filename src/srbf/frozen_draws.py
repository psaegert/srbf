"""Frozen draws: a run reads each problem's data from a render instead of drawing it.

symbolic-data renders a catalog's draws once (``symbolic-data render-draws``): for every problem its fitting and
held-out points, its per-run values with the ground truth written with them, a standard normal noise vector per side,
and for a data problem the rows of one draw from its pools. Every method, budget, shard and machine that reads the
same render evaluates on the same data. This module reads a render (format ``symbolic-data/frozen-draws``, version
1): a ``manifest.json`` with the sha256 of each catalog's source and of each draws file, and one
``<catalog>/repeat-<r>.npz`` per catalog and repeat.

The reader refuses rather than falls back: a catalog the render does not hold (matched by the sha256 of the catalog
file the run resolves), a repeat it does not hold, a file whose bytes differ from the manifest, sampling that asks for
something else than what was rendered (counts, more than one problem per expression, a noise mixture, holdouts), and
a manifest that differs from a pinned sha256 are errors. Nothing is ever drawn: a noise level ``s`` makes a formula
problem's targets ``y + s * std(y) * noise`` per side from the stored vector, the definition of symbolic-data's scalar
noise; data problems are used as rendered and not noised.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

FORMAT = "symbolic-data/frozen-draws"
FORMAT_VERSIONS = (1,)
MANIFEST = "manifest.json"

#: The ``sampling`` keys a run may give next to frozen draws; each must agree with the render.
_SAMPLING_KEYS = ("n_support", "n_validation", "noise", "problems_per_expression", "method", "layout", "max_trials")
_CORE = ("x_support", "y_support", "x_validation", "y_validation")
_OPTIONAL = ("y_support_noisy", "y_validation_noisy", "y_reference_support", "y_reference_validation",
             "outlier_mask_support", "outlier_mask_validation")


class FrozenDrawsError(ValueError):
    """The frozen draws a run asks for are absent, altered, or do not match the run's configuration."""


def sha256_file(path: str | os.PathLike[str]) -> str:
    """The sha256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_draws_config(value: Any) -> dict[str, Any]:
    """The ``data_source.draws`` block: ``{root, repeat (default 1), manifest_sha256 (optional)}``."""
    if not isinstance(value, Mapping):
        raise FrozenDrawsError("data_source.draws must be a mapping: {root: <rendered root>, repeat: 1}")
    unknown = sorted(set(value) - {"root", "repeat", "manifest_sha256"})
    if unknown:
        raise FrozenDrawsError(f"data_source.draws: unknown keys {unknown} (allowed: root, repeat, manifest_sha256)")
    root = value.get("root")
    if not isinstance(root, str) or not root:
        raise FrozenDrawsError("data_source.draws.root must name the rendered root (the folder holding manifest.json)")
    repeat = value.get("repeat", 1)
    if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 1:
        raise FrozenDrawsError(f"data_source.draws.repeat must be an integer >= 1, got {repeat!r}")
    pin = value.get("manifest_sha256")
    if pin is not None and not (isinstance(pin, str) and len(pin) == 64):
        raise FrozenDrawsError("data_source.draws.manifest_sha256 must be a sha256 hex digest")
    return {"root": root, "repeat": repeat, "manifest_sha256": pin}


def noised(y: np.ndarray, level: float, vector: np.ndarray) -> np.ndarray:
    """symbolic-data's scalar noise with the stored vector in place of a fresh draw: ``y + level * std(y) * vector``;
    an empty or constant side, or level 0, stays clean."""
    if y.size == 0 or level <= 0.0:
        return y.copy()
    std = float(np.std(y))
    if not np.isfinite(std) or std <= 0:
        return y.copy()
    return (y + level * std * vector).astype(np.float64)


class FrozenDrawsSource:
    """The problems of one catalog and repeat read from a frozen-draws render, in the render's order.

    ``catalog`` is the run's catalog reference; it is resolved (as symbolic-data resolves it) and its file's sha256
    picks the render's entry. ``sampling`` and ``holdouts`` are the run's, checked against the render.
    """

    def __init__(self, root: str | os.PathLike[str], *, catalog: Any, repeat: int = 1,
                 sampling: Mapping[str, Any] | None = None, holdouts: Sequence[Any] | None = None,
                 manifest_sha256: str | None = None) -> None:
        self.root = Path(root)
        self.repeat = int(repeat)
        manifest_path = self.root / MANIFEST
        if not manifest_path.is_file():
            raise FrozenDrawsError(f"no frozen draws at {self.root} ({MANIFEST} is missing)")
        self.manifest_sha256 = sha256_file(manifest_path)
        if manifest_sha256 is not None and manifest_sha256 != self.manifest_sha256:
            raise FrozenDrawsError(f"{manifest_path}: sha256 {self.manifest_sha256} differs from the pinned {manifest_sha256}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != FORMAT or manifest.get("format_version") not in FORMAT_VERSIONS:
            raise FrozenDrawsError(f"{manifest_path}: not a {FORMAT} manifest of version {FORMAT_VERSIONS}")
        self.settings = dict(manifest["settings"])
        self.acceptance = manifest.get("acceptance")
        self.level = self._check_sampling(sampling or {})
        if holdouts:
            raise FrozenDrawsError("data_source.holdouts cannot apply to frozen draws: the render is the problem set")
        self.catalog_sha256 = self._catalog_sha256(catalog)
        matches = [(name, e) for name, e in manifest["catalogs"].items() if e.get("source_sha256") == self.catalog_sha256]
        if not matches:
            held = ", ".join(f"{name} ({e.get('source')})" for name, e in sorted(manifest["catalogs"].items())) or "none"
            raise FrozenDrawsError(f"{self.root} holds no draws of catalog {catalog!r} (sha256 {self.catalog_sha256[:12]}); "
                                   f"it holds: {held}. Render this catalog's draws, or name the catalog file they were rendered from")
        self.catalog_name, entry = matches[0]
        info = entry["files"].get(str(self.repeat))
        if info is None:
            raise FrozenDrawsError(f"{self.root} holds repeat(s) {sorted(entry['files'])} of {self.catalog_name}, not {self.repeat}")
        self.path = self.root / info["path"]
        if not self.path.is_file():
            raise FrozenDrawsError(f"{self.path} is missing")
        self.sha256 = sha256_file(self.path)
        if self.sha256 != info["sha256"]:
            raise FrozenDrawsError(f"{self.path}: sha256 {self.sha256} differs from the manifest's {info['sha256']}")
        with np.load(self.path, allow_pickle=False) as data:
            meta = json.loads(data["_meta"].tobytes().decode("utf-8"))
        if (meta.get("catalog", {}).get("source_sha256") != self.catalog_sha256 or int(meta.get("repeat", -1)) != self.repeat
                or meta.get("settings") != self.settings):
            raise FrozenDrawsError(f"{self.path} names another catalog, repeat or settings than the manifest")
        self.records: list[dict[str, Any]] = list(meta["problems"])
        self.size = int(info.get("bytes", self.path.stat().st_size))

    # ----------------------------------------------------------------- checks
    def _check_sampling(self, sampling: Mapping[str, Any]) -> float:
        unknown = sorted(set(sampling) - set(_SAMPLING_KEYS))
        if unknown:
            raise FrozenDrawsError(f"sampling keys {unknown} do not apply to frozen draws (allowed: {', '.join(_SAMPLING_KEYS)})")
        for key in ("n_support", "n_validation", "layout", "max_trials"):
            if key in sampling and sampling[key] != self.settings[key]:
                raise FrozenDrawsError(f"sampling.{key} is {sampling[key]!r}; the draws were rendered with {self.settings[key]!r}")
        if sampling.get("problems_per_expression", 1) != 1:
            raise FrozenDrawsError("frozen draws hold one problem per expression (sampling.problems_per_expression: 1)")
        if sampling.get("method", "iterate") != "iterate":
            raise FrozenDrawsError("frozen draws are read in the render's order (sampling.method: iterate)")
        level = sampling.get("noise", 0.0)
        if isinstance(level, bool) or not isinstance(level, (int, float)) or not np.isfinite(level) or level < 0:
            raise FrozenDrawsError(f"frozen draws scale a stored noise vector: sampling.noise must be a number >= 0, got {level!r}")
        return float(level)

    @staticmethod
    def _catalog_sha256(catalog: Any) -> str:
        if not isinstance(catalog, (str, os.PathLike)):
            raise FrozenDrawsError("frozen draws need data_source.catalog to name a catalog file or a catalog name, not an inline catalog")
        from symbolic_data.resolver import resolve

        return sha256_file(resolve(os.fspath(catalog)).path)

    # ----------------------------------------------------------------- source protocol
    def size_hint(self) -> int:
        return len(self.records)

    def prepare(self, *, simplipy_engine: Any = None) -> None:
        """Nothing to prepare: the ground truth is read as rendered."""

    def identity(self) -> dict[str, Any]:
        """What a result records about the draws it read."""
        return {"format_version": FORMAT_VERSIONS[-1], "catalog": self.catalog_name, "catalog_sha256": self.catalog_sha256,
                "repeat": self.repeat, "path": str(self.path), "sha256": self.sha256, "size": self.size,
                "manifest_sha256": self.manifest_sha256, "settings": dict(self.settings), "noise": self.level,
                "acceptance": self.acceptance}

    def row_fields(self, index: int) -> dict[str, Any]:
        """What a result row records about its problem's draw: the attempt kept and whether the formula is flagged
        formula-inherent (no attempt let it be recognized)."""
        record = self.records[index]
        return {"draws_attempt": int(record.get("attempt", 1)), "draws_formula_inherent": bool(record.get("formula_inherent", False))}

    def __iter__(self) -> Iterator[Any]:
        from symbolic_data import Problem

        with np.load(self.path, allow_pickle=False) as data:
            for index, record in enumerate(self.records):
                if record["is_placeholder"]:
                    yield Problem.placeholder(variables=list(record["variables"]), reason=record["placeholder_reason"] or "placeholder",
                                              eq_id=record["eq_id"])
                    continue
                arrays = {name: data[f"p{index}__{name}"] for name in _CORE}
                optional = {name: data[f"p{index}__{name}"] for name in _OPTIONAL if f"p{index}__{name}" in data.files}
                if record["kind"] == "formula":
                    noise: Any = self.level
                    for side in ("y_support", "y_validation"):
                        optional.pop(f"{side}_noisy", None)
                    noisy = {"y_support_noisy": noised(arrays["y_support"], self.level, data[f"p{index}__noise_support"]),
                             "y_validation_noisy": noised(arrays["y_validation"], self.level, data[f"p{index}__noise_validation"])}
                else:
                    noise = record.get("noise", 0.0)
                    noisy = {f"{side}_noisy": optional.pop(f"{side}_noisy", arrays[side]).copy() for side in ("y_support", "y_validation")}
                yield Problem(
                    **arrays, **noisy, **optional,
                    skeleton=tuple(record["skeleton"]) if record["skeleton"] is not None else None,
                    expression=list(record["expression"]) if record["expression"] is not None else None,
                    constants=list(record["constants"]), variables=list(record["variables"]), complexity=record["complexity"],
                    noise=noise, eq_id=record["eq_id"], meta=dict(record["meta"] or {}), gt_kind=record["gt_kind"],
                )


def check_resume(existing: Mapping[str, Sequence[Any]], identity: Mapping[str, Any] | None) -> None:
    """Refuse to continue a result file whose rows were read from other draws (or drawn afresh) than this run reads."""
    rows = max((len(values) for key, values in existing.items() if key != "__meta__"), default=0)
    if rows == 0:
        return
    recorded = set(existing.get("draws_sha256") or [None] * rows)
    expected = identity["sha256"] if identity is not None else None
    if recorded != {expected}:
        def name(sha: Any) -> str:
            return "drawn afresh" if sha is None else str(sha)[:12]
        raise FrozenDrawsError(f"the result file's {rows} rows come from {', '.join(sorted(map(name, recorded)))}; this run "
                               f"reads {name(expected)}. Write to another output, or rerun from scratch")


__all__ = ["FORMAT", "FrozenDrawsError", "FrozenDrawsSource", "check_resume", "noised", "parse_draws_config", "sha256_file"]
