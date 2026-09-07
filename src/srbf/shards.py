"""Sharding one run across processes: the shard's slice of a catalog, its output name, the merge.

A unit of work that is too long for one GPU (erbench-syneq at 16,384 choices: 5,301 problems at
two to three minutes each) is split by ``srbf run --shard K/N``: shard ``K`` evaluates every
``N``-th problem starting at ``K`` (interleaved, so the shards balance across a catalog's
ordering), writes ``<output>.shard-K-of-N.<ext>`` and resumes on its own. ``srbf merge`` puts the
shards back into the unsharded output file the analysis reads. Each shard's ``__meta__`` carries
``shard: {index, count}``; the merged file carries ``shards`` instead.
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from flash_ansr.utils.paths import substitute_root_path

from srbf.store import ResultStore

_SHARD_SUFFIX = re.compile(r"\.shard-(\d+)-of-(\d+)$")


def parse_shard(text: str) -> tuple[int, int]:
    """``"K/N"`` -> ``(K, N)`` with ``0 <= K < N``."""
    parts = str(text).split("/")
    if len(parts) != 2 or not all(p.strip().isdigit() for p in parts):
        raise ValueError(f"--shard expects K/N with two non-negative integers, got {text!r}")
    index, count = int(parts[0]), int(parts[1])
    if count < 1 or index >= count:
        raise ValueError(f"--shard K/N needs 0 <= K < N, got {text!r}")
    return index, count


def shard_share(total: int, index: int, count: int) -> int:
    """How many of ``total`` interleaved problems fall to shard ``index`` of ``count``."""
    if total <= 0:
        return 0
    return len(range(index, int(total), count))


def shard_output_path(path: str, index: int, count: int) -> str:
    """``results/x/choices_016384.pkl`` -> ``results/x/choices_016384.shard-0-of-4.pkl``."""
    p = Path(path)
    return str(p.with_name(f"{p.stem}.shard-{index}-of-{count}{p.suffix}"))


def _load(path: str) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    resolved = Path(substitute_root_path(str(path)))
    with resolved.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: not a results mapping")
    meta = dict(payload.get("__meta__") or {})
    rows = {k: list(v) for k, v in payload.items() if k != "__meta__"}
    return rows, meta


def merge_shards(paths: Sequence[str], output: str, *, allow_partial: bool = False) -> dict[str, Any]:
    """Merge shard result files into ``output``; returns a summary of what was merged.

    Every input must carry ``__meta__.shard`` with one common ``count``; indices must be distinct
    and, unless ``allow_partial``, complete. Rows are concatenated and ordered by
    ``eval_row_index`` (which must be disjoint across shards); the columns must agree.
    """
    if not paths:
        raise ValueError("merge_shards: no shard files given")
    loaded: list[tuple[int, dict[str, list[Any]], dict[str, Any], str]] = []
    count: int | None = None
    for path in paths:
        rows, meta = _load(path)
        shard = meta.get("shard")
        if not isinstance(shard, Mapping) or "index" not in shard or "count" not in shard:
            raise ValueError(f"{path}: __meta__.shard is missing; only files written by `srbf run --shard` merge")
        index, this_count = int(shard["index"]), int(shard["count"])
        if count is None:
            count = this_count
        elif this_count != count:
            raise ValueError(f"{path}: shard count {this_count} differs from {count}")
        loaded.append((index, rows, meta, str(path)))
    assert count is not None
    indices = [index for index, _, _, _ in loaded]
    if len(set(indices)) != len(indices):
        raise ValueError(f"duplicate shard indices: {sorted(indices)}")
    missing = sorted(set(range(count)) - set(indices))
    if missing and not allow_partial:
        raise ValueError(f"missing shards {missing} of {count}; pass --allow-partial to merge what is there")
    columns = set(loaded[0][1])
    for index, rows, _, path in loaded[1:]:
        if set(rows) != columns:
            raise ValueError(f"{path}: columns differ from the first shard ({sorted(set(rows) ^ columns)})")
    merged: dict[str, list[Any]] = {column: [] for column in loaded[0][1]}
    for index, rows, _, _ in sorted(loaded, key=lambda item: item[0]):
        for column, values in rows.items():
            merged[column].extend(values)
    if "eval_row_index" in merged:
        order = merged["eval_row_index"]
        if len(set(order)) != len(order):
            raise ValueError("shards overlap: eval_row_index values repeat across the inputs")
        permutation = sorted(range(len(order)), key=lambda i: order[i])
        merged = {column: [values[i] for i in permutation] for column, values in merged.items()}
    base_meta = {k: v for k, v in loaded[0][2].items() if k != "shard"}
    base_meta["shards"] = {"count": count, "merged": sorted(indices), "partial": bool(missing),
                           "sources": [Path(p).name for _, _, _, p in sorted(loaded, key=lambda item: item[0])]}
    ResultStore(merged).save(output, meta=base_meta)
    n_rows = len(next(iter(merged.values()))) if merged else 0
    return {"output": str(output), "rows": n_rows, "shards": sorted(indices), "count": count, "missing": missing}


__all__ = ["merge_shards", "parse_shard", "shard_output_path", "shard_share"]
