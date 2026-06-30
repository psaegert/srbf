#!/usr/bin/env python
"""Compact, STREAMING columnar store for the standard-eval THOROUGH tier's save-all candidates.

Motivation (see ../STANDARD_EVAL.md Section 7):
  - At c=1M a problem yields ~0.5-1M unique candidates. The naive "list[str] expression + int64 tokens"
    pickle is ~105-170 B/candidate (measured on 24 real result pickles) and, worse, ~1 GB/problem LIVE as
    Python objects -> RAM, not disk, is the binding constraint.
  - Fix: store ONLY unique candidates (dedup is already free, pre-refinement) in a COMPACT COLUMNAR layout
    (uint8 tokens since vocab=83<256, float32 scalars, uint8 flags), and STREAM: flush ONE problem at a time
    to its own compressed file, never holding the full pool or accumulating across problems.

This module is a standalone, dependency-light prototype (numpy only). A capture hook in the
generation path can be wired once integrated (the eval harness currently keeps only the single
best candidate -- model_adapters.py:162).

Layout: one compressed .npz per (problem) under <out_dir>/, columns:
  tokens   uint8   [sum(len)]      flat concatenation of all candidates' token-ids
  offsets  int64   [n_cand + 1]    tokens[offsets[i]:offsets[i+1]] is candidate i  (CSR-style)
  fvu      float32 [n_cand]
  log_prob float32 [n_cand]
  valid    uint8   [n_cand]        1/0
  fit_status uint8 [n_cand]        small enum (0=ok, 1=failed, ...)
  const_vals  float32 [sum(n_const)]   (optional) flat fitted constants
  const_off   int64   [n_cand + 1]     (optional) CSR offsets into const_vals
Per-problem files keep both write-time and read-time (analysis) memory bounded to one problem.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np


class CandidateStoreWriter:
    """Streaming writer: one compressed .npz per problem. Bounded memory."""

    def __init__(self, out_dir: str | Path, *, vocab_size: int) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if vocab_size > 256:
            raise ValueError(f"vocab_size {vocab_size} > 256 does not fit uint8; bump the token dtype")
        self.vocab_size = vocab_size
        # RESUME-SAFE: the eval loop is restartable (resume:true, save_every), so a writer is
        # re-created mid-campaign over a dir that already holds earlier problems' files. Rebuild the
        # index from the existing problem_*.npz (cheap: no array reads) so close()'s manifest covers
        # ALL problems, not just this resume slice. (The reader globs the dir anyway -- the manifest is
        # advisory -- but a complete manifest keeps the stats honest.) n_candidates is left None for
        # pre-existing files to avoid decompressing every array on init.
        self._index: list[dict] = []
        self._problem_ids: set[int] = set()
        for p in sorted(self.out_dir.glob("problem_*.npz")):
            pid = self._parse_problem_id(p.name)
            if pid is None:
                continue
            self._problem_ids.add(pid)
            self._index.append({"problem_id": pid, "n_candidates": None, "bytes": p.stat().st_size})

    @staticmethod
    def _parse_problem_id(name: str) -> int | None:
        stem = name[len("problem_"):-len(".npz")] if name.startswith("problem_") and name.endswith(".npz") else ""
        return int(stem) if stem.isdigit() else None

    def has_problem(self, problem_id: int) -> bool:
        """Already written (resume skip)?"""
        return int(problem_id) in self._problem_ids

    def write_problem(
        self,
        problem_id: int,
        token_lists: Sequence[Sequence[int]],
        fvu: Sequence[float],
        log_prob: Sequence[float],
        *,
        valid: Sequence[int] | None = None,
        fit_status: Sequence[int] | None = None,
        constants: Sequence[Sequence[float]] | None = None,
    ) -> int:
        """Flush ONE problem's full (deduped) candidate set. Returns bytes written."""
        n = len(token_lists)
        if not (len(fvu) == len(log_prob) == n):
            raise ValueError("fvu / log_prob length must match token_lists")

        # CSR-pack the ragged token streams as flat uint8 + int64 offsets (no padding).
        offsets = np.empty(n + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum([len(t) for t in token_lists], out=offsets[1:])
        tokens = np.empty(int(offsets[-1]), dtype=np.uint8)
        for i, t in enumerate(token_lists):
            tokens[offsets[i]:offsets[i + 1]] = np.asarray(t, dtype=np.uint8)

        cols: dict[str, np.ndarray] = {
            "tokens": tokens,
            "offsets": offsets,
            "fvu": np.asarray(fvu, dtype=np.float32),
            "log_prob": np.asarray(log_prob, dtype=np.float32),
            "valid": (np.ones(n, np.uint8) if valid is None else np.asarray(valid, np.uint8)),
            "fit_status": (np.zeros(n, np.uint8) if fit_status is None else np.asarray(fit_status, np.uint8)),
        }
        if constants is not None:
            coff = np.empty(n + 1, dtype=np.int64)
            coff[0] = 0
            np.cumsum([len(c) for c in constants], out=coff[1:])
            cvals = np.empty(int(coff[-1]), dtype=np.float32)
            for i, c in enumerate(constants):
                cvals[coff[i]:coff[i + 1]] = np.asarray(c, dtype=np.float32)
            cols["const_vals"] = cvals
            cols["const_off"] = coff

        path = self.out_dir / f"problem_{problem_id:06d}.npz"
        # Atomic: write to a temp file then rename, so an interrupted write never leaves a truncated
        # .npz that the globbing reader would later choke on.
        tmp = path.with_suffix(".npz.tmp")
        with tmp.open("wb") as fh:
            # numpy's savez_compressed stub doesn't model **{name: array} unpacking (it types the
            # variadic as bool/ArrayLike); the call is correct at runtime.
            np.savez_compressed(fh, **cols)  # type: ignore[arg-type]
        tmp.replace(path)
        nbytes = path.stat().st_size
        pid = int(problem_id)
        self._index = [e for e in self._index if e["problem_id"] != pid]  # update-or-append (resume rewrite)
        self._index.append({"problem_id": pid, "n_candidates": int(n), "bytes": int(nbytes)})
        self._problem_ids.add(pid)
        # Refresh the (advisory) manifest periodically so one exists even if the run is killed before
        # close() -- there is no eval teardown hook. The reader globs the dir regardless, so this is
        # convenience, not correctness.
        if len(self._index) % 64 == 0:
            self._write_manifest()
        return nbytes

    def _write_manifest(self) -> dict:
        """Write the advisory manifest (the reader globs the dir, so this never gates correctness).
        n_candidates may be None for problems inherited from a prior run (not re-read on init)."""
        index = sorted(self._index, key=lambda p: p["problem_id"])
        manifest = {
            "vocab_size": self.vocab_size,
            "n_problems": len(index),
            "total_candidates": sum((p["n_candidates"] or 0) for p in index),
            "total_bytes": sum(p["bytes"] for p in index),
            "problems": index,
        }
        tmp = self.out_dir / "manifest.json.tmp"
        tmp.write_text(json.dumps(manifest, indent=0))
        tmp.replace(self.out_dir / "manifest.json")
        return manifest

    def close(self) -> dict:
        """Write the final manifest and return summary stats (no candidate data held in RAM)."""
        return self._write_manifest()


class CandidateStoreReader:
    """Streaming reader: yields one problem's columns at a time (bounded memory)."""

    def __init__(self, out_dir: str | Path) -> None:
        self.out_dir = Path(out_dir)
        mpath = self.out_dir / "manifest.json"
        self.manifest = json.loads(mpath.read_text()) if mpath.exists() else None

    def problem_ids(self) -> list[int]:
        """Authoritative problem list = the .npz files on disk (NOT the manifest, which may be a stale
        resume slice). Sorted ascending."""
        ids = []
        for p in sorted(self.out_dir.glob("problem_*.npz")):
            stem = p.name[len("problem_"):-len(".npz")]
            if stem.isdigit():
                ids.append(int(stem))
        return sorted(ids)

    def __iter__(self) -> Iterator[dict]:
        for pid in self.problem_ids():
            with np.load(self.out_dir / f"problem_{pid:06d}.npz") as z:
                yield {k: z[k] for k in z.files}

    @staticmethod
    def candidate_tokens(block: dict, i: int) -> np.ndarray:
        off = block["offsets"]
        return block["tokens"][off[i]:off[i + 1]]


def _self_test() -> None:
    """Synthesize a realistic problem and verify compact size + round-trip (no GPU/data needed)."""
    import tempfile

    rng = np.random.default_rng(0)  # test-only; NOT a data seed
    n_cand = 50_000              # conservative c=1M unique-count placeholder
    vocab = 83
    lengths = np.clip(rng.normal(23.6, 4.0, n_cand).round().astype(int), 4, 35)
    token_lists = [rng.integers(0, vocab, L, dtype=np.uint8).tolist() for L in lengths]
    fvu = rng.random(n_cand).astype(np.float32)
    log_prob = (-rng.random(n_cand) * 50).astype(np.float32)
    constants = [rng.random(int(k)).tolist() for k in rng.integers(0, 3, n_cand)]

    with tempfile.TemporaryDirectory() as d:
        w = CandidateStoreWriter(d, vocab_size=vocab)
        nbytes = w.write_problem(0, token_lists, fvu.tolist(), log_prob.tolist(), constants=constants)
        man = w.close()
        per_cand = nbytes / n_cand
        print(f"n_candidates={n_cand:,}  mean_len={lengths.mean():.1f}")
        print(f"compressed bytes/candidate = {per_cand:.1f} B   (investigation target ~13-25 B gzip)")
        print(f"=> projected c=1M cell (2200 problems): {per_cand * n_cand * 2200 / 1e9:.1f} GB compressed "
              f"at this unique-count")

        # round-trip
        r = CandidateStoreReader(d)
        blk = next(iter(r))
        assert blk["fvu"].shape[0] == n_cand
        assert np.array_equal(CandidateStoreReader.candidate_tokens(blk, 7), np.asarray(token_lists[7], np.uint8))
        assert np.allclose(blk["fvu"], fvu) and np.allclose(blk["log_prob"], log_prob)
        print("round-trip OK (tokens + scalars match);  manifest total_candidates =", man["total_candidates"])


if __name__ == "__main__":
    _self_test()
