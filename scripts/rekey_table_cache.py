"""Carry an `srbf table` cache over to a new judge fingerprint, for a change that leaves judging alone.

`srbf table` keys every cached file by the judge's fingerprint, a hash of srbf's source (srbf.table.judge_fingerprint).
A change to how files are read, cached or scheduled moves the fingerprint as much as a change to judging does, and the
next table judges every file again: for the results board about 23,700 files, most of a day. This script links each
file's entry under the fingerprint it was judged with (--from) to its key under the current one, for the same trees
and options the table is run with. It never judges a file into the cache and never replaces an entry that exists.

Use it only for a change that leaves judging alone, and check that claim: --check N judges N carried files afresh with
the installed srbf and compares their rows with the carried ones. Any difference exits 1 and names the file; the
carried links stay, so delete them (or the cache) before the next table.

  rekey_table_cache.py --cache DIR --from FINGERPRINT --tree METHOD:DRAW:PATH [--tree ...]
      [--index-variables METHOD=FIRST ...] [--engine acj-5-4-llm] [--max-rung R] [--check N]
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import sys

from srbf.table import (ResultTree, _cache_key, _cache_load, _cache_path, _worker_engine, judge_fingerprint,
                        judge_result_file, parse_index_bases, result_files)


def rekey(trees: list[ResultTree], cache_dir: str, old: str, engine: str = "acj-5-4-llm",
          max_rung: int | None = None) -> dict:
    """Link every file's entry under fingerprint ``old`` to its key under the current one.

    Returns the counts (files, carried, present: already under the current key, missing: no entry under ``old``) and
    the carried files as (tree, path) pairs."""
    new = judge_fingerprint(engine)
    out = {"fingerprint": new, "files": 0, "carried": [], "present": 0, "missing": 0}
    for tree in trees:
        for path in result_files(tree, max_rung=max_rung):
            out["files"] += 1
            dst = _cache_path(cache_dir, _cache_key(tree, path, new))
            if os.path.exists(dst):
                out["present"] += 1
                continue
            src = _cache_path(cache_dir, _cache_key(tree, path, old))
            if not os.path.exists(src):
                out["missing"] += 1
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                os.link(src, dst)
            except OSError:                               # a file system without hard links
                shutil.copy2(src, dst)
            out["carried"].append((tree, path))
    return out


def _same(x: object, y: object) -> bool:
    """Cell equality where a missing number (NaN) equals itself: a row with one is the same row judged twice."""
    return x == y or (isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y))


def same_rows(a: list[list[object]] | None, b: list[list[object]]) -> bool:
    return a is not None and len(a) == len(b) and all(
        len(r) == len(s) and all(_same(x, y) for x, y in zip(r, s)) for r, s in zip(a, b))


def check(carried: list[tuple[ResultTree, str]], cache_dir: str, engine: str, n: int) -> list[str]:
    """Judge ``n`` carried files afresh (spread over the list, so every method is likely in it) and compare their rows
    with the carried ones; returns the files that differ."""
    if not carried or n <= 0:
        return []
    fingerprint = judge_fingerprint(engine)
    picked = sorted(carried, key=lambda tp: (tp[0].method, tp[0].draw, tp[1]))
    picked = [picked[round(k * (len(picked) - 1) / max(1, n - 1))] for k in range(min(n, len(picked)))]
    differ = []
    for tree, path in dict.fromkeys(picked):
        stored = _cache_load(cache_dir, _cache_key(tree, path, fingerprint))
        fresh = judge_result_file(path, method=tree.method, draw=tree.draw, engine=_worker_engine(engine),
                                  first_index=tree.first_index)
        if not same_rows(stored, fresh):
            differ.append(path)
    return differ


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True, metavar="DIR")
    ap.add_argument("--from", dest="old", required=True, metavar="FINGERPRINT", help="the fingerprint the entries were judged with")
    ap.add_argument("--tree", action="append", required=True, metavar="METHOD:DRAW:PATH")
    ap.add_argument("--index-variables", action="append", default=None, metavar="METHOD=FIRST")
    ap.add_argument("--engine", default="acj-5-4-llm")
    ap.add_argument("--max-rung", type=int, default=None)
    ap.add_argument("--check", type=int, default=0, metavar="N", help="judge N carried files afresh and compare")
    a = ap.parse_args(argv)
    bases = parse_index_bases(a.index_variables)
    trees = [ResultTree.parse(spec) for spec in a.tree]
    trees = [ResultTree(t.method, t.draw, t.path, bases.get(t.method)) for t in trees]
    r = rekey(trees, a.cache, a.old, engine=a.engine, max_rung=a.max_rung)
    if r["fingerprint"] == a.old:
        print(f"the current fingerprint is {a.old}: nothing to carry")
        return 0
    print(f"{a.old} -> {r['fingerprint']}: {r['files']} files, {len(r['carried'])} carried, "
          f"{r['present']} already under the current fingerprint, {r['missing']} without an entry (the table judges those)")
    differ = check(r["carried"], a.cache, a.engine, a.check)
    if a.check:
        print(f"check: {min(a.check, len(r['carried']))} carried files judged afresh, {len(differ)} differ")
        for path in differ:
            print(f"DIFFERS {path}")
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main())
