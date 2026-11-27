"""Build the results site's time axis (timing.json) from reference-machine result files.

The site places a method's rung on a time axis only with seconds measured on the reference machine. Two kinds of
measurement, both read from result files:

  * a timing ladder on the frozen subset (scripts/freeze_timing_subset.py);
  * a method evaluated on the whole suite on the reference machine itself.

Either way the rung's seconds are the mean fit time over the problems answered, averaged over benchmarks > subsets
like every other number on the site (random effects, scripts/site_random_effects.py), so no benchmark decides the time
on its own. The manifest records each catalog's benchmark ("benchmark") and the subset of each of its problems
("subsets"), both written by scripts/freeze_timing_subset.py from the catalogs; a catalog without them is a benchmark
of its own and one subset.

A rung is timed only once it is complete: every catalog has its file with every problem (the subset's count, or the
catalog's full size). Failed problems do not count towards the time: a row with an error, prediction_success False, or
no finite fit_time is left out, and the number of timed rows is kept in __provenance__.

  site_timing.py --manifest configs/timing/timing_subset.json --out timing.json \\
      [--subset KEY=DIR ...] [--suite KEY=DIR ...] [--suite-pattern niter_{rung:05d}.pkl] [--pattern KEY=PATTERN ...]

KEY is the method's key on the site; DIR holds <catalog>/choices_<rung>.pkl (subset) or <catalog>/<suite-pattern>
(suite). --pattern names the rung files of one KEY instead: a method whose ladder counts iterations keeps its
niter_{rung:05d}.pkl files on the subset too. Every rung found in the files is considered; the output lists the
complete ones.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from site_random_effects import SetStat, combine_nested  # noqa: E402
from timing_readout import load_rung  # noqa: E402

import numpy as np  # noqa: E402

NOTE = ("Measured on one workstation (16 CPU cores, one RTX 4090 GPU), one method at a time, on a fixed sample of 262 problems. "
        "Each point averages over the problems where the method returned a usable formula, with the problem sets weighted "
        "as in every other average on the site.")


def over_sets(per_catalog: dict[str, Any], benchmarks: dict[str, str] | None = None,
              subsets: dict[str, Any] | None = None) -> float | None:
    """The seconds per problem: the mean over the answered (finite) problems, averaged over benchmarks > subsets as
    every average on the site (srbf.aggregate.combine_nested). ``per_catalog`` holds each catalog's times;
    ``benchmarks`` maps a catalog to its benchmark (a catalog it does not name is a benchmark of its own), and
    ``subsets`` a catalog to the subset of each of its times (aligned with them; a catalog it does not name is one
    subset, named after itself). Times of one benchmark and one subset name are one subset, whichever catalog holds
    them."""
    groups: dict[str, dict[str, list[np.ndarray]]] = {}
    for c, x in per_catalog.items():
        t = np.asarray(x, float)
        if not t.size:
            continue
        subs = groups.setdefault((benchmarks or {}).get(c, c), {})
        labels = (subsets or {}).get(c)
        if labels is None:
            subs.setdefault(c, []).append(t)
            continue
        labels = np.asarray(labels, dtype=object)
        if labels.shape != t.shape:
            raise ValueError(f"catalog {c!r}: {labels.size} subsets for {t.size} times")
        for u in dict.fromkeys(labels.tolist()):
            subs.setdefault(str(u), []).append(t[labels == u])
    stats = []
    for subs in groups.values():
        g = []
        for parts in subs.values():
            t = parts[0] if len(parts) == 1 else np.concatenate(parts)
            g.append(SetStat(int(t.size), float(t.sum()), float((t * t).sum())))
        stats.append(g)
    fit = combine_nested(stats, "normal")
    return None if fit is None else fit.mu


def benchmarks_of(catalogs: dict[str, Any]) -> dict[str, str]:
    """{catalog: benchmark} from the manifest's catalog entries; an entry without one is a benchmark of its own."""
    return {c: str(m.get("benchmark") or c) for c, m in catalogs.items()}


def subset_of_row(catalog: str, meta: dict[str, Any]) -> Any:
    """The subset of a catalog's problem from its row (its eval_row_index in the full catalog), as the manifest records
    the catalog's subsets (scripts/freeze_timing_subset.py: "subsets", {"names": [...], "rows": [each row's index into
    names]}, "rows" left out when there is one name); a catalog the manifest records none for is one subset, named
    after itself."""
    rec = meta.get("subsets") or {}
    names, rows = rec.get("names") or [catalog], rec.get("rows")

    def of(row: int) -> str:
        if rows is None:
            return str(names[0])
        if not 0 <= int(row) < len(rows):
            raise ValueError(f"catalog {catalog!r}: row {row} is not among its {len(rows)} problems")
        return str(names[rows[int(row)]])
    return of


def note(suite_keys: list[str]) -> str:
    """The time axis's note; a method timed on the whole suite instead of the subset is named in it."""
    if not suite_keys:
        return NOTE
    one = len(suite_keys) == 1
    return (f"{NOTE} {', '.join(suite_keys)} {'was' if one else 'were'} run on all problems on this workstation, so "
            f"{'its' if one else 'their'} points average over all problems, not over the sample.")


def rungs_in(directory: Path, pattern: str) -> list[int]:
    """Every rung some catalog of `directory` has a file for."""
    # A width pads to AT LEAST that many digits ({rung:07d} names 2^24 evals_16777216.pkl), so a file is a rung's when
    # the pattern spells that rung as exactly its name, whatever the width ({rung:05d}, {rung:06d}, {rung:07d}, {rung}).
    parts = re.split(r"\{rung(?::0?\d+d)?\}", pattern)
    if len(parts) != 2:
        raise ValueError(f"a rung-file pattern names the rung exactly once: {pattern!r}")
    head, tail = parts
    rx = re.compile("^" + re.escape(head) + r"(\d+)" + re.escape(tail) + "$")
    found = {int(m.group(1)) for p in directory.glob("*/*")
             if (m := rx.match(p.name)) and pattern.format(rung=int(m.group(1))) == p.name}
    return sorted(found)


def subset_rung(directory: Path, catalogs: dict[str, Any], rung: int, pattern: str) -> dict[str, Any] | None:
    """The seconds of one complete rung of a subset ladder, or None while a catalog is missing or short."""
    cols, missing, short = load_rung(directory.parent, "", directory.name, catalogs, rung, pattern, False)
    if missing or short:
        return None
    times = {e: c["fit_time"][np.isfinite(c["fit_time"])] for e, c in cols.items()}
    # the subset ladder's row j is the catalog's row source_row_index[j]
    labels = {e: np.asarray([subset_of_row(e, catalogs[e])(r) for r in catalogs[e].get("source_row_index", range(len(c["fit_time"])))],
                            dtype=object)[np.isfinite(c["fit_time"])] for e, c in cols.items()}
    seconds = over_sets(times, benchmarks_of(catalogs), labels)
    if seconds is None:
        return None
    return {"seconds": seconds, "n_ok": int(sum(t.size for t in times.values())), "n": int(sum(c["fit_time"].size for c in cols.values()))}


def suite_rung(directory: Path, catalogs: dict[str, Any], rung: int, pattern: str) -> dict[str, Any] | None:
    """The seconds of one complete rung of a whole-suite evaluation, or None."""
    per: dict[str, list[float]] = {}
    labels: dict[str, list[str]] = {}
    n, hung = 0, 0
    for catalog, meta in catalogs.items():
        path = directory / catalog / pattern.format(rung=rung)
        if not path.exists():
            return None
        with open(path, "rb") as fh:
            snap = pickle.load(fh)
        times = snap["fit_time"]
        if len(times) < int(meta["size"]):
            return None
        errors, success = snap.get("error") or [], snap.get("prediction_success") or []
        hangs = snap.get("worker_hangs") or []
        rows = snap.get("eval_row_index") or list(range(len(times)))
        of = subset_of_row(catalog, meta)
        per[catalog], labels[catalog] = [], []
        for j, t in enumerate(times):
            n += 1
            hung += 1 if j < len(hangs) and hangs[j] else 0
            failed = (j < len(errors) and errors[j]) or (j < len(success) and success[j] is False)
            if failed or t is None or t != t or t == float("inf"):
                continue
            per[catalog].append(float(t))
            labels[catalog].append(of(rows[j]))
    ok = sum(len(x) for x in per.values())
    seconds = over_sets({c: np.asarray(x) for c, x in per.items()}, benchmarks_of(catalogs), labels) if ok else None
    if seconds is None:
        return None
    return {"seconds": seconds, "n_ok": ok, "n": n, "hung": hung}


def build(manifest: dict[str, Any], subset: dict[str, Path], suite: dict[str, Path], suite_pattern: str,
          subset_pattern: str = "choices_{rung:06d}.pkl", patterns: dict[str, str] | None = None) -> dict[str, Any]:
    """``patterns`` overrides the rung file name of single keys (a subset ladder that keeps niter_ files, say)."""
    catalogs = manifest["catalogs"]
    patterns = patterns or {}
    timing: dict[str, Any] = {}
    rows_timed: dict[str, dict[str, list[int]]] = {}
    hung_rows: dict[str, dict[str, int]] = {}
    for key, directory in subset.items():
        pattern = patterns.get(key, subset_pattern)
        for rung in rungs_in(directory, pattern):
            cell = subset_rung(directory, catalogs, rung, pattern)
            if cell is None or cell["n"] < int(manifest["total_problems"]) or not math.isfinite(cell["seconds"]):
                continue
            timing.setdefault(key, {})[str(rung)] = round(cell["seconds"], 4)
            rows_timed.setdefault(key, {})[str(rung)] = [cell["n_ok"], cell["n"]]
    for key, directory in suite.items():
        pattern = patterns.get(key, suite_pattern)
        for rung in rungs_in(directory, pattern):
            cell = suite_rung(directory, catalogs, rung, pattern)
            if cell is None:
                continue
            timing.setdefault(key, {})[str(rung)] = round(cell["seconds"], 4)
            rows_timed.setdefault(key, {})[str(rung)] = [cell["n_ok"], cell["n"]]
            hung_rows.setdefault(key, {})[str(rung)] = cell["hung"]
    timing["note"] = note(list(suite))
    timing["__provenance__"] = {"rule": manifest.get("rule"), "total_problems": manifest.get("total_problems"),
                                "rows_timed": rows_timed, "hung_rows": hung_rows,
                                "updated": dt.datetime.now().strftime("%Y-%m-%d %H:%M")}
    return timing


def _pairs(specs: list[str] | None, flag: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for spec in specs or []:
        key, sep, path = spec.partition("=")
        if not sep:
            sys.exit(f"{flag} takes KEY=DIR, got {spec!r}")
        out[key] = Path(path)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="the frozen subset (catalogs: size, count, weight)")
    ap.add_argument("--subset", action="append", default=None, metavar="KEY=DIR", help="a timing ladder on the frozen subset")
    ap.add_argument("--suite", action="append", default=None, metavar="KEY=DIR", help="a whole-suite evaluation on the reference machine")
    ap.add_argument("--suite-pattern", default="niter_{rung:05d}.pkl", help="file name of a whole-suite rung")
    ap.add_argument("--pattern", action="append", default=None, metavar="KEY=PATTERN",
                    help="the rung file name of KEY's directory, e.g. PySR=niter_{rung:05d}.pkl for an iteration ladder on the subset")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    manifest = json.loads(Path(a.manifest).read_text())
    patterns = {k: str(v) for k, v in _pairs(a.pattern, "--pattern").items()}
    timing = build(manifest, _pairs(a.subset, "--subset"), _pairs(a.suite, "--suite"), a.suite_pattern, patterns=patterns)
    tmp = a.out + ".tmp"
    Path(tmp).write_text(json.dumps(timing, indent=1))
    os.replace(tmp, a.out)
    print("timed:", {k: max(map(int, v)) for k, v in timing.items() if isinstance(v, dict) and not k.startswith("__") and v})
    return 0


if __name__ == "__main__":
    sys.exit(main())
