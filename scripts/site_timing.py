"""Build the results site's time axis (timing.json) from reference-machine result files.

The site places a method's rung on a time axis only with seconds measured on the reference machine. Two kinds of
measurement, both read from result files:

  * a timing ladder on the frozen subset (scripts/freeze_timing_subset.py): the rung's seconds are the POOLED mean fit
    time over the subset, each catalog weighted by its full size (scripts/timing_readout.py), so the number estimates
    the whole suite's mean;
  * a method evaluated on the whole suite on the reference machine itself: the rung's seconds are the mean fit time
    over the problems it answered.

A rung is timed only once it is complete: every catalog has its file with every problem (the subset's count, or the
catalog's full size). Failed problems do not count towards the time: a row with an error, prediction_success False, or
no finite fit_time is left out, and the number of timed rows is kept in __provenance__.

  site_timing.py --manifest configs/timing/timing_subset.json --out timing.json \\
      [--subset KEY=DIR ...] [--suite KEY=DIR ...] [--suite-pattern niter_{rung:05d}.pkl]

KEY is the method's key on the site; DIR holds <catalog>/choices_<rung>.pkl (subset) or <catalog>/<suite-pattern>
(suite). Every rung found in the files is considered; the output lists the complete ones.
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
from timing_readout import estimates, load_rung  # noqa: E402

import numpy as np  # noqa: E402

NOTE = ("Reference machine: one workstation (16 cores / 32 threads, one RTX 4090), one method at a time, on a frozen 262-problem stratified subset; "
        "each point is the size-weighted pooled mean fit time over the suite's strata. PySR is evaluated on the reference "
        "machine itself: its points are the mean over the whole suite.")


def rungs_in(directory: Path, pattern: str) -> list[int]:
    """Every rung some catalog of `directory` has a file for."""
    rx = re.compile("^" + re.escape(pattern).replace(re.escape("{rung:06d}"), r"(\d{6})").replace(re.escape("{rung:05d}"), r"(\d{5})") + "$")
    found = {int(m.group(1)) for p in directory.glob("*/*") if (m := rx.match(p.name))}
    return sorted(found)


def subset_rung(directory: Path, catalogs: dict[str, Any], rung: int, pattern: str) -> dict[str, Any] | None:
    """The pooled mean of one complete rung of a subset ladder, or None while a catalog is missing or short."""
    cols, missing, short = load_rung(directory.parent, "", directory.name, catalogs, rung, pattern, False)
    if missing or short:
        return None
    weights = {e: float(m["weight"]) for e, m in catalogs.items()}
    ident = {e: np.arange(int(m["count"])) for e, m in catalogs.items()}
    est = estimates(cols, weights, ident)
    return {"seconds": est["pooled_mean"], "n_ok": est["n_ok"], "n": est["n"]}


def suite_rung(directory: Path, catalogs: dict[str, Any], rung: int, pattern: str) -> dict[str, Any] | None:
    """The mean fit time over the answered problems of one complete rung of a whole-suite evaluation, or None."""
    total, ok, n, hung = 0.0, 0, 0, 0
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
        for j, t in enumerate(times):
            n += 1
            hung += 1 if j < len(hangs) and hangs[j] else 0
            failed = (j < len(errors) and errors[j]) or (j < len(success) and success[j] is False)
            if failed or t is None or t != t or t == float("inf"):
                continue
            total += float(t)
            ok += 1
    if not ok:
        return None
    return {"seconds": total / ok, "n_ok": ok, "n": n, "hung": hung}


def build(manifest: dict[str, Any], subset: dict[str, Path], suite: dict[str, Path], suite_pattern: str,
          subset_pattern: str = "choices_{rung:06d}.pkl") -> dict[str, Any]:
    catalogs = manifest["catalogs"]
    timing: dict[str, Any] = {}
    rows_timed: dict[str, dict[str, list[int]]] = {}
    hung_rows: dict[str, dict[str, int]] = {}
    for key, directory in subset.items():
        for rung in rungs_in(directory, subset_pattern):
            cell = subset_rung(directory, catalogs, rung, subset_pattern)
            if cell is None or cell["n"] < int(manifest["total_problems"]) or not math.isfinite(cell["seconds"]):
                continue
            timing.setdefault(key, {})[str(rung)] = round(cell["seconds"], 4)
            rows_timed.setdefault(key, {})[str(rung)] = [cell["n_ok"], cell["n"]]
    for key, directory in suite.items():
        for rung in rungs_in(directory, suite_pattern):
            cell = suite_rung(directory, catalogs, rung, suite_pattern)
            if cell is None:
                continue
            timing.setdefault(key, {})[str(rung)] = round(cell["seconds"], 4)
            rows_timed.setdefault(key, {})[str(rung)] = [cell["n_ok"], cell["n"]]
            hung_rows.setdefault(key, {})[str(rung)] = cell["hung"]
    timing["note"] = NOTE
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
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    manifest = json.loads(Path(a.manifest).read_text())
    timing = build(manifest, _pairs(a.subset, "--subset"), _pairs(a.suite, "--suite"), a.suite_pattern)
    tmp = a.out + ".tmp"
    Path(tmp).write_text(json.dumps(timing, indent=1))
    os.replace(tmp, a.out)
    print("timed:", {k: max(map(int, v)) for k, v in timing.items() if isinstance(v, dict) and not k.startswith("__") and v})
    return 0


if __name__ == "__main__":
    sys.exit(main())
