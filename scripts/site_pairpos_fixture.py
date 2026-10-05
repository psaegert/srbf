"""Write the browser fixture for positions between budgets: results-site/tests/fixtures/pairpos/.

The site suite runs against the committed release, which may not hold pv/, ranks/ and paired/ yet, so its tests of the
Ranks and Paired views at a position route these files instead. They are written by the exporter's own code
(scripts/site_export_v2.py) from real rows: one problem set (feynman) and three methods whose budget ladders differ
(T8-20M and PySR run 1, 2, 4, ...; dsr runs 1000, 2000, ...), so positions between budgets interpolate.

  index.js        appended to the release's results.js: D.pv, D.slots, the methods' timing and their feynman cells
                  (so the page brackets them from the same results the files hold), and D.rank_keys
  pv/ ranks/ paired/
                  the files for log10_fvu_val and numeric_recovery_val, at their paths under data/<release>/
  expected.json   what the exporter's code and the page's averaging (scripts/site_random_effects.py) give at four
                  positions: per method pair the chance to beat on log10_fvu_val, and the paired contrast on
                  numeric_recovery_val of T8-20M minus PySR

    python scripts/site_pairpos_fixture.py <rows dir> <timing.json>

<rows dir> holds rows_full_*.csv with the feynman rows of the three methods (srbf table's columns; other rows are
ignored), <timing.json> the reference-machine timing (timing.json of the campaign root).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))
import site_export_v2 as sx  # noqa: E402
import site_random_effects as re_  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results-site", "tests", "fixtures", "pairpos")
REL = "2026-09"
CATALOG = "feynman"
METHODS = ["T8-20M", "PySR", "dsr"]
KEYS = ["log10_fvu_val", "numeric_recovery_val"]
# 1024: T8-20M and PySR ran it, dsr sits between 1000 and 2000; 2000: dsr ran it, the others sit between 1024 and
# 2048; 1448: all three between two budgets; t7: seven seconds per problem, all three between two timed budgets.
POSITIONS = ["1024", "2000", "1448", "t7"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("timing")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    loaded = sx.load_rows(a.rows)
    data = {m: {cr: rows for cr, rows in loaded.get(m, {}).items() if cr[0] == CATALOG} for m in METHODS}
    sizes = {c["key"]: c["laws"] for c in sx.catalog_meta(os.path.join(os.path.dirname(__file__), "..", "results-site", "data", "catalog_mu.json"), {})}
    sizes = {CATALOG: sizes[CATALOG]}
    timing = {m: v for m, v in json.load(open(a.timing)).items() if m in METHODS}
    order = [m[0] for m in sx.METHODS if m[0] in METHODS]
    pairs = [(ka, kb) for i, ka in enumerate(order) for kb in order[i + 1:]]
    rungs = sorted({r for m in METHODS for (_c, r) in data[m] if sx.usable(m, r)})
    slots = [str(r) for r in rungs] + [sx.budget_key(t) for t in sx.TIME_BUDGETS]
    at = sx.brackets(data, METHODS, sizes, timing, slots)
    ranks, paired = sx.slot_cells(data, pairs, at, slots, KEYS)
    cells = {m: {str(r): sx.summarize_cell(rows, sizes[CATALOG]) for (_c, r), rows in sorted(data[m].items()) if sx.usable(m, r)} for m in METHODS}

    out = os.path.abspath(a.out)
    if os.path.isdir(out):
        shutil.rmtree(out)
    with tempfile.TemporaryDirectory() as tmp:
        pv = sx.write_pair_values(tmp, REL, data, METHODS, sizes, KEYS)
        sx.write_slot_cells(tmp, REL, ranks, paired)
        for d, _sub, files in os.walk(tmp):
            for f in files:
                rel = os.path.relpath(os.path.join(d, f), tmp)
                keep = (rel.startswith("pv" + os.sep) and f in ["frame.js"] + [k + ".js" for k in KEYS]) or \
                    (rel.split(os.sep)[0] in ("ranks", "paired") and f in [k + ".js" for k in KEYS])
                if keep:
                    os.makedirs(os.path.dirname(os.path.join(out, rel)), exist_ok=True)
                    shutil.copy(os.path.join(tmp, rel), os.path.join(out, rel))
    index = {"pv": pv, "slots": {"rungs": rungs, "budgets": [sx.budget_key(t) for t in sx.TIME_BUDGETS], "seconds": sx.TIME_BUDGETS,
                                 "at": at, "stamp": sx.stamps(at), "basis": {}},
             "timing": timing, "cells": cells, "rank_keys": [k for k in sx.RANK_KEYS if k not in sx.COPY_OF]}
    with open(os.path.join(out, "index.js"), "w") as fh:
        fh.write(";(function(){var D=window.RESULTS_V2,I=%s;D.pv=I.pv;D.slots=I.slots;D.rank_keys=I.rank_keys;"
                 "Object.keys(I.timing).forEach(function(m){D.timing[m]=I.timing[m];});"
                 "Object.keys(I.cells).forEach(function(m){D.cells[m]=D.cells[m]||{};D.cells[m][%s]=I.cells[m];});})();\n"
                 % (json.dumps(index, separators=(",", ":")), json.dumps(CATALOG)))

    expected: dict[str, Any] = {"release": REL, "catalogs": [CATALOG], "methods": METHODS, "positions": {}}
    where = sx.brackets(data, METHODS, sizes, timing, POSITIONS)
    for pos in POSITIONS:
        b = {m: where[m].get(pos, {}).get(CATALOG) for m in METHODS}
        entry: dict[str, Any] = {"brackets": b, "chance_to_beat": {}, "paired": None}

        def at_pos(m: str) -> Any:
            r1, r2, w = b[m]
            return sx.position(data[m][(CATALOG, int(r1))], data[m][(CATALOG, int(r2))], w)
        for ka, kb in pairs:
            if b[ka] is None or b[kb] is None:
                continue
            n, s1, s2 = sx.rank_pair_cell_at(at_pos(ka), at_pos(kb), ["log10_fvu_val"])
            fit = re_.combine([re_.SetStat(int(n), s1, s2)], "normal")
            assert fit is not None
            entry["chance_to_beat"][ka + "|" + kb] = {"cell": [n, s1, s2], "mu": fit.mu, "p": (1 + fit.mu) / 2, "lo": (1 + fit.lo) / 2,
                                                      "hi": (1 + fit.hi) / 2, "pvalue": fit.p, "n": int(n)}
        if b["T8-20M"] is not None and b["PySR"] is not None:
            cell = sx.paired_cell_at(at_pos("T8-20M"), at_pos("PySR"))
            assert cell is not None
            t = cell["m"]["numeric_recovery_val"]
            fit = re_.combine([re_.SetStat(int(t[0]), t[1], t[2])], "normal")
            assert fit is not None
            entry["paired"] = {"pair": "T8-20M - PySR", "key": "numeric_recovery_val", "cell": t, "v": fit.mu, "lo": fit.lo, "hi": fit.hi,
                               "p": fit.p, "wins": t[3], "losses": t[4], "n": int(t[0])}
        expected["positions"][pos] = entry
    with open(os.path.join(out, "expected.json"), "w") as fh:
        json.dump(expected, fh, indent=1)
    with open(os.path.join(out, "README.md"), "w") as fh:
        fh.write("Browser fixture for the Ranks and Paired views at a position (feynman; T8-20M, PySR, dsr), written by "
                 "`python scripts/site_pairpos_fixture.py <rows dir> <timing.json>` (see that script's docstring).\n")
    total = sum(os.path.getsize(os.path.join(d, name)) for d, _s, fs in os.walk(out) for name in fs)
    print(f"{out}: {sum(len(fs) for _d, _s, fs in os.walk(out))} files, {total // 1024} kB")


if __name__ == "__main__":
    main()
