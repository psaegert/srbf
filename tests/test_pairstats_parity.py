"""results-site/pairstats.js computes the explorer's paired contrasts and pairwise rank outcomes in the browser, from the
per-problem files the exporter writes (pv/, scripts/site_export_v2.py write_pair_values). At a budget a method ran it
must return what paired_cell and rank_pair_cell return, bit for bit; between two budgets it must follow the rule its
header documents, which reduces to the exporter's at w = 0 and w = 1. The gate runs the page's own file under node.

PAIRSTATS_ROWS, when set, names a directory of rows_full_*.csv (load_rows's input, e.g. a few catalogs cut from the
board's table) to run the same parity on real results as well."""
import csv
import importlib.util
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
JS = ROOT / "results-site" / "pairstats.js"
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node runs the page's pairstats.js")


def _exporter() -> Any:
    spec = importlib.util.spec_from_file_location("site_export_v2", ROOT / "scripts" / "site_export_v2.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sx = _exporter()
REL = "test-release"

# The node side: load pairstats.js and the pv files, read every job's two positions, return the results as JSON.
DRIVER = r"""
const fs = require("fs"), vm = require("vm");
globalThis.window = globalThis;
const PS = require(process.argv[1]);
const job = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
for (const f of job.files) { vm.runInThisContext(fs.readFileSync(f, "utf8"), { filename: f }); }
const file = (m, c, r) => { const f = PS.stored(job.rel, m, c, r); if (!f) { throw new Error("no pv file " + [m, c, r]); } return f; };
const pos = (p) => PS.at(file(p[0], p[1], p[2]), p[3] === null ? null : file(p[0], p[1], p[3]), p[4]);
const hex = (x) => { const b = Buffer.alloc(8); b.writeDoubleLE(x); return b.toString("hex"); };
const out = { cells: job.jobs.map((j) => j.kind === "paired" ? PS.pairedCell(pos(j.a), pos(j.b)) : PS.rankPairCell(pos(j.a), pos(j.b), j.keys)) };
if (job.sums) { out.sums = job.sums.map(PS.sums); out.pairwise = job.sums.map((xs) => hex(PS.pairwise(xs, 0, xs.length))); }
if (job.round) { out.round = job.round.map(PS.round6); }
if (job.columns) { out.columns = job.columns.map((c) => Buffer.from(PS.column(c.col, c.n).buffer).toString("hex")); }
if (job.constants) { out.constants = { paired: PS.PAIRED_KEYS, rates: PS.RATE_KEYS, worst: PS.WORST, answered: PS.ANSWERED }; }
process.stdout.write(JSON.stringify(out));
"""


def run_js(tmp: Path, job: dict[str, Any], js: Path = JS) -> dict[str, Any]:
    spec = tmp / "job.json"
    spec.write_text(json.dumps(job))
    done = subprocess.run([str(NODE), "-e", DRIVER, str(js), str(spec)], capture_output=True, text=True, timeout=1800)
    assert done.returncode == 0, done.stderr[-4000:]
    return json.loads(done.stdout)


def write_pv(tmp: Path, data: dict[str, Any], sizes: dict[str, int]) -> list[str]:
    out = tmp / "release"
    sx.write_pair_values(str(out), REL, data, sorted(data), sizes)
    return sorted(str(p) for p in (out / "pv").rglob("*.js"))


def same(py: Any, js: Any, where: str = "") -> None:
    """Exactly equal: counts as whole numbers, floats bit for bit as JSON carries them (shortest round-trip repr; JSON
    has no -0, so a zero is a zero)."""
    if py is None:
        assert js is None, f"{where}: python None, js {js!r}"
    elif isinstance(py, dict):
        assert isinstance(js, dict) and list(py) == list(js), f"{where}: keys {list(py)} vs {js!r}"
        for k in py:
            same(py[k], js[k], f"{where}.{k}")
    elif isinstance(py, (list, tuple)):
        assert isinstance(js, list) and len(py) == len(js), f"{where}: {py} vs {js}"
        for i, (a, b) in enumerate(zip(py, js)):
            same(a, b, f"{where}[{i}]")
    elif isinstance(py, int):
        assert isinstance(js, (int, float)) and float(js).is_integer() and int(js) == py, f"{where}: count {py} vs {js!r}"
    else:
        assert isinstance(js, (int, float)) and not isinstance(js, bool) and float(js) == py, f"{where}: {py!r} vs {js!r}"


def mismatches(expect: list[Any], got: list[Any]) -> int:
    bad = 0
    for py, js in zip(expect, got):
        try:
            same(py, js)
        except AssertionError:
            bad += 1
    return bad


# ---- fixtures: rows as load_rows reads them from a table ---------------------------------------------------------------
TIES = [0.0, 0.25, 0.5, 1.0, 2.0, 1 / 3, 2 / 3, -1.0, 3.0, 0.0078125]


def _cell_text(rng: np.random.Generator, key: str, success: bool, truth: dict[str, float]) -> str:
    if key in sx.EVERY_PROBLEM:
        return repr(truth[key])
    if not success:
        if key in sx.WORST:
            return repr(sx.WORST[key])                                   # the rows carry a failed run's worst value
        return str(rng.choice(["", "", "nan"]))
    u = rng.random()
    if u < 0.04:
        return "nan"
    if u < 0.07:
        return "inf"
    if u < 0.09:
        return "-inf"
    if u < 0.13:
        return ""
    if key in ("log10_fvu_val", "log10_fvu_fit"):                         # exact fits, recovered ties, everything else
        return repr(float(rng.choice([sx.FVU_FLOOR, -9.0, -7.5, -6.92, -6.0, -3.0, 0.5]))) if u < 0.5 else repr(float(rng.normal(-4, 3)))
    if u < 0.55:
        return repr(float(rng.choice(TIES)))
    return repr(float(np.round(rng.normal(1, 2), int(rng.integers(1, 17)))))


def fixture_rows(tmp: Path, cells: dict[tuple[str, str, int], dict[str, Any]], seed: int = 0) -> dict[str, Any]:
    """cells: (method, catalog, rung) -> {"n": problems, "drop": {(draw, row), ...} the runs that are missing, "fail":
    the share of failed runs}; written as a table and read back by load_rows."""
    rng = np.random.default_rng(seed)
    cols = ["model", "draw", "catalog", "rung", "row"] + sx.RATE_KEYS + sx.CONT_KEYS
    truth = {c: {i: {k: float(rng.integers(0, 9)) for k in sx.EVERY_PROBLEM} for i in range(spec["n"])}
             for (_m, c, _r), spec in cells.items()}
    (tmp / "rows").mkdir(exist_ok=True)
    with open(tmp / "rows" / "rows_full_fixture.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for (m, c, r), spec in sorted(cells.items()):
            for d in (1, 2):
                for i in range(spec["n"]):
                    if (d, i) in spec.get("drop", set()):
                        continue
                    ok = rng.random() >= spec.get("fail", 0.2)
                    rates = [("1" if ok else "0") if k == "success" else ("1" if ok and rng.random() < 0.4 else str(rng.choice(["0", "0", ""])))
                             for k in sx.RATE_KEYS]
                    w.writerow([m, d, c, r, i] + rates + [_cell_text(rng, k, ok, truth[c][i]) for k in sx.CONT_KEYS])
    return sx.load_rows(str(tmp / "rows"))


def fixture(tmp: Path) -> tuple[dict[str, Any], dict[str, int]]:
    """Three methods on two problem sets (12 and 300 problems: numpy's summation blocks), both draws; failed runs, empty,
    NaN and infinite values, ties; a problem missing at one budget, problems with one run, a cell still in progress."""
    cells: dict[tuple[str, str, int], dict[str, Any]] = {}
    for c, n in (("small", 12), ("big", 300)):
        for m, rungs in (("A", (1, 2, 4)), ("B", (1, 4)), ("C", (2, 4))):
            for r in rungs:
                cells[(m, c, r)] = {"n": n, "fail": {"A": 0.15, "B": 0.3, "C": 0.5}[m]}
    cells[("A", "small", 2)]["drop"] = {(1, 5), (2, 5), (2, 7)}          # problem 5 missing, problem 7 with one run
    cells[("B", "small", 4)]["drop"] = {(2, i) for i in range(4)} | {(1, 11)}
    cells[("C", "big", 4)]["drop"] = {(d, i) for d in (1, 2) for i in range(290, 300)} | {(2, i) for i in range(0, 290, 7)}
    cells[("A", "big", 1)]["drop"] = {(1, 3), (2, 3), (1, 150)}
    return fixture_rows(tmp, cells), {"small": 12, "big": 300}


def positions(data: dict[str, Any], catalog: str) -> list[tuple[str, int]]:
    """The (method, budget) cells of a problem set the release publishes (usable), as pv/ holds them."""
    return sorted((m, r) for m in data for (c, r) in data[m] if c == catalog and sx.usable(m, r))


# ---- parity at the budgets the methods ran -------------------------------------------------------------------------
def exact_jobs(data: dict[str, Any], catalogs: list[str], keys: list[str]) -> tuple[list[dict[str, Any]], list[Any]]:
    """Every ordered pair of positions (method, budget) on each problem set: the same budget, or two different ones (a
    time limit pairs methods at different budgets)."""
    jobs, expect = [], []
    for c in catalogs:
        ps = positions(data, c)
        for ma, ra in ps:
            for mb, rb in ps:
                if (ma, ra) == (mb, rb):
                    continue
                rows_a, rows_b = data[ma][(c, ra)], data[mb][(c, rb)]
                a, b = [ma, c, ra, None, 0], [mb, c, rb, None, 0]
                jobs += [{"kind": "paired", "a": a, "b": b}, {"kind": "rank", "a": a, "b": b, "keys": keys}]
                expect += [sx.paired_cell(rows_a, rows_b), sx.rank_pair_cell(rows_a, rows_b, None, keys)]
    return jobs, expect


def test_pairstats_equals_the_exporter_at_every_budget(tmp_path: Path) -> None:
    data, sizes = fixture(tmp_path)
    files = write_pv(tmp_path, data, sizes)
    jobs, expect = exact_jobs(data, ["small", "big"], sx.RANK_KEYS)
    got = run_js(tmp_path, {"rel": REL, "files": files, "jobs": jobs})["cells"]
    assert len(got) == len(expect) == 2 * sum(len(positions(data, c)) * (len(positions(data, c)) - 1) for c in sizes) == 168
    for j, py, js in zip(jobs, expect, got):
        same(py, js, f"{j['kind']} {j['a'][:3]} vs {j['b'][:3]}")
    # the fixture reaches what it is meant to: a cell in progress, problems without a successful run on one side
    cell = sx.paired_cell(data["A"][("big", 1)], data["C"][("big", 4)])
    assert cell is not None and cell["n"] == 289 and cell["m"]["f1_score@answered"][3] < cell["m"]["f1_score"][3]


def test_at_w_0_and_1_a_method_is_its_budget(tmp_path: Path) -> None:
    """A bracket read at either end is the file at that end, whatever the other end holds."""
    data, sizes = fixture(tmp_path)
    files = write_pv(tmp_path, data, sizes)
    brackets = {"A": [(1, 2), (2, 4), (1, 4)], "B": [(1, 4)], "C": [(2, 4)]}
    jobs, expect = [], []
    for c in sizes:
        for ma, bra in brackets.items():
            for mb, brb in brackets.items():
                if ma == mb:
                    continue
                for lo_a, hi_a in bra:
                    for lo_b, hi_b in brb:
                        for wa in (0, 1):
                            for wb in (0, 1):
                                ra, rb = (lo_a, hi_a)[wa], (lo_b, hi_b)[wb]
                                rows_a, rows_b = data[ma][(c, ra)], data[mb][(c, rb)]
                                a, b = [ma, c, lo_a, hi_a, wa], [mb, c, lo_b, hi_b, wb]
                                jobs += [{"kind": "paired", "a": a, "b": b}, {"kind": "rank", "a": a, "b": b, "keys": sx.RANK_KEYS}]
                                expect += [sx.paired_cell(rows_a, rows_b), sx.rank_pair_cell(rows_a, rows_b, None, sx.RANK_KEYS)]
    got = run_js(tmp_path, {"rel": REL, "files": files, "jobs": jobs})["cells"]
    assert len(got) == len(expect) == 224
    for j, py, js in zip(jobs, expect, got):
        same(py, js, f"{j['kind']} {j['a']} vs {j['b']}")


# ---- between two budgets ---------------------------------------------------------------------------------------------
def _terms(data: dict[str, Any], m: str, c: str, lo: int, hi: int | None, w: float) -> tuple[list[int], list[tuple[float, Any, Any]]]:
    """A position as the rule reads it, from the rows themselves: its problems, and its terms (weight, runs per
    problem, rows)."""
    def runs_of(rows: Any) -> dict[int, Any]:
        ids, runs = sx._problem_runs(rows)
        return {int(i): rs for i, rs in zip(ids, runs)}
    if hi is None or w in (0, 1):
        rows = data[m][(c, hi if w == 1 else lo)]
        return sorted(runs_of(rows)), [(1.0, runs_of(rows), rows)]
    a, b = runs_of(data[m][(c, lo)]), runs_of(data[m][(c, hi)])
    return sorted(set(a) & set(b)), [(1 - w, a, data[m][(c, lo)]), (w, b, data[m][(c, hi)])]


def _per_problem(terms: list[Any], ids: list[int], fn: Any) -> list[float]:
    """fn(rows) -> (ids, values) as value_matrix / rate_matrix return them, interpolated as (1 - w) x_lo + w x_hi."""
    tables = [dict(zip((int(i) for i in fn(rows)[0]), fn(rows)[1])) for _w, _runs, rows in terms]
    if len(terms) == 1:
        return [float(tables[0][i]) for i in ids]
    return [terms[0][0] * float(tables[0][i]) + terms[1][0] * float(tables[1][i]) for i in ids]


def _superiority(ta: list[Any], tb: list[Any], i: int, key: str, answered: bool) -> float | None:
    """The bilinear mixture of the exporter's reference superiority() over the terms of both positions, renormalised
    over the defined ones; None where none is."""
    acc = wsum = 0.0
    for wa, runs_a, _ in ta:
        for wb, runs_b, _ in tb:
            s = sx.superiority(runs_a[i], runs_b[i], key, answered)
            if s is not None:
                acc += wa * wb * s
                wsum += wa * wb
    return acc / wsum if wsum > 0 else None


def reference_between(data: dict[str, Any], a: list[Any], b: list[Any], kind: str, keys: list[str]) -> Any:
    """The documented interpolation, written from the exporter's per-problem reference functions (superiority,
    value_matrix, rate_matrix) and _sums, independently of pairstats.js."""
    ids_a, ta = _terms(data, *a)
    ids_b, tb = _terms(data, *b)
    common = sorted(set(ids_a) & set(ids_b))
    if not common:
        return None
    if kind == "rank":
        out: list[Any] = [len(common)]
        for k in keys:
            sup = [_superiority(ta, tb, i, k, False) for i in common]
            out += sx._sums([0.0 if s is None else s for s in sup])
        return out
    m: dict[str, Any] = {}
    for k in sx.PAIRED_KEYS:
        if k in sx.RATE_KEYS:
            ra = _per_problem(ta, common, lambda rows, k=k: sx.rate_matrix(rows, k))
            rb = _per_problem(tb, common, lambda rows, k=k: sx.rate_matrix(rows, k))
            ds = [x - y for x, y in zip(ra, rb)]
            m[k] = [len(ds)] + sx._sums(ds) + [sum(d > 0 for d in ds), sum(d < 0 for d in ds)]
            continue
        for name, answered in [(k, False)] + ([(k + sx.ANSWERED, True)] if k in sx.WORST else []):
            va = _per_problem(ta, common, lambda rows, k=k, ans=answered: sx.value_matrix(rows, k, ans))
            vb = _per_problem(tb, common, lambda rows, k=k, ans=answered: sx.value_matrix(rows, k, ans))
            ds = [x - y for x, y in zip(va, vb) if math.isfinite(x) and math.isfinite(y)]
            ss = [s for s in (_superiority(ta, tb, i, k, answered) for i in common) if s is not None]
            m[name] = [len(ds)] + sx._sums(ds) + [len(ss)] + sx._sums(ss) + [sum(s > 0 for s in ss), sum(s < 0 for s in ss)]
    return {"n": len(common), "m": m}


INTERPOLATED = [(["A", "small", 1, 2, 0.3], ["B", "small", 1, 4, 0.71]),    # both between; A@2 lacks problem 5
                (["A", "small", 2, 4, 0.5], ["C", "small", 4, None, 0]),    # A between, C at a budget it ran
                (["C", "big", 2, 4, 0.25], ["A", "big", 1, 2, 0.9]),        # C@4 lacks problems 290-299, A@1 problem 3
                (["B", "big", 1, 4, 0.125], ["A", "big", 2, None, 0])]


def test_between_two_budgets_follows_the_documented_rule(tmp_path: Path) -> None:
    data, sizes = fixture(tmp_path)
    files = write_pv(tmp_path, data, sizes)
    keys = ["log10_fvu_val", "success", "mdl_ratio", "n_constants_delta", "f1_score", "edit_distance"]
    jobs, expect = [], []
    for a, b in INTERPOLATED:
        for kind in ("paired", "rank"):
            jobs.append({"kind": kind, "a": a, "b": b, "keys": keys})
            expect.append(reference_between(data, a, b, kind, keys))
    got = run_js(tmp_path, {"rel": REL, "files": files, "jobs": jobs})["cells"]
    for j, py, js in zip(jobs, expect, got):
        same(py, js, f"{j['kind']} {j['a']} vs {j['b']}")
    assert expect[0]["n"] == 11 and expect[4]["n"] == 289      # only the problems a method has at both budgets
    # the reference reduces to the exporter at a budget a method ran
    same(sx.paired_cell(data["A"][("big", 2)], data["C"][("big", 4)]), reference_between(data, ["A", "big", 2, None, 0], ["C", "big", 2, 4, 1], "paired", []))


def _run(success: bool, **vals: float | None) -> dict[str, Any]:
    row: dict[str, Any] = {k: 0.0 for k in sx.RATE_KEYS}
    row.update({k: None for k in sx.CONT_KEYS})
    row["success"] = 1.0 if success else 0.0
    row.update(vals)
    return row


def test_a_hand_computed_mixture(tmp_path: Path) -> None:
    """Two problems, both methods between their budgets 1 and 2 (wA = 0.25, wB = 0.5): the weights of the terms
    (A1,B1), (A1,B2), (A2,B1), (A2,B2) are 0.375, 0.375, 0.125, 0.125.
    log10 FVU (lower is better), runs per budget:
      problem 0:  A1 [-1, -3]  A2 [-4, -6]  B1 [-2, -5]  B2 [-3, -3]
        sup(A1,B1) = (-1 -1 +1 -1)/4 = -0.5   sup(A1,B2) = (-1 -1 0 0)/4 = -0.5
        sup(A2,B1) = (+1 -1 +1 +1)/4 = 0.5    sup(A2,B2) = 1
        mixture: 0.375 (-0.5) + 0.375 (-0.5) + 0.125 (0.5) + 0.125 (1) = -0.1875
        values (run means): A 0.75 (-2) + 0.25 (-5) = -2.75, B 0.5 (-3.5) + 0.5 (-3) = -3.25: difference 0.5
      problem 1:  A1 [-2, -2]  A2 two failed runs (no value, last)  B1 [-2, -2]  B2 [-2, -2]
        sup = 0.375 (0) + 0.375 (0) + 0.125 (-1) + 0.125 (-1) = -0.25; A2 has no value, so no difference
      -> [1 difference, 0.5, 0.25, 2 superiorities, -0.4375, 0.0976562|5 -> 0.097656, 0 better, 2 worse]
    Token F1 over the answered predictions only (failed runs left out):
      problem 0:  every run 0.5, B2's 0.7: sup(Ax,B1) = 0, sup(Ax,B2) = -1 -> 0.375 (-1) + 0.125 (-1) = -0.5
      problem 1:  A1 [0.9, 0.9], A2 none, B1 [0.5, 0.5], B2 [0.9, 0.9]: the A2 terms are undefined, the weights
                  renormalise over A1's: (0.375 (+1) + 0.375 (0)) / 0.75 = 0.5
      -> superiorities [2, 0.0, 0.5, 1 better, 1 worse]
    Token F1 over every run (the rank cell; a failed run holds the worst value, 0): problem 0 -0.5 as above, problem 1
      0.375 (+1) + 0.375 (0) + 0.125 (-1) + 0.125 (-1) = 0.125 -> [-0.375, 0.265625]"""
    def rows(*runs_per_problem: list[dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
        return {(d + 1, i): r for i, runs in enumerate(runs_per_problem) for d, r in enumerate(runs)}

    def fv(v: float | None, f1: float, ok: bool = True) -> dict[str, Any]:
        return _run(ok, log10_fvu_val=v, f1_score=f1)
    data = {"A": {("h", 1): rows([fv(-1, 0.5), fv(-3, 0.5)], [fv(-2, 0.9), fv(-2, 0.9)]),
                  ("h", 2): rows([fv(-4, 0.5), fv(-6, 0.5)], [fv(None, 0.0, False), fv(None, 0.0, False)])},
            "B": {("h", 1): rows([fv(-2, 0.5), fv(-5, 0.5)], [fv(-2, 0.5), fv(-2, 0.5)]),
                  ("h", 2): rows([fv(-3, 0.7), fv(-3, 0.7)], [fv(-2, 0.9), fv(-2, 0.9)])}}
    files = write_pv(tmp_path, data, {"h": 2})
    a, b = ["A", "h", 1, 2, 0.25], ["B", "h", 1, 2, 0.5]
    keys = ["log10_fvu_val", "f1_score"]
    got = run_js(tmp_path, {"rel": REL, "files": files, "jobs": [{"kind": "paired", "a": a, "b": b},
                                                                 {"kind": "rank", "a": a, "b": b, "keys": keys}]})["cells"]
    assert got[0]["n"] == 2
    assert got[0]["m"]["log10_fvu_val"] == [1, 0.5, 0.25, 2, -0.4375, 0.097656, 0, 2]
    assert got[0]["m"]["f1_score@answered"][3:] == [2, 0.0, 0.5, 1, 1]
    assert got[1] == [2, -0.4375, 0.097656, -0.375, 0.265625]
    same(reference_between(data, a, b, "paired", []), got[0])          # and the reference agrees with the hand
    same(reference_between(data, a, b, "rank", keys), got[1])


# ---- the pieces: the file format, _sums, the constants, the files -------------------------------------------------------
def test_columns_round_trip_bit_for_bit(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    cols = [np.array([-0.0, 0.0, np.nan, np.inf, -np.inf, 1.5, -0.0]), np.full(5, -np.inf), np.arange(300.0) % 7,
            np.tile(np.arange(300.0) / 7, 10), rng.normal(size=1000), np.zeros(0), np.array([np.nan])]
    encoded = [sx.pv_column(c) for c in cols]
    assert list(encoded[1]) == ["u"] and list(encoded[4]) == ["f"] and "i" in encoded[2] and "i" in encoded[3]
    got = run_js(tmp_path, {"rel": REL, "files": [], "jobs": [], "columns": [{"n": int(c.size), "col": e} for c, e in zip(cols, encoded)]})["columns"]
    for c, hexed in zip(cols, got):
        assert bytes.fromhex(hexed) == np.asarray(c, "<f8").tobytes()


def test_sums_are_numpys_and_pythons_round(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    arrays = [list(map(float, rng.normal(size=n) * 10 ** rng.uniform(-4, 4, size=n))) for n in (0, 1, 7, 8, 9, 15, 16, 17, 127, 128, 129, 255, 256, 1000, 5301)]
    arrays += [list(map(float, rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0, 1 / 3], size=n))) for n in (3, 64, 300, 5301)]
    ties = [m / 128 for m in range(-4001, 4001, 2)] + [m / 128 + 2 ** 20 for m in range(1, 200, 2)] + [2.5e-6, 1.5e-7, 0.5e-6, 1e22, -1e22]
    plain = list(map(float, rng.normal(size=4000) * 10 ** rng.uniform(-9, 9, size=4000)))
    got = run_js(tmp_path, {"rel": REL, "files": [], "jobs": [], "sums": arrays, "round": ties + plain})
    for xs, js, hexed in zip(arrays, got["sums"], got["pairwise"]):
        assert bytes.fromhex(hexed) == np.float64(np.asarray(xs, float).sum()).tobytes(), f"np.sum of {len(xs)}"   # bit for bit
        same(sx._sums(xs), js, f"_sums of {len(xs)}")
    for x, js in zip(ties + plain, got["round"]):
        assert float(js) == round(x, 6), (x, round(x, 6), js)


def test_the_page_knows_the_exporters_keys(tmp_path: Path) -> None:
    got = run_js(tmp_path, {"rel": REL, "files": [], "jobs": [], "constants": True})["constants"]
    assert got == {"paired": sx.PAIRED_KEYS, "rates": sx.RATE_KEYS, "worst": list(sx.WORST), "answered": sx.ANSWERED}


def test_the_index_and_the_files(tmp_path: Path) -> None:
    data, sizes = fixture(tmp_path)
    out = tmp_path / "release"
    stale = out / "pv" / "gone" / "small" / "1.js"
    stale.parent.mkdir(parents=True)
    stale.write_text("old")
    index = sx.write_pair_values(str(out), REL, data, ["A", "B"], sizes)
    assert set(index) == {"A", "B"} and not stale.exists() and not stale.parent.parent.exists()
    assert index["A"]["small|2"] == [11, 21] and index["A"]["big|4"] == [300, 600] and index["B"]["small|4"] == [12, 19]
    assert sorted(p.relative_to(out).as_posix() for p in (out / "pv").rglob("*.js")) == sorted(
        f"pv/{m}/{cr.split('|')[0]}/{cr.split('|')[1]}.js" for m in index for cr in index[m])
    first = {p: p.stat().st_mtime_ns for p in (out / "pv").rglob("*.js")}
    sx.write_pair_values(str(out), REL, data, ["A", "B"], sizes)     # an unchanged file is not rewritten
    assert first == {p: p.stat().st_mtime_ns for p in (out / "pv").rglob("*.js")}
    text = (out / "pv" / "A" / "small" / "2.js").read_text()
    assert text.startswith("window.RESULTS_V2_PV=window.RESULTS_V2_PV||{};") and '"A|small|2"' in text


# ---- the gate can fail ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("broken", [
    ("num += (x > sb[q]) - (sb[q] > x);", "num += (sb[q] > x) - (x > sb[q]);"),                       # superiority's sign
    ("out[i] = t[0].w * a[ia[i]] + t[1].w * b[ib[i]];", "out[i] = t[1].w * a[ia[i]] + t[0].w * b[ib[i]];"),   # weights swapped
    ("if (wsum > 0) { val[i] = acc / wsum; ok[i] = 1; }", "if (wsum > 0) { val[i] = acc; ok[i] = 1; }"),      # no renormalising
    ("res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));", "res = r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7;"),   # not numpy's sum
    ("if (last % 2 === 1) {", "if (false) {")])                                                       # ties away from zero
def test_a_broken_variant_fails_the_gate(tmp_path: Path, broken: tuple[str, str]) -> None:
    src = JS.read_text()
    assert src.count(broken[0]) == 1
    bad = tmp_path / "pairstats_broken.js"
    bad.write_text(src.replace(broken[0], broken[1]))
    data, sizes = fixture(tmp_path)
    files = write_pv(tmp_path, data, sizes)
    jobs, expect = exact_jobs(data, ["big"], sx.RANK_KEYS)
    for a, b in INTERPOLATED:
        jobs.append({"kind": "paired", "a": a, "b": b})
        expect.append(reference_between(data, a, b, "paired", []))
    rng = np.random.default_rng(5)
    arrays = [list(map(float, rng.normal(size=n) * 10 ** rng.uniform(-4, 4, size=n))) for n in (16, 129, 1000)]
    ties = [m / 128 for m in range(1, 400, 2)]
    got = run_js(tmp_path, {"rel": REL, "files": files, "jobs": jobs, "sums": arrays, "round": ties}, js=bad)
    bad_cells = mismatches(expect, got["cells"])
    bad_sums = sum(bytes.fromhex(h) != np.float64(np.asarray(xs).sum()).tobytes() for xs, h in zip(arrays, got["pairwise"]))
    bad_round = sum(float(js) != round(x, 6) for x, js in zip(ties, got["round"]))
    assert bad_cells + bad_sums + bad_round > 0


# ---- real results, when given ------------------------------------------------------------------------------------------
@pytest.mark.skipif(not os.environ.get("PAIRSTATS_ROWS"), reason="PAIRSTATS_ROWS names no rows directory")
def test_pairstats_equals_the_exporter_on_real_rows(tmp_path: Path) -> None:
    data = sx.load_rows(os.environ["PAIRSTATS_ROWS"])
    catalogs = sorted({c for m in data for (c, _r) in data[m]})
    sizes = {c: max(len(rs) for m in data for (cc, _r), rows in data[m].items() if cc == c for rs in sx.by_draw(rows).values()) for c in catalogs}
    files = write_pv(tmp_path, data, sizes)
    jobs, expect = [], []
    rng = np.random.default_rng(0)
    for c in catalogs:
        ps = positions(data, c)
        for i, (ma, ra) in enumerate(ps):
            for mb, rb in ps[i + 1:]:
                if ma == mb or (ra != rb and rng.random() > 0.05):   # every pair at one budget, and a sample of the others
                    continue
                rows_a, rows_b = data[ma][(c, ra)], data[mb][(c, rb)]
                a, b = [ma, c, ra, None, 0], [mb, c, rb, None, 0]
                jobs += [{"kind": "paired", "a": a, "b": b}, {"kind": "rank", "a": a, "b": b, "keys": sx.RANK_KEYS}]
                expect += [sx.paired_cell(rows_a, rows_b), sx.rank_pair_cell(rows_a, rows_b, None, sx.RANK_KEYS)]
        # between two budgets: a few methods with two budgets or more, against the next method
        laddered = sorted({m for m, _r in ps if sum(1 for mm, _ in ps if mm == m) >= 2})
        for ma, mb in zip(laddered, laddered[1:] + laddered[:1]):
            if ma == mb:
                continue
            ra, rb = [r for m, r in ps if m == ma][:2], [r for m, r in ps if m == mb][-2:]
            a, b = [ma, c, ra[0], ra[1], 0.37], [mb, c, rb[0], rb[1], 0.62]
            for kind in ("paired", "rank"):
                jobs.append({"kind": kind, "a": a, "b": b, "keys": sx.RANK_KEYS[:6]})
                expect.append(reference_between(data, a, b, kind, sx.RANK_KEYS[:6]))
    got = run_js(tmp_path, {"rel": REL, "files": files, "jobs": jobs})["cells"]
    for j, py, js in zip(jobs, expect, got):
        same(py, js, f"{j['kind']} {j['a']} vs {j['b']}")
    print(f"real rows: {len(jobs)} cells equal ({sum(1 for j in jobs if j['a'][3] is not None)} between budgets), {len(files)} pv files")
