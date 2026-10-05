"""results-site/pairstats.js computes the explorer's paired contrasts and pairwise rank outcomes in the browser, from the
per-problem files the exporter writes (pv/, scripts/site_export_v2.py write_pair_values). The exporter computes the same
cells at its slots (every budget of the release and every time limit, each method bracketed over its complete budgets:
brackets, slot_cells) for ranks/<key>.js and paired/<key>.js. Both must agree bit for bit: at a budget a method ran
with what paired_cell and rank_pair_cell have always returned, and between two budgets with the rule pairstats.js's
header documents. The gate runs the page's own file under node.

PAIRSTATS_ROWS, when set, names a directory of rows_full_*.csv (load_rows's input, e.g. a few catalogs cut from the
board's table, with its timing.json) to run the same parity on real results as well."""
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

# The node side: load pairstats.js and the files, read every job's two positions, return the results as JSON.
DRIVER = r"""
const fs = require("fs"), vm = require("vm");
globalThis.window = globalThis;
const PS = require(process.argv[1]);
const job = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
for (const f of job.files) { vm.runInThisContext(fs.readFileSync(f, "utf8"), { filename: f }); }
const hex = (x) => { const b = Buffer.alloc(8); b.writeDoubleLE(x); return b.toString("hex"); };
const file = (m, r, c) => { const f = PS.file(job.rel, m, r, c); if (!f) { throw new Error("no pv frame " + [m, r, c]); } return f; };
const pos = (p) => p.raw ? PS.at(PS.decode(p.raw[0]), p.raw[1] ? PS.decode(p.raw[1]) : null, p.w)
  : PS.at(file(p[0], p[2], p[1]), p[3] === null ? null : file(p[0], p[3], p[1]), p[4]);
const run = (j) => { try { return j.kind === "paired" ? PS.pairedCell(pos(j.a), pos(j.b)) : PS.rankPairCell(pos(j.a), pos(j.b), j.keys); }
  catch (e) { if (job.catch) { return { error: String(e.message) }; } throw e; } };
const out = { cells: (job.jobs || []).map(run) };
if (job.brackets) { out.brackets = job.brackets.map((b) => { const r = PS.bracketIn(b.points, b.x); return r ? [r.r1, r.r2, hex(r.w)] : null; }); }
if (job.sums) { out.sums = job.sums.map(PS.sums); out.pairwise = job.sums.map((xs) => hex(PS.pairwise(xs, 0, xs.length))); }
if (job.round) { out.round = job.round.map(PS.round6); }
if (job.logs) { out.logs = job.logs.map((x) => hex(PS.logd(x))); }
if (job.columns) { out.columns = job.columns.map((c) => Buffer.from(PS.column(c.col, c.n).buffer).toString("hex")); }
if (job.constants) { out.constants = { paired: PS.PAIRED_KEYS, rates: PS.RATE_KEYS, worst: PS.WORST, answered: PS.ANSWERED,
  frame: PS.framePath("T8-20M", 1024), key: PS.keyPath("T8-20M", 1024, "f1_score@answered") }; }
if (job.globals) { out.globals = { ranks: (globalThis.RESULTS_V2_RANKCELLS || {})[job.rel], paired: (globalThis.RESULTS_V2_PAIRCELLS || {})[job.rel] }; }
process.stdout.write(JSON.stringify(out));
"""


def run_js(tmp: Path, job: dict[str, Any], js: Path = JS) -> dict[str, Any]:
    spec = tmp / "job.json"
    spec.write_text(json.dumps(job))
    done = subprocess.run([str(NODE), "-e", DRIVER, str(js), str(spec)], capture_output=True, text=True, timeout=3600)
    assert done.returncode == 0, done.stderr[-4000:]
    return json.loads(done.stdout)


def write_pv(tmp: Path, data: dict[str, Any], sizes: dict[str, int], rank_keys: list[str] | None = None) -> list[str]:
    out = tmp / "release"
    sx.write_pair_values(str(out), REL, data, sorted(data), sizes, rank_keys)
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
    elif isinstance(py, str):
        assert py == js, f"{where}: {py!r} vs {js!r}"
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


# ---- today's cells, frozen (scripts/site_export_v2.py at be9134e1): a budget both methods ran must still give them ------
def paired_cell_today(rows_a: Any, rows_b: Any) -> dict[str, Any] | None:
    common, ja, jb = sx._align(sx._problem_runs(rows_a)[0], sx._problem_runs(rows_b)[0])
    if not common.size:
        return None
    out: dict[str, Any] = {}
    for k in sx.PAIRED_KEYS:
        if k in sx.RATE_KEYS:
            ds = sx.rate_matrix(rows_a, k)[1][ja] - sx.rate_matrix(rows_b, k)[1][jb]
            out[k] = [int(ds.size)] + sx._sums(list(ds)) + [int((ds > 0).sum()), int((ds < 0).sum())]
            continue
        for name, answered in [(k, False)] + ([(k + sx.ANSWERED, True)] if k in sx.WORST else []):
            va, vb = sx.value_matrix(rows_a, k, answered)[1][ja], sx.value_matrix(rows_b, k, answered)[1][jb]
            both = np.isfinite(va) & np.isfinite(vb)
            ds = (va - vb)[both]
            _, sa, ma = sx.score_matrix(rows_a, k, answered)
            _, sb, mb = sx.score_matrix(rows_b, k, answered)
            sup, ok = sx.superiority_of(sa[ja], ma[ja], sb[jb], mb[jb])
            ss = sup[ok]
            out[name] = [int(ds.size)] + sx._sums(list(ds)) + [int(ss.size)] + sx._sums(list(ss)) + [int((ss > 0).sum()), int((ss < 0).sum())]
    return {"n": int(common.size), "m": out}


def rank_pair_cell_today(rows_a: Any, rows_b: Any, keys: list[str]) -> list[float] | None:
    common, ja, jb = sx._align(sx._problem_runs(rows_a)[0], sx._problem_runs(rows_b)[0])
    if not common.size:
        return None
    out: list[float] = [int(common.size)]
    for k in keys:
        _, sa, ma = sx.score_matrix(rows_a, k)
        _, sb, mb = sx.score_matrix(rows_b, k)
        out += sx._sums(list(sx.superiority_of(sa[ja], ma[ja], sb[jb], mb[jb])[0]))
    return out


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


SIZES = {"small": 12, "big": 300}
# reference seconds per problem: B's 4 sits exactly on the 3 s limit, C's ladder starts above 0.3 s, A's 8 on small is
# not finished (so A has no 8 there); a time limit between two budgets interpolates
TIMING = {"A": {"1": 0.2, "2": 0.5, "4": 1.7, "8": 4.0}, "B": {"1": 0.05, "4": 3.0}, "C": {"2": 0.9, "4": 12.0, "8": 30.0}}


def fixture(tmp: Path) -> tuple[dict[str, Any], dict[str, int]]:
    """Three methods on two problem sets (12 and 300 problems: numpy's summation blocks), both draws; failed runs, empty,
    NaN and infinite values, ties; problems with one run; two cells still in progress (A at 8 on small, C at 8 on big)."""
    cells: dict[tuple[str, str, int], dict[str, Any]] = {}
    for c, n in SIZES.items():
        for m, rungs in (("A", (1, 2, 4, 8)), ("B", (1, 4)), ("C", (2, 4, 8))):
            for r in rungs:
                cells[(m, c, r)] = {"n": n, "fail": {"A": 0.15, "B": 0.3, "C": 0.5}[m]}
    cells[("A", "small", 2)]["drop"] = {(2, 5), (2, 7)}                    # problems with one run
    cells[("B", "small", 4)]["drop"] = {(2, i) for i in range(4)} | {(1, 11)}
    cells[("C", "big", 4)]["drop"] = {(2, i) for i in range(0, 300, 7)}
    cells[("A", "big", 1)]["drop"] = {(1, 150)}
    cells[("A", "small", 8)]["drop"] = {(1, 5), (2, 5)}                    # in progress: problem 5 has no run yet
    cells[("C", "big", 8)]["drop"] = {(d, i) for d in (1, 2) for i in range(290, 300)}
    return fixture_rows(tmp, cells), dict(SIZES)


def complete(data: dict[str, Any], sizes: dict[str, int], m: str, c: str, r: int) -> bool:
    rows = data.get(m, {}).get((c, r))
    return bool(rows) and sx.usable(m, r) and sx.problems_of(rows, sizes[c])[1]


def positions(data: dict[str, Any], sizes: dict[str, int], catalog: str) -> list[tuple[str, int]]:
    """The (method, budget) cells of a problem set that pv/ holds: published and complete."""
    return sorted((m, r) for m in data for (c, r) in data[m] if c == catalog and complete(data, sizes, m, c, r))


def page_points(data: dict[str, Any], sizes: dict[str, int], timing: dict[str, Any], m: str, c: str, by_time: bool) -> list[list[float]]:
    """explorer_v2.js points(): the method's complete budgets on c as [position, budget], by budget or by reference time,
    sorted by position (stably, from ascending budgets)."""
    pts = []
    for (cc, r) in sorted(data.get(m, {})):
        if cc == c and complete(data, sizes, m, c, r):
            v = sx.ref_time(timing, m, r) if by_time else float(r)
            if v is not None and v > 0:
                pts.append([v, r])
    return sorted(pts, key=lambda p: p[0])


# ---- parity at the budgets the methods ran -------------------------------------------------------------------------
def exact_jobs(data: dict[str, Any], sizes: dict[str, int], catalogs: list[str], keys: list[str]) -> tuple[list[dict[str, Any]], list[Any]]:
    """Every ordered pair of positions (method, budget) on each problem set: the same budget, or two different ones."""
    jobs, expect = [], []
    for c in catalogs:
        ps = positions(data, sizes, c)
        for ma, ra in ps:
            for mb, rb in ps:
                if (ma, ra) == (mb, rb):
                    continue
                rows_a, rows_b = data[ma][(c, ra)], data[mb][(c, rb)]
                a, b = [ma, c, ra, None, 0], [mb, c, rb, None, 0]
                jobs += [{"kind": "paired", "a": a, "b": b}, {"kind": "rank", "a": a, "b": b, "keys": keys}]
                expect += [paired_cell_today(rows_a, rows_b), rank_pair_cell_today(rows_a, rows_b, keys)]
    return jobs, expect


def test_pairstats_equals_the_exporter_at_every_budget(tmp_path: Path) -> None:
    data, sizes = fixture(tmp_path)
    files = write_pv(tmp_path, data, sizes)
    jobs, expect = exact_jobs(data, sizes, ["small", "big"], sx.RANK_KEYS)
    got = run_js(tmp_path, {"rel": REL, "files": files, "jobs": jobs})["cells"]
    n_pos = {c: len(positions(data, sizes, c)) for c in sizes}
    assert n_pos == {"small": 8, "big": 8} and len(got) == len(expect) == 2 * sum(n * (n - 1) for n in n_pos.values())
    for j, py, js in zip(jobs, expect, got):
        same(py, js, f"{j['kind']} {j['a'][:3]} vs {j['b'][:3]}")
    # the fixture reaches what it is meant to: problems without a successful run on one side
    cell = paired_cell_today(data["A"][("big", 1)], data["C"][("big", 4)])
    assert cell is not None and cell["n"] == 300 and cell["m"]["f1_score@answered"][3] < cell["m"]["f1_score"][3]


def test_todays_cells_are_unchanged(tmp_path: Path) -> None:
    """paired_cell and rank_pair_cell now run through the position code (one term each): every pair of cells, finished
    or not, gives what the code before it gave, and so does paired_cell_at at the exact positions."""
    data, _sizes = fixture(tmp_path)
    cells = [(m, cr) for m in sorted(data) for cr in sorted(data[m])]
    n = 0
    for ma, (c, ra) in cells:
        for mb, (cc, rb) in cells:
            if c != cc or (ma, ra) == (mb, rb):
                continue
            rows_a, rows_b = data[ma][(c, ra)], data[mb][(c, rb)]
            same(paired_cell_today(rows_a, rows_b), sx.paired_cell(rows_a, rows_b), f"paired {ma}{ra} {mb}{rb} {c}")
            same(rank_pair_cell_today(rows_a, rows_b, sx.RANK_KEYS), sx.rank_pair_cell(rows_a, rows_b, None, sx.RANK_KEYS), f"rank {ma}{ra} {mb}{rb}")
            same(paired_cell_today(rows_a, rows_b), sx.paired_cell_at(sx.position(rows_a, rows_b, 0.0), sx.position(rows_a, rows_b, 1.0)))
            n += 1
    assert n == 2 * 9 * 8


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
                                expect += [paired_cell_today(rows_a, rows_b), rank_pair_cell_today(rows_a, rows_b, sx.RANK_KEYS)]
                                same(expect[-2], sx.paired_cell_at(sx.position(data[ma][(c, lo_a)], data[ma][(c, hi_a)], wa),
                                                                   sx.position(data[mb][(c, lo_b)], data[mb][(c, hi_b)], wb)))
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
    value_matrix, rate_matrix) and _sums, independently of pairstats.js and of the exporter's vectorised code."""
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


def exporter_between(data: dict[str, Any], a: list[Any], b: list[Any], kind: str, keys: list[str]) -> Any:
    """The same through the exporter's own code (what ranks/ and paired/ hold at a slot)."""
    def pos(m: str, c: str, lo: int, hi: int | None, w: float) -> Any:
        return sx.position(data[m][(c, lo)], None if hi is None else data[m][(c, hi)], w)
    pa, pb = pos(*a), pos(*b)
    return sx.rank_pair_cell_at(pa, pb, keys) if kind == "rank" else sx.paired_cell_at(pa, pb)


INTERPOLATED = [(["A", "small", 1, 2, 0.3], ["B", "small", 1, 4, 0.71]),    # both between
                (["A", "small", 2, 4, 0.5], ["C", "small", 4, None, 0]),    # A between, C at a budget it ran
                (["C", "big", 2, 4, 0.25], ["A", "big", 1, 2, 0.9]),
                (["B", "big", 1, 4, 0.125], ["A", "big", 2, None, 0])]
KEYS6 = ["log10_fvu_val", "success", "mdl_ratio", "n_constants_delta", "f1_score", "edit_distance"]


def test_between_two_budgets_follows_the_documented_rule(tmp_path: Path) -> None:
    data, sizes = fixture(tmp_path)
    files = write_pv(tmp_path, data, sizes)
    jobs, expect = [], []
    for a, b in INTERPOLATED:
        for kind in ("paired", "rank"):
            jobs.append({"kind": kind, "a": a, "b": b, "keys": KEYS6})
            ref = reference_between(data, a, b, kind, KEYS6)
            same(ref, exporter_between(data, a, b, kind, KEYS6), f"exporter {kind} {a} {b}")   # one rule, two codes
            expect.append(ref)
    got = run_js(tmp_path, {"rel": REL, "files": files, "jobs": jobs})["cells"]
    for j, py, js in zip(jobs, expect, got):
        same(py, js, f"{j['kind']} {j['a']} vs {j['b']}")
    # the reference reduces to the cell of today at a budget a method ran
    same(paired_cell_today(data["A"][("big", 2)], data["C"][("big", 4)]), reference_between(data, ["A", "big", 2, None, 0], ["C", "big", 2, 4, 1], "paired", []))


def test_between_budgets_only_the_problems_of_both_take_part(tmp_path: Path) -> None:
    """A cell in progress never reaches pv/, but at() keeps the rule for any two files: only the problems the method has
    at both budgets take part (decoded from pv_cell's objects, both ends of A's bracket on small, 4 and the unfinished 8)."""
    data, _sizes = fixture(tmp_path)
    keys = list(dict.fromkeys(KEYS6 + sx.PAIRED_KEYS))
    raw = {r: sx.pv_cell(data["A"][("small", r)], keys) for r in (4, 8)}
    jobs, expect = [], []
    for kind in ("paired", "rank"):
        jobs.append({"kind": kind, "a": {"raw": [raw[4], raw[8]], "w": 0.4}, "b": {"raw": [sx.pv_cell(data["B"][("small", 4)], keys)], "w": 0}, "keys": KEYS6})
        expect.append(reference_between(data, ["A", "small", 4, 8, 0.4], ["B", "small", 4, None, 0], kind, KEYS6))
        same(expect[-1], exporter_between(data, ["A", "small", 4, 8, 0.4], ["B", "small", 4, None, 0], kind, KEYS6))
    got = run_js(tmp_path, {"rel": REL, "files": [], "jobs": jobs})["cells"]
    assert expect[0]["n"] == 11
    for py, js in zip(expect, got):
        same(py, js)


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
    for kind, js in (("paired", got[0]), ("rank", got[1])):              # and both Python codes agree with the hand
        same(reference_between(data, a, b, kind, keys), js)
        same(exporter_between(data, a, b, kind, keys), js)


# ---- the slots: ranks/<key>.js and paired/<key>.js -------------------------------------------------------------------
def slots_of(data: dict[str, Any]) -> list[str]:
    return [str(r) for r in sorted({r for m in data for (_c, r) in data[m]})] + [sx.budget_key(t) for t in sx.TIME_BUDGETS]


def slot_jobs(at: dict[str, Any], pairs: list[tuple[str, str]], slots: list[str], rank_keys: list[str],
              ranks: dict[str, Any], paired: dict[str, Any]) -> tuple[list[dict[str, Any]], list[Any]]:
    """For every cell the exporter wrote: the job that reads both methods at their brackets through pv/, and the cell
    as ranks/ and paired/ hold it (rank: [n, sum, sum of squares] of every key in turn; paired: {name: entry})."""
    jobs, expect = [], []
    for ka, kb in pairs:
        pk = ka + "|" + kb
        for c in sorted({c for k in rank_keys for c in ranks[k].get(pk, {})}):
            for slot in slots:
                rc = [ranks[k].get(pk, {}).get(c, {}).get(slot) for k in rank_keys]
                if rc[0] is None:
                    continue
                ba, bb = at[ka][slot][c], at[kb][slot][c]
                a, b = [ka, c, ba[0], ba[1], ba[2]], [kb, c, bb[0], bb[1], bb[2]]
                jobs.append({"kind": "rank", "a": a, "b": b, "keys": rank_keys})
                expect.append([rc[0][0]] + [x for t in rc for x in t[1:]])
                jobs.append({"kind": "paired", "a": a, "b": b})
                expect.append({"n": rc[0][0], "m": {name: per[pk][c][slot] for name, per in paired.items()}})
    return jobs, expect


def bracket_jobs(data: dict[str, Any], sizes: dict[str, int], timing: dict[str, Any], methods: list[str], slots: list[str],
                 at: dict[str, Any]) -> tuple[list[dict[str, Any]], list[Any]]:
    """Every method x problem set x slot: the page's points and position, and the exporter's bracket (or None)."""
    jobs, expect = [], []
    for m in methods:
        for c in sizes:
            for slot in slots:
                by_time, x = sx.slot_value(slot)
                jobs.append({"points": page_points(data, sizes, timing, m, c, by_time), "x": x})
                b = at.get(m, {}).get(slot, {}).get(c)
                expect.append(None if b is None else [b[0], b[1], np.float64(b[2]).tobytes().hex()])
    return jobs, expect


def slot_gate(tmp: Path, data: dict[str, Any], sizes: dict[str, int], timing: dict[str, Any], rank_keys: list[str],
              js: Path = JS) -> dict[str, Any]:
    """The exporter's slots against pairstats.js: brackets (the page's rule) and every cell. Returns the counts."""
    methods = sorted(data)
    slots = slots_of(data)
    at = sx.brackets(data, methods, sizes, timing, slots)
    pairs = [(ka, kb) for i, ka in enumerate(methods) for kb in methods[i + 1:]]
    ranks, paired = sx.slot_cells(data, pairs, at, slots, rank_keys)
    files = write_pv(tmp, data, sizes, rank_keys)
    jobs, expect = slot_jobs(at, pairs, slots, rank_keys, ranks, paired)
    bjobs, bexpect = bracket_jobs(data, sizes, timing, methods, slots, at)
    got = run_js(tmp, {"rel": REL, "files": files, "jobs": jobs, "brackets": bjobs}, js=js)
    return {"cells": mismatches(expect, got["cells"]), "n_cells": len(jobs), "brackets": mismatches(bexpect, got["brackets"]),
            "n_brackets": len(bjobs), "between": sum(1 for j in jobs if j["a"][4] or j["b"][4]),
            "w_differ": [(j["x"], e, g) for j, e, g in zip(bjobs, bexpect, got["brackets"]) if e != g],
            "at": at, "ranks": ranks, "paired": paired}


def test_slot_cells_are_what_pairstats_computes_at_the_slot(tmp_path: Path) -> None:
    data, sizes = fixture(tmp_path)
    g = slot_gate(tmp_path, data, sizes, TIMING, sx.RANK_KEYS)
    assert g["w_differ"] == [] and g["brackets"] == 0
    assert g["cells"] == 0 and g["n_cells"] == 2 * 26 and g["between"] == 2 * 16
    at = g["at"]
    # the brackets: exact hits, interpolation by budget and by time, nothing outside a ladder or at an unfinished budget
    assert at["B"]["2"]["small"] == [1, 4, 0.5] and at["C"].get("1") is None and at["B"]["t3"]["big"] == [4, 4, 0.0]
    assert at["A"]["8"]["big"] == [8, 8, 0.0] and "small" not in at["A"].get("8", {}) and at["C"]["8"].get("big") is None
    assert at["A"]["t1"]["small"] == [2, 4, math.log(1 / 0.5) / math.log(1.7 / 0.5)]
    # a budget both ran: exactly today's cell
    for (ka, kb, c, r) in (("A", "B", "small", 4), ("A", "C", "big", 2), ("B", "C", "big", 4)):
        today = rank_pair_cell_today(data[ka][(c, r)], data[kb][(c, r)], sx.RANK_KEYS)
        assert today is not None
        pk = ka + "|" + kb
        assert [g["ranks"][k][pk][c][str(r)] for k in sx.RANK_KEYS] == [[today[0], today[1 + 2 * i], today[2 + 2 * i]] for i in range(len(sx.RANK_KEYS))]
        same(paired_cell_today(data[ka][(c, r)], data[kb][(c, r)])["m"], {name: per[pk][c][str(r)] for name, per in g["paired"].items()})


def test_slot_files_merge_with_an_overlays(tmp_path: Path) -> None:
    """ranks/<key>.js and paired/<key>.js of the release and of an overlay add up pair by pair, in either order."""
    data, sizes = fixture(tmp_path)
    slots = slots_of(data)
    at = sx.brackets(data, sorted(data), sizes, TIMING, slots)
    pub = sx.slot_cells(data, [("A", "B")], at, slots, ["log10_fvu_val", "success"])
    own = sx.slot_cells(data, [("C", "A"), ("C", "B")], at, slots, ["log10_fvu_val", "success"])
    sx.write_slot_cells(str(tmp_path / "pub"), REL, *pub)
    sx.write_slot_cells(str(tmp_path / "own"), REL, *own)
    assert sorted(p.name for p in (tmp_path / "pub" / "paired").glob("*.js")) == sorted(k + ".js" for k in sx.PAIRED_KEYS)
    assert sorted(p.name for p in (tmp_path / "pub" / "ranks").glob("*.js")) == ["log10_fvu_val.js", "success.js"]
    for first, second in (("pub", "own"), ("own", "pub")):
        files = sorted(str(p) for p in (tmp_path / first).rglob("*.js")) + sorted(str(p) for p in (tmp_path / second).rglob("*.js"))
        got = run_js(tmp_path, {"rel": REL, "files": files, "globals": True})["globals"]
        assert sorted(got["ranks"]["log10_fvu_val"]) == ["A|B", "C|A", "C|B"]
        assert sorted(got["paired"]) == sorted(set(sx.PAIRED_KEYS) | {"f1_score@answered"})
        same(pub[1]["f1_score@answered"]["A|B"], got["paired"]["f1_score@answered"]["A|B"])
        same(own[0]["success"]["C|B"], got["ranks"]["success"]["C|B"])
    # the single files they replace are removed, and a key no longer written
    (tmp_path / "pub" / "ranks.js").write_text("old")
    (tmp_path / "pub" / "ranks" / "gone.js").write_text("old")
    sx.write_slot_cells(str(tmp_path / "pub"), REL, *pub)
    assert not (tmp_path / "pub" / "ranks.js").exists() and not (tmp_path / "pub" / "ranks" / "gone.js").exists()


# ---- the pieces: the file format, the loader, _sums, the constants ------------------------------------------------------
def test_columns_round_trip_bit_for_bit(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    cols = [np.array([-0.0, 0.0, np.nan, np.inf, -np.inf, 1.5, -0.0]), np.full(5, -np.inf), np.arange(300.0) % 7,
            np.tile(np.arange(300.0) / 7, 10), rng.normal(size=1000), np.zeros(0), np.array([np.nan])]
    encoded = [sx.pv_column(c) for c in cols]
    assert list(encoded[1]) == ["u"] and list(encoded[4]) == ["f"] and "i" in encoded[2] and "i" in encoded[3]
    got = run_js(tmp_path, {"rel": REL, "files": [], "columns": [{"n": int(c.size), "col": e} for c, e in zip(cols, encoded)]})["columns"]
    for c, hexed in zip(cols, got):
        assert bytes.fromhex(hexed) == np.asarray(c, "<f8").tobytes()


def test_a_key_file_not_loaded_is_named_never_read_as_zeros(tmp_path: Path) -> None:
    data, sizes = fixture(tmp_path)
    files = [f for f in write_pv(tmp_path, data, sizes) if f.endswith(("frame.js", "log10_fvu_val.js"))]
    a, b = ["A", "small", 1, 2, 0.5], ["B", "small", 4, None, 0]
    got = run_js(tmp_path, {"rel": REL, "files": files, "catch": True, "jobs": [
        {"kind": "rank", "a": a, "b": b, "keys": ["log10_fvu_val"]}, {"kind": "rank", "a": a, "b": b, "keys": ["log10_fvu_val", "success"]},
        {"kind": "paired", "a": a, "b": b}]})["cells"]
    same(exporter_between(data, a, b, "rank", ["log10_fvu_val"]), got[0])
    assert "pv/A/1/success.js is not loaded" in got[1]["error"]
    assert "pv/A/1/numeric_recovery_val.js is not loaded" in got[2]["error"]


def test_sums_are_numpys_and_pythons_round(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    arrays = [list(map(float, rng.normal(size=n) * 10 ** rng.uniform(-4, 4, size=n))) for n in (0, 1, 7, 8, 9, 15, 16, 17, 127, 128, 129, 255, 256, 1000, 5301)]
    arrays += [list(map(float, rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0, 1 / 3], size=n))) for n in (3, 64, 300, 5301)]
    ties = [m / 128 for m in range(-4001, 4001, 2)] + [m / 128 + 2 ** 20 for m in range(1, 200, 2)] + [2.5e-6, 1.5e-7, 0.5e-6, 1e22, -1e22]
    plain = list(map(float, rng.normal(size=4000) * 10 ** rng.uniform(-9, 9, size=4000)))
    got = run_js(tmp_path, {"rel": REL, "files": [], "sums": arrays, "round": ties + plain})
    for xs, js, hexed in zip(arrays, got["sums"], got["pairwise"]):
        assert bytes.fromhex(hexed) == np.float64(np.asarray(xs, float).sum()).tobytes(), f"np.sum of {len(xs)}"   # bit for bit
        same(sx._sums(xs), js, f"_sums of {len(xs)}")
    for x, js in zip(ties + plain, got["round"]):
        assert float(js) == round(x, 6), (x, round(x, 6), js)


def test_brackets_agree_with_the_page_bit_for_bit(tmp_path: Path) -> None:
    """Math.log and math.log over every reference time of the board's timing.json (when present) and a grid of
    positions: the exporter's w and pairstats.js's bracketIn are the same doubles."""
    path = Path.home() / "srbf_eval_local" / "baselines_d1" / "timing.json"
    timing = {k: v for k, v in json.load(open(path)).items() if isinstance(v, dict)} if path.exists() else TIMING
    xs = sorted(set(sx.TIME_BUDGETS) | {0.137 * 1.07 ** i for i in range(160)} | {2.0 ** (i / 4) for i in range(80)} | {1000.0 * 2 ** (i / 3) for i in range(30)})
    jobs, expect = [], []
    for m, per in sorted(timing.items()):
        for by_time in (True, False):
            pts = sorted(([sx.ref_time(timing, m, int(r)) if by_time else float(r), int(r)] for r in per if r.isdigit()), key=lambda p: p[0])
            pts = [p for p in pts if p[0] is not None and p[0] > 0]
            for x in xs:
                jobs.append({"points": pts, "x": x})
                b = sx.bracket_in([(p, r) for p, r in pts], x)
                expect.append(None if b is None else [b[0], b[1], np.float64(b[2]).tobytes().hex()])
    got = run_js(tmp_path, {"rel": REL, "files": [], "brackets": jobs})["brackets"]
    differ = [(j["x"], e, g) for j, e, g in zip(jobs, expect, got) if e != g]
    assert len(jobs) > 1000 and sum(e is not None and e[2] != "0" * 16 for e in expect) > 500
    assert differ == [], f"{len(differ)} brackets differ, e.g. {differ[:3]}"


def test_logd_is_one_double_in_both_and_a_log(tmp_path: Path) -> None:
    """The brackets' logarithm: the same bits in Python and in the browser, and within a few ulps of libm's log."""
    rng = np.random.default_rng(11)
    edges = [1.0, 2.0, 0.5, 1e-310, 5e-324, 1.7976931348623157e308, 1.0000000000000002, 0.9999999999999999, 1.4142135623730951]
    xs = list(map(float, np.exp(rng.uniform(-40, 40, size=20000)))) + edges
    xs += [float(b / a) for a in (0.1261, 2.0345, 1000.0, 5.0199) for b in np.linspace(a * 1.0001, a * 3, 400)]
    got = run_js(tmp_path, {"rel": REL, "files": [], "logs": xs})["logs"]
    assert [h for h, x in zip(got, xs) if h != np.float64(sx.logd(x)).tobytes().hex()] == []
    ulps = [abs(sx.logd(x) - math.log(x)) / math.ulp(math.log(x)) for x in xs if x != 1.0]
    assert max(ulps) <= 4 and sx.logd(1.0) == 0.0


def test_the_page_knows_the_exporters_keys(tmp_path: Path) -> None:
    got = run_js(tmp_path, {"rel": REL, "files": [], "constants": True})["constants"]
    assert got == {"paired": sx.PAIRED_KEYS, "rates": sx.RATE_KEYS, "worst": list(sx.WORST), "answered": sx.ANSWERED,
                   "frame": "pv/T8-20M/1024/frame.js", "key": "pv/T8-20M/1024/f1_score.js"}


def test_the_index_and_the_files(tmp_path: Path) -> None:
    data, sizes = fixture(tmp_path)
    out = tmp_path / "release"
    stale = out / "pv" / "gone" / "1" / "frame.js"
    stale.parent.mkdir(parents=True)
    stale.write_text("old")
    index = sx.write_pair_values(str(out), REL, data, ["A", "C"], sizes, ["log10_fvu_val"])
    assert not stale.exists() and not stale.parent.parent.exists()
    # only finished cells: A has no 8 on small, C no 8 on big
    assert index == {"A": {"1": ["big", "small"], "2": ["big", "small"], "4": ["big", "small"], "8": ["big"]},
                     "C": {"2": ["big", "small"], "4": ["big", "small"], "8": ["small"]}}
    keys = ["frame"] + list(dict.fromkeys(["log10_fvu_val"] + sx.PAIRED_KEYS))
    assert sorted(p.relative_to(out).as_posix() for p in (out / "pv").rglob("*.js")) == sorted(
        f"pv/{m}/{r}/{k}.js" for m in index for r in index[m] for k in keys)
    first = {p: p.stat().st_mtime_ns for p in (out / "pv").rglob("*.js")}
    sx.write_pair_values(str(out), REL, data, ["A", "C"], sizes, ["log10_fvu_val"])   # an unchanged file is not rewritten
    assert first == {p: p.stat().st_mtime_ns for p in (out / "pv").rglob("*.js")}
    text = (out / "pv" / "A" / "2" / "f1_score.js").read_text()
    assert text.startswith("window.RESULTS_V2_PV=window.RESULTS_V2_PV||{};") and '"A|2|f1_score"' in text
    obj = json.loads(text[text.index("]=", text.index('"A|2|f1_score"')) + 2:text.rindex(";})();")])
    assert sorted(obj) == ["big", "small"] and sorted(obj["small"]["v"]) == ["f1_score", "f1_score@answered"] and "r" not in obj["small"]


# ---- the gates can fail ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("broken", [
    ("num += (x > sb[q]) - (sb[q] > x);", "num += (sb[q] > x) - (x > sb[q]);"),                       # superiority's sign
    ("out[i] = t[0].w * a[ia[i]] + t[1].w * b[ib[i]];", "out[i] = t[1].w * a[ia[i]] + t[0].w * b[ib[i]];"),   # weights swapped
    ("if (wsum > 0) { val[i] = acc / wsum; ok[i] = 1; }", "if (wsum > 0) { val[i] = acc; ok[i] = 1; }"),      # no renormalising
    ("res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7));", "res = r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7;"),   # not numpy's sum
    ("if (last % 2 === 1) {", "if (false) {"),                                                        # ties away from zero
    ("w: logd(x / ps[i][0]) / logd(ps[i + 1][0] / ps[i][0])", "w: Math.log(x / ps[i][0]) / Math.log(ps[i + 1][0] / ps[i][0])"),   # libm's log
    ("if (m < SQRT_HALF) { m = m * 2; e = e - 1; }", "if (false) { m = m * 2; e = e - 1; }")])        # logd's range
def test_a_broken_page_variant_fails_the_gate(tmp_path: Path, broken: tuple[str, str]) -> None:
    src = JS.read_text()
    assert src.count(broken[0]) == 1
    bad = tmp_path / "pairstats_broken.js"
    bad.write_text(src.replace(broken[0], broken[1]))
    data, sizes = fixture(tmp_path)
    g = slot_gate(tmp_path, data, sizes, TIMING, KEYS6, js=bad)
    rng = np.random.default_rng(5)
    arrays = [list(map(float, rng.normal(size=n) * 10 ** rng.uniform(-4, 4, size=n))) for n in (16, 129, 1000)]
    ties = [m / 128 for m in range(1, 400, 2)]
    logs = list(map(float, np.exp(rng.uniform(-5, 5, size=200))))
    got = run_js(tmp_path, {"rel": REL, "files": [], "sums": arrays, "round": ties, "logs": logs}, js=bad)
    bad_sums = sum(bytes.fromhex(h) != np.float64(np.asarray(xs).sum()).tobytes() for xs, h in zip(arrays, got["pairwise"]))
    bad_round = sum(float(js) != round(x, 6) for x, js in zip(ties, got["round"]))
    bad_logs = sum(h != np.float64(sx.logd(x)).tobytes().hex() for x, h in zip(logs, got["logs"]))
    assert g["cells"] + g["brackets"] + bad_sums + bad_round + bad_logs > 0


ORIGINAL = {name: getattr(sx, name) for name in ("position", "superiority_at", "bracket_in", "brackets")}


def _swap_weights(rows_lo: Any, rows_hi: Any = None, w: float = 0.0) -> Any:
    p = ORIGINAL["position"](rows_lo, rows_hi, w)
    if len(p.terms) == 1:
        return p
    (w0, r0, i0), (w1, r1, i1) = p.terms
    return sx.Position(p.ids, [(w1, r0, i0), (w0, r1, i1)])


def _no_renormalising(a: Any, b: Any, common: Any, key: str, answered: bool = False, cache: Any = None) -> Any:
    sup, ok = ORIGINAL["superiority_at"](a, b, common, key, answered, cache)
    wsum = np.zeros(common.size)
    for wa, ra, _ in a.terms:
        for wb, rb, _ in b.terms:
            _, _, ids, _s, o = sx._superiority_table(ra, rb, key, answered, cache)
            wsum = wsum + np.where(o[np.searchsorted(ids, common)], wa * wb, 0.0)
    return sup * wsum, ok


def _w_flipped(points: Any, x: float) -> Any:
    b = ORIGINAL["bracket_in"](points, x)
    return b if b is None or b[0] == b[1] else (b[0], b[1], 1 - b[2])


def _unfinished_too(data: Any, keys: Any, sizes: Any, timing: Any, slots: Any) -> Any:
    return ORIGINAL["brackets"](data, keys, {c: 0 for c in sizes}, timing, slots)


@pytest.mark.parametrize("name,broken", [("position", _swap_weights), ("superiority_at", _no_renormalising),
                                         ("bracket_in", _w_flipped), ("brackets", _unfinished_too)])
def test_a_broken_slot_variant_fails_the_gate(tmp_path: Path, monkeypatch: Any, name: str, broken: Any) -> None:
    """The exporter's slot code broken one way at a time: the gate (brackets and cells against the page) must fail."""
    monkeypatch.setattr(sx, name, broken)
    data, sizes = fixture(tmp_path)
    try:
        g = slot_gate(tmp_path, data, sizes, TIMING, KEYS6)
    except (KeyError, AssertionError):
        return                                                            # failing loudly fails the gate too
    assert g["cells"] + g["brackets"] > 0


# ---- real results, when given ------------------------------------------------------------------------------------------
@pytest.mark.skipif(not os.environ.get("PAIRSTATS_ROWS"), reason="PAIRSTATS_ROWS names no rows directory")
def test_pairstats_equals_the_exporter_on_real_rows(tmp_path: Path) -> None:
    root = Path(os.environ["PAIRSTATS_ROWS"])
    data = sx.load_rows(str(root))
    catalogs = sorted({c for m in data for (c, _r) in data[m]})
    sizes = {c: max(len(rs) for m in data for (cc, _r), rows in data[m].items() if cc == c for rs in sx.by_draw(rows).values()) for c in catalogs}
    timing = {k: v for k, v in json.load(open(root / "timing.json")).items() if isinstance(v, dict)} if (root / "timing.json").exists() else {}
    methods = [m for m in os.environ.get("PAIRSTATS_METHODS", "").split(",") if m] or sorted(data)
    data = {m: data[m] for m in methods if m in data}
    g = slot_gate(tmp_path, data, sizes, timing, sx.RANK_KEYS)
    print(f"real rows: {g['n_cells']} slot cells ({g['between']} with a method between budgets), {g['n_brackets']} brackets; "
          f"cell mismatches {g['cells']}, bracket mismatches {g['brackets']}, w differing {len(g['w_differ'])}")
    assert g["w_differ"] == [] and g["brackets"] == 0 and g["cells"] == 0


# ---- the browser fixture (results-site/tests/fixtures/pairpos, scripts/site_pairpos_fixture.py) ---------------------
FIXTURE = ROOT / "results-site" / "tests" / "fixtures" / "pairpos"
FIXTURE_CHECK = r"""
const fs = require("fs"), vm = require("vm"), path = require("path");
globalThis.window = globalThis;
const [PSJ, RES, FX] = process.argv.slice(1), PS = require(PSJ);
vm.runInThisContext(fs.readFileSync(RES, "utf8") + fs.readFileSync(path.join(FX, "index.js"), "utf8"));
const walk = (d) => fs.readdirSync(d, { withFileTypes: true }).flatMap((e) => e.isDirectory() ? walk(path.join(d, e.name)) : [path.join(d, e.name)]);
["pv", "ranks", "paired"].forEach((s) => walk(path.join(FX, s)).forEach((f) => vm.runInThisContext(fs.readFileSync(f, "utf8"))));
const D = window.RESULTS_V2, E = JSON.parse(fs.readFileSync(path.join(FX, "expected.json"), "utf8")), c = E.catalogs[0], rel = E.release;
const refTime = (m, r) => { const t = (D.timing[m] || {})[String(r)]; return t > 0 ? t : null; };
const points = (m, time) => { const cs = (D.cells[m] || {})[c] || {}, out = [];
  Object.keys(cs).forEach((k) => { if (cs[k].state !== "complete") { return; } const v = time ? refTime(m, k) : +k; if (v > 0) { out.push([v, +k]); } });
  return out.sort((a, b) => a[0] - b[0]); };
const at = (m, pos) => { const time = pos[0] === "t", b = PS.bracketIn(points(m, time), time ? parseFloat(pos.slice(1)) : +pos);
  return b && { b: [b.r1, b.r2, b.w], P: PS.at(PS.file(rel, m, b.r1, c), PS.file(rel, m, b.r2, c), b.w) }; };
const out = { positions: {}, slots: [] };
for (const pos of Object.keys(E.positions)) {
  const e = E.positions[pos], o = out.positions[pos] = { brackets: {}, chance: {}, paired: null };
  E.methods.forEach((m) => { const x = at(m, pos); o.brackets[m] = x && x.b; });
  Object.keys(e.chance_to_beat).forEach((pk) => { const [a, b] = pk.split("|"); o.chance[pk] = PS.rankPairCell(at(a, pos).P, at(b, pos).P, ["log10_fvu_val"]); });
  o.paired = PS.pairedCell(at("T8-20M", pos).P, at("PySR", pos).P, { keys: ["numeric_recovery_val"] }).m.numeric_recovery_val;
}
const RK = window.RESULTS_V2_RANKCELLS[rel], PC = window.RESULTS_V2_PAIRCELLS[rel];
for (const key of Object.keys(RK)) { for (const pk of Object.keys(RK[key])) { const [a, b] = pk.split("|");
  for (const slot of Object.keys(RK[key][pk][c])) { const A = at(a, slot), B = at(b, slot);
    out.slots.push([key, pk, slot, RK[key][pk][c][slot], PS.rankPairCell(A.P, B.P, [key]), PC[key][pk][c][slot], PS.pairedCell(A.P, B.P, { keys: [key] }).m[key]]); } } }
process.stdout.write(JSON.stringify(out));
"""


@pytest.mark.skipif(not (FIXTURE / "expected.json").exists() or not (ROOT / "results-site" / "data" / "2026-09" / "results.js").exists(),
                    reason="no browser fixture or no release to append it to")
def test_the_browser_fixture_is_what_the_page_computes() -> None:
    """The page, holding the release with the fixture's index.js appended, brackets every method where expected.json
    says, computes the cells expected.json holds, and the slot files' cells at every slot; expected.json's numbers are
    the page's averaging (site_random_effects) of those cells."""
    import site_random_effects as re_
    done = subprocess.run([str(NODE), "-e", FIXTURE_CHECK, str(JS), str(ROOT / "results-site" / "data" / "2026-09" / "results.js"), str(FIXTURE)],
                          capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stderr[-4000:]
    got = json.loads(done.stdout)
    expected = json.loads((FIXTURE / "expected.json").read_text())
    assert set(expected["positions"]) == {"1024", "2000", "1448", "t7"}
    for pos, e in expected["positions"].items():
        g = got["positions"][pos]
        same(e["brackets"], g["brackets"], pos)
        for pk, ch in e["chance_to_beat"].items():
            same(ch["cell"], g["chance"][pk], f"{pos} {pk}")
            f = re_.combine([re_.SetStat(ch["cell"][0], ch["cell"][1], ch["cell"][2])], "normal")
            assert f is not None and (ch["p"], ch["lo"], ch["hi"]) == ((1 + f.mu) / 2, (1 + f.lo) / 2, (1 + f.hi) / 2)
        same(e["paired"]["cell"], g["paired"], f"{pos} paired")
        t = e["paired"]["cell"]
        f = re_.combine([re_.SetStat(t[0], t[1], t[2])], "normal")
        assert f is not None and (e["paired"]["v"], e["paired"]["lo"], e["paired"]["hi"], e["paired"]["p"]) == (f.mu, f.lo, f.hi, f.p)
        assert (e["paired"]["wins"], e["paired"]["losses"], e["paired"]["n"]) == (t[3], t[4], t[0])
    # a budget two methods ran exactly, one between; all between; a time
    b = {pos: e["brackets"] for pos, e in expected["positions"].items()}
    assert b["1024"]["T8-20M"][2] == 0 and b["1024"]["dsr"][2] > 0 and b["2000"]["dsr"][2] == 0 and all(x[2] > 0 for x in b["1448"].values())
    assert len(got["slots"]) > 50
    for key, pk, slot, rank_file, rank_js, paired_file, paired_js in got["slots"]:
        same(rank_file, rank_js, f"ranks/{key} {pk} {slot}")
        same(paired_file, paired_js, f"paired/{key} {pk} {slot}")
