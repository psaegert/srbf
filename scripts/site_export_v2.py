"""Export a srbf benchmark release for the results site's v2 explorer (results-site/explorer_v2.js).

Reads the judged per-law rows of a campaign root (rows2_<method>.csv per baseline, t8s1_rows2.csv for the
Flash-ANSR arms, markers/<key>.txt + unit files for progress, timing.json for the reference-machine time axis)
and writes per-cell summaries (method x catalog x rung: counts, sums, sums of squares, 64-bin histograms) so
the page pools any catalog subset with confidence bands on the client.

PUBLIC / PRIVATE SPLIT. Only the methods named in --public go into the release file that the site ships.
Methods named in --private are written to --private-out ONLY (a file outside the deployed tree; see
results-site/README.md "Local-only methods"): the public payload, page and repository carry no trace of them.

usage: site_export_v2.py <root> <release id> <out.js> [--title ...] [--notes ...] [--sizes suite_law_mu.json]
       [--public e2e,nesymres-100M,T8-3M,...] [--private diffsym-v4.0 --private-out private/results_v2_private.js]"""
import argparse, csv, datetime as dt, json, math, os, sys
import numpy as np

METHODS = [  # key, label, param, color, group, unit-file key
    ("e2e", "E2E", "candidates per bag", "#2f6fd0", "baseline", "e2e"),
    ("nesymres-100M", "NeSymReS 100M", "beam width", "#e8842a", "baseline", "nesymres"),
    ("PySR", "PySR", "seconds", "#d62728", "baseline", None),
    ("diffsym-v4.0", "diffsym v4.0", "samples", "#d6338f", "baseline", "diffsym"),
    ("T8-3M", "Flash-ANSR T8-3M", "draws", "#8fcf8a", "flash-ansr", None),
    ("T8-20M", "Flash-ANSR T8-20M", "draws", "#3e9b4a", "flash-ansr", None),
    ("T8-120M", "Flash-ANSR T8-120M", "draws", "#1b5e20", "flash-ansr", None),
    ("prior", "training prior", "draws", "#9a9a9a", "reference", None)]
RUNGS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 65536]
HIST = {"r2": (-1.0, 1.0), "lmdl": (-4.0, 4.0), "lfvu": (-16.0, 2.0)}
NB = 64
E2E_DEFAULT_MAX_RUNG = 256   # E2E is reported at its default settings only (owner 2026-09-16)


def hist_of(values, key):
    lo, hi = HIST[key]; v = np.asarray(values, float); v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    idx = np.clip(((v - lo) / (hi - lo) * NB).astype(int), 0, NB - 1)
    return np.bincount(idx, minlength=NB).tolist()


def load_rows(path, model=None):
    cells = {}
    if not os.path.exists(path):
        return cells
    for r in csv.DictReader(open(path)):
        if r["draw"] != "1" or (model and r["model"] != model):
            continue
        key = (r["catalog"], int(r["rung"]))
        c = cells.setdefault(key, {"rows": {}})
        r2 = float(r["r2"]) if r["r2"] not in ("", "nan") else float("nan")
        mdl = float(r["mdl_ratio"]) if r.get("mdl_ratio", "") not in ("", "nan") else float("nan")
        lf = float(r["log10_fvu_val"]) if r.get("log10_fvu_val", "") not in ("", "nan") else float("nan")
        c["rows"][int(r["row"])] = (int(r["exact"]), int(r["numeric"]), r2, int(r["success"]), mdl, lf)
    return cells


def summarize(rows):
    vals = list(rows.values()); ok = [x for x in vals if x[3]]
    r2 = np.array([x[2] for x in ok if np.isfinite(x[2])], float)
    lmdl = np.array([math.log2(x[4]) for x in ok if np.isfinite(x[4]) and x[4] > 0], float)
    lfvu = np.array([x[5] for x in ok if np.isfinite(x[5])], float)
    r2c = np.clip(r2, -1.0, 1.0)
    return {"n": len(vals), "ex": sum(x[0] for x in vals), "num": sum(x[1] for x in vals), "ok": len(ok),
            "r2": [int(r2c.size), float(r2c.sum()), float((r2c ** 2).sum()), hist_of(r2c, "r2")],
            "lmdl": [int(lmdl.size), float(lmdl.sum()), float((lmdl ** 2).sum()), hist_of(lmdl, "lmdl")],
            "lfvu": [int(lfvu.size), float(lfvu.sum()), float((lfvu ** 2).sum()), hist_of(lfvu, "lfvu")]}


def build(root, keys, sizes):
    data, status = {}, {}
    for key, label, param, color, group, ukey in METHODS:
        if key not in keys:
            continue
        if group == "baseline" and ukey:
            path = f"{root}/rows2_{key}.csv" if os.path.exists(f"{root}/rows2_{key}.csv") else f"{root}/rows_{key}.csv"
            cells = load_rows(path)
        elif key == "PySR":
            cells = load_rows(f"{root}/rows2_PySR.csv")
        else:
            cells = load_rows(f"{root}/t8s1_rows2.csv", key)
        per = {}
        for (c, r), cell in cells.items():
            if c not in sizes or r not in RUNGS:
                continue
            if key == "e2e" and r > E2E_DEFAULT_MAX_RUNG:
                continue
            rows = cell["rows"]; n = len(rows)
            state = "complete" if n >= sizes[c] else "partial"
            if state == "complete" and not any(x[3] for x in rows.values()):
                state = "failed"
            per.setdefault(c, {})[str(r)] = {"state": state, **summarize(rows)}
        data[key] = per
        mk = f"{root}/markers/{key}.txt"
        tot = f"{root}/units_{ukey}_d1.txt" if ukey else f"{root}/t8s1_units_draw1.txt"
        status[key] = [sum(1 for _ in open(mk)) if os.path.exists(mk) else 0, sum(1 for _ in open(tot)) if os.path.exists(tot) else None]
    return data, status


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root"); ap.add_argument("release"); ap.add_argument("out")
    ap.add_argument("--title", default=""); ap.add_argument("--notes", default="")
    ap.add_argument("--sizes", default=os.path.expanduser("~/srbf_eval_local/nscore_val/results/suite_law_mu.json"))
    ap.add_argument("--public", default="e2e,nesymres-100M,PySR,T8-3M,T8-20M,T8-120M,prior")
    ap.add_argument("--private", default=""); ap.add_argument("--private-out", default="")
    a = ap.parse_args()
    sizes = {c: len(v) for c, v in json.load(open(a.sizes))["per_catalog"].items()}
    timing = json.load(open(f"{a.root}/timing.json")) if os.path.exists(f"{a.root}/timing.json") else {}
    timing_note = timing.pop("note", "") if isinstance(timing, dict) else ""
    public = [k for k in a.public.split(",") if k]; private = [k for k in a.private.split(",") if k]
    if set(public) & set(private):
        sys.exit(f"a method cannot be both public and private: {sorted(set(public) & set(private))}")
    known = {m[0] for m in METHODS}
    unknown = sorted((set(public) | set(private)) - known)
    if unknown:
        sys.exit(f"unknown method keys {unknown}; known: {sorted(known)}")
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    def payload(keys, timing_keys):
        data, status = build(a.root, keys, sizes)
        return {"release": {"id": a.release, "title": a.title or a.release, "notes": a.notes, "generated": stamp},
                "sizes": sizes, "rungs": RUNGS, "hist": HIST, "nb": NB,
                "methods": [{"key": k, "label": l, "param": p, "color": col, "group": g} for k, l, p, col, g, _ in METHODS if k in keys],
                "data": data, "status": status,
                "timing": {k: v for k, v in timing.items() if k in timing_keys}, "timing_note": timing_note}

    pub = payload(public, set(public))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        fh.write("window.RESULTS_V2 = " + json.dumps(pub, separators=(",", ":")) + ";\n")
    print(f"public release {a.release}: {a.out} ({os.path.getsize(a.out) // 1024} kB); methods with data: "
          f"{[k for k in public if pub['data'].get(k)]}; status {pub['status']}")
    if private:
        if not a.private_out:
            sys.exit("--private needs --private-out")
        priv = payload(private, set(private))
        os.makedirs(os.path.dirname(os.path.abspath(a.private_out)), exist_ok=True)
        with open(a.private_out, "w") as fh:
            fh.write("window.RESULTS_V2_PRIVATE = " + json.dumps(priv, separators=(",", ":")) + ";\n")
        print(f"private overlay ({len(private)} method(s)): {a.private_out} -- never inside the deployed tree")


if __name__ == "__main__":
    main()
