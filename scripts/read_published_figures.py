"""Read the published numbers that the Reproductions page takes from figures, from the papers' own PDFs.

A figure in these PDFs is a vector drawing: every plotted marker is a path with exact page coordinates (in points,
1/72 inch). A value is the marker centre's position between two labelled positions of a linear axis:

    value = v0 + (v1 - v0) * (c - c0) / (c1 - c0)

where c is the marker centre and c0, c1 are the axis positions labelled v0, v1 (a label's position is the grid or tick
line it labels, or its own centre where the axis has no line there). The script downloads each PDF from its pinned
address, checks its SHA-256, finds the labels, the lines and the markers in the drawing, and prints every coordinate
and the arithmetic. It needs PyMuPDF:

    pip install pymupdf
    python scripts/read_published_figures.py            # every reading
    python scripts/read_published_figures.py e2e-fig5   # one
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

PAPERS = {
    "e2e": {"url": "https://proceedings.neurips.cc/paper_files/paper/2022/file/"
                   "42eb37cdbefd7abae0835f4b67548c39-Paper-Conference.pdf",
            "sha256": "5b3bc0688ea7045a"},
    "nesymres": {"url": "https://arxiv.org/pdf/2106.06427v1",
                 "sha256": "491da1f127f4ddf2"},
}


def fetch(key: str, cache: Path) -> Path:
    paper = PAPERS[key]
    path = cache / f"{key}.pdf"
    if not path.exists():
        req = urllib.request.Request(paper["url"], headers={"User-Agent": "srbf-read-published-figures"})
        with urllib.request.urlopen(req, timeout=60) as r:
            path.write_bytes(r.read())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if not digest.startswith(paper["sha256"]):
        raise SystemExit(f"{key}: {paper['url']} has SHA-256 {digest[:16]}..., not the {paper['sha256']}... read before")
    return path


Box = tuple[float, float, float, float]


def spans(page: Any) -> list[tuple[str, Box]]:
    out = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for s in line["spans"]:
                if s["text"].strip():
                    x0, y0, x1, y1 = s["bbox"]
                    out.append((s["text"].strip(), (float(x0), float(y0), float(x1), float(y1))))
    return out


def centre(b: tuple[float, float, float, float]) -> tuple[float, float]:
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def rgb(c: Any) -> tuple[float, ...] | None:
    return tuple(round(x, 3) for x in c) if c else None


def snap(page: Any, axis: str, pos: float, lo: float, hi: float) -> tuple[float, str]:
    """The line a label at `pos` labels: a straight line across the axis direction (a tick or a gridline) within
    1 pt of the label's centre, between `lo` and `hi` in the other direction; else the label's centre itself."""
    best = None
    for p in page.get_drawings():
        items = p.get("items") or []
        if len(items) != 1 or items[0][0] != "l":
            continue
        a, b = items[0][1], items[0][2]
        if axis == "x" and abs(a.x - b.x) < 0.01 and min(a.y, b.y) >= lo - 12 and max(a.y, b.y) <= hi + 12:
            d = abs(a.x - pos)
            if d < 1 and (best is None or d < best[0]):
                best = (d, a.x)
        if axis == "y" and abs(a.y - b.y) < 0.01 and min(a.x, b.x) >= lo - 12 and max(a.x, b.x) <= hi + 12:
            d = abs(a.y - pos)
            if d < 1 and (best is None or d < best[0]):
                best = (d, a.y)
    return (best[1], "line") if best else (pos, "label centre")


def markers(page: Any, colour: tuple[float, ...], box: tuple[float, float, float, float]) -> list[tuple[float, float]]:
    """Centres of the filled marker paths of one colour inside a box (x0, y0, x1, y1)."""
    out = []
    for p in page.get_drawings():
        r = p["rect"]
        if rgb(p.get("fill")) == colour and r.width < 8 and r.height < 8:
            cx, cy = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
            if box[0] <= cx <= box[2] and box[1] <= cy <= box[3]:
                out.append((cx, cy))
    return out


def legend_colour(page: Any, label: str, near: tuple[float, float, float, float]) -> tuple[float, ...]:
    """The fill colour of the legend marker just left of a legend entry's text."""
    sp = [b for t, b in spans(page) if t == label and near[0] <= b[0] <= near[2] and near[1] <= b[1] <= near[3]]
    if len(sp) != 1:
        raise SystemExit(f"legend entry {label!r}: {len(sp)} matches")
    b = sp[0]
    cy = (b[1] + b[3]) / 2
    cands: list[tuple[float, tuple[float, ...]]] = []
    for p in page.get_drawings():
        r = p["rect"]
        if p.get("fill") and r.width < 8 and r.height < 8 and r.x1 <= b[0] + 0.5 and b[0] - 25 <= r.x0:
            colour = rgb(p["fill"])
            if colour is not None and abs((r.y0 + r.y1) / 2 - cy) < 3:
                cands.append((b[0] - r.x1, colour))
    if not cands:
        raise SystemExit(f"legend entry {label!r}: no marker")
    return min(cands)[1]


def linear(v0: float, c0: float, v1: float, c1: float, c: float) -> float:
    return v0 + (v1 - v0) * (c - c0) / (c1 - c0)


def e2e_fig5(cache: Path) -> dict[str, Any]:
    """E2E, Fig. 5 (page 9): the noise-0 marker of 'Ours' in the 'Mean accuracy (R2 > 0.99)' panel."""
    import pymupdf
    page = pymupdf.open(fetch("e2e", cache))[8]
    sp = spans(page)
    panel = [b for t, b in sp if t.startswith("Mean accuracy")][0]
    ticks = {t: b for t, b in sp if t in ("0.0", "1.0") and panel[0] - 10 <= b[0] <= panel[2] + 60 and b[1] > panel[3] + 100}
    ours = [b for t, b in sp if t == "Ours" and b[2] <= panel[0] + 1]
    if len(ticks) != 2 or len(ours) != 1:
        raise SystemExit("e2e-fig5: the panel's axis labels or the 'Ours' row were not found")
    axis_y = ticks["0.0"][1]
    x0, how0 = snap(page, "x", centre(ticks["0.0"])[0], panel[3], axis_y)
    x1, how1 = snap(page, "x", centre(ticks["1.0"])[0], panel[3], axis_y)
    colour = legend_colour(page, "0.0", (300, 490, 400, 510))
    row_y = centre(ours[0])[1]
    found = markers(page, colour, (x0 - 5, row_y - 3, x1 + 5, row_y + 3))
    if len(found) != 1:
        raise SystemExit(f"e2e-fig5: {len(found)} noise-0 markers in the 'Ours' row")
    cx, cy = found[0]
    value = linear(0.0, x0, 1.0, x1, cx)
    return {"reading": "e2e-fig5", "page": 9, "colour": colour, "axis": {"0.0": (round(x0, 3), how0), "1.0": (round(x1, 3), how1)},
            "marker": (round(cx, 3), round(cy, 3)), "value": value,
            "arithmetic": f"(x {cx:.3f} - {x0:.3f}) / ({x1:.3f} - {x0:.3f}) = {value:.4f}"}


def nesymres(cache: Path, page_no: int, name: str) -> dict[str, Any]:
    """NeSymReS, the AI Feynman panel of Fig. 3 (page 7) or Fig. 8 (page 17): the NeSymReS markers, one per beam width
    1, 2, 4, ..., 256 (the paper's Table 1) in order of CPU time; the sixth is beam width 32."""
    import pymupdf
    page = pymupdf.open(fetch("nesymres", cache))[page_no - 1]
    sp = spans(page)
    titles = sorted((b for t, b in sp if t == "AI Feynman"), key=lambda b: b[1])
    title: Box = titles[0] if page_no == 7 else titles[1]
    colour = legend_colour(page, "NeSymReS (ours)", (0, title[3], 600, title[3] + 160))
    # the panel's y labels: numbers just left of the panel, between its title and the CPU-seconds axis
    xlab = [b for t, b in sp if t == "CPU seconds" and abs(centre(b)[0] - centre(title)[0]) < 40 and b[1] > title[3]]
    bottom = min(xlab, key=lambda b: b[1])[1]
    ylabs = [(float(t), b) for t, b in sp if re.fullmatch(r"\d\.\d+", t) and b[2] < title[0] + 10
             and title[3] < centre(b)[1] < bottom and title[0] - 40 < b[0]]
    ylabs.sort(key=lambda vb: vb[1][1])
    (vhi, bhi), (vlo, blo) = ylabs[0], ylabs[-1]
    yhi, how_hi = snap(page, "y", centre(bhi)[1], bhi[2], bhi[2] + 100)
    ylo, how_lo = snap(page, "y", centre(blo)[1], blo[2], blo[2] + 100)
    # the panel ends where the next panel's y-axis numbers begin
    right = min(b[0] for t, b in sp if re.fullmatch(r"\d\.\d+", t) and b[0] > title[2]
                and title[3] < centre(b)[1] < bottom)
    pts = sorted(markers(page, colour, (title[0] - 5, title[3], right - 5, bottom)))
    if len(pts) != 9:
        raise SystemExit(f"{name}: {len(pts)} NeSymReS markers in the AI Feynman panel, expected 9 (beam 1 to 256)")
    cx, cy = pts[5]
    value = linear(vlo, ylo, vhi, yhi, cy)
    return {"reading": name, "page": page_no, "colour": colour,
            "axis": {str(vlo): (round(ylo, 3), how_lo), str(vhi): (round(yhi, 3), how_hi)},
            "marker": (round(cx, 3), round(cy, 3)), "value": value, "of_52": value * 52,
            "arithmetic": f"{vlo} + ({vhi} - {vlo}) * (y {cy:.3f} - {ylo:.3f}) / ({yhi:.3f} - {ylo:.3f}) = {value:.4f};"
                          f" x 52 laws = {value * 52:.2f}",
            "all_beams": [(2 ** i, round(linear(vlo, ylo, vhi, yhi, y) * 52, 2)) for i, (_, y) in enumerate(pts)]}


READINGS = {
    "e2e-fig5": e2e_fig5,
    "nesymres-fig3": lambda cache: nesymres(cache, 7, "nesymres-fig3"),
    "nesymres-fig8": lambda cache: nesymres(cache, 17, "nesymres-fig8"),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("readings", nargs="*", choices=[[], *READINGS], help="default: all")
    ap.add_argument("--cache", type=Path, default=None, help="where to keep the downloaded PDFs (default: a temporary directory)")
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        cache = a.cache or Path(tmp)
        cache.mkdir(parents=True, exist_ok=True)
        for name in a.readings or list(READINGS):
            r = READINGS[name](cache)
            print(f"{r['reading']} (page {r['page']}, marker colour {r['colour']})")
            for v, (pos, how) in r["axis"].items():
                print(f"  axis {v}: {pos} pt ({how})")
            print(f"  marker centre: {r['marker']} pt")
            print(f"  {r['arithmetic']}")
            if "all_beams" in r:
                print("  every beam width, as laws of 52: " + ", ".join(f"{w}: {n}" for w, n in r["all_beams"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
