#!/usr/bin/env python3
"""Read curves out of a vector (PDF) figure exactly, instead of digitising by eye.

``mutool convert -F svg`` turns a PDF page into an SVG. Every stroked path is
a polyline in page points. The script then:
1. prints the stroke colours and the tick marks found along a chosen axes
   frame;
2. once the tick values are supplied (read off the rendered figure; ``--x-ticks``
   and ``--y-ticks``, linear or log), maps each polyline of the chosen colours
   to data coordinates and writes them as CSV.

Usage:
    digitise_vector_figure.py fig.pdf                     # list frames, colours, ticks
    digitise_vector_figure.py fig.pdf --frame 0 --x-ticks 120 140 160 \\
        --y-ticks 1e-5 1e-2 1e1 1e4 1e7 --y-log --colors '#d62728' --out curves.csv

Only straight segments are read (matplotlib writes data lines as M/L paths);
curve control points are skipped. A frame is a closed axis-aligned black
rectangle; ticks are short black segments perpendicular to its edges, within
1 pt of them. Major and minor ticks are told apart by length (``--major-only``).
"""
import argparse
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
TOK = re.compile(r"[MLHVCZmlhvcz]|" + NUM.pattern)
SVG = "{http://www.w3.org/2000/svg}"


def to_svg(pdf, page=1):
    out = Path(tempfile.mkdtemp()) / "page.svg"
    subprocess.run(["mutool", "convert", "-F", "svg", "-o", str(out), str(pdf), str(page)],
                   check=True, capture_output=True)
    return sorted(out.parent.glob("page*.svg"))[0]


def _matrix(transform):
    if not transform or not transform.startswith("matrix"):
        return np.array([[1.0, 0, 0], [0, 1.0, 0]])
    a, b, c, d, e, f = map(float, NUM.findall(transform))
    return np.array([[a, c, e], [b, d, f]])


def _polylines(d):
    """Subpaths of an SVG path as lists of (x, y); C segments keep their end point."""
    toks, out, cur, pos, cmd, i = TOK.findall(d), [], [], (0.0, 0.0), "M", 0
    while i < len(toks):
        t = toks[i]
        if t.isalpha():
            cmd = t
            i += 1
            if cmd in "Zz":
                if cur:
                    cur.append(cur[0])
                    out.append(cur)
                cur = []
            continue
        if cmd in "ML":
            pos = (float(toks[i]), float(toks[i + 1]))
            i += 2
            if cmd == "M":
                if cur:
                    out.append(cur)
                cur, cmd = [pos], "L"
            else:
                cur.append(pos)
        elif cmd == "H":
            pos = (float(toks[i]), pos[1])
            cur.append(pos)
            i += 1
        elif cmd == "V":
            pos = (pos[0], float(toks[i]))
            cur.append(pos)
            i += 1
        elif cmd == "C":
            pos = (float(toks[i + 4]), float(toks[i + 5]))
            cur.append(pos)
            i += 6
        else:
            raise ValueError(f"unsupported path command {cmd!r}")
    if cur:
        out.append(cur)
    return out


def strokes(svg_path):
    """[(colour, polyline (N, 2) in page points, y down)] for every stroked path."""
    out = []
    for el in ET.parse(svg_path).iter(f"{SVG}path"):
        color = el.get("stroke")
        if not color or color == "none" or not el.get("d"):
            continue
        M = _matrix(el.get("transform"))
        for poly in _polylines(el.get("d")):
            p = np.c_[np.asarray(poly), np.ones(len(poly))] @ M.T
            out.append((color.lower(), p))
    return out


def frames(lines, min_size=40.0, tol=0.5):
    """Axes frames (x0, y0, x1, y1), y down: closed black rectangles, or four
    black spines (matplotlib draws each spine as its own path)."""
    out, hor, ver = [], [], []
    for color, p in lines:
        if color != "#000000":
            continue
        x0, y0, x1, y1 = p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()
        if len(p) == 5 and np.allclose(p[0], p[-1]) and x1 - x0 > min_size and y1 - y0 > min_size:
            out.append((x0, y0, x1, y1))
        elif len(p) == 2 and y1 - y0 < 1e-3 and x1 - x0 > min_size:
            hor.append((x0, x1, y0))
        elif len(p) == 2 and x1 - x0 < 1e-3 and y1 - y0 > min_size:
            ver.append((y0, y1, x0))
    near = lambda a, b: abs(a - b) < tol
    for (ya, yb, xl) in ver:
        for (yc, yd, xr) in ver:
            if xr - xl < min_size or not (near(ya, yc) and near(yb, yd)):
                continue
            top = any(near(h[0], xl) and near(h[1], xr) and near(h[2], ya) for h in hor)
            bottom = any(near(h[0], xl) and near(h[1], xr) and near(h[2], yb) for h in hor)
            if top and bottom:
                out.append((xl, ya, xr, yb))
    return sorted({tuple(round(v, 3) for v in f) for f in out})


def ticks(lines, frame, tol=1.0):
    """Tick positions along the horizontal (x) and vertical (y) edges, with lengths.

    A tick on the bottom or top edge counts for x, one on the left or right
    edge for y; ticks mirrored on opposite edges are merged.
    """
    x0, y0, x1, y1 = frame
    tx, ty = [], []
    for color, p in lines:
        if color != "#000000" or len(p) != 2:
            continue
        (ax, ay), (bx, by) = p
        length = float(np.hypot(bx - ax, by - ay))
        if length > 10 or length < 0.5:
            continue
        on = lambda u, v, edge: min(abs(u - edge), abs(v - edge)) < tol
        if abs(ax - bx) < 1e-3 and x0 - tol <= ax <= x1 + tol and (on(ay, by, y1) or on(ay, by, y0)):
            tx.append((float(ax), length))
        if abs(ay - by) < 1e-3 and y0 - tol <= ay <= y1 + tol and (on(ax, bx, x0) or on(ax, bx, x1)):
            ty.append((float(ay), length))
    dedup = lambda v: sorted({(round(a, 2), round(b, 2)) for a, b in v})
    return dedup(tx), dedup(ty)


def marker_centres(lines, frame, colors):
    """(colour, x, y) of each marker: the bounding-box centre of every closed or
    multi-vertex stroked path of a selected colour, and of every black "x"
    (two crossing diagonal 2-point strokes, merged by their common centre)."""
    x0, y0, x1, y1 = frame
    out, crosses = [], []
    for color, p in lines:
        cx, cy = (p[:, 0].min() + p[:, 0].max()) / 2, (p[:, 1].min() + p[:, 1].max()) / 2
        if not (x0 <= cx <= x1 and y0 <= cy <= y1):
            continue
        size = max(np.ptp(p[:, 0]), np.ptp(p[:, 1]))
        if color == "#000000":
            if len(p) == 2 and np.ptp(p[:, 0]) > 1e-3 and np.ptp(p[:, 1]) > 1e-3 and size < 15:
                crosses.append((round(cx, 2), round(cy, 2)))
        elif color in colors and len(p) >= 3 and size < 15:
            out.append((color, cx, cy))
    if "x" in colors:
        out += [("x", cx, cy) for cx, cy in sorted(set(crosses))]
    return out


def calibrate(positions, values, log):
    """Linear map page coordinate -> data (log10 data if ``log``)."""
    v = np.log10(values) if log else np.asarray(values, float)
    slope, icept = np.polyfit(positions, v, 1)
    resid = np.max(np.abs(np.polyval([slope, icept], positions) - v))
    return (lambda q: 10 ** (slope * np.asarray(q) + icept)) if log else \
        (lambda q: slope * np.asarray(q) + icept), resid


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--page", type=int, default=1)
    ap.add_argument("--frame", type=int, default=None)
    ap.add_argument("--x-ticks", type=float, nargs="+")
    ap.add_argument("--y-ticks", type=float, nargs="+")
    ap.add_argument("--x-log", action="store_true")
    ap.add_argument("--y-log", action="store_true")
    ap.add_argument("--major-only", action="store_true", help="keep only the longest ticks")
    ap.add_argument("--colors", nargs="+", default=None, help="stroke colours; 'x' = black crosses (markers)")
    ap.add_argument("--markers", action="store_true", help="write marker centres instead of line vertices")
    ap.add_argument("--swap", action="store_true",
                    help="figure drawn rotated: data x runs along page y and data y along page x")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    lines = strokes(to_svg(args.pdf, args.page))
    fr = frames(lines)
    if args.frame is None:
        from collections import Counter
        print("stroke colours:", dict(Counter(c for c, _ in lines)))
        for i, f in enumerate(fr):
            tx, ty = ticks(lines, f)
            print(f"frame {i}: {np.round(f, 1)}\n  x ticks (pos, len): {tx}\n  y ticks (pos, len): {ty}")
        return
    f = fr[args.frame]
    tx, ty = ticks(lines, f)
    if args.major_only:
        tx = [t for t in tx if t[1] >= max(l for _, l in tx) - 0.01]
        ty = [t for t in ty if t[1] >= max(l for _, l in ty) - 0.01]
    px, py = sorted(t[0] for t in tx), sorted((t[0] for t in ty), reverse=True)   # y down: bottom first
    if args.swap:                           # data x along page y (top first), data y along page x
        px, py = sorted(t[0] for t in ty), sorted(t[0] for t in tx)
    if len(px) != len(args.x_ticks) or len(py) != len(args.y_ticks):
        raise SystemExit(f"{len(px)} x ticks {px} and {len(py)} y ticks {py} found; "
                         f"got {len(args.x_ticks)} and {len(args.y_ticks)} values")
    fx, rx = calibrate(px, args.x_ticks, args.x_log)
    fy, ry = calibrate(py, args.y_ticks, args.y_log)
    print(f"calibration residuals (data units{', log10' if args.x_log or args.y_log else ''}): x {rx:.2e}, y {ry:.2e}")
    x0, y0, x1, y1 = f
    if args.markers:
        rows = []
        for color, cx, cy in marker_centres(lines, f, [c.lower() for c in args.colors]):
            u, v = (cy, cx) if args.swap else (cx, cy)
            rows.append((color, -1, float(fx(u)), float(fy(v))))
        with open(args.out, "w") as fh:
            fh.write("color,path,x,y\n")
            for color, k, x, y in rows:
                fh.write(f"{color},{k},{x:.6g},{y:.6g}\n")
        print(f"wrote {len(rows)} markers to {args.out}")
        return
    rows = []
    for k, (color, p) in enumerate(lines):
        if args.colors and color not in [c.lower() for c in args.colors]:
            continue
        if len(p) < 3 or (color == "#000000" and not (args.colors and "#000000" in args.colors)):
            continue
        inside = (p[:, 0] >= x0 - 0.5) & (p[:, 0] <= x1 + 0.5) & (p[:, 1] >= y0 - 0.5) & (p[:, 1] <= y1 + 0.5)
        if inside.mean() < 0.5:
            continue
        for x, y in zip(fx(p[inside, 0]), fy(p[inside, 1])):
            rows.append((color, k, x, y))
    if args.out:
        with open(args.out, "w") as fh:
            fh.write("color,path,x,y\n")
            for color, k, x, y in rows:
                fh.write(f"{color},{k},{x:.6g},{y:.6g}\n")
        print(f"wrote {len(rows)} points to {args.out}")
    else:
        for r in rows[:20]:
            print(r)


if __name__ == "__main__":
    main()
