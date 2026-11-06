#!/usr/bin/env python3
"""Extract the CH-stretch cascade spectra of Mackie+18's supplement (vector PDF) into a CSV.

Each page of ``cascade_spectra.pdf`` (manual dataset ``mackie2018-supplement``) holds one molecule: four
panels, the top right the CH stretches (3150-2800 cm^-1), with the 3, 5, 7 and 9 eV cascades as strips
stacked bottom to top, each normalised to its own highest peak. The curves are vector paths: this reads them
from ``mutool convert -F svg`` (MuPDF) and calibrates the frequency axis on the panel's major ticks (3100,
3000, 2900, 2800 cm^-1). Writes data/external_data/pahchem/mackie2018_ch_cascades.csv (molecule, e_uv_ev,
nu_cm, intensity normalised to the strip's peak).

    digitise_mackie2018.py [--molecules 123678-hexahydropyrene 9-methylanthracene ...]   (needs mutool)
"""
import argparse
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from pycalima import _paths
from pycalima._datasets import ensure_dataset

COLOURS = {3: "#a52a2a", 5: "#ffa500", 7: "#008080", 9: "#ff0000"}
MAJOR_CM = (3100.0, 3000.0, 2900.0, 2800.0)     # left to right (the panel starts at 3150)


def _parse(d):
    pts, x, y = [], 0.0, 0.0
    for cmd, args in re.findall(r"([MLHVCZ])([^MLHVCZ]*)", d):
        v = [float(t) for t in re.findall(r"-?\d*\.?\d+(?:[eE][-+]?\d+)?", args)]
        if cmd in "ML":
            for i in range(0, len(v) - 1, 2):
                x, y = v[i], v[i + 1]
                pts.append((x, y))
        elif cmd == "H":
            for t in v:
                x = t
                pts.append((x, y))
        elif cmd == "V":
            for t in v:
                y = t
                pts.append((x, y))
    return np.array(pts)


def _paths_of(svg):
    out = []
    for el in re.findall(r"<path [^>]*>", svg):
        st = re.search(r'stroke="(#[0-9a-f]{6})"', el)
        d = re.search(r' d="([^"]*)"', el)
        tr = re.search(r'transform="matrix\(([^)]*)\)"', el)
        if not (st and d):
            continue
        m = [float(t) for t in tr.group(1).split(",")] if tr else [1, 0, 0, 1, 0, 0]
        p = _parse(d.group(1))
        if len(p):
            out.append((st.group(1), m[0] * p[:, 0] + m[2] * p[:, 1] + m[4], m[1] * p[:, 0] + m[3] * p[:, 1] + m[5]))
    return out


def page_titles(pdf):
    text = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True, check=True).stdout
    titles = {}
    for n, page in enumerate(text.split("\f"), start=1):
        m = re.search(r"cascade spectra of (\S+) at 3, 5, 7, and 9 eV", page)
        if m:
            titles[m.group(1)] = n
    return titles


def ch_panel(pdf, page):
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["mutool", "convert", "-F", "svg", "-o", f"{tmp}/p%d.svg", str(pdf), str(page)], check=True,
                       capture_output=True)
        svg = Path(f"{tmp}/p1.svg").read_text()
    paths = _paths_of(svg)
    coloured = [p for p in paths if p[0] in COLOURS.values()]
    right = [p for p in coloured if p[1].min() > 300 and p[2].max() < 300]            # top-right panel
    x0, x1 = min(p[1].min() for p in right), max(p[1].max() for p in right)
    # the panel's bottom axis: the long black horizontal line under the curves
    segs = []
    for c, X, Y in paths:
        if c == "#000000":
            segs += [(X[i], Y[i], X[i + 1], Y[i + 1]) for i in range(len(X) - 1)]
    segs = np.array(segs)
    horiz = segs[(np.abs(segs[:, 1] - segs[:, 3]) < 0.05) & (np.abs(segs[:, 2] - segs[:, 0]) > 0.8 * (x1 - x0))
                 & (np.minimum(segs[:, 0], segs[:, 2]) < x0 + 1) & (np.maximum(segs[:, 0], segs[:, 2]) > x1 - 1)]
    y_axis = max(y for y in horiz[:, 1] if y < max(p[2].max() for p in right) + 5)
    vert = segs[(np.abs(segs[:, 0] - segs[:, 2]) < 0.05) & (segs[:, 0] > x0 - 0.5) & (segs[:, 0] < x1 + 0.5)
                & (np.maximum(segs[:, 1], segs[:, 3]) > y_axis - 0.5) & (np.maximum(segs[:, 1], segs[:, 3]) < y_axis + 0.5)
                & (np.abs(segs[:, 3] - segs[:, 1]) > 0.1) & (np.abs(segs[:, 3] - segs[:, 1]) < 10)]
    lengths = np.abs(vert[:, 3] - vert[:, 1])
    major = sorted({round(float(x), 2) for x, ln in zip(vert[:, 0], lengths) if ln > 0.75 * lengths.max()})
    if len(major) not in (3, 4) or np.ptp(np.diff(major)) > 0.05:
        raise ValueError(f"page {page}: major ticks {major}")
    slope, icpt = np.polyfit(major, MAJOR_CM[:len(major)], 1)
    # four strips of equal height stacked from the axis, 3 eV at the bottom; each curve's peak is its strip top
    height = (y_axis - min(p[2].min() for p in right)) / 4.0
    tops = [y_axis - (k + 1) * height for k in range(4)]
    out = {}
    for k, (ev, col) in enumerate(COLOURS.items()):
        segs_c = [p for p in right if p[0] == col]
        X = np.concatenate([p[1] for p in segs_c])
        Y = np.concatenate([p[2] for p in segs_c])
        bot, top = y_axis - k * height, y_axis - (k + 1) * height
        nu = slope * X + icpt
        o = np.argsort(nu)
        out[ev] = (nu[o], ((bot - Y) / (bot - top))[o])
    return out, (slope, icpt), tops


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--molecules", nargs="*", default=["123678-hexahydropyrene", "9-methylanthracene",
                                                       "910-dimethylanthracene", "pyrene"])
    args = ap.parse_args()
    pdf = ensure_dataset("mackie2018-supplement") / "cascade_spectra.pdf"
    titles = page_titles(pdf)
    rows = ["molecule,e_uv_ev,nu_cm,intensity"]
    for mol in args.molecules:
        if mol not in titles:
            raise SystemExit(f"{mol}: not in the supplement ({sorted(titles)})")
        panel, (a, b), _ = ch_panel(pdf, titles[mol])
        print(f"{mol} (page {titles[mol]}): axis nu = {a:.4f} x + {b:.2f}")
        for ev, (nu, f) in panel.items():
            g = np.arange(2805.0, 3145.01, 0.5)
            rows += [f"{mol},{ev},{x:.1f},{y:.4f}" for x, y in zip(g, np.interp(g, nu, f))]
    path = _paths.get_external_data_path("pahchem") / "mackie2018_ch_cascades.csv"
    header = ("# Mackie et al. (2018), J. Chem. Phys. 149, 134302, supplementary cascade_spectra.pdf: CH-stretch\n"
              "# panels of the 3, 5, 7, 9 eV cascade spectra, read from the vector paths 2026-10-21 by\n"
              "# diagnostics/pahchem/digitise_mackie2018.py; intensity normalised to each strip's peak (theirs), on a\n"
              "# 0.5 cm^-1 grid (interpolated between the paths' points).\n")
    path.write_text(header + "\n".join(rows) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
