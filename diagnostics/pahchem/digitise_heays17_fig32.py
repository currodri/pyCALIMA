#!/usr/bin/env python3
"""Digitise Heays+17 Fig. 32 (lower panel): the cumulative cosmic-ray-induced photons per H2 ionisation.

Heays, Bosman & van Dishoeck 2017 (A&A 602, A105; arXiv:1701.04459) do not distribute their P(lambda) (Gredel+89's
spectrum) as a file; the figure is a raster image in the PDF. This renders p. 52 at 600 dpi (pdftoppm), finds the lower
panel's frame and ticks from the dark pixels and the curve from the red ones (mean row per column), calibrates x on
the 100-250 nm ticks and the 300 nm frame and y on the frame (0.0, 0.5; checked on the 0.1-0.4 ticks), restores the
Lyman alpha step that the plotted line's width smears (at 121.567 nm, from the local slopes either side) and removes the
digitisation noise with a running maximum. Writes ``data/external_data/pahchem/heays17_cr_uv_cumulative.csv``
(``pahchem.cr_uv``).

Result (2026-11-04): calibration residuals <= 0.04 nm and <= 0.001; total 0.3627 at 299.7 nm (the paper's "only 0.36");
Lyman alpha 0.049 photons, 13.5 % (the paper's "15%"). Below ~92 nm the curve runs along the frame and reads up to
0.003 high.

    digitise_heays17_fig32.py PATH/TO/heays17.pdf [--out CSV]
"""
import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from pycalima import _paths

LYMAN_ALPHA_NM = 121.567
HEADER = """# Heays, Bosman & van Dishoeck 2017, A&A 602, A105 (arXiv:1701.04459), Fig. 32, lower panel: the cumulative number of
# ultraviolet photons generated per primary cosmic-ray ionisation of H2, P(lambda) of Gredel et al. (1989) integrated
# from short wavelengths (para-H2, 1 km/s lines). The paper's P(lambda) is not distributed as a file (2026-11-04).
# Digitised from the raster figure on p. 52 of the arXiv PDF rendered at 600 dpi (diagnostics/pahchem/
# digitise_heays17_fig32.py): mean row of the curve's pixels per column; x calibrated on the 100-250 nm ticks and the
# 300 nm frame (residuals <= {xres:.2f} nm), y on the frame (0.0, 0.5), checked on the 0.1-0.4 ticks (<= {yres:.3f}). Lyman
# alpha, smeared over the plotted line's width, is restored as a step at 121.567 nm from the local slopes either side:
# {step:.4f} photons (Heays+17: "15% of the integrated flux"). A running maximum removes the digitisation noise (a
# cumulative cannot fall). The total, {total:.4f} at {lmax:.1f} nm, is the paper's "only 0.36". Below ~92 nm the curve runs
# along the frame and reads up to 0.003 high. 2026-11-04.
wavelength_nm,cumulative_photons_per_ionisation"""


def runs(idx):
    return [r for r in np.split(idx, np.nonzero(np.diff(idx) > 3)[0] + 1) if r.size]


def digitise(png):
    """(lambda [nm], cumulative, x residual [nm], y residual) of the lower panel."""
    im = np.asarray(Image.open(png).convert("RGB")).astype(int)
    R, G, B = im[..., 0], im[..., 1], im[..., 2]
    dark = (R < 90) & (G < 90) & (B < 90)
    red = (R > 150) & (R - G > 50) & (R - B > 20)
    w = dark.shape[1]
    frame_rows = runs(np.nonzero(dark.sum(axis=1) > 0.6 * w)[0])
    top, bottom = int(frame_rows[-2].mean()), int(frame_rows[-1].mean())
    band = dark[top + 5:bottom - 5]
    frame_cols = runs(np.nonzero(band.sum(axis=0) > 0.9 * band.shape[0])[0])
    left, right = int(frame_cols[0].mean()), int(frame_cols[-1].mean())
    ticks = [float(t.mean()) for t in runs(np.nonzero(dark[bottom - 14:bottom - 6, left + 8:right - 8].all(axis=0))[0]
                                           + left + 8)]
    xs = np.array(ticks[:4] + [right], dtype=float)
    lam = np.array([100.0, 150.0, 200.0, 250.0, 300.0])
    a, b = np.polyfit(xs, lam, 1)
    val = lambda r: 0.5 * (bottom - r) / (bottom - top)                     # noqa: E731
    yt = [val(float(t.mean())) for t in runs(np.nonzero(dark[top + 8:bottom - 8, left + 6:left + 14].all(axis=1))[0]
                                             + top + 8)]
    out = []
    for c in range(left + 4, right - 1):
        rr = np.nonzero(red[top + 3:bottom - 2, c])[0]
        if rr.size:
            out.append((a * c + b, val(rr.mean() + top + 3)))
    out = np.array(out)
    yres = float(np.max(np.abs(np.sort(yt) - np.array([0.1, 0.2, 0.3, 0.4]))))
    return out[:, 0], out[:, 1], float(np.max(np.abs(lam - (a * xs + b)))), yres


def restore(lam, cum):
    """The curve with Lyman alpha as a step at LYMAN_ALPHA_NM (from the local slopes) and made non-decreasing."""
    left, right = lam <= 120.75, lam >= 122.2
    sl = np.polyfit(lam[left & (lam > 118.7)], cum[left & (lam > 118.7)], 1)[0]
    sr = np.polyfit(lam[right & (lam < 124.3)], cum[right & (lam < 124.3)], 1)[0]
    pre = cum[left][-1] + sl * (LYMAN_ALPHA_NM - lam[left][-1])
    post = cum[right][0] - sr * (lam[right][0] - LYMAN_ALPHA_NM)
    L = np.concatenate([lam[left], [LYMAN_ALPHA_NM, LYMAN_ALPHA_NM + 1e-3], lam[right]])
    C = np.maximum(np.maximum.accumulate(np.concatenate([cum[left], [pre, post], cum[right]])), 0.0)
    return L, C, post - pre


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--out", type=Path, default=_paths.get_external_data_path("pahchem", "heays17_cr_uv_cumulative.csv"))
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        stem = Path(tmp) / "fig32"
        subprocess.run(["pdftoppm", "-f", "52", "-l", "52", "-r", "600", "-x", "2700", "-y", "600", "-W", "2100",
                        "-H", "1400", "-png", str(a.pdf), str(stem)], check=True)
        lam, cum, xres, yres = digitise(next(Path(tmp).glob("fig32*.png")))
    L, C, step = restore(lam, cum)
    with open(a.out, "w") as f:
        f.write(HEADER.format(xres=xres, yres=yres, step=step, total=C[-1], lmax=L[-1]) + "\n")
        f.writelines(f"{x:.3f},{y:.5f}\n" for x, y in zip(L, C))
    print(f"{L.size} rows; x residual {xres:.3f} nm, y residual {yres:.4f}; Lyman alpha {step:.4f} "
          f"({step / C[-1] * 100:.1f} %); total {C[-1]:.4f} -> {a.out}")


if __name__ == "__main__":
    main()
