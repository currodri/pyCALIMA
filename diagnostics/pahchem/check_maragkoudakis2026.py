#!/usr/bin/env python3
"""Maragkoudakis+26 (PDRs4All XVIII) recovered from their stated inputs, step by step.

Step A, their pipeline on the PDRs4All MIRI templates (MRS v2 additive
stitching, as their figure files name it):
1. Gas lines: removed with a running median (their masking is not stated).
2. Continuum: pybaselines morphological then imodpoly. The half-window and
   order are not stated, so a grid is run and each setting is scored against
   their own band ratios (digitised from their maps). This recovers one of
   their inputs from their own published data.
3. Decomposition: pyPAHdb's NNLS with the v4.00-alpha matrix
   (``maragkoudakis2026.decompose``), on two samplings: the template's own
   0.00046 um grid, and a cube-like grid (MRS channel steps 0.0008 / 0.0013 /
   0.0025 um), since an unweighted NNLS weights the points it is given.
4. Fractions against their Table 1. Table 1 averages per-pixel fits over
   zones; the templates are flux-weighted pixel averages, so a template fit is
   close to the zone value but not the same.

Needs pybaselines (``--pylibs`` to add a directory holding it), the matrix
(dataset ``pypahdb-matrix-v4-00-alpha``) and the templates file.

    check_maragkoudakis2026.py --templates .../PDRs4All_templates_MIRI_MRS_v2_additive_stitching.dat
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

from pycalima.pahchem import maragkoudakis2026 as M

TEMPLATES = ("HII", "Atomic", "DF1", "DF2", "DF3")
TABLE1_ZONE = {"HII": "HII", "Atomic": "APDR", "DF1": "DF1", "DF2": "DF2", "DF3": "DF3"}
CUBE_STEPS = ((4.9, 7.65, 0.0008), (7.65, 11.7, 0.0013), (11.7, 15.05, 0.0025))


def load_v2(path):
    """lambda [um], {template: I_nu [MJy/sr]}; columns HII, APDR, DF1, DF2, DF3 with errors."""
    v = np.loadtxt(path)
    return v[:, 0], {t: v[:, 1 + 2 * i] for i, t in enumerate(TEMPLATES)}


def remove_lines(lam, f, width_um=0.02):
    n = max(3, int(round(width_um / np.median(np.diff(lam)))) | 1)
    return median_filter(f, size=n, mode="nearest")


def cube_grid():
    return np.concatenate([np.arange(a, b, s) for a, b, s in CUBE_STEPS])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--templates", type=Path, required=True)
    ap.add_argument("--pylibs", type=Path, default=None, help="directory holding pybaselines")
    ap.add_argument("--matrix", type=Path, default=None)
    args = ap.parse_args()
    if args.pylibs:
        sys.path.append(str(args.pylibs))
    lam, tmpl = load_v2(args.templates)
    s = (lam > 4.95) & (lam < 15.5)
    lam = lam[s]
    obs = {t: remove_lines(lam, f[s]) for t, f in tmpl.items()}
    dig = M.templates_digitised().set_index("template")
    # -- 2. continuum settings scored against their own ratios --------------------------------------
    step = np.median(np.diff(lam))
    best = None
    print("continuum grid (half-window in um, order, variant): rms log10 offset of 6.2/7.7/8.6 over 11.2")
    for hw_um in (0.1, 0.2, 0.4, 0.8, 1.6):
        for order in (2, 3, 4, 5, 6):
            for variant in ("mor_then_imodpoly", "imodpoly"):
                if variant == "imodpoly" and hw_um != 0.1:
                    continue
                dev = []
                for t in TEMPLATES:
                    base = M.baseline_pybaselines(lam, obs[t], int(hw_um / step), order, variant)
                    r = M.band_ratios(lam, obs[t] - base)
                    for b, col in (("6.2", "r62_112"), ("7.7", "r77_112"), ("8.6", "r86_112")):
                        dev.append(np.log10(r[b] / dig.loc[t, col]))
                rms = float(np.sqrt(np.mean(np.square(dev))))
                print(f"  {hw_um:4.1f} {order} {variant:18s} rms {rms:.3f} dex (mean {np.mean(dev):+.3f})")
                if best is None or rms < best[0]:
                    best = (rms, hw_um, order, variant)
    rms, hw_um, order, variant = best
    print(f"best: half-window {hw_um} um, order {order}, {variant}: rms {rms:.3f} dex")
    # -- 3-4. decomposition ---------------------------------------------------------------------------
    pre = M.load_precomputed(args.matrix)
    t1 = M.table1().set_index("zone")
    keys = ("f_neut", "f_cat", "f_an", "f_sm", "f_med", "f_lar", "n_c")
    for grid_name in ("template grid", "cube-like grid"):
        print(f"\n=== decomposition on the {grid_name} (continuum: best setting) ===")
        print(f"  {'':8s} " + " ".join(f"{k:>7s}" for k in keys) + "   dication+ error")
        for t in TEMPLATES:
            base = M.baseline_pybaselines(lam, obs[t], int(hw_um / step), order, variant)
            y = obs[t] - base
            if grid_name == "cube-like grid":
                g = cube_grid()
                yy, ll = np.interp(g, lam, y), g
            else:
                yy, ll = y, lam
            w, fit = M.decompose(ll, yy, pre)
            fr = M.fractions(w, pre)
            err = M.fit_error(ll, yy, fit)
            print(f"  {t:8s} " + " ".join(f"{fr[k]:7.2f}" for k in keys) + f"   {fr['f_dication_plus']:.2f}     {err:.2f}")
            z = t1.loc[TABLE1_ZONE[t]]
            print(f"  {'Table 1':8s} " + " ".join(f"{z[k]:7.2f}" for k in keys))


if __name__ == "__main__":
    main()
