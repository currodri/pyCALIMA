#!/usr/bin/env python3
"""Murga+2022 model b (H loss and addition only) through pahchem.murga2022, against Figshare.

Prints:
- the deep-MC calibration (locations 134-151, 3.3 A, where H addition is
  negligible): the fitted bin-step rates against our photo-loss rates;
- the last-output (1e5 yr) X_H distribution of every (location, Z, size),
  f(a, Z)-weighted and summarised by region and size;
- the f- and n-weighted <X_H/X_H0>(distance) of Fig. 8.

Needs the murga2022-figshare dataset. The photo-loss tables (~3 min each)
are cached next to it.

Measured (2026-10-05): see the module docstring of pahchem.murga2022 and
tests/test_pahchem_murga2022.py.
"""
import argparse

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from pycalima._paths import get_external_data_path
from pycalima.pahchem import murga2022 as N
from pycalima.pahchem import murga_data as M
from pycalima.pahchem.murga_charge import region_of

REGIONS = ("HII", "PDR 1", "PDR 2", "DF", "MC")


def fit_chain(fracs, times_yr=N.OUTPUT_TIMES_YR):
    """(k1, k2) [s^-1] of normal -> middle -> lowest bin, an irreversible chain."""
    t = np.asarray(times_yr) * N.YR

    def model(p):
        k1, k2 = np.exp(p)
        a = np.exp(-k1 * t)
        b = k1 / (k2 - k1) * (np.exp(-k1 * t) - np.exp(-k2 * t))
        return np.r_[a - fracs[:, 2], b - fracs[:, 1]]

    s = least_squares(model, np.log([1e-12, 1.3e-12]))
    return np.exp(s.x), float(np.abs(s.fun).max())


def fractions(n):
    tot = n.sum(axis=-2, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tot > 0, n / tot, 0.0)


def summary(ours, theirs_n, f, reg, t=-1):
    """Per region and size: f-weighted bin fractions of ours and theirs at output t."""
    rows = []
    th = fractions(theirs_n[:, :, t])                      # loc, Z, bin, size
    w = f[:, :, 0, 0, :] * (theirs_n[:, :, 0].sum(axis=2) > 0)   # loc, Z, size
    for r in REGIONS:
        m = reg == r
        for s in range(th.shape[-1]):
            ww = w[m, :, s][:, :, None]
            o = (ours[m, :, t, :, s] * ww).sum(axis=(0, 1)) / ww.sum()
            p = (th[m, :, :, s] * ww).sum(axis=(0, 1)) / ww.sum()
            rows.append((r, s, o, p))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--temperature", choices=("dl01", "tielens"), default="dl01")
    ap.add_argument("--loss-scale", type=float, default=1.0)
    args = ap.parse_args()

    d = M.read_model_2022("b")
    prof = pd.read_csv(get_external_data_path("pahchem/murga2022_fig6_profile.csv"), comment="#")
    reg = np.array([region_of(x) for x in prof.distance_arcsec])
    R = N.photo_loss_table(d["a_cm"], temperature=args.temperature)

    print("Deep-MC calibration, 3.3 A neutral (k = fitted bin-step rate, R = our photo loss):")
    for loc in (134, 138, 142, 146, 150):
        nn = d["n"][loc, 0, :, :, 0]                       # t, bin
        (k1, k2), res = fit_chain(nn / nn.sum(axis=1, keepdims=True))
        Rl = N.photo_loss_at(R, d["chi"][loc])[0, :, 1]
        step = N.DX_FLUX * N.shiva.dl01_nh(N.nc_of(d["a_cm"][0]))
        print(f"  loc {loc}: k1 {k1:.3e}, k2/k1 {k2 / k1:.3f} | ours R(normal)/step {Rl[2] / step:.3e}"
              f" (theirs/ours {k1 / (Rl[2] / step):.2f}), R(mid)/R(normal) {Rl[1] / Rl[2]:.3f}")

    ours = N.run_model_b(d, prof, R, loss_scale=args.loss_scale)
    print(f"\nLast output (1e5 yr), f-weighted X_H bin fractions, ours | Figshare  (T(E) = {args.temperature},"
          f" loss x {args.loss_scale}):")
    for r, s, o, p in summary(ours, d["n"], d["f"], reg):
        if s == 0:
            print(f" {r}")
        print(f"   {d['a_cm'][s] * 1e8:5.1f} A  " + " ".join(f"{v:4.2f}" for v in o) + "  |  "
              + " ".join(f"{v:4.2f}" for v in p) + f"   d<x> {((o - p) * N.X_BINS).sum():+.2f}")

    # Fig. 8 quantity: <X_H/X_H0> weighted by n f over Z and sizes
    n0 = d["n"][:, :, 0].sum(axis=2)                      # loc, Z, size: total per population
    w = d["f"][:, :, 0, 0, :] * n0
    ours_x = (ours[:, :, -1] * N.X_BINS[None, None, :, None] * w[:, :, None, :]).sum(axis=(1, 2, 3)) / w.sum(axis=(1, 2))
    th = fractions(d["n"][:, :, -1])
    their_x = (th * N.X_BINS[None, None, :, None] * w[:, :, None, :]).sum(axis=(1, 2, 3)) / w.sum(axis=(1, 2))
    print("\n<X_H/X_H0>(distance), n f-weighted (Fig. 8 quantity): distance ours | Figshare")
    for loc in range(0, len(prof), 8):
        print(f"   {prof.distance_arcsec[loc]:6.1f}  {reg[loc]:6s} {ours_x[loc]:.3f} | {their_x[loc]:.3f}")
    for r in REGIONS:
        m = reg == r
        print(f"  {r:6s} median d<x> {np.median(ours_x[m] - their_x[m]):+.3f}, max |d<x>| {np.abs(ours_x[m] - their_x[m]).max():.3f}")


if __name__ == "__main__":
    main()
