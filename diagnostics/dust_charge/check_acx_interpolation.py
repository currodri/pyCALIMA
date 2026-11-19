"""Charging of a-C(:H) with tracked hydrogen: direct tables against interpolated ones.

The covalent bins carrying M_H (1-53 nm) need charging tables along x = H/C. Option 2
of the plan computes them directly for the intermediate material ``acx<E_g>``
(``wdb06_yields``; labelled assumptions for IE and electron affinity between the 'ac'
and 'ach' ends) at a few band-gap nodes, and RAMSES interpolates linearly in x between
the nearest two. Option 1 keeps only the two end tables (E_g = 0.1 and 2.5 eV). In both,
only the tabulated photon and hot-gas arrays are interpolated
(``rt_group_charging.interpolate_bin_tables``); IP, EA and the charge limits stay those
of the intermediate material.

For grains of 1.5, 4, 12 and 30 nm and band gaps between the nodes, this solves the
charge distribution in the four cells of the PAH/grain seam check (WNM, CNM, PDR,
dense; ISRF only) with the direct table and with each node set, and prints:

- the direct <Z>, sigma_Z, photoelectric heating Gamma and recombination cooling
  Lambda across the band-gap range (how much the material matters);
- the worst error of each node set over the test band gaps and cells.

    python diagnostics/dust_charge/check_acx_interpolation.py
"""
from __future__ import annotations

import argparse

import numpy as np

from pycalima.models.dust_charge import rt_group_charging as rg
from pycalima.models.dust_charge import wdb06_yields as wy
from pycalima.models.dust_radiation import jones2012_optics as jo

AMU = 1.66053906660e-24
CELLS = {"WNM": (8000.0, 0.4, 7.2e-3, "H+", 1.0),
         "CNM": (100.0, 30.0, 3e-2, "C+", 1.0),
         "PDR": (300.0, 1e4, 1.0, "C+", 1e3),
         "dense": (30.0, 1e3, 1e-3, "C+", 1e-2)}
ION_MASS = {"H+": 1.00794, "C+": 12.0107}
RADII_NM = (1.5, 4.0, 12.0, 30.0)
E_TEST = (0.25, 0.75, 1.25, 1.75, 2.25)
NODE_SETS = {"ends (option 1)": (0.1, 2.5), "3 nodes": (0.1, 1.3, 2.5), "5 nodes": (0.1, 0.7, 1.3, 1.9, 2.5)}


def solve(tab):
    gt = rg.GroupChargingTable(tab)
    nG = np.asarray(tab["L0"]).size
    out = {}
    for name, (T, n_h, n_e, ion, G0) in CELLS.items():
        r = gt.solve(np.zeros(nG), rg.C_CGS, T, n_e, [(n_e, 1.0, ION_MASS[ion] * AMU)], G0_bg=G0, recomb=None)
        out[name] = np.array([r["Zmean"], r["Zsigma"], r["Gamma"], r["Lambda"]])
    return out


def interpolated(tables, nodes, e, a_cm):
    """Table at band gap e from the two nodes around it, linear in H/C at radius a."""
    j = int(np.clip(np.searchsorted(nodes, e) - 1, 0, len(nodes) - 2))
    x0, x1, x = (float(jo.h_per_carbon(a_cm, v)) for v in (nodes[j], nodes[j + 1], e))
    return rg.interpolate_bin_tables(tables[nodes[j]], tables[nodes[j + 1]], (x - x0) / (x1 - x0), wy.acx_label(e))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.parse_args(argv)
    e_all = sorted(set(E_TEST) | {e for n in NODE_SETS.values() for e in n})
    print("columns: <Z>, sigma_Z, Gamma [erg/s per grain], Lambda [erg/s per grain]")
    for a_nm in RADII_NM:
        a = a_nm * 1e-7
        tables = {e: rg.build_bin_table(wy.acx_label(e), a) for e in e_all}
        direct = {e: solve(tables[e]) for e in e_all}
        print(f"\n=== a = {a_nm} nm; H/C from {float(jo.h_per_carbon(a, 0.1)):.3f} (E_g 0.1) "
              f"to {float(jo.h_per_carbon(a, 2.5)):.3f} (E_g 2.5)")
        for cell in CELLS:
            lo, hi = direct[0.1][cell], direct[2.5][cell]
            print(f"  {cell:5s} direct E_g 0.1: <Z> {lo[0]:8.3f} s {lo[1]:6.3f} G {lo[2]:10.3e} L {lo[3]:10.3e}"
                  f" | E_g 2.5: <Z> {hi[0]:8.3f} s {hi[1]:6.3f} G {hi[2]:10.3e} L {hi[3]:10.3e}")
        for name, nodes in NODE_SETS.items():
            dz = ds = dg = dl = 0.0
            for e in E_TEST:
                got = solve(interpolated(tables, nodes, e, a))
                for cell in CELLS:
                    d, g = direct[e][cell], got[cell]
                    dz, ds = max(dz, abs(g[0] - d[0])), max(ds, abs(g[1] - d[1]))
                    dg = max(dg, abs(g[2] / d[2] - 1) if d[2] > 0 else 0.0)
                    dl = max(dl, abs(g[3] / d[3] - 1) if d[3] > 0 else 0.0)
            print(f"  {name:16s} worst over E_g {E_TEST} and cells: |d<Z>| {dz:.3f}  |d sigma| {ds:.3f}  "
                  f"Gamma {100 * dg:5.1f}%  Lambda {100 * dl:5.1f}%")


if __name__ == "__main__":
    main()
