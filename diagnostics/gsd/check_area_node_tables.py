#!/usr/bin/env python3
"""How many radii per bin the area-node charging needs: blended against direct charge tables (A3c, phase 3).

RAMSES solves each bin's charge once, at its area node a*, which moves inside the bin with the in-bin state; the
charge kernels must then be known at any radius. ``gsd.charging.AreaNodeCharging`` blends them in ln a from tables at
n radii per bin, edges included (``blend_table``). For every grain bin of the carbon grid ('ac' with the seam's
sticking law) and of the silicate grid (0.45 nm - 1 um, 8 bins), this script
- draws radii inside the bin (uniform in ln a, seeded);
- solves the charge there with the blended table, for n = 2, 3 and 5 radii per bin, and with a table built
  directly at that radius;
- does so in the four cells of check_inbin_rates (Mathis background at the cell's G0, no RT photons; LG5 solve);
- reports the worst relative error of Gamma and of Lambda (with Lambda_auto), and the worst error of <Z> in units
  of 0.05 + 0.01 |<Z>| charges (absolute for small grains, relative for large ones, whose <Z> reaches hundreds).

Target: 1 % on Gamma and Lambda, a <Z> score of at most 1. Each bin gets the fewest radii that meet it. The table
memory per bin is n times one table's arrays (the kernels K, H on 12 groups x 31 T_s x 801 potentials, real*8
here; RAMSES plans 8 groups, ~11 T_s, a cut potential range and real*4, about 0.35 MiB per table).

    python diagnostics/gsd/check_area_node_tables.py [--n-random 6] [--out results/gsd_a3c/area_node_tables.txt]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

C_CGS = 2.99792458e10
AMU = 1.66053906660e-24
#: name: (T [K], n_e [cm^-3], ion carrying the electrons, G0 [Habing]); the cells of check_inbin_rates
CELLS = {"WNM": (8000.0, 7.2e-3, 1.00794, 1.0), "CNM": (100.0, 3e-2, 12.0107, 1.0),
         "PDR": (300.0, 1.0, 12.0107, 1e3), "dense": (30.0, 1e-3, 12.0107, 1e-2)}
RADII = (2, 3, 5, 9)
TARGET = {"Gamma": 0.01, "Lambda": 0.01, "Z": 1.0}


def populations():
    """(label, grid, GSD material, sticking law, bins tested): the carbon grain bins and the silicate bins."""
    from pycalima.gsd.carbon import CarbonLayout
    from pycalima.gsd.grid import MassGrid
    from pycalima.gsd.materials import get_material

    lay = CarbonLayout()
    cg = lay.grid()
    sil = get_material("silicate")
    m = sil.rho * 4.0 / 3.0 * np.pi * np.array([4.5e-8, 1e-4]) ** 3
    sg = MassGrid(np.geomspace(m[0], m[1], 9), sil.rho)
    return [("carbon (ac, seam sticking)", cg, lay.material, "seam", np.flatnonzero(~lay.pah_bins(cg))),
            ("silicate", sg, "silicate", "wd01", np.arange(sg.n_bins))]


def solve_cells(tab, photons):
    """{cell: (Gamma, Lambda + Lambda_auto, <Z>)} of one table."""
    from pycalima.models.dust_charge import rt_group_charging as rg

    gt = rg.GroupChargingTable(tab, egy=photons.egy)
    out = {}
    for cell, (T, n_e, m_ion, G0) in CELLS.items():
        r = gt.solve(photons.N, C_CGS, T, n_e, [(n_e, 1.0, m_ion * AMU)], G0_bg=G0, recomb=None, lattice=True)
        out[cell] = (r["Gamma"], r["Lambda"] + r["Lambda_auto"], r["Zmean"])
    return out


def meets(err):
    return err[0] <= TARGET["Gamma"] and err[1] <= TARGET["Lambda"] and err[2] <= TARGET["Z"]


def table_bytes(tab):
    return sum(np.asarray(v).nbytes for v in tab.values() if isinstance(v, np.ndarray))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-random", type=int, default=6, help="radii drawn per bin")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="results/gsd_a3c/area_node_tables.txt")
    args = ap.parse_args(argv)
    from pycalima.gsd import charging as ch

    rng = np.random.default_rng(args.seed)
    lines = ["Area-node charge tables: blended (n radii per bin, edges included) against direct tables at random radii",
             f"inside each bin; {args.n_random} radii per bin; cells {', '.join(CELLS)} (Mathis background, LG5).",
             "Worst over radii and cells: |dGamma/Gamma|, |dLambda/Lambda| [%], |d<Z>| / (0.05 + 0.01 |<Z>|).",
             "Target 1 %, 1 %, 1."]
    summary = []
    for label, grid, material, sticking, bins in populations():
        rho = grid.rho
        models = {n: ch.AreaNodeCharging(material, grid, radii_per_bin=n, sticking=sticking, rho=rho) for n in RADII}
        L0, L1 = models[RADII[0]].L0, models[RADII[0]].L1
        photons = ch.PhotonGroups(L0, L1, np.zeros(L0.size), np.sqrt(L0 * L1))
        wdb = ch.charging_material(material)
        lines.append(f"\n=== {label}")
        lines.append("bin   a range [nm]   " + "  ".join(f"n={n}: Gamma Lambda  <Z>" for n in RADII) + "   pick")
        worst = {n: np.zeros(3) for n in RADII}
        picks = []
        nbytes = 0
        for i in bins:
            a_lo, a_hi = grid.a_edges[i], grid.a_edges[i + 1]
            radii = np.exp(rng.uniform(np.log(a_lo), np.log(a_hi), args.n_random))
            direct = []
            for r in radii:
                tab = ch.bin_table(wdb, float(r), L0, L1, cache=False, sticking=sticking, rho=rho)
                nbytes = max(nbytes, table_bytes(tab))
                direct.append(solve_cells(tab, photons))
            row = f"{i:3d}  {a_lo * 1e7:7.2f}-{a_hi * 1e7:7.2f}"
            errs = {}
            for n in RADII:
                err = np.zeros(3)
                for r, d in zip(radii, direct, strict=True):
                    b = solve_cells(models[n].table_at(i, float(r)), photons)
                    for cell in CELLS:
                        g0, l0, z0 = d[cell]
                        g1, l1, z1 = b[cell]
                        err = np.maximum(err, [abs(g1 / g0 - 1.0) if g0 > 0 else 0.0,
                                               abs(l1 / l0 - 1.0) if l0 > 0 else 0.0,
                                               abs(z1 - z0) / (0.05 + 0.01 * abs(z0))])
                worst[n] = np.maximum(worst[n], err)
                errs[n] = err
                row += f"  {100 * err[0]:9.3f} {100 * err[1]:6.3f} {err[2]:5.2f}"
            pick = next((n for n in RADII if meets(errs[n])), None)
            picks.append(pick)
            row += f"   {pick}"
            lines.append(row)
            print(row, flush=True)
        lines.append("worst                " + "".join(
            f"  {100 * worst[n][0]:9.3f} {100 * worst[n][1]:6.3f} {worst[n][2]:5.2f}" for n in RADII))
        tot = sum(p for p in picks if p is not None)
        lines.append(f"radii per bin meeting the target: {picks} ({tot} tables; one is {nbytes / 2 ** 20:.2f} MiB "
                     f"here, ~0.35 MiB in RAMSES's planned format)")
        summary.append((label, picks))
    txt = "\n".join(lines)
    print(txt)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(txt + "\n")


if __name__ == "__main__":
    main()
