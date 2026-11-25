"""Per-bin optics of the covalent carbon bins with tracked hydrogen (rung M1, step 1b).

The bins from N_C,max (400 C, 1.06 nm) to a_H,max (53 nm) carry M_H next to M_C, so
their tables depend on the in-bin state s and on x = H atoms per C atom
(``gsd.carbon.HydrogenatedCarbon``: at each radius x sets the bulk band gap through
bulk plus surface hydrogen, and the gap the Jones (2012) optics, effective-gap size
interpolation). This builds kappa(lambda, s, x) per gram of carbon with the A2 settings
(128 nodes per bin, s and x grids refined to tol; ``gsd.optics.adaptive_kappa_tables_x``)
and prints what follows. ``--refine lambda`` (default) meets tol at every wavelength, so
the tables serve any RT groups RAMSES averages them over. ``--refine groups`` refines
until only the quantities RAMSES derives meet tol: the group means of a reference group set (``--groups``, eV edges;
default the 8 CALIMA RTZ groups) for 1e4-1e5 K blackbodies, and the Planck means at
3-3000 K. The tables keep all 241 wavelengths, so RAMSES can average them over any
groups (it needed more band-gap points, not fewer, in the tests).

- per bin, the s- and x-grid sizes and the x range (a-C floor to a-C:H saturation);
- kappa_abs at s = 0.5 at a few wavelengths, at the floor, the middle and saturation;
- the end members against the fixed-composition carriers of the same bins
  (``aromatic``: DustEM Q_CM20 below 20 nm, CM20 core/mantle above; ``aliphatic``:
  Jones E_g = 2.5 eV);
- the table size per bin for RAMSES's 8 groups (3 quantities, 8 bytes).

``--coordinate e_gap`` (default) indexes the tables by the bulk band gap the bin's
grains share, starting from the band gaps of the Jones (2012) tables, and RAMSES turns
M_H / M_C into it (``gsd.carbon.bin_h_per_c``); ``h_per_c`` indexes them by the H/C
itself, where each in-bin grain reaches its own end members, and crosses each tabulated
band gap, at a different H/C, and the kinks make the grids ~4x larger in each axis.
The tables are saved to ``--save`` (npz) as soon as they are built; the comparison with
the fixed-composition carriers needs DustEM ($CALIMA_DUSTEM_DIR) and is skipped without it.

    python diagnostics/gsd/check_hydrogenated_tables.py [--tol 3e-3] [--coordinate h_fraction]
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from pycalima.gsd.carbon import CARRIERS, M_C, CarbonLayout, HydrogenatedCarbon
from pycalima.gsd.grid import MassGrid
from pycalima.gsd.optics import adaptive_kappa_tables, adaptive_kappa_tables_x, group_reducer, lookup, lookup_x
from pycalima.models.dust_charge.rt_group_charging import DEFAULT_GROUP_L0, DEFAULT_GROUP_L1
from pycalima.models.dust_radiation import jones2012_optics as jo

LAM_UM = np.logspace(-3, 3, 241)          # the RAMSES optics grid
PROBE_UM = (0.1, 0.2175, 0.55, 3.4, 11.3, 100.0)
N_NODES = 128
A_H_MAX_CM = 53.1e-7                      # the default a_H,max bin edge


def covalent_bins(layout):
    grid = layout.grid()
    nc = grid.m_edges / M_C
    lo = np.nonzero(np.isclose(nc, layout.nc_pah_max))[0][0]
    hi = np.nonzero(grid.a_edges <= A_H_MAX_CM * (1 + 1e-3))[0][-1]
    return grid, MassGrid(grid.m_edges[lo:hi + 1], grid.rho), np.arange(lo, hi)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tol", type=float, default=3e-3)
    ap.add_argument("--refine", choices=("groups", "lambda"), default="lambda")
    ap.add_argument("--groups", type=float, nargs="+", default=None,
                    help="group edges [eV], ascending (default: the 8 CALIMA groups)")
    ap.add_argument("--coordinate", choices=("e_gap", "h_per_c"), default="e_gap")
    ap.add_argument("--nodes", type=int, default=N_NODES, help="in-bin nodes per bin")
    ap.add_argument("--save", default="results/carbon_plan/hydrogenated_tables.npz")
    args = ap.parse_args(argv)
    if args.groups is None:
        groups = list(zip(DEFAULT_GROUP_L0, DEFAULT_GROUP_L1))
    else:
        groups = list(zip(args.groups[:-1], args.groups[1:]))
    reduce = group_reducer(LAM_UM, groups) if args.refine == "groups" else None
    print(f"refine on {args.refine}" + (f": {len(groups)} groups {groups}" if reduce else ""))

    grid, sub, bins = covalent_bins(CarbonLayout())
    hc = HydrogenatedCarbon()
    if args.coordinate == "h_per_c":
        x_range = [hc.h_range(sub.m_edges[i], sub.m_edges[i + 1]) for i in range(sub.n_bins)]
        x_start = None
    else:
        x_range = [hc.e_range] * sub.n_bins
        cols = [e for e in jo.E_GAPS if hc.e_range[0] - 1e-9 <= e <= hc.e_range[1] + 1e-9]
        x_start = lambda i: cols                          # noqa: E731
    t0 = time.time()
    tab_x = adaptive_kappa_tables_x(sub, args.nodes, hc.per_gram_on(sub, LAM_UM, args.coordinate),
                                   lambda i: x_range[i], args.tol, reduce=reduce, x_start=x_start)
    print(f"tables built in {time.time() - t0:.0f} s (tol {args.tol}, coordinate {args.coordinate})", flush=True)
    np.savez(args.save, lam_um=LAM_UM, coordinate=args.coordinate, m_edges=sub.m_edges,
             **{f"bin{bins[i]}_{k}": v for i, t in enumerate(tab_x) for k, v in zip(("s", "c", "kappa"), t)})
    print(f"saved {args.save}")
    ip = [int(np.argmin(abs(LAM_UM - p))) for p in PROBE_UM]

    print(f"\n{'bin':>3s} {'a [nm]':>13s} {'N_C':>17s} {'n_s':>4s} {'n_c':>4s} {'coord. range':>14s} "
          f"{'group table':>14s} {'per-lambda file':>16s}")
    for i, (s, x, tab) in enumerate(tab_x):
        kb = s.size * x.size * len(groups) * 3 * 8
        print(f"{bins[i]:3d} {sub.a_edges[i] * 1e7:5.2f}-{sub.a_edges[i + 1] * 1e7:6.2f} "
              f"{sub.m_edges[i] / M_C:8.3g}-{sub.m_edges[i + 1] / M_C:8.3g} {s.size:4d} {x.size:4d} "
              f"{x[0]:6.3f}-{x[-1]:6.3f} {kb / 1024:11.1f} kB {tab.nbytes / 2**20:13.1f} MB", flush=True)

    print("\nkappa_abs per gram of carbon [cm^2/g] at s = 0.5:")
    print(f"{'bin':>3s} {'coord':>7s} " + " ".join(f"{p:>9.4g}um" for p in PROBE_UM))
    for i, (s, x, tab) in enumerate(tab_x):
        for x0 in (x[0], 0.5 * (x[0] + x[-1]), x[-1]):
            k = lookup_x(s, x, tab, 0.5, x0)[:, 0]
            print(f"{bins[i]:3d} {x0:7.3f} " + " ".join(f"{k[j]:11.4g}" for j in ip))

    try:
        fixed = {n: adaptive_kappa_tables(sub, args.nodes, CARRIERS[n].per_gram_on(sub, LAM_UM), args.tol)
                 for n in ("aromatic", "aliphatic")}
    except Exception as exc:                     # noqa: BLE001 - DustEM missing: skip the comparison
        print(f"\nskipping the comparison with the fixed carriers: {type(exc).__name__}")
        return
    print("\nend members / fixed-composition carriers at s = 0.5 (floor / aromatic, saturation / aliphatic):")
    print(f"{'bin':>3s} {'':>11s} " + " ".join(f"{p:>9.4g}um" for p in PROBE_UM))
    for i, (s, x, tab) in enumerate(tab_x):
        for x0, name in ((x[0], "aromatic"), (x[-1], "aliphatic")):
            sf, tf = fixed[name][i]
            r = lookup_x(s, x, tab, 0.5, x0)[:, 0] / lookup(sf, tf, 0.5)[0, :, 0]
            print(f"{bins[i]:3d} {name:>11s} " + " ".join(f"{r[j]:11.3f}" for j in ip))


if __name__ == "__main__":
    main()
