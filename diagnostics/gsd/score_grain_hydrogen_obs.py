"""Grain hydrogen scored like the PAH tables: equilibrium against the truth on observables, by path budget.

``check_hydrogen_tracking.py`` judged the covalent grains on their state (band gap) and on ln kappa at single
wavelengths; the PAH bins were judged on observables, by the factor error max(E/T, T/E) - 1 of the worst cell and of
the path integral (``check_pah_bins_hydrogen``). This script applies the PAH metric to the grains, on the same paths
as ``check_hydrogen_tracking`` (Murga+20 Orion Bar and Horsehead at 0.3, 1, 3 km/s in and out; CNM <-> WNM cycles of
1e4, 1e5, 1e6 yr), for the per-gram opacity of bins 4 (1.06-2.82 nm) and 7 (20-53 nm) at

* 0.15 um: FUV absorption, which drives the photoelectric heating and the grain's own UV processing;
* 0.55 um: the visible extinction (A_V);
* 3.4 um: the aliphatic band region;
* 100 um: the far-IR opacity, which sets the thermal emission.

Path integrals are over time along the path; the worst cell is over the whole path.

    python diagnostics/gsd/score_grain_hydrogen_obs.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_hydrogen_tracking as H   # noqa: E402

from pycalima.gsd.carbon import CarbonLayout   # noqa: E402
from pycalima.gsd.hydrogen import YR, BinHydrogen   # noqa: E402
from pycalima.gsd.optics import lookup_x   # noqa: E402

LAMS_UM = (0.15, 0.55, 3.4, 100.0)


def tables(path, bins):
    d = np.load(path)
    lam = d["lam_um"]
    il = [int(np.argmin(abs(lam - p))) for p in LAMS_UM]
    return {b: (d[f"bin{b}_s"], d[f"bin{b}_c"], d[f"bin{b}_kappa"][:, :, il, 0]) for b in bins}


def kappa(tab, e_gap, s=0.5):
    s_t, e_t, k = tab
    return np.array([lookup_x(s_t, e_t, k, s, float(e)) for e in np.atleast_1d(e_gap)]).reshape(-1, len(LAMS_UM))


def score(bh, tab, t_end, cond, x0=None, n_out=400):
    """(worst cell, path budget) factor errors of E against the truth, per wavelength."""
    t = np.linspace(0.0, t_end, n_out)
    x0 = bh.equilibrium(cond(0.0)) if x0 is None else x0
    k_true = kappa(tab, bh.e_gap(H._solve(bh, t_end, cond, x0, t)))
    k_eq = kappa(tab, bh.e_gap(np.array([bh.equilibrium(cond(tt)) for tt in t])))
    ratio = k_eq / k_true
    cell = np.max(np.maximum(ratio, 1.0 / ratio), axis=0) - 1.0
    budget = np.trapezoid(k_eq, t, axis=0) / np.trapezoid(k_true, t, axis=0)
    return cell, np.maximum(budget, 1.0 / budget) - 1.0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--murga", default=str(Path.home() / "Library/Caches/calima/datasets/pahchem/murga2020"))
    ap.add_argument("--tables", default="results/carbon_plan/hydrogenated_tables.npz")
    ap.add_argument("--out", default="results/gsd_overview/grain_hydrogen_obs.txt")
    args = ap.parse_args(argv)
    grid = CarbonLayout().grid()
    bins = (4, 7)
    tabs = tables(args.tables, bins)
    paths = []
    for site in ("orion_isobaric", "horsehead"):
        for v in H.SPEEDS_KMS:
            for inward in (True, False):
                t_end, cond = H.profile_path(site, args.murga, v, inward)
                paths.append((f"{site} {'in' if inward else 'out'} {v:g} km/s", t_end, cond, not inward))
    for period in (1e4, 1e5, 1e6):
        t_end, cond = H.cycle_path(period)
        paths.append((f"CNM<->WNM {period:.0e} yr", t_end, cond, False))
    lines = ["Grain hydrogen, equilibrium against the truth on per-gram opacity: factor error max(E/T, T/E) - 1, "
             "worst cell | path budget. Outward paths start at the H plateau (cloud material).",
             f"{'path':30s} " + " ".join(f"{p:>15g} um" for p in LAMS_UM)]
    for b in bins:
        bh = BinHydrogen(grid, b)
        lines.append(f"\nbin {b}: {grid.a_edges[b] * 1e7:.2f}-{grid.a_edges[b + 1] * 1e7:.1f} nm")
        for name, t_end, cond, out in paths:
            cell, budget = score(bh, tabs[b], t_end, cond, bh.x_max if out else None)
            lines.append(f"{name:30s} " + " ".join(f"{c:8.2f} | {g:6.2f}" for c, g in zip(cell, budget)))
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
