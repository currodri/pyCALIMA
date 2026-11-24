#!/usr/bin/env python3
"""Timescales of the processes that move carbon between the bins of the carbon grid (carbon-transfer study of
2026-11-23, check 2; results/carbon_plan/carbon_transfer_2026-11-23/).

The carbon grid is ``gsd.carbon.CarbonLayout`` (4 molecular bins N_C 20-400, then 7 a-C(:H) grain bins to 1 um,
rho 1.6), populated with THEMIS J13 carbon (DustEM GRAIN_J13 shapes, all at rho 1.6). For each environment:

* collisions (``gsd.collisions._terms`` with ``GrainCollisions`` on 'themis_ac', i.e. graphite strength as the a-C proxy,
  KT10, fragments a^-3.3 down to the grid's 0.39 nm, below it to the sink): the time to refill each bin's mass from
  the other bins (supply) and to empty it into the others (loss), at the J13 state;
* PAH carbon loss for the molecular bins (pahchem hydrogen tables, deep chain cap +8:
  results/carbon_plan/hydrogen_tables/deep_k8/<bin>_carbon.dat): the time to lose the molecule's carbon;
* the a-C(:H) hydrogen relaxation for the covalent bins (``gsd.hydrogen.BinHydrogen``, laboratory laws), against the
  Hirashita & Murga (2020) Eq. 21 aromatisation time (coefficients as extracted in lit_identity.md; to verify);
* classical sputtering (``gsd.sputtering.erosion_rate``, graphite yields, rho 1.6): a / |da/dt| (labelled invalid for
  N_C < ~1000, Bocchio+12; Micelotta+10b molecular rates are not in the code).

Environments: WIM, WNM, DC as in check_bin_convergence.PHASES (HY09 turbulence); CNM with T = 100 K, n_H = 30 and
Larson-law turbulence (V = 1.1 km/s (L/pc)^0.38 at L = 10 pc, i.e. 2.6 km/s; a labelled choice, the code has no CNM
collision phase); PDR (seam cell: T 300 K, n_H 1e4, G0 1e3) and HOT (3e6 K, n_H 1) for photolysis and sputtering.

    python diagnostics/gsd/check_carbon_transfer_timescales.py --out results/carbon_plan/carbon_transfer_2026-11-23/timescales.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

YR = 3.15576e7
PC = 3.0857e18
AMU = 1.66053906660e-24
M_C = 12.0107 * AMU
M_HATOM = 1.00794 * AMU
TABLES = Path("results/carbon_plan/hydrogen_tables/deep_k8")
MOL_LABELS = ("bin_nc20-42", "bin_nc42-89", "bin_nc89-189", "bin_nc189-400")

# T [K], n_H [cm^-3], turbulence L [pc] and V [km/s] (0: none), G0 [Habing], x_e, n(H I) [cm^-3]
ENVS = {
    "WIM": dict(T=8000.0, n_H=0.1, L=100.0, V=20.0, G0=1.0, xe=1.0, n_HI=0.01),
    "WNM": dict(T=6000.0, n_H=0.3, L=100.0, V=20.0, G0=1.0, xe=0.02, n_HI=0.3),
    "CNM": dict(T=100.0, n_H=30.0, L=10.0, V=2.6, G0=1.0, xe=1e-3, n_HI=30.0),
    "DC": dict(T=10.0, n_H=1e4, L=1.0, V=1.5, G0=1e-2, xe=1e-7, n_HI=1.0),
    "PDR": dict(T=300.0, n_H=1e4, L=0.0, V=0.0, G0=1e3, xe=1e-4, n_HI=1e4),
    "HOT": dict(T=3e6, n_H=1.0, L=0.0, V=0.0, G0=1.0, xe=1.2, n_HI=1e-6),
}
COLLIDE = ("WIM", "WNM", "CNM", "DC")
COAG = {"WIM": False, "WNM": True, "CNM": True, "DC": True}


def j13_dnda(a):
    """THEMIS J13 carbon number per H atom per cm of radius, both components at rho 1.6 (shapes and M_dust/M_H of
    GRAIN_J13)."""
    def plaw_ed(x):
        f = x ** -5.0
        return np.where(x > 10e-7, f * np.exp(-((x - 10e-7) / 50e-7)), f)

    def logn(x):
        return np.exp(-0.5 * np.log(x / 7e-7) ** 2) / x           # dn/dln a lognormal (as check_carbon_heating_share)

    out = np.zeros_like(a, dtype=float)
    for shape, mtot, lo in ((plaw_ed, 1.70e-3, 0.4e-7), (logn, 6.3e-4, 0.5e-7)):
        x = np.geomspace(lo, 4900e-7, 6000)
        m = 4.0 / 3.0 * np.pi * 1.6 * x ** 3
        norm = mtot * M_HATOM / np.trapezoid(m * shape(x), x)     # DustEM gives M_dust / M_H
        out += np.where((a >= lo) & (a <= 4900e-7), norm * shape(a), 0.0)
    return out


def bin_state(grid):
    """J13 (N, M) per H in each bin of the grid."""
    N, M = np.zeros(grid.n_bins), np.zeros(grid.n_bins)
    for i in range(grid.n_bins):
        x = np.geomspace(grid.a_edges[i], grid.a_edges[i + 1], 400)
        f = j13_dnda(x)
        N[i] = np.trapezoid(f, x)
        M[i] = np.trapezoid(4.0 / 3.0 * np.pi * grid.rho * x ** 3 * f, x)
    return N, M


def collision_times(grid, N, M, env):
    from pycalima.gsd.collisions import _terms
    from pycalima.gsd.materials import get_material
    from pycalima.gsd.shattering import GrainCollisions
    from pycalima.gsd.velocities import RelativeVelocity

    e = ENVS[env]
    rv = RelativeVelocity(T=e["T"], n_H=e["n_H"], mu=1.4, v_turb=e["V"] * 1e5, L=e["L"] * PC)
    model = GrainCollisions(get_material("themis_ac").shatter, rv, law="KT10", coagulation=COAG[env],
                            a_frag_min=float(grid.a_edges[0]), collector="remove", n_speed=8, table_ppd=96,
                            threads=1)
    n = e["n_H"]
    F, L, H = _terms(grid, N * n, M * n, model)
    nb = grid.n_bins
    Fb = F.copy()
    Fb[np.arange(F.shape[0]), np.arange(F.shape[0])] = 0.0
    inflow = Fb[:nb, :nb].sum(axis=1)                   # g cm^-3 s^-1 from the other bins
    outflow = Fb[:, :nb].sum(axis=0)                    # to other bins, overflow and sink
    with np.errstate(divide="ignore"):
        t_in = np.where(inflow > 0, M * n / inflow, np.inf) / YR
        t_out = np.where(outflow > 0, M * n / outflow, np.inf) / YR
    return t_in, t_out


def freeze_out_times(grid, N, M, env, n_mol=4, eps=1e-6):
    """Loss time of each molecular bin by collisions with the grain bins only (molecule + molecule excluded): the
    bin is a trace population (eps of its J13 state), every other molecular bin is empty, the grains are at J13, so
    collisions among molecules (~ eps^2) are negligible against molecule-grain ones (~ eps)."""
    from pycalima.gsd.collisions import _terms
    from pycalima.gsd.materials import get_material
    from pycalima.gsd.shattering import GrainCollisions
    from pycalima.gsd.velocities import RelativeVelocity

    e = ENVS[env]
    rv = RelativeVelocity(T=e["T"], n_H=e["n_H"], mu=1.4, v_turb=e["V"] * 1e5, L=e["L"] * PC)
    model = GrainCollisions(get_material("themis_ac").shatter, rv, law="KT10", coagulation=COAG[env],
                            a_frag_min=float(grid.a_edges[0]), collector="remove", n_speed=8, table_ppd=96,
                            threads=1)
    n = e["n_H"]
    out = []
    for i in range(n_mol):
        Nt, Mt = N.copy(), M.copy()
        Nt[:n_mol], Mt[:n_mol] = 0.0, 0.0
        Nt[i], Mt[i] = eps * N[i], eps * M[i]
        F, _, _ = _terms(grid, Nt * n, Mt * n, model)
        Fb = F.copy()
        Fb[np.arange(F.shape[0]), np.arange(F.shape[0])] = 0.0
        outflow = Fb[:, i].sum()
        out.append(Mt[i] * n / outflow / YR if outflow > 0 else np.inf)
    return out


def photolysis_times(env, layout):
    from pycalima.solvers.table_io import build_pah_photolysis_interpolator_nd

    e = ENVS[env]
    out = []
    for lab in MOL_LABELS:
        f, _ = build_pah_photolysis_interpolator_nd(TABLES / f"{lab}_carbon.dat")
        nc = float(np.load(TABLES / f"{lab}.npz", allow_pickle=True)["nc_mean"])
        v = f(np.log10(e["G0"]), np.log10(e["n_H"]), np.log10(min(e["T"], 1e4)), np.log10(e["xe"]))
        rate_c = 2.0 * 10.0 ** v                         # C atoms lost per molecule per s (table: log10(rate / 2))
        out.append(nc / rate_c / YR if rate_c > 0 else np.inf)
    return out


def hydrogen_times(grid, env, covalent):
    from pycalima.gsd.hydrogen import BinHydrogen, Conditions

    e = ENVS[env]
    cond = Conditions(u_mathis=e["G0"] / 1.14, n_hi=e["n_HI"], t_gas=min(e["T"], 1e4))
    out = []
    for i in covalent:
        bh = BinHydrogen(grid, i)
        x = bh.equilibrium(cond)
        out.append(bh.relaxation_time(x, cond) / YR)
    return out


def hm20_tau(a_cm, G0):
    a = a_cm * 1e4                                       # um
    return (3.0 * a ** -2 + 6.6e7 * a) / (G0 / 1.14)     # yr, U in Mathis units (HM20 Eq. 21 as extracted)


def sputter_times(grid, env):
    from pycalima.gsd.sputtering import erosion_rate

    e = ENVS[env]
    adot = erosion_rate("graphite", e["T"], e["n_H"], rho=grid.rho)
    a = grid.a_centres
    return np.abs(a / adot(a)) / YR


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="results/carbon_plan/carbon_transfer_2026-11-23/timescales.txt")
    args = ap.parse_args(argv)
    from pycalima.gsd.carbon import CarbonLayout

    layout = CarbonLayout()
    grid = layout.grid()
    N, M = bin_state(grid)
    nb = grid.n_bins
    a_c = grid.a_centres
    nc_c = 4.0 / 3.0 * np.pi * grid.rho * a_c ** 3 / M_C
    covalent = [i for i in range(nb) if a_c[i] > grid.a_edges[4] and a_c[i] < 53.1e-7]
    lines = ["Carbon grid (CarbonLayout, rho 1.6) with THEMIS J13 carbon; times in yr",
             "bin  a_lo-a_hi [nm]  N_C(centre)  mass share"]
    for i in range(nb):
        lines.append(f"{i + 1:3d}  {grid.a_edges[i] * 1e7:7.2f}-{grid.a_edges[i + 1] * 1e7:8.2f}  {nc_c[i]:10.3g}"
                     f"  {M[i] / M.sum() * 100:6.2f} %")
    lines.append("")
    lines.append("collisions (graphite strength, KT10, a^-3.3 to 0.39 nm): supply time M_i / inflow from the other bins, "
                 "loss time M_i / outflow (to other bins, overflow, sink); J13 state")
    for env in COLLIDE:
        t_in, t_out = collision_times(grid, N, M, env)
        lines.append(f"  {env:4s} supply " + " ".join(f"{t:8.1e}" for t in t_in))
        lines.append(f"  {env:4s} loss   " + " ".join(f"{t:8.1e}" for t in t_out))
    lines.append("molecular bins 1-4 lost to the grain bins only (trace molecules, no molecule + molecule collisions):")
    for env in COLLIDE:
        t = freeze_out_times(grid, N, M, env)
        lines.append(f"  {env:4s} freeze " + " ".join(f"{x:8.1e}" for x in t))
    lines.append("")
    lines.append("PAH carbon loss (pahchem tables, deep +8): time to lose the molecule's carbon, bins 1-4")
    for env in ENVS:
        t = photolysis_times(env, layout)
        lines.append(f"  {env:4s} " + " ".join(f"{x:8.1e}" for x in t))
    lines.append("")
    lines.append(f"a-C(:H) hydrogen relaxation 1/(k_UV + k_H) (lab laws, gsd.hydrogen), covalent bins "
                 f"{[i + 1 for i in covalent]}; HM20 Eq. 21 aromatisation in brackets")
    for env in ENVS:
        if env == "HOT":
            continue
        t = hydrogen_times(grid, env, covalent)
        h = [hm20_tau(a_c[i], ENVS[env]["G0"]) for i in covalent]
        lines.append(f"  {env:4s} " + " ".join(f"{x:8.1e} ({y:7.1e})" for x, y in zip(t, h)))
    lines.append("")
    lines.append("classical sputtering a/|da/dt| (graphite yields; invalid for N_C < ~1000, i.e. bins 1-5)")
    for env in ("WIM", "HOT"):
        t = sputter_times(grid, env)
        lines.append(f"  {env:4s} " + " ".join(f"{x:8.1e}" for x in t))
    lines.append("")
    lines.append("literature (not computed): Micelotta+10b PAH lifetime ~1e3 yr at n_H 0.01, 1e7 K (scales as 1/n); "
                 "Jones & Ysard 2025 a-C photo-fragmentation 1e6-1e7 yr for 40-60 C in the diffuse ISM, 1e2-1e4 times "
                 "shorter in PDRs, > 150 C stable; freeze-out 0.3-0.5 Myr dense clouds, 1e8 yr CNM (Seok+14)")
    txt = "\n".join(lines)
    print(txt)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(txt + "\n")


if __name__ == "__main__":
    main()
