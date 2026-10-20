#!/usr/bin/env python3
"""Stage-4 checks of the collapse model, written to ``_paths.get_plots_dir("collapse")``
as ``collapse_growth.png``:

1. the dust-free Omukai+05 tracks (T and f_H2), and our compressional heating
   against their Fig. 2a;
2. static dust: the cooling criterion C = Lambda_gd t_col / e_th along the Z = 0
   track for several dust populations at their critical D, against the
   Schneider+12a Eq. 16 constant S D_cr;
3. accretion: the dust-to-gas ratio of MgSiO3 (0.01 um, 10 % of Si condensed)
   for Z = 1e-4, 1e-5, 1e-6 Z_sun, against the Chiaki+13 growth densities;
4. Brownian coagulation of MRN silicates: S / S_0 vs n_H for Z = 1, 1e-2, 1e-4,
   against the Hirashita & Omukai (2009) critical densities.

    python diagnostics/collapse/check_collapse_growth.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pycalima import _paths
from pycalima.collapse.accretion import formula_mass, solar_abundances
from pycalima.collapse.grid import SizeGrid
from pycalima.collapse.integrator import (Population, brownian_kernel, critical_dust_to_gas,
                                          lognormal_population, run_collapse)
from pycalima.collapse.materials import get_material
from pycalima.collapse.tracks import Track
from pycalima.plotting_style import use_calima_style

RHO_PER_H = (1 + 4 * 0.0833) * 1.6735575e-24
COMPR = Path(__file__).resolve().parents[2] / "tests" / "data" / "omukai05_fig2a_compr.txt"


def main():
    use_calima_style()
    plt.rcParams["text.usetex"] = False
    z0 = Track.omukai2005("Z0")
    fig, ax = plt.subplots(2, 2, figsize=(13, 9.5))
    a1, a2, a3, a4 = ax.ravel()

    # 1. tracks and compression
    for key, ls, lab in (("Z0", "k-", "Z = 0"), ("m6", "b--", "[Z/H] = -6 (no dust)"),
                         ("m4", "g--", "[Z/H] = -4 (no dust)"), ("m2", "r--", "[Z/H] = -2 (no dust)")):
        tr = Track.omukai2005(key)
        a1.plot(10 ** tr.log_n, 10 ** tr.log_T, ls, lw=1, label=lab)
    a1b = a1.twinx()
    n = np.logspace(0, 16, 300)
    ln, lG = np.loadtxt(COMPR, unpack=True)
    a1b.plot(10 ** ln, 10 ** lG, "o", color="tab:orange", ms=3, mfc="none", label="Omukai+05 Fig. 2a compr")
    a1b.plot(n, z0.compression_heating(n) / z0.rho(n), "-", color="tab:orange", lw=1, label="ours")
    a1b.set(yscale="log", ylabel=r"$\Gamma_{\rm comp}/\rho$ [erg s$^{-1}$ g$^{-1}$]")
    a1.set(xscale="log", yscale="log", xlabel=r"$n_{\rm H}$ [cm$^{-3}$]", ylabel="T [K]",
           title="Omukai+05 dust-free tracks; compressional heating")
    a1.legend(fontsize=7, loc="upper left")
    a1b.legend(fontsize=7, loc="lower right")

    # 2. static dust at the critical D
    n_mol = z0.n_molecular(0.9)
    for mat, ap, c in (("MgSiO3", 3e-6, "tab:blue"), ("MgSiO3", 3e-5, "tab:cyan"), ("C", 1e-6, "tab:red"),
                       ("Fe", 1e-5, "tab:gray")):
        make = lambda D, m=mat, a=ap: ([lognormal_population(m, D * RHO_PER_H, a_peak=a)], None)
        Dcr = critical_dust_to_gas(make, z0, rtol=0.03, dlnn=0.1)
        h, _ = run_collapse(z0, make(Dcr)[0], n_end=1e16, dlnn=0.05)
        S = h.S[0]
        a2.plot(h.nH, h.C, color=c, lw=1.2,
                label=f"{mat} {ap * 1e4:g} um: D_cr={Dcr:.2e}, S D_cr={S * Dcr:.2e}")
    a2.axhline(1, color="k", lw=0.6)
    a2.axvspan(1e8, n_mol, color="0.92", zorder=0)
    a2.set(xscale="log", yscale="log", ylim=(1e-3, 30), xlabel=r"$n_{\rm H}$ [cm$^{-3}$]",
           ylabel=r"$C = \Lambda_{\rm gd}\,t_{\rm col}/e_{\rm th}$",
           title=f"Static dust at D_cr (grey: window)\nSchneider+12a Eq. 16: "
                 f"S D_cr = {1.4e-3 * (z0.T(1e12) / 1e3) ** -0.5:.2e} cm$^2$ g$^{{-1}}$")
    a2.legend(fontsize=7, loc="lower right")

    # 3. accretion
    mf = formula_mass("MgSiO3")
    for Z, c, ref in ((1e-4, "tab:green", 1e10), (1e-5, "tab:orange", 1e12), (1e-6, "tab:purple", 1e14)):
        y0 = solar_abundances(Z)
        nf = 0.1 * y0["Si"]
        y = dict(y0)
        y["Si"] -= nf
        y["Mg"] -= nf
        h, _ = run_collapse(z0, [lognormal_population("MgSiO3", nf * mf, a_peak=1e-6, sigma_ln=0.3)],
                            y_gas=y, n_end=1e17, dlnn=0.05, accretion=True)
        a3.plot(h.nH, np.asarray(h.D) / h.D[0], color=c, label=f"Z = {Z:g} Z_sun")
        a3.axvline(ref, color=c, ls=":", lw=1)
    a3.set(xscale="log", yscale="log", xlabel=r"$n_{\rm H}$ [cm$^{-3}$]", ylabel=r"$D / D_0$",
           title="MgSiO3 growth by accretion (dotted: Chiaki+13 growth densities)")
    a3.legend(fontsize=8)

    # 4. Brownian coagulation
    rho_s = get_material("Mg2SiO4").rho_s
    zeta = 50 ** (1 / 15)
    g = SizeGrid.from_radii(5e-7 * zeta ** -2, 2.5e-5 * zeta ** 25, 42, rho_s)
    ae = g.a_edges
    w = np.zeros(g.n_bins)
    w[2:17] = ae[3:18] ** 0.5 - ae[2:17] ** 0.5
    ns = np.logspace(2, 17, 3000)
    for Z, c in ((1.0, "k"), (1e-2, "tab:blue"), (1e-4, "tab:red")):
        h, _ = run_collapse(z0, [Population("Mg2SiO4", g, 0.006 * Z * RHO_PER_H * w / w.sum())],
                            n_end=1e17, dlnn=0.1, coagulation=True, coag_kernel=brownian_kernel)
        a4.plot(h.nH, np.asarray(h.S) / h.S[0], color=c, label=f"Z = {Z:g} Z_sun")
        crit = 1.6e7 * (rho_s / 3) ** 3 * Z ** -2 * (z0.T(ns) / 100) ** -1
        a4.axvline(ns[np.argmax(ns >= crit)], color=c, ls=":", lw=1)
    a4.set(xscale="log", ylim=(0, 1.05), xlabel=r"$n_{\rm H}$ [cm$^{-3}$]", ylabel=r"$S/S_0$",
           title="Brownian coagulation of MRN silicates (dotted: HO09 Eq. 15)")
    a4.legend(fontsize=8)
    fig.tight_layout()
    out = _paths.get_plots_dir("collapse") / "collapse_growth.png"
    fig.savefig(out, dpi=140)
    print("wrote", out)


if __name__ == "__main__":
    main()
