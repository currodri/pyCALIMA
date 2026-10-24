#!/usr/bin/env python3
"""Per-size grain charging in AGN illumination (pycalima.gsd.charging).

Grains of 3 A - 1 um at distance r from an AGN of L_bol = 1e46 erg/s
(SKIRTOR disk SED plus a Gamma = 1.9 corona with L_2-10 = L_bol / 30, which
the SKIRTOR SED lacks), in photoionised gas at T = 1e4 K and n_e = 1e2
cm^-3 (the polar-dust conditions of Tazaki, Ichikawa & Kokubo 2020, TIK20).
Charges are solved per radius with the RAMSES-CALIMA group solver extended
to 10 keV (WDB06 yields with inner shells and Auger electrons).

Left: grain potential and the ion-capture Coulomb factor <J~> versus size
(~ exp(-eU/kT): positive grains at hundreds of volts accrete no ions at
all, so growth in photoionised gas needs the neutral fraction).
Right: the Coulomb-explosion limit a_CE (largest radius with a non-zero
explosion rate) versus r / L45^1/2, for tensile strengths 1e9-1e11 dyn
cm^-2. TIK20 (their Figs. 1-2) find a_CE ~ 0.01-0.1 um, falling much more
slowly with distance than the sublimation radius (r^-2), with a flat
extinction curve inside ~10 L45^1/2 pc.

Usage
-----
    python diagnostics/gsd/check_charging_agn.py
"""
import argparse

import numpy as np

from pycalima import _paths
from pycalima.gsd.charging import ChargingModel, PhotonGroups, xray_power_law
from pycalima.models.dust_radiation.agn_hot_dust import AGNIllumination

PC = 3.086e18
L_BOL = 1e46


def photons(agn, r_pc):
    r = r_pc * PC
    ph = PhotonGroups.from_flux_lambda(agn.lam_cm, agn.flux(r))
    E, nE = xray_power_law(r, L_BOL / 30.0)
    return ph + PhotonGroups.from_photon_density(E, nE)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    agn = AGNIllumination(L_BOL, law="isotropic")
    a_grid = np.geomspace(3e-8, 1e-4, 22)
    r_list = np.geomspace(0.1, 100.0, 13)
    L45 = L_BOL / 1e45
    models = {mat: ChargingModel(mat, a_grid) for mat in ("silicate", "graphite")}

    profiles = {}
    a_ce = {(mat, S): [] for mat in models for S in (1e9, 1e10, 1e11)}
    for r_pc in r_list:
        ph = photons(agn, r_pc)
        for mat, cm in models.items():
            st = cm.solve(ph, T=1e4, n_e=1e2)
            if np.isclose(r_pc, r_list[[2, 6, 10]]).any():
                profiles[(mat, r_pc)] = (st.potential_volts(a_grid), st.ion_focusing())
            for S in (1e9, 1e10, 1e11):
                rate = st.explosion_rate(S)
                a_ce[(mat, S)].append(a_grid[rate > 0].max() if (rate > 0).any() else np.nan)
    print("a_CE [um] vs r / L45^1/2 [pc]:")
    x = r_list / np.sqrt(L45)
    print("  r/L45^1/2 " + " ".join(f"{v:7.2f}" for v in x))
    for key, vals in a_ce.items():
        print(f"  {key[0]:8s} S={key[1]:.0e} " + " ".join(f"{v * 1e4:7.4f}" for v in vals))

    if args.no_plot:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.5, 4.2))
    colors = {r_list[2]: "#2a78d6", r_list[6]: "#1baf7a", r_list[10]: "#eb6834"}
    for (mat, r_pc), (U, J) in profiles.items():
        ls = "-" if mat == "silicate" else "--"
        lab = f"r = {r_pc:.2g} pc" if mat == "silicate" else None
        ax1.plot(a_grid * 1e4, U, color=colors[r_pc], ls=ls, lw=1.5, label=lab)
        ax2.plot(a_grid * 1e4, J, color=colors[r_pc], ls=ls, lw=1.5, label=lab)
    for S, c in ((1e9, "#c3c2be"), (1e10, "#52514e"), (1e11, "#1c1c1a")):
        ax1.plot(a_grid * 1e4, a_grid * np.sqrt(4 * np.pi * S) * 299.792458, color=c, lw=1,
                 ls=":", label=rf"$U_{{\rm CE}}$, $S_{{\max}}$ = {S:.0e}")
        for mat, ls in (("silicate", "-"), ("graphite", "--")):
            ax3.plot(x, np.array(a_ce[(mat, S)]) * 1e4, color=c, ls=ls, lw=1.5,
                     label=rf"{mat}, $S_{{\max}}$ = {S:.0e}")
    ax1.set(xscale="log", yscale="log", xlabel=r"$a$ [$\mu$m]", ylabel=r"$\langle U\rangle$ [V]",
            title=r"Grain potential ($L = 10^{46}$, $T = 10^4$ K, $n_e = 10^2$)")
    ax2.set(xscale="log", yscale="log", xlabel=r"$a$ [$\mu$m]",
            ylabel=r"ion capture $\langle\tilde J\rangle$", ylim=(1e-300, 10),
            title=r"Ion accretion factor ($\sim e^{-eU/kT}$; 0 not shown)")
    ax3.set(xscale="log", yscale="log", xlabel=r"$r / L_{45}^{1/2}$ [pc]",
            ylabel=r"$a_{\rm CE}$ [$\mu$m]", title="Coulomb-explosion limit (TIK20: 0.01-0.1 um)")
    ax3.axhspan(0.01, 0.1, color="#efeeea", zorder=0)
    for ax in (ax1, ax2, ax3):
        ax.tick_params(which="both", direction="in", top=True, right=True)
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    out = _paths.get_plots_dir()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "gsd_charging_agn.png", dpi=150)
    print("Saved", out / "gsd_charging_agn.png")


if __name__ == "__main__":
    main()
