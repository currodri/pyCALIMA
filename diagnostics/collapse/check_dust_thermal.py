#!/usr/bin/env python3
"""Optics and dust temperatures of the collapse grain materials.

Writes ``dust_thermal.png`` to ``_paths.get_plots_dir("collapse")``:

* Q_abs / a vs wavelength for 10 nm grains (the Rayleigh regime, where it is
  independent of a for dielectrics);
* the Planck-mean opacity per gram of dust, kappa_P = 3 <Q> / (4 a rho_s), for
  0.1 um grains, against temperature;
* the dust temperature of each material vs grain radius in molecular gas at
  T = 600 K and T_CMB = 40 K (z ~ 14), for n(H2) = 1e8, 1e10, 1e12 cm^-3 and
  no continuum trapping (beta = 1).

and ``optical_constants.png``: per material, the tabulated n and k (points)
against the n and k used (lines; grey band: the tabulated range, beyond it
extrapolated), and Q_abs / a of a 10 nm grain against the electric-dipole
limit (the difference for Fe is magnetic-dipole absorption).

Materials whose optical constants are an unfetched dataset are left out.
Run ``python -m pycalima.collapse.export_collapse_optics`` first to cache the tables.

    python diagnostics/collapse/check_dust_thermal.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pycalima import _paths
from pycalima.collapse import gas_grain as gg
from pycalima.collapse.dust_temperature import dust_temperature
from pycalima.collapse.materials import materials
from pycalima.collapse.optics import LAMBDA_UM, load_nk, planck_table, qabs_grid, refractive_index
from pycalima.plotting_style import use_calima_style


def main():
    use_calima_style()
    plt.rcParams["text.usetex"] = False
    tabs = {}
    for name in sorted(materials()):
        try:
            tabs[name] = planck_table(name)
        except (FileNotFoundError, KeyError):
            print(f"{name}: optical constants not available, left out")
    colors = dict(zip(tabs, plt.cm.tab10(np.linspace(0, 1, 10))))
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.8))
    for name, tab in tabs.items():
        Q = qabs_grid(name, radii_cm=np.array([1e-6]))[0]
        lam_tab = load_nk(name)[0]
        ax1.plot(LAMBDA_UM, Q / 1e-6, color=colors[name], label=name)
        ax1.axvline(lam_tab.max(), color=colors[name], ls=":", lw=0.6)
        T = np.geomspace(5.0, 3000.0, 200)
        ax2.plot(T, 3 * tab(1e-5, T) / (4 * 1e-5 * materials()[name].rho_s), color=colors[name], label=name)
    ax1.set(xscale="log", yscale="log", xlim=(0.1, 2e4), ylim=(1e-3, 1e5),
            xlabel=r"$\lambda$ [$\mu$m]", ylabel=r"$Q_{\rm abs}/a$ [cm$^{-1}$], $a$ = 10 nm",
            title="Absorption (dotted: end of the data; beyond, extrapolated)")
    ax2.set(xscale="log", yscale="log", xlabel=r"$T$ [K]",
            ylabel=r"$\kappa_{\rm P}$ [cm$^2$ g$^{-1}$ of dust], $a$ = 0.1 $\mu$m",
            title="Planck-mean opacity")
    ax1.legend(fontsize=7, ncol=2)
    a = np.geomspace(1e-7, 1e-3, 60)
    T, Tr = 600.0, 40.0
    for nH2, ls in ((1e8, ":"), (1e10, "--"), (1e12, "-")):
        F = gg.collision_flux_factor(T, 0.0, nH2, 0.16 * nH2)
        for name, tab in tabs.items():
            ax3.plot(a * 1e4, dust_temperature(tab, a, T, F, Tr), ls, color=colors[name], lw=1.0)
        ax3.plot([], [], "k" + ls, label=rf"$n_{{\rm H_2}}=10^{{{int(np.log10(nH2))}}}$ cm$^{{-3}}$")
    ax3.axhline(T, color="0.6", lw=0.6)
    ax3.axhline(Tr, color="0.6", lw=0.6)
    ax3.set(xscale="log", yscale="log", xlabel=r"$a$ [$\mu$m]", ylabel=r"$T_{\rm d}$ [K]",
            title=r"Dust temperature: $T$ = 600 K, $T_{\rm CMB}$ = 40 K, $\beta$ = 1")
    ax3.legend(fontsize=8)
    fig.tight_layout()
    out = _paths.get_plots_dir("collapse") / "dust_thermal.png"
    fig.savefig(out, dpi=150)
    print("wrote", out)
    optical_constants_figure(list(tabs))


def optical_constants_figure(names):
    fig, axes = plt.subplots(len(names), 2, figsize=(12, 2.4 * len(names)), squeeze=False)
    for r, name in enumerate(names):
        lam, n, k = load_nk(name)
        m = refractive_index(LAMBDA_UM, lam, n, k)
        ax, ax2 = axes[r]
        for v in (ax, ax2):
            v.axvspan(lam.min(), lam.max(), color="0.93", zorder=0)
            v.set(xscale="log", yscale="log", xlim=(0.03, 2e4))
        ax.plot(lam, n, "b.", ms=2)
        ax.plot(lam, k, "r.", ms=2)
        ax.plot(LAMBDA_UM, m.real, "b-", lw=0.8)
        ax.plot(LAMBDA_UM, -m.imag, "r-", lw=0.8)
        ax.set_ylabel(name)
        a = 1e-6
        x = 2 * np.pi * a * 1e4 / LAMBDA_UM
        ax2.plot(LAMBDA_UM, qabs_grid(name, np.array([a]))[0] / a, "k-", lw=1, label=r"$Q_{\rm abs}/a$")
        ax2.plot(LAMBDA_UM, 4 * x * np.imag(-(m ** 2 - 1) / (m ** 2 + 2)) / a, "g--", lw=0.8,
                 label="electric dipole")
        ax2.set_ylim(1e-8, 1e7)
    axes[0, 0].set_title("n (blue), k (red): points tabulated, lines used")
    axes[0, 1].set_title(r"$a$ = 10 nm [cm$^{-1}$]")
    axes[0, 1].legend(fontsize=7)
    fig.tight_layout()
    out = _paths.get_plots_dir("collapse") / "optical_constants.png"
    fig.savefig(out, dpi=110)
    print("wrote", out)


if __name__ == "__main__":
    main()
