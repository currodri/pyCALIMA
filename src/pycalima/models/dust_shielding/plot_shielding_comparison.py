"""
Shielding factor comparison: old RAMSES formula vs. per-bin CALIMA.

Produces five figures saved as PDF in the same directory:

  fig1_kappa_LW.pdf             kappa_LW(a) vs grain size, graphite + silicate
  fig2_dust_shielding.pdf       comp_Sd (old vs new) vs total H column
  fig3_h2_shielding.pdf         Combined H2 shielding f_shd vs N_H
  fig4_co_shielding.pdf         CO self-shielding comp_SCO vs N_CO
  fig5_g0_selfshield.pdf        Dust self-shielding of G0 vs Sigma_dust

Usage
-----
Run from the CALIMA root:

    python -m models.dust_shielding.plot_shielding_comparison

or directly:

    python models/dust_shielding/plot_shielding_comparison.py
"""

from pathlib import Path

_HERE = Path(__file__).resolve().parent
_CALIMA_ROOT = _HERE.parents[1]

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pycalima.models.dust_shielding.shielding_functions import (
    comp_SH2,
    comp_Sd_old,
    comp_Sd_new,
    comp_SCO,
    comp_G0_selfshield,
    f_shd_old,
    f_shd_new,
    compute_kappa_LW_draine,
    compute_kappa_LW_precomp,
    SDEFF_MW,
    DGR_MW,
    KAPPA_LW_MW_EFF,
    m_H,
)

# ── Aesthetics ──────────────────────────────────────────────────────────────
sns.set_theme(style="ticks")
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

_OUT = _HERE   # save PDFs alongside this script

# Colour palette
C_GRA = "#2166ac"   # blue  — graphite
C_SIL = "#d6604d"   # red   — silicate
C_OLD = "#555555"   # grey  — old RAMSES formula


# ═══════════════════════════════════════════════════════════════════════════════
#   FIG 1 — kappa_LW(a) vs grain radius
# ═══════════════════════════════════════════════════════════════════════════════

def fig1_kappa_LW():
    """Plot LW-band mass opacity kappa_LW [cm^2/g] vs grain radius."""
    print("Figure 1: computing kappa_LW from Draine tables...")

    a_um   = np.logspace(-3, 1, 60)   # 0.001 to 10 um
    kap_gr = compute_kappa_LW_draine(a_um, composition='graphite')
    kap_si = compute_kappa_LW_draine(a_um, composition='silicate')

    # Pre-computed CALIMA bin values for the 4 default dust bins
    calima_bins = [
        ("DustBin_01", 2.2, 0.01,  C_GRA,  "Graphite bin 1 ($a_0=0.01$ um)"),
        ("DustBin_02", 2.2, 0.10,  C_GRA,  "Graphite bin 2 ($a_0=0.10$ um)"),
        ("DustBin_03", 3.3, 0.005, C_SIL,  "Silicate bin 1 ($a_0=0.005$ um)"),
        ("DustBin_04", 3.3, 0.10,  C_SIL,  "Silicate bin 2 ($a_0=0.10$ um)"),
    ]
    bin_kappas = {}
    for bin_id, rho_gr, a0, *_ in calima_bins:
        try:
            bin_kappas[bin_id] = compute_kappa_LW_precomp(bin_id, rho_gr, a0)
        except FileNotFoundError as e:
            print(f"  Warning: {e}")
            bin_kappas[bin_id] = None

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    ax.loglog(a_um, kap_gr, color=C_GRA, lw=2,
              label="Graphite (Draine Q-table sweep)")
    ax.loglog(a_um, kap_si, color=C_SIL, lw=2,
              label="Silicate (Draine Q-table sweep)")

    ax.axhline(KAPPA_LW_MW_EFF, color=C_OLD, lw=1.5, ls="--",
               label=(r"Old RAMSES: $\sigma_\mathrm{eff}/(D/G_\odot\,m_H)$"
                      rf"$={KAPPA_LW_MW_EFF:.1e}$ cm$^2$/g"))

    markers = ["o", "s", "^", "D"]
    for (bin_id, rho_gr, a0, col, label), mk in zip(calima_bins, markers):
        k = bin_kappas.get(bin_id)
        if k is not None:
            ax.plot(a0, k, mk, color=col, ms=8, zorder=5,
                    label=rf"{label}: $\kappa_{{LW}}={k:.1e}$ cm$^2$/g")

    ax.set_xlabel(r"Grain radius $a$ ($\mu$m)")
    ax.set_ylabel(r"$\kappa_{LW}$ (cm$^2$ g$^{-1}_\mathrm{dust}$)")
    ax.set_title(r"LW-band ($6$--$13.6\,\mathrm{eV}$) dust mass opacity")
    ax.set_xlim([1e-3, 10])
    ax.legend(loc="lower left", framealpha=0.8)
    ax.grid(which="both", alpha=0.25)
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(_OUT / "fig1_kappa_LW.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig1_kappa_LW.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
#   FIG 2 — Dust shielding factor comp_Sd vs N_H
# ═══════════════════════════════════════════════════════════════════════════════

def fig2_dust_shielding():
    """Compare old and new dust shielding factor vs total H column density."""
    print("Figure 2: dust shielding factor vs N_H...")

    N_H = np.logspace(18, 24, 300)   # cm^-2

    sizes_info = [
        (0.001, 'graphite', 2.2, "#1a78c2",
         r"$a=0.001\,\mu$m graphite (small)"),
        (0.01,  'graphite', 2.2, "#2166ac",
         r"$a=0.01\,\mu$m graphite (MW-like)"),
        (0.10,  'graphite', 2.2, "#7fcdff",
         r"$a=0.10\,\mu$m graphite (grown)"),
        (0.001, 'silicate', 3.3, "#c45030",
         r"$a=0.001\,\mu$m silicate (small)"),
        (0.10,  'silicate', 3.3, "#f4a582",
         r"$a=0.10\,\mu$m silicate (grown)"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    # Panel A: old formula for different metallicities Z
    for Z, ls in [(1.0, '-'), (0.1, '--'), (0.01, ':')]:
        f = comp_Sd_old(N_H, 0.0, Z=Z)
        ax1.loglog(N_H, f, color=C_OLD, ls=ls, lw=1.8,
                   label=f"Old formula, $Z={Z}$")

    ax1.set_title("Old RAMSES formula (fixed $\\sigma_\\mathrm{eff}$)")
    ax1.set_xlabel(r"$N_H = N_\mathrm{HI} + 2N_{H_2}$ (cm$^{-2}$)")
    ax1.set_ylabel(r"Dust LW shielding $f_{Sd}$")

    # Panel B: new formula — kappa_LW varies with grain size/composition
    for a_um, comp, rho_gr, col, label in sizes_info:
        kap = float(compute_kappa_LW_draine([a_um], comp,
                                             grain_density=rho_gr)[0])
        tau = kap * DGR_MW * m_H * N_H
        f   = comp_Sd_new(tau)
        ax2.loglog(N_H, f, color=col, lw=1.8, label=label)

    f_old = comp_Sd_old(N_H, 0.0, Z=1.0)
    ax2.loglog(N_H, f_old, color=C_OLD, ls="--", lw=1.5,
               label="Old formula $Z=1$ (reference)")

    ax2.set_title(r"New per-bin formula (same $D/G = D/G_\odot$)")
    ax2.set_xlabel(r"$N_H = N_\mathrm{HI} + 2N_{H_2}$ (cm$^{-2}$)")

    for ax in (ax1, ax2):
        ax.set_xlim([1e18, 1e24])
        ax.set_ylim([1e-5, 1.1])
        ax.grid(which="both", alpha=0.2)
        ax.legend(loc="lower left", framealpha=0.8)

    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(_OUT / "fig2_dust_shielding.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig2_dust_shielding.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
#   FIG 3 — Combined H2 shielding f_shd = f_SH2 x f_Sd
# ═══════════════════════════════════════════════════════════════════════════════

def fig3_h2_shielding():
    """Combined H2 self-shielding + dust shielding, old vs new."""
    print("Figure 3: combined H2 shielding factor...")

    # Mixed gas: x_H2 = 0.5 (half the H in H2)
    N_H  = np.logspace(19, 24, 300)
    N_HI = 0.5 * N_H
    N_H2 = 0.25 * N_H   # H2 molecules: n_H2 = 0.5 * x_H2 * n_H

    grain_models = [
        (0.001, 'graphite', 2.2, "#1a78c2",
         r"Small graphite ($a=0.001\,\mu$m)"),
        (0.01,  'graphite', 2.2, "#2166ac",
         r"MW graphite ($a=0.01\,\mu$m)"),
        (0.1,   'graphite', 2.2, "#7fcdff",
         r"Grown graphite ($a=0.1\,\mu$m)"),
        (0.1,   'silicate', 3.3, "#f4a582",
         r"Grown silicate ($a=0.1\,\mu$m)"),
    ]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    f_old = f_shd_old(N_HI, N_H2, Z=1.0)
    ax.loglog(N_H, f_old, color=C_OLD, ls="--", lw=2,
              label="Old formula $Z=1$ (current RAMSES)")

    f_h2only = comp_SH2(N_H2)
    ax.loglog(N_H, f_h2only, color=C_OLD, ls=":", lw=1.5,
              label=r"$f_{SH_2}$ only (no dust)")

    for a_um, comp, rho_gr, col, label in grain_models:
        kap = float(compute_kappa_LW_draine([a_um], comp,
                                             grain_density=rho_gr)[0])
        tau = kap * DGR_MW * m_H * N_H
        f   = f_shd_new(N_H2, tau)
        ax.loglog(N_H, f, color=col, lw=1.8, label=label)

    ax.set_xlabel(r"Total $N_H$ (cm$^{-2}$)")
    ax.set_ylabel(r"$f_{shd} = f_{SH_2} \times f_{Sd}$")
    ax.set_title(r"Combined H$_2$ + dust LW shielding ($x_{H_2}=0.5$, $D/G = D/G_\odot$)")
    ax.set_xlim([1e19, 1e24])
    ax.set_ylim([1e-6, 1.1])
    ax.legend(framealpha=0.8)
    ax.grid(which="both", alpha=0.2)
    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(_OUT / "fig3_h2_shielding.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig3_h2_shielding.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
#   FIG 4 — CO self-shielding comp_SCO
# ═══════════════════════════════════════════════════════════════════════════════

def fig4_co_shielding():
    """CO self-shielding factor as a function of N_CO for several N_H2."""
    print("Figure 4: CO self-shielding...")

    N_CO_arr  = np.logspace(12, 20, 400)
    N_H2_vals = [1e18, 1e19, 1e20, 1e21, 1e22]
    colours   = plt.cm.plasma(np.linspace(0.15, 0.85, len(N_H2_vals)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel A: shielding vs N_CO for different N_H2
    for N_H2, col in zip(N_H2_vals, colours):
        f = comp_SCO(N_CO_arr, N_H2)
        ax1.loglog(N_CO_arr, f, color=col, lw=1.8,
                   label=rf"$N_{{H_2}}=10^{{{int(np.log10(N_H2))}}}$ cm$^{{-2}}$")

    ax1.set_xlabel(r"$N_{CO}$ (cm$^{-2}$)")
    ax1.set_ylabel(r"CO self-shielding $f_{SCO}$")
    ax1.set_title("CO self-shielding (Lee 1996 / Visser et al. 2009 tables)")
    ax1.set_xlim([1e12, 1e20])
    ax1.set_ylim([1e-6, 1.1])
    ax1.legend(framealpha=0.8, fontsize=8)
    ax1.grid(which="both", alpha=0.2)

    # Panel B: H2 continuum shielding of CO vs N_H2
    N_H2_arr = np.logspace(13, 22, 400)
    f_H2_CO  = comp_SCO(1.0, N_H2_arr)   # N_CO -> 0: isolates H2 term
    ax2.loglog(N_H2_arr, f_H2_CO, color=C_GRA, lw=2)
    ax2.set_xlabel(r"$N_{H_2}$ (cm$^{-2}$)")
    ax2.set_ylabel(r"H$_2$ continuum shielding of CO, $f_{SH_2,CO}$")
    ax2.set_title(r"H$_2$ continuum shielding of CO photodissociation")
    ax2.set_xlim([1e13, 1e22])
    ax2.set_ylim([1e-3, 1.1])
    ax2.grid(which="both", alpha=0.2)

    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(_OUT / "fig4_co_shielding.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig4_co_shielding.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
#   FIG 5 — Dust self-shielding of G0
# ═══════════════════════════════════════════════════════════════════════════════

def fig5_g0_selfshield():
    """Dust self-shielding of G0 vs dust surface density."""
    print("Figure 5: G0 self-shielding by dust...")

    Sigma_dust = np.logspace(-4, 2, 400)   # g/cm^2

    grain_models = [
        (0.001, 'graphite', 2.2, "#1a78c2",
         r"Small graphite ($a=0.001\,\mu$m)"),
        (0.01,  'graphite', 2.2, "#2166ac",
         r"MW graphite ($a=0.01\,\mu$m)"),
        (0.1,   'graphite', 2.2, "#7fcdff",
         r"Grown graphite ($a=0.1\,\mu$m)"),
        (1.0,   'graphite', 2.2, "#d0e8ff",
         r"Large graphite ($a=1.0\,\mu$m)"),
        (0.001, 'silicate', 3.3, "#c45030",
         r"Small silicate ($a=0.001\,\mu$m)"),
        (0.1,   'silicate', 3.3, "#f4a582",
         r"MW silicate ($a=0.1\,\mu$m)"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    # Panel A: attenuation vs dust surface density
    for a_um, comp, rho_gr, col, label in grain_models:
        kap = float(compute_kappa_LW_draine([a_um], comp,
                                             grain_density=rho_gr)[0])
        tau = kap * Sigma_dust
        f   = comp_G0_selfshield(tau)
        ax1.plot(Sigma_dust, f, color=col, lw=1.8, label=label)

    ax1.set_xscale("log")
    ax1.set_xlabel(r"Dust column $\Sigma_\mathrm{dust}$ (g cm$^{-2}$)")
    ax1.set_ylabel(r"$G_0$ attenuation $\exp(-\tau_{FUV})$")
    ax1.set_title(r"Dust self-shielding of $G_0$")
    ax1.legend(framealpha=0.8, fontsize=8)
    ax1.set_xlim([1e-4, 1e2])
    ax1.set_ylim([-0.02, 1.05])
    ax1.grid(which="both", alpha=0.2)

    # Panel B: universal exp(-tau) curve — all grain models collapse onto this
    tau_arr = np.logspace(-3, 4, 300)
    f_tau   = comp_G0_selfshield(tau_arr)
    ax2.semilogx(tau_arr, f_tau, color="k", lw=2)
    ax2.axvline(1.0, color="grey", ls="--", lw=1,
                label=r"$\tau_{FUV}=1$")
    ax2.set_xlabel(r"$\tau_{FUV} = \kappa_{LW}\,\Sigma_\mathrm{dust}$")
    ax2.set_title(r"Universal attenuation $e^{-\tau}$")
    ax2.legend(framealpha=0.8)
    ax2.set_xlim([1e-3, 1e4])
    ax2.grid(which="both", alpha=0.2)

    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(_OUT / "fig5_g0_selfshield.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig5_g0_selfshield.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
#   Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Shielding factor comparison -- pyCALIMA")
    print(f"  Old sigma_eff = {SDEFF_MW:.2e} cm2/H  (fixed MW bare-grain-s)")
    print(f"  DGR_MW        = {DGR_MW:.4f}      (canonical MW dust-to-gas)")
    print(f"  kappa_LW_eff  = {KAPPA_LW_MW_EFF:.2e} cm2/g (implied by old formula)")
    print("=" * 60)

    fig1_kappa_LW()
    fig2_dust_shielding()
    fig3_h2_shielding()
    fig4_co_shielding()
    fig5_g0_selfshield()

    print("\nAll figures saved in:", str(_OUT))
