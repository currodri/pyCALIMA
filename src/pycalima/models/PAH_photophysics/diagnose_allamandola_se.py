"""
diagnose_allamandola_se.py
==========================
Compare two sticking-coefficient prescriptions for neutral PAH electron
attachment (Z=0 → Z=-1):

  1. Calibrated  — se = c × α^m  (backwards-reconstructed to match Andrews 2016 Fig. 8)
  2. Full formula — S = kr/(kr+kb) using Allamandola et al. (1989) detailed balance
                    with quantum harmonic oscillator density of states and kr from
                    Li & Draine IR cross-section

For each of the three reference PAHs (C24H12, C54H18, C96H24) at T = 500 K:
  • prints the key intermediate quantities
  • shows the sticking coefficient from both methods
  • plots S_full as a function of gas temperature (10 – 5000 K)

Usage
-----
    python -m models.PAH_photophysics.diagnose_allamandola_se
"""

from __future__ import annotations

import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

import numpy as np
import matplotlib.pyplot as plt
from math import lgamma

from pycalima.models.PAH_photophysics.pah_charge_utils import (
    afromNc,
    alpha_neutral_Cagliari,
    se_neutral_Andrews2016,
    se_neutral_Allamandola1989_full,
    ME_CGS, H_CGS, C_CGS, KB_CGS, EV2ERG, E_STATC, TINY,
)
from pycalima.models.PAH_photophysics.pah_temperature import get_absorption_cross_section
from scipy.optimize import brentq

# ── PAH definitions ──────────────────────────────────────────────────────────
PAH_DEFS = [
    dict(name='C24H12', Nc=24, Nh=12, EA=0.47, label=r'Coronene  $C_{24}H_{12}$'),
    dict(name='C54H18', Nc=54, Nh=18, EA=1.44, label=r'Circumcoronene  $C_{54}H_{18}$'),
    dict(name='C96H24', Nc=96, Nh=24, EA=3.11, label=r'Circumcircumcoronene  $C_{96}H_{24}$'),
]

T_GAS = 500.0   # K
T_ARR = np.logspace(1.0, np.log10(5000), 120)   # 10–5000 K

HBAR = H_CGS / (2.0 * np.pi)
HNU0 = H_CGS * C_CGS * 1000.0   # erg, ν₀ = 1000 cm⁻¹


def _intermediates(Nc, Nh, EA_eV, alpha_Ang3, T_K=500.0):
    """Return dict of intermediate Allamandola quantities."""
    alpha_cm3 = alpha_Ang3 * 1e-24
    kf = 2.0 * np.pi * np.sqrt(alpha_cm3 * E_STATC**2 / ME_CGS)

    eps_erg = KB_CGS * T_K
    v_e = np.sqrt(2.0 * eps_erg / ME_CGS)
    rho_e = ME_CGS**2 * v_e / (np.pi**2 * HBAR**3)

    s = 3*(Nc + Nh) - 6
    n = max(1, round(EA_eV * EV2ERG / HNU0))
    log_rho_m = lgamma(n + s) - lgamma(n + 1) - lgamma(s) - np.log(HNU0)
    rho_m = np.exp(log_rho_m)
    kb = kf * rho_e / rho_m

    a0 = afromNc(Nc)
    w_cm, C_abs = get_absorption_cross_section(0, a0)
    nu = C_CGS / w_cm
    hnu_eV = H_CGS * nu / EV2ERG
    ir = (w_cm > 1e-4) & (C_abs > 0)
    nu_ir, C_ir, hnu_ir = nu[ir], C_abs[ir], hnu_eV[ir]

    def U_of_T(T):
        x = H_CGS * nu_ir / (KB_CGS * T)
        occ = np.where(x > 50, np.exp(-x),
                       1./(np.exp(np.clip(x, 0, 50)) - 1 + TINY))
        return float(np.trapezoid(C_ir*hnu_ir*occ, nu_ir)
                     / np.trapezoid(C_ir, nu_ir)) * s / 2.0

    try:
        T_vib = brentq(lambda T: U_of_T(T) - EA_eV, 10., 10000., xtol=1.)
    except ValueError:
        T_vib = 500.

    x = H_CGS * nu_ir / (KB_CGS * T_vib)
    nb = np.where(x > 50, np.exp(-x),
                  1./(np.exp(np.clip(x, 0, 50)) - 1 + TINY))
    kr = float(np.trapezoid(
        (8*np.pi*nu_ir**2/C_CGS**2)*C_ir*nb, nu_ir))

    return dict(s=s, n=n, rho_m=rho_m, rho_e=rho_e, kf=kf, kb=kb, kr=kr,
                T_vib=T_vib, S_full=kr/(kr + kb))


def main():
    print("=" * 72)
    print("Allamandola (1989) sticking coefficient — diagnostics at T=500 K")
    print("=" * 72)
    print(f"{'PAH':12s}  {'s':>4}  {'n':>3}  {'ρ⁻ (erg⁻¹)':>12}  "
          f"{'kr (s⁻¹)':>10}  {'kb (s⁻¹)':>12}  {'S_full':>8}  {'se_calib':>8}")
    print("-" * 72)

    se_full_arr: dict[str, np.ndarray] = {}

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for p in PAH_DEFS:
            alpha = alpha_neutral_Cagliari(p['Nc'])
            d = _intermediates(p['Nc'], p['Nh'], p['EA'], alpha, T_GAS)
            se_c = se_neutral_Andrews2016(p['Nc'], p['EA'])
            print(f"{p['name']:12s}  {d['s']:4d}  {d['n']:3d}  "
                  f"{d['rho_m']:12.3e}  {d['kr']:10.2f}  {d['kb']:12.3e}  "
                  f"{d['S_full']:8.4f}  {se_c:8.4f}")

            # Temperature sweep
            se_T = np.array([
                se_neutral_Allamandola1989_full(
                    p['Nc'], p['Nh'], p['EA'], alpha, T)
                for T in T_ARR
            ])
            se_full_arr[p['name']] = se_T

    print()
    print("Notes:")
    print("  ρ⁻ = quantum harmonic oscillator DOS at E=EA, ν₀=1000 cm⁻¹")
    print("  kr = spontaneous IR emission rate at T_vib (Li & Draine C_abs)")
    print("  se_calib = c × α^m calibrated to Andrews+16 Fig. 8")
    print("  S_full: C24 and C96 give ≈0 because ρ⁻ small → kb >> kr at 500 K")

    # ── Plot: S_full(T) vs se_calib for the three PAHs ──────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5),
                             sharey=True, constrained_layout=True)
    colors = {'C24H12': '#1f77b4', 'C54H18': '#ff7f0e', 'C96H24': '#2ca02c'}

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for col, p in enumerate(PAH_DEFS):
            ax = axes[col]
            name = p['name']
            alpha = alpha_neutral_Cagliari(p['Nc'])
            se_c = se_neutral_Andrews2016(p['Nc'], p['EA'])

            ax.semilogx(T_ARR, se_full_arr[name],
                        color=colors[name], lw=2.0,
                        label='Full Allamandola+89\n(quantum DOS + IR $k_r$)')
            ax.axhline(se_c, color=colors[name], ls='--', lw=1.5,
                       label=f'Calibrated ($s_e={se_c:.3f}$)')
            ax.axvline(T_GAS, color='k', ls=':', lw=0.8)

            ax.set_xlabel(r'Gas temperature $T$ [K]', fontsize=10)
            ax.set_title(p['label'], fontsize=10)
            ax.set_xlim(T_ARR[0], T_ARR[-1])
            ax.set_ylim(-0.02, 1.05)
            ax.legend(fontsize=8)
            ax.grid(True, which='both', alpha=0.25, lw=0.5)

    axes[0].set_ylabel(r'Sticking coefficient $s_e$', fontsize=10)
    fig.suptitle(
        'Neutral PAH electron sticking coefficient: full Allamandola+89 vs calibrated\n'
        r'$k_r$ from Li \& Draine $C_\mathrm{abs}$;  $\rho^-$ from quantum harmonic DOS at $E=E_A$',
        fontsize=10,
    )

    out = ROOT / 'allamandola_se_comparison.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'\nFigure saved → {out}')
    plt.show()


if __name__ == '__main__':
    main()
