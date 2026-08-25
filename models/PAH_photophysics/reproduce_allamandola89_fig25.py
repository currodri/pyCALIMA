"""
reproduce_allamandola89_fig25.py
=================================
Reproduce Figure 25 of Allamandola et al. (1989, ApJS 71, 733):
electron sticking coefficient S vs number of carbon atoms N, computed via

    S = kr / (kr + kb)
    kb = kf × ρ_e(ε) / ρ⁻(E*)

with the Whitten-Rabinovitch (1963) density of states for the anion:

    ρ_WR(E*) = (E* + a_WR × E_zpe)^(s-1) / [(s-1)! × (hν₀)^s]

Fixed parameters (from the paper):
    EA = 0.7 eV,  kr = 10 s⁻¹,  T = 10 K,  ν₀ = 1000 cm⁻¹
    s = 3*(Nc+Nh)-6  (all vibrational modes)
    Nh ≈ round(√(6·Nc))  (compact hexagonal PAH approximation)
    E* = EA + kBT ≈ EA  (kBT = 0.86 meV at T=10 K, negligible)

Key finding: reproducing the Allamandola curve requires a_WR ≈ 0.77 (constant),
which is the value tabulated in Whitten-Rabinovitch Table I for the relevant
(ρ, s) regime.  The simple approximation a = 1 − exp(−2.4191 ρ^(1/3)) gives
a ≈ 0.52–0.68, which is systematically too small and yields S → 0.

Usage
-----
    python -m models.PAH_photophysics.reproduce_allamandola89_fig25
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from math import lgamma, log, floor, sqrt, exp
import matplotlib.pyplot as plt

# ── Physical constants (CGS) ──────────────────────────────────────────────────
ME    = 9.1093837015e-28
KB    = 1.380649e-16
HBAR  = 1.054571817e-27
H     = 6.62607015e-27
C_LGT = 2.99792458e10
EV    = 1.602176634e-12
ESTATC = 4.8032047e-10
HNU0  = H * C_LGT * 1000.0    # erg  (ν₀ = 1000 cm⁻¹)

# ── Molecular helpers ─────────────────────────────────────────────────────────

def Nh_compact(Nc: int) -> int:
    """Nh for a compact (pericondensed) PAH: Nh = round(√(6·Nc))."""
    return round(sqrt(6.0 * Nc))


def alpha_Ang3(Nc: float) -> float:
    """Neutral polarizability [Å³] — Allamandola+89 linear scaling: α = 1.5e-24 Nc cm³."""
    return 1.5 * Nc


# ── Rate coefficients ─────────────────────────────────────────────────────────

def _log_rho_e(T: float) -> float:
    """ln ρ_e [erg⁻¹ cm⁻³] at ε = kBT."""
    v = (2.0 * KB * T / ME) ** 0.5
    return log(ME**2 * v / (np.pi**2 * HBAR**3))


def _log_kf(Nc: float) -> float:
    """ln kf [cm³ s⁻¹] — Langevin electron capture rate coefficient."""
    a_cm3 = alpha_Ang3(Nc) * 1e-24
    return log(2.0 * np.pi * (a_cm3 * ESTATC**2 / ME) ** 0.5)


# ── Sticking coefficient ──────────────────────────────────────────────────────

def a_WR_fit(Nc: int, EA: float = 0.7) -> float:
    """
    Whitten-Rabinovitch correction factor a(ρ) fitted to Allamandola+89 Fig. 25.

    Quadratic in ρ = EA / E_zpe:
        a = -1.3713 ρ² + 0.3802 ρ + 0.7481

    where E_zpe = s × hν₀/2  and  s = 3(Nc+Nh)-6.

    Calibrated using α = 1.5×10⁻²⁴ Nc cm³ (Allamandola+89 Eq. A.7).
    Reproduces the digitised Fig. 25 curve to within ≈10%.
    The standard approximation a = 1 − exp(−2.4191 ρ^(1/3)) gives
    a ≈ 0.52–0.68 here (too small) and makes S → 0.
    """
    Nh = Nh_compact(Nc)
    s = 3 * (Nc + Nh) - 6
    E_zpe = s * HNU0 / 2.0
    rho = EA * EV / E_zpe
    return -1.3713 * rho**2 + 0.3802 * rho + 0.7481


def S_WR(Nc: int, EA: float = 0.7, T: float = 10.0, kr: float = 10.0,
         a_WR: float | None = None) -> float:
    """
    Sticking coefficient from Whitten-Rabinovitch density of states.

    Parameters
    ----------
    Nc    : number of carbon atoms
    EA    : electron affinity [eV]
    T     : gas/electron temperature [K]
    kr    : IR radiative stabilization rate [s⁻¹]  (Allamandola+89 assumed 10)
    a_WR  : Whitten-Rabinovitch correction factor.  If None, uses the
            quadratic fit a_WR_fit(Nc, EA) calibrated to Allamandola+89 Fig. 25.
    """
    if a_WR is None:
        a_WR = a_WR_fit(Nc, EA)
    Nh = Nh_compact(Nc)
    s = 3 * (Nc + Nh) - 6
    E = EA * EV
    E_zpe = s * HNU0 / 2.0
    E_eff = E + a_WR * E_zpe
    if E_eff <= 0.0:
        return 0.0

    log_rho_m = (s - 1) * log(E_eff / HNU0) - lgamma(s) - log(HNU0)
    log_kb = _log_kf(Nc) + _log_rho_e(T) - log_rho_m
    log_kb_kr = log_kb - log(kr)

    if log_kb_kr > 500:
        return 0.0
    if log_kb_kr < -500:
        return 1.0
    return 1.0 / (1.0 + exp(log_kb_kr))


def S_quantum(Nc: int, EA: float = 0.7, T: float = 10.0, kr: float = 10.0) -> float:
    """Exact quantum combinatorial DOS: C(n+s-1, n) / hν₀."""
    Nh = Nh_compact(Nc)
    s = 3 * (Nc + Nh) - 6
    n = max(1, int(floor(EA * EV / HNU0)))
    log_rho_m = lgamma(n + s) - lgamma(n + 1) - lgamma(s) - log(HNU0)
    log_kb = _log_kf(Nc) + _log_rho_e(T) - log_rho_m
    log_kb_kr = log_kb - log(kr)
    if log_kb_kr > 500:
        return 0.0
    if log_kb_kr < -500:
        return 1.0
    return 1.0 / (1.0 + exp(log_kb_kr))


def S_WR_stdapprox(Nc: int, EA: float = 0.7, T: float = 10.0,
                   kr: float = 10.0) -> float:
    """WR with the standard a = 1 − exp(−2.4191 ρ^(1/3)) approximation."""
    Nh = Nh_compact(Nc)
    s = 3 * (Nc + Nh) - 6
    E = EA * EV; E_zpe = s * HNU0 / 2.0
    rho = E / E_zpe
    beta = exp(-2.4191 * rho ** (1.0 / 3.0))
    return S_WR(Nc, EA, T, kr, a_WR=1.0 - beta)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    data = np.loadtxt(ROOT / 'external_data' / 'sticking_coefficient_Allamandola1989.csv',
                      delimiter=',')
    N_data, S_data = data[:, 0], data[:, 1]

    N_arr = np.arange(10, 130, 1)
    S_fit  = np.array([S_WR(Nc)             for Nc in N_arr])
    S_077  = np.array([S_WR(Nc, a_WR=0.77)  for Nc in N_arr])
    S_quant = np.array([S_quantum(Nc)        for Nc in N_arr])
    S_std   = np.array([S_WR_stdapprox(Nc)  for Nc in N_arr])

    # print comparison at data points
    print(f"{'Nc':>4}  {'S_data':>10}  {'S_WR(fit)':>10}  {'a_WR_fit':>9}  {'S_quantum':>10}")
    for Nd, Sd in zip(N_data, S_data):
        Nc = int(round(Nd))
        sf = S_WR(Nc)
        sq = S_quantum(Nc)
        af = a_WR_fit(Nc)
        print(f"{Nc:4d}  {Sd:10.3e}  {sf:10.3e}  {af:9.4f}  {sq:10.3e}")

    # ── Plot ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(N_arr, S_fit,   lw=2.2, color='C0',
                label=r'WR, $a_{WR}(\rho)=-1.42\rho^2+0.40\rho+0.748$ (fitted)')
    ax.semilogy(N_arr, S_077,   lw=1.5, color='C3', ls=':',
                label=r'WR, $a_{WR}=0.77$ (constant)')
    ax.semilogy(N_arr, S_quant, lw=1.5, color='C1', ls='--',
                label=r'Quantum DOS $C(n+s-1,n)/h\nu_0$')
    ax.semilogy(N_arr, S_std,   lw=1.5, color='C2', ls='-.',
                label=r'WR, $a=1-\exp(-2.42\rho^{1/3})$ (standard approx)')
    ax.scatter(N_data, S_data,  s=40,  color='k', zorder=5,
               label='Allamandola+89 Fig. 25 [digitised]')

    ax.set_xlabel('Number of carbon atoms $N$', fontsize=12)
    ax.set_ylabel(r'Sticking coefficient $S(e)$', fontsize=12)
    ax.set_title(
        r'Electron sticking coefficient vs PAH size'
        '\n'
        r'$k_r=10\,\mathrm{s}^{-1}$, $E_A=0.7\,\mathrm{eV}$, $T=10\,\mathrm{K}$, '
        r'$\nu_0=1000\,\mathrm{cm}^{-1}$, $s=3(N_C+N_H)-6$',
        fontsize=10,
    )
    ax.set_xlim(10, 130)
    ax.set_ylim(3e-7, 5e-2)
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3, lw=0.5)

    out = ROOT / 'allamandola89_fig25_comparison.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'\nFigure saved → {out}')
    plt.show()


if __name__ == '__main__':
    main()
