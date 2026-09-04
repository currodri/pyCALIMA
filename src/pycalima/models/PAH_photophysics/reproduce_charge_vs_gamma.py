"""
reproduce_charge_vs_gamma.py
============================
Equilibrium charge-state distribution (f_{Z} for Z = -1, 0, +1, +2) of the
normally-hydrogenated (Nh = Nh0) form of coronene, circumcoronene, and
circumcircumcoronene as a function of the ionisation parameter

    γ = G0 √T / ne

from 1 to 10^6.  Three panels in one figure.

Radiation field: Kurucz 15000 K stellar spectrum (as in Andrews 2016).
Gas temperature: T = 500 K (fixed).

Usage
-----
    python -m models.PAH_photophysics.reproduce_charge_vs_gamma
"""

from __future__ import annotations

import warnings
from pathlib import Path


import numpy as np
import matplotlib.pyplot as plt

from pycalima.models.PAH_photophysics.pah_charge_utils import (
    afromNc,
    recombination_rate_Bakes1994,
    attachment_rate_Bakes1994,
    se_neutral_Andrews2016,
    se_neutral_WR,
)
from pycalima.models.PAH_photophysics.pah_dissociation import compute_total_photoionisation_rate
from pycalima.models.PAH_photophysics.pah_temperature import (
    compute_base_g0,
    get_absorption_cross_section,
)
from pycalima.models.PAH_photophysics.pah_radiation import load_kurucz_u_E, load_kurucz_I_nu
from pycalima import _paths

# ─── PAH definitions ──────────────────────────────────────────────────────────
PAH_DEFS = {
    'C24': dict(Nc=24, Nh0=12,
                label='Coronene\n($C_{24}H_{12}$)',
                IP1=7.20, IP2=11.50, EA=0.47),
    'C54': dict(Nc=54, Nh0=18,
                label='Circumcoronene\n($C_{54}H_{18}$)',
                IP1=6.14, IP2=8.91,  EA=1.44),
    'C96': dict(Nc=96, Nh0=24,
                label='Circumcircumcoronene\n($C_{96}H_{24}$)',
                IP1=5.68, IP2=8.24,  EA=3.11),   # BT94: EA(1)=4.4-0.5×25.1/√96
}
_PAH_ORDER = ['C24', 'C54', 'C96']

T_GAS  = 500.0     # K
NE_REF = 1.0       # cm^-3  (reference; equilibrium depends only on γ)
_HV_EV = 1.23984193e-4   # h·c  [eV·cm]

NGAMMA = 300
GAMMA_GRID = np.logspace(0, 6, NGAMMA)


# ─── 4-species charge balance ─────────────────────────────────────────────────

def _solve_charge(k_det: float, k_ion0: float, k_ion1: float,
                  k_rec1_ne: float, k_rec2_ne: float, k_att_ne: float) -> np.ndarray:
    """
    Solve steady-state 4-species charge balance (Z = -1, 0, +1, +2).

    Rate matrix (dn/dt = A @ n = 0):
      Z=-1: gains from attachment (n_0), loses to photodetachment
      Z= 0: gains from detachment, Z=+1 recombination; loses to attachment, ionisation
      Z=+1: gains from Z=0 ionisation, Z=+2 recombination; loses to ionisation, recombination
      Z=+2: gains from Z=+1 ionisation; loses to recombination

    k_rec1_ne : e- recombination rate for Z=+1 [s^-1] (already × ne)
    k_rec2_ne : e- recombination rate for Z=+2 [s^-1] (already × ne; > k_rec1_ne from Coulomb)

    Returns normalised fractions [f_{-1}, f_0, f_{+1}, f_{+2}].
    """
    A = np.zeros((4, 4))
    # Z = -1
    A[0, 0] = -k_det
    A[0, 1] =  k_att_ne
    # Z = 0
    A[1, 0] =  k_det
    A[1, 1] = -(k_att_ne + k_ion0)
    A[1, 2] =  k_rec1_ne
    # Z = +1
    A[2, 1] =  k_ion0
    A[2, 2] = -(k_rec1_ne + k_ion1)
    A[2, 3] =  k_rec2_ne
    # Z = +2
    A[3, 2] =  k_ion1
    A[3, 3] = -k_rec2_ne

    # Conservation replaces last row
    A[-1, :] = 1.0
    b = np.zeros(4)
    b[-1] = 1.0

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        n = np.linalg.solve(A, b)

    n = np.clip(n, 0.0, None)
    s = n.sum()
    if s > 0:
        n /= s
    return n


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Computing Kurucz G0 base ...", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        G0_base = float(compute_base_g0(load_kurucz_u_E(15000)))
    kurucz_I = load_kurucz_I_nu(15000)
    print(f"  G0_base = {G0_base:.4e}", flush=True)

    # Reference field at G0=1: I_ν(ν) / G0_base
    def field_ref(nu: float) -> float:
        return kurucz_I(nu) / G0_base

    # ── Pre-compute photoionisation reference rates (scale linearly with G0) ──
    pah_rates: dict = {}
    for name, pdef in PAH_DEFS.items():
        a0 = afromNc(pdef['Nc'])
        w, C_abs = get_absorption_cross_section(0, a0)
        E_cs = _HV_EV / w
        idx  = np.argsort(E_cs)
        xsect = np.column_stack([E_cs[idx], C_abs[idx]])

        # Photodetachment threshold = EA (not IP1); k_det scales linearly with G0
        k_det = float(compute_total_photoionisation_rate(field_ref, xsect, IP=pdef['EA']))
        k1    = float(compute_total_photoionisation_rate(field_ref, xsect, IP=pdef['IP1']))
        k2    = float(compute_total_photoionisation_rate(field_ref, xsect, IP=pdef['IP2']))
        # se=1 for positively ionised species (attractive Coulomb)
        k_rec1 = recombination_rate_Bakes1994(pdef['Nc'], Z=1, se=1.0, T=T_GAS, ne=1.0)
        k_rec2 = recombination_rate_Bakes1994(pdef['Nc'], Z=2, se=1.0, T=T_GAS, ne=1.0)
        # Andrews+16 calibrated se (power law in Cagliari polarizability)
        se_calib = se_neutral_Andrews2016(pdef['Nc'], pdef['EA'])
        k_att_calib = attachment_rate_Bakes1994(pdef['Nc'], se=se_calib, T=T_GAS, ne=1.0)
        # Allamandola+89 WR density-of-states se
        se_wr = se_neutral_WR(pdef['Nc'], pdef['EA'], T_K=T_GAS)
        k_att_wr = attachment_rate_Bakes1994(pdef['Nc'], se=se_wr, T=T_GAS, ne=1.0)
        pah_rates[name] = dict(k_det_ref=k_det, k_ion1_ref=k1, k_ion2_ref=k2,
                               k_rec1_coeff=k_rec1, k_rec2_coeff=k_rec2,
                               k_att_calib=k_att_calib, k_att_wr=k_att_wr)
        print(f"  {name}: se_calib={se_calib:.4f}  se_WR={se_wr:.4e}  "
              f"k_att_calib={k_att_calib:.2e}  k_att_WR={k_att_wr:.2e}", flush=True)

    # ── Load digitised Andrews 2016 data for all PAHs ─────────────────────────
    _ext = _paths.get_external_data_path()
    _andrews_data: dict[str, dict] = {}
    for pah_key in _PAH_ORDER:
        _andrews_data[pah_key] = {}
        for Z, tag in [(-1, 'anion'), (0, 'neutral'), (1, 'cation'), (2, 'dication')]:
            f = _ext / f'{pah_key}HN_{tag}_andrews16.csv'
            if f.exists():
                _andrews_data[pah_key][Z] = np.loadtxt(f, delimiter=',')

    # ── Plot ──────────────────────────────────────────────────────────────────
    Z_vals   = [-1,  0,  1,  2]
    Z_colors = {-1: '#9467bd', 0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c'}
    Z_labels = {-1: r'$Z=-1$', 0: r'$Z=0$', 1: r'$Z=+1$', 2: r'$Z=+2$'}
    Z_ls     = {-1: '--', 0: '-', 1: '-', 2: '-.'}

    fig, axes = plt.subplots(
        1, 3, figsize=(14, 5),
        sharey=True, constrained_layout=True,
    )

    for col, name in enumerate(_PAH_ORDER):
        ax   = axes[col]
        pdef = PAH_DEFS[name]
        r    = pah_rates[name]

        fracs_calib = np.zeros((NGAMMA, 4))
        fracs_wr    = np.zeros((NGAMMA, 4))
        for ig, gamma in enumerate(GAMMA_GRID):
            G0 = gamma * NE_REF / np.sqrt(T_GAS)
            k_det     = r['k_det_ref']    * G0
            k_ion0    = r['k_ion1_ref']   * G0
            k_ion1    = r['k_ion2_ref']   * G0
            k_rec1_ne = r['k_rec1_coeff'] * NE_REF
            k_rec2_ne = r['k_rec2_coeff'] * NE_REF
            fracs_calib[ig] = _solve_charge(k_det, k_ion0, k_ion1, k_rec1_ne, k_rec2_ne,
                                            r['k_att_calib'] * NE_REF)
            fracs_wr[ig]    = _solve_charge(k_det, k_ion0, k_ion1, k_rec1_ne, k_rec2_ne,
                                            r['k_att_wr']    * NE_REF)

        for iz, Z in enumerate(Z_vals):
            lbl_c = (Z_labels[Z] + ' (calib)') if col == 0 else None
            lbl_w = (Z_labels[Z] + ' (WR)')    if col == 0 else None
            ax.plot(GAMMA_GRID, fracs_calib[:, iz],
                    color=Z_colors[Z], ls=Z_ls[Z], lw=1.5, alpha=0.5, label=lbl_c)
            ax.plot(GAMMA_GRID, fracs_wr[:, iz],
                    color=Z_colors[Z], ls=Z_ls[Z], lw=2.2, label=lbl_w)

        # Overlay digitised Andrews 2016 data
        for Z, arr in _andrews_data.get(name, {}).items():
            lbl = f'Andrews+16 $Z={Z:+d}$' if col == 0 else None
            ax.scatter(arr[:, 0], arr[:, 1],
                       color=Z_colors[Z], marker='o', s=30, zorder=5,
                       edgecolors='k', linewidths=0.4, label=lbl)

        ax.set_xscale('log')
        ax.set_xlim(GAMMA_GRID[0], GAMMA_GRID[-1])
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel(r'$\gamma = G_0\,\sqrt{T}\,/\,n_e$', fontsize=11)
        ax.set_title(pdef['label'], fontsize=11)
        if col == 0:
            ax.set_ylabel('Charge-state fraction', fontsize=11)
        ax.legend(fontsize=8, loc='upper left' if col == 0 else 'best')
        ax.tick_params(labelsize=9)
        ax.grid(True, which='both', alpha=0.25, lw=0.5)

    fig.suptitle(
        r'PAH charge distribution vs $\gamma=G_0\sqrt{T}/n_e$'
        r'  ($T=500\,\mathrm{K}$, $N_H=N_{H,0}$, Kurucz 15 kK)'
        '\n'
        r'Thick: WR $s_e$ (Allamandola+89, $k_r=10\,\mathrm{s}^{-1}$)'
        r'   Faded: calibrated $s_e$ (Andrews+16)'
        '\nCircles: Andrews et al. (2016) Fig. 8 [digitised]',
        fontsize=10,
    )

    out = _paths.get_plots_dir('pah_photophysics') / 'pah_charge_vs_gamma.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved → {out}")
    plt.show()


if __name__ == '__main__':
    main()
