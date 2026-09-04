"""
compare_c96h24_andrews16.py
===========================
Verify that our PAHdb C96H24 vibrational modes reproduce the microcanonical
k_IR(E), k_H(E), and k_H2(E) tables digitised from Andrews et al. (2016) for
circumcircumcoronene (C96H24, neutral, H_even_duo state, Nh=24).

Andrews (2016) state in their appendix that the PAH intrinsic IR spectra for
C96H24 neutral come from PAHdb, so our modes should be near-identical and the
reproduction should be near-exact modulo minor version/scaling differences.

Quantities compared
-------------------
k_IR(E) : IR cooling rate [s^-1] at internal energy E [eV].
    Our formula:  k_IR = Σ_i ε_i A_i <n_i(T_m)> / U(T_m)
    where T_m solves U_QHO(T_m) = E from the PAHdb mode set.

k_H(E)  : RRKM H-loss rate [s^-1] at internal energy E [eV].
    H_even_duo class (Nh=24, all duo positions): E_act = 4.60 eV, ΔS = 44.8 J/(mol·K)
    Formula: k_H = e × (k_B T_e / h) × exp(ΔS/R) × exp(−E_act / k_B T_e)
             T_e = T_m × (1 − 0.2 × E_act / E)   (Tielens 2005)

k_H2(E) : RRKM H2-loss rate [s^-1] at internal energy E [eV].
    H_even_duo H2 channel: E_act = 3.52 eV, ΔS = −53.1 J/(mol·K)

Data files
----------
external_data/kIR_C96H24_Andrews16.csv   — k_IR vs E [eV] for C96H24 neutral.
    Note: first 3 rows (E = 0.20–0.22 eV, non-monotonic) are bad digitisation
    artefacts and are dropped before comparison.
external_data/H-loss_C96H24_Andrews16.csv  — k_H  vs E [eV] (H_even_duo).
external_data/H2-loss_C96H24_Andrews16.csv — k_H2 vs E [eV] (H_even_duo H2 channel).

Usage
-----
    python -m models.PAH_photophysics.compare_c96h24_andrews16
"""

from __future__ import annotations

from pathlib import Path


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import brentq

from pycalima.models.PAH_photophysics.pah_mol_data import load_pah_modes
from pycalima import _paths
from pycalima.models.grain_size_config import get_model_data_dir

_EXT       = _paths.get_external_data_path()
_MODES     = get_model_data_dir() / 'PAH_states' / 'C96H24_0.dat'

# ── Physical constants ──────────────────────────────────────────────────────
KB_EV  = 8.61733326e-5   # eV/K
KB_J_K = 1.380649e-23    # J/K
H_J_S  = 6.62607015e-34  # J·s
R_GAS  = 8.31446261      # J/(mol·K)

# ── RRKM class: H_even_duo (Andrews 2016, Table B.3 footnote) ──────────────
E_ACT_H   = 4.60    # eV        — H-loss activation energy
DS_H      = 44.8    # J/(mol·K) — H-loss activation entropy
E_ACT_H2  = 3.52    # eV        — H2-loss activation energy
DS_H2     = -53.1   # J/(mol·K) — H2-loss activation entropy (negative = ordered TS)


# ── QHO helpers ────────────────────────────────────────────────────────────

def _u_qho(freq_ev: np.ndarray, T: float) -> float:
    """U_QHO(T) = Σ_i hν_i / (exp(hν_i / k_B T) − 1)  [eV]."""
    x   = freq_ev / (KB_EV * T)
    occ = np.where(x > 50.0, np.exp(-x) / (1 - np.exp(-x)), 1.0 / np.expm1(x))
    occ = np.where(np.isnan(occ) | np.isinf(occ), 0.0, occ)
    return float(np.sum(freq_ev * occ))


def _t_canonical(freq_ev: np.ndarray, E_eV: float) -> float:
    """Invert U_QHO(T_m) = E_eV to get canonical temperature T_m [K]."""
    return brentq(lambda T: _u_qho(freq_ev, T) - E_eV, 1.0, 15000.0)


def compute_kir(freq_ev: np.ndarray, einstein_A: np.ndarray, E_eV: float) -> float:
    """
    k_IR(E) = Σ_i ε_i A_i <n_i(T_m)> / U(T_m)   [s^-1]

    Evaluated at the canonical temperature T_m where U_QHO(T_m) = E_eV.
    """
    T_m = _t_canonical(freq_ev, E_eV)
    x   = freq_ev / (KB_EV * T_m)
    occ = np.where(x > 50.0, np.exp(-x) / (1 - np.exp(-x)), 1.0 / np.expm1(x))
    occ = np.where(np.isnan(occ) | np.isinf(occ), 0.0, occ)
    U   = float(np.sum(freq_ev * occ))
    if U <= 0:
        return 0.0
    return float(np.sum(freq_ev * einstein_A * occ) / U)


def compute_kh_rrkm(freq_ev: np.ndarray, E_eV: float,
                    E_act: float, dS: float) -> float:
    """
    RRKM Arrhenius H-loss rate at internal energy E_eV [eV]:

        k_H = e × (k_B T_e / h) × exp(ΔS/R) × exp(−E_act / k_B T_e)
        T_e = T_m × (1 − 0.2 × E_act / E)      (Tielens 2005 effective temperature)
    """
    if E_eV <= E_act:
        return 0.0
    T_m = _t_canonical(freq_ev, E_eV)
    T_e = T_m * (1.0 - 0.2 * E_act / E_eV)
    if T_e <= 0:
        return 0.0
    return (np.e
            * (KB_J_K * T_e / H_J_S)
            * np.exp(dS / R_GAS)
            * np.exp(-E_act / (KB_EV * T_e)))


# ── Load data ──────────────────────────────────────────────────────────────

def load_andrews_table(path: Path, skip_rows: int = 0) -> np.ndarray:
    """Load a two-column CSV (E [eV], rate [s^-1])."""
    data = np.loadtxt(path, delimiter=',')
    return data[skip_rows:]   # drop bad leading rows if requested


def main() -> None:
    print("Loading C96H24 PAHdb modes ...", flush=True)
    freq_ev, einstein_A = load_pah_modes(str(_MODES))
    print(f"  {len(freq_ev)} modes, "
          f"ν = {freq_ev.min()*8065:.0f}–{freq_ev.max()*8065:.0f} cm^-1\n")

    # Andrews tables.
    # kIR: first 3 rows have E = 0.20-0.22 eV with non-monotonic, physically
    # impossible values — bad digitisation artefacts; skip them.
    kir_a  = load_andrews_table(_EXT / 'kIR_C96H24_Andrews16.csv',   skip_rows=3)
    kh_a   = load_andrews_table(_EXT / 'H-loss_C96H24_Andrews16.csv')
    kh2_a  = load_andrews_table(_EXT / 'H2-loss_C96H24_Andrews16.csv')

    # ── k_IR comparison ────────────────────────────────────────────────────
    print("k_IR comparison  (C96H24 PAHdb  vs  Andrews 2016)")
    print(f"{'E [eV]':>10}  {'T_m [K]':>8}  {'kIR_PAHdb':>13}  "
          f"{'kIR_Andrews':>13}  {'ratio':>8}  {'log10':>8}")
    kir_ratios = []
    for E, k_a in kir_a:
        T_m = _t_canonical(freq_ev, E)
        k_p = compute_kir(freq_ev, einstein_A, E)
        r   = k_p / k_a
        kir_ratios.append(np.log10(r))
        print(f"  {E:8.3f}  {T_m:8.1f}  {k_p:13.4e}  {k_a:13.4e}  "
              f"{r:8.3f}  {np.log10(r):+8.3f}")
    print(f"  → mean log10(PAHdb/Andrews) = {np.mean(kir_ratios):+.3f} dex  "
          f"(RMS = {np.sqrt(np.mean(np.array(kir_ratios)**2)):.3f} dex)\n")

    # ── k_H comparison ─────────────────────────────────────────────────────
    print(f"k_H comparison  (RRKM H_even_duo: E_act={E_ACT_H} eV, "
          f"ΔS={DS_H} J/mol/K  vs  Andrews 2016)")
    print(f"{'E [eV]':>10}  {'T_m [K]':>8}  {'T_e [K]':>8}  {'kH_RRKM':>13}  "
          f"{'kH_Andrews':>13}  {'log10':>8}")
    kh_ratios = []
    for E, k_a in kh_a:
        T_m = _t_canonical(freq_ev, E)
        T_e = T_m * (1.0 - 0.2 * E_ACT_H / E)
        k_r = compute_kh_rrkm(freq_ev, E, E_ACT_H, DS_H)
        if k_r > 0 and k_a > 0:
            log_r = np.log10(k_r / k_a)
            kh_ratios.append(log_r)
            print(f"  {E:8.3f}  {T_m:8.1f}  {T_e:8.1f}  {k_r:13.4e}  "
                  f"{k_a:13.4e}  {log_r:+8.3f}")
    print(f"  → mean log10(RRKM/Andrews) = {np.mean(kh_ratios):+.3f} dex  "
          f"(RMS = {np.sqrt(np.mean(np.array(kh_ratios)**2)):.3f} dex)\n")

    # ── k_H2 comparison ────────────────────────────────────────────────────
    print(f"k_H2 comparison  (RRKM H_even_duo H2 channel: E_act={E_ACT_H2} eV, "
          f"ΔS={DS_H2} J/mol/K  vs  Andrews 2016)")
    print(f"{'E [eV]':>10}  {'T_m [K]':>8}  {'T_e [K]':>8}  {'kH2_RRKM':>13}  "
          f"{'kH2_Andrews':>13}  {'log10':>8}")
    kh2_ratios = []
    for E, k_a in kh2_a:
        T_m  = _t_canonical(freq_ev, E)
        T_e  = T_m * (1.0 - 0.2 * E_ACT_H2 / E)
        k_r  = compute_kh_rrkm(freq_ev, E, E_ACT_H2, DS_H2)
        if k_r > 0 and k_a > 0:
            log_r = np.log10(k_r / k_a)
            kh2_ratios.append(log_r)
            print(f"  {E:8.3f}  {T_m:8.1f}  {T_e:8.1f}  {k_r:13.4e}  "
                  f"{k_a:13.4e}  {log_r:+8.3f}")
    if kh2_ratios:
        print(f"  → mean log10(RRKM/Andrews) = {np.mean(kh2_ratios):+.3f} dex  "
              f"(RMS = {np.sqrt(np.mean(np.array(kh2_ratios)**2)):.3f} dex)\n")

    # ── Plot ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 9))
    gs  = gridspec.GridSpec(2, 3, hspace=0.38, wspace=0.35)

    ax_kir  = fig.add_subplot(gs[0, 0])
    ax_kh   = fig.add_subplot(gs[0, 1])
    ax_kh2  = fig.add_subplot(gs[0, 2])
    ax_rir  = fig.add_subplot(gs[1, 0])
    ax_rkh  = fig.add_subplot(gs[1, 1])
    ax_rkh2 = fig.add_subplot(gs[1, 2])

    # Dense energy grids for smooth RRKM curves
    E_kir_d  = np.linspace(kir_a[0, 0],  kir_a[-1, 0],  120)
    E_kh_d   = np.linspace(kh_a[0, 0],   kh_a[-1, 0],   120)
    E_kh2_d  = np.linspace(kh2_a[0, 0],  kh2_a[-1, 0],  120)

    kir_d  = np.array([compute_kir(freq_ev, einstein_A, E) for E in E_kir_d])
    kh_d   = np.array([compute_kh_rrkm(freq_ev, E, E_ACT_H,  DS_H)  for E in E_kh_d])
    kh2_d  = np.array([compute_kh_rrkm(freq_ev, E, E_ACT_H2, DS_H2) for E in E_kh2_d])

    # k_IR panel
    ax_kir.semilogy(E_kir_d,      kir_d,         'b-',  lw=2,  label='PAHdb C96H24')
    ax_kir.semilogy(kir_a[:, 0],  kir_a[:, 1],   'ko',  ms=5,  label='Andrews+16')
    ax_kir.set_xlabel('Internal energy E [eV]')
    ax_kir.set_ylabel(r'$k_\mathrm{IR}$ [s$^{-1}$]')
    ax_kir.set_title(r'IR cooling rate')
    ax_kir.legend(fontsize=9)
    ax_kir.grid(True, alpha=0.3)

    # k_H panel
    ax_kh.semilogy(E_kh_d,       kh_d,          'r-',  lw=2,
                   label=fr'RRKM  ($E_a={E_ACT_H}$, $\Delta S={DS_H}$)')
    ax_kh.semilogy(kh_a[:, 0],   kh_a[:, 1],    'ko',  ms=5, label='Andrews+16')
    ax_kh.set_xlabel('Internal energy E [eV]')
    ax_kh.set_ylabel(r'$k_H$ [s$^{-1}$]')
    ax_kh.set_title(r'H-loss rate  (H$_\mathrm{even,duo}$)')
    ax_kh.legend(fontsize=9)
    ax_kh.grid(True, alpha=0.3)

    # k_H2 panel
    ax_kh2.semilogy(E_kh2_d,     kh2_d,         'g-',  lw=2,
                    label=fr'RRKM  ($E_a={E_ACT_H2}$, $\Delta S={DS_H2}$)')
    ax_kh2.semilogy(kh2_a[:, 0], kh2_a[:, 1],   'ko',  ms=5, label='Andrews+16')
    ax_kh2.set_xlabel('Internal energy E [eV]')
    ax_kh2.set_ylabel(r'$k_{H_2}$ [s$^{-1}$]')
    ax_kh2.set_title(r'H$_2$-loss rate  (H$_\mathrm{even,duo}$)')
    ax_kh2.legend(fontsize=9)
    ax_kh2.grid(True, alpha=0.3)

    # Residual panels
    kir_p  = np.array([compute_kir(freq_ev, einstein_A, E) for E in kir_a[:, 0]])
    kh_p   = np.array([compute_kh_rrkm(freq_ev, E, E_ACT_H,  DS_H)  for E in kh_a[:, 0]])
    kh2_p  = np.array([compute_kh_rrkm(freq_ev, E, E_ACT_H2, DS_H2) for E in kh2_a[:, 0]])

    for ax, E_pts, ratio, color, title in [
        (ax_rir,  kir_a[:, 0],  kir_p  / kir_a[:, 1],  'b',
         r'$k_\mathrm{IR}$ residuals'),
        (ax_rkh,  kh_a[:, 0],   kh_p   / kh_a[:, 1],   'r',
         r'$k_H$ residuals'),
        (ax_rkh2, kh2_a[:, 0],  kh2_p  / kh2_a[:, 1],  'g',
         r'$k_{H_2}$ residuals'),
    ]:
        mask = ratio > 0
        ax.axhline(0,     color='k', lw=0.8)
        ax.axhline( 0.30, color='k', lw=0.5, ls=':', label='±factor 2')
        ax.axhline(-0.30, color='k', lw=0.5, ls=':')
        ax.plot(E_pts[mask], np.log10(ratio[mask]), f'{color}o-', ms=5)
        ax.set_xlabel('Internal energy E [eV]')
        ax.set_ylabel(r'$\log_{10}(\mathrm{model} / k_\mathrm{A16})$')
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.0, 1.0)

    fig.suptitle(
        r'C$_{96}$H$_{24}$ (neutral, H$_\mathrm{even,duo}$, $N_H=24$) — '
        'PAHdb modes vs Andrews+16\n'
        r'$k_\mathrm{IR}$: PAHdb C96H24 modes.  '
        r'$k_H$, $k_{H_2}$: RRKM Tielens (2005) with Andrews (2016) parameters.',
        fontsize=10,
    )

    out = _paths.get_plots_dir('pah_photophysics') / 'c96h24_kir_kh_vs_andrews16.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Figure saved → {out}")
    plt.close(fig)


if __name__ == '__main__':
    main()
