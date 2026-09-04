"""
compare_c96_states_andrews16.py
================================
Compare PAHdb-derived k_IR(E) and RRKM k_H(E) against digitised Andrews (2016)
tables for three hydrogenation states of circumcircumcoronene C96:

  C96H23 — odd H, H_odd class       (E_act = 4.10 eV, ΔS = +55.6 J/mol/K)
  C96H24 — even H, H_even_duo class (E_act = 4.60 eV, ΔS = +44.8 J/mol/K)
            H2-loss channel          (E_act = 3.52 eV, ΔS = −53.1 J/mol/K)
  C96H25 — super-H, superH_neutral  (E_act = 1.40 eV, ΔS = +55.6 J/mol/K)

All model curves use C96H24_0.dat PAHdb modes (no C96H23 or C96H25 files
available; the one-H difference has negligible impact on k_IR and the E-T
relation used for H_even_duo and H_odd rates).

Andrews data files
------------------
  kIR_C96H23_Andrews16.csv   — only 3 usable points at E > 35 eV (most rows
                                 have E=0 due to a digitisation failure).
  kIR_C96H24_Andrews16.csv   — 16 rows; first 3 (E=0.20–0.22 eV) are artefacts.
  kIR_C96H25_Andrews16.csv   — 16 rows; first 2 (E=0.07–0.22 eV) are artefacts.
  H-loss_C96H23_Andrews16.csv  — 18 points, E = 11.4–39.8 eV  (H_odd)
  H-loss_C96H24_Andrews16.csv  — 18 points, E = 14.5–39.6 eV  (H_even_duo)
  H2-loss_C96H24_Andrews16.csv — 15 points, E = 13.0–39.7 eV  (H_even_duo)
  H-loss_C96H25_Andrews16.csv  — 24 points, E =  2.6–39.7 eV  (superH_neutral)
                                  RRKM Arrhenius fails for this class
                                  (required ΔS varies −63 → +33 J/mol/K).

Usage
-----
    python -m models.PAH_photophysics.compare_c96_states_andrews16
"""

from __future__ import annotations

from pathlib import Path


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import brentq

from pycalima.models.PAH_photophysics.pah_mol_data import load_pah_modes
from pycalima import _paths
from pycalima.models.grain_size_config import get_model_data_dir

_EXT    = _paths.get_external_data_path()
_STATES = get_model_data_dir() / 'PAH_states'

# ── Physical constants ──────────────────────────────────────────────────────
KB_EV  = 8.61733326e-5
KB_J_K = 1.380649e-23
H_J_S  = 6.62607015e-34
R_GAS  = 8.31446261

# ── RRKM parameters (Andrews 2016 / Tielens 2005) ──────────────────────────
# modes_file: species-specific PAHdb UB3LYP modes (doublets for Nh=23,25)
#   C96H23 neutral → uid=693 (2-A, n_solo=11, n_duo=12)
#   C96H24 neutral → uid=108 (1-AG, n_solo=12, n_duo=12)
#   C96H25 neutral → uid=690 (2-A, n_solo=12, n_duo=12)
STATES = {
    'C96H23': dict(
        label=r'$\mathrm{C_{96}H_{23}}$ (H$_\mathrm{odd}$)',
        color='#e67e22',    # orange
        modes_file='C96H23_0.dat',
        E_act_H=4.10, dS_H=55.6,
        E_act_H2=100.0, dS_H2=0.0,   # no H2-loss channel
        kir_file='kIR_C96H23_Andrews16.csv',   kir_skip_zeros=True,
        kh_file='H-loss_C96H23_Andrews16.csv', kh2_file=None,
    ),
    'C96H24': dict(
        label=r'$\mathrm{C_{96}H_{24}}$ (H$_\mathrm{even,duo}$)',
        color='#2980b9',    # blue
        modes_file='C96H24_0.dat',
        E_act_H=4.60, dS_H=44.8,
        E_act_H2=3.52, dS_H2=-53.1,
        kir_file='kIR_C96H24_Andrews16.csv',   kir_skip_first=3,
        kh_file='H-loss_C96H24_Andrews16.csv', kh2_file='H2-loss_C96H24_Andrews16.csv',
    ),
    'C96H25': dict(
        label=r'$\mathrm{C_{96}H_{25}}$ (super-H$_\mathrm{neutral}$)',
        color='#27ae60',    # green
        modes_file='C96H25_0.dat',
        E_act_H=1.40, dS_H=55.6,
        E_act_H2=100.0, dS_H2=0.0,
        kir_file='kIR_C96H25_Andrews16.csv',   kir_skip_first=2,
        kh_file='H-loss_C96H25_Andrews16.csv', kh2_file=None,
    ),
}


# ── QHO / RRKM helpers ─────────────────────────────────────────────────────

def _u_qho(freq_ev, T):
    x = freq_ev / (KB_EV * T)
    occ = np.where(x > 50.0, np.exp(-x) / (1 - np.exp(-x)), 1.0 / np.expm1(x))
    occ = np.where(np.isnan(occ) | np.isinf(occ), 0.0, occ)
    return float(np.sum(freq_ev * occ))


def _t_canonical(freq_ev, E):
    return brentq(lambda T: _u_qho(freq_ev, T) - E, 1.0, 15000.0)


def compute_kir_curve(freq_ev, einstein_A, E_arr):
    """k_IR(E) = Σ ε_i A_i <n_i(T_m)> / U(T_m)  [s^-1]."""
    out = np.zeros(len(E_arr))
    for i, E in enumerate(E_arr):
        T_m = _t_canonical(freq_ev, E)
        x   = freq_ev / (KB_EV * T_m)
        occ = np.where(x > 50.0, np.exp(-x) / (1 - np.exp(-x)), 1.0 / np.expm1(x))
        occ = np.where(np.isnan(occ) | np.isinf(occ), 0.0, occ)
        U   = float(np.sum(freq_ev * occ))
        out[i] = float(np.sum(freq_ev * einstein_A * occ) / U) if U > 0 else 0.0
    return out


def compute_kh_curve(freq_ev, E_arr, E_act, dS):
    """RRKM Arrhenius k_H(E) [s^-1] (Tielens 2005)."""
    out = np.zeros(len(E_arr))
    for i, E in enumerate(E_arr):
        if E <= E_act:
            continue
        T_m = _t_canonical(freq_ev, E)
        T_e = T_m * (1.0 - 0.2 * E_act / E)
        if T_e <= 0:
            continue
        out[i] = (np.e
                  * (KB_J_K * T_e / H_J_S)
                  * np.exp(dS / R_GAS)
                  * np.exp(-E_act / (KB_EV * T_e)))
    return out


# ── Data loading ────────────────────────────────────────────────────────────

def load_table(path, skip_first=0, skip_zeros=False):
    """Load two-column CSV; optionally drop leading rows or rows with E=0."""
    data = np.loadtxt(path, delimiter=',')
    if skip_zeros:
        data = data[data[:, 0] > 0]
    elif skip_first > 0:
        data = data[skip_first:]
    return data


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    # ── Load per-species PAHdb modes ─────────────────────────────────────────
    modes = {}
    for key, cfg in STATES.items():
        path = _STATES / cfg['modes_file']
        freq_ev, einstein_A = load_pah_modes(str(path))
        modes[key] = (freq_ev, einstein_A)
        mult = 2 if key != 'C96H24' else 1
        print(f"  {key}  ({cfg['modes_file']}, mult={mult}): "
              f"{len(freq_ev)} modes, "
              f"ν = {freq_ev.min()*8065:.0f}–{freq_ev.max()*8065:.0f} cm^-1")
    print()

    # ── Load Andrews tables ──────────────────────────────────────────────────
    tables = {}
    for key, cfg in STATES.items():
        t = {}

        kir_path = _EXT / cfg['kir_file']
        if cfg.get('kir_skip_zeros'):
            t['kir'] = load_table(kir_path, skip_zeros=True)
        else:
            t['kir'] = load_table(kir_path, skip_first=cfg.get('kir_skip_first', 0))

        t['kh'] = load_table(_EXT / cfg['kh_file'])
        t['kh2'] = load_table(_EXT / cfg['kh2_file']) if cfg['kh2_file'] else None

        tables[key] = t

    # ── Dense E grids for smooth model curves ───────────────────────────────
    E_kir = np.linspace(0.3, 40.0, 200)
    E_kh  = np.linspace(1.5, 40.0, 200)
    E_kh2 = np.linspace(3.0, 40.0, 200)

    # ── Print comparison tables ──────────────────────────────────────────────
    for key, cfg in STATES.items():
        t = tables[key]
        freq_ev, einstein_A = modes[key]
        # Strip TeX markup for plain-text output. Hoisted out of the f-string:
        # a backslash inside a replacement field requires Python >= 3.12.
        label_plain = cfg['label']
        for _ch in ('$', '\\', '{', '}'):
            label_plain = label_plain.replace(_ch, '')
        print(f"{'─'*65}")
        print(f"  {label_plain}")
        print(f"{'─'*65}")

        kir_pts = compute_kir_curve(freq_ev, einstein_A, t['kir'][:, 0])
        ratios  = np.log10(kir_pts / t['kir'][:, 1])
        print(f"  k_IR:  mean = {np.mean(ratios):+.3f} dex  "
              f"RMS = {np.sqrt(np.mean(ratios**2)):.3f} dex  "
              f"({len(ratios)} points)")

        kh_pts = compute_kh_curve(freq_ev, t['kh'][:, 0], cfg['E_act_H'], cfg['dS_H'])
        mask   = (kh_pts > 0) & (t['kh'][:, 1] > 0)
        if mask.any():
            r = np.log10(kh_pts[mask] / t['kh'][mask, 1])
            print(f"  k_H:   mean = {np.mean(r):+.3f} dex  "
                  f"RMS = {np.sqrt(np.mean(r**2)):.3f} dex  "
                  f"({mask.sum()} points)"
                  + ("  ← RRKM fails (variable ΔS required)" if key == 'C96H25' else ""))

        if t['kh2'] is not None:
            kh2_pts = compute_kh_curve(freq_ev, t['kh2'][:, 0],
                                       cfg['E_act_H2'], cfg['dS_H2'])
            mask2   = (kh2_pts > 0) & (t['kh2'][:, 1] > 0)
            if mask2.any():
                r2 = np.log10(kh2_pts[mask2] / t['kh2'][mask2, 1])
                print(f"  k_H2:  mean = {np.mean(r2):+.3f} dex  "
                      f"RMS = {np.sqrt(np.mean(r2**2)):.3f} dex  "
                      f"({mask2.sum()} points)")
        print()

    # ── Figure: 2 rows × 3 columns ──────────────────────────────────────────
    # Row 0: absolute rates  (k_IR | k_H | k_H2)
    # Row 1: residuals       (log10 model/Andrews)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8),
                             gridspec_kw={'hspace': 0.42, 'wspace': 0.32})
    ax_kir,  ax_kh,  ax_kh2  = axes[0]
    ax_rkir, ax_rkh, ax_rkh2 = axes[1]

    # ── Overlay Andrews data and residuals for each state ───────────────────
    for key, cfg in STATES.items():
        t               = tables[key]
        freq_ev, ein_A  = modes[key]
        color           = cfg['color']
        label           = cfg['label']

        # k_IR: species-specific model curve + Andrews points
        kir_mod = compute_kir_curve(freq_ev, ein_A, E_kir)
        ax_kir.semilogy(E_kir, kir_mod, '-', color=color, lw=2)
        ax_kir.semilogy(t['kir'][:, 0], t['kir'][:, 1], 'o',
                        color=color, ms=5, zorder=4)
        kir_pts = compute_kir_curve(freq_ev, ein_A, t['kir'][:, 0])
        ax_rkir.plot(t['kir'][:, 0],
                     np.log10(kir_pts / t['kir'][:, 1]),
                     'o-', color=color, ms=4, lw=1.2)

        # k_H: model curve + Andrews points
        kh_mod = compute_kh_curve(freq_ev, E_kh, cfg['E_act_H'], cfg['dS_H'])
        valid  = kh_mod > 0
        ax_kh.semilogy(E_kh[valid], kh_mod[valid], '-', color=color, lw=2,
                       label=label)
        ax_kh.semilogy(t['kh'][:, 0], t['kh'][:, 1], 'o',
                       color=color, ms=5, zorder=4)

        kh_pts = compute_kh_curve(freq_ev, t['kh'][:, 0],
                                   cfg['E_act_H'], cfg['dS_H'])
        mask = (kh_pts > 0) & (t['kh'][:, 1] > 0)
        if mask.any():
            ax_rkh.plot(t['kh'][mask, 0],
                        np.log10(kh_pts[mask] / t['kh'][mask, 1]),
                        'o-', color=color, ms=4, lw=1.2, label=label)

        # k_H2 (C96H24 only)
        if t['kh2'] is not None:
            kh2_mod = compute_kh_curve(freq_ev, E_kh2,
                                       cfg['E_act_H2'], cfg['dS_H2'])
            valid2  = kh2_mod > 0
            ax_kh2.semilogy(E_kh2[valid2], kh2_mod[valid2], '-',
                            color=color, lw=2,
                            label=fr'{label}  ($E_a={cfg["E_act_H2"]}$ eV)')
            ax_kh2.semilogy(t['kh2'][:, 0], t['kh2'][:, 1], 'o',
                            color=color, ms=5, zorder=4)
            kh2_pts = compute_kh_curve(freq_ev, t['kh2'][:, 0],
                                        cfg['E_act_H2'], cfg['dS_H2'])
            mask2 = (kh2_pts > 0) & (t['kh2'][:, 1] > 0)
            if mask2.any():
                ax_rkh2.plot(t['kh2'][mask2, 0],
                             np.log10(kh2_pts[mask2] / t['kh2'][mask2, 1]),
                             'o-', color=color, ms=4, lw=1.2)

    # ── Reference lines on residual panels ──────────────────────────────────
    for ax in (ax_rkir, ax_rkh, ax_rkh2):
        ax.axhline(0,     color='k', lw=0.9)
        ax.axhline(+0.30, color='k', lw=0.5, ls=':', alpha=0.7)
        ax.axhline(-0.30, color='k', lw=0.5, ls=':', alpha=0.7,
                   label='±factor 2')
        ax.set_xlabel('Internal energy  E  [eV]', fontsize=10)
        ax.set_ylabel(r'$\log_{10}(\mathrm{model}/k_\mathrm{A16})$', fontsize=10)
        ax.grid(True, alpha=0.25, lw=0.5)

    # Force generous y-limits on k_H residuals to show superH divergence
    ax_rkh.set_ylim(-1.5, 8.0)

    # ── Axis labels / titles ─────────────────────────────────────────────────
    for ax in (ax_kir, ax_kh, ax_kh2):
        ax.set_xlabel('Internal energy  E  [eV]', fontsize=10)
        ax.grid(True, which='both', alpha=0.25, lw=0.5)
        ax.set_xlim(0, 41)

    ax_kir.set_ylabel(r'$k_\mathrm{IR}$  [s$^{-1}$]', fontsize=10)
    ax_kh.set_ylabel(r'$k_H$  [s$^{-1}$]', fontsize=10)
    ax_kh2.set_ylabel(r'$k_{H_2}$  [s$^{-1}$]', fontsize=10)

    ax_kir.set_title(r'IR cooling rate  $k_\mathrm{IR}(E)$', fontsize=10)
    ax_kh.set_title(r'H-loss rate  $k_H(E)$', fontsize=10)
    ax_kh2.set_title(r'H$_2$-loss rate  $k_{H_2}(E)$  —  C$_{96}$H$_{24}$ only',
                     fontsize=10)
    ax_rkir.set_title(r'$k_\mathrm{IR}$ residuals', fontsize=10)
    ax_rkh.set_title(r'$k_H$ residuals', fontsize=10)
    ax_rkh2.set_title(r'$k_{H_2}$ residuals', fontsize=10)

    # ── Legends ──────────────────────────────────────────────────────────────
    # k_IR: per-state model line + Andrews points
    handles_kir = []
    for key, cfg in STATES.items():
        handles_kir += [
            plt.Line2D([0], [0], color=cfg['color'], lw=2,
                       label=f'{key} model'),
            plt.Line2D([0], [0], color=cfg['color'], lw=0, marker='o', ms=6,
                       label=f'{key} Andrews+16'),
        ]
    ax_kir.legend(handles=handles_kir, fontsize=8, loc='lower right')

    # k_H: separate model per state (different RRKM params)
    handles_kh = []
    for key, cfg in STATES.items():
        e_str = f"$E_a={cfg['E_act_H']}$"
        handles_kh += [
            plt.Line2D([0], [0], color=cfg['color'], lw=2,
                       label=f'{key} model  ({e_str} eV)'),
            plt.Line2D([0], [0], color=cfg['color'], lw=0, marker='o', ms=6,
                       label=f'{key} Andrews+16'),
        ]
    ax_kh.legend(handles=handles_kh, fontsize=7.5, loc='upper left')

    # residuals legend
    ax_rkir.legend(fontsize=8)
    ax_rkh.legend(fontsize=8, loc='upper right')
    ax_rkh2.legend(fontsize=8)

    # Annotation on superH failure
    ax_rkh.annotate(
        r'RRKM fails for super-H:' '\n'
        r'required $\Delta S$ varies' '\n'
        r'$-63 \to +33$ J mol$^{-1}$ K$^{-1}$',
        xy=(5.0, 6.0), xytext=(15, 3.5),
        fontsize=7.5, color=STATES['C96H25']['color'],
        arrowprops=dict(arrowstyle='->', color=STATES['C96H25']['color'], lw=1),
    )

    fig.suptitle(
        r'C$_{96}$ hydrogenation states — RRKM model (PAHdb v4.00 species-specific modes) '
        'vs Andrews et al. (2016)\n'
        r'Solid lines: model.  Circles: Andrews+16 digitised data.',
        fontsize=10,
    )

    out = _paths.get_plots_dir('pah_photophysics') / 'c96_states_kir_kh_vs_andrews16.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Figure saved → {out}")
    plt.close(fig)


if __name__ == '__main__':
    main()
