"""
compare_c24_G0_rates.py
=======================
Compare C24 H-loss and H2-loss photodissociation rates (RRKM, GD89 energy
distribution) against the polynomial fits from Andrews et al. (2016) Table B.1
for coronene (C24H12) neutral states.

Two UV cross-sections are compared:
  - Li & Draine (2001) : grain-size extrapolation to C24 radius
  - Malloci et al. (2007): molecular TD-DFT for coronene neutral

Three RRKM classes evaluated with C24H12 PAHdb modes:
    H_odd          → C24H11 (Z=0, Nh=11),  not in Table B.1 neutral
    H_even_duo     → C24H12 (Z=0, Nh=12),  compare with Table B.1
    superH_neutral → C24H13 (Z=0, Nh=13),  compare with Table B.1

Output: compare_c24_G0_rates.png
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

from pycalima.models.PAH_photophysics.pah_temperature import (
    compute_base_g0,
    get_absorption_cross_section,
)
from pycalima.models.PAH_photophysics.pah_radiation import load_kurucz_u_E
from pycalima.models.PAH_photophysics.pah_charge_utils import afromNc
from pycalima.models.PAH_photophysics.reproduce_andrews16_fig9 import (
    _STATES_DIR, _HV_EV, _worker,
)

_EXT = ROOT / 'external_data'

# ── Config ────────────────────────────────────────────────────────────────────
Nc   = 24
IP1  = 7.20
IP2  = 11.50
MODES_FILE = str(_STATES_DIR / 'C24H12_0.dat')
G0_GRID = np.logspace(0, 5, 20)


# ── Cross-section loaders ─────────────────────────────────────────────────────

def xsect_li_draine() -> np.ndarray:
    """Li & Draine (2001) cross-section, sorted by energy [E_eV, sigma_cm2]."""
    a0_cm = afromNc(Nc)
    w_cm, C_abs = get_absorption_cross_section(0, a0_cm)
    E = _HV_EV / w_cm
    idx = np.argsort(E)
    return np.column_stack([E[idx], C_abs[idx]])


def xsect_malloci(charge: int = 0) -> np.ndarray:
    """
    Malloci et al. (2007) Cagliari TD-DFT cross-section for coronene.
    charge=0 → neutral, -1 → anion, +1 → cation.
    Returns [E_eV, sigma_cm2], energy 0–30 eV at 0.05 eV spacing.
    """
    label = {0: 'neutral', -1: 'anion', 1: 'cation'}[charge]
    fname = _EXT / f'malloci_coronene_{label}_sigma.dat'
    data = np.loadtxt(fname, comments='#')
    E   = data[:, 0]               # eV
    sig = data[:, 1] * 1e-18       # Mb → cm^2
    sig = np.maximum(sig, 0.0)     # clamp tiny negatives
    idx = np.argsort(E)
    return np.column_stack([E[idx], sig[idx]])


# ── Andrews Table B.1 polynomial fits ────────────────────────────────────────

def poly_rate(coeffs, G0: np.ndarray) -> np.ndarray:
    x = np.log10(G0)
    return 10.0 ** sum(float(coeffs[i]) * x**i for i in range(len(coeffs)))


def load_b1_poly(Z: int, Nh: int, loss: str):
    """Return polynomial coefficients from Table B.1, or None."""
    import pandas as pd
    df = pd.read_csv(_EXT / 'table_B1_coronene.csv')
    row = df[(df['Z'] == Z) & (df['NH'] == Nh)]
    if row.empty:
        return None
    prefix = 'H_loss' if loss == 'H' else 'H2_loss'
    vals = row.iloc[0][[f'{prefix}_p{i}' for i in range(5)]].values.astype(float)
    return None if np.isnan(vals[0]) else vals


# ── Run workers ───────────────────────────────────────────────────────────────

def run_workers(xsect: np.ndarray, G0_base: float, label: str):
    tasks = [
        ('C24', MODES_FILE, xsect, float(G0), G0_base, IP1, IP2)
        for G0 in G0_GRID
    ]
    print(f"  [{label}] Running {len(tasks)} workers ...", flush=True)
    with Pool(max(1, cpu_count() - 1)) as pool:
        results = pool.map(_worker, tasks)
    classes = ['H_odd', 'H_even_duo', 'superH_neutral']
    G0_arr = np.array([r[1] for r in results])
    YH  = {cls: np.array([r[2][cls][0] for r in results]) for cls in classes}
    YH2 = {cls: np.array([r[2][cls][1] for r in results]) for cls in classes}
    return G0_arr, YH, YH2


# ── Plotting helper ───────────────────────────────────────────────────────────

COLORS = {
    'H_odd':          '#2196F3',
    'H_even_duo':     '#4CAF50',
    'superH_neutral': '#FF5722',
}
TITLES = [
    r'C$_{24}$H$_{11}$ (H-odd, $N_H{=}11$)' + '\n(no Table B.1 neutral entry)',
    r'C$_{24}$H$_{12}$ (parent, $N_H{=}12$)',
    r'C$_{24}$H$_{13}$ (super-H, $N_H{=}13$)',
]
CLASSES = ['H_odd', 'H_even_duo', 'superH_neutral']


def _plot_panel_main(ax, cls, G0_ld, YH_ld, YH2_ld, G0_m, YH_m, YH2_m,
                     p_H, p_H2, G0_fit):
    """Top row: k vs G0."""
    if p_H is not None:
        ax.loglog(G0_fit, poly_rate(p_H,  G0_fit), '-',  color='k', lw=1.8,
                  label='Andrews B.1 H-loss')
    if p_H2 is not None:
        ax.loglog(G0_fit, poly_rate(p_H2, G0_fit), ':',  color='k', lw=1.8,
                  label=r'Andrews B.1 H$_2$-loss')

    ax.loglog(G0_ld, YH_ld[cls],  'o-',  color=COLORS[cls], ms=4,
              label='Li&Draine H-loss')
    if np.any(YH2_ld[cls] > 0):
        ax.loglog(G0_ld, YH2_ld[cls], 's--', color=COLORS[cls], ms=4,
                  alpha=0.7, mfc='white', label=r'Li&Draine H$_2$-loss')

    ax.loglog(G0_m, YH_m[cls],  '^-',  color=COLORS[cls], ms=5,
              alpha=0.6, mfc='white', label='Malloci H-loss')
    if np.any(YH2_m[cls] > 0):
        ax.loglog(G0_m, YH2_m[cls], 'v--', color=COLORS[cls], ms=5,
                  alpha=0.5, mfc='white', label=r'Malloci H$_2$-loss')

    ax.set_ylabel(r'$k$ [s$^{-1}$]', fontsize=10)
    ax.set_xticklabels([])
    ax.legend(fontsize=7, ncol=1)
    ax.grid(True, alpha=0.3, which='both')


def _plot_panel_res(ax, cls, G0_ld, YH_ld, YH2_ld, G0_m, YH_m, YH2_m,
                    p_H, p_H2):
    """Bottom row: log10(model/Andrews) residuals."""
    plotted = False
    if p_H is not None:
        k_a_H = poly_rate(p_H, G0_ld)
        res_ld = np.log10(YH_ld[cls] / k_a_H)
        res_m  = np.log10(YH_m[cls]  / poly_rate(p_H, G0_m))
        mu_ld = np.nanmean(res_ld)
        mu_m  = np.nanmean(res_m)
        ax.semilogx(G0_ld, res_ld, 'o-',  color=COLORS[cls], ms=4,
                    label=f'Li&Draine  {mu_ld:+.1f} dex')
        ax.semilogx(G0_m,  res_m,  '^-',  color=COLORS[cls], ms=5,
                    alpha=0.6, mfc='white', label=f'Malloci  {mu_m:+.1f} dex')
        print(f"  {cls} H-loss:  Li&Draine {mu_ld:+.2f} dex | Malloci {mu_m:+.2f} dex")
        plotted = True
    if p_H2 is not None:
        k_a_H2 = poly_rate(p_H2, G0_ld)
        res_ld2 = np.log10(YH2_ld[cls] / k_a_H2)
        res_m2  = np.log10(YH2_m[cls]  / poly_rate(p_H2, G0_m))
        mu_ld2 = np.nanmean(res_ld2)
        mu_m2  = np.nanmean(res_m2)
        ax.semilogx(G0_ld, res_ld2, 's--', color=COLORS[cls], ms=4, alpha=0.7,
                    mfc='white', label=f'Li&Draine H$_2$  {mu_ld2:+.1f} dex')
        ax.semilogx(G0_m,  res_m2,  'v--', color=COLORS[cls], ms=5, alpha=0.5,
                    mfc='none',  label=f'Malloci H$_2$  {mu_m2:+.1f} dex')
        print(f"  {cls} H2-loss: Li&Draine {mu_ld2:+.2f} dex | Malloci {mu_m2:+.2f} dex")
        plotted = True

    if not plotted:
        ax.text(0.5, 0.5, 'No Table B.1 comparison', transform=ax.transAxes,
                ha='center', va='center', fontsize=8, color='gray')
    else:
        ax.axhline(0, color='k', lw=0.8, ls='--')
        ax.legend(fontsize=7)

    ax.set_xlabel(r'$G_0$ (Habing)', fontsize=10)
    ax.set_ylabel(r'$\log_{10}$(Model/Andrews)', fontsize=9)
    ax.set_xscale('log')
    ax.set_xlim(G0_GRID[0], G0_GRID[-1])
    ax.grid(True, alpha=0.3, which='both')


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Computing Kurucz G0 base ...", flush=True)
    G0_base = float(compute_base_g0(load_kurucz_u_E(15000)))
    print(f"  G0_base = {G0_base:.4e}")

    xs_ld = xsect_li_draine()
    xs_m  = xsect_malloci(charge=0)

    # Print integrated cross-section comparison
    mask_uv = (xs_ld[:, 0] >= 6) & (xs_ld[:, 0] <= 13.6)
    mask_uv_m = (xs_m[:, 0] >= 6) & (xs_m[:, 0] <= 13.6)
    int_ld = np.trapezoid(xs_ld[mask_uv, 1], xs_ld[mask_uv, 0])
    int_m  = np.trapezoid(xs_m[mask_uv_m, 1], xs_m[mask_uv_m, 0])
    print(f"  σ integrated 6-13.6 eV: Li&Draine = {int_ld:.3e} cm²·eV,"
          f"  Malloci = {int_m:.3e} cm²·eV"
          f"  (ratio = {int_m/int_ld:.2f}×)")

    print("\nRunning Li & Draine workers ...")
    G0_ld, YH_ld, YH2_ld = run_workers(xs_ld, G0_base, 'Li&Draine')
    print("\nRunning Malloci workers ...")
    G0_m,  YH_m,  YH2_m  = run_workers(xs_m,  G0_base, 'Malloci')

    # Andrews polynomials
    p_H_12  = load_b1_poly(0, 12, 'H')
    p_H2_12 = load_b1_poly(0, 12, 'H2')
    p_H_11  = load_b1_poly(0, 11, 'H')   # likely None
    p_H_13  = load_b1_poly(0, 13, 'H')

    polys_H  = [p_H_11,  p_H_12,  p_H_13]
    polys_H2 = [None,    p_H2_12, None  ]
    G0_fit   = np.logspace(0, 5, 300)

    print("\nResiduals (mean log10 model/Andrews):")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 9),
                             gridspec_kw={'height_ratios': [3, 1.2]})
    fig.suptitle(
        'C$_{24}$ photodissociation rates vs $G_0$:\n'
        'GD89+RRKM model with Li\\&Draine vs Malloci UV cross-sections'
        ' compared to Andrews (2016) Table B.1',
        fontsize=12,
    )

    for col, (cls, title, p_H, p_H2) in enumerate(
            zip(CLASSES, TITLES, polys_H, polys_H2)):
        axes[0, col].set_title(title, fontsize=10)
        _plot_panel_main(axes[0, col], cls,
                         G0_ld, YH_ld, YH2_ld,
                         G0_m,  YH_m,  YH2_m,
                         p_H, p_H2, G0_fit)
        _plot_panel_res(axes[1, col], cls,
                        G0_ld, YH_ld, YH2_ld,
                        G0_m,  YH_m,  YH2_m,
                        p_H, p_H2)

    # Fix y-limits on residual panels
    axes[1, 1].set_ylim(-4, 1)
    axes[1, 2].set_ylim(-12, 1)

    fig.text(
        0.5, 0.005,
        r'Malloci $\sigma$ is 0.5$\times$ Li&Draine (integrated 6–13.6 eV) → similar offset from Andrews. '
        r'Super-H discrepancy ($\sim$10 dex) is independent of $\sigma$: '
        r'it reflects that every UV photon dissociates (yield$\approx$1), '
        r'so $k_H(G_0)\approx R_{\rm abs}$ and differs only in cross-section amplitude.',
        ha='center', fontsize=8, style='italic', color='gray',
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    out = ROOT / 'compare_c24_G0_rates.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nSaved → {out}")


if __name__ == '__main__':
    main()
