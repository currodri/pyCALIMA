#!/usr/bin/env python3
"""
TEST SCRIPT FOR TOTAL PHOTOELECTRIC HEATING RATE vs. WD01

This script computes the total photoelectric heating efficiency

    Gamma_tot / (G0 * nH) = integral[(Gamma - Lambda) / G0 * (1/nH) * dn/da  da]

where:
  - Gamma  is the photoelectric heating rate per grain [erg/s]
  - Lambda is the electron recombination cooling rate per grain [erg/s]
  - dn/da  is the WD01 (Rv=3.1, bc=6e-5) grain size distribution [nH^-1 cm^-1]

The result is compared against the digitised WD01 Figure 2 data:
  external_data/pehtotal_WD01_ISRF_100K_bc6e5.csv
where x = gamma = G0 * sqrt(T) / ne  and  y = Gamma_tot/(G0*nH) * 1e26 [erg/s cm^3].

The computation is parallelised: each (gamma, grain_type, grain_size) triplet is
an independent task dispatched to a ProcessPoolExecutor.

Usage:
    .venv/bin/python models/dust_charge/test_photoelectric_heating_total.py
    .venv/bin/python models/dust_charge/test_photoelectric_heating_total.py --T 100 --n-gamma 20
    .venv/bin/python models/dust_charge/test_photoelectric_heating_total.py --use-li-draine

By: Curro Rodriguez Montero (currodri@gmail.com)
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import quad
from scipy.special import erf
from concurrent.futures import ProcessPoolExecutor, as_completed
from pycalima.plotting_style import use_calima_style

# ── path setup ───────────────────────────────────────────────────────────────

# ── matplotlib style ─────────────────────────────────────────────────────────

# ── WD01 (Rv=3.1, bc=6e-5) grain size distribution parameters ────────────────
# (identical to test_grain_size_distribution.py)
M_C      = 12 * 1.66053892e-24   # mass of C atom [g]
RHO_GRA  = 2.24                  # graphite density [g/cm3]
BC_TOTAL = 6.0e-5                # total C abundance in very small grains
BC1      = 0.75 * BC_TOTAL
BC2      = 0.25 * BC_TOTAL
SIGMA    = 0.4
A_MIN_GRA = 3.5e-8               # 3.5 Å in cm  (minimum graphite size)
A_MIN_SIL = 3.5e-8               # 3.5 Å in cm  (minimum silicate size)
A01      = 3.5e-8
A02      = 30.0e-8
A_MAX    = 10.0e-4               # 10 μm in cm

# Graphite GSD parameters
ALPHA_G  = -1.54
BETA_G   = -0.165
AT_G     = 0.0107e-4
AC_G     = 0.428e-4
C_G_RAW  = 9.99e-12

# Silicate GSD parameters
ALPHA_S  = -2.21
BETA_S   = 0.300
AT_S     = 0.164e-4
AC_S     = 0.1e-4
C_S_RAW  = 1.0e-13

# Volume targets [cm3/H]
VG_TARGET = 1.092 * 2.07e-27
VS_TARGET = 1.322 * 2.98e-27


# ── GSD helper functions ─────────────────────────────────────────────────────

def _F_envelope(a, beta, at):
    if beta >= 0:
        return 1.0 + beta * (a / at)
    else:
        return 1.0 / (1.0 - beta * (a / at))


def _compute_B(bc, a0):
    erf_arg = (3.0 * SIGMA / np.sqrt(2.0)) + np.log(a0 / A_MIN_GRA) / SIGMA * np.sqrt(2.0)
    num = 3.0 * bc * M_C * np.exp(-4.5 * SIGMA**2)
    den = (2.0 * np.pi)**1.5 * (a0**3) * RHO_GRA * SIGMA * (1.0 + erf(erf_arg))
    return num / den


_B1 = _compute_B(BC1, A01)
_B2 = _compute_B(BC2, A02)


def graphite_dn_da_unscaled(a):
    """Unscaled graphite GSD [nH^-1 cm^-1]."""
    if a < A_MIN_GRA:
        return 0.0
    # lognormal (PAH / very small grains)
    D = sum((B / a) * np.exp(-0.5 * (np.log(a / a0) / SIGMA)**2)
            for B, a0 in [(_B1, A01), (_B2, A02)])
    # power-law with exponential cutoff
    M = 1.0 if a < AT_G else np.exp(-((a - AT_G) / AC_G)**3)
    F = _F_envelope(a, BETA_G, AT_G)
    return D + (C_G_RAW / a) * (a / AT_G)**ALPHA_G * F * M


def silicate_dn_da_unscaled(a):
    """Unscaled silicate GSD [nH^-1 cm^-1]."""
    if a < A_MIN_SIL:
        return 0.0
    M = 1.0 if a < AT_S else np.exp(-((a - AT_S) / AC_S)**3)
    F = _F_envelope(a, BETA_S, AT_S)
    return (C_S_RAW / a) * (a / AT_S)**ALPHA_S * F * M


# Compute volume-normalisation scaling factors
_Vg_raw, _ = quad(lambda a: (4 / 3) * np.pi * a**3 * graphite_dn_da_unscaled(a),
                  A_MIN_GRA, A_MAX)
_Vs_raw, _ = quad(lambda a: (4 / 3) * np.pi * a**3 * silicate_dn_da_unscaled(a),
                  A_MIN_SIL, A_MAX)

SCALE_G = VG_TARGET / _Vg_raw
SCALE_S = VS_TARGET / _Vs_raw


def graphite_dn_da(a):
    return graphite_dn_da_unscaled(a) * SCALE_G


def silicate_dn_da(a):
    return silicate_dn_da_unscaled(a) * SCALE_S


# ── Worker function (top-level for pickling) ──────────────────────────────────

def _compute_single_grain(args):
    """
    Compute (Gamma - Lambda) / G0 [erg/s] for a single grain of given size,
    type, and environment (G0, ne, T).

    Returns (gamma_idx, size_idx, grain_type_str, peh_minus_rec_over_G0, peh_norm, rec_norm, zmean, zsigma).
    """
    gamma_idx, size_idx, grain_type, a_cm, G0, ne, T, use_li_draine = args

    # Set Li & Draine flag inside worker process
    import pycalima.models.dust_radiation.dust_emission as de
    de.USE_LI_DRAINE_2001_CARBONACEOUS = use_li_draine

    from pycalima.models.dust_charge import dust_charging as _dc
    from pycalima.models.dust_charge.dust_photoelectric_heating import _compute_rates_point

    try:
        peh, rec, _Zm, _Zs, _ir, _ic = _compute_rates_point(
            (G0, ne, T, grain_type, a_cm, 'Mathis', [])
        )
        # Gamma - Lambda per grain [erg/s], normalised by G0
        result = (peh - rec) / max(1.13, 1e-30)
        peh_norm = peh / max(1.13, 1e-30)
        rec_norm = rec / max(1.13, 1e-30)
        zmean = _Zm
        zsigma = _Zs
    except Exception as exc:
        # Return 0 on failure so that the integration simply ignores this point
        import warnings
        warnings.warn(f"[PEH worker] grain_type={grain_type}, a={a_cm:.3e}, "
                      f"G0={G0:.2e}, ne={ne:.2e}, T={T:.1f}: {exc}")
        result = 0.0
        peh_norm = 0.0
        rec_norm = 0.0
        zmean = 0.0
        zsigma = 0.0

    return gamma_idx, size_idx, grain_type, result, peh_norm, rec_norm, zmean, zsigma


# ── Main computation ──────────────────────────────────────────────────────────

def compute_total_peh(T=100.0, n_gamma=20, n_sizes=30, use_li_draine=False,
                      n_workers=None, G0=1.0):
    """
    Compute Gamma_tot / (G0 * nH) over a range of gamma = G0*sqrt(T)/ne.

    Returns
    -------
    gamma_range : ndarray  [K^0.5 cm^3]
    Gamma_tot_per_G0nH : ndarray  [erg/s cm^3]
    debug_data : dict
        Detailed arrays for diagnostic plots and debugging.
    """
    gamma_range = np.logspace(2, 6, n_gamma)   # gamma = G0*sqrt(T)/ne

    # Grain size grids — same range as WD01
    # For Li & Draine we go down to 4 Å, otherwise 3.5 Å
    a_min_plot = 4e-8 if use_li_draine else A_MIN_GRA
    sizes_gra = np.logspace(np.log10(a_min_plot), np.log10(A_MAX), n_sizes)  # cm
    sizes_sil = np.logspace(np.log10(A_MIN_SIL),  np.log10(A_MAX), n_sizes)

    # Pre-compute GSD weights at each grain size
    dn_da_gra = np.array([graphite_dn_da(a) for a in sizes_gra])
    dn_da_sil = np.array([silicate_dn_da(a) for a in sizes_sil])

    # Build task list
    tasks = []
    for gi, gamma in enumerate(gamma_range):
        ne = G0 * np.sqrt(T) / gamma
        for si, a in enumerate(sizes_gra):
            tasks.append((gi, si, 'graphite', a, G0, ne, T, use_li_draine))
        for si, a in enumerate(sizes_sil):
            tasks.append((gi, si, 'silicate', a, G0, ne, T, use_li_draine))

    print(f"  Dispatching {len(tasks)} tasks "
          f"({n_gamma} gamma × {n_sizes} sizes × 2 grain types)…")

    # Result storage: (Gamma - Lambda)/G0 for each [gamma, size]
    peh_gra = np.zeros((n_gamma, n_sizes))
    peh_sil = np.zeros((n_gamma, n_sizes))
    peh_val_gra = np.zeros((n_gamma, n_sizes))
    peh_val_sil = np.zeros((n_gamma, n_sizes))
    rec_val_gra = np.zeros((n_gamma, n_sizes))
    rec_val_sil = np.zeros((n_gamma, n_sizes))
    zmean_gra = np.zeros((n_gamma, n_sizes))
    zmean_sil = np.zeros((n_gamma, n_sizes))
    zsigma_gra = np.zeros((n_gamma, n_sizes))
    zsigma_sil = np.zeros((n_gamma, n_sizes))

    n_done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as exe:
        future_map = {exe.submit(_compute_single_grain, t): t for t in tasks}
        for fut in as_completed(future_map):
            gi, si, gtype, val, peh_val, rec_val, zmean, zsigma = fut.result()
            if gtype == 'graphite':
                peh_gra[gi, si] = val
                peh_val_gra[gi, si] = peh_val
                rec_val_gra[gi, si] = rec_val
                zmean_gra[gi, si] = zmean
                zsigma_gra[gi, si] = zsigma
            else:
                peh_sil[gi, si] = val
                peh_val_sil[gi, si] = peh_val
                rec_val_sil[gi, si] = rec_val
                zmean_sil[gi, si] = zmean
                zsigma_sil[gi, si] = zsigma
            n_done += 1
            if n_done % max(1, len(tasks) // 20) == 0 or n_done == len(tasks):
                print(f"    [{n_done}/{len(tasks)}] completed", flush=True)

    # Integrate over grain sizes to get Gamma_tot / (G0 * nH)
    # integral[(Gamma-Lambda)/G0 * (1/nH) * dn/da  da]
    Gamma_tot = np.zeros(n_gamma)
    for gi in range(n_gamma):
        gra_integrand = peh_gra[gi, :] * dn_da_gra   # [erg/s cm^-1]
        sil_integrand = peh_sil[gi, :] * dn_da_sil
        Gamma_tot[gi] = (np.trapezoid(gra_integrand, sizes_gra) +
                         np.trapezoid(sil_integrand, sizes_sil))

    debug_data = {
        'sizes_gra': sizes_gra,
        'sizes_sil': sizes_sil,
        'dn_da_gra': dn_da_gra,
        'dn_da_sil': dn_da_sil,
        'peh_gra': peh_gra,
        'peh_sil': peh_sil,
        'peh_val_gra': peh_val_gra,
        'peh_val_sil': peh_val_sil,
        'rec_val_gra': rec_val_gra,
        'rec_val_sil': rec_val_sil,
        'zmean_gra': zmean_gra,
        'zmean_sil': zmean_sil,
        'zsigma_gra': zsigma_gra,
        'zsigma_sil': zsigma_sil,
    }

    return gamma_range, Gamma_tot, debug_data


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_total_peh(gamma_range, Gamma_tot, T, save_path, use_li_draine=False):
    """Plot Gamma_tot/(G0*nH) vs gamma and overlay WD01 digitised data."""
    use_calima_style()
    plt.rcParams["text.usetex"] = False

    # Load WD01 digitised comparison
    ext_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', '..', 'external_data',
                            'pehtotal_WD01_ISRF_100K_bc6e5.csv')
    wd01_loaded = False
    if os.path.exists(ext_path):
        try:
            wd01_data  = np.loadtxt(ext_path, delimiter=',')
            wd01_gamma = wd01_data[:, 0]
            wd01_val   = wd01_data[:, 1]  # already in units of 1e-26 erg/s cm^3
            wd01_loaded = True
        except Exception as e:
            print(f"Warning: could not load WD01 data: {e}")

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    # CALIMA result (convert to units of 1e-26 erg/s cm^3 for readability)
    calima_scaled = Gamma_tot * 1e26
    ax.loglog(gamma_range, np.abs(calima_scaled), 'C0-o', lw=2, ms=5,
              label=f'CALIMA numerical (T = {T:.0f} K'
                    + (', Li & Draine 2001)' if use_li_draine else ')'))

    if wd01_loaded:
        ax.loglog(wd01_gamma, wd01_val, 'k--x', lw=2, ms=7,
                  label='WD01 (digitised, T = 100 K, ISRF, Rv=3.1, bc=6e-5)')

    ax.set_xlabel(r'$\gamma = G_0 T^{1/2} / n_e$  [K$^{1/2}$ cm$^3$]')
    ax.set_ylabel(r'$\Gamma_{\rm tot}\,/\,(G_0\,n_H)$  [$10^{-26}$ erg s$^{-1}$ cm$^3$]')
    ax.set_title(f'Total Photoelectric Heating Efficiency  (T = {T:.0f} K, G0 = 1, ISRF)')
    ax.grid(True, which='both', linestyle=':', alpha=0.5)
    ax.legend(fontsize=10, loc='lower left')

    # Secondary x-axis: ne (for G0=1, T=100)
    ax2 = ax.twiny()
    gamma_ticks = ax.get_xticks()
    ne_ticks = 1.0 * np.sqrt(T) / np.clip(gamma_ticks, 1e-10, None)
    ax2.set_xscale('log')
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(gamma_ticks)
    ax2.set_xticklabels([f'{v:.1e}' for v in ne_ticks], fontsize=7)
    ax2.set_xlabel(r'$n_e$ [cm$^{-3}$]  (for G0=1)', fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved plot → {save_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def generate_peh_debug_diagnostics(gamma_range, sizes_gra, sizes_sil, dn_da_gra, dn_da_sil,
                                  peh_gra, peh_sil, peh_val_gra, peh_val_sil,
                                  rec_val_gra, rec_val_sil, zmean_gra, zmean_sil,
                                  zsigma_gra, zsigma_sil, T, save_dir, use_li_draine=False, G0=1.0):
    """
    Generate detailed debugging reports and plots to understand the uptick at high gamma.
    """
    use_calima_style()
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    
    # 1. Create debug directory if it doesn't exist
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = os.path.dirname(os.path.abspath(__file__))

    suffix = '_li_draine' if use_li_draine else ''
    
    # Selected gamma indices for integrand plotting
    # We want a few points spanning the range from low to high gamma.
    n_gamma = len(gamma_range)
    sel_gamma_indices = [0, n_gamma//4, n_gamma//2, 3*n_gamma//4, n_gamma-1]
    # Ensure they are unique and within bounds
    sel_gamma_indices = sorted(list(set(np.clip(sel_gamma_indices, 0, n_gamma-1))))
    
    # Selected grain sizes for charge and rate plotting
    # We want a range from small to large
    target_sizes = [3.5e-8, 1e-7, 3e-7, 1e-6, 1e-5]  # in cm
    sel_gra_indices = []
    sel_sil_indices = []
    for ts in target_sizes:
        idx_gra = np.argmin(np.abs(sizes_gra - ts))
        idx_sil = np.argmin(np.abs(sizes_sil - ts))
        sel_gra_indices.append(idx_gra)
        sel_sil_indices.append(idx_sil)
    # Deduplicate while preserving order
    sel_gra_indices = sorted(list(set(sel_gra_indices)))
    sel_sil_indices = sorted(list(set(sel_sil_indices)))
    
    # Plot 1: Integrand a * dn/da * (peh - rec)/G0 vs a (log-x, linear-y or log-y)
    # Since it can be positive or negative, let's plot a * integrand on log-x and linear-y,
    # because that shows the relative contribution of each ln(a) bin to the total integral.
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False, dpi=150)
    for idx in sel_gamma_indices:
        g = gamma_range[idx]
        # graphite
        y_gra = sizes_gra * dn_da_gra * peh_gra[idx, :]
        axes[0].plot(sizes_gra, y_gra, label=f'gamma={g:.1e}')
        # silicate
        y_sil = sizes_sil * dn_da_sil * peh_sil[idx, :]
        axes[1].plot(sizes_sil, y_sil, label=f'gamma={g:.1e}')
        
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Grain size a [cm]')
    axes[0].set_ylabel(r'$a (dn/da) (\Gamma - \Lambda)/G_0$ [cm$^{-3}$ s$^{-1}$]')
    axes[0].set_title('Graphite Integrand Contribution')
    axes[0].grid(True, which='both', linestyle=':', alpha=0.5)
    axes[0].legend(fontsize=8)
    
    axes[1].set_xscale('log')
    axes[1].set_xlabel('Grain size a [cm]')
    axes[1].set_ylabel(r'$a (dn/da) (\Gamma - \Lambda)/G_0$ [cm$^{-3}$ s$^{-1}$]')
    axes[1].set_title('Silicate Integrand Contribution')
    axes[1].grid(True, which='both', linestyle=':', alpha=0.5)
    axes[1].legend(fontsize=8)
    
    fig.suptitle(f'Integrand Contribution to Heating (T = {T:.0f} K)')
    fig.tight_layout()
    integrand_plot_path = os.path.join(save_dir, f'peh_total_debug_integrand{suffix}.png')
    fig.savefig(integrand_plot_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved debug integrand plot → {integrand_plot_path}")
    
    # Plot 2: Zmean vs Gamma for selected grain sizes
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True, dpi=150)
    for idx in sel_gra_indices:
        a_nm = sizes_gra[idx] * 1e7
        axes[0].plot(gamma_range, zmean_gra[:, idx], '-o', ms=4, label=f'a={a_nm:.1f} nm')
    for idx in sel_sil_indices:
        a_nm = sizes_sil[idx] * 1e7
        axes[1].plot(gamma_range, zmean_sil[:, idx], '-o', ms=4, label=f'a={a_nm:.1f} nm')
        
    axes[0].set_xscale('log')
    axes[0].set_xlabel(r'$\gamma$ [K$^{1/2}$ cm$^3$]')
    axes[0].set_ylabel('Mean Charge Zmean')
    axes[0].set_title('Graphite Mean Charge')
    axes[0].grid(True, which='both', linestyle=':', alpha=0.5)
    axes[0].legend(fontsize=8)
    
    axes[1].set_xscale('log')
    axes[1].set_xlabel(r'$\gamma$ [K$^{1/2}$ cm$^3$]')
    axes[1].set_title('Silicate Mean Charge')
    axes[1].grid(True, which='both', linestyle=':', alpha=0.5)
    axes[1].legend(fontsize=8)
    
    fig.suptitle(f'Mean Grain Charge vs Gamma (T = {T:.0f} K)')
    fig.tight_layout()
    charge_plot_path = os.path.join(save_dir, f'peh_total_debug_charge{suffix}.png')
    fig.savefig(charge_plot_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved debug charge plot → {charge_plot_path}")

    # Plot 3: Individual peh & rec rates vs Gamma for representative sizes
    # We'll plot for: minimum size (index 0) and a medium size (10 nm)
    sizes_to_plot = [0, np.argmin(np.abs(sizes_gra - 1e-6))] # smallest & ~10nm
    fig, axes = plt.subplots(len(sizes_to_plot), 2, figsize=(14, 4 * len(sizes_to_plot)), sharex=True, dpi=150)
    if len(sizes_to_plot) == 1:
        axes = np.expand_dims(axes, axis=0)
        
    for row_idx, size_idx in enumerate(sizes_to_plot):
        a_nm_gra = sizes_gra[size_idx] * 1e7
        a_nm_sil = sizes_sil[size_idx] * 1e7
        
        # Graphite row
        axes[row_idx, 0].loglog(gamma_range, peh_val_gra[:, size_idx], 'r-o', ms=4, label='Photoelectric (peh)')
        axes[row_idx, 0].loglog(gamma_range, rec_val_gra[:, size_idx], 'b-x', ms=4, label='Recombination (rec)')
        # net absolute
        net_gra = peh_val_gra[:, size_idx] - rec_val_gra[:, size_idx]
        axes[row_idx, 0].loglog(gamma_range, np.abs(net_gra), 'g--', label='|net|')
        axes[row_idx, 0].set_ylabel(f'Rate / G0 [erg/s]\na={a_nm_gra:.1f} nm')
        axes[row_idx, 0].grid(True, which='both', linestyle=':', alpha=0.5)
        axes[row_idx, 0].legend(fontsize=8)
        if row_idx == 0:
            axes[row_idx, 0].set_title('Graphite Rates')
            
        # Silicate row
        axes[row_idx, 1].loglog(gamma_range, peh_val_sil[:, size_idx], 'r-o', ms=4, label='Photoelectric (peh)')
        axes[row_idx, 1].loglog(gamma_range, rec_val_sil[:, size_idx], 'b-x', ms=4, label='Recombination (rec)')
        net_sil = peh_val_sil[:, size_idx] - rec_val_sil[:, size_idx]
        axes[row_idx, 1].loglog(gamma_range, np.abs(net_sil), 'g--', label='|net|')
        axes[row_idx, 1].grid(True, which='both', linestyle=':', alpha=0.5)
        axes[row_idx, 1].legend(fontsize=8)
        if row_idx == 0:
            axes[row_idx, 1].set_title('Silicate Rates')
            
    for col in [0, 1]:
        axes[-1, col].set_xscale('log')
        axes[-1, col].set_xlabel(r'$\gamma$ [K$^{1/2}$ cm$^3$]')
        
    fig.suptitle(f'Photoelectric Heating and Recombination Cooling Rates (T = {T:.0f} K)')
    fig.tight_layout()
    rates_plot_path = os.path.join(save_dir, f'peh_total_debug_rates{suffix}.png')
    fig.savefig(rates_plot_path, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved debug rates plot → {rates_plot_path}")

    # 2. Write Markdown Debugging Report
    report_path = os.path.join(save_dir, f'peh_total_debug_report{suffix}.md')
    with open(report_path, 'w') as f:
        f.write(f"# Photoelectric Heating Debugging Report (T = {T:.0f} K)\n\n")
        f.write("This report provides detailed diagnostics for the total photoelectric heating efficiency calculation, ")
        f.write("specifically investigating why the results show a deviation/uptick at high gamma ($\\gamma \\ge 10^5$).\n\n")
        
        f.write("## Integrand Contributions by Grain Size Bins\n\n")
        f.write("To see which sizes dominate the total heating rate at each gamma, we split the integration into size bins:\n")
        f.write("- **Very Small Grains** ($a < 1$ nm = $10$ Å)\n")
        f.write("- **Medium Grains** ($1 \\le a < 10$ nm = $10 - 100$ Å)\n")
        f.write("- **Large Grains** ($a \\ge 10$ nm = $100$ Å)\n\n")
        
        f.write("| gamma | ne [cm-3] | Total [1e-26] | Graphite [1e-26] | Silicate [1e-26] | Gra (<1nm) | Gra (1-10nm) | Gra (>=10nm) | Sil (<1nm) | Sil (1-10nm) | Sil (>=10nm) |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
        
        for gi in range(n_gamma):
            g = gamma_range[gi]
            ne = G0 * np.sqrt(T) / g
            
            # Full integrals
            tot_gra = np.trapezoid(peh_gra[gi, :] * dn_da_gra, sizes_gra) * 1e26
            tot_sil = np.trapezoid(peh_sil[gi, :] * dn_da_sil, sizes_sil) * 1e26
            total = tot_gra + tot_sil
            
            # Slices
            # Graphite
            m_gra_s = sizes_gra < 1e-7
            m_gra_m = (sizes_gra >= 1e-7) & (sizes_gra < 1e-6)
            m_gra_l = sizes_gra >= 1e-6
            
            # Silicate
            m_sil_s = sizes_sil < 1e-7
            m_sil_m = (sizes_sil >= 1e-7) & (sizes_sil < 1e-6)
            m_sil_l = sizes_sil >= 1e-6
            
            # Integrate slices
            def int_slice(y, x, mask):
                if np.sum(mask) >= 2:
                    return np.trapezoid(y[mask], x[mask]) * 1e26
                return 0.0
                
            gra_s = int_slice(peh_gra[gi, :] * dn_da_gra, sizes_gra, m_gra_s)
            gra_m = int_slice(peh_gra[gi, :] * dn_da_gra, sizes_gra, m_gra_m)
            gra_l = int_slice(peh_gra[gi, :] * dn_da_gra, sizes_gra, m_gra_l)
            
            sil_s = int_slice(peh_sil[gi, :] * dn_da_sil, sizes_sil, m_sil_s)
            sil_m = int_slice(peh_sil[gi, :] * dn_da_sil, sizes_sil, m_sil_m)
            sil_l = int_slice(peh_sil[gi, :] * dn_da_sil, sizes_sil, m_sil_l)
            
            f.write(f"| {g:.2e} | {ne:.2e} | {total:.4f} | {tot_gra:.4f} | {tot_sil:.4f} | {gra_s:.4f} | {gra_m:.4f} | {gra_l:.4f} | {sil_s:.4f} | {sil_m:.4f} | {sil_l:.4f} |\n")
            
        f.write("\n## Diagnostic Plots\n\n")
        f.write("### 1. Integrand Contribution vs. Size\n")
        f.write(f"![Integrand Contribution](peh_total_debug_integrand{suffix}.png)\n\n")
        f.write("### 2. Mean Grain Charge vs. Gamma\n")
        f.write(f"![Mean Grain Charge](peh_total_debug_charge{suffix}.png)\n\n")
        f.write("### 3. Individual Rates vs. Gamma\n")
        f.write(f"![Individual Rates](peh_total_debug_rates{suffix}.png)\n\n")
        
    print(f"  Saved debug report → {report_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Compute total photoelectric heating efficiency and compare to WD01.')
    parser.add_argument('--T', type=float, default=100.0,
                        help='Gas temperature in K (default: 100)')
    parser.add_argument('--G0', type=float, default=1.0,
                        help='Radiation field scaling (default: 1.0 = ISRF)')
    parser.add_argument('--n-gamma', type=int, default=20,
                        help='Number of gamma points (default: 20)')
    parser.add_argument('--n-sizes', type=int, default=30,
                        help='Number of grain sizes per species (default: 30)')
    parser.add_argument('--n-workers', type=int, default=None,
                        help='Number of parallel workers (default: all CPUs)')
    parser.add_argument('--use-li-draine', action='store_true',
                        help='Use Li & Draine (2001) carbonaceous cross-section blend')
    parser.add_argument('--debug-dir', type=str, default=None,
                        help='Directory to save debugging plots/reports (e.g. artifact folder)')
    args = parser.parse_args()

    # Apply Li & Draine flag in the main process (workers set it themselves)
    if args.use_li_draine:
        import pycalima.models.dust_radiation.dust_emission as de
        de.USE_LI_DRAINE_2001_CARBONACEOUS = True
        print("Using Li & Draine (2001) carbonaceous cross sections for graphite.")

    print("=" * 70)
    print("TOTAL PHOTOELECTRIC HEATING EFFICIENCY vs. WD01")
    print("=" * 70)
    print(f"  T = {args.T:.1f} K  |  G0 = {args.G0:.2f}  |  "
          f"n_gamma = {args.n_gamma}  |  n_sizes = {args.n_sizes}")
    print(f"  GSD:  WD01 Rv=3.1, bc=6e-5")
    print(f"  Volume scaling:  SCALE_G = {SCALE_G:.5f},  SCALE_S = {SCALE_S:.5f}")

    gamma_range, Gamma_tot, debug_data = compute_total_peh(
        T=args.T,
        n_gamma=args.n_gamma,
        n_sizes=args.n_sizes,
        use_li_draine=args.use_li_draine,
        n_workers=args.n_workers,
        G0=args.G0,
    )

    # Print table
    print(f"\n{'gamma':>12}  {'ne [cm-3]':>12}  {'Gamma/(G0*nH) [1e-26 erg/s cm3]':>35}")
    print("-" * 65)
    for g, G in zip(gamma_range, Gamma_tot):
        ne = args.G0 * np.sqrt(args.T) / g
        print(f"  {g:10.3e}  {ne:12.3e}  {G*1e26:35.4f}")

    # Save plot
    script_dir = os.path.dirname(os.path.abspath(__file__))
    suffix = '_li_draine' if args.use_li_draine else ''
    plot_path = os.path.join(script_dir, f'peh_total_comparison{suffix}.png')
    plot_total_peh(gamma_range, Gamma_tot, T=args.T, save_path=plot_path,
                   use_li_draine=args.use_li_draine)

    # Save debugging diagnostics
    debug_dir = args.debug_dir if args.debug_dir else script_dir
    generate_peh_debug_diagnostics(
        gamma_range=gamma_range,
        sizes_gra=debug_data['sizes_gra'],
        sizes_sil=debug_data['sizes_sil'],
        dn_da_gra=debug_data['dn_da_gra'],
        dn_da_sil=debug_data['dn_da_sil'],
        peh_gra=debug_data['peh_gra'],
        peh_sil=debug_data['peh_sil'],
        peh_val_gra=debug_data['peh_val_gra'],
        peh_val_sil=debug_data['peh_val_sil'],
        rec_val_gra=debug_data['rec_val_gra'],
        rec_val_sil=debug_data['rec_val_sil'],
        zmean_gra=debug_data['zmean_gra'],
        zmean_sil=debug_data['zmean_sil'],
        zsigma_gra=debug_data['zsigma_gra'],
        zsigma_sil=debug_data['zsigma_sil'],
        T=args.T,
        save_dir=debug_dir,
        use_li_draine=args.use_li_draine,
        G0=args.G0
    )

    print("\nDone.")


if __name__ == '__main__':
    main()
