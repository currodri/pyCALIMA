"""
validate_murga2020.py — Reproduce Murga et al. (2020) Figures 3 and 4.

This script computes the SHIVA model predictions for:
    Fig. 3 — Mean PAH charge ⟨Z⟩ as a function of molecule size (Nc)
             for three representative points along the PDR profile.
    Fig. 4 — C₂H₂ photo-dissociation rate per molecule [s⁻¹] vs. Nc.

PDR environmental conditions
-----------------------------
Taken from the representative positions shown in Murga+2020 Fig. 2.
The three points bracket the range from the UV-exposed PDR surface to
the more shielded interior (A_V ~ 0, 1, 2):

    Point 1 (surface):    G0 = 1000, nH = 1000 cm⁻³, T = 500 K
    Point 2 (A_V ~ 1):    G0 = 100,  nH = 1000 cm⁻³, T = 200 K
    Point 3 (A_V ~ 2):    G0 = 10,   nH = 1000 cm⁻³, T =  80 K

Electron density
-----------------
In a PDR, free electrons come mainly from photoionised carbon (x_C ≈ 1.6×10⁻⁴).
We use ne ≈ x_C × nH as a simple estimate; G0-dependent attenuation of
photoionisation deep in the slab reduces this at Point 3.

    ne_1 = 0.16 cm⁻³,  ne_2 = 0.10 cm⁻³,  ne_3 = 0.02 cm⁻³

Size range
----------
Nc from 20 (smallest PAHs considered stable, ~benzopyrene C20) to 200
(large coronene-type PAHs), with 20 sample points.

Usage
-----
    python -m models.shiva.validate_murga2020

Outputs three files in the current directory:
    murga2020_fig3_mean_charge.png
    murga2020_fig4_c2h2_rate.png
    murga2020_results.npz   (arrays for further analysis)

References
----------
Murga, M.S. et al. 2020, A&A, 644, A89
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Make sure CALIMA root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from models.shiva.isrf import make_isrf_callable
from models.shiva.shiva_charge import mean_charge, steady_state_charges
from models.shiva.shiva_dissociation import (
    C2H2_loss_rate_per_molecule,
    C2H2_loss_rate_charge_averaged,
)

# ── PDR conditions (Murga+2020 Fig. 2 representative points) ─────────────
PDR_POINTS = [
    dict(label='PDR surface ($A_V \\approx 0$)',   G0=1000, nH=1000, T=500, ne=0.16,
         color='tab:red',    ls='-'),
    dict(label='$A_V \\approx 1$',                  G0=100,  nH=1000, T=200, ne=0.10,
         color='tab:orange', ls='--'),
    dict(label='$A_V \\approx 2$',                  G0=10,   nH=1000, T=80,  ne=0.02,
         color='tab:blue',   ls='-.'),
]

# ── PAH size grid ─────────────────────────────────────────────────────────
NC_VALUES = np.array([20, 24, 30, 40, 54, 70, 96, 120, 150, 200], dtype=float)

# ── Number of temperature bins for GD89 (reduce for speed) ───────────────
N_T_BINS = 120


def compute_pdr_point(pdr, Nc_arr, u_E_fn, verbose=True):
    """
    Compute ⟨Z⟩(Nc) and k_C2H2(Nc) for a single PDR point.

    Returns
    -------
    z_mean : ndarray   shape (len(Nc_arr),)
    k_c2h2 : ndarray   shape (len(Nc_arr),)  [s⁻¹]
    """
    G0, T, ne = pdr['G0'], pdr['T'], pdr['ne']
    z_mean = np.zeros(len(Nc_arr))
    k_c2h2 = np.zeros(len(Nc_arr))

    for i, Nc in enumerate(Nc_arr):
        if verbose:
            print(f"  Nc = {int(Nc):4d}  G0={G0:.0f}  T={T:.0f} K ... ", end='', flush=True)

        # Mean charge
        z_mean[i] = mean_charge(Nc, T, ne, u_E_fn, G0=G0)

        # Charge-averaged C2H2 loss rate
        k_c2h2[i] = C2H2_loss_rate_charge_averaged(
            Nc, T, ne, u_E_fn, G0=G0, N_T=N_T_BINS
        )

        if verbose:
            print(f"⟨Z⟩ = {z_mean[i]:+.3f},  k_C2H2 = {k_c2h2[i]:.2e} s⁻¹")

    return z_mean, k_c2h2


def run_validation(Nc_arr=NC_VALUES, output_dir='.', verbose=True):
    """
    Run the full validation and produce Figs 3 & 4.

    Parameters
    ----------
    Nc_arr : array_like
        PAH sizes (number of C atoms) to evaluate.
    output_dir : str
        Directory for output files.
    verbose : bool
        Print progress.
    """
    u_E_fn = make_isrf_callable(G0=1.0, field='mathis83')

    # Storage
    all_z    = {}
    all_k    = {}

    for pdr in PDR_POINTS:
        lab = pdr['label']
        if verbose:
            print(f"\n=== {lab}  (G0={pdr['G0']}, T={pdr['T']} K, ne={pdr['ne']} cm⁻³) ===")
        z, k = compute_pdr_point(pdr, Nc_arr, u_E_fn, verbose=verbose)
        all_z[lab] = z
        all_k[lab] = k

    # ── Figure 3: mean charge ─────────────────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(6, 4.5))
    for pdr in PDR_POINTS:
        lab = pdr['label']
        ax3.plot(Nc_arr, all_z[lab],
                 color=pdr['color'], ls=pdr['ls'], lw=2, label=lab)
    ax3.axhline(0, color='k', lw=0.6, ls=':')
    ax3.set_xlabel('$N_C$ (number of C atoms)', fontsize=12)
    ax3.set_ylabel(r'$\langle Z \rangle$', fontsize=12)
    ax3.set_title('SHIVA: mean PAH charge (cf. Murga+2020 Fig. 3)', fontsize=11)
    ax3.legend(fontsize=9)
    ax3.set_xlim(Nc_arr[0], Nc_arr[-1])
    fig3.tight_layout()
    fig3_path = os.path.join(output_dir, 'murga2020_fig3_mean_charge.png')
    fig3.savefig(fig3_path, dpi=150)
    plt.close(fig3)
    print(f"\nSaved: {fig3_path}")

    # ── Figure 4: C2H2 loss rate ──────────────────────────────────────────
    fig4, ax4 = plt.subplots(figsize=(6, 4.5))
    for pdr in PDR_POINTS:
        lab = pdr['label']
        k_plot = np.where(all_k[lab] > 0, all_k[lab], 1e-40)
        ax4.semilogy(Nc_arr, k_plot,
                     color=pdr['color'], ls=pdr['ls'], lw=2, label=lab)
    ax4.set_xlabel('$N_C$ (number of C atoms)', fontsize=12)
    ax4.set_ylabel(r'$k_{\mathrm{C_2H_2}}$ [s$^{-1}$ per molecule]', fontsize=12)
    ax4.set_title('SHIVA: C$_2$H$_2$ loss rate (cf. Murga+2020 Fig. 4)', fontsize=11)
    ax4.legend(fontsize=9)
    ax4.set_xlim(Nc_arr[0], Nc_arr[-1])
    ax4.set_ylim(1e-25, 1e-5)
    fig4.tight_layout()
    fig4_path = os.path.join(output_dir, 'murga2020_fig4_c2h2_rate.png')
    fig4.savefig(fig4_path, dpi=150)
    plt.close(fig4)
    print(f"Saved: {fig4_path}")

    # ── Save numerical results ─────────────────────────────────────────────
    npz_path = os.path.join(output_dir, 'murga2020_results.npz')
    save_dict = {'Nc': Nc_arr}
    for pdr in PDR_POINTS:
        lab = pdr['label']
        key_z = 'z_' + pdr['label'][:4].replace(' ', '_').replace('$', '')
        key_k = 'k_' + pdr['label'][:4].replace(' ', '_').replace('$', '')
        save_dict[key_z] = all_z[lab]
        save_dict[key_k] = all_k[lab]
    np.savez(npz_path, **save_dict)
    print(f"Saved: {npz_path}")

    return all_z, all_k


# ── Standalone entry point ────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Reproduce Murga+2020 Figs 3 & 4 with the SHIVA model.')
    parser.add_argument('--output-dir', default='.', help='Output directory')
    parser.add_argument('--fast', action='store_true',
                        help='Use fewer Nc points for a quick sanity check')
    args = parser.parse_args()

    Nc_arr = np.array([24, 54, 96, 150], dtype=float) if args.fast else NC_VALUES
    run_validation(Nc_arr=Nc_arr, output_dir=args.output_dir, verbose=True)
