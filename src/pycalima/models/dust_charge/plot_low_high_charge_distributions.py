#!/usr/bin/env python3
"""
Compute and plot equilibrium charge distributions for a representative
low-temperature and high-temperature case at (approximately) the same
gamma = G0 * sqrt(T) / ne so the temperature-driven separation can be
inspected directly.

Saves two files in the repository root:
- charge_distributions_overlay.png  (linear y)
- charge_distributions_overlay_logy.png (log y)

Run: python3 plot_low_high_charge_distributions.py
"""
import numpy as np
import matplotlib.pyplot as plt
import os

from pycalima.models.dust_charge.dust_charging import equilibrium_charge_for_grain


def compute_and_plot(grain_type='silicate', a_micron=0.005, gamma_target=1e4, ne_cm3=1e-2):
    """Compute equilibrium P(Z) for low-T and high-T with same gamma_target.

    Parameters
    ----------
    grain_type : str
    a_micron : float
    gamma_target : float
    ne_cm3 : float
    """
    # choose low and high temperatures
    T_low = 50.0        # K (lowT < 100)
    T_high = 1e5        # K (highT > 1e4)

    # compute G0 so gamma = G0 * sqrt(T) / ne
    G0_low = float(gamma_target * ne_cm3 / np.sqrt(T_low))
    G0_high = float(gamma_target * ne_cm3 / np.sqrt(T_high))

    print(f"Using parameters (ne={ne_cm3} cm^-3):\n  low: T={T_low} K, G0={G0_low:.3g}\n  high: T={T_high} K, G0={G0_high:.3g}")

    # compute equilibrium charge distributions
    Zs_low, P_low, rates_low, Zmean_low, Zsigma_low = equilibrium_charge_for_grain(
        G0_low, ne_cm3, T_low, grain_type, a_micron, debug=False
    )
    Zs_high, P_high, rates_high, Zmean_high, Zsigma_high = equilibrium_charge_for_grain(
        G0_high, ne_cm3, T_high, grain_type, a_micron, debug=False
    )

    # ensure same Z grid for plotting by taking union
    Zmin = min(Zs_low.min(), Zs_high.min())
    Zmax = max(Zs_low.max(), Zs_high.max())
    Zs_union = np.arange(Zmin, Zmax + 1, dtype=int)

    def regrid(Zs, P, Z_union):
        out = np.zeros_like(Z_union, dtype=float)
        idx = {z: i for i, z in enumerate(Zs)}
        for i, z in enumerate(Z_union):
            if z in idx:
                out[i] = P[idx[z]]
        return out

    P_low_u = regrid(Zs_low, P_low, Zs_union)
    P_high_u = regrid(Zs_high, P_high, Zs_union)

    outdir = os.path.abspath('.')
    fname_lin = os.path.join(outdir, 'charge_distributions_overlay.png')
    fname_log = os.path.join(outdir, 'charge_distributions_overlay_logy.png')

    # Plot linear y (overlay)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.step(np.concatenate([[Zs_union[0]-0.5], np.repeat(Zs_union, 2), [Zs_union[-1]+0.5]]),
            np.concatenate([[0.0], np.repeat(P_low_u, 2), [0.0]]), where='pre', label=f'low-T (T={T_low} K), <Z>={Zmean_low:.2f}')
    ax.step(np.concatenate([[Zs_union[0]-0.5], np.repeat(Zs_union, 2), [Zs_union[-1]+0.5]]),
            np.concatenate([[0.0], np.repeat(P_high_u, 2), [0.0]]), where='pre', label=f'high-T (T={T_high} K), <Z>={Zmean_high:.2f}')
    ax.set_xlabel('Z')
    ax.set_ylabel('P(Z)')
    ax.set_title(rf'Equilibrium charge distributions — {grain_type}, a={a_micron} um, same $\gamma$={gamma_target:.3g}')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.5)
    fig.tight_layout()
    fig.savefig(fname_lin)
    plt.close(fig)

    # Plot log-y to show tails
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.step(np.concatenate([[Zs_union[0]-0.5], np.repeat(Zs_union, 2), [Zs_union[-1]+0.5]]),
            np.concatenate([[1e-20], np.repeat(P_low_u + 1e-300, 2), [1e-20]]), where='pre', label=f'low-T (T={T_low} K)')
    ax.step(np.concatenate([[Zs_union[0]-0.5], np.repeat(Zs_union, 2), [Zs_union[-1]+0.5]]),
            np.concatenate([[1e-20], np.repeat(P_high_u + 1e-300, 2), [1e-20]]), where='pre', label=f'high-T (T={T_high} K)')
    ax.set_yscale('log')
    ax.set_xlabel('Z')
    ax.set_ylabel('P(Z) (log scale)')
    ax.set_title(f'Equilibrium charge distributions (log y) — {grain_type}, a={a_micron} um')
    ax.legend()
    ax.grid(True, which='both', linestyle=':', alpha=0.5)
    fig.tight_layout()
    fig.savefig(fname_log)
    plt.close(fig)

    print('Saved:', fname_lin)
    print('Saved:', fname_log)


if __name__ == '__main__':
    compute_and_plot()
