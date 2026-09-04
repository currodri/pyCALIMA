"""
Example script to scan grain charge vs gamma = G0 * sqrt(T) / ne
"""
import numpy as np
import sys
import os
# ensure repository root is on sys.path so imports from project modules work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.dust_charge.dust_charging import compute_charge_vs_gamma

if __name__ == '__main__':
    grain_type = 'silicate'
    a_micron = 0.005*1e-4
    gamma_values = np.logspace(-4, 6, 10)
    results, fig = compute_charge_vs_gamma(
        grain_type, a_micron, gamma_values,
        combos_per_gamma=100, seed=1, debug=False,
        temp_bin_edges=np.logspace(1, 7, 10)
    )
    print('Done.')

    # create three scatter plots color-coded by Temperature, n_e and G0
    import matplotlib.pyplot as plt
    import numpy as np

    gam = np.array([r['gamma'] for r in results])
    Zmean = np.array([r['Zmean'] for r in results])
    T = np.array([r['T'] for r in results])
    ne = np.array([r['ne'] for r in results])
    G0 = np.array([r['G0'] for r in results])

    fig2, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)

    # helper: map a positive array to marker sizes in [smin, smax]
    def sizes_from_log(arr, smin=20, smax=220):
        arr = np.asarray(arr)
        arr_safe = np.maximum(arr, 1e-30)
        lv = np.log10(arr_safe)
        if np.all(np.isfinite(lv)) and (lv.max() - lv.min()) > 1e-12:
            sizes = smin + (lv - lv.min()) / (lv.max() - lv.min()) * (smax - smin)
        else:
            sizes = np.full_like(lv, (smin + smax) / 2.0)
        return sizes

    sizes_ne = sizes_from_log(ne)
    sizes_G0 = sizes_from_log(G0)
    sizes_T = sizes_from_log(T)

    # Top row: original three plots
    sc00 = axes[0, 0].scatter(gam, Zmean, c=np.log10(T), cmap='plasma', s=sizes_ne, alpha=0.9, edgecolors='none')
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_xlabel(r'$\gamma$')
    axes[0, 0].set_title('color = log10(T [K]) ; size ~ n_e')
    cb00 = fig2.colorbar(sc00, ax=axes[0, 0])
    cb00.set_label(r'log10(T)')

    sc01 = axes[0, 1].scatter(gam, Zmean, c=np.log10(ne), cmap='viridis', s=sizes_G0, alpha=0.9, edgecolors='none')
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_xlabel(r'$\gamma$')
    axes[0, 1].set_title('color = log10(n_e) ; size ~ G0')
    cb01 = fig2.colorbar(sc01, ax=axes[0, 1])
    cb01.set_label(r'log10(n_e)')

    sc02 = axes[0, 2].scatter(gam, Zmean, c=np.log10(G0), cmap='cividis', s=sizes_T, alpha=0.9, edgecolors='none')
    axes[0, 2].set_xscale('log')
    axes[0, 2].set_xlabel(r'$\gamma$')
    axes[0, 2].set_title('color = log10(G0) ; size ~ T')
    cb02 = fig2.colorbar(sc02, ax=axes[0, 2])
    cb02.set_label(r'log10(G0)')

    # Bottom row: derived quantities
    g0_over_ne = np.maximum(G0 / np.maximum(ne, 1e-30), 1e-30)
    g0_sqrtT = np.maximum(G0 * np.sqrt(np.maximum(T, 1e-30)), 1e-30)
    sqrtT_over_ne = np.maximum(np.sqrt(np.maximum(T, 1e-30)) / np.maximum(ne, 1e-30), 1e-30)

    sc10 = axes[1, 0].scatter(gam, Zmean, c=np.log10(g0_over_ne), cmap='coolwarm', s=sizes_T, alpha=0.9, edgecolors='none')
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_xlabel(r'$\gamma$')
    axes[1, 0].set_title('color = log10(G0 / n_e) ; size ~ T')
    cb10 = fig2.colorbar(sc10, ax=axes[1, 0])
    cb10.set_label(r'log10(G0 / n_e)')

    sc11 = axes[1, 1].scatter(gam, Zmean, c=np.log10(g0_sqrtT), cmap='inferno', s=sizes_ne, alpha=0.9, edgecolors='none')
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_xlabel(r'$\gamma$')
    axes[1, 1].set_title('color = log10(G0 * sqrt(T)) ; size ~ n_e')
    cb11 = fig2.colorbar(sc11, ax=axes[1, 1])
    cb11.set_label(r'log10(G0 * sqrt(T))')

    sc12 = axes[1, 2].scatter(gam, Zmean, c=np.log10(sqrtT_over_ne), cmap='magma', s=sizes_G0, alpha=0.9, edgecolors='none')
    axes[1, 2].set_xscale('log')
    axes[1, 2].set_xlabel(r'$\gamma$')
    axes[1, 2].set_title('color = log10(sqrt(T) / n_e) ; size ~ G0')
    cb12 = fig2.colorbar(sc12, ax=axes[1, 2])
    cb12.set_label(r'log10(sqrt(T) / n_e)')

    # common formatting
    for ax in axes.ravel():
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.minorticks_on()

    # label only left column with ylabel
    for ax in axes[:, 0]:
        ax.set_ylabel(r'$\langle Z \rangle$')

    fig2.tight_layout()
    outpath = f'model_data/dust_charging_data/gamma_scan_sixcols_{grain_type}_a{a_micron:.4g}um.png'
    fig2.savefig(outpath, dpi=200)
    print('Saved figure to', outpath)