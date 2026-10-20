#!/usr/bin/env python3
"""Grain potential in a quasar spectrum: compare with WDB06 Figs. 6a and 7a.

Weingartner, Draine & Barr (2006) sec. 10.1: the unobscured quasar spectrum of
Sazonov et al. (2004) (their eq. 29), ionisation parameter U = n_gamma(>13.6 eV)
/ n_H = 0.1, 1, 10, 100, gas at T = 1e4, 1e5, 1e6 K. We take a fully ionised
pure-H gas with n_e = n_p = n_H = 1 cm^-3 (WDB06 do not state n_e/n_H).

Paper values read off the figures at a = 0.1 um, U = 100:
carbonaceous ~53 / 78 / 100 V, silicate ~62 / 85 / 110 V (T = 1e4 / 1e5 / 1e6 K).

pyCALIMA averages the gas-electron secondary yield over thermal impact energies
(see wdb06_yields), where WDB06 evaluate it at the mean energy; expect small
differences from the figures where gas secondaries matter (low U, T >= 1e5 K).

The solve grid is cached to an .npz next to the plot; pass --recompute to redo it.
"""
import argparse
import warnings

import numpy as np

from pycalima import _paths
from pycalima.models.dust_charge.dust_charging import equilibrium_charge_for_grain

EV2ERG = 1.602176634e-12
C_CGS = 2.99792458e10
E2_OVER_A_V_CM = 1.439964e-7  # e / a in volts for a in cm, times Z

U_VALUES = (0.1, 1.0, 10.0, 100.0)
T_VALUES = (1e4, 1e5, 1e6)
A_UM = np.geomspace(1e-3, 1.0, 10)
MATERIALS = ("graphite", "silicate")


def sazonov_shape(E):
    """WDB06 eq. 29 without the normalisation alpha."""
    E = np.asarray(E, dtype=float)
    return np.where(E < 10.0, 0.0798 * E ** -0.6,
                    np.where(E < 2e3, E ** -1.7 * np.exp(E / 2e3),
                             2.94e-3 * E ** -0.8 * np.exp(-E / 2e5)))


def sazonov_field(U, n_H=1.0, E_min=1.0, E_max=2e4, n=1500):
    """[E_eV, lambda_nm, I_E (erg s^-1 cm^-2 eV^-1)] with I_E = c u_E and
    alpha = 2.26e-10 erg eV^-1 n_gamma(>13.6 eV) (WDB06 eq. 29)."""
    E = np.geomspace(E_min, E_max, n)
    alpha = 2.26e-10 * U * n_H
    return np.column_stack([E, 1239.84193 / E, C_CGS * alpha * sazonov_shape(E)])


def solve_grid():
    phi = np.full((len(MATERIALS), len(T_VALUES), len(U_VALUES), A_UM.size), np.nan)
    for im, mat in enumerate(MATERIALS):
        for iT, T in enumerate(T_VALUES):
            ions = [{'n': 1.0, 'T': T, 'm': 1.6726219e-27, 'z': 1}]
            for iU, U in enumerate(U_VALUES):
                field = sazonov_field(U)
                for ia, a_um in enumerate(A_UM):
                    a = a_um * 1e-4
                    out = equilibrium_charge_for_grain(
                        1.0, 1.0, T, mat, a, rad_field=field, ion_species=ions,
                        yield_params={'yield_model': 'WDB06'})
                    phi[im, iT, iU, ia] = out[3] * E2_OVER_A_V_CM / a
                print(f'{mat} T={T:.0e} U={U:g}: phi(0.1um) = '
                      f'{np.interp(-1, np.log10(A_UM), phi[im, iT, iU]):.1f} V', flush=True)
    return phi


def plot(phi, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colors = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100')  # categorical slots 1-4
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), sharex=True, sharey='row')
    for im, mat in enumerate(MATERIALS):
        for iT, T in enumerate(T_VALUES):
            ax = axes[im, iT]
            for iU, U in enumerate(U_VALUES):
                ax.plot(A_UM, phi[im, iT, iU], color=colors[iU], lw=2, marker='o', ms=4,
                        label=f'U = {U:g}')
                ax.annotate(f'U={U:g}', (A_UM[-1], phi[im, iT, iU, -1]), xytext=(4, 0),
                            textcoords='offset points', va='center', fontsize=8, color='0.3')
            ax.set_xscale('log')
            ax.set_title(f'{"carbonaceous" if mat == "graphite" else "silicate"}, T = {T:.0e} K',
                         fontsize=10)
            ax.grid(color='0.9', lw=0.6)
            ax.set_xlim(1e-3, 2.5)
            if iT == 0:
                ax.set_ylabel(r'$\phi$ (V)')
            if im == 1:
                ax.set_xlabel(r'$a$ ($\mu$m)')
    axes[0, 0].legend(frameon=False, fontsize=8, loc='upper left')
    fig.suptitle('WDB06 yields, Sazonov et al. (2004) quasar spectrum: compare WDB06 Figs. 6a, 7a',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# Left panels of page 25 of the arXiv preprint at 300 dpi: frame (top, bottom,
# left, right) px and (a_left, a_right, phi_bottom, phi_top). a ticks 0.01/0.1 um
# at 640/886 px; phi ticks every 20 V at 140 px (Fig. 6a) and 128 px (Fig. 7a).
OVERLAY = {'graphite': ((519, 1254, 394, 1133), (1e-3, 1.0, 0.0, 105.0)),
           'silicate': ((1998, 2733, 394, 1132), (1e-3, 1.0, 0.0, 115.0))}


def plot_overlay(phi, pdf, path, phi_ref=None, ref_label='reference'):
    """Our phi(a) (colour) over the paper's Figs. 6a and 7a (black)."""
    import subprocess
    import tempfile
    from pathlib import Path

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    colors = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100')
    with tempfile.TemporaryDirectory() as tmp:
        stem = Path(tmp) / 'p25'
        subprocess.run(['pdftoppm', '-f', '25', '-l', '25', '-r', '300', '-gray', '-png',
                        '-singlefile', str(pdf), str(stem)], check=True)
        page = mpimg.imread(stem.with_suffix('.png'))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    for ax, (im, mat) in zip(axes, enumerate(MATERIALS)):
        (t, b, l, r), (a0, a1, p0, p1) = OVERLAY[mat]
        ext = (np.log10(a0), np.log10(a1), p0, p1)
        ax.imshow(page[t:b + 1, l:r + 1], cmap='gray', extent=ext, aspect='auto',
                  vmin=0, vmax=1, alpha=0.75, zorder=0)
        for iU, U in enumerate(U_VALUES):
            for iT in range(len(T_VALUES)):
                ax.plot(np.log10(A_UM), phi[im, iT, iU], color=colors[iU], lw=1.8,
                        label=f'U = {U:g}' if iT == 0 else None, zorder=2)
                if phi_ref is not None:
                    ax.plot(np.log10(A_UM), phi_ref[im, iT, iU], color=colors[iU], lw=1.2,
                            ls=':', zorder=2)
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(p0, p1)
        ax.set_xticks([-3, -2, -1, 0], ['$10^{-3}$', '$10^{-2}$', '0.1', '1'])
        ax.set_xlabel(r'$a$ ($\mu$m)')
        ax.set_ylabel(r'$\phi$ (V)')
        ax.set_title(f'{"Fig. 6a: carbonaceous" if mat == "graphite" else "Fig. 7a: silicate"}'
                     ' (three curves per U: T = 1e4, 1e5, 1e6 K, bottom to top)', fontsize=9)
        ax.legend(frameon=True, framealpha=0.9, fontsize=8, loc='upper right')
    note = f'; dotted: {ref_label}' if phi_ref is not None else ''
    fig.suptitle('pyCALIMA (colour) over WDB06 Figs. 6a/7a (black)' + note, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--recompute', action='store_true')
    ap.add_argument('--pdf', help='WDB06 arXiv preprint, to overlay Figs. 6a/7a')
    ap.add_argument('--reference', help='another cached .npz grid to draw dotted on the overlay')
    ap.add_argument('--reference-label', default='reference', help='legend text for --reference')
    args = ap.parse_args()
    warnings.filterwarnings('ignore')

    out_dir = _paths.get_plots_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / 'wdb06_quasar_potential.npz'
    if cache.exists() and not args.recompute:
        phi = np.load(cache)['phi']
    else:
        phi = solve_grid()
        np.savez(cache, phi=phi, a_um=A_UM, U=U_VALUES, T=T_VALUES, materials=MATERIALS)
    png = out_dir / 'wdb06_quasar_potential.png'
    plot(phi, png)
    print('Saved', png)
    if args.pdf:
        ref = np.load(args.reference)['phi'] if args.reference else None
        over = out_dir / 'wdb06_quasar_potential_overlay.png'
        plot_overlay(phi, args.pdf, over, ref, args.reference_label)
        print('Saved', over)


if __name__ == '__main__':
    main()
