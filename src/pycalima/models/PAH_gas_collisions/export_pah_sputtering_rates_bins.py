#!/usr/bin/env python
"""Export PAH sputtering rate tables (phi=0) for all PAH bins.

This script reads all PAH bins from the grain-size configuration, computes
phi=0 sputtering rates using export_rates_simple, and writes outputs to
model_data/pah_sputtering_data. It also saves one quick-look plot per bin.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from pycalima.models.grain_size_config import set_config_path, get_bins, get_lognormal_parameters, get_export_parameters, get_model_data_dir
from pycalima.models.PAH_gas_collisions.PAH_sputtering import export_rates_simple


DEFAULT_EXPORT_PARAMS = {
    'Tmin': 1e3,
    'Tmax': 1e9,
    'nT': 100,
    'nbins_v': 300,
    'nbins_theta': 30,
}




def _save_quicklook_plot(output_path, T, J_e, J_electronic, J_ion, title):
    """Save a quick validation plot for PAH sputtering rates."""
    sns.set_theme(style="white")
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": "Computer Modern Roman",
    })

    fig, ax = plt.subplots(1, 1, figsize=(7, 5), dpi=180)

    ax.loglog(T, 2.0 * np.asarray(J_e), linewidth=2.2, label='Electrons ($2J_e$)')

    for ptype in ['H', 'He', 'C', 'O']:
        total = 2.0 * np.asarray(J_electronic[ptype]) + np.asarray(J_ion[ptype])
        ax.loglog(T, total, linewidth=2.0, label=f'{ptype} total')

    ax.set_xlabel(r'$T$ [K]')
    ax.set_ylabel(r'Rate [cm$^3$ s$^{-1}$]')
    ax.set_title(title)
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(frameon=False, fontsize=10)
    ax.set_ylim([1e-17,1e-2])
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main(config_path=None):
    if config_path:
        set_config_path(config_path)

    params_cfg = get_export_parameters('pah_sputtering', defaults=DEFAULT_EXPORT_PARAMS)
    Tmin = float(params_cfg['Tmin'])
    Tmax = float(params_cfg['Tmax'])
    nT = int(params_cfg['nT'])
    nbins_v = int(params_cfg['nbins_v'])
    nbins_theta = int(params_cfg['nbins_theta'])

    output_dir = get_model_data_dir() / 'pah_sputtering_data'
    output_dir.mkdir(parents=True, exist_ok=True)

    bins = sorted(
        get_bins(is_pah=True),
        key=lambda b: (b['composition'], b['bin_rank'], b['index']),
    )
    if not bins:
        raise RuntimeError('No PAH bins found in grain-size configuration.')

    print('=' * 80)
    print('Exporting PAH sputtering rates (phi=0) for PAH bins')
    print('=' * 80)
    print(f'Output directory: {output_dir}')
    print(f'Temperature grid: [{Tmin:.2e}, {Tmax:.2e}] with nT={nT}')
    print('=' * 80)

    created_tables = []
    created_figures = []

    for bin_info in bins:
        bin_id = bin_info['id']
        comp = bin_info['composition']
        rank = int(bin_info['bin_rank'])
        params = get_lognormal_parameters(bin_id, model_name='basic')
        grain_size_micron = float(params['a0'])
        pah_label = bin_id

        print(
            f"\n[bin={bin_id}] composition={comp}, rank={rank}, "
            f"grain_size={grain_size_micron:.4e} micron"
        )

        T, J_e, J_electronic, J_ion = export_rates_simple(
            RPAH=grain_size_micron,
            Tmin=Tmin,
            Tmax=Tmax,
            nT=nT,
            nbins_v=nbins_v,
            nbins_theta=nbins_theta,
            pah_label=pah_label,
            output_dir=str(output_dir),
        )

        # 1 electron file (Z=0) + 4 ion files (H,He,C,O)
        table_paths = [
            output_dir / f'pah_sputtering_{pah_label}_Z_0',
            output_dir / f'pah_sputtering_{pah_label}_Z_1',
            output_dir / f'pah_sputtering_{pah_label}_Z_2',
            output_dir / f'pah_sputtering_{pah_label}_Z_6',
            output_dir / f'pah_sputtering_{pah_label}_Z_8',
        ]
        for p in table_paths:
            if p.exists():
                renamed = output_dir / p.name.replace('pah_sputtering_', 'sputtering_', 1)
                p.replace(renamed)
                created_tables.append(str(renamed))

        fig_path = output_dir / f'sputtering_{pah_label}_quicklook.png'
        _save_quicklook_plot(
            fig_path,
            T,
            J_e,
            J_electronic,
            J_ion,
            title=f'PAH sputtering, bin {rank}, a0={grain_size_micron:.4g} micron',
        )
        created_figures.append(str(fig_path))
        print(f'    Saved quick-look plot: {fig_path}')

    print('\nDone.')
    print(f'Created {len(created_tables)} tables and {len(created_figures)} figures in {output_dir}')

    return {
        'output_dir': str(output_dir),
        'file_count': len(created_tables),
        'figure_count': len(created_figures),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Export PAH sputtering rates (phi=0) for all PAH bins.'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to JSON grain size configuration file. If not provided, uses default.'
    )
    args = parser.parse_args()

    main(config_path=args.config)
