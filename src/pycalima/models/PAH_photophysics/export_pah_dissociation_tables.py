#!/usr/bin/env python
"""Export PAH acetylene dissociation tables and plots for configured PAH bins.

Outputs are written directly into model_data/PAH_dissociation_data (flat layout,
no subfolders).
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime

from pycalima.models.grain_size_config import set_config_path, load_grain_size_config, get_bins, get_export_parameters, get_model_data_dir
from pycalima.models.PAH_photophysics.PAH_photophysics import plot_acetylene_dissociation_rate


DEFAULT_EXPORT_PARAMS = {
    'g0_min': 1e-2,
    'g0_max': 1e6,
    'nh_min': 1e-2,
    'nh_max': 1e6,
}


@contextmanager
def pushd(path: Path):
    """Temporarily change the process working directory."""
    import os

    old_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _select_pah_bins(config_path: str | None = None):
    if config_path:
        set_config_path(config_path)
        load_grain_size_config(config_path=config_path, reload=True)

    pah_bins = sorted(get_bins(is_pah=True), key=lambda b: (b['bin_rank'], b.get('index', 0)))
    if not pah_bins:
        raise RuntimeError('No PAH bins found in grain-size configuration.')
    return pah_bins


def main(
    output_dir: str | None = None,
    config_path: str | None = None,
    g0_min: float | None = None,
    g0_max: float | None = None,
    nh_min: float | None = None,
    nh_max: float | None = None,
    overwrite: bool = True,
):
    """Export acetylene dissociation products for all configured PAH bins."""
    out_dir = Path(output_dir) if output_dir else (get_model_data_dir() / 'PAH_dissociation_data')
    out_dir.mkdir(parents=True, exist_ok=True)

    export_cfg = get_export_parameters('pah_dissociation', defaults=DEFAULT_EXPORT_PARAMS)
    g0_min = float(export_cfg['g0_min'] if g0_min is None else g0_min)
    g0_max = float(export_cfg['g0_max'] if g0_max is None else g0_max)
    nh_min = float(export_cfg['nh_min'] if nh_min is None else nh_min)
    nh_max = float(export_cfg['nh_max'] if nh_max is None else nh_max)

    pah_bins = _select_pah_bins(config_path=config_path)

    print('=' * 80)
    print('EXPORTING PAH ACETYLENE DISSOCIATION TABLES')
    print('=' * 80)
    print(f'Output directory: {out_dir}')
    print(f'PAH bins: {len(pah_bins)}')
    print(f'G0 range: [{g0_min:.3e}, {g0_max:.3e}]')
    print(f'nH range: [{nh_min:.3e}, {nh_max:.3e}]')
    print('=' * 80)

    tables_generated = 0
    plots_generated = 0
    failed = []

    with pushd(out_dir):
        for pah_bin in pah_bins:
            pah_bin_id = pah_bin['id']
            print(f"\n[PAH bin: {pah_bin_id}]")
            try:
                table_before = set(Path('.').glob('acetylene_dissociation_table_*.dat'))
                plot_default = Path('C54_integrated_dissociation_rate.png')
                plot_exists_before = plot_default.exists()

                plot_acetylene_dissociation_rate(
                    G0min=g0_min,
                    G0max=g0_max,
                    nHmin=nh_min,
                    nHmax=nh_max,
                    pah_bin_id=pah_bin_id,
                )

                table_after = set(Path('.').glob('acetylene_dissociation_table_*.dat'))
                new_tables = sorted(table_after - table_before)

                if not new_tables:
                    new_tables = sorted(table_after)

                if new_tables:
                    for it, src_table in enumerate(new_tables):
                        if it == 0:
                            dst_table = Path(f'dissociation_{pah_bin_id}.dat')
                        else:
                            dst_table = Path(f'dissociation_{pah_bin_id}_{it}.dat')
                        if dst_table.exists() and not overwrite:
                            print(f'  - table exists, keeping existing: {dst_table.name}')
                        else:
                            src_table.replace(dst_table)
                            tables_generated += 1

                if plot_default.exists():
                    target_plot = Path(f'{pah_bin_id}_integrated_dissociation_rate.png')
                    if target_plot.exists() and not overwrite:
                        print(f'  - plot exists, keeping existing: {target_plot.name}')
                    else:
                        plot_default.replace(target_plot)
                        plots_generated += 1
                elif not plot_exists_before:
                    print('  - warning: dissociation plot was not generated')

            except Exception as exc:
                failed.append({'pah_bin_id': pah_bin_id, 'error': str(exc)})
                print(f'  - error: {exc}')

    all_files = list(out_dir.glob('*.dat')) + list(out_dir.glob('*.png'))

    print('\n' + '=' * 80)
    print('PAH DISSOCIATION EXPORT SUMMARY')
    print('=' * 80)
    print(f'  Bins attempted: {len(pah_bins)}')
    print(f'  Tables generated: {tables_generated}')
    print(f'  Plots generated: {plots_generated}')
    print(f'  Failed bins: {len(failed)}')
    print(f'  Files in output dir: {len(all_files)}')

    return {
        'output_dir': str(out_dir),
        'file_count': len(all_files),
        'tables_generated': tables_generated,
        'plots_generated': plots_generated,
        'bins_processed': len(pah_bins) - len(failed),
        'failed': len(failed),
        'failed_bins': failed,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Export PAH acetylene dissociation tables and plots for configured PAH bins.'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory. Default: model_data/PAH_dissociation_data'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to JSON grain-size configuration file.'
    )
    parser.add_argument('--g0-min', type=float, default=None, help='Minimum G0 value.')
    parser.add_argument('--g0-max', type=float, default=None, help='Maximum G0 value.')
    parser.add_argument('--nh-min', type=float, default=None, help='Minimum nH value [cm^-3].')
    parser.add_argument('--nh-max', type=float, default=None, help='Maximum nH value [cm^-3].')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite per-bin plot files if they exist.')
    args = parser.parse_args()

    result = main(
        output_dir=args.output,
        config_path=args.config,
        g0_min=args.g0_min,
        g0_max=args.g0_max,
        nh_min=args.nh_min,
        nh_max=args.nh_max,
        overwrite=args.overwrite,
    )

    print('\nExport completed!')
    print(f"Outputs saved to: {result['output_dir']}")
