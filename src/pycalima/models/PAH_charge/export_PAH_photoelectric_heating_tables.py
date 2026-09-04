#!/usr/bin/env python
"""Export PAH photoelectric heating efficiency and population tables.

This script generates photoelectric heating efficiency and ionization state tables
for PAH molecules under various radiation field conditions and ISM parameters.

Tables are computed as function of gamma = G0*sqrt(T)/ne for different stellar
radiation fields and are suitable for interpolation in astrophysical simulations.

Tables are saved in model_data/PAH_photoelectric_heating_data/ subdirectory.

Uses generalized PAH bin definitions from grain_size_distribution.json.
"""

import argparse
from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add models path for imports

from pycalima.models.PAH_charge.PAH_photoelectric_heating import compute_tables_ISRF
from pycalima.models.grain_size_config import load_grain_size_config, set_config_path, get_bins, get_lognormal_parameters, get_export_parameters, get_model_data_dir
from pycalima.models.tools.utils import Nc_from_size


DEFAULT_EXPORT_PARAMS = {
    'temperature': 1000.0,
    'G0': 1.0,
    'ne_min': 1e-5,
    'ne_max': 1e5,
    'n_ne': 100,
    'radiation_models': ['Draine', 'Habing', 'Mathis', 'O6V', 'B0V', 'A0'],
    'optical_models': ['Malloci', 'Draine'],
    'attachment_models': ['Berne'],
}

def _get_pah_bins(config_path=None):
    """Load PAH bins from grain_size_distribution.json.
    
    Returns
    -------
    list of dict
        PAH bin information with keys: id, composition, bin_rank, index, a0, amin, amax, sigma, s, Nc_estimate
    """
    if config_path is not None:
        set_config_path(config_path)
        load_grain_size_config(config_path=config_path, reload=True)

    pah_bins = get_bins(is_pah=True)
    
    # Add lognormal parameters for each PAH bin
    for bin_info in pah_bins:
        bin_id = bin_info['id']
        params = get_lognormal_parameters(bin_id, model_name='basic')
        a0_micron = float(params['a0'])
        a0_cm = a0_micron * 1e-4  # Convert microns to cm
        a0_angstrom = a0_micron * 1e4  # Convert microns to Angstrom
        
        bin_info['a0'] = a0_micron
        bin_info['a0_cm'] = a0_cm
        bin_info['amin'] = float(params['amin'])
        bin_info['amax'] = float(params['amax'])
        bin_info['sigma'] = float(params['sigma'])
        bin_info['s'] = float(params['s'])
        
        # Estimate Nc from grain size using models.tools.utils.Nc_from_size()
        # which follows Draine et al. (2021) Eq. 8: Nc = 418 * (a/10)^3
        # where a is in Angstrom
        bin_info['Nc_estimate'] = Nc_from_size(a0_angstrom)
    
    return pah_bins


def _repo_root():
    """Get repository root."""
    return Path(__file__).resolve().parents[2]


def _setup_plotting():
    """Configure matplotlib for publication-quality plots."""
    sns.set_theme(style="white")
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": "Computer Modern Roman",
    })


def main(output_root=None, radiation_models=None, optical_models=None, 
         attachment_models=None, T_gas=None, G0_field=None, pah_bins=None,
         config_path=None, overwrite=False):
    """
    Export PAH photoelectric heating tables for a single temperature and varying electron density.
    
    Parameters
    ----------
    output_root : Path, optional
        Root output directory. If None, uses model_data/PAH_photoelectric_heating_data/
    radiation_models : list, optional
        List of radiation model names to process. If None, uses default list.
    optical_models : list, optional
        List of optical model names. If None, uses Draine + Malloci.
    attachment_models : list, optional
        List of attachment models. If None, uses Berne.
    T_gas : float, optional
        Gas temperature in K. If None, uses default 1000 K.
    G0_field : float, optional
        Radiation field intensity in Habing units. If None, uses default 1.0.
    pah_bins : list, optional
        List of PAH bin dicts from get_bins(is_pah=True). If None, loads from JSON.
    config_path : str, optional
        Path to a grain size configuration JSON file. If provided, only this file is used.
    overwrite : bool, optional
        Whether to overwrite existing files. Default is False.
    
    Returns
    -------
    dict
        Summary of export results with counts and file paths
    """
    export_cfg = get_export_parameters('pah_photoelectric_heating', defaults=DEFAULT_EXPORT_PARAMS)

    # Use provided values or defaults from config
    T_gas = T_gas if T_gas is not None else float(export_cfg['temperature'])
    G0_field = G0_field if G0_field is not None else float(export_cfg['G0'])
    ne_min = float(export_cfg['ne_min'])
    ne_max = float(export_cfg['ne_max'])
    n_ne = int(export_cfg['n_ne'])
    
    if pah_bins is None:
        pah_bins = _get_pah_bins(config_path=config_path)
    
    if output_root is None:
        output_root = get_model_data_dir() / 'PAH_photoelectric_heating_data'
    else:
        output_root = Path(output_root)
    
    output_root.mkdir(parents=True, exist_ok=True)
    
    if radiation_models is None:
        radiation_models = list(export_cfg['radiation_models'])
    
    if optical_models is None:
        optical_models = list(export_cfg['optical_models'])
    
    if attachment_models is None:
        attachment_models = list(export_cfg['attachment_models'])
    
    print('=' * 80)
    print('Exporting PAH photoelectric heating tables')
    print('=' * 80)
    print(f'Output directory: {output_root}')
    print(f'Temperature: {T_gas:.1f} K')
    print(f'Radiation field (G0): {G0_field:.1f}')
    print(f'Electron density range: [{ne_min:.1f}, {ne_max:.1e}] cm⁻³ with n_ne={n_ne}')
    print(f'Radiation models: {radiation_models}')
    print(f'Optical models: {optical_models}')
    print(f'Attachment models: {attachment_models}')
    print(f'PAH bins: {len(pah_bins)} bins loaded from grain_size_distribution.json')
    for pb in pah_bins:
        print(f'  - {pb["id"]:20s} (a0={pb["a0"]:.2e} μm, Nc={pb.get("Nc_estimate", "?"):5})')
    print('=' * 80)
    
    _setup_plotting()
    
    created_files = []
    results_summary = []
    failed_tables = []
    
    import os
    
    print(f"\nComputing tables for T={T_gas:.1f} K, G0={G0_field:.1f}")
    
    for pah_bin in pah_bins:
        pah_bin_id = pah_bin['id']
        pah_a0 = pah_bin['a0']
        pah_Nc = pah_bin['Nc_estimate']
        
        print(f"\n[PAH bin: {pah_bin_id} (Nc={pah_Nc}, a0={pah_a0:.2e} μm)]")
        
        for rad_model in radiation_models:
            for opt_model in optical_models:
                for att_model in attachment_models:
                    try:
                        print(f"  {rad_model:15s} + {opt_model:10s} + {att_model:10s}...", end=' ', flush=True)
                        
                        compute_tables_ISRF(
                            Nc=pah_Nc,
                            a0=pah_bin['a0'],
                            amin=pah_bin['amin'],
                            amax=pah_bin['amax'],
                            sigma=pah_bin['sigma'],
                            s=pah_bin['s'],
                            T=T_gas,
                            ne_min=ne_min,
                            ne_max=ne_max,
                            n_ne=n_ne,
                            radiation_model=rad_model,
                            op_model=opt_model,
                            attach_model=att_model,
                            output_dir=output_root,
                            file_prefix=f'{pah_bin_id}',
                        )
                        
                        # Register created files written directly into output_root.
                        created_files.extend(str(path) for path in output_root.glob(f'peh_ISRF_{rad_model}_{opt_model}_{att_model}_*.dat'))
                        created_files.extend(str(path) for path in output_root.glob(f'peh_ISRF_{rad_model}_{opt_model}_{att_model}_*.pdf'))
                        
                        print(f"✓")
                        
                        results_summary.append({
                            'pah_bin_id': pah_bin_id,
                            'a0_micron': pah_a0,
                            'temperature_K': T_gas,
                            'G0': G0_field,
                            'radiation_model': rad_model,
                            'optical_model': opt_model,
                            'attachment_model': att_model,
                            'status': 'Success',
                        })
                        
                    except Exception as e:
                        print(f"✗ Error: {e}")
                        failed_tables.append({
                            'pah_bin_id': pah_bin_id,
                            'a0_micron': pah_a0,
                            'temperature_K': T_gas,
                            'G0': G0_field,
                            'radiation_model': rad_model,
                            'optical_model': opt_model,
                            'attachment_model': att_model,
                            'error': str(e),
                        })
                        results_summary.append({
                            'pah_bin_id': pah_bin_id,
                            'a0_micron': pah_a0,
                            'temperature_K': T_gas,
                            'G0': G0_field,
                            'radiation_model': rad_model,
                            'optical_model': opt_model,
                            'attachment_model': att_model,
                            'status': f'Error: {str(e)[:60]}',
                        })
    
    # Create index file with table metadata
    index_data = {
        'description': 'PAH photoelectric heating efficiency tables (generalized PAH bins)',
        'computation_parameters': {
            'temperature_K': float(T_gas),
            'G0': float(G0_field),
            'electron_density_cm3': {
                'min': float(ne_min),
                'max': float(ne_max),
                'n_points': int(n_ne),
            },
        },
        'pah_bins': [
            {
                'id': pb['id'],
                'composition': pb['composition'],
                'bin_rank': pb['bin_rank'],
                'a0_micron': pb['a0'],
                'a0_cm': pb['a0_cm'],
            }
            for pb in pah_bins
        ],
        'radiation_models': radiation_models,
        'optical_models': optical_models,
        'attachment_models': attachment_models,
        'table_structure': {
            'gamma': 'log10(G0 * sqrt(T) / ne) [K^0.5 cm^-3]',
            'efficiency': 'log10(photoelectric heating efficiency)',
            'Prad': 'log10(radiation pressure) [erg/s]',
            'f_anion': 'anion population fraction',
            'f_neutral': 'neutral population fraction',
            'f_cation': 'cation (Z=+1) population fraction',
            'f_dication': 'dication (Z=+2) population fraction',
        },
        'results': {
            'tables_generated': len([r for r in results_summary if r['status'] == 'Success']),
            'tables_failed': len(failed_tables),
            'output_directory': str(output_root),
        }
    }
    
    index_path = output_root / 'index.json'
    with open(index_path, 'w') as f:
        json.dump(index_data, f, indent=2)
    print(f"\n✓ Index file saved: {index_path.name}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("PAH PHOTOELECTRIC HEATING TABLE EXPORT SUMMARY")
    print("=" * 80)
    successful = sum(1 for r in results_summary if r['status'] == 'Success')
    failed = sum(1 for r in results_summary if 'Error' in r['status'])
    print(f"  Total configurations attempted: {len(results_summary)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Files created: {len(created_files)}")
    if failed_tables:
        print(f"\n  Failed configurations:")
        for fail in failed_tables[:5]:  # Show first 5 failures
            print(f"    - T={fail['temperature_K']:.1f}K {fail['radiation_model']} "
                  f"{fail['optical_model']} {fail['attachment_model']}: {fail['error'][:50]}")
        if len(failed_tables) > 5:
            print(f"    ... and {len(failed_tables) - 5} more")
    
    print("=" * 80)
    
    return {
        'output_dir': str(output_root),
        'file_count': len(created_files),
        'tables_generated': successful,
        'failed': failed,
        'index_file': str(index_path),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Export PAH photoelectric heating efficiency and population tables for a single temperature.'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory. If not provided, uses model_data/PAH_photoelectric_heating_data/'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=DEFAULT_EXPORT_PARAMS['temperature'],
        help=f"Gas temperature [K]. Default: {DEFAULT_EXPORT_PARAMS['temperature']}"
    )
    parser.add_argument(
        '--G0',
        type=float,
        default=DEFAULT_EXPORT_PARAMS['G0'],
        help=f"Radiation field intensity (Habing units). Default: {DEFAULT_EXPORT_PARAMS['G0']}"
    )
    parser.add_argument(
        '--radiation-models',
        type=str,
        nargs='+',
        default=None,
        help='Radiation models to process (e.g., Draine Habing O6V)'
    )
    parser.add_argument(
        '--optical-models',
        type=str,
        nargs='+',
        default=None,
        help='Optical models to use (default: Draine Malloci)'
    )
    parser.add_argument(
        '--attachment-models',
        type=str,
        nargs='+',
        default=None,
        help='Attachment models to use (default: Berne)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to grain size configuration JSON file. If not provided, uses the default grain_size_distribution.json.'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing files'
    )
    
    args = parser.parse_args()
    
    result = main(
        output_root=args.output,
        radiation_models=args.radiation_models,
        optical_models=args.optical_models,
        attachment_models=args.attachment_models,
        config_path=args.config,
        overwrite=args.overwrite,
        T_gas=args.temperature,
        G0_field=args.G0,
    )
    
    print("\nExport completed!")
    print(f"Tables saved to: {result['output_dir']}")
