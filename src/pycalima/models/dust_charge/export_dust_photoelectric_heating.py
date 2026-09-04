#!/usr/bin/env python
"""Export dust photoelectric heating and cooling rates for all dust bins.

This script reads all dust bins from the grain-size configuration and computes
heating and cooling rates as function of temperature and gamma = G0*sqrt(T)/ne.

Outputs are 2D rate tables (gamma × temperature grid) suitable for interpolation
in simulations, plus quick-look plots showing how rates vary with ISM conditions.
"""

import argparse
from pathlib import Path
import json
import concurrent.futures
import re

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pycalima.models.grain_size_config import set_config_path, get_bins, get_lognormal_parameters, get_export_parameters, get_model_data_dir
from pycalima.models.dust_charge.dust_photoelectric_heating import make_rate_gamma_T_tables


DEFAULT_EXPORT_PARAMS = {
    'Tmin': 10.0,
    'Tmax': 1e5,
    'nT': 50,
    'gamma_min': 1e-6,
    'gamma_max': 1e6,
    'n_gamma': 100,
    'radiation_model': 'Mathis',
    'mode': 'fix_G0',
    'fixed_value': 1.0,
    'n_workers': None,
}


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


def _save_rate_plot(output_path, T, gamma, rates, radiation_model, rate_type='heating', grain_type='graphite'):
    """
    Save quick-look plot showing heating/cooling rates vs temperature and gamma.
    
    Parameters
    ----------
    output_path : Path
        Output file path
    T : array
        Temperature grid [K]
    gamma : array
        Gamma grid [G0*sqrt(T)/ne]
    rates : dict
        Dictionary with keys like 'peh_rate', 'rec_rate', etc.
        Each is a 2D array (n_gamma, n_T) or 1D array
    rate_type : str
        'heating', 'cooling', or 'both'
    grain_type : str
        'graphite' or 'silicate'
    """
    _setup_plotting()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=180)
    
    # Extract rates and ensure proper shape
    peh_rate = rates.get('peh_rate', np.zeros(len(T)))
    rec_rate = rates.get('rec_rate', np.zeros(len(T)))
    
    # Handle array shapes
    if peh_rate.ndim == 1:
        # 1D array - replicate for all gamma values
        peh_rate = np.tile(peh_rate, (len(gamma), 1))
    elif peh_rate.size == len(T):
        # 1D array disguised as something else
        peh_rate = np.tile(peh_rate, (len(gamma), 1))
        
    if rec_rate.ndim == 1:
        # 1D array - replicate for all gamma values
        rec_rate = np.tile(rec_rate, (len(gamma), 1))
    elif rec_rate.size == len(T):
        # 1D array disguised as something else
        rec_rate = np.tile(rec_rate, (len(gamma), 1))
    
    # Ensure T and gamma are proper sizes
    T = np.atleast_1d(T).flatten()
    gamma = np.atleast_1d(gamma).flatten()
    
    # Verify shapes match
    if peh_rate.shape[1] != len(T):
        # Transpose or reshape if needed
        if peh_rate.shape[0] == len(T):
            peh_rate = peh_rate.T
        elif peh_rate.size == len(T):
            peh_rate = peh_rate.reshape(1, -1)
            peh_rate = np.repeat(peh_rate, len(gamma), axis=0)
    
    if rec_rate.shape[1] != len(T):
        # Transpose or reshape if needed
        if rec_rate.shape[0] == len(T):
            rec_rate = rec_rate.T
        elif rec_rate.size == len(T):
            rec_rate = rec_rate.reshape(1, -1) 
            rec_rate = np.repeat(rec_rate, len(gamma), axis=0)
    
    # Select representative gamma values for line plots
    if len(gamma) > 1:
        gamma_indices = [0, len(gamma)//4, len(gamma)//2, 3*len(gamma)//4, -1]
        gamma_vals_plot = [gamma[i] for i in gamma_indices if i < len(gamma)]
    else:
        gamma_vals_plot = [gamma[0]]
    
    # Plot 1: Heating rate vs T for different gamma
    ax = axes[0, 0]
    for gi, gval in enumerate(gamma_vals_plot):
        if gi == len(gamma_vals_plot) - 1:
            idx = min(-1, len(gamma)-1)
        else:
            idx = np.searchsorted(gamma, gval)
            idx = min(idx, len(gamma)-1)
        if idx < len(peh_rate):
            ax.loglog(T, np.abs(peh_rate[idx, :]), marker='o', markersize=3, 
                      label=rf'$\gamma$ = {gval:.2e}', linewidth=1.5, alpha=0.8)
    ax.set_xlabel(r'$T$ [K]', fontsize=12)
    ax.set_ylabel(r'$\Lambda_{\rm PEH}$ [erg cm$^3$ s$^{-1}$]', fontsize=12)
    ax.set_title('Photoelectric Heating Rate', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, frameon=False, loc='best')
    ax.grid(True, which='both', alpha=0.3)
    
    # Plot 2: Cooling rate vs T for different gamma
    ax = axes[0, 1]
    for gi, gval in enumerate(gamma_vals_plot):
        if gi == len(gamma_vals_plot) - 1:
            idx = min(-1, len(gamma)-1)
        else:
            idx = np.searchsorted(gamma, gval)
            idx = min(idx, len(gamma)-1)
        if idx < len(rec_rate):
            ax.loglog(T, np.abs(rec_rate[idx, :]), marker='s', markersize=3,
                      label=rf'$\gamma$ = {gval:.2e}', linewidth=1.5, alpha=0.8)
    ax.set_xlabel(r'$T$ [K]', fontsize=12)
    ax.set_ylabel(r'$\Lambda_{\rm rec}$ [erg cm$^3$ s$^{-1}$]', fontsize=12)
    ax.set_title('Recombination Cooling Rate', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, frameon=False, loc='best')
    ax.grid(True, which='both', alpha=0.3)
    
    # Plot 3: Heating rate 2D heatmap (gamma vs T)
    ax = axes[1, 0]
    peh_valid = np.maximum(np.abs(peh_rate), 1e-30)
    im = ax.pcolormesh(T, gamma, np.log10(peh_valid), shading='auto', cmap='viridis')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$T$ [K]', fontsize=12)
    ax.set_ylabel(r'$\gamma$ [G$_0$ $\sqrt{T}$ / $n_e$]', fontsize=12)
    ax.set_title('log10(PEH Rate)', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label=r'log10($\Lambda_{\rm PEH}$)')
    
    # Plot 4: Cooling rate 2D heatmap (gamma vs T)
    ax = axes[1, 1]
    rec_valid = np.maximum(np.abs(rec_rate), 1e-30)
    im = ax.pcolormesh(T, gamma, np.log10(rec_valid), shading='auto', cmap='plasma')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$T$ [K]', fontsize=12)
    ax.set_ylabel(r'$\gamma$ [G$_0$ $\sqrt{T}$ / $n_e$]', fontsize=12)
    ax.set_title('log10(Recombination Rate)', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label=r'log10($\Lambda_{\rm rec}$)')
    
    # Add grain info as text
    fig.suptitle(f'{grain_type.capitalize()} dust grain - {radiation_model} radiation field',
                 fontsize=14, fontweight='bold', y=0.995)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def main(config_path=None):
    """
    Export dust photoelectric heating and cooling rates for all dust bins.
    
    Parameters
    ----------
    config_path : str, optional
        Path to grain size configuration JSON file
    """
    if config_path:
        set_config_path(config_path)

    params_cfg = get_export_parameters('dust_photoelectric_heating', defaults=DEFAULT_EXPORT_PARAMS)
    Tmin = float(params_cfg['Tmin'])
    Tmax = float(params_cfg['Tmax'])
    nT = int(params_cfg['nT'])
    gamma_min = float(params_cfg['gamma_min'])
    gamma_max = float(params_cfg['gamma_max'])
    n_gamma = int(params_cfg['n_gamma'])
    radiation_model = str(params_cfg['radiation_model'])
    mode = str(params_cfg['mode'])
    fixed_value = float(params_cfg['fixed_value'])
    n_workers_cfg = params_cfg.get('n_workers')
    n_workers = None if n_workers_cfg is None else int(n_workers_cfg)

    output_dir = get_model_data_dir() / 'dust_photoelectric_heating_data'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove legacy filenames that used composition tags (Gra/suvSil).
    legacy_patterns = [
        re.compile(r'^log10_Ts_(Gra|suvSil)_.+\.dat$'),
        re.compile(r'^log10_gammas_(Gra|suvSil)_.+\.dat$'),
        re.compile(r'^dust_rates_(heating|cooling)_.+_(Gra|suvSil)_.+\.dat$'),
        re.compile(r'^dust_rates_vs_gamma_by_temperature_.+_(Gra|suvSil)_.+\.pdf$'),
        re.compile(r'^log10_Ts_.+\.dat$'),
        re.compile(r'^log10_gammas_.+\.dat$'),
        re.compile(r'^dust_rates_peh_.+\.dat$'),
        re.compile(r'^dust_rates_rec_.+\.dat$'),
    ]
    for existing in output_dir.iterdir():
        if existing.is_file() and any(p.match(existing.name) for p in legacy_patterns):
            existing.unlink()

    bins = sorted(
        get_bins(is_pah=False),
        key=lambda b: (b['composition'], b['bin_rank'], b['index']),
    )
    if not bins:
        raise RuntimeError('No dust bins found in grain-size configuration.')

    print('=' * 80)
    print('Exporting dust photoelectric heating rates for all dust bins')
    print('=' * 80)
    print(f'Output directory: {output_dir}')
    print(f'Temperature grid: [{Tmin:.2e}, {Tmax:.2e}] with nT={nT}')
    print(f'Gamma range: [{gamma_min:.2e}, {gamma_max:.2e}] with n_gamma={n_gamma}')
    print(f'Radiation model: {radiation_model}')
    print(f'Mode: {mode} with fixed_value={fixed_value}')
    print('=' * 80)

    _setup_plotting()

    created_files = []
    results_summary = []

    shared_executor = None
    if n_workers != 1:
        max_workers = None if n_workers is None or n_workers <= 0 else int(n_workers)
        shared_executor = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)

    try:
        for bin_info in bins:
            bin_id = bin_info['id']
            comp = bin_info['composition']
            rank = int(bin_info['bin_rank'])
            params = get_lognormal_parameters(bin_id, model_name='basic')
            grain_size_micron = float(params['a0'])
            grain_size_cm = grain_size_micron * 1e-4

            print(
                f"\n[bin={bin_id}] composition={comp}, rank={rank}, "
                f"grain_size={grain_size_micron:.4e} micron"
            )

            # Map composition to grain_type (graphite or silicate)
            if comp.lower() == 'silicate':
                grain_type = 'silicate'
            else:
                grain_type = 'graphite'

            try:
                # Compute heating/cooling rate tables
                result = make_rate_gamma_T_tables(
                    grain_type=grain_type,
                    a_cm=grain_size_cm,
                    radiation_model=radiation_model,
                    mode=mode,
                    fixed_value=fixed_value,
                    Tmin=Tmin,
                    Tmax=Tmax,
                    nT=nT,
                    gamma_min=gamma_min,
                    gamma_max=gamma_max,
                    n_gamma=n_gamma,
                    num_workers=n_workers,
                    out_dir=str(output_dir),
                    debug=False,
                    grain_label=bin_id,
                    executor=shared_executor,
                )

                # Extract gamma and T grids and rates
                T = result['T_vals']
                gamma = result['gamma_vals']
                G0_grid = result.get('G0_vals')
                ne_grid = result.get('ne_vals')
                zmean_grid = result.get('Zmean')
                zsigma_grid = result.get('Zsigma')

                log_peh = result.get('log_peh', np.zeros((len(gamma), len(T))))
                log_rec = result.get('log_rec', np.zeros((len(gamma), len(T))))

                # Convert from log space if needed
                peh_rate = np.power(10.0, log_peh) if np.any(np.isfinite(log_peh)) else np.zeros_like(log_peh)
                rec_rate = np.power(10.0, log_rec) if np.any(np.isfinite(log_rec)) else np.zeros_like(log_rec)

                # Save rates as binary tables with name_binname convention
                file_stem = f'heating_{bin_id}'
                npz_path = output_dir / f'{file_stem}.npz'
                np.savez(npz_path,
                         T=T, gamma=gamma, peh_rate=peh_rate, rec_rate=rec_rate,
                         G0_grid=G0_grid, ne_grid=ne_grid,
                         Zmean_grid=zmean_grid, Zsigma_grid=zsigma_grid,
                         mode=mode, fixed_value=fixed_value,
                         grain_type=grain_type, composition=comp, rank=rank,
                         bin_id=bin_id, grain_size_micron=grain_size_micron)
                created_files.append(str(npz_path))
                print(f"  ✓ Data saved (NPZ): {npz_path.name}")

                # Save figure with quick-look plots
                fig_path = output_dir / f'{file_stem}.png'
                _save_rate_plot(fig_path, T, gamma,
                                {'peh_rate': peh_rate, 'rec_rate': rec_rate},
                                radiation_model=radiation_model,
                                grain_type=grain_type)
                created_files.append(str(fig_path))
                print(f"  ✓ Figure saved: {fig_path.name}")

                # Save metadata as JSON
                json_data = {
                    'bin_id': bin_id,
                    'composition': comp,
                    'grain_type': grain_type,
                    'bin_rank': rank,
                    'grain_size_micron': grain_size_micron,
                    'radiation_model': radiation_model,
                    'temperature_grid': {
                        'Tmin': float(Tmin),
                        'Tmax': float(Tmax),
                        'nT': int(nT),
                    },
                    'gamma_grid': {
                        'gamma_min': float(gamma_min),
                        'gamma_max': float(gamma_max),
                        'n_gamma': int(n_gamma),
                    },
                    'mode': mode,
                    'fixed_value': float(fixed_value),
                }

                json_path = output_dir / f'{file_stem}.json'
                with open(json_path, 'w') as f:
                    json.dump(json_data, f, indent=2)
                created_files.append(str(json_path))
                print(f"  ✓ Metadata saved: {json_path.name}")

                results_summary.append({
                    'bin_id': bin_id,
                    'composition': comp,
                    'rank': rank,
                    'status': 'Success',
                })

            except Exception as e:
                print(f"  ✗ Error: {e}")
                import traceback
                traceback.print_exc()
                results_summary.append({
                    'bin_id': bin_id,
                    'composition': comp,
                    'rank': rank,
                    'status': f'Error: {str(e)}',
                })
    finally:
        if shared_executor is not None:
            shared_executor.shutdown(wait=True)

    # Print summary
    print("\n" + "=" * 80)
    print("PHOTOELECTRIC HEATING EXPORT SUMMARY")
    print("=" * 80)
    successful = sum(1 for r in results_summary if r['status'] == 'Success')
    failed = len(results_summary) - successful
    print(f"  Bins processed: {len(results_summary)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Files created: {len(created_files)}")
    print("=" * 80)

    return {
        'output_dir': str(output_dir),
        'file_count': len(created_files),
        'bins_processed': successful,
        'successful': successful,
        'failed': failed,
        'Tmin': float(Tmin),
        'Tmax': float(Tmax),
        'nT': int(nT),
        'gamma_min': float(gamma_min),
        'gamma_max': float(gamma_max),
        'n_gamma': int(n_gamma),
        'radiation_model': radiation_model,
        'mode': mode,
        'fixed_value': float(fixed_value),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Export dust photoelectric heating rates for all dust bins.'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to JSON grain size configuration file. If not provided, uses default.'
    )
    args = parser.parse_args()
    result = main(config_path=args.config)
    print("\nExport completed successfully!")
