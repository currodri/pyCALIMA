#!/usr/bin/env python
"""Export dust grain charge (Zmean) vs gamma for all dust bins.

This script reads all dust bins from the grain-size configuration, scans charge
values over the gamma parameter (G0 * sqrt(T) / ne), and saves scatter plots
showing how grain equilibrium charge varies with gamma.

Gamma characterizes different ISM conditions:
- Low gamma: cool, dense, low radiation
- High gamma: hot, diffuse, high radiation
"""

import argparse
from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from models.grain_size_config import set_config_path, get_bins, get_lognormal_parameters, get_export_parameters
from models.dust_charge.dust_charging import compute_charge_vs_gamma


DEFAULT_EXPORT_PARAMS = {
    'gamma_min': 1e-4,
    'gamma_max': 1e6,
    'n_gamma': 50,
    'combos_per_gamma': 20,
    'seed': 42,
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


def main(config_path=None):
    """
    Export dust charging vs gamma for all dust bins.
    
    Parameters
    ----------
    config_path : str, optional
        Path to grain size configuration JSON file
    """
    if config_path:
        set_config_path(config_path)

    params_cfg = get_export_parameters('dust_charging', defaults=DEFAULT_EXPORT_PARAMS)
    gamma_min = float(params_cfg['gamma_min'])
    gamma_max = float(params_cfg['gamma_max'])
    n_gamma = int(params_cfg['n_gamma'])
    combos_per_gamma = int(params_cfg['combos_per_gamma'])
    seed = int(params_cfg['seed'])

    repo_root = _repo_root()
    output_dir = repo_root / 'model_data' / 'dust_charging_data'
    output_dir.mkdir(parents=True, exist_ok=True)

    bins = sorted(
        get_bins(is_pah=False),
        key=lambda b: (b['composition'], b['bin_rank'], b['index']),
    )
    if not bins:
        raise RuntimeError('No dust bins found in grain-size configuration.')

    gamma_values = np.logspace(np.log10(gamma_min), np.log10(gamma_max), n_gamma)

    print('=' * 80)
    print('Exporting dust grain charge (Zmean) vs gamma for all dust bins')
    print('=' * 80)
    print(f'Output directory: {output_dir}')
    print(f'Gamma range: [{gamma_min:.2e}, {gamma_max:.2e}] ({n_gamma} points)')
    print(f'Combinations per gamma: {combos_per_gamma}')
    print('=' * 80)

    _setup_plotting()

    created_files = []
    results_summary = []

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
            # Compute charge vs gamma
            # Note: provide temp_bin_edges to ensure proper charge distribution binning
            print(f"  Computing charge distribution for {len(gamma_values)} gamma values...")
            results, fig = compute_charge_vs_gamma(
                grain_type=grain_type,
                a=grain_size_cm,
                gamma_values=gamma_values,
                combos_per_gamma=combos_per_gamma,
                temp_bin_edges=np.logspace(1, 7, 31),  # Temperature bins for distribution
                seed=seed,
                debug=False,
            )
            
            if results is None or fig is None:
                raise ValueError("compute_charge_vs_gamma returned None for results or figure")
            
            if len(results) == 0:
                raise ValueError("No results returned from compute_charge_vs_gamma")

            # Save figure
            fig_path = output_dir / f'charging_vs_gamma_{comp}_{rank:02d}_{grain_size_micron:.2e}_micron.png'
            fig.savefig(fig_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            created_files.append(str(fig_path))
            print(f"  ✓ Figure saved: {fig_path.name}")

            # Save results as JSON
            print(f"  Processing {len(results)} charge data points...")
            json_results = []
            error_summary = {}
            for r in results:
                # Skip entries with None Zmean/Zsigma
                if r.get('Zmean') is None or r.get('Zsigma') is None:
                    error_msg = r.get('error', 'Unknown error')
                    error_summary[error_msg[:100]] = error_summary.get(error_msg[:100], 0) + 1
                    continue
                    
                try:
                    json_results.append({
                        'gamma': float(r['gamma']),
                        'G0': float(r['G0']),
                        'T': float(r['T']),
                        'ne': float(r['ne']),
                        'Zmean': float(r['Zmean']),
                        'Zsigma': float(r['Zsigma']),
                    })
                except (ValueError, KeyError, TypeError) as entry_err:
                    print(f"    Warning: Skipping invalid result entry: {entry_err}")
                    continue
            
            # Report any errors that occurred
            if error_summary:
                print(f"  ⚠ {len([e for e in error_summary.values() if e > 0])} unique error types:")
                for err_msg, count in sorted(error_summary.items(), key=lambda x: -x[1])[:3]:
                    print(f"    - {count}x: {err_msg}")
            
            if len(json_results) == 0:
                print(f"  ✗ No valid results! All {len(results)} combinations failed.")
                raise ValueError(f'No valid charging results for bin {bin_id}')
            
            json_data = {
                'bin_id': bin_id,
                'composition': comp,
                'grain_type': grain_type,
                'bin_rank': rank,
                'grain_size_micron': grain_size_micron,
                'gamma_min': float(gamma_min),
                'gamma_max': float(gamma_max),
                'n_gamma': int(n_gamma),
                'n_samples_per_gamma': int(combos_per_gamma),
                'data_points': len(json_results),
                'results': json_results
            }

            json_path = output_dir / f'charging_vs_gamma_{comp}_{rank:02d}_{grain_size_micron:.2e}_micron.json'
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            created_files.append(str(json_path))
            print(f"  ✓ Data saved: {json_path.name}")

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

    # Print summary
    print("\n" + "=" * 80)
    print("DUST CHARGING EXPORT SUMMARY")
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
        'gamma_min': gamma_min,
        'gamma_max': gamma_max,
        'n_gamma': n_gamma,
        'combos_per_gamma': combos_per_gamma,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Export dust grain charge vs gamma for all dust bins.'
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
