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


def _load_results_from_heating_npz(npz_path, bin_id):
    """Load per-point charging results and metadata from a heating export NPZ file."""
    with np.load(npz_path, allow_pickle=False) as data:
        gamma = np.asarray(data['gamma'], dtype=float)
        T = np.asarray(data['T'], dtype=float)
        zmean_grid = np.asarray(data['Zmean_grid'], dtype=float)
        zsigma_grid = np.asarray(data['Zsigma_grid'], dtype=float)
        G0_grid = np.asarray(data['G0_grid'], dtype=float)
        ne_grid = np.asarray(data['ne_grid'], dtype=float)
        mode = str(data['mode']) if 'mode' in data else None
        fixed_value = float(data['fixed_value']) if 'fixed_value' in data else None

    if zmean_grid.shape != (len(T), len(gamma)):
        raise ValueError(
            f'Unexpected Zmean_grid shape for {bin_id}: {zmean_grid.shape}, expected {(len(T), len(gamma))}'
        )
    if zsigma_grid.shape != (len(T), len(gamma)):
        raise ValueError(
            f'Unexpected Zsigma_grid shape for {bin_id}: {zsigma_grid.shape}, expected {(len(T), len(gamma))}'
        )

    results = []
    for iT, T_val in enumerate(T):
        for ig, gamma_val in enumerate(gamma):
            zmean = zmean_grid[iT, ig]
            zsigma = zsigma_grid[iT, ig]
            if not np.isfinite(zmean) or not np.isfinite(zsigma):
                continue
            results.append({
                'gamma': float(gamma_val),
                'G0': float(G0_grid[iT, ig]),
                'T': float(T_val),
                'ne': float(ne_grid[iT, ig]),
                'Zmean': float(zmean),
                'Zsigma': float(zsigma),
            })
    metadata = {
        'gamma_min': float(np.nanmin(gamma)),
        'gamma_max': float(np.nanmax(gamma)),
        'n_gamma': int(len(gamma)),
        'temperature_min': float(np.nanmin(T)),
        'temperature_max': float(np.nanmax(T)),
        'n_temperature': int(len(T)),
        'mode': mode,
        'fixed_value': fixed_value,
    }
    return results, metadata


def _summarize_result_points(json_results):
    """Build grid-style metadata from point-wise charge results."""
    if not json_results:
        return {
            'gamma_min': None,
            'gamma_max': None,
            'n_gamma': 0,
            'n_samples_per_gamma': None,
            'n_samples_per_gamma_min': 0,
            'n_samples_per_gamma_max': 0,
            'temperature_min': None,
            'temperature_max': None,
            'n_temperature': 0,
        }

    gamma_arr = np.asarray([r['gamma'] for r in json_results], dtype=float)
    temp_arr = np.asarray([r['T'] for r in json_results], dtype=float)
    unique_gamma, gamma_counts = np.unique(gamma_arr, return_counts=True)
    unique_temp = np.unique(temp_arr)

    n_samples_per_gamma = None
    if len(gamma_counts) > 0 and np.all(gamma_counts == gamma_counts[0]):
        n_samples_per_gamma = int(gamma_counts[0])

    return {
        'gamma_min': float(np.nanmin(unique_gamma)),
        'gamma_max': float(np.nanmax(unique_gamma)),
        'n_gamma': int(len(unique_gamma)),
        'n_samples_per_gamma': n_samples_per_gamma,
        'n_samples_per_gamma_min': int(np.min(gamma_counts)),
        'n_samples_per_gamma_max': int(np.max(gamma_counts)),
        'temperature_min': float(np.nanmin(unique_temp)),
        'temperature_max': float(np.nanmax(unique_temp)),
        'n_temperature': int(len(unique_temp)),
    }


def _plot_results(results, grain_size_cm, out_png):
    """Plot Zmean and Zsigma vs gamma with T color coding and ne marker size."""
    gam_arr = np.array([r['gamma'] for r in results], dtype=float)
    zmean_arr = np.array([r['Zmean'] for r in results], dtype=float)
    zsig_arr = np.array([r['Zsigma'] for r in results], dtype=float)
    T_arr = np.array([r['T'] for r in results], dtype=float)
    ne_arr = np.array([r['ne'] for r in results], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=200)
    cmap = plt.get_cmap('viridis')

    tmin = np.nanmin(T_arr)
    tmax = np.nanmax(T_arr)
    tmin = max(tmin, 1e-30)
    tmax = max(tmax, tmin * 1.0001)
    norm = plt.Normalize(vmin=np.log10(tmin), vmax=np.log10(tmax))

    nemin = np.nanmin(ne_arr)
    nemax = np.nanmax(ne_arr)
    nemin = max(nemin, 1e-30)
    nemax = max(nemax, nemin * 1.0001)
    size_scale = (np.log10(np.clip(ne_arr, nemin, nemax)) - np.log10(nemin)) / (np.log10(nemax) - np.log10(nemin) + 1e-30)
    sizes = 10.0 + 40.0 * size_scale

    sc1 = axes[0].scatter(gam_arr, zmean_arr, c=np.log10(T_arr), cmap=cmap, norm=norm, s=sizes, alpha=0.9)
    axes[0].set_xscale('log')
    axes[0].set_xlabel(r'$\gamma = G_0\sqrt{T}/n_e$')
    axes[0].set_ylabel(r'$\langle Z \rangle$')
    axes[0].set_title('Mean Charge')

    axes[1].scatter(gam_arr, zsig_arr, c=np.log10(T_arr), cmap=cmap, norm=norm, s=sizes, alpha=0.9)
    axes[1].set_xscale('log')
    axes[1].set_xlabel(r'$\gamma = G_0\sqrt{T}/n_e$')
    axes[1].set_ylabel(r'$\sigma_Z$')
    axes[1].set_title('Charge Width')

    cbar = fig.colorbar(sc1, ax=axes.ravel().tolist(), pad=0.02)
    cbar.set_label(r'$\log_{10}(T/\mathrm{K})$')
    fig.suptitle(f'Grain size a={grain_size_cm:.3e} cm', fontsize=11)
    fig.subplots_adjust(top=0.86, bottom=0.14, left=0.09, right=0.95, wspace=0.28)
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main(config_path=None, reuse_heating_data=False):
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
    heating_dir = repo_root / 'model_data' / 'dust_photoelectric_heating_data'

    print('=' * 80)
    print('Exporting dust grain charge (Zmean) vs gamma for all dust bins')
    print('=' * 80)
    print(f'Output directory: {output_dir}')
    if reuse_heating_data:
        print('Gamma range: derived per bin from heating-table gamma grid')
        print('Combinations per gamma: derived per bin from heating-table temperature grid')
        print(f'Reusing precomputed charge data from: {heating_dir}')
    else:
        print(f'Gamma range: [{gamma_min:.2e}, {gamma_max:.2e}] ({n_gamma} points)')
        print(f'Combinations per gamma: {combos_per_gamma}')
    print('=' * 80)

    _setup_plotting()

    created_files = []
    results_summary = []

    first_grid_meta = None

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
            file_stem = f'charge_{bin_id}'
            fig_path = output_dir / f'{file_stem}.png'
            grid_meta = None
            if reuse_heating_data:
                print('  Building charge export from heating tables...')
                npz_path = heating_dir / f'heating_{bin_id}.npz'
                if not npz_path.exists():
                    raise FileNotFoundError(
                        f'Expected heating table file not found for bin {bin_id}: {npz_path}'
                    )
                results, grid_meta = _load_results_from_heating_npz(npz_path, bin_id)
                if first_grid_meta is None:
                    first_grid_meta = dict(grid_meta)
                if len(results) == 0:
                    raise ValueError(f'No reusable charging rows available in {npz_path.name}')
                _plot_results(results, grain_size_cm, fig_path)
            else:
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

            inferred_grid = _summarize_result_points(json_results)
            if reuse_heating_data and grid_meta is not None:
                inferred_grid['mode'] = grid_meta.get('mode')
                inferred_grid['fixed_value'] = grid_meta.get('fixed_value')
            
            json_data = {
                'bin_id': bin_id,
                'composition': comp,
                'grain_type': grain_type,
                'bin_rank': rank,
                'grain_size_micron': grain_size_micron,
                'source': 'heating_tables' if reuse_heating_data else 'direct_charge_solver',
                'grid': inferred_grid,
                'gamma_min': inferred_grid['gamma_min'],
                'gamma_max': inferred_grid['gamma_max'],
                'n_gamma': inferred_grid['n_gamma'],
                'n_samples_per_gamma': inferred_grid['n_samples_per_gamma'],
                'data_points': len(json_results),
                'results': json_results
            }

            json_path = output_dir / f'{file_stem}.json'
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

    summary_grid = {
        'gamma_min': float(gamma_min),
        'gamma_max': float(gamma_max),
        'n_gamma': int(n_gamma),
        'n_samples_per_gamma': int(combos_per_gamma),
        'n_samples_per_gamma_min': int(combos_per_gamma),
        'n_samples_per_gamma_max': int(combos_per_gamma),
        'temperature_min': None,
        'temperature_max': None,
        'n_temperature': None,
        'mode': None,
        'fixed_value': None,
    }
    if reuse_heating_data and first_grid_meta is not None:
        summary_grid.update({
            'gamma_min': first_grid_meta.get('gamma_min'),
            'gamma_max': first_grid_meta.get('gamma_max'),
            'n_gamma': first_grid_meta.get('n_gamma'),
            'n_samples_per_gamma': first_grid_meta.get('n_temperature'),
            'n_samples_per_gamma_min': first_grid_meta.get('n_temperature'),
            'n_samples_per_gamma_max': first_grid_meta.get('n_temperature'),
            'temperature_min': first_grid_meta.get('temperature_min'),
            'temperature_max': first_grid_meta.get('temperature_max'),
            'n_temperature': first_grid_meta.get('n_temperature'),
            'mode': first_grid_meta.get('mode'),
            'fixed_value': first_grid_meta.get('fixed_value'),
        })

    return {
        'output_dir': str(output_dir),
        'file_count': len(created_files),
        'bins_processed': successful,
        'successful': successful,
        'failed': failed,
        'source': 'heating_tables' if reuse_heating_data else 'direct_charge_solver',
        'gamma_min': summary_grid['gamma_min'],
        'gamma_max': summary_grid['gamma_max'],
        'n_gamma': summary_grid['n_gamma'],
        'combos_per_gamma': summary_grid['n_samples_per_gamma'],
        'combos_per_gamma_min': summary_grid['n_samples_per_gamma_min'],
        'combos_per_gamma_max': summary_grid['n_samples_per_gamma_max'],
        'temperature_min': summary_grid['temperature_min'],
        'temperature_max': summary_grid['temperature_max'],
        'n_temperature': summary_grid['n_temperature'],
        'mode': summary_grid['mode'],
        'fixed_value': summary_grid['fixed_value'],
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
    parser.add_argument(
        '--reuse-heating-data',
        action='store_true',
        help='Reuse precomputed charge moments stored by dust photoelectric heating export.'
    )
    args = parser.parse_args()
    result = main(config_path=args.config, reuse_heating_data=bool(args.reuse_heating_data))
    print("\nExport completed successfully!")
