#!/usr/bin/env python
"""Export dust-assisted ion recombination coefficient tables for all dust bins.

This script reads all dust bins from the grain-size configuration and computes the
ion recombination coefficient alpha (using case A from Weingartner & Draine 2001)
as a function of temperature and gamma = G0*sqrt(T)/ne.

Outputs are 2D ASCII tables (T × gamma grid) suitable for interpolation,
plus quick-look plots showing how coefficients vary with ISM conditions.
"""

import argparse
import json
import concurrent.futures
from pathlib import Path
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from models.grain_size_config import set_config_path, get_bins, get_lognormal_parameters, get_export_parameters, get_model_data_dir
from models.dust_charge.dust_ion_recombination import compute_ion_recombination_coefficients

DEFAULT_EXPORT_PARAMS = {
    'Tmin': 10.0,
    'Tmax': 1e5,
    'nT': 100,
    'gamma_min': 1e-6,
    'gamma_max': 1e6,
    'n_gamma': 100,
    'radiation_model': 'Mathis',
    'mode': 'fix_G0',
    'fixed_value': 1.0,
    'n_workers': None,
}

# The 11 ions ordered by atomic number
ION_ELEMENTS = ['H', 'He', 'C', 'Na', 'Mg', 'Si', 'S', 'K', 'Ca', 'Mn', 'Fe']
ATOMIC_WEIGHTS = {
    'H': 1.008, 'He': 4.003, 'C': 12.011, 'Na': 22.990, 'Mg': 24.305,
    'Si': 28.085, 'S': 32.06, 'K': 39.098, 'Ca': 40.078, 'Mn': 54.938, 'Fe': 55.845
}

# Context cache for workers
_DIR_WORKER_PREPARED_CONTEXTS = {}


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


def _compute_recomb_point(task):
    """
    Compute recombination coefficients for a single grid point (G0, ne, T).
    No background ions are passed to the charging solver (same as PE heating grid).
    Post-hoc Case A thresholding is applied to calculate the coefficients.
    """
    G0_used, ne_used, T_used, grain_type, a_cm, radiation_model = task[:6]
    
    from models.dust_charge import dust_charging as _dc
    
    global _DIR_WORKER_PREPARED_CONTEXTS
    ctx_key = (str(grain_type), float(a_cm), str(radiation_model))
    ctx = _DIR_WORKER_PREPARED_CONTEXTS.get(ctx_key)
    if ctx is None:
        scan_ctx = _dc._prepare_gamma_scan_context(
            grain_type, a_cm, radiation_model=radiation_model, yield_params=None
        )
        ctx = {
            'nu': np.asarray(scan_ctx['nu'], dtype=float),
            'J_nu_base': np.asarray(scan_ctx['J_nu'], dtype=float),
            'C_abs_nu': np.asarray(scan_ctx['C_abs_nu'], dtype=float),
            'yield_func': scan_ctx['yield_func'],
            'yield_params': dict(scan_ctx['yield_params']),
        }
        _DIR_WORKER_PREPARED_CONTEXTS[ctx_key] = ctx

    J_nu_scaled = ctx['J_nu_base'] * float(G0_used)

    # Solve charging distribution without background ions
    Zs, P, rates, Zmean, Zsigma = _dc.compute_equilibrium_charge_distribution_vectorized(
        float(a_cm), float(ne_used), float(T_used), [],
        ctx['nu'], J_nu_scaled, ctx['C_abs_nu'],
        yield_func=ctx['yield_func'],
        yield_params=ctx['yield_params'],
        Z_start=0,
        debug=False,
    )

    # Define the 11 ions dynamically at T_used
    ion_species = []
    for el in ION_ELEMENTS:
        ion_species.append({
            'name': f'{el}+',
            'z': 1.0,
            'T': T_used,
            'm': ATOMIC_WEIGHTS[el] * 1.66053906660e-24,
            's_i': 1.0
        })

    # Compute ion recombination coefficients using Case A
    _, recomb_coeffs = compute_ion_recombination_coefficients(
        Zs, P, a_cm, ion_species, grain_type=grain_type, recomb_mode='case_a'
    )

    return Zmean, Zsigma, np.array(recomb_coeffs, dtype=float)


def _compute_recomb_batch(batch_tasks):
    """Worker helper to compute a batch of grid points."""
    out = []
    for task in batch_tasks:
        pos, iT, ig, G0, ne, T_task, grain_type, a_cm, radiation_model = task
        try:
            Zm, Zs, recomb_coeffs = _compute_recomb_point((G0, ne, T_task, grain_type, a_cm, radiation_model))
        except Exception as exc:
            msg = (
                f'[__compute_recomb_batch] Worker task failed: '
                f'pos={pos}, iT={iT}, ig={ig}, T={T_task:.6e} K, '
                f'G0={G0:.6e}, ne={ne:.6e} cm^-3, '
                f'grain={grain_type}, a_cm={a_cm:.3e}, radiation_model={radiation_model}'
            )
            raise RuntimeError(msg) from exc
        out.append((pos, Zm, Zs, recomb_coeffs))
    return out


def _write_ion_recomb_tables(out_dir, size_tag, T_vals, gamma_vals, recomb_coeffs_grid, mode):
    """Write the ion recombination coefficient tables with metadata headers."""
    out_path = Path(out_dir) / f'dust_rates_ion_recomb_{size_tag}.dat'
    
    log_T = np.log10(T_vals)
    log_gamma = np.log10(gamma_vals)
    columns_str = ' '.join(ION_ELEMENTS)
    
    from models.grain_size_config import get_header_lines
    header_lines = get_header_lines(
        title=f"Ion recombination rate coefficient (alpha, Case A) table metadata (mode={mode})",
        script_name="models/dust_charge/export_dust_ion_recombination.py",
        bin_info=f"Dust Bin: {size_tag}",
        val_desc=f"log10(alpha [cm^3 s^-1 per grain]) for columns: {columns_str}",
        num_lines=6
    )
    
    nT = len(log_T)
    n_gamma = len(log_gamma)
    n_ions = len(ION_ELEMENTS)
    
    # Convert to log10 space, set non-positive or invalid values to -1e30
    fill_bad = -1e30
    with np.errstate(divide='ignore', invalid='ignore'):
        log_alpha = np.log10(np.where(recomb_coeffs_grid > 0.0, recomb_coeffs_grid, np.nan))
    log_alpha[~np.isfinite(log_alpha)] = fill_bad
    
    with open(out_path, 'w') as fh:
        fh.write('\n'.join(header_lines) + '\n')
        fh.write(f'{log_T.size} {log_gamma.size}\n')
        fh.write(' '.join(f'{value:.12e}' for value in log_T) + '\n')
        fh.write(' '.join(f'{value:.12e}' for value in log_gamma) + '\n')
        
        # Grid flattening matching the order temperature (outer) x gamma (inner)
        for iT in range(nT):
            for ig in range(n_gamma):
                row_vals = log_alpha[iT, ig, :]
                fh.write(' '.join(f'{value:.12e}' for value in row_vals) + '\n')
                
    return out_path


def _save_rate_plot(output_path, T, gamma, recomb_coeffs, radiation_model, grain_type, bin_id):
    """Save quick-look plot showing ion recombination coefficients vs temperature and gamma."""
    _setup_plotting()
    
    h_idx = ION_ELEMENTS.index('H')
    c_idx = ION_ELEMENTS.index('C')
    fe_idx = ION_ELEMENTS.index('Fe')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=180)
    
    # Select representative gamma values for line plots
    if len(gamma) > 1:
        gamma_indices = [0, len(gamma)//4, len(gamma)//2, 3*len(gamma)//4, -1]
        gamma_vals_plot = [gamma[i] for i in gamma_indices if i < len(gamma)]
    else:
        gamma_vals_plot = [gamma[0]]
        
    # Plot 1: H+ recombination coefficient vs T for different gamma
    ax = axes[0, 0]
    for gi, gval in enumerate(gamma_vals_plot):
        if gi == len(gamma_vals_plot) - 1:
            idx = min(-1, len(gamma)-1)
        else:
            idx = np.searchsorted(gamma, gval)
            idx = min(idx, len(gamma)-1)
        ax.loglog(T, np.abs(recomb_coeffs[:, idx, h_idx]), marker='o', markersize=3,
                  label=rf'$\gamma$ = {gval:.2e}', linewidth=1.5, alpha=0.8)
    ax.set_xlabel(r'$T$ [K]', fontsize=12)
    ax.set_ylabel(r'$\alpha(H^+)$ [cm$^3$ s$^{-1}$ per grain]', fontsize=12)
    ax.set_title(r'H$^+$ Recombination Coefficient (Case A)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, frameon=False, loc='best')
    ax.grid(True, which='both', alpha=0.3)
    
    # Plot 2: C+ recombination coefficient vs T for different gamma
    ax = axes[0, 1]
    for gi, gval in enumerate(gamma_vals_plot):
        if gi == len(gamma_vals_plot) - 1:
            idx = min(-1, len(gamma)-1)
        else:
            idx = np.searchsorted(gamma, gval)
            idx = min(idx, len(gamma)-1)
        ax.loglog(T, np.abs(recomb_coeffs[:, idx, c_idx]), marker='s', markersize=3,
                  label=rf'$\gamma$ = {gval:.2e}', linewidth=1.5, alpha=0.8)
    ax.set_xlabel(r'$T$ [K]', fontsize=12)
    ax.set_ylabel(r'$\alpha(C^+)$ [cm$^3$ s$^{-1}$ per grain]', fontsize=12)
    ax.set_title(r'C$^+$ Recombination Coefficient (Case A)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, frameon=False, loc='best')
    ax.grid(True, which='both', alpha=0.3)
    
    # Plot 3: Fe+ recombination coefficient vs T for different gamma
    ax = axes[1, 0]
    for gi, gval in enumerate(gamma_vals_plot):
        if gi == len(gamma_vals_plot) - 1:
            idx = min(-1, len(gamma)-1)
        else:
            idx = np.searchsorted(gamma, gval)
            idx = min(idx, len(gamma)-1)
        ax.loglog(T, np.abs(recomb_coeffs[:, idx, fe_idx]), marker='^', markersize=3,
                  label=rf'$\gamma$ = {gval:.2e}', linewidth=1.5, alpha=0.8)
    ax.set_xlabel(r'$T$ [K]', fontsize=12)
    ax.set_ylabel(r'$\alpha(Fe^+)$ [cm$^3$ s$^{-1}$ per grain]', fontsize=12)
    ax.set_title(r'Fe$^+$ Recombination Coefficient (Case A)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, frameon=False, loc='best')
    ax.grid(True, which='both', alpha=0.3)
    
    # Plot 4: H+ 2D heatmap (gamma vs T)
    ax = axes[1, 1]
    h_val = recomb_coeffs[:, :, h_idx].T
    h_valid = np.maximum(np.abs(h_val), 1e-30)
    im = ax.pcolormesh(T, gamma, np.log10(h_valid), shading='auto', cmap='viridis')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$T$ [K]', fontsize=12)
    ax.set_ylabel(r'$\gamma$ [G$_0$ $\sqrt{T}$ / $n_e$]', fontsize=12)
    ax.set_title(r'log10($\alpha(H^+)$)', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label=r'log10($\alpha(H^+)$)')
    
    fig.suptitle(f'{grain_type.capitalize()} dust grain ({bin_id}) - {radiation_model} radiation field',
                 fontsize=14, fontweight='bold', y=0.995)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def make_ion_recomb_gamma_T_tables(grain_type, a_cm, radiation_model='Mathis',
                                  mode='fix_G0', fixed_value=1.0,
                                  Tmin=10.0, Tmax=1e5, nT=100,
                                  gamma_min=1e-6, gamma_max=1e6, n_gamma=100,
                                  num_workers=None, out_dir='tables',
                                  grain_label=None, executor=None):
    """
    Compute grids of ion recombination coefficients on a log(T) x log(gamma) grid
    and write ASCII tables suitable for linear interpolation.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Build grids in log space
    T_vals = np.logspace(np.log10(Tmin), np.log10(Tmax), nT)
    gamma_vals = np.logspace(np.log10(gamma_min), np.log10(gamma_max), n_gamma)
    
    # Prepare tasks: rows over T, columns over gamma
    tasks = []
    for iT, T in enumerate(T_vals):
        sqrtT = np.sqrt(T)
        for ig, gamma in enumerate(gamma_vals):
            if mode == 'fix_G0':
                G0 = float(fixed_value)
                ne = max(1e-20, (G0 * sqrtT) / float(gamma))
            elif mode == 'fix_ne':
                ne = float(fixed_value)
                G0 = max(1e-20, (float(gamma) * ne) / sqrtT)
            else:
                raise ValueError('mode must be "fix_G0" or "fix_ne"')
            tasks.append((iT, ig, G0, ne, float(T)))
            
    N = len(tasks)
    G0_vals = np.full(N, np.nan)
    ne_vals = np.full(N, np.nan)
    Zmean_vals = np.full(N, np.nan)
    Zsigma_vals = np.full(N, np.nan)
    for pos, t in enumerate(tasks):
        G0_vals[pos] = t[2]
        ne_vals[pos] = t[3]
        
    n_ions = len(ION_ELEMENTS)
    recomb_coeffs_vals = [None] * N
    
    # Run in parallel/sequential
    if num_workers == 1:
        count = 0
        for pos, t in enumerate(tasks):
            iT, ig, G0, ne, T_task = t
            Zm, Zs, recomb_coeffs = _compute_recomb_point((G0, ne, T_task, grain_type, a_cm, radiation_model))
            Zmean_vals[pos] = Zm
            Zsigma_vals[pos] = Zs
            recomb_coeffs_vals[pos] = recomb_coeffs
            count += 1
            if count % 100 == 0 or count == len(tasks):
                print(f'[make_ion_recomb_gamma_T_tables] Processed {count}/{len(tasks)} tasks (in-process)')
    else:
        if num_workers is None or int(num_workers) <= 0:
            num_workers = os.cpu_count() or 1
            
        worker_inputs = []
        for pos, t in enumerate(tasks):
            iT, ig, G0, ne, T_task = t
            worker_inputs.append((pos, iT, ig, G0, ne, T_task, grain_type, a_cm, radiation_model))
            
        target_batches = max(4 * int(num_workers), 1)
        batch_size = max(16, int(np.ceil(len(worker_inputs) / float(target_batches))))
        batches = [worker_inputs[i:i + batch_size] for i in range(0, len(worker_inputs), batch_size)]
        
        exe = executor
        owns_executor = exe is None
        if exe is None:
            exe = concurrent.futures.ProcessPoolExecutor(max_workers=num_workers)
            
        try:
            try:
                from tqdm import tqdm
                iterator = tqdm(exe.map(_compute_recomb_batch, batches, chunksize=1),
                                total=len(batches), desc='Computing recombination coefficients')
                use_tqdm = True
            except Exception:
                iterator = exe.map(_compute_recomb_batch, batches, chunksize=1)
                use_tqdm = False
                
            count = 0
            for batch_out in iterator:
                for pos, Zm, Zs, recomb_coeffs in batch_out:
                    Zmean_vals[pos] = Zm
                    Zsigma_vals[pos] = Zs
                    recomb_coeffs_vals[pos] = recomb_coeffs
                count += len(batch_out)
                if not use_tqdm and (count % 100 == 0 or count == len(worker_inputs)):
                    print(f'[make_ion_recomb_gamma_T_tables] Processed {count}/{len(worker_inputs)} tasks')
        finally:
            if owns_executor:
                exe.shutdown(wait=True)
                
    # Reshape results back into 2D grids (T, gamma) or 3D grids (T, gamma, ions)
    G0_mat = G0_vals.reshape((nT, n_gamma))
    ne_mat = ne_vals.reshape((nT, n_gamma))
    Zmean_mat = Zmean_vals.reshape((nT, n_gamma))
    Zsigma_mat = Zsigma_vals.reshape((nT, n_gamma))
    recomb_coeffs_mat = np.array([c if c is not None else np.zeros(n_ions) for c in recomb_coeffs_vals]).reshape((nT, n_gamma, n_ions))
    
    # Save the ASCII table
    size_tag = grain_label if grain_label is not None else f'a_{a_cm:.3e}_cm'
    out_path = _write_ion_recomb_tables(
        out_dir=out_dir,
        size_tag=size_tag,
        T_vals=T_vals,
        gamma_vals=gamma_vals,
        recomb_coeffs_grid=recomb_coeffs_mat,
        mode=mode
    )
    
    return {
        'T_vals': T_vals,
        'gamma_vals': gamma_vals,
        'G0_vals': G0_mat,
        'ne_vals': ne_mat,
        'Zmean': Zmean_mat,
        'Zsigma': Zsigma_mat,
        'recomb_coeffs_grid': recomb_coeffs_mat,
        'out_path': str(out_path)
    }


def main(config_path=None):
    """
    Export dust-assisted ion recombination coefficient tables for all dust bins.
    """
    if config_path:
        set_config_path(config_path)

    # Use parameters under 'dust_ion_recombination' if they exist, fallback to DEFAULT_EXPORT_PARAMS
    params_cfg = get_export_parameters('dust_ion_recombination', defaults=DEFAULT_EXPORT_PARAMS)
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

    output_dir = get_model_data_dir() / 'dust_ion_recombination_data'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove pre-existing files to keep folder clean
    for existing in output_dir.iterdir():
        if existing.is_file() and (existing.name.startswith('dust_rates_ion_recomb_') or
                                   existing.name.startswith('ion_recomb_')):
            existing.unlink()

    bins = sorted(
        get_bins(is_pah=False),
        key=lambda b: (b['composition'], b['bin_rank'], b['index']),
    )
    if not bins:
        raise RuntimeError('No dust bins found in grain-size configuration.')

    print('=' * 80)
    print('Exporting dust-assisted ion recombination coefficients for all dust bins')
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

            if comp.lower() == 'silicate':
                grain_type = 'silicate'
            else:
                grain_type = 'graphite'

            try:
                # Compute recombination coefficient tables
                result = make_ion_recomb_gamma_T_tables(
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
                    grain_label=bin_id,
                    executor=shared_executor,
                )

                T = result['T_vals']
                gamma = result['gamma_vals']
                recomb_coeffs = result['recomb_coeffs_grid']
                Zmean = result['Zmean']
                Zsigma = result['Zsigma']
                G0_grid = result['G0_vals']
                ne_grid = result['ne_vals']

                # Save raw grids as numpy NPZ file
                file_stem = f'ion_recomb_{bin_id}'
                npz_path = output_dir / f'{file_stem}.npz'
                np.savez(
                    npz_path,
                    T=T, gamma=gamma, recomb_coeffs=recomb_coeffs,
                    Zmean_grid=Zmean, Zsigma_grid=Zsigma,
                    G0_grid=G0_grid, ne_grid=ne_grid,
                    mode=mode, fixed_value=fixed_value,
                    grain_type=grain_type, composition=comp, rank=rank,
                    bin_id=bin_id, grain_size_micron=grain_size_micron
                )
                created_files.append(str(npz_path))
                print(f"  ✓ Data saved (NPZ): {npz_path.name}")

                # Save quick-look plots
                fig_path = output_dir / f'{file_stem}.png'
                _save_rate_plot(
                    fig_path, T, gamma, recomb_coeffs,
                    radiation_model=radiation_model,
                    grain_type=grain_type, bin_id=bin_id
                )
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
                    'ion_species_ordered_by_atomic_number': ION_ELEMENTS
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
    print("ION RECOMBINATION COEFFICIENT EXPORT SUMMARY")
    print("=" * 80)
    successful = sum(1 for r in results_summary if r['status'] == 'Success')
    failed = len(results_summary) - successful
    print(f"  Bins processed: {len(results_summary)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Files created: {len(created_files)}")
    print("=" * 80)

    # Save a general README
    readme_path = output_dir / 'README.md'
    with open(readme_path, 'w') as fh:
        fh.write('# Dust-Assisted Ion Recombination Coefficient Tables\n\n')
        fh.write('This directory contains grid files of the ion recombination coefficient alpha (using Case A threshold)\n')
        fh.write('for each dust bin, as a function of temperature and gamma = G0*sqrt(T)/ne.\n\n')
        fh.write('## Table Formatting\n')
        fh.write('- Comments: First 6 lines start with `#`.\n')
        fh.write('- Size Line: Line 7 specifies `nT n_gamma` (grid dimensions).\n')
        fh.write('- Temp Grid: Line 8 contains the space-separated log10(T) values (increasing).\n')
        fh.write('- Gamma Grid: Line 9 contains the space-separated log10(gamma) values (increasing).\n')
        fh.write('- Data Rows: `nT * n_gamma` rows (outer loop log10(T), inner loop log10(gamma)).\n')
        fh.write('  Each row contains 11 space-separated values corresponding to the log10 of the recombination coefficient alpha\n')
        fh.write('  (in units of cm^3 s^-1 per grain) for the following elements ordered by atomic number:\n')
        fh.write('  `H He C Na Mg Si S K Ca Mn Fe`\n\n')
        fh.write('Missing/invalid values are encoded as `-1e30`.\n')
    
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
        description='Export dust-assisted ion recombination coefficients for all dust bins.'
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
