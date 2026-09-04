"""
MASTER EXPORT SCRIPT

This script orchestrates the export of all grain and PAH optical properties and
collision data. It calls the individual export functions in sequence, tracks
outputs, and generates a comprehensive README with metadata including:
  - Git repository information (branch, head commit)
  - Grain and PAH bin properties from the JSON configuration
  - Timestamps and locations of all exported data
  - Functions and methodologies used

By: Curro Rodriguez (currodri@gmail.com)
"""

import os
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
import json
import time

from pycalima.models.grain_size_config import set_config_path, load_grain_size_config, get_bins, get_bin_by_rank, get_model_data_dir


def get_git_info():
    """
    Retrieve current git repository information.
    
    Returns
    -------
    dict
        Dictionary with keys 'branch', 'commit_hash', 'commit_message'
        Returns empty values if not in a git repository.
    """
    try:
        # Get current branch name
        branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
        
        # Get current commit hash (short)
        commit_short = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
        
        # Get current commit hash (full)
        commit_full = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
        
        # Get commit message
        commit_msg = subprocess.check_output(
            ['git', 'log', '-1', '--pretty=%B'],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
        
        return {
            'branch': branch,
            'commit_short': commit_short,
            'commit_full': commit_full,
            'commit_message': commit_msg,
        }
    except Exception as e:
        print(f"Warning: Could not retrieve git information: {e}")
        return {
            'branch': 'unknown',
            'commit_short': 'unknown',
            'commit_full': 'unknown',
            'commit_message': 'unknown',
        }


def get_output_base():
    """Root of the generated-data tree this run writes into."""
    return get_model_data_dir()


def _run_profiled_stage(stage_name, stage_func, config_path, stage_profile, enable_profile=True, **stage_kwargs):
    """Run one export stage and capture wall-clock timing metadata."""
    t0 = time.perf_counter()
    result = stage_func(config_path, **stage_kwargs)
    t1 = time.perf_counter()

    if enable_profile:
        elapsed = float(t1 - t0)
        status = str(result.get('status', 'Unknown')) if isinstance(result, dict) else 'Unknown'
        stage_profile[stage_name] = {
            'seconds': elapsed,
            'status': status,
            'ok': status.startswith('Success'),
        }

    return result


def _print_profile_summary(stage_profile, total_seconds):
    """Print ranked wall-clock summary for full-export stages."""
    print("\n" + "="*80)
    print("EXPORT PROFILE SUMMARY (WALL-CLOCK)")
    print("="*80)

    if not stage_profile:
        print("No stage profile information collected.")
        print("="*80)
        return

    ranked = sorted(stage_profile.items(), key=lambda kv: kv[1]['seconds'], reverse=True)
    for idx, (name, meta) in enumerate(ranked, start=1):
        sec = float(meta['seconds'])
        pct = (100.0 * sec / total_seconds) if total_seconds > 0.0 else 0.0
        status = meta.get('status', 'Unknown')
        print(f"{idx:>2d}. {name:<36} {sec:>9.2f}s  ({pct:>5.1f}%)  [{status}]")

    print("-"*80)
    print(f"Total full export wall-clock: {total_seconds:.2f}s")
    print("="*80)


def _write_profile_json(stage_profile, total_seconds, output_path):
    """Write profile metrics to a JSON file for later analysis."""
    payload = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'total_seconds': float(total_seconds),
        'stages': stage_profile,
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(payload, f, indent=2)
    return str(out)


def generate_readme(export_results, config_data, git_info, output_base=None):
    """
    Generate comprehensive README documenting all exports.
    
    Parameters
    ----------
    export_results : dict
        Results from export functions with keys like:
        - 'dust_optical_properties': {'status': ..., 'timestamp': ..., 'dir': ...}
        - 'pah_optical_properties': {...}
        - 'collisional_cooling': {...}
        - 'sputtering_rates': {...}
    config_data : dict
        Grain configuration data from JSON
    git_info : dict
        Git repository information
    output_base : str or Path, optional
        Directory the README is written into. Defaults to the generated-data
        directory for the active configuration.
    """
    output_base = Path(output_base) if output_base else get_output_base()
    output_base.mkdir(parents=True, exist_ok=True)
    readme_path = output_base / 'README.md'
    
    timestamp = datetime.now()
    timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
    
    # Extract bin information
    dust_bins = get_bins(is_pah=False)
    pah_bins = get_bins(is_pah=True)
    
    # Start README content
    readme_lines = [
        "# Grain & PAH Optical Properties and Collision Data Repository",
        "",
        f"**Generated**: {timestamp_str}",
        f"**Repository**: {git_info['branch']} ({git_info['commit_short']})",
        "",
        "---",
        "",
        "## Overview",
        "",
        "This directory contains pre-computed optical properties and collision data for dust grains",
        "and PAHs across multiple size bins. Data is generated from the grain size configuration",
        "(`models/grain_size_distribution.json`) and computed using methods specified below.",
        "",
        "---",
        "",
        "## Repository Information",
        "",
        f"- **Branch**: `{git_info['branch']}`",
        f"- **Commit (short)**: `{git_info['commit_short']}`",
        f"- **Commit (full)**: `{git_info['commit_full']}`",
        f"- **Commit Message**: {git_info['commit_message']}",
        f"- **Export Date/Time**: {timestamp_str}",
        "",
        "---",
        "",
        "## Grain Size Configuration",
        "",
        "All grain and PAH bins are defined in `models/grain_size_distribution.json`.",
        "The configuration includes both size distribution parameters and composition metadata.",
        "",
        "### Dust (Non-PAH) Bins",
        "",
        "| Bin ID | Index | Composition | Bin Rank | a0 (μm) | a_min (μm) | a_max (μm) | σ | s (g/cm³) |",
        "|--------|-------|-------------|----------|---------|------------|------------|---|-----------|",
    ]
    
    # Add dust bins to table
    for bin_info in dust_bins:
        bin_id = bin_info['id']
        idx = bin_info.get('index', '?')
        comp = bin_info['composition']
        rank = bin_info['bin_rank']
        
        # Get lognormal parameters
        params = get_bin_by_rank(comp, rank, is_pah=False)
        if params:
            a0 = params.get('a0', '?')
            amin = params.get('amin', '?')
            amax = params.get('amax', '?')
            sigma = params.get('sigma', '?')
            s = params.get('s', '?')
        else:
            a0 = amin = amax = sigma = s = '?'
        
        readme_lines.append(
            f"| {bin_id} | {idx} | {comp} | {rank} | {a0} | {amin} | {amax} | {sigma} | {s} |"
        )
    
    readme_lines.extend([
        "",
        "### PAH Bins",
        "",
        "| Bin ID | Index | Composition | Bin Rank | a0 (μm) | a_min (μm) | a_max (μm) | σ | s (g/cm³) |",
        "|--------|-------|-------------|----------|---------|------------|------------|---|-----------|",
    ])
    
    # Add PAH bins to table
    for bin_info in pah_bins:
        bin_id = bin_info['id']
        idx = bin_info.get('index', '?')
        comp = bin_info['composition']
        rank = bin_info['bin_rank']
        
        # Get lognormal parameters
        params = get_bin_by_rank(comp, rank, is_pah=True)
        if params:
            a0 = params.get('a0', '?')
            amin = params.get('amin', '?')
            amax = params.get('amax', '?')
            sigma = params.get('sigma', '?')
            s = params.get('s', '?')
        else:
            a0 = amin = amax = sigma = s = '?'
        
        readme_lines.append(
            f"| {bin_id} | {idx} | {comp} | {rank} | {a0} | {amin} | {amax} | {sigma} | {s} |"
        )
    
    readme_lines.extend([
        "",
        "---",
        "",
        "## Exported Datasets",
        "",
        "### 1. Dust Optical Properties",
        "",
    ])
    
    if 'dust_optical_properties' in export_results:
        dust_result = export_results['dust_optical_properties']
        readme_lines.extend([
            f"- **Status**: {dust_result['status']}",
            f"- **Export Time**: {dust_result['timestamp']}",
            f"- **Output Directory**: `{dust_result['dir']}`",
            f"- **Function**: `export_dust_optical_properties()` from `models.dust_radiation.export_dust_optical_properties`",
            f"- **Description**: Cross-sections (C_abs, C_sca, C_rp) and efficiencies (Q_abs, Q_sca, Q_rp)",
            f"  for all non-PAH dust bins computed using Draine-Lee 1984 optical data.",
            f"- **File Format**: One file per bin with columns:",
            f"  ```",
            f"  lambda[Angstrom] Q_abs Q_sca Q_rp C_abs[cm^2] C_sca[cm^2] C_rp[cm^2]",
            f"  ```",
            f"- **Files**: {dust_result.get('file_count', '?')} files ({dust_result.get('successful', 0)} successful, {dust_result.get('failed', 0)} failed)",
            "",
        ])
    
    readme_lines.extend([
        "### 2. PAH Optical Properties",
        "",
    ])
    
    if 'pah_optical_properties' in export_results:
        pah_result = export_results['pah_optical_properties']
        readme_lines.extend([
            f"- **Status**: {pah_result['status']}",
            f"- **Export Time**: {pah_result['timestamp']}",
            f"- **Output Directory**: `{pah_result['dir']}`",
            f"- **Function**: `export_pah_optical_properties()` from `models.PAH_radiation.pah_oppacity`",
            f"- **Description**: Cross-sections and efficiencies for PAH bins using Li & Draine 2001 data.",
            f"- **File Format**: Same as dust optical properties",
            f"- **Files**: {pah_result.get('file_count', '?')} files ({pah_result.get('successful', 0)} successful, {pah_result.get('failed', 0)} failed)",
            "",
        ])
    
    readme_lines.extend([
        "### 3. Collisional Cooling Data",
        "",
    ])
    
    if 'collisional_cooling' in export_results:
        cooling_result = export_results['collisional_cooling']
        readme_lines.extend([
            f"- **Status**: {cooling_result['status']}",
            f"- **Export Time**: {cooling_result['timestamp']}",
            f"- **Output Directory**: `{cooling_result['dir']}`",
            f"- **Function**: `export_collisional_cooling_bins()` from `models.dust_gas_collisions.export_collisional_cooling_bins`",
            f"- **Description**: Collisional cooling rate tables as function of temperature and ion charge",
            f"  for all non-PAH dust bins across 10 ion species.",
            f"- **Temperature Grid**: {cooling_result.get('temp_range', 'dynamic')}",
            f"- **Ion Species**: H, He, C, N, O, Ne, Mg, Si, S, Fe",
            f"- **Files**: {cooling_result.get('file_count', '?')} files",
            "",
        ])
    
    readme_lines.extend([
        "### 4. Sputtering Rates (T-φ Tables)",
        "",
    ])
    
    if 'sputtering_rates' in export_results:
        sputter_result = export_results['sputtering_rates']
        readme_lines.extend([
            f"- **Status**: {sputter_result['status']}",
            f"- **Export Time**: {sputter_result['timestamp']}",
            f"- **Output Directory**: `{sputter_result['dir']}`",
            f"- **Function**: `export_rates_T_phi()` from `models.dust_gas_collisions.dust_sputtering`",
            f"- **Description**: Sputtering yield rates as function of temperature and ion charge",
            f"  for all non-PAH dust bins across 10 ion species.",
            f"- **Temperature Grid**: {sputter_result.get('temp_range', 'dynamic')}",
            f"- **Ion Species**: Same as collisional cooling",
            f"- **Files**: {sputter_result.get('file_count', '?')} files + {sputter_result.get('figure_count', 0)} figures",
            "",
        ])

    readme_lines.extend([
        "### 5. PAH Sputtering Rates (phi=0)",
        "",
    ])

    if 'pah_sputtering_rates' in export_results:
        pah_sputter_result = export_results['pah_sputtering_rates']
        readme_lines.extend([
            f"- **Status**: {pah_sputter_result['status']}",
            f"- **Export Time**: {pah_sputter_result['timestamp']}",
            f"- **Output Directory**: `{pah_sputter_result['dir']}`",
            f"- **Function**: `export_rates_simple()` from `models.PAH_gas_collisions.PAH_sputtering`",
            f"- **Description**: PAH sputtering rates for phi=0 (charge effects neglected)",
            f"  exported for all PAH bins and ion species e, H, He, C, O.",
            f"- **Temperature Grid**: {pah_sputter_result.get('temp_range', 'dynamic')}",
            f"- **Files**: {pah_sputter_result.get('file_count', '?')} tables + {pah_sputter_result.get('figure_count', 0)} figures",
            "",
        ])

    readme_lines.extend([
        "### 6. Dust Grain Charge (Zmean vs Gamma)",
        "",
    ])

    if 'dust_charging' in export_results:
        charging_result = export_results['dust_charging']
        charging_source = charging_result.get('source', 'direct_charge_solver')
        charging_gamma_min = charging_result.get('gamma_min')
        charging_gamma_max = charging_result.get('gamma_max')
        charging_n_gamma = charging_result.get('n_gamma')
        charging_samples = charging_result.get('combos_per_gamma')
        charging_samples_min = charging_result.get('combos_per_gamma_min')
        charging_samples_max = charging_result.get('combos_per_gamma_max')
        charging_mode = charging_result.get('mode')
        charging_fixed = charging_result.get('fixed_value')
        charging_temp_min = charging_result.get('temperature_min')
        charging_temp_max = charging_result.get('temperature_max')
        charging_n_temp = charging_result.get('n_temperature')

        if charging_source == 'heating_tables':
            function_line = "- **Function**: Reuse of `make_rate_gamma_T_tables()` outputs from `models.dust_charge.dust_photoelectric_heating`"
            description_lines = [
                "- **Description**: Equilibrium charge moments (Zmean, Zsigma) reused from dust-photoelectric-heating",
                "  tables, preserving the same gamma/temperature sampling across both exports.",
            ]
        else:
            function_line = "- **Function**: `compute_charge_vs_gamma()` from `models.dust_charge.dust_charging`"
            description_lines = [
                "- **Description**: Equilibrium charge (Zmean) distribution for dust grains as function of",
                "  gamma parameter gamma = G0 * sqrt(T) / ne, sampled with random (G0, T, ne) combinations.",
            ]

        if charging_gamma_min is not None and charging_gamma_max is not None and charging_n_gamma is not None:
            gamma_line = f"- **Gamma Range**: {charging_gamma_min:.2e} to {charging_gamma_max:.2e} ({int(charging_n_gamma)} points)"
        else:
            gamma_line = "- **Gamma Range**: dynamic"

        if charging_samples is not None:
            sampling_line = f"- **Samples per Gamma**: {int(charging_samples)}"
        elif charging_samples_min is not None and charging_samples_max is not None:
            sampling_line = f"- **Samples per Gamma**: variable ({int(charging_samples_min)} to {int(charging_samples_max)})"
        else:
            sampling_line = "- **Samples per Gamma**: dynamic"

        readme_lines.extend([
            f"- **Status**: {charging_result['status']}",
            f"- **Export Time**: {charging_result['timestamp']}",
            f"- **Output Directory**: `{charging_result['dir']}`",
            function_line,
            *description_lines,
            gamma_line,
            sampling_line,
            f"- **Output Format**: PNG scatter plots + JSON data files with Zmean and Zsigma",
            f"- **Files**: {charging_result.get('file_count', '?')} files ({charging_result.get('bins_processed', 0)} bins processed)",
            "",
        ])
        if charging_source == 'heating_tables' and charging_temp_min is not None and charging_temp_max is not None and charging_n_temp is not None:
            readme_lines.extend([
                f"- **Temperature Grid Reused**: {charging_temp_min:.2e} to {charging_temp_max:.2e} K ({int(charging_n_temp)} points)",
            ])
        if charging_mode is not None:
            if charging_fixed is None:
                readme_lines.extend([f"- **Heating Coupling Mode**: {charging_mode}"])
            else:
                readme_lines.extend([f"- **Heating Coupling Mode**: {charging_mode} (fixed_value={float(charging_fixed):.3g})"])
        if charging_source == 'heating_tables' or charging_mode is not None:
            readme_lines.extend([""])

    readme_lines.extend([
        "### 7. Dust Photoelectric Heating Rates",
        "",
    ])

    if 'dust_photoelectric_heating' in export_results:
        heating_result = export_results['dust_photoelectric_heating']
        h_tmin = heating_result.get('Tmin')
        h_tmax = heating_result.get('Tmax')
        h_nt = heating_result.get('nT')
        h_gmin = heating_result.get('gamma_min')
        h_gmax = heating_result.get('gamma_max')
        h_ng = heating_result.get('n_gamma')
        h_mode = heating_result.get('mode')
        h_fixed = heating_result.get('fixed_value')
        h_rad = heating_result.get('radiation_model')

        if h_tmin is not None and h_tmax is not None and h_nt is not None:
            temp_grid_line = f"- **Temperature Grid**: {float(h_tmin):.2e} to {float(h_tmax):.2e} K ({int(h_nt)} points log-spaced)"
        else:
            temp_grid_line = "- **Temperature Grid**: dynamic"

        if h_gmin is not None and h_gmax is not None and h_ng is not None:
            gamma_grid_line = f"- **Gamma Grid**: {float(h_gmin):.2e} to {float(h_gmax):.2e} ({int(h_ng)} points log-spaced)"
        else:
            gamma_grid_line = "- **Gamma Grid**: dynamic"

        readme_lines.extend([
            f"- **Status**: {heating_result['status']}",
            f"- **Export Time**: {heating_result['timestamp']}",
            f"- **Output Directory**: `{heating_result['dir']}`",
            f"- **Function**: `make_rate_gamma_T_tables()` from `models.dust_charge.dust_photoelectric_heating`",
            f"- **Description**: Photoelectric heating and recombination cooling rates for dust grains as 2D tables",
            f"  on gamma × temperature grid. Rates computed using equilibrium charge distribution.",
            temp_grid_line,
            gamma_grid_line,
            f"- **Radiation Model**: {h_rad if h_rad is not None else 'dynamic'}",
            f"- **Mode**: {h_mode if h_mode is not None else 'dynamic'}" + (f" (fixed_value={float(h_fixed):.3g})" if h_fixed is not None else ""),
            f"- **Output Format**: NPZ (binary numpy) + PNG quick-look plots with metadata",
            f"- **Files**: {heating_result.get('file_count', '?')} files ({heating_result.get('bins_processed', 0)} bins processed)",
            "",
        ])

    readme_lines.extend([
        "### 8. PAH Photoelectric Heating Tables",
        "",
    ])

    if 'pah_photoelectric_heating_tables' in export_results:
        pah_heating_result = export_results['pah_photoelectric_heating_tables']
        readme_lines.extend([
            f"- **Status**: {pah_heating_result['status']}",
            f"- **Export Time**: {pah_heating_result['timestamp']}",
            f"- **Output Directory**: `{pah_heating_result['dir']}`",
            f"- **Function**: `compute_tables_ISRF()` from `models.PAH_charge.PAH_photoelectric_heating`",
            f"- **Description**: PAH photoelectric heating efficiency and ionization state tables as function of",
            f"  gamma = G0*sqrt(T)/ne for various interstellar radiation fields and PAH sizes (C54, C418).",
            f"- **Radiation Models**: Draine, Habing, Mathis, O6V, B0V, A0, and more",
            f"- **PAH Sizes**: Small (C54, a0~0.5 nm) and Large (C418, a0~1.5 nm)",
            f"- **Temperature Range**: 100 K to 10 kK (10 points log-spaced)",
            f"- **Electron Density Range**: 1 to 1000 cm⁻³ (20 points log-spaced)",
            f"- **Output Format**: DAT tables + PDF visualizations with efficiency and population fractions",
            f"- **Files**: {pah_heating_result.get('file_count', '?')} files ({pah_heating_result.get('tables_generated', 0)} tables generated)",
            "",
        ])

    readme_lines.extend([
        "### 9. PAH Dissociation Tables",
        "",
    ])

    if 'pah_dissociation_tables' in export_results:
        pah_diss_result = export_results['pah_dissociation_tables']
        readme_lines.extend([
            f"- **Status**: {pah_diss_result['status']}",
            f"- **Export Time**: {pah_diss_result['timestamp']}",
            f"- **Output Directory**: `{pah_diss_result['dir']}`",
            f"- **Function**: `plot_acetylene_dissociation_rate()` from `models.PAH_photophysics.PAH_photophysics`",
            f"- **Description**: Acetylene dissociation-rate tables and integrated contour plots for PAH bins",
            f"  defined in the grain-size configuration.",
            f"- **Output Format**: DAT tables + PNG contour plots (flat directory, no subfolders)",
            f"- **Files**: {pah_diss_result.get('file_count', '?')} files ({pah_diss_result.get('tables_generated', 0)} tables, {pah_diss_result.get('plots_generated', 0)} plots)",
            "",
        ])

    readme_lines.extend([
        "### 10. Dust Sublimation",
        "",
    ])

    if 'dust_sublimation' in export_results:
        subl_result = export_results['dust_sublimation']
        readme_lines.extend([
            f"- **Status**: {subl_result['status']}",
            f"- **Export Time**: {subl_result['timestamp']}",
            f"- **Output Directory**: `{subl_result['dir']}`",
            f"- **Function**: `export_dust_sublimation()` from `models.dust_radiation.dust_sublimation`",
            f"- **Description**: Dust sublimation rate tables and plots of sublimation timescales and rates.",
            f"  Sublimation rates are filtered to exclude temperatures where the sublimation timescale is",
            f"  longer than 10 times the age of the Universe, preventing interpolation noise.",
            f"- **Output Format**: DAT tables + PDF plots (sublimation vs T, and G0 and O6V timescales)",
            f"- **Files**: {subl_result.get('file_count', '?')} tables, {subl_result.get('plots_generated', 0)} plots",
            "",
        ])

    readme_lines.extend([
        "### 11. Dust-Assisted Ion Recombination Coefficients",
        "",
    ])

    if 'dust_ion_recombination' in export_results:
        recomb_result = export_results['dust_ion_recombination']
        r_tmin = recomb_result.get('Tmin')
        r_tmax = recomb_result.get('Tmax')
        r_nt = recomb_result.get('nT')
        r_gmin = recomb_result.get('gamma_min')
        r_gmax = recomb_result.get('gamma_max')
        r_ng = recomb_result.get('n_gamma')
        r_mode = recomb_result.get('mode')
        r_fixed = recomb_result.get('fixed_value')
        r_rad = recomb_result.get('radiation_model')

        if r_tmin is not None and r_tmax is not None and r_nt is not None:
            temp_grid_line = f"- **Temperature Grid**: {float(r_tmin):.2e} to {float(r_tmax):.2e} K ({int(r_nt)} points log-spaced)"
        else:
            temp_grid_line = "- **Temperature Grid**: dynamic"

        if r_gmin is not None and r_gmax is not None and r_ng is not None:
            gamma_grid_line = f"- **Gamma Grid**: {float(r_gmin):.2e} to {float(r_gmax):.2e} ({int(r_ng)} points log-spaced)"
        else:
            gamma_grid_line = "- **Gamma Grid**: dynamic"

        readme_lines.extend([
            f"- **Status**: {recomb_result['status']}",
            f"- **Export Time**: {recomb_result['timestamp']}",
            f"- **Output Directory**: `{recomb_result['dir']}`",
            f"- **Function**: `main()` from `models.dust_charge.export_dust_ion_recombination`",
            f"- **Description**: Recombination coefficients alpha (using Weingartner & Draine 2001 Case A threshold)",
            f"  for 11 ions ordered by atomic number: H, He, C, Na, Mg, Si, S, K, Ca, Mn, Fe.",
            temp_grid_line,
            gamma_grid_line,
            f"- **Radiation Model**: {r_rad if r_rad is not None else 'dynamic'}",
            f"- **Mode**: {r_mode if r_mode is not None else 'dynamic'}" + (f" (fixed_value={float(r_fixed):.3g})" if r_fixed is not None else ""),
            f"- **Output Format**: DAT tables + PNG quick-look plots + JSON metadata + NPZ binary",
            f"- **Files**: {recomb_result.get('file_count', '?')} files ({recomb_result.get('bins_processed', 0)} bins processed)",
            "",
        ])
    
    readme_lines.extend([
        "### 12. Dust Band Luminosities",
        "",
    ])

    if 'dust_band_luminosities' in export_results:
        band_result = export_results['dust_band_luminosities']
        readme_lines.extend([
            f"- **Status**: {band_result['status']}",
            f"- **Export Time**: {band_result['timestamp']}",
            f"- **Output Directory**: `{band_result['dir']}`",
            f"- **Function**: `export_band_luminosities()` from `models.dust_radiation.export_dust_band_luminosities`",
            f"- **Description**: Band-integrated specific luminosities (in erg/s/g) for Spitzer and Herschel bands",
            f"  across a range of dust temperatures.",
            f"- **Temperature Grid**: {band_result.get('Tmin', 1.0):.1f} to {band_result.get('Tmax', 5000.0):.1f} K ({int(band_result.get('nT', 500))} points log-spaced)",
            f"- **Output Format**: ASCII tables with columns: log10(T) followed by log10(L_spec) for the 9 filters",
            f"  `Spitzer_MIPS_24`, `Spitzer_MIPS_70`, `Spitzer_MIPS_160`, `Herschel_Pacs_70`, `Herschel_Pacs_100`,",
            f"  `Herschel_Pacs_160`, `Herschel_SPIRE_250`, `Herschel_SPIRE_350`, `Herschel_SPIRE_500`.",
            f"- **Files**: {band_result.get('file_count', '?')} files ({band_result.get('successful', 0)} successful, {band_result.get('failed', 0)} failed)",
            "",
        ])

    readme_lines.extend([
        "---",
        "",
        "## Methodology",
        "",
        "All computations depend **only** on the fundamental grain properties:",
        "  - **Composition**: graphite or silicate",
        "  - **Grain Size**: a0 (micron) from lognormal distribution parameters",
        "",
        "This design enables full generalization without reliance on hardcoded size-category labels.",
        "",
        "### Optical Properties Computation",
        "",
        "1. Read Draine-Lee optical efficiency tables (Q_abs, Q_sca, g)",
        "2. Interpolate in wavelength and grain size to target bin properties",
        "3. Convert efficiencies to cross-sections using geometric grain area",
        "4. Compute Q_rp = Q_abs + (1 - g) × Q_sca (radiation pressure efficiency)",
        "",
        "**References**:",
        "  - Draine & Lee (1984): Optical properties of interstellar graphite and silicate grains",
        "  - Li & Draine (2001): Infrared emission from dust in the diffuse interstellar medium",
        "",
        "### Collisional Cooling Data Computation",
        "",
        "1. Calculate grain-ion collision rates as function of temperature",
        "2. Compute energy transfer efficiency for each ion species and charge state",
        "3. Tabulate cooling rates across temperature grid",
        "",
        "### Sputtering Rates Computation",
        "",
        "1. Calculate sputtering yields from ion impact",
        "2. Compute sputtering rates as function of ion energy (temperature, charge)",
        "3. Generate T-φ tables for rapid lookup in simulations",
        "",
        "### Dust Grain Charge Computation",
        "",
        "1. Compute equilibrium ionisation potential and electron affinity",
        "2. Solve for equilibrium charge using photoelectric balance",
        "3. Sample multiple (G0, T, ne) combinations per gamma value",
        "4. Calculate mean and RMS charge distributions",
        "",
        "### Photoelectric Heating Rate Computation",
        "",
        "1. Calculate photoelectric emission cross-sections and yields",
        "2. Compute electron velocity distributions after photoemission",
        "3. Calculate energy carried away by escaped electrons (heating rate)",
        "4. Integrate recombination cooling over velocity distributions",
        "5. Tabulate heating and cooling rates on gamma × T grid",
        "",
        "---",
        "",
        "## File Organization",
        "",
        "```",
        "model_data/",
        "├── optical_properties/",
        "│   ├── averaged_cross_section_<BinID>.txt",
        "│   ├── band_luminosity_<BinID>.txt",
        "│   └── ...",
        "├── collisional_cooling_data/",
        "│   ├── cooling_<BinID>_Z_*",
        "│   └── ...",
        "├── thermal_sputtering_data/",
        "│   ├── sputtering_<BinID>_Z_*",
        "│   ├── sputtering_Tphi_overview_<BinID>-*.png",
        "│   └── ...",
        "├── pah_sputtering_data/",
        "│   ├── sputtering_<PAHBinID>_Z_0",
        "│   ├── sputtering_<PAHBinID>_Z_1",
        "│   ├── sputtering_<PAHBinID>_Z_2",
        "│   ├── sputtering_<PAHBinID>_Z_6",
        "│   ├── sputtering_<PAHBinID>_Z_8",
        "│   ├── sputtering_<PAHBinID>_quicklook.png",
        "│   └── ...",
        "├── dust_charging_data/",
        "│   ├── charge_<BinID>.png",
        "│   ├── charge_<BinID>.json",
        "│   ├── dust_charge_Z_vs_T_dustbin_###",
        "│   ├── dust_charge_sigma_vs_T_dustbin_###",
        "│   └── ...",
        "├── dust_photoelectric_heating_data/",
        "│   ├── heating_<BinID>.npz",
        "│   ├── heating_<BinID>.png",
        "│   ├── heating_<BinID>.json",
        "│   ├── dust_rates_heating_<mode>_<BinID>.dat",
        "│   ├── dust_rates_cooling_<mode>_<BinID>.dat",
        "│   └── ...",
        "├── PAH_dissociation_data/",
        "│   ├── dissociation_pah_bin_*.dat",
        "│   ├── *_integrated_dissociation_rate.png",
        "│   └── ...",
        "├── dust_sublimation/",
        "│   ├── sublimation_rate_DustBin_*.dat",
        "│   ├── sublimation_rate_vs_T.pdf",
        "│   ├── dust_sublimation.pdf",
        "│   └── ...",
        "├── dust_ion_recombination_data/",
        "│   ├── dust_rates_ion_recomb_<BinID>.dat",
        "│   ├── ion_recomb_<BinID>.npz",
        "│   ├── ion_recomb_<BinID>.png",
        "│   ├── ion_recomb_<BinID>.json",
        "│   └── ...",
        "└── README.md (this file)",
        "```",
        "",
        "---",
        "",
        "## Dependencies",
        "",
        "- **numpy**: Numerical array operations",
        "- **matplotlib**: Visualization (sputtering and heating figures)",
        "- **seaborn**: Publication-quality plot styling",
        "- **models.grain_size_config**: Central grain configuration loader",
        "- **models.dust_radiation.dust_oppacity**: Dust optical property interpolation",
        "- **models.PAH_radiation.pah_oppacity**: PAH optical property interpolation",
        "- **models.dust_gas_collisions.dust_collisional_cooling**: Collisional cooling calculations",
        "- **models.dust_gas_collisions.dust_sputtering**: Sputtering yield calculations",
        "- **models.PAH_gas_collisions.PAH_sputtering**: PAH sputtering calculations (phi=0 mode)",
        "- **models.dust_charge.dust_charging**: Equilibrium grain charge calculations",
        "- **models.dust_charge.dust_photoelectric_heating**: Photoelectric heating rate calculations",
        "- **models.dust_radiation.dust_sublimation**: Dust sublimation rate and timescale calculations",
        "",
        "---",
        "",
        "## How to Use",
        "",
        "### Generate all datasets at once:",
        "",
        "```bash",
        "cd /path/to/CALIMA",
        "python -m models.export_all_grain_data",
        "```",
        "",
        "### Generate individual datasets:",
        "",
        "```bash",
        "# Dust optical properties only",
        "python -m models.dust_radiation.export_dust_optical_properties",
        "",
        "# PAH optical properties only",
        "python -m models.PAH_radiation.export_pah_optical_properties",
        "",
        "# Collisional cooling data",
        "python -m models.dust_gas_collisions.export_collisional_cooling_bins",
        "",
        "# Sputtering rates for dust grains",
        "python -m models.dust_gas_collisions.export_sputtering_rates_bins",

        "# PAH sputtering rates (phi=0)",
        "python -m models.PAH_gas_collisions.export_pah_sputtering_rates_bins",
        "",
        "# Dust grain charge vs gamma",
        "python -m models.dust_charge.export_dust_charging_vs_gamma",
        "",
        "# Dust photoelectric heating rates",
        "python -m models.dust_charge.export_dust_photoelectric_heating",

        "# PAH dissociation tables",
        "python -m models.PAH_photophysics.export_pah_dissociation_tables",
        "",
        "# Dust sublimation rates and plots",
        "python -m models.dust_radiation.dust_sublimation",
        "",
        "# Dust-assisted ion recombination coefficients",
        "python -m models.dust_charge.export_dust_ion_recombination",
        "",
        "# Dust band luminosities",
        "python -m models.dust_radiation.export_dust_band_luminosities",
        "```",
        "",
        "### With custom configuration file:",
        "",
        "```bash",
        "python -m models.export_all_grain_data --config /path/to/config.json",
        "python -m models.dust_charge.export_dust_charging_vs_gamma --config /path/to/config.json",
        "python -m models.dust_charge.export_dust_photoelectric_heating --config /path/to/config.json",
        "```",
        "",
        "---",
        "",
        "## Changelog",
        "",
        f"**Latest Update**: {timestamp_str}",
        "",
        "- Git Branch: `{git_info['branch']}`",
        "- Git Commit: `{git_info['commit_short']}`",
        "- All datasets regenerated from configuration",
        "",
        "---",
        "",
        "## Notes",
        "",
        "- All computations are deterministic and depend only on grain size configuration.",
        "- Files can be safely regenerated by running this master export script.",
        "- For changes to output, please modify `models/grain_size_distribution.json`.",
        "- Timestamps indicate when each dataset was generated.",
        "",
        "---",
        "",
        f"*Generated by CALIMA master export script on {timestamp_str}*",
    ])
    
    # Write README
    readme_content = '\n'.join(readme_lines)
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    return str(readme_path)


def export_dust_optical_properties_wrapper(config_path=None):
    """Wrapper for dust optical properties export with error handling."""
    from pycalima.models.dust_radiation.export_dust_optical_properties import export_dust_optical_properties
    
    start_time = datetime.now()
    timestamp_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
    
    try:
        print("\n" + "="*80)
        print("EXPORTING DUST OPTICAL PROPERTIES")
        print("="*80)
        export_dust_optical_properties(config_path=config_path)
        _odir = str(get_model_data_dir() / 'optical_properties')
        return {
            'status': 'Success',
            'timestamp': timestamp_str,
            'dir': _odir,
            'successful': 4,
            'failed': 0,
            'file_count': 4,
        }
    except Exception as e:
        print(f"Error: {e}")
        return {
            'status': f'Error: {str(e)}',
            'timestamp': timestamp_str,
            'dir': str(get_model_data_dir() / 'optical_properties'),
            'successful': 0,
            'failed': 4,
            'file_count': 0,
        }


def export_pah_optical_properties_wrapper(config_path=None):
    """Wrapper for PAH optical properties export with error handling."""
    from pycalima.models.PAH_radiation.pah_oppacity import export_pah_optical_properties
    
    start_time = datetime.now()
    timestamp_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
    
    try:
        print("\n" + "="*80)
        print("EXPORTING PAH OPTICAL PROPERTIES")
        print("="*80)
        export_pah_optical_properties(config_path=config_path)
        _odir = str(get_model_data_dir() / 'optical_properties')
        return {
            'status': 'Success',
            'timestamp': timestamp_str,
            'dir': _odir,
            'successful': 3,
            'failed': 0,
            'file_count': 3,
        }
    except Exception as e:
        print(f"Error: {e}")
        return {
            'status': f'Error: {str(e)}',
            'timestamp': timestamp_str,
            'dir': str(get_model_data_dir() / 'optical_properties'),
            'successful': 0,
            'failed': 3,
            'file_count': 0,
        }


def export_collisional_cooling_wrapper(config_path=None):
    """Wrapper for collisional cooling export with error handling."""
    from pycalima.models.dust_gas_collisions.export_collisional_cooling_bins import main as export_cooling
    
    start_time = datetime.now()
    timestamp_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
    
    try:
        print("\n" + "="*80)
        print("EXPORTING COLLISIONAL COOLING DATA")
        print("="*80)
        export_cooling(config_path=config_path)
        return {
            'status': 'Success',
            'timestamp': timestamp_str,
            'dir': 'model_data/collisional_cooling_data',
            'temp_range': '1e1 - 1e9 K (100 points)',
            'file_count': 40,  # 4 bins × 10 ion species
        }
    except Exception as e:
        print(f"Error: {e}")
        return {
            'status': f'Error: {str(e)}',
            'timestamp': timestamp_str,
            'dir': 'model_data/collisional_cooling_data',
            'file_count': 0,
        }


def export_sputtering_rates_wrapper(config_path=None):
    """Wrapper for sputtering rates export with error handling."""
    from pycalima.models.dust_gas_collisions.export_sputtering_rates_bins import main as export_sputtering
    
    start_time = datetime.now()
    timestamp_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
    
    try:
        print("\n" + "="*80)
        print("EXPORTING SPUTTERING RATES")
        print("="*80)
        export_sputtering(config_path=config_path)
        return {
            'status': 'Success',
            'timestamp': timestamp_str,
            'dir': 'model_data/thermal_sputtering_data',
            'temp_range': '1e3 - 1e9 K (100 points)',
            'file_count': 40,  # 4 bins × 10 ion species (tables)
            'figure_count': 4,  # One figure per bin
        }
    except Exception as e:
        print(f"Error: {e}")
        return {
            'status': f'Error: {str(e)}',
            'timestamp': timestamp_str,
            'dir': 'model_data/thermal_sputtering_data',
            'file_count': 0,
            'figure_count': 0,
        }


def export_pah_sputtering_rates_wrapper(config_path=None):
    """Wrapper for PAH sputtering export (phi=0) with error handling."""
    from pycalima.models.PAH_gas_collisions.export_pah_sputtering_rates_bins import main as export_pah_sputtering

    start_time = datetime.now()
    timestamp_str = start_time.strftime('%Y-%m-%d %H:%M:%S')

    try:
        print("\n" + "="*80)
        print("EXPORTING PAH SPUTTERING RATES (PHI=0)")
        print("="*80)
        result = export_pah_sputtering(config_path=config_path)
        return {
            'status': 'Success',
            'timestamp': timestamp_str,
            'dir': result.get('output_dir', 'model_data/pah_sputtering_data'),
            'temp_range': '1e3 - 1e9 K (100 points)',
            'file_count': result.get('file_count', 0),
            'figure_count': result.get('figure_count', 0),
        }
    except Exception as e:
        print(f"Error: {e}")
        return {
            'status': f'Error: {str(e)}',
            'timestamp': timestamp_str,
            'dir': 'model_data/pah_sputtering_data',
            'file_count': 0,
            'figure_count': 0,
        }


def export_dust_charging_wrapper(config_path=None, reuse_heating_data=False):
    """Wrapper for dust charging vs gamma export with error handling."""
    from pycalima.models.dust_charge.export_dust_charging_vs_gamma import main as export_charging

    start_time = datetime.now()
    timestamp_str = start_time.strftime('%Y-%m-%d %H:%M:%S')

    try:
        print("\n" + "="*80)
        print("EXPORTING DUST CHARGING (ZMEAN VS GAMMA)")
        print("="*80)
        result = export_charging(config_path=config_path, reuse_heating_data=reuse_heating_data)
        return {
            'status': 'Success',
            'timestamp': timestamp_str,
            'dir': result.get('output_dir', 'model_data/dust_charging_data'),
            'file_count': result.get('file_count', 0),
            'bins_processed': result.get('bins_processed', 0),
            'source': result.get('source', 'direct_charge_solver'),
            'gamma_min': result.get('gamma_min'),
            'gamma_max': result.get('gamma_max'),
            'n_gamma': result.get('n_gamma'),
            'combos_per_gamma': result.get('combos_per_gamma'),
            'combos_per_gamma_min': result.get('combos_per_gamma_min'),
            'combos_per_gamma_max': result.get('combos_per_gamma_max'),
            'temperature_min': result.get('temperature_min'),
            'temperature_max': result.get('temperature_max'),
            'n_temperature': result.get('n_temperature'),
            'mode': result.get('mode'),
            'fixed_value': result.get('fixed_value'),
        }
    except Exception as e:
        print(f"Error: {e}")
        return {
            'status': f'Error: {str(e)}',
            'timestamp': timestamp_str,
            'dir': 'model_data/dust_charging_data',
            'file_count': 0,
            'bins_processed': 0,
            'source': 'direct_charge_solver',
        }


def export_dust_photoelectric_heating_wrapper(config_path=None):
    """Wrapper for dust photoelectric heating export with error handling."""
    from pycalima.models.dust_charge.export_dust_photoelectric_heating import main as export_heating

    start_time = datetime.now()
    timestamp_str = start_time.strftime('%Y-%m-%d %H:%M:%S')

    try:
        print("\n" + "="*80)
        print("EXPORTING DUST PHOTOELECTRIC HEATING RATES")
        print("="*80)
        result = export_heating(config_path=config_path)
        return {
            'status': 'Success',
            'timestamp': timestamp_str,
            'dir': result.get('output_dir', 'model_data/dust_photoelectric_heating_data'),
            'file_count': result.get('file_count', 0),
            'bins_processed': result.get('bins_processed', 0),
            'Tmin': result.get('Tmin'),
            'Tmax': result.get('Tmax'),
            'nT': result.get('nT'),
            'gamma_min': result.get('gamma_min'),
            'gamma_max': result.get('gamma_max'),
            'n_gamma': result.get('n_gamma'),
            'mode': result.get('mode'),
            'fixed_value': result.get('fixed_value'),
            'radiation_model': result.get('radiation_model'),
        }
    except Exception as e:
        print(f"Error: {e}")
        return {
            'status': f'Error: {str(e)}',
            'timestamp': timestamp_str,
            'dir': 'model_data/dust_photoelectric_heating_data',
            'file_count': 0,
            'bins_processed': 0,
        }


def export_pah_photoelectric_heating_tables_wrapper(config_path=None):
    """Wrapper for PAH photoelectric heating tables export with error handling."""
    from pycalima.models.PAH_charge.export_PAH_photoelectric_heating_tables import main as export_pah_tables

    start_time = datetime.now()
    timestamp_str = start_time.strftime('%Y-%m-%d %H:%M:%S')

    try:
        print("\n" + "="*80)
        print("EXPORTING PAH PHOTOELECTRIC HEATING TABLES")
        print("="*80)
        result = export_pah_tables(config_path=config_path)
        failed_count = int(result.get('failed', 0))
        if failed_count > 0:
            status = f"Error: {failed_count} configuration(s) failed"
        else:
            status = 'Success'
        return {
            'status': status,
            'timestamp': timestamp_str,
            'dir': result.get('output_dir', 'model_data/PAH_photoelectric_heating_data'),
            'file_count': result.get('file_count', 0),
            'tables_generated': result.get('tables_generated', 0),
            'failed': failed_count,
        }
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            'status': f'Error: {str(e)}',
            'timestamp': timestamp_str,
            'dir': 'model_data/PAH_photoelectric_heating_data',
            'file_count': 0,
            'tables_generated': 0,
        }


def export_pah_dissociation_tables_wrapper(config_path=None):
    """Wrapper for PAH dissociation tables export with error handling."""
    from pycalima.models.PAH_photophysics.export_pah_dissociation_tables import main as export_pah_dissociation

    start_time = datetime.now()
    timestamp_str = start_time.strftime('%Y-%m-%d %H:%M:%S')

    try:
        print("\n" + "="*80)
        print("EXPORTING PAH DISSOCIATION TABLES")
        print("="*80)
        result = export_pah_dissociation(config_path=config_path, overwrite=True)
        return {
            'status': 'Success',
            'timestamp': timestamp_str,
            'dir': result.get('output_dir', 'model_data/PAH_dissociation_data'),
            'file_count': result.get('file_count', 0),
            'tables_generated': result.get('tables_generated', 0),
            'plots_generated': result.get('plots_generated', 0),
        }
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            'status': f'Error: {str(e)}',
            'timestamp': timestamp_str,
            'dir': 'model_data/PAH_dissociation_data',
            'file_count': 0,
            'tables_generated': 0,
            'plots_generated': 0,
        }


def export_dust_sublimation_wrapper(config_path=None):
    """Wrapper for dust sublimation rate tables and plots export with error handling."""
    from pycalima.models.dust_radiation.dust_sublimation import export_dust_sublimation

    start_time = datetime.now()
    timestamp_str = start_time.strftime('%Y-%m-%d %H:%M:%S')

    try:
        print("\n" + "="*80)
        print("EXPORTING DUST SUBLIMATION TABLES AND PLOTS")
        print("="*80)
        result = export_dust_sublimation(config_path=config_path)
        return {
            'status': 'Success',
            'timestamp': timestamp_str,
            'dir': result.get('output_dir', 'model_data/dust_sublimation'),
            'file_count': len(result.get('tables', [])),
            'tables_generated': len(result.get('tables', [])),
            'plots_generated': len(result.get('plots', [])),
        }
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            'status': f'Error: {str(e)}',
            'timestamp': timestamp_str,
            'dir': 'model_data/dust_sublimation',
            'file_count': 0,
            'tables_generated': 0,
            'plots_generated': 0,
        }


def export_dust_ion_recombination_wrapper(config_path=None):
    """Wrapper for dust-assisted ion recombination export with error handling."""
    from pycalima.models.dust_charge.export_dust_ion_recombination import main as export_recomb

    start_time = datetime.now()
    timestamp_str = start_time.strftime('%Y-%m-%d %H:%M:%S')

    try:
        print("\n" + "="*80)
        print("EXPORTING DUST-ASSISTED ION RECOMBINATION COEFFICIENTS")
        print("="*80)
        result = export_recomb(config_path=config_path)
        return {
            'status': 'Success',
            'timestamp': timestamp_str,
            'dir': result.get('output_dir', 'model_data/dust_ion_recombination_data'),
            'file_count': result.get('file_count', 0),
            'bins_processed': result.get('bins_processed', 0),
            'Tmin': result.get('Tmin'),
            'Tmax': result.get('Tmax'),
            'nT': result.get('nT'),
            'gamma_min': result.get('gamma_min'),
            'gamma_max': result.get('gamma_max'),
            'n_gamma': result.get('n_gamma'),
            'mode': result.get('mode'),
            'fixed_value': result.get('fixed_value'),
            'radiation_model': result.get('radiation_model'),
        }
    except Exception as e:
        print(f"Error: {e}")
        return {
            'status': f'Error: {str(e)}',
            'timestamp': timestamp_str,
            'dir': 'model_data/dust_ion_recombination_data',
            'file_count': 0,
            'bins_processed': 0,
        }
 
 
def export_dust_band_luminosities_wrapper(config_path=None):
    """Wrapper for dust band luminosities export with error handling."""
    from pycalima.models.dust_radiation.export_dust_band_luminosities import export_band_luminosities

    start_time = datetime.now()
    timestamp_str = start_time.strftime('%Y-%m-%d %H:%M:%S')

    try:
        print("\n" + "="*80)
        print("EXPORTING DUST BAND LUMINOSITIES")
        print("="*80)
        export_band_luminosities(config_path=config_path)
        _odir = str(get_model_data_dir() / 'optical_properties')
        return {
            'status': 'Success',
            'timestamp': timestamp_str,
            'dir': _odir,
            'file_count': 4,
            'nT': 500,
            'Tmin': 1.0,
            'Tmax': 5000.0,
            'successful': 4,
            'failed': 0,
        }
    except Exception as e:
        print(f"Error: {e}")
        return {
            'status': f'Error: {str(e)}',
            'timestamp': timestamp_str,
            'dir': str(get_model_data_dir() / 'optical_properties'),
            'file_count': 0,
            'successful': 0,
            'failed': 4,
        }
 
 
# ---------------------------------------------------------------------------
# Export stages
# ---------------------------------------------------------------------------
# Ordered stage table driving main(). The name is the key used by --stages,
# --skip-stages, the profile JSON and the generated README.
#
# Order is significant: 'dust_charging' is passed reuse_heating_data=True so
# that it reuses the equilibrium charge solves already performed by
# 'dust_photoelectric_heating' instead of repeating them (22.6s vs several
# minutes at 20 dust bins). _select_stages() therefore refuses a selection
# that keeps 'dust_charging' but drops the stage it reuses.
_STAGES = (
    ('dust_optical_properties',          export_dust_optical_properties_wrapper,          {}),
    ('pah_optical_properties',           export_pah_optical_properties_wrapper,           {}),
    ('collisional_cooling',              export_collisional_cooling_wrapper,              {}),
    ('sputtering_rates',                 export_sputtering_rates_wrapper,                 {}),
    ('pah_sputtering_rates',             export_pah_sputtering_rates_wrapper,             {}),
    ('dust_photoelectric_heating',       export_dust_photoelectric_heating_wrapper,       {}),
    ('dust_charging',                    export_dust_charging_wrapper,                    {'reuse_heating_data': True}),
    ('pah_photoelectric_heating_tables', export_pah_photoelectric_heating_tables_wrapper, {}),
    ('pah_dissociation_tables',          export_pah_dissociation_tables_wrapper,          {}),
    ('dust_sublimation',                 export_dust_sublimation_wrapper,                 {}),
    ('dust_ion_recombination',           export_dust_ion_recombination_wrapper,           {}),
    ('dust_band_luminosities',           export_dust_band_luminosities_wrapper,           {}),
)

STAGE_NAMES = tuple(name for name, _func, _kwargs in _STAGES)

# stage -> stages whose output it reuses within the same invocation
_STAGE_REUSES = {'dust_charging': ('dust_photoelectric_heating',)}


def _parse_stage_list(spec):
    """Parse a comma-separated stage list, rejecting unknown names."""
    names = [item.strip() for item in spec.split(',') if item.strip()]
    unknown = [n for n in names if n not in STAGE_NAMES]
    if unknown:
        raise ValueError(
            f"Unknown export stage(s): {', '.join(unknown)}. "
            f"Available stages: {', '.join(STAGE_NAMES)}."
        )
    return names


def _select_stages(stages=None, skip_stages=None):
    """Resolve the --stages / --skip-stages selection to an ordered tuple.

    Returns the entries of :data:`_STAGES` to run, in table order, so a
    partial selection cannot reorder a reuse dependency.
    """
    if stages and skip_stages:
        raise ValueError('--stages and --skip-stages are mutually exclusive.')

    if stages:
        chosen = set(_parse_stage_list(stages))
    elif skip_stages:
        chosen = set(STAGE_NAMES) - set(_parse_stage_list(skip_stages))
    else:
        chosen = set(STAGE_NAMES)

    for stage, reused in _STAGE_REUSES.items():
        if stage not in chosen:
            continue
        missing = [r for r in reused if r not in chosen]
        if missing:
            raise ValueError(
                f"Stage '{stage}' reuses results from "
                f"{', '.join(missing)}, which would not run. Either include "
                f"that stage, or skip '{stage}' as well."
            )

    return tuple(entry for entry in _STAGES if entry[0] in chosen)


def main(config_path=None, profile=True, profile_output=None,
         stages=None, skip_stages=None):
    """
    Master export script that coordinates all grain data exports.
    
    Parameters
    ----------
    config_path : str, optional
        Path to JSON grain size configuration file.
        If not provided, uses default grain_size_distribution.json
    profile : bool, optional
        Collect and print per-stage wall-clock timings. Default True.
    profile_output : str, optional
        Path to write the stage profile as JSON.
    stages : str, optional
        Comma-separated subset of :data:`STAGE_NAMES` to run. Default: all.
    skip_stages : str, optional
        Comma-separated stages to omit. Mutually exclusive with *stages*.
        Skipping ``dust_ion_recombination`` alone removes roughly 45% of a
        full export's wall-clock time.

    Returns
    -------
    None
        Prints progress and writes tables under the generated-data directory.
        Returns early without exporting if the configuration cannot be loaded
        or the stage selection is invalid.
    """
    print("\n" + "="*80)
    print("CALIMA MASTER GRAIN DATA EXPORT SCRIPT")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # Resolve the stage selection first, so a bad name fails before any of the
    # expensive imports and solves below.
    try:
        selected_stages = _select_stages(stages, skip_stages)
    except ValueError as exc:
        print(f"\n  ✗ {exc}")
        return
    if not selected_stages:
        print("\n  ✗ The stage selection is empty; nothing to export.")
        return
    if len(selected_stages) != len(_STAGES):
        names = ', '.join(name for name, _f, _k in selected_stages)
        print(f"\nRunning {len(selected_stages)} of {len(_STAGES)} stages: {names}")
    
    # Set config path if provided
    if config_path:
        set_config_path(config_path)
        print(f"\nUsing custom configuration: {config_path}")
    else:
        print(f"\nUsing default configuration")
    
    # Get git information
    git_info = get_git_info()
    print(f"\nGit Information:")
    print(f"  Branch: {git_info['branch']}")
    print(f"  Commit: {git_info['commit_short']}")
    
    # Load grain configuration
    print("\nLoading grain size configuration...")
    try:
        config = load_grain_size_config()
        dust_bins = get_bins(is_pah=False)
        pah_bins = get_bins(is_pah=True)
        print(f"  ✓ {len(dust_bins)} dust bins")
        print(f"  ✓ {len(pah_bins)} PAH bins")
    except Exception as e:
        print(f"  ✗ Error loading configuration: {e}")
        return
    
    # Run the selected exports
    export_results = {}
    stage_profile = {}
    t_full_start = time.perf_counter()

    for stage_name, stage_func, stage_kwargs in selected_stages:
        export_results[stage_name] = _run_profiled_stage(
            stage_name,
            stage_func,
            config_path,
            stage_profile,
            enable_profile=profile,
            **stage_kwargs,
        )

    t_full_end = time.perf_counter()
    total_seconds = float(t_full_end - t_full_start)
    
    # Generate README
    print("\n" + "="*80)
    print("GENERATING README")
    print("="*80)
    try:
        readme_path = generate_readme(export_results, config, git_info)
        print(f"  ✓ README generated: {readme_path}")
    except Exception as e:
        print(f"  ✗ Error generating README: {e}")
        return
    
    # Summary
    print("\n" + "="*80)
    print("EXPORT SUMMARY")
    print("="*80)
    for name, result in export_results.items():
        status = result['status']
        print(f"{name:.<40} {status}")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    if profile:
        _print_profile_summary(stage_profile, total_seconds)
        if profile_output:
            out_path = _write_profile_json(stage_profile, total_seconds, profile_output)
            print(f"Profile JSON written to: {out_path}")

def _build_parser():
    parser = argparse.ArgumentParser(
        prog='calima-export',
        description='Master export script for grain and PAH optical properties and collision data.'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='JSON grain size configuration: a path, or a bundled short name '
             '("default", "ramses4bin", "4C6Si", "test"). Run calima-paths to '
             'see what is available.'
    )
    parser.add_argument(
        '--no-profile',
        action='store_true',
        help='Disable wall-clock stage profiling output at the end of the run.'
    )
    parser.add_argument(
        '--profile-output',
        type=str,
        default=None,
        help='Optional JSON path to write stage profile metrics.'
    )
    parser.add_argument(
        '--stages',
        type=str,
        default=None,
        help='Comma-separated subset of stages to run (default: all). '
             'Stages: ' + ', '.join(STAGE_NAMES) + '.'
    )
    parser.add_argument(
        '--skip-stages',
        type=str,
        default=None,
        help='Comma-separated stages to omit. Mutually exclusive with '
             '--stages. Skipping dust_ion_recombination alone removes '
             'roughly 45%% of the total wall-clock time.'
    )
    parser.add_argument(
        '--list-stages',
        action='store_true',
        help='Print the export stage names in run order and exit.'
    )
    return parser


def cli(argv=None) -> int:
    """Console-script wrapper: ``calima-export``.

    The argparse setup lives here rather than under ``if __name__``, because a
    console_scripts entry point calls its target with no arguments and would
    otherwise never see sys.argv. main()'s signature is deliberately
    unchanged: it is the de-facto plugin interface that this module uses to
    call the nine sibling exporters, all of which expose main(config_path=None).
    """
    args = _build_parser().parse_args(argv)
    if args.list_stages:
        for name in STAGE_NAMES:
            print(name)
        return 0
    main(config_path=args.config,
         profile=(not args.no_profile),
         profile_output=args.profile_output,
         stages=args.stages,
         skip_stages=args.skip_stages)
    return 0


if __name__ == '__main__':
    raise SystemExit(cli())
