"""
DUST OPTICAL PROPERTIES BATCH EXPORTER

This script exports optical properties (cross-sections and efficiencies) for all
dust grain bins defined in the grain size configuration. Each bin's properties
are computed based on its size and composition and saved to individual files in
model_data/optical_properties/.

The computation depends only on grain size and composition, enabling full generalization
without reliance on hardcoded size-category labels like 'Small' or 'Large'.

By: Curro Rodriguez (currodri@gmail.com)
"""

import os
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from pycalima.models.grain_size_config import set_config_path, get_bins, get_lognormal_parameters, get_optical_props_path, get_header_lines, get_model_data_dir
from pycalima.models.dust_radiation.dust_oppacity import (
    dust_efficiencies,
    _compute_component_cross_sections_legacy,
    compute_component_cross_sections_mie,
    compute_isrf_averaged_cross_sections,
    export_dielectric_tables_for_bin,
)

PATH_OPTICS = str(get_optical_props_path())


def _save_optical_quicklook_plot(plot_path, wavelengths_cm, C_abs, C_sca, title):
    """Save a quick-look log-log plot of absorption, scattering and extinction."""
    wavelengths_micron = np.asarray(wavelengths_cm) * 1e4
    C_abs = np.asarray(C_abs)
    C_sca = np.asarray(C_sca)
    C_ext = C_abs + C_sca

    fig, ax = plt.subplots(figsize=(7, 5), dpi=180)
    ax.loglog(wavelengths_micron, C_abs, label='C_abs', linewidth=2)
    ax.loglog(wavelengths_micron, C_sca, label='C_sca', linewidth=2)
    ax.loglog(wavelengths_micron, C_ext, label='C_ext', linewidth=2)
    ax.set_xlabel('Wavelength [micron]')
    ax.set_ylabel(r'Cross section [cm$^2$]')
    ax.set_title(title)
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)


def export_dust_optical_properties(output_dir=None, config_path=None, cabs_method='precomputed', distribution_class=None, distribution_class_map=None):
    """
    Batch export optical properties for all dust grain bins.
    
    This function:
    1. Reads all non-PAH grain bins from grain_size_distribution.json (or specified config)
    2. For each dust bin (determined by composition and size)
    3. Computes optical cross-sections and efficiencies
    4. Saves results to output_dir/
    
    The computation depends ONLY on:
    - grain composition ('graphite' or 'silicate')
    - grain size a0 from lognormal parameters
    
    Parameters
    ----------
    output_dir : str
        Output directory for cross-section files. Default: 'model_data/optical_properties'
    config_path : str, optional
        Path to JSON configuration file. If provided, temporarily sets the config.
    cabs_method : str, optional
        Method to compute cross sections: 'precomputed' or 'mie'.
    """
    # Set config path if provided
    if config_path:
        set_config_path(config_path)
    if output_dir is None:
        output_dir = str(get_model_data_dir() / 'optical_properties')
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load non-PAH bins from configuration
    non_pah_bins = get_bins(is_pah=False)
    
    if not non_pah_bins:
        print("No non-PAH dust bins found in grain configuration.")
        return
    
    print(f"\nExporting optical properties for {len(non_pah_bins)} dust bins...")
    
    # Process each dust bin
    successful_exports = 0
    failed_exports = 0
    
    for bin_info in non_pah_bins:
        bin_id = bin_info['id']
        composition = bin_info['composition']
        bin_rank = bin_info['bin_rank']
        
        # Get grain size parameters for this bin
        lognormal_params = get_lognormal_parameters(bin_id)
        if not lognormal_params:
            print(f"  ✗ Could not find parameters for bin {bin_id}")
            failed_exports += 1
            continue
        
        a0 = lognormal_params.get('a0')
        if a0 is None:
            print(f"  ✗ No a0 parameter for bin {bin_id}")
            failed_exports += 1
            continue
        
        grain_size_micron = a0
        
        # Compute optical properties
        try:
            # Create a uniform log-spaced wavelength grid (241 points from 10^-3 to 10^3 micron)
            # to ensure constant bin separation in log space for downstream interpolation.
            target_wav_micron = np.logspace(-3, 3, 241)
            
            # Construct the distribution to get representative grain mass
            p = get_lognormal_parameters(bin_id)
            from pycalima.models.grain_distributions import LogNormal_Distribution
            _bin_dist_cls = (distribution_class_map or {}).get(bin_id, distribution_class)
            _dist_cls = _bin_dist_cls if _bin_dist_cls is not None else LogNormal_Distribution
            dist = _dist_cls(
                p['a0'] * 1e-4,
                p['amin'] * 1e-4,
                p['amax'] * 1e-4,
                p['sigma'],
                p['s'],
            )
            if cabs_method == 'mie':
                cabs_comps, csca_comps, crp_comps = compute_component_cross_sections_mie(
                    [bin_id], target_wavelengths=target_wav_micron, nsize_per_bin=30, verbose=False,
                    distribution_class=_bin_dist_cls,
                )
            else:
                cabs_comps, csca_comps, crp_comps = _compute_component_cross_sections_legacy(
                    [bin_id], target_wavelengths=target_wav_micron, nsize_per_bin=30, verbose=False,
                    distribution_class=_bin_dist_cls,
                )

            wavelengths_cm = target_wav_micron * 1e-4
            C_abs = cabs_comps[0]   # [cm^2/g_dust]
            C_sca = csca_comps[0]
            C_rp  = crp_comps[0]
            
        except Exception as e:
            print(f"  ✗ Error computing optical properties for bin {bin_id}: {e}")
            failed_exports += 1
            continue
        
        # Use a unified prefix for downstream tooling consistency.
        file_stem = f"averaged_cross_section_{bin_id}"
        output_filename = f"{file_stem}.txt"
        output_path = os.path.join(output_dir, output_filename)
        plot_filename = f"{file_stem}_quicklook.png"
        plot_path = os.path.join(output_dir, plot_filename)
        
        # Write to file
        try:
            isrf_avg = compute_isrf_averaged_cross_sections(
                wavelengths_cm=wavelengths_cm,
                C_abs=C_abs,
                C_sca=C_sca,
                C_rp=C_rp,
            )

            headers = get_header_lines(
                title="Dust optical properties",
                script_name="models/dust_radiation/export_dust_optical_properties.py",
                bin_info=f"Bin ID: {bin_id}, Composition: {composition}, Bin rank: {bin_rank}, Grain size a0: {a0} micron",
                val_desc="Columns: lambda[Angstrom] C_abs[cm^2/g_dust] C_sca[cm^2/g_dust] C_rp[cm^2/g_dust]"
            )

            with open(output_path, 'w') as f:
                for line in headers:
                    f.write(f"{line}\n")
                f.write(f"# NWAV\n")
                f.write(f"{len(wavelengths_cm):d}\n")
                f.write(f"# ISRF-average: Mathis83, energy range [0.1, 13.6] eV\n")
                f.write(f"# ISRF_AVG_CROSS_SECTIONS_CM2: C_abs_ISRF C_sca_ISRF C_rp_ISRF\n")
                f.write(f"{isrf_avg['C_abs_isrf']: .12E} {isrf_avg['C_sca_isrf']: .12E} {isrf_avg['C_rp_isrf']: .12E}\n")
                f.write(f"# \n")
                
                for j in range(len(wavelengths_cm)):
                    wavelength_angstrom = wavelengths_cm[j] * 1e8  # Convert cm to Angstroms
                    f.write(f"{wavelength_angstrom:20.12e} ")
                    f.write(f"{C_abs[j]:20.12e} ")
                    f.write(f"{C_sca[j]:20.12e} ")
                    f.write(f"{C_rp[j]:20.12e}\n")

            _save_optical_quicklook_plot(
                plot_path,
                wavelengths_cm,
                C_abs,
                C_sca,
                title=f"{composition} bin {bin_rank}, a0={a0:.4g} micron"
            )

            export_dielectric_tables_for_bin(
                bin_id=bin_id,
                composition=composition,
                output_dir=output_dir,
            )
            
            print(f"  ✓ {output_filename}")
            print(f"  ✓ {plot_filename}")
            successful_exports += 1
        
        except Exception as e:
            print(f"  ✗ Error writing {output_filename}: {e}")
            failed_exports += 1
    
    print(f"\nExport summary:")
    print(f"  Successful: {successful_exports}")
    print(f"  Failed: {failed_exports}")
    print(f"  Output directory: {output_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Export optical properties for all dust grain bins.'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to JSON grain size configuration file. If not provided, uses default.'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='mie',
        choices=['precomputed', 'mie'],
        help="Method to compute cross sections: 'precomputed' (legacy size interpolation) or 'mie' (direct Mie theory)"
    )
    args = parser.parse_args()
    
    if args.config:
        set_config_path(args.config)
    
    export_dust_optical_properties(cabs_method=args.method)
