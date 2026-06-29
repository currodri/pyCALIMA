"""
DUST BAND LUMINOSITY EXPORTER

This script exports band-integrated luminosities in erg/s/g for all dust grain bins
defined in the grain size configuration for a temperature range of 1 to 5000 K.
The output tables are plain ASCII, formatted to be easily readable by Fortran code.

By: Antigravity / Pair Programming
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np

# Inject repo root into sys.path to allow imports when run as script
if __package__ in (None, ''):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from models.grain_size_config import (
    set_config_path,
    get_bins,
    get_lognormal_parameters,
    get_optical_props_path,
    get_header_lines,
    get_repo_root,
)
from models.dust_radiation.dust_oppacity import _read_precomputed_cross_section_table
from models.dust_radiation.dust_emission import compute_energy_band_luminosity_from_table


def export_band_luminosities(
    output_dir='model_data/optical_properties',
    config_path=None,
    num_temperatures=500,
    T_min=1.0,
    T_max=5000.0,
    spacing='log',
):
    """
    Export band-integrated luminosities for all dust bins in the configuration.
    """
    if config_path:
        set_config_path(config_path)

    repo_root = get_repo_root()
    output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load non-PAH grain bins from configuration
    non_pah_bins = get_bins(is_pah=False)
    if not non_pah_bins:
        print("No non-PAH dust bins found in the configuration.")
        return

    print(f"\nExporting band luminosities for {len(non_pah_bins)} non-PAH dust bins...")

    # Define the band filters
    filter_dir_spitzer = repo_root / 'external_data' / 'Spitzer_filters'
    filter_dir_herschel = repo_root / 'external_data' / 'Herschel_filters'

    filters = {
        'Spitzer_MIPS_24': filter_dir_spitzer / 'Spitzer_MIPS.24mu.dat',
        'Spitzer_MIPS_70': filter_dir_spitzer / 'Spitzer_MIPS.70mu.dat',
        'Spitzer_MIPS_160': filter_dir_spitzer / 'Spitzer_MIPS.160mu.dat',
        'Herschel_Pacs_70': filter_dir_herschel / 'Herschel_Pacs.blue.dat',
        'Herschel_Pacs_100': filter_dir_herschel / 'Herschel_Pacs.green.dat',
        'Herschel_Pacs_160': filter_dir_herschel / 'Herschel_Pacs.red.dat',
        'Herschel_SPIRE_250': filter_dir_herschel / 'Herschel_SPIRE.PSW_ext.dat',
        'Herschel_SPIRE_350': filter_dir_herschel / 'Herschel_SPIRE.PMW_ext.dat',
        'Herschel_SPIRE_500': filter_dir_herschel / 'Herschel_SPIRE.PLW_ext.dat',
    }

    # Verify that all filter files exist
    for name, path in filters.items():
        if not path.exists():
            raise FileNotFoundError(f"Filter file for {name} not found at {path}")

    # Set up temperature grid
    if spacing == 'log':
        T_grid = np.logspace(np.log10(T_min), np.log10(T_max), num_temperatures)
    else:
        T_grid = np.linspace(T_min, T_max, num_temperatures)

    # Process each bin
    successful = 0
    failed = 0

    for bin_info in non_pah_bins:
        bin_id = bin_info['id']
        composition = bin_info['composition']
        bin_rank = bin_info['bin_rank']
        is_pah = bin_info['is_pah']

        try:
            # Read precomputed cross sections
            # We assume they are stored in the same output_dir
            wavs, C_abs, C_sca, C_rp = _read_precomputed_cross_section_table(
                bin_id, optical_dir=output_dir
            )

            # Compute the grain mass
            cfg = get_lognormal_parameters(bin_id)
            # a0 is in microns, convert to cm
            m_grain = 4. / 3. * np.pi * (cfg['a0'] * 1e-4)**3. * cfg['s']

            # Output file path
            output_filename = f"band_luminosity_{bin_id}.txt"
            output_path = output_dir / output_filename

            # Get standardized header
            headers = get_header_lines(
                title=f"Dust Band Luminosities for {bin_id}",
                script_name="models/dust_radiation/export_dust_band_luminosities.py",
                bin_info=f"Bin ID: {bin_id}, Composition: {composition}, Bin rank: {bin_rank}, is_pah: {is_pah}, a0: {cfg['a0']} micron, s: {cfg['s']} g/cm^3, Mass: {m_grain:.6e} g",
                val_desc="T_dust Spitzer_MIPS_24 Spitzer_MIPS_70 Spitzer_MIPS_160 Herschel_Pacs_70 Herschel_Pacs_100 Herschel_Pacs_160 Herschel_SPIRE_250 Herschel_SPIRE_350 Herschel_SPIRE_500"
            )

            # Open file and write
            with open(output_path, 'w', encoding='utf-8') as f:
                for line in headers:
                    f.write(f"{line}\n")
                f.write(f"# Temperature points: {len(T_grid)}\n")
                f.write(f"# Values are in log10: temperature in log10(K), luminosities in log10(erg/s/g)\n")
                f.write(f"#\n")

                for T_dust in T_grid:
                    row_vals = []
                    for filter_name, filter_file in filters.items():
                        L_band = compute_energy_band_luminosity_from_table(
                            bin_id, T_dust, filter_file, wavs, C_abs
                        )
                        # Specific luminosity in erg/s/g
                        L_spec = L_band / m_grain
                        # Convert to log10, handle zero/negative values safely by clipping
                        log_L_spec = np.log10(np.maximum(L_spec, 1e-300))
                        row_vals.append(log_L_spec)

                    # Write row (temperature in log10)
                    f.write(f"{np.log10(T_dust):20.12e} ")
                    f.write(" ".join(f"{val:20.12e}" for val in row_vals) + "\n")

            print(f"  ✓ {output_filename}")
            successful += 1

        except Exception as e:
            print(f"  ✗ Error exporting band luminosities for {bin_id}: {e}")
            failed += 1

    print(f"\nBand luminosity export summary:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Output directory: {output_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Export Spitzer and Herschel band luminosities in erg/s/g for all dust bins.'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to JSON grain size configuration file. If not provided, uses default.'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='model_data/optical_properties',
        help='Output directory. Default: model_data/optical_properties'
    )
    parser.add_argument(
        '--nT',
        type=int,
        default=500,
        help='Number of temperature points. Default: 500'
    )
    parser.add_argument(
        '--Tmin',
        type=float,
        default=1.0,
        help='Minimum temperature. Default: 1.0'
    )
    parser.add_argument(
        '--Tmax',
        type=float,
        default=5000.0,
        help='Maximum temperature. Default: 5000.0'
    )
    parser.add_argument(
        '--spacing',
        type=str,
        choices=['log', 'linear'],
        default='log',
        help='Temperature grid spacing: log or linear. Default: log'
    )
    args = parser.parse_args()

    export_band_luminosities(
        output_dir=args.output_dir,
        config_path=args.config,
        num_temperatures=args.nT,
        T_min=args.Tmin,
        T_max=args.Tmax,
        spacing=args.spacing,
    )
