#!/usr/bin/env python
"""Export collisional cooling tables for all non-PAH grain bins.

This script uses the global grain-size JSON configuration and loops over all
non-PAH bins, generating cooling tables in `model_data/collisional_cooling_data`.
"""

import argparse
import concurrent.futures
import os
from pathlib import Path
import re

import numpy as np

from pycalima.models.dust_gas_collisions.dust_collisional_cooling import export_collisional_cooling
from pycalima.models.grain_size_config import set_config_path, get_bins, get_lognormal_parameters, get_export_parameters, get_model_data_dir


DEFAULT_EXPORT_PARAMS = {
    'Tmin': 1e1,
    'Tmax': 1e9,
    'nT': 100,
    'nv': 200,
    'nphi': 100,
    'delta_max': 0.1,
}


# Ion set consistent with existing generalized test scripts
ION_SPECIES = [
    {"name": "H", "mass": 1.008, "Z": 1, "Z_max": 1},
    {"name": "He", "mass": 4.002602, "Z": 2, "Z_max": 2},
    {"name": "C", "mass": 12.011, "Z": 6, "Z_max": 6},
    {"name": "N", "mass": 14.007, "Z": 7, "Z_max": 7},
    {"name": "O", "mass": 15.999, "Z": 8, "Z_max": 8},
    {"name": "Ne", "mass": 20.180, "Z": 10, "Z_max": 10},
    {"name": "Mg", "mass": 24.305, "Z": 12, "Z_max": 12},
    {"name": "Si", "mass": 28.086, "Z": 14, "Z_max": 14},
    {"name": "S", "mass": 32.065, "Z": 16, "Z_max": 16},
    {"name": "Fe", "mass": 55.845, "Z": 26, "Z_max": 26},
]


# Composition properties expected by export_collisional_cooling
COMPOSITION_PROPERTIES = {
    "graphite": {
        "density": 2.24,
        "atomic_mass": 12.011,
        "atomic_number": 6.0,
    },
    "silicate": {
        "density": 3.3,
        "atomic_mass": (24.305 + 55.845 + 28.0855 + 4 * 15.999) / 7.0,
        "atomic_number": (4 * 8 + 14 + 26 + 12) / 7.0,
    },
}




def main(config_path=None):
    if config_path:
        set_config_path(config_path)

    params_cfg = get_export_parameters('collisional_cooling', defaults=DEFAULT_EXPORT_PARAMS)
    Tmin = float(params_cfg['Tmin'])
    Tmax = float(params_cfg['Tmax'])
    nT = int(params_cfg['nT'])
    nv = int(params_cfg['nv'])
    nphi = int(params_cfg['nphi'])
    delta_max = float(params_cfg['delta_max'])
    
    table_dir = get_model_data_dir() / "collisional_cooling_data"
    table_dir.mkdir(parents=True, exist_ok=True)

    # Remove legacy composition-based filenames to keep output naming consistent.
    legacy_pattern = re.compile(r"^cooling_(graphite|silicate)_bin_\d+_Z_\d+$")
    for existing in table_dir.iterdir():
        if existing.is_file() and legacy_pattern.match(existing.name):
            existing.unlink()

    ion_masses = np.array([sp["mass"] for sp in ION_SPECIES])
    ion_atomic_numbers = np.array([sp["Z"] for sp in ION_SPECIES])
    nZ_ion = np.array([sp["Z_max"] for sp in ION_SPECIES])

    bins = sorted(
        get_bins(is_pah=False),
        key=lambda b: (b["composition"], b["bin_rank"], b["index"]),
    )

    if not bins:
        raise RuntimeError("No non-PAH bins found in grain-size configuration.")

    print("=" * 80)
    print("Exporting collisional cooling tables for non-PAH bins")
    print("=" * 80)
    print(f"Output directory: {table_dir}")
    print(f"Temperature grid: [{Tmin:.2e}, {Tmax:.2e}] with nT={nT}")
    print(f"Velocity bins: nV={nv} | phi bins: nphi={nphi}")
    print(f"Ion species: {len(ION_SPECIES)}")
    print("=" * 80)

    max_workers = min(10, (os.cpu_count() or 1))
    print(f"Shared worker pool: {max_workers}")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for bin_info in bins:
            composition = bin_info["composition"]
            bin_id = bin_info["id"]
            bin_rank = int(bin_info["bin_rank"])

            if composition not in COMPOSITION_PROPERTIES:
                raise ValueError(
                    f"Unsupported composition '{composition}' for bin '{bin_id}'."
                )

            # Use bin central size a0 (micron) from JSON config.
            params = get_lognormal_parameters(bin_id, model_name="basic")
            grain_size_micron = float(params["a0"])

            props = COMPOSITION_PROPERTIES[composition]
            dust_label = bin_id

            print(f"\n[bin={bin_id}] composition={composition}, rank={bin_rank}, a0={grain_size_micron:.4e} micron")

            export_collisional_cooling(
                Tmin=Tmin,
                Tmax=Tmax,
                grain_size_micron=grain_size_micron,
                grain_density=props["density"],
                grain_atomic_mass=props["atomic_mass"],
                grain_atomic_number=props["atomic_number"],
                dust_label=dust_label,
                ion_atomic_masses=ion_masses,
                ion_atomic_numbers=ion_atomic_numbers,
                nZ_ion=nZ_ion,
                nT=nT,
                nv=nv,
                nphi=nphi,
                delta_max=delta_max,
                table_dir=str(table_dir),
                executor=executor,
            )

    print("\nAll non-PAH bins exported successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Export collisional cooling tables for all non-PAH dust bins.'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to JSON grain size configuration file. If not provided, uses default.'
    )
    args = parser.parse_args()
    main(config_path=args.config)
