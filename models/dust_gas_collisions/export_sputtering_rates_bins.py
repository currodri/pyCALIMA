#!/usr/bin/env python
"""Export sputtering T-phi tables for all non-PAH grain bins.

This script uses the global grain-size JSON configuration, iterates over
all non-PAH bins, and computes sputtering tables per ion species.
Generated files are copied into `model_data/thermal_sputtering_data`.
"""

import argparse
import concurrent.futures
import os
from pathlib import Path
import shutil

import numpy as np

import models.dust_gas_collisions.dust_sputtering as dust_sputtering
from models.grain_size_config import set_config_path, get_bins, get_lognormal_parameters, get_export_parameters


DEFAULT_EXPORT_PARAMS = {
    'Tmin': 1e3,
    'Tmax': 1e9,
    'nT': 100,
    'nphi': 100,
    'nbins_v': 200,
    'hnu_max_ev': 13.6,
}


# Per-element charge setup (same pattern as test_export_rates_t_phi.py)
ION_SPECIES = [
    {"name": "H", "mass": 1.008, "Z": 1, "Zk_min": 0, "Zk_max": 1},
    {"name": "He", "mass": 4.002602, "Z": 2, "Zk_min": 0, "Zk_max": 2},
    {"name": "C", "mass": 12.011, "Z": 6, "Zk_min": 0, "Zk_max": 7},
    {"name": "N", "mass": 14.007, "Z": 7, "Zk_min": 0, "Zk_max": 6},
    {"name": "O", "mass": 15.999, "Z": 8, "Zk_min": 0, "Zk_max": 6},
    {"name": "Ne", "mass": 20.180, "Z": 10, "Zk_min": 0, "Zk_max": 6},
    {"name": "Mg", "mass": 24.305, "Z": 12, "Zk_min": 0, "Zk_max": 6},
    {"name": "Si", "mass": 28.086, "Z": 14, "Zk_min": 0, "Zk_max": 6},
    {"name": "S", "mass": 32.065, "Z": 16, "Zk_min": 0, "Zk_max": 6},
    {"name": "Fe", "mass": 55.845, "Z": 26, "Zk_min": 0, "Zk_max": 6},
]


def _repo_root():
    # models/dust_gas_collisions/<this_file>.py -> repo root at parents[2]
    return Path(__file__).resolve().parents[2]


def _copy_if_exists(src_path, dst_dir, dst_name=None):
    src = Path(src_path)
    if src.exists():
        dst = dst_dir / (dst_name if dst_name is not None else src.name)
        shutil.copy2(src, dst)
        return str(dst)
    return None


def main(config_path=None):
    if config_path:
        set_config_path(config_path)

    params_cfg = get_export_parameters('dust_sputtering', defaults=DEFAULT_EXPORT_PARAMS)
    Tmin = float(params_cfg['Tmin'])
    Tmax = float(params_cfg['Tmax'])
    nT = int(params_cfg['nT'])
    nphi = int(params_cfg['nphi'])
    nbins_v = int(params_cfg['nbins_v'])
    hnu_max_ev = float(params_cfg['hnu_max_ev'])
    
    repo_root = _repo_root()
    output_dir = repo_root / "model_data" / "thermal_sputtering_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    bins = sorted(
        get_bins(is_pah=False),
        key=lambda b: (b["composition"], b["bin_rank"], b["index"]),
    )
    if not bins:
        raise RuntimeError("No non-PAH bins found in grain-size configuration.")

    print("=" * 80)
    print("Exporting sputtering T-phi tables for non-PAH bins")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Temperature grid: [{Tmin:.2e}, {Tmax:.2e}] with nT={nT}")
    print(f"nphi={nphi}, nbins_v={nbins_v}")
    print(f"Ion species per bin: {len(ION_SPECIES)}")
    print("=" * 80)

    copied_files = []

    max_workers = min(os.cpu_count() or 1, 5)
    print(f"Shared worker pool: {max_workers}")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for bin_info in bins:
            bin_id = bin_info["id"]
            comp = bin_info["composition"]
            rank = int(bin_info["bin_rank"])
            dustlabel = f"{comp}_bin_{rank:02d}"
            params = get_lognormal_parameters(bin_id, model_name="basic")
            grain_size_micron = float(params["a0"])

            print(
                f"\n[bin={bin_id}] composition={comp}, rank={rank}, "
                f"grain_size={grain_size_micron:.4e} micron"
            )

            for sp in ION_SPECIES:
                label = f"-{sp['name']}-Zk{sp['Zk_min']}to{sp['Zk_max']}"
                print(
                    f"  -> {sp['name']}: m={sp['mass']:.6f}, Z={sp['Z']}, "
                    f"Zk=[{sp['Zk_min']},{sp['Zk_max']}]"
                )

                result = dust_sputtering.export_rates_T_phi(
                    Tmin=Tmin,
                    Tmax=Tmax,
                    dust_type=None,
                    dustlabel=dustlabel,
                    composition=comp,
                    ion_atomic_masses=np.array([sp["mass"]]),
                    ion_atomic_numbers=np.array([sp["Z"]]),
                    Zk_min=sp["Zk_min"],
                    Zk_max=sp["Zk_max"],
                    grain_radius_micron=grain_size_micron,
                    hnu_max_ev=hnu_max_ev,
                    nT=nT,
                    nphi=nphi,
                    nbins_v=nbins_v,
                    do_size_correction=True,
                    label=label,
                    executor=executor,
                )

                for src in result["output_files"]:
                    src_name = Path(src).name
                    dst_name = src_name.replace('thermal_sputtering_', 'sputtering_', 1)
                    dst = _copy_if_exists(src, output_dir, dst_name=dst_name)
                    if dst is not None:
                        copied_files.append(dst)
                fig_src_name = Path(result["figure_file"]).name
                fig_dst_name = fig_src_name.replace('thermal_sputtering_', 'sputtering_', 1)
                fig_dst = _copy_if_exists(result["figure_file"], output_dir, dst_name=fig_dst_name)
                if fig_dst is not None:
                    copied_files.append(fig_dst)

    print("\nDone.")
    print(f"Copied {len(copied_files)} files to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Export sputtering rates for all non-PAH dust bins.'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to JSON grain size configuration file. If not provided, uses default.'
    )
    args = parser.parse_args()
    main(config_path=args.config)
