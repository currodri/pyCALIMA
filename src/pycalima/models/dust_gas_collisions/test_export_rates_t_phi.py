#!/usr/bin/env python
"""Hardcoded test script for export_rates_T_phi with H, He, C, and O.

This script runs one export per element so each species can have its own
ion-charge interval used to estimate phi bounds.
"""

import numpy as np

import pycalima.models.dust_gas_collisions.dust_sputtering as dust_sputtering


def main():
    # Global hardcoded setup.
    composition = 'silicate'
    grain_radius_micron = 0.1
    dustlabel = 'dustbin_004'
    Tmin = 1e3
    Tmax = 1e9
    hnu_max_ev = 13.6

    nT = 100
    nphi = 100
    nbins_v = 200

    # Per-element ion setup and per-element charge ranges.
    # Zk_min/Zk_max define the charge interval used to build phi bounds.
    species = [
        {'name': 'H', 'mass': 1.008, 'Z': 1, 'Zk_min': 0, 'Zk_max': 1},
        {'name': 'He', 'mass': 4.002602, 'Z': 2, 'Zk_min': 0, 'Zk_max': 2},
        {'name': 'C', 'mass': 12.011, 'Z': 6, 'Zk_min': 0, 'Zk_max': 6},
        {'name': 'N', 'mass': 14.007, 'Z': 7, 'Zk_min': 0, 'Zk_max': 7},
        {'name': 'O', 'mass': 15.999, 'Z': 8, 'Zk_min': 0, 'Zk_max': 8},
        {'name': 'Ne', 'mass': 20.180, 'Z': 10, 'Zk_min': 0, 'Zk_max': 10},
        {'name': 'Mg', 'mass': 24.305, 'Z': 12, 'Zk_min': 0, 'Zk_max': 12},
        {'name': 'Si', 'mass': 28.086, 'Z': 14, 'Zk_min': 0, 'Zk_max': 14},
        {'name': 'S', 'mass': 32.065, 'Z': 16, 'Zk_min': 0, 'Zk_max': 16},
        {'name': 'Fe', 'mass': 55.845, 'Z': 26, 'Zk_min': 0, 'Zk_max': 26},
    ]

    print('Running export_rates_T_phi with hardcoded settings')
    print(f'composition={composition}, grain_radius_micron={grain_radius_micron}, Tmin={Tmin:.2e}, Tmax={Tmax:.2e}')
    print(f'dustlabel={dustlabel}')
    print(f'hnu_max_ev={hnu_max_ev:.2f}, nT={nT}, nphi={nphi}, nbins_v={nbins_v}')

    all_tables = []
    all_figures = []

    for sp in species:
        label = f"-{sp['name']}-Zk{sp['Zk_min']}to{sp['Zk_max']}"
        print(
            f"\n[{sp['name']}] m={sp['mass']:.6f} a.u., Z={sp['Z']}, "
            f"Zk=[{sp['Zk_min']}, {sp['Zk_max']}]"
        )

        result = dust_sputtering.export_rates_T_phi(
            Tmin=Tmin,
            Tmax=Tmax,
            dust_type=None,
            composition=composition,
            dustlabel=dustlabel,
            ion_atomic_masses=np.array([sp['mass']]),
            ion_atomic_numbers=np.array([sp['Z']]),
            Zk_min=sp['Zk_min'],
            Zk_max=sp['Zk_max'],
            grain_radius_micron=grain_radius_micron,
            hnu_max_ev=hnu_max_ev,
            nT=nT,
            nphi=nphi,
            nbins_v=nbins_v,
            do_size_correction=True,
            label=label,
        )

        print(f"phi_min={result['phi_min']:.6e}, phi_max={result['phi_max']:.6e}")
        print(
            'grain Z range '
            f"[{result['grain_charge_min_allowed']}, {result['grain_charge_max_allowed']}]"
        )
        print(f"figure={result['figure_file']}")
        for path in result['output_files']:
            print(f'table={path}')
            all_tables.append(path)
        all_figures.append(result['figure_file'])

    print('\nDone. Generated files:')
    print(f'tables={len(all_tables)}')
    print(f'figures={len(all_figures)}')


if __name__ == "__main__":
    main()
