#!/usr/bin/env python
"""
Simplified test script for the generalized export_collisional_cooling function
Uses the same ion species as test_export_rates_t_phi.py
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/currodri/Documents/GitHub/DustRAMSES')

from models.dust_gas_collisions.dust_collisional_cooling import export_collisional_cooling


# ============================================================================
# USER PARAMETERS - modify these to test different configurations
# ============================================================================
Tmin = 1e1                    # Minimum temperature [K]
Tmax = 1e9                  # Maximum temperature [K]
nT = 100                         # Number of temperature bins
nv = 200                         # Number of velocity bins
nphi = 100                        # Number of phi (grain charge) bins

grain_size_micron = 0.1        # Grain size [micron]
composition = 'Silicate'        # 'Graphite' or 'Silicate'
dust_label = 'dustbin_004'   # Label for output files and plots

# ============================================================================
# GRAIN COMPOSITION DEFINITIONS
# ============================================================================
grain_properties = {
    'Graphite': {
        'density': 2.24,                                # g/cm^3
        'atomic_mass': 12.011,                          # a.u. (Carbon)
        'atomic_number': 6.0,                           # atomic number
    },
    'Silicate': {
        'density': 3.3,                                 # g/cm^3 (MgFeSiO4)
        'atomic_mass': (24.305 + 55.845 + 28.0855 + 4*15.999) / 7.,  # a.u. (avg)
        'atomic_number': (4*8 + 14 + 26 + 12) / 7.,    # atomic number (avg)
    }
}

# ============================================================================
# ION SPECIES (same as test_export_rates_t_phi.py)
# ============================================================================
ion_species = [
    {'name': 'H',  'mass': 1.008,      'Z': 1,  'Z_max': 1},
    {'name': 'He', 'mass': 4.002602,   'Z': 2,  'Z_max': 2},
    {'name': 'C',  'mass': 12.011,     'Z': 6,  'Z_max': 6},
    {'name': 'N',  'mass': 14.007,     'Z': 7,  'Z_max': 7},
    {'name': 'O',  'mass': 15.999,     'Z': 8,  'Z_max': 8},
    {'name': 'Ne', 'mass': 20.180,     'Z': 10, 'Z_max': 10},
    {'name': 'Mg', 'mass': 24.305,     'Z': 12, 'Z_max': 12},
    {'name': 'Si', 'mass': 28.086,     'Z': 14, 'Z_max': 14},
    {'name': 'S',  'mass': 32.065,     'Z': 16, 'Z_max': 16},
    {'name': 'Fe', 'mass': 55.845,     'Z': 26, 'Z_max': 26},
]


def main():
    # Validate composition
    if composition not in grain_properties:
        print(f"Error: composition '{composition}' not recognized.")
        print(f"Available options: {list(grain_properties.keys())}")
        return

    # Get grain properties
    grain_props = grain_properties[composition]
    
    # Prepare ion arrays
    ion_masses = np.array([sp['mass'] for sp in ion_species])
    ion_atomic_numbers = np.array([sp['Z'] for sp in ion_species])
    nZ_ion = np.array([sp['Z_max'] for sp in ion_species])
    
    print("=" * 80)
    print("Testing generalized export_collisional_cooling function")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Grain composition: {composition}")
    print(f"  Grain size: {grain_size_micron} micron")
    print(f"  Temperature range: {Tmin:.2e} - {Tmax:.2e} K ({nT} bins)")
    print(f"  Velocity bins: {nv}")
    print(f"  Ion species: {len(ion_species)} ({', '.join([sp['name'] for sp in ion_species])})")
    print(f"  Output label: {dust_label}")
    print("=" * 80)
    
    try:
        export_collisional_cooling(
            Tmin=Tmin,
            Tmax=Tmax,
            grain_size_micron=grain_size_micron,
            grain_density=grain_props['density'],
            grain_atomic_mass=grain_props['atomic_mass'],
            grain_atomic_number=grain_props['atomic_number'],
            dust_label=dust_label,
            ion_atomic_masses=ion_masses,
            ion_atomic_numbers=ion_atomic_numbers,
            nZ_ion=nZ_ion,
            nT=nT,
            nv=nv,
            nphi=nphi,
            delta_max=0.1
        )
        print("\n✓ Cooling table generation completed successfully!")
        
        # Check if output files were created
        import os
        import glob
        
        output_files = glob.glob('./collisional_cooling_data/cooling_*')
        if output_files:
            cooling_files = [f for f in output_files if dust_label in f]
            print(f"\nGenerated {len(cooling_files)} output files for {dust_label}:")
            for f in sorted(cooling_files):
                file_size = os.path.getsize(f)
                print(f"  - {os.path.basename(f)} ({file_size} bytes)")
        else:
            print("\nNo output files found in ./collisional_cooling_data/")
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
