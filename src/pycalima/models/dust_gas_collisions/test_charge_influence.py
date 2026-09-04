#!/usr/bin/env python
"""
Test script for the charge influence plotting function.

This script demonstrates how to use the `plot_erosion_rate_charge_influence` function
to explore how ion and grain charges affect the erosion rate of small carbonaceous grains.
"""

import numpy as np
import models.dust_gas_collisions.dust_sputtering as dust_sputtering
import matplotlib.pyplot as plt


def main():
    """Main function - required for multiprocessing on macOS."""
    # Set up ion parameters (default to H+)
    ion_atomic_masses = np.array([15.999])
    ion_atomic_numbers = np.array([8])
    ion_charges = np.array([1])
    ion_abundances = np.array([1.0])

    # Set radiation field and electron density (user-provided constants)
    G0 = 1.0  # Radiation field strength (Habing units)
    ne = 0.1  # Electron density [cm^-3]

    # Temperature range [K]
    Tmin = 1e4
    Tmax = 1e7

    # Create the plot
    print("Generating charge influence plot for smallC grains...")
    fig = dust_sputtering.plot_erosion_rate_charge_influence(
        Tmin=Tmin,
        Tmax=Tmax,
        G0=G0,
        ne=ne,
        ion_atomic_masses=ion_atomic_masses,
        ion_atomic_numbers=ion_atomic_numbers,
        ion_charges=ion_charges,
        ion_abundances=ion_abundances,
        nT=50,          # Number of temperature points
        nZ_ion=8,      # Ion charges 0-10
        nbins_v=100     # Velocity bins for integration
    )

    # Save the figure
    fig.savefig('erosion_rate_charge_influence_smallC.png', dpi=300, bbox_inches='tight',format='png')
    print("Figure saved as 'erosion_rate_charge_influence_smallC.png'")

if __name__ == '__main__':
    main()
