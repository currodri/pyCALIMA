#!/usr/bin/env python3
"""
Example script to test equilibrium charge computation and plotting.
"""
import os
import sys
import numpy as np

# ensure repository root is on sys.path so imports from project modules work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dust_charging import equilibrium_charge_for_grain, plot_charge_distribution
from dust_model import grain_charge_dist


def main():
    # example parameters
    G0 = 0.01                    # Habing units
    ne = 0.007       # cm^-3
    T = 33      # K
    grain_type = 'graphite'
    a_micron = 0.01

    # Simple WNM-like ion species for benchmarking (densities in cm^-3)
    # H+ and C+ approximations (very rough):
    ion_species = [
        {'n': 3e-3, 'T': T, 'm': 1.6726219e-27, 'z': 1},  # H+
        {'n': 0.0042, 'T': T, 'm': 12.0 * 1.66053906660e-27, 'z': 1},  # C+
    ]
    # ion_species = None  # use default ion species

    print('Computing equilibrium charge distribution...')
    Zs, P, rates, Zmean, Zsigma = equilibrium_charge_for_grain(G0, ne, T, grain_type, a_micron, ion_species=ion_species, debug=True,
                                                                radiation_model='Mathis')

    print(f'Zmean = {Zmean:.3f}, Zsigma = {Zsigma:.3f}')

    fig, ax = plot_charge_distribution(Zs, P, title=f'{grain_type}, a={a_micron} um, G0={G0}, ne={ne}, T={T} K')
    out = 'equilibrium_charge_histogram.png'
    
    # Add the data from WD01 Fig. 9 for comparison
    # [-1,0,1,2,3,4,5,6,7,8],[0.05,0.15,0.28,0.31,0.2,0.09,0.01,0,0,0]
    # ax.step([-2,-1,0,1,2,3,4],[0,0.08,0.25,0.4,0.22,0.07,0.0], where='mid', color='k', label='WD01 Fig. 9')

    # Add the results from Ibanez-Mejia et al. 2019 for comparison
    # grain_dist, Zdist = grain_charge_dist(G0, ne, T, grain_type, '10A')
    # ax.step(Zdist, grain_dist, where='mid', color='C1', label='IM19 10A')
    # ax.legend(loc='best',frameon=False)
    
    # Set the x axis limits based on the min/max Z values that have non-negligible probability
    significant = P > 1e-3
    if np.any(significant):
        Zmin = np.min(Zs[significant])
        Zmax = np.max(Zs[significant])
        ax.set_xlim(Zmin - 1, Zmax + 1)

    fig.savefig(out, dpi=200)
    print('Saved histogram to', out)


if __name__ == '__main__':
    main()
