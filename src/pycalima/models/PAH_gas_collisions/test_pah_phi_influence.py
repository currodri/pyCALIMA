#!/usr/bin/env python
"""Explore PAH nuclear destruction-rate dependence on phi for one ion setup.

This script builds a phi range from PAH charges in [-1, +2] for a chosen ion
charge and PAH size, then plots rate(T, phi) and representative T-slices.
"""

import os

import matplotlib.pyplot as plt

import pycalima.models.PAH_gas_collisions.PAH_sputtering as PAH_sputtering


def main():
    # User-editable setup
    ion_mass_amu = 12.011
    ion_atomic_number = 6
    ion_charge = 3

    RPAH_micron = 5e-4
    Tmin = 1e3
    Tmax = 1e9

    pah_charge_min = -1
    pah_charge_max = 2

    nT = 80
    # With PAH charges in [-1, 0, +1, +2], phi has 4 discrete values.
    nphi = 4

    threshold_energy = 7.5
    nbins_v = 400

    # Optimisation controls
    adaptive_nbins_v = True
    nbins_v_min = 80
    nbins_v_power = 1.0
    use_kernel_lookup = True
    nbins_v_lookup = 1200

    fig, T, phi_grid, rates = PAH_sputtering.plot_nuclear_phi_influence(
        Tmin=Tmin,
        Tmax=Tmax,
        ion_mass_amu=ion_mass_amu,
        ion_atomic_number=ion_atomic_number,
        ion_charge=ion_charge,
        RPAH_micron=RPAH_micron,
        pah_charge_min=pah_charge_min,
        pah_charge_max=pah_charge_max,
        nT=nT,
        nphi=nphi,
        threshold_energy=threshold_energy,
        nbins_v=nbins_v,
        adaptive_nbins_v=adaptive_nbins_v,
        nbins_v_min=nbins_v_min,
        nbins_v_power=nbins_v_power,
        use_kernel_lookup=use_kernel_lookup,
        nbins_v_lookup=nbins_v_lookup,
    )

    out_dir = './PAH_sputtering_data'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    out_file = os.path.join(out_dir, 'PAH_phi_influence_nuclear.png')
    fig.savefig(out_file, dpi=250)

    print('Saved:', out_file)
    print('T range [K]:', T[0], T[-1])
    print('phi range [eV]:', phi_grid[0], phi_grid[-1])
    print('phi values [eV]:', phi_grid)
    print('rates shape:', rates.shape)

    plt.close(fig)


if __name__ == '__main__':
    main()
