#!/usr/bin/env python
"""Quick test of the export_rates_simple function."""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from models.PAH_gas_collisions.PAH_sputtering import export_rates_simple


def main():
    print("Testing export_rates_simple function...")

    # Test parameters
    RPAH = 10e-4  # microns
    Tmin = 1e3
    Tmax = 1e9
    nT = 100

    # Call the function
    T, J_e, J_elec, J_ion = export_rates_simple(
        RPAH=RPAH,
        Tmin=Tmin,
        Tmax=Tmax,
        nT=nT,
        nbins_v=300,
        nbins_theta=30,
        pah_label='PAHbin_002'
    )

    print(f"\nTest Results:")
    print(f"  Temperature range: {T[0]:.2e} - {T[-1]:.2e} K")
    print(f"  Number of T points: {len(T)}")
    print(f"  Electron rates shape: {J_e.shape}")
    print(f"  Ion species computed: {list(J_ion.keys())}")

    # Check output files
    output_dir = './PAH_sputtering_data'
    if os.path.exists(output_dir):
        files = os.listdir(output_dir)
        print(f"\nOutput files created:")
        for f in sorted(files):
            fpath = os.path.join(output_dir, f)
            fsize = os.path.getsize(fpath)
            print(f"  - {f} ({fsize} bytes)")

    print("\n✓ Test completed successfully!")

    # Plot the exported tables
    print("\nPlotting exported tables...")

    sns.set_theme(style="white")
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": "Computer Modern Roman",
    })

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)
    fig.suptitle(f'PAH Sputtering Rates (R={RPAH*1e4:.1f} Å)', fontsize=14, fontweight='bold')

    # Plot electron rates
    ax = axes[0, 0]
    ax.plot(T, J_e, 'o-', linewidth=2.5, markersize=6, label='Electrons', color='C0')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Temperature [K]', fontsize=11)
    ax.set_ylabel('Rate [cm$^3$ s$^{-1}$]', fontsize=11)
    ax.set_title('Electron Sputtering', fontsize=12, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(frameon=False)

    # Plot ion electronic rates
    ax = axes[0, 1]
    for i, ptype in enumerate(['H', 'He', 'C', 'O']):
        ax.plot(T, J_elec[ptype], 'o-', linewidth=2.5, markersize=5,
                label=f'{ptype} electronic', color=f'C{i+1}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Temperature [K]', fontsize=11)
    ax.set_ylabel('Rate [cm$^3$ s$^{-1}$]', fontsize=11)
    ax.set_title('Ion Electronic Sputtering', fontsize=12, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(frameon=False, fontsize=9)

    # Plot ion nuclear rates
    ax = axes[1, 0]
    for i, ptype in enumerate(['H', 'He', 'C', 'O']):
        ax.plot(T, J_ion[ptype], 's-', linewidth=2.5, markersize=5,
                label=f'{ptype} nuclear', color=f'C{i+1}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Temperature [K]', fontsize=11)
    ax.set_ylabel('Rate [cm$^3$ s$^{-1}$]', fontsize=11)
    ax.set_title('Ion Nuclear Sputtering', fontsize=12, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(frameon=False, fontsize=9)

    # Plot total rates (electronic + nuclear)
    ax = axes[1, 1]
    ax.plot(T, 2.0*J_e, 'o-', linewidth=2.5, markersize=6, label='Electrons (×2)', color='C0')
    for i, ptype in enumerate(['H', 'He', 'C', 'O']):
        total = 2.0*J_elec[ptype] + J_ion[ptype]
        ax.plot(T, total, 'o-', linewidth=2.5, markersize=5,
                label=f'{ptype} total', color=f'C{i+1}', alpha=0.8)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Temperature [K]', fontsize=11)
    ax.set_ylabel('Rate [cm$^3$ s$^{-1}$]', fontsize=11)
    ax.set_title('Total Sputtering Rates', fontsize=12, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    plot_path = './PAH_sputtering_data/rates_comparison.pdf'
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved: {plot_path}")
    print("Done!")


if __name__ == '__main__':
    main()
