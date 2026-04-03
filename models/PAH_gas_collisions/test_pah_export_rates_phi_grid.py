#!/usr/bin/env python
"""Smoke test for PAH export_rates with multi-phi charge-grid support.

This test runs PAH_sputtering.export_rates with small grids and verifies that
per-element T-phi files are generated with a consistent phi grid containing
phi=0 and the expected number of discrete phi values.
"""

import os
import numpy as np

import models.PAH_gas_collisions.PAH_sputtering as PAH_sputtering


def read_tphi_header(path):
    """Read (nT, nphi, phi_grid) from a Fortran-style T-phi table."""

    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    nT, nphi = [int(x) for x in lines[0].split()[:2]]
    phi_grid = np.array([float(x) for x in lines[1].split()], dtype=float)
    return nT, nphi, phi_grid


def main():
    # Lightweight run configuration for a quick functional test.
    RPAH_micron = 10e-4
    Tmin = 1e3
    Tmax = 1e9
    nT = 100

    pah_charge_states = (-1, 0, 1, 2)
    ion_charge_ranges = {
        "H": [1],
        "He": [1, 2],
        "C": [1, 2, 3, 4, 5, 6, 7],
        "O": [1, 2, 3, 4, 5, 6],
    }

    print("Running export_rates multi-phi test...")
    PAH_sputtering.export_rates(
        RPAH=RPAH_micron,
        Tmin=Tmin,
        Tmax=Tmax,
        nT=nT,
        nbins_v=140,
        nbins_theta=16,
        radius_method="Draine21",
        pah_charge_states=pah_charge_states,
        ion_charge_ranges=ion_charge_ranges,
        adaptive_nbins_v=True,
        nbins_v_min=50,
        nbins_v_power=1.0,
        use_kernel_lookup=True,
        nbins_v_lookup=600,
        plot_rates=True,
        nH_plot=1.0,
        Z_plot=1,
        plot_phi_curves=True,
    )

    out_dir = "./PAH_sputtering_data"
    if not os.path.exists(out_dir):
        raise RuntimeError(f"Output folder not found: {out_dir}")

    # Build expected global phi grid from charge combinations.
    RPAH_cm = RPAH_micron * 1e-4
    all_ion_charges = sorted({z for values in ion_charge_ranges.values() for z in values})
    expected_phi = PAH_sputtering._build_phi_grid_from_charge_sets(
        all_ion_charges,
        list(pah_charge_states),
        RPAH_cm,
    )

    print(f"Expected phi values: {len(expected_phi)}")
    print(f"Expected phi grid: {expected_phi}")

    for species in ["H", "He", "C", "O"]:
        path = os.path.join(out_dir, f"{species}_sputtering_Tphi_{RPAH_micron:.4f}_micron_PAH")
        if not os.path.exists(path):
            raise RuntimeError(f"Missing T-phi table for {species}: {path}")

        nT_file, nphi_file, phi_grid = read_tphi_header(path)

        if nT_file != nT:
            raise RuntimeError(f"{species}: nT mismatch: got {nT_file}, expected {nT}")
        if nphi_file != len(expected_phi):
            raise RuntimeError(
                f"{species}: nphi mismatch: got {nphi_file}, expected {len(expected_phi)}"
            )
        if not np.any(np.isclose(phi_grid, 0.0, atol=1e-14, rtol=0.0)):
            raise RuntimeError(f"{species}: phi=0 not found in phi grid")
        # Export uses {:.8e} formatting for phi values, so allow small
        # roundoff-level differences when comparing against in-memory values.
        if not np.allclose(phi_grid, expected_phi, rtol=0.0, atol=1e-7):
            raise RuntimeError(
                f"{species}: phi grid values differ from expected.\n"
                f"got={phi_grid}\nexpected={expected_phi}"
            )

        print(f"OK: {species} | nT={nT_file}, nphi={nphi_file}")

    print("All checks passed.")


if __name__ == "__main__":
    main()
