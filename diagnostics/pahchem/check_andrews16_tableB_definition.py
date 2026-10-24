#!/usr/bin/env python3
"""What Andrews+2016 Table B (and Fig. 6) actually tabulate.

Table B.1 gives, for neutral C24H12, k_H(G0 = 1) = 10^-5.8 = 1.6e-6 s^-1 and,
for C24H13, 10^3.8 s^-1. Both exceed the photon absorption rate (~1e-8 s^-1),
which no per-molecule photo-rate can do. The hypothesis tested here: the table
holds eq. (7) of the paper, k_i = int k_i(T) G(T) dT, which has no dissociation
sink, so it counts k_i x (cooling time) "events" per absorbed photon. The
network rate is r_abs P_i (eq. 8), which is bounded by r_abs.

Coronene is in the single-photon regime at G0 = 1 (Andrews Sect. 2.4.3), so for
a photon of energy E the molecule cools along dE/dt = -k_IR(E) E (eq. 3) and
  N(E) = int_E_act^E k_i(E') / (k_IR(E') E') dE'     (no sink, eq. 7)
  P(E) = 1 - exp(-N(E))                              (with a sink)
Inputs are Andrews' own: Table 2 Arrhenius parameters with T_e = T_m(1 - 0.2
E_act/E), Malloci neutral coronene sigma, Draine field / 1.7 (Habing G0 = 1),
6-13.6 eV. PAHdb C24H12 modes stand in for C24H12 and C24H13. C24H3 (75 modes)
has no mode file here, so it is bracketed by C24 (66) and C24H12 (102): fewer
modes means a faster k(E).

The rates come from the legacy PAH_photophysics helpers; this script is the M0
audit of docs/physics/pahchem-spec.md and is superseded once pahchem.dos and
pahchem.rates exist.
"""
import argparse
from pathlib import Path

import numpy as np

from pycalima import _paths
from pycalima.models.PAH_photophysics.pah_mol_data import (
    compute_rrkm_dissociation_rate,
    compute_thermal_ir_rate,
)
from pycalima.models.tools.radiation_fields import Draine_1978_isrf

HC_EV_NM = 1239.84193

# Table B.1, Z = 0: (label, mode file, E_act [eV], dS [J/K/mol], p0 = log10 k(G0=1))
ROWS = [
    ("C24H12 H-loss, even", "C24H12_0.dat", 4.60, 44.8, -5.800),
    ("C24H3  H-loss, odd [C24]", "C24_0.dat", 4.10, 55.6, -1.850),
    ("C24H3  H-loss, odd [C24H12]", "C24H12_0.dat", 4.10, 55.6, -1.850),
    ("C24H13 H-loss, super-H", "C24H12_0.dat", 1.40, 55.6, 3.791),
]


def absorption_spectrum():
    """Photon absorption rate per eV [s^-1 eV^-1] of neutral coronene at G0 = 1."""
    sig = np.loadtxt(_paths.get_external_data_path("malloci_coronene_neutral_sigma.dat"))
    E = np.linspace(6.0, 13.6, 153)
    sigma = np.interp(E, sig[:, 0], sig[:, 1]) * 1e-18          # Mb -> cm^2
    F_lam = Draine_1978_isrf(HC_EV_NM / E) / 1.7                 # photons cm^-2 s^-1 nm^-1
    return E, sigma * F_lam * HC_EV_NM / E**2


def events_per_photon(modes, E_ph, E_act, dS, n_grid=4000):
    """N(E) of the module docstring, on the photon-energy grid."""
    Eg = np.linspace(E_act * 1.0001, E_ph.max(), n_grid)
    k = np.array([compute_rrkm_dissociation_rate(modes, e, E_act, dS) for e in Eg])
    k_ir = np.array([compute_thermal_ir_rate(modes, e)[2] for e in Eg])
    f = k / (k_ir * Eg)
    N = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(Eg))])
    return np.interp(E_ph, Eg, N, left=0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--states-dir", type=Path,
                    default=_paths.get_model_data_dir() / "PAH_states",
                    help="directory of PAHdb transitions files (default: %(default)s)")
    args = ap.parse_args()
    for f in sorted({row[1] for row in ROWS}):
        if not (args.states_dir / f).exists():
            ap.error(f"{args.states_dir / f} not found; extract it with pah_db_lookup")

    E, R = absorption_spectrum()
    r_abs = np.trapezoid(R, E)
    print(f"r_abs(C24H12, G0 = 1) = {r_abs:.2e} s^-1\n")
    print(f"{'state [modes]':28s} {'Table B.1':>10s} {'eq.7 no sink':>13s} {'dex':>6s} {'r_abs P':>10s}")
    for label, modes, E_act, dS, p0 in ROWS:
        N = events_per_photon(str(args.states_dir / modes), E, E_act, dS)
        k7 = np.trapezoid(R * N, E)
        k_phys = np.trapezoid(R * -np.expm1(-N), E)
        print(f"{label:28s} {10**p0:10.2e} {k7:13.2e} {np.log10(k7) - p0:+6.2f} {k_phys:10.2e}")
    print("\nTable B matches the sink-free integral; the per-molecule rate is r_abs P <= r_abs.")


if __name__ == "__main__":
    main()
