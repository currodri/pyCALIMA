#!/usr/bin/env python3
"""Murga & Wiebe (2022) Fig. 1: C66H20 H-loss rate R_H(chi) by four rate forms.

Their R_x = int k_diss(E) dp/dE dE (eq. 1) uses Draine & Li (2001) multiphoton
dp/dE in the Mathis field scaled by chi, with the Table 1 parameters:

- Arrh. 1: E0 = 3.3 eV, dS = 5 cal/K/mol;
- Arrh. 2: E0 = 4.3 eV, dS = 11.8;
- RRK: E0 = 2.8 eV, k0 = 1e16 s^-1;
- Laplace: E0 = 4.3 eV, k0 = 1.4e16.

Here dp/dE is the sink-free stationary P(E) of pahchem.energy_me (Li & Draine
sigma, all absorptions heating, G0 = 1.13 chi). The rate forms are those of
pahchem.rates. The "sink" column is the physical rate with a dissociation sink.

Measured (2026-10-04), log R_H [yr^-1], ours against MW22 Fig. 1 read by eye
(+-0.5 dex):

| | chi = 1 | chi = 1e4 |
|---|---|---|
| RRK | -7.7 vs ~-7.0 | +0.7 vs ~0.0 |
| Arrh. 1 | -2.5 vs ~-4.1 | +3.2 vs ~+0.3 |
| Arrh. 2 | -6.1 vs ~-9.3 | +1.4 vs ~-3.6 |
| Laplace | -7.1 vs ~-3.1 | +0.6 vs ~+1.7 |

RRK, the one form fully fixed by (s, k0, E0), agrees within 0.7 dex, which
validates the multiphoton P(E) against Draine & Li's dp/dE. The other three
differ by 1.6-5 dex, and their order differs from MW22's, so implementations
of the same named method differ by orders of magnitude on conventions the
paper does not state. Their Laplace behaves as if rho were taken with fewer
modes or including the ZPE; their Arrh. as if without the T_e correction.

Needs C66H20 modes, which are not shipped. Extract them from the PAHdb
(manual dataset pahdb-theoretical-v4-00) with
models.PAH_photophysics.pah_db_lookup: ``build_pahdb_catalog(xml, out.json,
nc_values=[66])``, then ``extract_modes_by_uid(xml, 115, "C66H20_0.dat")``.
"""
import argparse
from pathlib import Path

import numpy as np

from pycalima.pahchem import dos, energy_me, fields, photo, rates
from pycalima.pahchem.molecule import PAHSpecies

YR = 3.15576e7
CAL = 4.184


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("modes", type=Path, help="C66H20_0.dat (PAHdb transitions file)")
    args = ap.parse_args()

    EV = dos.CM_PER_EV
    sp = PAHSpecies.from_states_file(args.modes)
    cm = energy_me.CascadeModel(sp.freqs_cm, sp.einstein_a, e_max_cm=50 * EV, de_cm=100.0)
    E = cm.energy_cm
    Eg, counts = dos.beyer_swinehart(sp.freqs_cm, E[-1] + 200.0)
    k = {"Arrh. 1": rates.arrhenius_effective_temperature(sp.freqs_cm, E, 3.3 * EV, 5.0 * CAL),
         "Arrh. 2": rates.arrhenius_effective_temperature(sp.freqs_cm, E, 4.3 * EV, 11.8 * CAL),
         "RRK": rates.rrk(E, sp.n_modes, 1e16, 2.8 * EV),
         "Laplace": rates.inverse_laplace_arrhenius(Eg, counts, 1.4e16, 4.3 * EV)[np.rint(E).astype(int)]}
    E_ph = np.linspace(5.0, 13.6, 345)
    sigma = photo.sigma_li_draine(66, 0, E_ph)
    print(f"{'chi':>6s} " + " ".join(f"{n:>17s}" for n in k) + "   (log R_H [yr^-1]: sink-free / with sink)")
    for chi in (1.0, 1e1, 1e2, 1e3, 1e4, 1e5):
        h = photo.heating_spectrum(fields.mathis1983(g0=1.13 * chi), sigma, np.zeros_like(E_ph), E_ph)
        P = cm.steady_state(E_ph, h, {}).probability
        cells = []
        for name, kk in k.items():
            sink = cm.steady_state(E_ph, h, {name: kk}).rates[name]
            cells.append(f"{np.log10(P @ kk * YR):7.2f} /{np.log10(sink * YR):7.2f}")
        print(f"{chi:6.0e} " + " ".join(f"{c:>17s}" for c in cells), flush=True)


if __name__ == "__main__":
    main()
