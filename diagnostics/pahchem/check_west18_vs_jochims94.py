#!/usr/bin/env python3
"""H-loss thresholds of PAH cations: West+18 iPEPICO against Jochims+94 PIMS.

Jochims et al. (1994) measured the parent-ion internal energy E_int at the
(M-H)+ appearance potential and assumed the rate there is k ~ 1e4 s^-1. West
et al. (2018) give RRKM k(E) fitted to iPEPICO breakdown diagrams. For the
molecules in both, this prints the energy at which West's k(E) reaches 1e4 s^-1
next to Jochims' E_int, and West's k at Jochims' E_int.

The two disagree by 0.7-1.4 eV. That spread is the laboratory systematic on
"the" H-loss threshold, and it is larger than the error bars either paper
quotes. Needs the west2018-esi dataset (calima-fetch-data import ...).
"""
import argparse

import numpy as np

from pycalima.pahchem import dos, labdata

# West reaction number for M+ -> (M-H)+ + H, by Jochims molecule name
PAIRS = {"pyrene": 17, "fluoranthene": 11, "coronene": 20}


def main():
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    esi = labdata.west2018_esi()
    jochims = labdata.jochims94_table(1)
    print(f"{'molecule':14s} {'West E(k=1e4)':>14s} {'Jochims E_int':>14s} {'diff':>6s} {'West k at Jochims E':>20s}")
    for name, n in PAIRS.items():
        r = esi[n]
        E_ev, k = r.energy_cm / dos.CM_PER_EV, r.k_s
        ok = k > 0
        e4 = np.interp(4.0, np.log10(k[ok]), E_ev[ok])
        ej = jochims.loc[name, "eint_H_eV"]
        k_at = 10 ** np.interp(ej, E_ev[ok], np.log10(k[ok]))
        print(f"{name:14s} {e4:11.2f} eV {ej:11.2f} eV {e4 - ej:+6.2f} {k_at:17.2e} s^-1")


if __name__ == "__main__":
    main()
