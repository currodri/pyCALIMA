#!/usr/bin/env python3
"""Reference per-mode emission of the AmesPAHdb suite's ``cascade`` model.

Runs ``Transitions.cascade(E)`` (thermal approximation: the heat capacity
C(T) dT released as the molecule cools from the attained temperature,
shared among the modes as I nu^3 / (exp(h nu / kT) - 1)) for PAHdb species at
several excitation energies, and writes the fraction of E emitted in each
band (the PAHdb modes themselves are not written: PAHdb is never bundled).
The output is the bundled reference
``external_data/pahchem/amespahdb_cascade_band_fractions.csv`` for the emission
gates (tests/test_pahchem_emission.py), with the bands of ``BANDS``.

Needs the AmesPAHdb Python suite (requirements-dev.txt) and the PAHdb XML.

    make_pahdb_cascade_reference.py --xml pahdb-complete-theoretical-v4.00.xml --out ref.csv
"""
import argparse

import numpy as np
import pandas as pd

EV_ERG = 1.602176634e-12
SPECIES = {18: "C24H12", 37: "C54H18"}       # PAHdb v4.00 UIDs, neutral (as model_data/PAH_states)
ENERGIES_EV = (2.0, 4.0, 6.0, 8.0, 10.0, 12.0)
#: band -> [lo, hi) in cm^-1; must equal tests/test_pahchem_emission.py BANDS
BANDS = {"3.3": (2950, 3200), "6.2": (1500, 1700), "7.7": (1150, 1500), "8.6": (1000, 1150),
         "11.2": (850, 930), "12.7": (700, 850), "far-IR": (0, 700)}


def main():
    from amespahdbpythonsuite.amespahdb import AmesPAHdb

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--xml", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    db = AmesPAHdb(filename=args.xml, check=False, cache=True)
    rows = []
    for uid, formula in SPECIES.items():
        for e_ev in ENERGIES_EV:
            t = db.gettransitionsbyuid([uid])
            t.cascade(e_ev * EV_ERG, cache=False)
            d = t.data[uid]
            f = np.array([x["frequency"] for x in d])
            s = np.array([x["intensity"] for x in d])
            frac = s / s.sum()
            for band, (lo, hi) in BANDS.items():
                rows.append(dict(uid=uid, species=formula, e_ev=e_ev, band=band, lo_cm=lo, hi_cm=hi,
                                 energy_fraction=frac[(f >= lo) & (f < hi)].sum(),
                                 n_modes=int(((f >= lo) & (f < hi)).sum())))
            print(formula, e_ev, "T_max =", t.model["temperatures"], "modes", f.size)
    head = ("# AmesPAHdb Python suite Transitions.cascade(E) (Bauschlicher+18 emission model, thermal\n"
            "# approximation), PAHdb theoretical v4.00, no shift or broadening; energy_fraction = the\n"
            "# fraction of the excitation energy E emitted by the modes in [lo_cm, hi_cm). Written by\n"
            "# diagnostics/pahchem/make_pahdb_cascade_reference.py.\n")
    with open(args.out, "w") as fh:
        fh.write(head)
        pd.DataFrame(rows).to_csv(fh, index=False, float_format="%.6g")


if __name__ == "__main__":
    main()
