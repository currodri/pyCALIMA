#!/usr/bin/env python3
"""Emission of the aromatic overtone and combination bands into the 3.4 um window (PAHdb anharmonic v1.00).

Harmonic spectra (PAHdb v4.00, the population fit) put nothing at 3.36-3.60 um in a purely aromatic PAH. The
anharmonic library (0 K absorption: fundamentals, overtones, combinations; B3LYP/N07D) does. This measures what
emission it gives, with no transfer assumption, for its pure-CH species that have a PAHdb v4.00 counterpart:

- **Cascade.** One UV photon of ``--e-uv`` eV, pahchem's harmonic cascade on the PAHdb v4.00 modes of the same
  species (same UID): the residence time P_j in each bin (photons over rate of the CH modes, as
  ``fit_34um_sidegroups``).
- **Lines.** Every anharmonic line emits from bin j at P_j A rho(E_j - nu) / rho(E_j) (``cooling.mode_emission_rates``
  for a pseudo-mode at the line's wavenumber): for a combination 1_i 1_j, rho(E - nu_i - nu_j) / rho(E) is the
  fraction of states with both modes excited, the leading term of its rate. First order: the lines do not feed
  back into the cascade's cooling, nor into its density of states.

Printed per species and E_UV: emitted energy in 3.36-3.60 um over 3.20-3.36 um, and the same for the 0 K
absorption; and the absorption ratios against the 6-9 um band, for a later size/charge argument.

    check_anharmonic_34um.py [--e-uv 5 8 11]    (needs pahdb-anharmonic-v1-00 and pahdb-theoretical-v4-00)
"""
import argparse

import numpy as np

from pycalima import _paths
from pycalima.pahchem import ames_suite, cooling, dos, emission, energy_me
from pycalima.pahchem.molecule import _EINSTEIN_A_PER_KM_MOL

WIN_33, WIN_34, WIN_CC = (3.20, 3.36), (3.36, 3.60), (6.0, 9.0)
#: PAHdb anharmonic v1.00 phenanthrene (UID 273) carries pyrene's 2675 transitions (UID 334), line for line
BAD_UIDS = {273}
DE_CM, GRAIN_CM, WINDOW_CM = 20.0, 2.0, 100.0


def window(f, x, lo_um, hi_um):
    return float(x[(f >= 1e4 / hi_um) & (f < 1e4 / lo_um)].sum())


def emitted(harm_f, harm_i, lines_f, lines_i, e_uv_ev):
    """Emitted energy per line [cm^-1 per absorbed photon, arbitrary normalisation] after one e_uv_ev photon."""
    a = _EINSTEIN_A_PER_KM_MOL * harm_f**2 * harm_i
    e_uv = e_uv_ev * dos.CM_PER_EV
    cm = energy_me.CascadeModel(harm_f, a, e_uv + 5 * DE_CM, DE_CM, GRAIN_CM, WINDOW_CM)
    start = np.zeros(cm.n)
    x = e_uv / DE_CM
    lo = int(np.floor(x))
    start[lo], start[lo + 1] = 1.0 - (x - lo), x - lo
    X = emission.emission_by_energy(cm, start)
    ch = harm_f > 2600.0
    kc = cm.k_modes[:cm.n_ir][ch].sum(axis=0)
    P = np.divide(X[ch].sum(axis=0), kc, out=np.zeros(cm.n), where=kc > 0)
    E, counts = dos.beyer_swinehart(harm_f, cm.energy_cm[-1] + WINDOW_CM, GRAIN_CM)
    sel = lines_i > 0
    k = cooling.mode_emission_rates(E, counts, lines_f[sel], _EINSTEIN_A_PER_KM_MOL * lines_f[sel] ** 2 * lines_i[sel],
                                    window_cm=WINDOW_CM, at_cm=cm.energy_cm)
    out = np.zeros(lines_f.size)
    out[sel] = (k @ P) * lines_f[sel]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--e-uv", type=float, nargs="+", default=[5.0, 8.0, 11.0])
    args = ap.parse_args()
    harm = {r["uid"]: r for r in ames_suite.xml_index(ames_suite.xml_path("pahdb-theoretical-v4-00"),
                                                      _paths.get_model_data_dir() / "pahchem_pahdb_index")}
    anh = [r for r in ames_suite.read_xml(ames_suite.xml_path("pahdb-anharmonic-v1-00"))
           if r["pure_ch"] and r["uid"] in harm and r["uid"] not in BAD_UIDS]
    head = "  ".join(f"{e:.0f} eV" for e in args.e_uv)
    print(f"species        Z | absorption: 3.4/3.3  3.4/(6-9 um)  3.3/(6-9 um) | emitted 3.4/3.3 at {head}")
    for r in sorted(anh, key=lambda r: (r["nc"], r["charge"])):
        h = harm[r["uid"]]
        lf, li = np.asarray(r["freqs_cm"], float), np.asarray(r["intensity"], float)
        hf, hi = np.asarray(h["freqs_cm"], float), np.asarray(h["intensity"], float)
        a33, a34, acc = window(lf, li, *WIN_33), window(lf, li, *WIN_34), window(lf, li, *WIN_CC)
        em = []
        for e in args.e_uv:
            p = emitted(hf, hi, lf, li, e)
            em.append(window(lf, p, *WIN_34) / window(lf, p, *WIN_33))
        print(f"{r['formula']:10s} uid {r['uid']:3d} {int(r['charge']):+d} | {a34 / a33:8.3f}  {a34 / acc:10.4f}  {a33 / acc:10.3f}   | "
              + "  ".join(f"{x:6.3f}" for x in em), flush=True)


if __name__ == "__main__":
    main()
