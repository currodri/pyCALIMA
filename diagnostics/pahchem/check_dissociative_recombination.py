#!/usr/bin/env python3
"""Dissociative recombination and the Auger excitation (gas-coupling section 2.2c): H and H2 loss from the deposits the loss tables do
not hold, and how they move the hydrogenation.

For C24H12, C54H18, C96H24, C150H30 and C216H36 (as ``check_charge_ladder``):
- per event, ``population.deposit_response``: P(H) and P(H2) of a capture into charge Z (deposit IP(Z)) and of a C 1s Auger decay
  into it (``K_AUGER_EXCITATION_EV``), for the intact molecule (N_H0);
- in fields: the photon-driven H loss rate (the loss tables), the rate the captures and Auger decays add, and the intact share
  f(N_H0) and mean H lost, with ``Params.recombination_dissociation`` off and on:
  - PDRs: Kurucz 40 kK cut at 13.6 eV, G0 = 1e3-1e5, n_H = 1e4, n_e = 1e-4 n_H, T = 500 K;
  - ionised gas: the 100 eV field of ``check_charge_ladder_euv`` (G0 = 1e4, T = 1e4 K, n(H+) = n_e), atomic H 1e3;
  - X-rays only: F_nu ~ nu^-1 from 300 eV to 10 keV (``fields.powerlaw``, G0 its energy flux in Habing units, 1e2-1e4),
    n_H = 1e4, n_e = 1e-2, T = 1e3 K.
Nothing is fitted.

Result (2026-11-03; 3 min on 4 cores with the bases built, 1.9 GB):
- Per capture into N_H0: coronene+ + e- -> coronene loses H with P = 2e-6 (non-dissociative, as Le Page's statistical model has
  pyrene+ and larger); the loss grows with the deposit: C24 0.84 into +1 (11.2 eV), C54 0.32 at +5 and ~1 at +9, C96 5e-3 at +10
  and ~1 at +18, C150 1e-3 at +20, C216 0.06 at its cold limit (70 eV). Per Auger decay: C24 ~1, C54 4e-3-3e-2, C96 up < 1e-8.
- PDRs: no change (<H lost> <= +0.04, C24, stripped by the photons anyway). EUV: C96 f(N_H0) 0.980 -> 0.967 at psi = 1e5,
  <H lost> 21.4 -> 23.5 at 1e6. X-rays only (photon-driven loss exactly 0), with the outgoing electrons' collisions (the C 1s
  products keep 32-41 eV): at G0-equivalent 1e4 f(N_H0) of C24-C216 0.959, 0.900, 0.869, 0.895, 0.927 without dissociative
  recombination, 0.889, 0.773, 0.820, 0.894, 0.927 with it (rerun 2026-11-04 with the escape tables' measured partition;
  0.960 ... 0.931 and 0.889 ... 0.931 before; 11 min on 4 cores, 3.8 GB, the heat-node responses built).

    check_dissociative_recombination.py [--workers 4]
"""
import argparse
import sys
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import check_charge_ladder as L
import fit_pdrs4all_population as FP

from pycalima.pahchem import population as P
from pycalima.pahchem.baseline import Environment, Params


def envs():
    out = {}
    for g0 in (1e3, 1e4, 1e5):
        out[f"PDR G0 {g0:.0e}"] = Environment("pdr", g0, 1e4, 1.0, 500.0, 40000)
    for psi in (1e4, 1e5, 1e6):
        ne = 1e4 * 100.0 / psi
        out[f"EUV psi {psi:.0e}"] = Environment("euv", 1e4, 1e3, ne, 1e4, 40000, field_e_max_ev=100.0, ions=(("H+", ne),))
    for g0 in (1e2, 1e3, 1e4):
        out[f"X-rays G0eq {g0:.0e}"] = Environment("xdr", g0, 1e4, 1e-2, 1e3, None, field_e_min_ev=300.0,
                                                   field_e_max_ev=1e4, field_powerlaw=1.0)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    warnings.simplefilter("ignore")
    lib = L.library()
    pops = {d: P.Population(lib, L.WINDOWS, params=replace(Params(), recombination_dissociation=d), cache_dir=FP.BASIS_CACHE)
            for d in (False, True)}
    on = pops[True]
    print("Per event on N_H0: P(H) / P(H2) of a capture into Z (deposit IP(Z)) and of a C 1s Auger decay into Z (17.3 eV)")
    for mol in on.library:
        tops = max(mol.charges)
        row = []
        for z in sorted({0, 1, 2, 3, 5, 10, 20, tops - 1} & set(range(-1, tops))):
            rec = on._response(mol, z, "rec")[1][0]
            aug = on._response(mol, z, "aug")[1][0]                    # node 0: the Auger excitation alone
            row.append(f"{z:+d} [IP {P.state_threshold(mol, z):5.1f}]: {rec[0][1]:.1e}/{rec[0][2]:.1e} | Auger {aug[0][1]:.1e}/{aug[0][2]:.1e}")
        print(f"  {mol.name}:\n    " + "\n    ".join(row))
    for name, e in envs().items():
        shape = P.field_shape(e)
        for pop in pops.values():
            pop.prepare(shape, workers=args.workers)
            pop.prepare_responses(shape, workers=args.workers)
        f = on.charge_matrix(e)
        photon = np.einsum("sz,szk->sk", f, on._interp_g0(on._ladders(shape)[0], e.g0))[:, 0]
        dlh = on._deposit_losses(e)[0][:, 0]
        print(f"\n{name}: <Z>, k_H photon / deposits [s^-1] (N_H0), f(N_H0) and <H lost> without -> with dissociative recombination")
        h = {d: pop.hydrogenation_fractions(e) for d, pop in pops.items()}
        chains = {d: pop._chain_distributions(e) for d, pop in pops.items()}       # f(N_H), N_H = 0 ... N_H0
        zbar = f @ np.array(on.zs, float)
        for s, mol in enumerate(on.library):
            lost = {d: mol.nh - float(np.arange(mol.nh + 1) @ np.asarray(chains[d][s])) for d in chains}
            print(f"  {mol.name:14s} <Z> {zbar[s]:5.2f}  k_H {photon[s]:.1e} / {dlh[s]:.1e}  f(N_H0) {h[False][0][s, 0]:.3f} -> "
                  f"{h[True][0][s, 0]:.3f}  <H lost> {lost[False]:.2f} -> {lost[True]:.2f}")


if __name__ == "__main__":
    main()
