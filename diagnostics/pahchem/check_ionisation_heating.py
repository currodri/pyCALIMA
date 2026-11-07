#!/usr/bin/env python3
"""The internal energy a photoionisation leaves its product (gas-coupling section 2.2d): ``Params.ionisation_heating`` off and on.

Each valence photoionisation (detachment) out of charge Z leaves the ion Z + 1 (Z + 2 for a double ionisation) with
E - IP - <E_k> (``photo.photoelectron_energy``: 54 % of the excess below 13.6 eV, the Hrodmarsson+25 shares to 21.9 eV, then
constant), followed down the product's own cascade (``population.ionisation_deposits``, ``Population._deposit_flows``). For
C24H12, C54H18, C96H24, C150H30 and C216H36 (as ``check_charge_ladder``):
- per event: the mean deposit <E_int> into each product and its P(H) on N_H0;
- in fields, off -> on: the total IR power per molecule, the 3.3/11.2, 6.2/11.2 and 7.7/11.2 band ratios, f(N_H0) and the mean H
  lost, in the fields of ``check_dissociative_recombination`` (PDRs: Kurucz 40 kK cut at 13.6 eV, G0 1e3-1e5; ionised gas: to
  100 eV, psi 1e4-1e6; X-rays only, 300 eV-10 keV, where no valence photon is absorbed and nothing changes but the C 1s
  products' valence-free chain).
Nothing is fitted.

Result (2026-11-03; ~5 min on 4 cores with the bases built, 1.6 GB):
- Per event: <E_int> into +1 2.4-3.4 eV cut at 13.6 eV, 4.3-4.7 eV to 100 eV, falling with the charge; into the neutral from the
  anion 4.9-7.5 eV. P(H) on N_H0 <= 2e-5 except neutral C24 (0.03): these deposits heat, they do not strip.
- PDRs: IR power per molecule x1.04-1.08 at G0 = 1e3 (3.3/11.2 -4 to -8 %, 7.7/11.2 +2 to +4 %), x1.01 at 1e4, x1.001 at 1e5.
- EUV (measured with photoelectron_reference "photon"; the "neutral" default adds x1.003-1.010): x1.26-1.28 at
  psi = 1e4 (3.3/11.2, 6.2/11.2 -5 %), x1.03 at 1e5, x1.000 at 1e6, where every state above IP 21.9 eV keeps
  nothing under "photon" (the hole held at 21.9 eV photon energy; docs section 2.2c).
- X-rays only: x1.003-1.006. f(N_H0) and <H lost> unchanged everywhere. C24 (and C54, C96 at psi = 1e6) emit nothing: stripped
  below the emitting ladder (nan ratios).

    check_ionisation_heating.py [--workers 4]
"""
import argparse
import sys
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import check_charge_ladder as L
import check_dissociative_recombination as DR
import fit_pdrs4all_population as FP

from pycalima.pahchem import population as P
from pycalima.pahchem.baseline import Params

WINDOWS = {"all": (1.0, 1e4), "3.3": (3.2, 3.36), "6.2": (6.0, 6.5), "7.7": (7.3, 8.2), "11.2": (11.05, 11.5)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    warnings.simplefilter("ignore")
    lib = L.library()
    pops = {h: P.Population(lib, WINDOWS, params=replace(Params(), ionisation_heating=h), cache_dir=FP.BASIS_CACHE)
            for h in (False, True)}
    on = pops[True]
    for name, e in DR.envs().items():
        shape = P.field_shape(e)
        for pop in pops.values():
            pop.prepare(shape, workers=args.workers)
            pop.prepare_responses(shape, workers=args.workers)
        if name.startswith("EUV psi 1e+05") or name.startswith("PDR G0 1e+04"):
            print(f"\n{name}: per photoionisation landing on Z, <E_int> [eV] and P(H) on N_H0 (dz = 1; dz = 2 if any)")
            ion = on._ion_stack(shape)
            for s, mol in enumerate(on.library):
                row = []
                for z in on._live_charges(mol, shape):
                    if (s, z - on.zmin) not in ion:
                        continue
                    rows = P.ionisation_products(mol, z, on.p, shape)
                    resp = on._response(mol, z, ("ion", shape))[1]
                    row.append(" ".join(f"{z:+d}<-{dz}: {float(w @ en):.2f} eV P(H) {resp[r][0][1]:.1e}"
                                        for r, (dz, _, en, w) in enumerate(rows)))
                print(f"  {mol.name}: " + "; ".join(row[:8]) + (" ..." if len(row) > 8 else ""))
        F = {h: pop.band_fluxes(e)[0] for h, pop in pops.items()}
        b = {k: i for i, k in enumerate(WINDOWS)}
        chains = {h: pop._chain_distributions(e) for h, pop in pops.items()}
        hf = {h: pop.hydrogenation_fractions(e)[0] for h, pop in pops.items()}
        zbar = on.charge_matrix(e) @ np.array(on.zs, float)
        print(f"\n{name}: IR power on/off, band ratios off -> on, f(N_H0) and <H lost> off -> on")
        for s, mol in enumerate(on.library):
            ratio = lambda h, a: F[h][s, b[a]] / F[h][s, b["11.2"]]          # noqa: E731
            lost = {h: mol.nh - float(np.arange(mol.nh + 1) @ np.asarray(chains[h][s])) for h in chains}
            print(f"  {mol.name:14s} <Z> {zbar[s]:5.2f}  P_IR x{F[True][s, b['all']] / F[False][s, b['all']]:.3f}  "
                  + "  ".join(f"{a}/11.2 {ratio(False, a):.3f}->{ratio(True, a):.3f}" for a in ("3.3", "6.2", "7.7"))
                  + f"  f(N_H0) {hf[False][s, 0]:.3f}->{hf[True][s, 0]:.3f}  <H lost> {lost[False]:.2f}->{lost[True]:.2f}")


if __name__ == "__main__":
    main()
