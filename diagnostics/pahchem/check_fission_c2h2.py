#!/usr/bin/env python3
"""Proton emission by hot ions and C2H2 loss (gas-coupling section 2.2e): thresholds, and their effect off -> on.

``population.loss_extras`` adds to every cascade (the bases' loss tables, the deposits' single-event yields):
- proton emission from Z >= +1 (``Params.fission``; to Z - 1, one H less) over max(D, B): D = the H loss's E0 + IE(H) -
  IP(Z - 1), B = ``impact.fission_barrier`` (Holm+11 Fig. 7, B = 6.30 - 2.88 Z / sqrt(N_C) eV), with the H-loss transition
  state; it is a down-step of the charge chain, solved with the hydrogenation chain it depends on (``charge_matrix``);
- C2H2 loss for skeletons with catacondensed end rings (``Params.c2h2_loss``, Ling & Lifshitz 1998 anthracene+), a
  destruction of the molecule as the library holds it.
The ladder's top is min(floor(0.2 N_C), the barrier's limit 2.19 sqrt(N_C)) (``impact.coulomb_limit``).
For C24H12-C216H36 (``check_charge_ladder``) and three catacondensed v4.00 molecules:
1. per charge: E0 of proton emission against the H loss's 4.41 eV, and the top of the ladder;
2. in the fields of ``check_dissociative_recombination`` (PDRs, EUV, X-rays only), off -> on: <Z>, f(N_H0), <H lost>, the
   protons and C2H2 losses per molecule, and the fixed point's iterations.
Nothing is fitted.

Result (2026-11-04):
1. Proton emission falls below the H loss's 4.41 eV from +4 in C24 (3.95 eV), +5 in C54, +8 in C96, +12 in C216; the tops of
   C96-C216 are at 0.03-0.7 eV. Tops: C150 +26 (0.2 N_C: 30), C216 +32 (43).
2. PDRs and EUV: <Z> moves <= 0.01; intact highly charged molecules are stripped (C96 psi 1e5 f(N_H0) 0.967 -> 0.463, C150
   1e6 1.000 -> 0.292), the small ones are bare already. X-rays only, G0-equivalent 1e4: <Z> -0.16 (C24), -0.91 (C54),
   -1.18 (C96), -0.99 (C150), -0.74 (C216); f(N_H0) of C54, C96 0.773 -> 0.872, 0.820 -> 0.865 (less heat on capture).
   C2H2 loss of the catacondensed C22H14 isomers ~5e-8 s^-1 there (C 1s products), 1e-12-1e-10 s^-1 in PDRs (stripped).

    check_fission_c2h2.py [--workers 4]
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
import check_wd01_per_size as WS
import fit_pdrs4all_population as FP

from pycalima.pahchem import impact
from pycalima.pahchem import population as P
from pycalima.pahchem.baseline import Params


def library():
    """The ladder molecules and the three smallest v4.00 molecules of N_C 22-40 with catacondensed end rings."""
    lib = P.shift_thresholds(FP.library("v400"), WS.W.IP_SHIFTS)
    cata = sorted((m for m in lib if m.edges.get("quartet", 0) > 0 and 22 <= m.nc <= 40), key=lambda m: (m.nc, m.name))[:3]
    return L.library() + cata


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    warnings.simplefilter("ignore")
    lib = library()
    p = Params()
    print("1. Proton emission E0 [eV] per charge (the H loss's 4.41 eV even-H, 3.16 eV odd-H) and the top of the ladder")
    for m0 in lib:
        mol = P.extend_ladder(m0)
        e0 = {z: dict((c[0], c[1]) for c in P.loss_extras(mol, z, mol.nh, p)).get("Hp") for z in mol.charges if z >= 1}
        shown = "  ".join(f"{z:+d} {v:.2f}" for z, v in e0.items() if v is not None and (z <= 6 or z % 4 == 0 or z == max(e0)))
        print(f"   {mol.name:14s} top {max(mol.charges):+d} (0.2 N_C {int(0.2 * mol.nc)}, barrier "
              f"{int(np.ceil(impact.FISSION_BARRIER[0] / impact.FISSION_BARRIER[1] * np.sqrt(mol.nc))) - 1}); "
              f"quartet H {mol.edges.get('quartet', 0)}: {shown}")
    pops = {on: P.Population(lib, {"all": (1.0, 1e4)}, params=replace(p, fission=on, c2h2_loss=p.c2h2_loss if on else None),
                             cache_dir=FP.BASIS_CACHE) for on in (False, True)}
    zs = np.array(pops[True].zs, float)
    for name, e in DR.envs().items():
        shape = P.field_shape(e)
        for pop in pops.values():
            pop.prepare(shape, workers=args.workers)
            pop.prepare_responses(shape, workers=args.workers)
        f = {on: pop.charge_matrix(e) for on, pop in pops.items()}
        h = {on: pop.hydrogenation_fractions(e)[0][:, 0] for on, pop in pops.items()}
        ch = {on: pop._chain_distributions(e) for on, pop in pops.items()}
        lr = pops[True].loss_rates(e)
        print(f"\n2. {name}: <Z>, f(N_H0), <H lost> off -> on; protons, C2H2 losses [s^-1 per molecule]")
        for s, mol in enumerate(pops[True].library):
            lost = {on: mol.nh - float(np.arange(mol.nh + 1) @ np.asarray(ch[on][s])) for on in ch}
            print(f"   {mol.name:14s} <Z> {f[False][s] @ zs:6.2f} -> {f[True][s] @ zs:6.2f}  f(N_H0) {h[False][s]:.3f} -> "
                  f"{h[True][s]:.3f}  <H lost> {lost[False]:5.2f} -> {lost[True]:5.2f}  protons {lr['Hp'][s]:.2e}  "
                  f"C2H2 {lr['C2H2'][s]:.2e}")


if __name__ == "__main__":
    main()
