#!/usr/bin/env python3
"""Past a barrier-limited top a step emits a proton, not a Coulomb explosion (pahchem audit A2; gas-coupling section 2.2e).

``impact.coulomb_limit`` is min(floor(0.2 N_C), the proton-emission barrier's limit): the barrier for N_C >~ 115. With
``Params.fission``, a step past such a top (``Population._routed``) lands back on it with an H less, and only the bare
skeleton's share f(N_H = 0) of it explodes (``Population.chain_overflow``); a molecule whose top is the cold limit, and every
molecule without fission, explodes on all of them. The flag-level test (``test_past_a_barrier_limited_top_...``) does not
build the large molecules; this check does, in the field where it matters:

- the ladder molecules C24H12, C150H30 and C216H36 (``check_charge_ladder``; C24's top is the cold limit, the others' the
  barrier's);
- X-rays only, G0-equivalent 1e4 (``check_dissociative_recombination``: F_nu ~ nu^-1 from 300 eV to 10 keV, n_H = 1e4,
  n_e = 1e-2, T = 1e3 K), where the large molecules keep their H. In hot gas (1e7 K) they are bare, so there is nothing
  to route (C130H28: f(N_H = 0) = 1.000, ``results/pahchem/audit_open_2026-11-06/a2_c130_hot_1e7K.log``).

Per molecule, with fission on and off: <Z>, the steps past the top [s^-1], the explosions, the bare share, and the
lifetime as the library holds it. Gates: on, explosions = bare share x steps for the routed molecules and = steps for the
others; off, = steps for all; and the routed molecules explode less than their steps whenever they keep any H.

Result (2026-11-06, every gate passes): with fission C150H30 and C216H36 take 4.3e-15 and 2.9e-16 steps past their
tops per second and keep their H (bare share ~1e-33), so they explode at ~1e-48 s^-1; without it they explode at 2.9e-11 and
5.4e-12 s^-1 (lifetimes 1.1e3 and 5.9e3 yr), at <Z> 7.65 and 9.43 against 6.83 and 8.85 with the protons. C24H12, whose top
(+4) is the cold limit, explodes on every step either way (lifetime ~1 yr).

Cost: the bases of every state of C150 and C216 (+26, +32) in the X-ray shape; 68 s and 1.7 GB peak with 2 workers here.
Kept in ``fit_pdrs4all_population.BASIS_CACHE``, so a rerun is fast.

    check_a2_routing.py [--workers 2]
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

NAMES = ("C24H12 (18)", "C150H30 (612)", "C216H36 (615)")
FIELD = "X-rays G0eq 1e+04"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    warnings.simplefilter("ignore")
    lib = [m for m in L.library() if m.name in NAMES]
    env = DR.envs()[FIELD]
    shape = P.field_shape(env)
    ok = True
    print(f"{FIELD}: per molecule, steps past the top, explosions [s^-1] and the bare share f(N_H = 0)")
    for on in (True, False):
        pop = P.Population(lib, L.WINDOWS, params=replace(Params(), fission=on), cache_dir=FP.BASIS_CACHE)
        pop.prepare(shape, workers=args.workers)
        pop.prepare_responses(shape, workers=args.workers)
        f = pop.charge_matrix(env)
        steps = (f * pop._past_top(env, pop._single_up(env))[0]).sum(axis=1)
        boom = pop.chain_overflow(env)
        bare = pop._bare_share(env)
        routed = pop._routed()
        d = pop.destruction_rates(env)
        zs = np.array(pop.zs, dtype=float)
        print(f"\n  fission {'on' if on else 'off'}:")
        for s, mol in enumerate(pop.library):
            want = bare[s] * steps[s] if routed[s] else steps[s]
            good = bool(np.isclose(boom[s], want, rtol=1e-12, atol=0.0))
            if routed[s] and bare[s] < 1.0 and steps[s] > 0.0:
                good = good and boom[s] < steps[s]
            ok = ok and good
            print(f"   {mol.name:14s} top {max(mol.charges):+d} routed {bool(routed[s])!s:5} <Z> {f[s] @ zs:6.2f}  "
                  f"steps {steps[s]:.3e}  explosions {boom[s]:.3e}  bare share {bare[s]:.3f}  "
                  f"lifetime {1.0 / d['events'][s] / 3.156e7:.2e} yr  {'PASS' if good else 'FAIL'}")
    print("\nall gates pass" if ok else "\nGATE FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
