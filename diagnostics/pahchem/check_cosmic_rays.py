#!/usr/bin/env python3
"""Cosmic rays in the population (gas-coupling section 2.2e): per unit zeta, as RAMSES-RTZ gives it, and their effect.

``Params.cosmic_rays`` follows ``impact.passage_spectrum`` of ``cosmic_rays.standard_spectra`` (the Padovani+18 protons with
the PDG nuclei and primary electrons, attenuated through ``STANDARD_SPECTRUM``'s column, scaled to the primary rate per H
zeta_H of ``Environment.zeta_cr``, zeta(H2) = 2 zeta_H) in the charge chain and as deposits. For C24H12-C216H36
(``check_charge_ladder``):
1. the shapes: primary zeta(H2) of models L and H through N(H2) = 1e20-1e22, and how much of it the PAH passages miss;
2. per unit zeta_H, for Z = 0 and +1: ionising passages, the mean charges and heat they leave, passages beyond the heat nodes
   (E* > 320 eV or > 15 charges), for each shape: the per-zeta rates' dependence on the shape (``STANDARD_SPECTRUM``);
3. the population with zeta_H = 1e-17-1e-15 in a dark cloud (Draine G0 1e-3, no old stars, n_H 1e4, n_e 1e-4, T 10 K) and a
   translucent one (G0 1, n_H 1e3, n_e 0.1, T 50 K): <Z>, the IR the passages' heat adds, f(N_H0), Coulomb explosions and
   passages beyond (destruction rates [s^-1]; lifetimes 1 / rate).
Primary cosmic rays only: no secondary electrons, no cosmic-ray-induced UV. Nothing is fitted.

Result (2026-11-04; 4 cores, every charge's bases in the two cloud fields):
1. zeta(H2) of the shapes: L 7.8e-17, 3.0e-17, 1.7e-17 and H 1.4e-15, 6.0e-16, 2.3e-16 s^-1 through 1e20, 1e21, 1e22 cm^-2.
2. Per unit zeta_H, L through 1e21: ionising passages 172 (C24), 395 (C54), 713 (C96), 1121 (C150), 1624 (C216) for Z = 0
   (+1: 8-9 % fewer), <dz> 1.12-1.14, heat 1.8-15.7 keV; beyond 0.19-2.6. H gives 0.55-0.64x the ionisations,
   0.70-0.77x the heat and 2-5x the beyond; within a model the columns move them +-15-20 %.
3. Dark and translucent clouds, the L/H blend: <Z> moves <= 0.003, f(N_H0) not at all; the IR +0.1-0.2 % at zeta_H = 1e-16
   in the dark (G0 1e-3), none at G0 = 1. Destruction (explosions for C24, C54; beyond for N_C >= 96): lifetimes
   1.1-2.8e9 yr at zeta_H 1e-17 (L scaled), 0.64-1.6e8 yr at 1e-16 (w ~ 0.3), 0.61-1.5e7 yr at 1e-15 (H scaled).

    check_cosmic_rays.py [--workers 4] [--skip-shapes]
"""
import argparse
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import check_charge_ladder as L
import fit_pdrs4all_population as FP

from pycalima.pahchem import cosmic_rays as CR
from pycalima.pahchem import population as P
from pycalima.pahchem.baseline import Environment, Params

SHAPES = [(m, n) for m in ("L", "H") for n in (1e20, 1e21, 1e22)]
YR = 3.15576e7


def _job(job):
    """Per unit zeta_H of the shape (model, N(H2)): its passages over its own zeta_H = zeta(H2) / H2_PER_H."""
    mol, z, (model, n_h2) = job
    k_node, beyond, power = P.cr_passages(mol, z, Params(), model, n_h2)
    per = CR.H2_PER_H / CR.shape_zeta(model, n_h2)
    k_node, beyond, power = k_node * per, beyond * per, power * per
    ion = k_node[1:].sum()
    return (float(ion), float((np.arange(k_node.shape[0])[:, None] * k_node).sum() / max(ion, 1e-300)),
            float(k_node.sum()), power, beyond)


def shapes(workers):
    print("1. Primary zeta(H2) [s^-1] of the shapes (unscaled)")
    for m, n in SHAPES:
        print(f"   {m} N(H2) {n:.0e}: {CR.primary_ionisation_rate(CR.projectile_spectra(m, n)):.3e}")
    lib = [P.extend_ladder(m) for m in L.library()]
    jobs = [(mol, z, sh) for mol in lib for z in (0, 1) for sh in SHAPES]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = dict(zip([(j[0].name, j[1], j[2]) for j in jobs], ex.map(_job, jobs)))
    print("\n2. Per unit zeta_H [s^-1 per (s^-1)]: ionising passages, <dz> of those, all passages, heat [eV], beyond")
    for mol in lib:
        for z in (0, 1):
            print(f"   {mol.name} {z:+d}:")
            for sh in SHAPES:
                ion, dz, allp, heat, beyond = res[(mol.name, z, sh)]
                print(f"     {sh[0]} {sh[1]:.0e}: ion {ion:8.1f}  <dz> {dz:.3f}  passages {allp:8.1f}  heat {heat:9.1f}  "
                      f"beyond {beyond:.3e}")


def environments(workers):
    lib = L.library()
    pop = P.Population(lib, {"all": (1.0, 1e4)}, params=replace(Params(), cosmic_rays=True), cache_dir=FP.BASIS_CACHE)
    pop.prepare_cr(workers=workers)
    zs = np.array(pop.zs, float)
    clouds = {"dark": lambda zeta: Environment("dark", 1e-3, 1e4, 1e-4, 10.0, None, optical_isrf=0.0, zeta_cr=zeta),
              "translucent": lambda zeta: Environment("transl", 1.0, 1e3, 0.1, 50.0, None, optical_isrf=0.0, zeta_cr=zeta)}
    print("\n3. Populations: <Z>, IR with/without cosmic rays, f(N_H0), explosions, beyond [s^-1], lifetime [yr]")
    for name, mk in clouds.items():
        for zeta in (0.0, 1e-17, 1e-16, 1e-15):
            e = mk(zeta)
            shape = P.field_shape(e)
            pop.prepare(shape, workers=workers)
            pop.prepare_responses(shape, workers=workers)
            f = pop.charge_matrix(e)
            F = pop.band_fluxes(e)[0][:, 0]
            if zeta == 0.0:
                F0 = F
            h = pop.hydrogenation_fractions(e)[0][:, 0]
            boom, beyond = pop.chain_overflow(e), pop.cr_beyond(e)
            print(f"  {name} zeta_H {zeta:.0e}:")
            for s, mol in enumerate(pop.library):
                life = 1.0 / (boom[s] + beyond[s]) / YR if boom[s] + beyond[s] > 0 else np.inf
                print(f"    {mol.name:14s} <Z> {f[s] @ zs:+.4f}  IR x{F[s] / F0[s]:.4f}  f(N_H0) {h[s]:.4f}  "
                      f"explosions {boom[s]:.2e}  beyond {beyond[s]:.2e}  lifetime {life:.2e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-shapes", action="store_true")
    a = ap.parse_args()
    warnings.simplefilter("ignore")
    if not a.skip_shapes:
        shapes(a.workers)
    environments(a.workers)


if __name__ == "__main__":
    main()
