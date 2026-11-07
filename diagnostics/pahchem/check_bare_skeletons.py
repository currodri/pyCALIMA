#!/usr/bin/env python3
"""The bare skeleton (N_H = 0) and the destruction of PAHs (gas-coupling section 2.2f): calibration and lifetimes.

1. The cooling calibration ``population.BARE_IR_LAW``: for every PAHdb v4.00 neutral parent whose bare skeleton PAHdb
   computed (same carbon framework, ``carbon_graph``), the thermal IR power at 500, 800 and 1500 K of that skeleton over the
   parent's ``stripped(0)``, and the fit ln R(800 K) = ln a + b ln N_C.
2. ``Population.destruction_rates`` for C24H12-C216H36 (``check_charge_ladder``; PAHdb skeletons for C24, C54, C96, the
   calibrated ones for C150, C216) in the fields of ``check_dissociative_recombination``: each channel (Léger+89's C, C2, C3
   loss and Berné+15's folding of the bare skeleton, weighted by its share f(N_H = 0) of the chain; C2H2 loss; Coulomb
   explosions; cosmic-ray passages beyond the heat nodes), the lifetime 1 / events and the carbon lost.
Nothing is fitted but the calibration.

Result (2026-11-05):
1. 166 pairs (9 superhydrogenated skipped), N_C 14-96: R(800 K) ~12 at N_C 14-20, 8-11 at 20-35, 5.5 (C54), 3.4 (C66), 4.4
   (C96); fit 54.68 N_C^-0.522, rms x1.98 (BARE_IR_LAW).
2. Lifetimes: PDRs (Kurucz 40 kK) bare C24 22 yr at G0 1e3, 1.2 yr at 1e4, 11 days at 1e5; C54 (10 % bare) 1.9e3 yr at 1e5;
   N_C >= 96 hydrogenated, intact. EUV: C54 220, 6.9, 0.04 yr at psi 1e4, 1e5, 1e6; stripped C96 folds in ~3.5 days at
   1e6 (230, 10, 0.07 yr and ~5 days before the captured electrons' kinetic energy was deposited, pahchem audit M2).
   X-rays only (G0-equivalent 1e4): hydrogenated, Coulomb explosions from C 1s multi-ionisation, C24 1.3 yr, C54 41 yr,
   C96 1.2e5 yr (1.2, 20, 7e3 yr before the outgoing electrons' collisions were fixed, audit C1 and M1); C150 and C216
   emit protons past their barrier-limited tops instead (audit A2): > 1e30 yr (C216 9e5 yr before).
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import check_charge_ladder as L
import check_dissociative_recombination as DR
import fit_pdrs4all_population as FP

from pycalima import _paths
from pycalima.pahchem import ames_suite
from pycalima.pahchem import population as P

HC_K = 1.4387769                                  # c2 = hc/k [cm K]
YR = 3.15576e7


def thermal_power(sp, T):
    """sum_i A_i nu_i / (exp(hc nu_i / kT) - 1): the thermal IR power of a species' modes at T, up to constants."""
    f, a = np.asarray(sp.freqs_cm, float), np.asarray(sp.einstein_a, float)
    return float(np.sum(a * f / np.expm1(HC_K * f / T)))


def calibration():
    recs = ames_suite.xml_index(ames_suite.xml_path("pahdb-theoretical-v4-00"), _paths.get_model_data_dir() / "pahchem_pahdb_index")
    bare = {}
    for r in recs:
        if r["pure_ch"] and r["nh"] == 0 and r.get("carbon_graph") and r["uid"] is not None and r["charge"] == 0:
            bare.setdefault(r["carbon_graph"], r)
    rows, skipped = [], 0
    for r in recs:
        if not (r["pure_ch"] and r["nh"] > 0 and r["charge"] == 0 and r.get("carbon_graph") in bare
                and r["n_ch2"] == 0 and r["n_ch3"] == 0):
            continue
        try:
            st = ames_suite.species_from_record(r, "parent").stripped(0)
        except ValueError:                        # superhydrogenated: too few aromatic C-H stretches
            skipped += 1
            continue
        b = ames_suite.species_from_record(bare[r["carbon_graph"]], "bare")
        rows.append((r["nc"], *(thermal_power(b, T) / thermal_power(st, T) for T in (500.0, 800.0, 1500.0))))
    a = np.array(rows)
    print(f"1. PAHdb bare skeleton / stripped(0), thermal IR power: {len(a)} pairs ({skipped} skipped), N_C "
          f"{a[:, 0].min():.0f}-{a[:, 0].max():.0f}")
    for lo, hi in ((14, 20), (20, 26), (26, 30), (30, 35), (50, 60), (60, 70), (90, 100)):
        k = (a[:, 0] >= lo) & (a[:, 0] < hi)
        if k.any():
            print(f"   N_C {lo}-{hi}: n {k.sum():3d}  500 K {np.median(a[k, 1]):5.2f}  800 K {np.median(a[k, 2]):5.2f} "
                  f"[{a[k, 2].min():.2f}-{a[k, 2].max():.2f}]  1500 K {np.median(a[k, 3]):5.2f}")
    c = np.polyfit(np.log(a[:, 0]), np.log(a[:, 2]), 1)
    res = np.log(a[:, 2]) - np.polyval(c, np.log(a[:, 0]))
    print(f"   fit R(800 K) = {np.exp(c[1]):.2f} N_C^{c[0]:.3f}, rms x{np.exp(res.std()):.2f}; population.BARE_IR_LAW = "
          f"{P.BARE_IR_LAW}")


def destruction(workers):
    pop = P.Population(L.library(), {"all": (1.0, 1e4)}, cache_dir=FP.BASIS_CACHE)
    print("\n2. Destruction per molecule [s^-1]: the bare skeleton's channels x f(N_H = 0), C2H2, explosions, CR beyond")
    for m in pop.library:
        print(f"   {m.name}: {'PAHdb bare skeleton' if m.bare else 'calibrated stripped(0)'}; channels "
              f"{[c[0] for c in P.bare_channels(m, pop.p)]}")
    for name, e in DR.envs().items():
        shape = P.field_shape(e)
        pop.prepare(shape, workers=workers)
        pop.prepare_responses(shape, workers=workers)
        pop.prepare_bare(shape, workers=workers)
        d = pop.destruction_rates(e)
        p0 = [float(c[0]) for c in pop._chain_distributions(e)]
        print(f"  {name}:")
        for s, m in enumerate(pop.library):
            life = 1.0 / d["events"][s] / YR if d["events"][s] > 0 else np.inf
            parts = "  ".join(f"{c} {d[c][s]:.1e}" for c in d if c not in ("events", "carbon") and d[c][s] > 0)
            print(f"    {m.name:14s} f(N_H=0) {p0[s]:.3f}  {parts or 'none'}  -> lifetime {life:.1e} yr, "
                  f"carbon {d['carbon'][s]:.1e} s^-1")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    warnings.simplefilter("ignore")
    calibration()
    destruction(a.workers)


if __name__ == "__main__":
    main()
