#!/usr/bin/env python3
"""One-at-a-time sensitivity of C54H18 hydrogenation and charge (pahchem M5).

Runs pahchem.baseline.sensitivity at the NGC 7023 NW edge and in the Orion
Bar atomic PDR. Each input in baseline.PERTURBATIONS is moved across its
literature range and the steady-state observables are compared with the
baseline:

- f_dehydrogenated (N_H < N_H0), shown as d log10;
- f_cation (Z > 0);
- f_anion;
- the H2 formation rate per PAH.

Rows are ranked by the largest |d log10 f_dehydro| over both environments;
the ranking orders the data requests (pahchem-data-requests.md).
Takes ~25 min.
"""
import argparse
from pathlib import Path

import numpy as np

from pycalima import _paths
from pycalima.pahchem.baseline import NGC7023_EDGE, ORION_BAR_ATOMIC, sensitivity
from pycalima.pahchem.molecule import PAHSpecies


def dlog(a, b):
    return np.log10(max(a, 1e-30) / max(b, 1e-30))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--states-dir", type=Path, default=_paths.get_model_data_dir() / "PAH_states")
    args = ap.parse_args()
    sp = PAHSpecies.from_states_file(args.states_dir / "C54H18_0.dat")
    anion = PAHSpecies.from_states_file(args.states_dir / "C54H18_-1.dat")

    results = {env.name: sensitivity("C54", sp, env, anion=anion) for env in (NGC7023_EDGE, ORION_BAR_ATOMIC)}
    names = [n for n in next(iter(results.values())) if n != "baseline"]
    for env, res in results.items():
        b = res["baseline"]
        print(f"\n{env}: baseline f_dehydro = {b['f_dehydrogenated']:.3g}, f_cation = {b['f_cation']:.3f}, "
              f"f_anion = {b['f_anion']:.3f}, H2 = {b['h2_per_pah_per_s']:.3g} /PAH/s")

    def score(name):
        s = 0.0
        for res in results.values():
            for r in res[name]:
                if r is not None:
                    s = max(s, abs(dlog(r["f_dehydrogenated"], res["baseline"]["f_dehydrogenated"])))
        return s

    print("\n{:36s} | {:>32s} | {:>32s}".format("perturbation (low, high)", *results))
    print("{:36s} | {:>32s} | {:>32s}".format("", *["dlog f_dehydro   d f_cation"] * 2))
    for name in sorted(names, key=score, reverse=True):
        cells = []
        for res in results.values():
            b = res["baseline"]
            parts = []
            for r in res[name]:
                parts.append("    --      --  " if r is None else
                             f"{dlog(r['f_dehydrogenated'], b['f_dehydrogenated']):+6.2f} {r['f_cation'] - b['f_cation']:+6.3f}")
            cells.append(" ".join(parts))
        print(f"{name:36s} | {cells[0]:>32s} | {cells[1]:>32s}", flush=True)


if __name__ == "__main__":
    main()
