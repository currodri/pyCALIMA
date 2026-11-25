#!/usr/bin/env python3
"""The deep super-hydrogenated chain (Params.superh_chemistry = "deep", 2026-11-15) against its cap.

For the compact parents C24H12 (18), C54H18 (37), C96H24 (108) and C150H30 (612) of PAHdb v4.00, in the Draine field
at 100 K with n_e = 1.4e-4 n_H, over the environments where the +4 chain piles up at its top (shielded and dense gas,
and the GSD session's shielded Orion Bar layers, G0 3-10, n(H) 2-8e3), and two lit ones for reference. Each molecule
is solved with BaselineModel's super-H chemistry at a cap of 4 (the joint chain's default), and with the deep one at
caps of 4, 8, 12 and 24 (``--levels``). The deep chain is also run at the top cap with Murga+22's C2H2 loss of the
super-hydrogenated states (``superh_carbon_loss = "murga22"``, a labelled bracket).

It prints and writes <N_H>/N_H0, the super-H share, "at_cap" (the share at the ladder's top: where it is more than
~1 %, the cap, not the chemistry, sets the answer), the H2 released, the carbon loss and the photoelectric heating.

Usage: check_deep_superh.py [--levels 4 8 12 24] [--out results/pahchem/deep_superh/caps.csv]
"""
import argparse
import csv
from dataclasses import replace
from pathlib import Path

from pycalima import _paths
from pycalima.pahchem import population as P
from pycalima.pahchem.baseline import Environment, Params

PARENTS = ("C24H12 (18)", "C54H18 (37)", "C96H24 (108)", "C150H30 (612)")
ENVS = (("diffuse", 1.0, 30.0), ("pdr", 1e4, 1e4), ("orion_shielded", 5.0, 5e3), ("shielded", 1e-2, 1e2),
        ("dense", 1e-3, 1e4))


def variants(levels):
    base = replace(Params(), hydrogen_chain="joint")
    out = {"andrews16 K=4": replace(base, superh_levels=4)}
    for k in levels:
        out[f"deep K={k}"] = replace(base, superh_levels=k, superh_chemistry="deep")
    out[f"deep K={max(levels)} murga22"] = replace(base, superh_levels=max(levels), superh_chemistry="deep",
                                                  superh_carbon_loss="murga22")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--levels", type=int, nargs="+", default=[4, 8, 12, 24])
    ap.add_argument("--out", type=Path, default=_paths.get_results_dir() / "pahchem" / "deep_superh" / "caps.csv")
    args = ap.parse_args()
    lib = P.library_v400(cache_dir=_paths.get_model_data_dir() / "pahchem_pahdb_index")
    lib = [m for m in lib if m.name in PARENTS]
    cache = _paths.get_model_data_dir() / "pahchem_population_basis"
    rows = []
    for name, params in variants(args.levels).items():
        pop = P.Population(lib, {"3.3": (3.2, 3.36)}, params=params, cache_dir=cache)
        for label, g0, n in ENVS:
            env = Environment(label, g0, n, 1.4e-4 * n, 100.0, None)
            sm, d = pop.hydrogenation_summary(env), pop.destruction_rates(env)
            heat = pop.photoelectric_heating(env)[0]
            for s, mol in enumerate(pop.library):
                rows.append({"variant": name, "env": label, "g0": g0, "n_h": n, "molecule": mol.name,
                             "x": sm["x"][s], "superh": sm["superh"][s], "at_cap": sm["at_cap"][s], "h2": sm["h2"][s],
                             "carbon": d["carbon"][s], "heating": heat[s]})
                r = rows[-1]
                print(f"{name:22s} {label:15s} {mol.name:15s} x {r['x']:.3f} superh {r['superh']:.3f} "
                      f"at_cap {r['at_cap']:.2e} h2 {r['h2']:.2e} C {r['carbon']:.2e} heat {r['heating']:.2e}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
