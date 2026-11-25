#!/usr/bin/env python3
"""Start geometries and jobs of the Delta-SCF ladder of an odd-N_H0 parent (a pi radical).

The super-H threshold laws (``threshold_shifts.law_shift``) were fitted on the closed-shell parents C24H12, C54H18 and
C96H24. On an odd-N_H0 parent they give its super-hydrogenated states shifts of the wrong sign (pahchem audit of
2026-11-25, bug 4): C89H27 + 1 H gets IP -0.52 eV and EA +0.20 eV, while C89H28 is closed-shell. This ladder measures
those shifts on one odd parent.

- **States.** Those of ``make_ladder.states``, the duo-pair convention of the C24/C54/C96 ladders:
  - N_H0 - 1, the parent, N_H0 + 1 and N_H0 + 2;
  - with ``--edge-sites``, also ``make_transfer``'s N_H0 -/+ 1 on one H of each other edge class, for the site
    scatter.
- **Charges** -1, 0, +1, for IP(0) and the EA. ``law_shift`` takes the dications' shifts from IP1's.
- **Spin** from the electron count. Odd counts are doublets, even ones singlets. Triplets are added where the lowest
  multiplicity is not known a priori:
  - the parent's anion and cation, closed shells of a pi radical;
  - the neutral N_H0 - 1, a sigma + pi pair.
- **Single points.** 6-31+G* for Z = -1 and 0, the EA pairs, as in the other ladders.

Usage: make_odd_parent.py OUT_DIR [--uid 3226] [--name C31] [--edge-sites] [--xml PATH]

It writes OUT_DIR/<name>/<state>/start.xyz and OUT_DIR/jobs.txt, in ``make_slurm.py``'s format (pass
``--per-node <name>=... --layout <name>=...``). The default parent, C31H15 (UID 3226), is small and odd. PAHdb computed
it at every charge from -1 to +2, and its ladder step is within 6 % of ``population.LADDER_STEP_FIT``, so it shows no
RB3LYP/UB3LYP mix. It has solo, duo and trio edges.
"""
import argparse
from pathlib import Path

import numpy as np

from make_ladder import read_geometries, states, write_xyz
from make_transfer import add_ch2, edge_sites

CHARGES = (-1, 0, 1)


def spins(n_elec, kind):
    """2S of the jobs of one (state, charge) of an odd-N_H0 parent (module docstring)."""
    if n_elec % 2:
        return (1,)
    return (0, 2) if kind in ("parent", "sigma") else (0,)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--uid", type=int, default=3226)
    ap.add_argument("--name", default="C31", help="the parent's directory, make_slurm.py's key")
    ap.add_argument("--edge-sites", action="store_true", help="also N_H0 -/+ 1 on the other edge classes")
    ap.add_argument("--xml", type=Path, default=None, help="PAHdb v4.00 XML (default: $CALIMA_PAHDB_DIR's)")
    args = ap.parse_args()
    if args.xml is None:
        from pycalima.pahchem import ames_suite
        args.xml = Path(ames_suite.xml_path("pahdb-theoretical-v4-00"))
    types, xyz = read_geometries(args.xml, [args.uid])[args.uid]
    xyz = xyz - xyz.mean(axis=0)
    nc, nh0 = int((types == 6).sum()), int((types == 1).sum())
    if nh0 % 2 == 0:
        raise ValueError(f"UID {args.uid}: C{nc}H{nh0} is a closed-shell parent; use make_ladder.py")
    ladder, _ = states(types, xyz)
    st = {f"H{nh0 - 1}": (*ladder[f"H{nh0 - 1}"], "sigma"), f"H{nh0}": (*ladder[f"H{nh0}"], "parent"),
          f"H{nh0 + 1}": (*ladder[f"H{nh0 + 1}"], "superh"), f"H{nh0 + 2}": (*ladder[f"H{nh0 + 2}"], "superh")}
    if args.edge_sites:
        sites, host, counts = edge_sites(types, xyz)
        print(f"C{nc}H{nh0} (UID {args.uid}): edge H by class {counts}; sites {sites}")
        for cls, h in sites.items():
            if cls == "duo":                           # the ladder's own states sit on a duo pair
                continue
            keep = np.setdiff1d(np.arange(types.size), [h])
            st[f"H{nh0 - 1}_{cls}"] = (types[keep], xyz[keep], "sigma")
            st[f"H{nh0 + 1}_{cls}"] = (*add_ch2(types, xyz, h, host[h]), "superh")
    jobs = []
    for name, (t, x, kind) in st.items():
        d = args.out_dir / args.name / name
        d.mkdir(parents=True, exist_ok=True)
        write_xyz(d / "start.xyz", t, x, f"C{nc}H{nh0} (UID {args.uid}) {name} start (make_odd_parent.py)")
        nh = int((t == 1).sum())
        for z in CHARGES:
            for two_s in spins(6 * nc + nh - z, kind):
                jobs.append(f"{args.name}/{name} {z} {two_s} {int(z in (-1, 0))}\n")
    (args.out_dir / "jobs.txt").write_text("".join(jobs))
    print(f"{len(jobs)} jobs in {args.out_dir / 'jobs.txt'}")


if __name__ == "__main__":
    main()
