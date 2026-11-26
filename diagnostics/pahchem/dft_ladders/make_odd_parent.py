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

The first run (2026-11-25) showed that the extra H's site decides the state. An odd alternant radical's singly
occupied orbital is nonbonding (Hueckel) and lives on alternate carbons:
- On a carbon where it has density, H closes the shell. C31H15 + H at its trio site: IP0 +0.01, EA -0.26 eV.
- On one where it has a node, H leaves a pi diradical. C31H15's duo and solo sites, whose restricted-singlet energies
  then fake IP0 -0.88, EA +0.5-0.7 eV.

``--radical-site`` puts N_H0 + 1 (and N_H0 + 2, with the next H on the neighbouring CH carbon of most density) on the
CH carbon of most density, where H adds. ``--triplet-checks STATE ...`` adds the neutral's triplet of those states, to
show which are diradicals.

The follow-up (2026-11-25) confirmed it: the radical site's isomer is the lowest +1 by 0.6-1.3 eV, and the odd parent
takes the even ladder one H later (``threshold_shifts.ODD_LAWS``). To test that law past +2 H and at Z = +2:
- ``--sequence`` adds the path that addition takes after the radical site. N_H0 + 2 ``_seq`` puts the next H on one
  carbon of the ladder's duo pair (the first one not holding the radical site), N_H0 + 3 ``_rad`` on its partner, and
  N_H0 + 4 ``_rad`` on one carbon of the second pair.
- ``--charges -1 0 1 2`` adds the dications for IP1. As in ``make_ladder``, an even count at Z = +2 also gets its
  triplet, and a dehydrogenated odd one its quartet.
- ``--states NAME ...`` keeps only those states' jobs (and geometries), so a parent already run gets only its new ones;
  ``run_state.py`` also skips a job whose json shows a converged optimisation.

Usage: make_odd_parent.py OUT_DIR [--uid 3226] [--name C31] [--edge-sites] [--radical-site] [--sequence]
       [--charges -1 0 1] [--states H15 H16_rad ...] [--triplet-checks H16 H16_solo] [--jobs jobs.txt] [--xml PATH]

It writes OUT_DIR/<name>/<state>/start.xyz and OUT_DIR/jobs.txt, in ``make_slurm.py``'s format (pass
``--per-node <name>=... --layout <name>=...``). The default parent, C31H15 (UID 3226), is small and odd. PAHdb computed
it at every charge from -1 to +2, and its ladder step is within 6 % of ``population.LADDER_STEP_FIT``, so it shows no
RB3LYP/UB3LYP mix. It has solo, duo and trio edges.
"""
import argparse
from pathlib import Path

import numpy as np

from make_ladder import duo_pairs, read_geometries, states, write_xyz
from make_transfer import add_ch2, edge_sites

CHARGES = (-1, 0, 1)


def spins(n_elec, kind, z=0):
    """2S of the jobs of one (state, charge) of an odd-N_H0 parent (module docstring)."""
    if n_elec % 2:
        return (1, 3) if (z == 2 and kind == "sigma") else (1,)
    return (0, 2) if (kind in ("parent", "sigma", "radsuperh") or z == 2) else (0,)


def somo_density(types, xyz):
    """{carbon index: c^2}: the Hueckel singly occupied orbital of the carbon skeleton (nonbonding for an odd alternant
    radical), on each carbon."""
    C = np.nonzero(types == 6)[0]
    D = np.linalg.norm(xyz[C][:, None] - xyz[C][None], axis=2)
    w, v = np.linalg.eigh(-((D > 0.5) & (D < 1.75)).astype(float))
    somo = v[:, C.size // 2]
    return {int(c): float(somo[i] ** 2) for i, c in enumerate(C)}


def radical_sites(types, xyz):
    """(h1, c1, h2, c2): the CH carbon of most singly-occupied density (c1, its H h1), and the next one (c2, h2): its CH
    neighbour of most density, else the CH carbon of most density elsewhere. Past N_H0 + 1 (a closed shell) any site
    makes a radical, as an even parent's first extra H does."""
    C, H = np.nonzero(types == 6)[0], np.nonzero(types == 1)[0]
    host = {int(h): int(C[np.argmin(np.linalg.norm(xyz[C] - xyz[h], axis=1))]) for h in H}
    dens = somo_density(types, xyz)
    h1 = max(host, key=lambda h: dens[host[h]])
    near = [h for h in host if h != h1 and np.linalg.norm(xyz[host[h]] - xyz[host[h1]]) < 1.75]
    h2 = max(near or [h for h in host if h != h1], key=lambda h: dens[host[h]])
    return h1, host[h1], h2, host[h2]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--uid", type=int, default=3226)
    ap.add_argument("--name", default="C31", help="the parent's directory, make_slurm.py's key")
    ap.add_argument("--edge-sites", action="store_true", help="also N_H0 -/+ 1 on the other edge classes")
    ap.add_argument("--radical-site", action="store_true",
                    help="N_H0 + 1 and + 2 on the CH carbons of most singly-occupied density (H<n>_rad)")
    ap.add_argument("--sequence", action="store_true",
                    help="with --radical-site: N_H0 + 2 _seq, + 3 _rad and + 4 _rad along the ladder's duo pairs")
    ap.add_argument("--charges", nargs="+", type=int, default=list(CHARGES))
    ap.add_argument("--states", nargs="*", default=None, help="only these states (default: all)")
    ap.add_argument("--triplet-checks", nargs="*", default=[], help="states whose neutral also gets its triplet")
    ap.add_argument("--jobs", default="jobs.txt", help="the job file's name in OUT_DIR")
    ap.add_argument("--only-new", action="store_true", help="only the --radical-site and --triplet-checks jobs")
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
    if args.radical_site:
        h1, c1, h2, c2 = radical_sites(types, xyz)
        dens = somo_density(types, xyz)
        print(f"radical site: C{c1} (density {dens[c1]:.3f}); the second extra H on C{c2} ({dens[c2]:.3f})")
        t1, x1 = add_ch2(types, xyz, h1, c1)
        st[f"H{nh0 + 1}_rad"] = (t1, x1, "radsuperh")
        st[f"H{nh0 + 2}_rad"] = (*add_ch2(t1, x1, h2, c2), "superh")
        if args.sequence:                                  # along the ladder's duo pairs, after the radical site
            pairs, host = duo_pairs(types, xyz)
            pairs = [p for p in pairs if c1 not in (host[p[0]], host[p[1]])]
            mid = lambda p: 0.5 * (xyz[p[0]] + xyz[p[1]])                     # noqa: E731
            first = max(pairs, key=lambda p: mid(p)[0])
            second = max(pairs, key=lambda p: np.linalg.norm(mid(p) - mid(first)))
            print(f"sequence: C{c1}, then duo pair C{host[first[0]]}/C{host[first[1]]}, then C{host[second[0]]}")
            t2, x2 = add_ch2(t1, x1, first[0], host[first[0]])
            t3, x3 = add_ch2(t2, x2, first[1], host[first[1]])
            st[f"H{nh0 + 2}_seq"] = (t2, x2, "superh")
            st[f"H{nh0 + 3}_rad"] = (t3, x3, "radsuperh")
            st[f"H{nh0 + 4}_rad"] = (*add_ch2(t3, x3, second[0], host[second[0]]), "superh")
    if args.only_new:
        st = {k: v for k, v in st.items() if k.endswith("_rad")}
    if args.states is not None:
        missing = set(args.states) - set(st)
        if missing:
            raise ValueError(f"no such states: {sorted(missing)} (have {sorted(st)})")
        st = {k: v for k, v in st.items() if k in args.states}
    jobs = []
    for name in args.triplet_checks:                       # the neutral's triplet of a state already run
        jobs.append(f"{args.name}/{name} 0 2 1\n")
    for name, (t, x, kind) in st.items():
        d = args.out_dir / args.name / name
        d.mkdir(parents=True, exist_ok=True)
        write_xyz(d / "start.xyz", t, x, f"C{nc}H{nh0} (UID {args.uid}) {name} start (make_odd_parent.py)")
        nh = int((t == 1).sum())
        for z in args.charges:
            for two_s in spins(6 * nc + nh - z, kind, z):
                jobs.append(f"{args.name}/{name} {z} {two_s} {int(z in (-1, 0))}\n")
    (args.out_dir / args.jobs).write_text("".join(jobs))
    print(f"{len(jobs)} jobs in {args.out_dir / args.jobs}")


if __name__ == "__main__":
    main()
