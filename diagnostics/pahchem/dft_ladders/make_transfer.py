#!/usr/bin/env python3
"""Start geometries and jobs of the ladder transfer test: do the C24/C54/C96 shift laws hold for other shapes and sites?

The C24/C54/C96 ladders (``make_ladder.py``) are one family (compact, D6h, duo edges only). Here, for PAHdb v4.00
molecules of other shapes and edges, one site of each edge class (solo, duo, trio, quartet: the length of the run of
CH carbons the H sits on; the H at the end of the run) gets:

- ``H<N_H0-1>_<class>``: that H removed (a sigma radical);
- ``H<N_H0+1>_<class>``: an extra H on its carbon (a CH2 group, the two H at +-55 deg out of the plane).

With the parent, at charges -1, 0, +1 (no dications: their shifts were small): the parent's anion and cation are
doublets; the sigma radical's anion singlet and triplet (C96H23-'s were within 0.04 eV), its cation triplet (lowest at
every size so far); the CH2 radical's ions singlets. 6-31+G* single points for z = -1, 0 (the EAs).

Usage: make_transfer.py OUT_DIR [--uids 1969 2226 4 105] [--xml PATH]
Writes OUT_DIR/<formula>_<uid>/<state>/start.xyz and OUT_DIR/jobs.txt (largest first).
"""
import argparse
from pathlib import Path

import numpy as np

from make_ladder import CC_BOND, read_geometries, write_xyz

CLASSES = {1: "solo", 2: "duo", 3: "trio", 4: "quartet"}


def edge_sites(types, xyz):
    """{class: H index} (one H per edge class, at the end of its run of CH carbons) and {H: its carbon}."""
    C, H = np.nonzero(types == 6)[0], np.nonzero(types == 1)[0]
    host = {h: C[np.argmin(np.linalg.norm(xyz[C] - xyz[h], axis=1))] for h in H}
    ch = {host[h]: h for h in H}
    near = lambda a, b: np.linalg.norm(xyz[a] - xyz[b]) < CC_BOND                  # noqa: E731
    seen, runs = set(), []
    for c in ch:                                       # runs of bonded CH carbons
        if c in seen:
            continue
        run, todo = [], [c]
        while todo:
            a = todo.pop()
            if a in seen:
                continue
            seen.add(a)
            run.append(a)
            todo += [b for b in ch if b not in seen and near(a, b)]
        runs.append(run)
    out = {}
    for run in sorted(runs, key=lambda r: min(r)):
        cls = CLASSES[len(run)]
        if cls in out:
            continue
        ends = [a for a in run if sum(near(a, b) for b in run if b != a) <= 1]      # a run's end carbons
        out[cls] = ch[min(ends)]
    counts = {}
    for run in runs:
        counts[CLASSES[len(run)]] = counts.get(CLASSES[len(run)], 0) + len(run)
    return out, host, counts


def add_ch2(types, xyz, h, c):
    """The geometry with an extra H on carbon c (H h's): the CH2 group's two H at +-55 deg out of the plane."""
    normal = np.linalg.svd(xyz[types == 6] - xyz[types == 6].mean(axis=0))[2][-1]
    radial = (xyz[h] - xyz[c]) / np.linalg.norm(xyz[h] - xyz[c])
    ang = np.deg2rad(55.0)
    x = xyz.copy()
    x[h] = xyz[c] + 1.10 * (np.cos(ang) * radial - np.sin(ang) * normal)
    x = np.vstack([x, xyz[c] + 1.10 * (np.cos(ang) * radial + np.sin(ang) * normal)])
    return np.append(types, 1), x


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--uids", nargs="+", type=int, default=[1969, 2226, 4, 105])
    ap.add_argument("--xml", type=Path, default=None, help="PAHdb v4.00 XML (default: $CALIMA_PAHDB_DIR's)")
    args = ap.parse_args()
    if args.xml is None:
        from pycalima.pahchem import ames_suite
        args.xml = Path(ames_suite.xml_path("pahdb-theoretical-v4-00"))
    geo = read_geometries(args.xml, args.uids)
    jobs = []
    for uid in args.uids:
        types, xyz = geo[uid]
        xyz = xyz - xyz.mean(axis=0)
        nc, nh0 = int((types == 6).sum()), int((types == 1).sum())
        parent = f"C{nc}H{nh0}_{uid}"
        sites, host, counts = edge_sites(types, xyz)
        print(f"{parent}: edge H by class {counts}; sites {sites}")
        st = {f"H{nh0}": (types, xyz, "parent")}
        for cls, h in sites.items():
            keep = np.setdiff1d(np.arange(types.size), [h])
            st[f"H{nh0 - 1}_{cls}"] = (types[keep], xyz[keep], "sigma")
            st[f"H{nh0 + 1}_{cls}"] = (*add_ch2(types, xyz, h, host[h]), "ch2")
        for name, (t, x, kind) in st.items():
            d = args.out_dir / parent / name
            d.mkdir(parents=True, exist_ok=True)
            write_xyz(d / "start.xyz", t, x, f"{parent} {name} start (make_transfer.py)")
            spins = {"parent": {-1: (1,), 0: (0,), 1: (1,)}, "sigma": {-1: (0, 2), 0: (1,), 1: (2,)},
                     "ch2": {-1: (0,), 0: (1,), 1: (0,)}}[kind]
            for z, ss in spins.items():
                for two_s in ss:
                    jobs.append((nc, f"{parent}/{name}", z, two_s, int(z in (-1, 0))))
    jobs.sort(key=lambda j: -j[0])
    with open(args.out_dir / "jobs.txt", "w") as f:
        for _, rel, z, two_s, diffuse in jobs:
            f.write(f"{rel} {z} {two_s} {diffuse}\n")
    print(f"{len(jobs)} jobs in {args.out_dir / 'jobs.txt'}")


if __name__ == "__main__":
    main()
