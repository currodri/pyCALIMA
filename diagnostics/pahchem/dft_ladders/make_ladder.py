#!/usr/bin/env python3
"""Start geometries and the job list of the Delta-SCF IP/EA ladders (data-requests T1 item 3, computed here).

For each parent (PAHdb v4.00 neutral geometry: C24H12 UID 18, C54H18 UID 37, C96H24 UID 108) it writes the start
geometry of every hydrogenation state, and one job per (state, charge, spin) to ``jobs.txt``:

- **States.** The parent; N_H0-1 ... N_H0-4 dehydrogenated by removing duo pairs, Malloci+08's paired convention
  (N_H0-1: one H of the duo pair nearest the +x axis; -2: its partner; -3, -4: the same for the pair farthest from it);
  the bare skeleton; N_H0+1, N_H0+2 super-hydrogenated, the extra H on the carbons of the first pair (CH2 groups, one
  above the plane, one below: a local minimum, not searched).
- **Charges** -1, 0, +1, +2.
- **Spin.** Odd electron counts are doublets. Even counts are singlets, and also triplets for the dehydrogenated, bare
  and doubly charged states, whose lowest multiplicity is not known a priori.

The geometries are only start points: ``run_state.py`` optimises each (state, charge, spin). Sites and multiplicities
are one choice per state, labelled in the job names; a site search is not done.

Usage: make_ladder.py OUT_DIR [--xml PATH] [--parents C24 C54 C96]
"""
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

PARENTS = {"C24": 18, "C54": 37, "C96": 108}
CHARGES = (-1, 0, 1, 2)
CH_BOND, CC_BOND = 1.25, 1.75     # bond-search cut-offs [A]


def read_geometries(xml_path, uids):
    """{uid: (types, xyz [A])} of the PAHdb species ``uids``."""
    want, out = set(uids), {}
    for _, el in ET.iterparse(xml_path, events=("end",)):
        if el.tag.rsplit("}", 1)[-1] != "specie":
            continue
        uid = int(el.get("uid")) if el.get("uid") else None
        if uid in want:
            types, xyz = [], []
            for c in el:
                if c.tag.rsplit("}", 1)[-1] == "geometry":
                    for atom in c:
                        v = {a.tag.rsplit("}", 1)[-1]: a.text for a in atom}
                        types.append(int(v["type"]))
                        xyz.append((float(v["x"]), float(v["y"]), float(v["z"])))
            out[uid] = (np.array(types), np.array(xyz))
            if len(out) == len(want):
                break
        el.clear()
    return out


def duo_pairs(types, xyz):
    """[(h1, h2)]: the H atoms on adjacent carbons (duo pairs), with each H's carbon."""
    C, H = np.nonzero(types == 6)[0], np.nonzero(types == 1)[0]
    host = {h: C[np.argmin(np.linalg.norm(xyz[C] - xyz[h], axis=1))] for h in H}
    pairs = []
    for i, a in enumerate(H):
        for b in H[i + 1:]:
            if np.linalg.norm(xyz[host[a]] - xyz[host[b]]) < CC_BOND:
                pairs.append((a, b))
    return pairs, host


def states(types, xyz):
    """{name: (types, xyz)} of the hydrogenation states of one parent (module docstring)."""
    xyz = xyz - xyz.mean(axis=0)
    pairs, host = duo_pairs(types, xyz)
    if len(pairs) < 2:
        raise ValueError("need two duo pairs")
    mid = lambda p: 0.5 * (xyz[p[0]] + xyz[p[1]])                     # noqa: E731
    first = max(pairs, key=lambda p: mid(p)[0])                         # the pair nearest +x
    second = max(pairs, key=lambda p: np.linalg.norm(mid(p) - mid(first)))
    nh0 = int((types == 1).sum())
    out = {f"H{nh0}": (types, xyz)}
    removed = [first[0], first[1], second[0], second[1]]
    for k in range(1, 5):
        keep = np.setdiff1d(np.arange(types.size), removed[:k])
        out[f"H{nh0 - k}"] = (types[keep], xyz[keep])
    c_only = types == 6
    out["H0"] = (types[c_only], xyz[c_only])
    # super-H: an extra H on each carbon of the first pair, the CH2 group's two H at +-55 deg out of the plane (its normal
    # from the carbons: PAHdb's geometries are not all in the xy plane)
    normal = np.linalg.svd(xyz[types == 6] - xyz[types == 6].mean(axis=0))[2][-1]
    t, x = list(types), [row for row in xyz]
    for n_extra, (h, side) in enumerate(((first[0], +1.0), (first[1], -1.0)), start=1):
        c = host[h]
        radial = xyz[h] - xyz[c]
        radial /= np.linalg.norm(radial)
        ang = np.deg2rad(55.0)
        x[h] = xyz[c] + 1.10 * (np.cos(ang) * radial - side * np.sin(ang) * normal)
        x.append(xyz[c] + 1.10 * (np.cos(ang) * radial + side * np.sin(ang) * normal))
        t.append(1)
        out[f"H{nh0 + n_extra}"] = (np.array(t), np.array(x))
    return out, nh0


def spins(n_elec, nh, nh0, z):
    """2S of the jobs of one (state, charge): doublet if odd, plus quartet for a dehydrogenated z = +2 (a sigma radical
    with two pi holes: the doublets came out with <S^2> ~ 1.8); singlet, plus triplet for dehydrogenated, bare or z = +2."""
    if n_elec % 2:
        return (1, 3) if (nh < nh0 and z == 2) else (1,)
    return (0, 2) if (nh < nh0 or z == 2) else (0,)


def write_xyz(path, types, xyz, comment=""):
    sym = {1: "H", 6: "C"}
    with open(path, "w") as f:
        f.write(f"{len(types)}\n{comment}\n")
        for t, (a, b, c) in zip(types, xyz):
            f.write(f"{sym[int(t)]} {a:.6f} {b:.6f} {c:.6f}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--xml", type=Path, default=None, help="PAHdb v4.00 XML (default: $CALIMA_PAHDB_DIR's)")
    ap.add_argument("--parents", nargs="+", default=list(PARENTS))
    args = ap.parse_args()
    if args.xml is None:
        from pycalima.pahchem import ames_suite
        args.xml = Path(ames_suite.xml_path("pahdb-theoretical-v4-00"))
    geo = read_geometries(args.xml, [PARENTS[p] for p in args.parents])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for parent in args.parents:
        types, xyz = geo[PARENTS[parent]]
        st, nh0 = states(types, xyz)
        for name, (t, x) in st.items():
            nc, nh = int((t == 6).sum()), int((t == 1).sum())
            d = args.out_dir / parent / name
            d.mkdir(parents=True, exist_ok=True)
            write_xyz(d / "start.xyz", t, x, f"{parent} {name} start (make_ladder.py)")
            for z in CHARGES:
                for two_s in spins(6 * nc + nh - z, nh, nh0, z):
                    diffuse = z in (-1, 0)
                    jobs.append((nc, nh == 0, f"{parent}/{name}", z, two_s, diffuse))
    # smallest first; the bare skeletons last (hardest SCFs)
    jobs.sort(key=lambda j: (j[1], j[0]))
    with open(args.out_dir / "jobs.txt", "w") as f:
        for nc, _, rel, z, two_s, diffuse in jobs:
            f.write(f"{rel} {z} {two_s} {int(diffuse)}\n")
    print(f"{len(jobs)} jobs in {args.out_dir / 'jobs.txt'}")


if __name__ == "__main__":
    main()
