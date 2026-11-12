#!/usr/bin/env python3
"""The Delta-SCF IP/EA ladders from the ``run_state.py`` jsons: one row per (parent, N_H), thresholds in eV.

For each (state, charge) the lowest-energy spin among the jobs whose optimisation and SCF converged is kept, per basis
(a job out of steps is reported, not used). IP_z = E(z+1) - E(z) for z = 0, 1; EA = E(0) - E(-1), at 6-31G* and, where
run, 6-31+G* (single points at the 6-31G* geometries). Adiabatic: each charge at its own optimised geometry.

Also: dIP0 = IP_0(N_H) - IP_0(N_H0), the quantity pahchem's ``ip_shift_per_h`` rule sets to 0.1 eV x (N_H0 - N_H)
for dehydrogenated states (Params.ip_shift_per_h; Malloci+08), and the spin kept per charge (with <S^2>).

Usage: collect.py ROOT [--csv OUT]
"""
import argparse
import csv
import json
from pathlib import Path

HARTREE_EV = 27.211386245988
NH0 = {"C24": 12, "C54": 18, "C96": 24}


def best(root, key="e_tot"):
    """{(parent, nh, z): (rec, others)}: the converged job lowest in ``key`` per (state, charge) (the spin order can
    differ between the bases: C96H23-'s singlet and triplet swap at 6-31+G*); others = all its jobs."""
    jobs = {}
    for f in sorted(root.glob("C*/H*/q*_s*.json")):
        r = json.loads(f.read_text())
        parent, h = r["state"].split("/")
        jobs.setdefault((parent, int(h[1:]), r["charge"]), []).append(r)
    out = {}
    for k, rs in jobs.items():
        ok = [r for r in rs if r.get("opt_converged") and r.get("scf_converged") and r.get(key) is not None
              and (key == "e_tot" or r.get("scf_converged_631pgs"))]
        out[k] = (min(ok, key=lambda r: r[key]) if ok else None, rs)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", type=Path)
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()
    b, bd = best(args.root), best(args.root, "e_tot_631pgs")
    rows = []
    for parent in sorted({k[0] for k in b}, key=lambda p: int(p[1:])):
        nh0 = NH0[parent]
        for nh in sorted({k[1] for k in b if k[0] == parent}):
            e = {z: b[(parent, nh, z)][0] for z in (-1, 0, 1, 2) if (parent, nh, z) in b}
            ed = {z: bd[(parent, nh, z)][0] for z in (-1, 0, 1, 2) if (parent, nh, z) in bd}

            def gap(hi, lo, key="e_tot"):
                a, c = (e if key == "e_tot" else ed).get(hi), (e if key == "e_tot" else ed).get(lo)
                if a is None or c is None or a.get(key) is None or c.get(key) is None:
                    return None
                return (a[key] - c[key]) * HARTREE_EV

            row = {"parent": parent, "nh": nh, "dnh": nh - nh0, "ip0": gap(1, 0), "ip1": gap(2, 1),
                   "ea": gap(0, -1), "ea_631pgs": gap(0, -1, "e_tot_631pgs")}
            for z in (-1, 0, 1, 2):
                r = e.get(z)
                row[f"spin_{z:+d}"] = "" if r is None else f"{r['two_s']}({r['s2']:.2f})"
                jobs = b.get((parent, nh, z), (None, []))[1]
                bad = [f"s{j['two_s']}" for j in jobs if not j.get("opt_converged")]
                row[f"unconv_{z:+d}"] = " ".join(bad)
            rows.append(row)
    ref = {(r["parent"]): r for r in rows if r["dnh"] == 0}
    for r in rows:
        p = ref.get(r["parent"])
        r["dip0"] = None if (p is None or p["ip0"] is None or r["ip0"] is None) else r["ip0"] - p["ip0"]
        r["dip0_rule"] = 0.1 * max(-r["dnh"], 0) if r["nh"] > 0 else None
    fmt = lambda x: "  --  " if x is None else f"{x:6.3f}"
    print(f"{'state':10s} {'IP0':>6s} {'IP1':>6s} {'EA':>6s} {'EA+':>6s} {'dIP0':>6s} {'rule':>6s}  spins(-1,0,+1,+2)  unconverged")
    for r in rows:
        unc = "; ".join(f"{z}:{r[f'unconv_{z}']}" for z in ("-1", "+0", "+1", "+2") if r[f"unconv_{z}"])
        spins = " ".join(r[f"spin_{z}"] or "--" for z in ("-1", "+0", "+1", "+2"))
        print(f"{r['parent']}H{r['nh']:<5d} {fmt(r['ip0'])} {fmt(r['ip1'])} {fmt(r['ea'])} {fmt(r['ea_631pgs'])} "
              f"{fmt(r['dip0'])} {fmt(r['dip0_rule'])}  {spins}  {unc}")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
