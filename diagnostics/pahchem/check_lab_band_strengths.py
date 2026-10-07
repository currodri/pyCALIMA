#!/usr/bin/env python3
"""DFT band strengths (PAHdb v4.00, 6-31G*) against the matrix-isolation library (PAHdb experimental v3.10).

Each laboratory species is matched to the theoretical species with the same
bond graph and charge (``ames_pahdb.bond_graph_hash``). Integrated intensities
(km/mol) are summed in four regions (cm^-1): CH stretch 2950-3150, CC stretch
1500-1650, CH in-plane / CC 1100-1500, CH out-of-plane 650-950.

Lab absolute cation intensities are often normalised with theory (the ion
column is not measured), so the test is the ratios *within* one spectrum.
- CC / oop and stretch / oop, lab against DFT, per charge.
- The neutral-to-cation change of those ratios, which is what makes cations
  bright at 6-9 um in the fits.
Ar-matrix shifts (a few cm^-1) do not move intensity between regions.
"""
from collections import defaultdict

import numpy as np

from pycalima import _paths
from pycalima.pahchem import ames_pahdb

REGIONS = {"stretch": (2950.0, 3150.0), "CC": (1500.0, 1650.0), "ip": (1100.0, 1500.0), "oop": (650.0, 950.0)}


def regions(freqs, inten):
    f, i = np.asarray(freqs), np.asarray(inten)
    return {k: float(i[(f >= lo) & (f < hi)].sum()) for k, (lo, hi) in REGIONS.items()}


def main():
    lab = ames_pahdb.read_xml(ames_pahdb.xml_path("pahdb-experimental-v3-10"))
    theory = ames_pahdb.xml_index(ames_pahdb.xml_path("pahdb-theoretical-v4-00"),
                                  _paths.get_model_data_dir() / "pahchem_pahdb_index")
    by = defaultdict(list)
    for r in theory:
        if r.get("graph"):
            by[(r["graph"], r["charge"])].append(r)
    rows = []
    for r in lab:
        if not r["pure_ch"] or r["freqs_cm"].size == 0:
            continue
        th = by.get((r["graph"], r["charge"]))
        if not th:
            print(f"  no v4.00 match: {r['formula']} Z={r['charge']} (lab UID {r['uid']})")
            continue
        t = min(th, key=lambda x: x["uid"])
        a, b = regions(r["freqs_cm"], r["intensity"]), regions(t["freqs_cm"], t["intensity"])
        rows.append((r, t, a, b))
    print(f"\n{len(rows)} lab species matched to v4.00\n")
    print(f"{'formula':10s} {'Z':>2s} {'nc':>3s} | lab CC/oop  DFT CC/oop | lab str/oop  DFT str/oop | lab ip/oop DFT ip/oop")
    stats = defaultdict(list)
    for r, t, a, b in sorted(rows, key=lambda x: (x[0]["charge"], x[0]["nc"])):
        if a["oop"] <= 0 or b["oop"] <= 0:
            continue
        lab_r = {k: a[k] / a["oop"] for k in ("CC", "stretch", "ip")}
        dft_r = {k: b[k] / b["oop"] for k in ("CC", "stretch", "ip")}
        print(f"{r['formula']:10s} {r['charge']:+2d} {r['nc']:3d} | {lab_r['CC']:9.2f} {dft_r['CC']:10.2f} | "
              f"{lab_r['stretch']:10.2f} {dft_r['stretch']:11.2f} | {lab_r['ip']:9.2f} {dft_r['ip']:9.2f}")
        for k in ("CC", "stretch", "ip"):
            if lab_r[k] > 0 and dft_r[k] > 0:
                stats[(r["charge"], k)].append(np.log10(dft_r[k] / lab_r[k]))
    print("\nlog10(DFT / lab) of the ratio to the CH oop intensity: median [16-84 %] (n)")
    for (z, k), v in sorted(stats.items()):
        v = np.array(v)
        print(f"  Z={z:+d} {k:8s}/oop: {np.median(v):+.2f} [{np.percentile(v, 16):+.2f}, {np.percentile(v, 84):+.2f}] ({v.size})")
    # neutral-cation pairs: how much brighter the CC band gets on ionisation, lab vs DFT
    pairs = defaultdict(dict)
    for r, t, a, b in rows:
        pairs[r["graph"]][r["charge"]] = (a, b)
    contrast = []
    for g, d in pairs.items():
        if 0 in d and 1 in d and all(x["oop"] > 0 for x in (d[0][0], d[0][1], d[1][0], d[1][1])):
            lab_c = (d[1][0]["CC"] / d[1][0]["oop"]) / (d[0][0]["CC"] / d[0][0]["oop"])
            dft_c = (d[1][1]["CC"] / d[1][1]["oop"]) / (d[0][1]["CC"] / d[0][1]["oop"])
            contrast.append(np.log10(dft_c / lab_c))
    if contrast:
        c = np.array(contrast)
        print(f"\ncation/neutral contrast of CC/oop, log10(DFT / lab): median {np.median(c):+.2f} "
              f"[{np.percentile(c, 16):+.2f}, {np.percentile(c, 84):+.2f}] over {c.size} neutral-cation pairs")


if __name__ == "__main__":
    main()
