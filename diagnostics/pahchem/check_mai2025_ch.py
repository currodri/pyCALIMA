#!/usr/bin/env python3
"""The 3.3 um CH-stretch band against size in the Mai+25 machine-learning MD spectra (pahchem.mai2025).

For each pure-aromatic molecule of their 1704 (PAHdb v4.00 says no CH2 / CH3), the smoothed band's peak and
centroid at 50, 300 and 600 K; printed as medians in N_C bins. The runs are classical: 300 / 600 K carry the
vibrational energy per mode of a quantum molecule at ~750 / ~1250 K, and the excited CH stretches add their
self-anharmonicity, so the absolute shifts are upper bounds; the size trend at equal energy per mode is the
result.

    check_mai2025_ch.py      (needs the fetch dataset mai2025-mlmd and PAHdb v4.00)
"""
import numpy as np

from pycalima import _paths
from pycalima.pahchem import ames_suite, mai2025

BINS = ((9, 24), (24, 40), (40, 60), (60, 90), (90, 130), (130, 217))


def main():
    recs = {r["uid"]: r for r in ames_suite.xml_index(ames_suite.xml_path("pahdb-theoretical-v4-00"),
                                                       _paths.get_model_data_dir() / "pahchem_pahdb_index")}
    band = {}
    for T in mai2025.TEMPERATURES:
        for name, nc, nh, uid, nu, inten in mai2025.spectra(T):
            r = recs.get(uid)
            if r is None or r["nc"] != nc or r["n_ch2"] + r["n_ch3"] > 0:
                continue
            band.setdefault(name, {"nc": nc})[T] = mai2025.ch_band(nu, inten)
    rows = [v for v in band.values() if all(T in v for T in mai2025.TEMPERATURES)]
    print(f"{len(rows)} pure-aromatic molecules")
    print("  N_C bin    n | peak shift 50->300, 50->600 K | centroid shift 50->600 K [cm^-1]")
    for lo, hi in BINS:
        s = [v for v in rows if lo <= v["nc"] < hi]
        pk3 = np.median([v[300][0] - v[50][0] for v in s])
        pk6 = np.median([v[600][0] - v[50][0] for v in s])
        c6 = np.median([v[600][1] - v[50][1] for v in s])
        print(f"  {lo:3d}-{hi - 1:3d} {len(s):4d} | {pk3:+6.1f}, {pk6:+6.1f} | {c6:+6.1f}")


if __name__ == "__main__":
    main()
