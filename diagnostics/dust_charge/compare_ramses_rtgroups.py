#!/usr/bin/env python3
"""Compare RAMSES/CALIMA 'WDB06rt' grain charging with the pyCALIMA reference solver.

RAMSES, with dust_rtgroups_debug=.true., writes dust_rtgroups_debug_XXXXX.dat:
one line per (call, dust bin) with the inputs it used (T, n_e, H+/He+/He++
densities and masses, c_red, N_g, group energies) and its outputs (mean charge,
width, PE heating, recombination and autoionisation cooling). This script
re-solves every sampled line with rt_group_charging.GroupChargingTable on the
same table files and reports the relative differences, which should be at
round-off level.

    python compare_ramses_rtgroups.py RUN_DIR [RUN_DIR ...] --tables TABLES_DIR

With ``--original-tables DIR`` it also shows, for the last dumped state of
each run, the mean charge the original (log gamma, log T) tables give there
(charging_model='RM2026': G0 from the groups with 5.6 < egy < 13.6 eV, as
rtz_cooling_module's advected_G0), i.e. how much the new path changes it.
"""
import argparse
from pathlib import Path

import numpy as np

from pycalima.models.dust_charge import rt_group_charging as rg

KEYS = ("Zmean", "Zsigma", "Gamma", "Lambda", "Lambda_auto")


def read_dump(path):
    rows = []
    for line in open(path):
        if line.startswith("#"):
            continue
        p = line.split()
        nG = int(p[9])
        v = np.array(p[1:9], dtype=float)
        rows.append(dict(bin=int(p[0]), T=v[0], ne=v[1], n_Hp=v[2], n_Hep=v[3], n_Hepp=v[4],
                         m_H=v[5], m_He=v[6], c_red=v[7],
                         Np=np.array(p[10:10 + nG], dtype=float), egy=np.array(p[10 + nG:10 + 2 * nG], dtype=float),
                         out=dict(zip(KEYS + ("Zstar",), np.array(p[10 + 2 * nG:16 + 2 * nG], dtype=float))),
                         discrete=p[16 + 2 * nG] == "T"))
    return rows


def original_table_charge(path, G0, T, ne):
    """Bilinear (log gamma, log T) lookup of dust_charge_Z_vs_T_DustBin_XX, as
    RAMSES compute_mean_dust_charge (clamped at the edges)."""
    with open(path) as fh:
        rows = [ln for ln in fh if not ln.startswith("#")]
    ng, nT = (int(x) for x in rows[0].split())
    lT = np.array(rows[1].split(), dtype=float)
    lg = np.array(rows[2].split(), dtype=float)
    Z = np.array([r.split() for r in rows[3:3 + ng]], dtype=float)
    x = np.log10(max(G0, 1e-6) * np.sqrt(T) / max(ne, 1e-20))
    y = np.log10(T)
    fx = np.clip((x - lg[0]) / (lg[1] - lg[0]), 0, ng - 1)
    fy = np.clip((y - lT[0]) / (lT[1] - lT[0]), 0, nT - 1)
    i, j = min(int(fx), ng - 2), min(int(fy), nT - 2)
    u, v = fx - i, fy - j
    return ((1 - u) * (1 - v) * Z[i, j] + u * (1 - v) * Z[i + 1, j]
            + (1 - u) * v * Z[i, j + 1] + u * v * Z[i + 1, j + 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", help="run directories holding dust_rtgroups_debug_*.dat")
    ap.add_argument("--tables", required=True, help="directory with dust_charging_rtgroups_DustBin_XX.dat")
    ap.add_argument("--stride", type=int, default=20, help="compare every Nth dumped line")
    ap.add_argument("--original-tables", default=None,
                    help="CALIMA table directory with dust_charge_Z_vs_T_DustBin_XX (original path)")
    args = ap.parse_args()

    tables = {}
    worst_all = {k: 0.0 for k in KEYS}
    for run in args.runs:
        dumps = sorted(Path(run).glob("dust_rtgroups_debug_*.dat"))
        if not dumps:
            print(f"{run}: no debug dump")
            continue
        all_rows = [r for d in dumps for r in read_dump(d)]
        rows = all_rows[::args.stride]
        worst = {k: 0.0 for k in KEYS}
        n_disc_mismatch = 0
        for r in rows:
            b = r["bin"]
            if b not in tables:
                tables[b] = rg.read_bin_table(Path(args.tables) / f"dust_charging_rtgroups_DustBin_{b:02d}.dat")
            gt = rg.GroupChargingTable(tables[b], egy=r["egy"])
            ions = [(r["n_Hp"], 1.0, r["m_H"]), (r["n_Hep"], 1.0, r["m_He"]), (r["n_Hepp"], 2.0, r["m_He"])]
            py = gt.solve(r["Np"], r["c_red"], r["T"], r["ne"], ions)
            n_disc_mismatch += int(py["discrete"] != r["discrete"])
            for k in KEYS:
                a, f = py[k], r["out"][k]
                scale = max(abs(a), abs(f))
                if k == "Zmean":
                    scale = max(scale, 1.0)
                d = abs(a - f) / scale if scale > 0 else 0.0
                worst[k] = max(worst[k], d)
        for k in KEYS:
            worst_all[k] = max(worst_all[k], worst[k])
        print(f"{Path(run).name:22s} {len(rows):5d} calls | max rel diff " +
              " ".join(f"{k}={worst[k]:.1e}" for k in KEYS) + f" | discrete-flag mismatches {n_disc_mismatch}")
        if args.original_tables:
            last = {}
            for r in all_rows:
                last[r["bin"]] = r
            for b, r in sorted(last.items()):
                sel = (r["egy"] > 5.6) & (r["egy"] < 13.6)
                G0 = float(np.sum(r["Np"][sel] * r["c_red"] * r["egy"][sel]) * 1.602176634e-12 / 1.6e-3)
                Zo = original_table_charge(Path(args.original_tables) / f"dust_charge_Z_vs_T_DustBin_{b:02d}", G0, r["T"], r["ne"])
                a = tables[b]["a"] if b in tables else rg.read_bin_table(
                    Path(args.tables) / f"dust_charging_rtgroups_DustBin_{b:02d}.dat")["a"]
                print(f"    last state bin {b}: T={r['T']:.0f} ne={r['ne']:.3g} G0={G0:.3g} | Z new {r['out']['Zmean']:9.2f} "
                      f"({r['out']['Zmean'] * rg.E2_EV_CM / a:7.2f} V)  Z original tables {Zo:9.2f} ({Zo * rg.E2_EV_CM / a:7.2f} V)")
    print("overall max rel diff: " + " ".join(f"{k}={worst_all[k]:.1e}" for k in KEYS))


if __name__ == "__main__":
    main()
