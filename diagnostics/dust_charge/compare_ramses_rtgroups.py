#!/usr/bin/env python3
"""Compare RAMSES/CALIMA 'WDB06rt' grain charging with the pyCALIMA reference solver.

RAMSES, with dust_rtgroups_debug=.true. (or dust_rtgroups_verify=.true., which
also stops the run after dust_rtgroups_debug_max calls), writes
dust_rtgroups_debug_XXXXX.dat (format 2): one line per (call, dust bin) with
the inputs (T, n_e, H+/He+/He++ densities and masses, c_red, G0_bg, N_g, group
energies, the cold-start guess), the warm state the call started from (the
previous Z* of this bin in this cell, its discrete window, the recombination
fit reference) and the outputs (mean charge, width, PE heating, recombination
and autoionisation cooling, Z*, their d/d ln T, the grain-assisted
recombination coefficients of the RECOMB_IONS and whether they were fitted).

Each full solve (path 0) is re-solved with GroupChargingTable.solve
(kernels='pchip') from the same warm state, so every branch is the same and
the differences should be round-off. Calls served from the d/d ln T of the
bin's last full solve (path 2, the cooling solver's T(1 +- 1e-5) calls) are
checked against the Python derivatives of that solve. Lines from the
uniform-G0 table (path 1: no local photons) are compared with a direct solve,
so their differences are the table's interpolation error.

    python compare_ramses_rtgroups.py RUN_DIR [RUN_DIR ...] --tables TABLES_DIR [--stride N]

With ``--original-tables DIR`` it also shows, for the last dumped state of
each bin, the mean charge of the original (log gamma, log T) tables there
(charging_model='RM2026': G0 from the groups with 5.6 < egy < 13.6 eV, as
rtz_cooling_module's advected_G0).
"""
import argparse
from pathlib import Path

import numpy as np

from pycalima.models.dust_charge import rt_group_charging as rg

KEYS = ("Zmean", "Zsigma", "Gamma", "Lambda", "Lambda_auto", "Zstar",
        "dZmean_dlnT", "dZsigma_dlnT", "dGamma_dlnT", "dLambda_dlnT", "alpha")
DLNT = 1e-5          # dust_interface RTG_DLNT


def read_dump(path):
    """Calls of one dump file, in order (warm-state resets 'R' are not needed:
    each line carries the warm state it used)."""
    rows = []
    for line in open(path):
        if line.startswith("#") or line.startswith("R"):
            continue
        p = line.split()
        it = iter(p)
        nxt = lambda: next(it)                                           # noqa: E731
        r = dict(icell=int(nxt()), bin=int(nxt()), path=int(nxt()))
        for k in ("T", "ne", "n_Hp", "n_Hep", "n_Hepp", "m_H", "m_He", "c_red", "G0_bg"):
            r[k] = float(nxt())
        nG = int(nxt())
        r["Np"] = np.array([float(nxt()) for _ in range(nG)])
        r["egy"] = np.array([float(nxt()) for _ in range(nG)])
        zg = float(nxt())
        r["Z_guess"] = None if zg <= -1e299 else zg
        w = dict(valid=nxt() == "T", discrete=nxt() == "T", Zstar=float(nxt()), wlo=float(nxt()), whi=float(nxt()))
        nx = int(nxt())
        w["x"] = np.array([float(nxt()) for _ in range(nx)])
        w["alpha"] = np.array([float(nxt()) for _ in range(len(rg.RECOMB_IONS))])
        r["warm"] = w
        out = {k: float(nxt()) for k in ("Zmean", "Zsigma", "Gamma", "Lambda", "Lambda_auto", "Zstar")}
        out["discrete"] = nxt() == "T"
        for k in ("dZmean_dlnT", "dZsigma_dlnT", "dGamma_dlnT", "dLambda_dlnT"):
            out[k] = float(nxt())
        out["nfit"] = int(nxt())
        out["alpha"] = np.array([float(nxt()) for _ in range(len(rg.RECOMB_IONS))])
        r["out"] = out
        rows.append(r)
    return rows


def warm_dict(w):
    """The pyCALIMA warm result equivalent to a dumped RTGState."""
    if not w["valid"]:
        return None
    d = dict(Zstar=w["Zstar"], discrete=w["discrete"], nodes=np.array([w["wlo"], w["whi"]]))
    if w["x"].size:
        d["recomb_ref"] = (w["x"], w["alpha"])
    return d


def rel(a, b, floor=0.0):
    a, b = np.atleast_1d(a), np.atleast_1d(b)
    s = np.maximum(np.maximum(np.abs(a), np.abs(b)), floor)
    return float(np.max(np.where(s > 0, np.abs(a - b) / np.where(s > 0, s, 1.0), 0.0)))


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
    ap.add_argument("--stride", type=int, default=1, help="check every Nth full solve (each line is independent)")
    ap.add_argument("--worst", type=int, default=3, help="show the N worst full solves")
    ap.add_argument("--original-tables", default=None,
                    help="CALIMA table directory with dust_charge_Z_vs_T_DustBin_XX (original path)")
    args = ap.parse_args()

    tabs, gts = {}, {}

    def table(b, egy):
        if b not in tabs:
            tabs[b] = rg.read_bin_table(Path(args.tables) / f"dust_charging_rtgroups_DustBin_{b:02d}.dat")
        key = (b, tuple(egy))
        if key not in gts:
            gts[key] = rg.GroupChargingTable(tabs[b], egy=egy, kernels="pchip")
        return gts[key]

    for run in args.runs:
        dumps = sorted(Path(run).glob("dust_rtgroups_debug_*.dat"))
        if not dumps:
            print(f"{run}: no debug dump")
            continue
        worst = {k: 0.0 for k in KEYS}
        worst_dd = {k: 0.0 for k in KEYS[6:10]}             # derivatives of discrete P(Z) (analytic)
        worst_served = {k: 0.0 for k in KEYS[:4]}
        worst_tab = {k: 0.0 for k in KEYS[:4]}
        npath = np.zeros(3, int)
        nchk = ndisc = nfit = nmis_disc = nmis_fit = 0
        bad = []
        last = {}
        for d in dumps:
            rows = read_dump(d)
            k0 = 0
            for r in rows:
                npath[r["path"]] += 1
                gt = table(r["bin"], r["egy"])
                ions = [(r["n_Hp"], 1.0, r["m_H"]), (r["n_Hep"], 1.0, r["m_He"]), (r["n_Hepp"], 2.0, r["m_He"])]
                o = r["out"]
                if r["path"] == 0:
                    k0 += 1
                    if (k0 - 1) % args.stride:
                        last.pop((d, r["bin"]), None)
                        continue
                    warm = warm_dict(r["warm"])
                    py = gt.solve(r["Np"], r["c_red"], r["T"], r["ne"], ions, G0_bg=r["G0_bg"],
                                  Z_guess=r["Z_guess"], recomb="fit", warm=warm, dlnT=DLNT)
                    last[(d, r["bin"])] = py
                    nchk += 1
                    ndisc += int(o["discrete"])
                    nfit += o["nfit"]
                    nmis_disc += int(py["discrete"] != o["discrete"])
                    refit = (not py["discrete"]) and (warm is None or "recomb_ref" not in warm
                                                       or py["recomb_ref"] is not warm["recomb_ref"])
                    nmis_fit += int(refit != bool(o["nfit"]))
                    if py["discrete"] != o["discrete"]:
                        continue
                    diffs = {}
                    for k in KEYS:
                        floor = 1.0 if k in ("Zmean", "Zstar") else 0.0
                        if k == "dZmean_dlnT":
                            floor = 1e-6 * max(abs(o["Zmean"]), 1.0)
                        if k == "dZsigma_dlnT":
                            floor = 1e-6 * max(o["Zsigma"], 1.0)
                        if k in ("dGamma_dlnT", "dLambda_dlnT"):
                            floor = 1e-6 * abs(o[k[1:-5]])     # relative to Gamma / Lambda themselves
                        diffs[k] = rel(py["alpha_rec"] if k == "alpha" else py[k], o[k], floor)
                        worst[k] = max(worst[k], diffs[k])
                        if o["discrete"] and k in worst_dd:
                            worst_dd[k] = max(worst_dd[k], diffs[k])
                    bad.append((max(diffs.values()), r, diffs))
                elif r["path"] == 2:
                    py = last.get((d, r["bin"]))
                    if py is None:
                        continue
                    h = o["Zstar"]                          # RAMSES writes ln(T / T_last) there
                    pred = dict(Zmean=py["Zmean"] + h * py["dZmean_dlnT"], Zsigma=py["Zsigma"] + h * py["dZsigma_dlnT"],
                                Gamma=py["Gamma"] + h * py["dGamma_dlnT"],
                                Lambda=py["Lambda"] + py["Lambda_auto"] + h * py["dLambda_dlnT"])
                    for k in pred:
                        worst_served[k] = max(worst_served[k], rel(pred[k], o[k], 1.0 if k == "Zmean" else 0.0))
                else:
                    py = gt.solve(np.zeros_like(r["Np"]), 0.0, r["T"], r["ne"], [(r["ne"], 1.0, r["m_H"])],
                                  G0_bg=r["G0_bg"], recomb=None)
                    py["Lambda"] += py["Lambda_auto"]
                    for k in KEYS[:4]:
                        worst_tab[k] = max(worst_tab[k], rel(py[k], o[k], 1.0 if k == "Zmean" else 0.0))
        print(f"== {run}: {sum(npath)} calls (full {npath[0]}, uniform table {npath[1]}, from d/dlnT {npath[2]})")
        print(f"   {nchk} full solves re-solved ({ndisc} discrete, {nfit} recombination fits): "
              f"discrete-flag mismatches {nmis_disc}, fit/reuse mismatches {nmis_fit}")
        print("   max rel diff, full solves: " + " ".join(f"{k}={v:.1e}" for k, v in worst.items()))
        print("     of which d/dlnT of discrete P(Z) (analytic; wide: forward difference): "
              + " ".join(f"{k}={v:.1e}" for k, v in worst_dd.items()))
        print("   max rel diff, from d/dlnT: " + " ".join(f"{k}={v:.1e}" for k, v in worst_served.items()))
        if npath[1]:
            print("   max rel diff, uniform table vs direct solve (interpolation): "
                  + " ".join(f"{k}={v:.1e}" for k, v in worst_tab.items()))
        for m, r, diffs in sorted(bad, key=lambda t: -t[0])[:args.worst]:
            k = max(diffs, key=diffs.get)
            print(f"   worst: {m:.1e} in {k} | bin {r['bin']} T {r['T']:.4g} ne {r['ne']:.3g} warm {r['warm']['valid']}"
                  f"/{r['warm']['discrete']} discrete {r['out']['discrete']} <Z> {r['out']['Zmean']:.4f}")
        if args.original_tables:
            for b in sorted({r["bin"] for d in dumps for r in read_dump(d)[-8:]}):
                r = [r for d in dumps for r in read_dump(d) if r["bin"] == b][-1]
                sel = (r["egy"] > 5.6) & (r["egy"] < 13.6)
                G0 = float(np.sum(r["Np"][sel] * r["c_red"] * r["egy"][sel]) * 1.602176634e-12 / 1.6e-3)
                Zo = original_table_charge(Path(args.original_tables) / f"dust_charge_Z_vs_T_DustBin_{b:02d}", G0, r["T"], r["ne"])
                a = tabs[b]["a"]
                print(f"    last state bin {b}: T={r['T']:.0f} ne={r['ne']:.3g} G0={G0:.3g} | Z new {r['out']['Zmean']:9.2f} "
                      f"({r['out']['Zmean'] * rg.E2_EV_CM / a:7.2f} V)  Z original tables {Zo:9.2f} ({Zo * rg.E2_EV_CM / a:7.2f} V)")


if __name__ == "__main__":
    main()
