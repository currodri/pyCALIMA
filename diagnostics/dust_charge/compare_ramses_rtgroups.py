#!/usr/bin/env python3
"""Compare RAMSES/CALIMA 'WDB06rt' grain charging with the pyCALIMA reference solver.

RAMSES, with dust_rtgroups_debug=.true. (or dust_rtgroups_verify=.true., which
also stops the run after dust_rtgroups_debug_max calls), writes
dust_rtgroups_debug_XXXXX.dat (format 5): one line per (call, dust bin) with
the inputs (T, n_e, H+/He+/He++ densities and masses, c_red, G0_bg, N_g, group
energies, the cold-start guess), the warm state the call started from (the
previous Z* of this bin in this cell, its discrete window and width, the recombination
fit reference: inputs, alpha and Gaussian estimate) and the outputs (mean charge, width, PE heating, recombination
and autoionisation cooling, Z*, their d/d ln T, the grain-assisted
recombination coefficients of the RECOMB_IONS, whether the bin's recombination
was refined by the fit (solve_cell's importance gate) and whether it fitted), then
the d/d ln S and d/d ln N of the predictor (zero where it is off).

Each full solve (path 0) is re-solved with GroupChargingTable.solve
(kernels='pchip') from the same warm state, so every branch is the same and
the differences should be round-off. Calls served from the d/d ln T of the
bin's last full solve (path 2, the cooling solver's T(1 +- 1e-5) calls) are
checked against the Python derivatives of that solve. Lines from the
uniform-G0 table (path 1: no local photons) are compared with a direct solve,
so their differences are the table's interpolation error.

With the predictor on (rtg_predict_tol > 0), each full solve also carries
d/d ln S and d/d ln N, compared with solve(..., sens=True). A predicted call
(path 4) is replayed with rt_group_charging.predict from the Python solve of its
anchor (the last full solve of that bin in that cell since the warm states were
reset); a full solve that had an anchor is a refused prediction, and predict
must refuse it too.

    python compare_ramses_rtgroups.py RUN_DIR [RUN_DIR ...] --tables TABLES_DIR [--stride N]
        [--predict-tol 0.1 --predict-tol-shape 0.3]   (the run's rtg_predict_tol, _shape)

With ``--psitab DIR`` it instead checks the lookups of charging_model='WDB06tab'
(dust_psitab_debug_XXXXX.dat) against charging_psi_tables.psi_lookup on the
tables in DIR.

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
SENS = tuple(f"d{q}_dln{x}" for x in "SN" for q in ("Zmean", "Zsigma", "Gamma", "Lambda"))
DLNT = 1e-5          # dust_interface RTG_DLNT


def read_dump(path):
    """Calls of one dump file, in order. Each line carries the warm state it used; the
    warm-state resets 'R' only number the rows' epochs (a predictor anchor is that of
    the same cell and bin in the same epoch)."""
    rows = []
    epoch = 0
    for line in open(path):
        if line.startswith("R"):
            epoch += 1
            continue
        if line.startswith("#"):
            continue
        p = line.split()
        it = iter(p)
        nxt = lambda: next(it)                                           # noqa: E731
        r = dict(icell=int(nxt()), bin=int(nxt()), path=int(nxt()), epoch=epoch)
        for k in ("T", "ne", "n_Hp", "n_Hep", "n_Hepp", "m_H", "m_He", "c_red", "G0_bg"):
            r[k] = float(nxt())
        nG = int(nxt())
        r["Np"] = np.array([float(nxt()) for _ in range(nG)])
        r["egy"] = np.array([float(nxt()) for _ in range(nG)])
        zg = float(nxt())
        r["Z_guess"] = None if zg <= -1e299 else zg
        w = dict(valid=nxt() == "T", discrete=nxt() == "T", Zstar=float(nxt()), wlo=float(nxt()), whi=float(nxt()),
                 Zsigma=float(nxt()))
        nx = int(nxt())
        w["x"] = np.array([float(nxt()) for _ in range(nx)])
        w["alpha"] = np.array([float(nxt()) for _ in range(len(rg.RECOMB_IONS))])
        w["lg"] = np.array([float(nxt()) for _ in range(len(rg.RECOMB_IONS))])
        r["warm"] = w
        out = {k: float(nxt()) for k in ("Zmean", "Zsigma", "Gamma", "Lambda", "Lambda_auto", "Zstar")}
        out["discrete"] = nxt() == "T"
        for k in ("dZmean_dlnT", "dZsigma_dlnT", "dGamma_dlnT", "dLambda_dlnT"):
            out[k] = float(nxt())
        out["nfit"] = int(nxt())
        out["refined"] = nxt() == "T"
        out["alpha"] = np.array([float(nxt()) for _ in range(len(rg.RECOMB_IONS))])
        sens = [float(x) for x in it]                                  # absent before the predictor
        out["sens"] = len(sens) == len(SENS) and any(sens)
        if out["sens"]:
            out.update(zip(SENS, sens))
        r["out"] = out
        rows.append(r)
    return rows


def warm_dict(w):
    """The pyCALIMA warm result equivalent to a dumped RTGState."""
    if not w["valid"]:
        return None
    d = dict(Zstar=w["Zstar"], discrete=w["discrete"], Zsigma=w["Zsigma"], nodes=np.array([w["wlo"], w["whi"]]))
    if w["x"].size:
        d["recomb_ref"] = (w["x"], w["alpha"], w["lg"])
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


def compare_psitab(runs, tabdir):
    """RAMSES WDB06tab lookups against charging_psi_tables.psi_lookup."""
    from pycalima.models.dust_charge import charging_psi_tables as cpt
    pts = {}
    for run in runs:
        worst = dict(Zmean=0.0, Zsigma=0.0, Gamma=0.0, Lambda=0.0, alpha=0.0)
        n = 0
        for d in sorted(Path(run).glob("dust_psitab_debug_*.dat")):
            for line in open(d):
                if line.startswith("#"):
                    continue
                p = line.split()
                b = int(p[0])
                T, ne, GF, hF, GE, hE1, hE2, Zm, Zs, Gm, Lm = (float(x) for x in p[1:12])
                al = np.array(p[12:22], dtype=float)
                if b not in pts:
                    pts[b] = cpt.read_psi_table(Path(tabdir) / f"dust_charging_psitab_DustBin_{b:02d}.dat")
                py = cpt.psi_lookup(pts[b], T, ne, GF, hF, GE, hE1, hE2)
                for k, f in (("Zmean", Zm), ("Zsigma", Zs), ("Gamma", Gm), ("Lambda", Lm)):
                    worst[k] = max(worst[k], rel(py[k], f, 1.0 if k == "Zmean" else 0.0))
                worst["alpha"] = max(worst["alpha"], rel(py["alpha_rec"], al))
                n += 1
        print(f"== {run}: {n} WDB06tab lookups, max rel diff vs psi_lookup: "
              + " ".join(f"{k}={v:.1e}" for k, v in worst.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", help="run directories holding dust_rtgroups_debug_*.dat")
    ap.add_argument("--tables", required=True, help="directory with dust_charging_rtgroups_DustBin_XX.dat")
    ap.add_argument("--stride", type=int, default=1, help="check every Nth full solve (each line is independent)")
    ap.add_argument("--worst", type=int, default=3, help="show the N worst full solves")
    ap.add_argument("--original-tables", default=None,
                    help="CALIMA table directory with dust_charge_Z_vs_T_DustBin_XX (original path)")
    ap.add_argument("--psitab", default=None, help="WDB06tab table directory: check dust_psitab_debug_*.dat instead")
    ap.add_argument("--predict-tol", type=float, default=rg.PREDICT_TOL, help="the run's rtg_predict_tol")
    ap.add_argument("--predict-tol-shape", type=float, default=rg.PREDICT_TOL_SHAPE, help="the run's rtg_predict_tol_shape")
    args = ap.parse_args()
    if args.psitab:
        return compare_psitab(args.runs, args.psitab)

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
        worst_sens = {k: 0.0 for k in SENS}
        worst_pred = {k: 0.0 for k in KEYS[:6] + ("alpha",)}
        npath = np.zeros(5, int)
        nchk = ndisc = nfit = nmis_disc = nmis_fit = 0
        nref = nmis_ref = nmis_pred = 0
        bad = []
        last = {}

        def ions_of(r):
            return [(r["n_Hp"], 1.0, r["m_H"]), (r["n_Hep"], 1.0, r["m_He"]), (r["n_Hepp"], 2.0, r["m_He"])]

        def solve_full(r):
            """The full solve of a dumped call, refined where RAMSES refined it (its importance
            gate), so that a predictor anchor carries the same alpha."""
            gt_, warm_ = table(r["bin"], r["egy"]), warm_dict(r["warm"])
            py_ = gt_.solve(r["Np"], r["c_red"], r["T"], r["ne"], ions_of(r), G0_bg=r["G0_bg"], Z_guess=r["Z_guess"],
                            recomb="gauss", warm=warm_, dlnT=DLNT, sens=r["out"]["sens"])
            py_["fitted"] = gt_.refine_recombination(py_, warm_) if r["out"]["refined"] else False
            return py_

        def predicted(an, r):
            """rt_group_charging.predict at r's inputs from anchor an = [row, its Python solve or None]."""
            if an[1] is None:
                an[1] = solve_full(an[0])
            return rg.predict(an[1], r["Np"], r["c_red"], r["T"], r["ne"], ions_of(r), G0_bg=r["G0_bg"],
                              tol=args.predict_tol, tol_shape=args.predict_tol_shape)

        for d in dumps:
            rows = read_dump(d)
            k0 = 0
            anchors = {}     # (icell, bin, epoch) -> [row, Python solve]: the last full solve with sensitivities
            for r in rows:
                npath[r["path"]] += 1
                gt = table(r["bin"], r["egy"])
                ions = ions_of(r)
                o = r["out"]
                akey = (r["icell"], r["bin"], r["epoch"])
                if r["path"] == 0:
                    an = anchors.pop(akey, None)
                    if an is not None:
                        nref += 1
                        nmis_ref += int(predicted(an, r) is not None)    # RAMSES refused this prediction
                    if o["sens"]:
                        anchors[akey] = an = [r, None]
                    k0 += 1
                    if (k0 - 1) % args.stride:
                        last.pop((d, r["bin"]), None)
                        continue
                    py = solve_full(r)
                    fitted = py["fitted"]
                    if o["sens"]:
                        an[1] = py
                    last[(d, r["bin"])] = py
                    nchk += 1
                    ndisc += int(o["discrete"])
                    nfit += o["nfit"]
                    nmis_disc += int(py["discrete"] != o["discrete"])
                    nmis_fit += int(fitted != bool(o["nfit"]))
                    if py["discrete"] != o["discrete"]:
                        continue
                    diffs = {}
                    for k in KEYS:
                        if k == "alpha" and not o["alpha"].any():
                            continue                            # dust_ion_recombination off (or alpha = 0 exactly)
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
                    if o["sens"]:
                        scale = dict(Zmean=max(abs(o["Zmean"]), 1.0), Zsigma=max(o["Zsigma"], 1.0), Gamma=abs(o["Gamma"]),
                                     Lambda=abs(o["Lambda"] + o["Lambda_auto"]))
                        for k in SENS:
                            worst_sens[k] = max(worst_sens[k], rel(py[k], o[k], 1e-6 * scale[k[1:-5]]))
                    bad.append((max(diffs.values()), r, diffs))
                elif r["path"] == 4:
                    an = anchors.get(akey)
                    py = predicted(an, r) if an is not None else None
                    if py is None:
                        nmis_pred += 1                           # RAMSES predicted what predict refuses
                        last.pop((d, r["bin"]), None)
                        continue
                    last[(d, r["bin"])] = py                     # the T +- 1e-5 calls it serves
                    for k in worst_pred:
                        if k == "alpha":
                            if o["alpha"].any():                 # the run has the grain recombination
                                worst_pred[k] = max(worst_pred[k], rel(py["alpha_rec"], o["alpha"]))
                            continue
                        worst_pred[k] = max(worst_pred[k], rel(py[k], o[k], 1.0 if k in ("Zmean", "Zstar") else 0.0))
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
        print(f"== {run}: {sum(npath)} calls (full {npath[0]}, uniform table {npath[1]}, from d/dlnT {npath[2]}, "
              f"predicted {npath[4]})")
        print(f"   {nchk} full solves re-solved ({ndisc} discrete, {nfit} recombination fits): "
              f"discrete-flag mismatches {nmis_disc}, fit/reuse mismatches {nmis_fit}")
        print("   max rel diff, full solves: " + " ".join(f"{k}={v:.1e}" for k, v in worst.items()))
        print("     of which d/dlnT of discrete P(Z) (analytic; wide: forward difference): "
              + " ".join(f"{k}={v:.1e}" for k, v in worst_dd.items()))
        if any(worst_sens.values()):
            print("     d/dlnS, d/dlnN (predictor): " + " ".join(f"{k}={v:.1e}" for k, v in worst_sens.items()))
        print("   max rel diff, from d/dlnT: " + " ".join(f"{k}={v:.1e}" for k, v in worst_served.items()))
        if npath[4] or nref:
            print(f"   predictor: {npath[4]} predicted (refused by predict: {nmis_pred}), {nref} refused "
                  f"(predicted by predict: {nmis_ref})")
            print("   max rel diff, predicted vs predict: " + " ".join(f"{k}={v:.1e}" for k, v in worst_pred.items()))
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
