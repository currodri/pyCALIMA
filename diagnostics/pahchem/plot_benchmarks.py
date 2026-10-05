#!/usr/bin/env python3
"""Benchmark figures for pahchem: our results over the published ones.

Writes one PNG per benchmark to ``<plots dir>/pahchem/`` (see ``calima-paths``).
The calculations are those of the gated tests (tests/test_pahchem.py) and of
the check_*.py scripts, whose docstrings record the measured numbers.

| figure | benchmark | milestone | time |
|---|---|---|---|
| west18 | RRKM k(E) from West+18's own frequencies and Table 1 E0, against their ESI k(E) | M2 | s |
| jochims94 | RRK E(k = 1e4 s^-1) against Jochims+94 Table 2; West+18 thresholds | M2 | s |
| montillaud13_fig3 | coronene k(E), inverse Laplace with their A and E0 | M2 | s |
| andrews16_fig5 | k_H, k_H2 (Table 2, T_e form) and k_IR of the three parents | M2-M3 | ~1 min |
| andrews16_tableB | Table B against eq. 7, the network rate r_abs P and r_abs, against G0 | M0, M5 | ~3 min |
| montillaud13_fig10 | 10/50/90 % totally dehydrogenated contours of C54H18 | M5 | ~15 min |
| andrews16_ngc7023 | Figs. 7 (N_H) and 8 (charge) of C54H18 in NGC 7023 NW | M5 | ~7 min |
| murga_wiebe22_fig1 | R_H(chi) of C66H20 by four rate forms | M5 | ~2 min |
| sensitivity | the one-at-a-time ranking at NGC 7023 and the Orion Bar | M5 | ~25 min |
| allain96_fig7 | ionised fraction vs N_C, Orion Bar and NGC 7027 (Allain+96 II Fig. 7) | M5 | ~30 s |
| allain96_fig10 | hydrogenation coefficient vs N_C, Orion Bar (by charge) and NGC 7027 (II Fig. 10) | M5 | ~30 s |
| lepage01_fig10 | H-loss photodissociation rate vs N_C, six classes (Le Page+01 Fig. 10) | M5 | ~1 min |
| lepage03_fig6_7 | C2 / C2H2 / H loss vs N_C, dehydrogenated and normal PAHs (LP03 Figs. 6-7) | M5 | ~1 min |

Inputs that are not shipped:
- ``--west-esi``: the West+18 ESI workbook (default: the west2018-esi dataset);
- ``--c66-modes``: C66H20_0.dat, extracted from PAHdb as in check_murga_wiebe22.py.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pycalima import _paths
from pycalima.pahchem import cooling, dos, energy_me, fields, labdata, photo, rates
from pycalima.pahchem.molecule import PAHSpecies

EV = dos.CM_PER_EV
E_PH = np.linspace(5.0, 13.6, 345)
EXT = _paths.get_external_data_path


def _save(fig, out, name):
    path = out / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}", flush=True)


def _residual_axes(figsize=(7, 6)):
    return plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [3, 1]})


# ---------------------------------------------------------------------------
# M2: unimolecular rates
# ---------------------------------------------------------------------------

#: reaction -> path degeneracy; the reactions whose ESI k(E) uses Table 1 E0 (test_pahchem.py)
WEST_CLEAN = {1: 1, 2: 6, 3: 2, 4: 1, 7: 1, 24: 2, 25: 1}


def fig_west18(args, out):
    esi, t1 = labdata.west2018_esi(args.west_esi), labdata.west2018_table1()
    fig, (ax, ar) = _residual_axes()
    for n, sigma in WEST_CLEAN.items():
        r = esi[n]
        grid, counts = dos.beyer_swinehart(r.reactant_freqs_cm, r.energy_cm.max() + 10.0)
        k = rates.rrkm(grid, counts, r.ts_freqs_cm, t1.loc[n, "E0_eV"] * EV, degeneracy=sigma, window_cm=1.0)
        ours = k[np.rint(r.energy_cm).astype(int)]
        ok = (r.k_s > 0) & (ours > 0)
        E = r.energy_cm[ok] / EV
        row = t1.loc[n]
        line, = ax.semilogy(E, r.k_s[ok], lw=4, alpha=0.35,
                            label=f"R{n}: {row['reactant']} $\\to$ {row['products']}")
        ax.semilogy(E, ours[ok], "--", lw=1, color=line.get_color())
        ar.plot(E, np.log10(ours[ok] / r.k_s[ok]), color=line.get_color(), lw=1)
    ax.set_ylim(1e-2, 1e12)
    ax.set_ylabel("$k(E)$ [s$^{-1}$]")
    ax.set_title("West+18 iPEPICO RRKM (thick) vs pahchem.rates.rrkm (dashed)\n"
                 "same B3LYP frequencies and TS, Table 1 $E_0$: a code check", fontsize=9)
    ax.legend(fontsize=7, ncol=2)
    ar.set_ylabel("$\\Delta\\log k$")
    ar.set_xlabel("ion internal energy [eV]")
    ar.set_ylim(-0.06, 0.06)
    ar.axhline(0, color="k", lw=0.5)
    _save(fig, out, "west18")


def fig_jochims94(args, out):
    from scipy.optimize import brentq

    t = labdata.jochims94_table(2)
    ours = [brentq(lambda e: np.log(rates.rrk(e * EV, int(s) + 1, 1e16, 2.8 * EV)) - np.log(1e4), 2.81, 60.0)
            for s in t["s_minus_1"]]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(t["n_c"], t["eint_exp_eV"], "ko", label="Jochims+94 measured $E_{\\rm int}$ at the H-loss AP")
    ax.plot(t["n_c"], t["eint_calc_eV"], "s", mfc="none", ms=9, color="C0",
            label="Jochims+94 RRK fit (printed), $k = 10^4$ s$^{-1}$")
    ax.plot(t["n_c"], ours, "x", color="C1", ms=8, label="pahchem.rates.rrk, same $\\nu$, $E_0$")
    try:
        esi = labdata.west2018_esi(args.west_esi)
        for name, n, nc in (("pyrene", 17, 16), ("fluoranthene", 11, 16), ("coronene", 20, 24)):
            r = esi[n]
            ok = r.k_s > 0
            e4 = np.interp(4.0, np.log10(r.k_s[ok]), r.energy_cm[ok] / EV)
            ax.plot(nc, e4, "*", color="C3", ms=12, label="West+18 iPEPICO RRKM, $k = 10^4$ s$^{-1}$"
                    if name == "pyrene" else None)
            ax.annotate(name, (nc, e4), xytext=(4, 4), textcoords="offset points", fontsize=7, color="C3")
    except Exception as exc:                       # the ESI is optional here
        print(f"  (no West+18 ESI: {exc})")
    ax.set_xlabel("$N_{\\rm C}$")
    ax.set_ylabel("internal energy at $k = 10^4$ s$^{-1}$ [eV]")
    ax.set_title("H loss from PAH cations: RRK reproduction (crosses in squares)\n"
                 "and the West vs Jochims laboratory systematic (stars vs dots)", fontsize=9)
    ax.legend(fontsize=7)
    _save(fig, out, "jochims94")


def fig_montillaud13_fig3(args, out):
    sp = PAHSpecies.from_states_file(args.states_dir / "C24H12_0.dat")
    E, counts = dos.beyer_swinehart(sp.freqs_cm, 16.0 * EV)
    k = rates.inverse_laplace_arrhenius(E, counts, 6.8e17, 4.8 * EV)
    d = pd.read_csv(EXT("dissoc_rate_coronene_Montillaud.csv"), skiprows=1)
    fig, (ax, ar) = _residual_axes()
    for (xc, yc), label, style in ((("X", "Y"), "Montillaud+13 Fig. 3", dict(color="k", marker="o", ls="none")),
                                   (("X.1", "Y.1"), "Le Page+01 Fig. 9 (in their Fig. 3)", dict(color="0.6", ls=":")),
                                   (("X.2", "Y.2"), "Berné & Tielens (in their Fig. 3)", dict(color="0.6", ls="--"))):
        x, y = d[xc].dropna().to_numpy(), d[yc].dropna().to_numpy()
        ax.semilogy(x, y, label=label, **style)
        if xc == "X":
            ar.plot(x, np.log10(k[(x * EV).astype(int)] / y), "ko")
    sel = E > 5.0 * EV
    ax.semilogy(E[sel] / EV, k[sel], color="C1", label="pahchem inverse Laplace, $A = 6.8\\times10^{17}$, $E_0 = 4.8$ eV")
    ax.set_ylim(1e-6, 1e10)
    ax.set_ylabel("$k_{\\rm H}(E)$ [s$^{-1}$]")
    ax.set_title("Coronene H loss: PAHdb modes for Montillaud's Malloci modes", fontsize=9)
    ax.legend(fontsize=7)
    ar.axhline(0, color="k", lw=0.5)
    ar.set_ylabel("$\\Delta\\log k$")
    ar.set_xlabel("$E$ [eV]")
    _save(fig, out, "montillaud13_fig3")


def fig_andrews16_fig5(args, out):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    E = np.linspace(0.2, 40.0, 400)
    for ax, name in zip(axes, ("C24H12", "C54H18", "C96H24")):
        sp = PAHSpecies.from_states_file(args.states_dir / f"{name}_0.dat")
        Eg, counts = dos.beyer_swinehart(sp.freqs_cm, E.max() * EV + 200.0, 2.0)
        k_ir = cooling.cooling_rate(sp.freqs_cm, cooling.mode_emission_rates(
            Eg, counts, sp.freqs_cm, sp.einstein_a, at_cm=E * EV)) / (E * EV)
        ir_file = f"kIR_{name}_Andrews16.csv" if name != "C54H18" else "IR_C54H18_Andrews16.csv"
        for label, file, ours, color in (
                ("H loss", f"H-loss_{name}_Andrews16.csv",
                 rates.arrhenius_effective_temperature(sp.freqs_cm, E * EV, 4.6 * EV, 44.8), "C0"),
                ("H$_2$ loss", f"H2-loss_{name}_Andrews16.csv",
                 rates.arrhenius_effective_temperature(sp.freqs_cm, E * EV, 3.52 * EV, -53.1), "C2"),
                ("$k_{\\rm IR}$", ir_file, k_ir, "C3")):
            d = np.loadtxt(EXT(file), delimiter=",")
            d = d[np.argsort(d[:, 0])]
            ax.semilogy(d[:, 0], d[:, 1], "o", ms=3, color=color, label=f"Andrews+16 {label}")
            ax.semilogy(E, np.where(ours > 0, ours, np.nan), color=color, label=f"pahchem {label}")
        ax.set_title(f"{name} (PAHdb neutral modes)", fontsize=9)
        ax.set_xlabel("$E$ [eV]")
        ax.set_ylim(1e-10, 1e13)
    axes[0].set_ylabel("rate [s$^{-1}$]")
    axes[0].legend(fontsize=7)
    fig.suptitle("Andrews+16 Fig. 5: Table 2 Arrhenius ($T_e$ form) and microcanonical IR emission", fontsize=10)
    _save(fig, out, "andrews16_fig5")


# ---------------------------------------------------------------------------
# M0/M5: Andrews Table B, Montillaud Fig. 10, Andrews NGC 7023, Murga & Wiebe
# ---------------------------------------------------------------------------

def fig_andrews16_tableB(args, out):
    from pycalima.pahchem.andrews16 import Andrews16State, channels

    cases = (("C24H12", 12, 12, 0, "table_B1_coronene.csv"),
             ("C24H12", 13, 12, 0, "table_B1_coronene.csv"),
             ("C54H18", 18, 18, 6, "table_B2_circumcoronene.csv"),
             ("C96H24", 24, 24, 12, "table_B3_circumcircumcoronene.csv"))
    lg = np.arange(0.0, 5.01, 0.5)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.5), sharey=True)
    for ax, (name, nh, nh0, n_solo, table) in zip(axes, cases):
        sp = PAHSpecies.from_states_file(args.states_dir / f"{name}_0.dat")
        st = Andrews16State(sp, nh0, n_solo, 0)
        st.k = {c: rates.arrhenius_effective_temperature(sp.freqs_cm, st.cm.energy_cm, Ea * EV, dS)
                for c, (Ea, dS) in channels(nh, nh0, n_solo, 0).items()}
        sigma = photo.sigma_malloci_coronene(0, E_PH) if sp.nc == 24 else photo.sigma_li_draine(sp.nc, 0, E_PH)
        res = [st.rates(E_PH, sigma, fields.kurucz(15000, 10**g)) for g in lg]
        row = pd.read_csv(EXT(table)).query("Z == 0 and NH == @nh").iloc[0]
        for i, c in enumerate(st.k):
            color = f"C{i}"
            theirs = sum(row[f"{c}_loss_p{p}"] * lg**p for p in range(5))
            ax.plot(lg, theirs, color=color, lw=4, alpha=0.35, label=f"Table B, {c} loss")
            ax.plot(lg, [np.log10(r.k_eq7[c]) for r in res], "o", color=color, label=f"eq. 7 emulation, {c}")
            ax.plot(lg, [np.log10(r.network_rates[c]) for r in res], "--", color=color,
                    label=f"$r_{{\\rm abs}} P$ (network), {c}")
        ax.plot(lg, [np.log10(r.r_abs) for r in res], "k:", label="$r_{\\rm abs}$")
        ax.set_title(f"{name[:3]}H{nh}, Z = 0", fontsize=9)
        ax.set_xlabel("$\\log G_0$ (Kurucz 15000 K)")
    axes[0].set_ylabel("$\\log$ rate [s$^{-1}$]")
    axes[0].legend(fontsize=6)
    fig.suptitle("Andrews+16 Table B is eq. 7 (sink-free, can exceed $r_{\\rm abs}$); "
                 "their network uses $r_{\\rm abs}P \\leq r_{\\rm abs}$", fontsize=10)
    _save(fig, out, "andrews16_tableB")


def fig_montillaud13_fig10(args, out):
    from pycalima.models.PAH_photophysics.PAH_photophysics import read_sections
    from pycalima.pahchem.montillaud13 import Montillaud13

    sp = PAHSpecies.from_states_file(args.states_dir / "C54H18_0.dat")
    g0 = 10 ** np.arange(0.0, 5.51, 0.25)
    nh = 10 ** np.arange(1.0, 5.01, 0.25)
    fig, ax = plt.subplots(figsize=(7, 6))
    for level, (x, y), color in zip((0.1, 0.5, 0.9), read_sections("montillaud13_hydrogenation_fraction_C54.csv"),
                                    ("C0", "C1", "C2")):
        ax.plot(np.log10(np.array(x, float)), np.log10(np.array(y, float)), "o-", color=color, ms=3,
                label=f"Montillaud+13 Fig. 10, {level:.0%}")
    for modes, ls in (("parent", "-"), ("stripped", "--")):
        if modes == "stripped" and args.quick:
            continue
        F = Montillaud13(sp, "C54", modes=modes).hydrogenation_grid(g0, nh)
        cs = ax.contour(np.log10(g0), np.log10(nh), F.T, levels=[0.1, 0.5, 0.9],
                        colors=["C0", "C1", "C2"], linestyles=ls, linewidths=1.5)
        ax.plot([], [], "k", ls=ls, label=f"pahchem, {modes} modes" + (" (their model)" if modes == "parent" else ""))
        np.savez(out / f"montillaud13_fig10_{modes}.npz", g0=g0, nh=nh, f_bare=F)
    ax.set_xlabel("$\\log G_0$ (Mathis)")
    ax.set_ylabel("$\\log n_{\\rm H}$ [cm$^{-3}$]")
    ax.set_title("C$_{54}$H$_{18}$: fraction totally dehydrogenated (T = 100 K)", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    _save(fig, out, "montillaud13_fig10")


def fig_andrews16_ngc7023(args, out):
    from pycalima.pahchem.andrews16 import Andrews16Model

    modes = {}
    for f in sorted(args.states_dir.glob("C54*.dat")):
        s = PAHSpecies.from_states_file(f)
        modes[(s.charge, s.nh)] = s
    model = Andrews16Model("C54", modes, de_cm=150.0)
    D = Path(EXT("NWPDR_NGC7023"))
    prof = {n: np.loadtxt(D / f"{n}.csv", delimiter=",") for n in ("G0nH", "nH", "ne", "temperature")}
    at = lambda n, a: np.interp(a, prof[n][:, 0], prof[n][:, 1])
    rows = list(csv.reader(open(D / "Nh_circumcoronene.csv")))
    fig7 = {}
    for j in range(0, len(rows[0]), 2):
        pts = np.array([[float(r[j]), float(r[j + 1])] for r in rows[2:] if len(r) > j + 1 and r[j] and r[j + 1]])
        fig7[int(rows[0][j].split("=")[1])] = pts[np.argsort(pts[:, 0])]
    names = {-1: "anion", 0: "neutral", 1: "cation", 2: "dication"}
    fig8 = {}
    for z, name in names.items():
        d = np.loadtxt(EXT(f"C54HN_{name}_andrews16.csv"), delimiter=",")
        fig8[z] = d[np.argsort(d[:, 0])]

    av = np.array([0.05, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
    fh, fz, gam = [], [], []
    for a in av:
        n_h, n_e, T = at("nH", a), at("ne", a), at("temperature", a)
        g0 = at("G0nH", a) * n_h
        net = model.network(model.photo_rates(g0), n_h, n_e, T)
        f = net.steady_state()
        fh.append(net.hydrogenation_fractions(f))
        fz.append(net.charge_fractions(f))
        gam.append(g0 * np.sqrt(T) / n_e)
        print(f"  A_V = {a}: done", flush=True)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    for i, n in enumerate(sorted(fig7)):
        a1.plot(fig7[n][:, 0], fig7[n][:, 1], color=f"C{i}", lw=3, alpha=0.4, label=f"Andrews+16, N$_{{\\rm H}}$ = {n}")
        a1.plot(av, [h.get(n, 0.0) for h in fh], "o--", color=f"C{i}", label=f"pahchem, N$_{{\\rm H}}$ = {n}")
    a1.plot(av, [h.get(18, 0.0) for h in fh], "ko--", label="pahchem, N$_{\\rm H}$ = 18")
    a1.set_yscale("log")
    a1.set_ylim(1e-4, 1.5)
    a1.set_xlabel("$A_V$")
    a1.set_ylabel("fraction")
    a1.set_title("Fig. 7: hydrogenation of C$_{54}$H$_{18}$ along the NW PDR", fontsize=9)
    a1.legend(fontsize=6, ncol=2)
    for z, name in names.items():
        a2.plot(fig8[z][:, 0], fig8[z][:, 1], color=f"C{z + 1}", lw=3, alpha=0.4, label=f"Andrews+16, {name}")
        a2.plot(gam, [f[z] for f in fz], "o--", color=f"C{z + 1}", label=f"pahchem, {name}")
    a2.set_xscale("log")
    a2.set_xlabel("$\\gamma = G_0 T^{1/2}/n_e$")
    a2.set_ylabel("charge fraction")
    a2.set_title("Fig. 8: charge of C$_{54}$H$_{18}$", fontsize=9)
    a2.legend(fontsize=6, ncol=2)
    fig.suptitle("Andrews+16 NGC 7023 NW, in their configuration (Andrews16Model; eq. 8 k$_{\\rm IR}$ along the cooling path)",
                 fontsize=10)
    _save(fig, out, "andrews16_ngc7023")



def fig_murga_wiebe22_fig1(args, out):
    yr, cal = 3.15576e7, 4.184
    sp = PAHSpecies.from_states_file(args.c66_modes)
    cm = energy_me.CascadeModel(sp.freqs_cm, sp.einstein_a, e_max_cm=50 * EV, de_cm=100.0)
    E = cm.energy_cm
    Eg, counts = dos.beyer_swinehart(sp.freqs_cm, E[-1] + 200.0)
    k = {"Arrh. 1": rates.arrhenius_effective_temperature(sp.freqs_cm, E, 3.3 * EV, 5.0 * cal),
         "Arrh. 2": rates.arrhenius_effective_temperature(sp.freqs_cm, E, 4.3 * EV, 11.8 * cal),
         "RRK": rates.rrk(E, sp.n_modes, 1e16, 2.8 * EV),
         "Laplace": rates.inverse_laplace_arrhenius(Eg, counts, 1.4e16, 4.3 * EV)[np.rint(E).astype(int)]}
    sigma = photo.sigma_li_draine(66, 0, E_PH)
    chi = 10 ** np.arange(0.0, 5.01, 0.5)
    free = {n: [] for n in k}
    sink = {n: [] for n in k}
    for c in chi:
        h = photo.heating_spectrum(fields.mathis1983(g0=1.13 * c), sigma, np.zeros_like(E_PH), E_PH)
        P = cm.steady_state(E_PH, h, {}).probability
        for n, kk in k.items():
            free[n].append(np.log10(P @ kk * yr))
            sink[n].append(np.log10(cm.steady_state(E_PH, h, {n: kk}).rates[n] * yr))
    mw = pd.read_csv(EXT("pahchem/murga_wiebe22_fig1.csv"), comment="#")
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for i, n in enumerate(k):
        ax.plot(np.log10(chi), free[n], color=f"C{i}", label=f"pahchem {n} (sink-free, their eq. 1)")
        ax.plot(np.log10(chi), sink[n], ":", color=f"C{i}")
        c = mw[(mw["product"] == "H") & (mw["method"] == n)].sort_values("chi")
        ax.plot(np.log10(c.chi), np.log10(c.rate_per_yr), "s-", lw=4, alpha=0.35, color=f"C{i}",
                label=f"MW22 Fig. 1 {n}")
    ax.plot([], [], "k:", label="with a dissociation sink (physical)")
    ax.set_xlabel("$\\log \\chi$ (Mathis)")
    ax.set_ylabel("$\\log R_{\\rm H}$ [yr$^{-1}$]")
    ax.set_title("Murga & Wiebe 2022 Fig. 1: C$_{66}$H$_{20}$ H loss by four rate forms", fontsize=9)
    ax.legend(fontsize=6, ncol=2)
    _save(fig, out, "murga_wiebe22_fig1")


def fig_sensitivity(args, out):
    from pycalima.pahchem.baseline import NGC7023_EDGE, ORION_BAR_ATOMIC, sensitivity

    cache = out / "sensitivity.json"
    if cache.exists() and not args.recompute:
        results = json.loads(cache.read_text())
    else:
        sp = PAHSpecies.from_states_file(args.states_dir / "C54H18_0.dat")
        results = {env.name: sensitivity("C54", sp, env) for env in (NGC7023_EDGE, ORION_BAR_ATOMIC)}
        cache.write_text(json.dumps(results, default=float, indent=1))
    dlog = lambda a, b: np.log10(max(a, 1e-30) / max(b, 1e-30))
    names = [n for n in next(iter(results.values())) if n != "baseline"]

    def score(n):
        return max((abs(dlog(r["f_dehydrogenated"], res["baseline"]["f_dehydrogenated"]))
                    for res in results.values() for r in res[n] if r is not None), default=0.0)

    names.sort(key=score)
    y = np.arange(len(names))
    fig, axes = plt.subplots(1, 4, figsize=(18, 0.35 * len(names) + 2), sharey=True)
    for col, (env, res) in enumerate(results.items()):
        b = res["baseline"]
        for k, (key, fn, label) in enumerate((
                ("f_dehydrogenated", dlog, "$\\Delta\\log f_{\\rm dehydro}$"),
                ("f_cation", lambda a, c: a - c, "$\\Delta f_+$"))):
            ax = axes[2 * col + k]
            for side, color in ((0, "C0"), (1, "C3")):
                vals = [fn(res[n][side][key], b[key]) if res[n][side] is not None else 0.0 for n in names]
                ax.barh(y + (0.2 if side else -0.2), vals, height=0.4, color=color,
                        label=("low" if side == 0 else "high") + " end of range")
            ax.axvline(0, color="k", lw=0.5)
            ax.set_xlabel(label)
            ax.set_title(f"{env}\nbaseline {key} = {b[key]:.3g}", fontsize=8)
    axes[0].set_yticks(y, names, fontsize=7)
    axes[0].legend(fontsize=7)
    fig.suptitle("One-at-a-time sensitivity of C$_{54}$H$_{18}$ (pahchem baseline)", fontsize=10)
    _save(fig, out, "sensitivity")


def fig_allain96_fig7(args, out):
    from pycalima.pahchem import allain96 as A

    names = ["C10H8", "C14H10", "C16H10", "C24H12", "C32H14", "PAH 50 C", "PAH 100 C"]
    d = pd.read_csv(EXT("pahchem/allain96_fig7_alpha_ion.csv"), comment="#")
    fams = {n: A.AllainFamily(n, A.load_modes(n, args.states_dir)) for n in names}
    nc = np.array([A.SPECIES[n][0] for n in names])
    fig, ax = plt.subplots(figsize=(7, 5))
    for med, color in (("Orion Bar", "C0"), ("NGC 7027", "C3")):
        c = d[d.medium == med].sort_values("n_c")
        ax.plot(c.n_c, c.alpha_ion_percent, color=color, lw=4, alpha=0.35, label=f"Allain+96 II Fig. 7, {med}")
        alpha = np.array([fams[n].steady(med)["alpha_ion"] for n in names])
        ax.plot(nc, 100 * alpha, "o--", color=color, label=f"pahchem.allain96, {med}")
        odds = alpha / (1 - alpha) * 10**-0.2               # large PAHs: remove the I Table 5 k_ion excess
        large = nc >= 50
        ax.plot(nc[large], 100 * (odds / (1 + odds))[large], "x", ms=9, color=color,
                label=f"{med}, 50/100 C with $k_{{\\rm ion}}$ / 10$^{{0.2}}$")
    ax.set_xlabel("$N_{\\rm C}$")
    ax.set_ylabel("$\\alpha_{\\rm ion}$ [%]")
    ax.set_title("Allain+96 II Fig. 7: ionised fraction in their configuration\n"
                 "(Malloci $\\sigma$ up to 32 C; Li & Draine $\\sigma$ at 50 and 100 C)", fontsize=9)
    ax.legend(fontsize=7)
    _save(fig, out, "allain96_fig7")


def fig_allain96_fig10(args, out):
    from pycalima.pahchem import allain96 as A

    names = ["C10H8", "C14H10", "C16H10", "C24H12", "C32H14", "PAH 50 C", "PAH 100 C"]
    d = pd.read_csv(EXT("pahchem/allain96_fig10_alpha_h.csv"), comment="#")
    fams = {n: A.AllainFamily(n, A.load_modes(n, args.states_dir)) for n in names}
    nc = [A.SPECIES[n][0] for n in names]

    def alpha_h(fam, med, charge):
        s = fam.steady(med)
        if charge == "both":
            return s["alpha_H"]
        f = s["f"][{"neutral": 0, "cation": 1}[charge]]
        return float(f @ np.arange(fam.nh0 + 1) / (fam.nh0 * f.sum()))

    fig, ax = plt.subplots(figsize=(7, 5))
    for med, charge, color in (("Orion Bar", "cation", "C0"), ("Orion Bar", "neutral", "C2"),
                               ("NGC 7027", "both", "C3")):
        c = d[(d.medium == med) & (d.charge == charge)].sort_values("n_c")
        ax.plot(c.n_c, c.alpha_h_percent, color=color, lw=4, alpha=0.35, label=f"Allain+96 II Fig. 10, {med} ({charge})")
        ax.plot(nc, [100 * alpha_h(fams[n], med, charge) for n in names], "o--", color=color,
                label=f"pahchem.allain96, {med} ({charge})")
    ax.set_xlabel("$N_{\\rm C}$")
    ax.set_ylabel("$\\alpha_{\\rm H}$ [%]")
    ax.set_title("Allain+96 II Fig. 10: hydrogenation coefficient in their configuration", fontsize=9)
    ax.legend(fontsize=7, loc="center right")
    _save(fig, out, "allain96_fig10")


LP_FAMILY = [(6, 6), (10, 8), (14, 10), (16, 10), (24, 12), (32, 14), (50, 20), (80, 24), (100, 28),
             (120, 30), (150, 40), (200, 50)]


def fig_lepage01_fig10(args, out):
    from pycalima.pahchem import lepage01 as L

    d = pd.read_csv(EXT("pahchem/lepage01_fig10_kph.csv"), comment="#")
    classes = {"PAH": (0, lambda y: y), "PAH+": (1, lambda y: y), "XdPAH": (0, lambda y: 1),
               "XdPAH+": (1, lambda y: 1), "PAHH": (0, lambda y: y + 1), "PAHH+": (1, lambda y: y + 1)}
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (cls, (z, nh_of)) in enumerate(classes.items()):
        c = d[d.species_class == cls].sort_values("n_c")
        ax.semilogy(c.n_c, c.k_ph_s, color=f"C{i}", lw=4, alpha=0.35, label=f"Le Page+01 Fig. 10, {cls}")
        nc = [n for n, _ in LP_FAMILY if n <= c.n_c.max() + 1]
        k = [L.k_photodissociation(n, nh_of(y), y, z)["H"] for n, y in LP_FAMILY if n <= c.n_c.max() + 1]
        ax.semilogy(nc, k, "o--", ms=4, color=f"C{i}", label=f"pahchem.lepage01, {cls}")
    ax.set_ylim(1e-16, 1e-6)
    ax.set_xlabel("$N_{\\rm C}$")
    ax.set_ylabel("$k_{\\rm ph}$ (H loss) [s$^{-1}$]")
    ax.set_title("Le Page+01 Fig. 10 in their configuration: their $\\sigma$ (Fig. 11), whole field,\n"
                 "$(1 - Y_{\\rm ion})$ for neutrals; PAH: normal, Xd: one H left, PAHH: one extra H", fontsize=9)
    ax.legend(fontsize=6, ncol=2)
    _save(fig, out, "lepage01_fig10")


def fig_lepage03_fig6_7(args, out):
    from pycalima.pahchem import lepage01 as L

    f6 = pd.read_csv(EXT("pahchem/lepage03_fig6_dehydrogenated_rates.csv"), comment="#")
    f7 = pd.read_csv(EXT("pahchem/lepage03_fig7_normal_rates.csv"), comment="#")
    fig, (a6, a7) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    ncs = np.arange(6, 41)
    c = f6[f6.channel == "C2"].sort_values("n_c")
    a6.semilogy(c.n_c, c.k_s, color="C0", lw=4, alpha=0.35, label="LP03 Fig. 6, C$_2$ loss")
    c = f6[f6.channel == "C2H2"].sort_values("n_c")
    a6.semilogy(c.n_c, c.k_s, color="C1", lw=4, alpha=0.35, label="LP03 Fig. 6, C$_2$H$_2$ loss (not modelled)")
    for z, ls in ((0, "--"), (1, ":")):
        a6.semilogy(ncs, [L.k_carbon_loss(int(n), 0, 20, z, *L.C2_LOSS) for n in ncs], ls, color="C0",
                    label=f"pahchem, bare C$_n$, Z = {z} (8 eV, 5 cal/K)")
    for k, lab in ((L.H_ADDITION_FIG6, "H addition"), (L.C_INSERTION_FIG6, "C$^+$ insertion")):
        a6.axhline(k, color="0.5", lw=0.8)
        a6.text(38, k * 1.3, lab, fontsize=7, ha="right", color="0.4")
    a6.set_title("LP03 Fig. 6: extremely dehydrogenated PAHs", fontsize=9)
    c = f7[f7.channel == "H"].sort_values("n_c")
    a7.semilogy(c.n_c, c.k_s, color="C2", lw=4, alpha=0.35, label="LP03 Fig. 7, H loss")
    c = f7[f7.channel == "C2H2"].sort_values("n_c")
    a7.semilogy(c.n_c, c.k_s, color="C1", lw=4, alpha=0.35, label="LP03 Fig. 7, C$_2$H$_2$ loss")
    fam = [(14, 10), (16, 10), (24, 12), (32, 14), (50, 20)]
    for z, ls in ((1, "o--"), (0, "s:")):
        a7.semilogy([n for n, _ in fam], [L.k_photodissociation(n, y, y, z)["H"] for n, y in fam], ls,
                    color="C2", label=f"pahchem H loss, Z = {z}")
    for e0, ls in ((4.5, "^--"), (6.0, "v--")):
        a7.semilogy([n for n, _ in fam], [L.k_carbon_loss(n, y, y, 1, e0, L.C2H2_LOSS_DS, L.C2H2_LOSS_DEGENERACY)
                                          for n, y in fam], ls, color="C1",
                    label=f"pahchem C$_2$H$_2$ loss, Z = 1, $E_0$ = {e0} eV")
    a7.set_title("LP03 Fig. 7: normal H coverage", fontsize=9)
    for a in (a6, a7):
        a.set_xlabel("$N_{\\rm C}$")
        a.set_ylim(1e-16, 1e-7)
        a.legend(fontsize=6)
    a6.set_ylabel("rate [s$^{-1}$]")
    fig.suptitle("Le Page+03 (n$_t$ = 25, $\\chi$ = 1): carbon-skeleton survival", fontsize=10)
    _save(fig, out, "lepage03_fig6_7")


FIGURES = {f.__name__[4:]: f for f in (
    fig_west18, fig_jochims94, fig_montillaud13_fig3, fig_andrews16_fig5, fig_andrews16_tableB,
    fig_montillaud13_fig10, fig_andrews16_ngc7023, fig_murga_wiebe22_fig1, fig_sensitivity,
    fig_allain96_fig7, fig_allain96_fig10, fig_lepage01_fig10, fig_lepage03_fig6_7)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("figures", nargs="*", help=f"any of {', '.join(FIGURES)} (default: all)")
    ap.add_argument("--states-dir", type=Path, default=_paths.get_model_data_dir() / "PAH_states")
    ap.add_argument("--west-esi", type=Path, default=None, help="West+18 ESI .xlsx")
    ap.add_argument("--c66-modes", type=Path, default=None, help="C66H20_0.dat")
    ap.add_argument("--out", type=Path, default=None, help="default: <plots dir>/pahchem")
    ap.add_argument("--quick", action="store_true", help="montillaud13_fig10: parent modes only")
    ap.add_argument("--recompute", action="store_true", help="sensitivity: ignore the cached json")
    args = ap.parse_args()
    unknown = set(args.figures) - set(FIGURES)
    if unknown:
        ap.error(f"unknown figures: {', '.join(sorted(unknown))}")
    out = args.out or _paths.get_plots_dir("pahchem")
    out.mkdir(parents=True, exist_ok=True)
    for name in args.figures or FIGURES:
        if name == "murga_wiebe22_fig1" and args.c66_modes is None:
            print("skipping murga_wiebe22_fig1: needs --c66-modes")
            continue
        print(f"[{name}]", flush=True)
        FIGURES[name](args, out)


if __name__ == "__main__":
    main()
