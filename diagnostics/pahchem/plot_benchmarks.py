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
| murga2019_shiva_inputs | SHIVA's T(E) (DL01 Fig. 7) and Y_ion (Murga+2019 Fig. 2) | M5 | s |
| murga_wiebe22_shiva | MW22 Fig. 1, H and C loss, with SHIVA's DL01 and Tielens T(E) | M5 | ~1 min |
| murga2020_fig3 | mean charge vs A_V, Orion Bar and Horsehead | M5 | s |
| murga2020_fig4 | C2H2 per molecule for the 20 parents, independent vs competing channels | M5 | minutes |
| murga2022_charge | <Z> vs distance by size against the Figshare f(a, Z) | M5 | ~1 min |
| murga2022_model_b | <X_H/X_H0> and the hydrogenation bins vs distance, model b | M5 | ~1 min |

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
        anion = PAHSpecies.from_states_file(args.states_dir / "C54H18_-1.dat")
        results = {env.name: sensitivity("C54", sp, env, anion=anion) for env in (NGC7023_EDGE, ORION_BAR_ATOMIC)}
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


# --- Murga et al. (SHIVA): the figures reuse the calculations of the check_* scripts ---

def _checks():
    import sys

    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)


def fig_murga2019_shiva_inputs(args, out):
    from pycalima.pahchem import shiva as S

    t = pd.read_csv(EXT("pahchem/shiva_dl01_fig7_temperature.csv"), comment="#")
    y = pd.read_csv(EXT("pahchem/murga2019_fig2_yion.csv"), comment="#")
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.8))
    for i, nc in enumerate(sorted(t.n_c.unique())):
        g = t[t.n_c == nc].sort_values("energy_cm")
        a.loglog(g.energy_cm, g.temperature_k, color=f"C{i}", lw=4, alpha=0.35, label=f"DL01 Fig. 7, N_C = {nc}")
        E = np.logspace(np.log10(g.energy_cm.min()), np.log10(g.energy_cm.max()), 200)
        a.loglog(E, S.Temperature(S.dl01_modes(int(nc), S.dl01_nh(int(nc))))(E / dos.CM_PER_EV), "--",
                 color=f"C{i}", label=f"pahchem.shiva, N_C = {nc}")
    a.set_xlabel("$E$ [cm$^{-1}$]")
    a.set_ylabel("$T_m$ [K]")
    a.set_title("SHIVA's T(E): Draine & Li 2001 synthetic modes", fontsize=9)
    a.legend(fontsize=6)
    for z in sorted(y.z.unique()):
        g = y[y.z == z].sort_values("energy_eV")
        b.plot(g.energy_eV, g.y_ion, color=f"C{z}", lw=4, alpha=0.35, label=f"Murga+19 Fig. 2, Z = {z}")
        E = np.linspace(g.energy_eV.min(), g.energy_eV.max(), 300)
        b.plot(E, S.y_ion_wd01(E, 3.3e-8, int(z)), "--", color=f"C{z}", label=f"WD01a (pyCALIMA), Z = {z}")
    b.set_xlabel("$h\\nu$ [eV]")
    b.set_ylabel("$Y_{\\rm ion}$")
    b.set_title("Photoelectric yield, a = 3.3 A (Murga+2019 Fig. 2)", fontsize=9)
    b.legend(fontsize=6)
    _save(fig, out, "murga2019_shiva_inputs")


def fig_murga_wiebe22_shiva(args, out):
    _checks()
    import check_shiva as CS
    from pycalima.pahchem import shiva as S

    mw = pd.read_csv(EXT("pahchem/murga_wiebe22_fig1.csv"), comment="#")
    chi = 10 ** np.arange(0.0, 5.01, 0.5)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for temp, ls in (("dl01", ":"), ("tielens", "--")):
        m = S.ShivaMolecule(66, 20, 0, temperature=temp)
        forms = CS.mw22_forms(m, args.c66_modes)
        p = [m.distribution(S.mmp83(c), ionising_photons_heat=True)[0] for c in chi]
        for ax, prod in zip(axes, ("H", "C")):
            for i, (name, k) in enumerate(forms[prod].items()):
                ax.plot(np.log10(chi), [np.log10(pp @ k * CS.YR + 1e-300) for pp in p], ls, color=f"C{i}",
                        label=f"SHIVA, {name}, T(E) {temp}")
    for ax, prod in zip(axes, ("H", "C")):
        for i, name in enumerate(("Arrh. 1", "Arrh. 2", "RRK", "Laplace")):
            g = mw[(mw["product"] == prod) & (mw.method == name)].sort_values("chi")
            ax.plot(np.log10(g.chi), np.log10(g.rate_per_yr), "s-", lw=4, alpha=0.35, color=f"C{i}",
                    label=f"MW22 Fig. 1, {name}")
        ax.set_xlabel("$\\log \\chi$ (MMP83)")
        ax.set_title(f"C$_{{66}}$H$_{{20}}$, {'H loss' if prod == 'H' else 'C-skeleton loss'}", fontsize=9)
        ax.set_ylim(-12, 7)
    axes[0].set_ylabel("$\\log R$ [yr$^{-1}$]")
    axes[0].legend(fontsize=5.5, ncol=2)
    fig.suptitle("Murga & Wiebe 2022 Fig. 1 (exact) against SHIVA's own pieces: DL01 vs Tielens T(E)", fontsize=10)
    _save(fig, out, "murga_wiebe22_shiva")


def fig_murga2020_fig3(args, out):
    _checks()
    import check_murga_charge as CC

    res = CC.m20_fig3()
    av = np.array([0.1] + list(np.arange(0.5, 5.01, 0.5)))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, site in zip(axes, ("orion_isochoric", "horsehead")):
        for i, mol in enumerate(("C24H12", "C66H20", "C128H28")):
            theirs, rounded, _mode = res[(site, mol)]
            off = (i - 1) * 0.04
            ax.plot(av + off, theirs + off, "o", ms=9, mfc="none", color=f"C{i}", label=f"Murga+20 Fig. 3, {mol}")
            ax.plot(av - off, rounded - off, "x", ms=7, color=f"C{i}", label=f"pahchem (WD01, M22 config.), {mol}")
        ax.set_xlabel("$A_V$")
        ax.set_title(site.replace("_", " "), fontsize=9)
    axes[0].set_ylabel("$\\langle Z \\rangle$ (rounded; markers offset for clarity)")
    axes[0].legend(fontsize=7)
    fig.suptitle("Murga+2020 Fig. 3: mean charge (48/66 points agree; the two papers used different conventions)",
                 fontsize=10)
    _save(fig, out, "murga2020_fig3")


def fig_murga2020_fig4(args, out):
    _checks()
    import check_murga2020 as C20
    from pycalima.pahchem import murga2020 as T

    variants = (("independent channels (default)", T.Options()), ("competing channels", T.Options(compete=True)))
    res = {label: C20.compare(T.PARENTS, opts, verbose=False) for label, opts in variants}
    x = np.arange(len(T.PARENTS))
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    styles = {(0.1, 0): ("C3", "s"), (2.0, 0): ("C0", "o"), (0.1, 1): ("C2", "D"), (2.0, 1): ("k", "x")}
    for ax, site in zip(axes, ("orion_isochoric", "horsehead")):
        for (av, z), (color, mk) in styles.items():
            theirs = [res[variants[0][0]][(site, av, z, p)][1] for p in T.PARENTS]
            ax.semilogy(x, theirs, mk, ms=7, mfc="none", color=color, label=f"Murga+20 Fig. 4, A_V = {av}, Z = {z}")
            for (label, _), ls in zip(variants, ("-", ":")):
                ours = [max(res[label][(site, av, z, p)][0], 1e-20) for p in T.PARENTS]
                ax.semilogy(x, ours, ls, color=color, lw=1.2)
        for (label, _), ls in zip(variants, ("-", ":")):
            ax.plot([], [], ls, color="0.3", label=f"pahchem, {label} (colour = case)")
        ax.set_xticks(x, T.PARENTS, rotation=60, fontsize=7)
        ax.set_ylabel("C$_2$H$_2$ per molecule in $10^6$ yr")
        ax.set_title(site.replace("_", " "), fontsize=9)
        ax.legend(fontsize=6, ncol=3)
    fig.suptitle("Murga+2020 Fig. 4: full destruction (Orion A_V = 0.1, Z = 0) is exact; partial destruction is not",
                 fontsize=10)
    fig.tight_layout()
    _save(fig, out, "murga2020_fig4")


def fig_murga2022_charge(args, out):
    _checks()
    import check_murga_charge as CC

    from pycalima.pahchem import murga_charge as C

    t = CC.m22_table()
    p = pd.read_csv(EXT("pahchem/murga2022_fig6_profile.csv"), comment="#")
    t["distance"] = p.distance_arcsec.to_numpy()[t["loc"].to_numpy()]
    hii = p.chi.to_numpy() >= 3.99e4
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, a in enumerate((3.3, 5.0, 7.8, 11.7, 19.9)):
        g = t[np.isclose(t.a_A, a)].sort_values("distance")
        ax.plot(g.distance, g.z_murga, color=f"C{i}", lw=4, alpha=0.35, label=f"Murga+22 Figshare, a = {a} A")
        ax.plot(g.distance, g.z_ours, "--", color=f"C{i}", label=f"pahchem (WD01, inferred config.), a = {a} A")
        h = g[hii[g["loc"].to_numpy()]]
        z_euv = [float(np.dot(*C.f_wd01(a * 1e-8, p.chi[l], p.t_gas[l], p.n_e[l], z_range=C.Z_GRID_2022,
                                        euv_factor=C.EUV_FACTOR_HII, **C.MURGA2022))) for l in h["loc"]]
        ax.plot(h.distance, z_euv, "o", ms=3, color=f"C{i}")
    ax.plot([], [], "o", ms=3, color="0.3", label=f"HII with O6 EUV (Kurucz 40 kK x {C.EUV_FACTOR_HII:g})")
    for x, lab in ((113, "IF"), (130, "DF")):
        ax.axvline(x, color="0.6", lw=0.8)
        ax.text(x + 0.5, ax.get_ylim()[1] * 0.9, lab, fontsize=7, color="0.4")
    ax.set_xlabel("distance [arcsec]")
    ax.set_ylabel("$\\langle Z \\rangle$")
    ax.set_title("Murga+2022: WD01 charge along the Orion Bar (dots: HII region with the O6 EUV added)", fontsize=9)
    ax.legend(fontsize=6, ncol=2)
    _save(fig, out, "murga2022_charge")


def fig_murga2022_model_b(args, out):
    _checks()
    import check_murga2022 as C22
    from pycalima.pahchem import murga2022 as N
    from pycalima.pahchem import murga_data as MD

    d = MD.read_model_2022("b")
    prof = pd.read_csv(EXT("pahchem/murga2022_fig6_profile.csv"), comment="#")
    R = N.photo_loss_table(d["a_cm"])
    ours = N.run_model_b(d, prof, R)
    n0 = d["n"][:, :, 0].sum(axis=2)
    w = d["f"][:, :, 0, 0, :] * n0                                       # loc, Z, size
    th = C22.fractions(d["n"][:, :, -1])
    xb = N.X_BINS[None, None, :, None]
    ours_x = (ours[:, :, -1] * xb * w[:, :, None, :]).sum(axis=(1, 2, 3)) / w.sum(axis=(1, 2))
    their_x = (th * xb * w[:, :, None, :]).sum(axis=(1, 2, 3)) / w.sum(axis=(1, 2))
    dist = prof.distance_arcsec.to_numpy()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    a1.plot(dist, their_x, color="k", lw=4, alpha=0.35, label="Murga+22 model b (Figshare)")
    a1.plot(dist, ours_x, "--", color="k", label="pahchem.murga2022 model b")
    a1.set_xlabel("distance [arcsec]")
    a1.set_ylabel("$\\langle X_{\\rm H}/X_{\\rm H0} \\rangle$ (n f-weighted, Fig. 8 quantity)")
    a1.set_title("Mean hydrogenation at 1e5 yr", fontsize=9)
    a1.legend(fontsize=7)
    for i, (m, a) in enumerate(((0, 3.3), (3, 7.0), (6, 11.7))):
        ww = w[:, :, m][:, :, None]
        o = (ours[:, :, -1, :, m] * ww).sum(axis=1) / ww.sum(axis=1)       # loc, bin
        t = (th[:, :, :, m] * ww).sum(axis=1) / ww.sum(axis=1)
        for j, ls in ((0, "-"), (2, "--"), (3, ":")):
            a2.plot(dist, t[:, j], ls, color=f"C{i}", lw=3, alpha=0.35)
            a2.plot(dist, o[:, j], ls, color=f"C{i}", lw=1,
                    label=f"{a} A, bin X_H/X_H0 = {N.X_BINS[j]:.2f} (thick: Figshare)")
    a2.set_xlabel("distance [arcsec]")
    a2.set_ylabel("fraction in bin")
    a2.set_title("Bins: lowest (solid), normal (dashed), first super-H (dotted)", fontsize=9)
    a2.legend(fontsize=6)
    fig.suptitle("Murga+2022 model b (H loss only): small PAHs reproduced; >= 6 A over-super-hydrogenated (open)",
                 fontsize=10)
    _save(fig, out, "murga2022_model_b")


def fig_emission_cascade(args, out):
    """Per-photon IR emission: the exact cascade against the AmesPAHdb thermal cascade."""
    from pycalima.pahchem import emission

    ref = pd.read_csv(EXT("pahchem/amespahdb_cascade_band_fractions.csv"), comment="#")
    grid = np.linspace(500.0, 3300.0, 5601)
    wl = 1e4 / grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), gridspec_kw={"width_ratios": [2.2, 1]})
    for row, formula in zip(axes, ("C24H12", "C54H18")):
        sp = PAHSpecies.from_states_file(args.states_dir / f"{formula}_0.dat")
        cm = energy_me.CascadeModel(sp.freqs_cm, sp.einstein_a, e_max_cm=13 * dos.CM_PER_EV, de_cm=20.0)
        N = emission.cascade_photons(cm)
        r = ref[ref.species == formula]
        bands = list(dict.fromkeys(r.band))
        for e, c in ((2.0, "C0"), (8.0, "C3")):
            j = int(round(e * dos.CM_PER_EV / cm.de_cm))
            ours = sp.freqs_cm * N[:, j] / cm.energy_cm[j]
            row[0].plot(wl, emission.spectrum(sp.freqs_cm, ours, grid) * grid**2 / 1e4, color=c, lw=1.2,
                        label=f"{e:.0f} eV")
            rb = r[r.e_ev == e].set_index("band")
            ob = [ours[(sp.freqs_cm >= rb.lo_cm[b]) & (sp.freqs_cm < rb.hi_cm[b])].sum() for b in bands]
            row[1].plot(bands, np.array(ob) / rb.energy_fraction[bands].to_numpy(), "o-", color=c, label=f"{e:.0f} eV")
        row[0].set_ylabel(r"$(dE/d\lambda)\,/\,E$ [$\mu$m$^{-1}$]")
        row[0].set_title(f"{formula}: per-photon emission, exact cascade", fontsize=10)
        row[0].set_xlim(3, 20)
        row[0].legend(fontsize=8)
        row[1].axhline(1.0, color="k", lw=0.8)
        row[1].set_ylabel("pahchem / AmesPAHdb")
        row[1].set_title("band fraction vs the thermal cascade", fontsize=10)
        row[1].set_ylim(0.7, 1.1)
        row[1].legend(fontsize=8)
    axes[1, 0].set_xlabel(r"$\lambda$ [$\mu$m]")
    axes[1, 1].set_xlabel("band [$\\mu$m]")
    _save(fig, out, "emission_cascade")


def fig_maragkoudakis2020_fig2(args, out):
    """Maragkoudakis+20 Fig. 2 (ISRF): their markers, their model (thermal) and pahchem (exact)."""
    from pycalima.pahchem import maragkoudakis2020 as M
    from pycalima.pahchem import ames_pahdb as pahdb

    species = pahdb.read_ascii(pahdb.ascii_path(), M.sample().uid)
    rows = []
    for uid, sp in species.items():
        for model in ("thermal", "exact"):
            b = M.band_fluxes(sp, model, energies_ev=())["isrf"]
            rows.append(dict(nc=sp.nc, charge=sp.charge, model=model, ratio=b["11.2"] / b["3.3"]))
    t = pd.DataFrame(rows)
    fig2 = pd.read_csv(EXT("pahchem/maragkoudakis2020_fig2.csv"), comment="#")
    f, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, panel in zip(axes, ("left", "right")):
        for z, group, c in ((0, "neutral_11.2/3.3", "C0"), (1, "cation_CHoop/3.3", "C3"), (-1, "anion_CHoop/3.3", "C2")):
            g = fig2[(fig2.panel == panel) & (fig2.group == group)]
            if g.empty:
                continue
            ax.plot(g.nc, g.ratio, "o", mfc="none", color=c, ms=7, label=f"{group.split('_')[0]}: paper")
            for model, mk in (("thermal", "+"), ("exact", "x")):
                o = t[(t.charge == z) & (t.model == model)]
                ax.plot(o.nc, o.ratio, mk, color=c, ms=5, label=f"{group.split('_')[0]}: pahchem {model}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$N_{\rm C}$")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("11.2/3.3 (neutral), CH$_{oop}$/3.3 (ions)")
    f.suptitle("Maragkoudakis+20 Fig. 2 (ISRF): + their thermal model, x pahchem exact cascade", fontsize=10)
    _save(f, out, "maragkoudakis2020_fig2")


def fig_rigopoulou2021_fig1(args, out):
    """Rigopoulou+21 Fig. 1: UID 742 at 5 and 12.4 eV."""
    from pycalima.pahchem import ames_pahdb as pahdb
    from pycalima.pahchem import emission
    from pycalima.pahchem import rigopoulou2021 as R

    sp = pahdb.read_ascii(pahdb.ascii_path(), [R.FIG1_UID])[R.FIG1_UID]
    dig = R.fig1_digitised()
    th = emission.ThermalCascade(sp.freqs_cm, sp.einstein_a)
    grid = R.GRID_CM
    f, ax = plt.subplots(figsize=(10, 5))
    spectra = {k: emission.spectrum(sp.freqs_cm, th.energy_per_mode(e * dos.CM_PER_EV), grid, 15.0, 0.0)
               for k, e in R.FIG1_ENERGIES_EV.items()}
    scale = dig[dig.curve == "12eV"].intensity.max() / spectra["12eV"].max()
    for k, c in (("5eV", "C0"), ("12eV", "C3")):
        g = dig[dig.curve == k]
        ax.plot(g.lambda_um, g.intensity, color=c, lw=2.5, alpha=0.35, label=f"paper, {k}")
        ax.plot(1e4 / grid, spectra[k] * scale, color=c, lw=0.9, label=f"thermal cascade, FWHM 15 cm$^{{-1}}$, {k}")
    ax.set_xlim(3, 14)
    ax.set_xlabel(r"$\lambda$ [$\mu$m]")
    ax.set_ylabel("relative intensity (per cm$^{-1}$)")
    ax.set_title("Rigopoulou+21 Fig. 1: PAHdb UID 742 (C$_{210}$H$_{36}$), one common scale", fontsize=10)
    ax.legend(fontsize=8)
    _save(f, out, "rigopoulou2021_fig1")


def fig_family_emission(args, out):
    """M7b: population-weighted spectra and band ratios of the C54 and C96 families."""
    from pycalima.pahchem.baseline import NGC7023_EDGE, ORION_BAR_ATOMIC, BaselineModel, Environment
    from pycalima.pahchem.family_emission import FamilyEmission, band_fluxes, load_superh_species

    grid = np.arange(500.0, 3500.0, 0.5)
    fams = {}
    for parent, formula, charges in (("C54", "C54H18", (-1, 0, 1, 2)), ("C96", "C96H24", (-1, 0, 1, 2, 3))):
        sp = {z: PAHSpecies.from_states_file(args.states_dir / f"{formula}_{z}.dat") for z in charges}
        sh = load_superh_species(args.states_dir / "superh", formula)
        fams[parent] = FamilyEmission(BaselineModel(parent, sp[0], anion=sp[-1]), sp, sh)
    f, axes = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1.4, 1.4, 1]})
    fam = fams["C54"]
    for ax, env in zip(axes[:2], (NGC7023_EDGE, ORION_BAR_ATOMIC)):
        fr = fam.fractions(env)
        total, _ = fam.spectrum(env, grid, fr)
        ax.plot(1e4 / grid, total * grid**2 / 1e4, "k", lw=1.2, label="family")
        for z in fam.model.charges:
            part = {k: (v if k[0] == z else 0.0) for k, v in fr.items()}
            fz = sum(part.values())
            s, _ = fam.spectrum(env, grid, part)
            ax.plot(1e4 / grid, s * grid**2 / 1e4, lw=0.8, label=f"Z = {z:+d} (f = {fz:.2f})")
        ax.set_xlim(3, 15)
        ax.set_xlabel(r"$\lambda$ [$\mu$m]")
        ax.set_ylabel(r"$dP/d\lambda$ per molecule [cm$^{-1}$ s$^{-1}$ $\mu$m$^{-1}$]")
        ax.set_title(f"C$_{{54}}$H$_{{18}}$ family, {env.name}", fontsize=9)
        ax.legend(fontsize=7)
    g0s = np.logspace(1, 5, 9)
    for parent, c in (("C54", "C0"), ("C96", "C3")):
        r62, r33 = [], []
        for g0 in g0s:
            b = band_fluxes(grid, fams[parent].spectrum(Environment("t", g0, 1e4, 3.0, 300.0, teff_k=None), grid)[0])
            r62.append(b["6.2"] / b["11.2"])
            r33.append(b["3.3"] / b["11.2"])
        axes[2].loglog(g0s, r62, "o-", color=c, label=f"{parent}: 6.2/11.2")
        axes[2].loglog(g0s, r33, "s--", color=c, label=f"{parent}: 3.3/11.2")
    axes[2].set_xlabel(r"$G_0$ ($n_{\rm H} = 10^4$, $n_e = 3$, $T = 300$ K)")
    axes[2].set_ylabel("band ratio")
    axes[2].set_title("family band ratios: charge, then dehydrogenation", fontsize=9)
    axes[2].legend(fontsize=7)
    _save(f, out, "family_emission")


def fig_rf_borja2026(args, out):
    """Recurrent fluorescence: pahchem's rate with Borja+26 Table I inputs against their Fig. 8."""
    t = pd.read_csv(EXT("pahchem/borja2026_table1.csv"), comment="#")
    curves = pd.read_csv(EXT("pahchem/borja2026_fig8_krf.csv"), comment="#")
    files = {"Np+": "C10H8_1", "An+": "C14H10_1", "Py+": "C16H10_1"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    e_ev = np.linspace(2.0, 13.5, 116)
    for ax, (mol, fname) in zip(axes, files.items()):
        sp = PAHSpecies.from_states_file(args.states_dir / f"{fname}.dat")
        E, counts = dos.beyer_swinehart(sp.freqs_cm, 15 * dos.CM_PER_EV, 2.0)
        rows = t[t.molecule == mol].set_index("state")
        d0 = rows.loc["D0", "transition_energy_exp_eV"]
        T = dos.canonical_temperature(sp.freqs_cm, e_ev * dos.CM_PER_EV)
        for k, state in enumerate(s for s in rows.index if s != "D0"):
            r = rows.loc[state]
            f_fc = 0.0 if np.isnan(r.fFC_x1e3) else r.fFC_x1e3 * 1e-3
            f = f_fc + 1e-3 * (r.fHT_0K_x1e3 + (r.fHT_2000K_x1e3 - r.fHT_0K_x1e3) * np.minimum(T, 2000.0) / 2000.0)
            k_rf = [cooling.recurrent_fluorescence_rates(E, counts, [(r.transition_energy_exp_eV - d0) * dos.CM_PER_EV],
                                                         [fi], window_cm=50.0, at_cm=[ei * dos.CM_PER_EV])[0, 0]
                    for ei, fi in zip(e_ev, f)]
            c = curves[(curves.molecule == mol) & (curves.state == state)]
            col = f"C{k}"
            ax.semilogy(c.e_ev, 10 ** c.log10_k_rf, color=col, lw=2.5, alpha=0.4,
                        label=f"{state}: Borja+26 ({'allowed' if f_fc else 'HT only'})")
            ax.semilogy(e_ev, k_rf, "--", color=col, lw=1.2, label=f"{state}: pahchem")
        ax.set_title(mol, fontsize=10)
        ax.set_xlabel("internal energy [eV]")
        ax.set_ylim(1e-2, 1e5)
        ax.legend(fontsize=7)
    axes[0].set_ylabel(r"$k_{\rm RF}$ [s$^{-1}$]")
    fig.suptitle("Recurrent fluorescence: k = 0.667 nu^2 f rho(E-E_n)/rho(E) with their Table I inputs, "
                 "against their full Herzberg-Teller/Duschinsky treatment", fontsize=10)
    _save(fig, out, "rf_borja2026")


def fig_ere(args, out):
    """Extended Red Emission test: RF photon efficiency and ERE/IR power against observations."""
    _checks()
    import check_ere as E

    from pycalima.pahchem.baseline import BaselineModel
    from pycalima.pahchem.family_emission import FamilyEmission

    sp = {z: PAHSpecies.from_states_file(args.states_dir / f"C24H12_{z}.dat") for z in (-1, 0, 1)}
    fam = FamilyEmission(BaselineModel("C24", sp[0], anion=sp[-1]), sp)
    names = [e.name for e in E.ENVIRONMENTS]
    res = [E.ere_observables(fam, env) for env in E.ENVIRONMENTS]
    obs_eta = {"diffuse ISM": (0.10, 1.0), "IC 63-like (G0=650)": (0.047, 0.047),
               "NGC 7023 NW edge (A_V ~ 0)": (0.002, 0.006), "Orion Bar atomic PDR": (0.004, 0.013)}
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].semilogy(x, [r["eta"] for r in res], "o-", color="C3", label="pahchem: C24 family RF at 540-900 nm")
    for i, n in enumerate(names):
        lo, hi = obs_eta[n]
        axes[0].plot([i, i], [lo, hi], color="k", lw=6, alpha=0.4, solid_capstyle="butt",
                     label="observed (Gordon+98; Smith & Witt 02)" if i == 0 else None)
    axes[0].set_ylabel("ERE photons per absorbed UV photon")
    axes[0].set_title("ERE efficiency: model rises with U, observed falls", fontsize=10)
    axes[1].semilogy(x, [r["ere_over_ir"] for r in res], "o-", color="C3", label="pahchem")
    axes[1].axhspan(0.0035, 0.06, color="k", alpha=0.15, label="observed ERE/mid-IR (Berne+08: 0.35-1.6 %, <~ 6 %)")
    axes[1].set_ylabel("ERE / IR power")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([n.split(" (")[0] for n in names], rotation=20, fontsize=8)
        ax.legend(fontsize=7)
    _save(fig, out, "ere_test")


def fig_pdrs4all_scorecard(args, out):
    """PDRs4All Orion Bar: observed band ratios against the model range over the stated conditions."""
    _checks()
    import compare_pdrs4all as P

    from pycalima.pahchem import conditions
    from pycalima.pahchem.baseline import BaselineModel
    from pycalima.pahchem.family_emission import FamilyEmission, load_superh_species

    obs = pd.read_csv(EXT("pahchem/pdrs4all_orionbar_band_ratios.csv"), comment="#").set_index("ratio")
    grid = np.arange(500.0, 3500.0, 0.5)
    ratios = ["3.3/11.2", "6.2/11.2", "7.7/11.2", "8.6/11.2", "11.0/11.2", "12.7/11.2", "3.4/3.29"]
    fams = {}
    for parent, (formula, charges) in P.FAMILIES.items():
        sp = {z: PAHSpecies.from_states_file(args.states_dir / f"{formula}_{z}.dat") for z in charges}
        fams[parent] = FamilyEmission(BaselineModel(parent, sp[0], anion=sp[-1]), sp,
                                      load_superh_species(args.states_dir / "superh", formula))
    model = {}
    for t in P.TEMPLATES:
        for parent, fam in fams.items():
            for corner, env in conditions.bracketing_environments("orion_bar", t).items():
                b = P.model_bands(grid, fam.spectrum(env, grid)[0])
                model[(t, parent, corner)] = {r: b[r.split("/")[0]] / b[r.split("/")[1]] for r in ratios}
    fig, axes = plt.subplots(1, len(ratios), figsize=(19, 4.2))
    x = np.arange(len(P.TEMPLATES))
    for ax, r in zip(axes, ratios):
        ax.plot(x, obs.loc[r, list(P.TEMPLATES)].to_numpy(float), "ks", ms=7, label="observed")
        for dx, (parent, c) in zip((-0.15, 0.15), (("C54", "C0"), ("C96", "C3"))):
            lo = np.array([model[(t, parent, "least ionising")][r] for t in P.TEMPLATES])
            hi = np.array([model[(t, parent, "most ionising")][r] for t in P.TEMPLATES])
            ax.vlines(x + dx, np.minimum(lo, hi), np.maximum(lo, hi), color=c, lw=3, alpha=0.6,
                      label=f"{parent}: range over stated conditions")
            ax.plot(x + dx, lo, "o", color=c, ms=4)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(P.TEMPLATES, rotation=45, fontsize=8)
        ax.set_title(r, fontsize=10)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("PDRs4All Orion Bar templates (squares) vs pahchem families; dots: least-ionising corner "
                 "(environments/orion_bar.toml)", fontsize=10)
    _save(fig, out, "pdrs4all_scorecard")


FIGURES = {f.__name__[4:]: f for f in (
    fig_west18, fig_jochims94, fig_montillaud13_fig3, fig_andrews16_fig5, fig_andrews16_tableB,
    fig_montillaud13_fig10, fig_andrews16_ngc7023, fig_murga_wiebe22_fig1, fig_sensitivity,
    fig_allain96_fig7, fig_allain96_fig10, fig_lepage01_fig10, fig_lepage03_fig6_7,
    fig_murga2019_shiva_inputs, fig_murga_wiebe22_shiva, fig_murga2020_fig3, fig_murga2020_fig4,
    fig_murga2022_charge, fig_murga2022_model_b, fig_emission_cascade, fig_maragkoudakis2020_fig2,
    fig_rigopoulou2021_fig1, fig_family_emission, fig_rf_borja2026, fig_ere, fig_pdrs4all_scorecard)}


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
