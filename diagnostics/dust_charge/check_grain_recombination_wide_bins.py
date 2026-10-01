#!/usr/bin/env python3
"""How much grain-assisted recombination is lost by skipping wide-P(Z) dust bins?

For RAMSES the recombination sum of a dust bin with a wide charge distribution
(sigma_Z >= SIGMA_DISCRETE) is expensive (rt_group_charging._recombination_fit).
This script measures f_skip,i = (wide dust bins) / (all dust bins) of the rate
coefficient per H of each ion X+, i.e. the largest error that skipping those
bins can make in the contribution of the dust grains (PAHs are a separate part
of CALIMA and are left out), in three sets of cells:

  grid        synthetic: neutral ISM in the Mathis ISRF (G0 = 1e-3 - 1e4),
              photoionised gas lit by 3e4-1e5 K blackbodies, hot gas (1e5-1e8 K)
  equilibrium the RAMSES RTZ equilibrium tests at G0 = 1 and 1e-3 (their T,
              ion fractions and dust densities)
  stromgren   the 2D RAMSES Stromgren test (tests/rt/stromgren2d_rtz_calima),
              all snapshots (local RT groups, ion fractions, dust densities)

and, where RAMSES ion fractions are known, the fraction of the element in X+
(if X+ is a minor stage, a wrong X+ rate barely matters). With --gas-recomb (a
table of the RAMSES RTZ X+ -> X radiative + dielectronic rates, columns T and
the ions of RECOMB_IONS) it also gives the error that skipping makes in the
total X+ recombination rate, impact = f_skip R / (1 + R) with
R = n_H alpha_dust / (n_e alpha_gas) (charge exchange is ignored, so this is an
upper bound). Dust bins use the pyCALIMA RT-group tables.

It also tests the proposed RAMSES rule for the wide bins ("switch"): where the
ion capture rate A(Z) is smooth across P(Z), sigma |d ln A/dZ| < SWITCH, take
the Gaussian estimate A(mu) exp(b^2 s^2/2) Phi((Zth + 1/2 - mu - b s^2)/s)
(mu = Z* + 1/2, b = d ln A/dZ across mu +- sigma), otherwise skip the bin;
an ion whose threshold lies more than TAIL_CUT widths below the shifted mean,
(Zth + 1/2 - mu - b s^2)/s < -TAIL_CUT, gets zero (the Gaussian tail says
nothing there and the true rate is negligible).

And a per-bin-accurate alternative ("coarse"): where sigma |d ln A/dZ| < COARSE_SKIP,
sample d ln P/dZ = g(Z - 1/2) every h = max(1, sigma min(1, 2/(sigma |d ln A/dZ|)))
charges (so that ln A changes by at most ~2 per step) from mu both ways (until
P and P A are e^-30 below their peaks), take g linear between samples so that
ln P is piecewise quadratic, and sum P A Theta over the integer charges; skip the
bin beyond COARSE_SKIP (cold gas, where its rate is negligible).

    python check_grain_recombination_wide_bins.py --equilibrium DIR --stromgren DIR --stromgren-plot SCRIPT
"""
import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pycalima import _paths
from pycalima.models.dust_charge import rt_group_charging as rg

M_P, C = 1.67262192e-24, 2.99792458e10
X_H, Y_HE = 0.73738783, 0.24924204
DUST = ("DustBin_01", "DustBin_02", "DustBin_03", "DustBin_04")
RHO_GR = {"graphite": 2.2, "silicate": 3.3}
# Milky Way mass fractions of the RAMSES RTZ setups (dust; gas-phase metals after depletion)
MW_DUST = (5.106243e-4, 5.106243e-4, 1.835089e-3, 1.835089e-3)
GAS_METALS = {"C": 1.185171e-3, "N": 6.929047e-4, "O": 4.368967e-3, "Ne": 1.256482e-3, "Mg": 1.900819e-4,
              "Si": 6.649476e-5, "S": 3.092675e-4, "Fe": 1.016061e-4}
AMU = {el: m for el, m, _ in rg.RECOMB_IONS}
NSTAGE = {"H": 2, "He": 3, "C": 7, "N": 6, "O": 6, "Ne": 6, "Mg": 6, "Si": 6, "S": 6, "Fe": 6}
ELEM_Z = {"H": 1, "He": 2, "C": 6, "N": 7, "O": 8, "Ne": 10, "Mg": 12, "Si": 14, "S": 16, "Fe": 26}
IONS = [el for el, _, _ in rg.RECOMB_IONS]
SWITCH = 1.0
TAIL_CUT = 3.0
COARSE_SKIP = 10.0


GAS_RECOMB = None     # (log10 T, log10 alpha) of the RAMSES gas-phase X+ -> X rates


def alpha_gas(T, k):
    lt, la = GAS_RECOMB
    return 10.0 ** np.interp(np.log10(T), lt, la[:, k])


def load_tables():
    tabdir = _paths.get_model_data_dir() / "dust_charging_rtgroups_data"
    return [rg.read_bin_table(tabdir / f"dust_charging_rtgroups_{b}.dat") for b in DUST]


def n_per_H(rho_frac, tab, rho_gr):
    """grains per H nucleus for a bin mass fraction (per unit gas mass)."""
    return rho_frac * M_P / X_H / (4.0 / 3.0 * np.pi * tab["a"] ** 3 * rho_gr)


def ln_capture_z1(gt, Z):
    """ln of pi a^2 (8kT/pi m_u)^1/2 J~(Z, z=1) in log form (DS87, theta_nu =
    nu/(1+nu^-1/2)), so that the repulsive branch cannot underflow."""
    a, T = gt.a, gt.cell["T"]
    tau = a * rg.KB_CGS * T / rg.E_STATC ** 2
    if Z > 0:
        th = Z / (1.0 + 1.0 / np.sqrt(Z))
        lnJ = 2.0 * np.log(1.0 + 1.0 / np.sqrt(4.0 * tau + 3.0 * Z)) - th / tau
    elif Z < 0:
        lnJ = np.log((1.0 - Z / tau) * (1.0 + np.sqrt(2.0 / (tau - 2.0 * Z))))
    else:
        lnJ = np.log(1.0 + np.sqrt(np.pi / (2.0 * tau)))
    return np.log(np.pi * a * a * np.sqrt(8.0 * rg.KB_CGS * T / (np.pi * rg.M_U))) + lnJ


def gaussian_estimate(gt, Zs, sigma):
    """Recombination coefficients of a wide P(Z) as a Gaussian of mean Z* + 1/2
    and width sigma with ln A linear across mu +- sigma; also sigma |d ln A/dZ|."""
    from math import erf, sqrt
    t = gt.t
    mu = Zs + 0.5
    zl, zh = max(mu - sigma, t["Zmin"]), min(mu + sigma, t["Zmax"])
    b = (ln_capture_z1(gt, zh) - ln_capture_z1(gt, zl)) / (zh - zl) if zh > zl else 0.0
    pref = np.exp(min(ln_capture_z1(gt, mu) + 0.5 * (b * sigma) ** 2, 700.0))
    cdf = lambda x: 0.5 * (1.0 + erf(x / (sqrt(2.0) * sigma)))  # noqa: E731
    lower = cdf(t["Zmin"] - 0.5 - mu - b * sigma ** 2)
    mass = np.array([m for _, m, _ in rg.RECOMB_IONS])
    x = (t["recomb_Zth"] + 0.5 - mu - b * sigma ** 2) / sigma
    est = np.array([0.0 if xi < -TAIL_CUT else pref * max(cdf(xi * sigma) - lower, 0.0) for xi in x])
    return est / np.sqrt(mass), sigma * abs(b)


def coarse_sum(gt, Zs, sigma, sb, cut=30.0, max_nodes=64):
    """Coarse-sampled detailed-balance recombination sum; also the g(Z) calls.
    sb: sigma |d ln A/dZ| (sets the step)."""
    t = gt.t
    Zlo, Zhi = t["Zmin"], t["Zmax"] - 1.0
    mu, h = Zs + 0.5, max(1.0, sigma * min(1.0, 2.0 / max(sb, 1e-30)))
    X, G, LP = [mu], [0.0], [0.0]                 # d ln P/dZ at mu is g(Z*) = 0
    vmax, ncall = ln_capture_z1(gt, mu), 0
    for sgn in (1.0, -1.0):
        x, gprev, lp = mu, 0.0, 0.0
        for _ in range(max_nodes):
            xn = x + sgn * h
            if xn - 0.5 > Zhi or xn - 0.5 < Zlo:
                break
            gn = gt._g(xn - 0.5)
            ncall += 1
            lp += sgn * h * 0.5 * (gprev + gn)     # exact for g linear between samples
            x, gprev = xn, gn
            v = lp + ln_capture_z1(gt, x)
            vmax = max(vmax, v)
            X.append(x), G.append(gn), LP.append(lp)
            if lp < -cut and v < vmax - cut:
                break
    o = np.argsort(X)
    X, G, LP = np.array(X)[o], np.array(G)[o], np.array(LP)[o]
    Z = np.arange(np.ceil(X[0]), np.floor(X[-1]) + 1.0)
    j = np.clip(np.searchsorted(X, Z) - 1, 0, max(len(X) - 2, 0))
    d = Z - X[j]
    hh = np.where(len(X) > 1, X[np.minimum(j + 1, len(X) - 1)] - X[j], 1.0)
    dG = G[np.minimum(j + 1, len(X) - 1)] - G[j]
    lnP = LP[j] + G[j] * d + dG * d * d / (2.0 * np.maximum(hh, 1e-30))
    P = np.exp(lnP - lnP.max())
    return gt._recombination(Z, P / P.sum()), ncall


def cell_rates(dust, N, egy, c_red, T, ne, ions_charge, G0_bg, fd):
    """alpha per H of RECOMB_IONS: all dust bins (exact), wide dust bins (exact),
    wide dust bins with the switch; and the per-bin sigma. fd: dust mass fractions
    (per unit gas mass)."""
    tot = np.zeros(len(IONS))
    wide = np.zeros(len(IONS))
    corr = np.zeros(len(IONS))
    sig = []
    per_bin = []
    for ib, (tab, f) in enumerate(zip(dust, fd)):
        gt = rg.GroupChargingTable(tab, egy=egy)
        r = gt.solve(N, c_red, T, ne, ions_charge, G0_bg=G0_bg, recomb="exact")
        n = n_per_H(f, tab, RHO_GR[tab["material"]])
        tot += n * r["alpha_rec"]
        if not r["discrete"]:
            wide += n * r["alpha_rec"]
            est, sb = gaussian_estimate(gt, r["Zstar"], r["Zsigma"])
            used = sb < SWITCH
            if used:
                corr += n * est
            if sb < COARSE_SKIP:
                co, nc = coarse_sum(gt, r["Zstar"], r["Zsigma"], sb)
            else:
                co, nc = 0.0 * est, 0
            geo = np.pi * tab["a"] ** 2 * np.sqrt(8.0 * rg.KB_CGS * T / (np.pi * np.array(
                [m for _, m, _ in rg.RECOMB_IONS]) * rg.M_U))
            per_bin.append(dict(bin=ib, sb=float(sb), used=bool(used), sigma=float(r["Zsigma"]),
                                exact=r["alpha_rec"].tolist(), switch=(est if used else 0.0 * est).tolist(),
                                coarse=co.tolist(), ncall=int(nc), geo=geo.tolist()))
        sig.append(r["Zsigma"])
    cell_rates.per_bin = per_bin
    return tot, wide, corr, sig


def bb_groups(Ts, G0, L0, L1):
    E = np.geomspace(0.1, 500.0, 6000)
    uE = E ** 3 / np.expm1(E / (8.617333262e-5 * Ts))
    b = (E >= 6) & (E <= 13.6)
    uE *= G0 * rg.HABING_FLUX / C / np.trapezoid(uE[b], E[b])
    N, egy = [], []
    for l0, l1 in zip(L0, L1):
        m = (E >= l0) & (E < l1)
        n = np.trapezoid(uE[m] / (E[m] * rg.EV2ERG), E[m])
        N.append(n)
        egy.append(np.trapezoid(uE[m], E[m]) / rg.EV2ERG / n)
    return np.array(N), np.array(egy)


def run_grid(dust):
    rows = []
    L0, L1 = dust[0]["L0"], dust[0]["L1"]
    egy0 = dust[0]["egy"][:, dust[0]["egy"].shape[1] // 2]
    nG = L0.size
    for G0 in (1e-3, 1.0, 1e2, 1e4):                       # neutral ISM, Mathis ISRF
        for T in np.geomspace(10.0, 1e4, 13):
            for ne in np.geomspace(1e-4, 1.0, 9):
                tot, wide, corr, sig = cell_rates(dust, np.zeros(nG), egy0, 0.0, T, ne, [(ne, 1.0, M_P)], G0,
                                            MW_DUST)
                rows.append(dict(set="grid: neutral ISM", T=T, ne=ne, G0=G0, tot=tot, wide=wide, corr=corr, bins=cell_rates.per_bin, sig=sig))
    for Ts in (3e4, 4e4, 1e5):                              # photoionised gas
        for G0 in (1.0, 1e2, 1e4):
            N, egy = bb_groups(Ts, G0, L0, L1)
            for T in np.geomspace(5e3, 2e4, 4):
                for ne in np.geomspace(1e-1, 1e3, 5):
                    tot, wide, corr, sig = cell_rates(dust, N, egy, C, T, ne, [(ne, 1.0, M_P)], 0.0, MW_DUST)
                    rows.append(dict(set="grid: HII", T=T, ne=ne, nH=ne / 1.1, G0=G0, tot=tot, wide=wide, corr=corr, bins=cell_rates.per_bin, sig=sig))
    for T in np.geomspace(1e5, 1e8, 10):                    # hot gas
        for ne in np.geomspace(1e-3, 1.0, 4):
            tot, wide, corr, sig = cell_rates(dust, np.zeros(nG), egy0, 0.0, T, ne, [(ne, 1.0, M_P)], 1.0,
                                        MW_DUST)
            rows.append(dict(set="grid: hot", T=T, ne=ne, nH=ne / 1.2, G0=1.0, tot=tot, wide=wide, corr=corr, bins=cell_rates.per_bin, sig=sig))
    return rows


def _read_cols(path):
    rows = []
    for ln in open(path):
        try:
            rows.append([float(x) for x in ln.replace(",", " ").split()])
        except ValueError:
            continue
    n = min(len(r) for r in rows)
    return np.array([r[:n] for r in rows])


def electron_density(nH, frac):
    """n_e from the ion fractions frac[el] (n_rows x n_stages) and the gas-phase abundances."""
    ne = nH * frac["H"][:, 1]
    nHe = nH * Y_HE / (4.002602 * X_H) * 1.00794
    ne += nHe * (frac["He"][:, 1] + 2.0 * frac["He"][:, 2])
    for el, mf in GAS_METALS.items():
        n_el = nH * mf / (AMU[el] * X_H) * 1.00794
        ne += n_el * (frac[el] * np.arange(frac[el].shape[1])).sum(axis=1)
    return ne


def run_equilibrium(dust, root):
    rows = []
    egy0 = dust[0]["egy"][:, dust[0]["egy"].shape[1] // 2]
    for case, G0 in (("new_calima_cooling", 1.0), ("new_calima_lowG0", 1e-3)):
        d = Path(root) / case
        frac = {}
        n = None
        for el in IONS:
            e = _read_cols(d / f"element_{ELEM_Z[el]}_ions.dat")
            frac[el] = e[:, 3:3 + NSTAGE[el]]
            n = e.shape[0] if n is None else min(n, e.shape[0])
        du = _read_cols(d / "dust_rho.dat")
        n = min(n, du.shape[0])
        e = _read_cols(d / "element_1_ions.dat")[:n]
        nH, T = e[:, 0], e[:, 1]
        frac = {k: v[:n] for k, v in frac.items()}
        ne = electron_density(nH, frac)
        rho_gas = nH * M_P / X_H
        for k in range(0, n, 3):
            fd = du[k, 4:8] / rho_gas[k]
            tot, wide, corr, sig = cell_rates(dust, np.zeros(8), egy0, 0.0, T[k], ne[k], [(ne[k], 1.0, M_P)], G0, fd)
            rows.append(dict(set=f"equilibrium G0={G0:g}", T=T[k], ne=ne[k], nH=nH[k], G0=G0, tot=tot, wide=wide, corr=corr, bins=cell_rates.per_bin,
                             sig=sig, xplus={el: float(frac[el][k, 1]) for el in IONS}))
    return rows


def run_stromgren(dust, root, plot_script, n_cells=500, seed=2):
    spec = importlib.util.spec_from_file_location("strom", plot_script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rng = np.random.default_rng(seed)
    rows = []
    for out in sorted((Path(root) / "new").glob("output_*"))[1:]:
        s = mod.cell_state(out)
        cells = rng.choice(s["r"].size, size=min(n_cells, s["r"].size), replace=False)
        for c in cells:
            nH = s["n_H"][c]
            ions_charge = [(nH * s["x_H"][c, 1], 1.0, M_P), (s["n_He"][c] * s["x_He"][c, 1], 1.0, 6.6446573e-24),
                           (s["n_He"][c] * s["x_He"][c, 2], 2.0, 6.6446573e-24)]
            fd = s["rho_dust"][c] / s["rho"][c]
            tot, wide, corr, sig = cell_rates(dust, s["Np"][c], s["egy"], s["c_red"], s["T"][c], s["ne"][c],
                                        ions_charge, 0.0, fd)
            xplus = {el: float(s[f"x_{el}"][c, 1]) for el in IONS}
            rows.append(dict(set=f"stromgren t={s['t']:.1f} Myr", T=s["T"][c], ne=s["ne"][c], nH=nH, r=s["r"][c],
                             tot=tot, wide=wide, corr=corr, bins=cell_rates.per_bin, sig=sig, xplus=xplus))
    return rows


def errors(r, k):
    """(skip, switch) errors: of the dust-grain rate, and of the total X+ -> X
    recombination rate (NaN without n_H or gas rates)."""
    if r["tot"][k] <= 0:
        return 0.0, 0.0, np.nan, np.nan
    d_skip = r["wide"][k] / r["tot"][k]
    d_sw = abs(r["corr"][k] - r["wide"][k]) / r["tot"][k]
    if GAS_RECOMB is None or "nH" not in r:
        return d_skip, d_sw, np.nan, np.nan
    R = r["nH"] * r["tot"][k] / (r["ne"] * alpha_gas(r["T"], k))
    return d_skip, d_sw, d_skip * R / (1.0 + R), d_sw * R / (1.0 + R)


def summarise(rows):
    print("max (95th percentile) errors; 'total' = total X+ -> X recombination rate")
    print(f"{'set':24s} {'ion':3s} {'n':>4s} | {'dust: skip':>19s} {'switch':>19s} | {'total: skip':>19s} {'switch':>19s}")
    fmt = lambda v: f"{np.nanmax(v):8.1e} ({np.nanpercentile(v, 95):8.1e})" if np.any(np.isfinite(v)) else f"{'-':>19s}"  # noqa: E731
    for st in dict.fromkeys(r["set"] for r in rows):
        rs = [r for r in rows if r["set"] == st]
        for k, el in enumerate(IONS):
            e = np.array([errors(r, k) for r in rs])
            print(f"{st:24s} {el:3s} {len(rs):4d} | {fmt(e[:, 0]):>19s} {fmt(e[:, 1]):>19s} | "
                  f"{fmt(e[:, 2]):>19s} {fmt(e[:, 3]):>19s}")


def plot(rows, path):
    fams = (("grid: neutral", "#5b8def"), ("grid: HII", "#eb6834"), ("grid: hot", "#c0392b"),
            ("equilibrium", "#2e9e5b"), ("stromgren", "#8e5bd6"))
    fam = lambda st: next(f for f, _ in fams if st.startswith(f))  # noqa: E731
    ions = ("H", "He", "C", "Mg", "Si", "Fe")
    fig, axes = plt.subplots(2, len(ions), figsize=(3.9 * len(ions), 7.6), sharex=True, sharey=True, squeeze=False)
    floor = 1e-8
    for c, el in enumerate(ions):
        k = IONS.index(el)
        for row, (i_skip, i_sw, lab) in enumerate(((0, 1, "dust-grain rate"), (2, 3, "total X+ recombination rate"))):
            ax = axes[row, c]
            T = np.array([r["T"] for r in rows])
            e = np.array([errors(r, k) for r in rows])
            ok = np.isfinite(e[:, i_skip])
            ax.scatter(T[ok], np.maximum(e[ok, i_skip], floor), s=7, marker="x", color="0.72", lw=0.6,
                       label="skip wide bins" if c == 0 else None)
            for f, col in fams:
                m = ok & np.array([fam(r["set"]) == f for r in rows])
                ax.scatter(T[m], np.maximum(e[m, i_sw], floor), s=6, color=col, alpha=0.75, lw=0,
                           label=f"switch: {f}" if c == 0 else None)
            ax.axhline(0.01, color="0.4", lw=0.8, ls="--")
            ax.axhline(0.1, color="0.4", lw=0.8, ls=":")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim(floor * 0.7, 1e4)
            ax.grid(color="0.93", lw=0.6)
            if row == 0:
                ax.set_title(f"{el}+", fontsize=11)
            if row == 1:
                ax.set_xlabel("T (K)")
            if c == 0:
                ax.set_ylabel(f"relative error of the\n{lab}")
    axes[0, 0].legend(frameon=False, fontsize=7, loc="lower right", markerscale=2)
    fig.suptitle(f"Wide-P(Z) dust bins: skipped (grey) vs switch (Gaussian estimate if sigma|dlnA/dZ| < {SWITCH:g}, "
                 f"thresholds beyond {TAIL_CUT:g} widths set to 0, else skipped); floor 1e-8, lines 1% and 10%", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print("Saved", path)


def summarise_per_bin(rows):
    """Per wide bin and ion: |method - exact| / exact of that bin, over the cases
    where the bin's rate matters (exact > 1e-3 of its geometric rate pi a^2 v)."""
    nc = np.array([e["ncall"] for r in rows for e in r.get("bins", [])])
    print(f"per wide bin, cases with alpha > 1e-3 pi a^2 v; coarse g(Z) calls per wide bin: median {np.median(nc):.0f}, "
          f"95% {np.percentile(nc, 95):.0f}, max {nc.max()}")
    print(f"{'bin':10s} {'ion':3s} {'n':>5s} | {'Gaussian switch: med':>20s} {'95%':>8s} {'max':>8s} | "
          f"{'coarse: med':>11s} {'95%':>8s} {'max':>8s}")
    for ib, b in enumerate(DUST):
        for k, el in enumerate(IONS):
            ent = [e for r in rows for e in r.get("bins", []) if e["bin"] == ib and e["exact"][k] > 1e-3 * e["geo"][k]]
            if not ent:
                continue
            er = {m: np.array([abs(e[m][k] - e["exact"][k]) / e["exact"][k] for e in ent]) for m in ("switch", "coarse")}
            print(f"{b:10s} {el:3s} {len(ent):5d} | {np.median(er['switch']):20.1e} {np.percentile(er['switch'], 95):8.1e} "
                  f"{er['switch'].max():8.1e} | {np.median(er['coarse']):11.1e} {np.percentile(er['coarse'], 95):8.1e} "
                  f"{er['coarse'].max():8.1e}")


def plot_per_bin(rows, path):
    fams = (("grid: neutral", "#5b8def"), ("grid: HII", "#eb6834"), ("grid: hot", "#c0392b"),
            ("equilibrium", "#2e9e5b"), ("stromgren", "#8e5bd6"))
    fam = lambda st: next(f for f, _ in fams if st.startswith(f))  # noqa: E731
    ions = ("H", "He", "C", "Mg", "Si", "Fe")
    labels = {"DustBin_01": "graphite 0.01 um", "DustBin_02": "graphite 0.1 um",
              "DustBin_03": "silicate 0.005 um", "DustBin_04": "silicate 0.1 um"}
    floor = 1e-8
    for method, title in (("coarse", "coarse detailed-balance sum (step max(1, sigma min(1, 2/(sigma|dlnA/dZ|))); "
                                     f"skipped if sigma|dlnA/dZ| >= {COARSE_SKIP:g})"),
                          ("switch", f"Gaussian switch (sigma|dlnA/dZ| < {SWITCH:g}, tail cut {TAIL_CUT:g})")):
        fig, axes = plt.subplots(len(ions), len(DUST), figsize=(4.3 * len(DUST), 2.5 * len(ions)), sharex=True,
                                 sharey=True, squeeze=False)
        for c, b in enumerate(DUST):
            for rr, el in enumerate(ions):
                k = IONS.index(el)
                ax = axes[rr, c]
                for f, col in fams:
                    pts = [(r["T"], e) for r in rows if fam(r["set"]) == f for e in r.get("bins", [])
                           if e["bin"] == c and e["exact"][k] > 1e-300]
                    if not pts:
                        continue
                    T = np.array([p[0] for p in pts])
                    err = np.array([max(abs(e[method][k] - e["exact"][k]) / e["exact"][k], floor) for _, e in pts])
                    sig = np.array([e["exact"][k] > 1e-3 * e["geo"][k] for _, e in pts])
                    ax.scatter(T[sig], err[sig], s=7, color=col, alpha=0.8, lw=0,
                               label=f"{f}" if (c == 0 and rr == 0) else None)
                    ax.scatter(T[~sig], err[~sig], s=7, facecolors="none", edgecolors=col, alpha=0.35, lw=0.5)
                ax.axhline(0.01, color="0.4", lw=0.8, ls="--")
                ax.axhline(0.1, color="0.4", lw=0.8, ls=":")
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_ylim(floor * 0.7, 1e2)
                ax.grid(color="0.93", lw=0.6)
                if rr == 0:
                    ax.set_title(f"{b} ({labels[b]})", fontsize=10)
                if c == 0:
                    ax.set_ylabel(f"{el}+: |est - exact|/exact")
                if rr == len(ions) - 1:
                    ax.set_xlabel("T (K)")
        axes[0, 0].legend(frameon=False, fontsize=6, loc="upper left", markerscale=2)
        fig.suptitle(f"Per-bin error for wide-P(Z) bins: {title}. Filled: the bin's rate matters "
                     "(> 1e-3 of pi a^2 v); hollow: it does not. Lines 1%, 10%", fontsize=11)
        fig.tight_layout()
        out = path.with_name(path.stem + f"_{method}.png")
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print("Saved", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--equilibrium", help="RAMSES_dev/equilibrium/rtgroups_compare")
    ap.add_argument("--stromgren", help="run directory holding new/ of the 2D Stromgren test")
    ap.add_argument("--stromgren-plot", help="tests/rt/stromgren2d_rtz_calima/plot-stromgren2d-rtz.py (reader)")
    ap.add_argument("--gas-recomb", help="RAMSES RTZ X+ -> X rates: T, then the ions of RECOMB_IONS")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    global GAS_RECOMB
    if args.gas_recomb:
        g = np.loadtxt(args.gas_recomb)
        GAS_RECOMB = (np.log10(g[:, 0]), np.log10(g[:, 1:]))
    js = _paths.get_results_dir() / "grain_recombination_wide_bins.json"
    if not args.plot_only:
        dust = load_tables()
        rows = run_grid(dust)
        if args.equilibrium:
            rows += run_equilibrium(dust, args.equilibrium)
        if args.stromgren:
            rows += run_stromgren(dust, args.stromgren, args.stromgren_plot)
        js.parent.mkdir(parents=True, exist_ok=True)
        js.write_text(json.dumps([{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}
                                  for r in rows]))
    rows = [{k: (np.array(v) if k in ("tot", "wide", "corr") else v) for k, v in r.items()}
            for r in json.loads(js.read_text())]
    summarise(rows)
    plot(rows, _paths.get_plots_dir() / "grain_recombination_wide_bins.png")
    summarise_per_bin(rows)
    plot_per_bin(rows, _paths.get_plots_dir() / "grain_recombination_wide_bins_per_bin.png")


if __name__ == "__main__":
    main()
