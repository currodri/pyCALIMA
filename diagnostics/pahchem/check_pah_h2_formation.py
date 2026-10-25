#!/usr/bin/env python3
"""H2 formation on PAHs: pahchem's baseline against the literature targets (validation V11).

pahchem's H2 per PAH is ``baseline.BaselineModel(...).network(...).h2_formation_rate(f)``: the sum over the
hydrogenation x charge chain of photolytic H2 loss (RRKM, West+18 E0 = 4.58 eV, even N_H <= N_H0 above the solo
count) and Eley-Rideal abstraction (0.06 A^2 v_H n(H) once per superhydrogenated molecule, N_H0 < N_H <= N_H0+4).
Molecules: C24H12, C54H18 and C96H24 (model_data/PAH_states), default Params.

Definitions used here (all explicit, per target):
- h2 [H2 per PAH per s]; R_PAH = x_PAH h2 / n(H) [cm^3 s^-1], so that dn(H2)/dt = R_PAH n_H n(H) (spec 2.4,
  item 4); x_PAH = 2.5e-5 / N_C per H nucleus (f_C = 0.1 of C/H = 2.5e-4, 4.6e-7 for C54, 2.6e-7 for C96) unless
  stated;
- Andrews+16 eq. 11: xi(Z) = sum_i 2 k_H2(Z, i) f(Z, i) / k_col(Z), f normalised within charge Z, k_col (App. E):
  neutrals sigma_geo v_H n(H) (sigma_geo = 35.7 A^2 N_C/24, pahchem's scaling), anions 7.8e-10 n(H), cations
  1.4e-10 n(H); eq. 13: R_PAH = 1/2 xi k_col n_PAH;
- Castellanos+18b eq. 2: xi = sum_i 2 k_H2(i) f(i) / (kappa_add n(H)), kappa_add = 1.4e-10, in a cation-only chain
  (H loss, H2 loss, H addition up to N_H0, as their model), and for the cation part of pahchem's full chain;
- Boschman+15 Table A.2: photo-rates per G0 [s^-1] of coronene states, pahchem at Habing G0 = 1 (Draine shape);
- Habart+04 Table 3 R_f [cm^3 s^-1] (a total: grains + PAHs) at their Table 1 chi (Draine units, G0 = 1.71 chi),
  n_H, T_gas, with n(H) = n_H and n_e = 1.6e-4 n_H (assumed: the H2-emitting layer is partly molecular);
- Le Bourlot+12 k_ER(T) = 2.8e-17 T^1/2 a e^(-300/T) / (1 + a e^(-300/T)), a = 1/(1 + (T/464)^1.5) [cm^3 s^-1];
- Thi+20 eq. C.2, R_max ~ 1e-16 f_PAH (sigma/10 A^2)(T/100)^1/2 (xbar/10) "s^-1", from sigma v_th x_PAH xbar with
  x_PAH = 3e-7 f_PAH (50/N_C);
- D3 bound (h2_formation.md): every H2 needs two added H, 2 h2 <= H-addition flux (incl. associative detachment).

Result (2026-10-25; R_PAH at x_PAH = 2.5e-5/N_C, n(H) = n_H, x_e = 1.6e-4; ~1 min, 1 core):
- ER constant: 0.9995 x 8.7e-13 (T/100)^1/2 (passes). D3: 2 h2 / H-addition flux <= 0.974 (passes).
- Andrews+16 Fig. 10: xi_ER plateaus at 1.5e-3 (C54) / 8.4e-4 (C96) for neutrals and 5.0e-3 (500 K) / 1.6e-3
  (50 K) for anions, size-independent: < 1 % (passes), and identical to ``Andrews16Model`` at G0/n(H) <= 1e-2.
  Super-H sets in at a ~3x higher G0/n(H) than theirs (f(super-H) 0.5-1.0 at 1e-2). xi_photo of normal C54 <
  1e-4 for G0/n(H) <= 1 (passes); neutral coronene at G0 = 10 reaches 1.1-2.3e-2 against their < 1e-2.
- eq. 13, NGC 7023 NW (f_C 0.15): R_PAH 6.4e-20 (C54) and 3.3e-20 (C96) against "a few 1e-19" for N_C ~ 40 at
  xi = 1 % (their C54/C96 lower still); grains 6e-18.
- Castellanos+18b xi (1-2 % at every G0): coronene+ 0.30 at the peak (cation-only chain; 0.07 in the full chain),
  15-30x too high, its peak at the bare skeleton; circumcoronene+ 9e-4 (G0 = 1) to 3e-2 (1e5), only near 1-2 %
  for G0 >= 1e4, the peak at G0/n(H) = 50-1000 and N_H 14-18, not at N_H0 - 6 (fails).
- Boschman+15 Table A.2 (C24, per G0): normal neutral H/H2 1.3/1.4x, cation 3.2/3.6x, dehydrogenated H 2.0-2.4x,
  super-H H 0.74-0.89x; pahchem has no H2 loss from odd-N_H or super-H states (their 1.2e-9 and 4.9e-9).
  Their Fig. 3 (R vs A_V) needs their PDR profile: not reproducible; their peak "a few 1e-16" is >= 100x pahchem's
  largest R_PAH on any grid here (2e-18, coronene).
- Habart+04: R_PAH 2e-21-1e-18, <= 0.7 % of R_f = 1.5e-16 (Oph W, S140, IC 63) and <= 2.4 % of 3e-17 (NGC 2023,
  Orion Bar): pahchem's PAHs cannot supply the excess over 3e-17. Chamaeleon: no chi, n_H in the repo (skipped).
- Le Bourlot+12 k_ER: pahchem 1e-4 to 2e-2 of it over 100-1000 K.
- Thi+20 R_max: the printed form is 1/4 of sigma v_th x_PAH xbar with the mean speed (4.0e-16 at 10 A^2, xbar 10);
  pahchem's ER ceiling 2.4e-19 (T/100)^1/2 equals the xbar = 1, 0.06 A^2 value, reached where f(super-H) = 1.

    check_pah_h2_formation.py [--andrews16] [--out DIR]
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pycalima import _paths
from pycalima.pahchem import chemistry
from pycalima.pahchem.andrews16 import CORONENE_GEOMETRIC_A2, Andrews16Model
from pycalima.pahchem.baseline import NGC7023_EDGE, BaselineModel, Environment, Params
from pycalima.pahchem.molecule import PAHSpecies
from pycalima.pahchem.network import PAHNetwork
from pycalima.plotting_style import use_calima_style

FORMULA = {"C24": "C24H12", "C54": "C54H18", "C96": "C96H24"}
G0S = (1.0, 10.0, 1e2, 1e3, 1e4, 1e5)
RATIOS = np.logspace(-4.0, 1.0, 11)                  # G0 / n(H)
CAST_RATIOS = np.logspace(-4.0, 3.0, 29)             # G0 / n(H) for Castellanos+18b (their peak may lie high)
X_E = 1.6e-4
K_ADD_CATION = 1.4e-10
#: Boschman+15 Table A.2 [s^-1 per G0]: (Z, N_H offset from N_H0, channel) -> k
BOSCHMAN_A2 = {(0, 0, "H"): 9.29e-10, (0, 0, "H2"): 9.29e-10, (1, 0, "H"): 3.38e-10, (1, 0, "H2"): 3.38e-10,
               (0, -1, "H"): 3.49e-9, (0, -1, "H2"): 3.49e-9, (1, -1, "H"): 5.05e-9, (1, -1, "H2"): 5.05e-9,
               (0, 1, "H"): 2.58e-8, (0, 1, "H2"): 1.23e-9, (1, 1, "H"): 2.52e-8, (1, 1, "H2"): 4.91e-9}
#: Habart+04: region -> (chi Draine (lo, hi), n_H (lo, hi), T_gas, R_f Table 3)
HABART = {"Oph W": ((250, 250), (1e4, 1e4), 330.0, 1.5e-16), "S140": ((100, 250), (1e4, 5e4), 500.0, 1.5e-16),
          "IC 63": ((650, 650), (5e4, 1e5), 620.0, 1.5e-16), "NGC 2023": ((500, 3000), (1e4, 1e5), 330.0, 3e-17),
          "Orion Bar": ((5e3, 2.4e4), (5e4, 3e5), 390.0, 3e-17)}


def x_pah(nc, f_c=0.1, c_h=2.5e-4):
    return f_c * c_h / nc


def le_bourlot12(t):
    a = 1.0 / (1.0 + (t / 464.0) ** 1.5)
    q = a * np.exp(-300.0 / t)
    return 2.8e-17 * np.sqrt(t) * q / (1.0 + q)


def thi20_rmax(t, sigma_a2, xbar, f_pah=1.0):
    return 1e-16 * f_pah * (sigma_a2 / 10.0) * np.sqrt(t / 100.0) * (xbar / 10.0)


def models():
    d = _paths.get_model_data_dir() / "PAH_states"
    out = {}
    for p, f in FORMULA.items():
        sp = {z: PAHSpecies.from_states_file(d / f"{f}_{z}.dat") for z in (-1, 0)}
        out[p] = BaselineModel(p, sp[0], Params(), anion=sp[-1])
    return out


def k_col(model, z, t, n_h):
    if z < 0:
        return 7.8e-10 * n_h
    if z > 0:
        return K_ADD_CATION * n_h
    return chemistry.cross_section_rate(t, CORONENE_GEOMETRIC_A2 * model.nc / 24.0) * n_h


def state(model, pr, env):
    """Steady state of the full chain: dict of h2, per-charge xi (eq. 11), fractions, fluxes."""
    net = model.network(pr, env)
    f = net.steady_state()
    proc = {p.name: p for p in net.processes}
    nh = np.array(model.nh_values)
    out = dict(h2=net.h2_formation_rate(f), fz=net.charge_fractions(f),
               f_superh=float(f[:, nh > model.nh0].sum()), xi={}, xi_er={}, xi_photo={})
    for i, z in enumerate(model.charges):
        fz = f[i].sum()
        er = float(np.sum(f[i] * proc["Eley-Rideal"].rate[i]))
        ph = float(np.sum(f[i] * proc["H2 loss"].rate[i]))
        kc = k_col(model, z, env.temperature_k, env.n_h)
        out["xi_er"][z] = 2.0 * er / (fz * kc) if fz > 0 else np.nan
        out["xi_photo"][z] = 2.0 * ph / (fz * kc) if fz > 0 else np.nan
    flux = net.fluxes(f)
    added = flux["H addition, cations"] + flux["H addition, neutrals"] + flux["associative detachment"]
    out["d3"] = 2.0 * out["h2"] / added if added > 0 else 0.0
    out["er_share"] = flux["Eley-Rideal"] / out["h2"] if out["h2"] > 0 else np.nan
    return out


def castellanos_chain(model, pr, n_h):
    """xi of eq. 2 in a cation-only chain: H loss, H2 loss, H addition (1.4e-10 n(H)) up to N_H0."""
    nhs = tuple(range(0, model.nh0 + 1))
    net = PAHNetwork((1,), nhs)
    net.add("H loss", np.array([[pr[(1, n)].get("H", 0.0) for n in nhs]]), dnh=-1)
    h2 = np.array([[pr[(1, n)].get("H2", 0.0) for n in nhs]])
    net.add("H2 loss", h2, dnh=-2, h2_released=1)
    net.add("H addition", np.array([[K_ADD_CATION * n_h if n < model.nh0 else 0.0 for n in nhs]]), dnh=+1)
    f = net.steady_state()
    return 2.0 * float(np.sum(f * h2)) / (K_ADD_CATION * n_h), f[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--andrews16", action="store_true", help="also run andrews16.Andrews16Model (C54, ~1 min per G0)")
    ap.add_argument("--out", type=Path, default=None, help="plot directory (default results/plots/pahchem)")
    args = ap.parse_args()
    mods = models()

    print("D1. Eley-Rideal constant: 0.06 A^2 v_H against Andrews+16 eq. 10, 8.7e-13 (T/100)^1/2 [cm^3 s^-1]")
    for t in (50.0, 100.0, 500.0, 2000.0):
        k = chemistry.cross_section_rate(t, 0.06)
        print(f"  T = {t:6.0f} K: {k:.4e}, ratio {k / (8.7e-13 * np.sqrt(t / 100.0)):.4f}")

    print("\nBoschman+15 Table A.2: coronene photo-rates per Habing G0 [s^-1], pahchem at G0 = 1 (Draine) / theirs")
    m24 = mods["C24"]
    pr1 = m24.photo_rates(Environment("G0 = 1", 1.0, 1e3, 1.0, 100.0, None))
    pr2 = m24.photo_rates(Environment("G0 = 100", 1e2, 1e3, 1.0, 100.0, None))
    for (z, dn, ch), k in BOSCHMAN_A2.items():
        v1, v2 = pr1[(z, m24.nh0 + dn)].get(ch, 0.0), pr2[(z, m24.nh0 + dn)].get(ch, 0.0) / 1e2
        lab = {-1: "dehydrogenated", 0: "normal", 1: "super-H"}[dn]
        print(f"  Z = {z:+d} {lab:15s} {ch:2s}: Boschman {k:.2e}; pahchem {v1:.2e} (G0 = 100: {v2:.2e} per G0); "
              f"ratio {v1 / k:.1e}")

    print("\nComputing photo-rates (Kurucz 15000 K) at G0 =", G0S)
    prs = {(p, g): m.photo_rates(Environment("grid", g, 1.0, 1.0, 100.0, 15000)) for p, m in mods.items() for g in G0S}

    print("\nAndrews+16 Fig. 10 / eq. 11, x_e = 1.6e-4: xi_ER(Z = 0, -1), xi_photo(Z = 0, +1), f(super-H), "
          "R_PAH; their xi_ER < 1e-2 (plateau), xi_photo < 1e-4 for normal C54")
    fig10 = {}
    d3_max, rmax_all = 0.0, {}
    for p in ("C24", "C54", "C96"):
        m = mods[p]
        for t in (50.0, 500.0):
            for g in (10.0, 1e3):
                for r in RATIOS:
                    n = g / r
                    s = state(m, prs[(p, g)], Environment("fig10", g, n, X_E * n, t, 15000))
                    s["R"] = x_pah(m.nc) * s["h2"] / n
                    fig10[(p, t, g, r)] = s
                    d3_max = max(d3_max, s["d3"])
                    rmax_all[(p, t)] = max(rmax_all.get((p, t), 0.0), s["R"])
            for g in (10.0, 1e3):
                print(f"  {p} T = {t:3.0f} K, G0 = {g:g}:")
                for r in RATIOS[::2]:
                    s = fig10[(p, t, g, r)]
                    print(f"    G0/n(H) {r:7.1e}: xi_ER(0, -1) {s['xi_er'][0]:.1e}, {s['xi_er'][-1]:.1e}; "
                          f"xi_photo(0, +1) {s['xi_photo'][0]:.1e}, {s['xi_photo'][1]:.1e}; f(super-H) {s['f_superh']:.2f}; "
                          f"f(-1, 0, +1) {s['fz'][-1]:.2f} {s['fz'][0]:.2f} {s['fz'][1]:.2f}; R_PAH {s['R']:.1e}")
    xi_er_max = max(max(s["xi_er"][0], s["xi_er"][-1]) for k, s in fig10.items() if k[0] != "C24")
    print(f"  max xi_ER over the grid {xi_er_max:.2e} (Andrews: < 1e-2)")

    if args.andrews16:
        d = _paths.get_model_data_dir() / "PAH_states"
        modes = {}
        for f in sorted(d.glob("C54*.dat")):
            sp = PAHSpecies.from_states_file(f)
            modes[(sp.charge, sp.nh)] = sp
        a16 = Andrews16Model("C54", modes, de_cm=150.0)
        print("\n  Andrews16Model (C54, their network as implemented in pahchem.andrews16), T = 500 K:")
        for g in (10.0, 1e3):
            pr = a16.photo_rates(g)
            for r in RATIOS[::2]:
                n = g / r
                net = a16.network(pr, n, X_E * n, 500.0)
                f = net.steady_state()
                proc = {q.name: q for q in net.processes}
                cells = []
                for i, z in enumerate(a16.charges):
                    if z in (-1, 0) and f[i].sum() > 0:
                        er = float(np.sum(f[i] * proc["Eley-Rideal"].rate[i]))
                        cells.append(f"xi_ER({z:+d}) {2 * er / (f[i].sum() * k_col(a16, z, 500.0, n)):.1e}")
                print(f"    G0 = {g:g}, G0/n(H) {r:7.1e}: " + ", ".join(cells)
                      + f"; h2 {net.h2_formation_rate(f):.1e} per PAH per s (baseline "
                        f"{fig10[('C54', 500.0, g, r)]['h2']:.1e})")

    print("\nCastellanos+18b eq. 2, xi of the cations (1-2 % for coronene+ and circumcoronene+ at the peak):")
    cast = {}
    for p in ("C24", "C54"):
        m = mods[p]
        for g in G0S:
            row = []
            for r in CAST_RATIOS:
                n = g / r
                xi_c, fc = castellanos_chain(m, prs[(p, g)], n)
                full = state(m, prs[(p, g)], Environment("cast", g, n, X_E * n, 300.0, 15000))
                cast[(p, g, r)] = (xi_c, full["xi_photo"][1], fc)
                row.append(xi_c)
            i = int(np.argmax(row))
            fc = cast[(p, g, CAST_RATIOS[i])][2]
            print(f"  {p}+ G0 = {g:7.0e}: peak xi {row[i]:.2e} at G0/n(H) = {CAST_RATIOS[i]:.1e} (most populated N_H "
                  f"there {int(np.argmax(fc))}, N_H0 = {m.nh0}; Castellanos: N_H0 - 6); full chain cation xi there "
                  f"{cast[(p, g, CAST_RATIOS[i])][1]:.2e}; xi at G0/n(H) = 1e-2 {row[int(np.argmin(np.abs(np.log(CAST_RATIOS / 1e-2))))]:.2e}")

    print("\nAndrews+16 eq. 13 / Castellanos: R_PAH in NGC 7023 NW (pahchem NGC7023_EDGE: G0 4100, n(H) 1.35e4, "
          "T 470 K), x_PAH = 0.15 x 3.5e-4 / N_C (Andrews' f_C)")
    for p in ("C54", "C96"):
        m = mods[p]
        s = state(m, m.photo_rates(NGC7023_EDGE), NGC7023_EDGE)
        xp = 0.15 * 3.5e-4 / m.nc
        print(f"  {p}: h2 {s['h2']:.2e} per PAH per s (ER share {s['er_share']:.2f}); R_PAH {xp * s['h2'] / NGC7023_EDGE.n_h:.1e} "
              f"(Andrews: 'a few 1e-19' for N_C ~ 40, xi = 1 %; their grains 6e-18; Castellanos ovalene ~1e-17); "
              f"xi_photo(+1) {s['xi_photo'][1]:.1e}, xi_ER(0) {s['xi_er'][0]:.1e}")

    print("\nHabart+04 Table 3 (R_f, grains + PAHs; standard 3e-17): pahchem R_PAH over the Table 1 ranges")
    for name, (chi, nh, t, rf) in HABART.items():
        vals = []
        for p in ("C54", "C96"):
            m = mods[p]
            for c in sorted(set(chi)):
                for n in sorted(set(nh)):
                    g = 1.71 * c
                    s = state(m, m.photo_rates(Environment(name, g, n, X_E * n, t, None)),
                              Environment(name, g, n, X_E * n, t, None))
                    vals.append((p, x_pah(m.nc) * s["h2"] / n, s["er_share"]))
        lo, hi = min(v[1] for v in vals), max(v[1] for v in vals)
        print(f"  {name:9s} (chi {chi[0]:g}-{chi[1]:g}, n_H {nh[0]:.0e}-{nh[1]:.0e}, T {t:g} K): Habart R_f {rf:.1e}; "
              f"pahchem R_PAH {lo:.1e}-{hi:.1e} ({hi / rf:.1e} of R_f at most); ER share "
              f"{min(v[2] for v in vals):.2f}-{max(v[2] for v in vals):.2f}")

    print("\nLe Bourlot+12 k_ER(T) against pahchem R_PAH(T) (C54 and C96, G0 = 1e2 and 1e3, n(H) = 1e4)")
    ts = np.array([100.0, 200.0, 330.0, 500.0, 1000.0])
    lb = {}
    for p in ("C54", "C96"):
        m = mods[p]
        for g in (1e2, 1e3):
            rr = [x_pah(m.nc) * state(m, prs[(p, g)], Environment("lb12", g, 1e4, X_E * 1e4, t, 15000))["h2"] / 1e4
                  for t in ts]
            lb[(p, g)] = rr
            print(f"  {p} G0 = {g:g}: " + ", ".join(f"T {t:4.0f}: {r:.1e} ({r / le_bourlot12(t):.1e} x LB12)"
                                                    for t, r in zip(ts, rr)))
    print("  LB12: " + ", ".join(f"{le_bourlot12(t):.2e}" for t in ts) + " (h2_formation.md: 1.2e-17 at 100 K, 1.3e-16 at 500 K)")

    print("\nThi+20 R_max (eq. C.2) at T = 100 K, N_C = 54 (x_PAH = 3e-7 (50/54) for f_PAH = 1):")
    xp_thi = 3e-7 * 50.0 / 54.0
    v100 = chemistry.thermal_speed(100.0, 1.00794 * chemistry.AMU_G)
    for sig, xbar in ((10.0, 10.0), (6.0, 10.0), (0.6, 10.0), (0.06, 10.0), (0.06, 1.0)):
        print(f"  sigma {sig:5.2f} A^2, xbar {xbar:4.0f}: printed {thi20_rmax(100.0, sig, xbar):.2e}; "
              f"sigma v_th x_PAH xbar (mean speed) {sig * 1e-16 * v100 * xp_thi * xbar:.2e}")
    er_cap = xp_thi * chemistry.cross_section_rate(100.0, 0.06)
    print(f"  pahchem's ER ceiling (once per molecule, every PAH super-H) x_PAH 0.06 A^2 v_H = {er_cap:.2e}; reached "
          f"share f(super-H) <= {max(s['f_superh'] for s in fig10.values()):.2f} on the Fig. 10 grid")

    print(f"\nD3 bound 2 h2 / (H-addition flux): max {d3_max:.3f} over the Fig. 10 grid (must be <= 1)")
    print("Max R_PAH on the Fig. 10 grid: " + ", ".join(f"{p} {t:g} K {v:.1e}" for (p, t), v in rmax_all.items())
          + " (Mennella+12 upper limit 1.2e-16 at 330 K; Habart 3e-17-1.5e-16)")

    use_calima_style()
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 8.0), sharex=True)
    colours = {("C54", 50.0): "#2a78d6", ("C54", 500.0): "#eb6834", ("C96", 50.0): "#1baf7a", ("C96", 500.0): "#8e44ad"}
    for (p, t), c in colours.items():
        lab = rf"C$_{{{p[1:]}}}$, $T = {t:g}$ K"
        axes[0].loglog(RATIOS, [fig10[(p, t, 1e3, r)]["xi_er"][0] for r in RATIOS], color=c, lw=1.8, label=lab + ", PAH$^0$")
        axes[0].loglog(RATIOS, [fig10[(p, t, 1e3, r)]["xi_er"][-1] for r in RATIOS], color=c, lw=1.2, ls="--",
                       label=lab + ", PAH$^-$")
        axes[1].loglog(RATIOS, [fig10[(p, t, 1e3, r)]["R"] for r in RATIOS], color=c, lw=1.8, label=lab)
    axes[0].axhline(1e-2, color="0.5", ls=":", lw=1.0)
    axes[0].set_ylabel(r"$\xi_{\rm ER}$ (Andrews+16 eq. 11)", fontsize=14)
    axes[0].set_ylim(1e-7, 1e-1)
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    for v, lab in ((3e-17, "standard"), (1.5e-16, "Habart+04 max")):
        axes[1].axhline(v, color="0.5", ls=":", lw=1.0)
        axes[1].text(RATIOS[0], v * 1.2, lab, fontsize=9, color="0.4")
    axes[1].set_xlabel(r"$G_0/n({\rm H})$ [cm$^{3}$], $G_0 = 10^3$", fontsize=14)
    axes[1].set_ylabel(r"$R_{\rm PAH}$ [cm$^3$ s$^{-1}$]", fontsize=14)
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.minorticks_on()
        ax.tick_params(which="both", axis="both", direction="in")
    fig.subplots_adjust(hspace=0)
    out = args.out or _paths.get_plots_dir("pahchem")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "pahchem_pah_h2_formation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out / "pahchem_pah_h2_formation.png")


if __name__ == "__main__":
    main()
