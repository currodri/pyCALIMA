#!/usr/bin/env python3
"""pahchem against WD01 one ingredient at a time, per grain: where the charge balance, alpha(H+), Gamma_PE and Lambda_rec differ.

Both models are a birth-death chain over the charge Z: up rates (photoionisation or photodetachment, H+ collisions) and down rates
(electron capture), with heating per photo event and cooling per capture. Here both are put on one charge axis:

- **pahchem:** the one-molecule table of a PAHdb v4.00 molecule (``group_tables.molecule_kernels``, calibrated IPs ``W.IP_SHIFTS``,
  defaults): photo rates ``K_bg`` and heat ``H_bg`` in the Mathis field at G0 = 1 Habing, capture ``alpha`` and its energy ``L`` onto
  Z = 0, +1, +2, H+ rates ``k_mn`` (on -1) and ``k_ct`` (on 0); states -1..+2.
- **WD01:** pyCALIMA's WD01 grain physics for one graphite sphere of the same carbon count, a = 10 A (N_C/468)^(1/3)
  (``check_wd01b_alpha._alpha_full``): photo rates ``compute_Rpe_vectorized`` (Draine & Lee graphite absorption, WD01 yields,
  photodetachment for Z < 0), capture with WD01's sticking, H+ capture at sticking 1, heating and cooling per Z; states from WD01's
  most negative allowed charge up.

Validation first: the chain on pahchem's rates reproduces ``check_wd01_per_size.molecule_job`` and on WD01's rates reproduces WD01's
solver (``check_wd01_per_size.wd01_job``). Then the rates are swapped block by block, cumulatively from pahchem to WD01 (the path),
and one at a time into pahchem (the single swaps; the effects are not additive):

1. cation sticking: pahchem's s = 1 x eta on its own geometry -> WD01's 0.5 (1 - exp(-a/10 A));
2. cation capture geometry: 0.82 pi a_disc^2 (a_disc = 0.9 sqrt(N_C) A) -> pi a^2 of the sphere (J~ on that radius);
3. neutral attachment: statistical sticking x f on the disc -> WD01's sticking x pi a^2 v J~(0);
4. anion detachment: pahchem's sigma_UV x valence-detachment yield -> WD01's photodetachment + valence-band yield of Z = -1;
5. H+ collisions on -1 and 0: mutual neutralisation (0.58 DS87) and charge transfer (0.9 Langevin) -> WD01's capture (s = 1, all Z);
6. photoionisation absorption: the library sigma -> WD01's graphite C_abs (pahchem's thresholds and yields kept);
7. photoionisation thresholds and yields: -> WD01's full R_pe (IP(Z) of the grain formula, yield y2 min(y0 y1, 1));
8. charge ladder: WD01's states beyond -1..+2, with WD01's rates (including +2 -> +3);
9. energies: heat per photo event and energy per capture -> WD01's.

After step 9 every rate is WD01's, so the path ends on WD01's result (checked). Nothing is tuned.

Result (2026-11-01; C24H12, C54H18, C96H24, C150H30; T = 100, 1000 K; psi = 1e2-1e5; f = 1). The chain reproduces pahchem's code to
<= 4e-16 and WD01's solver to <= 8e-4 in every case. Budgets along the path (WD01 - pahchem; alpha and Gamma in dex):
- psi >= 1e4: the cation sticking (pahchem's s = 1 against WD01's 0.16-0.25) is the largest term in every case: <Z> +0.39 to +0.65,
  alpha -0.23 to -1.04, Gamma -0.22 to -0.67. Disc against sphere adds up to +0.16 in <Z> (C150). The charge ladder matters only at
  psi = 1e5 for N_C >= 96: +0.31 (C96), +0.60 (C150) in <Z>, -0.13 to -0.30 in alpha; at psi = 1e4 it is <= 0.06.
- psi <= 1e3: the neutral attachment (pahchem's statistical sticking ~1 against WD01's ~0.2) is the largest term for N_C >= 54: <Z>
  +0.17 to +0.51, alpha -0.18 to -0.55; for C24H12 it has the other sign (pahchem's statistical sticking on coronene is small). The
  anion detachment gives -0.02 to -0.15 in <Z>; WD01's H+ rates per state are 1.3-1.4x pahchem's (+0.06 to +0.18 in alpha); WD01's
  photoelectrons carry more energy per event (Gamma +0.04 to +0.33).
- The photon physics nearly agrees: absorption (library against graphite C_abs) and thresholds/yields move <Z> by <= 0.1 for
  N_C >= 54 (up to 0.2 for C24H12, where graphite absorbs more) and Gamma by <= 0.1.
The two models differ almost only in the electron sticking on cations and neutrals.

    check_wd01_ingredients.py [--molecules "C54H18 (37)" ...] [--T 100 1000] [--psi 1e2 1e3 1e4 1e5]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "dust_charge"))
import check_wd01_per_size as S

from pycalima import _paths
from pycalima.models.dust_charge import dust_charging as DC
from pycalima.models.dust_charge import shared_physics as SP
from pycalima.models.dust_charge.dust_photoelectric_heating import (compute_photoelectric_heating_rate,
                                                                    compute_recombination_cooling_rate)
from pycalima.pahchem import chemistry, photo
from pycalima.pahchem import group_tables as G
from pycalima.pahchem import population as pop
from pycalima.pahchem.baseline import Params

W = S.W
MOLECULES = ("C24H12 (18)", "C54H18 (37)", "C96H24 (108)", "C150H30 (612)")
Z_LO, Z_HI = -4, 10
ZS = np.arange(Z_LO, Z_HI + 1)
M_H = 1.67262192e-24
EV = chemistry.EV_ERG
STEPS = ("pahchem", "1 cation sticking", "2 cation geometry", "3 neutral attachment", "4 anion detachment", "5 H+ collisions",
         "6 absorption", "7 thresholds and yields", "8 charge ladder", "9 energies")
OUT = _paths.get_results_dir("pahchem") / "wd01_ingredients_2026-11-01"


def idx(z):
    return int(z - Z_LO)


def pahchem_rates(mol, params, T):
    """Per-state rate arrays on ZS (NaN outside -1..+2): photo up [s^-1], heat per photo event [eV], capture coefficient onto z
    [cm^3 s^-1], energy per capture [eV], H+ coefficient raising z [cm^3 s^-1], and the pieces the swaps need."""
    L0, L1, nE = G.rg.DEFAULT_GROUP_L0, G.rg.DEFAULT_GROUP_L1, G.N_E_PER_GROUP
    k = G.molecule_kernels((mol, params, photo.PHOTOELECTRON_PARTITION, L0, L1, nE, G.LT))
    tab = G.PAHGroupTable(G.combine([k], np.array([1.0]), L0, L1, nE, G.rg.TS_GRID, G.LT, mol.name, ""))
    r = {key: np.full(ZS.size, np.nan) for key in ("up", "e_photo", "cap", "e_cap", "ion")}
    alpha, L = tab._in_T("alpha", T), tab._in_T("L", T)
    j = G.ION_NAMES.index("H+")
    for i, z in enumerate(G.CHARGES):
        r["up"][idx(z)] = tab.t["K_bg"][i]
        r["e_photo"][idx(z)] = tab.t["H_bg"][i] / tab.t["K_bg"][i] if tab.t["K_bg"][i] > 0 else 0.0
        r["ion"][idx(z)] = 0.0
    for i, z in enumerate(G.CHARGES[1:]):
        r["cap"][idx(z)] = alpha[i]
        r["e_cap"][idx(z)] = L[i] / alpha[i]
    r["cap"][idx(-1)], r["e_cap"][idx(-1)] = 0.0, 0.0
    r["ion"][idx(-1)], r["ion"][idx(0)] = tab._in_T("k_mn", T)[j], tab.t["k_ct"][j]
    # what the absorption swap needs: pahchem's yield on the Mathis grid, and its own photo rate recomputed from it
    chain_mol = G._chain(mol)
    E, Fb = G.BG_GRID, G._mathis_flux()
    r["yield"] = {z: pop._yield(chain_mol, z, params, energy_ev=E) for z in G.CHARGES}
    r["sigma"] = {z: photo.sigma_uv(mol.nc, z, E, params.uv_cross_section) * params.sigma_scale for z in G.CHARGES}
    r["check_up"] = {z: float(np.trapezoid(r["sigma"][z] * r["yield"][z] * Fb, E)) for z in G.CHARGES}
    r["E"], r["Fb"], r["table"] = E, Fb, tab
    return r


def wd01_rates(nc, T):
    """WD01's per-state arrays on ZS for one graphite sphere of nc carbons (``check_wd01b_alpha._alpha_full``'s physics)."""
    a = float(W.radius_cm(nc))
    gm = W.mathis_G()
    ctx = DC._prepare_gamma_scan_context("graphite", a)
    nu, J, C = ctx["nu"], ctx["J_nu"] / gm, ctx["C_abs_nu"]
    yp = ctx["yield_params"]
    zmin = int(SP.most_negative_allowed_charge_graphite(a))
    ok = ZS >= zmin
    up = np.where(ok, DC.compute_Rpe_vectorized(nu, J, C, a, ZS, ctx["yield_func"], yp), np.nan)
    cap = np.where(ok, SP.collisional_rates_electrons_vector(np.array([a]), ZS, 1.0, T, SP.electron_sticking_coefficient_graphite), np.nan)
    ion_sp = [{"n": 1.0, "T": T, "m": M_H, "z": 1.0}]
    ion = np.where(ok, SP.collisional_rates_ions_vector(np.array([a]), ZS, ion_sp), np.nan)
    # alpha(H+) "case a": an H+ capture neutralises only where IP(H) exceeds the grain's (as _alpha_full)
    ip_grain = np.where(ZS >= 0, SP.ionisation_potential_valence_vec(4.4, ZS, a), SP.electron_affinity_graphite_vec(ZS + 1, a))
    v = np.sqrt(8.0 * SP.KB_CGS * T / (np.pi * M_H))
    k_alpha = np.where(ok, np.pi * a * a * v * SP.DS87_J_function_vec(ZS, np.array([1.0]), a, T) * (13.6 - ip_grain >= 0.0), np.nan)
    E_eV = DC.h_cgs * nu / DC.eV2erg
    field = np.column_stack([E_eV, DC.c_cgs / nu, J * nu * DC.eV2erg])
    im = np.column_stack([yp["Imperp"], yp["Impar"]])
    heat = np.array([compute_photoelectric_heating_rate((int(z), a, field, "graphite", im, C)) if o else np.nan for z, o in zip(ZS, ok)])
    rec = np.array([compute_recombination_cooling_rate((int(z), a, 1.0, T, "graphite")) if o else np.nan for z, o in zip(ZS, ok)])
    with np.errstate(divide="ignore", invalid="ignore"):
        e_photo = np.where(up > 0, heat / up / EV, 0.0)
        e_cap = np.where(cap > 0, rec / cap / EV, 0.0)
    return {"up": up, "cap": cap, "ion": ion, "k_alpha": k_alpha, "e_photo": e_photo, "e_cap": e_cap, "zmin": zmin, "a": a,
            "E": E_eV, "C_abs": C, "J_E": J * DC.eV2erg / DC.h_cgs}


def solve(r, ne, n_ion, zlo, zhi):
    """f(Z) on ZS for states zlo..zhi, and <Z>, alpha(H+) [cm^3 s^-1], Gamma, Lambda [erg s^-1] per grain."""
    lo, hi = idx(zlo), idx(zhi)
    up = r["up"] + n_ion * r["ion"]
    down = ne * r["cap"]
    logf = np.full(ZS.size, -np.inf)
    logf[lo] = 0.0
    for i in range(lo, hi):
        if up[i] > 0 and down[i + 1] > 0 and np.isfinite(logf[i]):
            logf[i + 1] = logf[i] + np.log(up[i]) - np.log(down[i + 1])
    f = np.exp(logf - logf.max())
    f /= f.sum()
    g = lambda x: float(np.nansum(f * np.nan_to_num(x)))       # noqa: E731
    return {"f": f, "zbar": g(ZS), "alpha": g(r["k_alpha"]), "gamma": g(r["up"] * r["e_photo"]) * EV,
            "lam": ne * g(r["cap"] * r["e_cap"]) * EV}


def blend(p, w, steps):
    """The rates after the cumulative swaps ``steps`` (names from STEPS); pahchem's states -1..+2 unless the ladder is swapped."""
    r = {k: p[k].copy() for k in ("up", "e_photo", "cap", "e_cap", "ion")}
    r["k_alpha"] = p["ion"].copy()
    s = lambda n: n in steps                                        # noqa: E731
    cat = [idx(z) for z in (1, 2)]
    a_w = w["a"]
    if s("1 cation sticking") and not s("2 cation geometry"):
        for i in cat:
            r["cap"][i] = p["cap"][i] * float(SP.electron_sticking_coefficient_graphite(np.array([ZS[i]]), a_w)[0])
    if s("2 cation geometry"):
        for i in cat:
            r["cap"][i] = w["cap"][i]
    if s("3 neutral attachment"):
        r["cap"][idx(0)] = w["cap"][idx(0)]
    if s("4 anion detachment"):
        r["up"][idx(-1)] = w["up"][idx(-1)]
    if s("5 H+ collisions"):
        for z in (-1, 0, 1, 2):
            r["ion"][idx(z)], r["k_alpha"][idx(z)] = w["ion"][idx(z)], w["k_alpha"][idx(z)]
    if s("6 absorption") and not s("7 thresholds and yields"):
        C = np.interp(p["E"], w["E"], w["C_abs"])
        for z in (0, 1, 2):
            r["up"][idx(z)] = float(np.trapezoid(C * p["yield"][z] * p["Fb"], p["E"]))
    if s("7 thresholds and yields"):
        for z in (0, 1, 2):
            r["up"][idx(z)] = w["up"][idx(z)]
    zlo, zhi = -1, 2
    if s("8 charge ladder"):
        zlo, zhi = w["zmin"], Z_HI
        out = [i for i, z in enumerate(ZS) if z < -1 or z >= 2]
        for i in out:
            for k in ("up", "cap", "ion", "k_alpha", "e_photo", "e_cap"):
                if not (k == "cap" and ZS[i] == 2):                  # capture onto +2 stays as the steps above set it
                    r[k][i] = w[k][i]
        r["cap"][idx(-1)] = w["cap"][idx(-1)]                       # -1 -> -2 where WD01 allows it
    if s("9 energies"):
        r["e_photo"], r["e_cap"] = w["e_photo"].copy(), w["e_cap"].copy()
    for k in r:
        r[k] = np.where(np.isfinite(r[k]), r[k], 0.0)
    return r, zlo, zhi


def validate(mol, params, p, w, T, psi):
    """The chain on pure pahchem and pure WD01 rates against the full codes (max relative difference)."""
    ne = np.sqrt(T) / psi
    rp, zl, zh = blend(p, w, ())
    a = solve(rp, ne, ne, zl, zh)
    ref = p["table"].solve(np.zeros(p["table"].t["L0"].size), T, ne, 0.0, (("H+", ne),), G0_bg=1.0)
    zp = float(ref["f"] @ np.array(G.CHARGES, dtype=float))
    dp = max(abs(a["zbar"] - zp), abs(a["gamma"] / ref["gamma"] - 1), abs(a["lam"] / ref["lam"] - 1),
             abs(a["alpha"] / (ref["alpha_mn"]["H+"] + ref["alpha_ct"]["H+"]) - 1))
    rw, zl, zh = blend(p, w, STEPS[1:])
    b = solve(rw, ne, ne, zl, zh)
    wd = S.wd01_job((1.0 / W.mathis_G(), ne, T, w["a"]))
    dw = max(abs(b["zbar"] - wd[3]), abs(b["gamma"] / wd[1] - 1), abs(b["lam"] / wd[2] - 1), abs(b["alpha"] / wd[0] - 1))
    return dp, dw, wd


def plot_path(res, T, path):
    """The cumulative path pahchem -> WD01 at temperature T: <Z> - <Z>_WD01 and the ratios to WD01, one panel column per psi."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pycalima.plotting_style import use_calima_style

    use_calima_style()
    keys = [k for k in res if float(k.split("|")[1]) == T]
    mols = list(dict.fromkeys(k.split("|")[0] for k in keys))
    psis = sorted({float(k.split("|")[2]) for k in keys})
    rows = (("zbar", r"$\langle Z\rangle - \langle Z\rangle_{\rm WD01}$", 3, False), ("alpha", r"$\alpha({\rm H^+})$ / WD01", 0, True),
            ("gamma", r"$\Gamma_{\rm PE}$ / WD01", 1, True), ("lam", r"$\Lambda_{\rm rec}$ / WD01", 2, True))
    colors = ("#2a78d6", "#1baf7a", "#eb6834", "#8a5cc2")
    fig, axes = plt.subplots(len(rows), len(psis), figsize=(4.2 * len(psis), 11), sharex=True)
    x = np.arange(len(STEPS))
    for c, psi in enumerate(psis):
        for r_i, (q, lab, iw, ratio) in enumerate(rows):
            ax = axes[r_i, c]
            for m, col in zip(mols, colors):
                d = res.get(f"{m}|{T:g}|{psi:g}")
                if d is None:
                    continue
                ref = d["wd01"][iw]
                y = [s[q] / ref if ratio else s[q] - ref for s in d["path"]]
                ax.plot(x, y, "o-", color=col, ms=3.5, lw=1.6, label=m.split(" ")[0])
            ax.axhline(1.0 if ratio else 0.0, color="k", lw=0.8)
            if ratio:
                ax.set_yscale("log")
            ax.minorticks_on()
            ax.tick_params(which="both", direction="in", top=True, right=True)
            ax.tick_params(axis="x", which="minor", bottom=False, top=False)
            if c == 0:
                ax.set_ylabel(lab)
            if r_i == 0:
                ax.set_title(rf"$T = {T:g}$ K, $\psi = 10^{{{int(round(np.log10(psi)))}}}$")
        axes[-1, c].set_xticks(x)
        axes[-1, c].set_xticklabels([s.split(" ", 1)[-1] if s[0].isdigit() else s for s in STEPS], rotation=60, ha="right", fontsize=8)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("pahchem to WD01, one ingredient at a time (cumulative, left to right; per grain)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--molecules", nargs="+", default=list(MOLECULES))
    ap.add_argument("--T", nargs="+", type=float, default=[100.0, 1000.0])
    ap.add_argument("--psi", nargs="+", type=float, default=[1e2, 1e3, 1e4, 1e5])
    ap.add_argument("--attachment-fraction", type=float, default=1.0)
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    params = Params(attachment_fraction=a.attachment_fraction)
    tag = "" if params.attachment_fraction == 1.0 else f"_f{params.attachment_fraction:g}"
    if a.plot_only:
        res = json.loads((OUT / f"ingredients{tag}.json").read_text())
        for T in a.T:
            plot_path(res, T, _paths.get_plots_dir("pahchem") / f"pahchem_wd01_ingredients_T{T:g}{tag}.png")
        return
    lib = {m.name: m for m in pop.shift_thresholds(pop.library_v400(), W.IP_SHIFTS)}
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    for name in a.molecules:
        mol = lib[name]
        for T in a.T:
            p, w = pahchem_rates(mol, params, T), wd01_rates(mol.nc, T)
            dup = max(abs(p["check_up"][z] / p["up"][idx(z)] - 1) for z in G.CHARGES if p["up"][idx(z)] > 0)
            print(f"\n=== {name} (N_C {mol.nc}, WD01 a = {w['a'] * 1e8:.2f} A, Z_min {w['zmin']}), T = {T:.0f} K, f = {params.attachment_fraction}; "
                  f"pahchem photo rates recomputed from sigma x yield: max rel. diff {dup:.1e}")
            for psi in a.psi:
                ne = np.sqrt(T) / psi
                dp, dw, wd = validate(mol, params, p, w, T, psi)
                print(f"\npsi = {psi:.0e} (n_e = n(H+) = {ne:.3g}): validation, chain vs code: pahchem {dp:.1e}, WD01 {dw:.1e}")
                print(f"  {'step':26s} {'<Z>':>7s} {'alpha(H+)':>10s} {'Gamma':>10s} {'Lambda':>10s}   /WD01: {'dZ':>6s} {'alpha':>7s} {'Gamma':>7s} {'Lambda':>7s}   "
                      f"f(-1)  f(0)  f(+1) f(+2) f(>=3)")
                rows = []
                for k in range(len(STEPS)):
                    r, zl, zh = blend(p, w, STEPS[1:k + 1])
                    q = solve(r, ne, ne, zl, zh)
                    f = q["f"]
                    rows.append({"step": STEPS[k], **{x: q[x] for x in ("zbar", "alpha", "gamma", "lam")}, "f": f.tolist()})
                    print(f"  {STEPS[k]:26s} {q['zbar']:7.3f} {q['alpha']:10.3e} {q['gamma']:10.3e} {q['lam']:10.3e}   "
                          f"       {q['zbar'] - wd[3]:+6.2f} {q['alpha'] / wd[0]:7.3f} {q['gamma'] / wd[1]:7.3f} {q['lam'] / wd[2]:7.3f}   "
                          f"{f[idx(-1)]:.3f} {f[idx(0)]:.3f} {f[idx(1)]:.3f} {f[idx(2)]:.3f} {f[idx(3):].sum():.3f}")
                single = []
                for st in STEPS[1:]:
                    deps = {"2 cation geometry": ("1 cation sticking", "2 cation geometry"),
                            "7 thresholds and yields": ("6 absorption", "7 thresholds and yields")}.get(st, (st,))
                    r, zl, zh = blend(p, w, deps)
                    q = solve(r, ne, ne, zl, zh)
                    single.append({"step": st, **{x: q[x] for x in ("zbar", "alpha", "gamma", "lam")}})
                base = rows[0]
                print("  single swaps into pahchem (change of <Z>; ratio to pahchem of alpha, Gamma, Lambda):")
                for sw in single:
                    print(f"    {sw['step']:26s} dZ {sw['zbar'] - base['zbar']:+6.3f}  alpha x{sw['alpha'] / base['alpha']:6.3f}  "
                          f"Gamma x{sw['gamma'] / base['gamma']:6.3f}  Lambda x{sw['lam'] / base['lam']:6.3f}")
                res[f"{name}|{T:g}|{psi:g}"] = {"path": rows, "single": single, "wd01": list(map(float, wd)),
                                               "validation": {"pahchem": dp, "wd01": dw}}
    (OUT / f"ingredients{tag}.json").write_text(json.dumps(res))
    for T in a.T:
        plot_path(res, T, _paths.get_plots_dir("pahchem") / f"pahchem_wd01_ingredients_T{T:g}{tag}.png")
    print(f"\nresults in {OUT}")


if __name__ == "__main__":
    main()
