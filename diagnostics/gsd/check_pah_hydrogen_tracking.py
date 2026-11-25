"""Tabulate or track the hydrogen of PAHs? P2 gates G1-G2 and tests T1-T6, PAH half.

The molecular bins (N_C 20-400) would store M_H next to M_C and N. A PAH's hydrogen is not one number but a
distribution over (Z, N_H), so the questions are whether its equilibrium is enough (E), whether one tracked scalar
<N_H> with a closure is (M), or neither (F), and whether the charge tables need an N_H axis. The networks are
pahchem's own (``baseline.BaselineModel``, default ``Params``: the full (Z, N_H) network with super-hydrogenated
states and the IP shift of dehydrogenated ones) for C24H12, C54H18 and C96H24, one per bin of the default
``CarbonLayout`` below 189 C; bin 4 (189-400 C) has no parent of its own (none in ``andrews16.PARENTS``). The tools
are ``pahchem.hydrogen_tracking``. It prints:

- G1, Montillaud+13 Fig. 9 (``montillaud13``, their card, C66H20 in the NGC 7023 NW profile at A_V = 0 and 1):
  charge and hydrogen settling times from C66H20 neutral, with and without their cation + H2 reaction, with their
  inputs and with the transition placed where theirs is (stripped modes, G0 x 10^0.4: our threshold energies are
  higher than theirs, check_montillaud13_fig10.py);
- G2, Murga+22 model b (Figshare): per region and size, <X_H/X_H0> at 1e4 and 1e5 yr from the normal state, the
  share of the change still to come at 1e5 yr, and the e-folding time of its last decade;
- T1: on a (G0, n(H)) grid (Draine shape, T = 100 K, n_e = 1.4e-4 n(H), Montillaud's Fig. 10 conditions), the
  steady state and the settling time (90 %) of the hydrogen observables after a x10 step in G0, up and down, and
  the charge relaxation time at the populated N_H; with a figure;
- T2: fast charge. The steady states of the N_H chain with charge averaged at each N_H (FC), and at N_H0 for every
  N_H (FC0, what ``population`` and a charge table without an N_H axis assume), against the full network, over the
  T1 grid and along the paths;
- T3/T4/T6: parcels carried at 0.3, 1 and 3 km/s into and out of the NGC 7023 NW (Meudon, digitised), Orion Bar
  (isobaric) and Horsehead (Murga+20) profiles, cycling CNM <-> WNM, and cycling across the C54 and C96
  dehydrogenation transitions (``TRANSITIONS``, T1's conditions). The reference is the full network with the
  conditions held over 50 A_V cells (what a cell code samples). Against it, at the cell ends: E, the equilibrium;
  FC and FC0; C, the one-scalar closure (its ODE, exact per cell); and "exp", RAMSES-like steps of 1e2, 1e3 and
  1e4 yr inside each cell (fewer where the cell is crossed faster), the closure's update x_eq + (x - x_eq)
  exp(-k dt) with its rate k = -R(x) / (x - x_eq) frozen at the step's start; "step" is that update against the
  exact closure (the stepping error alone). Errors are the largest absolute differences of x = <N_H>/N_H0 and the
  shares f_intact, f_bare, f_superh and f_cation, split by the G0 of the cell (>= 100, 1-100, < 1);
- T5: two parcels mixed 1:1, the true mixture against the closure's distribution at the mean <N_H>, and how long
  the difference lasts;
- the treatment map: per molecule and G0 class, the worst error of each treatment over all paths.

Inputs that are not the profiles' own (labelled): field shapes (Kurucz 15000 K for NGC 7023, 40000 K for the Orion
Bar, 30000 K for the Horsehead, Draine for the diffuse phases), held with depth (no reddening); N_H = 1.87e21 A_V
for the path lengths; chi (MMP83) -> G0 = 1.126 chi; CNM G0 1.126, n(H) 50, n_e 0.03, 80 K and WNM G0 1.126,
n(H) 0.5, n_e 0.02, 8000 K (typical solar-circle phases, Wolfire+03). The Horsehead profile prints x(H) = 0 beyond
A_V = 3: the profile is cut there.

Needs model_data/PAH_states (C24H12, C54H18, C96H24, C66H20; $CALIMA_MODEL_DATA) and the Murga+20/22 Figshare
data (``--murga2020``, ``--murga2022``). About an hour on 1 core.

    python diagnostics/gsd/check_pah_hydrogen_tracking.py [--parents C24 C54] [--skip-gates]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import scipy.linalg

from pycalima import _paths
from pycalima.pahchem import hydrogen_tracking as ht
from pycalima.pahchem.baseline import BaselineModel, Environment, Params
from pycalima.pahchem.molecule import PAHSpecies

YR = ht.YR
NH_PER_AV = 1.87e21
MATHIS_G0 = 1.126                       # Habing G0 of the Mathis (1983) ISRF at U = 1 (fields.mathis1983().g0())
PARENTS = {"C24": "C24H12", "C54": "C54H18", "C96": "C96H24"}
SPEEDS_KMS = (0.3, 1.0, 3.0)
STEPS_YR = (1e2, 1e3, 1e4)
SCORED = ("x", "f_intact", "f_bare", "f_superh", "f_cation")
G0_CLASSES = (("G0>=100", 100.0, np.inf), ("1<=G0<100", 1.0, 100.0), ("G0<1", 0.0, 1.0))
CNM = (MATHIS_G0, 50.0, 0.03, 80.0, None)
#: cycles across the dehydrogenation transitions of C54 (G0 / n(H) ~ 10-100) and C96 (~ 1e3-1e4), T1's conditions
TRANSITIONS = {"G0 1e2<->1e3 at n(H)=10": ((1e2, 10.0, 1.4e-3, 100.0, None), (1e3, 10.0, 1.4e-3, 100.0, None)),
               "G0 1e3<->1e4 at n(H)=1": ((1e3, 1.0, 1.4e-4, 100.0, None), (1e4, 1.0, 1.4e-4, 100.0, None))}
PLOTTED = (("C24", "Orion Bar out 3 km/s"), ("C24", "Horsehead out 1 km/s"),
           ("C54", "G0 1e2<->1e3 at n(H)=10, period 1e+04 yr"), ("C96", "G0 1e3<->1e4 at n(H)=1, period 1e+04 yr"))
WNM = (MATHIS_G0, 0.5, 0.02, 8000.0, None)
SITES = {"NGC 7023": 15000, "Orion Bar": 40000, "Horsehead": 30000}
T_EVOLVE_YR = np.geomspace(1e-6, 1e8, 57)
N_NODES = 50


class Family:
    """A BaselineModel parent; its photo rates tabulated on ``G0`` per field shape (log-log; linear in G0 below)."""
    G0 = 10.0 ** np.arange(-4.0, 5.51, 0.25)

    def __init__(self, parent, states_dir, params=Params()):
        f = PARENTS[parent]
        sp = PAHSpecies.from_states_file(states_dir / f"{f}_0.dat")
        an = PAHSpecies.from_states_file(states_dir / f"{f}_-1.dat")
        self.parent, self.m = parent, BaselineModel(parent, sp, params, an)
        self.nh0 = self.m.nh0
        self.keys = [(z, n) for z in self.m.charges for n in self.m.nh_values]
        self._tab, self._nets, self._closures, self._prop = {}, {}, {}, {}

    def _table(self, teff):
        if teff not in self._tab:
            t0 = time.time()
            arr = {k: np.zeros((self.G0.size, len(self.keys))) for k in ("H", "H2", "ion")}
            for i, g in enumerate(self.G0):
                pr = self.m.photo_rates(Environment("table", g, 1.0, 1.0, 100.0, teff))
                for c, key in enumerate(self.keys):
                    for k in arr:
                        arr[k][i, c] = pr[key].get(k, 0.0)
            self._tab[teff] = arr
            print(f"  [{self.parent}: photo rates on {self.G0.size} G0, field {teff or 'Draine'}: "
                  f"{time.time() - t0:.0f} s]", flush=True)
        return self._tab[teff]

    def photo(self, g0, teff):
        arr = self._table(teff)
        lg, x = np.log10(self.G0), np.log10(g0)
        if x <= lg[0]:
            vals = {k: a[0] * g0 / self.G0[0] for k, a in arr.items()}
        else:
            i = int(np.clip(np.searchsorted(lg, x), 1, lg.size - 1))
            w = min((x - lg[i - 1]) / (lg[i] - lg[i - 1]), 1.0)
            vals = {}
            for k, a in arr.items():
                lo, hi = a[i - 1], a[i]
                ok = (lo > 0) & (hi > 0)
                v = np.where(ok, np.exp((1 - w) * np.log(np.where(ok, lo, 1.0)) + w * np.log(np.where(ok, hi, 1.0))),
                             (1 - w) * lo + w * hi)
                vals[k] = v
        return {key: {k: float(vals[k][c]) for k in vals} for c, key in enumerate(self.keys)}

    def net(self, cond):
        if cond not in self._nets:
            g0, n_h, n_e, t, teff = cond
            self._nets[cond] = self.m.network(self.photo(g0, teff), Environment("cell", g0, n_h, n_e, t, teff))
        return self._nets[cond]

    def closure(self, cond):
        if cond not in self._closures:
            self._closures[cond] = ht.Closure(self.net(cond), self.nh0, np.linspace(-30.0, 30.0, 61))
        return self._closures[cond]

    def propagate(self, cond, dt, f):
        key = (cond, float(f"{dt:.10e}"))
        if key not in self._prop:
            if len(self._prop) > 4000:
                self._prop.clear()
            self._prop[key] = scipy.linalg.expm(self.net(cond).generator().toarray() * dt)
        g = np.clip(self._prop[key] @ np.asarray(f).ravel(), 0.0, None)
        return (g / g.sum()).reshape(self.net(cond).shape)


def cond_key(g0, n_h, n_e, t, teff):
    r = lambda v: float(f"{v:.6e}")                                                  # noqa: E731
    return (r(g0), r(n_h), r(n_e), r(t), teff)


# -- profiles ----------------------------------------------------------------------------------------
def murga_nodes(site, root):
    from pycalima.pahchem.murga_data import read_profile_2020

    p = read_profile_2020({"Orion Bar": "orion_isobaric", "Horsehead": "horsehead"}[site], root)
    u = p["n_tot"] / (p["x_h"] + p["x_h2"])         # the abundances' unit, whatever their normalisation
    q = {"g0": MATHIS_G0 * p["chi"], "n_h": u * p["x_h"], "n_e": u * p["x_e"], "t": p["t_gas"],
         "n_nuc": u * (p["x_h"] + 2.0 * p["x_h2"])}
    last = np.nonzero(p["x_h"] > 0)[0][-1]          # the Horsehead prints x(H) = 0 beyond A_V 3: cut there
    av = np.linspace(p["av"][0], p["av"][last], N_NODES)
    return av, {k: np.exp(np.interp(av, p["av"][:last + 1], np.log(v[:last + 1]))) for k, v in q.items()}


def ngc7023_nodes():
    D = Path(_paths.get_external_data_path("NWPDR_NGC7023"))
    prof = {n: np.loadtxt(D / f"{n}.csv", delimiter=",") for n in ("G0nH", "nH", "nH2", "ne", "temperature")}
    hi = min(p[:, 0].max() for p in prof.values())
    av = np.linspace(0.0, hi, N_NODES)
    at = lambda n: np.exp(np.interp(av, prof[n][:, 0], np.log(prof[n][:, 1])))        # noqa: E731
    n_h = at("nH")
    return av, {"g0": at("G0nH") * n_h, "n_h": n_h, "n_e": at("ne"), "t": at("temperature"),
                "n_nuc": n_h + 2.0 * at("nH2")}


def profile_path(av, q, teff, speed_kms, inward):
    """(segment conditions, durations [s], A_V at segment starts): node k held from node k to k + 1."""
    n_mid = np.sqrt(q["n_nuc"][1:] * q["n_nuc"][:-1])
    dt = np.diff(av) * NH_PER_AV / n_mid / (speed_kms * 1e5)
    conds = [cond_key(q["g0"][i], q["n_h"][i], q["n_e"][i], q["t"][i], teff) for i in range(av.size)]
    if inward:
        return conds[:-1], dt, av[:-1]
    return conds[::-1][:-1], dt[::-1], av[::-1][:-1]


def cycle_path(period_yr, a=CNM, b=WNM, n_cycles=4):
    c, w = cond_key(*a), cond_key(*b)
    return [c, w] * n_cycles, np.full(2 * n_cycles, 0.5 * period_yr * YR), None


# -- scoring ----------------------------------------------------------------------------------------
def classes_of(conds):
    g0 = np.array([c[0] for c in conds])
    return {name: (g0 >= lo) & (g0 < hi) for name, lo, hi in G0_CLASSES}


def score(a, b, masks):
    """{class: worst |a - b| over SCORED} (and the observable), a and b dicts of arrays on the same times."""
    out = {}
    for name, m in masks.items():
        if not m.any():
            out[name] = (np.nan, "")
            continue
        errs = {k: float(np.max(np.abs(np.asarray(a[k])[m] - np.asarray(b[k])[m]))) for k in SCORED}
        k = max(errs, key=errs.get)
        out[name] = (errs[k], k)
    return out


def run_path(F, conds, durs, f0):
    """Scores of every treatment along one path (module docstring)."""
    net0 = F.net(conds[0])
    obs = lambda f: ht.observables(net0, f, F.nh0)                                    # noqa: E731
    t_bounds = np.concatenate([[0.0], np.cumsum(durs)])
    # reference: the full network, conditions held per segment; compared at the segment ends
    ref = [f0]
    for c, dt in zip(conds, durs):
        ref.append(F.propagate(c, dt, ref[-1]))
    ref = np.array(ref[1:])
    seg_masks = classes_of(conds)
    res = {"E": score(obs(np.array([F.net(c).steady_state() for c in conds])), obs(ref), seg_masks)}
    p_true = {}
    for tag, conditional in (("FC", True), ("FC0", False)):
        p, out = f0.sum(axis=0), []
        for c, dt in zip(conds, durs):
            chain, fz = ht.eliminate_charge(F.net(c), conditional, F.nh0)
            p = np.clip(scipy.linalg.expm(chain.generator().toarray() * dt) @ p, 0.0, None)
            p /= p.sum()
            out.append(fz * p[None, :])
        res[tag] = score(obs(np.array(out)), obs(ref), seg_masks)
        p_true[tag] = out
    x, cl_f = float(obs(f0)["x"]), []
    for c, dt in zip(conds, durs):
        cl = F.closure(c)
        x = cl.step(x, dt)
        cl_f.append(cl.distribution(x))
    res["C"] = score(obs(np.array(cl_f)), obs(ref), seg_masks)
    # RAMSES-like steps inside each cell (its conditions held): the closure's exponential update with the rate frozen
    # at the step's start, sub-stepped at dt; against the full network ("exp") and the exact closure ("step")
    for dt_yr in STEPS_YR:
        x, f_exp = float(obs(f0)["x"]), []
        for c, dt in zip(conds, durs):
            cl = F.closure(c)
            n = max(int(np.ceil(dt / (dt_yr * YR))), 1)
            for _ in range(n):
                x = cl.exp_step(x, dt / n)
            f_exp.append(cl.distribution(x))
        o_exp = obs(np.array(f_exp))
        res[f"exp {dt_yr:.0e}"] = score(o_exp, obs(ref), seg_masks)
        res[f"step {dt_yr:.0e}"] = score(o_exp, obs(np.array(cl_f)), seg_masks)
    o = obs(ref)
    return res, (float(o["x"].min()), float(o["x"].max())), (t_bounds[1:], ref, p_true)


# -- gates ------------------------------------------------------------------------------------------
def gate_montillaud(states_dir):
    from pycalima.pahchem.montillaud13 import Montillaud13

    D = Path(_paths.get_external_data_path("NWPDR_NGC7023"))
    prof = {n: np.loadtxt(D / f"{n}.csv", delimiter=",") for n in ("G0nH", "nH", "ne", "temperature", "nH2")}
    at = lambda n, a: np.interp(a, prof[n][:, 0], prof[n][:, 1])                       # noqa: E731
    sp = PAHSpecies.from_states_file(states_dir / "C66H20_0.dat")
    ts = T_EVOLVE_YR * YR
    print("\nG1: Montillaud+13 Fig. 9, C66H20 in NGC 7023 NW (their card; theirs: charge in hours, H ~1e2 yr at "
          "A_V = 0 and ~1e4 yr at A_V = 1, 1e2-1e3 yr with cation + H2)")
    print(f"{'variant':34s} {'A_V':>4s} {'f(bare) ss':>10s} {'charge [h]':>10s} {'H [yr]':>9s} {'H, +H2 [yr]':>11s}")
    for label, modes, shift in (("their inputs (parent modes)", "parent", 1.0),
                                ("transition matched (stripped, G0x2.5)", "stripped", 10 ** 0.4)):
        m = Montillaud13(sp, "C66", modes=modes)
        for av in (0.0, 1.0):
            nh, ne, T, nh2 = at("nH", av), at("ne", av), at("temperature", av), at("nH2", av)
            pr = m.photo_rates(at("G0nH", av) * nh * shift)
            row = []
            for h2 in (False, True):
                net = m.network(pr, nh, ne, T)
                if h2:
                    k = np.zeros(net.shape)
                    k[1, :m.nh0] = 5e-13 * nh2                 # Sect. 3.4: cation + H2, two H at a time
                    net.add("cation + H2", k, dnh=+2)
                f0 = net.zeros()
                f0[0, m.nh0] = 1.0
                F = net.evolve(f0, ts)
                o = ht.observables(net, F, m.nh0)
                ss = net.steady_state()
                fz = F[:, 1].sum(axis=1)
                row.append((ss[:, 0].sum(), ht.settling_time(ts, fz, y_inf=ss[1].sum()) / 3600.0,
                            ht.settling_time(ts, o["x"], y_inf=ht.observables(net, ss, m.nh0)["x"]) / YR))
            print(f"{label:34s} {av:4.1f} {row[0][0]:10.2f} {row[0][1]:10.3g} {row[0][2]:9.3g} {row[1][2]:11.3g}")


def gate_murga2022(root):
    import pandas as pd

    from pycalima._paths import get_external_data_path
    from pycalima.pahchem import murga_data as M
    from pycalima.pahchem.murga_charge import region_of

    d = M.read_model_2022("b", root)
    prof = pd.read_csv(get_external_data_path("pahchem/murga2022_fig6_profile.csv"), comment="#")
    reg = np.array([region_of(x) for x in prof.distance_arcsec])
    n = d["n"]                                                    # loc, Z, t, X_H, size
    xb = d["x_h"] / d["x_h"][2]
    w = d["f"][:, :, 0, 0, :] * n[:, :, 0].sum(axis=2)            # loc, Z, size
    tot = n.sum(axis=3)
    with np.errstate(invalid="ignore", divide="ignore"):
        x = np.where(tot > 0, (n * xb[None, None, None, :, None]).sum(axis=3) / tot, np.nan)   # loc, Z, t, size
    t_yr = np.arange(n.shape[2]) * 1e4
    print("\nG2: Murga+22 model b (Figshare), from the normal state: f-weighted <X_H/X_H0> at 1e4 and 1e5 yr, the share "
          "of the change from 9e4 to 1e5 yr, and the e-folding time of that decade")
    print(f"{'region':6s} {'a [A]':>6s} {'<x>(1e4)':>9s} {'<x>(1e5)':>9s} {'last 1e4 yr':>11s} {'tau [yr]':>9s}")
    for r in ("PDR 1", "PDR 2", "DF", "MC"):
        mloc = reg == r
        for s in range(d["a_cm"].size):
            ww = w[mloc, :, s]
            xs = np.nansum(x[mloc, :, :, s] * ww[:, :, None], axis=(0, 1)) / ww.sum()
            if not np.isfinite(xs).all() or ww.sum() == 0:
                continue
            last = (xs[-1] - xs[-2]) / (xs[-1] - xs[0]) if xs[-1] != xs[0] else 0.0
            r1, r2 = xs[-2] - xs[-3], xs[-1] - xs[-2]
            tau = 1e4 / np.log(r1 / r2) if r1 * r2 > 0 and abs(r2) < abs(r1) else np.inf
            if s % 2 == 0 or s == d["a_cm"].size - 1:
                print(f"{r:6s} {d['a_cm'][s] * 1e8:6.1f} {xs[1]:9.3f} {xs[-1]:9.3f} {100 * last:10.1f}% {tau:9.3g}")
    del t_yr


# -- T1 ---------------------------------------------------------------------------------------------
def t1_maps(F, g0s, nhs, T=100.0):
    """Steady state and settling times on the grid; G0 steps of x10 (two grid points of 0.5 dex)."""
    lg = np.log10(g0s)
    pad = 10.0 ** np.arange(lg[0] - 1.0, lg[-1] + 1.0 + 1e-9, 0.5)
    out = {k: np.zeros((g0s.size, nhs.size)) for k in ("x", "f_bare", "f_superh", "t_up", "t_down", "tau_z",
                                                       "fc", "fc0")}
    ts = T_EVOLVE_YR * YR
    for j, nh in enumerate(nhs):
        ss = {}
        for g in pad:
            c = cond_key(g, nh, 1.4e-4 * nh, T, None)
            ss[c[0]] = (c, F.net(c).steady_state())
        for i, g in enumerate(g0s):
            c, f = ss[cond_key(g, nh, 1.4e-4 * nh, T, None)[0]]
            net = F.net(c)
            o = ht.observables(net, f, F.nh0)
            for k in ("x", "f_bare", "f_superh"):
                out[k][i, j] = o[k]
            for tag, g_from in (("t_up", g / 10.0), ("t_down", g * 10.0)):
                f_from = ss[cond_key(g_from, nh, 1.4e-4 * nh, T, None)[0]][1]
                Fz = net.evolve(f_from, ts)
                oo = ht.observables(net, Fz, F.nh0)
                t = [ht.settling_time(ts, oo[k], y_inf=o[k]) for k in ("x", "f_intact", "f_bare", "f_superh")
                     if abs(oo[k][0] - o[k]) > 0.01]
                out[tag][i, j] = max(t) / YR if t else 0.0
            pz = f.sum(axis=0)
            cr = ht.charge_rates(net)
            out["tau_z"][i, j] = 1.0 / cr[np.argmax(pz)]
            for tag, cnd in (("fc", True), ("fc0", False)):
                chain, fz = ht.eliminate_charge(net, cnd, F.nh0)
                oc = ht.observables(net, fz * chain.steady_state()[0][None, :], F.nh0)
                out[tag][i, j] = max(abs(oc[k] - o[k]) for k in SCORED)
    return out


def plot_t1(maps, g0s, nhs, out_png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    from pycalima.plotting_style import use_calima_style

    use_calima_style()
    fig, axs = plt.subplots(3, len(maps), figsize=(4.4 * len(maps), 11.5), squeeze=False)
    X, Y = np.meshgrid(np.log10(nhs), np.log10(g0s))
    for c, (parent, mp) in enumerate(maps.items()):
        t = np.clip(np.maximum(mp["t_up"], mp["t_down"]), 1e-6, 1e6)
        panels = ((t, LogNorm(1e-4, 1e6), "viridis", "H settling (90 %) [yr]", f"{parent}: hydrogen"),
                  (np.maximum(mp["tau_z"] / YR, 1e-8), LogNorm(1e-5, 1e3), "magma", "charge relaxation [yr]",
                   f"{parent}: charge"),
                  (np.maximum(mp["fc"], 1e-4), LogNorm(1e-3, 1.0), "cividis", "worst steady-state error",
                   f"{parent}: fast-charge chain (T2)"))
        for r, (v, norm, cmap, lab, title) in enumerate(panels):
            ax = axs[r, c]
            im = ax.pcolormesh(X, Y, v, norm=norm, cmap=cmap, shading="nearest")
            fig.colorbar(im, ax=ax, label=lab)
            cs = ax.contour(X, Y, mp["x"], levels=[0.1, 0.5, 0.9], colors="w", linewidths=1.0)
            ax.clabel(cs, fmt="x=%.1f", fontsize=8)
            ax.set_title(title, fontsize=12)
            ax.set_xlabel(r"$\log\,n$(H) [cm$^{-3}$]", fontsize=13)
            ax.set_ylabel(r"$\log\,G_0$ [Habing]", fontsize=13)
            ax.minorticks_on()
            ax.tick_params(which="both", axis="both", direction="in")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    print(f"wrote {out_png}")


def plot_paths(curves, out_png):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pycalima.plotting_style import use_calima_style

    use_calima_style()
    fig, axs = plt.subplots(2, len(curves), figsize=(4.6 * len(curves), 7.5), squeeze=False, sharex="col")
    for c, (title, (t, o_ref, o_eq, o_cl, o_fc)) in enumerate(curves.items()):
        for r, k, lab in ((0, "x", r"$\langle N_{\rm H}\rangle / N_{\rm H,0}$"), (1, "f_bare", r"$f$(bare)")):
            ax = axs[r, c]
            ax.plot(t / YR, o_ref[k], "C0-", lw=2, label="full network")
            ax.plot(t / YR, o_eq[k], "k--", lw=1.2, label="equilibrium")
            ax.plot(t / YR, o_cl[k], "C3:", lw=1.8, label="one-scalar closure")
            ax.plot(t / YR, o_fc[k], "C2-.", lw=1.2, label="fast-charge chain")
            ax.set_ylabel(lab, fontsize=13)
            ax.set_xscale("log")
            ax.minorticks_on()
            ax.tick_params(which="both", axis="both", direction="in")
        axs[0, c].set_title(title, fontsize=11)
        axs[1, c].set_xlabel("time [yr]", fontsize=13)
        axs[0, c].legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    print(f"wrote {out_png}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parents", nargs="+", default=list(PARENTS), choices=list(PARENTS))
    ap.add_argument("--states-dir", type=Path, default=_paths.get_model_data_dir() / "PAH_states")
    ap.add_argument("--murga2020", default=str(Path.home() / "Library/Caches/calima/datasets/pahchem/murga2020"))
    ap.add_argument("--murga2022", default=str(Path.home() / "Library/Caches/calima/datasets/pahchem/murga2022"))
    ap.add_argument("--skip-gates", action="store_true")
    ap.add_argument("--out", default="results/carbon_plan/check_pah_hydrogen_tracking")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    detail = open(out.with_suffix(".txt"), "w")

    from pycalima.gsd.carbon import M_C, CarbonLayout

    g = CarbonLayout().grid()
    print("molecular bins (N_C): " + ", ".join(f"{a:.0f}-{b:.0f}" for a, b in
                                                 zip(g.m_edges[:4] / M_C, g.m_edges[1:5] / M_C))
          + "; parents C24 (bin 1), C54 (bin 2), C96 (bin 3), none for bin 4")
    if not args.skip_gates:
        gate_montillaud(args.states_dir)
        gate_murga2022(args.murga2022)

    fams = {p: Family(p, args.states_dir) for p in args.parents}
    for F in fams.values():                       # interpolation check of the photo tables
        worst = 0.0
        for g0 in (3.3e-3, 0.7, 47.0, 2.2e3, 9e4):
            direct = F.m.photo_rates(Environment("chk", g0, 1.0, 1.0, 100.0, None))
            tab = F.photo(g0, None)
            for key in F.keys:
                for k in ("H", "ion"):
                    a, b = direct[key].get(k, 0.0), tab[key][k]
                    if a > 1e-20 * max(1e-300, max(v.get(k, 0.0) for v in direct.values())):
                        worst = max(worst, abs(np.log(b / a)))
        print(f"{F.parent}: photo-rate table against direct, worst |d ln k| {worst:.3f} (Draine shape)")

    # T1 and T2 on the grid
    g0s, nhs = 10.0 ** np.arange(-3.0, 5.01, 0.5), 10.0 ** np.arange(-1.0, 6.01, 0.5)
    maps = {}
    print("\nT1: steady state and settling (90 %) after a x10 step in G0 (Draine, 100 K, n_e = 1.4e-4 n(H));"
          " T2: worst steady-state error of FC / FC0 over x and the shares")
    for p, F in fams.items():
        t0 = time.time()
        maps[p] = mp = t1_maps(F, g0s, nhs)
        print(f"-- {p} ({time.time() - t0:.0f} s): rows G0, at n(H) = 1e1 / 1e3 / 1e5: "
              "x, f_bare | t_settle up, down [yr] | tau_Z [h] | T2 FC, FC0")
        for i, g0 in enumerate(g0s):
            cells = []
            for nh in (1e1, 1e3, 1e5):
                j = int(np.argmin(abs(np.log(nhs / nh))))
                cells.append(f"{mp['x'][i, j]:5.2f} {mp['f_bare'][i, j]:4.2f} | {mp['t_up'][i, j]:8.2g} "
                             f"{mp['t_down'][i, j]:8.2g} | {mp['tau_z'][i, j] / 3600:8.2g} | {mp['fc'][i, j]:.3f} "
                             f"{mp['fc0'][i, j]:.3f}")
            print(f"  G0 {g0:8.2g}: " + "  ||  ".join(cells))
        print(f"   T2 over the grid: FC worst {mp['fc'].max():.3f}, FC0 worst {mp['fc0'].max():.3f}")
    np.savez(out.with_name(out.name + "_T1.npz"), g0=g0s, n_h=nhs,
             **{f"{p}_{k}": v for p, mp in maps.items() for k, v in mp.items()})
    plot_t1(maps, g0s, nhs, out.with_name(out.name + "_T1.png"))

    # T3/T4/T6 paths
    paths = []
    nodes = {"NGC 7023": ngc7023_nodes(), "Orion Bar": murga_nodes("Orion Bar", args.murga2020),
             "Horsehead": murga_nodes("Horsehead", args.murga2020)}
    for site, (av, q) in nodes.items():
        for v in SPEEDS_KMS:
            for inward in (True, False):
                conds, durs, _ = profile_path(av, q, SITES[site], v, inward)
                paths.append((f"{site} {'in ' if inward else 'out'} {v:g} km/s ({durs.sum() / YR:.2g} yr)", conds, durs))
    for period in (1e2, 1e3, 1e4):
        conds, durs, _ = cycle_path(period)
        paths.append((f"CNM<->WNM, period {period:.0e} yr", conds, durs))
    for label, (lo, hi) in TRANSITIONS.items():
        for period in (1e3, 1e4, 1e5):
            conds, durs, _ = cycle_path(period, lo, hi)
            paths.append((f"{label}, period {period:.0e} yr", conds, durs))
    schemes = ["E", "FC", "FC0", "C"] + [f"{k} {d:.0e}" for d in STEPS_YR for k in ("exp", "step")]
    worst = {p: {(s, c[0]): (0.0, "") for s in schemes for c in G0_CLASSES} for p in fams}
    curves = {}
    print("\nT3/T4/T6: worst |error| over x and the shares along each path, by G0 class (>=100 / 1-100 / <1);"
          " details in " + str(out.with_suffix(".txt")))
    show = ["E", "FC", "FC0", "C", "exp 1e+03", "exp 1e+04", "step 1e+04"]
    print(f"{'':44s} " + " ".join(f"{s:>17s}" for s in show))
    for p, F in fams.items():
        print(f"-- {p} --")
        for name, conds, durs in paths:
            t0 = time.time()
            f0 = F.net(conds[0]).steady_state()
            res, (xlo, xhi), (t, ref, fc) = run_path(F, conds, durs, f0)
            cells = []
            for s in show:
                cells.append("/".join(f"{res[s][c[0]][0]:.2f}" if np.isfinite(res[s][c[0]][0]) else "  - "
                                      for c in G0_CLASSES))
            print(f"{name:44s} " + " ".join(f"{c:>17s}" for c in cells) + f"  x {xlo:.2f}-{xhi:.2f}"
                  f" [{time.time() - t0:.0f} s]", flush=True)
            detail.write(f"{p} {name}: true x {xlo:.3f}-{xhi:.3f}\n")
            for s in schemes:
                detail.write(f"    {s:12s} " + "  ".join(f"{c[0]}: {res[s][c[0]][0]:.3f} ({res[s][c[0]][1]})"
                                                       for c in G0_CLASSES) + "\n")
                for c in G0_CLASSES:
                    v = res[s][c[0]]
                    if np.isfinite(v[0]) and v[0] > worst[p][(s, c[0])][0]:
                        worst[p][(s, c[0])] = (v[0], f"{v[1]}, {name.split(' (')[0]}")
            if (p, name.split(" (")[0]) in PLOTTED:
                net0 = F.net(conds[0])
                o_ref = ht.observables(net0, ref, F.nh0)
                o_eq = ht.observables(net0, np.array([F.net(c).steady_state() for c in conds]), F.nh0)
                x, cl_f = float(ht.observables(net0, f0, F.nh0)["x"]), []
                for c, dt in zip(conds, durs):
                    x = F.closure(c).step(x, dt)
                    cl_f.append(F.closure(c).distribution(x))
                curves[f"{p}, {name.split(' (')[0]}"] = (t, o_ref, o_eq, ht.observables(net0, np.array(cl_f), F.nh0),
                                                         ht.observables(net0, np.array(fc["FC"]), F.nh0))
    plot_paths(curves, out.with_name(out.name + "_paths.png"))

    # T5 mixing
    print("\nT5: two parcels mixed 1:1, the closure's distribution at the mean <N_H> against the true mixture"
          " (at the first parcel's conditions): |error| of x, f_intact, f_bare, f_superh, f_cation; settling of the mix")
    av, q = nodes["NGC 7023"]
    edge = cond_key(q["g0"][0], q["n_h"][0], q["n_e"][0], q["t"][0], SITES["NGC 7023"])
    deep = cond_key(q["g0"][-1], q["n_h"][-1], q["n_e"][-1], q["t"][-1], SITES["NGC 7023"])
    for p, F in fams.items():
        net_edge = F.net(edge)
        bare, intact = net_edge.zeros(), net_edge.zeros()
        bare[F.m.charges.index(0), 0] = 1.0
        intact[F.m.charges.index(0), F.m.nh_values.index(F.nh0)] = 1.0
        pairs = {"NGC 7023 edge + cloud (A_V 5) material": (edge, F.net(edge).steady_state(), F.net(deep).steady_state()),
                 "CNM + WNM (in the CNM)": (cond_key(*CNM), F.net(cond_key(*CNM)).steady_state(),
                                            F.net(cond_key(*WNM)).steady_state()),
                 "bare + intact neutrals (in the CNM)": (cond_key(*CNM), bare, intact)}
        for name, (c, f1, f2) in pairs.items():
            net = F.net(c)
            mix = 0.5 * (f1 + f2)
            om = ht.observables(net, mix, F.nh0)
            oc = ht.observables(net, F.closure(c).distribution(float(om["x"])), F.nh0)
            ts = T_EVOLVE_YR * YR
            ev = ht.observables(net, net.evolve(mix, ts), F.nh0)
            o_inf = ht.observables(net, net.steady_state(), F.nh0)
            settle = max(ht.settling_time(ts, ev[k], y_inf=o_inf[k]) for k in SCORED) / YR
            print(f"{p} {name:42s} " + " ".join(f"{abs(oc[k] - om[k]):.3f}" for k in SCORED)
                  + f"   mix settles in {settle:.3g} yr")

    print("\nTreatment map: worst |error| over all paths (and where), per molecule and G0 class")
    for p in fams:
        print(f"-- {p} --")
        for s in ("E", "FC", "FC0", "C", "exp 1e+02", "exp 1e+03", "exp 1e+04", "step 1e+04"):
            print(f"   {s:10s} " + " | ".join(f"{c[0]}: {worst[p][(s, c[0])][0]:.3f} ({worst[p][(s, c[0])][1]})"
                                             for c in G0_CLASSES))
    detail.close()


if __name__ == "__main__":
    main()
