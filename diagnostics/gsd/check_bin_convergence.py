#!/usr/bin/env python3
"""How few size bins does the GSD need? A bin-convergence and cost study.

RAMSES-CALIMA has to evolve the grain size distribution in millions of cells
over millions of steps, so the question is the cheapest representation
(scheme x number of bins) that still follows a high-resolution reference
under accretion, shattering, coagulation and thermal sputtering.

Size range. Carbon starts at 0.35 nm (N_C ~ 20 with a = 10 A (N_C/468)^1/3,
Draine & Li 2001; the WD01/DL07/Zubko+04 minimum), silicate at 0.45 nm (the
astrodust truncation of Hensley & Draine 2023); ``--a-min-c 5e-8`` gives the
N_C ~ 50-60 photostability limit (Montillaud+13, Andrews+15, Murga+22). The
coarse grids end at 1 um, where the FIR-emission to extinction budget leaves
almost no mass (Zubko+04; the WD01 tail); grains grown past it are kept as
overflow. The reference runs to 5 um at 30 bins per mass decade, so the mass
that crosses 1 um is measured.

Schemes, all through the same ``pycalima.gsd`` operators:
  pl2m        two moments (N, M) per bin, power law inside (2 scalars/bin)
  1m-mrn      one moment M; inside each bin dn/da ~ a^-3.5 (1 scalar/bin)
  1m-limiter  one moment M; in-bin slope from the neighbours, harmonic-mean
              limiter of Marchand et al. (2021, Ishinisan; collapse.grid)
A one-moment scheme stores only M: before every operator its N is rebuilt
from the closure and afterwards discarded, as a RAMSES passive scalar would.

Operators per step (Lie splitting, stiff exchange last; gsd review s. 7.5):
collisions (MPRK22; KT10 shattering above v_shat and Chokshi sticking
below v_coag, Brownian + Ormel & Cuzzi 2007 velocities, fragments below
a_min returned to the gas), accretion (``exchange_step``, backward Euler in
the gas, sticking S), thermal sputtering (``gsd.sputtering``, Bocchio size
correction). The coarse runs step at a fixed hydro dt (``--dt-hydro``), the
cadence at which RAMSES would re-impose the in-bin shape; the reference takes
the per-phase steps of PHASES (``--ref-check`` halves them).

Scenarios: each process alone from the WD01 distribution (accretion in a
molecular cloud, shattering in the WNM and the WIM, coagulation in a dense
cloud, sputtering in hot gas), and the four in series. Phases are HY09
Table 1 (Hirashita & Yan 2009, MNRAS 394, 1061: T, n_H, injection scale L
and velocity V, here the Ormel & Cuzzi driving), plus hot post-shock gas.
Abundances per H are kept and scaled by each phase's n_H.

Accuracy, at fixed snapshot times: per composition the dust mass, the
power-law slope of dn/da (mass-weighted fit over 3 nm - 0.25 um) and the
shape (mass-weighted mean and width of ln a); for the mixture A_V/N_H, R_V,
the 2175 A bump over its continuum, the far-UV slope A(0.1)/A(0.15) and the
curve A_lambda/A_V from 0.1 to 2.2 um (Mie, Draine & Lee 1984 optics,
integrated over the in-bin profiles). Errors are relative. Cost: scalars
per cell, collision sub-steps and wall time per hydro step.

Usage
-----
    python diagnostics/gsd/check_bin_convergence.py run --scenario accretion --quick
    python diagnostics/gsd/check_bin_convergence.py run --scenario all --workers 4
    python diagnostics/gsd/check_bin_convergence.py report
"""
from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace

import numpy as np

from pycalima import _paths

YR = 3.156e7
MYR = 1e6 * YR
PC = 3.086e18
M_AMU = 1.66053906660e-24

A_MAX = 1e-4                 # coarse grids end at 1 um
A_REF_MAX = 5e-4             # the reference at 5 um
REF_BPD = 30                 # reference bins per mass decade
TABLE_PPD = 96               # pair-rate table per radius decade, shared by all grids
N_SPEED = 8
ERRMAX = 1e-3
STICKING = 0.3               # accretion sticking (Hirashita & Kuo 2011)
T_DUST = 20.0
F_DEP0 = 0.5                 # initial fraction of the key element in dust
SOLAR = {"C": 2.69e-4, "Si": 3.24e-5}   # Asplund et al. 2009, per H
N_BINS = (2, 3, 4, 6, 8, 12, 16, 24, 32)
SCHEMES = ("pl2m", "1m-mrn", "1m-limiter")
LAMBDAS_UM = (0.1, 0.12, 0.15, 0.18, 0.2175, 0.25, 0.3, 0.36, 0.44, 0.55, 0.7, 0.9, 1.25,
              1.65, 2.2)
SLOPE_RANGE = (3e-7, 2.5e-5)  # [cm] range of the power-law fit of dn/da

# HY09 Table 1 (T [K], n_H [cm^-3], L [pc], V [km/s]) and a hot phase.
# dt_ref [yr]: the reference macro-step.
PHASES = {
    "CNM": dict(T=100.0, n_H=30.0, L=0.64, V=2.0, dt_ref=5e4),
    "WNM": dict(T=6000.0, n_H=0.3, L=100.0, V=20.0, dt_ref=2e5),
    "WIM": dict(T=8000.0, n_H=0.1, L=100.0, V=20.0, dt_ref=2e5),
    "MC": dict(T=25.0, n_H=300.0, L=1.0, V=1.2, dt_ref=2e4),
    "DC": dict(T=10.0, n_H=1e4, L=1.0, V=1.5, dt_ref=2e3),
    "HOT": dict(T=3e6, n_H=1.0, L=0.0, V=0.0, dt_ref=2e2),
}

# each process alone, then all four in series: segments of (phase,
# duration [Myr], processes switched on)
_ACC = ("MC", 100.0, {"accretion"})
_SHAT_WNM = ("WNM", 100.0, {"shattering"})
_SHAT_WIM = ("WIM", 30.0, {"shattering"})
_COAG = ("DC", 30.0, {"coagulation"})
_SPUT = ("HOT", 0.03, {"sputtering"})
SCENARIOS = {
    "accretion": [_ACC],
    "shattering_wnm": [_SHAT_WNM],
    "shattering_wim": [_SHAT_WIM],
    "coagulation": [_COAG],
    "sputtering": [_SPUT],
    "series": [_ACC, _SHAT_WNM, _COAG, _SPUT],
}
SNAP_FRACTIONS = (0.1, 0.3, 1.0)


# ---------------------------------------------------------------------------
# materials and initial conditions
# ---------------------------------------------------------------------------

@dataclass
class Material:
    name: str          # gsd.shattering / dust_sputtering name
    key: str           # gas key species
    a_min: float       # [cm]
    shatter: object = None
    sublimation: object = None


def materials(a_min_c, a_min_si):
    from pycalima.gsd.shattering import GRAPHITE, SILICATE
    from pycalima.models.dust_radiation.sublimation_kinetics import get_sublimation_model

    out = []
    for name, key, a_min, sh in (("graphite", "C", a_min_c, GRAPHITE),
                                 ("silicate", "Si", a_min_si, SILICATE)):
        # GD89 kinetics: the condensation coefficient is the sticking; the
        # bulk density is the shattering material's so that all operators agree
        sub = replace(get_sublimation_model(name, "GD89"), alpha_gd89=STICKING,
                      density_gcm3=sh.rho)
        out.append(Material(name, key, a_min, sh, sub))
    return out


def wd01_dnda(name, a):
    """Weingartner & Draine (2001) R_V = 3.1, b_C = 6e-5 shape, 1/n_H dn/da
    [cm^-1]; normalised later to the dust mass."""
    a = np.asarray(a, float)
    um = 1e-4
    if name == "graphite":
        al, be, at, ac, C = -1.54, -0.165, 0.0107 * um, 0.428 * um, 9.99e-12
        sig, rho, mC, bC = 0.4, 2.24, 12.011 * M_AMU, 6e-5
        from scipy.special import erf
        D = np.zeros_like(a)
        for b, a0 in ((0.75 * bC, 3.5e-8), (0.25 * bC, 3.0e-7)):
            B = (3.0 / (2 * np.pi) ** 1.5 * np.exp(-4.5 * sig ** 2) / (rho * a0 ** 3 * sig)
                 * b * mC / (1 + erf(3 * sig / np.sqrt(2) + np.log(a0 / 3.5e-8) / (sig * np.sqrt(2)))))
            D += B / a * np.exp(-0.5 * (np.log(a / a0) / sig) ** 2)
    else:
        al, be, at, ac, C = -2.21, 0.300, 0.164 * um, 0.1 * um, 1.0e-13
        D = 0.0
    F = 1 + be * a / at if be >= 0 else 1.0 / (1 - be * a / at)
    cut = np.where(a < at, 1.0, np.exp(-np.clip((a - at) / ac, 0, None) ** 3))
    return D + C / a * (a / at) ** al * F * cut


def initial_bins(grid, mat, ic):
    """(N, M) per bin of the IC between a_min and 1 um, per H, holding
    F_DEP0 of the key element."""
    N = np.zeros(grid.n_bins)
    M = np.zeros(grid.n_bins)
    for i in range(grid.n_bins):
        lo, hi = max(grid.a_edges[i], mat.a_min), min(grid.a_edges[i + 1], A_MAX)
        if hi <= lo:
            continue
        a = np.geomspace(lo, hi, 400)
        f = wd01_dnda(mat.name, a) if ic == "wd01" else a ** -3.5
        N[i] = np.trapezoid(f * a, np.log(a))
        M[i] = np.trapezoid(f * a * grid.mass(a), np.log(a))
    per_gram = mat.sublimation.key_per_unit / (mat.sublimation.unit_mass_amu * M_AMU)
    scale = F_DEP0 * SOLAR[mat.key] / per_gram / M.sum()
    return N * scale, M * scale


# ---------------------------------------------------------------------------
# grids and schemes
# ---------------------------------------------------------------------------

def make_grid(mat, n_bins):
    from pycalima.gsd.grid import MassGrid

    if n_bins is None:
        return MassGrid.from_radii(mat.a_min, A_REF_MAX, REF_BPD, mat.shatter.rho)
    m = 4.0 / 3.0 * np.pi * mat.shatter.rho * np.array([mat.a_min, A_MAX]) ** 3
    return MassGrid(np.geomspace(m[0], m[1], n_bins + 1), mat.shatter.rho)


def closure_N(grid, M, scheme):
    """Number per bin of a one-moment scheme, from its in-bin slope."""
    from pycalima.collapse.grid import slopes
    from pycalima.gsd.pl2m import mean_mass_ratio

    if scheme == "1m-mrn":
        beta = np.full(grid.n_bins, -11.0 / 6.0)        # dn/da ~ a^-3.5
    else:
        beta = slopes(np.ascontiguousarray(M, dtype=float), np.log(grid.eta))
    ratio = mean_mass_ratio(beta, grid.eta)
    return np.where(M > 0, M / (grid.m_edges[:-1] * ratio), 0.0)


@dataclass
class Composition:
    mat: Material
    grid: object
    scheme: str
    M: np.ndarray                        # per H
    N: np.ndarray                        # per H (pl2m only; otherwise from the closure)
    over_M: float = 0.0                  # grown past the grid, per H
    models: dict = field(default_factory=dict)
    adots: dict = field(default_factory=dict)

    @property
    def per_gram(self):
        s = self.mat.sublimation
        return s.key_per_unit / (s.unit_mass_amu * M_AMU)

    def number(self):
        return self.N if self.scheme == "pl2m" else closure_N(self.grid, self.M, self.scheme)


def collision_model(c, phase_name, processes, threads):
    from pycalima.gsd.shattering import GrainCollisions
    from pycalima.gsd.velocities import RelativeVelocity

    key = phase_name
    if key not in c.models:
        p = PHASES[phase_name]
        rv = RelativeVelocity(T=p["T"], n_H=p["n_H"], mu=1.4, v_turb=p["V"] * 1e5, L=p["L"] * PC)
        c.models[key] = GrainCollisions(
            c.mat.shatter, rv, law="KT10", coagulation="coagulation" in processes,
            v_shat=None if "shattering" in processes else 1e30, a_frag_min=c.mat.a_min,
            collector="remove", n_speed=N_SPEED, table_ppd=TABLE_PPD, threads=threads)
    return c.models[key]


def step(comps, gas, phase_name, dt, processes, threads, stats):
    """One split step of all compositions in a phase; gas and state per H."""
    from pycalima.gsd.collisions import collide
    from pycalima.gsd.exchange import Population, exchange_step
    from pycalima.gsd.sputtering import erosion_rate, sputter

    p = PHASES[phase_name]
    nH = p["n_H"]
    hot = p["T"] >= 1e5
    for c in comps:
        N, M = c.number() * nH, c.M * nH
        t0 = time.perf_counter()
        if not hot and processes & {"shattering", "coagulation"}:
            r = collide(c.grid, N, M, collision_model(c, phase_name, processes, threads), dt,
                        errmax=ERRMAX)
            N, M = r["N"], r["M"]
            c.over_M += r["overflow_M"] / nH
            gas[c.mat.key] += r["sink_M"] / nH * c.per_gram
            stats["coll_substeps"] += r["n_steps"]
        t1 = time.perf_counter()
        if not hot and "accretion" in processes and gas[c.mat.key] > 0.0:
            pop = Population(c.mat.sublimation, c.grid, N, M, np.full(c.grid.n_bins, T_DUST),
                             nucleation=False)
            g = {c.mat.key: gas[c.mat.key] * nH}
            exchange_step([pop], g, p["T"], dt)
            N, M = pop.N, pop.M
            c.over_M += pop.overflow_M / nH
            gas[c.mat.key] = g[c.mat.key] / nH
        t2 = time.perf_counter()
        if hot and "sputtering" in processes:
            if phase_name not in c.adots:
                c.adots[phase_name] = erosion_rate(c.mat.name, p["T"], nH, rho=c.grid.rho)
            N, M, dM = sputter(c.grid, N, M, c.adots[phase_name], dt)
            gas[c.mat.key] += dM / nH * c.per_gram
        t3 = time.perf_counter()
        c.M = M / nH
        if c.scheme == "pl2m":
            c.N = N / nH
        stats["t_coll"] += t1 - t0
        stats["t_acc"] += t2 - t1
        stats["t_sput"] += t3 - t2


# ---------------------------------------------------------------------------
# observables
# ---------------------------------------------------------------------------

def mie_table():
    """C_ext(a, lambda) [cm^2] on a log grid of radii, per material (cached)."""
    from pycalima.models.dust_radiation.dust_oppacity import compute_cross_sections_mie

    path = os.path.join(_out_dir(), "mie_table.npz")
    a = np.geomspace(3e-8, 6e-4, 160)
    if os.path.exists(path):
        d = np.load(path)
        if d["a"].shape == a.shape and np.allclose(d["lam"], LAMBDAS_UM):
            return {"a": d["a"], "graphite": d["graphite"], "silicate": d["silicate"]}
    tab = {"a": a}
    for name in ("graphite", "silicate"):
        C = np.empty((a.size, len(LAMBDAS_UM)))
        for k, ak in enumerate(a):
            with contextlib.redirect_stdout(io.StringIO()):
                _, _, s, ab, _ = compute_cross_sections_mie(name, ak * 1e4, list(LAMBDAS_UM))
            C[k] = s + ab
        tab[name] = C
    np.savez(path, lam=np.array(LAMBDAS_UM), **tab)
    return tab


def _integrate(prof, f):
    """sum over grains of f(a) (f vectorised, trailing axes allowed)."""
    tot = 0.0
    for i in np.nonzero(prof.N > 0)[0]:
        if prof.delta[i]:
            tot = tot + prof.N[i] * f(np.array([prof.aL[i] * prof.u_delta[i]]))[0]
            continue
        n = max(12, int(np.ceil(np.log(prof.uR) / 0.01)))
        u = np.geomspace(1.0, prof.uR, n + 1)
        w = prof.A[i] * u ** (prof.lam[i] + 1.0)              # dN / d ln u
        fa = f(prof.aL[i] * u)
        y = fa * (w if fa.ndim == 1 else w[:, None])
        tot = tot + np.trapezoid(y, np.log(u), axis=0)
    return tot


def metrics(comps, mie):
    """Named observables of the state (all per H)."""
    from pycalima.gsd.pl2m import BinProfile

    out = {}
    A_lam = 0.0
    for c in comps:
        prof = BinProfile(c.grid, c.number(), c.M)
        g = c.grid
        n = c.mat.name
        out[f"{n}:M_dust"] = c.M.sum() + c.over_M
        out[f"{n}:slope"] = _power_law_slope(prof, g)
        # shape: mass-weighted mean and width of ln a
        Mg = max(c.M.sum(), 1e-300)
        l1 = _integrate(prof, lambda a: g.mass(a) * np.log(a)) / Mg
        l2 = _integrate(prof, lambda a: g.mass(a) * np.log(a) ** 2) / Mg
        out[f"{n}:a_m"] = np.exp(l1)
        out[f"{n}:width"] = np.sqrt(max(l2 - l1 ** 2, 0.0))
        la = np.log(mie["a"])
        lC = np.log(mie[n])
        A_lam = A_lam + _integrate(prof, lambda a: np.exp(np.array(
            [np.interp(np.log(a), la, lC[:, k]) for k in range(lC.shape[1])]).T))
    A = 1.086 * A_lam                                          # A_lambda / N_H [mag cm^2]
    i = {lam: k for k, lam in enumerate(LAMBDAS_UM)}
    AV = A[i[0.55]]
    out["ext:A_V/N_H"] = AV
    out["ext:R_V"] = AV / (A[i[0.44]] - AV)
    # 2175 A bump over the continuum interpolated linearly in 1/lambda
    # between 0.15 and 0.3 um; UV slope as the far-UV rise
    t = (1 / 0.2175 - 1 / 0.3) / (1 / 0.15 - 1 / 0.3)
    out["ext:bump"] = A[i[0.2175]] / (A[i[0.3]] + t * (A[i[0.15]] - A[i[0.3]]))
    out["ext:UV_slope"] = A[i[0.1]] / A[i[0.15]]
    for k, lam in enumerate(LAMBDAS_UM):         # the curve's shape, A_lambda / A_V
        if lam != 0.55:
            out[f"ext:curve({lam})"] = A[k] / AV
    return out


def _power_law_slope(prof, grid):
    """Exponent of the best power law dn/da ~ a^p over SLOPE_RANGE: a fit of
    ln(dN/d ln a) on 12 log intervals, weighted by the mass in each."""
    e = np.geomspace(*SLOPE_RANGE, 13)
    Nk = np.zeros(12)
    Mk = np.zeros(12)
    for i in np.nonzero(prof.N > 0)[0]:
        for k in range(12):
            u1 = max(e[k] / prof.aL[i], 1.0)
            u2 = min(e[k + 1] / prof.aL[i], prof.uR)
            if u2 > u1:
                Nk[k] += prof.number(i, u1, u2)
                Mk[k] += prof.moment(i, 3, u1, u2) * grid.m_edges[i]
    ok = (Nk > 0) & (Mk > 0)
    if ok.sum() < 3:
        return np.nan
    x = 0.5 * np.log(e[:-1] * e[1:])[ok]
    y = np.log(Nk[ok] / np.log(e[1] / e[0]))
    w = Mk[ok] / Mk[ok].sum()
    xm, ym = (w * x).sum(), (w * y).sum()
    return (w * (x - xm) * (y - ym)).sum() / (w * (x - xm) ** 2).sum() - 1.0


def errors(m, ref):
    """Relative error of each metric against the reference (the slope
    relative to max(|p|, 1)); an undefined value where the reference has one
    counts as infinite."""
    out = {}
    for k, v in ref.items():
        x = m[k]
        if np.isnan(v) or np.isnan(x):
            out[k] = 0.0 if np.isnan(v) and np.isnan(x) else np.inf
            continue
        scale = max(abs(v), 1.0) if k.endswith(":slope") else abs(v)
        out[k] = abs(x - v) / max(scale, 1e-300)
    return out


# ---------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------

def schedule(scenario, quick):
    """[(phase, processes, t_start, t_end)] intervals between snapshots [s]."""
    out, t = [], 0.0
    for ph, dur, processes in SCENARIOS[scenario]:
        d = dur * MYR * (0.1 if quick else 1.0)
        prev = 0.0
        for f in SNAP_FRACTIONS:
            out.append((ph, processes, t + prev * d, t + f * d))
            prev = f
        t += d
    return out


def _out_dir():
    d = os.path.join(_paths.get_results_dir(), "gsd_bin_convergence")
    os.makedirs(d, exist_ok=True)
    return d


def run_name(scenario, scheme, n_bins, a_min_c, tag=""):
    nb = "ref" if n_bins is None else f"N{n_bins:02d}"
    return f"{scenario}_{scheme}_{nb}_amin{a_min_c * 1e8:.2f}A{tag}"


def run_one(scenario, scheme, n_bins, a_min_c, a_min_si, ic, dt_hydro_yr, quick, threads,
            ref_dt_factor=1.0, tag=""):
    """Evolve one scheme/grid through a scenario; write and return its npz path."""
    os.environ.setdefault("PYCALIMA_GSD_THREADS", str(threads))
    mats = materials(a_min_c, a_min_si)
    comps = []
    gas = {}
    for mat in mats:
        g = make_grid(mat, n_bins)
        N, M = initial_bins(g, mat, ic)
        c = Composition(mat, g, scheme, M, N)
        if scheme != "pl2m":
            c.N = None
        comps.append(c)
        gas[mat.key] = SOLAR[mat.key] * (1.0 - F_DEP0)
    totals0 = {c.mat.key: gas[c.mat.key] + c.M.sum() * c.per_gram for c in comps}
    mie = mie_table()
    stats = dict(coll_substeps=0, t_coll=0.0, t_acc=0.0, t_sput=0.0, hydro_steps=0)
    times, rows, gases = [0.0], [metrics(comps, mie)], [dict(gas)]
    wall0 = time.perf_counter()
    for ph, processes, t0, t1 in schedule(scenario, quick):
        if n_bins is None:
            # small steps only where operators are split or the gas couples
            # (the backward-Euler exchange); collisions alone sub-step
            # themselves and sputtering at fixed T is exact
            hot = PHASES[ph]["T"] >= 1e5
            ops = ({"sputtering"} & processes if hot else
                   ({"collisions"} if processes & {"shattering", "coagulation"} else set())
                   | ({"accretion"} & processes))
            split = len(ops) > 1 or "accretion" in ops
            dt_target = PHASES[ph]["dt_ref"] * YR * ref_dt_factor if split else t1 - t0
        else:
            dt_target = dt_hydro_yr * YR
        n = max(int(np.ceil((t1 - t0) / dt_target - 1e-9)), 1)
        for _ in range(n):
            step(comps, gas, ph, (t1 - t0) / n, processes, threads, stats)
        stats["hydro_steps"] += n
        times.append(t1)
        rows.append(metrics(comps, mie))
        gases.append(dict(gas))
    wall = time.perf_counter() - wall0
    cons = {k: abs(gas[k] + sum(c.M.sum() + c.over_M for c in comps if c.mat.key == k)
                   * next(c.per_gram for c in comps if c.mat.key == k) - v) / v
            for k, v in totals0.items()}
    names = list(rows[0])
    path = os.path.join(_out_dir(), run_name(scenario, scheme, n_bins, a_min_c, tag) + ".npz")
    final = {f"{c.mat.name}:{x}": getattr(c, x) if x != "N" else c.number()
             for c in comps for x in ("M", "N")}
    edges = {f"{c.mat.name}:m_edges": c.grid.m_edges for c in comps}
    np.savez(path, times=np.array(times), names=np.array(names),
             values=np.array([[r[k] for k in names] for r in rows]),
             gas_C=np.array([g["C"] for g in gases]), gas_Si=np.array([g["Si"] for g in gases]),
             meta=json.dumps(dict(scenario=scenario, scheme=scheme, n_bins=n_bins,
                                  a_min_c=a_min_c, a_min_si=a_min_si, ic=ic,
                                  dt_hydro_yr=dt_hydro_yr, quick=quick, wall=wall,
                                  conservation=cons, ref_dt_factor=ref_dt_factor,
                                  bins={c.mat.name: c.grid.n_bins for c in comps}, **stats)),
             **final, **edges)
    print(f"{os.path.basename(path)}: {wall:.1f} s, {stats['hydro_steps']} steps, "
          f"conservation {max(cons.values()):.1e}", flush=True)
    return path


def _run_task(kw):
    return run_one(**kw)


def cmd_run(args):
    scenarios = list(SCENARIOS) if args.scenario == "all" else args.scenario.split(",")
    schemes = SCHEMES if args.schemes == "all" else tuple(args.schemes.split(","))
    n_list = (4, 8, 16) if args.quick and not args.n_bins else (
        tuple(int(x) for x in args.n_bins.split(",")) if args.n_bins else N_BINS)
    base = dict(a_min_c=args.a_min_c, a_min_si=args.a_min_si, ic=args.ic,
                dt_hydro_yr=args.dt_hydro, quick=args.quick, threads=args.threads)
    mie_table()                      # build the cache once, before any workers
    tasks = []
    for sc in scenarios:
        if not args.no_ref:
            tasks.append(dict(scenario=sc, scheme="pl2m", n_bins=None, **base))
            if args.ref_check:
                tasks.append(dict(scenario=sc, scheme="pl2m", n_bins=None, ref_dt_factor=0.5,
                                  tag="_halfdt", **base))
        if args.ref_only:
            continue
        for scheme, nb in itertools.product(schemes, n_list):
            tasks.append(dict(scenario=sc, scheme=scheme, n_bins=nb, **base))
    if args.workers <= 1:
        for t in tasks:
            run_one(**t)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(_run_task, tasks))


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def _load(path):
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    names = [str(x) for x in d["names"]]
    return meta, d["times"], names, d["values"]


def cmd_report(args):
    import glob

    out = _out_dir()
    files = sorted(glob.glob(os.path.join(out, "*.npz")))
    runs = [(_load(f), f) for f in files if not f.endswith("mie_table.npz")]
    refs = {(m["scenario"], round(m["a_min_c"] * 1e8, 2)): (t, n, v)
            for (m, t, n, v), f in runs if m["n_bins"] is None and m["ref_dt_factor"] == 1.0}
    rows = []
    for (m, t, names, v), f in runs:
        key = (m["scenario"], round(m["a_min_c"] * 1e8, 2))
        if key not in refs:
            continue
        tr, nr, vr = refs[key]
        if len(tr) != len(t) or not np.allclose(tr, t):
            continue
        if m["n_bins"] is None and m["ref_dt_factor"] == 1.0:
            continue
        err = np.array([[errors(dict(zip(names, vi)), dict(zip(nr, vri)))[k] for k in nr]
                        for vi, vri in zip(v[1:], vr[1:])])
        worst = err.max(axis=0)
        nb = m["n_bins"]
        scal = (sum(m["bins"].values()) * (2 if m["scheme"] == "pl2m" else 1)) if nb else None
        rows.append(dict(scenario=m["scenario"], scheme=m["scheme"] + ("-halfdt" if nb is None else ""),
                         n_bins=nb, a_min_c=key[1], err=float(worst.max()),
                         worst_metric=nr[int(worst.argmax())], per_metric=dict(zip(nr, worst.tolist())),
                         scalars=scal, wall=m["wall"],
                         wall_per_step=m["wall"] / max(m["hydro_steps"], 1),
                         substeps_per_step=m["coll_substeps"] / max(m["hydro_steps"], 1) / 2,
                         conservation=max(m["conservation"].values())))
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    lines = ["# GSD bin convergence", "",
             "| scenario | a_min(C) [A] | scheme | N | scalars | max err | worst metric | "
             "wall/step [ms] | coll. sub-steps/step | conservation |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["scenario"], r["a_min_c"], r["scheme"], r["n_bins"] or 0)):
        lines.append(f"| {r['scenario']} | {r['a_min_c']} | {r['scheme']} | {r['n_bins'] or 'ref'} | "
                     f"{r['scalars'] or ''} | {r['err']:.3g} | {r['worst_metric']} | "
                     f"{1e3 * r['wall_per_step']:.2f} | {r['substeps_per_step']:.2f} | "
                     f"{r['conservation']:.1e} |")
    lines += ["", "## Smallest N within tolerance (that N and every larger one tested)", "",
              "| scenario | a_min(C) [A] | scheme | " + " | ".join(f"tol {t:g}" for t in args.tol) + " |",
              "|---|---|---|" + "---|" * len(args.tol)]
    groups = {}
    for r in rows:
        if r["n_bins"]:
            groups.setdefault((r["scenario"], r["a_min_c"], r["scheme"]), []).append(r)
    for (sc, am, scheme), rs in sorted(groups.items()):
        rs = sorted(rs, key=lambda r: r["n_bins"])
        cells = []
        for tol in args.tol:
            ok = [r["err"] <= tol for r in rs]
            n_min = next((rs[i]["n_bins"] for i in range(len(rs)) if all(ok[i:])), None)
            cells.append(str(n_min) if n_min else f"> {rs[-1]['n_bins']}")
        lines.append(f"| {sc} | {am} | {scheme} | " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(out, "summary.md"), "w") as fh:
        fh.write(text)
    print(text)
    if not args.no_plots:
        plot(rows, refs)


def plot(rows, refs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter, ScalarFormatter
    from pycalima.plotting_style import use_calima_style

    use_calima_style()
    scen = sorted({r["scenario"] for r in rows if r["n_bins"]})
    if not scen:
        return
    fig, axes = plt.subplots(2, len(scen), figsize=(3.6 * len(scen), 6.4), squeeze=False)
    for j, sc in enumerate(scen):
        for scheme, mk in zip(SCHEMES, ("o", "s", "^")):
            rs = sorted([r for r in rows if r["scenario"] == sc and r["scheme"] == scheme
                         and r["n_bins"] and r["a_min_c"] == min(x["a_min_c"] for x in rows)],
                        key=lambda r: r["n_bins"])
            if not rs:
                continue
            lab = scheme.replace("-", " ")
            axes[0, j].loglog([r["n_bins"] for r in rs], [r["err"] for r in rs], marker=mk, label=lab)
            axes[1, j].loglog([r["scalars"] for r in rs], [r["err"] for r in rs], marker=mk, label=lab)
        for ax in axes[:, j]:
            ax.axhline(0.05, color="0.5", lw=0.8, ls="--")
            ax.minorticks_on()
            ax.tick_params(which="both", axis="both", direction="in")
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(NullFormatter())
        axes[0, j].set_xticks([2, 4, 8, 16, 32])
        axes[0, j].set_title(sc.replace("_", " "), fontsize=13)
        axes[0, j].set_xlabel(r"bins per composition", fontsize=13)
        axes[1, j].set_xlabel(r"scalars per cell", fontsize=13)
    axes[0, 0].set_ylabel(r"max relative error", fontsize=14)
    axes[1, 0].set_ylabel(r"max relative error", fontsize=14)
    axes[0, 0].legend(frameon=False, fontsize=10)
    fig.tight_layout()
    path = os.path.join(_paths.get_plots_dir(), "gsd_bin_convergence.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="evolve the reference and the coarse grids")
    r.add_argument("--scenario", default="all", help=f"one of {', '.join(SCENARIOS)}, comma list, or all")
    r.add_argument("--schemes", default="all", help=f"comma list of {', '.join(SCHEMES)}, or all")
    r.add_argument("--n-bins", default="", help="comma list of bins per composition (default: the full set)")
    r.add_argument("--a-min-c", type=float, default=3.5e-8, help="carbon a_min [cm]")
    r.add_argument("--a-min-si", type=float, default=4.5e-8, help="silicate a_min [cm]")
    r.add_argument("--ic", choices=("wd01", "mrn"), default="wd01")
    r.add_argument("--dt-hydro", type=float, default=1e4, help="coarse-run step [yr]")
    r.add_argument("--quick", action="store_true", help="durations x0.1 and N in (4, 8, 16)")
    r.add_argument("--no-ref", action="store_true", help="skip the reference run")
    r.add_argument("--ref-only", action="store_true", help="run only the reference")
    r.add_argument("--ref-check", action="store_true", help="also run the reference at half its steps")
    r.add_argument("--workers", type=int, default=1, help="runs in parallel (processes)")
    r.add_argument("--threads", type=int, default=1, help="threads per run for the pair sums")
    r.set_defaults(func=cmd_run)
    p = sub.add_parser("report", help="errors against the reference, N_min table, figure")
    p.add_argument("--tol", type=float, nargs="+", default=[0.05, 0.10])
    p.add_argument("--no-plots", action="store_true")
    p.set_defaults(func=cmd_report)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
