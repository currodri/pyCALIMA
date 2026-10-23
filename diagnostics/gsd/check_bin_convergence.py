#!/usr/bin/env python3
"""How few size bins does the GSD need? Fast convergence cases for RAMSES.

RAMSES-CALIMA has to evolve the grain size distribution in millions of cells
over millions of steps, so the question is the cheapest representation
(scheme x number of bins) that still follows the exact or a high-resolution
answer under accretion, shattering, coagulation and thermal sputtering.

Cases: a simple initial distribution in one gas, for a few Myr.
  sn_diffuse        SN dust (lognormal, a0 = 0.1 um, sigma = 0.47; Asano et
                    al. 2013b) in the WNM: shattering, coagulation, accretion
  mrn_accretion     MRN (a^-3.5, 5 nm - 0.25 um) in a dense cloud: accretion
  mrn_coagulation   MRN in a dense cloud: coagulation
  mrn_shattering    MRN in the turbulent WIM: shattering
  mrn_sputtering    MRN in hot gas: thermal sputtering at constant da/dt
Accretion (da/dt independent of a, with gas depletion) and sputtering at a
constant rate move every grain by the same Delta a(t), so their reference is
exact: the continuous MRN mapped along the characteristics, with Delta a from
the closed-box ODE for accretion (``analytic``). The other cases compare with
the high-resolution run, which the exact cases also check.

Gas: HY09 Table 1 phases (Hirashita & Yan 2009, MNRAS 394, 1061: T, n_H,
injection scale L and velocity V, the Ormel & Cuzzi 2007 driving) and hot
post-shock gas. Size range: carbon from 0.35 nm (N_C ~ 20, Draine & Li 2001;
``--a-min-c 5e-8`` for the N_C ~ 50-60 photostability limit), silicate from
0.45 nm (Hensley & Draine 2023); coarse grids end at 1 um (the FIR-emission
to extinction budget; Zubko+04), the reference runs to 5 um at 30 bins per
mass decade.

Schemes, all through the same ``pycalima.gsd`` operators:
  pl2m        two moments (N, M) per bin, power law inside (2 scalars/bin)
  1m-mrn      one moment M; inside each bin dn/da ~ a^-3.5 (1 scalar/bin)
  1m-limiter  one moment M; in-bin slope from the neighbours, harmonic-mean
              limiter of Marchand et al. (2021, Ishinisan; collapse.grid)
A one-moment scheme stores only M: before every operator its N is rebuilt
from the closure and afterwards discarded, as a RAMSES passive scalar would.
Coarse runs step at a fixed hydro dt (``--dt-hydro``), the cadence at which
RAMSES would re-impose the in-bin shape.

Metrics at 10%, 30% and 100% of the run: per composition the dust mass, the
power-law slope of dn/da (mass-weighted fit over 3 nm - 0.25 um) and the
shape (mass-weighted mean and width of ln a); for the mixture A_V/N_H, R_V,
the 2175 A bump over its continuum, the far-UV slope A(0.1)/A(0.15) and the
curve A_lambda/A_V from 0.1 to 2.2 um (Mie, Draine & Lee 1984 optics).
Errors are relative. Cost: scalars per cell, collision sub-steps and wall
time per hydro step.

Usage
-----
    python diagnostics/gsd/check_bin_convergence.py run --case mrn_accretion
    python diagnostics/gsd/check_bin_convergence.py run --workers 2
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
MRN_RANGE = (5e-7, 2.5e-5)    # Mathis, Rumpl & Nordsieck (1977)
SN_A0, SN_SIGMA = 1e-5, 0.47  # Asano et al. 2013b

# HY09 Table 1 (T [K], n_H [cm^-3], L [pc], V [km/s]) and hot gas;
# dt_ref [yr]: the reference step where operators are split or gas couples
PHASES = {
    "WNM": dict(T=6000.0, n_H=0.3, L=100.0, V=20.0, dt_ref=2e5),
    "WIM": dict(T=8000.0, n_H=0.1, L=100.0, V=20.0, dt_ref=2e5),
    "DC": dict(T=10.0, n_H=1e4, L=1.0, V=1.5, dt_ref=2e3),
    "HOT": dict(T=3e6, n_H=1.0, L=0.0, V=0.0, dt_ref=2e2),
}

CASES = {
    "sn_diffuse": dict(ic="sn", phase="WNM", myr=5.0,
                       processes={"shattering", "coagulation", "accretion"}),
    "mrn_accretion": dict(ic="mrn", phase="DC", myr=5.0, processes={"accretion"},
                          exact="accretion"),
    "mrn_coagulation": dict(ic="mrn", phase="DC", myr=10.0, processes={"coagulation"}),
    "mrn_shattering": dict(ic="mrn", phase="WIM", myr=10.0, processes={"shattering"}),
    "mrn_sputtering": dict(ic="mrn", phase="HOT", myr=0.03, processes={"sputtering"},
                           exact="sputtering"),
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

    @property
    def per_gram(self):
        """Key-species atoms per gram of grain material."""
        s = self.sublimation
        return s.key_per_unit / (s.unit_mass_amu * M_AMU)

    def mass(self, a):
        return 4.0 / 3.0 * np.pi * self.shatter.rho * np.asarray(a) ** 3


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


def ic_dnda(ic, a):
    """Shape of the initial dn/da (normalised later to the dust mass)."""
    a = np.asarray(a, float)
    if ic == "mrn":
        return np.where((a >= MRN_RANGE[0]) & (a <= MRN_RANGE[1]), a ** -3.5, 0.0)
    return a ** -4 * np.exp(-np.log(a / SN_A0) ** 2 / (2 * SN_SIGMA ** 2))


def ic_samples(mat, ic, n=4001):
    """The continuous initial distribution as quadrature nodes (a, dN) per H,
    holding F_DEP0 of the key element."""
    lo, hi = (max(MRN_RANGE[0], mat.a_min), MRN_RANGE[1]) if ic == "mrn" else (mat.a_min, A_MAX)
    a = np.geomspace(lo, hi, n)
    w = np.full(n, np.log(hi / lo) / (n - 1))
    w[[0, -1]] *= 0.5
    dN = ic_dnda(ic, a) * a * w
    return a, dN * F_DEP0 * SOLAR[mat.key] / mat.per_gram / (dN * mat.mass(a)).sum()


def initial_bins(grid, mat, ic):
    """(N, M) per bin of the initial distribution, per H."""
    a, dN = ic_samples(mat, ic, n=200001)
    k = np.clip(grid.bin_of_radius(a), 0, grid.n_bins - 1)
    return (np.bincount(k, dN, grid.n_bins),
            np.bincount(k, dN * mat.mass(a), grid.n_bins))


# ---------------------------------------------------------------------------
# grids and schemes
# ---------------------------------------------------------------------------

def make_grid(mat, n_bins):
    from pycalima.gsd.grid import MassGrid

    if n_bins is None:
        return MassGrid.from_radii(mat.a_min, A_REF_MAX, REF_BPD, mat.shatter.rho)
    m = mat.mass(np.array([mat.a_min, A_MAX]))
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

    def number(self):
        return self.N if self.scheme == "pl2m" else closure_N(self.grid, self.M, self.scheme)


def collision_model(c, phase_name, processes, threads):
    from pycalima.gsd.shattering import GrainCollisions
    from pycalima.gsd.velocities import RelativeVelocity

    if phase_name not in c.models:
        p = PHASES[phase_name]
        rv = RelativeVelocity(T=p["T"], n_H=p["n_H"], mu=1.4, v_turb=p["V"] * 1e5, L=p["L"] * PC)
        c.models[phase_name] = GrainCollisions(
            c.mat.shatter, rv, law="KT10", coagulation="coagulation" in processes,
            v_shat=None if "shattering" in processes else 1e30, a_frag_min=c.mat.a_min,
            collector="remove", n_speed=N_SPEED, table_ppd=TABLE_PPD, threads=threads)
    return c.models[phase_name]


def erosion(mat, phase_name, rho):
    """Constant sputtering da/dt (no size correction), so the case is exact."""
    from pycalima.gsd.sputtering import erosion_rate

    p = PHASES[phase_name]
    return erosion_rate(mat.name, p["T"], p["n_H"], rho=rho, size_correction=False)


def step(comps, gas, phase_name, dt, processes, threads, stats):
    """One split step of all compositions; gas and state per H."""
    from pycalima.gsd.collisions import collide
    from pycalima.gsd.exchange import Population, exchange_step
    from pycalima.gsd.sputtering import sputter

    p = PHASES[phase_name]
    nH = p["n_H"]
    for c in comps:
        N, M = c.number() * nH, c.M * nH
        t0 = time.perf_counter()
        if processes & {"shattering", "coagulation"}:
            r = collide(c.grid, N, M, collision_model(c, phase_name, processes, threads), dt,
                        errmax=ERRMAX)
            N, M = r["N"], r["M"]
            c.over_M += r["overflow_M"] / nH
            gas[c.mat.key] += r["sink_M"] / nH * c.mat.per_gram
            stats["coll_substeps"] += r["n_steps"]
        t1 = time.perf_counter()
        if "accretion" in processes and gas[c.mat.key] > 0.0:
            pop = Population(c.mat.sublimation, c.grid, N, M, np.full(c.grid.n_bins, T_DUST),
                             nucleation=False)
            g = {c.mat.key: gas[c.mat.key] * nH}
            exchange_step([pop], g, p["T"], dt)
            N, M = pop.N, pop.M
            c.over_M += pop.overflow_M / nH
            gas[c.mat.key] = g[c.mat.key] / nH
        t2 = time.perf_counter()
        if "sputtering" in processes:
            if phase_name not in c.adots:
                c.adots[phase_name] = erosion(c.mat, phase_name, c.grid.rho)
            N, M, dM = sputter(c.grid, N, M, c.adots[phase_name], dt)
            gas[c.mat.key] += dM / nH * c.mat.per_gram
        t3 = time.perf_counter()
        c.M = M / nH
        if c.scheme == "pl2m":
            c.N = N / nH
        stats["t_coll"] += t1 - t0
        stats["t_acc"] += t2 - t1
        stats["t_sput"] += t3 - t2


# ---------------------------------------------------------------------------
# observables, from quadrature nodes (a, dN) of any distribution
# ---------------------------------------------------------------------------

def mie_table():
    """C_ext(a, lambda) [cm^2] on a log grid of radii, per material (cached)."""
    from pycalima.models.dust_radiation.dust_oppacity import compute_cross_sections_mie

    path = os.path.join(_out_dir(), "mie_table.npz")
    a = np.geomspace(3e-8, 6e-4, 160)
    if os.path.exists(path):
        d = np.load(path)
        if d["a"].shape == a.shape and d["lam"].shape == (len(LAMBDAS_UM),) \
                and np.allclose(d["lam"], LAMBDAS_UM):
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


def bin_samples(grid, N, M):
    """Quadrature nodes (a, dN) of the in-bin profiles (1% steps in a)."""
    from pycalima.gsd.pl2m import BinProfile

    prof = BinProfile(grid, N, M)
    a_out, w_out = [], []
    for i in np.nonzero(prof.N > 0)[0]:
        if prof.delta[i]:
            a_out.append([prof.aL[i] * prof.u_delta[i]])
            w_out.append([prof.N[i]])
            continue
        n = max(12, int(np.ceil(np.log(prof.uR) / 0.01)))
        u = np.geomspace(1.0, prof.uR, n + 1)
        w = np.full(n + 1, np.log(prof.uR) / n)
        w[[0, -1]] *= 0.5
        a_out.append(prof.aL[i] * u)
        w_out.append(prof.A[i] * u ** (prof.lam[i] + 1.0) * w)
    if not a_out:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(a_out), np.concatenate(w_out)


def metrics(dists, mie):
    """Named observables of {material: (Material, a, dN, overflow mass)}."""
    out = {}
    A = 0.0
    for n, (mat, a, dN, over) in dists.items():
        m = mat.mass(a) * dN
        Mg = max(m.sum(), 1e-300)
        out[f"{n}:M_dust"] = m.sum() + over
        out[f"{n}:slope"] = _power_law_slope(a, dN, m)
        l1 = (m * np.log(a)).sum() / Mg                      # shape: mass-weighted
        l2 = (m * np.log(a) ** 2).sum() / Mg                 # mean and width of ln a
        out[f"{n}:a_m"] = np.exp(l1)
        out[f"{n}:width"] = np.sqrt(max(l2 - l1 ** 2, 0.0))
        la, lC = np.log(mie["a"]), np.log(mie[n])
        C = np.exp(np.array([np.interp(np.log(a), la, lC[:, k]) for k in range(lC.shape[1])]))
        A = A + 1.086 * (C * dN).sum(axis=1)                  # A_lambda / N_H [mag cm^2]
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


def _power_law_slope(a, dN, m):
    """Exponent of the best power law dn/da ~ a^p over SLOPE_RANGE: a fit of
    ln(dN/d ln a) on 12 log intervals, weighted by the mass in each."""
    e = np.geomspace(*SLOPE_RANGE, 13)
    k = np.searchsorted(e, a) - 1
    inside = (k >= 0) & (k < 12)
    Nk = np.bincount(k[inside], dN[inside], 12)
    Mk = np.bincount(k[inside], m[inside], 12)
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

def snapshot_times(case):
    return [f * CASES[case]["myr"] * MYR for f in SNAP_FRACTIONS]


def _out_dir():
    d = os.path.join(_paths.get_results_dir(), "gsd_bin_convergence")
    os.makedirs(d, exist_ok=True)
    return d


def run_name(case, scheme, n_bins, a_min_c, tag=""):
    nb = "ref" if n_bins is None else f"N{n_bins:02d}"
    return f"{case}_{scheme}_{nb}_amin{a_min_c * 1e8:.2f}A{tag}"


def _save(case, scheme, n_bins, a_min_c, tag, times, rows, meta, extra=None):
    names = list(rows[0])
    path = os.path.join(_out_dir(), run_name(case, scheme, n_bins, a_min_c, tag) + ".npz")
    np.savez(path, times=np.array(times), names=np.array(names),
             values=np.array([[r[k] for k in names] for r in rows]),
             meta=json.dumps(dict(case=case, scheme=scheme, n_bins=n_bins, a_min_c=a_min_c,
                                  **meta)), **(extra or {}))
    return path


def run_one(case, scheme, n_bins, a_min_c, a_min_si, dt_hydro_yr, threads,
            ref_dt_factor=1.0, tag=""):
    """Evolve one scheme/grid through a case; write and return its npz path."""
    os.environ.setdefault("PYCALIMA_GSD_THREADS", str(threads))
    cs = CASES[case]
    ph, processes = cs["phase"], cs["processes"]
    comps, gas = [], {}
    for mat in materials(a_min_c, a_min_si):
        g = make_grid(mat, n_bins)
        N, M = initial_bins(g, mat, cs["ic"])
        comps.append(Composition(mat, g, scheme, M, N if scheme == "pl2m" else None))
        gas[mat.key] = SOLAR[mat.key] * (1.0 - F_DEP0)
    totals0 = {c.mat.key: gas[c.mat.key] + c.M.sum() * c.mat.per_gram for c in comps}
    mie = mie_table()

    def state():
        return {c.mat.name: (c.mat, *bin_samples(c.grid, c.number(), c.M), c.over_M)
                for c in comps}

    stats = dict(coll_substeps=0, t_coll=0.0, t_acc=0.0, t_sput=0.0, hydro_steps=0)
    times, rows = [0.0], [metrics(state(), mie)]
    if n_bins is None:
        # small steps only where operators are split or the gas couples (the
        # backward-Euler exchange); collisions alone sub-step themselves and
        # sputtering at fixed T is exact
        ops = ({"collisions"} if processes & {"shattering", "coagulation"} else set()) \
            | (processes & {"accretion", "sputtering"})
        split = len(ops) > 1 or "accretion" in ops
        dt_target = PHASES[ph]["dt_ref"] * YR * ref_dt_factor if split else np.inf
    else:
        dt_target = dt_hydro_yr * YR
    wall0 = time.perf_counter()
    t = 0.0
    for t1 in snapshot_times(case):
        n = max(int(np.ceil((t1 - t) / dt_target - 1e-9)), 1)
        for _ in range(n):
            step(comps, gas, ph, (t1 - t) / n, processes, threads, stats)
        stats["hydro_steps"] += n
        t = t1
        times.append(t1)
        rows.append(metrics(state(), mie))
    wall = time.perf_counter() - wall0
    cons = {k: abs(gas[k] + sum((c.M.sum() + c.over_M) * c.mat.per_gram
                                for c in comps if c.mat.key == k) - v) / v
            for k, v in totals0.items()}
    extra = {f"{c.mat.name}:{x}": (c.M if x == "M" else c.number()) for c in comps
             for x in ("M", "N")}
    extra.update({f"{c.mat.name}:m_edges": c.grid.m_edges for c in comps})
    path = _save(case, scheme, n_bins, a_min_c, tag, times, rows,
                 dict(a_min_si=a_min_si, dt_hydro_yr=dt_hydro_yr, wall=wall, conservation=cons,
                      ref_dt_factor=ref_dt_factor,
                      bins={c.mat.name: c.grid.n_bins for c in comps}, **stats), extra)
    print(f"{os.path.basename(path)}: {wall:.1f} s, {stats['hydro_steps']} steps, "
          f"conservation {max(cons.values()):.1e}", flush=True)
    return path


def run_exact(case, a_min_c, a_min_si):
    """The exact solution of an accretion or constant-rate sputtering case:
    every grain moves by the same Delta a(t)."""
    from scipy.integrate import solve_ivp

    cs = CASES[case]
    p = PHASES[cs["phase"]]
    mats = materials(a_min_c, a_min_si)
    samples = {m.name: ic_samples(m, cs["ic"]) for m in mats}
    mie = mie_table()
    times = [0.0] + snapshot_times(case)
    shift = {}
    for m in mats:
        a0, dN = samples[m.name]
        if cs["exact"] == "sputtering":
            rate = abs(float(erosion(m, cs["phase"], m.shatter.rho)(1e-5)))
            shift[m.name] = [rate * t for t in times]
            continue
        # accretion: da/dt = alpha V v_X n_H x_gas / 4k (exchange.Population.adot),
        # x_gas = x_tot - key atoms in the grains grown by Delta a (closed box)
        s = m.sublimation
        v_X = np.sqrt(8 * 1.380649e-16 * p["T"] / (np.pi * s.key_mass_amu * M_AMU))
        c = STICKING * s.unit_volume_cm3 * v_X * p["n_H"] / (4.0 * s.key_per_unit)
        x_tot = SOLAR[m.key]

        def rhs(_, y):
            dust = (dN * m.mass(a0 + y[0])).sum() * m.per_gram
            return [c * max(x_tot - dust, 0.0)]

        sol = solve_ivp(rhs, (0.0, times[-1]), [0.0], t_eval=times, rtol=1e-11, atol=1e-20,
                        method="LSODA")
        shift[m.name] = list(sol.y[0])
    rows = []
    for k in range(len(times)):
        dists = {}
        for m in mats:
            a0, dN = samples[m.name]
            a = a0 + shift[m.name][k] * (1 if cs["exact"] == "accretion" else -1)
            keep = a > m.a_min                       # eroded below the grid: gone
            dists[m.name] = (m, a[keep], dN[keep], 0.0)
        rows.append(metrics(dists, mie))
    path = _save(case, "exact", None, a_min_c, "", times, rows,
                 dict(a_min_si=a_min_si, shift_cm={k: v for k, v in shift.items()}))
    print(f"{os.path.basename(path)}: exact", flush=True)
    return path


def _run_task(kw):
    return run_one(**kw)


def cmd_run(args):
    cases = list(CASES) if args.case == "all" else args.case.split(",")
    schemes = SCHEMES if args.schemes == "all" else tuple(args.schemes.split(","))
    n_list = tuple(int(x) for x in args.n_bins.split(",")) if args.n_bins else N_BINS
    base = dict(a_min_c=args.a_min_c, a_min_si=args.a_min_si, dt_hydro_yr=args.dt_hydro,
                threads=args.threads)
    mie_table()                      # build the cache once, before any workers
    tasks = []
    for cs in cases:
        if CASES[cs].get("exact"):
            run_exact(cs, args.a_min_c, args.a_min_si)
        if not args.no_ref:
            tasks.append(dict(case=cs, scheme="pl2m", n_bins=None, **base))
            if args.ref_check:
                tasks.append(dict(case=cs, scheme="pl2m", n_bins=None, ref_dt_factor=0.5,
                                  tag="_halfdt", **base))
        if args.ref_only:
            continue
        for scheme, nb in itertools.product(schemes, n_list):
            tasks.append(dict(case=cs, scheme=scheme, n_bins=nb, **base))
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
    return json.loads(str(d["meta"])), d["times"], [str(x) for x in d["names"]], d["values"]


def cmd_report(args):
    import glob

    out = _out_dir()
    runs = [_load(f) for f in sorted(glob.glob(os.path.join(out, "*.npz")))
            if not f.endswith("mie_table.npz")]
    key = lambda m: (m["case"], round(m["a_min_c"] * 1e8, 2))  # noqa: E731
    highres = {key(m): (t, n, v) for m, t, n, v in runs
               if m["scheme"] == "pl2m" and m["n_bins"] is None and m.get("ref_dt_factor") == 1.0}
    exact = {key(m): (t, n, v) for m, t, n, v in runs if m["scheme"] == "exact"}
    rows = []
    for m, t, names, v in runs:
        k = key(m)
        is_ref = m["n_bins"] is None
        if m["scheme"] == "exact" or (is_ref and m.get("ref_dt_factor") == 1.0 and k not in exact):
            continue
        tr, nr, vr = exact.get(k) or highres.get(k) or (None, None, None)
        if tr is None or len(tr) != len(t) or not np.allclose(tr, t):
            continue
        err = np.array([[errors(dict(zip(names, vi)), dict(zip(nr, vri)))[q] for q in nr]
                        for vi, vri in zip(v[1:], vr[1:])])
        worst = err.max(axis=0)
        label = m["scheme"] if not is_ref else ("ref, half dt" if m["ref_dt_factor"] != 1.0
                                                 else f"ref ({REF_BPD}/dec)")
        scal = (sum(m["bins"].values()) * (2 if m["scheme"] == "pl2m" else 1)) if not is_ref else None
        rows.append(dict(case=m["case"], scheme=label, n_bins=m["n_bins"], a_min_c=k[1],
                         against="exact" if k in exact else "high-res",
                         err=float(worst.max()), worst_metric=nr[int(worst.argmax())],
                         per_metric=dict(zip(nr, worst.tolist())), scalars=scal, wall=m["wall"],
                         wall_per_step=m["wall"] / max(m["hydro_steps"], 1),
                         substeps_per_step=m["coll_substeps"] / max(m["hydro_steps"], 1) / 2,
                         conservation=max(m["conservation"].values())))
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    lines = ["# GSD bin convergence", "",
             "| case | a_min(C) [A] | vs | scheme | N | scalars | max err | worst metric | "
             "wall/step [ms] | coll. sub-steps/step | conservation |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["case"], r["a_min_c"], r["scheme"], r["n_bins"] or 0)):
        lines.append(f"| {r['case']} | {r['a_min_c']} | {r['against']} | {r['scheme']} | "
                     f"{r['n_bins'] or ''} | {r['scalars'] or ''} | {r['err']:.3g} | "
                     f"{r['worst_metric']} | {1e3 * r['wall_per_step']:.2f} | "
                     f"{r['substeps_per_step']:.2f} | {r['conservation']:.1e} |")
    lines += ["", "## Smallest N within tolerance (that N and every larger one tested)", "",
              "| case | a_min(C) [A] | scheme | " + " | ".join(f"tol {t:g}" for t in args.tol) + " |",
              "|---|---|---|" + "---|" * len(args.tol)]
    groups = {}
    for r in rows:
        if r["n_bins"]:
            groups.setdefault((r["case"], r["a_min_c"], r["scheme"]), []).append(r)
    for (cs, am, scheme), rs in sorted(groups.items()):
        rs = sorted(rs, key=lambda r: r["n_bins"])
        cells = []
        for tol in args.tol:
            ok = [r["err"] <= tol for r in rs]
            n_min = next((rs[i]["n_bins"] for i in range(len(rs)) if all(ok[i:])), None)
            cells.append(str(n_min) if n_min else f"> {rs[-1]['n_bins']}")
        lines.append(f"| {cs} | {am} | {scheme} | " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(out, "summary.md"), "w") as fh:
        fh.write(text)
    print(text)
    if not args.no_plots:
        plot(rows)


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter, ScalarFormatter
    from pycalima.plotting_style import use_calima_style

    use_calima_style()
    cases = [c for c in CASES if any(r["case"] == c and r["n_bins"] for r in rows)]
    if not cases:
        return
    a_min = min(r["a_min_c"] for r in rows)
    fig, axes = plt.subplots(2, len(cases), figsize=(3.6 * len(cases), 6.4), squeeze=False)
    for j, cs in enumerate(cases):
        for scheme, mk in zip(SCHEMES, ("o", "s", "^")):
            rs = sorted([r for r in rows if r["case"] == cs and r["scheme"] == scheme
                         and r["n_bins"] and r["a_min_c"] == a_min], key=lambda r: r["n_bins"])
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
        axes[0, j].set_title(cs.replace("_", " "), fontsize=13)
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
    r = sub.add_parser("run", help="evolve the references and the coarse grids")
    r.add_argument("--case", default="all", help=f"one of {', '.join(CASES)}, comma list, or all")
    r.add_argument("--schemes", default="all", help=f"comma list of {', '.join(SCHEMES)}, or all")
    r.add_argument("--n-bins", default="", help="comma list of bins per composition (default: the full set)")
    r.add_argument("--a-min-c", type=float, default=3.5e-8, help="carbon a_min [cm]")
    r.add_argument("--a-min-si", type=float, default=4.5e-8, help="silicate a_min [cm]")
    r.add_argument("--dt-hydro", type=float, default=1e4, help="coarse-run step [yr]")
    r.add_argument("--no-ref", action="store_true", help="skip the high-resolution run")
    r.add_argument("--ref-only", action="store_true", help="run only the references")
    r.add_argument("--ref-check", action="store_true", help="also run the reference at half its steps")
    r.add_argument("--workers", type=int, default=1, help="runs in parallel (processes)")
    r.add_argument("--threads", type=int, default=1, help="threads per run for the pair sums")
    r.set_defaults(func=cmd_run)
    p = sub.add_parser("report", help="errors against the references, N_min table, figure")
    p.add_argument("--tol", type=float, nargs="+", default=[0.05, 0.10])
    p.add_argument("--no-plots", action="store_true")
    p.set_defaults(func=cmd_report)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
