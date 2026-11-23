#!/usr/bin/env python3
"""One-zone runs of the carbon grid under the candidate collision rules (carbon-transfer study of 2026-11-23, check 3;
results/carbon_plan/carbon_transfer_2026-11-23/).

Start: THEMIS J13 carbon on ``CarbonLayout`` (as check_carbon_transfer_timescales). Processes: grain collisions
(``gsd.collisions.collide`` with ``GrainCollisions`` on 'themis_ac': graphite strength as the a-C proxy, KT10, fragments
a^-3.3 to the grid's 0.39 nm, below it to the gas; HY09 sticking; neutral grains, no Coulomb factors) and the molecular
bins' carbon loss (pahchem hydrogen tables, deep +8, as a downward shift with each bin's per-molecule C loss rate,
``gsd.sputtering.sputter`` with that da/dt; below the grid to the gas). Operator split per step: collisions, then
carbon loss.

Collision rules compared (applied by masking the rate coefficients of the model's channels, i.e. the generic pair-sum
path of ``collisions._terms``; ``gsd`` itself is unchanged):
- O0: the code as it is: every pair collides with grain physics (molecule + molecule included);
- O1: no molecule + molecule collisions; molecule + grain collisions only stick (freeze-out), never shatter;
- O2: O1 and molecules stick only to grains >= 20 nm (no molecule + nano-grain sticking).

Reported per environment and rule: the carbon mass in the molecular bins (N_C < 400), in the nano-grain bins (1-20
nm), in the grains, and returned to the gas, at the end; and the total grain photoelectric heating with the molecular
bins charged by the PAH tables over that with the 'ac' tables (per-C values at the bin centres for the seam check's
cells; molecular bins from results/carbon_plan/check_pah_grain_seam.txt).

    python diagnostics/gsd/check_carbon_transfer_options.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import check_carbon_heating_share as chs
import check_carbon_transfer_timescales as cts

YR = 3.15576e7
RUNS = {"WNM": dict(myr=30.0, steps=300, cell="WNM"),
        "CNM": dict(myr=30.0, steps=300, cell="CNM"),
        "DC": dict(myr=1.0, steps=100, cell="dense")}
RULES = ("O0", "O1", "O2", "O3")
STEPS_SCALE = 1.0
BROWNIAN = True


class MaskedCollisions:
    """A GrainCollisions whose channels are masked by pair class: molecular (a < a_mol), nano-grain (a_mol <= a <
    a_nano) and grain. Exposes only ``channels`` (the generic pair-sum path)."""

    def __init__(self, base, a_mol, a_nano, rule):
        self.base, self.a_mol, self.a_nano, self.rule = base, a_mol, a_nano, rule
        self.collector, self.alpha_f = base.collector, base.alpha_f

    def channels(self, a, m):
        from pycalima.gsd.collisions import Channel

        chans = self.base.channels(a, m)
        if self.rule == "O0":
            return chans
        mol = a < self.a_mol
        nano = (a >= self.a_mol) & (a < self.a_nano)
        mm = mol[:, None] & mol[None, :]
        if self.rule == "O3":
            return self._o3(chans, a, m, mol, mm)
        any_mol = mol[:, None] | mol[None, :]
        mol_nano = (mol[:, None] & nano[None, :]) | (nano[:, None] & mol[None, :])
        out = []
        for ch in chans:
            shatter = np.any(ch.sets[0][1] > 0) or np.any(ch.sets[1][1] > 0)
            keep = ~mm
            if shatter:
                keep &= ~any_mol                       # molecules never shatter or crater
            elif self.rule == "O2":
                keep &= ~mol_nano                      # molecules stick only to grains >= a_nano
            K = np.where(keep, ch.K, 0.0)
            (r1, f1, h1, l1), (r2, f2, h2, l2) = ch.sets
            out.append(Channel(K, rem1=r1, frag1=f1, a_hi1=h1, a_lo1=l1, rem2=r2, frag2=f2, a_hi2=h2, a_lo2=l2))
        return out


def _o3_channels(self, chans, a, m, mol, mm):
    """O3: grain-grain pairs as the base model; molecule + molecule nothing; molecule + non-molecule by impact speed
    (molecule-surface impacts, carbon_transfer_impacts notes): stick below V_STICK, rebound intact between, molecule
    destroyed above V_DESTROY (its carbon to the collector, i.e. the gas; the partner intact)."""
    from pycalima.gsd.collisions import Channel

    mixed = mol[:, None] ^ mol[None, :]
    out = []
    for ch in chans:
        (r1, f1, h1, l1), (r2, f2, h2, l2) = ch.sets
        out.append(Channel(np.where(mixed | mm, 0.0, ch.K), rem1=r1, frag1=f1, a_hi1=h1, a_lo1=l1, rem2=r2, frag2=f2,
                           a_hi2=h2, a_lo2=l2))
    rv = self.base.velocity
    dV, mu = rv.pair_statistics(a, m, self.rho)
    mu = np.maximum(mu, 1e-10)
    vmean, f_st, _ = rv.collision_speeds(dV, mu, V_STICK, 1)
    _, f_de, _ = rv.collision_speeds(dV, mu, V_DESTROY, 1)
    sig = np.pi * (a[:, None] + a[None, :]) ** 2
    K = sig * vmean
    M1, M2 = np.broadcast_to(m[:, None], K.shape), np.broadcast_to(m[None, :], K.shape)
    out.append(Channel(np.where(mixed, K * (1.0 - f_st), 0.0), rem1=M1 + M2))
    tiny = np.full(K.shape, 0.5 * self.a_min)
    out.append(Channel(np.where(mixed, K * f_de, 0.0),
                       rem1=np.where(mol[:, None], 0.0, M1), frag1=np.where(mol[:, None], M1, 0.0), a_hi1=tiny, a_lo1=tiny,
                       rem2=np.where(mol[None, :], 0.0, M2), frag2=np.where(mol[None, :], M2, 0.0), a_hi2=tiny, a_lo2=tiny))
    return out


MaskedCollisions._o3 = _o3_channels
V_STICK, V_DESTROY = 4.0e5, 7.0e5      # cm/s: ~1 and ~3 eV per atom (C60 and cluster surface impacts)


def carbon_loss_adot(grid, env, n_mol=4):
    """da/dt(a) [cm/s] of the molecular bins' carbon loss: each bin's per-molecule C loss rate, zero above."""
    from pycalima.solvers.table_io import build_pah_photolysis_interpolator_nd

    e = cts.ENVS[env]
    dmdt = []
    for lab in cts.MOL_LABELS:
        f, _ = build_pah_photolysis_interpolator_nd(cts.TABLES / f"{lab}_carbon.dat")
        v = f(np.log10(e["G0"]), np.log10(e["n_H"]), np.log10(min(e["T"], 1e4)), np.log10(e["xe"]))
        dmdt.append(-2.0 * 10.0 ** v * cts.M_C)
    edges = grid.a_edges

    def adot(a):
        a = np.asarray(a, float)
        i = np.clip(np.searchsorted(edges, a, side="right") - 1, 0, grid.n_bins - 1)
        rate = np.where(i < n_mol, np.take(np.array(dmdt + [0.0] * (grid.n_bins - n_mol)), i), 0.0)
        return rate / (4.0 * np.pi * grid.rho * a ** 2)

    return adot


def run(grid, N0, M0, env, rule):
    from pycalima.gsd.collisions import collide
    from pycalima.gsd.materials import get_material
    from pycalima.gsd.shattering import GrainCollisions
    from pycalima.gsd.sputtering import sputter
    from pycalima.gsd.velocities import RelativeVelocity

    e = cts.ENVS[env]
    rv = RelativeVelocity(T=e["T"], n_H=e["n_H"], mu=1.4, v_turb=e["V"] * 1e5, L=e["L"] * cts.PC, brownian=BROWNIAN)
    base = GrainCollisions(get_material("themis_ac").shatter, rv, law="KT10", coagulation=cts.COAG[env],
                           a_frag_min=float(grid.a_edges[0]), collector="remove", n_speed=8, table_ppd=None,
                           threads=1)
    model = MaskedCollisions(base, float(grid.a_edges[4]), 20e-7, rule)
    model.rho, model.a_min = grid.rho, float(grid.a_edges[0])
    adot = carbon_loss_adot(grid, env)
    n = e["n_H"]
    N, M = N0 * n, M0 * n
    gas = 0.0
    r = RUNS[env]
    steps = max(1, int(r["steps"] * STEPS_SCALE))
    dt = r["myr"] * 1e6 * YR / r["steps"]
    substeps = 0
    for _ in range(steps):
        c = collide(grid, N, M, model, dt, errmax=1e-3)
        N, M = c["N"], c["M"]
        gas += c["sink_M"] + c["overflow_M"]
        substeps += c["n_steps"]
        N, M, dM = sputter(grid, N, M, adot, dt)
        gas += dM
    return N / n, M / n, gas / n, substeps


_AC = {}


def heating_ratio(grid, M, cell):
    """Total PE heating with the molecular bins on PAH tables over that with 'ac' (per-C at the bin centres)."""
    pah = chs.pah_per_c("results/carbon_plan/check_pah_grain_seam.txt")[cell]
    if not _AC:
        _AC.update(chs.ac_per_c(grid.a_centres))
    ac = _AC[cell]
    C = M / cts.M_C
    g_ac = float(np.sum(ac[:, 0] * C))
    g_p = float(np.sum(np.where(np.arange(grid.n_bins) < 4, [p[0] for p in pah] + [0.0] * (grid.n_bins - 4),
                                ac[:, 0]) * C))
    return g_p / g_ac


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="results/carbon_plan/carbon_transfer_2026-11-23/options_runs.txt")
    ap.add_argument("--envs", default=",".join(RUNS))
    ap.add_argument("--rules", default=",".join(RULES))
    ap.add_argument("--steps-scale", type=float, default=1.0, help="scale the number of steps (timing tests)")
    ap.add_argument("--no-brownian", action="store_true", help="turbulent (and drift) velocities only, as most galaxy-scale models")
    args = ap.parse_args(argv)
    global STEPS_SCALE, BROWNIAN
    STEPS_SCALE = args.steps_scale
    BROWNIAN = not args.no_brownian
    from pycalima.gsd.carbon import CarbonLayout

    grid = CarbonLayout().grid()
    N0, M0 = cts.bin_state(grid)
    tot = M0.sum()
    lines = [f"One-zone carbon-grid runs from THEMIS J13 (Brownian velocities {'on' if BROWNIAN else 'off'}); carbon shares of the initial carbon mass at the end",
             "env   rule  t[Myr]  molecular  nano(1-20nm)  grains  gas   PE heating PAH/ac   substeps"]
    s0 = (M0[:4].sum() / tot, M0[4:7].sum() / tot, M0[7:].sum() / tot)
    lines.append(f"start             {s0[0]:8.3f}  {s0[1]:10.3f}  {s0[2]:7.3f}  0.000")
    for env in args.envs.split(","):
        for rule in args.rules.split(","):
            N, M, gas, nsub = run(grid, N0, M0, env, rule)
            hr = heating_ratio(grid, M, RUNS[env]["cell"])
            lines.append(f"{env:5s} {rule:4s} {RUNS[env]['myr']:6.1f}  {M[:4].sum() / tot:8.3f}  "
                         f"{M[4:7].sum() / tot:10.3f}  {M[7:].sum() / tot:7.3f}  {gas / tot:5.3f}   {hr:6.2f}"
                         f"          {nsub}")
            lines.append("      bins " + " ".join(f"{x / tot:7.4f}" for x in M))
            print(lines[-2], flush=True)
    txt = "\n".join(lines)
    print(txt)
    Path(args.out).write_text(txt + "\n")


if __name__ == "__main__":
    main()
