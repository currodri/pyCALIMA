#!/usr/bin/env python3
"""Timings of the GSD solver's main workloads (for the performance pass).

    python diagnostics/gsd/benchmark_gsd.py [--quick]

Each line: workload, wall and CPU time per call [ms] (median of repeats
after one warm-up call that includes numba compilation). CPU time is the
process total, so with threads it exceeds the wall time; on a loaded machine
the wall time of threaded workloads is not meaningful (the pair sums then
fall back to fewer threads, see ``collisions.default_threads``).
"""
import argparse
import time

import numpy as np

PC = 3.086e18
MYR = 3.156e13


def timeit(fn, n=5):
    fn()
    ts, cs = [], []
    for _ in range(n):
        t0, c0 = time.perf_counter(), time.process_time()
        fn()
        ts.append(time.perf_counter() - t0)
        cs.append(time.process_time() - c0)
    return 1e3 * float(np.median(ts)), 1e3 * float(np.median(cs))


def mrn(g, rho, D=4e-3):
    a_e = g.a_edges
    N, M = np.zeros(g.n_bins), np.zeros(g.n_bins)
    for i in range(g.n_bins):
        x1, x2 = max(5e-7, a_e[i]), min(2.5e-5, a_e[i + 1])
        if x2 > x1:
            N[i] = (x1 ** -2.5 - x2 ** -2.5) / 2.5
            M[i] = 4 / 3 * np.pi * rho * (x2 ** 0.5 - x1 ** 0.5) / 0.5
    s = D * 1.4 * 1.6735e-24 / M.sum()
    return N * s, M * s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    n = 3 if args.quick else 5
    from pycalima.gsd.grid import MassGrid
    from pycalima.gsd.collisions import _terms, collide
    from pycalima.gsd.coagulation import coagulate
    from pycalima.gsd.exchange import Population, exchange_step
    from pycalima.gsd.shattering import SILICATE, GrainCollisions
    from pycalima.gsd.velocities import RelativeVelocity
    from pycalima.gsd.charging import ChargingModel, PhotonGroups, xray_power_law
    from pycalima.models.dust_radiation.sublimation_kinetics import get_sublimation_model
    from pycalima.models.dust_radiation.agn_hot_dust import AGNIllumination

    rows = []
    # 1. exchange step, graphite, 136 bins
    model = get_sublimation_model("graphite", "GD89")
    pop = Population.empty(model, 3e-8, 1e-3, 10, T_d=1700.0, nucleation=True)
    a = pop.grid.a_centres
    w = np.exp(-np.log(a / 1e-5) ** 2 / (2 * 0.15 ** 2))
    nC = 10 ** -3.44 * 1e10
    pop.M = w / w.sum() * 0.5 * nC * 12.011 * 1.66054e-24
    pop.N = pop.M / pop.grid.mass(a)
    gas = {"C": 0.5 * nC}
    rows.append(("exchange_step (136 bins)", timeit(lambda: exchange_step([pop], gas, 1e4, 1e6), n)))

    # 2-4. collisions on a 120-bin silicate grid
    g = MassGrid.from_radii(3e-8, 3e-4, 10, SILICATE.rho)
    N, M = mrn(g, SILICATE.rho)
    m0 = g.mass(3e-6)
    rows.append(("coagulate additive, 1 coll. time (150-bin grid)",
                 timeit(lambda: coagulate(MassGrid.from_radii(1e-7, 1e-2, 10, 1.0),
                                          *(_exp_bins(1e-7, 1e-2, m0)), lambda x, y: 1e-10 * (x + y) / m0,
                                          1.0 / (1e-10 * _exp_bins(1e-7, 1e-2, m0)[1].sum() / m0)), 1)))
    one = lambda x: (np.full(np.shape(x) + (3,), 20.0), np.array([1, 4, 1]) / 6.0)
    for ppd, ch in ((None, None), (None, one), (24, one)):
        rv = RelativeVelocity(T=8000, n_H=1.0, mu=1.4, v_turb=1e6, L=100 * PC, charges=ch)
        s = GrainCollisions(SILICATE, rv, coagulation=True, table_ppd=ppd)
        lab = f"rate evaluation, shatter+stick, charges={'yes' if ch else 'no'}, table={ppd}"
        rows.append((lab, timeit(lambda: _terms(g, N, M, s), n)))
    rv = RelativeVelocity(T=8000, n_H=1.0, mu=1.4, v_turb=1e6, L=100 * PC)
    s = GrainCollisions(SILICATE, rv, coagulation=True, table_ppd=24)
    t0, c0 = time.perf_counter(), time.process_time()
    r = collide(g, N, M, s, 10 * MYR, errmax=1e-3)
    rows.append((f"collide WIM 10 Myr, table 24 ({r['n_steps']} steps)",
                 (1e3 * (time.perf_counter() - t0), 1e3 * (time.process_time() - c0))))

    # 5. charging, 22 radii, AGN
    agn = AGNIllumination(1e46, law="isotropic")
    cm = ChargingModel("silicate", np.geomspace(3e-8, 1e-4, 22))
    r_cm = 3 * PC
    ph = PhotonGroups.from_flux_lambda(agn.lam_cm, agn.flux(r_cm))
    E, nE = xray_power_law(r_cm, 1e46 / 30)
    ph = ph + PhotonGroups.from_photon_density(E, nE)
    rows.append(("charging solve (22 radii)", timeit(lambda: cm.solve(ph, T=1e4, n_e=1e2), n)))
    st = cm.solve(ph, T=1e4, n_e=1e2)
    rows.append(("   ion_focusing", timeit(lambda: st.ion_focusing(), n)))
    rows.append(("   explosion_rate", timeit(lambda: st.explosion_rate(1e10), n)))
    import os

    from pycalima.gsd.collisions import default_threads, performance_cores
    print(f"performance cores {performance_cores()}, load {os.getloadavg()[0]:.1f}, "
          f"pair-sum threads now {default_threads()}")
    print(f"{'workload':62s} {'wall':>10s} {'CPU':>10s}")
    for lab, (ms, cpu) in rows:
        print(f"{lab:62s} {ms:8.2f} ms {cpu:8.2f} ms")


def _exp_bins(lo, hi, m0):
    from pycalima.gsd.grid import MassGrid
    g = MassGrid.from_radii(lo, hi, 10, 1.0)
    me = g.m_edges
    N = np.exp(-me[:-1] / m0) - np.exp(-me[1:] / m0)
    M = (me[:-1] + m0) * np.exp(-me[:-1] / m0) - (me[1:] + m0) * np.exp(-me[1:] / m0)
    return N, M


def _nodes(g, N, M):
    from pycalima.gsd.collisions import bin_nodes
    m, w = bin_nodes(g, N, M)
    mf = m.ravel()
    return g.radius(mf), mf


if __name__ == "__main__":
    main()
