"""Time stepping of ``gsd.collisions.collide``: order, tolerance, failure modes.

The reference is the same ODE (collide's own rates, ``_terms``) integrated
with classical RK4, which shares none of collide's Patankar weighting or
number update.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from pycalima.gsd.collisions import CoagulationModel, _terms, collide
from pycalima.gsd.grid import MassGrid

RHO = 3.0


def _sweep_up_case():
    """MRN grains, 5-250 nm, under a geometric kernel with Delta v growing as
    the larger grain's a^(1/2) (intermediate turbulence): big grains sweep up
    small ones and mostly stay in their bin, as in Marchand et al. (2021)."""
    a = np.geomspace(2e-7, 2e-4, 31)
    g = MassGrid(4.0 / 3.0 * np.pi * RHO * a ** 3, RHO)
    lo, hi = np.clip(a[:-1], 5e-7, 2.5e-5), np.clip(a[1:], 5e-7, 2.5e-5)
    N = (lo ** -2.5 - hi ** -2.5) / 2.5
    M = 4.0 / 3.0 * np.pi * RHO * (hi ** 0.5 - lo ** 0.5) / 0.5
    N, M = N / M.sum() * 1e-20, M / M.sum() * 1e-20

    def kernel(m1, m2):
        a1, a2 = (3.0 * m1 / (4.0 * np.pi * RHO)) ** (1 / 3), (3.0 * m2 / (4.0 * np.pi * RHO)) ** (1 / 3)
        return np.pi * (a1 + a2) ** 2 * 1e2 * np.sqrt(np.maximum(a1, a2) / 1e-6)

    return g, N, M, CoagulationModel(kernel), 3e13


def _rhs(g, model, nb):
    def f(y):
        N, M = np.maximum(y[:nb], 0.0), np.maximum(y[nb:], 0.0)
        F, L, H = _terms(g, N, M, model)
        return np.concatenate([(H @ np.ones(nb))[:nb] - L, (F.sum(axis=1) - F.sum(axis=0))[:nb]])
    return f


def _rk4(g, N, M, model, times):
    f, y = _rhs(g, model, g.n_bins), np.concatenate([N, M])
    for t0, t1 in zip(times[:-1], times[1:]):
        h = t1 - t0
        k1 = f(y)
        k2 = f(y + 0.5 * h * k1)
        k3 = f(y + 0.5 * h * k2)
        k4 = f(y + h * k3)
        y = y + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return np.maximum(y[:g.n_bins], 0.0), np.maximum(y[g.n_bins:], 0.0)


def _steps(t_end, n):
    return np.concatenate([[0.0], np.geomspace(1e-6 * t_end, t_end, n)])


def _mean_log_a(g, N, M):
    f = N > 0
    return float(np.sum(M[f] * np.log10(g.radius(M[f] / N[f]))) / M[f].sum())


@pytest.fixture(scope="module")
def sweep_up():
    g, N, M, model, t_end = _sweep_up_case()
    ref = _rk4(g, N, M, model, _steps(t_end, 1600))
    return g, N, M, model, t_end, ref


def test_the_reference_has_grown_the_grains(sweep_up):
    g, N, M, model, t_end, (Nr, Mr) = sweep_up
    assert _mean_log_a(g, Nr, Mr) - _mean_log_a(g, N, M) > 0.5


def test_default_tolerance_matches_the_reference(sweep_up):
    """Before the in-bin number transfers were netted out (2026-11), errmax =
    1e-3 left <log a> 0.18 dex low on Marchand+21 Fig. 1."""
    g, N, M, model, t_end, (Nr, Mr) = sweep_up
    r = collide(g, N, M, model, t_end)
    assert abs(_mean_log_a(g, r["N"], r["M"]) - _mean_log_a(g, Nr, Mr)) < 1e-3
    assert np.abs(r["M"] - Mr).sum() < 1e-3 * M.sum()
    assert r["n_forced"] == 0


def test_fixed_steps_converge_at_second_order(sweep_up):
    """One collide call per step (errmax = inf): doubling the steps cuts the
    error about fourfold (it was about 1.1-fold, first order at best)."""
    g, N, M, model, t_end, (Nr, Mr) = sweep_up
    err = []
    for n in (100, 200):
        Nn, Mn = N, M
        times = _steps(t_end, n)
        for t0, t1 in zip(times[:-1], times[1:]):
            r = collide(g, Nn, Mn, model, t1 - t0, errmax=np.inf)
            Nn, Mn = r["N"], r["M"]
        err.append(np.abs(Mn - Mr).sum())
    assert err[0] / err[1] > 3.0


def test_drained_bins_are_emptied_not_left_without_mass():
    """A bin whose number underflowed to a denormal and its mass to zero is a
    node of zero mass (1/m = inf in a Brownian kernel). collide empties such
    bins, on input and after every step: N > 0 wherever M > 0 and vice versa."""
    a = np.geomspace(1e-7, 1e-4, 13)
    g = MassGrid(4.0 / 3.0 * np.pi * RHO * a ** 3, RHO)
    lo, hi = np.clip(a[:-1], 1e-7, 1e-5), np.clip(a[1:], 1e-7, 1e-5)
    N = (hi ** -2.5 - lo ** -2.5) / -2.5
    M = 4.0 / 3.0 * np.pi * RHO * (hi ** 0.5 - lo ** 0.5) / 0.5
    N[0], M[0] = 1e-310, 0.0
    brownian = CoagulationModel(lambda m1, m2: 1e-20 * np.sqrt(1.0 / m1 + 1.0 / m2))
    r = collide(g, N, M, brownian, 1.0)
    assert np.all(np.isfinite(r["N"])) and np.all(np.isfinite(r["M"]))
    assert np.array_equal(r["N"] > 0, r["M"] > 0)
    assert abs(r["M"].sum() + r["overflow_M"] - M.sum()) < 1e-12 * M.sum()


def test_running_out_of_steps_raises():
    g, N, M, model, t_end = _sweep_up_case()
    with pytest.raises(RuntimeError, match="max_steps"):
        collide(g, N, M, model, t_end, max_steps=5)


def test_a_step_far_longer_than_the_collision_time_warns_and_conserves_mass():
    """Everything coagulates past the grid in under a second, but dt = 1e12 s:
    the smallest step (1e-12 dt) cannot resolve it. Mass is still conserved
    and the bins stay realisable; a RuntimeWarning says the steps were forced."""
    a = np.geomspace(1e-7, 1e-4, 9)
    g = MassGrid(4.0 / 3.0 * np.pi * RHO * a ** 3, RHO)
    lo, hi = np.clip(a[:-1], 1e-7, 2e-5), np.clip(a[1:], 1e-7, 2e-5)
    N = 1e-6 * (hi ** -2.5 - lo ** -2.5) / -2.5
    M = 1e-6 * 4.0 / 3.0 * np.pi * RHO * (hi ** 0.5 - lo ** 0.5) / 0.5
    model = CoagulationModel(lambda m1, m2: 3e-10 * (m1 + m2) / g.m_edges[0])
    with pytest.warns(RuntimeWarning, match="smallest step"):
        r = collide(g, N, M, model, 1e12)
    assert r["n_forced"] > 0
    assert abs(r["M"].sum() + r["overflow_M"] - M.sum()) < 1e-12 * M.sum()
    filled = r["N"] > 0
    m_mean = r["M"][filled] / r["N"][filled]
    assert np.all(m_mean >= g.m_edges[:-1][filled] * (1 - 1e-12))
    assert np.all(m_mean <= g.m_edges[1:][filled] * (1 + 1e-12))


def test_a_resolved_run_does_not_warn(sweep_up):
    g, N, M, model, t_end, _ = sweep_up
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        collide(g, N, M, model, 0.01 * t_end)
