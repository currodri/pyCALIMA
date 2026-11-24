"""Per-moment size-dependent sinks (``pycalima.gsd.sinks``)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from pycalima.gsd import closures as C
from pycalima.gsd.grid import MassGrid
from pycalima.gsd.sinks import sink_per_moment

pytestmark = pytest.mark.gsd_quick

RHO = 1.6
METHOD_NAMES = tuple(C.METHODS)


@pytest.fixture(autouse=True)
def _clean_env(isolated_env):
    yield


def grid_and_state(n=8, seed=3):
    a = np.geomspace(3.9e-8, 1e-4, n + 1)
    g = MassGrid(4.0 / 3.0 * np.pi * RHO * a ** 3, RHO)
    rng = np.random.default_rng(seed)
    N = rng.random(n) * (rng.random(n) > 0.2)
    s = rng.uniform(1e-3, 1.0 - 1e-3, n)
    return a, g, N, N * g.m_edges[:-1] * (1.0 + s * (g.etas - 1.0))


def exact_moments(g, N, M, method, f):
    """Per bin int f(a) dN and int m f(a) dN over the reconstruction, by fine quadrature of every piece."""
    P = C.pieces(g, N, M, method)
    outN, outM = np.zeros(g.n_bins), np.zeros(g.n_bins)
    for i in range(g.n_bins):
        for j in range(P.kp):
            nij = P.n[i, j]
            if nij == 0.0:
                continue
            if P.spike[i, j]:
                a = P.aL[i] * P.u1[i, j]
                outN[i] += nij * f(a)
                outM[i] += nij * g.mass(a) * f(a)
                continue
            u = np.geomspace(P.u1[i, j], P.u2[i, j], 4001)
            dn = P.A[i, j] * u ** (P.lam[i, j] + 1.0)              # dN / d ln u
            a = P.aL[i] * u
            lnu = np.log(u)
            outN[i] += np.trapezoid(dn * f(a), lnu)
            outM[i] += np.trapezoid(dn * g.mass(a) * f(a), lnu)
    return outN, outM


@pytest.mark.parametrize("name", METHOD_NAMES)
def test_a_constant_rate_decays_both_moments_exactly(name):
    _, g, N, M = grid_and_state()
    k, dt = 3e-12, 2e11
    N1, M1, dN, dM = sink_per_moment(g, N, M, lambda r: np.full(np.shape(r), k), dt, name)
    np.testing.assert_allclose(N1, N * math.exp(-k * dt), rtol=1e-12)
    np.testing.assert_allclose(M1, M * math.exp(-k * dt), rtol=1e-12)
    np.testing.assert_allclose(N1 + dN, N, rtol=1e-14)
    np.testing.assert_allclose(M1 + dM, M, rtol=1e-14)


@pytest.mark.parametrize("name", METHOD_NAMES)
@pytest.mark.parametrize("p", [2.0, -1.0])
def test_a_power_law_rate_matches_the_exact_in_bin_integral(name, p):
    """k ~ a^p: the surviving moments are the in-bin integrals of exp(-k dt). The error, measured against what the
    bin held (the share of it removed wrongly), stays below 3e-4 with the default nodes where k dt runs from ~0.5 to
    ~4 across a bin (second order in the node spacing; zero for constant k)."""
    _, g, N, M = grid_and_state()
    k = lambda r: 1e-12 * (np.asarray(r) / 1e-6) ** p         # noqa: E731
    dt = 1e12
    N1, M1, _, _ = sink_per_moment(g, N, M, k, dt, name)
    refN, refM = exact_moments(g, N, M, name, lambda r: np.exp(-k(r) * dt))
    filled = N > 0
    assert np.all(np.abs(N1 - refN)[filled] <= 3e-4 * N[filled])
    assert np.all(np.abs(M1 - refM)[filled] <= 3e-4 * M[filled])


@pytest.mark.parametrize("name", METHOD_NAMES)
def test_a_steep_sink_keeps_the_bins_realisable_and_removes_the_large_grains_first(name):
    """An explosion-like rate rising steeply with a, over a step far beyond its timescale: what survives keeps a mean
    mass inside its bin, lower than before (the in-bin state moves down)."""
    _, g, N, M = grid_and_state()
    k = lambda r: 1e-10 * (np.asarray(r) / 1e-6) ** 6          # noqa: E731
    N1, M1, dN, dM = sink_per_moment(g, N, M, k, 1e12, name)
    ok = N1 > 1e-300 * N.max()
    mbar1 = M1[ok] / N1[ok]
    assert np.all(mbar1 >= g.m_edges[:-1][ok] * (1 - 1e-12))
    assert np.all(mbar1 <= g.m_edges[1:][ok] * (1 + 1e-12))
    assert np.all(mbar1 <= (M / np.where(N > 0, N, 1.0))[ok] * (1 + 1e-12))
    assert np.all(dN >= 0) and np.all(dM >= 0)


def test_a_short_step_removes_the_number_and_mass_weighted_rates():
    """For k dt << 1 the removal rates are int k dN and int k m dN (the per-moment rule)."""
    _, g, N, M = grid_and_state()
    k = lambda r: 1e-12 * (np.asarray(r) / 1e-6) ** 2          # noqa: E731
    dt = 1e3
    _, _, dN, dM = sink_per_moment(g, N, M, k, dt)
    kN, kM = exact_moments(g, N, M, None, k)
    filled = N > 0
    np.testing.assert_allclose(dN[filled] / dt, kN[filled], rtol=1e-3)
    np.testing.assert_allclose(dM[filled] / dt, kM[filled], rtol=1e-3)


def test_empty_bins_stay_empty():
    _, g, N, M = grid_and_state()
    N[[1, 4]] = 0.0
    M[[1, 4]] = 0.0
    N1, M1, _, dM = sink_per_moment(g, N, M, lambda r: 1e-12 + 0 * np.asarray(r), 1e12)
    assert np.all(N1[[1, 4]] == 0.0) and np.all(M1[[1, 4]] == 0.0) and np.all(dM[[1, 4]] == 0.0)
