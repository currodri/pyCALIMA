"""The discretised grain-size-distribution solver (pycalima.gsd).

Verification follows docs/development/gsd_solver_review.md s. 7.6: exact
transport tests first (advection, sublimation, size-dependent growth,
nucleation top-hat), then gas coupling, coagulation and shattering.
"""

from __future__ import annotations

import numpy as np
import pytest

from pycalima.gsd.grid import MassGrid
from pycalima.gsd.pl2m import BinProfile, int_pow, mean_mass_ratio, solve_beta
from pycalima.gsd.remap import flow_map, inject, remap, trajectory

RHO = 2.24


def _grid(bpd=10):
    return MassGrid.from_radii(3e-8, 1e-3, bpd, RHO)


def _lognormal(grid, a0=2e-6, sig=0.4, n_tot=1.0):
    """(N, M) per bin of a log-normal in radius, integrated per bin."""
    N = np.zeros(grid.n_bins)
    M = np.zeros(grid.n_bins)
    for i in range(grid.n_bins):
        a = np.geomspace(grid.a_edges[i], grid.a_edges[i + 1], 401)
        f = n_tot / (np.sqrt(2 * np.pi) * sig * a) * np.exp(-np.log(a / a0) ** 2 / (2 * sig ** 2))
        N[i] = np.trapezoid(f, a)
        M[i] = np.trapezoid(f * grid.mass(a), a)
    keep = N > 1e-30 * n_tot
    return np.where(keep, N, 0.0), np.where(keep, M, 0.0)


def _mapped_third_moment(grid, N, M, image, n=2001):
    """Sum over grains of X(a)^3 for the PL2M profile, by fine quadrature."""
    prof = BinProfile(grid, N, M)
    tot = 0.0
    for i in np.nonzero(N > 0)[0]:
        u = np.geomspace(1.0, prof.uR, n)
        dn = prof.A[i] * u ** prof.lam[i]
        X = image(prof.aL[i] * u)
        tot += np.trapezoid(dn * np.where(X > grid.a_edges[0], X ** 3, 0.0), u)
    return tot * 4.0 / 3.0 * np.pi * grid.rho


# ---------------------------------------------------------------------------
# two-moment power law
# ---------------------------------------------------------------------------

def test_pl2m_exponent_round_trips_and_spans_the_bin():
    eta = 10 ** 0.1
    beta = np.array([-120.0, -10.0, -2.0, -1.0, 0.0, 3.0, 50.0, 120.0])
    np.testing.assert_allclose(solve_beta(mean_mass_ratio(beta, eta), eta), beta, atol=1e-3)
    r = mean_mass_ratio(np.array([-149.0, 149.0]), eta)
    assert 1.0 < r[0] < 1.01 and eta / 1.01 < r[1] < eta     # beyond: spikes
    assert np.all(np.diff(mean_mass_ratio(np.linspace(-50, 50, 101), eta)) > 0)


def test_pl2m_reproduces_the_bin_moments():
    g = _grid()
    N, M = _lognormal(g)
    prof = BinProfile(g, N, M)
    filled = N > 0
    n = np.array([prof.number(i, 1.0, prof.uR) for i in range(g.n_bins)])
    m = np.array([prof.moment(i, 3, 1.0, prof.uR) * g.mass(prof.aL[i])
                  for i in range(g.n_bins)])
    np.testing.assert_allclose(n[filled], N[filled], rtol=1e-12)
    np.testing.assert_allclose(m[filled], M[filled], rtol=1e-6)


# ---------------------------------------------------------------------------
# transport along characteristics
# ---------------------------------------------------------------------------

def test_constant_growth_moves_number_and_mass_exactly():
    g = _grid()
    N, M = _lognormal(g)
    shift = 3e-6
    image = lambda a: a + shift
    N1, M1, lost, oN, oM = remap(g, N, M, image)
    assert lost == 0.0 and oN == 0.0
    assert N1.sum() == pytest.approx(N.sum(), rel=1e-13)
    assert M1.sum() == pytest.approx(_mapped_third_moment(g, N, M, image), rel=1e-6)
    # every bin stays realisable: mean mass inside the bin
    f = N1 > 0
    mbar = M1[f] / N1[f]
    assert np.all(mbar >= g.m_edges[:-1][f] * (1 - 1e-12))
    assert np.all(mbar <= g.m_edges[1:][f] * (1 + 1e-12))


def test_a_growing_burst_stays_narrow():
    """Monodisperse grains grown 11-fold in radius in 200 small steps keep
    their number exactly and stay within ~4 bins (the one-moment upwind
    scheme spreads them over the whole range). Two moments cannot encode a
    spike inside a bin, so it is reconstructed as a power law of the same N
    and M; that spread costs <1% in mass over the run (<(a+da)^3> differs
    from (a+da)^3), the price of the scheme."""
    g = _grid()
    N = np.zeros(g.n_bins)
    M = np.zeros(g.n_bins)
    a0 = 1e-7
    i0 = g.bin_of_radius(a0)
    N[i0], M[i0] = 1.0, g.mass(a0)
    step = 5e-8
    for _ in range(200):
        N, M, *_ = remap(g, N, M, lambda a: a + step)
    a_end = a0 + 200 * step
    assert N.sum() == pytest.approx(1.0, rel=1e-12)
    assert (M.sum() / g.mass(a_end)) == pytest.approx(1.0, rel=1e-2)
    order = np.argsort(M)[::-1]
    assert np.cumsum(M[order])[3] > 0.99 * M.sum()        # 99% of the mass in 4 bins


def test_constant_sublimation_removes_the_smallest_grains_exactly():
    g = _grid()
    N, M = _lognormal(g)
    loss = 1.5e-6
    image = lambda a: np.maximum(a - loss, 0.0)
    N1, M1, lost, oN, oM = remap(g, N, M, image)
    prof = BinProfile(g, N, M)
    # grains below a_min + loss vanish (their mapped radius is below the grid)
    cut = g.a_edges[0] + loss
    expect_lost = 0.0
    for i in np.nonzero(N > 0)[0]:
        hi = min(cut / prof.aL[i], prof.uR)
        if hi > 1.0:
            expect_lost += prof.number(i, 1.0, hi)
    assert lost == pytest.approx(expect_lost, rel=1e-10)
    assert N1.sum() + lost == pytest.approx(N.sum(), rel=1e-13)
    assert M1.sum() == pytest.approx(_mapped_third_moment(g, N, M, image), rel=1e-6)


def test_size_dependent_growth_follows_the_exact_characteristic():
    """adot = G/a: a^2 = a0^2 + 2 G t, both for the flow map and the remap."""
    g = _grid()
    N, M = _lognormal(g)
    G, dt = 1e-12, 2e3                                   # 2 G dt = (0.63 um)^2
    exact = lambda a: np.sqrt(a ** 2 + 2 * G * dt)
    a = np.geomspace(1e-7, 1e-4, 50)
    np.testing.assert_allclose(flow_map(lambda x: G / np.maximum(x, 1e-30), a, dt),
                               exact(a), rtol=1e-6)
    N1, M1, *_ = remap(g, N, M, exact)
    assert N1.sum() == pytest.approx(N.sum(), rel=1e-12)
    assert M1.sum() == pytest.approx(_mapped_third_moment(g, N, M, exact), rel=1e-4)


def test_nucleation_at_constant_rate_and_growth_gives_the_top_hat():
    """Constant J and adot: n(a) = J / adot on [a*, a* + adot dt]."""
    g = _grid()
    J, adot, dt, a_star = 2.0, 1e-9, 3e3, 4e-8
    s, y = trajectory(lambda a: np.full_like(a, adot), a_star, dt, n_nodes=65)
    dN, dM = inject(g, J * dt, y[::-1])
    assert dN.sum() == pytest.approx(J * dt, rel=1e-13)
    a_hi = a_star + adot * dt
    for j in np.nonzero(dN)[0]:
        x1, x2 = max(a_star, g.a_edges[j]), min(a_hi, g.a_edges[j + 1])
        assert dN[j] == pytest.approx(J / adot * (x2 - x1), rel=1e-9)
    mtot = J / adot * 4 / 3 * np.pi * RHO * (a_hi ** 4 - a_star ** 4) / 4
    assert dM.sum() == pytest.approx(mtot, rel=1e-9)
