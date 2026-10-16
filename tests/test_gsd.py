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


# ---------------------------------------------------------------------------
# gas-grain exchange
# ---------------------------------------------------------------------------

def _carbon_box(T_d, nH=1e10, dust_frac=0.99, nucleation=False, a0=1e-5, bpd=8):
    from pycalima.gsd.exchange import Population
    from pycalima.models.dust_radiation.sublimation_kinetics import get_sublimation_model

    model = get_sublimation_model("graphite", "GD89")
    pop = Population.empty(model, 3e-8, 1e-3, bpd, T_d=T_d, nucleation=nucleation)
    n_C = 10 ** -3.44 * nH
    if dust_frac > 0:
        a = pop.grid.a_centres
        w = np.exp(-np.log(a / a0) ** 2 / (2 * 0.15 ** 2))
        pop.M = w / w.sum() * dust_frac * n_C * 12.011 * 1.66053906660e-24
        pop.N = pop.M / pop.grid.mass(a)
    return pop, {"C": (1 - dust_frac) * n_C}, n_C


def test_closed_box_relaxes_to_the_saturated_density_conserving_carbon():
    from pycalima.gsd.exchange import exchange_step
    from pycalima.models.dust_radiation import dust_sublimation as ds

    pop, gas, n_C = _carbon_box(1700.0)
    total = gas["C"] + pop.key_atoms()
    for _ in range(25):
        exchange_step([pop], gas, 1e4, 1e7)
        assert gas["C"] + pop.key_atoms() == pytest.approx(total, rel=1e-13)
    n_sat = float(ds.saturated_key_density(pop.model, 1e-5, 1700.0, 1e4))
    assert gas["C"] == pytest.approx(n_sat, rel=0.01)


def test_above_saturation_all_dust_evaporates():
    from pycalima.gsd.exchange import exchange_step

    pop, gas, n_C = _carbon_box(2100.0)
    for _ in range(5):
        exchange_step([pop], gas, 1e4, 1e8)
    assert pop.M.sum() < 1e-10 * n_C * 12.011 * 1.66053906660e-24
    assert gas["C"] == pytest.approx(n_C, rel=1e-10)


def test_one_huge_step_is_stable_and_positive():
    """The gas is implicit: a step 1e4 times the relaxation time neither goes
    negative nor overshoots saturation."""
    from pycalima.gsd.exchange import exchange_step
    from pycalima.models.dust_radiation import dust_sublimation as ds

    pop, gas, n_C = _carbon_box(1700.0, dust_frac=0.5)
    exchange_step([pop], gas, 1e4, 1e12)
    n_sat = float(ds.saturated_key_density(pop.model, 1e-5, 1700.0, 1e4))
    assert 0.0 < gas["C"] < 2 * n_sat
    assert np.all(pop.N >= 0) and np.all(pop.M >= 0)


def test_dust_nucleates_from_pure_gas_and_grows():
    """No initial dust: at n_H = 1e10 and 1400 K clusters nucleate, grow and
    lock up the carbon down to the saturated density."""
    from pycalima.gsd.exchange import exchange_step

    pop, gas, n_C = _carbon_box(1400.0, dust_frac=0.0, nucleation=True)
    total = gas["C"]
    for _ in range(30):
        exchange_step([pop], gas, 1e4, 3e5)
    assert gas["C"] + pop.key_atoms() == pytest.approx(total, rel=1e-12)
    assert pop.key_atoms() > 0.9 * total
    assert pop.N.sum() > 0


def test_ostwald_ripening_follows_wagner():
    """Interface-limited (Wagner 1961) ripening: with the Kelvin term,
    da/dt = K (1/a* - 1/a) for a >> c, K = V J_bulk c, c = theta (3V/4pi)^1/3 / T.
    Started on the self-similar distribution P(rho) ~ rho (2-rho)^-5
    exp(-3 rho/(2-rho)), rho = a/a*, it must stay on it: a*^2 = K t / 2 and
    abar = (8/9) a*. a* is read from the gas, a* = c / ln(n / n_inf)."""
    from pycalima.gsd.exchange import Population, exchange_step
    from pycalima.models.dust_radiation import dust_sublimation as ds
    from pycalima.models.dust_radiation.sublimation_kinetics import get_sublimation_model

    T = 1800.0
    m = get_sublimation_model("graphite", "GD89")
    V = m.unit_volume_cm3
    c = m.theta * (3 * V / (4 * np.pi)) ** (1 / 3) / T
    K = V * float(m.flux(T)) * c
    pop = Population.empty(m, 3e-8, 3e-2, 20, T_d=T, nucleation=False)
    g, s0 = pop.grid, 50 * c
    N, M = np.zeros(g.n_bins), np.zeros(g.n_bins)
    for i in range(g.n_bins):
        a = np.linspace(g.a_edges[i], g.a_edges[i + 1], 801)
        r = np.clip(a / s0, 0, 2 - 1e-12)
        f = np.where(a / s0 < 2, r * (2 - r) ** -5.0 * np.exp(-3 * r / (2 - r)), 0.0)
        N[i], M[i] = np.trapezoid(f, a), np.trapezoid(f * g.mass(a), a)
    keep = N > 1e-14 * N.sum()
    scale = 0.95 * 10 ** -3.44 * 1e10 * 12.011 * 1.66053906660e-24 / M.sum()
    pop.N, pop.M = np.where(keep, N, 0.0) * scale, np.where(keep, M, 0.0) * scale
    n_inf = float(ds.saturated_key_density(m, 1.0, T, T))
    gas = {"C": n_inf * np.exp(c / s0)}

    t_r, t, dt = s0 ** 2 / K, 0.0, 1e-3 * s0 ** 2 / K
    while t < 10 * t_r:                                  # a* grows sqrt(6)-fold
        exchange_step([pop], gas, T, dt)
        t += dt
        dt = max(0.02 * t, 1e-3 * t_r)
    a_star = c / np.log(gas["C"] / n_inf)
    assert a_star == pytest.approx(np.sqrt(K * (t + 2 * t_r) / 2), rel=0.01)
    prof = BinProfile(g, pop.N, pop.M)
    a1 = sum(prof.moment(i, 1, 1.0, prof.uR) * prof.aL[i] for i in np.nonzero(pop.N)[0])
    assert a1 / pop.N.sum() / a_star == pytest.approx(8 / 9, rel=0.01)


def _nk13_carbon():
    """Nozawa et al. (2003) carbon as used by Nozawa & Kozasa (2013): the N03
    dataset with sticking s = 1 and a0 = 1.281 A per atom."""
    from dataclasses import replace

    from pycalima.models.dust_radiation.sublimation_kinetics import get_sublimation_model

    m = get_sublimation_model("graphite", "N03")
    rho = 12.011 * 1.66053906660e-24 / (4 * np.pi / 3 * 1.281e-8 ** 3)
    return replace(m, channels=(replace(m.channels[0], alpha0=1.0),), density_gcm3=rho)


@pytest.mark.parametrize("kelvin", [True, False])
def test_nucleation_in_expanding_gas_reproduces_nozawa_kozasa_2013(kelvin):
    """NK13 (ApJ 776, 24) s. 4: gas expanding homologously, c ~ t^-3 and
    T = T0 (t/t0)^-3(gamma-1), t0 = 300 d at ln S = 0. For c10 = 1e7 cm^-3,
    gamma = 1.25 (Lambda_on = 609) their eq. 64 gives a_ave = 0.00936 um, their
    steady model 0.00997 um and their non-steady cluster model f_con = 1.
    Through the solver (exchange steps + dilution), with and without the
    Kelvin term in growth, a_ave must land within 5% of the steady model."""
    from scipy.optimize import brentq

    from pycalima.gsd.exchange import Population, exchange_step

    model = _nk13_carbon()
    c10, gamma, t0 = 1e7, 1.25, 300 * 86400.0
    lnS = lambda c, T: 8.64726e4 / T - 19.0422 + np.log(c * 1.380649e-16 * T / 1e6)
    T0 = brentq(lambda T: lnS(c10, T), 300.0, 1e4)
    pop = Population.empty(model, 0.9e-8, 1e-3, 10, T_d=T0, kelvin=kelvin)
    gas, t = {"C": c10}, t0
    while t < 20 * t0:
        dt = 4e-3 * t
        d = (t / (t + dt)) ** 3
        t += dt
        pop.N, pop.M, gas["C"] = pop.N * d, pop.M * d, gas["C"] * d
        pop.overflow_N *= d
        pop.overflow_M *= d
        T = T0 * (t / t0) ** (-3 * (gamma - 1))
        pop.T_d = np.full(pop.grid.n_bins, T)
        exchange_step([pop], gas, T, dt)
    f_con = pop.key_atoms() / (c10 * (t / t0) ** -3)
    a_ave = (pop.M.sum() / pop.N.sum() / (4 / 3 * np.pi * model.density_gcm3)) ** (1 / 3)
    assert f_con == pytest.approx(1.0, abs=1e-3)
    assert a_ave * 1e4 == pytest.approx(0.00997, rel=0.05)


# ---------------------------------------------------------------------------
# coagulation (Smoluchowski analytic solutions)
# ---------------------------------------------------------------------------

def _exponential_bins(grid, N0, m0):
    """Exact bin integrals of n(m) = N0/m0 exp(-m/m0)."""
    me = grid.m_edges
    N = N0 * (np.exp(-me[:-1] / m0) - np.exp(-me[1:] / m0))
    M = N0 * ((me[:-1] + m0) * np.exp(-me[:-1] / m0) - (me[1:] + m0) * np.exp(-me[1:] / m0))
    return N, M


def test_constant_kernel_coagulation_matches_the_exact_solution():
    """K = K0, exponential start: n stays exponential, N = N0/(1+tau) and
    mean mass m0 (1+tau), tau = K0 N0 t / 2. Mass is conserved to round-off."""
    from pycalima.gsd.coagulation import coagulate

    g = MassGrid.from_radii(1e-7, 1e-3, 10, 1.0)
    m0, K0 = g.mass(3e-6), 1e-10
    N, M = _exponential_bins(g, 1.0, m0)
    tau = 10.0
    N1, M1, oN, oM, _ = coagulate(g, N, M, lambda a, b: np.full(np.broadcast(a, b).shape, K0),
                                  2 * tau / K0)
    assert M1.sum() + oM == pytest.approx(M.sum(), rel=1e-13)
    assert N1.sum() == pytest.approx(1.0 / (1 + tau), rel=1e-4)
    Ne, _ = _exponential_bins(g, 1.0 / (1 + tau), m0 * (1 + tau))
    assert np.abs(N1 - Ne).sum() / Ne.sum() < 0.01


def test_additive_and_multiplicative_kernels_follow_their_number_laws():
    """K = B (m + m'): N = N0 exp(-B M t). K = B m m' before gelation:
    N = N0 - B M^2 t / 2 (gelation at B M2 t = 1, M2 = 2 m0 M here)."""
    from pycalima.gsd.coagulation import coagulate

    g = MassGrid.from_radii(1e-7, 1e-2, 10, 1.0)
    m0 = g.mass(3e-6)
    N, M = _exponential_bins(g, 1.0, m0)
    Mt = M.sum()
    B = 1e-10 / m0
    N1, M1, *_ = coagulate(g, N, M, lambda a, b: B * (a + b), 2.0 / (B * Mt))
    assert M1.sum() == pytest.approx(Mt, rel=1e-13)
    assert N1.sum() == pytest.approx(N.sum() * np.exp(-2.0), rel=1e-4)
    B = 1e-10 / m0 ** 2
    t = 0.8 / (2 * B * m0 * Mt)
    N1, M1, *_ = coagulate(g, N, M, lambda a, b: B * a * b, t)
    assert M1.sum() == pytest.approx(Mt, rel=1e-13)
    assert N1.sum() == pytest.approx(N.sum() - 0.5 * B * Mt ** 2 * t, rel=1e-4)
