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


class _FullShatter:
    """Every collision shatters both grains into a^-alpha fragments on
    [a_hi/30, a_hi], with a_hi fixed."""

    def __init__(self, K, a_hi, alpha=3.3, collector="smallest"):
        self.K, self.a_hi, self.alpha_f, self.collector = K, a_hi, alpha, collector

    def channels(self, a, m):
        from pycalima.gsd.collisions import Channel

        P = m.size
        K = np.full((P, P), self.K)
        hi = np.full((P, P), self.a_hi)
        return [Channel(K, frag1=m[:, None] + 0 * m[None, :], a_hi1=hi, a_lo1=hi / 30,
                        frag2=m[None, :] + 0 * m[:, None], a_hi2=hi, a_lo2=hi / 30)]


@pytest.mark.parametrize("collector", ["smallest", "remove"])
def test_fragments_are_deposited_with_the_exact_power_law(collector):
    """One bin of grains shattering into a^-3.3 fragments spanning ~15 bins
    and the bottom edge: per-bin fragment number and mass equal the
    power-law integrals; mass below the grid goes to the collector."""
    from pycalima.gsd.collisions import _terms

    g = MassGrid.from_radii(1e-7, 1e-4, 10, RHO)
    N = np.zeros(g.n_bins)
    M = np.zeros(g.n_bins)
    k = g.bin_of_radius(5e-5)
    N[k] = 2.0
    M[k] = 2.0 * g.mass(5e-5)
    a_hi, alpha = 2e-6, 3.3
    model = _FullShatter(1e-9, a_hi, alpha, collector)
    F, L, H = _terms(g, N, M, model)
    rate = 0.5 * 1e-9 * N[k] ** 2                      # collisions per cm^3 s
    mfrag = rate * 2 * g.mass(5e-5)                    # nodes sit at one mass
    lo = a_hi / 30
    ipow = lambda e, x1, x2: (x2 ** (e + 1) - x1 ** (e + 1)) / (e + 1)
    norm = ipow(3 - alpha, lo, a_hi)
    a_e = g.a_edges
    for j in range(g.n_bins):
        x1, x2 = max(lo, a_e[j]), min(a_hi, a_e[j + 1])
        if j == k or x2 <= x1:
            continue
        dm = mfrag * ipow(3 - alpha, x1, x2) / norm
        dn = mfrag * ipow(-alpha, x1, x2) / norm / (4 / 3 * np.pi * RHO)
        if j == 0 and collector == "smallest":
            below = mfrag * ipow(3 - alpha, lo, a_e[0]) / norm
            dm += below
            dn += below / g.m_edges[0]
        assert F[j, k] == pytest.approx(dm, rel=1e-10)
        assert H[j, k] == pytest.approx(dn, rel=1e-10)
    below = mfrag * ipow(3 - alpha, lo, a_e[0]) / norm
    assert F[g.n_bins + 1, k] == pytest.approx(below if collector == "remove" else 0.0, abs=1e-30)
    assert F[:, k].sum() == pytest.approx(mfrag, rel=1e-12)
    assert L[k] == pytest.approx(2 * rate, rel=1e-12)


def test_shattering_conserves_mass_into_the_collector_and_sink():
    """Integrated over many collision times: grid + sink mass is constant
    to round-off; with collector 'smallest' nothing leaves the grid."""
    from pycalima.gsd.collisions import collide

    g = MassGrid.from_radii(1e-7, 1e-4, 10, RHO)
    N, M = _lognormal(g, a0=3e-5, sig=0.2)
    for collector in ("smallest", "remove"):
        r = collide(g, N, M, _FullShatter(1.0 / N.sum(), 2e-6, 3.3, collector), 2.0)
        assert r["M"].sum() + r["sink_M"] == pytest.approx(M.sum(), rel=1e-12)
        assert np.all(r["N"] >= 0) and np.all(r["M"] >= 0)
        if collector == "smallest":
            assert r["sink_M"] == 0.0
        else:
            assert r["sink_M"] > 0.0


# ---------------------------------------------------------------------------
# shattering
# ---------------------------------------------------------------------------

class _SelfSimilarCascade:
    """Constant v and Q*_D, KT10 ejecta from the larger grain (target),
    projectile shattered, fragments a^-alpha up to (f/2)^(1/3) a_t: a
    collision outcome that depends only on the mass ratio."""

    collector = "remove"

    def __init__(self, alpha, v=1e5, Q=1e9):
        self.alpha_f, self.v, self.Q = alpha, v, Q

    def channels(self, a, m):
        from pycalima.gsd.collisions import Channel

        A1, A2 = np.meshgrid(a, a, indexing="ij")
        M1, M2 = np.meshgrid(m, m, indexing="ij")
        big = M1 >= M2
        mt, mp, at = np.where(big, M1, M2), np.where(big, M2, M1), np.where(big, A1, A2)
        K = np.pi * (A1 + A2) ** 2 * self.v
        phi = 0.5 * mt * mp / (mt + mp) * self.v ** 2 / (mt * self.Q)
        mej = phi / (1 + phi) * mt
        ahi = (0.5 * np.minimum((mej + mp) / mt, 1.0)) ** (1 / 3) * at
        # target remnant and all fragments as one product set (attribution
        # to the two donors is by mass in the operator anyway)
        return [Channel(K, rem1=mt - mej, frag1=mej + mp, a_hi1=ahi, a_lo1=ahi / 100)]


@pytest.mark.parametrize("alpha_f", [2.5, 3.5])
def test_collisional_cascade_reaches_the_dohnanyi_slope(alpha_f):
    """Mass fed at the top of a self-similar cascade (constant v, Q*_D) and
    removed below the grid: the steady state is dN/da ~ a^-3.5 whatever the
    fragment law (Dohnanyi 1969; Tanaka, Inaba & Nakazawa 1996)."""
    from pycalima.gsd.collisions import collide

    g = MassGrid.from_radii(1e-6, 1e-2, 6, 3.0)
    top = g.n_bins - 2
    sN, sM = np.zeros(g.n_bins), np.zeros(g.n_bins)
    sM[top] = 1e-25
    sN[top] = sM[top] / g.m_centres[top]
    N, M = sN * 1e5, sM * 1e5
    model = _SelfSimilarCascade(alpha_f)
    t, dt = 0.0, 1e6
    while t < 3e11:
        r = collide(g, N, M, model, dt, errmax=1e-2, source=(sN, sM))
        N, M = r["N"], r["M"]
        t += dt
        dt *= 1.1
    a = g.a_centres
    sel = (a > 1e-5) & (a < 1e-3)
    slope = np.polyfit(np.log(a[sel]), np.log(N[sel] / np.diff(g.a_edges)[sel]), 1)[0]
    assert slope == pytest.approx(-3.5, abs=0.03)


def test_kt10_ejecta_reproduce_the_hk13_worked_example():
    """HK13 s. 3: carbon, a1 = 0.3 um hit by a2 = 0.03 um at 20 km/s:
    phi = 0.22 and M_ej / m1 = 0.18."""
    from pycalima.gsd.shattering import GRAPHITE, ejected_mass

    m = lambda a: 4 / 3 * np.pi * a ** 3 * GRAPHITE.rho
    assert ejected_mass(GRAPHITE, m(3e-5), m(3e-6), 2e6) / m(3e-5) == pytest.approx(0.18, abs=0.005)


def test_small_grain_production_scales_with_P1_t():
    """Cratering mass ~ 1/P1, so runs with equal P1/t give the same small
    grains (HK13 Fig. 7). MRN silicate in the WIM, v = 20 km/s above
    0.1 um (HK13 Model H), KT10 ejecta; mass is conserved with the sink.
    HK13 find n(a) within a factor 3; away from the removal edge it is 25%."""
    from dataclasses import replace

    from pycalima.gsd.collisions import collide
    from pycalima.gsd.shattering import SILICATE, GrainCollisions, hy09_directions

    g = MassGrid.from_radii(3e-8, 3e-4, 5, SILICATE.rho)
    a_e = g.a_edges
    N, M = np.zeros(g.n_bins), np.zeros(g.n_bins)
    for i in range(g.n_bins):
        x1, x2 = max(5e-7, a_e[i]), min(2.5e-5, a_e[i + 1])
        if x2 > x1:
            N[i] = (x1 ** -2.5 - x2 ** -2.5) / 2.5
            M[i] = 4 / 3 * np.pi * SILICATE.rho * (x2 ** 0.5 - x1 ** 0.5) / 0.5
    scale = 4e-3 * 1.4 * 1.6735e-24 / M.sum()
    N, M = N * scale, M * scale
    # HK13's step, smoothed over ~10% in a: a node sitting on a discontinuous
    # velocity law makes the rates jump and the step control chatter
    vel = hy09_directions(lambda a: 1e6 * (1 + np.tanh(np.log(np.asarray(a) / 1e-5) / 0.1)))
    myr = 3.156e13
    out = []
    for P1, t in ((3e11, 10.0), (1e11, 10.0 / 3)):
        model = GrainCollisions(replace(SILICATE, P1=P1), vel, law="KT10")
        r = collide(g, N, M, model, t * myr, errmax=1e-2)
        assert r["M"].sum() + r["sink_M"] == pytest.approx(M.sum(), rel=1e-12)
        out.append(r["N"])
    small = (g.a_centres < 3e-6) & (out[0] > 0)
    ratio = out[1][small] / out[0][small]
    assert np.all((ratio > 1 / 3) & (ratio < 3.0))       # HK13: within a factor 3
    mid = (g.a_centres > 3e-7) & (g.a_centres < 3e-6)    # away from the removal edge
    assert np.all(np.abs(out[1][mid] / out[0][mid] - 1) < 0.25)
    gained = (out[0][g.a_centres < 3e-7] * g.m_centres[g.a_centres < 3e-7]).sum()
    assert gained > 0.03 * M.sum()                      # small grains are produced
