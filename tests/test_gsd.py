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


# ---------------------------------------------------------------------------
# continuous relative velocities (Ormel & Cuzzi 2007 + skewed Maxwellian)
# ---------------------------------------------------------------------------

def test_collision_flux_above_a_speed_matches_quadrature():
    """C(u; s) and the mean speed of the skewed Maxwellian (Guillet+20 App.
    C), including s -> 0 and the deep tail."""
    from scipy.integrate import quad

    from pycalima.gsd.velocities import flux_above, log_flux_above, mean_speed_normalised

    def flux_density(v, s):
        if s < 1e-8:
            return 4 / np.sqrt(np.pi) * v ** 3 * np.exp(-v ** 2)
        return v ** 2 / (np.sqrt(np.pi) * s) * (np.exp(-(v - s) ** 2) - np.exp(-(v + s) ** 2))

    for s in (0.0, 1e-3, 0.7, 5.0, 50.0):
        lo, hi = max(0.0, s - 30), s + 30
        pts = [s] if s > lo else None
        tot = quad(flux_density, lo, hi, args=(s,), points=pts, limit=400, epsrel=1e-13)[0]
        assert float(mean_speed_normalised(s)) == pytest.approx(tot, rel=1e-10)
        for u in (lo, s + 0.5, s + 3.0):
            num = quad(flux_density, u, hi, args=(s,), points=[s] if u < s else None,
                       limit=400, epsabs=0, epsrel=1e-12)[0]
            assert float(flux_above(u, s)) == pytest.approx(num / tot, rel=1e-7)
    assert float(log_flux_above(20.0, 0.0)) == pytest.approx(np.log(401.0) - 400.0, rel=1e-12)


def test_flux_quantiles_invert_the_collision_flux():
    from pycalima.gsd.velocities import flux_quantile, log_flux_above

    rng = np.random.default_rng(1)
    s = np.concatenate([np.zeros(200), 10 ** rng.uniform(-4, 2.9, 2000)])
    y = np.exp(rng.uniform(np.log(1e-6), np.log(250), s.size))
    u = flux_quantile(s, y)
    lo, hi = np.zeros_like(u), s + 40.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        up = -log_flux_above(mid, s) < y
        lo, hi = np.where(up, mid, lo), np.where(up, hi, mid)
    ref = 0.5 * (lo + hi)
    assert np.max(np.abs(u - ref) / np.maximum(ref, 1e-3)) < 2e-3


def _wim_velocity(**kw):
    from pycalima.gsd.velocities import RelativeVelocity

    return RelativeVelocity(T=8000.0, n_H=1.0, mu=1.4, v_turb=1e6, L=100 * 3.086e18, **kw)


def test_relative_velocity_reproduces_the_collapse_coagulation_kernel():
    """Without thresholds the summed collision rate is sigma <v> with the rms
    speed of collapse.integrator.turbulent_kernel (Brownian + full OC07)."""
    from pycalima.collapse import velocities as cvel

    rv = _wim_velocity()
    a = np.geomspace(1e-7, 1e-4, 30)
    m = 4 / 3 * np.pi * a ** 3 * 3.3
    dV, mu = rv.pair_statistics(a, m, 3.3)
    t_s = cvel.epstein_stopping_time(a, 3.3, rv.T, rv.rho_gas, rv.mu)
    ref = np.sqrt(cvel.brownian_rms(rv.T, m) ** 2
                  + cvel.ormel_cuzzi_dv(t_s, rv.v_turb, rv.tau_L, rv.tau_eta) ** 2)
    np.testing.assert_allclose(dV, ref, rtol=1e-12)
    assert np.all(mu == 0)
    v_mean, c, v_q = rv.collision_speeds(dV, mu, 0.0, 4)
    np.testing.assert_allclose(v_mean, cvel.mean_collision_speed(dV, np.inf), rtol=1e-12)
    np.testing.assert_allclose(c, 1.0)


def test_shattering_rates_are_the_flux_above_threshold_and_continuous():
    """The shattering channels carry sigma <v> C(v_shat) in equal parts at
    speeds above v_shat, and the total rate varies smoothly with the gas
    state (no thresholds switching at the nodes)."""
    from pycalima.gsd.shattering import GrainCollisions, SILICATE

    a = np.geomspace(1e-7, 1e-4, 24)
    m = 4 / 3 * np.pi * a ** 3 * SILICATE.rho
    totals = []
    for T in np.linspace(7000.0, 9000.0, 41):
        rv = _wim_velocity()
        rv.T = T
        model = GrainCollisions("silicate", rv, n_speed=4, table_ppd=None)
        ch = model.channels(a, m)
        dV, mu = rv.pair_statistics(a, m, SILICATE.rho)
        v_mean, c, v_q = rv.collision_speeds(dV, mu, SILICATE.v_shat, 4)
        K = sum(c_.K for c_ in ch)
        sig = np.pi * (a[:, None] + a[None, :]) ** 2
        np.testing.assert_allclose(K, sig * v_mean * c, rtol=1e-12)
        assert np.all(v_q[:, c > 1e-12] > SILICATE.v_shat * (1 - 2e-3))
        totals.append(K.sum())
    d2 = np.diff(np.log(totals), 2)
    assert np.max(np.abs(d2)) < 1e-3          # smooth: no jumps in the second difference


def test_turbulent_shattering_and_sticking_run_smoothly_and_conserve_mass():
    """MRN silicate in the WIM with Ormel & Cuzzi velocities (v_turb = 10
    km/s, L = 100 pc): large grains are shattered into small ones in 10 Myr
    with few sub-steps, and grid + sink mass is conserved."""
    from pycalima.gsd.collisions import collide
    from pycalima.gsd.shattering import SILICATE, GrainCollisions

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
    model = GrainCollisions("silicate", _wim_velocity(), coagulation=True)
    r = collide(g, N, M, model, 10 * 3.156e13, errmax=1e-2)
    assert r["M"].sum() + r["sink_M"] == pytest.approx(M.sum(), rel=1e-12)
    assert r["n_steps"] < 200
    small = g.a_centres < 5e-7
    assert (r["M"][small].sum() + r["sink_M"]) > 0.01 * M.sum()


# ---------------------------------------------------------------------------
# charging
# ---------------------------------------------------------------------------

PC_CM = 3.086e18


def _agn_photons(L_bol, r_pc, xray=True):
    from pycalima.gsd.charging import PhotonGroups, xray_power_law
    from pycalima.models.dust_radiation.agn_hot_dust import AGNIllumination

    agn = AGNIllumination(L_bol, law="isotropic")
    ph = PhotonGroups.from_flux_lambda(agn.lam_cm, agn.flux(r_pc * PC_CM))
    if xray:
        E, nE = xray_power_law(r_pc * PC_CM, L_bol / 30.0)
        ph = ph + PhotonGroups.from_photon_density(E, nE)
    return ph


@pytest.fixture(scope="module")
def silicate_charging():
    from pycalima.gsd.charging import ChargingModel

    return ChargingModel("silicate", np.geomspace(3e-8, 1e-4, 7))


def test_agn_ionising_photon_density_matches_published_seds():
    """n(>13.6 eV) = 2.0e6 (WDB06 eq. 34) - 2.6e6 (TIK20 eq. 2) L45 (r/pc)^-2."""
    ph = _agn_photons(1e45, 1.0, xray=False)
    n_ion = ph.N[ph.L0 >= 13.6].sum()
    assert 1.8e6 < n_ion < 3.0e6


def test_charge_state_reproduces_the_group_solver(silicate_charging):
    """Per radius the state is RAMSES-CALIMA's solve; the charge points keep
    <Z> and sigma, and the ion factor is <J~> over the exact P(Z)."""
    from pycalima.models.dust_charge.shared_physics import DS87_J_function_vec

    ph = _agn_photons(1e46, 3.0)
    st = silicate_charging.solve(ph, T=1e4, n_e=1e2)
    for k, gt in enumerate(silicate_charging.tables):
        r = gt.solve(ph.N, 2.99792458e10, 1e4, 1e2, [(1e2, 1.0, 1.67262192e-24)], recomb=None)
        assert st.Zmean[k] == pytest.approx(r["Zmean"], rel=1e-6, abs=1e-6)
    Z, w = st.charge_points(st.a)
    np.testing.assert_allclose(Z @ w, st.Zmean, rtol=1e-12)
    np.testing.assert_allclose(np.sqrt(((Z - st.Zmean[:, None]) ** 2) @ w), st.Zsigma, rtol=1e-12)
    J = st.ion_focusing()
    k = 3
    Zk, Pk = st.exact_distribution(k)
    ref = P_ref = Pk @ DS87_J_function_vec(Zk, 1.0 + 0 * Zk, st.a[k] + 0 * Zk, 1e4 + 0 * Zk)
    assert J[k] == pytest.approx(ref, rel=1e-12)
    assert np.all(J[st.Zmean > 5] < 1e-3)       # positive grains repel ions


def test_coulomb_explosion_follows_tazaki_ichikawa_kokubo_2020(silicate_charging):
    """TIK20 (L45 = 10, T = 1e4 K, n_H = 1e2, S_max = 1e10): grains below
    a_CE ~ 0.01-0.1 um explode; a_CE falls much more slowly than the
    sublimation radius (r^-2) with distance."""
    a_ce = []
    for r_pc in (0.3, 3.0, 30.0):
        st = silicate_charging.solve(_agn_photons(1e46, r_pc), T=1e4, n_e=1e2)
        rate = st.explosion_rate(1e10)
        a_ce.append(st.a[rate > 0].max())
        assert np.all(rate[st.a > a_ce[-1]] == 0)
        assert np.all(st.explosion_rate(1e11) <= rate + 1e-300)
    assert 1e-6 < a_ce[1] < 1e-5
    assert a_ce[0] >= a_ce[1] >= a_ce[2]
    assert a_ce[0] / a_ce[2] < 100.0                 # vs 1e4 for an r^-2 law


def test_destroy_removes_grains_exponentially():
    from pycalima.gsd.charging import destroy

    g = _grid()
    N, M = _lognormal(g)
    N1, M1, lost = destroy(g, N, M, lambda a: np.full_like(a, 2.0), 0.5)
    np.testing.assert_allclose(N1, N * np.exp(-1.0), rtol=1e-12)
    assert M1.sum() + lost == pytest.approx(M.sum(), rel=1e-12)


def test_coulomb_rate_factor_matches_quadrature():
    from scipy.integrate import quad

    from pycalima.gsd.velocities import coulomb_rate_factor, mean_speed_normalised

    def pdf(v, s):
        if s < 1e-8:
            return 4 / np.sqrt(np.pi) * v ** 2 * np.exp(-v ** 2)
        return v / (np.sqrt(np.pi) * s) * (np.exp(-(v - s) ** 2) - np.exp(-(v + s) ** 2))

    for s in (0.0, 0.5, 3.0):
        mb = float(mean_speed_normalised(s))
        for vC2 in (-4.0, 0.0, 0.5, 9.0):
            for lo, hi in ((0.0, np.inf), (1.0, np.inf), (0.0, 2.0)):
                def f(v):
                    vi2 = v * v - vC2
                    if v <= 0 or vi2 < lo * lo or vi2 > hi * hi:
                        return 0.0
                    return v * pdf(v, s) * (1 - vC2 / v ** 2)
                pts = [np.sqrt(max(lo * lo + vC2, 0.0)), s]
                if np.isfinite(hi):
                    pts.append(np.sqrt(hi * hi + vC2))
                ref = quad(f, 0, s + 30, points=pts, limit=500, epsrel=1e-11)[0] / mb
                assert float(coulomb_rate_factor(1.0, s, vC2, lo, hi)) == pytest.approx(ref, rel=1e-9, abs=1e-14)


def test_grain_charges_suppress_or_focus_sticking():
    """Like charges cut the sticking rate of small grains, opposite charges
    raise it; the pair table reproduces the direct evaluation."""
    from pycalima.gsd.shattering import GrainCollisions
    from pycalima.gsd.velocities import RelativeVelocity

    a = np.geomspace(3e-8, 1e-5, 20)
    m = 4 / 3 * np.pi * a ** 3 * 3.3
    base = dict(T=100.0, n_H=1e4, mu=2.3, v_turb=3e4, L=3e17)
    def stick(Zfun, ppd=None):  # ppd None: direct evaluation
        rv = RelativeVelocity(**base, charges=Zfun)
        model = GrainCollisions("silicate", rv, coagulation=True, table_ppd=ppd)
        return model._fields(a, m)[2]
    one = lambda q: (lambda x: (np.full(np.shape(x) + (3,), q), np.array([1, 4, 1]) / 6.0))
    neutral, plus = stick(one(0.0)), stick(one(1.0))
    assert np.all(plus[:5, :5] < 0.5 * neutral[:5, :5])
    mixed = lambda x: (np.stack([np.full(np.shape(x), -1.0), np.zeros(np.shape(x)),
                                 np.ones(np.shape(x))], -1), np.array([0.5, 0.0, 0.5]))
    focused = stick(mixed)
    assert focused[0, 0] > neutral[0, 0]
    # the pair table: sticking rates through the square-root onset of the
    # Ormel & Cuzzi relative velocity of similar grains (t* = t_eta) converge
    # in the median only; shattering rates and impact speeds to ~1%
    tab = stick(one(1.0), ppd=24)
    big = plus > 1e-3 * plus.max()
    assert np.median(np.abs(tab[big] / plus[big] - 1)) < 0.01
    rv = _wim_velocity(charges=one(1.0))
    direct = GrainCollisions("silicate", rv, table_ppd=None)._fields(a, m)
    table = GrainCollisions("silicate", rv, table_ppd=24)._fields(a, m)
    big = direct[0] > 1e-3 * direct[0].max()
    np.testing.assert_allclose(table[0][big], direct[0][big], rtol=0.02)
    np.testing.assert_allclose(table[1][:, big], direct[1][:, big], rtol=0.005)


def test_ion_accretion_is_slowed_by_positive_grains():
    """An ionised key species accretes onto positive grains at x_n + x_i <J~>."""
    pop, gas, n_C = _carbon_box(1500.0, dust_frac=0.5)
    a = pop.grid.a_centres
    neutral = pop.adot(gas["C"], 1500.0)(a)
    pop.neutral_fraction = 0.25
    pop.ion_focusing = lambda x: np.full_like(np.asarray(x, float), 1e-3)
    ionised = pop.adot(gas["C"], 1500.0)(a)
    m = pop.model
    grow_n = m.condensation_alpha(1500.0) * m.unit_volume_cm3 * gas["C"] * np.sqrt(
        8 * 1.380649e-16 * 1500.0 / (np.pi * m.key_mass_amu * 1.66053906660e-24)) / (4 * m.key_per_unit)
    np.testing.assert_allclose(ionised - neutral, (0.25 + 0.75e-3 - 1.0) * grow_n, rtol=1e-10)


def test_terminal_drift_velocity_limits():
    """Subsonic: v = F / A (Epstein); supersonic: v -> (F / (pi a^2 rho))^1/2."""
    from pycalima.collapse import velocities as cvel
    from pycalima.gsd.velocities import terminal_drift_velocity

    a, T, rho, mu = 1e-5, 1e4, 1e-14, 1.4
    m = mu * cvel.MH_CGS
    v_th = np.sqrt(8 * cvel.KB_CGS * T / (np.pi * m))
    A = 4 * np.pi / 3 * a ** 2 * rho * v_th
    F_small = 1e-6 * A * v_th
    assert float(terminal_drift_velocity(F_small, a, T, rho, mu)) == pytest.approx(F_small / A, rel=1e-6)
    F_big = 1e8 * A * v_th
    v = float(terminal_drift_velocity(F_big, a, T, rho, mu))
    assert v == pytest.approx(np.sqrt(F_big / (np.pi * a ** 2 * rho)), rel=1e-3)


def test_hot_grain_from_optical_constants():
    """Mie absorption from Kitzmann & Heng (2018) constants: an absorbing Fe
    grain has Q_abs ~ 1 in the UV, a transparent MgO one far less; both give
    finite equilibrium temperatures in the AGN field at their rim distances."""
    from pycalima.models.dust_radiation.agn_hot_dust import PC_CM, AGNIllumination, HotGrain

    agn = AGNIllumination(1e46, law="isotropic")
    fe = HotGrain.from_optical_constants("iron", 0.1, "Fe.dat")
    mgo = HotGrain.from_optical_constants("periclase", 0.1, "MgO.dat")
    q = lambda g, lam_um: float(g.cross_section(lam_um * 1e-4)) / (np.pi * (1e-5) ** 2)
    assert 0.5 < q(fe, 0.3) < 3.0
    assert q(mgo, 0.3) < 1e-3 * q(fe, 0.3)
    # MgO barely radiates below ~8 um, so its rim lies at ~6.5 pc for L = 1e46
    for g, r_pc in ((fe, 1.0), (mgo, 10.0)):
        T = float(agn.temperature(g, r_pc * PC_CM))
        assert np.isfinite(T) and 300 < T < 5000
