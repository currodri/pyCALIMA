"""Invariants of the selectable GSD methods (``pycalima.gsd.closures``): closures, node rules, kernel averaging."""
from __future__ import annotations

import numpy as np
import pytest

from pycalima.gsd import closures as C
from pycalima.gsd.collisions import CoagulationModel, _terms, bin_nodes, collide
from pycalima.gsd.grid import MassGrid
from pycalima.gsd.optics import (combine_components, component_kappa_tables, kappa_direct, node_u,
                                 profile_weights, state_s)
from pycalima.gsd.remap import remap

RHO = 2.4
METHOD_NAMES = tuple(C.METHODS)


@pytest.fixture(autouse=True)
def _clean_env(isolated_env):
    """Every test runs in a clean $CALIMA_DATA root (nothing here reads or writes data)."""
    yield


def grid_of(n, a_lo=1e-7, a_hi=1e-4):
    a = np.geomspace(a_lo, a_hi, n + 1)
    return a, MassGrid(4.0 / 3.0 * np.pi * RHO * a ** 3, RHO)


def segment(a, lo, hi, p):
    """Exact (N, M) per bin of dn/da = a^p on [lo, hi]."""
    x0, x1 = np.clip(a[:-1], lo, hi), np.clip(a[1:], lo, hi)
    return ((x1 ** (p + 1) - x0 ** (p + 1)) / (p + 1),
            4.0 / 3.0 * np.pi * RHO * (x1 ** (p + 4) - x0 ** (p + 4)) / (p + 4))


def random_state(g, rng):
    nb = g.n_bins
    N = rng.random(nb) * (rng.random(nb) > 0.3)
    s = np.where(rng.random(nb) < 0.3, 10.0 ** rng.uniform(-12, 0, nb), rng.random(nb))
    s = np.where(rng.random(nb) < 0.15, 1.0 - s, s)
    return N, N * g.m_edges[:-1] * (1.0 + np.clip(s, 0.0, 1.0) * (g.etas - 1.0))


def test_default_method_is_ea_with_median_collision_nodes_and_gauss_rate_nodes():
    m = C.DEFAULT_METHOD
    assert (m.closure, m.eps, m.front_exponent, m.nodes, m.rate_nodes, m.kernel_average) == \
        ("ea", 1e-6, -3.5, "median", "gauss", 1)
    assert C.resolve(None) is C.DEFAULT_METHOD
    assert C.resolve("pl2m").closure == "pl2m" and C.resolve("pl2m").rate_nodes == "median"


@pytest.mark.parametrize("kwargs", [dict(closure="tc"), dict(nodes="gauss3"), dict(rate_nodes="centre"),
                                    dict(eps=0.0), dict(eps=np.inf), dict(kernel_average=0),
                                    dict(kernel_average=1.5), dict(front_exponent=np.nan)])
def test_unknown_names_and_bad_parameters_raise(kwargs):
    with pytest.raises(ValueError):
        C.GSDMethod(**kwargs)


def test_unknown_method_names_raise_everywhere():
    a, g = grid_of(8)
    N, M = segment(a, 2e-7, 2e-5, -3.5)
    with pytest.raises(ValueError):
        C.resolve("tc")
    with pytest.raises(TypeError):
        C.resolve(3)
    with pytest.raises(ValueError):
        remap(g, N, M, lambda x: x, method="pl2m_v2")
    with pytest.raises(ValueError):
        bin_nodes(g, N, M, method="eaf", rule="centre")
    with pytest.raises(ValueError):
        C.component_pieces(g.eta, [0.5], "right", "ea")        # ea's fronts are not tabulable in s


@pytest.mark.parametrize("name", METHOD_NAMES)
@pytest.mark.parametrize("n_bins", [4, 8, 32])
def test_every_closure_keeps_N_and_M_positive_and_inside_the_bin(name, n_bins):
    rng = np.random.default_rng(n_bins)
    m_lo, m_hi = 4.0 / 3.0 * np.pi * RHO * np.array([3.5e-8, 1e-4]) ** 3
    for g in (grid_of(n_bins)[1], MassGrid.graded([m_lo, 1e3 * m_lo, m_hi], [n_bins // 2, n_bins // 2], RHO)):
        uR = g.etas ** (1.0 / 3.0)
        for _ in range(40):
            N, M = random_state(g, rng)
            P = C.pieces(g, N, M, name)
            filled = N > 0
            assert np.all(P.n >= 0.0) and np.all(np.isfinite(P.A))
            assert np.all(P.u1 >= 1.0 - 1e-12) and np.all(P.u2 <= uR[:, None] * (1 + 1e-12))
            assert np.all(P.u2 >= P.u1)
            np.testing.assert_allclose(C.bin_moments(P, 0)[filled], N[filled], rtol=1e-11)
            np.testing.assert_allclose(C.bin_moments(P, 3)[filled] * P.mL[filled], M[filled], rtol=1e-9)


@pytest.mark.parametrize("name", METHOD_NAMES)
def test_remap_round_trip_and_conservation(name):
    rng = np.random.default_rng(3)
    a, g = grid_of(12)
    N, M = random_state(g, rng)
    N1, M1, lost, oN, oM = remap(g, N, M, lambda x: x, method=name)          # the identity map
    np.testing.assert_allclose(N1, N, rtol=1e-12, atol=1e-300)
    np.testing.assert_allclose(M1, M, rtol=1e-11, atol=1e-300)
    N2, M2, lost, oN, oM = remap(g, N, M, lambda x: 1.3 * x + 3e-7, method=name)
    assert abs(N2.sum() + oN + lost - N.sum()) < 1e-12 * N.sum()
    filled = N2 > 0
    assert np.all(M2[filled] >= N2[filled] * g.m_edges[:-1][filled] * (1 - 1e-12))
    assert np.all(M2[filled] <= N2[filled] * g.m_edges[1:][filled] * (1 + 1e-12))


def test_pl2m_paths_agree_and_eaf_is_pl2m_away_from_fronts():
    a, g = grid_of(10)
    N, M = segment(a, a[0], a[-1], -3.5)                 # filled everywhere, no fronts
    img = lambda x: 1.02 * x + 1e-8                       # noqa: E731
    fast = remap(g, N, M, img, method="pl2m")
    generic = C.remap_pieces(g, C.pieces(g, N, M, "pl2m"), img)
    np.testing.assert_allclose(fast[0], generic[0], rtol=1e-12)
    np.testing.assert_allclose(fast[1], generic[1], rtol=1e-12)
    m, w, wid = bin_nodes(g, N, M, widths=True, method="pl2m")
    m2, w2, wid2, _ = C.nodes(g, N, M, "pl2m", "median")
    np.testing.assert_allclose(m2, m, rtol=1e-12)
    np.testing.assert_allclose(wid2, wid, rtol=1e-10, atol=1e-12 * g.m_edges[-1])
    # interior bins only: the grid ends count as empty neighbours, so the end bins may be fronts
    eaf = remap(g, N, M, img)
    np.testing.assert_allclose(eaf[1][2:-2], fast[1][2:-2], rtol=1e-12)
    u = node_u(g.eta, 8)
    np.testing.assert_array_equal(profile_weights(g.eta, np.linspace(0, 1, 9), u),
                                  profile_weights(g.eta, np.linspace(0, 1, 9), u, method="pl2m"))


@pytest.mark.parametrize("name,kw", [("ea", {}), ("eaf", dict(front_exponent=0.0))])
@pytest.mark.parametrize("n_bins", [64, 128])
def test_edge_aware_growth_of_a_boxcar_is_exact_from_64_bins(name, kw, n_bins):
    """McKinnon+18's growth test (dn/da flat on [1, 5.6] nm, shifted by 25 nm in 100 steps): exact for ea, and for
    eaf when its front exponent is the boxcar's (PL2M: 2.5e-3 at 64 bins)."""
    method = C.GSDMethod(closure=name, **kw)
    a, g = grid_of(n_bins)
    lo, hi, da = 1e-7, 1e-7 * 1000 ** 0.25, 0.025e-4
    N, M = segment(a, lo, hi, 0.0)
    for _ in range(100):
        N, M, *_ = remap(g, N, M, lambda x: x + da / 100, method=method)
    exact = np.pi * RHO / 3.0 * ((hi + da) ** 4 - (lo + da) ** 4)
    assert abs(M.sum() / exact - 1.0) < 1e-10


@pytest.mark.parametrize("n_bins", [8, 64])
def test_eaf_keeps_an_mrn_segment_exact_under_multiplicative_growth(n_bins):
    """X = c a keeps dn/da ~ a^-3.5 a power law, so the eaf fronts (exponent -3.5) are exact: the binned N and M
    match the scaled segment, where PL2M's front bins do not."""
    a, g = grid_of(n_bins)
    lo, hi, c = 2.3e-7, 3.0e-6, 1.015
    N, M = segment(a, lo, hi, -3.5)
    Np, Mp = N.copy(), M.copy()
    for _ in range(100):
        N, M, *_ = remap(g, N, M, lambda x: c * x)
        Np, Mp, *_ = remap(g, Np, Mp, lambda x: c * x, method="pl2m")
    Ne, Me = segment(a, lo * c ** 100, hi * c ** 100, -3.5)          # the scaled segment, up to normalisation
    Ne, Me = Ne / Ne.sum() * Np.sum(), Me / Me.sum() * Mp.sum()        # (total N and M are exact for any closure)
    assert np.abs(M - Me).sum() / Me.sum() < 1e-8                      # round-off over 100 remaps
    if n_bins == 8:
        assert np.abs(Mp - Me).sum() / Me.sum() > 1e-3


def test_edge_aware_closures_beat_pl2m_on_the_growth_test_at_8_bins():
    a, g = grid_of(8)
    lo, hi, da = 1e-7, 1e-7 * 1000 ** 0.25, 0.025e-4
    exact = np.pi * RHO / 3.0 * ((hi + da) ** 4 - (lo + da) ** 4)
    err = {}
    for name in ("ea", "eaf", "pl2m"):
        N, M = segment(a, lo, hi, 0.0)
        for _ in range(100):
            N, M, *_ = remap(g, N, M, lambda x: x + da / 100, method=name)
        err[name] = abs(M.sum() / exact - 1.0)
    assert err["pl2m"] > 0.1 and err["ea"] < err["pl2m"] / 5 and err["eaf"] < err["pl2m"] / 5


@pytest.mark.parametrize("name", METHOD_NAMES)
def test_gauss_nodes_reproduce_the_radius_moments_up_to_the_third(name):
    """Exact to round-off for any non-degenerate in-bin measure (states within 1e-3 of an edge, near-spikes, lose
    digits in the variance; N and M stay exact there, checked below for all states)."""
    rng = np.random.default_rng(11)
    a, g = grid_of(10)
    N = rng.random(10) * (rng.random(10) > 0.2)
    M = N * g.m_edges[:-1] * (1.0 + rng.uniform(1e-3, 1.0 - 1e-3, 10) * (g.eta - 1.0))
    P = C.pieces(g, N, M, name)
    m, w, _, _ = C.nodes(g, N, M, name, "gauss")
    r = g.radius(m)
    filled = N > 1e-300
    for k in range(4):
        ref = C.radius_moments(P, k)[filled]
        np.testing.assert_allclose((w * r ** k).sum(axis=1)[filled], ref, rtol=1e-9)
    N, M = random_state(g, rng)                                            # includes near-spike states
    filled = N > 1e-300
    for rule in C.NODE_RULES:
        mm, wm, _, _ = C.nodes(g, N, M, name, rule)
        np.testing.assert_allclose(wm.sum(axis=1)[filled], N[filled], rtol=1e-13)
        np.testing.assert_allclose((wm * mm).sum(axis=1)[filled], M[filled], rtol=1e-11)


def test_kernel_averaging_changes_nothing_for_a_constant_kernel():
    a, g = grid_of(16)
    N, M = segment(a, 1e-7, 1e-5, -3.5)
    model = CoagulationModel(lambda m1, m2: 1e-8 + 0.0 * m1)
    base = _terms(g, N, M, model, C.GSDMethod(closure="pl2m", rate_nodes="median"))
    for n in (2, 4):
        avg = _terms(g, N, M, model, C.GSDMethod(closure="pl2m", rate_nodes="median", kernel_average=n))
        for x, y in zip(base, avg):
            np.testing.assert_allclose(y, x, rtol=1e-10, atol=1e-12 * np.abs(x).max())


@pytest.mark.parametrize("method", ["eaf", "ea", C.GSDMethod(nodes="gauss"), C.GSDMethod(kernel_average=2)])
def test_collisions_conserve_mass_with_every_method(method):
    a, g = grid_of(8)
    N, M = segment(a, 1e-7, 2e-5, -3.5)
    model = CoagulationModel(lambda m1, m2: 3e-10 * (m1 + m2) / g.m_edges[0])
    r = collide(g, N * 1e-6, M * 1e-6, model, 1e12, method=method)
    assert abs(r["M"].sum() + r["overflow_M"] - M.sum() * 1e-6) < 1e-12 * M.sum() * 1e-6
    assert np.all(r["N"] >= 0) and np.all(r["M"] >= 0)


def test_drift_velocities_use_the_local_base_closure_not_the_isolated_spike():
    a, g = grid_of(6)
    N = np.zeros(6)
    M = np.zeros(6)
    N[2], M[2] = 1.0, g.m_edges[2] * (1.0 + 0.2 * (g.eta - 1.0))        # one isolated bin
    P = C.pieces(g, N, M, "eaf")
    assert P.spike[2].any() and P.n[2, 3] == pytest.approx(1.0)            # the closure: a spike
    w = lambda r: r / a[2]                                                  # noqa: E731
    wN, wM = C.moment_velocities(g, N, M, w, "eaf")
    wN2, wM2 = C.moment_velocities(g, N, M, w, "pl2m")
    assert wM[2] > wN[2] * 1.05                                             # big grains still run ahead
    np.testing.assert_allclose([wN[2], wM[2]], [wN2[2], wM2[2]], rtol=1e-14)
    Pb = C.base_pieces(g, N, M, "eaf")
    np.testing.assert_allclose(wN[2], C.radius_moments(Pb, 1)[2] / a[2], rtol=1e-12)   # exact for w linear in a


def test_eaf_component_tables_reproduce_direct_integration():
    a, g = grid_of(8)
    N, M = segment(a, 3.3e-7, 2.1e-5, -3.5)                                 # both cut-offs inside bins: fronts
    fr = C.component_fractions(g, N, M, "eaf")
    assert fr[:, 1].max() == 1.0 and fr[:, 2].max() == 1.0
    lam = np.array([0.1, 0.55, 2.2])

    def per_gram(r):                                    # any smooth per-gram quantity, here vectorised in radius
        return np.exp(-np.outer(np.log(r / 1e-5) ** 2, lam))

    s = state_s(g, N, M)
    s_tab = np.unique(np.concatenate([np.linspace(0.0, 1.0, 5), s]))
    tabs = component_kappa_tables(g, s_tab, 16, per_gram, "eaf")
    filled = N > 0
    via_tables = combine_components(tabs, s_tab, s, fr)[filled]
    direct = kappa_direct(g, N, M, 16, per_gram, "eaf")[filled]
    np.testing.assert_allclose(via_tables, direct, rtol=1e-10)


def test_destroy_takes_the_rate_nodes():
    """charging.destroy evaluates the rate at the method's rate nodes: Gauss in radius by default, the median split
    for 'pl2m' (as before)."""
    from pycalima.gsd.charging import destroy

    a, g = grid_of(8)
    N, M = segment(a, 1e-7, 1e-5, -3.5)
    rate = lambda r: 1e-3 * (r / 1e-5) ** 2                                 # noqa: E731
    dt = 1e3
    for method, rule in ((None, "gauss"), ("pl2m", "median")):
        m, w = bin_nodes(g, N, M, method=method, rule=rule)
        keep = np.exp(-rate(g.radius(m)) * dt)
        N1, M1, dM = destroy(g, N, M, rate, dt, method=method)
        np.testing.assert_allclose(N1, (w * keep).sum(axis=1), rtol=1e-14)
        np.testing.assert_allclose(M1, (w * m * keep).sum(axis=1), rtol=1e-14)
    assert C.resolve(None).rate_nodes == "gauss" and C.resolve("pl2m").rate_nodes == "median"


# --- per-radius node tables (the closure-independent optics tables of the default ea) ----------------------------

def _smooth_per_gram(r):
    lam = np.array([0.1, 0.55, 2.2])
    return np.exp(-np.outer(np.log(r / 1e-5) ** 2, lam)) * (1e-5 / r)[:, None]


def test_node_average_equals_direct_integration_on_the_same_nodes():
    from pycalima.gsd.optics import NodeTable, kappa_direct, node_average, node_u

    a, g = grid_of(8)
    u = [node_u(e, 16) for e in g.etas]
    tab = NodeTable(g, u, [_smooth_per_gram(g.a_edges[i] * u[i]) for i in range(g.n_bins)], ["uniform 16"] * g.n_bins)
    rng = np.random.default_rng(5)
    for meth in ("ea", "eaf", "pl2m"):
        for _ in range(5):
            N, M = random_state(g, rng)
            filled = N > 0
            np.testing.assert_allclose(node_average(g, N, M, tab, meth)[filled],
                                       kappa_direct(g, N, M, 16, _smooth_per_gram, meth)[filled], rtol=1e-12)


def test_node_table_meets_its_tolerance_on_ea_states():
    from pycalima.gsd.optics import kappa_direct, node_average, node_table

    a, g = grid_of(8)
    tab = node_table(g, _smooth_per_gram, tol=3e-3)
    assert all(3 <= n <= 513 for n in tab.sizes)
    rng = np.random.default_rng(11)
    worst = 0.0
    for p in (-3.5, 0.0, 1.0):
        N, M = segment(a, a[1] * 1.3, a[-2] * 1.7, p)
        filled = N > 0
        ref = kappa_direct(g, N, M, 512, _smooth_per_gram, "ea")[filled]
        worst = max(worst, float(np.max(np.abs(node_average(g, N, M, tab, "ea")[filled] / ref - 1.0))))
    for _ in range(5):
        N, M = random_state(g, rng)
        filled = N > 0
        ref = kappa_direct(g, N, M, 512, _smooth_per_gram, "ea")[filled]
        worst = max(worst, float(np.max(np.abs(node_average(g, N, M, tab, "ea")[filled] / ref - 1.0))))
    assert worst <= 3e-3
