"""Per-moment dust drift (``pycalima.gsd.drift``): the in-bin law's moments, the beta tables, the TVA step mirror."""
from __future__ import annotations

import numpy as np
import pytest

from pycalima.gsd.grid import MassGrid

pytestmark = pytest.mark.gsd_quick

RHO = 1.6


def grid(n=6):
    a = np.geomspace(1e-7, 1e-4, n + 1)
    return MassGrid(4.0 / 3.0 * np.pi * RHO * a ** 3, RHO)


def state(g, s=0.3, scale=1.0):
    N = scale * np.geomspace(1.0, 1e-4, g.n_bins)
    return N, N * g.m_edges[:-1] * (1.0 + s * (g.etas - 1.0))


def law_of(g, w_nodes, law):
    """The in-bin law as a function w(i, u) of bin i and u = a / a_L (gsd.drift.law_values)."""
    from pycalima.gsd.drift import law_values

    def w(i, u):
        a = np.zeros((u.size, g.n_bins))
        a[:, i] = u * g.a_edges[i]
        return law_values(g, np.broadcast_to(w_nodes, (u.size, *w_nodes.shape)), a, law)[:, i]

    return w


def profile_integrals(g, N, M, w_of_u, n=160001):
    """int f dN / N and int f m dN / M over each bin's PL2M profile for f = w_of_u(i, u), by quadrature in u."""
    from pycalima.gsd.drift import _lambda

    lam = _lambda(g, N, M)
    fN, fM = [], []
    for i in range(g.n_bins):
        u = np.geomspace(1.0, g.etas[i] ** (1.0 / 3.0), n)
        dn = u ** lam[i]
        f = w_of_u(i, u)
        fN.append(np.trapezoid(f * dn, u) / np.trapezoid(dn, u))
        fM.append(np.trapezoid(f * u ** 3 * dn, u) / np.trapezoid(u ** 3 * dn, u))
    return np.array(fN), np.array(fM)


# drift shapes over the bin as functions of t, its fraction in ln a; each law samples them at its own nodes
SHAPES = {
    "rising": lambda t: 1.0 + 2.0 * t,
    "falling, negative": lambda t: -3.0 + 2.0 * t ** 2,
    "direction change, lower part": lambda t: -1.0 + 3.0 * t,
    "direction change, upper part": lambda t: 2.0 - 3.5 * t ** 1.5,
    "maximum inside, two changes": lambda t: -1.0 + 9.0 * t * (1.0 - t) + 0.5 * t,
}
LAW_T = {"linear": (0.0, 0.5, 1.0), "quadratic": (0.0, 0.5, 1.0), "cubic": (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)}


@pytest.mark.parametrize("law", list(LAW_T))
@pytest.mark.parametrize("shape", list(SHAPES), ids=list(SHAPES))
def test_the_flux_moments_are_the_profile_integrals(law, shape):
    """int w dN and int w^2 dN of the right- and left-moving grains (per unit N; with m dN per unit M) against
    quadrature of the law over the PL2M profile, for one-signed laws and direction changes inside the bin, for every
    law. The first moments add up to w^N and w^M."""
    from pycalima.gsd.drift import _flux_moments, _profile, moment_velocities

    g = grid(3)
    w_nodes = 1e4 * np.broadcast_to(np.array([SHAPES[shape](t) for t in LAW_T[law]]), (g.n_bins, len(LAW_T[law])))
    law_fn = law_of(g, w_nodes, law)
    for s in (0.1, 0.6, 0.95):
        N, M = state(g, s)
        P, Q = _flux_moments(_profile(g, N, M), g, w_nodes, law)

        def part(sign, power, law_fn=law_fn):
            return lambda i, u: (np.maximum if sign > 0 else np.minimum)(law_fn(i, u), 0.0) ** power

        for j, (fp, fq) in enumerate(((part(1, 1), part(-1, 1)), (part(1, 2), part(-1, 2)))):
            atol = 1e-7 * 1e4 ** (j + 1)
            for got, f in (((P[2 * j], P[2 * j + 1]), fp), ((Q[2 * j], Q[2 * j + 1]), fq)):
                eN, eM = profile_integrals(g, N, M, f)
                np.testing.assert_allclose(got[0], eN, rtol=1e-6, atol=atol)
                np.testing.assert_allclose(got[1], eM, rtol=1e-6, atol=atol)
        wN, wM = moment_velocities(g, N, M, w_nodes, law=law)
        np.testing.assert_allclose(P[0] + Q[0], wN, rtol=1e-12)
        np.testing.assert_allclose(P[1] + Q[1], wM, rtol=1e-12)
        assert all(np.all(x >= 0) for x in P) and np.all(Q[0] <= 0) and np.all(Q[1] <= 0) and np.all(Q[2] >= 0)


@pytest.mark.parametrize("law", list(LAW_T))
def test_the_law_passes_through_its_nodes_and_its_maximum_is_found(law):
    """law_values at the law radii gives the node values back; law_max_abs is the maximum of |w| over the bin (a dense
    scan), at an interior maximum too."""
    from pycalima.gsd.drift import law_max_abs, law_radii, law_values

    g = grid(3)
    w_nodes = 1e4 * np.broadcast_to(np.array([SHAPES["maximum inside, two changes"](t) for t in LAW_T[law]]),
                                    (g.n_bins, len(LAW_T[law])))
    r = law_radii(g, law)
    for k in range(r.shape[1]):
        np.testing.assert_allclose(law_values(g, w_nodes, r[:, k], law), w_nodes[:, k], rtol=1e-10)
    a = np.geomspace(g.a_edges[:-1], g.a_edges[1:], 20001)
    scan = np.abs(law_values(g, np.broadcast_to(w_nodes, (a.shape[0], *w_nodes.shape)), a, law)).max(axis=0)
    np.testing.assert_allclose(law_max_abs(g, w_nodes, law), scan, rtol=1e-6)


@pytest.mark.parametrize("law", list(LAW_T))
def test_the_law_is_exact_for_w_proportional_to_a(law):
    """Epstein drag under one acceleration: w ~ a is linear in a, so every law holds it exactly, and the mass, in the
    larger grains, moves faster than the number."""
    from pycalima.gsd.drift import law_radii, moment_velocities

    g = grid()
    r = law_radii(g, law)
    np.testing.assert_allclose(r[:, 0], g.a_edges[:-1])
    np.testing.assert_allclose(r[:, -1], g.a_edges[1:])
    assert np.all(np.diff(r, axis=1) > 0)
    N, M = state(g, 0.3)
    wN, wM = moment_velocities(g, N, M, 1e4 * r / 1e-7, law=law)
    eN, eM = profile_integrals(g, N, M, lambda i, u: 1e4 * u * g.a_edges[i] / 1e-7)
    np.testing.assert_allclose(wN, eN, rtol=1e-6)
    np.testing.assert_allclose(wM, eM, rtol=1e-6)
    assert np.all(wM > wN)


def test_gauss_node_velocities_are_exact_in_number_for_w_proportional_to_a():
    from pycalima.gsd.drift import gauss_velocities, law_radii, moment_velocities

    g = grid()
    N, M = state(g, 0.4)
    wN_g, wM_g = gauss_velocities(g, N, M, lambda a: 1e4 * a / 1e-7, method="pl2m")
    wN, wM = moment_velocities(g, N, M, 1e4 * law_radii(g) / 1e-7)
    np.testing.assert_allclose(wN_g, wN, rtol=1e-6)                   # int a dN: inside the Gauss rule's exactness
    np.testing.assert_allclose(wM_g, wM, rtol=5e-2)                   # int a^4 dN: not exact, close


def test_the_beta_tables_give_the_solved_velocities():
    """RAMSES interpolates beta from per-bin tables in the log mean mass q = ln(m_bar/m_L)/ln(eta). Over bins from
    eta 1.5 to 1.25e5 and in-bin states q = 0.002-0.998, the moment velocities from the tables agree with the solved
    ones to 1e-3 of max|w|, for a one-signed drift and one changing sign inside the bin. A table in the linear
    s = (m_bar/m_L - 1)/(eta - 1) fails this in wide bins: at eta 1.25e5 its first interval holds q < 0.149, and there
    w^M was off by 11 % of max|w| (audit studies/drift_audit_20261127, bug 1)."""
    from pycalima.gsd.drift import beta_tables, law_radii, moment_velocities

    for eta in (1.5, 18.8, 1e3, 1.25e5):
        g = MassGrid(np.array([1.0, eta]), 1.0)
        tabs = beta_tables(g)
        u = law_radii(g, "cubic") / law_radii(g, "cubic")[0, 0]
        for w_nodes in (u, u - np.sqrt(eta ** (1.0 / 3.0))):        # w ~ a, and changing sign at the centre
            q = np.linspace(0.002, 0.998, 499)
            M = (eta ** q)[:, None]
            N = np.ones_like(M)
            w = np.broadcast_to(w_nodes, (len(q), 1, 4))
            exact = moment_velocities(g, N, M, w, law="cubic")
            table = moment_velocities(g, N, M, w, tables=tabs, law="cubic")
            scale = np.abs(w_nodes).max()
            for e, t in zip(exact, table, strict=True):
                assert np.abs(t - e).max() / scale < 1e-3, (eta, np.abs(t - e).max() / scale)


def test_states_at_the_bin_edges_move_as_spikes():
    """A state beyond the power law's reach (here at s = 0 and s = 1 exactly) is a spike at its mean mass: N and M
    move at the drift there."""
    from pycalima.gsd.drift import moment_velocities

    g = grid()
    w_nodes = 1e4 * np.broadcast_to(np.array([-1.0, 2.0, 0.5]), (g.n_bins, 3))
    for s, w_edge in ((0.0, w_nodes[:, 0]), (1.0, w_nodes[:, 2])):
        N, M = state(g, s)
        wN, wM = moment_velocities(g, N, M, w_nodes)
        np.testing.assert_allclose(wN, w_edge, rtol=1e-12)
        np.testing.assert_allclose(wM, w_edge, rtol=1e-12)


def test_a_uniform_drift_gives_the_cell_velocity_step():
    """With the same drift at every node every grain moves alike: the split step is the single-velocity step."""
    from pycalima.gsd.drift import tva_step, tva_step_velocities

    g = grid(4)
    x = np.linspace(0.0, 1.0, 32, endpoint=False)
    prof = 1.0 + 0.8 * np.sin(2 * np.pi * x)
    N0, M0 = state(g, 0.4)
    N, M = prof[:, None] * N0[None], prof[:, None] * M0[None]
    w = 1e4 * (1.0 + 0.5 * np.cos(2 * np.pi * x))[:, None] * np.ones(g.n_bins)
    c = 0.4 / w.max()
    a = tva_step(g, N, M, np.repeat(w[..., None], 3, axis=-1), c, slope_type=2)
    b = tva_step_velocities(g, N, M, w, w, c, slope_type=2)
    np.testing.assert_allclose(a[0], b[0], rtol=1e-13)
    np.testing.assert_allclose(a[1], b[1], rtol=1e-13)


def test_a_uniform_column_is_left_exactly_as_it_is():
    """The same state and drift in every cell: every face carries the same flux, and the step leaves each cell bit for
    bit (what lets RAMSES's uniform regions keep the one-zone cell's result)."""
    from pycalima.gsd.drift import law_radii, tva_step

    g = grid(4)
    N0, M0 = state(g, 0.37)
    N, M = np.tile(N0, (16, 1)), np.tile(M0, (16, 1))
    w = np.broadcast_to(np.array([-1e4, 2e4, 0.7e4]) * (law_radii(g) / 1e-7) ** 0, (16, g.n_bins, 3))
    N1, M1, _ = tva_step(g, N, M, w, 0.4 / 2e4, slope_type=2)
    assert np.array_equal(N1, N) and np.array_equal(M1, M)


def _drift_run(g, w_of_x, slope_type, n_steps=60, n_cells=64):
    """A top-hat with a structured top drifting at the node values w_of_x(x) [cell, 3] (every bin alike)."""
    from pycalima.gsd.drift import tva_step

    x = np.linspace(0.0, 1.0, n_cells, endpoint=False)
    prof = np.where((x > 0.2) & (x < 0.5), 1.0, 1e-3) * (1.0 + 0.3 * np.sin(6 * np.pi * x))
    N0, M0 = state(g, 0.4)
    N, M = prof[:, None] * N0[None], prof[:, None] * M0[None]
    w_nodes = np.repeat(w_of_x(x)[:, None, :], g.n_bins, axis=1)
    c = 0.4 / np.max(np.abs(w_nodes))
    clips = 0
    for _ in range(n_steps):
        N, M, k = tva_step(g, N, M, w_nodes, c, slope_type=slope_type)
        clips += k
    return N, M, prof[:, None] * N0[None], prof[:, None] * M0[None], clips


@pytest.mark.parametrize("slope_type", [0, 1, 2])
def test_the_tva_step_conserves_and_stays_realisable(slope_type):
    """A top-hat with a structured top under w ~ a: N and M of every bin are conserved to round-off, nothing is
    negative, and no cell needs a clip."""
    g = grid(4)
    u = np.array([1.0, g.etas[0] ** (1 / 6), g.etas[0] ** (1 / 3)])
    N, M, N0, M0, clips = _drift_run(g, lambda x: 1e4 * np.broadcast_to(u, (x.size, 3)), slope_type)
    np.testing.assert_allclose(N.sum(axis=0), N0.sum(axis=0), rtol=1e-13)
    np.testing.assert_allclose(M.sum(axis=0), M0.sum(axis=0), rtol=1e-13)
    assert np.all(N >= 0) and np.all(M >= 0) and clips == 0


@pytest.mark.parametrize("slope_type", [1, 2])
def test_a_direction_change_inside_the_bin_stays_realisable(slope_type):
    """Small grains one way and large grains the other, with the drift varying in space: the bin's grains separate in
    both directions, conserved, and every cell's mean mass stays inside its bin."""
    g = grid(4)

    def w(x):
        return np.stack([-1e4 * (1.0 + 0.5 * np.sin(2 * np.pi * x)), 0.5e4 * np.ones_like(x),
                         2e4 * (1.0 + 0.3 * np.cos(2 * np.pi * x))], axis=1)

    N, M, N0, M0, clips = _drift_run(g, w, slope_type)
    np.testing.assert_allclose(N.sum(axis=0), N0.sum(axis=0), rtol=1e-13)
    np.testing.assert_allclose(M.sum(axis=0), M0.sum(axis=0), rtol=1e-13)
    assert np.all(N >= 0) and np.all(M >= 0) and clips == 0


@pytest.mark.parametrize("law", ["quadratic", "cubic"])
def test_the_polynomial_laws_are_realisable_at_the_cfl_bound(law):
    """As the next test, with the quadratic and the cubic laws: the drift has a maximum inside the bin and two
    direction changes, and the CFL is on the law's own maximum (law_max_abs, above the nodes)."""
    from pycalima.gsd.drift import law_max_abs, tva_step

    g = grid(3)
    rng = np.random.default_rng(7)
    n = 96
    N = rng.uniform(0.0, 1.0, (n, g.n_bins)) ** 3
    s = rng.uniform(0.0, 1.0, (n, g.n_bins))
    s[::7], s[3::11] = 0.0, 1.0
    M = N * g.m_edges[:-1] * (1.0 + s * (g.etas - 1.0))
    x = np.arange(n) / n
    amp = (1.0 + 0.5 * np.sin(2 * np.pi * x))[:, None, None]
    w_nodes = 1e4 * amp * np.array([SHAPES["maximum inside, two changes"](t) for t in LAW_T[law]])[None, None, :]
    w_nodes = np.broadcast_to(w_nodes, (n, g.n_bins, len(LAW_T[law])))
    c = 0.5 / np.max(law_max_abs(g, w_nodes, law))
    tot = N.sum(axis=0), M.sum(axis=0)
    clips = 0
    for _ in range(200):
        N, M, k = tva_step(g, N, M, w_nodes, c, slope_type=2, law=law)
        clips += k
    np.testing.assert_allclose(N.sum(axis=0), tot[0], rtol=1e-13)
    np.testing.assert_allclose(M.sum(axis=0), tot[1], rtol=1e-13)
    assert clips == 0 and np.all(N >= 0)


def test_the_step_is_realisable_at_the_cfl_bound():
    """Random N and s (some cells exactly at s = 0 and s = 1), a drift with a maximum inside the bin and two direction
    changes, varying in space, MC slopes, at c max|w| = 1/2: every cell stays inside its bin with no clip (the module
    doc's argument), and N and M are conserved."""
    from pycalima.gsd.drift import tva_step

    g = grid(3)
    rng = np.random.default_rng(7)
    n = 96
    N = rng.uniform(0.0, 1.0, (n, g.n_bins)) ** 3
    s = rng.uniform(0.0, 1.0, (n, g.n_bins))
    s[::7], s[3::11] = 0.0, 1.0
    M = N * g.m_edges[:-1] * (1.0 + s * (g.etas - 1.0))
    x = np.arange(n) / n
    w = np.stack([-1e4 * (1.0 + 0.5 * np.sin(2 * np.pi * x)), 2e4 * (1.0 + 0.4 * np.cos(6 * np.pi * x)),
                  -0.5e4 * (1.0 + 0.8 * np.sin(4 * np.pi * x))], axis=1)
    w_nodes = np.repeat(w[:, None, :], g.n_bins, axis=1)
    c = 0.5 / np.max(np.abs(w_nodes))
    tot = N.sum(axis=0), M.sum(axis=0)
    clips = 0
    for _ in range(300):
        N, M, k = tva_step(g, N, M, w_nodes, c, slope_type=2)
        clips += k
    np.testing.assert_allclose(N.sum(axis=0), tot[0], rtol=1e-13)
    np.testing.assert_allclose(M.sum(axis=0), tot[1], rtol=1e-13)
    assert clips == 0 and np.all(N >= 0)


def test_a_wide_bin_stays_realisable_where_its_states_reach_the_edges():
    """The D1 bin (20-53 nm, eta 18.8). A top-hat under w ~ a sorts its grains until the front holds nearly only the
    largest (s > 0.98: the faces' mean masses stay within their neighbours', so the front reaches the edge as the
    two-moment equations carry it, not by a jump of a face state); and states next to both edges (s = 1e-6 and
    1 - 1e-6, the spike branch) drifting both ways. No clip, N and M conserved; without the spike branch these
    clipped."""
    from pycalima.gsd.drift import law_radii, tva_step

    a = np.array([19.96e-7, 53.1e-7])
    g = MassGrid(4.0 / 3.0 * np.pi * RHO * a ** 3, RHO)
    n = 200
    x = (np.arange(n) + 0.5) / n
    sval = lambda N, M: (M / np.where(N > 0, N, 1.0) / g.m_edges[0] - 1.0) / (g.etas - 1.0)    # noqa: E731
    N = ((x > 0.1) & (x < 0.3)).astype(float)[:, None]
    M = N * g.m_edges[0] * (1.0 + 0.138 * (g.etas - 1.0))
    w_nodes = np.broadcast_to(1e4 * law_radii(g) / a[0], (n, 1, 3))
    c = 0.4 / w_nodes.max()
    tot, clips = (N.sum(), M.sum()), 0
    for _ in range(300):
        N, M, k = tva_step(g, N, M, w_nodes, c, slope_type=2)
        clips += k
    assert clips == 0 and sval(N, M)[N > 1e-6].max() > 0.98
    np.testing.assert_allclose((N.sum(), M.sum()), tot, rtol=1e-13)

    s0 = np.where(x < 0.5, 1e-6, 1.0 - 1e-6)[:, None]
    N = (1.0 + 0.5 * np.sin(2 * np.pi * x))[:, None]
    M = N * g.m_edges[0] * (1.0 + s0 * (g.etas - 1.0))
    w_nodes = np.broadcast_to(1e4 * (law_radii(g) / a[0] - 1.6), (n, 1, 3))
    c = 0.4 / np.abs(w_nodes).max()
    tot, clips = (N.sum(), M.sum()), 0
    for _ in range(200):
        N, M, k = tva_step(g, N, M, w_nodes, c, slope_type=2)
        clips += k
    assert clips == 0
    np.testing.assert_allclose((N.sum(), M.sum()), tot, rtol=1e-13)


def test_the_face_states_follow_n_and_hold_the_mean_mass_within_the_neighbours():
    """``_reconstruct``: N takes its full limited slope; at a uniform in-bin state with a density gradient M's slope is
    the mean mass times N's (second order, no round-off theta); with a varying in-bin state both faces' mean masses lie
    within the cell's and its neighbours' range; and the face states always average to the cell in N and M."""
    from pycalima.gsd.drift import _limited_slope, _mean_mass_range, _reconstruct

    g = grid()
    x = (np.arange(64) + 0.5) / 64
    N = (1.0 + 0.6 * np.sin(2 * np.pi * x))[:, None] * np.ones((1, g.n_bins))
    for s in (1e-3, 0.37, 0.999):
        M = N * g.m_edges[:-1] * (1.0 + s * (g.etas - 1.0))
        (NL, ML), (NR, MR) = _reconstruct(g, N, M, 2, True)
        np.testing.assert_allclose(NR - NL, _limited_slope(N, 2, True), rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(MR - ML, (M / N) * (NR - NL), rtol=1e-9, atol=1e-300)
    s = 0.5 + 0.45 * np.sin(2 * np.pi * 3 * x + 0.4)[:, None]
    M = N * g.m_edges[:-1] * (1.0 + s * (g.etas - 1.0))
    lo, hi = _mean_mass_range(g, N, M, True)
    (NL, ML), (NR, MR) = _reconstruct(g, N, M, 2, True)
    for Nf, Mf in ((NL, ML), (NR, MR)):
        assert np.all(Mf / Nf >= lo * (1 - 1e-12)) and np.all(Mf / Nf <= hi * (1 + 1e-12))
    np.testing.assert_allclose(0.5 * (NL + NR), N, rtol=1e-14)
    np.testing.assert_allclose(0.5 * (ML + MR), M, rtol=1e-14)


def test_prolongation_and_restriction_are_exact():
    """Parent s: the children keep the parent's in-bin state and average to it, with or without a slope."""
    from pycalima.gsd.drift import prolong, restrict

    g = grid()
    N, M = state(g, 0.7)
    for dM in (None, 0.8 * M, -0.3 * M):
        Nc, Mc = prolong(N, M, dM)
        np.testing.assert_allclose(Mc / Nc, np.broadcast_to(M / N, Nc.shape), rtol=1e-15)
        np.testing.assert_allclose(Nc.mean(axis=0), N, rtol=1e-15)
        np.testing.assert_allclose(Mc.mean(axis=0), M, rtol=1e-15)
        assert np.all(Nc >= 0)
        Nr, Mr = restrict(Nc, Mc)
        np.testing.assert_allclose(Nr, N, rtol=1e-15)
        np.testing.assert_allclose(Mr, M, rtol=1e-15)
