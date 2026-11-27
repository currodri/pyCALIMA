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


def pwl(g, w_nodes):
    """The in-bin law as a function w(i, u) of bin i and u = a / a_L: linear in a between the law radii."""
    from pycalima.gsd.drift import law_radii

    u_nodes = law_radii(g) / g.a_edges[:-1, None]
    return lambda i, u: np.interp(u, u_nodes[i], w_nodes[i])


def profile_integrals(g, N, M, w_of_u, n=40001):
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


NODE_SETS = {
    "rising": (1.0, 2.0, 3.0),
    "falling, negative": (-3.0, -2.5, -1.0),
    "direction change, lower piece": (-1.0, 0.5, 2.0),
    "direction change, upper piece": (2.0, 1.0, -1.5),
    "maximum inside, two changes": (-1.0, 2.0, -0.5),
}


@pytest.mark.parametrize("nodes", list(NODE_SETS), ids=list(NODE_SETS))
def test_the_flux_moments_are_the_profile_integrals(nodes):
    """int w dN and int w^2 dN of the right- and left-moving grains (per unit N; with m dN per unit M) against
    quadrature of the piecewise-linear law over the PL2M profile, for one-signed laws and direction changes in either
    piece. The first moments add up to w^N and w^M."""
    from pycalima.gsd.drift import _flux_moments, _law_u, _profile, moment_velocities

    g = grid()
    for s in (0.1, 0.6, 0.95):
        N, M = state(g, s)
        w_nodes = 1e4 * np.broadcast_to(np.array(NODE_SETS[nodes]), (g.n_bins, 3))
        P, Q = _flux_moments(_profile(g, N, M), _law_u(g), w_nodes)
        law = pwl(g, w_nodes)
        def part(sign, power, law=law):
            return lambda i, u: (np.maximum if sign > 0 else np.minimum)(law(i, u), 0.0) ** power

        for j, (fp, fq) in enumerate(((part(1, 1), part(-1, 1)), (part(1, 2), part(-1, 2)))):
            atol = 1e-7 * 1e4 ** (j + 1)
            for got, f in (((P[2 * j], P[2 * j + 1]), fp), ((Q[2 * j], Q[2 * j + 1]), fq)):
                eN, eM = profile_integrals(g, N, M, f)
                np.testing.assert_allclose(got[0], eN, rtol=1e-6, atol=atol)
                np.testing.assert_allclose(got[1], eM, rtol=1e-6, atol=atol)
        wN, wM = moment_velocities(g, N, M, w_nodes)
        np.testing.assert_allclose(P[0] + Q[0], wN, rtol=1e-12)
        np.testing.assert_allclose(P[1] + Q[1], wM, rtol=1e-12)
        assert all(np.all(x >= 0) for x in P) and np.all(Q[0] <= 0) and np.all(Q[1] <= 0) and np.all(Q[2] >= 0)


def test_the_law_is_exact_for_w_proportional_to_a():
    """Epstein drag under one acceleration: w ~ a is linear in a, so the law holds it exactly, and the mass, in the
    larger grains, moves faster than the number."""
    from pycalima.gsd.drift import law_radii, moment_velocities

    g = grid()
    r = law_radii(g)
    np.testing.assert_allclose(r[:, 0], g.a_edges[:-1])
    np.testing.assert_allclose(r[:, 2], g.a_edges[1:])
    assert np.all((r[:, 1] > r[:, 0]) & (r[:, 1] < r[:, 2]))
    N, M = state(g, 0.3)
    wN, wM = moment_velocities(g, N, M, 1e4 * r / 1e-7)
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
    """RAMSES interpolates beta(s) from per-bin tables. beta itself is steep near s = 0 and 1, but there the profile
    collapses onto an edge and the moment velocities hardly depend on it. With 257 points they agree with the solved
    ones to 4e-4 at s = 0.005 and better inside (1e-5 with 1025 points)."""
    from pycalima.gsd.drift import beta_tables, law_radii, moment_velocities

    g = grid()
    tabs = beta_tables(g)
    w_nodes = 1e4 * law_radii(g) / 1e-7
    for s in (0.005, 0.1, 0.5, 0.9, 0.995):
        N, M = state(g, s)
        exact = moment_velocities(g, N, M, w_nodes)
        table = moment_velocities(g, N, M, w_nodes, tables=tabs)
        for e, t in zip(exact, table, strict=True):
            np.testing.assert_allclose(t, e, rtol=1e-3)


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
    """The D1 bin (20-53 nm, eta 18.8): a top-hat under w ~ a sorts its grains until the front holds only the largest
    (s -> 1, beyond the power law's reach). No clip, N and M conserved; without the spike branch this case clipped."""
    from pycalima.gsd.drift import law_radii, tva_step

    a = np.array([19.96e-7, 53.1e-7])
    g = MassGrid(4.0 / 3.0 * np.pi * RHO * a ** 3, RHO)
    n = 200
    x = (np.arange(n) + 0.5) / n
    N = ((x > 0.1) & (x < 0.3)).astype(float)[:, None]
    M = N * g.m_edges[0] * (1.0 + 0.138 * (g.etas - 1.0))
    w_nodes = np.broadcast_to(1e4 * law_radii(g) / a[0], (n, 1, 3))
    c = 0.4 / w_nodes.max()
    tot, clips = (N.sum(), M.sum()), 0
    for _ in range(300):
        N, M, k = tva_step(g, N, M, w_nodes, c, slope_type=2)
        clips += k
    s = (M / np.where(N > 0, N, 1.0) / g.m_edges[0] - 1.0) / (g.etas - 1.0)
    assert clips == 0 and s[N > 1e-6].max() > 0.99
    np.testing.assert_allclose((N.sum(), M.sum()), tot, rtol=1e-13)


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
