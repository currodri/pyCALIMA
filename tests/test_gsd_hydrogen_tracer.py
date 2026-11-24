"""The hydrogen tracer M_H (A3c phase 5): carried by every operator at the donor bin's H/C, conserved with the gas."""
from __future__ import annotations

import numpy as np
import pytest
from test_gsd_cell_optics import jones  # noqa: F401  (the Jones 2012 tables fixture)

from pycalima.gsd.grid import MassGrid
from pycalima.gsd.remap import remap, remap_hydrogen

pytestmark = pytest.mark.gsd_quick

RHO = 1.6


@pytest.fixture(autouse=True)
def _clean_env(isolated_env):
    yield


def grid(n=8):
    m = 4.0 / 3.0 * np.pi * RHO * np.array([1e-7, 1e-4]) ** 3
    return MassGrid(np.geomspace(m[0], m[1], n + 1), RHO)


def state(g):
    a = g.a_centres
    N = a ** -3.0
    m = np.sqrt(g.m_edges[:-1] * g.m_edges[1:])
    return N, N * m


def moved(g, N, M, H, image):
    N1, M1, *_ = remap(g, N, M, image)
    return N1, M1, *remap_hydrogen(g, N, M, H, image, N1)


def test_growing_grains_keep_their_hydrogen():
    g = grid()
    N, M = state(g)
    H = 0.3 * M
    _, M1, H1, gas, over = moved(g, N, M, H, lambda a: a + 2e-7)
    assert gas == pytest.approx(0.0, abs=1e-14 * H.sum())
    assert H1.sum() + over == pytest.approx(H.sum(), rel=1e-12)
    filled = M1 > 0
    assert np.all(H1[filled] / M1[filled] <= 0.3 * (1 + 1e-12))       # accreted carbon brings no hydrogen
    assert H1[filled].sum() / M1[filled].sum() < 0.3


def test_shrinking_grains_keep_their_h_per_c_and_release_the_rest():
    g = grid()
    N, M = state(g)
    H = 0.3 * M
    _, M1, H1, gas, over = moved(g, N, M, H, lambda a: np.maximum(a - 2e-7, 0.0))
    np.testing.assert_allclose(H1, 0.3 * M1, rtol=1e-10, atol=1e-14 * H.sum())
    assert H1.sum() + gas + over == pytest.approx(H.sum(), rel=1e-12)
    assert gas > 0.0


def test_mixed_h_per_c_stays_within_its_bounds_and_is_conserved():
    g = grid()
    N, M = state(g)
    x = np.linspace(0.05, 0.6, g.n_bins)
    H = x * M
    for image in (lambda a: 1.3 * a, lambda a: 0.8 * a):
        _, M1, H1, gas, over = moved(g, N, M, H, image)
        assert H1.sum() + gas + over == pytest.approx(H.sum(), rel=1e-12)
        filled = M1 > 1e-12 * M.sum()
        r = H1[filled] / M1[filled]
        assert np.all(r <= x.max() * (1 + 1e-10))
        assert np.all(r >= 0.0)


def test_pl2m_is_not_traced():
    g = grid()
    N, M = state(g)
    with pytest.raises(NotImplementedError):
        remap_hydrogen(g, N, M, 0.3 * M, lambda a: a, N, method="pl2m")


def test_sputtering_keeps_the_h_per_c_and_returns_the_rest():
    from pycalima.gsd.sputtering import sputter

    g = grid()
    N, M = state(g)
    H = 0.3 * M
    adot = lambda a: np.full(np.shape(a), -1e-15)                     # noqa: E731
    N1, M1, _, H1, dH = sputter(g, N, M, adot, 3e7 * 3.15576e7, H=H)
    ref = sputter(g, N, M, adot, 3e7 * 3.15576e7)
    np.testing.assert_array_equal(N1, ref[0])
    np.testing.assert_array_equal(M1, ref[1])
    np.testing.assert_allclose(H1, 0.3 * M1, rtol=1e-10, atol=1e-14 * H.sum())
    assert H1.sum() + dH == pytest.approx(H.sum(), rel=1e-12)


def test_accretion_dilutes_the_hydrogen_and_conserves_it():
    from dataclasses import replace

    from pycalima.gsd.exchange import Population, exchange_step
    from pycalima.models.dust_radiation.sublimation_kinetics import get_sublimation_model

    g = grid()
    N, M = state(g)
    sub = replace(get_sublimation_model("graphite", "GD89"), alpha_gd89=0.3, density_gcm3=RHO)
    ref = Population(sub, g, N.copy(), M.copy(), np.full(g.n_bins, 20.0), nucleation=False)
    pop = Population(sub, g, N.copy(), M.copy(), np.full(g.n_bins, 20.0), nucleation=False, H=0.3 * M)
    gas_ref, gas = {"C": 1e-2}, {"C": 1e-2}
    exchange_step([ref], gas_ref, 100.0, 1e14)
    exchange_step([pop], gas, 100.0, 1e14)
    np.testing.assert_array_equal(pop.M, ref.M)
    assert gas["C"] == gas_ref["C"] and pop.M.sum() > M.sum()
    assert pop.H.sum() + pop.overflow_H + pop.h_to_gas == pytest.approx(0.3 * M.sum(), rel=1e-12)
    assert pop.H.sum() / pop.M.sum() < 0.3


def collision_case():
    from test_gsd_cell import PC, carbon

    from pycalima.gsd.shattering import GrainCollisions
    from pycalima.gsd.velocities import RelativeVelocity

    c = carbon(n_H=30.0)
    g = c.grid
    rv = RelativeVelocity(T=100.0, n_H=30.0, mu=1.4, v_turb=1e6, L=10 * PC)
    model = GrainCollisions(c.material, rv, law="KT10", coagulation=True, a_frag_min=float(g.a_edges[0]),
                            collector="remove", n_speed=4, table_ppd=None, threads=1)
    return g, c.N, c.M, model


def test_collisions_carry_a_uniform_h_per_c_exactly():
    from pycalima.gsd.collisions import collide

    g, N, M, model = collision_case()
    dt = 3e6 * 3.15576e7
    ref = collide(g, N, M, model, dt)
    r = collide(g, N, M, model, dt, H=0.3 * M)
    np.testing.assert_array_equal(r["M"], ref["M"])
    np.testing.assert_array_equal(r["N"], ref["N"])
    assert ref["sink_M"] > 0.0
    np.testing.assert_allclose(r["H"], 0.3 * r["M"], rtol=1e-9, atol=1e-14 * M.sum())
    assert r["sink_H"] == pytest.approx(0.3 * r["sink_M"], rel=1e-9)
    assert r["overflow_H"] == pytest.approx(0.3 * r["overflow_M"], rel=1e-9, abs=1e-14 * M.sum())


def test_collisions_conserve_a_mixed_h_per_c_within_its_bounds():
    from pycalima.gsd.collisions import collide

    g, N, M, model = collision_case()
    x = np.linspace(0.05, 0.6, g.n_bins)
    r = collide(g, N, M, model, 3e6 * 3.15576e7, H=x * M)
    assert r["H"].sum() + r["overflow_H"] + r["sink_H"] == pytest.approx((x * M).sum(), rel=1e-10)
    filled = r["M"] > 1e-12 * M.sum()
    ratio = r["H"][filled] / r["M"][filled]
    assert np.all((ratio >= x.min() * (1 - 1e-9)) & (ratio <= x.max() * (1 + 1e-9)))


def test_update_conserves_the_tracked_hydrogen_with_the_gas():
    from dataclasses import replace

    from test_gsd_cell import PC, YR, carbon, inputs

    from pycalima.gsd import cell

    c = carbon(n_H=1.0)
    x = np.linspace(0.05, 0.6, c.grid.n_bins)
    c.M_H = x * c.M
    inp = replace(inputs(n_H=1.0, T=1e6, gas_C=1e-4), v_turb=1e6, L=100 * PC)
    res = cell.update([c], inp, None, 1e6 * YR, cell.Options(table_ppd=None))
    assert abs(res.conservation["H"]) < 1e-12 and abs(res.conservation["C"]) < 1e-12
    assert res.gas["H"] > 0.0                                           # sputtering released some


def test_a_uniform_h_per_c_survives_collisions_erosion_and_explosion():
    from dataclasses import replace

    from test_gsd_cell import PC, YR, carbon, inputs

    from pycalima.gsd import cell

    c = carbon(n_H=1.0)
    c.M_H = 0.3 * c.M
    inp = replace(inputs(n_H=1.0, T=1e6, gas_C=1e-4), v_turb=1e6, L=100 * PC)
    g = c.grid
    state = type("S", (), {"a": g.a_centres, "charge_points": None, "ion_focusing_of": None})()
    pre = cell.Precool([0.0], [0.0], [state], [None], [np.full(g.n_bins, 1e-14)], [0.0])
    opt = cell.Options(table_ppd=None, processes=frozenset({"coagulation", "shattering", "sputtering", "explosion"}))
    res = cell.update([c], inp, pre, 1e6 * YR, opt)
    filled = c.M > 1e-12 * c.M.sum()
    np.testing.assert_allclose(c.M_H[filled] / c.M[filled], 0.3, rtol=1e-8)
    assert abs(res.conservation["H"]) < 1e-12


def test_the_hydrogen_step_is_exact_and_relaxes_to_the_equilibrium_gap(jones):  # noqa: F811
    from dataclasses import replace

    from test_gsd_cell import YR, inputs
    from test_gsd_cell_optics import optical_carbon, photons

    from pycalima.gsd import cell
    from pycalima.gsd.hydrogen import Y_CH, k_h

    c = optical_carbon(gap=True)
    lit = replace(inputs(), photons=photons(1e-4), G0_bg=10.0, n_HI=30.0)
    eq = cell.precool([c], lit).e_gap[0]                               # untracked: the equilibrium band gap
    bh = c.hydrogen
    x = np.full(c.grid.n_bins, 0.3)
    x[c.optics.gap_bins] = [0.5 * (b.x_min + b.x_max) for b in bh]          # each bin inside its own range
    c.M_H = c.M * x
    pre = cell.precool([c], lit)
    np.testing.assert_allclose(pre.e_gap[0], [float(b.e_gap(0.5 * (b.x_min + b.x_max))) for b in bh], rtol=1e-12)
    # one short step: the frozen-rate solution
    i, b, dt = 4, bh[0], 30.0 * YR
    x0 = c.M_H[i] / c.M[i]
    ku = Y_CH * float(np.interp(b.e_gap(x0), c.optics.gap_nodes, pre.uv_gap[0][0]))
    kh = k_h(30.0, lit.T)
    x_eq = (ku * b.x_min + kh * b.x_max) / (ku + kh)
    opt = cell.Options(processes=frozenset({"hydrogen"}))
    res = cell.update([c], lit, pre, dt, opt)
    assert c.M_H[i] / c.M[i] == pytest.approx(x_eq + (x0 - x_eq) * np.exp(-(ku + kh) * dt), rel=1e-12)
    assert abs(res.conservation["H"]) < 1e-12
    for _ in range(40):                                                 # long steps: the fixed point
        cell.update([c], lit, cell.precool([c], lit), 1e9 * YR, opt)
    np.testing.assert_allclose(cell.precool([c], lit).e_gap[0], eq, rtol=1e-3)
