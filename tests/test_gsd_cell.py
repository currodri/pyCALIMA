"""The per-cell GSD operator (gsd.cell): area nodes, area-node charging, and precool/update against the modules they
call (the oracle), with conservation and the switches."""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.gsd_quick

M_H = 1.6735575e-24
M_C = 12.011 * 1.66053906660e-24
YR = 3.15576e7
PC = 3.0857e18


def _carbon_state(grid, n_H, dust_to_gas=0.005):
    """A realisable MRN-like state: N ∝ a^-2.5 per bin, each bin's mean mass at the geometric mean of its edges."""
    a = grid.a_centres
    m = np.sqrt(grid.m_edges[:-1] * grid.m_edges[1:])
    N = a ** -2.5
    N *= dust_to_gas * 1.4 * M_H * n_H / np.sum(N * m)
    return N, N * m


@pytest.fixture(scope="module")
def carbon():
    from pycalima.gsd.carbon import CarbonLayout

    grid = CarbonLayout().grid()
    N, M = _carbon_state(grid, 30.0)
    return grid, N, M


def test_area_node_is_exact_for_area_and_volume(carbon):
    from pycalima.gsd.closures import area_node, pieces, radius_moments

    grid, N, M = carbon
    N, M = N.copy(), M.copy()
    N[3], M[3] = 0.0, 0.0
    a_star, w = area_node(grid, N, M)
    P = pieces(grid, N, M)
    filled = N > 0
    np.testing.assert_allclose((w * a_star ** 2)[filled], radius_moments(P, 2)[filled], rtol=1e-12)
    np.testing.assert_allclose((w * a_star ** 3)[filled], radius_moments(P, 3)[filled], rtol=1e-12)
    a_e = grid.a_edges
    assert np.all((a_star >= a_e[:-1]) & (a_star <= a_e[1:]))
    assert w[3] == 0.0 and a_star[3] == grid.a_centres[3]


@pytest.fixture(scope="module")
def silicate_charging():
    """Area-node charging of silicate on 3 bins, 2 table radii each (the bin edges; cached on disk)."""
    from pycalima.gsd.charging import AreaNodeCharging
    from pycalima.gsd.grid import MassGrid
    from pycalima.models.dust_charge import rt_group_charging as rg

    grid = MassGrid.from_radii(3e-7, 3e-5, 1.0, 3.3)
    groups = (rg.DEFAULT_GROUP_L0, rg.DEFAULT_GROUP_L1)
    return grid, AreaNodeCharging("silicate", grid, radii_per_bin=2, groups=groups)


def _isrf_photons(L0, L1, G0=1.0):
    from pycalima.gsd.charging import PhotonGroups

    ev, kb = 1.602176634e-12, 8.617333262e-5
    E = np.geomspace(0.1, 500.0, 6000)
    uE = E ** 3 / np.expm1(E / (kb * 3e4))
    b = (E >= 6) & (E <= 13.6)
    uE *= G0 * 5.33e-14 / np.trapezoid(uE[b], E[b])
    nE = uE / (E * ev)                                  # photons cm^-3 eV^-1
    return PhotonGroups.from_photon_density(E, nE, np.asarray(L0, float), np.asarray(L1, float))


def test_blend_table_returns_the_node_table_at_a_node(silicate_charging):
    from pycalima.models.dust_charge import rt_group_charging as rg

    _, ch = silicate_charging
    for i, radii in enumerate(ch.node_radii):
        for j, r in enumerate(radii):
            t = ch.table_at(i, r)
            ref = ch.node_tables[i][j]
            for k in rg.TABULATED_KEYS:
                np.testing.assert_allclose(t[k], ref[k], rtol=1e-12, atol=0.0, err_msg=k)
            assert t["Zmin"] == ref["Zmin"] and t["Zmax"] == ref["Zmax"]


def test_area_node_charge_at_a_node_radius_is_the_charging_models(silicate_charging):
    """At a table radius the blended table is that radius's own, so its solve is ``ChargingModel``'s there."""
    from pycalima.gsd.charging import ChargingModel
    from pycalima.models.dust_charge import rt_group_charging as rg

    _, ch = silicate_charging
    r = float(ch.node_radii[1][0])
    ph = _isrf_photons(ch.L0, ch.L1)
    ref = ChargingModel("silicate", [r], groups=(ch.L0, ch.L1)).solve(ph, 100.0, 1e-2)
    gt = rg.GroupChargingTable(ch.table_at(1, r), egy=ph.egy)
    got = gt.solve(ph.N, 2.99792458e10, 100.0, 1e-2, [(1e-2, 1.0, 1.67262192e-24)], recomb=None,
                   lattice=ch.lattice)
    np.testing.assert_allclose([got["Zmean"], got["Gamma"]], [ref.Zmean[0], ref.Gamma[0]], rtol=1e-10)


def test_precool_is_linear_in_the_dust_and_keeps_the_charges(silicate_charging):
    from pycalima.gsd.cell import CellInputs, Composition, precool

    grid, ch = silicate_charging
    N, M = _carbon_state(grid, 1.0)
    inp = CellInputs(T=100.0, n_H=1.0, n_e=1e-2, gas={"Si": 0.0}, photons=_isrf_photons(ch.L0, ch.L1))
    p1 = precool([Composition(grid, N, M, "Si", 1.0, None, charging=ch)], inp)
    p3 = precool([Composition(grid, 3 * N, 3 * M, "Si", 1.0, None, charging=ch)], inp)
    assert np.all(p1.heating[0] > 0) and np.all(p1.cooling[0] > 0)
    np.testing.assert_allclose(p3.heating[0], 3 * p1.heating[0], rtol=1e-12)
    np.testing.assert_allclose(p3.cooling[0], 3 * p1.cooling[0], rtol=1e-12)
    np.testing.assert_allclose(p3.states[0].Zmean, p1.states[0].Zmean, rtol=1e-12)
    # each state keeps its own cell: a later solve does not change an earlier state's tables
    z_before = p1.states[0].ion_focusing()
    precool([Composition(grid, N, M, "Si", 1.0, None, charging=ch)],
            CellInputs(T=8e3, n_H=1.0, n_e=1.0, gas={"Si": 0.0}, photons=_isrf_photons(ch.L0, ch.L1, 1e3)))
    np.testing.assert_allclose(p1.states[0].ion_focusing(), z_before, rtol=1e-12)


def test_precool_neutral_grains(carbon):
    from pycalima.gsd.cell import CellInputs, Composition, precool
    from pycalima.gsd.closures import pieces, radius_moments

    grid, N, M = carbon
    p = precool([Composition(grid, N, M, "C", 1.0 / M_C, None)], CellInputs(100.0, 30.0, 1e-3, {"C": 1.0}))
    assert p.states[0] is None and p.total_heating == 0.0 and p.total_cooling == 0.0
    np.testing.assert_allclose(p.area[0], np.pi * radius_moments(pieces(grid, N, M), 2), rtol=1e-12)


def _carbon_composition(grid, N, M):
    from pycalima.gsd.cell import Composition
    from pycalima.gsd.materials import get_material
    from pycalima.models.dust_radiation.sublimation_kinetics import get_sublimation_model

    return Composition(grid, N.copy(), M.copy(), "C", 1.0 / M_C, get_material("themis_ac").shatter,
                       sputtering="graphite", sublimation=get_sublimation_model("graphite", "GD89"))


def test_update_equals_the_modules_in_order(carbon):
    """The oracle: collide, then sputter, then exchange_step, called directly, give update's result to round-off."""
    from pycalima.gsd.cell import CellInputs, update
    from pycalima.gsd.collisions import collide
    from pycalima.gsd.exchange import Population, exchange_step
    from pycalima.gsd.materials import get_material
    from pycalima.gsd.shattering import GrainCollisions
    from pycalima.gsd.sputtering import erosion_rate, sputter
    from pycalima.gsd.velocities import RelativeVelocity
    from pycalima.models.dust_radiation.sublimation_kinetics import get_sublimation_model

    grid, N, M = carbon
    T, n_H, dt = 3e4, 30.0, 1e6 * YR
    gas0 = {"C": 1e-4 * n_H}
    c = _carbon_composition(grid, N, M)
    inp = CellInputs(T=T, n_H=n_H, n_e=1e-2, gas=gas0, v_turb=2.6e5, L=10.0 * PC)
    res = update([c], inp, None, dt)

    rv = RelativeVelocity(T=T, n_H=n_H, mu=1.4, v_turb=2.6e5, L=10.0 * PC)
    model = GrainCollisions(get_material("themis_ac").shatter, rv, law="KT10", alpha_f=3.3, coagulation=True,
                            a_frag_min=float(grid.a_edges[0]), collector="remove", n_speed=4, table_ppd=24, threads=1)
    r = collide(grid, N.copy(), M.copy(), model, dt, errmax=1e-3)
    gas = {"C": gas0["C"] + r["sink_M"] / M_C}
    N1, M1, dM = sputter(grid, r["N"], r["M"], erosion_rate("graphite", T, n_H, rho=grid.rho), dt)
    gas["C"] += dM / M_C
    pop = Population(get_sublimation_model("graphite", "GD89"), grid, N1.copy(), M1.copy(), np.full(grid.n_bins, 20.0),
                     nucleation=False)
    exchange_step([pop], gas, T, dt)

    np.testing.assert_allclose(c.N, pop.N, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(c.M, pop.M, rtol=1e-12, atol=0.0)
    assert res.gas["C"] == pytest.approx(gas["C"], rel=1e-12)
    assert c.overflow_M == pytest.approx(r["overflow_M"] + pop.overflow_M, rel=1e-12, abs=1e-300)
    assert abs(res.conservation["C"]) < 1e-12
    assert res.duu > 0.0


def test_update_with_every_process_off_changes_nothing(carbon):
    from pycalima.gsd.cell import CellInputs, Options, update

    grid, N, M = carbon
    c = _carbon_composition(grid, N, M)
    res = update([c], CellInputs(1e4, 30.0, 1e-2, {"C": 1.0}), None, 1e6 * YR, Options(processes=frozenset()))
    assert np.array_equal(c.N, N) and np.array_equal(c.M, M)
    assert res.gas == {"C": 1.0} and res.duu == 0.0


def test_update_rejects_unknown_processes(carbon):
    from pycalima.gsd.cell import CellInputs, Options, update

    grid, N, M = carbon
    with pytest.raises(ValueError, match="unknown processes"):
        update([_carbon_composition(grid, N, M)], CellInputs(1e4, 30.0, 1e-2, {"C": 1.0}), None, 1.0,
               Options(processes=frozenset({"melting"})))
