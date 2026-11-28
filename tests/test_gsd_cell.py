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


def _silicate_composition(grid, charging, n_H, refine=True):
    from pycalima.gsd.cell import Composition
    from pycalima.gsd.materials import get_material

    N, M = _carbon_state(grid, n_H)
    return Composition(grid, N, M, "Si", 1.0, get_material("silicate").shatter, charging=charging,
                       explosion_refine=refine)


def test_the_explosion_samples_are_the_area_nodes_without_refinement(silicate_charging):
    """explosion_refine off: precool's explosion rate is the area nodes' alone (the behaviour before 2026-11-28)."""
    from pycalima.gsd.cell import CellInputs, precool

    grid, ch = silicate_charging
    inp = CellInputs(T=600.0, n_H=5e4, n_e=7.0, gas={"Si": 1.0}, photons=_isrf_photons(ch.L0, ch.L1, 2e4))
    c = _silicate_composition(grid, ch, 5e4, refine=False)
    pre = precool([c], inp)
    a, k = pre.explosion[0]
    np.testing.assert_array_equal(a, pre.states[0].a)
    np.testing.assert_array_equal(k, pre.states[0].explosion_rate(S_max=c.s_max))


def test_the_explosion_is_also_sampled_inside_the_bins_that_explode(silicate_charging, monkeypatch):
    """explosion_refine on. With a step-like rate (a stand-in for ChargeState.explosion_rate: 1e-10 s^-1 below 5 nm,
    0 above), the area nodes keep their rates, and each bin whose area node explodes, and the next bin up, adds its
    table radii, so the sink integrates the step over the bin. With no explosion anywhere, only the lowest bin's lower
    edge is added (one solve), at a zero rate."""
    from pycalima.gsd import charging as CH
    from pycalima.gsd.cell import K_EXPLODE_MIN, CellInputs, precool

    grid, ch = silicate_charging
    inp = CellInputs(T=600.0, n_H=5e4, n_e=7.0, gas={"Si": 1.0}, photons=_isrf_photons(ch.L0, ch.L1, 2e4))
    a_cut = 5e-7
    monkeypatch.setattr(CH.ChargeState, "explosion_rate",
                        lambda self, S_max=1e10, **kw: np.where(np.asarray(self.a) < a_cut, 1e-10, 0.0))
    pre = precool([_silicate_composition(grid, ch, 5e4)], inp)
    a, k = pre.explosion[0]
    st = pre.states[0]
    k_node = np.where(st.a < a_cut, 1e-10, 0.0)
    for ai, ki in zip(st.a, k_node, strict=True):
        assert k[np.searchsorted(a, ai)] == ki
    hot = np.flatnonzero(k_node > K_EXPLODE_MIN)
    if hot.size:
        for i in set(hot.tolist()) | {min(int(hot.max()) + 1, grid.n_bins - 1)}:
            assert np.all(np.isin(ch.node_radii[i], a))
    else:                                   # the step lies below every node: the lowest bin's edge finds it
        assert np.all(np.isin(ch.node_radii[0], a))
    assert np.all(np.diff(a) > 0)

    monkeypatch.setattr(CH.ChargeState, "explosion_rate", lambda self, S_max=1e10, **kw: np.zeros(np.size(self.a)))
    pre = precool([_silicate_composition(grid, ch, 5e4)], inp)
    a, k = pre.explosion[0]
    np.testing.assert_array_equal(np.setdiff1d(a, pre.states[0].a), grid.a_edges[:1])
    assert np.all(k == 0.0)


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

    mat = get_material("themis_ac")
    return Composition(grid, N.copy(), M.copy(), "C", 1.0 / M_C, mat.shatter, sputtering="graphite",
                       sublimation=mat.sublimation_model())


def test_update_equals_the_modules_in_order(carbon):
    """The oracle: collide, then sputter, then exchange_step, called directly, give update's result to round-off."""
    from pycalima.gsd.cell import CellInputs, update
    from pycalima.gsd.collisions import collide
    from pycalima.gsd.exchange import Population, exchange_step
    from pycalima.gsd.materials import get_material
    from pycalima.gsd.shattering import GrainCollisions
    from pycalima.gsd.sputtering import erosion_rate, sputter
    from pycalima.gsd.velocities import RelativeVelocity

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
    pop = Population(get_material("themis_ac").sublimation_model(), grid, N1.copy(), M1.copy(),
                     np.full(grid.n_bins, 20.0), nucleation=False)
    exchange_step([pop], gas, T, dt)

    np.testing.assert_allclose(c.N, pop.N, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(c.M, pop.M, rtol=1e-12, atol=0.0)
    assert res.gas["C"] == pytest.approx(gas["C"], rel=1e-12)
    assert c.overflow_M == pytest.approx(r["overflow_M"] + pop.overflow_M, rel=1e-12, abs=1e-300)
    assert abs(res.conservation["C"]) < 1e-14
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


def _precool_with_explosion(grid, N, M, k_of_a):
    """A precool by hand: neutral grains with a given explosion rate at the area nodes."""
    from pycalima.gsd.cell import Precool
    from pycalima.gsd.closures import area_node

    a_star, w = area_node(grid, N, M)
    z = np.zeros(grid.n_bins)
    return Precool([z], [z], [None], [w], [np.pi * w * a_star ** 2], [(a_star, k_of_a(a_star))])


def test_explosion_at_a_constant_rate_removes_both_moments_exactly(carbon):
    from pycalima.gsd.cell import CellInputs, Options, update

    grid, N, M = carbon
    k, dt = 2e-14, 3e13
    pre = _precool_with_explosion(grid, N, M, lambda a: np.full(np.shape(a), k))
    c = _carbon_composition(grid, N, M)
    res = update([c], CellInputs(100.0, 30.0, 1e-2, {"C": 1.0}), pre, dt,
                 Options(processes=frozenset({"explosion"}), explosion_route="gas"))
    np.testing.assert_allclose(c.N, N * np.exp(-k * dt), rtol=1e-12)
    np.testing.assert_allclose(c.M, M * np.exp(-k * dt), rtol=1e-12)
    assert res.gas["C"] == pytest.approx(1.0 + M.sum() * (1 - np.exp(-k * dt)) / M_C, rel=1e-12)
    assert abs(res.conservation["C"]) < 1e-14


def test_explosion_to_fragments_conserves_carbon_and_keeps_the_bins_realisable(carbon):
    """A steep rate (large grains explode): the fragments stay on the grid, carbon is conserved, every bin keeps a
    mean mass inside its edges, and the mass moves to smaller grains."""
    from pycalima.gsd.cell import CellInputs, Options, update

    grid, N, M = carbon
    pre = _precool_with_explosion(grid, N, M, lambda a: 1e-13 * (np.asarray(a) / 1e-5) ** 4)
    c = _carbon_composition(grid, N, M)
    res = update([c], CellInputs(100.0, 30.0, 1e-2, {"C": 1.0}), pre, 1e13, Options(processes=frozenset({"explosion"})))
    assert abs(res.conservation["C"]) < 1e-14
    assert res.gas["C"] == pytest.approx(1.0, rel=1e-12)          # fragments from the grid's lowest edge up
    filled = c.N > 0
    mbar = c.M[filled] / c.N[filled]
    assert np.all(mbar >= grid.m_edges[:-1][filled] * (1 - 1e-12)) and np.all(mbar <= grid.m_edges[1:][filled] * (1 + 1e-12))
    a = grid.a_centres
    assert np.sum(c.M * a) / c.M.sum() < np.sum(M * a) / M.sum()
    assert c.M[-1] < M[-1] and c.M[0] > M[0]


def test_explosion_route_must_be_known(carbon):
    from pycalima.gsd.cell import CellInputs, Options, update

    grid, N, M = carbon
    pre = _precool_with_explosion(grid, N, M, lambda a: np.full(np.shape(a), 1e-14))
    with pytest.raises(ValueError, match="explosion_route"):
        update([_carbon_composition(grid, N, M)], CellInputs(100.0, 30.0, 1e-2, {"C": 1.0}), pre, 1e13,
               Options(processes=frozenset({"explosion"}), explosion_route="ions"))


def test_update_with_the_pair_rule_equals_collide_with_it():
    """Carbon's pair rule through the cell: in cold dense gas the molecules freeze onto the grains, as collide gives
    with the same rule, and carbon is conserved."""
    from pycalima.gsd.carbon import CarbonLayout
    from pycalima.gsd.cell import CellInputs, Options, update
    from pycalima.gsd.collisions import collide
    from pycalima.gsd.materials import get_material
    from pycalima.gsd.pairs import PairRule
    from pycalima.gsd.shattering import GrainCollisions
    from pycalima.gsd.velocities import RelativeVelocity

    layout = CarbonLayout()
    grid = layout.grid()
    N, M = _carbon_state(grid, 1e4)
    rule = PairRule.from_layout(layout, grid)
    c = _carbon_composition(grid, N, M)
    c.pair_rule = rule
    dt = 1e5 * YR
    res = update([c], CellInputs(10.0, 1e4, 1e-4, {"C": 1.0}), None, dt,
                 Options(processes=frozenset({"coagulation", "shattering"})))
    model = GrainCollisions(get_material("themis_ac").shatter, RelativeVelocity(T=10.0, n_H=1e4, mu=1.4),
                            coagulation=True, a_frag_min=float(grid.a_edges[0]), collector="remove", threads=1,
                            pair_rule=rule)
    r = collide(grid, N.copy(), M.copy(), model, dt, errmax=1e-3)
    np.testing.assert_allclose(c.N, r["N"], rtol=1e-12)
    np.testing.assert_allclose(c.M, r["M"], rtol=1e-12)
    assert c.M[:4].sum() < M[:4].sum() * (1 - 1e-3)                # the molecules freeze out
    assert abs(res.conservation["C"]) < 1e-14


def _accreting_inputs():
    """Cold dense gas with all the carbon available: the largest grains grow past 1 um within the step."""
    from pycalima.gsd.cell import CellInputs

    return CellInputs(T=20.0, n_H=1e4, n_e=1e-4, gas={"C": 3e-4 * 1e4})


def test_a_passive_overflow_bin_is_the_grid_below_plus_what_grew_past_it(carbon):
    """While the overflow bin is (nearly) empty, update runs on the grid below it: bit-identical to a run without the
    bin, whose top-of-grid collector holds exactly what the passive bin receives."""
    from pycalima.gsd.cell import Options, update

    grid, N, M = carbon
    opt = Options(processes=frozenset({"accretion"}))
    plain = _carbon_composition(grid, N, M)
    r0 = update([plain], _accreting_inputs(), None, 3e7 * YR, opt)
    go = grid.with_overflow(1e-3)
    over = _carbon_composition(go, np.append(N, 0.0), np.append(M, 0.0))
    r1 = update([over], _accreting_inputs(), None, 3e7 * YR, opt)
    assert plain.overflow_M > 0.0                                   # grains did grow past 1 um
    assert np.array_equal(over.N[:-1], plain.N) and np.array_equal(over.M[:-1], plain.M)
    assert over.M[-1] == plain.overflow_M and over.overflow_M == 0.0
    assert r1.gas == r0.gas and abs(r1.conservation["C"]) < 1e-14
    assert r1.overflow_share[0] == pytest.approx(over.M[-1] / over.M.sum()) and r0.overflow_share == [0.0]


def test_an_active_overflow_bin_is_an_ordinary_bin(carbon):
    """Above the gate the bin is computed like any other: update equals a run on the same edges with no overflow
    flag."""
    from pycalima.gsd.cell import Options, update
    from pycalima.gsd.grid import MassGrid

    grid, N, M = carbon
    go = grid.with_overflow(1e-3)
    N1, M1 = np.append(N, 1e-3 * N[-1]), np.append(M, 1e-3 * N[-1] * go.m_edges[-2] * 2.0)
    opt = Options(processes=frozenset({"accretion"}))
    a = _carbon_composition(go, N1, M1)
    ref = _carbon_composition(MassGrid(go.m_edges, go.rho), N1, M1)
    assert M1[-1] > opt.overflow_gate * M1.sum()                    # active from the start
    ra = update([a], _accreting_inputs(), None, 3e7 * YR, opt)
    rr = update([ref], _accreting_inputs(), None, 3e7 * YR, opt)
    assert np.array_equal(a.N, ref.N) and np.array_equal(a.M, ref.M) and ra.gas == rr.gas
    assert a.overflow_M == ref.overflow_M and ra.overflow_share[0] == pytest.approx(a.M[-1] / a.M.sum())


def test_strict_mode_raises_when_the_overflow_bin_fills_up(carbon):
    from pycalima.gsd.cell import Options, update

    grid, N, M = carbon
    go = grid.with_overflow(1e-3)
    c = _carbon_composition(go, np.append(N, N[-1]), np.append(M, M[-1] * 5.0))
    with pytest.raises(RuntimeError, match="overflow bin holds"):
        update([c], _accreting_inputs(), None, 1e3 * YR, Options(processes=frozenset({"accretion"}), strict=True))


@pytest.mark.parametrize("route", ["fragments", "gas"])
def test_a_vanishing_explosion_rate_leaves_the_bins_finite_and_conserves(carbon, route):
    """k dt down to 1e-30: the surviving number fraction rounds to 1 before the mass fraction does. That mass without
    number stays in its bin; nothing becomes NaN, and carbon is conserved."""
    from pycalima.gsd.cell import CellInputs, Options, update

    grid, N, M = carbon
    pre = _precool_with_explosion(grid, N, M, lambda a: np.geomspace(1e-60, 1e-20, np.size(a)))
    c = _carbon_composition(grid, N, M)
    res = update([c], CellInputs(100.0, 30.0, 1e-2, {"C": 1.0}), pre, 1e10,
                 Options(processes=frozenset({"explosion"}), explosion_route=route))
    assert np.all(np.isfinite(c.N)) and np.all(np.isfinite(c.M))
    assert abs(res.conservation["C"]) < 1e-14
    np.testing.assert_allclose(c.M, M, rtol=1e-9)


def test_a_sublimation_model_of_another_density_is_refused(carbon):
    """graphite's GD89 model (2.24 g/cm3) on THEMIS a-C's grid (1.6) would scale the uptake rate by the density ratio."""
    from pycalima.gsd.cell import CellInputs, Options, update
    from pycalima.models.dust_radiation.sublimation_kinetics import get_sublimation_model

    grid, N, M = carbon
    c = _carbon_composition(grid, N, M)
    c.sublimation = get_sublimation_model("graphite", "GD89")
    with pytest.raises(ValueError, match="density"):
        update([c], CellInputs(20.0, 1e4, 1e-4, {"C": 1.0}), None, 1e10, Options(processes=frozenset({"accretion"})))


def test_molecular_bins_skip_the_grain_explosion_and_accretion(carbon):
    """With molecular_bins = 4 (CarbonLayout's PAH bins), a constant explosion rate and accretion both leave bins 0-3
    exactly as they were: a molecule's charge and fission come from the PAH operator (R1), and molecular growth is off.
    The grain bins above still explode and grow, and carbon is conserved."""
    from pycalima.gsd.cell import CellInputs, Options, update

    grid, N, M = carbon
    pre = _precool_with_explosion(grid, N, M, lambda a: np.full(np.shape(a), 2e-14))
    c = _carbon_composition(grid, N, M)
    c.molecular_bins = 4
    res = update([c], CellInputs(100.0, 30.0, 1e-2, {"C": 1.0}), pre, 3e13,
                 Options(processes=frozenset({"explosion"}), explosion_route="gas"))
    assert np.array_equal(c.N[:4], N[:4]) and np.array_equal(c.M[:4], M[:4])
    np.testing.assert_allclose(c.M[4:], M[4:] * np.exp(-2e-14 * 3e13), rtol=1e-12)
    assert abs(res.conservation["C"]) < 1e-14
    c = _carbon_composition(grid, N, M)
    c.molecular_bins = 4
    res = update([c], _accreting_inputs(), None, 3e7 * YR, Options(processes=frozenset({"accretion"})))
    assert np.array_equal(c.N[:4], N[:4]) and np.array_equal(c.M[:4], M[:4])
    assert c.M[4:].sum() > M[4:].sum() * (1 + 1e-3) and abs(res.conservation["C"]) < 1e-14
