"""The per-cell GSD operator (``pycalima.gsd.cell``)."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from pycalima.gsd import cell
from pycalima.gsd.carbon import CarbonLayout
from pycalima.gsd.collisions import collide
from pycalima.gsd.exchange import Population, exchange_step
from pycalima.gsd.materials import get_material
from pycalima.gsd.shattering import GrainCollisions
from pycalima.gsd.sinks import deposit_power_law
from pycalima.gsd.sputtering import sputter
from pycalima.gsd.velocities import RelativeVelocity

pytestmark = pytest.mark.gsd_quick

YR = 3.15576e7
PC = 3.0857e18
AMU = 1.66053906660e-24
PER_GRAM_C = 1.0 / (12.011 * AMU)


@pytest.fixture(autouse=True)
def _clean_env(isolated_env):
    yield


def carbon(overflow=False, n_H=1.0):
    from pycalima.models.dust_radiation.sublimation_kinetics import get_sublimation_model

    lay = CarbonLayout(overflow_a_max_cm=1e-3 if overflow else None)
    g = lay.grid()
    a = g.a_centres
    m = np.sqrt(g.m_edges[:-1] * g.m_edges[1:])
    N = a ** -2.5 * (a < 1e-4)
    N *= 0.005 * 1.4 * 1.6735575e-24 * n_H / np.sum(N * m)
    sub = replace(get_sublimation_model("graphite", "GD89"), alpha_gd89=0.3, density_gcm3=g.rho)
    return cell.Composition(g, N, N * m, "C", PER_GRAM_C, get_material("themis_ac").shatter, sputtering="graphite",
                            sublimation=sub)


def inputs(n_H=30.0, T=100.0, gas_C=None):
    return cell.CellInputs(T=T, n_H=n_H, n_e=0.03 * n_H, gas={"C": 1e-4 * n_H if gas_C is None else gas_C},
                           T_d=20.0, v_turb=2.6e5, L=10 * PC)


def test_update_is_the_operators_in_their_order():
    """collide, then sputter, then the gas exchange, with the same models: the operator adds nothing of its own."""
    c, inp, dt = carbon(n_H=30.0), inputs(), 1e5 * YR
    ref = carbon(n_H=30.0)
    opt = cell.Options(table_ppd=None)
    res = cell.update([c], inp, None, dt, opt)
    g = ref.grid
    rv = RelativeVelocity(T=inp.T, n_H=inp.n_H, mu=1.4, v_turb=inp.v_turb, L=inp.L)
    model = GrainCollisions(ref.material, rv, law="KT10", coagulation=True, a_frag_min=float(g.a_edges[0]),
                            collector="remove", n_speed=4, table_ppd=None, threads=1)
    r = collide(g, ref.N, ref.M, model, dt, errmax=1e-3)
    gas = inp.gas["C"] + r["sink_M"] * PER_GRAM_C
    N, M, dM = sputter(g, r["N"], r["M"], ref.erosion(inp.T, inp.n_H), dt)
    gas += dM * PER_GRAM_C
    pop = Population(ref.sublimation, g, N, M, np.full(g.n_bins, 20.0), nucleation=False)
    gk = {"C": gas}
    exchange_step([pop], gk, inp.T, dt)
    np.testing.assert_array_equal(c.N, pop.N)
    np.testing.assert_array_equal(c.M, pop.M)
    assert res.gas["C"] == gk["C"]


def test_every_element_is_conserved():
    c = carbon(n_H=1.0)
    inp = inputs(n_H=1.0, T=1e6, gas_C=1e-4)
    res = cell.update([c], replace(inp, v_turb=1e6, L=100 * PC), None, 1e6 * YR, cell.Options(table_ppd=None))
    assert abs(res.conservation["C"]) < 1e-12
    assert res.duu > 0.0


def test_the_passive_overflow_bin_takes_what_grows_past_the_grid():
    """Accretion pushes grains past 1 um: with the overflow bin passive the grid below runs alone and the bin
    collects the grown grains; carbon is conserved, the share is reported, and strict mode raises past 1 %."""
    c = carbon(overflow=True, n_H=1e4)
    c.N[-1] = c.M[-1] = 0.0
    inp = cell.CellInputs(T=10.0, n_H=1e4, n_e=1e-4, gas={"C": 1e-4 * 1e4}, T_d=10.0)
    opt = cell.Options(processes=frozenset({"accretion"}))
    res = cell.update([c], inp, None, 3e7 * YR, opt)
    assert c.M[-1] > 0.0 and res.overflow_share[0] == pytest.approx(c.M[-1] / c.M.sum())
    assert abs(res.conservation["C"]) < 1e-12
    c.M[-1] = 0.5 * c.M.sum()
    c.N[-1] = c.M[-1] / c.grid.mass(3e-4)
    with pytest.raises(RuntimeError, match="overflow"):
        cell.update([c], inp, None, 1e3 * YR, cell.Options(processes=frozenset({"accretion"}), strict=True))


def test_coulomb_explosion_feeds_fragments_or_the_gas():
    """Explosion empties the exploding bins per moment; 'fragments' deposits their mass down the grid as a^-3.3
    (the part below the grid to the gas), 'gas' returns all of it; carbon is conserved both ways."""
    for route in ("fragments", "gas"):
        c = carbon(n_H=1.0)
        g = c.grid
        k = np.zeros(g.n_bins)
        k[8] = 1e-10                                            # 53-141 nm grains explode
        state = type("S", (), {"a": g.a_centres, "charge_points": None})()
        pre = cell.Precool([0.0], [0.0], [state], [None], [k], [0.0])
        M8 = c.M[8]
        res = cell.update([c], inputs(n_H=1.0, gas_C=1e-4), pre, 1e3 * YR,
                          cell.Options(processes=frozenset({"explosion"}), explosion_route=route))
        assert c.M[8] < M8 * 0.99
        assert abs(res.conservation["C"]) < 1e-12
        if route == "gas":
            assert res.gas["C"] - 1e-4 == pytest.approx((M8 - c.M[8]) * PER_GRAM_C, rel=1e-9)


def test_power_law_deposition_is_exact():
    g = CarbonLayout().grid()
    dN, dM, below, above = deposit_power_law(g, 1.0, 2e-8, 3e-6, 3.3)
    assert dM.sum() + below + above == pytest.approx(1.0, rel=1e-12)
    assert below > 0.0 and above == 0.0
    inside = (dN > 0)
    mbar = dM[inside] / dN[inside]
    assert np.all(mbar >= g.m_edges[:-1][inside] * (1 - 1e-12)) and np.all(mbar <= g.m_edges[1:][inside] * (1 + 1e-12))
    assert deposit_power_law(g, 1.0, 1e-8, 3e-8, 3.3)[2] == 1.0          # all below the grid


def test_precool_is_the_area_node_solve_weighted_by_the_bins():
    """Spike bins (all grains at the lower edge): heating and cooling are N times the single-grain rates there, the
    area pi N a_L^2; neutral compositions give none."""
    from pycalima.gsd.charging import AreaNodeCharging, ChargingModel, PhotonGroups
    from pycalima.gsd.grid import MassGrid

    g = MassGrid.from_radii(1e-7, 1e-6, 2.0 / 3.0, 2.2)
    N = np.array([1e-6, 2e-7])
    comp = cell.Composition(g, N.copy(), N * g.m_edges[:-1], "C", PER_GRAM_C, get_material("themis_ac").shatter,
                            charging=AreaNodeCharging("graphite", g, radii_per_bin=2))
    L0, L1 = comp.charging.L0, comp.charging.L1
    ph = PhotonGroups(L0, L1, np.zeros(L0.size), np.sqrt(L0 * L1))
    inp = cell.CellInputs(T=100.0, n_H=30.0, n_e=0.03, gas={"C": 3e-3}, photons=ph, G0_bg=1.0)
    pre = cell.precool([comp, carbon()], inp)
    ref = ChargingModel("graphite", g.a_edges[:-1]).solve(ph, T=100.0, n_e=0.03, G0_bg=1.0)
    lam = np.array([r["Lambda"] + r["Lambda_auto"] for r in ref.results])
    np.testing.assert_allclose(pre.heating[0], N * ref.Gamma, rtol=1e-9)
    np.testing.assert_allclose(pre.cooling[0], N * lam, rtol=1e-9)
    np.testing.assert_allclose(pre.area[0], np.pi * N * g.a_edges[:-1] ** 2, rtol=1e-12)
    assert pre.explosion[0].shape == (2,) and pre.total_heating == pytest.approx(float(np.sum(N * ref.Gamma)))
    assert pre.states[1] is None and np.all(pre.heating[1] == 0.0) and pre.explosion[1] is None


def test_molecular_bins_skip_the_grain_processes():
    """The 4 molecular bins take no accretion and no grain explosion, and classical sputtering stops at N_C,max: grains
    eroded from above join the molecular bins and nothing in them erodes; explosion fragments of larger grains may
    land in them. Carbon is conserved each way."""
    for procs in ({"accretion"}, {"sputtering"}, {"explosion"}):
        c = carbon(n_H=1.0)
        c.molecular_bins = 4
        M0, N0 = c.M.copy(), c.N.copy()
        g = c.grid
        k = np.full(g.n_bins, 1e-10)
        state = type("S", (), {"a": g.a_centres, "charge_points": None, "ion_focusing_of": None})()
        pre = cell.Precool([0.0], [0.0], [state], [None], [k], [0.0])
        T = 1e6 if procs == {"sputtering"} else 100.0
        res = cell.update([c], inputs(n_H=1.0, T=T, gas_C=1e-4), pre, 1e5 * YR, cell.Options(processes=frozenset(procs)))
        assert abs(res.conservation["C"]) < 1e-12
        if procs == {"accretion"}:
            np.testing.assert_array_equal(c.M[:4], M0[:4])
            np.testing.assert_array_equal(c.N[:4], N0[:4])
        else:                                                  # they only gain (eroded grains, fragments)
            assert np.all(c.M[:4] >= M0[:4] * (1 - 1e-12)) and c.M[4:].sum() < M0[4:].sum()
