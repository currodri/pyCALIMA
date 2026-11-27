"""The carbon pair rule in the collisions (``pycalima.gsd.pairs``, ``GrainCollisions(pair_rule=...)``)."""
from __future__ import annotations

import numpy as np
import pytest

from pycalima.gsd.carbon import CarbonLayout
from pycalima.gsd.collisions import collide
from pycalima.gsd.materials import get_material
from pycalima.gsd.pairs import PairRule
from pycalima.gsd.shattering import GrainCollisions
from pycalima.gsd.velocities import RelativeVelocity

pytestmark = pytest.mark.gsd_quick

YR = 3.15576e7
M_H = 1.6735575e-24
LAYOUT = CarbonLayout()
GRID = LAYOUT.grid()
RULE = PairRule.from_layout(LAYOUT, GRID)


@pytest.fixture(autouse=True)
def _clean_env(isolated_env):
    yield


def state(bins, n_H, dust_to_gas=0.005):
    """Mass in the listed bins only, shared equally, each bin's mean mass at the geometric mean of its edges."""
    m = np.sqrt(GRID.m_edges[:-1] * GRID.m_edges[1:])
    M = np.zeros(GRID.n_bins)
    M[list(bins)] = dust_to_gas * 1.4 * M_H * n_H / len(bins)
    return M / m, M


def model(velocity, rule=RULE, **kw):
    return GrainCollisions(get_material("themis_ac").shatter, velocity, coagulation=True,
                           a_frag_min=float(GRID.a_edges[0]), collector="remove", table_ppd=None, threads=1,
                           pair_rule=rule, **kw)


def cold():
    """10 K, no turbulence or drift: molecules meet grains at their thermal speeds, ~0.03 km/s."""
    return RelativeVelocity(T=10.0, n_H=1e4)


def fast():
    """Grains above 1.06 nm drift at 30 km/s through the molecules (and not relative to one another)."""
    a_mol = RULE.a_mol
    return RelativeVelocity(T=100.0, n_H=1e2, drift=lambda a: np.where(np.asarray(a) > a_mol, 3e6, 0.0))


def test_the_classes_follow_the_layout():
    assert RULE.a_mol == pytest.approx(1.06e-7, rel=1e-3)
    assert RULE.a_nano == pytest.approx(19.96e-7, rel=1e-3)
    mol, nano = RULE.classes(GRID.a_centres)
    assert mol.sum() == 4 and np.array_equal(np.flatnonzero(nano), [4, 5, 6])
    mol_e, nano_e = RULE.classes(GRID.a_edges[[4, 7]])            # a grain at a boundary takes the smaller class
    assert mol_e[0] and nano_e[1]


def test_molecules_do_not_react_with_each_other():
    N, M = state([0, 1, 2, 3], 1e4)
    c = collide(GRID, N, M, model(cold()), 1e6 * YR)
    np.testing.assert_array_equal(c["N"], N)
    np.testing.assert_array_equal(c["M"], M)
    assert c["sink_M"] == 0.0
    free = collide(GRID, N, M, model(cold(), rule=None), 1e6 * YR)      # without the rule they coagulate
    assert np.abs(free["M"] - M).max() > 1e-3 * M.max()


def test_cold_gas_freezes_the_molecules_onto_the_grains():
    N, M = state([0, 1, 2, 3, 8], 1e4)
    c = collide(GRID, N, M, model(cold()), 1e6 * YR)
    hosts = M.copy()
    hosts[:4] = 0.0
    alone = collide(GRID, np.where(hosts > 0, N, 0.0), hosts, model(cold()), 1e6 * YR)   # the hosts' own evolution
    lost = M[:4].sum() - c["M"][:4].sum()
    assert lost > 1e-3 * M[:4].sum()
    assert c["sink_M"] < 1e-12 * M.sum()                                       # nothing destroyed
    assert c["M"].sum() + c["sink_M"] + c["overflow_M"] == pytest.approx(M.sum(), rel=1e-12)
    assert c["N"][8] == pytest.approx(alone["N"][8], rel=1e-9)                 # the molecules add no hosts
    assert c["M"][8] - alone["M"][8] == pytest.approx(lost, rel=1e-6)
    assert np.all(c["N"][:4] <= N[:4])


def test_fast_impacts_destroy_the_molecules_and_return_their_carbon_to_the_gas():
    N, M = state([0, 1, 2, 3, 8], 1e2)
    c = collide(GRID, N, M, model(fast()), 1e7 * YR)
    lost = M[:4].sum() - c["M"][:4].sum()
    assert lost > 1e-3 * M[:4].sum()
    assert c["sink_M"] == pytest.approx(lost, rel=1e-9)
    assert c["N"][8] == pytest.approx(N[8], rel=1e-12) and c["M"][8] == pytest.approx(M[8], rel=1e-12)


def test_release_acts_on_every_host():
    """S_eff applies to molecules on nano-grains and on grains alike (photodesorption acts on the whole population,
    2026-11-24): S = 0 on grains above 20 nm only keeps them clean and lets the nano-grains collect; S = 0 everywhere
    keeps every host clean."""
    N, M = state([1, 5, 8], 1e4)
    on = collide(GRID, N, M, model(cold()), 1e6 * YR)
    assert on["M"][5] > M[5] * (1 + 1e-9) and on["M"][8] > M[8] * (1 + 1e-9)
    grains_off = PairRule.from_layout(LAYOUT, GRID, release=lambda a_m, a_h: np.where(a_h > RULE.a_nano, 0.0, 1.0)
                                      + 0.0 * a_m)
    c = collide(GRID, N, M, model(cold(), rule=grains_off), 1e6 * YR)
    assert c["M"][8] == pytest.approx(M[8], rel=1e-12) and c["M"][5] > M[5] * (1 + 1e-9)
    off = PairRule.from_layout(LAYOUT, GRID, release=lambda a_m, a_h: 0.0 * a_m * a_h)
    c = collide(GRID, N, M, model(cold(), rule=off), 1e6 * YR)
    assert c["M"][5] == pytest.approx(M[5], rel=1e-12) and c["M"][8] == pytest.approx(M[8], rel=1e-12)


@pytest.mark.parametrize("velocity", [cold, fast])
def test_neutral_charges_change_nothing_and_the_shares_are_the_flux_fractions(velocity):
    """With every charge 0 the Coulomb sums reduce to the neutral flux fractions; stick + destroy never exceed the
    whole flux (the rest rebounds)."""
    vel = velocity()
    a = GRID.a_centres
    m = GRID.mass(a)
    base = model(vel).channels(a, m)
    vq = velocity()
    vq.charges = lambda r: (np.zeros((np.size(r), 2)), np.array([0.5, 0.5]))
    charged = model(vq).channels(a, m)
    for b, c in zip(base, charged, strict=True):
        np.testing.assert_allclose(c.K, b.K, rtol=1e-12, atol=0.0)
    mol, _ = RULE.classes(a)
    mixed = mol[:, None] ^ mol[None, :]
    dV, mu = vel.pair_statistics(a, m, get_material("themis_ac").shatter.rho)
    K_all = np.pi * (a[:, None] + a[None, :]) ** 2 * vel.collision_speeds(dV, mu, 0.0, 1)[0]
    stick, destroy = base[-2].K, base[-1].K
    assert np.all(stick[mixed] + destroy[mixed] <= K_all[mixed] * (1 + 1e-12))


def test_the_rule_needs_a_relative_velocity_and_the_remove_collector():
    with pytest.raises(ValueError, match="collector"):
        GrainCollisions(get_material("themis_ac").shatter, cold(), collector="smallest", pair_rule=RULE)
    with pytest.raises(ValueError, match="RelativeVelocity"):
        GrainCollisions(get_material("themis_ac").shatter, lambda a1, a2: [(1.0, 1e5)], pair_rule=RULE)
