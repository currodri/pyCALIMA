"""The cell's pseudo-bin optics (``pycalima.gsd.optics.PseudoBinOptics``) on analytic per-gram absorption."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pycalima.gsd.charging import AGN_GROUP_L0, AGN_GROUP_L1, PhotonGroups
from pycalima.gsd.grid import MassGrid
from pycalima.gsd.optics import HC_EV_UM, SIGMA_SB, PseudoBinOptics

pytestmark = pytest.mark.gsd_quick

GROUPS = (np.array(AGN_GROUP_L0), np.array(AGN_GROUP_L1))
C = 2.99792458e10
EV = 1.602176634e-12


def _jones_root():
    from pycalima._datasets import get_dataset

    found = get_dataset("optical-jones-2012-size").locate()
    return None if found is None else Path(found).parents[1]


_JONES_ROOT = _jones_root()


@pytest.fixture(autouse=True)
def _clean_env(isolated_env):
    yield


@pytest.fixture
def jones(monkeypatch):
    """The Jones (2012) a-C(:H) tables (``hydrogen.BinHydrogen``'s optics) inside isolated_env, or skip."""
    from pycalima._datasets import get_dataset

    if _JONES_ROOT is not None:
        monkeypatch.setenv("CALIMA_DATASETS", str(_JONES_ROOT))
    if get_dataset("optical-jones-2012-size").locate() is None:
        pytest.skip("Jones (2012) tables not fetched; run `calima-fetch-data fetch optical-jones-2012-size`")


def grid():
    m = 4.0 / 3.0 * np.pi * 1.6 * np.array([1e-7, 1e-4]) ** 3
    return MassGrid(np.geomspace(m[0], m[1], 7), 1.6)


def flat(k0=1e4):
    return lambda a, lam: np.full((np.size(a), np.size(lam)), k0)


def photons(N=1e-3):
    L0, L1 = GROUPS
    return PhotonGroups(L0, L1, np.full(L0.size, N), np.sqrt(L0 * L1))


def test_flat_absorption_gives_the_flux_and_the_exact_temperature():
    g = grid()
    po = PseudoBinOptics(g, flat(), GROUPS, n_nodes=8)
    s = np.full(g.n_bins, 0.5)
    ph = photons()
    kg, power, uv = po.absorption(s, ph, 0.0)
    np.testing.assert_allclose(kg, 1e4, rtol=1e-12)
    assert power[0] == pytest.approx(1e4 * np.sum(C * ph.N * ph.egy * EV), rel=1e-12)
    np.testing.assert_allclose(po.planck_mean(s), 1e4, rtol=1e-12)
    T = po.dust_temperature(s, power)
    np.testing.assert_allclose(T, (power / (4.0 * SIGMA_SB * 1e4)) ** 0.25, rtol=1e-10)
    _, p_bg, uv_bg = po.absorption(s, None, 1.0)
    assert p_bg[0] > 1e4 * 1.6e-3                          # 1-13.6 eV, more than the 6-13.6 eV Habing flux
    assert uv_bg[0] > 0.0 and np.all(uv > 0.0)


def test_pseudo_bins_blend_linear_in_s_from_the_lower_to_the_upper_edge():
    """kappa = a (per gram): the bin value runs from kappa(a_L) at s = 0 to kappa(a_R) at s = 1."""
    g = grid()
    po = PseudoBinOptics(g, lambda a, lam: np.repeat(np.asarray(a, float)[:, None], np.size(lam), 1), GROUPS,
                         n_nodes=8)
    k0 = po.absorption(np.zeros(g.n_bins), None, 1.0)[0][:, 0]
    k1 = po.absorption(np.ones(g.n_bins), None, 1.0)[0][:, 0]
    np.testing.assert_allclose(k0, g.a_edges[:-1], rtol=1e-9)
    np.testing.assert_allclose(k1, g.a_edges[1:], rtol=1e-9)
    kh = po.absorption(np.full(g.n_bins, 0.5), None, 1.0)[0][:, 0]
    assert np.all((kh > k0) & (kh < k1))


def test_group_shape_follows_the_mean_energy():
    """kappa proportional to the photon energy: a group's mean is its photon-weighted mean energy, the egy the cell
    gives it (inside the range the in-group shapes span)."""
    g = grid()
    po = PseudoBinOptics(g, lambda a, lam: np.repeat((HC_EV_UM / np.asarray(lam))[None, :], np.size(a), 0), GROUPS,
                         n_nodes=8, n_e=64)
    ph = photons()
    egy = 0.5 * (po.egy[:, 3] + po.egy[:, 4])
    ph = PhotonGroups(ph.L0, ph.L1, ph.N, egy)
    kg = po.absorption(np.full(g.n_bins, 0.5), ph, 0.0)[0]
    np.testing.assert_allclose(kg[0], egy, rtol=2e-3)


def test_the_covalent_bins_take_their_band_gap():
    g = grid()
    po = PseudoBinOptics(g, flat(), GROUPS, gap_bins=(1, 2), n_nodes=8,
                         per_gram_gap=lambda a, lam, e: np.full((np.size(a), np.size(lam)), 1e4 * (1.0 + e)),
                         gap_nodes=np.linspace(0.1, 2.5, 5))
    s = np.full(g.n_bins, 0.5)
    kg, _, _ = po.absorption(s, photons(), 1.0, gap=np.array([0.4, 1.9]))
    np.testing.assert_allclose(kg[[1, 2], 0], [1.4e4, 2.9e4], rtol=1e-12)
    np.testing.assert_allclose(kg[[0, 3, 4, 5], 0], 1e4, rtol=1e-12)
    uv = po.uv_photons(s, photons(), 1.0)
    assert uv.shape == (2, 5) and np.all(np.diff(uv, axis=1) > 0)
    with pytest.raises(ValueError):
        po.absorption(s, photons(), 1.0)
    with pytest.raises(ValueError):
        PseudoBinOptics(g, flat(), GROUPS, gap_bins=(1, 3), per_gram_gap=flat(), n_nodes=8)


def optical_carbon(gap=False):
    from dataclasses import replace

    from test_gsd_cell import carbon

    from pycalima.gsd.hydrogen import BinHydrogen

    c = carbon(n_H=30.0)
    g = c.grid
    if not gap:
        c.optics = PseudoBinOptics(g, flat(), GROUPS, n_nodes=8)
        return c
    c.optics = PseudoBinOptics(g, flat(), GROUPS, gap_bins=(4, 5, 6, 7), n_nodes=8,
                               per_gram_gap=lambda a, lam, e: np.full((np.size(a), np.size(lam)), 1e4 * (1.0 + e)),
                               gap_nodes=np.linspace(0.1, 2.5, 7))
    c.hydrogen = [BinHydrogen(g, i, s=0.5, n_nodes=8, n_e=13) for i in range(4, 8)]
    return replace(c)


def test_precool_gives_the_dust_temperature_and_update_uses_it():
    from dataclasses import replace

    from test_gsd_cell import YR, carbon, inputs

    from pycalima.gsd import cell

    c = optical_carbon()
    inp = replace(inputs(), photons=photons(3e6), G0_bg=1e2)           # ~1500 K: sublimation depends on T_d
    pre = cell.precool([c], inp)
    s = np.full(c.grid.n_bins, 0.5)
    _, power, _ = c.optics.absorption(s, inp.photons, inp.G0_bg)
    T = (power[0] / (4.0 * SIGMA_SB * 1e4)) ** 0.25
    np.testing.assert_allclose(pre.t_dust[0], T, rtol=1e-10)
    np.testing.assert_allclose(pre.absorption[0], c.M.sum() * 1e4, rtol=1e-12)
    assert pre.e_gap == [None]
    opt = cell.Options(processes=frozenset({"accretion"}))
    cell.update([c], inp, pre, 1e5 * YR, opt)
    ref = carbon(n_H=30.0)
    cell.update([ref], replace(inp, T_d=T), None, 1e5 * YR, opt)
    np.testing.assert_array_equal(c.M, ref.M)
    hot = carbon(n_H=30.0)
    cell.update([hot], inp, None, 1e5 * YR, opt)                     # inputs.T_d = 20 K: no sublimation
    assert not np.array_equal(hot.M, ref.M)


def test_the_covalent_band_gap_is_the_hydrogen_equilibrium(jones):
    from dataclasses import replace

    from test_gsd_cell import inputs

    from pycalima.gsd import cell
    from pycalima.gsd.hydrogen import E_FLOOR, E_PLATEAU, Y_CH, equilibrium_gap, k_h
    from pycalima.gsd.optics import state_s

    c = optical_carbon(gap=True)
    dark = cell.precool([c], inputs()).e_gap[0]
    np.testing.assert_allclose(dark, E_PLATEAU, atol=2e-3)                 # no UV: hydrogenated to the plateau
    lit = replace(inputs(), photons=photons(1e-3), G0_bg=1e2)
    np.testing.assert_allclose(cell.precool([c], replace(lit, n_HI=0.0)).e_gap[0], E_FLOOR, atol=2e-3)
    gap = cell.precool([c], replace(lit, n_HI=10.0)).e_gap[0]
    uv = c.optics.uv_photons(state_s(c.grid, c.N, c.M), lit.photons, lit.G0_bg) * 12.011 * 1.66053906660e-24
    want = [equilibrium_gap(bh, Y_CH * np.interp(bh.e_grid, c.optics.gap_nodes, uv[j]), k_h(10.0, lit.T))[1]
            for j, bh in enumerate(c.hydrogen)]
    np.testing.assert_allclose(gap, want, rtol=1e-12)
    assert np.all((gap > E_FLOOR) & (gap < E_PLATEAU))
    with pytest.raises(ValueError):
        cell.precool([replace(c, hydrogen=None)], lit)
