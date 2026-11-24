"""The cell's pseudo-bin optics (``pycalima.gsd.optics.PseudoBinOptics``) on analytic per-gram absorption."""
from __future__ import annotations

import numpy as np
import pytest

from pycalima.gsd.charging import AGN_GROUP_L0, AGN_GROUP_L1, PhotonGroups
from pycalima.gsd.grid import MassGrid
from pycalima.gsd.optics import HC_EV_UM, SIGMA_SB, PseudoBinOptics

pytestmark = pytest.mark.gsd_quick

GROUPS = (np.array(AGN_GROUP_L0), np.array(AGN_GROUP_L1))
C = 2.99792458e10
EV = 1.602176634e-12


@pytest.fixture(autouse=True)
def _clean_env(isolated_env):
    yield


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
