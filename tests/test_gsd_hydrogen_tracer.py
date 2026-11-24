"""The hydrogen tracer M_H (A3c phase 5): carried by every operator at the donor bin's H/C, conserved with the gas."""
from __future__ import annotations

import numpy as np
import pytest

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
