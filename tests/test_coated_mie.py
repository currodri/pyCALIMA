"""Coated-sphere Mie (pycalima.models.tools.coated_mie) against homogeneous spheres."""
import numpy as np
import pytest

from pycalima.models.tools import coated_mie as cm

LAM = np.geomspace(0.02, 1000.0, 120)


@pytest.mark.parametrize("m", [1.5 - 0.01j, 2.0 - 0.8j, 2.4 - 1.2j])
@pytest.mark.parametrize("a", [1e-3, 0.05, 1.0, 3.0])
def test_equal_layers_are_the_homogeneous_sphere(m, a):
    miepython = pytest.importorskip("miepython")
    qe0, qs0, _, g0 = miepython.efficiencies(np.full(LAM.size, m), 2 * a, LAM)
    for frac in (0.0, 0.6):
        qa, qs, g = cm.efficiencies(frac * a, a, LAM, m, m)
        np.testing.assert_allclose(qa, qe0 - qs0, rtol=1e-5)
        ok = qs0 > 1e-12
        np.testing.assert_allclose(qs[ok], qs0[ok], rtol=1e-5)
        np.testing.assert_allclose(g[ok], g0[ok], atol=1e-6)


def test_a_vanishing_mantle_leaves_the_core_sphere():
    miepython = pytest.importorskip("miepython")
    m_core, m_mantle = 2.0 - 0.8j, 1.5 - 0.01j
    for a in (0.05, 0.5, 2.0):
        qa, qs, g = cm.efficiencies(a, a * (1 + 1e-10), LAM, m_core, m_mantle)
        qe0, qs0, _, g0 = miepython.efficiencies(np.full(LAM.size, m_core), 2 * a, LAM)
        np.testing.assert_allclose(qa, qe0 - qs0, rtol=1e-6)
        np.testing.assert_allclose(qs, qs0, rtol=1e-6)


def test_strongly_absorbing_large_coated_grains_stay_physical():
    """|Im(m)| y up to ~250, where an upward Neumann recursion (BHCOAT) fails."""
    qa, qs, g = cm.efficiencies(4.98, 5.0, np.array([0.0912, 0.1, 0.15]), 1.6 - 0.05j, 2.3 - 1.0j)
    assert np.all((qa > 0.7) & (qa < 1.1)) and np.all((qs > 0.9) & (qs < 1.4)) and np.all(np.abs(g) < 1)
