"""Photo-release of molecules from nano-grains (``pycalima.gsd.release``)."""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from pycalima.gsd import release as R

pytestmark = pytest.mark.gsd_quick

M_C = 12.011 * 1.66053906660e-24


def n_host(a):
    return 4.0 / 3.0 * np.pi * 1.6 * a ** 3 / M_C


def test_peak_temperatures_follow_the_literature_estimates():
    """A UV photon heats a ~1 nm host to a few hundred K, a 3 nm host to ~100-140 K (the review's estimates)."""
    assert 350.0 < R.peak_temperature(10.0, 24 + n_host(1e-7)) < 700.0
    assert 100.0 < R.peak_temperature(10.0, 24 + n_host(3e-7)) < 140.0
    assert R.peak_temperature(10.0, 100) > R.peak_temperature(10.0, 1000)


def test_release_is_a_small_molecule_small_host_effect(themis_cm20):
    """(The aggregate's IR emission uses the THEMIS-like a-C optics, hence the DustEM data.)"""
    assert R.p_thermal(24, 0.6e-7, 10.0) == pytest.approx(1.0)
    assert R.p_thermal(24, 3e-7, 10.0) < 1e-30
    assert R.p_thermal(50, 1e-7, 10.0) < 1e-10
    for nc in (24, 50):
        p = [R.p_thermal(nc, a, 10.0) for a in (0.6e-7, 1e-7, 2e-7, 5e-7)]
        assert all(x >= y for x, y in itertools.pairwise(p))
    assert R.p_thermal(24, 1e-7, 13.0) >= R.p_thermal(24, 1e-7, 6.0)


def test_ir_power_is_extensive_and_rises_with_temperature(themis_cm20):
    assert R.ir_power(2000, 300.0) == pytest.approx(2.0 * R.ir_power(1000, 300.0))
    assert R.ir_power(1000, 600.0) > 10.0 * R.ir_power(1000, 300.0)


def test_s_eff_limits():
    assert float(R.s_eff(1e-8, 0.0, 1e14)) == 1.0                    # no release: the molecule stays
    assert float(R.s_eff(1e-8, 1e-3, np.inf)) == 0.0                 # release and no burial: it never stays
    assert float(R.s_eff(1e-8, 0.0, np.inf)) == 1.0
    assert float(R.s_eff(1e-8, 1e-3, 1e11)) == pytest.approx(1.0 / (1.0 + 1e-8 * 1e-3 * 1e11))
    assert float(R.s_eff(1e-8, 0.0, 1e11, y_pd=1e-6)) < 1.0
    assert np.isinf(R.burial_time(0.0)) and R.burial_time(1e-15) == pytest.approx(0.34e-7 / 1e-15)
