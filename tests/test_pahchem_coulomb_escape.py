"""The Coulomb-window escape fraction of positive PAH ions (``photo.yield_coulomb_window``): its limits and calibration."""

import numpy as np
import pytest
from scipy.optimize import brentq

from pycalima.models.dust_charge import wdb06_yields
from pycalima.pahchem import photo
from pycalima.pahchem.population import LADDER_STEP_FIT

E = np.linspace(5.0, 120.0, 231)


@pytest.mark.parametrize("depth", [2.0, 7.0, 30.0])
def test_window_limits_are_wd01_y2_and_unity(depth):
    """u_c = inf: no bound electron autoionises, WD01's y2 on the same parabola (wdb06_yields); u_c = 0: all do."""
    ip = 9.0
    y2 = wdb06_yields._parabolic_y2_and_mean(-depth * np.ones_like(E), E - ip)[0]
    assert np.allclose(photo.yield_coulomb_window(E, ip, depth, np.inf), np.where(E > ip, y2, 0.0), atol=1e-12)
    assert np.allclose(photo.yield_coulomb_window(E, ip, depth, 0.0), (E > ip).astype(float), atol=1e-12)


def test_window_yield_is_bounded_ordered_and_a_function_of_the_excess_over_the_depth():
    ip = 20.0
    ys = [photo.yield_coulomb_window(E, ip, 15.0, u) for u in (0.0, 0.1, photo.ESCAPE_DEPTH, 1.0, np.inf)]
    for a, b in zip(ys, ys[1:]):                  # a deeper autoionisation cut loses more
        assert np.all(b <= a + 1e-12)
    for y in ys:
        assert np.all((y >= 0.0) & (y <= 1.0))
    # the Coulomb scaling: Y depends on (E - IP) / D alone, at threshold it is 6 I2(u_c), far above it 1
    x = np.linspace(0.01, 5.0, 50)
    for depth in (3.0, 40.0):
        assert np.allclose(photo.yield_coulomb_window(ip + x * depth, ip, depth), photo.yield_coulomb_window(ip + x * 15.0, ip, 15.0))
    assert float(photo.yield_coulomb_window(ip + 1e-6, ip, 15.0)) == pytest.approx(6.0 * photo._window_moments(photo.ESCAPE_DEPTH)[1])
    assert float(photo.yield_coulomb_window(ip + 300.0, ip, 15.0)) > 0.99


def test_escape_depth_reproduces_the_lowest_cation_yield_less_its_error():
    """C30H14+ (Wenzel+20): 0.83 at 15 eV, AE2+ 10.2 eV, less the 21 % calibration error of their cross sections."""
    a, b = LADDER_STEP_FIT
    depth = 2.0 * (a + b / np.sqrt(30.0))
    u_c = brentq(lambda u: float(photo.yield_coulomb_window(15.0, 10.2, depth, u)) - 0.83 * 0.79, 1e-4, 50.0)
    assert u_c == pytest.approx(photo.ESCAPE_DEPTH, rel=2e-3)
