"""Invariants of the Coulomb focusing factors over a Gaussian charge distribution
(pycalima.models.dust_charge.coulomb_focusing, the reference of RAMSES
compute_Coulomb_focusing_ions)."""
import math

import numpy as np
import pytest

from pycalima.models.dust_charge import coulomb_focusing as cf


def gaussian_sum(mu, sig, T, a, jmax, k=15.0):
    """The discrete Gaussian on every integer within k max(sig, 1/2) of mu, untapered."""
    sw = max(sig, 0.5)
    Z = np.arange(math.floor(mu - k * sw), math.ceil(mu + k * sw) + 1, dtype=float)
    q = (Z - mu) ** 2
    p = np.exp(-0.5 * (q - q.min()) / sig ** 2)
    return np.maximum(cf.coulomb_factors_exact(Z, p / p.sum(), T, a, jmax), cf.D_FLOOR)


@pytest.mark.parametrize("mu,sig,T,a", [(0.3, 0.4, 100.0, 1e-6), (-1.7, 0.8, 1e3, 5e-7), (5.2, 0.6, 1e4, 1e-6),
                                        (0.0, 0.9, 50.0, 1e-5), (-12.4, 0.5, 300.0, 1e-5)])
def test_narrow_distributions_are_the_discrete_gaussian(mu, sig, T, a):
    # below SIG_LO only weights below exp(-12.5) of the peak are tapered
    np.testing.assert_allclose(cf.coulomb_factors(mu, sig, T, a, 4), gaussian_sum(mu, sig, T, a, 4), rtol=1e-4)


@pytest.mark.parametrize("mu,T,a", [(3.0, 1e4, 1e-6), (40.0, 1e4, 1e-5), (-200.0, 100.0, 1e-5), (0.4, 1e4, 5e-7)])
def test_wide_distributions_close_to_the_discrete_gaussian(mu, T, a):
    # above SIG_HI the Gaussian integrals stand for the sum over the integers
    sig = cf.SIG_HI + 0.5
    np.testing.assert_allclose(cf.coulomb_factors(mu, sig, T, a, 3), gaussian_sum(mu, sig, T, a, 3), rtol=2e-2)


@pytest.mark.parametrize("sig", [0.3, 0.8, 1.2, 3.0, 4.0, 6.0])
def test_continuous_where_charges_enter_or_leave(sig):
    # the places a hard window would jump: the taper's outer ends, the +-3 sigma edges and the
    # half-integer means of a rounded window
    T, a, eps = 300.0, 1e-6, 1e-9
    sw = max(sig, 0.5)
    for k in range(-3, 4):
        for mu in (k + cf.K_WIN * sw, k - cf.K_WIN * sw, k + 0.5 + 3.0 * sig, k + 0.5):
            lo = cf.coulomb_factors(mu - eps, sig, T, a, 3)
            hi = cf.coulomb_factors(mu + eps, sig, T, a, 3)
            assert np.max(np.abs(np.log(hi / lo))) < 1e-6, (sig, mu)


@pytest.mark.parametrize("mu", [-2.3, 0.2, 0.5, 4.7])
def test_continuous_across_the_blend(mu):
    T, a, eps = 1e3, 1e-6, 1e-9
    for sig in (cf.SIG_LO, 0.5 * (cf.SIG_LO + cf.SIG_HI), cf.SIG_HI):
        lo = cf.coulomb_factors(mu, sig - eps, T, a, 3)
        hi = cf.coulomb_factors(mu, sig + eps, T, a, 3)
        assert np.max(np.abs(np.log(hi / lo))) < 1e-6, (mu, sig)


@pytest.mark.parametrize("mu,sig", [(2.3, 0.7), (-5.1, 1.2), (14.0, 4.0)])
def test_mirror_symmetry(mu, sig):
    # a negative impactor on a grain of charge Z is a positive one on -Z
    Dp = cf.coulomb_factors(mu, sig, 1e3, 1e-6, 1)
    Dm = cf.coulomb_factors(-mu, sig, 1e3, 1e-6, 1)
    assert Dp[0] == pytest.approx(Dm[2], rel=1e-12)


def test_attractive_far_from_zero_is_linear_in_the_mean():
    mu, sig, T, a = -60.0, 5.0, 1e4, 1e-5
    C1 = cf.E2 / (cf.KB * T * a)
    np.testing.assert_allclose(cf.coulomb_factors(mu, sig, T, a, 2)[2:], [1 + j * C1 * abs(mu) for j in (1, 2)],
                               rtol=1e-12)


def test_neutral_impactor_and_floor():
    D = cf.coulomb_factors(30.0, 2.0, 50.0, 5e-7, 5)
    assert D[1] == 1.0
    assert np.all(D >= cf.D_FLOOR)
    assert D[-1] == cf.D_FLOOR          # a +5 impactor on a +30 grain at 50 K
