"""Double photoionisation (``Params.double_ionisation``): the share curve and the charge chain with +2 steps."""

import numpy as np
import pytest

from pycalima.pahchem import photo
from pycalima.pahchem.population import _chain_fractions


def _dense_steady_state(k1, k2, down, live):
    """The same chain solved as a linear system: the null vector of its generator on the live slots."""
    m = live.size
    Q = np.zeros((m, m))
    for i in range(m):
        if not live[i]:
            continue
        for j, k in ((i + 1, k1[i]), (i + 2, k2[i]), (i - 1, down[i])):
            if 0 <= j < m and live[j] and k > 0:
                Q[j, i] += k
                Q[i, i] -= k
    idx = np.flatnonzero(live)
    A = Q[np.ix_(idx, idx)]
    A[-1] = 1.0                                   # replace one balance row by the normalisation
    b = np.zeros(idx.size)
    b[-1] = 1.0
    f = np.zeros(m)
    f[idx] = np.linalg.solve(A, b)
    return f


def test_chain_with_double_steps_is_the_exact_steady_state():
    rng = np.random.default_rng(3)
    n, m = 6, 9
    live = np.zeros((n, m), dtype=bool)
    for s, (lo, hi) in enumerate([(0, 9), (1, 9), (0, 5), (2, 7), (0, 3), (1, 2)]):
        live[s, lo:hi] = True
    k1 = 10.0 ** rng.uniform(-2, 1, (n, m))
    k2 = 10.0 ** rng.uniform(-3, 0, (n, m))
    down = 10.0 ** rng.uniform(-1, 2, (n, m))
    f = _chain_fractions(k1, k2, down, live)
    for s in range(n):
        assert np.allclose(f[s], _dense_steady_state(k1[s], k2[s], down[s], live[s]), rtol=1e-9, atol=1e-14)
    # without double steps: the birth-death product f(i+1) / f(i) = k1(i) / down(i+1)
    f1 = _chain_fractions(k1, np.zeros_like(k2), down, live)
    s = 0
    assert np.allclose(f1[s, 1:] / f1[s, :-1], k1[s, :-1] / down[s, 1:], rtol=1e-12)


def test_double_fraction_rises_as_the_square_and_saturates():
    dip = 20.0
    E = dip + np.array([-5.0, 0.0, photo.DOUBLE_HALF_EV, 1e4])
    p = photo.double_fraction(E, dip)
    assert p[0] == 0.0 and p[1] == 0.0
    assert p[2] == pytest.approx(0.5 * photo.DOUBLE_MAX) and p[3] == pytest.approx(photo.DOUBLE_MAX, rel=1e-4)
    small = photo.double_fraction(dip + np.array([1.0, 2.0]), dip)
    assert small[1] / small[0] == pytest.approx(4.0, rel=1e-2)
