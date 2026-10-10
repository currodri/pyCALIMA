"""The collapse size grid and its Smoluchowski coagulation scheme.

Invariants (power-law round-trip, exact mass conservation, positivity, number
decrease) and the two coagulation problems with exact solutions, both from an
initial n(m) = exp(-m) (total number and mass 1):

* constant kernel K = 2: n(m, t) = exp(-m / s) / s**2, s = 1 + t;
* additive kernel K = m + m' (Golovin 1963):
  n(m, t) = (1 - tau) / (m sqrt(tau)) exp(-(1 + tau) m) I_1(2 m sqrt(tau)),
  tau = 1 - exp(-t).

Marchand et al. (2021, A&A 649, A50) test their scheme on the same pair
(their Fig. 5).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ive

from pycalima.collapse.coagulation import WEIGHTS, _upsilon, advance, collector_bins
from pycalima.collapse.grid import SizeGrid, bin_means, pint, slopes


def _rho_constant(grid, t):
    x = grid.m_edges / (1.0 + t)
    F = (1.0 + x) * np.exp(-x)
    return F[:-1] - F[1:]


def _rho_additive(grid, t):
    tau = -np.expm1(-t)
    st = np.sqrt(tau)

    def mass_per_lnm(lm):
        m = np.exp(lm)
        # ive(1, x) = I_1(x) exp(-x): the exponent combines to -m (1 - sqrt(tau))**2
        return m * (1.0 - tau) / st * ive(1, 2.0 * m * st) * np.exp(-m * (1.0 - st) ** 2)

    return np.array([quad(mass_per_lnm, np.log(lo), np.log(hi), limit=200)[0]
                     for lo, hi in zip(grid.m_edges[:-1], grid.m_edges[1:])])


def _const(mm, aa):
    return np.full((mm.size, mm.size), 2.0)


def _additive(mm, aa):
    return mm[:, None] + mm[None, :]


def _l1(rho, exact):
    return np.abs(rho - exact).sum() / exact.sum()


# ---------------------------------------------------------------------------
# grid
# ---------------------------------------------------------------------------

def test_pint_is_continuous_through_minus_one():
    r = 2.7
    assert pint(r, -1.0) == pytest.approx(np.log(r), rel=1e-14)
    for e in (-1.0 - 1e-9, -1.0 + 1e-9):
        assert pint(r, e) == pytest.approx(np.log(r), rel=1e-8)
    assert pint(r, 2.0) == pytest.approx((r ** 3 - 1.0) / 3.0, rel=1e-14)


@pytest.mark.parametrize("beta", [-11.0 / 6.0, -2.0, -1.0, 0.5])
def test_slopes_recover_a_power_law_in_every_bin(beta):
    """MRN (dn/da ∝ a^-3.5) is beta = -11/6; the end bins use one-sided slopes."""
    g = SizeGrid.from_radii(5e-7, 2.5e-5, 13, 2.3)
    rho = g.power_law_rho(beta, 1.0)
    np.testing.assert_allclose(g.slopes(rho), beta, rtol=0, atol=1e-10)


def test_bin_means_match_quadrature_and_stay_in_the_bin():
    g = SizeGrid.from_masses(1.0, 1e6, 12)
    beta = np.linspace(-6.0, 4.0, 12)
    mean_m = g.mean_mass(beta)
    for lo, hi, b, mm in zip(g.m_edges[:-1], g.m_edges[1:], beta, mean_m):
        num = quad(lambda m: m ** (b + 1.0), lo, hi)[0]
        den = quad(lambda m: m ** b, lo, hi)[0]
        assert mm == pytest.approx(num / den, rel=1e-10)
        assert lo < mm < hi


def test_upsilon_is_a_fraction_and_the_collector_holds_the_product():
    g = SizeGrid.from_radii(1e-7, 1e-2, 40, 3.0)
    C = collector_bins(g.m_edges)
    rng = np.random.default_rng(1)
    beta = rng.uniform(-8.0, 6.0, g.n_bins)
    mean_m = bin_means(g.m_lo, g.eta, beta, 1.0)
    for i in range(g.n_bins):
        for j in range(i + 1):
            c = C[i, j]
            lo = g.m_lo[i] + g.m_lo[j]
            assert g.m_lo[c] <= lo * (1 + 1e-12)
            # the largest product, eta (m_lo_i + m_lo_j), never passes the top of c + 1
            assert lo * g.eta <= g.m_lo[c] * g.eta ** 2 * (1 + 1e-12)
            if c < g.n_bins:
                u = _upsilon(i, j, c, g.m_lo, g.eta, beta, mean_m, WEIGHTS["turbulent"])
                assert 0.0 <= u <= 1.0


# ---------------------------------------------------------------------------
# conservation and positivity
# ---------------------------------------------------------------------------

def test_mass_is_conserved_exactly_including_overflow():
    g = SizeGrid.from_radii(1e-7, 1e-4, 30, 3.0)
    rng = np.random.default_rng(7)
    rho0 = rng.uniform(0.0, 1.0, g.n_bins) * (rng.uniform(size=g.n_bins) > 0.3)
    A = rng.uniform(0.5, 2.0, (g.n_bins, g.n_bins))
    A = A + A.T

    def kernel(mm, aa):
        return A * (aa[:, None] + aa[None, :]) ** 2

    rho, over, steps = advance(g, rho0, 5.0, kernel, eps=0.05)
    assert steps > 10
    assert over > 0.0          # this run does push mass off the top of the grid
    assert (rho >= 0.0).all()
    assert rho.sum() + over == pytest.approx(rho0.sum(), rel=1e-12)   # round-off over the steps


def test_number_decreases_and_follows_the_constant_kernel_law():
    """Constant K = 2: N(t) = 1 / (1 + t)."""
    g = SizeGrid.from_masses(4e-6, 4e8, 60)
    rho0 = _rho_constant(g, 0.0)
    N_prev = g.number_density(rho0, g.slopes(rho0)).sum()
    for t in (1.0, 10.0, 100.0):
        rho, _, _ = advance(g, rho0, t, _const, weights="constant")
        N = g.number_density(rho, g.slopes(rho)).sum()
        assert N < N_prev
        assert N == pytest.approx(1.0 / (1.0 + t), rel=0.02)
        N_prev = N


# ---------------------------------------------------------------------------
# exact solutions (Marchand+21 Fig. 5)
# ---------------------------------------------------------------------------

def test_constant_kernel_matches_the_exact_solution():
    g = SizeGrid.from_masses(4e-6, 4e8, 60)
    rho, _, _ = advance(g, _rho_constant(g, 0.0), 1e3, _const, weights="constant")
    assert _l1(rho, _rho_constant(g, 1e3)) < 0.01


def test_additive_kernel_matches_the_exact_solution_and_converges():
    """t = 6: the mean mass grows by e^6 ~ 400 and the tail by ~1e5. The error
    sits in the steep exponential cut-off; it must fall with resolution."""
    t = 6.0
    errs = []
    for nb in (30, 60):
        g = SizeGrid.from_masses(4e-6, 4e8, nb)
        rho, _, _ = advance(g, _rho_constant(g, 0.0), t, _additive, weights="additive")
        errs.append(_l1(rho, _rho_additive(g, t)))
    assert errs[1] < 0.05
    assert errs[1] < 0.6 * errs[0]


def test_the_tail_threshold_does_not_change_the_answer():
    """Bins below ``tail`` do not set the step; the pair limiter keeps them
    positive. That must be harmless, even for the additive kernel, where the
    sparse high-mass tail drives the growth."""
    g = SizeGrid.from_masses(4e-6, 4e8, 40)
    r0 = _rho_constant(g, 0.0)
    a, _, _ = advance(g, r0, 4.0, _additive, weights="additive", tail=1e-12)
    b, _, _ = advance(g, r0, 4.0, _additive, weights="additive", tail=1e-4)
    assert _l1(b, a) < 0.01
