"""Mass-conserving discrete Smoluchowski coagulation on a ``SizeGrid``.

Marchand et al. (2021, A&A 649, A50, appendix B). A collision between grains
of bins ``i >= j`` produces a grain of mass in ``[m_lo[i] + m_lo[j],
eta * (m_lo[i] + m_lo[j])]``, which always lies within the collector bin ``C``
(the bin of ``m_lo[i] + m_lo[j]``) and ``C + 1``. A fraction ``upsilon`` of
the product mass goes to ``C`` and the rest to ``C + 1``, so mass is
conserved exactly, pair by pair.

``upsilon`` is the mass-weighted fraction of the ``(i, j)`` collisions whose
product stays below the top edge of ``C``, integrated over the in-bin power
laws with the mass of the ``j`` grain held at its bin mean (Marchand+21,
Eqs. 82, 85). The integrand is the kernel's *shape* inside the bin pair,
written as a sum of monomials ``c * m**p * m'**q`` (``m`` the grain from the
larger bin); ``WEIGHTS`` holds the ones used here. Marchand+21 Eq. 86 prints
the turbulent expansion without its binomial coefficients 1, 2, 1; they are
included here (they follow from expanding ``(m**(1/3) + m'**(1/3))**2``).

The rate itself uses the kernel evaluated at the bin mean sizes: a pair
``(i, j)`` collides at ``n_i n_j K_ij`` per unit volume (``K_ii / 2`` for
``i = j``). ``advance`` integrates with explicit steps limited to a relative
change ``eps`` per bin (Marchand+21, Eq. 91), with a pair-level limiter in the
sparsely filled tail (see ``advance``).
"""

from __future__ import annotations

import math

import numpy as np

from pycalima.collapse._jit import njit
from pycalima.collapse.grid import pint, slopes, bin_means

# Kernel shapes inside a bin pair: rows (c, p, q) of K ∝ sum c m**p m'**q.
WEIGHTS = {
    "constant": np.array([[1.0, 0.0, 0.0]]),
    "additive": np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]]),
    # (m^1/3 + m'^1/3)^2 m^1/6: geometric cross-section times the Ormel &
    # Cuzzi (2007) intermediate-regime velocity, which scales with the larger grain
    # (m^1/3 + m'^1/3)^2: the geometric cross-section alone (Brownian and other
    # kernels whose velocity factor varies little across a bin pair)
    "geometric": np.array([[1.0, 2.0 / 3.0, 0.0],
                           [2.0, 1.0 / 3.0, 1.0 / 3.0],
                           [1.0, 0.0, 2.0 / 3.0]]),
    "turbulent": np.array([[1.0, 5.0 / 6.0, 0.0],
                           [2.0, 0.5, 1.0 / 3.0],
                           [1.0, 1.0 / 6.0, 2.0 / 3.0]]),
}


def collector_bins(m_edges):
    """``C[i, j]``: the bin of ``m_lo[i] + m_lo[j]`` (may be ``>= n_bins``)."""
    m_lo = m_edges[:-1]
    log_eta = math.log(m_edges[1] / m_edges[0])
    s = m_lo[:, None] + m_lo[None, :]
    # the tolerance keeps a sum that falls exactly on an edge in the bin above it
    return np.floor(np.log(s / m_edges[0]) / log_eta + 1e-9).astype(np.int64)


@njit(cache=True)
def _r(x, eta, m_i, m_j, b_i, b_j, wts):
    """Mass-weighted kernel-shape integral over ``m in [m_i, x m_i]``, ``m' in bin j``.

    The factor ``m_i**(b_i + 1) m_j**(b_j + 1)`` common to numerator and
    denominator of ``upsilon`` is dropped.
    """
    tot = 0.0
    for k in range(wts.shape[0]):
        c, p, q = wts[k, 0], wts[k, 1], wts[k, 2]
        # (m + m') K: the m term, then the m' term
        tot += c * (m_i ** (p + 1.0) * m_j ** q * pint(x, b_i + p + 1.0) * pint(eta, b_j + q)
                    + m_i ** p * m_j ** (q + 1.0) * pint(x, b_i + p) * pint(eta, b_j + q + 1.0))
    return tot


@njit(cache=True)
def _upsilon(i, j, c, m_lo, eta, beta, mean_m, wts):
    x = (m_lo[c] * eta - mean_m[j]) / m_lo[i]
    if x <= 1.0:
        return 0.0
    if x >= eta:
        return 1.0
    return (_r(x, eta, m_lo[i], m_lo[j], beta[i], beta[j], wts)
            / _r(eta, eta, m_lo[i], m_lo[j], beta[i], beta[j], wts))


@njit(cache=True)
def _transfer(n, m_lo, eta, beta, mean_m, K, collector, wts, scale, drho, loss):
    """Add every pair's mass transfer rate, times ``min(scale_i, scale_j)``, to
    ``drho``, and each pair's net removal from a bin to that bin's ``loss``.
    Returns the overflow rate."""
    nb = n.size
    overflow = 0.0
    for i in range(nb):
        if n[i] == 0.0:
            continue
        for j in range(i + 1):
            if n[j] == 0.0:
                continue
            rate = n[i] * n[j] * K[i, j] * min(scale[i], scale[j])
            if i == j:
                rate *= 0.5
            if rate == 0.0:
                continue
            mt = (mean_m[i] + mean_m[j]) * rate
            c = collector[i, j]
            ups = _upsilon(i, j, c, m_lo, eta, beta, mean_m, wts) if c < nb else 1.0
            # net change of every bin touched by the pair: the product may land
            # back in bin i (c >= i always), e.g. a large grain sweeping up a small one
            d_i = -mean_m[i] * rate
            d_j = -mean_m[j] * rate
            d_c = ups * mt
            d_c1 = (1.0 - ups) * mt
            if c == i:
                d_i += d_c
                d_c = 0.0
            if i == j:
                d_i += d_j
                d_j = 0.0
            drho[i] += d_i
            drho[j] += d_j
            if d_i < 0.0:
                loss[i] -= d_i
            if d_j < 0.0:
                loss[j] -= d_j
            if c < nb:
                drho[c] += d_c
            else:
                overflow += d_c
            if c + 1 < nb:
                drho[c + 1] += d_c1
            else:
                overflow += d_c1
    return overflow


def coagulation_rates(grid, rho, K, weights="turbulent"):
    """``(drho/dt, overflow rate)`` at the current state, unlimited."""
    rho = np.asarray(rho, dtype=float)
    beta = slopes(rho, grid.log_eta)
    mean_m = bin_means(grid.m_lo, grid.eta, beta, 1.0)
    n = np.where(rho > 0.0, rho / mean_m, 0.0)
    drho, loss = np.zeros(grid.n_bins), np.zeros(grid.n_bins)
    over = _transfer(n, grid.m_lo, grid.eta, beta, mean_m, K, collector_bins(grid.m_edges),
                     _weights(weights), np.ones(grid.n_bins), drho, loss)
    return drho, over


def _weights(weights):
    return WEIGHTS[weights] if isinstance(weights, str) else np.asarray(weights, dtype=float)


def advance(grid, rho, t_end, kernel, weights="turbulent", eps=0.01, tail=1e-8,
            max_steps=1_000_000):
    """Coagulate ``rho`` over a time ``t_end``.

    ``kernel(mean_m, mean_a) -> K`` gives the symmetric kernel matrix [cm^3 s^-1]
    at the current bin means; it is re-evaluated every step, since the means
    move with the in-bin slopes. Returns ``(rho, overflow_mass, n_steps)``.

    Step size: a relative net change ``eps`` in every bin holding more than a
    fraction ``tail`` of the mass (Marchand+21, Eq. 91). Bins below that are
    not allowed to set the step: in the sparsely filled tail the loss rate per
    grain, sum_j n_j K_ij, can exceed the bulk's by orders of magnitude (it
    grows as m N for the additive kernel). Instead, a bin that would lose more
    than half its mass in the step has the transfers of every pair it takes
    part in scaled down until it loses half (counting, pair by pair, only the
    net removal: a product landing back in the same bin is not a loss), so mass
    stays exactly conserved and every bin positive.
    """
    wts = _weights(weights)
    rho = np.array(rho, dtype=float)
    collector = collector_bins(grid.m_edges)
    m_lo, eta, log_eta = grid.m_lo, grid.eta, grid.log_eta
    a_lo = grid.a_edges[:-1]
    nb = grid.n_bins
    ones = np.ones(nb)
    t = 0.0
    overflow = 0.0
    steps = 0
    while t < t_end:
        if steps >= max_steps:
            raise RuntimeError(f"coagulation: {max_steps} steps reached at t = {t:g} of {t_end:g}")
        beta = slopes(rho, log_eta)
        mean_m = bin_means(m_lo, eta, beta, 1.0)
        mean_a = a_lo * bin_means(ones, eta, beta, 1.0 / 3.0)
        n = np.where(rho > 0.0, rho / mean_m, 0.0)
        K = kernel(mean_m, mean_a)
        drho, loss = np.zeros(nb), np.zeros(nb)
        dover = _transfer(n, m_lo, eta, beta, mean_m, K, collector, wts, ones, drho, loss)
        ctrl = (drho != 0.0) & (rho > tail * rho.sum())
        if not ctrl.any():
            break
        dt = min(t_end - t, eps * np.min(rho[ctrl] / np.abs(drho[ctrl])))
        scale = np.ones(nb)
        lim = loss * dt > 0.5 * rho
        if lim.any():
            scale[lim] = 0.5 * rho[lim] / (loss[lim] * dt)
            drho[:], loss[:] = 0.0, 0.0
            dover = _transfer(n, m_lo, eta, beta, mean_m, K, collector, wts, scale, drho, loss)
        rho += drho * dt
        np.maximum(rho, 0.0, out=rho)   # round-off only
        overflow += dover * dt
        t += dt
        steps += 1
    return rho, overflow, steps


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------

KB_CGS = 1.380649e-16      # erg K^-1
G_CGS = 6.67430e-8         # cm^3 g^-1 s^-2
MH_CGS = 1.6735575e-24     # g (H atom)


def intermediate_turbulence_kernel(nH, T, rho_s, mu=2.3, gamma=5.0 / 3.0, z0=2.97,
                                   T_power=0.25):
    """Turbulent kernel in the intermediate regime of Ormel & Cuzzi (2007), as
    set up by Marchand+21 (their Eqs. 9-14): turbulence injected at the Jeans
    length with velocity ~ c_s, Epstein drag, so that

        K = sqrt(8 / 3 pi) pi (a + a')**2 dV,   dV = C2 nH**-1/4 T**T_power a_max**1/2,
        C2 = [(3 / sqrt 8) z0 (k_B G)**1/2 gamma rho_s / (mu m_H)]**1/2.

    Eqs. 10-13 give ``T_power = +1/4`` (dV**2 = 3/2 c_s**2 z0 tau_s / tau_L,
    with c_s**2 ∝ T and tau_s ∝ T**-1/2). Their Eqs. 14, 17 and 21 print
    ``-1/4``, and their Table 1 times (and the machine-precision agreement of
    their 1e5 K and 10 K runs) are consistent only with that, so reproducing
    their figures needs ``T_power=-0.25``.

    Returns ``kernel(mean_m, mean_a) -> K`` for ``advance``, with the
    ``"turbulent"`` weights.
    """
    C2 = np.sqrt(3.0 / np.sqrt(8.0) * z0 * np.sqrt(KB_CGS * G_CGS) * gamma * rho_s / (mu * MH_CGS))
    pre = np.sqrt(8.0 / (3.0 * np.pi)) * np.pi * C2 * nH ** -0.25 * T ** T_power

    def kernel(mean_m, mean_a):
        a_big = np.maximum(mean_a[:, None], mean_a[None, :])
        return pre * (mean_a[:, None] + mean_a[None, :]) ** 2 * np.sqrt(a_big)

    return kernel
