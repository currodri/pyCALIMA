"""Logarithmic grain size grid with a power law in mass inside each bin.

The scheme of Marchand et al. (2021, A&A 649, A50, appendix B; the Ishinisan
code). Bin ``i`` spans the masses ``[m_lo[i], eta * m_lo[i]]``; the state is
the dust mass density ``rho[i]`` of each bin. Inside a bin the number
distribution is the power law ``dn/dm ∝ m**beta[i]``, with ``beta[i]`` set
from the neighbouring bins (``slopes``), so every bin quantity (mean mass,
mean radius, cross-section) is an exact integral over that power law.

For a power law ``dn/dm ∝ m**beta`` across bins, ``rho[i+1] / rho[i] =
eta**(beta + 2)``; ``slopes`` inverts this on each side of a bin and takes the
harmonic mean of the two (a monotonicity-preserving limiter). Appendix B of
the arXiv version omits the ``- 1`` in the one-sided cases (its Eqs. 72-74);
it is restored here, since without it a pure power law is not recovered at the
grid ends.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pycalima.collapse._jit import njit

BETA_MAX = 15.0       # |beta| clamp: steeper than this puts all of a bin on one edge
BETA_EMPTY = -2.0     # slope of a bin with no non-empty neighbour: flat in rho per ln m
_TINY_EXPONENT = 1e-10


@njit(cache=True)
def pint(r, e):
    """``∫_1^r x**e dx`` for ``r >= 1``; continuous through ``e = -1``."""
    s = e + 1.0
    lr = math.log(r)
    if abs(s * lr) < _TINY_EXPONENT:
        return lr * (1.0 + 0.5 * s * lr)
    return math.expm1(s * lr) / s


@njit(cache=True)
def slopes(rho, log_eta):
    """In-bin slopes ``beta`` of ``dn/dm ∝ m**beta`` (Marchand+21, Eqs. 68-74)."""
    nb = rho.size
    beta = np.full(nb, BETA_EMPTY)
    for i in range(nb):
        if rho[i] <= 0.0:
            continue
        has_m = i > 0 and rho[i - 1] > 0.0
        has_p = i < nb - 1 and rho[i + 1] > 0.0
        dm = math.log(rho[i] / rho[i - 1]) - log_eta if has_m else 0.0
        dp = math.log(rho[i + 1] / rho[i]) - log_eta if has_p else 0.0
        if has_m and has_p:
            if dm * dp > 0.0:
                d = 2.0 * dm * dp / (dm + dp)
            else:
                d = dp
        elif has_p:
            d = dp
        elif has_m:
            d = dm
        else:
            continue
        b = d / log_eta - 1.0
        beta[i] = min(max(b, -BETA_MAX), BETA_MAX)
    return beta


@njit(cache=True)
def bin_means(m_lo, eta, beta, power):
    """Number-weighted ``<(m / m_lo)**power>`` of each bin, times ``m_lo**power``."""
    out = np.empty(m_lo.size)
    for i in range(m_lo.size):
        out[i] = m_lo[i] ** power * pint(eta, beta[i] + power) / pint(eta, beta[i])
    return out


@dataclass(frozen=True)
class SizeGrid:
    """Bin edges in mass (and radius, for a bulk density ``rho_s``).

    Build with ``from_radii`` (physical grids) or ``from_masses``
    (dimensionless coagulation tests, ``rho_s`` irrelevant).
    """

    m_edges: np.ndarray
    rho_s: float = 1.0

    @classmethod
    def from_radii(cls, a_min, a_max, n_bins, rho_s):
        a = np.geomspace(a_min, a_max, n_bins + 1)
        return cls(4.0 / 3.0 * np.pi * rho_s * a ** 3, float(rho_s))

    @classmethod
    def from_masses(cls, m_min, m_max, n_bins, rho_s=1.0):
        return cls(np.geomspace(m_min, m_max, n_bins + 1), float(rho_s))

    @property
    def n_bins(self):
        return self.m_edges.size - 1

    @property
    def m_lo(self):
        return self.m_edges[:-1]

    @property
    def eta(self):
        return self.m_edges[1] / self.m_edges[0]

    @property
    def log_eta(self):
        return math.log(self.eta)

    @property
    def a_edges(self):
        return (3.0 * self.m_edges / (4.0 * np.pi * self.rho_s)) ** (1.0 / 3.0)

    def slopes(self, rho):
        return slopes(np.asarray(rho, dtype=float), self.log_eta)

    def mean_mass(self, beta):
        return bin_means(self.m_lo, self.eta, beta, 1.0)

    def mean_radius(self, beta):
        return self.a_edges[:-1] * bin_means(np.ones(self.n_bins), self.eta, beta, 1.0 / 3.0)

    def mean_area(self, beta):
        """Number-weighted geometric cross-section ``<pi a**2>`` per grain."""
        a_lo = self.a_edges[:-1]
        return np.pi * a_lo ** 2 * bin_means(np.ones(self.n_bins), self.eta, beta, 2.0 / 3.0)

    def number_density(self, rho, beta):
        rho = np.asarray(rho, dtype=float)
        return np.where(rho > 0.0, rho / self.mean_mass(beta), 0.0)

    def power_law_rho(self, beta, total):
        """Bin mass densities of ``dn/dm ∝ m**beta`` over the whole grid, summing to ``total``."""
        w = self.m_lo ** (beta + 2.0) * pint(self.eta, beta + 1.0)
        return total * w / w.sum()
