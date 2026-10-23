"""Two-moment power-law reconstruction inside a bin.

In bin [m_L, m_R] (ratio eta = m_R/m_L) the number distribution is
dN/dm = A m^beta. Its mean mass,

    mbar / m_L = ((beta+1)/(beta+2)) (eta^(beta+2) - 1) / (eta^(beta+1) - 1),

rises monotonically from 1 (beta -> -inf, a spike at m_L) to eta (beta ->
+inf, a spike at m_R), so every realisable (N, M) pair has exactly one beta.
In radius, dN/da = A' a^lam with lam = 3 beta + 2. Integrals are taken in
u = a / a_L, which keeps the powers of order one for ``|beta|`` <= BETA_MAX.
"""

from __future__ import annotations

import numpy as np

BETA_MAX = 150.0


def mean_mass_ratio(beta, eta):
    """mbar / m_L for exponent beta in a bin of mass ratio eta (vectorised)."""
    beta = np.asarray(beta, float)
    le = np.log(eta)
    out = np.empty_like(beta)
    m1 = np.isclose(beta, -1.0, atol=1e-9)
    m2 = np.isclose(beta, -2.0, atol=1e-9)
    gen = ~(m1 | m2)
    b = beta[gen]
    # (b+1)/(b+2) * expm1((b+2) le) / expm1((b+1) le), stable for large |b|
    out[gen] = (b + 1) / (b + 2) * np.expm1((b + 2) * le) / np.expm1((b + 1) * le)
    out[m1] = (eta - 1.0) / le
    out[m2] = le / (1.0 - 1.0 / eta)
    return out


def solve_beta(mbar_ratio, eta, n_iter=56):
    """Exponent beta whose mean mass ratio is ``mbar_ratio`` (in (1, eta)),
    by bisection on [-BETA_MAX, BETA_MAX] (compiled kernel)."""
    from ._kernels import solve_beta as _solve

    r = np.atleast_1d(np.asarray(mbar_ratio, float))
    out = _solve(r.ravel().copy(), float(eta), BETA_MAX, int(n_iter)).reshape(r.shape)
    return out if np.ndim(mbar_ratio) else float(out[0])


def int_pow(lam, u1, u2):
    """Integral of u^lam du from u1 to u2 (u > 0), with lam = -1 handled."""
    lam = np.asarray(lam, float)
    near = np.abs(lam + 1.0) < 1e-12
    p = lam + 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        gen = (np.power(u2, p) - np.power(u1, p)) / np.where(near, 1.0, p)
    return np.where(near, np.log(u2 / u1), gen)


class BinProfile:
    """The power-law profiles of all bins of a grid for given (N, M).

    ``lam[i]`` is the radius exponent of bin i and ``A[i]`` the normalisation
    such that dN/du = A[i] u^lam[i] on u in [1, u_R], u = a / a_L,i.
    Empty bins get A = 0. A bin whose mean mass lies closer to an edge than
    the power law with ``|beta|`` <= BETA_MAX can reach (a near-monodisperse
    population) is represented exactly as a spike: ``delta[i]`` is True and
    all its grains sit at u_delta[i].
    """

    def __init__(self, grid, N, M):
        self.grid = grid
        N = np.asarray(N, float)
        M = np.asarray(M, float)
        self.N, self.M = N, M
        mL = grid.m_edges[:-1]
        filled = N > 0
        ratio = np.where(filled, M / np.where(filled, N, 1.0) / mL, 0.5 * (1 + grid.eta))
        r_lo, r_hi = mean_mass_ratio(np.array([-BETA_MAX, BETA_MAX]), grid.eta)
        self.delta = filled & ((ratio <= r_lo) | (ratio >= r_hi))
        self.u_delta = np.clip(ratio, 1.0, grid.eta) ** (1.0 / 3.0)
        self.beta = solve_beta(ratio, grid.eta)
        self.lam = 3.0 * self.beta + 2.0
        self.uR = grid.eta ** (1.0 / 3.0)
        norm = int_pow(self.lam, 1.0, self.uR)
        self.A = np.where(filled & ~self.delta, N / norm, 0.0)
        self.aL = grid.a_edges[:-1]

    def number(self, i, u1, u2):
        """Number in bin i between u1 and u2 (u = a / a_L,i)."""
        if self.delta[i]:
            return self.N[i] if u1 <= self.u_delta[i] < u2 else 0.0
        return self.A[i] * int_pow(self.lam[i], u1, u2)

    def moment(self, i, k, u1, u2):
        """Integral of u^k dN between u1 and u2 in bin i."""
        if self.delta[i]:
            ud = self.u_delta[i]
            return self.N[i] * ud ** k if u1 <= ud < u2 else 0.0
        return self.A[i] * int_pow(self.lam[i] + k, u1, u2)
