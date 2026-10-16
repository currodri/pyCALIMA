"""Compiled kernels of the GSD solver (numba when installed, Python otherwise).

``advance``: exact characteristics of da/dt = f(a) for f tabulated on
ascending radii and linear in a between table points. In a cell with
f = f0 + g (a - a0) a grain moves as a(t) = a + f(a) (e^{g t} - 1) / g,
so it is integrated cell by cell without time steps: the cost does not
depend on stiffness, and a root of f (an equilibrium radius) is approached
exponentially and never crossed. Grains carried below the table return 0.

``remap_profiles``: the analytic remap of two-moment power-law bin profiles
along those characteristics (see ``remap``).
"""

from __future__ import annotations

import math

import numpy as np

from pycalima.collapse._jit import njit


@njit(cache=True)
def _cell(ta, x):
    lo, hi = 0, ta.size - 1
    if x < ta[0]:
        return -1
    if x >= ta[hi]:
        return hi
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ta[mid] <= x:
            lo = mid
        else:
            hi = mid
    return lo


@njit(cache=True)
def _phi(z):
    # expm1(z) / z, continuous at 0
    if abs(z) < 1e-8:
        return 1.0 + 0.5 * z
    return math.expm1(z) / z


@njit(cache=True)
def advance_one(ta, tf, x, dt):
    """Radius after dt of a grain at x (0 if it fell below the table).

    In a cell the grain moves as x(t) = x + f(x) t expm1(g t) / (g t); the
    time to reach a cell edge is log1p(g (edge - x) / f(x)) / g. Written so,
    neither form cancels when f is nearly constant across the cell, so
    displacements far smaller than x are resolved to round-off.
    """
    n = ta.size
    t = dt
    k = _cell(ta, x)
    if k < 0:
        return 0.0
    for _ in range(4 * n + 8):
        if k >= n - 1:
            # above the table: constant velocity of the last point
            x = x + tf[n - 1] * t
            return x if x > 0.0 else 0.0
        a0, a1 = ta[k], ta[k + 1]
        f0, f1 = tf[k], tf[k + 1]
        g = (f1 - f0) / (a1 - a0)
        fx = f0 + g * (x - a0)
        if fx == 0.0:
            return x
        up = fx > 0.0
        target = a1 if up else a0
        ft = f1 if up else f0
        if ft == 0.0 or (ft > 0.0) != up:
            # a root of f lies ahead in this cell: approach it, never cross
            # (signs, not fx * ft: the product underflows for tiny rates)
            return x + fx * t * _phi(g * t)
        z = g * (target - x) / fx            # > -1: f keeps its sign to the edge
        tau = (target - x) / fx if abs(z) < 1e-8 else math.log1p(z) / g
        if tau >= t:
            return x + fx * t * _phi(g * t)
        t -= tau
        x = target
        if up:
            k += 1
        else:
            k -= 1
            if k < 0:
                return 0.0
    return x


@njit(cache=True)
def advance(ta, tf, xs, dt):
    out = np.empty_like(xs)
    for i in range(xs.size):
        out[i] = advance_one(ta, tf, xs[i], dt)
    return out


@njit(cache=True)
def _int_pow(lam, u1, u2):
    p = lam + 1.0
    if abs(p) < 1e-12:
        return math.log(u2 / u1)
    return (u2 ** p - u1 ** p) / p


@njit(cache=True)
def _cubic(A, lam, aL, d0, d1, u1, u2):
    # integral of (d0 + d1 u)^3 u^lam du * A, image radius = d0 + d1 u
    return A * (d0 ** 3 * _int_pow(lam, u1, u2)
                + 3.0 * d0 ** 2 * d1 * _int_pow(lam + 1.0, u1, u2)
                + 3.0 * d0 * d1 ** 2 * _int_pow(lam + 2.0, u1, u2)
                + d1 ** 3 * _int_pow(lam + 3.0, u1, u2))


@njit(cache=True)
def _bin_of(a_e, x):
    nb = a_e.size - 1
    if x < a_e[0]:
        return -1
    if x >= a_e[nb]:
        return nb
    lo, hi = 0, nb
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if a_e[mid] <= x:
            lo = mid
        else:
            hi = mid
    return lo


@njit(cache=True)
def remap_profiles(a_e, rho, N, A, lam, delta, u_delta, nodes, X):
    """Remap bin profiles given node radii ``nodes[i, :]`` of each source bin
    and their images ``X[i, :]``. Returns (N_new, M_new, lost, overN, overM)."""
    nb = a_e.size - 1
    mfac = 4.0 / 3.0 * math.pi * rho
    N_new = np.zeros(nb)
    M_new = np.zeros(nb)
    lost = 0.0
    oN = 0.0
    oM = 0.0
    npc = nodes.shape[1] - 1
    for i in range(nb):
        if N[i] <= 0.0:
            continue
        aL = a_e[i]
        if delta[i]:
            Xd = X[i, 0]
            j = _bin_of(a_e, Xd)
            if j < 0 or Xd <= 0.0:
                lost += N[i]
            elif j >= nb:
                oN += N[i]
                oM += N[i] * mfac * Xd ** 3
            else:
                N_new[j] += N[i]
                M_new[j] += N[i] * mfac * Xd ** 3
            continue
        for k in range(npc):
            p, q = nodes[i, k], nodes[i, k + 1]
            Xp, Xq = X[i, k], X[i, k + 1]
            up_, uq_ = p / aL, q / aL
            if Xq - Xp <= 1e-14 * max(Xq, 1e-300):
                n_piece = A[i] * _int_pow(lam[i], up_, uq_)
                j = _bin_of(a_e, Xp)
                if j < 0 or Xp <= 0.0:
                    lost += n_piece
                elif j >= nb:
                    oN += n_piece
                    oM += n_piece * mfac * Xp ** 3
                else:
                    N_new[j] += n_piece
                    M_new[j] += n_piece * mfac * Xp ** 3
                continue
            s = (Xq - Xp) / (q - p)
            c0 = Xp - s * p
            d1 = s * aL
            if Xp < a_e[0]:
                x_hi = min(Xq, a_e[0])
                u_hi = (p + (x_hi - Xp) / s) / aL
                lost += A[i] * _int_pow(lam[i], up_, u_hi)
            if Xq > a_e[nb]:
                x_lo = max(Xp, a_e[nb])
                u_lo = (p + (x_lo - Xp) / s) / aL
                oN += A[i] * _int_pow(lam[i], u_lo, uq_)
                oM += mfac * _cubic(A[i], lam[i], aL, c0, d1, u_lo, uq_)
            j0 = max(_bin_of(a_e, Xp), 0)
            j1 = min(_bin_of(a_e, Xq), nb - 1)
            for j in range(j0, j1 + 1):
                x1 = max(Xp, a_e[j])
                x2 = min(Xq, a_e[j + 1])
                if x2 <= x1:
                    continue
                u1 = (p + (x1 - Xp) / s) / aL
                u2 = (p + (x2 - Xp) / s) / aL
                N_new[j] += A[i] * _int_pow(lam[i], u1, u2)
                M_new[j] += mfac * _cubic(A[i], lam[i], aL, c0, d1, u1, u2)
    return N_new, M_new, lost, oN, oM


@njit(cache=True)
def inject_segments(a_e, rho, J, y):
    """Bin J grains spread uniformly along the polyline of radii y."""
    nb = a_e.size - 1
    mfac = 4.0 / 3.0 * math.pi * rho
    dN = np.zeros(nb)
    dM = np.zeros(nb)
    n_seg = y.size - 1
    per = J / n_seg
    for k in range(n_seg):
        lo = min(y[k], y[k + 1])
        hi = max(y[k], y[k + 1])
        if hi - lo <= 1e-14 * hi:
            j = _bin_of(a_e, lo)
            if 0 <= j < nb:
                dN[j] += per
                dM[j] += per * mfac * lo ** 3
            continue
        dens = per / (hi - lo)
        j0 = max(_bin_of(a_e, lo), 0)
        j1 = min(_bin_of(a_e, hi), nb - 1)
        for j in range(j0, j1 + 1):
            x1 = max(lo, a_e[j])
            x2 = min(hi, a_e[j + 1])
            if x2 > x1:
                dN[j] += dens * (x2 - x1)
                dM[j] += dens * mfac * (x2 ** 4 - x1 ** 4) / 4.0
    return dN, dM


@njit(cache=True)
def _mean_ratio(b, le):
    # mbar / m_L for exponent b in a bin with ln(eta) = le
    if abs(b + 1.0) < 1e-9:
        return (math.exp(le) - 1.0) / le
    if abs(b + 2.0) < 1e-9:
        return le / (1.0 - math.exp(-le))
    return (b + 1.0) / (b + 2.0) * math.expm1((b + 2.0) * le) / math.expm1((b + 1.0) * le)


@njit(cache=True)
def solve_beta(ratio, eta, beta_max, n_iter):
    """Exponents whose mean mass ratios are ``ratio`` (bisection)."""
    le = math.log(eta)
    out = np.empty_like(ratio)
    r_lo = _mean_ratio(-beta_max, le)
    r_hi = _mean_ratio(beta_max, le)
    for i in range(ratio.size):
        r = min(max(ratio[i], r_lo), r_hi)
        lo, hi = -beta_max, beta_max
        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)
            if _mean_ratio(mid, le) < r:
                lo = mid
            else:
                hi = mid
        out[i] = 0.5 * (lo + hi)
    return out
