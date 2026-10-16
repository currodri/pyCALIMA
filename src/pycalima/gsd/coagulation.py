"""Coagulation of one population: two-moment pair transfers, MPRK22 in time.

Each bin's power-law profile (``pl2m``) is represented by two nodes, the
lower and upper halves of its number split at the median, each carrying
N/2 at the mean mass of its half. The two nodes reproduce the bin's N and M
exactly. Every ordered node pair (p, q) collides at rate
r = K(p, q) w_p w_q / 2 [cm^-3 s^-1] (``K`` in cm^3 s^-1), removing one
grain from each and depositing one of mass m_p + m_q in the bin that
contains it, or in the overflow above the grid.

Mass then moves between bins as a conservative production-destruction
system, integrated with the modified Patankar-Runge-Kutta scheme MPRK22
(Kopecz & Meister 2018): the mass flows out of a bin are weighted by its
new-to-old ratio, so mass is conserved to round-off and stays positive for
any step. Number is not conserved by coagulation; its losses are taken
Patankar-implicitly in each bin (always positive), and its gains follow the
mass delivered, so a product's number and mass arrive together. The
difference between the first- and second-order stages sizes the sub-steps.

Collisions are within one population only: grains of different materials
do not coagulate with each other here.
"""

from __future__ import annotations

import math

import numpy as np

from pycalima.collapse._jit import njit

from .pl2m import BinProfile, int_pow
from .remap import realisable


def bin_nodes(grid, N, M):
    """Two nodes per bin: masses m[nb, 2] and weights w[nb, 2] (N/2 each)
    reproducing N and M of every bin. Empty bins get zero weight."""
    N = np.asarray(N, float)
    M = np.asarray(M, float)
    prof = BinProfile(grid, N, M)
    nb = grid.n_bins
    mL = grid.m_edges[:-1]
    m = np.tile(grid.m_centres[:, None], (1, 2)).astype(float)
    w = np.zeros((nb, 2))
    filled = N > 0
    w[filled] = 0.5 * N[filled, None]
    uR = prof.uR
    p = prof.lam + 1.0
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        u_med = np.where(np.abs(p) < 1e-12, np.sqrt(uR), (0.5 * (1.0 + uR ** p)) ** (1.0 / p))
    pl = filled & ~prof.delta
    lo = int_pow(prof.lam[pl] + 3.0, 1.0, u_med[pl]) / int_pow(prof.lam[pl], 1.0, u_med[pl])
    m[pl, 0] = mL[pl] * lo
    # the upper node takes the rest of the mass exactly
    m[pl, 1] = 2.0 * M[pl] / N[pl] - m[pl, 0]
    sp = filled & prof.delta
    m[sp, :] = (mL[sp] * prof.u_delta[sp] ** 3)[:, None]
    return m, w


@njit(cache=True)
def _pair_terms(m, w, K, m_edges):
    """Mass flows F[k, i] (from bin i to bin k; k = nb is the overflow),
    number losses L[i] and number gains H[k, i] per unit donor weight."""
    nb = m.shape[0]
    F = np.zeros((nb + 1, nb + 1))
    L = np.zeros(nb)
    H = np.zeros((nb + 1, nb))
    P = 2 * nb
    for p in range(P):
        i, a = p // 2, p % 2
        wp = w[i, a]
        if wp <= 0.0:
            continue
        mp = m[i, a]
        for q in range(P):
            j, b = q // 2, q % 2
            wq = w[j, b]
            if wq <= 0.0:
                continue
            mq = m[j, b]
            r = 0.5 * K[p, q] * wp * wq
            if r <= 0.0:
                continue
            mt = mp + mq
            # bin of the product
            if mt >= m_edges[nb]:
                k = nb
            else:
                lo, hi = 0, nb
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    if m_edges[mid] <= mt:
                        lo = mid
                    else:
                        hi = mid
                k = lo
            L[i] += r
            L[j] += r
            if k != i:
                F[k, i] += r * mp
            if k != j:
                F[k, j] += r * mq
            H[k, i] += r * mp / mt
            H[k, j] += r * mq / mt
    return F, L, H


def _terms(grid, N, M, kernel):
    m, w = bin_nodes(grid, N, M)
    mf = m.reshape(-1)
    K = np.asarray(kernel(mf[:, None], mf[None, :]), float)
    if K.ndim == 0:
        K = np.full((mf.size, mf.size), float(K))
    return _pair_terms(m, w, np.ascontiguousarray(K), grid.m_edges)


def _patankar(z_old, z_den, F, h):
    """Solve z = z_old + h [F z/z_den - diag(colsum F) z/z_den] (columns of
    the system matrix sum to one: conservative and positive)."""
    n = z_old.size
    # bins below 1e-30 of the total do not donate (and 1/z would overflow)
    tiny = 1e-30 * max(z_den.sum(), 1e-300)
    inv = np.zeros(n)
    big = z_den > tiny
    inv[big] = 1.0 / z_den[big]
    A = -h * F * inv[None, :]
    A[np.diag_indices(n)] = 1.0 + h * F.sum(axis=0) * inv
    return np.maximum(np.linalg.solve(A, z_old), 0.0)


def _number(N_old, N_den, theta, L, H, h, nb):
    """Number after a Patankar step: losses implicit in N, gains carried by
    the delivered mass (donor ratios theta). Returns (N[nb], N_overflow)."""
    gain = H @ theta[:nb]
    with np.errstate(divide="ignore", invalid="ignore"):
        loss = np.where(N_den > 0, L / N_den, 0.0)
    N = (N_old + h * gain[:nb]) / (1.0 + h * loss)
    return N, h * gain[nb]


def coagulate(grid, N, M, kernel, dt, errmax=1e-3, atol=1e-8, h0=None, max_steps=100000):
    """Coagulate (N, M) over dt with kernel(m1, m2) [cm^3 s^-1] (vectorised
    in grain masses). The local error of every bin, relative to its value or
    to ``atol`` times the total (number or mass), is held below ``errmax``.
    Returns (N, M, overflow_N, overflow_M, n_steps)."""
    nb = grid.n_bins
    N = np.asarray(N, float).copy()
    z = np.append(np.asarray(M, float), 0.0)          # bin masses + overflow
    oN = 0.0
    t = 0.0
    h = dt if h0 is None else min(h0, dt)
    steps = 0
    while t < dt * (1.0 - 1e-12) and steps < max_steps:
        h = min(h, dt - t)
        F0, L0, H0 = _terms(grid, N, z[:nb], kernel)
        if not F0.any() and not L0.any():
            break
        # stage 1 (first order): rates at the old state
        z1 = _patankar(z, z, F0, h)
        with np.errstate(divide="ignore", invalid="ignore"):
            th1 = np.where(z > 0, z1 / z, 0.0)
        N1, oN1 = _number(N, N, th1, L0, H0, h, nb)
        N1, M1 = realisable(grid, N1, z1[:nb])
        # stage 2: mean of the rates, denominators at stage 1
        F1, L1, H1 = _terms(grid, N1, M1, kernel)
        z1[:nb] = M1
        Fm, Lm, Hm = 0.5 * (F0 + F1), 0.5 * (L0 + L1), 0.5 * (H0 + H1)
        z2 = _patankar(z, z1, Fm, h)
        with np.errstate(divide="ignore", invalid="ignore"):
            th2 = np.where(z1 > 0, z2 / z1, 0.0)
        N2, oN2 = _number(N, N1, th2, Lm, Hm, h, nb)
        scale_z = np.maximum(np.maximum(z, z2), atol * max(z.sum(), 1e-300))
        scale_N = np.maximum(np.maximum(N, N2), atol * max(N.sum(), 1e-300))
        err = max(np.max(np.abs(z2 - z1) / scale_z),
                  np.max(np.abs(N2 - N1) / scale_N))
        if err <= errmax or h <= 1e-12 * dt:
            t += h
            steps += 1
            N, Mb = realisable(grid, N2, z2[:nb])
            z = np.append(Mb, z2[nb])
            oN += oN2
        fac = 0.9 * math.sqrt(errmax / max(err, 1e-300))
        h *= min(5.0, max(0.2, fac))
    return N, z[:nb], oN, z[nb], steps
