"""Mie efficiencies of a coated sphere, by a recursion that is stable for absorbing layers.

A core of radius ``a_core`` and index ``m_core`` inside a mantle of outer radius
``a`` and index ``m_mantle`` (relative to the medium). Indices may be given as
n - ik or n + ik: the absorbing sign is taken (the convention here is n + ik, as
in Bohren & Huffman 1983).

Bohren & Huffman's BHCOAT recurs the Riccati-Neumann functions chi_n(m2 x)
upwards, which breaks down once abs(Im m) x exceeds a few tens (it returns NaN or
Q_sca of several times the geometric limit for 1 um carbon grains in the far
UV). Here the coefficients use only logarithmic derivatives and ratios of
Riccati-Bessel functions (Yang 2003, Appl. Opt. 42, 1710):

- D1_n(z) = psi_n'/psi_n by downward recurrence, D3_n(z) = xi_n'/xi_n upwards;
- in the mantle, the log-derivative is carried from the core boundary z1 = m2 x
  to the surface z2 = m2 y through Q_n = R_n(z1) / R_n(z2), R_n = psi_n / xi_n,
  recurred from the closed form Q_0, which cannot overflow;
- a_n and b_n are the homogeneous-sphere expressions with the carried
  log-derivatives H^a, H^b in place of D_n(m y).

Equal core and mantle indices give the homogeneous sphere, and a vanishing
mantle the core sphere (tests compare both with ``miepython``). The asymmetry
parameter g comes from the same coefficients (BH eq. 4.62).
"""
from __future__ import annotations

import numpy as np


def _absorbing(m):
    m = np.asarray(m, complex)
    return m.real + 1j * np.abs(m.imag)


def _d1_down(z, n_max):
    """psi_n'/psi_n, n = 0..n_max, of every z (shape (nz,)), downward from D = 0."""
    n_start = int(max(n_max, np.abs(z).max()) + 16)
    d = np.zeros(z.shape, complex)
    out = np.empty((n_max + 1,) + z.shape, complex)
    for n in range(n_start, 0, -1):
        d = n / z - 1.0 / (d + n / z)
        if n - 1 <= n_max:
            out[n - 1] = d
    return out


def _d3_up(z, n_max):
    """xi_n'/xi_n, xi_n = psi_n - i chi_n, n = 0..n_max (xi_0 = -i e^{iz})."""
    out = np.empty((n_max + 1,) + z.shape, complex)
    out[0] = 1j
    for n in range(1, n_max + 1):
        out[n] = 1.0 / (n / z - out[n - 1]) - n / z
    return out


def _q_ratio(z1, z2, d1a, d3a, d1b, d3b, n_max):
    """Q_n = [psi_n(z1)/xi_n(z1)] / [psi_n(z2)/xi_n(z2)], n = 1..n_max."""
    w1, w2 = np.exp(2j * z1), np.exp(2j * z2)
    q = (w1 - 1.0) / (w2 - 1.0) * np.exp(2j * (z2 - z1))     # n = 0
    out = np.empty((n_max,) + z1.shape, complex)
    for n in range(1, n_max + 1):
        q = q * ((d3a[n] + n / z1) / (d1a[n] + n / z1)) / ((d3b[n] + n / z2) / (d1b[n] + n / z2))
        out[n - 1] = q
    return out


def _coefficients(x, y, m1, m2, n_max, n_stop):
    """a_n, b_n, shape (n_max, nz), for size parameters x (core) and y (total);
    column j is valid up to order n_stop[j] (above it the real-argument recursion
    is frozen, as chi_n(y) grows without bound for n >> y)."""
    z0, z1, z2 = m1 * x, m2 * x, m2 * y
    d1_core = _d1_down(z0, n_max)
    d1_z1, d3_z1 = _d1_down(z1, n_max), _d3_up(z1, n_max)
    d1_z2, d3_z2 = _d1_down(z2, n_max), _d3_up(z2, n_max)
    Q = _q_ratio(z1, z2, d1_z1, d3_z1, d1_z2, d3_z2, n_max)
    n = np.arange(1, n_max + 1)[:, None]
    D0, D1a, D3a, D1b, D3b = d1_core[1:], d1_z1[1:], d3_z1[1:], d1_z2[1:], d3_z2[1:]
    # TE (a_n): H_eff = (m2/m1) D1(m1 x) at the interface; TM (b_n): (m1/m2) D1(m1 x)
    G1, G2 = m2 * D0 - m1 * D1a, m2 * D0 - m1 * D3a
    Ha = (G2 * D1b - Q * G1 * D3b) / (G2 - Q * G1)
    G1, G2 = m1 * D0 - m2 * D1a, m1 * D0 - m2 * D3a
    Hb = (G2 * D1b - Q * G1 * D3b) / (G2 - Q * G1)
    # Riccati-Bessel functions of the real y, upwards (stable for n <~ y + 4 y^1/3)
    psi = np.empty((n_max + 2,) + y.shape)
    chi = np.empty((n_max + 2,) + y.shape)
    psi[0], psi[1] = np.cos(y), np.sin(y)          # index k holds order k - 1
    chi[0], chi[1] = -np.sin(y), np.cos(y)
    for k in range(2, n_max + 2):
        live = k - 1 <= n_stop
        psi[k] = np.where(live, (2 * k - 3) / y * psi[k - 1] - psi[k - 2], psi[k - 1])
        chi[k] = np.where(live, (2 * k - 3) / y * chi[k - 1] - chi[k - 2], chi[k - 1])
    xi = psi - 1j * chi
    fa = Ha / m2 + n / y
    fb = m2 * Hb + n / y
    a = (fa * psi[2:] - psi[1:-1]) / (fa * xi[2:] - xi[1:-1])
    b = (fb * psi[2:] - psi[1:-1]) / (fb * xi[2:] - xi[1:-1])
    return a, b


def efficiencies(a_core, a, lam, m_core, m_mantle):
    """(Q_abs, Q_sca, g) of coated spheres; lengths in the same units. ``lam`` and
    the indices may be arrays of one shape (one sphere at many wavelengths, say);
    the radii are scalars or broadcast with them."""
    lam = np.atleast_1d(np.asarray(lam, float))
    shape = np.broadcast(lam, np.asarray(m_core), np.asarray(m_mantle), np.asarray(a), np.asarray(a_core)).shape
    m1 = np.broadcast_to(_absorbing(m_core), shape).ravel()
    m2 = np.broadcast_to(_absorbing(m_mantle), shape).ravel()
    y = np.broadcast_to(2.0 * np.pi * np.asarray(a, float) / lam, shape).ravel()
    x = np.broadcast_to(2.0 * np.pi * np.asarray(a_core, float) / lam, shape).ravel().copy()
    no_core = x <= 0.0                              # a homogeneous mantle sphere
    x[no_core] = 1e-3 * y[no_core]
    m1 = np.where(no_core, m2, m1)
    n_stop = (y + 4.0 * y ** (1.0 / 3.0) + 2.0).astype(int)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        an, bn = _coefficients(x, y, m1, m2, int(n_stop.max()), n_stop)
    n = np.arange(1, an.shape[0] + 1, dtype=float)[:, None]
    keep = n <= n_stop[None, :]
    an, bn = np.where(keep, an, 0.0), np.where(keep, bn, 0.0)
    q_ext = 2.0 / y ** 2 * np.sum((2 * n + 1) * (an + bn).real, axis=0)
    q_sca = 2.0 / y ** 2 * np.sum((2 * n + 1) * (abs(an) ** 2 + abs(bn) ** 2), axis=0)
    t1 = n[:-1] * (n[:-1] + 2) / (n[:-1] + 1) * (an[:-1] * np.conj(an[1:]) + bn[:-1] * np.conj(bn[1:])).real
    t2 = (2 * n + 1) / (n * (n + 1)) * (an * np.conj(bn)).real
    g = np.where(q_sca > 0, 4.0 / (y ** 2 * np.maximum(q_sca, 1e-300)) * (t1.sum(axis=0) + t2.sum(axis=0)), 0.0)
    out = (q_ext - q_sca, q_sca, g)
    return tuple(v.reshape(shape) for v in out)
