"""Metallicity-dependent stellar lifetimes (Raiteri, Villata & Navarro 1996).

    log10(t/yr) = a0(Z) + a1(Z) log10(m) + a2(Z) log10(m)^2

fitted to the Padova tracks for 0.6-120 Msun and 7e-5 <= Z <= 0.03. Z is
clamped to that range. The quadratic turns over near 100 Msun; above the
turning mass the lifetime is held at its minimum so t(m) stays monotonic.
"""

from __future__ import annotations

import numpy as np

Z_MIN, Z_MAX = 7e-5, 0.03


def _coefficients(Z):
    lz = np.log10(np.clip(np.asarray(Z, dtype=float), Z_MIN, Z_MAX))
    a0 = 10.13 + 0.07547 * lz - 0.008084 * lz ** 2
    a1 = -4.424 - 0.7939 * lz - 0.1187 * lz ** 2
    a2 = 1.262 + 0.3385 * lz + 0.05417 * lz ** 2
    return a0, a1, a2


def lifetime(m, Z):
    """Main-sequence plus post-MS lifetime [yr] of a star of mass ``m`` [Msun]."""
    a0, a1, a2 = _coefficients(Z)
    x_turn = -a1 / (2.0 * a2)
    x = np.minimum(np.log10(np.asarray(m, dtype=float)), x_turn)
    return 10.0 ** (a0 + a1 * x + a2 * x ** 2)


def dying_mass(t, Z):
    """Initial mass [Msun] of the stars that die at age ``t`` [yr].

    Inverse of :func:`lifetime`. Ages shorter than the minimum lifetime
    return ``inf`` (no star has died yet).
    """
    a0, a1, a2 = _coefficients(Z)
    lt = np.log10(np.maximum(np.asarray(t, dtype=float), 1.0))
    disc = a1 ** 2 - 4.0 * a2 * (a0 - lt)
    x = (-a1 - np.sqrt(np.maximum(disc, 0.0))) / (2.0 * a2)
    return np.where(disc > 0.0, 10.0 ** x, np.inf)
