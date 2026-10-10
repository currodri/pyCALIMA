"""Dust temperature of every bin from gas-grain collisions and continuum radiation.

Each grain of radius a balances collisional heating by the gas against its
net thermal emission (Omukai 2000; Chiaki et al. 2015, Eqs. 5-6), with the
radiation field a blackbody at T_rad (the CMB) and the continuum escape
fraction beta:

    pi a^2 F alpha_T 2 k (T - T_d) = 4 pi a^2 sigma beta [<Q>(a, T_d) T_d^4 - <Q>(a, T_rad) T_rad^4].

As beta -> 0 (optically thick) or F -> infinity, T_d -> T; with no gas,
T_d -> T_rad. The left side falls and the right side rises with T_d, so the
root is bracketed by T and T_rad and found by vectorised bisection in ln T_d.

``escape_fraction`` is beta = min(1, tau^-2) (Omukai 2000).
"""

from __future__ import annotations

import numpy as np

from pycalima.collapse.gas_grain import KB_CGS

SIGMA_SB = 5.670374419e-5


def escape_fraction(tau):
    return np.minimum(1.0, np.asarray(tau, dtype=float) ** -2.0)


def dust_temperature(table, a, T, F, T_rad, beta=1.0, alpha_T=1.0, n_iter=60):
    """T_d of grains of radii ``a`` [cm] of one material (``optics.PlanckTable``)."""
    a = np.asarray(a, dtype=float)
    if beta <= 0.0:          # no radiative losses: the grains sit at the gas temperature
        return np.full(a.shape, float(T))
    lo = np.full(a.shape, np.log(min(T, T_rad)))
    hi = np.full(a.shape, np.log(max(T, T_rad)))
    if hi[0] == lo[0]:
        return np.exp(lo)
    absorbed = table(a, T_rad) * T_rad ** 4
    gain = F * alpha_T * KB_CGS / (2.0 * SIGMA_SB * beta)    # per 4 pi a^2 sigma beta
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        Td = np.exp(mid)
        f = gain * (T - Td) - (table(a, Td) * Td ** 4 - absorbed)
        lo = np.where(f > 0.0, mid, lo)
        hi = np.where(f > 0.0, hi, mid)
    return np.exp(0.5 * (lo + hi))
