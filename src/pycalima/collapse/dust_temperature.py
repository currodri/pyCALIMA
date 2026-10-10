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

import math

import numpy as np

from pycalima.collapse._jit import njit
from pycalima.collapse.gas_grain import KB_CGS

SIGMA_SB = 5.670374419e-5


def escape_fraction(tau):
    return np.minimum(1.0, np.asarray(tau, dtype=float) ** -2.0)


@njit(cache=True)
def _q(lnQa, la0, dla, lt0, dlt, a, T):
    """<Q>(a, T) from the PlanckTable grid (bilinear in ln a, ln T of ln(<Q>/a))."""
    na, nt = lnQa.shape
    fa = min(max((math.log(a) - la0) / dla, 0.0), na - 1.000001)
    ft = min(max((math.log(T) - lt0) / dlt, 0.0), nt - 1.000001)
    i = int(fa)
    j = int(ft)
    wa = fa - i
    wt = ft - j
    v = ((1 - wa) * (1 - wt) * lnQa[i, j] + wa * (1 - wt) * lnQa[i + 1, j]
         + (1 - wa) * wt * lnQa[i, j + 1] + wa * wt * lnQa[i + 1, j + 1])
    return a * math.exp(v)


@njit(cache=True)
def _solve(lnQa, la0, dla, lt0, dlt, a, T, T_rad, gain, n_iter):
    out = np.empty(a.size)
    llo, lhi = math.log(min(T, T_rad)), math.log(max(T, T_rad))
    for k in range(a.size):
        absorbed = _q(lnQa, la0, dla, lt0, dlt, a[k], T_rad) * T_rad ** 4
        lo, hi = llo, lhi
        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)
            Td = math.exp(mid)
            f = gain * (T - Td) - (_q(lnQa, la0, dla, lt0, dlt, a[k], Td) * Td ** 4 - absorbed)
            if f > 0.0:
                lo = mid
            else:
                hi = mid
        out[k] = math.exp(0.5 * (lo + hi))
    return out


def dust_temperature(table, a, T, F, T_rad, beta=1.0, alpha_T=1.0, n_iter=40):
    """T_d of grains of radii ``a`` [cm] of one material (``optics.PlanckTable``).

    Bisection in ln T_d between T and T_rad; 40 halvings resolve T_d to
    ~1e-12 of ln(T / T_rad)."""
    a = np.atleast_1d(np.asarray(a, dtype=float))
    if beta <= 0.0:          # no radiative losses: the grains sit at the gas temperature
        return np.full(a.shape, float(T))
    if T == T_rad:
        return np.full(a.shape, float(T))
    la, lt = np.log(table.radii), np.log(table.T)
    gain = F * alpha_T * KB_CGS / (2.0 * SIGMA_SB * beta)    # per 4 pi a^2 sigma beta
    return _solve(np.ascontiguousarray(table.lnQ_over_a), la[0], la[1] - la[0], lt[0], lt[1] - lt[0],
                  a, float(T), float(T_rad), float(gain), n_iter)
