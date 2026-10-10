"""Collapse histories n_H(t): pressure-free free fall (Guillet et al. 2020).

A uniform sphere collapsing from rest has x = R / R_0 = cos**2 beta at

    t / tau_ff = (2 / pi) (beta + sin beta cos beta),  tau_ff = (3 pi / 32 G rho_0)**1/2,

the solution of Guillet+20's dx/dt = -(pi / 2 tau_ff) (1/x - 1)**1/2, with
n_H = n_H,0 x**-3. Their Appendix A shows that a Larson-type collapse gives the
same grain size distribution at equal density, so this history is enough to
compare against.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

G_CGS = 6.67430e-8
MH_CGS = 1.6735575e-24


def freefall_time(nH, mu):
    """tau_ff = (3 pi / 32 G rho)**1/2 with rho = mu m_H n_H."""
    return math.sqrt(3.0 * math.pi / (32.0 * G_CGS * mu * MH_CGS * nH))


def freefall_time_to_density(nH, nH0, mu):
    """Time for a sphere starting at rest at nH0 to reach nH (pressure-free)."""
    x = (nH0 / nH) ** (1.0 / 3.0)
    beta = math.acos(math.sqrt(x))
    return freefall_time(nH0, mu) * 2.0 / math.pi * (beta + math.sin(beta) * math.cos(beta))


def freefall_density(t, nH0, mu):
    """n_H at time t of the pressure-free collapse from rest at nH0 (t < tau_ff)."""
    tff = freefall_time(nH0, mu)
    if not 0.0 <= t < tff:
        raise ValueError(f"t = {t:g} s outside [0, tau_ff = {tff:g} s)")
    target = 0.5 * math.pi * t / tff
    beta = brentq(lambda b: b + math.sin(b) * math.cos(b) - target, 0.0, 0.5 * math.pi)
    return nH0 * math.cos(beta) ** -6


def freefall_steps(nH0, nH_end, mu, dlnn=0.02):
    """Step edges (t [s], n_H) uniform in ln n_H, from nH0 to nH_end."""
    n = np.exp(np.linspace(math.log(nH0), math.log(nH_end),
                           max(int(math.ceil(math.log(nH_end / nH0) / dlnn)), 1) + 1))
    t = np.array([freefall_time_to_density(v, nH0, mu) for v in n])
    return t, n
