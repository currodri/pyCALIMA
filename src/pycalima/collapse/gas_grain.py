"""Gas-grain collisional energy transfer.

A neutral gas particle of species s hitting a grain at gas temperature T and
dust temperature T_d carries on average 2 k (T - T_d) times the thermal
accommodation coefficient alpha_T (Hollenbach & McKee 1979). Per grain of
radius a,

    H(a) = pi a^2 F alpha_T 2 k (T - T_d),   F = sum_s n_s (8 k T / pi m_s)^1/2,

and the gas loses Lambda_gd = sum_k n_k H(a_k) per unit volume (heats, if
negative). With alpha_T = 1 and s = H, H2, He this is Chiaki et al. (2015,
Eq. 5): F = [n(H) + n(H2)/sqrt 2 + n(He)/2] (8 k T / pi m_H)^1/2.

Electrons and ions are left out: in the dense, UV-shielded gas where dust
cooling matters their fraction is < 1e-6 (``charge.charge_balance``), so they
change Lambda_gd by less than that.
"""

from __future__ import annotations

import math

import numpy as np

KB_CGS = 1.380649e-16
MH_CGS = 1.6735575e-24


def collision_flux_factor(T, n_H, n_H2, n_He):
    """F = sum_s n_s (8 k T / pi m_s)^1/2 [cm^-2 s^-1] for H, H2 and He."""
    v_H = math.sqrt(8.0 * KB_CGS * T / (math.pi * MH_CGS))
    return v_H * (n_H + n_H2 / math.sqrt(2.0) + n_He / 2.0)


def heating_per_grain(a, T, T_d, F, alpha_T=1.0):
    """Energy the gas gives one grain per unit time, H(a) [erg s^-1]."""
    return math.pi * np.asarray(a) ** 2 * F * alpha_T * 2.0 * KB_CGS * (T - np.asarray(T_d))


def cooling_rate(n_grains, a, T, T_d, F, alpha_T=1.0):
    """Lambda_gd = sum_k n_k H(a_k) [erg cm^-3 s^-1]; > 0 when the gas cools."""
    return float(np.sum(np.asarray(n_grains) * heating_per_grain(a, T, T_d, F, alpha_T)))
