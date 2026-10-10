"""Grain-grain relative velocities and collision energetics, as matrices over bins.

Components, added in quadrature (Guillet et al. 2020, A&A 643, A17, Sect. 3):

* Brownian motion, dv = (8 k T / pi m_red)**1/2;
* turbulence, Ormel & Cuzzi (2007), their full closed-form expression
  (continuous across the tightly coupled, intermediate and heavy regimes),
  driven at a scale L with velocity v_g (Guillet+20 and Marchand+21 take the
  Jeans length and v_g = c_s);
* ambipolar drift of charged grains through the neutrals (Guillet+20, Eq. 18):
  a grain with Hall factor Gamma = omega_c t_s moves at
  -Gamma**2/(1+Gamma**2) V_AD along the drift and Gamma/(1+Gamma**2) V_AD across it.

Stopping times are Epstein, t_s = rho_s a / (rho_g v_th), v_th = (8 k T / pi mu m_H)**1/2.

With ``y_star="approx"`` the result matches ``solvers.grain_dynamics._ormel2007``
(the RAMSES mirror) for the same driving (v_g, tau_L, tau_eta);
``tests/test_collapse_velocities.py`` holds the two together.
"""

from __future__ import annotations

import math

import numpy as np

KB_CGS = 1.380649e-16
MH_CGS = 1.6735575e-24
G_CGS = 6.67430e-8
C_CGS = 2.99792458e10
E_STATC = 4.8032047e-10


def thermal_speed(T, mu):
    return math.sqrt(8.0 * KB_CGS * T / (math.pi * mu * MH_CGS))


def sound_speed(T, mu, gamma=5.0 / 3.0):
    return math.sqrt(gamma * KB_CGS * T / (mu * MH_CGS))


def jeans_length(T, rho_gas, mu, gamma=5.0 / 3.0):
    """L_J = c_s (pi / G rho)**1/2."""
    return sound_speed(T, mu, gamma) * math.sqrt(math.pi / (G_CGS * rho_gas))


def reynolds_marchand21(nH):
    """Re = 6.2e7 (n_H / 1e5 cm^-3)**1/2 (Marchand+21, Eq. 49; Guillet+20)."""
    return 6.2e7 * math.sqrt(nH / 1e5)


def epstein_stopping_time(a, rho_s, T, rho_gas, mu):
    return rho_s * a / (rho_gas * thermal_speed(T, mu))


def brownian_dv(T, m):
    m = np.asarray(m, dtype=float)
    return np.sqrt(8.0 * KB_CGS * T * (m[:, None] + m[None, :]) / (math.pi * m[:, None] * m[None, :]))


def ormel_cuzzi_y_star(St):
    """y* = t* / t_s, the boundary between class I and class II eddies in units of
    the stopping time: root in (0, 1.6] of (2/3) y (y-1)**2 - 1/(1+y) = -St/(1+St)
    (Ormel & Cuzzi 2007, Eq. 21d; the left side increases monotonically). Tends to
    y_a = 1.6 for St << 1 and to 0 for St >> 1. Vectorised bisection."""
    St = np.asarray(St, dtype=float)
    rhs = -St / (1.0 + St)
    lo, hi = np.zeros_like(St), np.full_like(St, 1.6)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        f = 2.0 / 3.0 * mid * (mid - 1.0) ** 2 - 1.0 / (1.0 + mid) - rhs
        lo = np.where(f < 0.0, mid, lo)
        hi = np.where(f < 0.0, hi, mid)
    return 0.5 * (lo + hi)


def ormel_cuzzi_dv(t_s, v_g, tau_L, tau_eta, y_star="exact"):
    """Turbulent relative velocity of every pair, from the full closed-form
    expression of Ormel & Cuzzi (2007, Eqs. 16-18), continuous in the stopping
    times, with V_g**2 = 3/2 v_g**2 (as ``_ormel2007``):

        dV**2 = dV_I**2 + dV_II**2,
        dV_I**2 / V_g**2 = (St1 - St2)/(St1 + St2) [St1**2/(St12* + St1) - St1**2/(1 + St1) - (1 <-> 2)],
        dV_II**2 / V_g**2 = sum_i [St12* - Re**-1/2 + St_i**2/(St_i + St12*) - St_i**2/(St_i + Re**-1/2)],

    St = t_s / tau_L, Re**-1/2 = tau_eta / tau_L, St12* = max(St1*, St2*)
    clipped to [Re**-1/2, 1], with St* = y*(St) St below St = 1 and 1 above. ``y_star``: ``"exact"`` (Eq. 21d) or
    ``"approx"`` (y* = y_a = 1.6, their Sect. 3.2.1; < 10 % different). Their
    regime formulas (Eqs. 26-29) are the limits St12* = Re**-1/2, the fully
    intermediate regime, and St12* = 1; switching between them, as
    ``_ormel2007`` once did, leaves jumps of up to x2.4 at the boundaries.
    """
    t_s = np.asarray(t_s, dtype=float)
    St = t_s / tau_L
    Se = tau_eta / tau_L
    ys = ormel_cuzzi_y_star(St) if y_star == "exact" else np.full_like(St, 1.6)
    # t* = t_L once t_s >= t_L (their Sect. 3.2.1): all eddies are class II.
    # Continuous, since y*(St = 1) = 1 exactly
    Sstar = np.where(St >= 1.0, 1.0, ys * St)
    St1, St2 = St[:, None], St[None, :]
    S12 = np.clip(np.maximum(Sstar[:, None], Sstar[None, :]), Se, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cls1 = lambda S: S ** 2 / (S12 + S) - S ** 2 / (1.0 + S)
        dI = np.where(St1 + St2 > 0.0, (St1 - St2) / (St1 + St2) * (cls1(St1) - cls1(St2)), 0.0)
    cls2 = lambda S: S12 - Se + S ** 2 / (S + S12) - S ** 2 / (S + Se)
    dII = cls2(St1) + cls2(St2)
    return np.sqrt(1.5 * v_g ** 2 * np.maximum(dI + dII, 0.0))


def ambipolar_dv(t_s, m, Z, B, V_AD):
    """Relative ambipolar drift velocity of every pair (Guillet+20, Eq. 18).

    Gamma = omega_c t_s with omega_c = Z e B / (m c); the sign of Z sets the
    sense of the gyration, so oppositely charged grains drift apart across the field.
    """
    t_s, m, Z = (np.asarray(v, dtype=float) for v in (t_s, m, Z))
    Gam = Z * E_STATC * B / (m * C_CGS) * t_s
    par = Gam ** 2 / (1.0 + Gam ** 2)
    perp = Gam / (1.0 + Gam ** 2)
    return V_AD * np.sqrt((par[:, None] - par[None, :]) ** 2 + (perp[:, None] - perp[None, :]) ** 2)


def reduced_mass(m):
    m = np.asarray(m, dtype=float)
    return m[:, None] * m[None, :] / (m[:, None] + m[None, :])


def coulomb_factor(a, m, Z, dv):
    """Coulomb focusing of point charges, max(1 - E_C / E_kin, 0), with
    E_C = Z_i Z_j e**2 / (a_i + a_j) and E_kin = m_red dv**2 / 2 (as in
    Vallucci-Goy et al. 2024): 0 for a barrier the pair cannot climb, > 1 for
    opposite charges."""
    a, Z = np.asarray(a, dtype=float), np.asarray(Z, dtype=float)
    E_C = Z[:, None] * Z[None, :] * E_STATC ** 2 / (a[:, None] + a[None, :])
    E_kin = 0.5 * reduced_mass(m) * dv ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(E_kin > 0.0, 1.0 - E_C / E_kin, np.where(E_C > 0.0, 0.0, 1.0))
    return np.maximum(f, 0.0)


def rolling_energy(gamma_s, a_mono, xi_crit=2e-7):
    """E_roll = 6 pi**2 xi_crit gamma_s a_mu for monomers of radius 2 a_mu
    (Dominik & Tielens 1997; Marchand+21, Eq. 51). a_mono: reduced monomer radius."""
    return 6.0 * math.pi ** 2 * xi_crit * gamma_s * a_mono
