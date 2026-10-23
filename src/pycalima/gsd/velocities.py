"""Grain-grain relative velocities for the collision operator: continuous in
grain size and gas state.

``RelativeVelocity`` gives, for every pair of node radii, the rms random
relative speed dV and a systematic relative drift mu:

* dV adds in quadrature Brownian motion, (3 k T / m_red)^1/2, and turbulence
  from the full closed form of Ormel & Cuzzi (2007, eqs. 16-18;
  ``collapse.velocities.ormel_cuzzi_dv``), continuous across the tightly
  coupled, intermediate and heavy regimes, with Epstein stopping times;
* mu = ``|v_d(a1) - v_d(a2)|`` for a drift velocity v_d(a) of each grain along a
  common direction (e.g. radiation pressure; ``drift``), 0 by default.

The relative velocity is then an isotropic Gaussian of rms dV shifted by mu,
whose speed follows the skewed Maxwellian of Guillet et al. (2020, A&A 643,
A17, App. C),

    f(v) = (alpha / (sqrt(pi) mu)) v [exp(-alpha^2 (v - mu)^2) - exp(-alpha^2 (v + mu)^2)],

alpha = (3/2)^1/2 / dV. Collisions happen with the flux v f(v), so a
threshold v_c splits them smoothly: the fraction of the collision rate with
v > v_c is ``flux_above``. Shattering is represented by n_q channels of
equal flux above v_shat, each at its flux quantile speed
(``flux_quantile``); sticking by the flux below v_coag. Both are continuous
in the node radii and the gas state, unlike fixed velocity directions or a
step in v(a), which make the sub-step control chatter.

Normalised variables: u = alpha v, s = alpha mu. The flux above u is::

    C(u; s) = [J(u; s) - J(u; -s)] / [J(0; s) - J(0; -s)],
    J(u; c) = int_u^inf v^2 exp(-(v - c)^2) dv
            = e^{-L^2} [L/2 + c] + (sqrt(pi)/4) (1 + 2 c^2) erfc(L),  L = u - c,

evaluated in log space with erfcx in the tail, and with its s -> 0 limit
(1 + u^2) e^{-u^2} for s < 1e-4.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
from scipy.special import erfcx

from pycalima.collapse import velocities as cvel
from pycalima.collapse._jit import njit

_SQPI = math.sqrt(math.pi)


def _log_J(u, c):
    """ln J(u; c), J = int_u^inf v^2 exp(-(v - c)^2) dv (u >= 0, J > 0)."""
    L = u - c
    out = np.empty(np.broadcast(u, c).shape)
    L, c = np.broadcast_arrays(L, c)
    pos = L >= 0.0
    # L >= 0: J = e^{-L^2} [L/2 + c + (sqrt(pi)/4) (1 + 2 c^2) erfcx(L)]
    br = L[pos] / 2.0 + c[pos] + _SQPI / 4.0 * (1.0 + 2.0 * c[pos] ** 2) * erfcx(L[pos])
    out[pos] = -L[pos] ** 2 + np.log(np.maximum(br, 1e-300))
    # L < 0: erfc(L) = 2 - erfc(-L), no underflow
    Ln, cn = L[~pos], c[~pos]
    from scipy.special import erfc
    J = np.exp(-Ln ** 2) * (Ln / 2.0 + cn) + _SQPI / 4.0 * (1.0 + 2.0 * cn ** 2) * erfc(Ln)
    out[~pos] = np.log(np.maximum(J, 1e-300))
    return out


def log_flux_above(u, s):
    """ln C(u; s): ln of the fraction of the collision flux with normalised
    speed above u, for normalised drift s (arrays broadcast)."""
    u, s = np.broadcast_arrays(np.maximum(np.asarray(u, float), 0.0),
                               np.maximum(np.asarray(s, float), 0.0))
    out = np.empty(u.shape)
    small = s < 1e-4
    # s -> 0: C = (1 + u^2) exp(-u^2)
    out[small] = np.log1p(u[small] ** 2) - u[small] ** 2
    if (~small).any():
        uu, ss = u[~small], s[~small]
        lp, lm = _log_J(uu, ss), _log_J(uu, -ss)
        l0p, l0m = _log_J(0.0 * uu, ss), _log_J(0.0 * uu, -ss)
        num = lp + np.log1p(-np.minimum(np.exp(lm - lp), 1.0 - 1e-16))
        den = l0p + np.log1p(-np.minimum(np.exp(l0m - l0p), 1.0 - 1e-16))
        out[~small] = np.minimum(num - den, 0.0)
    return out


def flux_above(u, s):
    return np.exp(log_flux_above(u, s))


def _log_erfc_diff(u, s):
    """ln [erfc(u - s) - erfc(u + s)], s > 0, stable in the tail."""
    from scipy.special import erfc

    L1, L2 = u - s, u + s
    out = np.empty(np.broadcast(u, s).shape)
    L1, L2 = np.broadcast_arrays(L1, L2)
    pos = L1 >= 0.0
    # both positive: erfc(L) = erfcx(L) e^{-L^2}
    l1 = -L1[pos] ** 2 + np.log(erfcx(L1[pos]))
    l2 = -L2[pos] ** 2 + np.log(erfcx(L2[pos]))
    out[pos] = l1 + np.log1p(-np.minimum(np.exp(l2 - l1), 1.0 - 1e-16))
    d = erfc(L1[~pos]) - erfc(L2[~pos])
    out[~pos] = np.log(np.maximum(d, 1e-300))
    return out


def inverse_moment_above(u, s):
    """(int_u^inf f(v)/v dv) / <v> in normalised units (alpha = 1), the term
    of the Coulomb-focusing integral: 2 e^{-u^2} / (sqrt(pi) <u>) as s -> 0,
    else [erfc(u - s) - erfc(u + s)] / (2 s <u>)."""
    u, s = np.broadcast_arrays(np.maximum(np.asarray(u, float), 0.0),
                               np.maximum(np.asarray(s, float), 0.0))
    out = np.empty(u.shape)
    small = s < 1e-4
    mbar = mean_speed_normalised(s)
    out[small] = 2.0 / _SQPI * np.exp(-u[small] ** 2) / mbar[small]
    if (~small).any():
        out[~small] = np.exp(_log_erfc_diff(u[~small], s[~small])) / (2.0 * s[~small] * mbar[~small])
    return out


def coulomb_rate_factor(alpha, s, vC2, v_lo, v_hi=np.inf):
    """Fraction of the collision flux sigma <v> whose impact speed (at
    contact) lies in [v_lo, v_hi], with Coulomb focusing (vC2 < 0,
    attraction) or a barrier (vC2 > 0, repulsion) of point charges:
    sigma_eff = sigma (1 - vC2 / v^2) and v_imp^2 = v^2 - vC2,
    vC2 = 2 Z1 Z2 e^2 / ((a1 + a2) m_red). Arrays broadcast."""
    alpha, s, vC2, v_lo, v_hi = np.broadcast_arrays(*(np.asarray(x, float)
                                                      for x in (alpha, s, vC2, v_lo, v_hi)))
    uC2 = alpha ** 2 * vC2
    Ua = alpha * np.sqrt(np.maximum(v_lo ** 2 + vC2, 0.0))
    fin = np.isfinite(v_hi)
    Ub = np.where(fin, alpha * np.sqrt(np.maximum(np.where(fin, v_hi, 0.0) ** 2 + vC2, 0.0)), np.inf)
    def part(U):
        inf = ~np.isfinite(U)
        Uf = np.where(inf, 0.0, U)
        val = flux_above(Uf, s) - uC2 * inverse_moment_above(Uf, s)
        return np.where(inf, 0.0, val)
    return np.maximum(part(Ua) - part(Ub), 0.0)


def mean_speed_normalised(s):
    """alpha <v> for normalised drift s (Maxwellian: 2 / sqrt(pi))."""
    s = np.asarray(s, float)
    return np.asarray(cvel.mean_collision_speed(math.sqrt(1.5) + 0 * s, np.inf, s), float)


class _QuantileTable:
    """u(s, y) with C(u; s) = e^-y, tabulated once on s in [0, S_MAX] (in
    asinh s) and z = ln y in [Z_MIN, ln Y_MAX]; bilinear in (asinh s, z)."""

    S_MAX = 1e3
    Y_MAX = 300.0
    Z_MIN = -30.0

    def __init__(self, n_s=321, n_u=4001, n_z=1201):
        self.xs = np.linspace(0.0, math.asinh(self.S_MAX), n_s)
        s = np.sinh(self.xs)
        self.z = np.linspace(self.Z_MIN, math.log(self.Y_MAX), n_z)
        self.U = np.empty((n_s, n_z))
        for k, sk in enumerate(s):
            lo = max(sk - 9.0, 0.0)
            u = lo + np.linspace(0.0, 1.0, n_u) ** 2 * (math.sqrt(self.Y_MAX) + 4.0 + sk - lo)
            yy = -log_flux_above(u, sk)
            ok = yy > 0.0
            zz = np.log(yy[ok])
            zz = np.maximum.accumulate(zz)                       # monotone against round-off
            keep = np.concatenate([[True], np.diff(zz) > 0])
            self.U[k] = np.interp(self.z, zz[keep], u[ok][keep])

    def __call__(self, s, y):
        s, y = np.broadcast_arrays(np.asarray(s, float), np.asarray(y, float))
        out = _bilinear(self.U, self.xs[-1], self.Z_MIN, self.z[-1], self.S_MAX, self.Y_MAX,
                        np.ascontiguousarray(s).ravel(), np.ascontiguousarray(y).ravel())
        return out.reshape(s.shape)


@njit(cache=True)
def _bilinear(U, xs_max, z_min, z_max, s_max, y_max, s, y):
    n_s, n_z = U.shape
    out = np.empty(s.size)
    for k in range(s.size):
        sk = min(max(s[k], 0.0), s_max)
        zk = math.log(min(max(y[k], math.exp(z_min)), y_max))
        fs = min(max(math.asinh(sk) / xs_max * (n_s - 1), 0.0), n_s - 1 - 1e-9)
        fz = min(max((zk - z_min) / (z_max - z_min) * (n_z - 1), 0.0), n_z - 1 - 1e-9)
        i, j = int(fs), int(fz)
        ts, tz = fs - i, fz - j
        out[k] = ((1 - ts) * ((1 - tz) * U[i, j] + tz * U[i, j + 1])
                  + ts * ((1 - tz) * U[i + 1, j] + tz * U[i + 1, j + 1]))
    return out


_TABLE: Optional[_QuantileTable] = None


def flux_quantile(s, y):
    """Normalised speed u above which a fraction e^-y of the collision flux
    lies, for normalised drift s (tabulated, built once in ~1 s)."""
    global _TABLE
    if _TABLE is None:
        _TABLE = _QuantileTable()
    return _TABLE(s, y)


class RelativeVelocity:
    """Relative-velocity statistics of grain pairs in a gas.

    T [K], n_H [cm^-3], mu: mean molecular weight; v_turb [cm/s] and L [cm]:
    turbulent velocity and driving scale (no turbulence if either is 0);
    sigma_neutral: H/H2 cross section for the kinematic viscosity, nu =
    v_th / (3 n_H sigma). ``drift(a)``: systematic velocity of a grain of
    radius a along a common direction [cm/s]. ``y_star``: Ormel & Cuzzi's
    class boundary, ``"exact"`` (their eq. 21d) or ``"approx"`` (1.6).
    ``charges(a)``: charge points and weights per radius (e.g.
    ``charging.ChargeState.charge_points``) for Coulomb focusing and barriers
    between grains (``coulomb_rate_factor``); None for neutral grains.
    """

    def __init__(self, T, n_H, mu=1.4, v_turb=0.0, L=0.0, sigma_neutral=2.0e-15,
                 drift: Optional[Callable] = None, y_star="exact", brownian=True,
                 charges: Optional[Callable] = None):
        self.T, self.n_H, self.mu = float(T), float(n_H), float(mu)
        self.rho_gas = self.n_H * self.mu * cvel.MH_CGS
        self.v_turb, self.L = float(v_turb), float(L)
        self.sigma_neutral = sigma_neutral
        self.drift = drift
        self.y_star = y_star
        self.brownian = brownian
        self.charges = charges
        if self.v_turb > 0.0 and self.L > 0.0:
            self.tau_L = self.L / self.v_turb
            nu = cvel.thermal_speed(self.T, self.mu) / (3.0 * self.n_H * sigma_neutral)
            self.tau_eta = self.tau_L / math.sqrt(max(self.v_turb * self.L / nu, 1.0))
        else:
            self.tau_L = self.tau_eta = None

    def pair_statistics(self, a, m, rho_s):
        """(dV, mu) [cm/s] for all pairs of grains of radii a and masses m."""
        a = np.asarray(a, float)
        m = np.asarray(m, float)
        dV2 = cvel.brownian_rms(self.T, m) ** 2 if self.brownian else np.zeros((a.size, a.size))
        if self.tau_L is not None:
            t_s = cvel.epstein_stopping_time(a, rho_s, self.T, self.rho_gas, self.mu)
            dV2 = dV2 + cvel.ormel_cuzzi_dv(t_s, self.v_turb, self.tau_L, self.tau_eta,
                                            y_star=self.y_star) ** 2
        if self.drift is None:
            mu = np.zeros_like(dV2)
        else:
            vd = np.asarray(self.drift(a), float)
            mu = np.abs(vd[:, None] - vd[None, :])
        return np.sqrt(dV2), mu

    def collision_speeds(self, dV, mu, v_cut, n_q):
        """Mean collision speed <v>, the flux fraction above v_cut, and n_q
        equal-flux quantile speeds above it (arrays [n_q, ...])."""
        alpha = math.sqrt(1.5) / np.maximum(dV, 1e-300)
        s = alpha * mu
        v_mean = mean_speed_normalised(s) / alpha
        lc = log_flux_above(alpha * v_cut, s)
        ks = (np.arange(n_q) + 0.5) / n_q
        y = -(lc[None] + np.log1p(-ks)[(slice(None),) + (None,) * lc.ndim])
        v_q = flux_quantile(np.broadcast_to(s, y.shape), y) / alpha
        return v_mean, np.exp(lc), v_q


def terminal_drift_velocity(F_rad, a, T_gas, rho_gas, mu=1.4):
    """Drift speed [cm/s] of a grain of radius a [cm] through gas at which the
    collisional drag balances a force F_rad [dyn] (e.g. radiation pressure,
    absorbed power / c). Drag of Draine & Salpeter (1979) for neutral gas,
    F = A v (1 + B v^2)^1/2 with A = (4 pi/3) a^2 rho v_th (Epstein, v_th =
    (8kT/pi m)^1/2) and B = 9 pi m / (128 k T) (the supersonic correction,
    pi a^2 rho v^2 at v >> v_th), solved in closed form. Coulomb drag on
    charged grains in ionised gas is not included (it lowers the drift)."""
    F, a = np.broadcast_arrays(np.asarray(F_rad, float), np.asarray(a, float))
    m = mu * cvel.MH_CGS
    v_th = math.sqrt(8.0 * cvel.KB_CGS * T_gas / (math.pi * m))
    A = 4.0 * math.pi / 3.0 * a ** 2 * rho_gas * v_th
    B = 9.0 * math.pi * m / (128.0 * cvel.KB_CGS * T_gas)
    A2 = A * A
    x = 2.0 * F * F / (A2 + np.sqrt(A2 * A2 + 4.0 * B * A2 * F * F))
    return np.sqrt(x)
