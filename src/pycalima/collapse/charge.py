"""Grain charge in UV-shielded gas: exact WDB06-dark P(Z), tables, and closure.

With no photons, the WDB06 charging of ``rt_group_charging`` reduces to
collisional capture: electrons with WD01 sticking and DS87 focusing, singly
charged ions with DS87 focusing (sticking 1), and the WDB06 most negative
charge. Its two remaining hot-gas terms, gas-electron secondaries and electron
pass-through, are left out: that reproduces ``GroupChargingTable.solve`` with
no photons to < 1 % at T <= 1e3 K, and for negative grains up to 3e3 K, but
above 1e3 K the secondaries raise the charge of positive grains by a few %
(3 % at 3e3 K). Detailed balance then
gives the discrete distribution exactly,

    P(Z+1) / P(Z) = xi J(Z, +1) / [s_e(Z+1) J(Z+1, -1)],
    xi = sum_i n_i v_i / (n_e v_e) = sum_i n_i (m_e / m_i)**1/2 / n_e,

so at fixed grain size and material P(Z) depends on (T, xi) only: the ion
masses enter only through xi. ``dark_charge_moments`` sums it exactly (no
Gaussian approximation) and ``DarkChargeTable`` tabulates, per bin, <Z>,
sigma_Z and the mean ion focusing factor <J(Z, +1)>, the grain-assisted
recombination rate per grain divided by n_i v_i pi a**2, on (log T, log xi).

``charge_balance`` closes the system for a population of bins (Marchand et
al. 2021, A&A 649, A50, Sect. 3): ionisation balance

    zeta n_H = k_rec n_i n_e + n_i v_i sum_k n_k pi a_k**2 <J>_k

and neutrality n_i - n_e + sum_k n_k <Z>_k = 0. For a trial xi the first is a
quadratic in n_e; the neutrality residual is then monotonic in xi, so a
bracketed root in log xi is robust, including the grain-poor limit
(n_i -> n_e) where a Newton iteration on the grain potential is ill-posed.

``marchand21_charge_balance`` is their analytic approximation (Eqs. 45-47),
kept as the reference for their Fig. 2.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

from pycalima.collapse._jit import njit
from pycalima.models.dust_charge.shared_physics import (
    _ds87_j_scalar_impl,
    most_negative_allowed_charge_graphite,
    most_negative_allowed_charge_silicate,
)

KB_CGS = 1.380649e-16
ME_CGS = 9.1093837015e-28
MH_CGS = 1.6735575e-24
E2_CGS = 4.8032047e-10 ** 2
ESCAPE_LENGTH_CM = 1e-7          # WD01 electron escape length (as in shared_physics)
P_TAIL = 1e-14                   # the exact sum stops where P falls below this x its maximum

MATERIALS = ("graphite", "silicate")


def most_negative_charge(material, a):
    """WDB06 most negative allowed charge (autoionisation limit), as in rt_group_charging."""
    if material == "graphite":
        return float(most_negative_allowed_charge_graphite(a))
    if material == "silicate":
        return float(most_negative_allowed_charge_silicate(a))
    raise ValueError(f"no charging parameters for material {material!r}; have {MATERIALS}")


@njit(cache=True)
def _s_e(Z, a, zmin, s_const):
    """Electron sticking: constant ``s_const`` if > 0, else WD01 eqs. 27-30
    exactly as ``GroupChargingTable._sticking``."""
    if s_const > 0.0:
        return s_const
    if Z <= zmin:
        return 0.0
    base = 0.5 * (1.0 - math.exp(-a / ESCAPE_LENGTH_CM))
    if Z > 0.0:
        return base
    Nc = 468.0 * (a / 1e-7) ** 3
    return base / (1.0 + math.exp(min(20.0 - Nc, 700.0)))


@njit(cache=True)
def _g(Z, a, T, xi, zmin, s_const):
    """ln P(Z+1) - ln P(Z), decreasing in Z."""
    up = xi * _ds87_j_scalar_impl(Z, 1.0, a, T)
    down = _s_e(Z + 1.0, a, zmin, s_const) * _ds87_j_scalar_impl(Z + 1.0, -1.0, a, T)
    if down <= 0.0:
        return 1e300
    if up <= 0.0:
        return -1e300
    return math.log(up) - math.log(down)


@njit(cache=True)
def dark_charge_moments(a, T, xi, zmin, s_const):
    """Exact (<Z>, sigma_Z, <J(Z,+1)>, <s_e J(Z,-1)>) of the dark P(Z).

    ``zmin``: lowest charge (use -1e9 with ``s_const > 0`` for no limit). In
    steady state xi <J(Z,+1)> = <s_e J(Z,-1)>: equal ion and electron currents.
    """
    lo = zmin if s_const <= 0.0 else -1e9
    # the mode: the smallest integer Z with g(Z) < 0
    if _g(lo, a, T, xi, zmin, s_const) < 0.0:
        mode = lo
    else:
        hi = 0.0
        step = 1.0
        while _g(hi, a, T, xi, zmin, s_const) >= 0.0:
            hi += step
            step *= 2.0
        lo_b = max(lo, -1.0)
        step = 1.0
        while lo_b > lo and _g(lo_b, a, T, xi, zmin, s_const) < 0.0:
            lo_b = max(lo_b - step, lo)
            step *= 2.0
        # g(lo_b) >= 0 > g(hi)
        while hi - lo_b > 1.0:
            mid = math.floor(0.5 * (lo_b + hi))
            if _g(mid, a, T, xi, zmin, s_const) < 0.0:
                hi = mid
            else:
                lo_b = mid
        mode = hi
    cut = math.log(P_TAIL)
    w0 = 0.0
    w1 = 0.0
    w2 = 0.0
    wi = 0.0
    we = 0.0
    # upward from the mode
    lnp = 0.0
    Z = mode
    while True:
        p = math.exp(lnp)
        w0 += p
        w1 += p * Z
        w2 += p * Z * Z
        wi += p * _ds87_j_scalar_impl(Z, 1.0, a, T)
        we += p * _s_e(Z, a, zmin, s_const) * _ds87_j_scalar_impl(Z, -1.0, a, T)
        gz = _g(Z, a, T, xi, zmin, s_const)
        lnp += gz
        Z += 1.0
        if lnp < cut or gz < -700.0:
            break
    # downward
    lnp = 0.0
    Z = mode
    while Z - 1.0 >= lo:
        gz = _g(Z - 1.0, a, T, xi, zmin, s_const)
        lnp -= gz
        Z -= 1.0
        if lnp < cut:
            break
        p = math.exp(lnp)
        w0 += p
        w1 += p * Z
        w2 += p * Z * Z
        wi += p * _ds87_j_scalar_impl(Z, 1.0, a, T)
        we += p * _s_e(Z, a, zmin, s_const) * _ds87_j_scalar_impl(Z, -1.0, a, T)
    zm = w1 / w0
    var = max(w2 / w0 - zm * zm, 0.0)
    return zm, math.sqrt(var), wi / w0, we / w0


class DarkChargeTable:
    """Per-bin <Z>, sigma_Z and ln <J(Z,+1)> on (log10 T, log10 xi).

    ``materials``: one per bin (``"graphite"`` or ``"silicate"``).
    ``s_e``: ``None`` for WD01 sticking with the WDB06 charge limit, or a
    constant (Marchand+21 use 0.5, with no charge limit).
    """

    def __init__(self, a, materials, log_T=np.linspace(0.5, 4.0, 71),
                 log_xi=np.linspace(-4.0, 4.0, 161), s_e=None):
        self.a = np.asarray(a, dtype=float)
        self.log_T = np.asarray(log_T, dtype=float)
        self.log_xi = np.asarray(log_xi, dtype=float)
        nb, nT, nx = self.a.size, self.log_T.size, self.log_xi.size
        s_const = 0.0 if s_e is None else float(s_e)
        self.zmin = np.array([most_negative_charge(m, ak) if s_e is None else -1e9
                              for m, ak in zip(materials, self.a)])
        self.Zmean = np.empty((nb, nT, nx))
        self.sigma = np.empty((nb, nT, nx))
        self.lnJ = np.empty((nb, nT, nx))
        for k in range(nb):
            for i, lt in enumerate(self.log_T):
                for j, lx in enumerate(self.log_xi):
                    zm, sg, ji, _ = dark_charge_moments(self.a[k], 10.0 ** lt, 10.0 ** lx,
                                                        self.zmin[k], s_const)
                    self.Zmean[k, i, j], self.sigma[k, i, j], self.lnJ[k, i, j] = zm, sg, math.log(ji)

    def at_temperature(self, T):
        """``(log_xi, Zmean[k, :], sigma[k, :], lnJ[k, :])`` interpolated linearly in log T."""
        lt = math.log10(T)
        if not self.log_T[0] <= lt <= self.log_T[-1]:
            raise ValueError(f"T = {T:g} K outside the table ({10 ** self.log_T[0]:g}-{10 ** self.log_T[-1]:g} K)")
        i = min(int(np.searchsorted(self.log_T, lt) - 1), self.log_T.size - 2)
        i = max(i, 0)
        w = (lt - self.log_T[i]) / (self.log_T[i + 1] - self.log_T[i])
        mix = lambda y: (1.0 - w) * y[:, i] + w * y[:, i + 1]
        return self.log_xi, mix(self.Zmean), mix(self.sigma), mix(self.lnJ)


def recombination_rate_coefficient(T):
    """Dissociative recombination of molecular ions, 2e-7 (T / 300 K)**-1/2 cm^3 s^-1
    (the value used by Marchand+21, Eq. 28)."""
    return 2e-7 * (T / 300.0) ** -0.5


def charge_balance(table, n_grains, T, nH, zeta, mu_i, k_rec=None):
    """Electron and ion densities and per-bin charges for grains ``n_grains`` [cm^-3].

    One effective singly charged ion of mass ``mu_i m_H``; ``k_rec`` its
    gas-phase recombination coefficient [cm^3 s^-1] (default: Marchand+21).
    Returns a dict: n_e, n_i, xi, Zmean, sigma, Jion (per bin).
    """
    n_grains = np.asarray(n_grains, dtype=float)
    k = recombination_rate_coefficient(T) if k_rec is None else k_rec
    lx, Zt, St, Jt = table.at_temperature(T)
    m_i = mu_i * MH_CGS
    v_i = math.sqrt(8.0 * KB_CGS * T / (math.pi * m_i))
    r_per_xi = math.sqrt(m_i / ME_CGS)
    area = n_grains * math.pi * table.a ** 2
    S = zeta * nH

    def state(lxi):
        j = min(max(int((lxi - lx[0]) / (lx[1] - lx[0])), 0), lx.size - 2)
        w = (lxi - lx[j]) / (lx[j + 1] - lx[j])
        Z = (1.0 - w) * Zt[:, j] + w * Zt[:, j + 1]
        J = np.exp((1.0 - w) * Jt[:, j] + w * Jt[:, j + 1])
        r = 10.0 ** lxi * r_per_xi
        b = r * v_i * np.dot(area, J)
        n_e = 2.0 * S / (b + math.sqrt(b * b + 4.0 * k * r * S))
        return Z, J, r, n_e, w, j

    def resid(lxi):
        Z, _, r, n_e, _, _ = state(lxi)
        return n_e * (r - 1.0) + np.dot(n_grains, Z)

    f0, f1 = resid(lx[0]), resid(lx[-1])
    if not f0 < 0.0 < f1:
        raise ValueError(f"charge balance not bracketed by the table's xi range "
                         f"(residuals {f0:.3g}, {f1:.3g}): extend log_xi")
    lxi = brentq(resid, lx[0], lx[-1], xtol=1e-12, rtol=1e-12)
    Z, J, r, n_e, w, j = state(lxi)
    sig = (1.0 - w) * St[:, j] + w * St[:, j + 1]
    return dict(n_e=n_e, n_i=r * n_e, xi=10.0 ** lxi, Zmean=Z, sigma=sig, Jion=J)


def marchand21_charge_balance(a, n_grains, T, nH, zeta, mu_i, s_e=0.5):
    """Marchand+21 analytic grain charges (Eqs. 45-47, root in psi).

    Returns a dict: n_e, n_i, psi, Zmean, Jion (per bin).
    """
    a = np.asarray(a, dtype=float)
    n_grains = np.asarray(n_grains, dtype=float)
    tau = a * KB_CGS * T / E2_CGS
    alpha = np.sqrt(8.0 / (math.pi * tau))
    Theta = s_e * math.sqrt(mu_i * MH_CGS / ME_CGS)
    v_i = math.sqrt(8.0 * KB_CGS * T / (math.pi * mu_i * MH_CGS))
    k = recombination_rate_coefficient(T)
    S = zeta * nH

    def parts(psi):
        eps = (1.0 - psi) / (Theta * math.exp(psi))
        eT = eps * Theta
        Z = psi * tau + (1.0 - eT ** 2) / (1.0 + eT * alpha + eT ** 2)
        J = (1.0 - psi) + (2.0 / tau) * (1.0 + 1.0 / eT) / (1.0 + alpha / eT + 1.0 / eT ** 2)
        n_i = -np.dot(n_grains, Z) / (1.0 - eps)
        return eps, Z, J, n_i

    def f(psi):
        eps, Z, J, n_i = parts(psi)
        return k * eps * n_i ** 2 / S + n_i * v_i * np.dot(n_grains * math.pi * a ** 2, J) / S - 1.0

    # eps < 1 (n_e < n_i) requires psi above psi0, the root of 1 - psi = Theta e^psi
    # (their Newton starting point: -2.504 for mu_i = 1, -3.799 for mu_i = 25)
    psi0 = brentq(lambda p: (1.0 - p) - Theta * math.exp(p), -50.0, 1.0)
    psi = brentq(f, psi0 + 1e-10, -1e-12, xtol=1e-14)
    eps, Z, J, n_i = parts(psi)
    return dict(n_e=eps * n_i, n_i=n_i, psi=psi, Zmean=Z, Jion=J)
