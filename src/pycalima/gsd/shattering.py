"""Shattering and sticking of grains of one material (collision models for
``collisions.collide``).

Every collision is evaluated from the side of each grain, as in Hirashita &
Kobayashi (2013, HK13; Earth Planets Space 65, 1083): grain 1 colliding
with grain 2 at relative speed v loses an ejected mass M_ej,1 and keeps a
remnant m1 - M_ej,1, and likewise for grain 2. Two ejecta laws:

* ``"KT10"`` (Kobayashi & Tanaka 2010, Icarus 206, 735, in the form of HK13
  eqs. 6-9): M_ej = phi / (1 + phi) m1, phi = E_imp / (m1 Q*_D),
  E_imp = m1 m2 v^2 / (2 (m1 + m2)), Q*_D = P1 / (2 rho).
* ``"JTH96"`` (Jones, Tielens & Hollenbach 1996 as written in HK13 eqs.
  2-4; Hirashita & Yan 2009, HY09): M_ej = f_M M with the shocked mass

      M / m2 = (1 + 2R) / (2 (1 + R)^(9/16)) sigma_r^(-1/9)
               [M_r^2 / (sigma_1 M_1^2)]^(8/9),

  M_r = v / c0, M_1 = 2 phi1 / (1 + (1 + 4 s phi1)^(1/2)), phi1 = P1 /
  (rho c0^2), sigma(M) = 0.30 (s + 1/M - 0.11)^1.3 / (s + 1/M - 1),
  sigma_1 = sigma(M_1), sigma_r = sigma(M_r / (1 + R)), R = 1 (same
  material); the whole grain is ejected if M > m1 / 2. HY09 prints the
  exponent as 0.13 and omits the factor 2; HK13's form is used.

Fragments follow dN/da ~ a^-alpha_f (alpha_f = 3.3) from ``a_frag_min`` to
a_f,max = (0.0204 f)^(1/3) a1, f = M_ej / m1 (Guillet et al. 2009, HK13 eq.
5); if a_f,max < a_frag_min they go to the collector (HK13 removes them,
HY09 keeps them in the smallest bin).

Shattering occurs above ``v_shat`` (HY09 Table 2). With ``coagulation`` on,
grains stick below the Chokshi et al. (1993) threshold of HY09 eq. 8,

    v_coag = 2.14 F_stick [(a1^3 + a2^3) / (a1 + a2)^3]^(1/2)
             gamma^(5/6) / (E^(1/3) R12^(5/6) rho^(1/2)),

R12 = a1 a2 / (a1 + a2), 1/E = 2 (1 - nu)^2 / E_Y for one material (as
printed in HY09; Chokshi et al. write (1 - nu^2)), F_stick = 10; between
the thresholds they bounce.

Relative velocities come either from a ``velocities.RelativeVelocity``
(Brownian + Ormel & Cuzzi 2007 turbulence + optional drift, with the
skewed-Maxwellian speed distribution): shattering is then ``n_speed``
channels of equal collision flux above v_shat at their quantile speeds, and
sticking the flux below v_coag, both continuous in grain size and gas
state; or from a callable ``velocity(a1, a2)`` returning a list of (weight,
v [cm/s]) arrays, e.g. ``hy09_directions(v_of_a)``, the four cases v1 + v2,
|v1 - v2|, v1, v2 of HY09/HK13 with weight 1/4 each, with hard thresholds
(kept for the HK13 benchmarks).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .collisions import Channel
from .velocities import RelativeVelocity, coulomb_rate_factor, flux_quantile, log_flux_above

_E2 = 4.80320471e-10 ** 2


@dataclass(frozen=True)
class ShatterMaterial:
    """Material constants of HY09 Table 2 (cgs; v_shat in cm/s)."""

    name: str
    rho: float
    c0: float
    s: float
    v_shat: float
    P1: float
    P_v: float
    gamma: float
    E_young: float
    nu: float

    @property
    def Q_D(self) -> float:
        """Q*_D = P1 / (2 rho) [erg/g] (HK13 eq. 9)."""
        return self.P1 / (2.0 * self.rho)


SILICATE = ShatterMaterial("silicate", 3.3, 5.0e5, 1.2, 2.7e5, 3.0e11, 5.4e12, 25.0, 5.4e11, 0.17)
GRAPHITE = ShatterMaterial("graphite", 2.2, 1.8e5, 1.9, 1.2e5, 4.0e10, 5.8e12, 12.0, 3.4e10, 0.5)
MATERIALS = {"silicate": SILICATE, "graphite": GRAPHITE}


def _sigma(Mach, s):
    x = s + 1.0 / Mach
    return 0.30 * (x - 0.11) ** 1.3 / (x - 1.0)


def jth96_shocked_mass_ratio(material: ShatterMaterial, v):
    """M / m_projectile, the mass shocked to P1 per unit projectile mass
    (HK13 eq. 2 without f_M), R = 1."""
    v = np.asarray(v, float)
    R = 1.0
    phi1 = material.P1 / (material.rho * material.c0 ** 2)
    M1 = 2.0 * phi1 / (1.0 + np.sqrt(1.0 + 4.0 * material.s * phi1))
    sig1 = _sigma(M1, material.s)
    Mr = np.maximum(v, 1e-30) / material.c0
    sigr = _sigma(Mr / (1.0 + R), material.s)
    return ((1.0 + 2.0 * R) / (2.0 * (1.0 + R) ** (9.0 / 16.0)) * sigr ** (-1.0 / 9.0)
            * (Mr ** 2 / (sig1 * M1 ** 2)) ** (8.0 / 9.0))


def ejected_mass(material, m1, m2, v, law="KT10", f_M=0.4):
    """Mass ejected from grain m1 hit by m2 at relative speed v."""
    m1 = np.asarray(m1, float)
    m2 = np.asarray(m2, float)
    if law == "KT10":
        E = 0.5 * m1 * m2 / (m1 + m2) * np.asarray(v, float) ** 2
        phi = E / (m1 * material.Q_D)
        return phi / (1.0 + phi) * m1
    if law == "JTH96":
        M = m2 * jth96_shocked_mass_ratio(material, v)
        return np.where(M > 0.5 * m1, m1, np.minimum(f_M * M, m1))
    raise ValueError(f"unknown ejecta law {law!r}")


def coagulation_threshold(material, a1, a2, F_stick=10.0):
    """HY09 eq. 8 (Chokshi et al. 1993) [cm/s]."""
    a1 = np.asarray(a1, float)
    a2 = np.asarray(a2, float)
    R12 = a1 * a2 / (a1 + a2)
    E = material.E_young / (2.0 * (1.0 - material.nu) ** 2)
    return (2.14 * F_stick * np.sqrt((a1 ** 3 + a2 ** 3) / (a1 + a2) ** 3)
            * material.gamma ** (5.0 / 6.0) / (E ** (1.0 / 3.0) * R12 ** (5.0 / 6.0)
                                              * np.sqrt(material.rho)))


def hy09_directions(v_of_a: Callable):
    """Relative velocities v1 + v2, |v1 - v2|, v1 and v2, weight 1/4 each,
    from a velocity dispersion v(a) (HY09 s. 2.2; HK13 s. 2.2).

    v(a) should be continuous: rates are evaluated at the bin nodes, and a
    node sitting on a jump (e.g. HK13's step at a_c) switches them on and off
    as it moves, which makes the sub-step control chatter. Smooth steps over
    ~10% in a."""

    def velocity(a1, a2):
        v1 = np.asarray(v_of_a(a1), float)
        v2 = np.asarray(v_of_a(a2), float)
        v1, v2 = np.broadcast_arrays(v1, v2)
        return [(0.25, v1 + v2), (0.25, np.abs(v1 - v2)), (0.25, v1), (0.25, v2)]

    return velocity


class GrainCollisions:
    """Shattering (and optionally sticking) for one material.

    ``velocity``: a ``RelativeVelocity`` (continuous; recommended) or a
    callable (a1, a2) -> list of (weight, v) arrays (``hy09_directions``).
    ``n_speed``: flux-quantile channels for shattering with a
    ``RelativeVelocity``; with its ``charges``, rates include Coulomb
    focusing/barriers averaged over the charge points of both grains, and
    impact speeds the mean-charge barrier. ``table_ppd``: if set, the rates
    and impact speeds are tabulated per environment on a log-radius grid with
    this many points per decade and interpolated (call ``refresh()`` after
    changing the gas, radiation or charges).
    ``a_frag_min`` [cm] is the smallest fragment (HK13: the grid minimum).
    ``collector``: ``"remove"`` (HK13) or ``"smallest"`` (HY09)."""

    def __init__(self, material, velocity, law="KT10", alpha_f=3.3, f_M=0.4,
                 a_frag_min=3e-8, collector="remove", coagulation=False, F_stick=10.0,
                 v_shat: Optional[float] = None, n_speed=4, table_ppd: Optional[float] = None):
        self.material = MATERIALS[material] if isinstance(material, str) else material
        self.velocity = velocity
        self.law = law
        self.alpha_f = float(alpha_f)
        self.f_M = f_M
        self.a_frag_min = float(a_frag_min)
        self.collector = collector
        self.coagulation = coagulation
        self.F_stick = F_stick
        self.v_shat = self.material.v_shat if v_shat is None else float(v_shat)
        self.n_speed = int(n_speed)
        self.table_ppd = table_ppd
        self._table = None

    def _shatter_products(self, a1, m1, m2, v):
        mej = ejected_mass(self.material, m1, m2, v, self.law, self.f_M)
        f = np.clip(mej / m1, 0.0, 1.0)
        a_hi = (0.0204 * f) ** (1.0 / 3.0) * a1
        return m1 - mej, mej, a_hi

    def channels(self, a, m):
        if isinstance(self.velocity, RelativeVelocity):
            return self._distributed_channels(a, m)
        A1, A2 = a[:, None], a[None, :]
        M1, M2 = m[:, None], m[None, :]
        sig = np.pi * (A1 + A2) ** 2
        lo = np.full(np.broadcast(A1, A2).shape, self.a_frag_min)
        v_c = coagulation_threshold(self.material, A1, A2, self.F_stick) if self.coagulation else None
        out = []
        for wt, v in self.velocity(A1, A2):
            v = np.broadcast_to(np.asarray(v, float), lo.shape)
            K = np.where(v > self.v_shat, wt * sig * v, 0.0)
            if np.any(K > 0):
                r1, e1, h1 = self._shatter_products(A1, M1, M2, v)
                r2, e2, h2 = self._shatter_products(A2, M2, M1, v)
                out.append(Channel(K, rem1=r1, frag1=e1, a_hi1=h1, a_lo1=lo,
                                   rem2=r2, frag2=e2, a_hi2=h2, a_lo2=lo))
            if self.coagulation:
                Kc = np.where(v < v_c, wt * sig * v, 0.0)
                if np.any(Kc > 0):
                    out.append(Channel(Kc, rem1=M1 + M2))
        if not out:
            out.append(Channel(np.zeros(lo.shape)))
        return out

    def refresh(self):
        """Drop the pair table (the environment changed)."""
        self._table = None

    def _pair_fields(self, a, m):
        """Total shattering rate coefficient, impact speeds of the n_speed
        flux-quantile channels, and sticking rate coefficient for all pairs."""
        vel = self.velocity
        A1, A2 = a[:, None], a[None, :]
        M1, M2 = m[:, None], m[None, :]
        sig = np.pi * (A1 + A2) ** 2
        dV, mu = vel.pair_statistics(a, m, self.material.rho)
        alpha = np.sqrt(1.5) / np.maximum(dV, 1e-300)
        s = alpha * mu
        v_mean = vel.collision_speeds(dV, mu, 0.0, 1)[0]
        m_red = M1 * M2 / (M1 + M2)
        v_c = coagulation_threshold(self.material, A1, A2, self.F_stick) if self.coagulation else None
        if vel.charges is None:
            vC2m = np.zeros_like(sig)
            f_shat = flux_above_(alpha * self.v_shat, s)
            f_stick = -np.expm1(log_flux_above(alpha * v_c, s)) if self.coagulation else None
        else:
            Z, wz = vel.charges(a)
            Zm = Z @ wz
            vC2m = 2.0 * _E2 * Zm[:, None] * Zm[None, :] / ((A1 + A2) * m_red)
            f_shat = np.zeros_like(sig)
            f_stick = np.zeros_like(sig) if self.coagulation else None
            for i, wi in enumerate(wz):
                for j, wj in enumerate(wz):
                    vC2 = 2.0 * _E2 * Z[:, i][:, None] * Z[:, j][None, :] / ((A1 + A2) * m_red)
                    f_shat += wi * wj * coulomb_rate_factor(alpha, s, vC2, self.v_shat)
                    if self.coagulation:
                        # sticking on the relative speed at infinity, v < v_coag (as
                        # collapse.velocities / Guillet+20), with the Coulomb factor:
                        # the impact-speed window of v in [0, v_coag]
                        lo_i = np.sqrt(np.maximum(-vC2, 0.0))
                        hi_i = np.sqrt(np.maximum(v_c ** 2 - vC2, 0.0))
                        f_stick += wi * wj * coulomb_rate_factor(alpha, s, vC2, lo_i, hi_i)
        # channel speeds: equal-flux quantiles above the threshold, as impact speeds
        Ua = alpha * np.sqrt(np.maximum(self.v_shat ** 2 + vC2m, 0.0))
        lc = log_flux_above(Ua, s)
        ks = (np.arange(self.n_speed) + 0.5) / self.n_speed
        y = -(lc[None] + np.log1p(-ks)[:, None, None])
        v_q = flux_quantile(np.broadcast_to(s, y.shape), y) / alpha
        v_imp = np.sqrt(np.maximum(v_q ** 2 - vC2m, self.v_shat ** 2))
        K_shat = sig * v_mean * f_shat
        K_stick = sig * v_mean * f_stick if self.coagulation else None
        return K_shat, v_imp, K_stick

    def _fields(self, a, m):
        """``_pair_fields``, or its interpolation from a pair table on a
        log-radius grid, linear on the triangles of each cell split along its
        diagonal: the turbulent relative velocity has a cusp at a1 = a2
        (Ormel & Cuzzi's class I term ~ |St1 - St2|), which then lies on
        triangle edges and is never interpolated across."""
        if not self.table_ppd:
            return self._pair_fields(a, m)
        lo, hi = a.min(), a.max()
        t = self._table
        if t is None or lo < t["a"][0] or hi > t["a"][-1]:
            n = max(int(np.ceil(np.log10(4.0 * hi / lo) * self.table_ppd)) + 1, 4)
            at = np.geomspace(lo / 2.0, hi * 2.0, n)
            K, v, Kc = self._pair_fields(at, 4.0 / 3.0 * np.pi * self.material.rho * at ** 3)
            t = self._table = {"a": at, "lK": np.log(np.maximum(K, 1e-300)), "v": v,
                               "lKc": None if Kc is None else np.log(np.maximum(Kc, 1e-300))}
        la = np.log(t["a"])
        f = np.clip((np.log(a) - la[0]) / (la[1] - la[0]), 0.0, la.size - 1 - 1e-9)
        x, y = f[:, None] + 0 * f[None, :], f[None, :] + 0 * f[:, None]
        i, j = x.astype(int), y.astype(int)
        p, q = x - i, y - j
        low = p >= q

        def tri(T):
            T00, T11 = T[..., i, j], T[..., i + 1, j + 1]
            T10, T01 = T[..., i + 1, j], T[..., i, j + 1]
            return np.where(low, T00 + p * (T10 - T00) + q * (T11 - T10),
                            T00 + q * (T01 - T00) + p * (T11 - T01))

        def pos(lT):
            v_ = np.exp(tri(lT))
            return np.where(v_ < 1e-299, 0.0, v_)

        return pos(t["lK"]), tri(t["v"]), None if t["lKc"] is None else pos(t["lKc"])

    def _distributed_channels(self, a, m):
        """Channels from the speed distribution of every pair."""
        A1, A2 = a[:, None], a[None, :]
        M1, M2 = m[:, None], m[None, :]
        K_shat, v_imp, K_stick = self._fields(a, m)
        lo = np.full(K_shat.shape, self.a_frag_min)
        out = []
        K = K_shat / self.n_speed
        if np.any(K > 0):
            for v in v_imp:
                r1, e1, h1 = self._shatter_products(A1, M1, M2, v)
                r2, e2, h2 = self._shatter_products(A2, M2, M1, v)
                out.append(Channel(K, rem1=r1, frag1=e1, a_hi1=h1, a_lo1=lo,
                                   rem2=r2, frag2=e2, a_hi2=h2, a_lo2=lo))
        if self.coagulation:
            out.append(Channel(K_stick, rem1=M1 + M2))
        if not out:
            out.append(Channel(np.zeros(K_shat.shape)))
        return out


def flux_above_(u, s):
    return np.exp(log_flux_above(u, s))
