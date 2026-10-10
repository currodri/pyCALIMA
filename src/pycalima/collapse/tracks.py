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


# ---------------------------------------------------------------------------
# Prescribed thermal tracks (environment cards) and Omukai et al. (2005) dynamics
# ---------------------------------------------------------------------------

KB_CGS = 1.380649e-16
Y_HE = 0.0833            # n(He) / n_H (Omukai 2000)


def omukai_f(gamma):
    """Ratio of pressure to gravity at the centre of the Larson-Penston-type
    self-similar solution (Omukai et al. 2005, Eq. 9), capped at 0.95 as they do."""
    g = np.asarray(gamma, dtype=float)
    f = np.where(g < 0.83, 0.0,
                 np.where(g < 1.0, 0.6 + 2.5 * (g - 1.0) - 6.0 * (g - 1.0) ** 2,
                          1.0 + 0.2 * (g - 4.0 / 3.0) - 2.9 * (g - 4.0 / 3.0) ** 2))
    return np.clip(f, 0.0, 0.95)


TRACK_COLUMNS = {"Z0": 1, "m6": 2, "m4": 3, "m2": 4, "0": 5}
GAMMA_HALF_WIDTH = 0.25     # dex in n_H


class Track:
    """A prescribed collapse: T(n_H) and f_H2(n_H) from a sourced environment card,
    with the Omukai et al. (2005) collapse time.

    Gas: H (atomic and molecular), He with y_He = 0.0833. Mean molecular weight
    mu = (1 + 4 y_He) / (1 - f_H2 / 2 + y_He); p = n_tot k T; rho = (1 + 4 y_He) m_H n_H.
    gamma = d ln p / d ln rho along the track (fitted over +-0.25 dex);
    t_col = t_ff / (1 - f(gamma))**1/2
    (Omukai+05, Eqs. 6-9), with f = 0 (free fall) until the first temperature
    minimum after the initial heating, "before the clouds experience the first
    fragmentation" (their Sect. 2). Compressional heating per unit volume:
    Gamma_comp = p / t_col.
    """

    def __init__(self, log_n, log_T, log_fH2, name=""):
        ok = np.isfinite(log_T)
        self.log_n = np.asarray(log_n)[ok]
        self.log_T = np.asarray(log_T)[ok]
        self.log_fH2 = np.asarray(log_fH2)[ok]
        self.name = name
        lp = np.log10(self._n_tot(self.log_n) * KB_CGS * 10.0 ** self.log_T)
        # d ln p / d ln rho (rho ∝ n_H): the slope of a straight-line fit over
        # +-GAMMA_HALF_WIDTH dex, since a digitised track is too noisy to
        # differentiate point to point
        self.gamma = np.array([np.polyfit(self.log_n[w], lp[w], 1)[0] for w in
                               (np.abs(self.log_n - x) <= GAMMA_HALF_WIDTH for x in self.log_n)])
        dT = np.gradient(self.log_T, self.log_n)
        first_min = np.nonzero((dT[:-1] < 0.0) & (dT[1:] >= 0.0))[0]
        self.log_n_frag = self.log_n[first_min[0] + 1] if first_min.size else self.log_n[0]
        # specific heat per H nucleus in units of k, and its log-slope along the track
        f = 10.0 ** self.log_fH2
        self.c_v = 1.5 * (1.0 - f) + 1.5 * Y_HE + 1.25 * f
        lu = np.log10(self.c_v * 10.0 ** self.log_T)
        self.dlnu = np.array([np.polyfit(self.log_n[w], lu[w], 1)[0] for w in
                              (np.abs(self.log_n - x) <= GAMMA_HALF_WIDTH for x in self.log_n)])
        self.dlnc = np.array([np.polyfit(self.log_n[w], np.log10(self.c_v[w]), 1)[0] for w in
                              (np.abs(self.log_n - x) <= GAMMA_HALF_WIDTH for x in self.log_n)])

    @classmethod
    def omukai2005(cls, which="Z0"):
        """Dust-free Omukai+05 track: ``"Z0"`` (Z = 0), or [Z/H] = ``"m6"``, ``"m4"``,
        ``"m2"``, ``"0"``, all with the Z = 0 H2 fraction (see the data file)."""
        from pycalima import _paths
        d = np.loadtxt(_paths.get_external_data_path("collapse_tracks", "omukai2005_dustfree.txt"))
        return cls(d[:, 0], d[:, TRACK_COLUMNS[which]], d[:, 6], name=f"Omukai+05 {which}")

    def _n_tot(self, log_n):
        fH2 = 10.0 ** np.interp(log_n, self.log_n, self.log_fH2)
        return 10.0 ** np.asarray(log_n) * (1.0 - fH2 / 2.0 + Y_HE)

    def T(self, nH):
        return 10.0 ** np.interp(np.log10(nH), self.log_n, self.log_T)

    def f_H2(self, nH):
        return 10.0 ** np.interp(np.log10(nH), self.log_n, self.log_fH2)

    def species(self, nH):
        """(n(H), n(H2), n(He)) [cm^-3]."""
        f = self.f_H2(nH)
        return nH * (1.0 - f), nH * f / 2.0, nH * Y_HE

    def mu(self, nH):
        return (1.0 + 4.0 * Y_HE) / (1.0 - self.f_H2(nH) / 2.0 + Y_HE)

    def rho(self, nH):
        return (1.0 + 4.0 * Y_HE) * MH_CGS * nH

    def pressure(self, nH):
        return self._n_tot(np.log10(nH)) * KB_CGS * self.T(nH)

    def t_col(self, nH):
        lg = np.log10(nH)
        f = np.where(lg < self.log_n_frag, 0.0, omukai_f(np.interp(lg, self.log_n, self.gamma)))
        return math.sqrt(3.0 * math.pi / (32.0 * G_CGS)) / np.sqrt(self.rho(nH)) / np.sqrt(1.0 - f)

    def compression_heating(self, nH):
        """Gamma_comp = p / t_col [erg cm^-3 s^-1]."""
        return self.pressure(nH) / self.t_col(nH)

    def specific_heat(self, nH):
        """(c_v, d ln c_v / d ln n_H) per H nucleus, c_v in units of k."""
        lg = np.log10(nH)
        return np.interp(lg, self.log_n, self.c_v), np.interp(lg, self.log_n, self.dlnc)

    def implied_cooling(self, nH):
        """Net non-dust cooling per H nucleus along the track [erg s^-1]:
        lambda = [p / n_H - du/d ln n_H] / t_col, u = c_v k T, which makes the
        energy equation du/dt = p / (n_H t_col) - lambda hold on the track. It
        includes chemical heating (H2 formation) and can be negative."""
        lg = np.log10(nH)
        T = self.T(nH)
        u = np.interp(lg, self.log_n, self.c_v) * KB_CGS * T
        du = u * np.interp(lg, self.log_n, self.dlnu)
        return (self.pressure(nH) / nH - du) / self.t_col(nH)

    def thermal_energy(self, nH):
        """e_th = sum_s n_s (f_s / 2) k T [erg cm^-3]: 3/2 kT for H and He, 5/2 kT for H2."""
        nH_, nH2, nHe = self.species(nH)
        return (1.5 * (nH_ + nHe) + 2.5 * nH2) * KB_CGS * self.T(nH)

    def n_molecular(self, fraction=0.9):
        """First n_H at which f_H2 reaches ``fraction`` (inf if never on the track)."""
        hit = np.nonzero(10.0 ** self.log_fH2 >= fraction)[0]
        return 10.0 ** self.log_n[hit[0]] if hit.size else math.inf

    def steps(self, n0, n_end, dlnn=0.02):
        """Step edges (t [s], n_H) uniform in ln n_H, with dt = t_col d ln n_H (midpoint)."""
        n = np.exp(np.linspace(math.log(n0), math.log(n_end),
                               max(int(math.ceil(math.log(n_end / n0) / dlnn)), 1) + 1))
        mid = np.sqrt(n[1:] * n[:-1])
        dt = self.t_col(mid) * np.diff(np.log(n))
        return np.r_[0.0, np.cumsum(dt)], n
