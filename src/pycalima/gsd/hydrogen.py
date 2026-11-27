"""Tracked hydrogen of the covalent a-C(:H) bins (rung M1): rates, equilibrium, steps.

A bin's grains share a bulk band gap E_g (``gsd.carbon.HydrogenatedCarbon``); the bin
stores M_H, i.e. its carbon-weighted H/C x (``gsd.carbon.bin_h_per_c``). The grain
hydrogen follows the laboratory laws of the plan (all labelled as such):

    dx/dt = -k_UV(x) (x - x_min) + k_H (x_max - x)

- UV loss of C-H bonds, first order in the bonds (Mennella+01: the 3.4 um band decays
  exponentially with fluence), at the per-bond rate k_UV = Y_CH (photons absorbed per C
  atom per s, 6.8-13.6 eV). Y_CH = 0.02 C-H bonds per absorbed photon at H/C ~ 1: Alata+14
  (one bond per ~70 photons, H2 yield 0.025) and Mate 2014, range 0.014-0.025; it gives
  sigma_d = Y_CH sigma_abs/C ~ 1.4e-19 cm^2, inside Mennella's and Alata's 1-3e-19. The lamp
  data stop at 10.5 eV; 10.5-13.6 eV is assumed to act alike. The absorption is the bin's
  own at its current E_g, so k_UV depends on x.
- the floor x_min: the lab limit of UV processing, X_H ~ 0.05 (E_g = 0.215 eV; Jones 2012
  III's recommendation, from Adel+89, Gadallah+11, Godard+11).
- re-hydrogenation by H atoms: k_H = sigma_f(T) n(H I) v_H / 4 (the H flux per unit
  surface), sigma_f = 1.9e-18 cm^2 per H atom (Mennella+02), up to Mennella's plateau of
  bulk H/C 0.65 (x_max, at E_g = 1.69 eV). sigma_f falls with the H-atom temperature with
  Mennella+06's 6 meV (70 K) activation energy, normalised at 300 K (it gives the ~1e3 fall
  at 10 K quoted by Godard+11).

The H lost to UV leaves mostly as H2 (Alata+14/15; CH4 is 3% of the H2), the H gained
comes from H I; those exchanges are left to the callers. Charge is not involved.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .carbon import M_C, HydrogenatedCarbon, bin_h_per_c

HC_EV_UM = 1.23984198
KB = 1.380649e-16
M_H_G = 1.6735575e-24
YR = 3.15576e7

Y_CH = 0.02                 # C-H bonds broken per photon absorbed (per C atom, at H/C ~ 1); 0.014-0.025
UV_RANGE_EV = (6.8, 13.6)
X_H_FLOOR = 0.05            # lab floor of UV processing (E_g = 0.215 eV)
SIGMA_F = 1.9e-18           # cm^2 per H atom (Mennella+02)
T_ACT_K, T_REF_K = 70.0, 300.0
H_PER_C_PLATEAU = 0.65      # bulk H/C at saturation under H atoms (Mennella+02)


def e_gap_of_bulk_x_h(x_h):
    return 4.3 * x_h


E_FLOOR = e_gap_of_bulk_x_h(X_H_FLOOR)
E_PLATEAU = e_gap_of_bulk_x_h(H_PER_C_PLATEAU / (1.0 + H_PER_C_PLATEAU))


def sigma_f(T):
    """Mennella's H-addition cross-section [cm^2 per H atom] at H-atom temperature T [K]."""
    return SIGMA_F * np.exp(-T_ACT_K * (1.0 / np.asarray(T, float) - 1.0 / T_REF_K))


def k_h(n_hi, T):
    """Re-hydrogenation rate [s^-1]: sigma_f(T) n(H I) v_H / 4."""
    v = np.sqrt(8.0 * KB * np.asarray(T, float) / (np.pi * M_H_G))
    return sigma_f(T) * np.asarray(n_hi, float) * v / 4.0


def k_uv_groups(sigma_per_c, N_g, c_red, uv_weight):
    """UV loss rate per C-H bond [s^-1] from RT photon groups: Y_CH c sum_g w_g N_g sigma_g.

    sigma_per_c[..., g]: the bin's group-mean absorption cross-section per C atom [cm^2] (from its pseudo-bins at the
    cell's in-bin state, on the band-gap nodes); N_g: the cell's group photon densities [cm^-3]; c_red: the RT speed
    of light [cm/s]; uv_weight[g]: the share of each group's photons inside the lab range ``UV_RANGE_EV`` (a property
    of the group limits and the assumed in-group spectrum, labelled). Returns k_UV[...]."""
    return Y_CH * c_red * np.asarray(sigma_per_c, float) @ (np.asarray(uv_weight, float) * np.asarray(N_g, float))


def equilibrium_gap(bin_h, k_uv_at_gap, k_h_rate, n_secant=6):
    """The steady H/C and band gap of a covalent bin, with the UV-loss rate given on the band-gap nodes.

    Solves x (k_UV(x) + k_H) = k_UV(x) x_min + k_H x_max, the equilibrium of the lab law (module doc), with k_UV
    linear in E_g between the nodes ``bin_h.e_grid`` (``k_uv_at_gap`` there, e.g. from ``k_uv_groups``) and k_H
    [s^-1] fixed. The root is bracketed on the nodes, then refined by regula falsi (Illinois) within the bracket: a
    few table lookups per bin, as RAMSES needs. Returns (x_eq, E_g)."""
    kg = np.asarray(k_uv_at_gap, float)
    lo, hi = bin_h.x_min, bin_h.x_max

    def f(x):
        ku = np.interp(bin_h.e_gap(x), bin_h.e_grid, kg)
        return -ku * (x - lo) + k_h_rate * (hi - x)

    xs = np.concatenate([[lo], bin_h.x_grid[(bin_h.x_grid > lo) & (bin_h.x_grid < hi)], [hi]])
    fs = f(xs)                                           # f(x_min) >= 0 >= f(x_max)
    j = int(np.flatnonzero(fs <= 0.0)[0])
    if j == 0:
        return float(lo), float(bin_h.e_gap(lo))
    a, b, fa, fb = xs[j - 1], xs[j], fs[j - 1], fs[j]
    side = 0
    for _ in range(n_secant):
        if fa == fb:
            break
        x = (a * fb - b * fa) / (fb - fa)
        fx = f(x)
        if fx == 0.0:
            a = b = x
            break
        if fx > 0.0:
            a, fa = x, fx
            if side == 1:
                fb *= 0.5
            side = 1
        else:
            b, fb = x, fx
            if side == -1:
                fa *= 0.5
            side = -1
    x = (a * fb - b * fa) / (fb - fa) if fb != fa else 0.5 * (a + b)
    return float(x), float(bin_h.e_gap(x))


@dataclass
class Conditions:
    """A cell: FUV in units of the Mathis (1983) ISRF (``u_mathis``), n(H I) [cm^-3], T_gas [K]."""
    u_mathis: float
    n_hi: float
    t_gas: float


@dataclass
class BinHydrogen:
    """The hydrogen of one covalent bin at in-bin state ``s``: x(E_g) and the UV absorption."""
    grid: object
    i: int
    s: float = 0.5
    n_nodes: int = 32
    n_e: int = 49
    carrier: HydrogenatedCarbon = field(default_factory=HydrogenatedCarbon)

    def __post_init__(self):
        from pycalima.pahchem.fields import mathis1983

        from .optics import node_u, profile_weights

        lo, hi = self.carrier.e_range
        self.e_grid = np.linspace(lo, hi, self.n_e)
        self.x_grid = np.array([float(bin_h_per_c(self.carrier, self.grid, self.i, self.s, e, self.n_nodes)[0])
                                for e in self.e_grid])
        if np.any(np.diff(self.x_grid) <= 0):
            raise RuntimeError("bin H/C is not increasing in E_g")
        self.x_min = float(np.interp(E_FLOOR, self.e_grid, self.x_grid))
        self.x_max = float(np.interp(E_PLATEAU, self.e_grid, self.x_grid))
        # photons absorbed per C atom per s in the Mathis ISRF (U = 1), against E_g
        f = mathis1983()
        E = np.linspace(*UV_RANGE_EV, 40)
        lam = HC_EV_UM / E
        eta = self.grid.etas[self.i]
        u = node_u(eta, self.n_nodes)
        W = profile_weights(eta, np.array([self.s]), u)
        m = self.grid.mass(self.grid.a_edges[self.i] * u)
        absorbed = []
        for e in self.e_grid:
            kap = W @ self.carrier.per_gram_carbon_at_gap(m, lam, e)[..., 0]     # cm^2 per g of C, (1, nE)
            absorbed.append(f.photon_rate(kap[0] * M_C, E))
        self.abs_per_c = np.array(absorbed)

    def e_gap(self, x):
        """Bulk E_g [eV] of the bin's grains at bin H/C x."""
        return np.interp(x, self.x_grid, self.e_grid)

    def k_uv(self, x, cond):
        """UV loss rate per C-H bond [s^-1] at H/C x."""
        return Y_CH * cond.u_mathis * np.interp(self.e_gap(x), self.e_grid, self.abs_per_c)

    def rhs(self, x, cond):
        return -self.k_uv(x, cond) * (x - self.x_min) + k_h(cond.n_hi, cond.t_gas) * (self.x_max - x)

    def relaxation_time(self, x, cond):
        """1 / (k_UV + k_H) [s] at H/C x (the e-folding time of the linearised law)."""
        return 1.0 / (self.k_uv(x, cond) + k_h(cond.n_hi, cond.t_gas))

    def equilibrium(self, cond, n_iter=60):
        """The steady H/C: x = (k_UV(x) x_min + k_H x_max) / (k_UV(x) + k_H), by bisection."""
        lo, hi = self.x_min, self.x_max
        for _ in range(n_iter):
            mid = 0.5 * (lo + hi)
            if self.rhs(mid, cond) > 0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def step(self, x, cond, dt):
        """The exact solution over dt with the rates frozen at x (what RAMSES applies)."""
        ku, kh = self.k_uv(x, cond), k_h(cond.n_hi, cond.t_gas)
        k = ku + kh
        x_eq = (ku * self.x_min + kh * self.x_max) / k
        return x_eq + (x - x_eq) * np.exp(-k * dt)
