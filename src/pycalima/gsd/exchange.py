"""Gas-grain exchange: recondensation, sublimation and nucleation on the GSD.

For a population of one material (a ``sublimation_kinetics`` model), grains
of radius a at dust temperature T_d(a) change size as

    da/dt = alpha(T_d) V_unit n_X v_X(T_gas) / (4 k)            (condensation)
          - V_unit J_bulk(T_d) exp(theta N^-1/3 / T_d) [f_corr(a, T_d)]
                                                               (sublimation)

with X the key species (k per unit), V_unit the volume per evaporating
unit, J_bulk the dataset's bulk evaporation flux and theta the GD89
small-grain (Kelvin) term for N units in the grain. This is the per-grain
form of ``dust_sublimation.saturated_key_density``: da/dt = 0 at
n_X = n_sat(a, T_d).

The GD89 microcanonical factor f_corr is off by default
(``microcanonical=False``). It describes isolated, stochastically heated
grains: at a fixed T_d it vanishes for clusters whose thermal energy is
below the binding energy, which would freeze tens-of-atoms clusters in
place, and the steady-state nucleation rate assumes canonical evaporation
with the same Kelvin term, so canonical rates keep the two consistent.
Switch it on together with a P(T) treatment of the smallest grains.
``kelvin=False`` drops the small-grain term from growth and evaporation (it
still sets the nucleation barrier), as in the growth law of Nozawa & Kozasa
(2013, eq. 2).

New grains form by steady-state nucleation (``steady_state_nucleation_rate``)
and are laid along their growth characteristic (``remap.inject``) from the
injection size, or the bottom of the grid if smaller. With the Kelvin term
that size is n* + 3 n*^(2/3) / mu^(1/2) monomers, where Delta G has fallen
by kT below the barrier (the Zeldovich width): J_s already counts the
clusters that cross the barrier, and at n* itself da/dt = 0, so grains
injected there would stall while nucleation continued. Without it
(``kelvin=False``) grains start at n*, as in NK13's steady model.

``exchange_step`` advances all populations sharing a key species over dt
with the gas density held at its end-of-step value n (backward Euler in the
gas): n is the root of n + X_dust(n) = X_total, which is monotone in n, so
the step is stable for any dt and keeps n >= 0, and the gas is then set to
X_total - X_dust so that every key element is conserved to round-off.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import brentq

from pycalima.models.dust_radiation import dust_sublimation as ds

from .grid import MassGrid
from .pl2m import BinProfile
from .remap import advance, inject, remap, trajectory, velocity_table

_KB = 1.380649e-16
_AMU = 1.66053906660e-24


@dataclass
class Population:
    """Grains of one material on its own mass grid.

    ``model`` is a ``sublimation_kinetics.SublimationModel``; ``T_d`` the dust
    temperature per bin [K]; ``T_cluster`` the temperature of nucleating
    clusters (default: the smallest bin's T_d). ``overflow_N/M`` collect
    grains grown past the top of the grid.
    """

    model: object
    grid: MassGrid
    N: np.ndarray
    M: np.ndarray
    T_d: np.ndarray
    T_cluster: Optional[float] = None
    nucleation: bool = True
    neutral_fraction: float = 1.0
    overflow_N: float = 0.0
    overflow_M: float = 0.0
    table_points: int = 2000
    microcanonical: bool = False
    kelvin: bool = True
    _fcorr: tuple = field(default=None, repr=False)
    _profile: tuple = field(default=None, repr=False)

    @classmethod
    def empty(cls, model, a_min_cm, a_max_cm, bins_per_mass_decade, T_d=1000.0, **kw):
        grid = MassGrid.from_radii(a_min_cm, a_max_cm, bins_per_mass_decade, model.density_gcm3)
        nb = grid.n_bins
        return cls(model, grid, np.zeros(nb), np.zeros(nb),
                   np.full(nb, float(T_d)) if np.isscalar(T_d) else np.asarray(T_d, float), **kw)

    @property
    def key_atoms_per_gram(self) -> float:
        return self.model.key_per_unit / (self.model.unit_mass_amu * _AMU)

    def key_atoms(self) -> float:
        """Key-species atoms locked in these grains [cm^-3]."""
        return (self.M.sum() + self.overflow_M) * self.key_atoms_per_gram

    # -- size-space velocity ---------------------------------------------------
    def temperature_of_radius(self, a):
        la = np.log(self.grid.a_centres)
        return np.interp(np.log(np.maximum(a, 1e-30)), la, self.T_d)

    def _correction_table(self):
        """f_corr on a log-a table at T_d(a); computed once per T_d."""
        key = tuple(np.round(self.T_d, 6))
        if self._fcorr is not None and self._fcorr[0] == key:
            return self._fcorr[1], self._fcorr[2]
        a_tab = np.geomspace(self.grid.a_edges[0] * 0.3, self.grid.a_edges[-1], 240)
        T_tab = self.temperature_of_radius(a_tab)
        f_tab = np.array([float(ds.correction_factor(self.model, a, T))
                          for a, T in zip(a_tab, T_tab)])
        self._fcorr = (key, np.log(a_tab), f_tab)
        return np.log(a_tab), f_tab

    def adot(self, n_key, T_gas, coulomb=1.0):
        """Vectorised da/dt(a) [cm s^-1] for key-species density n_key [cm^-3]."""
        m = self.model
        V = m.unit_volume_cm3
        m_X = m.key_mass_amu * _AMU
        v_X = np.sqrt(8.0 * _KB * T_gas / (np.pi * m_X))
        la_tab, f_tab = self._correction_table() if self.microcanonical else (None, None)

        def rate(a):
            a = np.asarray(a, float)
            T = self.temperature_of_radius(a)
            grow = m.condensation_alpha(T) * V * n_key * v_X * coulomb / (4.0 * m.key_per_unit)
            N_units = (4.0 / 3.0) * np.pi * np.maximum(a, 1e-30) ** 3 / V
            kelvin = (np.exp(np.minimum(m.theta * N_units ** (-1.0 / 3.0) / T, 700.0))
                  if self.kelvin else 1.0)
            fcorr = (np.interp(np.log(np.maximum(a, 1e-30)), la_tab, f_tab)
                     if la_tab is not None else 1.0)
            return grow - V * m.flux(T) * kelvin * fcorr

        return rate

    # -- one step at fixed gas ---------------------------------------------------
    def advanced(self, n_key, T_gas, dt, nucleate=True):
        """(N, M, overflow_N, overflow_M, born) after dt at fixed gas n_key (no
        mutation); ``born`` is the number nucleated in the step."""
        rate = self.adot(n_key, T_gas)
        table = velocity_table(rate, self.grid.a_edges[0] * 0.3, self.grid.a_edges[-1] * 3.0,
                               self.table_points)
        if self._profile is None or self._profile[0] is not self.N or self._profile[1] is not self.M:
            self._profile = (self.N, self.M, BinProfile(self.grid, self.N, self.M))
        N1, M1, _lost, oN, oM = remap(self.grid, self.N, self.M,
                                      lambda a: advance(table, a, dt),
                                      profile=self._profile[2])
        born = 0.0
        if nucleate and self.nucleation and n_key > 0.0:
            T_c = self.T_cluster if self.T_cluster is not None else float(self.T_d[0])
            J, n_star = ds.steady_state_nucleation_rate(
                self.model, self.neutral_fraction * n_key, T_c, T_gas)
            if J > 0.0 and np.isfinite(n_star):
                kpu = self.model.key_per_unit
                V_mon = self.model.unit_volume_cm3 / kpu
                n_inj = max(n_star, 1.0)
                if self.kelvin:
                    # past the barrier top by the Zeldovich width, where
                    # Delta G has fallen by kT: there the drift carries the
                    # grains on (at n* itself adot = 0 and they would stall)
                    mu = 1.5 * self.model.theta * kpu ** (-2.0 / 3.0) / T_c
                    n_inj = n_inj + 3.0 * n_inj ** (2.0 / 3.0) / np.sqrt(mu)
                a_star = (3.0 * n_inj * V_mon / (4.0 * np.pi)) ** (1.0 / 3.0)
                a0 = max(a_star, self.grid.a_edges[0] * (1.0 + 1e-9))
                _, y = trajectory(rate, a0, dt, n_nodes=33, table=table)
                dN, dM = inject(self.grid, J * dt, y[::-1])
                N1 = N1 + dN
                M1 = M1 + dM
                born = float(dN.sum())
        return N1, M1, self.overflow_N + oN, self.overflow_M + oM, born


def exchange_step(populations: List[Population], gas: Dict[str, float], T_gas: float,
                  dt: float, rtol: float = 1e-10, dlnn_max: float = 0.005,
                  max_depth: int = 20) -> Dict[str, float]:
    """Advance the populations and the gas over dt, implicitly in the gas.

    ``gas`` maps key species to number density [cm^-3]; it is updated in place
    and returned. Populations sharing a key species are solved together.

    Nucleation depends steeply on the gas density, so while newly formed
    grains are a non-negligible part (>1e-3) of a population the step is
    halved (up to ``max_depth`` times) until the key density changes by less
    than ``dlnn_max`` in ln per sub-step; the error of the backward-Euler
    gas update is then ~dlnn_max instead of the full depletion. Steps without
    nucleation are taken whole at any length.
    """
    by_key: Dict[str, List[Population]] = {}
    for p in populations:
        by_key.setdefault(p.model.key_species, []).append(p)
    for key, pops in by_key.items():
        _advance_key(pops, gas, key, T_gas, dt, rtol, dlnn_max, max_depth)
    return gas


def _advance_key(pops, gas, key, T_gas, dt, rtol, dlnn_max, depth):
    n0 = gas.get(key, 0.0)
    saved = [(p.N, p.M, p.overflow_N, p.overflow_M) for p in pops]
    born = _implicit_step(pops, gas, key, T_gas, dt, rtol)
    n1 = gas.get(key, 0.0)
    if depth <= 0 or n0 <= 0.0:
        return
    N_tot = sum(p.N.sum() for p in pops)
    if born > 1e-3 * N_tot and abs(np.log(max(n1, 1e-300) / n0)) > dlnn_max:
        for p, st in zip(pops, saved):
            p.N, p.M, p.overflow_N, p.overflow_M = st
        gas[key] = n0
        _advance_key(pops, gas, key, T_gas, 0.5 * dt, rtol, dlnn_max, depth - 1)
        _advance_key(pops, gas, key, T_gas, 0.5 * dt, rtol, dlnn_max, depth - 1)


def _implicit_step(pops, gas, key, T_gas, dt, rtol):
    """One backward-Euler step of the populations of one key species; returns
    the number nucleated."""
    total = gas.get(key, 0.0) + sum(p.key_atoms() for p in pops)
    if total <= 0.0:
        return 0.0
    cache = {}

    def dust_after(n):
        if n not in cache:
            cache[n] = [p.advanced(n, T_gas, dt) for p in pops]
        return cache[n]

    def residual(n):
        res = dust_after(n)
        return n + sum((r[1].sum() + r[3]) * p.key_atoms_per_gram
                       for r, p in zip(res, pops)) - total

    lo, hi = 0.0, total
    if residual(lo) >= 0.0:
        n = lo
    else:
        n = brentq(residual, lo, hi, xtol=rtol * total, rtol=rtol, maxiter=200)
    born = 0.0
    for p, r in zip(pops, dust_after(n)):
        p.N, p.M, p.overflow_N, p.overflow_M, b = r
        born += b
    gas[key] = max(total - sum(p.key_atoms() for p in pops), 0.0)
    return born
