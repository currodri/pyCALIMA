"""The per-cell GSD operator, as RAMSES will run it (A3c): two calls per cooling substep.

RAMSES-CALIMA calls the dust twice per cooling substep (``dust_interface.f90``):
- ``precool`` (``compute_dust_precool``), at the substep's starting state, gives what the gas needs and what the
  dust update will use: charges, photoelectric heating, recombination cooling, the Coulomb factors, the grain area
  (for H2 formation and gas-grain collisions) and the Coulomb-explosion rates.
- ``update`` (``compute_dust_update``) advances the dust over the substep with the precool charges, in the agreed
  operator splitting:
  1. collisions (``collisions.collide``, with carbon's pair rule and the precool charges in the Coulomb factors);
  2. size-dependent sinks per moment (Coulomb explosion, ``sinks``), whose grains go to fragments or to the gas;
  3. erosion (thermal sputtering, ``sputtering.sputter``);
  4. the exchange with the gas (accretion and sublimation, ``exchange.exchange_step``), last, as the stiffest
     coupling.

These two calls are the whole interface: their inputs and outputs are plain per-cell quantities, and nothing
outside this module calls the operators below for a cell. A composition's grid may carry the overflow bin
(``grid.MassGrid.with_overflow``): while it holds at most ``Options.overflow_gate`` of the composition's mass it is
carried passively (the operators run on the grid below it, and what grows past that grid's top is added to it); above
the gate it is an ordinary bin. Its share is reported, with a warning above 1 % (``strict``: an error).
"""
from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .grid import MassGrid

#: the processes ``update`` knows; ``Options.processes`` switches them
PROCESSES = frozenset({"shattering", "coagulation", "explosion", "sputtering", "accretion"})


@dataclass
class Composition:
    """One composition's grains in a cell, and what acts on them.

    ``N`` [cm^-3] and ``M`` [g cm^-3] per bin of ``grid``. ``key``: the gas element the grains are made of and return
    to ('C', 'Si'), ``per_gram`` its atoms per gram of grains. ``material``: the shattering material
    (``shattering.MATERIALS`` entry, or a ``materials.GrainMaterial``'s ``shatter``). ``sputtering``: the
    ``sputtering.erosion_rate`` material ('graphite', 'silicate') or None; ``sublimation``: the
    ``sublimation_kinetics`` model of accretion and sublimation, or None; ``charging``: a
    ``charging.AreaNodeCharging`` (precool), or None for neutral grains; ``pair_rule``: carbon's ``pairs.PairRule``,
    or None. ``overflow_M`` collects what leaves the top of the grid (past the overflow bin, if any)."""

    grid: MassGrid
    N: np.ndarray
    M: np.ndarray
    key: str
    per_gram: float
    material: object
    method: object = None
    sputtering: str | None = None
    sublimation: object = None
    charging: object = None
    pair_rule: object = None
    overflow_M: float = 0.0
    _erosion: dict = field(default_factory=dict, repr=False)

    def erosion(self, T, n_H) -> Callable:
        """The thermal sputtering da/dt(a) [cm/s] at (T, n_H), cached per cell condition."""
        from .sputtering import erosion_rate

        key = (float(T), float(n_H))
        if key not in self._erosion:
            self._erosion[key] = erosion_rate(self.sputtering, T, n_H, rho=self.grid.rho)
        return self._erosion[key]


@dataclass
class CellInputs:
    """A cell's conditions over the substep: T [K], n(H) [cm^-3], n_e [cm^-3], ``gas`` the number densities of the
    compositions' key elements in the gas [cm^-3] (``update`` returns their new values); the dust temperature T_d
    [K] (given until precool computes it); turbulence v_turb [cm/s] and its driving scale L [cm]; the RT photon
    groups (``charging.PhotonGroups``), the UV background G0_bg [Habing] and the ions [(n, z, m)] for the charge."""

    T: float
    n_H: float
    n_e: float
    gas: dict
    T_d: float = 20.0
    v_turb: float = 0.0
    L: float = 0.0
    photons: object = None
    G0_bg: float = 0.0
    ions: list | None = None


@dataclass
class Options:
    """Switches and numerical parameters of the operator."""

    processes: frozenset = PROCESSES
    errmax: float = 1e-3                 # collide's error control
    n_speed: int = 4                     # GrainCollisions' shattering channels
    table_ppd: float | None = 24         # GrainCollisions' pair table (ignored with a pair rule)
    threads: int | None = 1
    alpha_frag: float = 3.3              # fragments' dN/da ~ a^-alpha (shattering and explosion)
    explosion_route: str = "fragments"   # 'fragments' (user, 2026-11-23) or 'gas'
    overflow_gate: float = 1e-6          # the overflow bin is passive at or below this share of the mass
    overflow_warn: float = 0.01
    strict: bool = False                 # raise instead of warning when the overflow bin fills up


@dataclass
class Precool:
    """What precool gives the gas and the update, per composition (lists in the order of the compositions):
    heating and cooling [erg s^-1 cm^-3], the charge state on the area nodes (``charging.ChargeState``) and the bins'
    weights, the explosion rate per bin [s^-1] and the grain area per unit volume [cm^2 cm^-3]."""

    heating: list
    cooling: list
    states: list
    weights: list
    explosion: list
    area: list

    @property
    def total_heating(self):
        return float(sum(np.sum(h) for h in self.heating))

    @property
    def total_cooling(self):
        return float(sum(np.sum(c) for c in self.cooling))


@dataclass
class UpdateResult:
    """The update's bookkeeping: the gas key densities after the substep, the overflow bins' shares of their
    compositions' mass, the largest fractional change of any bin's mass (dUU, as RTZ uses it) and the relative
    change of each key element (dust + gas + overflow), which is round-off."""

    gas: dict
    overflow_share: list
    duu: float
    conservation: dict
    warnings: list


def _active(c: Composition, gate):
    """True if the composition's overflow bin (if any) is computed this substep."""
    if not c.grid.overflow:
        return True
    tot = float(np.sum(c.M))
    return tot > 0.0 and c.M[-1] > gate * tot


def _subgrid(c: Composition, active):
    if active or not c.grid.overflow:
        return c.grid, c.N.copy(), c.M.copy()
    return MassGrid(c.grid.m_edges[:-1], c.grid.rho), c.N[:-1].copy(), c.M[:-1].copy()


def _collisions(c, g, inputs, opt, pre_state):
    from .shattering import GrainCollisions
    from .velocities import RelativeVelocity

    charges = None if pre_state is None else pre_state.charge_points
    rv = RelativeVelocity(T=inputs.T, n_H=inputs.n_H, mu=1.4, v_turb=inputs.v_turb, L=inputs.L, charges=charges)
    return GrainCollisions(c.material, rv, law="KT10", alpha_f=opt.alpha_frag,
                           coagulation="coagulation" in opt.processes,
                           v_shat=None if "shattering" in opt.processes else 1e30,
                           a_frag_min=float(g.a_edges[0]), collector="remove", n_speed=opt.n_speed,
                           table_ppd=opt.table_ppd, threads=opt.threads, pair_rule=c.pair_rule)


def _explode(g, N, M, k_bin, dt, opt, a_star):
    """Coulomb explosion at the per-bin rates k_bin [s^-1] (precool's rate at each bin's area node, held over the
    bin: labelled), a sink that removes N and M alike, exactly; returns (N, M, mass to the gas)."""
    from .sinks import deposit_power_law

    nb = g.n_bins
    k = np.asarray(k_bin, float)[:nb]
    if not np.any(k > 0.0):
        return N, M, 0.0
    edges = g.a_edges
    keep = np.exp(-np.maximum(k, 0.0) * dt)
    dM = M * (1.0 - keep)
    N, M = N * keep, M * keep
    if opt.explosion_route == "gas":
        return N, M, float(dM.sum())
    gas = 0.0
    for i in np.flatnonzero(dM > 0.0):
        # fragments a^-alpha from the grid's lowest edge up to half the parent's mass (labelled)
        dn, dm, below, _ = deposit_power_law(g, float(dM[i]), float(edges[0]), 2.0 ** (-1.0 / 3.0) * a_star[i],
                                             opt.alpha_frag)
        N, M = N + dn, M + dm
        gas += below
    return N, M, gas


def update(compositions, inputs: CellInputs, pre: Precool | None, dt: float, opt: Options | None = None):
    """Advance every composition over dt [s] (module doc). The compositions' N, M and overflow_M are updated in
    place; returns an ``UpdateResult``."""
    from .collisions import collide
    from .exchange import Population, exchange_step
    from .sputtering import sputter

    opt = opt or Options()
    unknown = set(opt.processes) - PROCESSES
    if unknown:
        raise ValueError(f"unknown processes {sorted(unknown)}; known: {sorted(PROCESSES)}")
    gas = dict(inputs.gas)
    before = _element_totals(compositions, gas)
    shares, notes, duu = [], [], 0.0
    for k, c in enumerate(compositions):
        active = _active(c, opt.overflow_gate)
        g, N, M = _subgrid(c, active)
        M0 = M.copy()
        over_N, over_M = 0.0, 0.0                       # grown past g's top this substep
        if opt.processes & {"shattering", "coagulation"}:
            st = None if pre is None else pre.states[k]
            r = collide(g, N, M, _collisions(c, g, inputs, opt, st), dt, errmax=opt.errmax, method=c.method)
            N, M = r["N"], r["M"]
            over_N, over_M = over_N + r["overflow_N"], over_M + r["overflow_M"]
            gas[c.key] += r["sink_M"] * c.per_gram
        if "explosion" in opt.processes and pre is not None and pre.explosion[k] is not None:
            N, M, to_gas = _explode(g, N, M, pre.explosion[k], dt, opt, pre.states[k].a)
            gas[c.key] += to_gas * c.per_gram
        if "sputtering" in opt.processes and c.sputtering is not None:
            N, M, dM = sputter(g, N, M, c.erosion(inputs.T, inputs.n_H), dt, method=c.method)
            gas[c.key] += dM * c.per_gram
        if "accretion" in opt.processes and c.sublimation is not None:
            pop = Population(c.sublimation, g, N, M, np.broadcast_to(np.asarray(inputs.T_d, float), (g.n_bins,)).copy(),
                             nucleation=False, method=c.method)
            gk = {c.key: gas[c.key]}
            exchange_step([pop], gk, inputs.T, dt)
            N, M = pop.N, pop.M
            over_N, over_M = over_N + pop.overflow_N, over_M + pop.overflow_M
            gas[c.key] = gk[c.key]
        with np.errstate(divide="ignore", invalid="ignore"):
            ch = np.where(M0 > 0, np.abs(M - M0) / M0, 0.0)
        duu = max(duu, float(ch.max()) if ch.size else 0.0)
        if c.grid.overflow and not active:              # the passive overflow bin takes what grew past g
            c.N[:-1], c.M[:-1] = N, M
            c.N[-1] += over_N
            c.M[-1] += over_M
        else:
            c.N, c.M = N, M
            c.overflow_M += over_M
        share = float(c.M[-1] / c.M.sum()) if c.grid.overflow and c.M.sum() > 0 else 0.0
        shares.append(share)
        if share > opt.overflow_warn:
            msg = f"composition {k} ({c.key}): the overflow bin holds {100 * share:.2f} % of the mass"
            if opt.strict:
                raise RuntimeError(msg)
            warnings.warn(msg, stacklevel=2)
            notes.append(msg)
    after = _element_totals(compositions, gas)
    cons = {e: (after[e] - before[e]) / before[e] if before[e] > 0 else 0.0 for e in before}
    return UpdateResult(gas, shares, duu, cons, notes)


def _element_totals(compositions, gas):
    tot = {e: float(v) for e, v in gas.items()}
    for c in compositions:
        tot[c.key] = tot.get(c.key, 0.0) + (float(np.sum(c.M)) + c.overflow_M) * c.per_gram
    return tot
