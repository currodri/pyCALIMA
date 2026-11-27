"""The per-cell GSD operator: the two calls RAMSES-CALIMA makes per cooling substep.

RAMSES calls the dust twice per cooling substep (``calima/dust_interface.f90``):
- ``precool`` mirrors ``compute_dust_precool``. At the substep's starting state it gives what the gas needs and what
  the dust update uses, per bin: the charge state at the bin's area node, the photoelectric heating, the
  recombination cooling, the grain area and the Coulomb-explosion rate.
- ``update`` mirrors ``compute_dust_update``. It advances the dust over the substep with the precool charges, in
  this operator splitting:
  1. collisions (``collisions.collide``), with the precool charges in the Coulomb factors;
  2. Coulomb explosion, a size-dependent sink applied per moment (``sinks.sink_per_moment``); the exploded grains go
     to a^-alpha fragments (``sinks.deposit_power_law``) or to the gas;
  3. erosion by thermal sputtering (``sputtering.sputter``);
  4. the exchange with the gas (accretion and sublimation, ``exchange.exchange_step``), last, as the stiffest
     coupling.

These two calls are the whole interface. Their inputs and outputs are plain per-cell quantities, and nothing outside
this module calls the operators below for a cell.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .grid import MassGrid

#: the processes ``update`` knows; ``Options.processes`` switches them
PROCESSES = frozenset({"shattering", "coagulation", "explosion", "sputtering", "accretion"})


@dataclass
class Composition:
    """One composition's grains in a cell, and what acts on them.

    - ``N`` [cm^-3] and ``M`` [g cm^-3]: per bin of ``grid``.
    - ``key``: the gas element the grains are made of and return to ('C', 'Si'); ``per_gram``: its atoms per gram of
      grains.
    - ``material``: the shattering material (a ``shattering.MATERIALS`` entry, or a ``materials.GrainMaterial``'s
      ``shatter``).
    - ``sputtering``: the ``sputtering.erosion_rate`` material ('graphite', 'silicate'), a callable
      (T, n_H) -> da/dt(a), or None.
    - ``sublimation``: the ``sublimation_kinetics`` model of accretion and sublimation, or None.
    - ``charging``: a ``charging.AreaNodeCharging``, or None for neutral grains.
    - ``a_frag_min``: the smallest fragment radius [cm]; default the grid's lowest edge.
    - ``s_max``: the tensile strength [erg cm^-3] that sets the Coulomb-explosion charge (``ChargeState.explosion_rate``;
      1e10 for flawed grains, labelled).
    - ``overflow_M``: collects the mass that leaves the top of the grid."""

    grid: MassGrid
    N: np.ndarray
    M: np.ndarray
    key: str
    per_gram: float
    material: object
    method: object = None
    sputtering: str | Callable | None = None
    sublimation: object = None
    charging: object = None
    a_frag_min: float | None = None
    s_max: float = 1e10
    overflow_M: float = 0.0
    _erosion: dict = field(default_factory=dict, repr=False)
    _collision_model: object = field(default=None, repr=False)
    _collision_key: tuple = field(default=None, repr=False)

    def erosion(self, T, n_H) -> Callable:
        """The thermal sputtering da/dt(a) [cm/s] at (T, n_H), cached per cell condition."""
        from .sputtering import erosion_rate

        key = (float(T), float(n_H))
        if key not in self._erosion:
            self._erosion[key] = (self.sputtering(T, n_H) if callable(self.sputtering)
                                  else erosion_rate(self.sputtering, T, n_H, rho=self.grid.rho))
        return self._erosion[key]


@dataclass
class CellInputs:
    """A cell's conditions over the substep.

    - ``T`` [K], ``n_H`` [cm^-3], ``n_e`` [cm^-3].
    - ``gas``: the number densities of the compositions' key elements in the gas [cm^-3]; ``update`` returns their new
      values.
    - ``T_d``: the dust temperature [K].
    - ``v_turb`` [cm/s] and ``L`` [cm]: the turbulence and its driving scale.
    - ``photons`` (``charging.PhotonGroups``), ``G0_bg`` [Habing] and ``ions`` [(n, z, m)]: for the charge.
    - ``neutral``: the neutral fraction of each key element in the gas (1 if absent). Its ionised part accretes with
      the grains' Coulomb factor from precool."""

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
    neutral: dict | None = None


@dataclass
class Options:
    """Switches and numerical parameters of the operator."""

    processes: frozenset = PROCESSES
    errmax: float = 1e-3                 # collide's error control
    n_speed: int = 4                     # GrainCollisions' shattering channels
    table_ppd: float | None = 24         # GrainCollisions' pair table
    threads: int | None = 1
    alpha_frag: float = 3.3              # fragments' dN/da ~ a^-alpha (shattering and explosion)
    explosion_route: str = "fragments"   # 'fragments' (decided 2026-11-23) or 'gas'


@dataclass
class Precool:
    """What precool gives the gas and the update, per composition (lists in the order of the compositions):
    - heating and cooling [erg s^-1 cm^-3] per bin;
    - the charge state on the area nodes (``charging.ChargeState``, or None for neutral grains);
    - the bins' weights w = int a^2 dN / a*^2 [cm^-3];
    - the grain area per unit volume [cm^2 cm^-3] per bin;
    - the Coulomb-explosion rate per grain [s^-1] at each bin's area node, as (a*, k), or None for neutral grains."""

    heating: list
    cooling: list
    states: list
    weights: list
    area: list
    explosion: list

    @property
    def total_heating(self):
        return float(sum(np.sum(h) for h in self.heating))

    @property
    def total_cooling(self):
        return float(sum(np.sum(c) for c in self.cooling))


@dataclass
class UpdateResult:
    """The update's bookkeeping:
    - ``gas``: the key elements' gas densities after the substep;
    - ``duu``: the largest fractional change of any bin's mass (dUU, as RTZ uses it);
    - ``conservation``: the relative change of each key element (dust + gas + overflow), which is round-off."""

    gas: dict
    duu: float
    conservation: dict


def precool(compositions, inputs: CellInputs) -> Precool:
    """The dust at the substep's start, for the gas and the update (module doc).

    For a composition with a charging model, there is one charge solve per bin at its area node
    (``charging.AreaNodeCharging``). The heating is Gamma_i = w_i Gamma(a*_i) and the cooling
    Lambda_i = w_i (Lambda + Lambda_auto)(a*_i). Without a charging model the grains are neutral: no heating or
    cooling. The grain area pi int a^2 dN per bin is always given. The explosion rate is computed at each area node
    with the composition's tensile strength (``Composition.s_max``)."""
    from .charging import PhotonGroups
    from .closures import area_node

    out = Precool([], [], [], [], [], [])
    for c in compositions:
        a_star, w = area_node(c.grid, c.N, c.M, c.method)
        st, expl = None, None
        heat, cool = np.zeros(c.grid.n_bins), np.zeros(c.grid.n_bins)
        if c.charging is not None:
            ph = inputs.photons
            if ph is None:
                L0, L1 = c.charging.L0, c.charging.L1
                ph = PhotonGroups(L0, L1, np.zeros(L0.size), np.sqrt(L0 * L1))
            st, w = c.charging.solve(c.N, c.M, ph, inputs.T, inputs.n_e, inputs.ions, inputs.G0_bg)
            heat = w * st.Gamma
            cool = w * np.array([r["Lambda"] + r["Lambda_auto"] for r in st.results])
            expl = (st.a, st.explosion_rate(S_max=c.s_max))
        for lst, v in ((out.heating, heat), (out.cooling, cool), (out.states, st), (out.weights, w),
                       (out.area, np.pi * w * a_star ** 2), (out.explosion, expl)):
            lst.append(v)
    return out


def update(compositions, inputs: CellInputs, pre: Precool | None, dt: float, opt: Options | None = None):
    """Advance every composition over dt [s] (module doc). The compositions' N, M and overflow_M are updated in
    place; returns an ``UpdateResult``."""
    from .collisions import collide
    from .exchange import exchange_step
    from .sputtering import sputter

    opt = opt or Options()
    unknown = set(opt.processes) - PROCESSES
    if unknown:
        raise ValueError(f"unknown processes {sorted(unknown)}; known: {sorted(PROCESSES)}")
    gas = dict(inputs.gas)
    before = _element_totals(compositions, gas)
    duu = 0.0
    for k, c in enumerate(compositions):
        g, N, M = c.grid, c.N.copy(), c.M.copy()
        M0 = M.copy()
        state = None if pre is None else pre.states[k]
        if opt.processes & {"shattering", "coagulation"}:
            r = collide(g, N, M, _collisions(c, inputs, opt, state), dt, errmax=opt.errmax, method=c.method)
            N, M = r["N"], r["M"]
            c.overflow_M += r["overflow_M"]
            gas[c.key] += r["sink_M"] * c.per_gram
        expl = None if pre is None else pre.explosion[k]
        if "explosion" in opt.processes and expl is not None and np.any(expl[1] > 0.0):
            N, M, to_gas = _explode(c, N, M, expl, dt, opt)
            gas[c.key] += to_gas * c.per_gram
        if "sputtering" in opt.processes and c.sputtering is not None:
            N, M, dM = sputter(g, N, M, c.erosion(inputs.T, inputs.n_H), dt, method=c.method)
            gas[c.key] += dM * c.per_gram
        if "accretion" in opt.processes and c.sublimation is not None:
            pop = _grain_population(c, N, M, inputs, state)
            gk = {c.key: gas[c.key]}
            exchange_step([pop], gk, inputs.T, dt)
            N, M = pop.N, pop.M
            c.overflow_M += pop.overflow_M
            gas[c.key] = gk[c.key]
        with np.errstate(divide="ignore", invalid="ignore"):
            ch = np.where(M0 > 0, np.abs(M - M0) / M0, 0.0)
        duu = max(duu, float(ch.max()) if ch.size else 0.0)
        c.N, c.M = N, M
    after = _element_totals(compositions, gas)
    cons = {e: (after[e] - before[e]) / before[e] if before[e] > 0 else 0.0 for e in before}
    return UpdateResult(gas, duu, cons)


def _collisions(c, inputs, opt, state):
    """The composition's collision model for the cell's conditions. It is kept while they do not change: its pair
    table is built once per condition, as RAMSES's tables are. A new precool charge state makes a new one."""
    from .shattering import GrainCollisions
    from .velocities import RelativeVelocity

    key = (inputs.T, inputs.n_H, inputs.v_turb, inputs.L, id(state), opt.processes, opt.n_speed, opt.table_ppd,
           opt.alpha_frag)
    if c._collision_key != key:
        rv = RelativeVelocity(T=inputs.T, n_H=inputs.n_H, mu=1.4, v_turb=inputs.v_turb, L=inputs.L,
                              charges=None if state is None else state.charge_points)
        c._collision_model = GrainCollisions(
            c.material, rv, law="KT10", alpha_f=opt.alpha_frag, coagulation="coagulation" in opt.processes,
            v_shat=None if "shattering" in opt.processes else 1e30,
            a_frag_min=float(c.grid.a_edges[0] if c.a_frag_min is None else c.a_frag_min), collector="remove",
            n_speed=opt.n_speed, table_ppd=opt.table_ppd, threads=opt.threads)
        c._collision_key = key
    return c._collision_model


def _explode(c, N, M, expl, dt, opt):
    """Coulomb explosion over dt, a sink per moment (``sinks.sink_per_moment``). The rate between the area nodes is
    taken log-linear in a, and constant beyond the first and last nodes (labelled). Returns (N, M, mass to the gas).

    With ``Options.explosion_route`` 'fragments', each bin's exploded mass becomes a^-alpha fragments from
    ``a_frag_min`` up to the radius of half the exploded grains' mean mass (labelled); fragments below the grid go to
    the gas. With 'gas', all of it goes to the gas."""
    from .sinks import deposit_power_law, sink_per_moment

    if opt.explosion_route not in ("fragments", "gas"):
        raise ValueError("explosion_route must be 'fragments' or 'gas'")
    g = c.grid
    la, lk = np.log(expl[0]), np.log(np.maximum(expl[1], 1e-300))
    N, M, dN, dM = sink_per_moment(g, N, M, lambda a: np.exp(np.interp(np.log(a), la, lk)), dt, c.method)
    if opt.explosion_route == "gas":
        return N, M, float(dM.sum())
    a_lo = float(g.a_edges[0] if c.a_frag_min is None else c.a_frag_min)
    to_gas = 0.0
    for i in np.flatnonzero(dM > 0.0):
        a_hi = float(g.radius(0.5 * dM[i] / dN[i]))
        if a_hi <= a_lo:
            to_gas += float(dM[i])
            continue
        fN, fM, below, above = deposit_power_law(g, float(dM[i]), a_lo, a_hi, opt.alpha_frag)
        N, M = N + fN, M + fM
        to_gas += below + above
    return N, M, to_gas


def _grain_population(c, N, M, inputs, state):
    """The composition as an ``exchange.Population`` for accretion and sublimation. The ionised share of the key
    element accretes with the grains' ion Coulomb factor from precool's ``state``."""
    from .exchange import Population

    x_n = 1.0 if inputs.neutral is None else float(inputs.neutral.get(c.key, 1.0))
    focus = None if state is None or x_n >= 1.0 else state.ion_focusing_of()
    T_d = np.full(c.grid.n_bins, float(inputs.T_d))
    return Population(c.sublimation, c.grid, N.copy(), M.copy(), T_d, nucleation=False, method=c.method,
                      neutral_fraction=x_n, ion_focusing=focus)


def _element_totals(compositions, gas):
    tot = {e: float(v) for e, v in gas.items()}
    for c in compositions:
        tot[c.key] = tot.get(c.key, 0.0) + (float(np.sum(c.M)) + c.overflow_M) * c.per_gram
    return tot
