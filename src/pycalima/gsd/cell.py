"""The per-cell GSD operator: the two calls RAMSES-CALIMA makes per cooling substep.

RAMSES calls the dust twice per cooling substep (``calima/dust_interface.f90``):
- ``precool`` mirrors ``compute_dust_precool``. At the substep's starting state it gives what the gas needs and what
  the dust update uses, per bin: the charge state at the bin's area node, the photoelectric heating, the
  recombination cooling, the grain area and the Coulomb-explosion rate.
- ``update`` mirrors ``compute_dust_update``. It advances the dust over the substep with the precool charges, in
  this operator splitting:
  1. collisions (``collisions.collide``), with the precool charges in the Coulomb factors and, for carbon, the
     pair rule of its molecules (``pairs``);
  2. Coulomb explosion, a size-dependent sink applied per moment (``sinks.sink_per_moment``); the exploded grains go
     to a^-alpha fragments (``sinks.deposit_power_law``) or to the gas;
  3. erosion by thermal sputtering (``sputtering.sputter``);
  4. the exchange with the gas (accretion and sublimation, ``exchange.exchange_step``), last, as the stiffest
     coupling.

Carbon's molecular bins (``Composition.molecular_bins``) skip the grain explosion and accretion. Their own physics
comes from the PAH operator (J1).

These two calls are the whole interface. Their inputs and outputs are plain per-cell quantities, and nothing outside
this module calls the operators below for a cell.

A composition's grid may end in the overflow bin (``grid.MassGrid.with_overflow``, 1-10 um, decided 2026-11-23).
While it holds at most ``Options.overflow_gate`` of the composition's mass it is carried passively: the operators
run on the grid below it, and what grows past that grid's top is added to it. Above the gate it is an ordinary bin.
``update`` reports its share, with a warning above ``Options.overflow_warn`` (``strict``: an error).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import warnings

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
    - ``optics``: an ``optics.PseudoBinOptics`` on ``grid``, or None. With it, precool computes the dust temperature
      per bin (which the exchange uses) and the group absorption. ``hydrogen`` is then the ``hydrogen.BinHydrogen`` of
      its covalent bins (``optics.gap_bins``, in order): precool also finds their equilibrium band gap, at which the
      optics and the charging (with the same ``gap_bins``) are taken.
    - ``pair_rule``: carbon's ``pairs.PairRule`` (what molecules do in collisions), or None: every pair as grains.
    - ``molecular_bins``: how many of the lowest bins are molecules (carbon's PAH bins, N_C up to 400; 4 on
      ``CarbonLayout``). Two grain processes skip them:
      - the grain Coulomb explosion, since a molecule's charge and fission follow pahchem's routing in the PAH
        operator (R1);
      - accretion, since molecular growth is off (decided 2026-11-23).
      Fragments of exploding grains still land in them, and collisions treat them by the pair rule.
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
    optics: object = None
    hydrogen: list | None = None
    pair_rule: object = None
    molecular_bins: int = 0
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
      the grains' Coulomb factor from precool.
    - ``n_HI``: the atomic hydrogen [cm^-3] that re-hydrogenates the covalent bins (None: n_H, all atomic, labelled)."""

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
    n_HI: float | None = None


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
    overflow_gate: float = 1e-6          # the overflow bin is passive at or below this share of the mass
    overflow_warn: float = 0.01          # warn above this share
    strict: bool = False                 # raise instead of warning


@dataclass
class Precool:
    """What precool gives the gas and the update, per composition (lists in the order of the compositions):
    - heating and cooling [erg s^-1 cm^-3] per bin;
    - the charge state on the area nodes (``charging.ChargeState``, or None for neutral grains);
    - the bins' weights w = int a^2 dN / a*^2 [cm^-3];
    - the grain area per unit volume [cm^2 cm^-3] per bin;
    - the Coulomb-explosion rate per grain [s^-1] at each bin's area node, as (a*, k), or None for neutral grains;
    - with optics: the dust temperature per bin [K], the group absorption kappa_g = sum_i M_i kappa_g,i [cm^-1] per
      group and, with covalent bins, their band gaps [eV] (None otherwise)."""

    heating: list
    cooling: list
    states: list
    weights: list
    area: list
    explosion: list
    t_dust: list = field(default_factory=list)
    absorption: list = field(default_factory=list)
    e_gap: list = field(default_factory=list)

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
    - ``conservation``: the relative change of each key element (dust + gas + overflow), which is round-off;
    - ``overflow_share``: each composition's overflow-bin share of its mass (0 without one)."""

    gas: dict
    duu: float
    conservation: dict
    overflow_share: list


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
        t_d, kap, gap = _optics(c, inputs) if c.optics is not None else (None, None, None)
        heat, cool = np.zeros(c.grid.n_bins), np.zeros(c.grid.n_bins)
        if c.charging is not None:
            ph = inputs.photons
            if ph is None:
                L0, L1 = c.charging.L0, c.charging.L1
                ph = PhotonGroups(L0, L1, np.zeros(L0.size), np.sqrt(L0 * L1))
            gaps = None
            if gap is not None and c.charging.gap_bins.size:
                if not np.array_equal(c.charging.gap_bins, c.optics.gap_bins):
                    raise ValueError("the charging's covalent bins differ from the optics'")
                gaps = gap
            st, w = c.charging.solve(c.N, c.M, ph, inputs.T, inputs.n_e, inputs.ions, inputs.G0_bg, gaps=gaps)
            heat = w * st.Gamma
            cool = w * np.array([r["Lambda"] + r["Lambda_auto"] for r in st.results])
            expl = (st.a, st.explosion_rate(S_max=c.s_max))
        for lst, v in ((out.heating, heat), (out.cooling, cool), (out.states, st), (out.weights, w),
                       (out.area, np.pi * w * a_star ** 2), (out.explosion, expl), (out.t_dust, t_d),
                       (out.absorption, kap), (out.e_gap, gap)):
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
    duu, shares = 0.0, []
    for k, c in enumerate(compositions):
        passive = c.grid.overflow and not _overflow_active(c, opt.overflow_gate)
        if passive:
            g, N, M = MassGrid(c.grid.m_edges[:-1], c.grid.rho), c.N[:-1].copy(), c.M[:-1].copy()
        else:
            g, N, M = c.grid, c.N.copy(), c.M.copy()
        M0 = M.copy()
        over_N, over_M = 0.0, 0.0                       # grown past g's top this substep
        state = None if pre is None else pre.states[k]
        if opt.processes & {"shattering", "coagulation"}:
            r = collide(g, N, M, _collisions(c, inputs, opt, state), dt, errmax=opt.errmax, method=c.method)
            N, M = r["N"], r["M"]
            over_N, over_M = over_N + r["overflow_N"], over_M + r["overflow_M"]
            gas[c.key] += r["sink_M"] * c.per_gram
        expl = None if pre is None else pre.explosion[k]
        if "explosion" in opt.processes and expl is not None and np.any(expl[1] > 0.0):
            N, M, to_gas = _explode(c, g, N, M, expl, dt, opt)
            gas[c.key] += to_gas * c.per_gram
        if "sputtering" in opt.processes and c.sputtering is not None:
            N, M, dM = sputter(g, N, M, c.erosion(inputs.T, inputs.n_H), dt, method=c.method)
            gas[c.key] += dM * c.per_gram
        if "accretion" in opt.processes and c.sublimation is not None:
            mb = c.molecular_bins
            t_d = None if pre is None or k >= len(pre.t_dust) or pre.t_dust[k] is None else pre.t_dust[k][mb:g.n_bins]
            pop = _grain_population(c, MassGrid(g.m_edges[mb:], g.rho), N[mb:], M[mb:], inputs, state, t_d)
            gk = {c.key: gas[c.key]}
            exchange_step([pop], gk, inputs.T, dt)
            N, M = np.concatenate([N[:mb], pop.N]), np.concatenate([M[:mb], pop.M])
            over_N, over_M = over_N + pop.overflow_N, over_M + pop.overflow_M
            gas[c.key] = gk[c.key]
        with np.errstate(divide="ignore", invalid="ignore"):
            ch = np.where(M0 > 0, np.abs(M - M0) / M0, 0.0)
        duu = max(duu, float(ch.max()) if ch.size else 0.0)
        if passive:                                     # the passive overflow bin takes what grew past g
            c.N, c.M = np.append(N, c.N[-1] + over_N), np.append(M, c.M[-1] + over_M)
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
    after = _element_totals(compositions, gas)
    cons = {e: (after[e] - before[e]) / before[e] if before[e] > 0 else 0.0 for e in before}
    return UpdateResult(gas, duu, cons, shares)


def _overflow_active(c, gate):
    """True if the composition's overflow bin holds more than ``gate`` of its mass."""
    tot = float(np.sum(c.M))
    return tot > 0.0 and c.M[-1] > gate * tot


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
            n_speed=opt.n_speed, table_ppd=opt.table_ppd, threads=opt.threads, pair_rule=c.pair_rule)
        c._collision_key = key
    return c._collision_model


def _optics(c, inputs):
    """(T_d per bin [K], the group absorption kappa_g [cm^-1] per group, the covalent bins' band gaps [eV] or None)
    of a composition with optics.

    Each covalent bin's band gap is the equilibrium of its hydrogen (``hydrogen.equilibrium_gap``): k_UV is Y_CH
    times the 6.8-13.6 eV photons it absorbs per C atom at each band-gap node, and k_H comes from n(H I) and T. Then
    the absorbed power at those gaps gives the temperature that radiates it."""
    from .hydrogen import Y_CH, equilibrium_gap, k_h
    from .optics import state_s

    po = c.optics
    s = state_s(c.grid, c.N, c.M)
    gap = None
    if po.gap is not None:
        if c.hydrogen is None or len(c.hydrogen) != po.gap_bins.size:
            raise ValueError("a composition with covalent bins needs their BinHydrogen (Composition.hydrogen)")
        uv = po.uv_photons(s, inputs.photons, inputs.G0_bg) * _M_CARBON          # photons per C atom per s
        kh = k_h(inputs.n_H if inputs.n_HI is None else inputs.n_HI, inputs.T)
        gap = np.array([equilibrium_gap(bh, Y_CH * np.interp(bh.e_grid, po.gap_nodes, uv[j]), kh)[1]
                        for j, bh in enumerate(c.hydrogen)])
    kg, power, _ = po.absorption(s, inputs.photons, inputs.G0_bg, gap)
    return po.dust_temperature(s, power, gap), c.M @ kg, gap


_M_CARBON = 12.011 * 1.66053906660e-24


def _explode(c, g, N, M, expl, dt, opt):
    """Coulomb explosion over dt, a sink per moment (``sinks.sink_per_moment``). The rate between the area nodes is
    taken log-linear in a, and constant beyond the first and last nodes (labelled). It is zero on the molecular bins
    (``Composition.molecular_bins``). Returns (N, M, mass to the gas).

    With ``Options.explosion_route`` 'fragments', each bin's exploded mass becomes a^-alpha fragments from
    ``a_frag_min`` up to the radius of half the exploded grains' mean mass (labelled); fragments below the grid go to
    the gas. With 'gas', all of it goes to the gas."""
    from .sinks import deposit_power_law, sink_per_moment

    if opt.explosion_route not in ("fragments", "gas"):
        raise ValueError("explosion_route must be 'fragments' or 'gas'")
    la, lk = np.log(expl[0]), np.log(np.maximum(expl[1], 1e-300))
    N, M, dN, dM = sink_per_moment(g, N, M, lambda a: np.exp(np.interp(np.log(a), la, lk)), dt, c.method,
                                   active=np.arange(g.n_bins) >= c.molecular_bins)
    # At k dt ~ 1e-16 or less the surviving number fraction rounds to 1 while the mass fraction does not: mass removed
    # without number, a round-off artefact. It stays in its bin (exact conservation, no 0/0 below).
    lone = (dM > 0.0) & (dN <= 0.0)
    M, dM = np.where(lone, M + dM, M), np.where(lone, 0.0, dM)
    if opt.explosion_route == "gas":
        return N, M, float(dM.sum())
    a_lo = float(g.a_edges[0] if c.a_frag_min is None else c.a_frag_min)
    to_gas = 0.0
    for i in np.flatnonzero(dM > 0.0):
        m_bar = min(max(dM[i] / dN[i], g.m_edges[i]), g.m_edges[i + 1])     # the exploded grains' mean mass
        a_hi = float(g.radius(0.5 * m_bar))
        if a_hi <= a_lo:
            to_gas += float(dM[i])
            continue
        fN, fM, below, above = deposit_power_law(g, float(dM[i]), a_lo, a_hi, opt.alpha_frag)
        N, M = N + fN, M + fM
        to_gas += below + above
    return N, M, to_gas


def _grain_population(c, g, N, M, inputs, state, t_dust=None):
    """The composition as an ``exchange.Population`` for accretion and sublimation. The ionised share of the key
    element accretes with the grains' ion Coulomb factor from precool's ``state``.

    The sublimation model's volume per unit must be the grains' own. A model of another density (e.g. graphite's 2.24
    on THEMIS a-C's grid at 1.6) would scale the uptake rate by the density ratio, while the mass bookkeeping still
    balances. So it raises; ``materials.get_material(...).sublimation_model()`` gives the matching model."""
    from .exchange import Population

    if not np.isclose(c.sublimation.density_gcm3, g.rho, rtol=1e-6):
        raise ValueError(f"the sublimation model's density ({c.sublimation.density_gcm3} g/cm3) differs from the "
                         f"grid's ({g.rho} g/cm3)")

    x_n = 1.0 if inputs.neutral is None else float(inputs.neutral.get(c.key, 1.0))
    focus = None if state is None or x_n >= 1.0 else state.ion_focusing_of()
    T_d = np.full(g.n_bins, float(inputs.T_d)) if t_dust is None else np.asarray(t_dust, float).copy()
    return Population(c.sublimation, g, N.copy(), M.copy(), T_d, nucleation=False, method=c.method,
                      neutral_fraction=x_n, ion_focusing=focus)


def _element_totals(compositions, gas):
    tot = {e: float(v) for e, v in gas.items()}
    for c in compositions:
        tot[c.key] = tot.get(c.key, 0.0) + (float(np.sum(c.M)) + c.overflow_M) * c.per_gram
    return tot
