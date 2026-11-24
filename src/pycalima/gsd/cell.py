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

Carbon's molecular bins (``Composition.molecular_bins``) take their charge, heating, cooling and photon absorption
from pahchem's group tables and their carbon loss from the hydrogen tables (``molecular.MolecularTables``); precool
then also gives the collisions the molecules' charges and the photo-release factor S_eff of molecules sticking to
larger particles (``release``), per molecular bin and host bin.

These two calls are the whole interface: their inputs and outputs are plain per-cell quantities, and nothing
outside this module calls the operators below for a cell. A composition's grid may carry the overflow bin
(``grid.MassGrid.with_overflow``): while it holds at most ``Options.overflow_gate`` of the composition's mass it is
carried passively (the operators run on the grid below it, and what grows past that grid's top is added to it); above
the gate it is an ordinary bin. Its share is reported, with a warning above 1 % (``strict``: an error).
"""
from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .grid import MassGrid

#: the processes ``update`` knows; ``Options.processes`` switches them
PROCESSES = frozenset({"shattering", "coagulation", "explosion", "sputtering", "carbon_loss", "accretion", "hydrogen"})
#: the mass of an H atom [g] (tracked hydrogen exchanged with the gas)
M_H_ATOM = 1.6735575e-24


@dataclass
class Composition:
    """One composition's grains in a cell, and what acts on them.

    ``N`` [cm^-3] and ``M`` [g cm^-3] per bin of ``grid``. ``key``: the gas element the grains are made of and return
    to ('C', 'Si'), ``per_gram`` its atoms per gram of grains. ``material``: the shattering material
    (``shattering.MATERIALS`` entry, or a ``materials.GrainMaterial``'s ``shatter``). ``sputtering``: the
    ``sputtering.erosion_rate`` material ('graphite', 'silicate'), a callable (T, n_H) -> da/dt(a), or None;
    ``a_frag_min``: the smallest fragment radius [cm] (default the grid's lowest edge); ``molecular_bins``: how many
    of the lowest bins are molecular (carbon below N_C,max): the grain processes skip them (accretion, since molecular
    growth is off by default; classical sputtering and grain Coulomb explosion, since their loss follows the molecular
    tables), while collisions treat them by the pair rule and grains eroded from above join them;
    ``molecular_tables``: their tables (``molecular.MolecularTables``): carbon loss (process 'carbon_loss'), and with
    group tables their charge, heating and cooling in precool and S_eff in the pair rule; ``sublimation``: the
    ``sublimation_kinetics`` model of accretion and sublimation, or None; ``charging``: a
    ``charging.AreaNodeCharging`` (precool), or None for neutral grains; ``pair_rule``: carbon's ``pairs.PairRule``,
    or None; ``optics``: an ``optics.PseudoBinOptics`` on ``grid`` (precool then computes the dust temperature per
    bin, which update uses, and the group absorption), and ``hydrogen`` the ``hydrogen.BinHydrogen`` of its covalent
    bins (``optics.gap_bins``, in order) for their equilibrium band gap; ``M_H``: the hydrogen mass per bin [g cm^-3]
    (tracer, rung M1; None: not tracked). Tracked, it rides with the grains through every operator at each donor
    bin's H/C (growing grains keep their hydrogen, shrinking ones their H/C) and the covalent bins with ``hydrogen``
    take its exact step (process 'hydrogen'; their band gap then follows their H/C instead of the equilibrium);
    ``overflow_H`` collects what leaves the top of the grid. ``overflow_M`` collects what leaves the top of the grid (past the overflow bin, if any)."""

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
    a_frag_min: float | None = None
    molecular_bins: int = 0
    molecular_tables: object = None
    optics: object = None
    hydrogen: list | None = None
    M_H: np.ndarray | None = None
    overflow_M: float = 0.0
    overflow_H: float = 0.0
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
    """A cell's conditions over the substep: T [K], n(H) [cm^-3], n_e [cm^-3], ``gas`` the number densities of the
    compositions' key elements in the gas [cm^-3] (``update`` returns their new values); the dust temperature T_d
    [K] (for compositions without ``optics``; with them precool computes it per bin); turbulence v_turb [cm/s] and its
    driving scale L [cm]; the RT photon groups (``charging.PhotonGroups``), the UV background G0_bg [Habing] and the
    ions [(n, z, m)] for the charge; ``neutral`` the neutral fraction of each key element in the gas (1 if absent):
    its ionised part accretes with the grains' Coulomb factor from precool; ``n_HI`` [cm^-3] the atomic hydrogen
    that re-hydrogenates the covalent bins (None: n_H, all atomic, labelled). With tracked hydrogen
    (``Composition.M_H``) ``gas['H']`` [atoms cm^-3] takes the H the grains release and gives what they take up (0
    if absent: it then holds the net exchange of the substep)."""

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
    table_ppd: float | None = 24         # GrainCollisions' pair table (ignored with a pair rule)
    threads: int | None = 1
    alpha_frag: float = 3.3              # fragments' dN/da ~ a^-alpha (shattering and explosion)
    explosion_route: str = "fragments"   # 'fragments' (user, 2026-11-23) or 'gas'
    overflow_gate: float = 1e-6          # the overflow bin is passive at or below this share of the mass
    overflow_warn: float = 0.01
    strict: bool = False                 # raise instead of warning when the overflow bin fills up
    s_max: float = 1e10                  # tensile strength of the grains [erg cm^-3] (flawed a-C, labelled)
    y_pd: float = 0.0                    # photodesorption yield of molecules on nano-grains ('calibrated', A5)
    t_bury_max: float = 3.15576e16       # [s] cap of the burial time (1 Gyr, labelled): slower release counts as none


def precool(compositions, inputs: CellInputs, opt: Options | None = None) -> Precool:
    """The dust at the substep's start, for the gas and the update (module doc).

    Per composition with a charging model: one charge solve per bin at its area node (``charging.AreaNodeCharging``;
    its covalent bins at precool's band gaps when it and the optics have them);
    heating Gamma_i = w_i Gamma(a*_i) and cooling Lambda_i = w_i (Lambda + Lambda_auto)(a*_i) with the bins' weights
    w_i = int a^2 dN / a*_i^2; the explosion rate at each area node (``ChargeState.explosion_rate`` with
    ``Options.s_max``), held over the bin. Without a charging model the grains are neutral: no heating, cooling or
    explosion. Always: the grain area pi int a^2 dN per bin (``closures.area_node``).

    With molecular group tables (``Composition.molecular_tables``): the molecular bins' heating and cooling are
    N_i Gamma_i and N_i Lambda_i of their charge balance (``MolecularTables.charge``), in place of the grain tables';
    their charges enter the collisions; and with a pair rule, S_eff of each molecular bin on each larger bin
    (``_release``)."""
    from .charging import PhotonGroups
    from .closures import area_node

    opt = opt or Options()
    out = Precool([], [], [], [], [], [])
    for c in compositions:
        a_star, w = area_node(c.grid, c.N, c.M, c.method)
        out.area.append(np.pi * w * a_star ** 2)
        t_d, kap, gap, uvg = _optics(c, inputs) if c.optics is not None else (None, None, None, None)
        if c.charging is None:
            st, expl = None, None
            heat, cool = np.zeros(c.grid.n_bins), np.zeros(c.grid.n_bins)
        else:
            ph = inputs.photons
            if ph is None:
                L0, L1 = c.charging.L0, c.charging.L1
                ph = PhotonGroups(L0, L1, np.zeros(L0.size), np.sqrt(L0 * L1))
            gaps = None
            if gap is not None and getattr(c.charging, "gap_bins", np.zeros(0)).size:
                if not np.array_equal(c.charging.gap_bins, c.optics.gap_bins):
                    raise ValueError("the charging's covalent bins differ from the optics'")
                gaps = gap
            st, w = c.charging.solve(c.N, c.M, ph, inputs.T, inputs.n_e, inputs.ions, inputs.G0_bg, gaps=gaps)
            heat = w * st.Gamma
            cool = w * np.array([r["Lambda"] + r["Lambda_auto"] for r in st.results])
            expl = st.explosion_rate(S_max=opt.s_max) if "explosion" in opt.processes else None
        mol, S, h2 = None, None, None
        mt, mb = c.molecular_tables, c.molecular_bins
        if mt is not None and mb > 0:
            from .molecular import habing_g0

            g0 = inputs.G0_bg + habing_g0(inputs.photons)
            h2 = c.N[:mb] * mt.observable("h2", g0, inputs.n_H, inputs.T, inputs.n_e)[:mb]
        if mt is not None and mt.groups is not None and mb > 0:
            mol = mt.charge(inputs.photons, inputs.T, inputs.n_e, inputs.ions, inputs.G0_bg)
            heat[:mb] = c.N[:mb] * mol.gamma
            cool[:mb] = c.N[:mb] * mol.lam
            if c.pair_rule is not None:
                S = _release(c, inputs, opt, st, mol, a_star, t_d)
        for lst, v in ((out.heating, heat), (out.cooling, cool), (out.states, st), (out.weights, w),
                       (out.explosion, expl), (out.molecular, mol), (out.release, S), (out.h2, h2),
                       (out.t_dust, t_d), (out.absorption, kap), (out.e_gap, gap), (out.uv_gap, uvg)):
            lst.append(v)
    return out


def _release(c, inputs, opt, state, mol, a_star, t_dust=None):
    """S_eff [molecular bins, larger bins] (``release``): 1 / (1 + release_rate t_bury) on every host bin above the
    molecular ones (photodesorption acts on the whole population, 2026-11-24), each molecular bin at its mean N_C,
    each host at its area node a*, t_bury from the host's growth by accretion in the cell (the
    accretion law of ``update``, net of sublimation; growth by coagulation and by the molecules themselves is not
    counted, labelled: it would bury faster), capped at ``Options.t_bury_max`` (without growth it is infinite, and a
    release rate that underflows to ~1e-300 s^-1 on a 10 nm host would still empty it)."""
    from .release import burial_time, release_rate, s_eff

    g, mb = c.grid, c.molecular_bins
    hosts = list(range(mb, g.n_bins))
    m_c = np.sqrt(g.m_edges[:-1] * g.m_edges[1:])
    with np.errstate(divide="ignore", invalid="ignore"):
        nc = np.where(c.N > 0, c.M / np.maximum(c.N, 1e-300), m_c) / M_CARBON
        a_h = np.where(np.isfinite(a_star) & (a_star > 0), a_star, g.a_centres)[hosts]
    if c.sublimation is not None and "accretion" in opt.processes and hosts:
        pop = _grain_population(c, g, c.N, c.M, inputs, state, t_dust)
        t_bury = burial_time(np.maximum(pop.adot(inputs.gas[c.key], inputs.T)(a_h), 0.0))
    else:
        t_bury = np.full(len(hosts), np.inf)
    t_bury = np.minimum(t_bury, opt.t_bury_max)
    S = np.ones((mb, len(hosts)))
    for i in range(mb):
        for jj in range(len(hosts)):
            rate = release_rate(nc[i], a_h[jj], mol.absorbed[i], mol.energy, rho=g.rho, y_pd=opt.y_pd)
            S[i, jj] = float(s_eff(rate, 1.0, t_bury[jj], 0.0))
    return S


def _optics(c, inputs):
    """(T_d per bin [K], group absorption [cm^-1] per group, the covalent bins' band gaps [eV] and their UV photons
    absorbed per C atom at every band-gap node [s^-1], or None) of a composition with optics: the band gap of each
    covalent bin from the equilibrium of its hydrogen (``hydrogen.equilibrium_gap``: k_UV = Y_CH times those
    photons, k_H from n(H I) and T), or from its tracked H/C (``Composition.M_H``), then the absorbed power at those
    gaps and the temperature that radiates it."""
    from .hydrogen import Y_CH, equilibrium_gap, k_h
    from .optics import state_s

    po = c.optics
    s = state_s(c.grid, c.N, c.M)
    gap, uv = None, None
    if po.gap is not None:
        if c.hydrogen is None or len(c.hydrogen) != po.gap_bins.size:
            raise ValueError("a composition with covalent bins needs their BinHydrogen (Composition.hydrogen)")
        uv = po.uv_photons(s, inputs.photons, inputs.G0_bg) * M_CARBON           # photons per C atom per s
        if c.M_H is not None:
            gap = np.array([float(bh.e_gap(_h_per_c(c.M_H[i], c.M[i], bh))) for i, bh in zip(po.gap_bins, c.hydrogen,
                                                                                             strict=True)])
        else:
            n_hi = inputs.n_H if inputs.n_HI is None else inputs.n_HI
            kh = k_h(n_hi, inputs.T)
            gap = np.array([equilibrium_gap(bh, Y_CH * np.interp(bh.e_grid, po.gap_nodes, uv[j]), kh)[1]
                            for j, bh in enumerate(c.hydrogen)])
    kg, power, _ = po.absorption(s, inputs.photons, inputs.G0_bg, gap)
    return po.dust_temperature(s, power, gap), c.M @ kg, gap, uv


def _h_per_c(m_h, m, bh):
    """A bin's H/C (H mass per grain mass), inside the bin's range [x_min, x_max] (x_max if empty)."""
    return bh.x_max if m <= 0.0 else float(np.clip(m_h / m, bh.x_min, bh.x_max))


def _hydrogen_step(c, H, M, inputs, uv, dt):
    """The exact step of the tracked covalent bins' hydrogen with the rates frozen at their H/C (hydrogen module:
    dx/dt = -k_UV (x - x_min) + k_H (x_max - x), k_UV = Y_CH times the UV photons absorbed per C atom at the bin's
    band gap, k_H from n(H I) and T). Returns (H, H given to the gas [g cm^-3])."""
    from .hydrogen import Y_CH, k_h

    po = c.optics
    kh = k_h(inputs.n_H if inputs.n_HI is None else inputs.n_HI, inputs.T)
    H = H.copy()
    released = 0.0
    for j, (i, bh) in enumerate(zip(po.gap_bins, c.hydrogen, strict=True)):
        if i >= H.size or M[i] <= 0.0:
            continue
        x = float(H[i] / M[i])
        ku = Y_CH * float(np.interp(bh.e_gap(min(max(x, bh.x_min), bh.x_max)), po.gap_nodes, uv[j]))
        k = ku + kh
        if k <= 0.0:
            continue
        x_eq = (ku * bh.x_min + kh * bh.x_max) / k
        x_new = x_eq + (x - x_eq) * math.exp(-k * dt)
        released += (x - x_new) * M[i]
        H[i] = x_new * M[i]
    return H, released


def _grain_population(c, g, N, M, inputs, state, t_dust=None):
    """The grain bins (above the molecular ones) as an ``exchange.Population`` for accretion and sublimation: the
    ionised share of the key element accretes with the grains' ion Coulomb factor from precool's ``state``; the
    dust temperature per bin is precool's ``t_dust`` if given, else ``inputs.T_d``."""
    from .exchange import Population

    mb = c.molecular_bins
    gg = g if mb == 0 else MassGrid(g.m_edges[mb:], g.rho)
    x_n = 1.0 if inputs.neutral is None else float(inputs.neutral.get(c.key, 1.0))
    focus = None if state is None or x_n >= 1.0 else state.ion_focusing_of()
    T_d = inputs.T_d if t_dust is None else np.asarray(t_dust, float)[:g.n_bins]
    return Population(c.sublimation, gg, N[mb:].copy(), M[mb:].copy(),
                      np.broadcast_to(np.asarray(T_d, float), (g.n_bins,))[mb:].copy(),
                      nucleation=False, method=c.method, neutral_fraction=x_n, ion_focusing=focus)


@dataclass
class Precool:
    """What precool gives the gas and the update, per composition (lists in the order of the compositions):
    heating and cooling [erg s^-1 cm^-3], the charge state on the area nodes (``charging.ChargeState``) and the bins'
    weights, the explosion rate per bin [s^-1] and the grain area per unit volume [cm^2 cm^-3]; with molecular group
    tables the molecular bins' state (``molecular.MolecularState``) and S_eff [molecular bins, larger bins]; with
    molecular tables the H2 the molecular bins release [cm^-3 s^-1] per bin (the hydrogen tables' 'h2', at the cell's
    G0 = G0_bg + the groups' 6-13.6 eV); with optics the dust temperature per bin [K], the group absorption
    kappa_g = sum_i M_i kappa_g,i [cm^-1] per group, and with hydrogen the covalent bins' band gaps [eV]; None
    otherwise, or when absent."""

    heating: list
    cooling: list
    states: list
    weights: list
    explosion: list
    area: list
    molecular: list = field(default_factory=list)
    release: list = field(default_factory=list)
    h2: list = field(default_factory=list)
    t_dust: list = field(default_factory=list)
    absorption: list = field(default_factory=list)
    e_gap: list = field(default_factory=list)
    uv_gap: list = field(default_factory=list)

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


def _collisions(c, g, inputs, opt, pre_state, mol=None, S=None):
    """The composition's collision model for the cell's conditions, kept while they do not change (its pair table is
    built once per condition, as RAMSES's tables are); a charge state from precool makes a new one. ``mol``: the
    molecular bins' charges replace the grain state's on the molecular radii; ``S``: the pair rule's S_eff."""
    from dataclasses import replace

    from .shattering import GrainCollisions
    from .velocities import RelativeVelocity

    mb = c.molecular_bins
    charges = _charge_points(pre_state, mol, g.a_edges, mb)
    key = (inputs.T, inputs.n_H, inputs.v_turb, inputs.L, id(pre_state), g.n_bins, opt.processes, opt.n_speed,
           opt.table_ppd, opt.alpha_frag, id(mol), id(S))
    if c._collision_key != key:
        rule = c.pair_rule
        if rule is not None and S is not None:
            rule = replace(rule, release=_release_lookup(S, g.a_edges, mb))
        rv = RelativeVelocity(T=inputs.T, n_H=inputs.n_H, mu=1.4, v_turb=inputs.v_turb, L=inputs.L, charges=charges)
        c._collision_model = GrainCollisions(
            c.material, rv, law="KT10", alpha_f=opt.alpha_frag, coagulation="coagulation" in opt.processes,
            v_shat=None if "shattering" in opt.processes else 1e30,
            a_frag_min=float(g.a_edges[0] if c.a_frag_min is None else c.a_frag_min), collector="remove",
            n_speed=opt.n_speed, table_ppd=opt.table_ppd, threads=opt.threads, pair_rule=rule)
        c._collision_key = key
    return c._collision_model


_GH_OFFSETS = np.sqrt(3.0) * np.array([-1.0, 0.0, 1.0])
_GH_WEIGHTS = np.array([1.0, 4.0, 1.0]) / 6.0


def _charge_points(state, mol, edges, mb):
    """The collisions' charges(a) -> (Z[..., 3], w[3]) (``ChargeState.charge_points``): the grain state's, with the
    molecular bins' three Gauss-Hermite charges of their mean and width below N_C,max (grains neutral without a
    state); None if neither."""
    if mol is None:
        return None if state is None else state.charge_points
    a_mol = float(edges[mb])

    def points(a):
        a = np.asarray(a, float)
        Z = np.zeros((*a.shape, 3)) if state is None else np.array(state.charge_points(a)[0], float)
        i = np.clip(np.searchsorted(edges, a, side="right") - 1, 0, mb - 1)
        Zm = mol.mean[i][..., None] + mol.width[i][..., None] * _GH_OFFSETS
        return np.where((a < a_mol)[..., None], Zm, Z), _GH_WEIGHTS

    return points


def _release_lookup(S, edges, mb):
    """The pair rule's release(a_molecule, a_host): S_eff of the molecular bin and host bin holding the radii."""
    nh = S.shape[1]

    def release(a_m, a_h):
        i = np.clip(np.searchsorted(edges, a_m, side="right") - 1, 0, mb - 1)
        j = np.clip(np.searchsorted(edges, a_h, side="right") - 1 - mb, 0, nh - 1)
        return S[i, j]

    return release


def _at(lst, k):
    return lst[k] if lst is not None and k < len(lst) else None


def _erosion_rate(c, g, inputs, grains, molecules):
    """(da/dt(a) [cm/s], knots) of the erosion step: classical sputtering of the grain bins (stopping at N_C,max, so
    that grains eroded from above join the molecular bins) and the molecular bins' tabulated carbon loss, each C atom
    lost per molecule a mass m_C, as a shift (what erodes below the grid goes to the gas). The molecular rate jumps at
    the molecular bins' edges; the knots (a pair at each) keep the velocity table from smearing a jump into the bin
    above, which would leak the first grain bin into the molecules where only the molecules erode."""
    from .molecular import habing_g0

    mb = c.molecular_bins
    edges = g.a_edges
    a_mol = float(edges[mb]) if mb else 0.0
    f_gr = c.erosion(inputs.T, inputs.n_H) if grains else None
    dmdt = None
    if molecules:
        g0 = inputs.G0_bg + habing_g0(inputs.photons)
        dmdt = -c.molecular_tables.carbon_loss(g0, inputs.n_H, inputs.T, inputs.n_e)[:mb] * M_CARBON

    def adot(a):
        a = np.asarray(a, float)
        out = np.zeros(a.shape)
        if f_gr is not None:                    # with the molecules on, grains at N_C,max keep moving into them
            out = np.where(a >= a_mol if molecules else a > a_mol, f_gr(a), 0.0)
        if dmdt is not None:
            i = np.clip(np.searchsorted(edges, a, side="right") - 1, 0, mb - 1)
            out = np.where(a < a_mol, dmdt[i] / (4.0 * np.pi * g.rho * a * a), out)
        return out

    knots = None
    if molecules:
        e = np.asarray(edges[1:mb + 1], float)
        knots = np.concatenate([e * (1.0 - 1e-9), e])
    return adot, knots


M_CARBON = 12.011 * 1.66053906660e-24


def _explode(g, N, M, k_bin, dt, opt, a_star, H=None):
    """Coulomb explosion at the per-bin rates k_bin [s^-1] (precool's rate at each bin's area node, held over the
    bin: labelled), a sink that removes N and M alike, exactly; returns (N, M, mass to the gas), and with ``H``
    (H, H to the gas): the fragments carry their parent's H/C."""
    from .sinks import deposit_power_law

    nb = g.n_bins
    k = np.asarray(k_bin, float)[:nb]
    if not np.any(k > 0.0):
        return (N, M, 0.0) if H is None else (N, M, 0.0, H, 0.0)
    edges = g.a_edges
    keep = np.exp(-np.maximum(k, 0.0) * dt)
    dM = M * (1.0 - keep)
    N, M = N * keep, M * keep
    dH = None if H is None else H * (1.0 - keep)
    H = None if H is None else H * keep
    if opt.explosion_route == "gas":
        return (N, M, float(dM.sum())) if H is None else (N, M, float(dM.sum()), H, float(dH.sum()))
    gas, gas_h = 0.0, 0.0
    for i in np.flatnonzero(dM > 0.0):
        # fragments a^-alpha from the grid's lowest edge up to half the parent's mass (labelled)
        dn, dm, below, _ = deposit_power_law(g, float(dM[i]), float(edges[0]), 2.0 ** (-1.0 / 3.0) * a_star[i],
                                             opt.alpha_frag)
        N, M = N + dn, M + dm
        gas += below
        if H is not None:
            x = dH[i] / dM[i]
            H = H + x * dm
            gas_h += x * below
    return (N, M, gas) if H is None else (N, M, gas, H, gas_h)


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
    if any(c.M_H is not None for c in compositions):
        gas.setdefault("H", 0.0)
    before = _element_totals(compositions, gas)
    shares, notes, duu = [], [], 0.0
    for k, c in enumerate(compositions):
        active = _active(c, opt.overflow_gate)
        g, N, M = _subgrid(c, active)
        M0 = M.copy()
        over_N, over_M = 0.0, 0.0                       # grown past g's top this substep
        Hs, over_H = None, 0.0
        if c.M_H is not None:
            Hs = np.asarray(c.M_H, float)[:g.n_bins].copy()
        if opt.processes & {"shattering", "coagulation"}:
            st = None if pre is None else pre.states[k]
            mol, S = (None, None) if pre is None else (_at(pre.molecular, k), _at(pre.release, k))
            r = collide(g, N, M, _collisions(c, g, inputs, opt, st, mol, S), dt, errmax=opt.errmax, method=c.method,
                        H=Hs)
            N, M = r["N"], r["M"]
            over_N, over_M = over_N + r["overflow_N"], over_M + r["overflow_M"]
            gas[c.key] += r["sink_M"] * c.per_gram
            if Hs is not None:
                Hs, over_H = r["H"], over_H + r["overflow_H"]
                gas["H"] += r["sink_H"] / M_H_ATOM
        mb = c.molecular_bins
        if "explosion" in opt.processes and pre is not None and pre.explosion[k] is not None:
            kb = np.array(pre.explosion[k], float)
            kb[:mb] = 0.0
            if Hs is None:
                N, M, to_gas = _explode(g, N, M, kb, dt, opt, pre.states[k].a)
            else:
                N, M, to_gas, Hs, h_gas = _explode(g, N, M, kb, dt, opt, pre.states[k].a, Hs)
                gas["H"] += h_gas / M_H_ATOM
            gas[c.key] += to_gas * c.per_gram
        erode_gr = "sputtering" in opt.processes and c.sputtering is not None
        erode_mol = "carbon_loss" in opt.processes and c.molecular_tables is not None and mb > 0
        if erode_gr or erode_mol:
            adot, knots = _erosion_rate(c, g, inputs, erode_gr, erode_mol)
            if Hs is None:
                N, M, dM = sputter(g, N, M, adot, dt, method=c.method, knots=knots)
            else:
                N, M, dM, Hs, h_gas = sputter(g, N, M, adot, dt, method=c.method, knots=knots, H=Hs)
                gas["H"] += h_gas / M_H_ATOM
            gas[c.key] += dM * c.per_gram
        if "accretion" in opt.processes and c.sublimation is not None:
            pop = _grain_population(c, g, N, M, inputs, None if pre is None else pre.states[k],
                                    None if pre is None else _at(pre.t_dust, k))
            if Hs is not None:
                pop.H = Hs[mb:].copy()
            gk = {c.key: gas[c.key]}
            exchange_step([pop], gk, inputs.T, dt)
            N[mb:], M[mb:] = pop.N, pop.M
            over_N, over_M = over_N + pop.overflow_N, over_M + pop.overflow_M
            gas[c.key] = gk[c.key]
            if Hs is not None:
                Hs[mb:], over_H = pop.H, over_H + pop.overflow_H
                gas["H"] += pop.h_to_gas / M_H_ATOM
        uvg = None if pre is None else _at(pre.uv_gap, k)
        if Hs is not None and "hydrogen" in opt.processes and c.hydrogen is not None and uvg is not None:
            Hs, released = _hydrogen_step(c, Hs, M, inputs, uvg, dt)
            gas["H"] += released / M_H_ATOM
        with np.errstate(divide="ignore", invalid="ignore"):
            ch = np.where(M0 > 0, np.abs(M - M0) / M0, 0.0)
        duu = max(duu, float(ch.max()) if ch.size else 0.0)
        if c.grid.overflow and not active:              # the passive overflow bin takes what grew past g
            c.N[:-1], c.M[:-1] = N, M
            c.N[-1] += over_N
            c.M[-1] += over_M
            if Hs is not None:
                c.M_H = np.asarray(c.M_H, float).copy()
                c.M_H[:-1] = Hs
                c.M_H[-1] += over_H
        else:
            c.N, c.M = N, M
            c.overflow_M += over_M
            if Hs is not None:
                c.M_H, c.overflow_H = Hs, c.overflow_H + over_H
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
        if c.M_H is not None:
            tot["H"] = tot.get("H", 0.0) + (float(np.sum(c.M_H)) + c.overflow_H) / M_H_ATOM
    return tot
