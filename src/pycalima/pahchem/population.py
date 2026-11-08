"""Which PAH population explains a set of observed bands, at stated conditions.

A library of molecules is given. Each molecule's charge balance and
hydrogenation follow from the local conditions through pahchem's physics. Only
the molecules' abundances are fitted. In outline:

- **Library** (``library_m20``). The Maragkoudakis+20 sample of PAHdb v3.20:
  133 molecules with neutral and cation, 38 with the anion too. It runs from
  N_C 22 to 216 and includes the Ricca+18 straight-edge species. A molecule's
  IP and EA are the DFT total-energy differences of its charge states
  (TOTAL_E, adiabatic; grade C), where both states share one functional and
  basis (mostly B3LYP/4-31G, PAHdb's default, assumed where COMMENTS names no
  basis). Pairs computed at different levels take the IP
  of a fit a + b / sqrt(N_C) to the consistent pairs (``ip_source = "fit"``)
  and drop the anion. Its edge-H counts (solo, duo, trio, quartet) are PAHdb's.
- **Charge** (``charge_fractions``). A chain over the molecule's charge states
  (``Molecule.ionisation_ev`` holds the threshold of each step z -> z+1). The
  ingredients are the baseline's:

  - Li & Draine sigma;
  - Jochims+96 yields at the DFT threshold (photodetachment at the EA), and
    Wenzel+20 for the cation where it applies (N_C >= 32, IP < 11.3 eV; it was
    measured on cation -> dication only, so the dication takes Jochims');
  - BT94 electron capture, with sticking 1 for cations and the statistical s_e
    of ``attachment`` for neutrals;
  - associative detachment of anions with H only with ``Params.associative_detachment = "all"``
    (``anion_associative_detachment``: measured on dehydrogenated anions, and the chain holds N_H0).

  The chain runs to the cold Coulomb limit floor(0.2 N_C) (``extend_ladder``,
  ``Params.charge_ladder``). The states above the highest computed charge are
  stand-ins with its modes, and the states a field cannot reach get no basis
  (``Population._live``). Ionisation out of the top is Coulomb explosion
  (``chain_overflow``). With ``charge_ladder = "library"`` (before 2026-11-02)
  the chain stops at the library's highest charge (+1 for PAHdb v3.20, which
  has no dications): that state holds the higher charges, emitting as itself,
  and absorbs without ionising below Y_UNITY_EV.
- **Hydrogenation** (``intact_fractions``). A dehydrogenation chain per
  molecule, N_H = N_H0 ... 0, with H loss (one step), H2 loss (two steps) and
  H addition, solved in steady state at each environment; f(N_H0) is the
  intact fraction. The baseline's rules apply:

  - H loss uses West+18 even-H (E0, dS) with degeneracy N_H for even N_H, and
    the odd-H radical values with degeneracy 1;
  - H2 loss (degeneracy 2) applies for even N_H above the solo count;
  - H addition is 1.4e-10 n_H for cations and 0.07 of the geometric rate for
    neutrals.

  The rates are RRKM on the molecule's own modes for N_H0, and on
  ``PAHSpecies.stripped`` modes for N_H0-1 ... N_H0-3, each with its own
  multiphoton cascade (tabulated on ``K_H_G0_GRID``). Deeper even (odd)
  states take the N_H0-2 (N_H0-3) rates, the even-H degeneracy scaled by N_H.
  Rates are averaged over the intact molecule's charge fractions (charge and
  hydrogenation independent, no super-hydrogenated states). The
  single first-step ratio k_add / (k_add + k_H) of earlier versions
  (``intact_fractions(method="single")``) misses the radical runaway near
  threshold (review B: 0.87 against 0.00 for C54H18 at the Atomic
  most-ionising corner).
- **Joint hydrogenation** (``Params.hydrogen_chain = "joint"``, opt-in,
  2026-11-08; ``Population._joint_distributions``). One (Z, N_H) network per
  molecule, N_H = 0 ... N_H0 + ``superh_levels``, solved for its steady state
  (``network.gth_banded``), as Montillaud+13, Boschman+15 and Andrews+16 do;
  no published model averages over charge. It adds the super-hydrogenated
  states with BaselineModel's (Andrews+16) chemistry, the IPs of the
  dehydrogenated states raised by ``ip_shift_per_h`` per missing H
  (Malloci+08; the EA and the super-H thresholds unshifted), associative
  detachment of dehydrogenated anions, and Eley-Rideal H2. Valid for PDR and
  diffuse gas. In UV-shielded gas H addition outruns the photo-loss of the
  extra H and the molecules climb to the ladder's top ("at_cap" of
  ``hydrogenation_summary``), whose chemistry past +4 the literature
  contradicts (alternating barriers, closed-shell cation barriers, abstraction
  per CH2 site, skeleton loss from ~+6): not modelled yet.
- **Emission** (``Population.band_fluxes``). The exact microcanonical cascade
  of each charge state for one absorbed photon (``emission.cascade_photons``),
  folded with that state's heating spectrum. With ``dehydrogenated_emission``
  (the default since 2026-10-25) N_H0-1 ... N_H0-3 emit too, on their stripped
  modes, in the chain's proportions (``emission_weights``); the share below
  N_H0-3 is not modelled (``hydrogenation_fractions``), as the stripped modes
  of a bare skeleton are 5-12x too weak against PAHdb's (``PAHSpecies.stripped``).
  The chain is nearly all-or-nothing: a molecule is mostly intact or mostly
  past N_H0-3, so the unmodelled share, not N_H0-1 ... N_H0-3, decides where
  dehydrogenation matters. Single-photon heating holds for
  these sizes at G0 <~ 1e5. The bands use the PAHdb Lorentzian (FWHM 15 cm^-1,
  redshift 15 cm^-1), integrated analytically over the caller's wavelength
  windows, which should be the windows of the observations.
- **Cost.** Heating is single-photon and every field is a fixed shape times G0
  (a Draine field with old-star light, which does not scale with G0, is one
  shape per G0: ``field_shape``), so per charge state and field shape a
  ``StateBasis`` at G0 = 1 holds the IR
  power per mode, the photo-ionisation (or detachment) rate and the absorbed
  power, each G0 times its basis value. H loss is not linear in G0: for small
  PAHs two photons within one cooling time dominate it. So the basis
  tabulates k_H(G0) from the full multiphoton steady state on
  ``K_H_G0_GRID`` and interpolates in log-log.
- **Recombination heating** (``Params.recombination_heating``). In steady state
  each ionisation (or detachment) out of state z is balanced by a recombination
  (or attachment) into z, which leaves the threshold energy (IP, or EA) in
  vibrations. So state z is also heated at rate k_up(z) with that energy, and
  its cascade adds k_up(z) x (emission after an energy deposit IP_z) to the IR
  power. This too is linear in G0. Not in the bases: the electron's kinetic
  energy (~kT); the energy the ion keeps after photoionisation (E - IP - KE)
  and the H loss the recombination energy drives, which ``Population._deposit_flows``
  adds per event (``Params.ionisation_heating``, ``recombination_dissociation``;
  the k_H table is photon-driven only). Bases are built once (in parallel, ``Population.prepare``) and kept
  on disk (``cache_dir``), keyed by a hash of their inputs. An environment
  then costs a few vectorised array operations over the whole library.
- **Fit** (``fit_abundances``). Non-negative least squares on the band fluxes
  relative to a reference band, each weighted by its relative uncertainty.
  With fewer bands than molecules the solution is not unique: NNLS returns at
  most n_bands molecules. ``bootstrap`` refits under perturbed observations to
  show which conclusions survive. Size and edge statistics
  (``population_statistics``) are the robust outputs, not the individual
  species.

"""

from __future__ import annotations

import functools
import hashlib
import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from pycalima.pahchem import chemistry, dos, emission, energy_me, fields, impact, photo, rates
from pycalima.pahchem.andrews16 import CORONENE_GEOMETRIC_A2
from pycalima.pahchem.baseline import E_PH as E_PH_FUV
from pycalima.pahchem.baseline import CH_STRETCH_CM, LAB_STRETCH_SCALE, Params, scaled_einstein_a  # noqa: F401

try:  # optional accelerator (the ``accel`` extra); the NumPy path gives the same numbers
    from numba import njit
    HAVE_NUMBA = True
except ModuleNotFoundError:  # pragma: no cover - exercised only without numba
    HAVE_NUMBA = False

HARTREE_EV = 27.211386
_FORMULA_CH = re.compile(r"C\d+H\d+[+-]*")
EDGES = ("solo", "duo", "trio", "quartet")

#: the pools' size when none is given: $CALIMA_WORKERS, else the cores up to this. Each worker holds one cascade at a
#: time, 1.0-3.4 GB for a C24-C216 deposit response (perf audit of 2026-11-05), so a laptop's memory sets the bound;
#: a larger machine raises it through the variable.
POOL_WORKERS_MAX = 8


def default_workers(workers=None):
    """``workers`` if given, else $CALIMA_WORKERS, else min(POOL_WORKERS_MAX, the cores)."""
    if workers is not None:
        return int(workers)
    env = os.environ.get("CALIMA_WORKERS")
    return int(env) if env else min(POOL_WORKERS_MAX, os.cpu_count() or 1)


#: the work [s] done in this process before the rest of a batch goes to a pool (``_run_jobs``): a pool's start and its
#: workers' empty memos cost ~1-3 s, more than a few cheap jobs here (16 responses in 1.5 s after an X-ray cell)
POOL_AFTER_S = 2.0


def _run_jobs(build, jobs, workers, keep):
    """keep(job, build(job)) for each of ``jobs`` in order: in this process while the work takes under POOL_AFTER_S,
    the rest (if more than 4) in a pool of ``workers``."""
    import time

    t0, i = time.perf_counter(), 0
    while i < len(jobs) and (workers <= 1 or len(jobs) - i <= 4 or time.perf_counter() - t0 < POOL_AFTER_S):
        keep(jobs[i], build(jobs[i]))
        i += 1
    if i < len(jobs):
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers) as ex:
            for job, r in zip(jobs[i:], ex.map(build, jobs[i:])):
                keep(job, r)

#: above this every state ionises (or detaches) on absorption: the lab yields reach ~1 by 15-20 eV, of neutrals
#: (Jochims+96) and of cations (Zhen+16 0.8-0.9 at 18 eV; Wenzel+20 0.87-0.98 at 20 eV). Below it the lab laws hold.
#: Not WDB06: their eq. 15 (a <= 6 A) restores WD01's y0 only and keeps the Coulomb-escape factor y2, which the cation
#: yields exceed ~2x. A state whose threshold lies above it (highly charged) ionises from its threshold on, not below
#: (``_yield``); no yield above +1 -> +2 has been measured.
Y_UNITY_EV = 20.0
#: the RAMSES-RTZ group edges above 13.6 eV, the PARTITION_NODES energies and the C K edge [eV]
EUV_NODES_EV = (15.2, 18.0, 20.0, 21.9, 24.59, 54.42, 284.2, 291.0)


def _euv_grid(e_max_ev=500.0, n=161):
    """Log-spaced from 13.6 eV with EUV_NODES_EV, plus a node 1e-6 eV above 13.6 eV: fields cut at the Lyman limit
    then fall to zero at once, and no trapezoid or interpolation spreads their edge over the first EUV step."""
    g = np.unique(np.concatenate([np.geomspace(E_PH_FUV[-1], e_max_ev, n), EUV_NODES_EV, [E_PH_FUV[-1] + 1e-6]]))
    return g[g > E_PH_FUV[-1]]


_D_STEP = (E_PH_FUV[-1] - E_PH_FUV[0]) / (E_PH_FUV.size - 1)
#: lowest photon energy of the population [eV]: the RTZ 1-5.6 eV group's edge; below 5 eV photons detach anions
#: (EA ~0.5-2.5 eV) and heat the cations, which absorb in the visible
E_PH_MIN_EV = 1.0
#: photon-energy grid of the population [eV]: 1-5 eV in the baseline's 0.025 eV steps with a node 1e-6 eV below
#: 5 eV (fields cut at 5 eV then fall to zero at once), the baseline's 5-13.6 eV grid, then log-spaced to 500 eV
E_PH_VIS = np.concatenate([E_PH_FUV[0] + _D_STEP * np.arange(-round(float(E_PH_FUV[0] - E_PH_MIN_EV) / _D_STEP), 0),
                           [E_PH_FUV[0] - 1e-6]])
E_PH = np.concatenate([E_PH_VIS, E_PH_FUV, _euv_grid()])
#: energy-deposit grid of the cascades [eV]. Non-ionising absorptions deposit E (always < Y_UNITY_EV); ionisations
#: deposit the ion's internal energy (``photo.photoelectron_energy``) if ``Params.ionising_photons_heat`` (in the absorbing
#: state; ``Params.ionisation_heating`` puts it in the product, ``ionisation_deposits``). It holds
#: every 1-13.6 eV photon node (so cut fields integrate exactly as before), steps of the same 0.025 eV from 0.1 to
#: 21 eV, and nodes 1e-6 eV outside 5 and 13.6 eV for sharp edges; the lowest 0.1 eV (a fraction of a cascade
#: bin) is left out.
E_DEP = np.unique(np.concatenate([E_PH_FUV[0] + _D_STEP * np.arange(-int(round((E_PH_FUV[0] - 0.1) / _D_STEP)),
                                                                    int(round((21.0 - E_PH_FUV[0]) / _D_STEP)) + 1),
                                  E_PH_FUV, [E_PH_FUV[0] - 1e-6, E_PH_FUV[-1] + 1e-6]]))


@dataclass
class Molecule:
    """One PAH skeleton and its charge states in the library."""

    name: str                      # formula plus PAHdb neutral UID, e.g. "C54H18 (37)"
    nc: int
    nh: int
    edges: dict                    # {"solo": n_H, ...}, edge H atoms by class
    species: dict                  # {Z: PAHSpecies}
    uids: dict                     # {Z: PAHdb UID}
    ip_ev: float                   # neutral -> cation
    ea_ev: float | None = None     # anion -> neutral, if the anion is in the library
    ip_source: str = "dft"         # or "fit" (module docstring)
    ionisation_ev: dict | None = None   # {z: threshold of z -> z+1 [eV]}; default from ip_ev, ea_ev
    source: str = "pahdb-v3.20"
    stand_in_from: int | None = None    # the highest computed charge, where ``extend_ladder`` added stand-ins above it
    bare: dict | None = None            # {Z: PAHSpecies} of the bare skeleton (N_H = 0) where PAHdb computed it (``library_v400``)
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self):
        if self.ionisation_ev is None:
            self.ionisation_ev = {0: self.ip_ev} | ({-1: self.ea_ev} if self.ea_ev is not None else {})
        zs = self.charges
        if zs and zs != tuple(range(zs[0], zs[-1] + 1)):
            raise ValueError(f"{self.name}: charge states {zs} are not contiguous")

    @property
    def charges(self):
        return tuple(sorted(self.species))

    @property
    def stand_ins(self):
        """The charges whose modes are a stand-in (the highest computed charge's; ``extend_ladder``)."""
        return () if self.stand_in_from is None else tuple(z for z in self.charges if z > self.stand_in_from)


def library_m20(path=None, min_nc=0, all_anions=False):
    """[Molecule] of the Maragkoudakis+20 sample with N_C >= ``min_nc``.

    Charge states are matched by bond graph (``ames_suite.read_ascii_graphs``),
    not by formula and edge-H counts, which isomers share (review A).

    - The cation is the sample cation with the neutral's bond graph. This
      corrects one M20 row pairing, C66H18 (714).
    - The anion is the anion with that graph: from Table H3, or with
      ``all_anions`` from the whole of PAHdb v3.20, Table H3's first.

    ``path`` is the PAHdb v3.20 ASCII file (default ``ames_suite.ascii_path()``).
    """
    from pycalima.pahchem import ames_suite, maragkoudakis2020

    path = ames_suite.ascii_path() if path is None else path
    sample = maragkoudakis2020.sample()
    if all_anions:
        with open(path, encoding="utf-8", errors="replace") as fh:
            every = [int(line.split()[1]) for line in fh if line.startswith("UID:")]
        full_meta = ames_suite.read_metadata(path, every)
        pool = [u for u, m in full_meta.items() if m.get("CHARGE") == -1 and _FORMULA_CH.fullmatch(m.get("FORMULA", ""))]
        meta = {u: full_meta[u] for u in set(sample.uid) | set(pool)}
    else:
        full_meta = None
        meta = ames_suite.read_metadata(path, sample.uid)
        pool = list(sample[sample.charge == -1].uid)
    sp = ames_suite.read_ascii(path, sample.uid)
    key = lambda u: (meta[u]["FORMULA"].rstrip("+-"),) + tuple(meta[u][f"N_{e.upper()}"] for e in EDGES)
    pairs = sample[sample.charge >= 0].uid.to_numpy().reshape(-1, 2).copy()
    for u0, u1 in pairs:
        if meta[u0]["CHARGE"] != 0 or meta[u1]["CHARGE"] != 1 or key(u0) != key(u1):
            raise ValueError(f"PAHdb UIDs {u0}, {u1} are not a neutral-cation pair")
    graph = ames_suite.read_ascii_graphs(path, set(sample.uid) | set(pool))
    cations = {}
    for u in sample[sample.charge == 1].uid:
        cations.setdefault(graph[u], []).append(u)
    fixed = []
    for i, (u0, u1) in enumerate(pairs):
        match = cations.get(graph[u0], [])
        if u1 not in match and len(match) == 1:
            pairs[i, 1] = match[0]
            fixed.append((int(u0), int(u1), int(match[0])))
    unmatched = [i for i, (u0, u1) in enumerate(pairs) if graph[u1] != graph[u0]]
    if unmatched:                                    # the partner is not in the sample: search all v3.20 cations
        if full_meta is None:
            with open(path, encoding="utf-8", errors="replace") as fh:
                every = [int(line.split()[1]) for line in fh if line.startswith("UID:")]
            full_meta = ames_suite.read_metadata(path, every)
        allmeta = full_meta
        extra = [u for u, m in allmeta.items() if m.get("CHARGE") == 1 and _FORMULA_CH.fullmatch(m.get("FORMULA", ""))]
        g_extra = ames_suite.read_ascii_graphs(path, extra)
        for i in unmatched:
            u0, u1 = pairs[i]
            match = [u for u in extra if g_extra[u] == graph[u0]]
            if len(match) == 1:
                pairs[i, 1] = match[0]
                graph[match[0]] = g_extra[match[0]]
                meta[match[0]] = allmeta[match[0]]
                fixed.append((int(u0), int(u1), int(match[0])))
    anions_by_graph = {}
    for u in pool:
        anions_by_graph.setdefault(graph[u], []).append(u)
    h3 = set(sample[sample.charge == -1].uid)
    pick = lambda us: us if len(us) == 1 else [u for u in us if u in h3]       # Table H3 first among isomers
    anion = {}
    for u0, _ in pairs:
        us = pick(anions_by_graph.get(graph[u0], []))
        if len(us) == 1:
            anion[u0] = us[0]
    sp |= ames_suite.read_ascii(path, [u for u in set(anion.values()) | set(pairs[:, 1]) if u not in sp])
    level = lambda u: (meta[u]["METHOD"].lstrip("RU").upper(), meta[u]["BASIS"].replace("?", "4-31G"))
    same = lambda a, b: level(a) == level(b)
    de = lambda a, b: (meta[b]["TOTAL_E"] - meta[a]["TOTAL_E"]) * HARTREE_EV
    good = np.array([[sp[u0].nc, de(u0, u1)] for u0, u1 in pairs if same(u0, u1)])
    coef = np.polyfit(good[:, 0] ** -0.5, good[:, 1], 1)
    out = []
    for u0, u1 in sorted(pairs.tolist(), key=lambda p: (sp[p[0]].nc, sp[p[0]].nh, p[0])):
        if sp[u0].nc < min_nc:
            continue
        ok = same(u0, u1)
        uids = {0: u0, 1: u1} | ({-1: anion[u0]} if ok and u0 in anion and same(anion[u0], u0) else {})
        out.append(Molecule(f"{key(u0)[0]} ({u0})", sp[u0].nc, sp[u0].nh, dict(zip(EDGES, key(u0)[1:])),
                            {z: sp[u] for z, u in uids.items()}, uids,
                            de(u0, u1) if ok else float(np.polyval(coef, sp[u0].nc ** -0.5)),
                            de(uids[-1], u0) if -1 in uids else None, "dft" if ok else "fit"))
    return out


def library_v400(records=None, min_nc=21, max_nc=384, include_superh=False, cache_dir=None):
    """[Molecule] of PAHdb v4.00: every pure C/H skeleton with a neutral and a cation.

    - **Grouping.** Charge states are grouped by ``ames_suite.bond_graph_hash``
      (one bond graph is one molecule). Where PAHdb holds two entries of one
      graph and charge, the lower UID is kept.
    - **Charges.** The contiguous ladder around Z = 0, from -1 up to +3.
    - **Thresholds.** Step z -> z+1 is E(z+1) - E(z) from TOTAL_E where both
      states share a functional and basis (6-31G* for nearly all). Otherwise a
      fit a + b / sqrt(N_C) of that step over the consistent molecules is used,
      and the molecule records it in ``ip_source`` ("fit" if any step needed it).
    - **Selection.** min_nc <= N_C <= max_nc, H-bearing; species with H off
      the aromatic edges (super-hydrogenated, methylated, side chains) only
      with ``include_superh`` (they are separate molecules: their bond graph
      differs). Such H are those beyond PAHdb's solo...quintet counts, or
      tagged CH2/CH3: the tags miss the extra H of seven C96H25 isomers
      (UIDs 684-690). ``edges`` then also holds "aliphatic", the number of
      those H (vinyl H included), and "ch3", PAHdb's count of H in CH3 groups.
    - **Dropped.** Molecules whose thresholds do not rise with charge (three in
      v4.00: C127H31 and C277H45 with IP2 < IP1, C212H44 with EA > IP), and the
      1522 skeletons with a neutral and a dication but no cation, which the
      charge chain cannot reach.
    - **Bare skeletons** (``Molecule.bare``): PAHdb's N_H = 0 entries with the
      molecule's carbon framework (``carbon_graph``), by charge: 204 of v4.00's
      3107 H-bearing neutrals have one (N_C 14-34, 54, 66, 96).

    ``records`` defaults to ``ames_suite.xml_index`` of the v4.00 dataset
    (cached in ``cache_dir``).
    """
    from pycalima.pahchem import ames_suite

    if records is None:
        records = ames_suite.xml_index(ames_suite.xml_path("pahdb-theoretical-v4-00"), cache_dir)
    # the bare skeletons PAHdb computed (N_H = 0, pure carbon), by their carbon framework: a molecule's shares its graph
    bare = {}
    for r in sorted(records, key=lambda r: r["uid"] if r["uid"] is not None else 10**9):
        if r["pure_ch"] and r["nh"] == 0 and r.get("carbon_graph") and r["uid"] is not None:
            bare.setdefault(r["carbon_graph"], {}).setdefault(r["charge"], r)
    off_edge = lambda r: r["nh"] - sum(r[f"n_{e}"] for e in (*EDGES, "quintet"))
    keep = [r for r in records if r["pure_ch"] and min_nc <= r["nc"] <= max_nc and r["nh"] > 0
            and (include_superh or (r["n_ch2"] == 0 and r["n_ch3"] == 0 and off_edge(r) == 0))
            and -1 <= r["charge"] <= 3]
    groups = {}
    for r in sorted(keep, key=lambda r: r["uid"]):
        groups.setdefault(r["graph"], {}).setdefault(r["charge"], r)
    level = lambda r: (str(r["method"]).lstrip("RU").upper(), r["basis"])
    # per-step fits over consistent pairs
    steps = {}
    for g in groups.values():
        for z in range(-1, 3):
            if z in g and z + 1 in g and level(g[z]) == level(g[z + 1]) and g[z]["basis"] != "?":
                steps.setdefault(z, []).append((g[z]["nc"], (g[z + 1]["total_e"] - g[z]["total_e"]) * HARTREE_EV))
    coef = {z: np.polyfit(np.array(v)[:, 0] ** -0.5, np.array(v)[:, 1], 1) for z, v in steps.items() if len(v) > 10}
    out = []
    for g in groups.values():
        if 0 not in g or 1 not in g:
            continue
        lo = 0
        while lo - 1 in g:
            lo -= 1
        hi = 1
        while hi + 1 in g:
            hi += 1
        zs = range(lo, hi + 1)
        neu = g[0]
        thresholds, fitted = {}, False
        for z in zs[:-1]:
            a, b = g[z], g[z + 1]
            if level(a) == level(b) and a["basis"] != "?":
                thresholds[z] = (b["total_e"] - a["total_e"]) * HARTREE_EV
            elif z in coef:
                thresholds[z] = float(np.polyval(coef[z], neu["nc"] ** -0.5))
                fitted = True
            else:
                break
        zs = [z for z in zs if z == min(zs) or z - 1 in thresholds]
        th = [thresholds[z] for z in zs[:-1]]
        if any(b <= a for a, b in zip(th, th[1:])):
            continue                                     # non-physical ladder (module docstring)
        species = {z: ames_suite.species_from_record(g[z], f"PAHdb v4.00 UID {g[z]['uid']}") for z in zs}
        formula = re.sub(r"[+-]\d*$", "", neu["formula"])
        edges = {e: neu[f"n_{e}"] for e in EDGES} | (
            {"aliphatic": off_edge(neu), "ch3": neu["n_ch3"]} if include_superh else {})
        sk = bare.get(neu.get("carbon_graph"))
        out.append(Molecule(f"{formula} ({neu['uid']})", neu["nc"], neu["nh"], edges, species, {z: g[z]["uid"] for z in zs},
                            thresholds[0], thresholds.get(-1), "fit" if fitted else "dft",
                            {z: thresholds[z] for z in zs[:-1]}, "pahdb-v4.00",
                            bare={z: ames_suite.species_from_record(r, f"PAHdb v4.00 UID {r['uid']} (bare)")
                                  for z, r in sk.items()} if sk else None))
    return sorted(out, key=lambda m: (m.nc, m.nh, m.uids[0]))


# -- charge ----------------------------------------------------------------------------------------
def _yield(mol, z, params=Params(), energy_ev=E_PH):
    """Ionisation (or detachment) yield out of charge z on ``energy_ev`` (default E_PH); zero at the top of the chain.

    As the baseline (``Params``): Jochims+96 (or ``yield_model="lepage"``, Le
    Page+01 eq. 14 as printed), and Wenzel+20 for the cation where it applies
    (``yield_cation``; measured on cation -> dication only, as the baseline). Anions detach with the same yield at the EA,
    or with unit yield above it (``detachment="unity"``).
    """
    E = np.asarray(energy_ev, dtype=float)
    if z == max(mol.charges):
        # the top state ionises out of the chain (``chain_overflow``) only above Y_UNITY_EV and its own threshold
        return _coulomb_escape(mol, z, params, E, (E >= max(Y_UNITY_EV, top_threshold(mol))).astype(float))
    threshold = mol.ionisation_ev[z]
    if z == -1 and params.detachment == "unity":
        return (E >= threshold).astype(float)
    if z == -1 and params.detachment in ("statistical", "valence"):
        return statistical_detachment(mol, params, energy_ev=E, valence=params.detachment == "valence")[0]
    if z == 1 and params.yield_cation == "wenzel" and mol.nc >= 32 and threshold < 11.3:   # cation -> dication only
        y = photo.yield_wenzel2020(E, threshold, mol.nc)
    elif params.yield_model == "lepage":
        # Le Page's law is fitted up to 13.6 eV (Verstraete+90's yields) and is a bump about d = 14.9 eV, falling
        # back to zero above it; above 13.6 eV it rises linearly to unity at Y_UNITY_EV instead (as the anion's
        # statistical detachment does)
        lyman = E_PH_FUV[-1]
        y_ly = float(photo.yield_lepage2001(lyman, threshold, form="lepage01"))
        y = np.where(E <= lyman, photo.yield_lepage2001(E, threshold, form="lepage01"),
                     y_ly + (1.0 - y_ly) * np.clip((E - lyman) / (Y_UNITY_EV - lyman), 0.0, 1.0))
    else:
        y = photo.yield_jochims1996(E, threshold)
    # unity above Y_UNITY_EV only above the threshold too (thresholds above 20 eV: highly charged states), none below it
    return _coulomb_escape(mol, z, params, E, np.where(E >= max(Y_UNITY_EV, threshold), 1.0, np.where(E < threshold, 0.0, y)))


#: ionisation energy of atomic H [eV]: proton emission costs the H loss plus IE(H) less the ion's IP(z - 1 -> z)
IE_H_EV = 13.598


def loss_extras(mol, z, n_h, params):
    """((name, E0 [eV], dS(1000 K) [J/K/mol], path degeneracy, reaction-coordinate mode [cm^-1]), ...): the loss channels of
    hydrogenation state N_H = ``n_h`` of charge z besides H and H2, as ``rates.rrkm`` takes them.

    - "C2H2" (``Params.c2h2_loss``): only for a skeleton with catacondensed end rings (PAHdb's quartet H), degeneracy 1/2
      per end ring (Ling & Lifshitz 1998 fitted 1 for anthracene's two), the C-C stretch as reaction coordinate (as
      ``impact``'s West+18 channel); none for pericondensed parents, which West+18 see lose no hydrocarbon fragment.
      Only for cations and above (pahchem audit m9, the user's choice of 2026-11-06): the parameters are anthracene+'s, and
      West+18's cations (Table 1, E0 2.6-5.2 eV) bracket them; no barrier is measured or computed for a neutral or an anion,
      whose hot relatives (benzene, 1-naphthol, azulene at 193 nm) lose H instead. Dications and above take the cation's,
      untested (their C2Hx+ losses are mostly charge separation).
    - "Hp" (``Params.fission``): proton emission from z >= +1 (to z - 1, one H less), over max(D, B): D = the member's H-loss
      E0 + IE_H_EV - IP(z - 1 -> z) (the ladder's; within 0.1-1 eV of Holm+11's Table V), B = ``impact.fission_barrier``;
      the H-loss transition state (dS, degeneracy N_H for even N_H, the radical's for odd), an assumption: Holm+11
      computed barriers, not transition-state frequencies. A barrier at or below 0 (above the barrier's charge limit)
      leaves the barrierless RRKM rate, E0 = 0."""
    out = []
    quartets = mol.edges.get("quartet", 0) / 4.0
    if params.c2h2_loss is not None and quartets > 0 and n_h >= 2 and z >= 1:
        out.append(("C2H2", float(params.c2h2_loss[0]), float(params.c2h2_loss[1]), 0.5 * quartets, 1600.0))
    if params.fission and z >= 1 and n_h >= 1:
        (e0_h, ds_h), deg = (params.h_even, n_h) if n_h % 2 == 0 else (params.h_odd, 1)
        d_hp = e0_h + IE_H_EV - float(state_threshold(mol, z - 1))
        out.append(("Hp", max(d_hp, float(impact.fission_barrier(mol.nc, z)), 0.0), float(ds_h), float(deg), 3050.0))
    return tuple(out)


def _extra_rates(freqs, E, counts, pick, beta, extra):
    """{name: k(E) on the loss grid} of ``loss_extras``' channels."""
    return {name: rates.rrkm(E, counts, rates.transition_state_for_entropy(freqs, coord, ds, n_scaled=5), e0 * dos.CM_PER_EV,
                             degeneracy=deg, beta=beta)[pick]
            for name, e0, ds, deg, coord in extra}


def _coulomb_escape(mol, z, params, E, y):
    """``y`` capped for z >= 2 by ``photo.yield_coulomb_window`` with D = (z+1) times the state's IP step
    (``Params.coulomb_escape``: "window" at u_c = ``photo.ESCAPE_DEPTH``, "y2" at u_c = inf); "none" leaves it."""
    if params.coulomb_escape not in ("none", "window", "y2"):
        raise ValueError(f"coulomb_escape must be 'none', 'window' or 'y2', not {params.coulomb_escape!r}")
    if params.coulomb_escape == "none" or z < 2:
        return y
    ip = float(state_threshold(mol, z))
    depth = (z + 1) * (ip - float(state_threshold(mol, z - 1)))
    u_c = photo.ESCAPE_DEPTH if params.coulomb_escape == "window" else np.inf
    return np.minimum(y, photo.yield_coulomb_window(E, ip, depth, u_c))


def double_threshold(mol, z):
    """IP(z) + IP(z+1) [eV]: the photon energy that takes charge z to z + 2 (z below the top of the chain)."""
    return float(state_threshold(mol, z)) + float(state_threshold(mol, z + 1))


def _lands(live, j):
    """lands[s, slot]: a step of +j from the slot reaches a live slot of the same molecule."""
    out = np.zeros_like(live)
    if j < live.shape[1]:                          # a step longer than the chain lands nowhere
        out[:, : live.shape[1] - j] = live[:, j:]
    return out


def _chain_fractions(ups, down, live):
    """f[s, slot]: the steady state of charge chains with up steps of +1, +2, ... (``ups[j-1]``[s, slot]: the rate of
    slot -> slot+j) and single down steps (down, slot -> slot-1), each molecule on its live slots (contiguous from its
    lowest). No net flux crosses any cut: f(i+1) down(i+1) = sum over j and 0 <= m < j of f(i-m) k_j(i-m), summed in logs;
    a step that would land on no live slot is a loss the chain does not follow (``Population.chain_overflow``)."""
    n, m = live.shape
    ks = [np.where(_lands(live, j), k, 0.0) for j, k in enumerate(ups, start=1)]
    with np.errstate(divide="ignore"):
        lk, l_down = [np.log(k) for k in ks], np.log(down)
    first = live.argmax(axis=1)
    logf = np.full((n, m), -np.inf)
    logf[np.arange(n), first] = 0.0
    for i in range(m - 1):
        terms = [logf[:, i - mm] + lk[j - 1][:, i - mm] for j in range(1, len(ks) + 1) for mm in range(j) if i - mm >= 0]
        new = np.logaddexp.reduce(np.stack(terms), axis=0) - l_down[:, i + 1]
        on = live[:, i + 1] & (first < i + 1)
        logf[on, i + 1] = new[on]
    f = np.exp(logf - logf.max(axis=1, keepdims=True))
    return f / f.sum(axis=1, keepdims=True)


# -- C 1s (K-shell) absorption (stage 3, 2026-11-03) ------------------------------------------------------------------
#: where the C 1s continuum starts in the library's cross section: Verner's atomic C 1s [eV]
K_EDGE_EV = 291.0
#: per K vacancy (docs/physics/pahchem-gas-coupling.md section 2.2c): the fluorescence yield of C (Krause 1979, JPCRD 8, 307,
#: Table 3: 2.8e-3), and the share of the Auger decays that eject two electrons, 9 % (range 7-12 %): the C 1s hydrocarbons of
#: Sci. Rep. 8, 16405 (2018), Table 1 (CH4 6.3, C2H2 8.2, C2H4 8.5, C2H6 8.8 %), at an aromatic carbon's valence-electron count
K_FLUORESCENCE, K_DOUBLE_AUGER = 2.8e-3, 0.09
#: the doubly charged ion's mean excitation after a C KVV Auger decay [eV]: ``impact.K_AUGER_EXCITATION_EV`` (17.3 eV,
#: Huo+23), its one owner, for the photoabsorptions here and the collisions there
K_AUGER_EXCITATION_EV = impact.K_AUGER_EXCITATION_EV
#: energies above K_AUGER_EXCITATION_EV [eV] on which a C 1s product's extra heat from its outgoing electrons is held: each
#: has its ``deposit_response``, and ``impact.escape_outcomes``' E* is split between neighbouring nodes keeping its mean
HEAT_NODES_EV = np.concatenate([np.arange(0.0, 20.0, 2.5), np.geomspace(20.0, 320.0, 13)])
#: kinetic energies [eV] on which the outgoing electrons' outcomes are tabulated (``escape_table``), interpolated in ln T.
#: Below the first an electron cannot collide in the model: a loss is at most T/2 (Moeller) and ``impact``'s lowest
#: oscillator is at 0.5 eV
ESCAPE_T_GRID = np.geomspace(1.0, 1.2e4, 33)
#: E* bin width of the outcomes [eV], and the charges a single outgoing electron's collisions may add that are followed
ESCAPE_DE, ESCAPE_DZ_MAX = 0.5, 12
#: the probability an outcome may leave the grids (E* above the top heat node, more charges than followed) before it is
#: an error: none is dropped silently
ESCAPE_BEYOND_TOL = 1e-4
#: bump when ``escape_table`` or the electron collisions of ``impact`` change, so stored tables are rebuilt
ESCAPE_VERSION = 1

#: ``escape_table`` and ``deposit_response`` results of this process, keyed by what they depend on, so that every
#: Population shares them (populations differing only in other parameters, the tests, the checks' on/off pairs)
_ESCAPE_TABLES, _RESPONSE_MEMO = {}, {}


def impact_ladder(mol, z, params, dz_max):
    """[(IP, IP2, yield, hole) of charge z + k, k = 0 ... min(dz_max, top - z)]: the states a passage or an outgoing
    electron ionises the molecule through (``impact.passage_spectrum``, ``impact.escape_outcomes``), so that its k-th
    ionisation costs IP(z + k) (pahchem audit M1). Past the top a passage destroys the molecule, and its collisions there
    take the top state's (``impact._poisson_chain``). ``hole``: None (``impact.deposition``'s partition), or for the
    anion the internal energy its photodetachment leaves at W (``ionisation_deposits``: W - EA less ``electron_energy``,
    0 under Params.detachment "jochims"; pahchem audit m7)."""
    top = max(mol.charges)
    out = []
    for k in range(min(dz_max, max(top - z, 0)) + 1):
        zk = z + k
        hole = None
        if zk < 0:
            thr = float(state_threshold(mol, zk))
            e1 = np.where(E_PH > thr, E_PH - thr - electron_energy(mol, zk, params), 0.0)   # 0 where nothing ionises
            hole = lambda w, e1=e1: np.interp(w, E_PH, e1)                     # noqa: E731
        out.append((_ladder_ip(mol, zk), _ladder_ip(mol, zk + 1),
                    (lambda w, zk=zk: _yield(mol, min(zk, top), params, energy_ev=w)), hole))
    return out


def escape_table(mol, z, params):
    """P[T, dz, E* bin] on ESCAPE_T_GRID: what an outgoing electron of each energy does to charge z on its way out,
    ``impact.escape_outcomes`` with ``impact.impact_collisions`` on the molecule (Bethe on its oscillator strengths, below
    ``impact.join_ev`` its photons' cross section ``Params.uv_cross_section``, pahchem audit m6; the polystyrene low-speed correction, Kim's binary-encounter scaling for slow electrons) and z's threshold and yield.
    A distant ionising collision leaves the photoionisation's mean hole (partition "measured", ``photo.photoelectron_energy``
    under ``Params.photoelectron_reference``), as the population's photons do (``Params.ionisation_heating``; "none", no E*,
    before 2026-11-04); close collisions keep the hole (``impact.deposition``). Cached per process under ``_escape_key``
    (``Population.prepare_escape`` keeps them on disk)."""
    key = _escape_key(mol, z, params)
    if key not in _ESCAPE_TABLES:
        ip, ip2 = _ladder_ip(mol, z), _ladder_ip(mol, z + 1)
        y = lambda w: _yield(mol, z, params, energy_ev=w)               # noqa: E731
        ladder = impact_ladder(mol, z, params, ESCAPE_DZ_MAX)
        rows = []
        for t in ESCAPE_T_GRID:
            chain = [(impact.impact_collisions(impact.ELECTRON, float(t), mol.nc, mol.nh, z + k, ip_ev=ipk,
                                               uv_model=params.uv_cross_section), ipk, ip2k, yk, hk)
                     for k, (ipk, ip2k, yk, hk) in enumerate(ladder)]
            P, beyond = impact.escape_outcomes(chain[0][0], mol.nc, ip, ip2, y, de=ESCAPE_DE,
                                               e_max=float(HEAT_NODES_EV[-1]), dz_max=ESCAPE_DZ_MAX, chain=chain,
                                               partition="measured", reference_ev=photoelectron_reference(mol, params))
            if beyond > ESCAPE_BEYOND_TOL:
                raise ValueError(f"{mol.name} {z:+d}: an outgoing {t:.0f} eV electron leaves the outcome grid with P {beyond:.1e}")
            rows.append(P)
        _ESCAPE_TABLES[key] = np.array(rows)
    return _ESCAPE_TABLES[key]


def _escape_key(mol, z, params):
    """Key of ``escape_table`` (memory and disk): everything it depends on, as ``_cr_key``, so that populations differing
    in parameters it does not read (fission, C2H2 loss, cosmic rays, ...) share it."""
    h = hashlib.sha1(f"escape{ESCAPE_VERSION}".encode())
    w = impact.w_grid()
    for v in (mol.nc, mol.nh, z, tuple(sorted(mol.ionisation_ev.items())), photoelectron_reference(mol, params),
              ESCAPE_T_GRID, ESCAPE_DE, ESCAPE_DZ_MAX, HEAT_NODES_EV, photo.PARTITION_NODES, photo.PHOTOELECTRON_PARTITION,
              photo.uv_model_key(params.uv_cross_section)):
        _feed(h, v)
    for ip, ip2, y, hole in impact_ladder(mol, z, params, ESCAPE_DZ_MAX):   # every state its collisions ionise through
        _feed(h, (ip, ip2))
        _feed(h, np.asarray(y(w), dtype=float))
        if hole is not None:
            _feed(h, np.asarray(hole(w), dtype=float))
    return h.hexdigest()[:20]


def _build_escape(job):
    return escape_table(*job)


def _log_t_weights(t_ev):
    """(collide, i, w): whether each energy is on ESCAPE_T_GRID (below its first node no collision is possible), and its
    neighbours i, i+1 and weight w of linear interpolation in ln T there. Energies above the grid are an error."""
    t = np.asarray(t_ev, dtype=float)
    if np.any(t > ESCAPE_T_GRID[-1]):
        raise ValueError(f"outgoing electron of {t.max():.0f} eV above ESCAPE_T_GRID")
    collide = t >= ESCAPE_T_GRID[0]
    lt = np.log(ESCAPE_T_GRID)
    x = np.log(np.where(collide, t, ESCAPE_T_GRID[0]))
    i = np.minimum(np.searchsorted(lt, x, side="right") - 1, lt.size - 2)   # x = the last node: its lower interval
    return collide, i, (x - lt[i]) / (lt[i + 1] - lt[i])


def _node_split(e_ev):
    """S[bin, node]: E* bin centres split between the neighbouring HEAT_NODES_EV keeping the mean (all inside the nodes)."""
    nodes = HEAT_NODES_EV
    if e_ev.min() < nodes[0] or e_ev.max() > nodes[-1]:
        raise ValueError("E* outside the heat nodes")
    j = np.minimum(np.searchsorted(nodes, e_ev, side="right") - 1, nodes.size - 2)
    w = (e_ev - nodes[j]) / (nodes[j + 1] - nodes[j])
    S = np.zeros((e_ev.size, nodes.size))
    S[np.arange(e_ev.size), j] = 1.0 - w
    S[np.arange(e_ev.size), j + 1] = w
    return S


def _convolve(a, b):
    """Linear 2-D convolution of non-negative (dz, E*) distributions on their full support, a.shape + b.shape - 1 along
    each axis (nothing is cut); FFT round-off below 0 is set to 0 after checking it is round-off (``impact.FFT_ROUND_OFF``)."""
    shape = (a.shape[0] + b.shape[0] - 1, a.shape[1] + b.shape[1] - 1)
    full = np.fft.irfft2(np.fft.rfft2(a, s=shape) * np.fft.rfft2(b, s=shape), s=shape)
    if full.min() < -impact.FFT_ROUND_OFF * max(full.max(), 1e-300):
        raise ValueError("convolution negative beyond round-off")
    return np.maximum(full, 0.0)


#: photon grid of the K-shell rates [eV]: the measured edge from its first point (282.1 eV), then to 10 keV
K_GRID = np.unique(np.concatenate([np.arange(282.0, 320.0 + 1e-9, 0.05), np.geomspace(320.0, 1e4, 400), [K_EDGE_EV]]))


def sigma_kshell(nc, params, energy_ev):
    """The C 1s part of the state's cross section [cm^2]: the library's (``uvlib.sigma_k_shell``, the measured edge when its
    anchors carry it) for the Octopus models, N_C times Verner's atomic C 1s for the others."""
    if str(params.uv_cross_section).startswith("octopus_library"):
        from pycalima.pahchem import uvlib
        return uvlib.sigma_k_shell(nc, energy_ev)
    from pycalima.models.dust_charge import wdb06_atomic
    return nc * wdb06_atomic.photoionisation_cross_section("C", "1s", np.asarray(energy_ev, dtype=float))


def _ladder_ip(mol, z):
    """IP(z) [eV] on the cold-limit ladder, continued past the top state with its last step."""
    top = max(mol.charges)
    if z <= top:
        return float(state_threshold(mol, z))
    step = float(state_threshold(mol, top)) - float(state_threshold(mol, top - 1))
    return float(state_threshold(mol, top)) + (z - top) * step


def _photoelectron(mol, z, params, energy_ev):
    """(pe, free, e_k, depth): the probability a C 1s photoelectron of charge z escapes at each energy, whether it leaves
    from the continuum (above the edge), the edge E_K(z) [eV] and the Coulomb window's depth D [eV] (``kshell_branching``)."""
    E = np.asarray(energy_ev, dtype=float)
    ip = _ladder_ip(mol, z)
    e_k = K_EDGE_EV + ip - float(mol.ionisation_ev[0])
    depth = max(z + 1, 0) * (_ladder_ip(mol, z + 1) - ip)
    free = E >= e_k
    in_window = ~free & (E >= e_k - depth) if depth > 0 else np.zeros_like(free)
    u = (e_k - E[in_window]) / depth if depth > 0 else np.zeros(0)
    if params.coulomb_escape == "none":
        bound = np.ones_like(u)
    elif params.coulomb_escape == "window":
        bound = 1.0 / (1.0 + (photo.ESCAPE_DEPTH / u) ** 1.5)          # u > 0 inside the window
    else:
        bound = np.zeros_like(u)
    pe = free.astype(float)
    pe[in_window] = bound
    return pe, free, e_k, depth


def kshell_branching(mol, z, params, energy_ev):
    """(P[0..3] of dz per C 1s absorption on ``energy_ev``, E_K(z) [eV], depth D [eV]).

    The 1s electron lands at E' = E - E_K(z), E_K(z) = K_EDGE_EV + IP(z) - IP(0) (the core level shifts with the charge as the
    valence does). Above 0 it escapes. In the Coulomb window, -D <= E' < 0 with D = (z+1) times the ion's IP step, it is bound
    and escapes as the valence electrons do there (``Params.coulomb_escape``: all with "none", 1 / (1 + (u_c/u)^1.5) with
    "window", none with "y2"). Below the window (the 1s -> pi* resonances) it stays. The hole then fluoresces (K_FLUORESCENCE)
    or Auger-decays, ejecting one electron or two (K_DOUBLE_AUGER). dz counts the escaping electrons, before the outgoing
    electrons' collisions (``kshell_rates``)."""
    pe, _, e_k, depth = _photoelectron(mol, z, params, energy_ev)
    w, f = K_FLUORESCENCE, K_DOUBLE_AUGER
    P = np.array([(1 - pe) * w, pe * w + (1 - pe) * (1 - w) * (1 - f), pe * (1 - w) * (1 - f) + (1 - pe) * (1 - w) * f,
                  pe * (1 - w) * f])
    return P, e_k, depth


def _escape_mix(tab, t_ev, weights):
    """sum_E weights(E) P(T(E)): the outgoing-electron outcome of ``escape_table`` mixed over energies ``t_ev`` with
    ``weights`` (no collision, P = delta(0, 0), for those below ESCAPE_T_GRID)."""
    collide, i, w = _log_t_weights(t_ev)
    wt = np.asarray(weights, dtype=float)
    node = np.zeros(ESCAPE_T_GRID.size)
    np.add.at(node, i[collide], (wt * (1.0 - w))[collide])
    np.add.at(node, i[collide] + 1, (wt * w)[collide])
    out = np.tensordot(node, tab, axes=1)
    out[0, 0] += float(wt[~collide].sum())
    return out


def kshell_rates(mol, z, params, field1):
    """C 1s absorption by charge z in ``field1`` (G0 = 1): (k[j - 1] [s^-1] for dz = j = 1 ... J on K_GRID, k_node[j - 1, i]
    the same with the product's extra heat on HEAT_NODES_EV[i], the over-count in the valence rate (the 1s absorption on E_PH
    that k_up counts as one ionisation at the valence yield), the electrons' power [eV s^-1], the power absorbed above E_PH's
    top [eV s^-1]).

    Per absorption (``kshell_branching``): the photoelectron, at E - E_K(z), escapes from the continuum above the edge,
    escapes slowly (~0 eV) from the Coulomb window as ``Params.coulomb_escape`` says, or stays below it; the hole fluoresces
    or Auger-decays with one electron at E_K - IP(z) - IP(z+1) - K_AUGER_EXCITATION_EV, or two sharing that less IP(z+2).
    Each outgoing electron collides on its way out (``escape_table`` of the product z+2): its extra charges and E* are
    convolved over the electrons, the photoelectron's mixed over the spectrum first (the Auger part does not depend on E).
    The product keeps K_AUGER_EXCITATION_EV + E*; the fluorescence branch (0.3 %) is given the same, an overestimate for it.
    The electrons lose what they leave: E* and the extra charges' ionisation energies (the product's IP)."""
    E = K_GRID
    tw = _trapezoid_weights(E)
    rate = sigma_kshell(mol.nc, params, E) * params.sigma_scale * field1.flux_at(E) * tw   # per K_GRID node [s^-1]
    pe, free, e_k, _ = _photoelectron(mol, z, params, E)
    ip = [_ladder_ip(mol, z + i) for i in range(3)]
    e_a1 = e_k - ip[0] - ip[1] - K_AUGER_EXCITATION_EV
    e_a2 = e_a1 - ip[2]
    if e_a2 < 0.0:                      # >= 150 eV over the cold-limit ladders; negative only for inconsistent thresholds
        raise ValueError(f"C 1s Auger energy {e_a2:.1f} eV < 0 for {mol.name} at {z:+d}: thresholds inconsistent")
    w, f = K_FLUORESCENCE, K_DOUBLE_AUGER
    zp = min(z + 2, max(mol.charges))
    tab = escape_table(mol, zp, params)
    n_dz, n_e = tab.shape[1], tab.shape[2]
    t_photo = np.where(free, E - e_k, 0.0)
    # the photo part [dz, E*]: free photoelectrons collide (dz +1 and their outcome); window ones escape (dz +1, ~0 eV);
    # the rest stay (dz 0)
    M = np.zeros((n_dz + 1, n_e))
    M[1:] += _escape_mix(tab, t_photo, rate * free)
    M[1, 0] += float(np.sum(rate * np.where(free, 0.0, pe)))
    M[0, 0] += float(np.sum(rate * (1.0 - pe)))
    # the Auger part, the same at every E: fluorescence (no electron), one electron, two
    d1 = _escape_mix(tab, np.array([e_a1]), np.array([1.0]))
    d2 = _escape_mix(tab, np.array([0.5 * e_a2]), np.array([1.0]))
    A = np.zeros((2 * n_dz + 1, n_e))
    A[0, 0] += w
    A[1:1 + n_dz] += (1 - w) * (1 - f) * d1
    A[2:] += (1 - w) * f * _convolve(d2, d2)[:, :n_e]
    full = _convolve(M, A)
    total = float(full.sum())
    R = full[:, :n_e]
    beyond = 1.0 - float(R.sum()) / total if total > 0.0 else 0.0
    if beyond > ESCAPE_BEYOND_TOL:
        raise ValueError(f"{mol.name} {z:+d}: C 1s products beyond {HEAT_NODES_EV[-1]:.0f} eV of E* with P {beyond:.1e}")
    e_star = np.arange(n_e) * ESCAPE_DE
    k_node = R[1:] @ _node_split(e_star)              # [dz - 1, node]; dz = 0 changes no charge and is not followed
    k = k_node.sum(axis=1)
    # what the electrons give the molecule per outgoing electron of energy T: E* plus the extra charges' IP
    x_t = np.einsum("tde,d->t", tab, np.arange(n_dz) * _ladder_ip(mol, zp)) + np.einsum("tde,e->t", tab, e_star)
    collide, i, wi = _log_t_weights(t_photo)
    x_photo = np.where(collide, (1.0 - wi) * x_t[i] + wi * x_t[i + 1], 0.0)

    def x_at(t):
        c, j, u = _log_t_weights(np.array([t]))
        return float(np.where(c, (1.0 - u) * x_t[j] + u * x_t[j + 1], 0.0)[0])
    k_tot = float(rate.sum())
    lost = float(np.sum(rate * free * x_photo)) + k_tot * (1 - w) * ((1 - f) * x_at(e_a1) + 2.0 * f * x_at(0.5 * e_a2))
    power = float(np.sum(rate * free * t_photo)) + k_tot * (1 - w) * ((1 - f) * e_a1 + f * e_a2) - lost
    hi = E > E_PH[-1]
    absorbed_hi = float(np.sum(np.where(hi, rate * E, 0.0)))
    s_ph = sigma_kshell(mol.nc, params, E_PH) * params.sigma_scale
    over = photo.ionisation_rate(field1, s_ph, _yield(mol, z, params), E_PH)
    return k, k_node, float(over), power, absorbed_hi


# -- cosmic rays (Params.cosmic_rays, Environment.zeta_cr; 2026-11-04) -----------------------------------------------
#: the charges a passage may add that are followed, and the E* grid of ``impact.passage_outcomes`` [eV]
CR_DZ_MAX, CR_DE_EV, CR_E_MAX_EV = 15, 1.0, 512.0
#: bump when ``cr_passages`` changes, so stored passages are rebuilt (4: an anion's collisions leave its photodetachment
#: hole, pahchem audit m7)
CR_VERSION = 4


def cr_passages(mol, z, params, protons="L", n_h2=None):
    """Cosmic-ray passages through charge z in the unscaled shape ``cosmic_rays.projectile_spectra(protons, n_h2)`` (n_h2
    ``cosmic_rays.STANDARD_COLUMN`` by default; the population blends L and H per zeta, ``cosmic_rays.blend_weights``):
    (k_node[dz, node] [s^-1] for dz = 0 ... CR_DZ_MAX with the product's E* on HEAT_NODES_EV, split keeping the mean;
    the rate of passages beyond them, E* above the top node or more than CR_DZ_MAX charges; the heat deposited inside
    [eV s^-1]). ``impact.passage_spectrum`` on the molecule's oscillator strengths (below ``impact.join_ev`` its photons'
    cross section ``Params.uv_cross_section``, pahchem audit m6), Poisson collisions per passage, each
    distant ionisation leaving the photons' mean hole (partition "measured"), with z's threshold and yield."""
    from pycalima.pahchem import cosmic_rays

    ip, ip2 = _ladder_ip(mol, z), _ladder_ip(mol, z + 1)
    y = lambda w: _yield(mol, z, params, energy_ev=w)                   # noqa: E731
    spectra = cosmic_rays.projectile_spectra(protons, cosmic_rays.STANDARD_COLUMN if n_h2 is None else n_h2)
    ps = impact.passage_spectrum(mol.nc, mol.nh, z, ip, ip2, spectra, y, de=CR_DE_EV,
                                 e_max=CR_E_MAX_EV, dz_max=CR_DZ_MAX, partition="measured",
                                 ladder=impact_ladder(mol, z, params, CR_DZ_MAX), uv_model=params.uv_cross_section,
                                 reference_ev=photoelectron_reference(mol, params))
    return _passage_nodes(ps)


def _passage_nodes(ps):
    """(k_node[dz, node], beyond, heat) of ``impact.passage_spectrum``'s {projectile: Passages}: the products' E* on
    HEAT_NODES_EV (split keeping the mean), the passages beyond them, the heat deposited inside."""
    ps = [v for v in ps.values() if v is not None]
    if not ps:
        return np.zeros((CR_DZ_MAX + 1, HEAT_NODES_EV.size)), 0.0, 0.0
    e = ps[0].e
    R = sum(v.P for v in ps) * ps[0].de                                  # [dz, k] passages [s^-1]
    inside = e <= HEAT_NODES_EV[-1]
    k_node = R[:, inside] @ _node_split(e[inside])
    beyond = float(sum(v.beyond for v in ps)) + float(R[:, ~inside].sum())
    return k_node, beyond, float((R[:, inside] * e[inside]).sum())


#: thermal electrons' kinetic energies [eV] from 1 eV (below which the model allows no collision: ESCAPE_T_GRID) to
#: HOT_T_SPAN kT, at HOT_N_T log-spaced points
HOT_T_SPAN, HOT_N_T = 40.0, 48


def hot_passages(mol, z, params, temperature_k):
    """Thermal electrons' passages through charge z per unit n_e [cm^-3] at ``temperature_k`` (Maxwellian,
    ``impact.maxwellian_flux``, Coulomb-focused on the charged PAH, ``impact.coulomb_focus``): as ``cr_passages``."""
    kt = 8.617333262e-5 * float(temperature_k)
    t = np.geomspace(1.0, max(HOT_T_SPAN * kt, 10.0), HOT_N_T)
    ip, ip2 = _ladder_ip(mol, z), _ladder_ip(mol, z + 1)
    y = lambda w: _yield(mol, z, params, energy_ev=w)                   # noqa: E731
    spectra = [(impact.ELECTRON, impact.maxwellian_flux(impact.ELECTRON, temperature_k, 1.0))]
    ps = impact.passage_spectrum(mol.nc, mol.nh, z, ip, ip2, spectra, y, t_grid={"e": t}, focus=True, de=CR_DE_EV,
                                 e_max=CR_E_MAX_EV, dz_max=CR_DZ_MAX, partition="measured",
                                 ladder=impact_ladder(mol, z, params, CR_DZ_MAX), uv_model=params.uv_cross_section,
                                 reference_ev=photoelectron_reference(mol, params))
    return _passage_nodes(ps)


def hot_ion_passages(mol, z, temperature_k, ion):
    """Thermal ions ``ion`` (a name of ``chemistry.IONS``, its charge the count of "+") through charge z per unit ion
    density [cm^-3] at ``temperature_k``: ``impact.thermal_ion_passages`` (heat at dz = 0: friction on the valence
    electrons and elastic collisions below threshold), as ``cr_passages``, and {"C", "H": knock-outs [s^-1]}."""
    el = ion.rstrip("+")
    ps, ko = impact.thermal_ion_passages(mol.nc, el, temperature_k, 1.0, z, len(ion) - len(el), nh=mol.nh,
                                         de=CR_DE_EV, e_max=CR_E_MAX_EV, dz_max=CR_DZ_MAX)
    return (*_passage_nodes({ion: ps}), ko)


#: bump when ``hot_passages``, ``hot_ion_passages`` or the collisions of ``impact`` they take change, so stored passages
#: are rebuilt
HOT_VERSION = 1
#: the hot passages of this process by ``_hot_key`` (populations differing in parameters they do not read share them)
_HOT_MEMO = {}


def _hot_key(mol, z, params, temperature_k, ion):
    """Key of the passages through charge z at ``temperature_k`` (memory and disk): of the thermal electrons
    (``hot_passages``) when ``ion`` is None, else of that ion (``hot_ion_passages``); everything they depend on, as
    ``_cr_key``."""
    h = hashlib.sha1(f"hot{HOT_VERSION}".encode())
    for v in (mol.nc, mol.nh, z, float(temperature_k), ion, CR_DZ_MAX, CR_DE_EV, CR_E_MAX_EV, HEAT_NODES_EV):
        _feed(h, v)
    if ion is not None:
        _feed(h, tuple(sorted(impact.KNOCKOUT_T0_EV.items())))
        return h.hexdigest()[:20]
    w = impact.w_grid()
    for v in (tuple(sorted(mol.ionisation_ev.items())), photoelectron_reference(mol, params), HOT_T_SPAN, HOT_N_T,
              photo.PARTITION_NODES, photo.PHOTOELECTRON_PARTITION, photo.uv_model_key(params.uv_cross_section)):
        _feed(h, v)
    for ip, ip2, y, hole in impact_ladder(mol, z, params, CR_DZ_MAX):   # every state the passages ionise through
        _feed(h, (ip, ip2))
        _feed(h, np.asarray(y(w), dtype=float))
        if hole is not None:
            _feed(h, np.asarray(hole(w), dtype=float))
    return h.hexdigest()[:20]


def _build_stop(job):
    return capture_stopping(*job)


def _build_hot(job):
    mol, z, params, temperature_k, ion = job
    return hot_passages(mol, z, params, temperature_k) if ion is None else hot_ion_passages(mol, z, temperature_k, ion)


def _cr_key(mol, z, params, protons):
    """Disk key of ``cr_passages`` of shape ``protons``: everything it depends on."""
    h = hashlib.sha1(f"cr{CR_VERSION}".encode())
    from pycalima.pahchem import cosmic_rays
    w = impact.w_grid()
    for v in (mol.nc, mol.nh, z, tuple(sorted(mol.ionisation_ev.items())), protons, cosmic_rays.STANDARD_COLUMN,
              photoelectron_reference(mol, params), CR_DZ_MAX, CR_DE_EV, CR_E_MAX_EV, HEAT_NODES_EV,
              impact.K_AUGER_EXCITATION_EV, photo.PARTITION_NODES, photo.PHOTOELECTRON_PARTITION,
              photo.uv_model_key(params.uv_cross_section)):
        _feed(h, v)
    for ip, ip2, y, hole in impact_ladder(mol, z, params, CR_DZ_MAX):   # every state the passages ionise through
        _feed(h, (ip, ip2))
        _feed(h, np.asarray(y(w), dtype=float))
        if hole is not None:
            _feed(h, np.asarray(hole(w), dtype=float))
    return h.hexdigest()[:20]


def _build_cr(job):
    mol, z, params, protons = job
    return cr_passages(mol, z, params, protons)


#: the two shapes the standard spectrum blends (``cosmic_rays.blend_weights``)
CR_SHAPES = ("L", "H")


#: (a, b) of IP(+1 -> +2) - IP(0 -> +1) = a + b / sqrt(N_C) [eV], over the 743 PAHdb v4.00 molecules of ``library_v400``
#: whose neutral, cation and dication have DFT thresholds (rms 0.24 eV; 3.9, 2.8, 2.2 eV at N_C 24, 54, 96): the step of
#: a molecule with no computed dication (``extend_ladder``)
LADDER_STEP_FIT = (0.4640, 16.9086)


def extend_ladder(mol):
    """``mol`` with its charges up to the cold Coulomb limit floor(0.2 N_C) (``impact.coulomb_limit``;
    ``Params.charge_ladder = "cold_limit"``); ``mol`` itself if it reaches it already.

    - **Thresholds** continue the last computed step, as for a charged conductor (each charge adds e^2/C), and as
      ``top_threshold`` assumed: they reproduce PAHdb's +2 -> +3 of C54H18 and C96H24 to 0.07 eV, and with the
      calibrated shifts of the PDRs4All fits give 13.4-14.1 eV for the isomers of C30H14, C32H14 and C36H18 against
      Wenzel+20's measured 13.9-14.1 eV. A molecule with no computed dication takes its +1 -> +2 step from
      LADDER_STEP_FIT.
    - **Stand-ins:** the states above the highest computed charge take its modes (``Molecule.stand_ins``):
      multiply charged spectra are cation-like (Bakes+01, Malloci+07, Banhatti+21) and no prescription for their
      change beyond +2 is published. Their UV cross section is +2's (``photo.sigma_uv``), as WD01 and Draine & Li 2007
      give every ion one.
    """
    from pycalima.pahchem import impact

    z_top, top = impact.coulomb_limit(mol.nc), max(mol.charges)
    if z_top <= top:
        return mol
    if top < 1:
        raise ValueError(f"{mol.name}: extend_ladder needs a computed cation")
    th = dict(mol.ionisation_ev)
    if top == 1:
        step = LADDER_STEP_FIT[0] + LADDER_STEP_FIT[1] / np.sqrt(mol.nc)
        th[1] = th[0] + step
    else:
        step = th[top - 1] - th[top - 2]
    for z in range(max(top, 2), z_top):
        th[z] = th[z - 1] + step
    species = mol.species | {z: mol.species[top] for z in range(top + 1, z_top + 1)}
    return replace(mol, species=species, ionisation_ev=th, stand_in_from=top, _cache={})


def top_threshold(mol):
    """Ionisation energy out of the top state [eV], extrapolated from the last two steps of the ladder (or
    the last step plus 3 eV): only for the electron energy of the EUV ionisations that leave the chain."""
    zs = sorted(mol.ionisation_ev)
    if len(zs) >= 2:
        return 2.0 * mol.ionisation_ev[zs[-1]] - mol.ionisation_ev[zs[-2]]
    return mol.ionisation_ev[zs[-1]] + 3.0


def state_threshold(mol, z):
    return top_threshold(mol) if z == max(mol.charges) else mol.ionisation_ev[z]


def electron_energy(mol, z, params=Params(), partition=photo.PHOTOELECTRON_PARTITION, energy_ev=E_PH):
    """<E_k> [eV] of the electron of each ionisation (detachment) out of charge z on ``energy_ev``: the share
    ``partition`` of the excess over ``state_threshold`` for Z >= 0, all of it for the anion below 13.6 eV
    (``photo.photoelectron_energy``), or the energies of ``statistical_detachment``."""
    if z == -1 and params.detachment in ("statistical", "valence"):
        return statistical_detachment(mol, params, energy_ev=energy_ev, valence=params.detachment == "valence")[1]
    return photo.photoelectron_energy(energy_ev, state_threshold(mol, z), 1.0 if z < 0 else partition,
                                      reference_ev=photoelectron_reference(mol, params))[0]


def photoelectron_reference(mol, params):
    """The ``reference_ev`` of ``photo.photoelectron_energy`` under ``Params.photoelectron_reference``: the molecule's neutral
    IP with "neutral", None (the photon energy) with "photon"."""
    if params.photoelectron_reference not in ("photon", "neutral"):
        raise ValueError(f"photoelectron_reference must be 'photon' or 'neutral', not {params.photoelectron_reference!r}")
    return float(mol.ionisation_ev[0]) if params.photoelectron_reference == "neutral" else None


def kshell_applies(params, shape):
    """True if C 1s absorption follows its own branching (``Params.kshell`` "auger") and the field of ``shape`` reaches the edge."""
    if params.kshell not in ("auger", "single"):
        raise ValueError(f"kshell must be 'auger' or 'single', not {params.kshell!r}")
    return params.kshell == "auger" and _unit_field(shape).energy_ev.max() >= K_GRID[0]


def absorption_sigma(nc, z, energy_ev, uv_cross_section="li_draine", sigma_scale=1.0, detach_ea=None):
    """The absorption cross section [cm^2] of charge z on ``energy_ev``: ``photo.sigma_uv`` times ``sigma_scale``, and for an
    anion whose EA ``detach_ea`` is given (``_detach_ea``) at least the WD01 photodetachment cross section (eq. 20) that
    ``statistical_detachment`` takes as its prompt share, which is part of the absorption. Before 2026-11-06 the share was
    clipped at 1 where it lay above (pahchem audit m8: C24H12- up to 2.9x at 1.5-2.4 eV with the Octopus library, 61x with
    Li & Draine; none from N_C 96)."""
    sigma = photo.sigma_uv(nc, z, energy_ev, uv_cross_section) * sigma_scale
    if detach_ea is None:
        return sigma
    from pycalima.models.dust_charge.shared_physics import photodetachment_cross_section_vec

    return np.maximum(sigma, photodetachment_cross_section_vec(np.asarray(energy_ev, dtype=float), detach_ea, -1))


def _detach_ea(mol, z, params):
    """The EA ``absorption_sigma`` raises the anion's absorption with: with ``Params.detachment`` "statistical" or
    "valence" (WD01's prompt photodetachment), else None."""
    return float(mol.ionisation_ev[-1]) if z == -1 and params.detachment in ("statistical", "valence") else None


def state_sigma(mol, z, params, energy_ev=E_PH):
    """``absorption_sigma`` of charge z of ``mol`` under ``params``."""
    return absorption_sigma(mol.nc, z, energy_ev, params.uv_cross_section, params.sigma_scale, _detach_ea(mol, z, params))


def valence_sigma(mol, z, params, shape):
    """The state's cross section on E_PH less its C 1s part when that has its own branching (``kshell_applies``)."""
    sigma = state_sigma(mol, z, params)
    if kshell_applies(params, shape):
        sigma = sigma - sigma_kshell(mol.nc, params, E_PH) * params.sigma_scale
    return sigma


def ionisation_deposits(mol, z, params, shape):
    """[(dz, k [s^-1], E_int [eV], w)] of the valence photoionisations (detachments) out of charge z under ``shape`` at G0 = 1
    (none out of the top state, whose ionisations leave the chain): their rate k and, on the photon nodes of E_PH where they
    happen, the internal energy E_int = E - threshold - <E_k> each leaves the product z + dz, with the share w of the events
    there (sum 1).

    - dz = 1: one electron with the <E_k> of ``electron_energy`` (``photo.photoelectron_energy``, the measured partition at
      PHOTOELECTRON_PARTITION; for the anion ``statistical_detachment``'s mean over its prompt, valence and delayed
      branches, whose deposits are not followed one by one).
    - dz = 2: the double ionisations of ``Params.double_ionisation``, the two electrons sharing the excess over
      ``double_threshold`` as one would.

    The same sigma (``valence_sigma``), yield and partition as the charge balance (``Population._jump_rates``) and the
    photoelectric heating (``Population._pe_stack``), so the absorbed energy closes: deposits, electrons and ionisation
    work. E_int is the mean at each photon energy; the spread of the photoelectron spectra about it is not followed, which
    for the H loss of the product (convex in E_int) makes it a lower bound. The C 1s products are ``kshell_rates``'."""
    key = ("ion_deposits", z, params, shape)
    if key in mol._cache:
        return mol._cache[key]
    out = []
    if z < max(mol.charges):
        field1 = _unit_field(shape)
        rate = valence_sigma(mol, z, params, shape) * _yield(mol, z, params) * field1.flux_at(E_PH) * _trapezoid_weights(E_PH)
        thr = float(state_threshold(mol, z))
        if z >= 0:
            e1 = photo.photoelectron_energy(E_PH, thr, reference_ev=photoelectron_reference(mol, params))[1]
            p2 = (photo.double_fraction(E_PH, double_threshold(mol, z)) if params.double_ionisation == "knockout"
                  else np.zeros(E_PH.size))
        else:
            e1 = E_PH - thr - electron_energy(mol, z, params)
            p2 = np.zeros(E_PH.size)
        rows = [(1, rate * (1.0 - p2), e1)]
        if np.any(p2 > 0.0):
            rows.append((2, rate * p2, photo.photoelectron_energy(E_PH, double_threshold(mol, z),
                                                                  reference_ev=photoelectron_reference(mol, params))[1]))
        for dz, w, e in rows:
            on = w > 0.0
            k = float(w.sum())
            if k > 0.0:
                out.append((dz, k, _round_off(e[on], 0.0, None, f"{mol.name} {z:+d}: internal energy after ionisation [eV]"),
                            w[on] / k))
    mol._cache[key] = out
    return out


def ionisation_products(mol, z, params, shape):
    """[(dz, k, E_int, w)] of the photoionisations that land on charge z: ``ionisation_deposits`` of z - 1 (dz = 1) and of
    z - 2 (dz = 2), in that order (the rows of the state's ("ion", shape) ``deposit_response``)."""
    return [d for dz in (1, 2) if z - dz in mol.species for d in ionisation_deposits(mol, z - dz, params, shape) if d[0] == dz]


def anion_valence_threshold(mol):
    """Threshold [eV] for removing a valence electron (not the extra one) from the anion, which leaves the neutral
    excited: the molecule's own ladder one Coulomb step down, 2 IP(0) - IP(+1), as WD01 eq. 2 steps IP_V(Z) with
    the charge. Without an IP(+1) (no computed dication, the "library" ladder) the step is LADDER_STEP_FIT's, the one
    ``extend_ladder`` gives such a molecule, so both ladders agree. At least the EA. C24H12 3.55, C54H18 3.75, C96H24
    3.88 eV with the calibrated shifts. Until 2026-11-02 the step without an IP(+1) was WD01's e^2/a (1 + 0.3 A/a) at
    the disc radius, which is 0.78x (0.73-0.85x) the DFT steps: the threshold was ~0.5 eV higher."""
    th = mol.ionisation_ev
    if 1 in th:
        ipv = 2.0 * th[0] - th[1]
    else:
        ipv = th[0] - (LADDER_STEP_FIT[0] + LADDER_STEP_FIT[1] / np.sqrt(mol.nc))
    return max(float(ipv), float(th[-1]))


def statistical_detachment(mol, params=Params(), grain_cm=10.0, energy_ev=E_PH, valence=False):
    """(Y, <E_k> [eV]) on ``energy_ev`` (default E_PH) of photodetachment of the anion: prompt plus delayed
    (``Params.detachment`` = "statistical"), and with ``valence`` (``Params.detachment`` = "valence") the prompt
    removal of a valence electron too. Any grid works, also below 5 eV, where the photons above the EA detach too.

    - Prompt: the WD01 photodetachment cross section (eq. 20, ``shared_physics.photodetachment_cross_section_vec``,
      threshold EA), as a share of the absorption (``absorption_sigma``, at least that cross section); the electron takes
      E - EA.
    - Valence (``valence``): of the other absorptions, a share y_v leaves promptly with a valence electron, the
      neutral left excited: WD01's valence-band photoemission of a negative grain (their eq. 2 and Sect. 2), with
      pahchem's photoionisation yield (``photo.yield_jochims1996``) above ``anion_valence_threshold`` and the
      photoelectron share of the excess (``photo.photoelectron_energy``). Without it a large anion, whose delayed
      detachment loses to IR emission, keeps its electron in the UV (C96H24 0.75 anionic in Tielens' diffuse
      cloud, against his 0.13).
    - Delayed: every other absorption converts internally and the anion, at internal energy E, autodetaches with
      the detailed-balance probability P_det(E) before IR emission takes it below EA (``attachment``,
      ``Attachment.delayed_detachment``); the electron leaves with the autodetachment spectrum's mean energy, a
      few 0.1 eV (thermionic; Bull+15 for UV-excited PAH anions). The rest stays as vibrational energy.
    """
    grid = np.asarray(energy_ev, dtype=float)
    f_att = attachment_fraction(mol, params)
    ref = photoelectron_reference(mol, params)
    key = ("detach_stat", params.sigma_scale, photo.uv_model_key(params.uv_cross_section), float(grain_cm), grid.tobytes(), bool(valence),
           f_att, ref)
    if key not in mol._cache:
        from pycalima.models.dust_charge.shared_physics import photodetachment_cross_section_vec
        from pycalima.pahchem.attachment import Attachment

        ea = mol.ionisation_ev[-1]
        lyman = E_PH_FUV[-1]
        lo = grid <= lyman
        E = np.append(grid[lo], lyman)                    # with 13.6 eV, where the EUV ramp starts
        sigma = absorption_sigma(mol.nc, -1, E, params.uv_cross_section, params.sigma_scale, ea)
        y_p = np.divide(photodetachment_cross_section_vec(E, ea, -1), sigma, out=np.zeros_like(E), where=sigma > 0)
        # not kept after: its arrays are ~0.5 MB a molecule (0.4 GB for v4.00), and the result is cached by grid
        at = Attachment(mol.species[0], mol.species[-1], ea, chemistry.disc_radius_cm(mol.nc),
                        e_excess_cm=(lyman - ea + 0.5) * dos.CM_PER_EV, grain_cm=grain_cm, capture_fraction=f_att)
        p_det, eps_cm = at.delayed_detachment(E * dos.CM_PER_EV)
        if valence:
            ipv = anion_valence_threshold(mol)
            y_v = photo.yield_jochims1996(E, ipv)
            ek_v = photo.photoelectron_energy(E, ipv, reference_ev=ref)[0]
        else:
            ipv, y_v, ek_v = ea, np.zeros_like(E), np.zeros_like(E)
        rest = (1.0 - y_p) * (1.0 - y_v)                 # absorptions that convert internally
        y = y_p + (1.0 - y_p) * y_v + rest * p_det
        ek = np.divide(y_p * np.clip(E - ea, 0.0, None) + (1.0 - y_p) * y_v * ek_v
                       + rest * p_det * eps_cm / dos.CM_PER_EV, y, out=np.zeros_like(E), where=y > 0)
        # above 13.6 eV: valence photoemission from the anion, rising to unity at Y_UNITY_EV, with the
        # electron energy of photo.photoelectron_energy (threshold the valence one, else EA)
        hi = grid[~lo]
        y_out, ek_out = np.empty_like(grid), np.empty_like(grid)
        y_out[lo], ek_out[lo] = y[:-1], ek[:-1]          # in the input order: the grid need not be sorted
        y_out[~lo] = np.clip(y[-1] + (1.0 - y[-1]) * (hi - lyman) / (Y_UNITY_EV - lyman), 0.0, 1.0)
        ek_out[~lo] = photo.photoelectron_energy(hi, ipv, reference_ev=ref if valence else None)[0]   # EA: not the ladder
        mol._cache[key] = (y_out, ek_out)
    return mol._cache[key]


def shift_thresholds(library, shifts_ev):
    """Copies of the molecules with each step z -> z+1 raised by ``shifts_ev[z]`` [eV] (fresh caches).

    For calibrating DFT thresholds (diagnostics/pahchem/calibrate_charge_inputs.py).
    """
    from dataclasses import replace

    out = []
    for m in library:
        th = {z: v + shifts_ev.get(z, 0.0) for z, v in m.ionisation_ev.items()}
        out.append(replace(m, ionisation_ev=th, ip_ev=th[0], ea_ev=th.get(-1), _cache={}))
    return out


#: the ``attachment.Attachment`` of each neutral-anion pair of this process, by its modes, EA, size and grid (populations
#: copy their molecules, ``extend_ladder``, and would each build it again: 0.2 s per molecule), the last STICKING_MEMO
_STICKING_MEMO, STICKING_MEMO = {}, 2048


def neutral_sticking(mol, temperature_k, capture_fraction=1.0):
    """Statistical s_e of the neutral (``attachment``), the neutral in its vibrational ground state, per anion
    formed with ``capture_fraction`` f of the classical captures (``attachment.Attachment``; ``attachment_sticking``
    gives f s_e, what multiplies the capture rate).

    The baseline instead uses the neutral's radiative P(E) under the Draine
    field at G0 = 1. That steady state costs ~7 s per molecule, has a mean
    energy of 5-11 cm^-1, and gives the same s_e as the ground state (checked
    for C26H16, C72H22 and C216H36 of the library). Between photons an
    interstellar PAH is cold.
    """
    from pycalima.pahchem.attachment import Attachment, grid_excess_cm

    if mol.ea_ev is None:
        raise ValueError(f"{mol.name}: no anion in the library, so no neutral sticking")
    akey = ("attachment", grid_excess_cm(temperature_k), float(capture_fraction))   # the grid holds the Maxwellian
    if akey not in mol._cache:
        h = hashlib.sha1(b"attachment")
        neutral, anion = mol.species[0], mol.species[-1]
        for v in (np.asarray(neutral.freqs_cm, dtype=float), np.asarray(anion.freqs_cm, dtype=float),
                  np.asarray(anion.einstein_a, dtype=float), float(mol.ea_ev), mol.nc, akey):
            _feed(h, v)
        key = h.hexdigest()
        if key not in _STICKING_MEMO:
            if len(_STICKING_MEMO) >= STICKING_MEMO:
                del _STICKING_MEMO[next(iter(_STICKING_MEMO))]
            _STICKING_MEMO[key] = (Attachment(neutral, anion, mol.ea_ev, chemistry.disc_radius_cm(mol.nc),
                                              e_excess_cm=akey[1], capture_fraction=capture_fraction), {})
        mol._cache[akey] = _STICKING_MEMO[key]
    at, done = mol._cache[akey]
    t = round(float(temperature_k), 6)
    if t not in done:
        done[t] = at.sticking(temperature_k, np.array([0.0]), np.array([1.0]))
    return done[t]


def attachment_fraction(mol, params=Params()):
    """f of this molecule: a logistic in N_C (``attachment_model="size"``, the default since 2026-11-02; measured at N_C <= 24 and
    at C60 only, an extrapolation in between), ``Params.attachment_fraction`` ("constant") or a logistic in its electron affinity
    ("ea"; the anion's detachment threshold ``ionisation_ev[-1]``). The same f sets the attachment and, by detailed balance, the
    anion's autodetachment (``statistical_detachment``)."""
    if params.attachment_model == "constant":
        return params.attachment_fraction
    lo = params.attachment_f_low
    if params.attachment_model == "size":
        return float(lo + (1.0 - lo) / (1.0 + (params.attachment_nc_half / mol.nc) ** params.attachment_nc_power))
    if params.attachment_model != "ea":
        raise ValueError(f"attachment_model must be 'constant', 'ea' or 'size', not {params.attachment_model!r}")
    ea = mol.ionisation_ev.get(-1) if mol.ea_ev is not None else None
    if ea is None:
        return params.attachment_fraction
    return float(lo + (1.0 - lo) / (1.0 + np.exp(-(ea - params.attachment_ea_half) / params.attachment_ea_width)))


def attachment_sticking(mol, temperature_k, params=Params()):
    """What multiplies the classical capture on the neutral: f (``attachment_fraction``) times the statistical s_e at that f
    (``neutral_sticking``)."""
    f = attachment_fraction(mol, params)
    return f * neutral_sticking(mol, temperature_k, f)


def capture_coefficient(mol, z, temperature_k, params=Params()):
    """Electron capture coefficient of charge z [cm^3 s^-1]: BT94/DS87 (``chemistry.electron_capture``), with the
    statistical sticking on the neutral (``attachment_sticking``) and the cation factor of ``Params.capture_model``."""
    st = attachment_sticking(mol, temperature_k, params) if z == 0 else 1.0
    return chemistry.electron_capture(mol.nc, z, temperature_k, sticking=st) * _capture_factor(mol, z, params, temperature_k)


def capture_energy(mol, z, temperature_k):
    """Mean kinetic energy a captured electron takes from the gas [eV]: kT Lambda~/J~ of Draine & Sutin 1987
    (eqs. 3.4-3.6) at the capture radius (``chemistry.disc_radius_cm``)."""
    from pycalima.models.dust_charge.shared_physics import DS87_lambda_scalar
    from pycalima.models.PAH_photophysics.pah_charge_utils import J_function_DS87

    a = chemistry.disc_radius_cm(mol.nc)
    return (chemistry.KB_EV * temperature_k * DS87_lambda_scalar(z, -1, a, temperature_k)
            / J_function_DS87(z, -1, a, temperature_k))


#: kinetic energies at infinity [eV] on which an arriving electron's stopping probability is tabulated (``capture_stopping``)
CAPTURE_STOP_EV = np.geomspace(1.0, 1.0e5, 41)


def capture_stopping(mol, z, params=Params()):
    """P_stop on CAPTURE_STOP_EV: the probability that an electron arriving on charge z with that kinetic energy at
    infinity loses enough crossing the molecule to stay (``impact.stopping_probability``): its energy at infinity, or on
    an anion all it has at the surface. Its collisions are at the impact energy (``impact.coulomb_focus``), with z's
    threshold for Kim's scaling and the oscillator strengths of ``Params.uv_cross_section``, as in ``hot_passages``."""
    u = -float(z) * impact.E2_EV_CM / impact.disc_radius_cm(mol.nc)       # the electron's potential energy there
    ip = _ladder_ip(mol, z)
    w = impact.w_grid()
    df = impact.oscillator_strength(mol.nc, mol.nh, z, w, params.uv_cross_section)
    out = np.zeros(CAPTURE_STOP_EV.size)
    for i, e in enumerate(CAPTURE_STOP_EV):
        t = e - u
        if t <= 0.0:
            continue
        c = impact.collisions(impact.ELECTRON, t, df, w, correction=impact.low_speed_correction(impact.ELECTRON, t),
                              ip_ev=ip)
        out[i] = impact.stopping_probability(c, mol.nc, e - max(u, 0.0))
    return out


def capture_stop_factors(mol, z, temperature_k, params, p_stop):
    """(rate, energy): the factors on ``capture_coefficient`` and ``capture_energy`` of charge z in hot gas
    (``Params.hot_gas``) when a captured electron must lose its energy in the molecule (Weingartner & Draine 2001,
    Sect. 3.1; ``p_stop`` = ``capture_stopping``). An arrival at E sticks with the effective low-energy sticking s
    (``attachment_sticking`` x ``_capture_factor``) up to the energy where P_stop peaks, ~20-26 eV (below it the
    collisions lack the channels that capture slow electrons: vibrations, resonances), and with min(s, P_stop(E)) above it,
    the bound of energy conservation (the user's choice, 2026-11-04). Arrivals weigh E exp(-E/kT) (1 - U/E), classical
    focusing; the factors are 1 below ~1e4 K. Returns (rate, energy, energy_low): the factor on the capture rate, on the
    mean kinetic energy of the captured electrons, and on the part of it the passages do not hold, the captures at or
    below the peak (those above it lose their energy in passages that ``hot_passages`` counts already). Recombination
    cooling and the "rec" deposit take energy_low (pahchem audit M2)."""
    kt = chemistry.KB_EV * float(temperature_k)
    u = -float(z) * impact.E2_EV_CM / impact.disc_radius_cm(mol.nc)
    s = (attachment_sticking(mol, temperature_k, params) if z == 0 else 1.0) * _capture_factor(mol, z, params,
                                                                                               temperature_k)
    e = np.geomspace(1e-3 * kt, 50.0 * kt, 400)
    w = impact._trapezoid_weights(np.log(e)) * e * e * np.exp(-e / kt) * np.clip(1.0 - u / e, 0.0, None)
    if s <= 0.0 or w.sum() <= 0.0:
        return 1.0, 1.0, 1.0
    e_peak = CAPTURE_STOP_EV[int(np.argmax(p_stop))]
    ps = np.interp(np.log(e), np.log(CAPTURE_STOP_EV), p_stop)
    low = e <= e_peak
    stick = np.where(low, s, np.minimum(s, ps))
    if (w * stick).sum() <= 0.0:
        return 0.0, 1.0, 1.0
    rate = float((w * stick).sum() / (w.sum() * s))
    mean = (w * e).sum() / w.sum()
    energy = float(((w * stick * e).sum() / (w * stick).sum()) / mean)
    energy_low = float(((w * np.where(low, stick, 0.0) * e).sum() / (w * stick).sum()) / mean)
    return rate, energy, energy_low


def ion_up_rate(mol, z, env, params=Params()):
    """Rate [s^-1] at which ions take the molecule from charge z to z + 1 at ``env`` (``env.ions``).

    Anions: mutual neutralisation with every ion (``chemistry.mutual_neutralisation``). Neutrals: the ion-neutral
    reaction (``chemistry.ion_neutral_rate``) for ions whose atom's IP exceeds the molecule's, plus the [M-PAH]+
    adducts of ``chemistry.ADDUCT_IONS`` below it if ``Params.metal_adducts``. Cations and above: none without
    ``Params.hot_gas`` (a Coulomb barrier, below ~1e4 K); with it the transfers of ``chemistry.ion_cation_transfer``
    (anchored on H+ + C60, Moretto-Capelle+04): exothermic where the atom's IP lies above the state's threshold, and
    endothermic below it, by the threshold law (``deficit_ev``, the user's choice of 2026-11-06; before, they were not
    followed). Out of the top state it is an overflow (``chain_overflow``, pahchem audit m10).
    """
    up = 0.0
    for ion, n in getattr(env, "ions", ()):
        if z == -1:
            up += n * chemistry.mutual_neutralisation(mol.nc, mol.nh, -1, ion, env.temperature_k, params.mn_efficiency)
        elif z == 0:
            if chemistry.IONS[ion][1] > mol.ionisation_ev[0] or (params.metal_adducts and ion in chemistry.ADDUCT_IONS):
                up += n * chemistry.ion_neutral_rate(mol.nc, mol.nh, ion, params.ion_neutral_efficiency)
        elif z >= 1 and params.hot_gas:
            up += n * chemistry.ion_cation_transfer(mol.nc, mol.nh, z, ion, env.temperature_k,
                                                    deficit_ev=transfer_deficit(mol, z, ion))
    return up


def transfer_deficit(mol, z, ion):
    """[eV]: by how much ``ion``'s charge transfer onto charge z (z >= 1) is endothermic, the state's threshold less the
    atom's IP; 0 if it is exothermic."""
    return max(float(state_threshold(mol, z)) - chemistry.IONS[ion][1], 0.0)


def transfer_cooling(mol, z, env, params=Params()):
    """[eV s^-1]: what the endothermic transfers out of charge z (``ion_up_rate``) take from the gas, each its deficit
    (``transfer_deficit``) from the pair's kinetic energy; the product keeps none of it."""
    if z < 1 or not params.hot_gas:
        return 0.0
    out = 0.0
    for ion, n in getattr(env, "ions", ()):
        d = transfer_deficit(mol, z, ion)
        if d > 0.0:
            out += n * d * chemistry.ion_cation_transfer(mol.nc, mol.nh, z, ion, env.temperature_k, deficit_ev=d)
    return out


def env_field(env):
    """The radiation field of a ``baseline.Environment`` as the population sees it: G0 times ``_unit_field`` of its
    ``field_shape`` (with the 1 eV start and the unscaled old stars of a Draine cell)."""
    f1 = _unit_field(field_shape(env))
    return fields.RadiationField(f1.energy_ev, env.g0 * f1.photon_flux, f1.name)


#: PAH- + H -> PAH(0, N_H+1) + e- [cm^3 s^-1] (Demarais+12, measured on dehydrogenated anions)
ASSOCIATIVE_DETACHMENT = 7.8e-10


def anion_associative_detachment(params=Params()):
    """k [cm^3 s^-1] of associative detachment by H out of the anion of the charge chain, which holds N_H0: zero
    with ``Params.associative_detachment = "dehydrogenated"`` (the default; Demarais+12 measured it on
    dehydrogenated anions only), ``ASSOCIATIVE_DETACHMENT`` with "all" (every anion, as Andrews+16 and pahchem
    before 2026-10-26)."""
    if params.associative_detachment not in ("dehydrogenated", "all"):
        raise ValueError(f"associative_detachment must be 'dehydrogenated' or 'all', not {params.associative_detachment!r}")
    return ASSOCIATIVE_DETACHMENT if params.associative_detachment == "all" else 0.0


def charge_fractions(mol, env, params=Params()):
    """{Z: f} of the chain -1/0/+1 at ``env`` (a ``baseline.Environment``)."""
    field_ = env_field(env)
    T = env.temperature_k
    zs = mol.charges
    f = {zs[0]: 1.0}
    for z in zs[:-1]:
        sigma = state_sigma(mol, z, params)
        up = photo.ionisation_rate(field_, sigma, _yield(mol, z, params), E_PH)
        if z == -1:
            up += anion_associative_detachment(params) * env.n_h
        up += ion_up_rate(mol, z, env, params)
        s = attachment_sticking(mol, T, params) if z + 1 == 0 else 1.0
        down = chemistry.electron_capture(mol.nc, z + 1, T, sticking=s) * _capture_factor(mol, z + 1, params, T) * env.n_e
        f[z + 1] = f[z] * up / down
    total = sum(f.values())
    return {z: v / total for z, v in f.items()}


# -- emission and hydrogenation ---------------------------------------------------------------
def _band_matrix(freqs_cm, windows_um, local=(), fwhm_cm=15.0, shift_cm=-15.0):
    """M[i, b]: fraction of a Lorentzian line at mode i inside window b (``emission.spectrum`` profile).

    For the bands in ``local``, the line's own value at the window edges is
    taken off as a baseline linear in wavelength (F_nu against lambda, as in
    the observed measurement), integrated over nu.
    """
    nu = np.asarray(freqs_cm, dtype=float)[:, None] + shift_cm
    lam1 = np.array([w[0] for w in windows_um.values()])[None, :]
    lam2 = np.array([w[1] for w in windows_um.values()])[None, :]
    nu1, nu2 = 1e4 / lam1, 1e4 / lam2
    g = 0.5 * fwhm_cm
    M = (np.arctan((nu1 - nu) / g) - np.arctan((nu2 - nu) / g)) / np.pi
    is_local = np.array([b in local for b in windows_um])[None, :]
    if is_local.any():
        L1, L2 = (g / np.pi / ((x - nu) ** 2 + g**2) for x in (nu1, nu2))
        base = L1 * (nu1 - nu2) + (L2 - L1) / (lam2 - lam1) * (1e4 * np.log(nu1 / nu2) - lam1 * (nu1 - nu2))
        M = np.where(is_local, M - base, M)
    return M


@functools.lru_cache(maxsize=None)
def _unit_field(shape):
    """The field of ``shape`` at G0 = 1.

    - ("kurucz", T_eff[, e_max_ev[, e_min_ev]]): the star's whole spectrum from ``E_PH_MIN_EV`` (or e_min_ev);
    - ("draine",): Draine's (1978) far-UV fit from 5 eV, nothing below;
    - ("draine", r[, e_min_ev]): the same plus r x the solar-circle old-star light below 5 eV (``fields.old_stars_flux``),
      r = ``Environment.optical_isrf`` / G0, so that G0 times this field leaves the old stars unscaled;
    - ("cr_uv", source, x_h2, signature): the cosmic-ray-induced field at zeta(H2) = ``cr_uv.ZETA_REF`` (``cr_uv.field``
      on E_PH), so that "G0" is zeta(H2) / 1e-16, RTZ's scale factor; ``cr_uv.signature`` keys the cached bases.
    """
    if shape[0] == "cr_uv":
        from pycalima.pahchem import cr_uv

        return cr_uv.field(cr_uv.ZETA_REF, shape[2], shape[1], energy_ev=E_PH)
    if shape[0] == "kurucz":
        e_max = shape[2] if len(shape) > 2 else fields.LYMAN_EV
        e_min = shape[3] if len(shape) > 3 else E_PH_MIN_EV
        return fields.kurucz(shape[1], 1.0, e_max=e_max, e_min=e_min)
    if shape[0] == "powerlaw":                 # ("powerlaw", alpha, e_max_ev, e_min_ev)
        return fields.powerlaw(shape[1], 1.0, e_min=shape[3], e_max=shape[2])
    if len(shape) == 1:
        return fields.draine1978(g0=1.0)
    e_min = shape[2] if len(shape) > 2 else E_PH_MIN_EV
    return fields.draine1978(g0=1.0, e_min=e_min, old_stars=shape[1])


def field_shape(env):
    """The field-shape key of a ``baseline.Environment``; the keys carry their cuts only where they differ from
    13.6 eV and ``E_PH_MIN_EV``.

    The Draine shape carries the old-star light per unit G0, ``optical_isrf / g0`` (none if the field starts at
    5 eV or above): one shape per G0 for such cells, within which every rate is again G0 times its basis value.
    """
    if getattr(env, "field_cr_uv", None) is not None:
        from pycalima.pahchem import cr_uv

        return ("cr_uv", str(env.field_cr_uv), float(getattr(env, "cr_uv_x_h2", 0.5)), cr_uv.signature(env.field_cr_uv))
    e_min = float(getattr(env, "field_e_min_ev", E_PH_MIN_EV))
    if getattr(env, "field_powerlaw", None) is not None:
        return ("powerlaw", float(env.field_powerlaw), float(env.field_e_max_ev), e_min)
    if env.teff_k is None:
        optical = float(getattr(env, "optical_isrf", 0.0))
        if optical <= 0.0 or e_min >= fields.DRAINE_E_MIN_EV:
            return ("draine",)
        r = optical / env.g0
        return ("draine", r) if e_min == E_PH_MIN_EV else ("draine", r, e_min)
    e_max = float(getattr(env, "field_e_max_ev", fields.LYMAN_EV))
    if e_min != E_PH_MIN_EV:
        return ("kurucz", int(env.teff_k), e_max, e_min)
    return ("kurucz", int(env.teff_k)) if e_max == fields.LYMAN_EV else ("kurucz", int(env.teff_k), e_max)


#: G0 at which ``StateBasis.k_h`` is tabulated (the multiphoton steady state at each)
K_H_G0_GRID = np.geomspace(1.0, 1e6, 13)


@dataclass(frozen=True)
class StateBasis:
    """One charge state under one field shape; rates at G0 = 1, except k_h on ``K_H_G0_GRID``."""

    freqs_cm: np.ndarray
    emitted: np.ndarray          # IR power per mode at G0 = 1 [cm^-1 s^-1]
    k_up: float                  # photo-ionisation or detachment rate at G0 = 1 [s^-1]
    k_h: np.ndarray              # first-H-loss photo-rate at each K_H_G0_GRID [s^-1]
    absorbed: float              # photon heating power at G0 = 1 [cm^-1 s^-1]
    recombination: float = 0.0   # recombination heating power at G0 = 1 [cm^-1 s^-1] (included in emitted)
    ladder_h: np.ndarray | None = None    # (4, K_H_G0_GRID): H-loss rates of N_H0 ... N_H0-3 [s^-1]
    ladder_h2: np.ndarray | None = None   # (4, K_H_G0_GRID): H2-loss rates of the same states [s^-1]
    # N_H0-1 ... N_H0-3 (as far as the ladder goes): their modes and IR power per mode at G0 = 1, heated as
    # this state (the same sigma, yield and recombinations: ``build_state_basis``)
    ladder_freqs: tuple = ()
    ladder_emitted: tuple = ()
    ladder_c2h2: np.ndarray | None = None   # (4, K_H_G0_GRID): C2H2-loss rates of N_H0 ... N_H0-3 [s^-1] (``loss_extras``)
    ladder_hp: np.ndarray | None = None     # (4, K_H_G0_GRID): proton-emission rates of the same states [s^-1]

    def k_h_at(self, g0):
        """k_H [s^-1] at ``g0`` (array), log-log interpolated; rates below 1e-300 s^-1 count as zero."""
        g = np.asarray(g0, dtype=float)
        if (g < K_H_G0_GRID[0]).any() or (g > K_H_G0_GRID[-1]).any():
            raise ValueError(f"G0 outside the k_H table {K_H_G0_GRID[0]:g}-{K_H_G0_GRID[-1]:g}")
        lk = np.log(np.maximum(self.k_h, 1e-300))
        return np.exp(np.interp(np.log(g), np.log(K_H_G0_GRID), lk))


#: the CH-stretch scale and its window live in ``baseline`` (every cascade applies them; pahchem-audit C7)
_einstein_a = scaled_einstein_a


#: classical upper limit of the factor on BT94 capture. BT94 with sticking 1 already holds the Coulomb
#: focusing and the image potential (Draine & Sutin J~), so only the 0.82 disc factor can rise to 1 (review B).
CAPTURE_MAX = 1.0 / 0.82


#: ``Params.capture_model``: "bt94" (sticking 1), "lab_trend" (``chemistry.lab_recombination_factor``, a power law over all nine
#: lab points), "lab_cold" (the default since 2026-11-02) and "lab_pahs" (``chemistry.lab_sticking``, saturating at 1, fitted to
#: the photo-ion or to every PAH point at N_C 10-16; above that an extrapolation), "wd01" (WD01's capture by a graphite sphere
#: of the same carbon count over BT94's, for comparison only)
CAPTURE_MODELS = ("bt94", "lab_trend", "lab_cold", "lab_pahs", "wd01")


def wd01_capture_ratio(nc, z, temperature_k):
    """WD01's electron capture by a graphite sphere of radius 10 A (N_C/468)^(1/3) with WD01's sticking, over BT94's
    (``chemistry.electron_capture``, sticking 1) at charge z: the "wd01" capture model, a comparison, not physics of a PAH."""
    from pycalima.models.dust_charge import shared_physics as sp

    a = 1e-7 * (nc / 468.0) ** (1.0 / 3.0)
    k = sp.collisional_rates_electrons_vector(np.array([a]), np.array([float(z), float(z) + 1.0]), 1.0, float(temperature_k),
                                              sp.electron_sticking_coefficient_graphite)[0]
    return float(k) / chemistry.electron_capture(nc, z, temperature_k)


def _capture_factor(mol, z, params, temperature_k=300.0):
    """Factor on BT94 capture by charge z: the cation sticking of ``Params.capture_model`` and ``capture_t_power`` (cations and
    dications only).

    The lab trend is capped at ``CAPTURE_MAX`` (and at ``capture_cap`` if
    lower). A ``capture_scale`` that alone exceeds the classical limit raises.
    """
    if params.capture_scale > CAPTURE_MAX * (1 + 1e-12):
        raise ValueError(f"capture_scale {params.capture_scale} exceeds the classical limit {CAPTURE_MAX:.3f}")
    if params.capture_model not in CAPTURE_MODELS:
        raise ValueError(f"capture_model must be one of {CAPTURE_MODELS}, not {params.capture_model!r}")
    f = params.capture_scale
    if z >= 1 and params.capture_model == "lab_trend":
        cap = CAPTURE_MAX if params.capture_cap is None else min(params.capture_cap, CAPTURE_MAX)
        f *= float(chemistry.lab_recombination_factor(mol.nc, cap))
    elif z >= 1 and params.capture_model in ("lab_cold", "lab_pahs"):
        f *= float(chemistry.lab_sticking(mol.nc, params.capture_model.split("_")[1]))
    elif z >= 1 and params.capture_model == "wd01":
        f *= wd01_capture_ratio(mol.nc, z, temperature_k)
    if z >= 1 and params.capture_t_power:
        f *= (float(temperature_k) / 300.0) ** (-params.capture_t_power)
    return min(f, CAPTURE_MAX)


def _basis_inputs(mol, z, shape, de_cm, e_max_cm, params):
    sp = mol.species[z]
    top = z == max(mol.charges)
    recomb = float(mol.ionisation_ev[z]) if (params.recombination_heating and not top) else 0.0
    ladder, extras = [], []
    for k in range(4):
        n = mol.nh - k
        if n < 1:
            break
        s_ = sp if k == 0 else sp.stripped(n)
        ladder.append((s_.freqs_cm, _einstein_a(s_, params), n, bool(n % 2 == 0 and n > mol.edges.get("solo", 0))))
        extras.append(loss_extras(mol, z, n, params))
    return dict(freqs=sp.freqs_cm, einstein_a=_einstein_a(sp, params), nc=mol.nc, nh=mol.nh, z=z, yield_=_yield(mol, z, params),
                shape=shape, de_cm=de_cm, e_max_cm=e_max_cm, h_even=tuple(params.h_even),
                sigma_scale=params.sigma_scale, recomb_ev=recomb, ladder=tuple(ladder),
                h_odd=tuple(params.h_odd), h2=tuple(params.h2), threshold_ev=float(state_threshold(mol, z)), extras=tuple(extras),
                ionising_heat=bool(params.ionising_photons_heat),
                # the UV cross section enters the inputs (and the key) unless it is Li & Draine's, the bases before 2026-10-26
                **({"uv_cross_section": photo.uv_model_key(params.uv_cross_section)} if params.uv_cross_section != "li_draine" else {}),
                # the anion's absorption raised to its photodetachment (``absorption_sigma``; enters the key only then)
                **({"detach_ea": _detach_ea(mol, z, params)} if _detach_ea(mol, z, params) is not None else {}),
                # only then does the electron's share enter the basis (and its key)
                **({"electron_ev": electron_energy(mol, z, params)} if params.ionising_photons_heat else {}))


#: bump when build_state_basis changes, so stored bases are rebuilt (with the grids, part of every key).
#: The result caches of the diagnostics key on it too: 6 is the photon grid from 1 eV; 7 the Kurucz fields
#: normalised over the whole 6-13.6 eV band (``fields.RadiationField.g0``, +0.15 %); 8 the emission of the
#: dehydrogenated states N_H0-1 ... N_H0-3; 9 (2026-11-02) the loss ceiling of 45 eV at every size (tilted counts,
#: ``dos.count_tilt``), each state's deposit grid and ceilings up to its hardest non-ionising deposit, and yields
#: unity only above the threshold; 10 (2026-11-04) the loss channels of ``loss_extras`` (C2H2 loss, proton emission).
BASIS_VERSION = 10
#: bump when emission_temperature_shares changes, so stored shares are rebuilt (keyed with the basis key)
TSHARES_VERSION = 1


def _feed(h, v):
    """Arrays by their bytes, also inside tuples (the ladder): repr printed them, slowly and to 8 digits."""
    if isinstance(v, np.ndarray):
        h.update(np.ascontiguousarray(v).tobytes())
    elif isinstance(v, tuple) and any(isinstance(u, (np.ndarray, tuple)) for u in v):
        for u in v:
            h.update(b"(")
            _feed(h, u)
            h.update(b")")
    else:
        h.update(repr(v).encode())


def _basis_key(inp, legacy=False):
    """Cache key of a basis; ``legacy``: the key before ``_feed`` (tuples through repr), to find old files."""
    h = hashlib.sha1()
    h.update(f"v{BASIS_VERSION}".encode())
    h.update(np.ascontiguousarray(E_PH).tobytes())
    h.update(np.ascontiguousarray(K_H_G0_GRID).tobytes())
    if not legacy:                     # every grid of build_state_basis, not only the photon grid
        h.update(np.ascontiguousarray(E_DEP).tobytes())
        h.update(repr((LOSS_E_MAX_EV, LOSS_DE_CM, LOSS_OVERFLOW_TOL, LOSS_STEP_EV, LOSS_E_CAP_EV,
                       photo.PARTITION_NODES)).encode())
    for k in sorted(inp):
        v = inp[k]
        h.update(k.encode())
        if legacy:
            h.update(np.ascontiguousarray(v).tobytes() if isinstance(v, np.ndarray) else repr(v).encode())
        else:
            _feed(h, v)
    return h.hexdigest()[:20]


#: energy grid of the multiphoton loss cascades (the baseline's): two-photon states reach ~27 eV, and the
#: 16 eV emission grid cut that tail off, underestimating k_H by orders of magnitude near threshold. 45 eV holds
#: three 13.6 eV photons. A state that absorbs harder photons without ionising (``deposit_grid``) starts with the same
#: headroom above its hardest deposit (``_loss_ceiling``). The ceiling then rises by LOSS_STEP_EV until the jumps
#: clamped at it at the top of K_H_G0_GRID (``SteadyState.overflow``, an upper bound on the losses they could add) are
#: below LOSS_OVERFLOW_TOL of the loss rate there, up to LOSS_E_CAP_EV (raising there): at G0 = 1e6 C54 and C96 converge at 45 eV,
#: C150 at 60 and C268 at 75 (multiphoton heating of large PAHs). The state counts are tilted where they would
#: overflow (``dos.count_tilt``). Until 2026-11-02 the ceiling dropped to 30 or 20 eV for N_C >~ 150 instead, which
#: underestimated their k_H at G0 = 1e6 46x (C150).
LOSS_E_MAX_EV, LOSS_DE_CM = 45.0, 200.0
LOSS_OVERFLOW_TOL, LOSS_STEP_EV, LOSS_E_CAP_EV = 1e-2, 15.0, 150.0


def _loss_converged(state, e_max_ev):
    """True if a loss cascade's ``SteadyState`` at the top of K_H_G0_GRID clamps few enough jumps (LOSS_OVERFLOW_TOL);
    raises if not at LOSS_E_CAP_EV (pahchem audit m5: it returned silently there)."""
    loss = sum(state.rates.values())
    if state.overflow <= LOSS_OVERFLOW_TOL * loss:
        return True
    if e_max_ev >= LOSS_E_CAP_EV:
        raise ValueError(f"multiphoton loss cascade not converged at {e_max_ev:g} eV (LOSS_E_CAP_EV): clamped jumps "
                         f"{state.overflow:.3e} s^-1 against a loss rate of {loss:.3e} s^-1")
    return False

#: hydrogenation states that emit per charge state: N_H0 and the ladder's N_H0-1 ... N_H0-3 (``StateBasis``)
N_EMITTING = 4
#: environments whose hydrogenation chains a Population keeps (``_chain_distributions``): the 48 cells of a PDRs4All
#: template, ~0.5 MB each for v4.00
CHAIN_MEMO = 64
#: environments whose flow coefficients and loss kernel a Population keeps (``_flow_terms``, ``_loss_kernel``): tens of
#: MB each for v4.00, and the charge fixed point and the flows' consumers ask for one environment at a time
FLOW_MEMO = 2
#: the charge fractions and the protons hot ions emit are iterated until no fraction moves by more than FISSION_TOL
#: (absolute, relative to itself): the small fractions to 1e-12 of themselves
FISSION_TOL, FISSION_ITER_MAX = (1e-18, 1e-12), 400


@dataclass(frozen=True)
class StateSpectra:
    """Spectra of the states that hold molecules, one row each: ``rows[r]`` on the caller's grid is state (molecule
    ``s[r]``, charge slot ``slot[r]`` = z - zmin, hydrogenation N_H0 - ``k[r]``), at G0 = 1. The dense S[s, slot, k, i]
    is zero everywhere else: with the cold-limit ladder 91 % of v4.00 under the Orion Bar field (7329 of 78320 charge
    states live), 1.6 GB on the fit grid against these rows' 0.16 GB."""

    rows: np.ndarray             # (n_states, n_nu)
    s: np.ndarray
    slot: np.ndarray
    k: np.ndarray
    n_mol: int

    def _sum(self, values, groups, n_groups):
        from scipy.sparse import csr_matrix

        return np.asarray(csr_matrix((values, (groups, np.arange(self.s.size))), shape=(n_groups, self.s.size)) @ self.rows)

    def combine(self, W):
        """P[s, i] = sum over slot and k of W[s, slot, k] S[s, slot, k, i] (W: ``Population.emission_weights``)."""
        return self._sum(W[self.s, self.slot, self.k], self.s, self.n_mol)

    def by_slot(self, W, weights, n_slots):
        """Z[slot, i] = sum over s and k of weights[s] W[s, slot, k] S[s, slot, k, i]: the spectrum by charge slot."""
        return self._sum(np.asarray(weights)[self.s] * W[self.s, self.slot, self.k], self.slot, n_slots)

    def state(self, s, slot, k):
        """S[s, slot, k, :] (zeros for a state without a row)."""
        r = np.flatnonzero((self.s == s) & (self.slot == slot) & (self.k == k))
        return self.rows[r[0]] if r.size else np.zeros(self.rows.shape[1])


def deposit_grid(field, sigma, yield_):
    """The deposit grid of one state: E_DEP, and above it, where the state absorbs without ionising (a threshold above
    E_DEP's top, highly charged states in EUV fields), the photon nodes of E_PH up to the first at which every
    absorption ionises again."""
    non = np.asarray(sigma) * (1.0 - np.asarray(yield_, dtype=float)) * field.flux_at(E_PH)
    above = np.nonzero((E_PH > E_DEP[-1]) & (non > 0.0))[0]
    if not above.size:
        return E_DEP
    last = min(int(above[-1]) + 1, E_PH.size - 1)
    return np.concatenate([E_DEP, E_PH[(E_PH > E_DEP[-1]) & (np.arange(E_PH.size) <= last)]])


def deposit_spectrum(field, sigma, yield_, threshold_ev, ionising_heat, electron_ev=None, e_dep=None):
    """h(e_dep) [s^-1 eV^-1]: the rate at which absorptions leave internal energy E_dep in the molecule.

    Non-ionising absorptions, sigma (1 - Y) F, deposit the photon energy. Ionisations, sigma Y F, deposit the ion's
    internal energy if ``ionising_heat``, else nothing (as before 2026-10-22): E - threshold - <E_k>, with the
    electron's <E_k> on E_PH from ``electron_ev`` (``electron_energy``, so that the anion's detached electron and
    the energy left behind add up to the excess), by default that of ``photo.photoelectron_energy``.
    Each photon bin's rate is split between its two neighbouring deposit bins, conserving the rate. ``e_dep``: the
    state's grid (``deposit_grid``; E_DEP by default).
    """
    e_dep = E_DEP if e_dep is None else e_dep
    F = field.flux_at(E_PH)
    y = np.asarray(yield_, dtype=float)
    non = np.asarray(sigma) * (1.0 - y) * F
    if np.any(non[E_PH > e_dep[-1]] > 0.0):
        raise ValueError("non-ionising absorption above the deposit grid (see deposit_grid)")
    h = np.interp(e_dep, E_PH, non, left=0.0, right=0.0)
    if ionising_heat:
        w = np.asarray(sigma) * y * F * _trapezoid_weights(E_PH)          # ionisations per photon bin [s^-1]
        if electron_ev is None:
            _, e_int = photo.photoelectron_energy(E_PH, threshold_ev)
        else:
            e_int = np.clip(E_PH - threshold_ev - np.asarray(electron_ev, dtype=float), 0.0, None)
        keep = (w > 0) & (e_int >= e_dep[0]) & (e_int <= e_dep[-1])
        i = np.clip(np.searchsorted(e_dep, e_int[keep], side="right") - 1, 0, e_dep.size - 2)
        f = (e_int[keep] - e_dep[i]) / (e_dep[i + 1] - e_dep[i])
        dep = np.zeros(e_dep.size)
        np.add.at(dep, i, (1.0 - f) * w[keep])
        np.add.at(dep, i + 1, f * w[keep])
        h = h + dep / _trapezoid_weights(e_dep)
    return h


def _trapezoid_weights(x):
    """w_i with sum_i w_i f_i = trapezoid(f, x)."""
    x = np.asarray(x, dtype=float)
    w = np.zeros(x.size)
    dx = np.diff(x)
    w[:-1] += 0.5 * dx
    w[1:] += 0.5 * dx
    return w


def _loss_ceiling(e_dep, h):
    """Ceiling of a state's loss cascade [eV]: LOSS_E_MAX_EV, or the same headroom above its hardest deposit."""
    top = float(e_dep[np.nonzero(h > 0.0)[0][-1]]) if np.any(h > 0.0) else 0.0
    return max(LOSS_E_MAX_EV, top + LOSS_E_MAX_EV - fields.LYMAN_EV)


def _loss_grid(freqs, einstein_a, e_max_ev=LOSS_E_MAX_EV):
    """(CascadeModel, E, counts, beta) of the multiphoton loss cascade up to ``e_max_ev``: the state counts tilted
    by beta where they would overflow (``dos.count_tilt``; the rates are those of the untilted counts)."""
    cm = energy_me.CascadeModel(freqs, einstein_a, e_max_ev * dos.CM_PER_EV, LOSS_DE_CM)
    top = cm.energy_cm[-1] + 200.0
    beta = dos.count_tilt(freqs, top)
    E, counts = dos.beyer_swinehart(freqs, top, beta=beta)
    if not (np.isfinite(counts).all() and np.isfinite(cm.k_modes).all()):
        raise ValueError(f"loss cascade to {e_max_ev:g} eV is not finite")
    return cm, E, counts, beta


def _h_loss_rate(freqs, E, counts, pick, nh, h_even, h_odd, beta=0.0):
    """RRKM H loss on the loss grid: even-H (E0, dS) with degeneracy N_H, odd-H radical values with 1."""
    (e0, ds), deg = (h_even, nh) if nh % 2 == 0 else (h_odd, 1)
    return rates.rrkm(E, counts, rates.transition_state_for_entropy(freqs, 3050.0, ds, n_scaled=5),
                      e0 * dos.CM_PER_EV, degeneracy=deg, beta=beta)[pick]


def _loss_tables(freqs, einstein_a, nh, h2_ok, h, h_even, h_odd, h2, grid=None, e_dep=None, e_max_ev=None, extra=()):
    """(k_H, k_H2, {name: k}) on K_H_G0_GRID for one hydrogenation state: RRKM sinks competing in its multiphoton cascade,
    with the channels ``extra`` of ``loss_extras``.
    ``grid``: this state's ``_loss_grid`` (to ``e_max_ev``) and H-loss rate, if already built ((cm, E, counts, beta),
    k_H(E)); ``e_dep``: the grid of the deposits ``h`` (``deposit_grid``; E_DEP by default); ``e_max_ev``: the
    ceiling to start from (``_loss_ceiling`` by default), raised until converged (LOSS_OVERFLOW_TOL)."""
    e_dep = E_DEP if e_dep is None else e_dep
    e_max = _loss_ceiling(e_dep, h) if e_max_ev is None else e_max_ev
    while True:
        if grid is None:
            cm, E, counts, beta = _loss_grid(freqs, einstein_a, e_max)
            pick = np.rint(cm.energy_cm / (E[1] - E[0])).astype(int)
            k_h = _h_loss_rate(freqs, E, counts, pick, nh, h_even, h_odd, beta)
        else:
            (cm, E, counts, beta), k_h = grid
            pick = np.rint(cm.energy_cm / (E[1] - E[0])).astype(int)
            grid = None                                  # rebuilt above it if it does not converge
        ch = {"H": k_h}
        if h2_ok:
            ch["H2"] = rates.rrkm(E, counts, rates.transition_state_for_entropy(freqs, 3050.0, h2[1], n_scaled=5),
                                  h2[0] * dos.CM_PER_EV, degeneracy=2, beta=beta)[pick]
        ch |= _extra_rates(freqs, E, counts, pick, beta, extra)
        out = {c: [] for c in ch}
        for g in K_H_G0_GRID:
            s = cm.steady_state(e_dep, g * h, ch)
            for c in ch:
                out[c].append(s.rates[c])
        if _loss_converged(s, e_max):
            kh2 = np.array(out["H2"]) if "H2" in out else np.zeros(K_H_G0_GRID.size)
            return np.array(out["H"]), kh2, {c: np.array(v) for c, v in out.items() if c not in ("H", "H2")}
        e_max = min(e_max + LOSS_STEP_EV, LOSS_E_CAP_EV)


# -- the bare skeleton (N_H = 0) and its destruction (2026-11-04) ----------------------------------------------------
#: (a, b): a bare skeleton without PAHdb modes takes its parent's ``stripped(0)`` modes (their frequencies set its state
#: counts) with the Einstein A times R = a N_C^b, the ratio of the thermal IR power at 800 K of PAHdb v4.00's bare skeletons
#: to their parents' stripped(0), over the 166 matched pairs (neutral parents, N_C 14-96): ~12 at N_C ~ 17, 8-11 at 20-35,
#: 5.5 (C54), 3.4 (C66), 4.4 (C96); rms x1.98 about the law. stripped(0) lacks the C-C bands that carry a bare skeleton's
#: emission (``PAHSpecies.stripped``), so it would cool 4-12x too slowly
BARE_IR_LAW = (54.66, -0.522)
#: folding of a bare flake into a cage (a fullerene: the PAH gone), (A [s^-1], E_a [eV]) in the inverse-Laplace form:
#: Berné, Montillaud & Joblin 2015 (A&A 577, A133) Table 2, from the MD of C96 by Lebedeva+12; for N_C >= BARE_FOLD_MIN_NC
#: (C60, the smallest stable fullerene)
BARE_FOLD, BARE_FOLD_MIN_NC = (1.0e15, 4.0), 60
#: bump when ``bare_loss_table`` changes, so stored tables are rebuilt
BARE_VERSION = 1


def bare_species(mol, z):
    """The bare skeleton of ``mol`` (N_H = 0) at charge z: PAHdb's (``Molecule.bare``, at z or the nearest charge it
    computed), else the parent's ``stripped(0)`` with its Einstein A times BARE_IR_LAW."""
    if mol.bare:
        return mol.bare[min(mol.bare, key=lambda q: (abs(q - z), q))]
    st = mol.species[z].stripped(0)
    a, b = BARE_IR_LAW
    return replace(st, einstein_a=st.einstein_a * a * mol.nc ** b, source=f"{st.source} (bare, BARE_IR_LAW)")


def bare_channels(mol, params):
    """((name, carbon atoms lost, form, A [s^-1] or dS [J/K/mol], E0 [eV]), ...): the bare skeleton's destruction channels,
    ``baseline.CARBON_LOSS[params.carbon_loss]`` (Léger+89 Table 2 by default: C, C2 and C3 from graphite sublimation) and,
    for N_C >= BARE_FOLD_MIN_NC, folding into a cage (BARE_FOLD, no carbon lost)."""
    from pycalima.pahchem.baseline import CARBON_LOSS

    form, frags = CARBON_LOSS[params.carbon_loss]
    out = [(name, int(n), form, float(a), float(e0)) for name, (n, a, e0) in frags.items()]
    if mol.nc >= BARE_FOLD_MIN_NC:
        out.append(("fold", 0, "ilt", BARE_FOLD[0], BARE_FOLD[1]))
    return tuple(out)


def _bare_rates(freqs, E, counts, pick, beta, channels):
    """{name: k(E) on the loss grid} of ``bare_channels``: inverse-Laplace (Léger+89 eq. 8) or RRKM with the C-C stretch."""
    out = {}
    for name, _, form, a, e0 in channels:
        if form == "ilt":
            k = rates.inverse_laplace_arrhenius(E, counts, a, e0 * dos.CM_PER_EV, beta=beta)
        else:
            k = rates.rrkm(E, counts, rates.transition_state_for_entropy(freqs, 1600.0, a, n_scaled=5), e0 * dos.CM_PER_EV,
                           beta=beta)
        out[name] = k[pick]
    return out


def bare_loss_table(freqs, einstein_a, channels, nc, z, yield_, shape, sigma_scale, threshold_ev, ionising_heat=False,
                    electron_ev=None, uv_cross_section="li_draine", detach_ea=None):
    """{channel: rate [s^-1] on K_H_G0_GRID} of the bare skeleton (``freqs``, ``einstein_a``) at charge z under ``shape``:
    its ``bare_channels`` competing in its multiphoton cascade, heated as the state (its sigma, yield and deposits); the
    ceiling rises as ``_loss_tables``' does. Top-level, so it runs in worker processes."""
    field1 = _unit_field(shape)
    sigma = absorption_sigma(nc, z, E_PH, uv_cross_section, sigma_scale, detach_ea)
    e_dep = deposit_grid(field1, sigma, yield_)
    h = deposit_spectrum(field1, sigma, yield_, threshold_ev, ionising_heat, electron_ev, e_dep)
    if not np.any(h > 0.0):                          # no photon deposits (X-rays only): no photon-driven destruction
        return {c[0]: np.zeros(K_H_G0_GRID.size) for c in channels}
    f_ = np.asarray(freqs, dtype=float)
    e_max = _loss_ceiling(e_dep, h)
    while True:
        cm, E, counts, beta = _loss_grid(f_, einstein_a, e_max)
        pick = np.rint(cm.energy_cm / (E[1] - E[0])).astype(int)
        ch = _bare_rates(f_, E, counts, pick, beta, channels)
        out = {c: [] for c in ch}
        for g in K_H_G0_GRID:
            s = cm.steady_state(e_dep, g * h, ch)
            for c in ch:
                out[c].append(s.rates[c])
        if _loss_converged(s, e_max):
            return {c: np.array(v) for c, v in out.items()}
        e_max = min(e_max + LOSS_STEP_EV, LOSS_E_CAP_EV)


def _bare_inputs(mol, z, shape, de_cm, e_max_cm, params):
    """``bare_loss_table``'s inputs for state z of ``mol`` under ``shape``."""
    inp = _basis_inputs(mol, z, shape, de_cm, e_max_cm, params)
    sp = bare_species(mol, z)
    keep = ("nc", "z", "yield_", "shape", "sigma_scale", "threshold_ev", "ionising_heat", "electron_ev", "uv_cross_section",
            "detach_ea")
    return {k: inp[k] for k in keep if k in inp} | dict(freqs=np.asarray(sp.freqs_cm, dtype=float),
                                                        einstein_a=np.asarray(sp.einstein_a, dtype=float),
                                                        channels=bare_channels(mol, params))


def _bare_key(inp):
    h = hashlib.sha1(f"bare{BARE_VERSION}".encode())
    h.update(np.ascontiguousarray(E_PH).tobytes())
    h.update(np.ascontiguousarray(E_DEP).tobytes())
    h.update(np.ascontiguousarray(K_H_G0_GRID).tobytes())
    h.update(repr((LOSS_E_MAX_EV, LOSS_DE_CM, LOSS_OVERFLOW_TOL, LOSS_STEP_EV, LOSS_E_CAP_EV, photo.PARTITION_NODES,
                   photo.PHOTOELECTRON_PARTITION)).encode())
    for k in sorted(inp):
        h.update(k.encode())
        _feed(h, inp[k])
    return h.hexdigest()[:20]


def _build_bare(inp):
    return bare_loss_table(**inp)


def bare_event_yields(freqs, einstein_a, channels, e_max_ev):
    """(loss-grid energies [cm^-1], {channel: P(E)}): the single-event destruction yields of a bare skeleton after a deposit
    E (``CascadeModel.photon_yields``), for the deposits the tables do not hold (``Population._deposit_flows``)."""
    h = hashlib.sha1(b"bare-yields")
    for v in (np.asarray(freqs, dtype=float), np.asarray(einstein_a, dtype=float), tuple(channels), float(e_max_ev),
              LOSS_DE_CM):
        _feed(h, v)
    key = h.hexdigest()
    if key not in _YIELD_MEMO:
        f_ = np.asarray(freqs, dtype=float)
        cl, E, counts, beta = _loss_grid(f_, einstein_a, e_max_ev)
        pick = np.rint(cl.energy_cm / (E[1] - E[0])).astype(int)
        _YIELD_MEMO[key] = (cl.energy_cm, cl.photon_yields(_bare_rates(f_, E, counts, pick, beta, channels)))
    return _YIELD_MEMO[key]


# -- super-hydrogenated states (Params.hydrogen_chain = "joint", 2026-11-08) ---------------------------------------------
#: bump when ``superh_loss_table`` changes, so stored tables are rebuilt
SUPERH_VERSION = 1


def superh_e0(params, z):
    """E0 [eV] of the loss of an extra H at charge z: ``Params.h_superh_e0`` (Z <= 0, Z > 0), Andrews+16 as BaselineModel."""
    return float(params.h_superh_e0[1 if z > 0 else 0])


def _superh_rates(freqs, E, counts, pick, beta, levels, e0_ev, ds):
    """[k_j(E) on the loss grid], j = 1 ... ``levels``: the loss of one of the j extra H of state N_H0 + j. RRKM with E0
    ``e0_ev`` and dS(1000 K) ``ds``, its transition state built as the aromatic H loss's, path degeneracy j (as
    BaselineModel), on the parent's modes: the extra H's own (CH2 groups) are not computed."""
    k1 = rates.rrkm(E, counts, rates.transition_state_for_entropy(freqs, 3050.0, ds, n_scaled=5), e0_ev * dos.CM_PER_EV,
                    degeneracy=1, beta=beta)[pick]
    return [j * k1 for j in range(1, levels + 1)]


def superh_loss_table(freqs, einstein_a, nc, z, yield_, shape, sigma_scale, threshold_ev, levels, e0_ev, ds,
                      ionising_heat=False, electron_ev=None, uv_cross_section="li_draine", detach_ea=None):
    """k[j - 1, K_H_G0_GRID] [s^-1]: the photon-driven loss of an extra H of the super-hydrogenated states N_H0 + j,
    j = 1 ... ``levels``, of charge z under ``shape`` (``_superh_rates`` in the state's multiphoton cascade, heated as the
    state: its sigma, yield and deposits); the ceiling rises as ``_loss_tables``' does. Top-level, so it runs in worker
    processes."""
    field1 = _unit_field(shape)
    sigma = absorption_sigma(nc, z, E_PH, uv_cross_section, sigma_scale, detach_ea)
    e_dep = deposit_grid(field1, sigma, yield_)
    h = deposit_spectrum(field1, sigma, yield_, threshold_ev, ionising_heat, electron_ev, e_dep)
    out = np.zeros((levels, K_H_G0_GRID.size))
    if levels == 0 or not np.any(h > 0.0):           # no photon deposits (X-rays only): no photon-driven loss
        return out
    f_ = np.asarray(freqs, dtype=float)
    e_max = _loss_ceiling(e_dep, h)
    while True:
        cm, E, counts, beta = _loss_grid(f_, einstein_a, e_max)
        pick = np.rint(cm.energy_cm / (E[1] - E[0])).astype(int)
        done = True
        for j, k in enumerate(_superh_rates(f_, E, counts, pick, beta, levels, e0_ev, ds)):
            for g, gi in enumerate(K_H_G0_GRID):
                s = cm.steady_state(e_dep, gi * h, {"H": k})
                out[j, g] = s.rates["H"]
            done = done and _loss_converged(s, e_max)
        if done:
            return out
        e_max = min(e_max + LOSS_STEP_EV, LOSS_E_CAP_EV)


def _superh_inputs(mol, z, shape, de_cm, e_max_cm, params):
    """``superh_loss_table``'s inputs for state z of ``mol`` under ``shape``: the state's own (parent) modes."""
    inp = _basis_inputs(mol, z, shape, de_cm, e_max_cm, params)
    keep = ("nc", "z", "yield_", "shape", "sigma_scale", "threshold_ev", "ionising_heat", "electron_ev", "uv_cross_section",
            "detach_ea", "freqs", "einstein_a")
    return {k: inp[k] for k in keep if k in inp} | dict(levels=int(params.superh_levels), e0_ev=superh_e0(params, z),
                                                        ds=float(params.h_superh_ds))


def _superh_key(inp):
    h = hashlib.sha1(f"superh{SUPERH_VERSION}".encode())
    h.update(np.ascontiguousarray(E_PH).tobytes())
    h.update(np.ascontiguousarray(E_DEP).tobytes())
    h.update(np.ascontiguousarray(K_H_G0_GRID).tobytes())
    h.update(repr((LOSS_E_MAX_EV, LOSS_DE_CM, LOSS_OVERFLOW_TOL, LOSS_STEP_EV, LOSS_E_CAP_EV, photo.PARTITION_NODES,
                   photo.PHOTOELECTRON_PARTITION)).encode())
    for k in sorted(inp):
        h.update(k.encode())
        _feed(h, inp[k])
    return h.hexdigest()[:20]


def _build_superh(inp):
    return superh_loss_table(**inp)


def superh_event_yields(freqs, einstein_a, levels, e0_ev, ds, e_max_ev):
    """(loss-grid energies [cm^-1], [P_j(E)], j = 1 ... levels): the single-event yields of the loss of an extra H after a
    deposit E (``CascadeModel.photon_yields``), for the deposits the tables do not hold (``Population._deposit_flows``)."""
    h = hashlib.sha1(b"superh-yields")
    for v in (np.asarray(freqs, dtype=float), np.asarray(einstein_a, dtype=float), int(levels), float(e0_ev), float(ds),
              float(e_max_ev), LOSS_DE_CM):
        _feed(h, v)
    key = h.hexdigest()
    if key not in _YIELD_MEMO:
        f_ = np.asarray(freqs, dtype=float)
        cl, E, counts, beta = _loss_grid(f_, einstein_a, e_max_ev)
        pick = np.rint(cl.energy_cm / (E[1] - E[0])).astype(int)
        ks = _superh_rates(f_, E, counts, pick, beta, levels, e0_ev, ds)
        _YIELD_MEMO[key] = (cl.energy_cm, [cl.photon_yields({"H": k})["H"] for k in ks])
    return _YIELD_MEMO[key]


def _cascade_start(freqs, einstein_a, nc, z, yield_, shape, de_cm, e_max_cm, sigma_scale, recomb_ev, threshold_ev,
                   ionising_heat, electron_ev, uv_cross_section="li_draine", detach_ea=None):
    """(h(e_dep), k_up, recombination power, CascadeModel, start, e_dep) of one state at G0 = 1, e_dep its
    ``deposit_grid``; the model and start are None when nothing is left to emit (every absorption ionises or
    detaches, and no recombination heats). The cascade reaches ``e_max_cm``, or 0.5 eV above the state's hardest
    deposit and its recombination energy where those lie higher (highly charged states)."""
    field1 = _unit_field(shape)
    sigma = absorption_sigma(nc, z, E_PH, uv_cross_section, sigma_scale, detach_ea)
    e_dep = deposit_grid(field1, sigma, yield_)
    h = deposit_spectrum(field1, sigma, yield_, threshold_ev, ionising_heat, electron_ev, e_dep)
    k_up = float(photo.ionisation_rate(field1, sigma, yield_, E_PH))
    recomb_power = k_up * recomb_ev * dos.CM_PER_EV
    if not np.any(h > 0.0) and recomb_power <= 0.0:
        return h, k_up, recomb_power, None, None, e_dep
    need = (max(float(e_dep[-1]), recomb_ev) + 0.5) * dos.CM_PER_EV
    cm = energy_me.CascadeModel(freqs, einstein_a, e_max_cm if need <= e_max_cm else need, de_cm)
    # molecules placed on the cascade grid by the deposits (trapezoid over e_dep, split between the two
    # neighbouring bins) and the recombinations, then followed down once: the same photons as weighting
    # emission.cascade_photons by them, in O(modes x bins) instead of O(modes x bins^2)
    e = e_dep * dos.CM_PER_EV / cm.de_cm
    j = np.floor(e).astype(int)
    w = e - j
    start = np.zeros(cm.n)
    hw = h * _trapezoid_weights(e_dep)
    np.add.at(start, j, (1.0 - w) * hw)
    np.add.at(start, j + 1, w * hw)
    if recomb_power > 0.0:
        x = recomb_ev * dos.CM_PER_EV / cm.de_cm
        i0 = int(np.floor(x))
        start[i0] += k_up * (1.0 - (x - i0))
        start[i0 + 1] += k_up * (x - i0)
    return h, k_up, recomb_power, cm, start, e_dep


def emission_temperature_shares(freqs, einstein_a, nc, z, yield_, shape, de_cm, e_max_cm, sigma_scale, recomb_ev=0.0,
                                threshold_ev=0.0, ionising_heat=False, electron_ev=None, uv_cross_section="li_draine",
                                detach_ea=None, **_):
    """W[class, T bin]: how each mode class's IR power is spread over emission temperature (``band_conventions``)
    for the state of these ``_basis_inputs`` (N_H0; the dehydrogenated states take the same shares). The cascade
    of ``build_state_basis``, followed per bin; zeros when the state does not emit."""
    from pycalima.pahchem import band_conventions as bc

    _, _, _, cm, start, _ = _cascade_start(np.asarray(freqs, dtype=float), einstein_a, nc, z, yield_, shape, de_cm, e_max_cm,
                                           sigma_scale, recomb_ev, threshold_ev, ionising_heat, electron_ev, uv_cross_section,
                                           detach_ea)
    if cm is None:
        return np.zeros((bc.N_CLASSES, bc.T_MID.size))
    X = emission.emission_by_energy(cm, start)
    return bc.class_temperature_shares(freqs, X, bc.bin_temperatures(freqs, cm.energy_cm))


def build_state_basis(freqs, einstein_a, nc, nh, z, yield_, shape, de_cm, e_max_cm, h_even, sigma_scale, recomb_ev=0.0,
                      ladder=(), h_odd=(3.16, 10.0), h2=(4.58, 40.0), threshold_ev=0.0, ionising_heat=False,
                      electron_ev=None, uv_cross_section="li_draine", extras=(), detach_ea=None):
    """``StateBasis`` of one charge state (module docstring); top-level, so it runs in worker processes.

    ``recomb_ev`` is the energy each recombination (attachment) into this state
    deposits; the heating rate is this state's own k_up (steady state).
    """
    h, k_up, recomb_power, cm, start, e_dep = _cascade_start(freqs, einstein_a, nc, z, yield_, shape, de_cm, e_max_cm,
                                                             sigma_scale, recomb_ev, threshold_ev, ionising_heat,
                                                             electron_ev, uv_cross_section, detach_ea)
    if cm is None:                                     # every absorption ionises or detaches: nothing to emit
        return StateBasis(np.asarray(freqs, dtype=float), np.zeros(len(freqs)), k_up,
                          np.zeros(K_H_G0_GRID.size), 0.0)
    emitted = freqs * emission.emission_by_energy(cm, start).sum(axis=1)
    # the dehydrogenated states of the ladder, heated by the same deposits (sigma and yield do not depend on N_H
    # here), each on its own modes (``PAHSpecies.stripped``), on the same cascade grid
    l_freqs, l_emitted = [], []
    for f_, a_, _, _ in ladder[1:]:
        cm_k = energy_me.CascadeModel(f_, a_, (cm.n - 1) * cm.de_cm, de_cm)
        l_freqs.append(np.asarray(f_, dtype=float))
        l_emitted.append(l_freqs[-1] * emission.emission_by_energy(cm_k, start).sum(axis=1))
    if not np.any(h > 0.0):           # no photon deposits (X-rays only: every absorption ionises): no photon-driven loss
        return StateBasis(np.asarray(freqs, dtype=float), emitted, k_up, np.zeros(K_H_G0_GRID.size), 0.0,
                          float(recomb_power), np.zeros((4, K_H_G0_GRID.size)), np.zeros((4, K_H_G0_GRID.size)),
                          tuple(l_freqs), tuple(l_emitted))
    # the state's loss grid serves its k_h and the first ladder step (the same modes); its ceiling rises until the
    # multiphoton cascade converges (LOSS_OVERFLOW_TOL), and the ladder starts from it
    e_loss = _loss_ceiling(e_dep, h)
    while True:
        cl, E, counts, beta = _loss_grid(freqs, einstein_a, e_loss)
        pick = np.rint(cl.energy_cm / (E[1] - E[0])).astype(int)
        # k_h (``intact_fractions(method="single")``) takes the even-H values with degeneracy N_H for any N_H;
        # ladder[0] the odd-H ones for odd N_H, so it shares this rate only for even N_H
        k = rates.rrkm(E, counts, rates.transition_state_for_entropy(freqs, 3050.0, h_even[1], n_scaled=5),
                       h_even[0] * dos.CM_PER_EV, degeneracy=nh, beta=beta)[pick]
        states = [cl.steady_state(e_dep, g * h, {"H": k}) for g in K_H_G0_GRID]
        if _loss_converged(states[-1], e_loss):
            break
        e_loss = min(e_loss + LOSS_STEP_EV, LOSS_E_CAP_EV)
    k_h = np.array([s.rates["H"] for s in states])
    lh, lh2 = np.zeros((4, K_H_G0_GRID.size)), np.zeros((4, K_H_G0_GRID.size))
    lx = {"C2H2": np.zeros((4, K_H_G0_GRID.size)), "Hp": np.zeros((4, K_H_G0_GRID.size))}
    for i, (f_, a_, n_, ok_) in enumerate(ladder):
        grid = None
        if i == 0 and n_ == nh and np.array_equal(f_, freqs) and np.array_equal(a_, einstein_a):
            grid = ((cl, E, counts, beta),
                    k if nh % 2 == 0 else _h_loss_rate(freqs, E, counts, pick, nh, h_even, h_odd, beta))
        lh[i], lh2[i], kx = _loss_tables(f_, a_, n_, ok_, h, h_even, h_odd, h2, grid, e_dep, e_loss,
                                         extras[i] if extras else ())
        for c, v in kx.items():
            lx[c][i] = v
    out = StateBasis(np.asarray(freqs, dtype=float), emitted, k_up, k_h,
                     float(np.trapezoid(h * e_dep * dos.CM_PER_EV, e_dep)), float(recomb_power), lh, lh2,
                     tuple(l_freqs), tuple(l_emitted), lx["C2H2"], lx["Hp"])
    if not (np.isfinite(out.emitted).all() and np.isfinite(out.k_h).all()
            and all(np.isfinite(e).all() for e in out.ladder_emitted)
            and np.isfinite([out.k_up, out.absorbed]).all()):
        raise ValueError(f"basis of N_C {nc}, Z {z} is not finite")
    return out


#: how far a quantity bounded by construction may stray past its bound by round-off before it is an error (``_round_off``)
ROUND_OFF = 1e-12


def _round_off(x, lo, hi, what):
    """``x`` clipped to [lo, hi] (None: open) after checking it strays past them by round-off only (ROUND_OFF, relative to
    the bound's scale), so that a clip never hides an error. The recursions behind the probabilities and rates it guards
    sum the same terms in different orders: 1-2 ulp past their bounds (audit of 2026-11-03)."""
    x = np.asarray(x, dtype=float)
    scale = max(abs(lo) if lo is not None else 0.0, abs(hi) if hi is not None else 0.0, 1.0)
    if (not np.all(np.isfinite(x)) or (lo is not None and np.any(x < lo - ROUND_OFF * scale))
            or (hi is not None and np.any(x > hi + ROUND_OFF * scale))):
        raise ValueError(f"{what} outside [{lo}, {hi}] by more than round-off: {x.min():.17g} .. {x.max():.17g}")
    return np.clip(x, lo, hi)


def _emission_many(model, starts):
    """Photons per IR mode emitted down the cascade of ``model`` from each row of ``starts`` (molecules over its bins):
    ``emission.emission_by_energy`` summed over the bins, for several starts in one pass."""
    A = np.array(starts, dtype=float)
    if energy_me.HAVE_NUMBA:
        return energy_me._cascade_down_numba(model.k_modes, model._lo, model._hi, model._w, A, model.n_ir, False)[:, :, 0]
    X = np.zeros((A.shape[0], model.n_ir))
    for j in range(model.n - 1, 0, -1):
        a = A[:, j]
        if not np.any(a > 0.0):
            continue
        km = model.k_modes[:, j]
        lo, hi, w = model._lo[:, j], model._hi[:, j], model._w[:, j]
        to_self = km * ((1.0 - w) * (lo == j) + w * (hi == j))
        denom = km.sum() - to_self.sum()
        if denom <= 0.0:
            continue                                   # cannot emit: the cascade ends here
        X += np.outer(a, km[: model.n_ir] / denom)
        np.add.at(A.T, lo, np.outer(np.where(lo < j, (1.0 - w) * km, 0.0) / denom, a))
        np.add.at(A.T, hi, np.outer(np.where(hi < j, w * km, 0.0) / denom, a))
    return X


def deposit_response(ladder, de_cm, h_even, h_odd, h2, extras, energies_ev, weights=None):
    """What one deposit of each of ``energies_ev`` does to a state, per molecule placed there: for each member of its
    hydrogenation ladder (``_basis_inputs``' ``ladder``: N_H0, N_H0-1, ..., with ``extras``' channels), (IR power per mode
    [cm^-1 per deposit], P(H loss), P(H2 loss), P(C2H2 loss), P(proton emission)), as [[(emitted, p_h, p_h2, p_c2h2, p_hp)
    per member] per energy]. With ``weights`` (rows x energies, each
    row summing to 1) one result per row instead: the mean over that row's deposits, exact, the cascade and the single-
    event yields being linear in the starts (one event of a spread of deposits, ``ionisation_products``).

    The emission follows the cascade of ``build_state_basis`` (``_emission_many``, sinks not taken out), every deposit
    split between its two neighbouring bins; the losses are the single-event yields of the loss cascade
    (``CascadeModel.photon_yields``, the same RRKM rates as the basis's ``_loss_tables``): field-independent, for
    deposits the bases do not hold (``Population._deposit_flows``)."""
    energies_ev = np.asarray(energies_ev, dtype=float)
    W = None if weights is None else np.asarray(weights, dtype=float)
    if W is not None:
        _round_off(W.sum(axis=1), 1.0, 1.0, "deposit weights per row")
    out = [[] for _ in range(energies_ev.size if W is None else W.shape[0])]
    top = float(energies_ev.max())
    for m, (f_, a_, n_, h2_ok) in enumerate(ladder):
        f_ = np.asarray(f_, dtype=float)
        cm = energy_me.CascadeModel(f_, a_, (top + 0.5) * dos.CM_PER_EV, de_cm)   # every deposit's upper bin inside
        x = energies_ev * dos.CM_PER_EV / cm.de_cm
        i0 = np.floor(x).astype(int)
        starts = np.zeros((energies_ev.size, cm.n))
        starts[np.arange(energies_ev.size), i0] = 1.0 - (x - i0)
        starts[np.arange(energies_ev.size), i0 + 1] += x - i0
        emitted = f_[None, :] * _emission_many(cm, starts if W is None else W @ starts)
        e_cl, Y = _single_event_yields(f_, a_, n_, h2_ok, h_even, h_odd, h2, max(LOSS_E_MAX_EV, top + 5.0),
                                       extras[m] if extras else ())
        p = {c: _round_off(np.interp(energies_ev * dos.CM_PER_EV, e_cl, y), 0.0, 1.0, f"P({c}) per deposit")
             for c, y in Y.items()}
        if W is not None:
            p = {c: _round_off(W @ v, 0.0, 1.0, f"P({c}) per event") for c, v in p.items()}
        get = lambda c, i: float(p[c][i]) if c in p else 0.0              # noqa: E731
        for i in range(len(out)):
            out[i].append((emitted[i], get("H", i), get("H2", i), get("C2H2", i), get("Hp", i)))
    return out


#: ``_single_event_yields`` of this process, keyed by its inputs: they depend on the ladder member alone, not on the field,
#: so every state sharing its modes (stand-ins) and every response set and field shape reuse them (0.2-0.4 MB each,
#: pahchem audit m1)
_YIELD_MEMO = {}


def _single_event_yields(f_, a_, n_, h2_ok, h_even, h_odd, h2, e_max_ev, extra=()):
    """(loss-grid energies [cm^-1], {"H": P(E), "H2": P(E), and those of ``extra``}): the single-event loss yields of one
    ladder member (``CascadeModel.photon_yields`` with the RRKM rates of the basis's ``_loss_tables``) on its loss grid to
    ``e_max_ev``."""
    h = hashlib.sha1()
    for v in (f_, np.asarray(a_, dtype=float), n_, h2_ok, tuple(h_even), tuple(h_odd), tuple(h2), float(e_max_ev),
              tuple(extra), LOSS_DE_CM):
        _feed(h, v)
    key = h.hexdigest()
    if key not in _YIELD_MEMO:
        cl, E, counts, beta = _loss_grid(f_, a_, e_max_ev)
        pick = np.rint(cl.energy_cm / (E[1] - E[0])).astype(int)
        ch = {"H": _h_loss_rate(f_, E, counts, pick, n_, h_even, h_odd, beta)}
        if h2_ok:
            ch["H2"] = rates.rrkm(E, counts, rates.transition_state_for_entropy(f_, 3050.0, h2[1], n_scaled=5),
                                  h2[0] * dos.CM_PER_EV, degeneracy=2, beta=beta)[pick]
        ch |= _extra_rates(f_, E, counts, pick, beta, extra)
        _YIELD_MEMO[key] = (cl.energy_cm, cl.photon_yields(ch))
    return _YIELD_MEMO[key]


#: bump when deposit_response changes, so stored responses are rebuilt (4: the channels of ``loss_extras``)
RESPONSE_VERSION = 5
#: kinetic energies [eV] above IP at which a capture's deposit is held ("rec"): the captured electron brings its own
#: (``capture_energy``, ~kT-2kT), split between the two nodes about it keeping its mean (pahchem audit M2; until
#: 2026-11-05 a capture deposited the IP alone, while the recombination cooling took the kinetic energy from the gas)
REC_KE_NODES_EV = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
#: the field-independent sets of responses a state may need: "rec", one capture into it (its threshold plus the
#: REC_KE_NODES_EV of the captured electron; none for the top
#: state); "aug", a C 1s product, K_AUGER_EXCITATION_EV plus each of HEAT_NODES_EV (only in fields that reach the edge);
#: "heat", a cosmic-ray passage's product, each of HEAT_NODES_EV (``Params.cosmic_rays``).
#: A third, ("ion", shape), holds the photoionisations landing on it in that field shape (``ionisation_products``)
RESPONSE_SETS = ("rec", "aug")


def _response_inputs(mol, z, de_cm, e_max_cm, params, which):
    """(ladder, de_cm, h_even, h_odd, h2, extras, energies[, weights]) of ``deposit_response`` for set ``which`` of state z."""
    inp = _basis_inputs(mol, z, ("draine",), de_cm, e_max_cm, params)
    head = (inp["ladder"], de_cm, tuple(params.h_even), tuple(params.h_odd), tuple(params.h2), inp["extras"])
    if which == "rec":
        ip = float(state_threshold(mol, z)) if z < max(mol.charges) else 0.0
        return head + (tuple(ip + REC_KE_NODES_EV),)
    if which == "aug":
        return head + (tuple(K_AUGER_EXCITATION_EV + HEAT_NODES_EV),)
    if which == "heat":
        return head + (tuple(HEAT_NODES_EV),)
    rows = ionisation_products(mol, z, params, which[1])
    energies = np.concatenate([e for _, _, e, _ in rows])
    W = np.zeros((len(rows), energies.size))
    at = np.cumsum([0] + [e.size for _, _, e, _ in rows])
    for r, (_, _, _, w) in enumerate(rows):
        W[r, at[r]:at[r + 1]] = w
    return head + (energies, W)


def _response_key(inp):
    h = hashlib.sha1(f"r{RESPONSE_VERSION}".encode())
    for v in (LOSS_E_MAX_EV, LOSS_DE_CM, *inp):
        _feed(h, v)
    return h.hexdigest()[:20]


def _build_response(inp):
    return deposit_response(*inp)


def _build(inp):
    return build_state_basis(**inp)


#: ``StateBasis`` of this process by ``_basis_key``, the last BASIS_MEMO of them: populations without a ``cache_dir`` and
#: with the same basis inputs (differing in capture_scale, kshell, double_ionisation, ...) build each once (0.02-0.05 MB
#: each); with one, the disk cache does this
_BASIS_MEMO, BASIS_MEMO = {}, 4096


def _remember_basis(key, b):
    if len(_BASIS_MEMO) >= BASIS_MEMO:
        del _BASIS_MEMO[next(iter(_BASIS_MEMO))]
    _BASIS_MEMO[key] = b


def _build_shares(inp):
    return emission_temperature_shares(**inp)


class Population:
    """Band fluxes, charge and intact fractions of a library at an environment (module docstring).

    ``windows_um`` is {band: (lo, hi)} in um, the windows of the observations;
    the bands in ``local`` take off a linear baseline across their window, as
    the observations did (``_band_matrix``). ``cache_dir`` keeps the state
    bases on disk (None: memory only). ``band_convention`` (``band_conventions``; a name or a
    ``BandConvention``) sets the band positions and profiles of the band fluxes and spectra; "legacy", the default,
    is the Lorentzian 15 / -15 cm^-1 of ``_band_matrix``.
    """

    def __init__(self, library, windows_um, local=(), de_cm=100.0, e_max_ev=21.5, params=Params(), cache_dir=None,
                 dehydrogenated_emission=True, band_convention="legacy"):
        from pycalima.pahchem import band_conventions

        self.band_convention = band_conventions.get(band_convention)
        given = list(library)
        if params.charge_ladder not in ("cold_limit", "library"):
            raise ValueError(f"charge_ladder must be 'cold_limit' or 'library', not {params.charge_ladder!r}")
        if params.hydrogen_chain not in ("charge_averaged", "joint"):
            raise ValueError(f"hydrogen_chain must be 'charge_averaged' or 'joint', not {params.hydrogen_chain!r}")
        if params.anion_h_addition not in ("as_neutral", "none"):
            raise ValueError(f"anion_h_addition must be 'as_neutral' or 'none', not {params.anion_h_addition!r}")
        if not (isinstance(params.superh_levels, int) and params.superh_levels >= 0):
            raise ValueError(f"superh_levels must be an integer >= 0, not {params.superh_levels!r}")
        if params.ionisation_heating and params.ionising_photons_heat:
            raise ValueError("ionisation_heating (the product's cascade) and ionising_photons_heat (the absorbing state's) "
                             "would count the energy an ionisation leaves twice: choose one")
        # the ladder to the cold limit on copies (``extend_ladder``); the caller's molecules map to them (``_own``)
        self.library = [extend_ladder(m) for m in given] if params.charge_ladder == "cold_limit" else given
        self.windows, self.p = dict(windows_um), params
        # True: N_H0 ... N_H0-3 emit in proportion to the hydrogenation chain (``emission_weights``); False: the
        # intact molecule alone, as before 2026-10-25 (its callers then filter on ``intact_fractions``)
        self.dehydrogenated_emission = bool(dehydrogenated_emission)
        self.local = tuple(local)
        self.bands = tuple(self.windows)
        self.de_cm, self.e_max_cm = float(de_cm), e_max_ev * dos.CM_PER_EV
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        self.zmin = min(min(m.charges) for m in self.library)
        self.zs = tuple(range(self.zmin, max(max(m.charges) for m in self.library) + 1))
        self.present = np.array([[z in m.species for z in self.zs] for m in self.library])
        self.nc = np.array([m.nc for m in self.library], dtype=float)
        self._index = {id(m): s for s, m in enumerate(given)} | {id(m): s for s, m in enumerate(self.library)}
        self._arrays = {}
        self._capture = {}
        self._chains = {}                # environment -> ``_chain_distributions``, the last CHAIN_MEMO of them
        self._charges = {}               # environment -> ``charge_matrix``, the last CHAIN_MEMO of them
        self._ion_ups = {}               # environment -> ``_ion_up``, the last CHAIN_MEMO of them
        self._ion_heats = {}             # environment -> ``_ion_heat``, the last CHAIN_MEMO of them
        self._terms = {}                 # environment -> ``_flow_terms``, the last FLOW_MEMO of them
        self._kernels = {}               # environment -> ``_loss_kernel``, the last FLOW_MEMO of them
        self._resp = {}                  # (id(mol), z) -> ``_response``
        self._joints = {}                # environment -> ``_joint_distributions``, the last CHAIN_MEMO of them

    def _own(self, mol):
        """The population's copy of ``mol`` (its ladder extended, ``extend_ladder``), or ``mol`` if not in it."""
        s = self._index.get(id(mol))
        return mol if s is None else self.library[s]

    def _live(self, shape):
        """live[s, slot]: the states that can hold molecules under ``shape``. Above +1 only photoionisation raises the
        charge (ions and associative detachment act on -1 and 0), so a state is live if the one below is and ionises in
        this field shape (k_up > 0); the others have f = 0 at every G0 and get no basis. A ladder to the cold limit then
        costs only the states the field reaches (in a field cut at 13.6 eV, those whose threshold from below is under
        13.6 eV, as WD01 eq. 22)."""
        key = ("live", shape)
        if key not in self._arrays and (self.p.cosmic_rays or self.p.hot_gas):
            self._arrays[key] = self.present.copy()        # passages reach every charge (+1 ... +CR_DZ_MAX steps)
        if key not in self._arrays:
            field1 = _unit_field(shape)
            live = self.present.copy()
            for s, mol in enumerate(self.library):
                for z in mol.charges:
                    if z < 2:
                        continue
                    i = z - self.zmin
                    if live[s, i - 1]:
                        sigma = photo.sigma_uv(mol.nc, z - 1, E_PH, self.p.uv_cross_section) * self.p.sigma_scale
                        live[s, i] = photo.ionisation_rate(field1, sigma, _yield(mol, z - 1, self.p), E_PH) > 0.0
                    else:
                        live[s, i] = False
            self._arrays[key] = live
        return self._arrays[key]

    def _live_charges(self, mol, shape):
        """The live charges of ``mol`` (one of the population's) under ``shape`` (``_live``)."""
        row = self._live(shape)[self._index[id(mol)]]
        return tuple(z for z in mol.charges if row[z - self.zmin])

    # -- bases -----------------------------------------------------------------------------------
    def _ckey(self, shape, z, mol=None):
        """Memory-cache key: everything a basis can depend on (the disk cache keys on the inputs themselves)."""
        th = tuple(sorted(mol.ionisation_ev.items())) if mol is not None else ()
        return ("basis", shape, z, self.de_cm, self.e_max_cm, self.p, th)

    def _load(self, key):
        f = None if self.cache_dir is None else self.cache_dir / f"{key}.npz"
        if f is None or not f.exists():
            return None
        d = np.load(f)
        n_l = sum(1 for k in range(1, 4) if f"ladder_freqs_{k}" in d)
        return StateBasis(d["freqs_cm"], d["emitted"], float(d["k_up"]), d["k_h"], float(d["absorbed"]),
                          float(d["recombination"]) if "recombination" in d else 0.0,
                          d["ladder_h"] if "ladder_h" in d else None, d["ladder_h2"] if "ladder_h2" in d else None,
                          tuple(d[f"ladder_freqs_{k}"] for k in range(1, n_l + 1)),
                          tuple(d[f"ladder_emitted_{k}"] for k in range(1, n_l + 1)),
                          d["ladder_c2h2"] if "ladder_c2h2" in d else None, d["ladder_hp"] if "ladder_hp" in d else None)

    def _stored(self, inp):
        """(key, stored basis or None) of ``inp``: on disk, or without ``cache_dir`` this process's (``_BASIS_MEMO``); a
        basis stored under its legacy key gets a link to the new."""
        key = _basis_key(inp)
        if self.cache_dir is None:
            return key, _BASIS_MEMO.get(key)
        b = self._load(key)
        if b is None and self.cache_dir is not None:
            old = self.cache_dir / f"{_basis_key(inp, legacy=True)}.npz"
            if old.exists():
                b = self._load(old.stem)
                try:
                    os.link(old, self.cache_dir / f"{key}.npz")
                except FileExistsError:          # another process linked it first
                    pass
        return key, b

    def _save(self, key, b):
        if self.cache_dir is None:
            _remember_basis(key, b)
        else:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            # written under a temporary name and renamed, so a concurrent reader never sees half a file
            tmp = self.cache_dir / f"{key}.{os.getpid()}.tmp.npz"
            np.savez(tmp, freqs_cm=b.freqs_cm, emitted=b.emitted, k_up=b.k_up,
                     k_h=b.k_h, absorbed=b.absorbed, recombination=b.recombination,
                     ladder_h=b.ladder_h if b.ladder_h is not None else np.zeros((4, K_H_G0_GRID.size)),
                     ladder_h2=b.ladder_h2 if b.ladder_h2 is not None else np.zeros((4, K_H_G0_GRID.size)),
                     ladder_c2h2=b.ladder_c2h2 if b.ladder_c2h2 is not None else np.zeros((4, K_H_G0_GRID.size)),
                     ladder_hp=b.ladder_hp if b.ladder_hp is not None else np.zeros((4, K_H_G0_GRID.size)),
                     **{f"ladder_freqs_{k + 1}": f_ for k, f_ in enumerate(b.ladder_freqs)},
                     **{f"ladder_emitted_{k + 1}": e for k, e in enumerate(b.ladder_emitted)})
            os.replace(tmp, self.cache_dir / f"{key}.npz")

    def prepare(self, shape, workers=None):
        """Give every live state (``_live``) its basis under ``shape``: from memory, the disk cache, or built in parallel."""
        todo = []
        for mol in self.library:
            for z in self._live_charges(mol, shape):
                ck = self._ckey(shape, z, mol)
                if ck in mol._cache:
                    continue
                inp = _basis_inputs(mol, z, shape, self.de_cm, self.e_max_cm, self.p)
                key, b = self._stored(inp)
                if b is None:
                    todo.append((mol, ck, key, inp))
                else:
                    mol._cache[ck] = b
        if not todo:
            return
        workers = default_workers(workers)

        def keep(t, b):
            mol, ck, key, _ = t
            mol._cache[ck] = b
            self._save(key, b)

        if workers > 1 and len(todo) > 4:
            from concurrent.futures import ProcessPoolExecutor, as_completed

            # each basis is stored as it arrives, so a failing state does not discard the others
            failed = None
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futures = {ex.submit(_build, t[3]): t for t in todo}
                for fut in as_completed(futures):
                    try:
                        keep(futures[fut], fut.result())
                    except Exception as exc:      # noqa: BLE001 - re-raised once the rest are stored
                        failed = failed or exc
            if failed is not None:
                raise failed
        else:
            for t in todo:
                keep(t, _build(t[3]))

    def basis(self, mol, z, shape):
        """``StateBasis`` of ``mol`` (the population's copy, ``_own``) at charge z under ``shape``."""
        mol = self._own(mol)
        ck = self._ckey(shape, z, mol)
        if ck not in mol._cache:
            inp = _basis_inputs(mol, z, shape, self.de_cm, self.e_max_cm, self.p)
            key, b = self._stored(inp)
            if b is None:
                b = _build(inp)
                self._save(key, b)
            mol._cache[ck] = b
        return mol._cache[ck]

    # -- band conventions --------------------------------------------------------------------------
    def _shares_file(self, inp):
        from pycalima.pahchem import band_conventions as bc

        if self.cache_dir is None:
            return None
        return self.cache_dir / "tshares" / f"{_basis_key(inp)}-t{TSHARES_VERSION}-{bc.T_EDGES.size}.npy"

    def temperature_shares(self, mol, z, shape):
        """W[class, T bin] of ``emission_temperature_shares`` for ``mol`` at charge z (memory and disk cached)."""
        mol = self._own(mol)
        ck = ("tshares",) + self._ckey(shape, z, mol)
        if ck not in mol._cache:
            inp = _basis_inputs(mol, z, shape, self.de_cm, self.e_max_cm, self.p)
            f = self._shares_file(inp)
            if f is not None and f.exists():
                W = np.load(f)
            else:
                W = emission_temperature_shares(**inp)
                self._save_shares(f, W)
            mol._cache[ck] = W
        return mol._cache[ck]

    @staticmethod
    def _save_shares(f, W):
        if f is not None:
            f.parent.mkdir(parents=True, exist_ok=True)
            tmp = f.with_name(f"{f.stem}.{os.getpid()}.tmp.npy")
            np.save(tmp, W)
            os.replace(tmp, f)

    def prepare_temperature_shares(self, shape, workers=None):
        """Build the ``temperature_shares`` of every state under ``shape`` (only "tdep" conventions use them),
        in parallel like ``prepare``."""
        todo = []
        for mol in self.library:
            for z in self._live_charges(mol, shape):
                ck = ("tshares",) + self._ckey(shape, z, mol)
                if ck in mol._cache:
                    continue
                inp = _basis_inputs(mol, z, shape, self.de_cm, self.e_max_cm, self.p)
                f = self._shares_file(inp)
                if f is not None and f.exists():
                    mol._cache[ck] = np.load(f)
                else:
                    todo.append((mol, ck, f, inp))
        workers = default_workers(workers)
        if workers > 1 and len(todo) > 4:
            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor(max_workers=workers) as ex:
                for (mol, ck, f, _), W in zip(todo, ex.map(_build_shares, [t[3] for t in todo])):
                    mol._cache[ck] = W
                    self._save_shares(f, W)
        else:
            for mol, ck, f, inp in todo:
                mol._cache[ck] = W = emission_temperature_shares(**inp)
                self._save_shares(f, W)

    def _window_matrix(self, mol, z, shape, freqs):
        """M[i, band] of the population's band convention for modes ``freqs`` of ``mol`` at charge z."""
        from pycalima.pahchem import band_conventions as bc

        conv = self.band_convention
        if conv.kind == "legacy":
            return _band_matrix(freqs, self.windows, self.local)
        if conv.kind == "constant":
            sh, fw, et = conv.constant_lines(freqs, mol.nc, z)
            return bc.window_matrix(freqs, self.windows, self.local, sh, fw, et)
        return bc.tdep_window_matrix(conv, freqs, mol.nc, z, self.windows, self.local,
                                     self.temperature_shares(mol, z, shape))

    def _line_spectrum(self, mol, z, shape, freqs, emitted, nu, fwhm_cm, shift_cm):
        """IR power per cm^-1 on ``nu``: fixed Lorentzians (``fwhm_cm``, ``shift_cm``) if given, else the convention."""
        from pycalima.pahchem import band_conventions as bc

        conv = self.band_convention
        if fwhm_cm is not None or shift_cm is not None or conv.kind == "legacy":
            return emission.spectrum(freqs, emitted, nu, 15.0 if fwhm_cm is None else fwhm_cm,
                                     -15.0 if shift_cm is None else shift_cm)
        if conv.kind == "constant":
            sh, fw, et = conv.constant_lines(freqs, mol.nc, z)
            return bc.line_spectrum(freqs, emitted, nu, sh, fw, et)
        return bc.tdep_spectrum(conv, freqs, emitted, mol.nc, z, nu, self.temperature_shares(mol, z, shape))

    def _stack(self, shape):
        """(k_up at G0 = 1, log k_H on K_H_G0_GRID, band power B[s, slot, band] at G0 = 1); 0 where absent."""
        key = (shape, tuple(self.windows.items()), self.local, self.band_convention)
        if key not in self._arrays:
            self.prepare(shape)
            n, m = len(self.library), len(self.zs)
            k_up, B = np.zeros((n, m)), np.zeros((n, m, len(self.bands)))
            lkh = np.full((n, m, K_H_G0_GRID.size), np.log(1e-300))
            for s, mol in enumerate(self.library):
                for z in self._live_charges(mol, shape):
                    b = self.basis(mol, z, shape)
                    i = z - self.zmin
                    k_up[s, i] = b.k_up
                    lkh[s, i] = np.log(np.maximum(b.k_h, 1e-300))
                    B[s, i] = b.emitted @ self._window_matrix(mol, z, shape, b.freqs_cm)
            self._arrays[key] = (k_up, lkh, B)
        return self._arrays[key]

    def _k_up(self, shape):
        """k_up[s, slot] at G0 = 1 [s^-1]: each live state's photoionisation (detachment) rate, the one its basis holds
        (``_cascade_start``: the same sigma, yield and field), without building the basis: the charge chain alone needs no
        more (with ``Params.fission`` its proton emission needs the loss tables, and so the bases)."""
        key = ("k_up", shape)
        if key not in self._arrays:
            field1 = _unit_field(shape)
            k = np.zeros((len(self.library), len(self.zs)))
            for s, mol in enumerate(self.library):
                for z in self._live_charges(mol, shape):
                    sigma = state_sigma(mol, z, self.p)
                    k[s, z - self.zmin] = photo.ionisation_rate(field1, sigma, _yield(mol, z, self.p), E_PH)
            self._arrays[key] = k
        return self._arrays[key]

    def _double_stack(self, shape):
        """k2[s, slot]: the part of k_up (G0 = 1) that removes two electrons (``Params.double_ionisation``), the same
        sigma, field and yield times ``photo.double_fraction`` above ``double_threshold``; 0 for anions, the top state
        (every ionisation out of it is an explosion) and with "none"."""
        key = ("double", shape)
        if key not in self._arrays:
            if self.p.double_ionisation not in ("none", "knockout"):
                raise ValueError(f"double_ionisation must be 'none' or 'knockout', not {self.p.double_ionisation!r}")
            if self.p.double_ionisation != "none" and self.p.ionising_photons_heat:
                raise ValueError("double_ionisation is not combined with ionising_photons_heat: the bases' deposits take one electron")
            k2 = np.zeros((len(self.library), len(self.zs)))
            if self.p.double_ionisation == "knockout":
                field1 = _unit_field(shape)
                for s, mol in enumerate(self.library):
                    for z in self._live_charges(mol, shape):
                        if 0 <= z < max(mol.charges):
                            sigma = self._valence_sigma(mol, z, shape)
                            y2 = _yield(mol, z, self.p) * photo.double_fraction(E_PH, double_threshold(mol, z))
                            k2[s, z - self.zmin] = photo.ionisation_rate(field1, sigma, y2, E_PH)
            self._arrays[key] = k2
        return self._arrays[key]

    def _kshell_on(self, shape):
        """True if C 1s absorption follows its own branching (``Params.kshell`` "auger") and the field reaches the edge."""
        return kshell_applies(self.p, shape)

    def _valence_sigma(self, mol, z, shape):
        """The state's cross section on E_PH less its C 1s part when that has its own branching (``_kshell_on``)."""
        return valence_sigma(mol, z, self.p, shape)

    def _kshell_stack(self, shape):
        """``kshell_rates`` of every live state at G0 = 1: {"k": [J, s, slot] (dz = 1 ... J), "k_node": {(s, slot):
        [J, node]}, "over", "power", "absorbed_hi": [s, slot]}; None unless ``_kshell_on``."""
        key = ("kshell", shape)
        if key not in self._arrays:
            out = None
            if self._kshell_on(shape):
                self.prepare_escape(shape)
                field1 = _unit_field(shape)
                n, m = len(self.library), len(self.zs)
                rows = {}
                over, power, hi = np.zeros((n, m)), np.zeros((n, m)), np.zeros((n, m))
                for s, mol in enumerate(self.library):
                    for z in self._live_charges(mol, shape):
                        i = z - self.zmin
                        k, k_node, over[s, i], power[s, i], hi[s, i] = kshell_rates(mol, z, self.p, field1)
                        rows[(s, i)] = (k, k_node)
                J = max(r[0].size for r in rows.values())
                K = np.zeros((J, n, m))
                for (s, i), (k, _) in rows.items():
                    K[: k.size, s, i] = k
                out = {"k": K, "k_node": {si: r[1] for si, r in rows.items()}, "over": over, "power": power,
                       "absorbed_hi": hi}
            self._arrays[key] = out
        return self._arrays[key]

    def _escape_file(self, key):
        return None if self.cache_dir is None else self.cache_dir / "escape" / f"{key}.npz"

    def prepare_escape(self, shape, workers=None):
        """``escape_table`` of the C 1s products of every live state under ``shape`` (z + 2, at most the top:
        ``kshell_rates``): from memory, the disk cache (``cache_dir``/escape) or computed (``_run_jobs``)."""
        todo = {}
        for mol in self.library:
            for z in self._live_charges(mol, shape):
                zp = min(z + 2, max(mol.charges))
                key = _escape_key(mol, zp, self.p)
                if key in _ESCAPE_TABLES or key in todo:
                    continue
                path = self._escape_file(key)
                if path is not None and path.exists():
                    with np.load(path) as d:
                        _ESCAPE_TABLES[key] = d["P"]
                else:
                    todo[key] = (replace(mol, _cache={}), zp, self.p)    # light copies: no bases travel to the workers

        def keep(job, P):
            key = _escape_key(*job)
            _ESCAPE_TABLES[key] = P
            path = self._escape_file(key)
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(f".{os.getpid()}.tmp.npz")
                np.savez_compressed(tmp, P=P)
                os.replace(tmp, path)
        _run_jobs(_build_escape, list(todo.values()), default_workers(workers), keep)

    def _ion_stack(self, shape):
        """{(s, slot): [(dz, k at G0 = 1)]}: the photoionisations landing on each live state (``ionisation_products``), the
        rows of its ("ion", shape) response; empty without ``Params.ionisation_heating``."""
        key = ("ion", shape)
        if key not in self._arrays:
            out = {}
            if self.p.ionisation_heating:
                for s, mol in enumerate(self.library):
                    for z in self._live_charges(mol, shape):
                        rows = ionisation_products(mol, z, self.p, shape)
                        if rows:
                            out[(s, z - self.zmin)] = [(dz, k) for dz, k, _, _ in rows]
            self._arrays[key] = out
        return self._arrays[key]

    # -- cosmic rays ------------------------------------------------------------------------------------------------
    def _cr_file(self, key):
        return None if self.cache_dir is None else self.cache_dir / "cosmic_rays" / f"{key}.npz"

    def prepare_cr(self, workers=None):
        """``cr_passages`` of every state in both shapes of the blend (CR_SHAPES; field-independent): from memory, the disk
        cache (``cache_dir``/cosmic_rays) or computed in parallel."""
        todo = []
        for mol in self.library:
            for z in mol.charges:
                for shape in CR_SHAPES:
                    ck = ("cr", z, shape, self.p)
                    if ck in mol._cache:
                        continue
                    key = _cr_key(mol, z, self.p, shape)
                    f = self._cr_file(key)
                    if f is not None and f.exists():
                        with np.load(f) as d:
                            mol._cache[ck] = (d["k_node"], float(d["beyond"]), float(d["power"]))
                    else:
                        todo.append((mol, z, ck, f, shape))
        workers = default_workers(workers)

        def keep(t, r):
            mol, _, ck, f, _ = t
            mol._cache[ck] = r
            if f is not None:
                f.parent.mkdir(parents=True, exist_ok=True)
                tmp = f.with_suffix(f".{os.getpid()}.tmp.npz")
                np.savez(tmp, k_node=r[0], beyond=r[1], power=r[2])
                os.replace(tmp, f)
        if workers > 1 and len(todo) > 2:
            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor(max_workers=workers) as ex:     # light copies: no bases travel to the workers
                for t, r in zip(todo, ex.map(_build_cr, [(replace(t[0], _cache={}), t[1], self.p, t[4]) for t in todo])):
                    keep(t, r)
        else:
            for t in todo:
                keep(t, cr_passages(t[0], t[1], self.p, t[4]))

    def _cr_stack(self, shape):
        """{"k": [J, s, slot] (dz = 1 ... J), "k_node": {(s, slot): [dz, node]} (dz = 0 ... J), "beyond", "power": [s, slot]}:
        ``cr_passages`` of every state in the unscaled ``shape`` ("L" or "H"); None without ``Params.cosmic_rays``."""
        key = ("cr", shape)
        if key not in self._arrays:
            out = None
            if self.p.cosmic_rays:
                self.prepare_cr()
                n, m = len(self.library), len(self.zs)
                K = np.zeros((CR_DZ_MAX, n, m))
                beyond, power, nodes = np.zeros((n, m)), np.zeros((n, m)), {}
                for s, mol in enumerate(self.library):
                    for z in mol.charges:
                        k_node, beyond[s, z - self.zmin], power[s, z - self.zmin] = mol._cache[("cr", z, shape, self.p)]
                        nodes[(s, z - self.zmin)] = k_node
                        K[: k_node.shape[0] - 1, s, z - self.zmin] = k_node[1:].sum(axis=1)
                out = {"k": K, "k_node": nodes, "beyond": beyond, "power": power}
            self._arrays[key] = out
        return self._arrays[key]

    def _cr_rates(self, env):
        """The passages of ``env``'s cosmic rays [s^-1 per molecule], ``_cr_stack``'s keys: the standard spectrum of its
        zeta_cr, a_L L + a_H H (``cosmic_rays.blend_weights``); None without cosmic rays."""
        if getattr(env, "zeta_cr", 0.0) <= 0.0:
            return None
        if not self.p.cosmic_rays:
            raise ValueError("Environment.zeta_cr > 0 needs a Population with Params.cosmic_rays")
        from pycalima.pahchem import cosmic_rays

        key = ("cr_rates", float(env.zeta_cr))
        if key not in self._arrays:
            a = dict(zip(CR_SHAPES, cosmic_rays.blend_weights(env.zeta_cr)))
            st = {sh: self._cr_stack(sh) for sh in CR_SHAPES}
            mix = lambda name: sum(a[sh] * st[sh][name] for sh in CR_SHAPES)          # noqa: E731
            self._arrays[key] = {"k": mix("k"), "beyond": mix("beyond"), "power": mix("power"),
                                 "k_node": {si: sum(a[sh] * st[sh]["k_node"][si] for sh in CR_SHAPES) for si in st["L"]["k_node"]}}
        return self._arrays[key]

    def prepare_hot(self, temperature_k, workers=None, ions=()):
        """``hot_passages`` of every state at ``temperature_k`` (per unit n_e, field-independent), ``hot_ion_passages``
        of each of ``ions`` (names of ``chemistry.IONS``; per unit ion density) and the captures' ``capture_stopping``:
        from memory, the disk cache (``cache_dir``/hot) or computed in parallel."""
        t = round(float(temperature_k), 6)
        key = lambda z, ion: ("hot", z, t, self.p) if ion is None else ("hot_ion", z, t, ion)       # noqa: E731
        todo = []
        for mol in self.library:
            for z in mol.charges:
                for ion in (None, *ions):
                    if key(z, ion) in mol._cache:
                        continue
                    hk = _hot_key(mol, z, self.p, t, ion)
                    path = None if self.cache_dir is None else self.cache_dir / "hot" / f"{hk}.npz"
                    if hk not in _HOT_MEMO and path is not None and path.exists():
                        with np.load(path) as d:
                            r = (d["k_node"], float(d["beyond"]), float(d["power"]))
                            _HOT_MEMO[hk] = r if ion is None else (*r, {x[3:]: float(d[x]) for x in d.files if x[:3] == "ko_"})
                    if hk in _HOT_MEMO:
                        mol._cache[key(z, ion)] = _HOT_MEMO[hk]
                    else:
                        todo.append((mol, z, ion, hk, path))
        workers = default_workers(workers)

        def keep(t_, r):
            mol, z, ion, hk, path = t_
            mol._cache[key(z, ion)] = _HOT_MEMO[hk] = r
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(f".{os.getpid()}.tmp.npz")
                np.savez(tmp, k_node=r[0], beyond=r[1], power=r[2], **({} if ion is None else {f"ko_{x}": v for x, v in r[3].items()}))
                os.replace(tmp, path)
        if workers > 1 and len(todo) > 2:
            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor(max_workers=workers) as ex:
                jobs = [(replace(m, _cache={}), z, self.p, t, ion) for m, z, ion, *_ in todo]
                for t_, r in zip(todo, ex.map(_build_hot, jobs)):
                    keep(t_, r)
        else:
            for t_ in todo:
                keep(t_, _build_hot((t_[0], t_[1], self.p, t, t_[2])))
        self.prepare_stopping(workers)

    def _hot_rates(self, env):
        """The thermal electrons' and ions' passages at ``env`` [s^-1 per molecule], ``_cr_stack``'s keys: ``hot_passages``
        at its temperature times n_e plus ``hot_ion_passages`` of each of ``env.ions`` times its density; and
        "knockout_c" [s, slot] (C knock-outs per molecule), "knockout_h" (H knock-outs per H atom) [s^-1]. None without
        ``Params.hot_gas``."""
        if not self.p.hot_gas:
            return None
        t = round(float(env.temperature_k), 6)
        ions = tuple((name, float(n)) for name, n in env.ions if n > 0.0)
        key = ("hot_rates", t, float(env.n_e), ions)
        if key not in self._arrays:
            self.prepare_hot(t, ions=tuple(dict(ions)))
            n, m = len(self.library), len(self.zs)
            K = np.zeros((CR_DZ_MAX, n, m))
            beyond, power, nodes = np.zeros((n, m)), np.zeros((n, m)), {}
            ko_c, ko_h = np.zeros((n, m)), np.zeros((n, m))
            for s, mol in enumerate(self.library):
                for z in mol.charges:
                    k_node, b, pw = (env.n_e * x for x in mol._cache[("hot", z, t, self.p)])
                    i = z - self.zmin
                    for name, dens in ions:
                        kn, bi, pi, ko = mol._cache[("hot_ion", z, t, name)]
                        k_node, b, pw = k_node + dens * kn, b + dens * bi, pw + dens * pi
                        ko_c[s, i] += dens * ko.get("C", 0.0)
                        ko_h[s, i] += (dens * ko.get("H", 0.0) / mol.nh) if mol.nh else 0.0
                    nodes[(s, i)] = k_node
                    beyond[s, i], power[s, i] = b, pw
                    K[: k_node.shape[0] - 1, s, i] = k_node[1:].sum(axis=1)
            self._arrays[key] = {"k": K, "k_node": nodes, "beyond": beyond, "power": power,
                                 "knockout_c": ko_c, "knockout_h": ko_h}
        return self._arrays[key]

    def _passages(self, env):
        """The passages of every source at ``env`` (cosmic rays, ``_cr_rates``; thermal electrons and ions,
        ``_hot_rates``), summed; None when there are none."""
        srcs = [r for r in (self._cr_rates(env), self._hot_rates(env)) if r is not None]
        if not srcs:
            return None
        if len(srcs) == 1:
            return srcs[0]
        a, b = srcs
        return {"k": a["k"] + b["k"], "beyond": a["beyond"] + b["beyond"], "power": a["power"] + b["power"],
                "k_node": {si: a["k_node"][si] + b["k_node"][si] for si in a["k_node"]}}

    def hot_beyond(self, env):
        """Thermal electrons' and ions' passages per molecule [s^-1] beyond the heat nodes or CR_DZ_MAX charges
        (``cr_beyond``)."""
        hot = self._hot_rates(env)
        if hot is None:
            return np.zeros(len(self.library))
        return (self.charge_matrix(env) * hot["beyond"]).sum(axis=1)

    def cr_beyond(self, env):
        """Cosmic-ray passages per molecule [s^-1] that leave more than HEAT_NODES_EV's top or more than CR_DZ_MAX charges
        (``cr_passages``): not followed (their charge steps and heat are not in the chain), a destruction rate for most."""
        cr = self._cr_rates(env)
        if cr is None:
            return np.zeros(len(self.library))
        return (self.charge_matrix(env) * cr["beyond"]).sum(axis=1)

    # -- deposits the bases do not hold (captures, photoionisation products, the C 1s Auger excitation) ----------------
    def _response_file(self, key):
        return None if self.cache_dir is None else self.cache_dir / "responses" / f"{key}.npz"

    def _response(self, mol, z, which):
        """(ladder, ``deposit_response``) of set ``which`` (RESPONSE_SETS) of state z: from memory, the disk cache
        (``cache_dir``/responses) or computed."""
        k = (id(mol), z, which)
        if k not in self._resp:
            inp = _response_inputs(mol, z, self.de_cm, self.e_max_cm, self.p, which)
            key = _response_key(inp)
            path = self._response_file(key)
            if key in _RESPONSE_MEMO:
                r = _RESPONSE_MEMO[key]
            elif path is not None and path.exists():
                with np.load(path) as d:
                    r = [[(d[f"e_{i}_{m}"], float(d["ph"][i, m]), float(d["ph2"][i, m]), float(d["pc"][i, m]),
                           float(d["php"][i, m])) for m in range(len(inp[0]))] for i in range(len(inp[-1]))]
            else:
                r = _build_response(inp)
                self._save_response(path, r)
            _RESPONSE_MEMO[key] = r
            self._resp[k] = (inp[0], r)
        return self._resp[k]

    @staticmethod
    def _save_response(path, r):
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp.npz")
        np.savez(tmp, ph=np.array([[x[1] for x in row] for row in r]), ph2=np.array([[x[2] for x in row] for row in r]),
                 pc=np.array([[x[3] for x in row] for row in r]), php=np.array([[x[4] for x in row] for row in r]),
                 **{f"e_{i}_{m}": x[0] for i, row in enumerate(r) for m, x in enumerate(row)})
        os.replace(tmp, path)

    def _response_sets(self, s, z, shape):
        """The response sets state z of molecule s needs under ``shape``: "rec"; "aug" where ``_kshell_on``; ("ion", shape)
        where photoionisations land on it (``_ion_stack``)."""
        sets = list(RESPONSE_SETS if self._kshell_on(shape) else ("rec",)) + (
            ["heat"] if self.p.cosmic_rays or self.p.hot_gas else [])
        return sets + ([("ion", shape)] if (s, z - self.zmin) in self._ion_stack(shape) else [])

    def prepare_responses(self, shape, workers=None):
        """``_response`` of every live state under ``shape`` (``_response_sets``), the missing ones computed in parallel."""
        self._responses([(s, z, which) for s, mol in enumerate(self.library) for z in self._live_charges(mol, shape)
                         for which in self._response_sets(s, z, shape)], workers)

    def _responses(self, sets, workers=None):
        """``_response`` of each (s, z, which) of ``sets``: those in neither memory nor the disk cache computed together
        (``_run_jobs``, ``default_workers``), so that the flows' consumers never build them one by one."""
        todo, seen = [], set()
        for s, z, which in sets:
            mol = self.library[s]
            if (id(mol), z, which) in self._resp:
                continue
            inp = _response_inputs(mol, z, self.de_cm, self.e_max_cm, self.p, which)
            key = _response_key(inp)
            path = self._response_file(key)
            if key not in _RESPONSE_MEMO and key not in seen and (path is None or not path.exists()):
                seen.add(key)
                todo.append(inp)

        def keep(inp, r):
            key = _response_key(inp)
            self._save_response(self._response_file(key), r)
            _RESPONSE_MEMO[key] = r
        _run_jobs(_build_response, todo, default_workers(workers), keep)
        for s, z, which in sets:
            self._response(self.library[s], z, which)

    def _deposit_flows(self, env, f=None):
        """[(s, z, which, index, rate, baked)] of the deposits the bases do not hold, per molecule [s^-1] (charge fractions
        ``f``, by default ``charge_matrix``'s): captures into z
        (``recombination_heating``; "rec"), at f(z+1) alpha(z+1) n_e, with the rate the bases emit them at
        (f(z) G0 k_up(z)) to be replaced; the photoionisations landing on z (``ionisation_heating``; ("ion", shape), row
        ``index`` of ``_ion_stack``), at f(z - dz) G0 k; the C 1s products z + j ("aug", heat node ``index``), at
        f(z) G0 k_node; the passages' products ("heat", ``_passages``) and the ions' charge transfer onto cations
        ("heat", ``_ion_heat``)."""
        f = self.charge_matrix(env) if f is None else f
        flows = []
        for s, plain, passages in self._flow_terms(env):
            for z, which, idx, i, a, b, keep, ib, ba, bb in plain:
                r = f[s, i] * a * b
                if keep or r > 0.0:
                    flows.append((s, z, which, idx, r, 0.0 if ib is None else f[s, ib] * ba * bb))
            for zp, parents in passages:                  # the passages' products, summed over their parents per node
                r = 0.0
                for i, k in parents:
                    r = r + f[s, i] * k
                for node in np.nonzero(r > 0.0)[0]:
                    if HEAT_NODES_EV[node] > 0.0:          # E* = 0 heats nothing (its charge step is in the chain)
                        flows.append((s, zp, "heat", int(node), float(r[node]), 0.0))
        return flows

    def _flow_terms(self, env):
        """[(s, plain, passages)]: ``_deposit_flows`` at ``env`` as coefficients of the charge fractions, the part that
        does not change while ``charge_matrix`` iterates. ``plain``: (z, which, index, slot, a, b, keep, baked slot, ba,
        bb), a flow of rate f[s, slot] a b and baked f[s, baked slot] ba bb (none: 0), kept at rate 0 only if ``keep``;
        ``passages``: (z', [(slot, k_node[node])]), the passages' products summed over their parents. The last FLOW_MEMO
        environments are kept."""
        if env in self._terms:
            return self._terms[env]
        shape = field_shape(env)
        cap = self._capture_rates(env.temperature_k) * env.n_e
        ke = np.minimum(self._capture_energy(env.temperature_k), REC_KE_NODES_EV[-1])
        k_up = self._stack(shape)[0]
        ks = self._kshell_stack(shape)
        ion = self._ion_stack(shape)
        cr = self._passages(env)
        ion_heat = self._ion_heat(env)
        out = []
        for s, mol in enumerate(self.library):
            live = set(self._live_charges(mol, shape))
            plain = []
            for z in live:
                i = z - self.zmin
                if z + 1 in live:                           # out of the top: an overflow (chain_overflow)
                    for node, k in ion_heat.get((s, i), ()):
                        plain.append((z + 1, "heat", node, i, k, 1.0, True, None, 0.0, 0.0))
                if self.p.recombination_heating and z < max(mol.charges) and z + 1 in live:
                    # the capture deposits IP + the electron's kinetic energy, at the two nodes about it; the bases emit
                    # the IP alone, which is replaced once (node 0)
                    e = float(ke[s, i + 1])
                    j = min(int(np.searchsorted(REC_KE_NODES_EV, e, side="right")) - 1, REC_KE_NODES_EV.size - 2)
                    u = (e - REC_KE_NODES_EV[j]) / (REC_KE_NODES_EV[j + 1] - REC_KE_NODES_EV[j])
                    c = cap[s, i + 1]
                    base = (i, env.g0, k_up[s, i])
                    if j == 0:
                        plain.append((z, "rec", 0, i + 1, c, 1.0 - u, True, *base))
                    else:
                        plain += [(z, "rec", 0, i + 1, 0.0, 0.0, True, *base), (z, "rec", j, i + 1, c, 1.0 - u, False, None, 0.0, 0.0)]
                    plain.append((z, "rec", j + 1, i + 1, c, u, False, None, 0.0, 0.0))
                for r, (dz, k) in enumerate(ion.get((s, i), ())):
                    plain.append((z, ("ion", shape), r, i - dz, env.g0, k, True, None, 0.0, 0.0))
                if ks is not None and (s, i) in ks["k_node"]:
                    k_node = ks["k_node"][(s, i)]
                    for j in range(1, k_node.shape[0] + 1):
                        if z + j not in live:
                            continue                             # past the cold limit: an explosion (chain_overflow)
                        for node in np.nonzero(k_node[j - 1] > 0.0)[0]:
                            plain.append((z + j, "aug", int(node), i, env.g0, k_node[j - 1, node], True, None, 0.0, 0.0))
            passages = {}
            if cr is not None:
                for z in live:
                    k_node = cr["k_node"][(s, z - self.zmin)]
                    for dz in range(k_node.shape[0]):
                        if z + dz in live:                # past the cold limit: an explosion (chain_overflow)
                            passages.setdefault(z + dz, []).append((z - self.zmin, k_node[dz]))
            out.append((s, plain, list(passages.items())))
        if len(self._terms) >= FLOW_MEMO:
            del self._terms[next(iter(self._terms))]
        self._terms[env] = out
        return out

    def _deposit_losses(self, env, f=None, by_state=False):
        """(dlh, dlh2, dlc, dlhp)[s, k]: the H, H2, C2H2 loss and proton-emission rates [s^-1 per molecule, as the
        charge-averaged chain takes them] of ``_deposit_flows`` (charge fractions ``f``): dissociative recombination
        (``Params.recombination_dissociation``), the photoionisation, C 1s and cosmic-ray products' heat; the loss tables
        hold the photon deposits only (thermal ions' H knock-outs, per H atom, are the chain's own term:
        ``_chain_distributions``). ``by_state``: [s, slot, k], per product state."""
        f = self.charge_matrix(env) if f is None else f
        dst, src, P = self._loss_kernel(env)
        x = f.ravel()[src]
        size = len(self.library) * len(self.zs) * N_EMITTING
        # float: without any flow (a molecule with one charge state) bincount of the empty weights returns integers
        out = np.array([np.bincount(dst, weights=P[:, c] * x, minlength=size) for c in range(4)], dtype=float)
        out = out.reshape(4, len(self.library), len(self.zs), N_EMITTING)
        return tuple(out) if by_state else tuple(out.sum(axis=2))

    def _loss_kernel(self, env):
        """(dst, src, P[entry, 4]): ``_deposit_losses`` as the linear map of the charge fractions it is,
        out[c].ravel()[dst] += P[:, c] f.ravel()[src] (dst over [s, slot, member], src over [s, slot]), from
        ``_flow_terms``: the flows' responses summed once per environment rather than once per iteration of
        ``charge_matrix`` (the same sums to round-off). The last FLOW_MEMO environments are kept."""
        if env in self._kernels:
            return self._kernels[env]
        m = len(self.zs)
        terms = self._flow_terms(env)
        use = lambda which: which != "rec" or self.p.recombination_dissociation      # noqa: E731
        self._responses(dict.fromkeys([(s, z, w) for s, plain, _ in terms for z, w, *_ in plain if use(w)]
                                      + [(s, zp, "heat") for s, _, passages in terms for zp, _ in passages]))
        rows = {}                                   # (s, z, which) -> [members, 4] per index
        dst, src, P = [], [], []

        def add(s, z, which, idx, i, c):
            if (s, z, which) not in rows:
                rows[(s, z, which)] = [np.array([p for _, *p in members], dtype=float).reshape(-1, 4)
                                       for members in self._response(self.library[s], z, which)[1]]
            R = rows[(s, z, which)][idx]
            dst.append((s * m + z - self.zmin) * N_EMITTING + np.arange(R.shape[0]))
            src.append(np.full(R.shape[0], s * m + i))
            P.append(c * R)
        for s, plain, passages in terms:
            for z, which, idx, i, a, b, *_ in plain:
                if use(which) and a * b != 0.0:
                    add(s, z, which, idx, i, a * b)
            for zp, parents in passages:
                for i, k in parents:
                    for node in np.nonzero((k > 0.0) & (HEAT_NODES_EV > 0.0))[0]:
                        add(s, zp, "heat", int(node), i, float(k[node]))
        cat = lambda v, shape: np.concatenate(v) if v else np.zeros(shape)           # noqa: E731
        kernel = (cat(dst, 0).astype(np.int64), cat(src, 0).astype(np.int64), cat(P, (0, 4)))
        if len(self._kernels) >= FLOW_MEMO:
            del self._kernels[next(iter(self._kernels))]
        self._kernels[env] = kernel
        return kernel

    def _differing_flows(self, env):
        """``_deposit_flows`` that differ from what the bases emit by more than round-off (none in a plain chain: its
        captures run at k_up), their responses prepared (``_responses``)."""
        flows = [fl for fl in self._deposit_flows(env) if abs(fl[4] - fl[5]) > ROUND_OFF * max(abs(fl[4]), abs(fl[5]))]
        self._responses(dict.fromkeys((s, z, which) for s, z, which, *_ in flows))
        return flows

    def _deposit_bands(self, env, h):
        """dF[s, band]: the IR band power of ``_deposit_flows`` per molecule, weighted by the hydrogenation shares
        ``h``[s, k], less the captures the bases already emit."""
        shape = field_shape(env)
        key = ("deposit_bands", shape, tuple(self.windows.items()), self.local, self.band_convention)
        RB = self._arrays.setdefault(key, {})       # (s, z, which) -> band powers [index, k, band] per deposit
        dF = np.zeros((len(self.library), len(self.bands)))
        for s, z, which, idx, rate, baked in self._differing_flows(env):
            if (s, z, which) not in RB:
                mol = self.library[s]
                ladder, resp = self._response(mol, z, which)
                RB[(s, z, which)] = [np.array([em @ self._window_matrix(mol, z, shape, np.asarray(ladder[k][0], dtype=float))
                                               for k, (em, *_) in enumerate(members[:N_EMITTING])]) for members in resp]
            b = RB[(s, z, which)][idx]
            dF[s] += (rate - baked) * (h[s, : b.shape[0]] @ b)
        return dF

    def deposit_modes(self, env):
        """{s: [(freqs [cm^-1], IR power per mode [cm^-1 s^-1 per molecule])]}: the modes of ``_deposit_flows``, weighted by
        the hydrogenation shares and less the captures the bases emit, as ``deposit_spectra`` takes them (its spectrum is
        these modes under its line profiles); what ``spectral_fit.molecule_table`` packs with the bases' modes."""
        h = self.emission_weights(env).sum(axis=1)
        out = {}
        for s, z, which, idx, rate, baked in self._differing_flows(env):
            w = rate - baked
            ladder, resp = self._response(self.library[s], z, which)
            for k, (em, *_) in enumerate(resp[idx][:N_EMITTING]):
                if h[s, k] > 0.0:
                    out.setdefault(s, []).append((np.asarray(ladder[k][0], dtype=float), w * h[s, k] * np.asarray(em, dtype=float)))
        return out

    def deposit_spectra(self, env, nu_cm, fwhm_cm=None, shift_cm=None, band_shape=None):
        """dP[s, i]: the IR spectrum of ``_deposit_flows`` per molecule [cm^-1 s^-1 per cm^-1], weighted by the hydrogenation
        shares and less the captures the bases emit, as ``_deposit_bands``: with the line profiles of ``_line_spectrum``,
        or ``band_shape`` (``spectral_fit.BandShape``, the fitted spectra's). Rows are built only for the flows that differ
        from what the bases hold by more than round-off (none in a plain chain: its captures run at k_up)."""
        from pycalima.pahchem import spectral_fit

        nu = np.asarray(nu_cm, dtype=float)
        shape = field_shape(env)
        conv = self.band_convention if fwhm_cm is None and shift_cm is None and band_shape is None else None
        key = ("deposit_spectra", shape, hashlib.sha1(nu.tobytes()).hexdigest(), fwhm_cm, shift_cm, conv, band_shape)
        rows = self._arrays.setdefault(key, {})
        h = self.emission_weights(env).sum(axis=1)
        dP = np.zeros((len(self.library), nu.size))
        for s, z, which, idx, rate, baked in self._differing_flows(env):
            w = rate - baked
            if (s, z, which, idx) not in rows:
                mol = self.library[s]
                ladder, resp = self._response(mol, z, which)
                rows[(s, z, which, idx)] = np.array([
                    spectral_fit.shaped_spectrum(np.asarray(ladder[k][0], dtype=float), em, nu, band_shape)
                    if band_shape is not None else
                    self._line_spectrum(mol, z, shape, np.asarray(ladder[k][0], dtype=float), em, nu, fwhm_cm, shift_cm)
                    for k, (em, *_) in enumerate(resp[idx][:N_EMITTING])])
            R = rows[(s, z, which, idx)]
            dP[s] += w * (h[s, : R.shape[0]] @ R)
        return dP

    def _jump_rates(self, env, up):
        """[k1, ..., kJ] [s^-1] at ``env``: ``up`` (single steps at env: photo, and ions where given) less the C 1s
        absorption and the double ionisations it counts, with those as their +1 ... +J steps (``_kshell_stack``,
        ``_double_stack``); None when there are none, the plain chain."""
        shape = field_shape(env)
        ks, k2v = self._kshell_stack(shape), self._double_stack(shape)
        cr = self._passages(env)
        if ks is None and not k2v.any() and cr is None:
            return None
        jk = ks["k"] if ks is not None else np.zeros((2,) + up.shape)
        over = ks["over"] if ks is not None else 0.0
        single = up - env.g0 * (over + k2v)          # >= 0: sigma - sigma_K >= 0 and P2 <= DOUBLE_MAX
        bad = single < -ROUND_OFF * np.abs(up)
        if np.any(bad):
            raise ValueError(f"valence single ionisation < 0 beyond round-off ({single[bad].min():.3e} s^-1)")
        steps = [np.maximum(single, 0.0) + env.g0 * jk[0]]                  # the round-off below 0 only
        steps += [env.g0 * jk[j] for j in range(1, jk.shape[0])]
        steps[1] = steps[1] + env.g0 * k2v
        if cr is not None:                                # +1 ... +CR_DZ_MAX steps of the passages
            steps += [np.zeros_like(up) for _ in range(cr["k"].shape[0] - len(steps))]
            for j in range(cr["k"].shape[0]):
                steps[j] = steps[j] + cr["k"][j]
        return steps

    def _state_modes(self, b):
        """[(freqs, IR power per mode at G0 = 1)] of N_H0 ... N_H0-3 of basis ``b``; empty past the ladder."""
        out = [(b.freqs_cm, b.emitted)] + list(zip(b.ladder_freqs, b.ladder_emitted))
        return out + [(np.zeros(0), np.zeros(0))] * (N_EMITTING - len(out))

    def _hydrogenation_bands(self, shape):
        """B[s, slot, k, band]: band power of N_H0 - k at G0 = 1, k < ``N_EMITTING``; 0 where absent."""
        key = ("hbands", shape, tuple(self.windows.items()), self.local, self.band_convention)
        if key not in self._arrays:
            self.prepare(shape)
            if self.band_convention.kind == "tdep":
                self.prepare_temperature_shares(shape)
            B = np.zeros((len(self.library), len(self.zs), N_EMITTING, len(self.bands)))
            for s, mol in enumerate(self.library):
                for z in self._live_charges(mol, shape):
                    for k, (f_, e_) in enumerate(self._state_modes(self.basis(mol, z, shape))):
                        if f_.size:
                            B[s, z - self.zmin, k] = e_ @ self._window_matrix(mol, z, shape, f_)
            self._arrays[key] = B
        return self._arrays[key]

    def state_powers(self, shape):
        """E[s, slot, k]: total IR power of N_H0 - k at G0 = 1 [cm^-1 s^-1] (photon and recombination heating)."""
        key = ("hpower", shape)
        if key not in self._arrays:
            self.prepare(shape)
            E = np.zeros((len(self.library), len(self.zs), N_EMITTING))
            for s, mol in enumerate(self.library):
                for z in self._live_charges(mol, shape):
                    for k, (_, e_) in enumerate(self._state_modes(self.basis(mol, z, shape))):
                        E[s, z - self.zmin, k] = float(np.sum(e_))
            self._arrays[key] = E
        return self._arrays[key]

    def _ladders(self, shape):
        """H, H2, C2H2-loss and proton-emission tables of the ladders N_H0 ... N_H0-3, each (s, slot, 4, K_H_G0_GRID); 0 where
        absent."""
        key = ("ladders", shape)
        if key not in self._arrays:
            self.prepare(shape)
            tabs = np.zeros((4, len(self.library), len(self.zs), 4, K_H_G0_GRID.size))
            for s, mol in enumerate(self.library):
                for z in self._live_charges(mol, shape):
                    b = self.basis(mol, z, shape)
                    if b.absorbed > 0.0:          # no photon deposits, no photon-driven loss (bases before 2026-11-03: NaN)
                        for t, v in enumerate((b.ladder_h, b.ladder_h2, b.ladder_c2h2, b.ladder_hp)):
                            if v is not None:
                                tabs[t, s, z - self.zmin] = v
            self._arrays[key] = tuple(tabs)
        return self._arrays[key]

    def _capture_rates(self, temperature_k):
        """alpha[s, slot]: electron capture coefficient of each state [cm^3 s^-1] (0 for a lowest state)."""
        t = round(float(temperature_k), 6)
        if t not in self._capture:
            a = np.zeros((len(self.library), len(self.zs)))
            if self.p.hot_gas:
                self.prepare_stopping()
            for s, mol in enumerate(self.library):
                for z in mol.charges[1:]:
                    a[s, z - self.zmin] = capture_coefficient(mol, z, temperature_k, self.p)
                    if self.p.hot_gas:                           # a fast electron must lose its energy to stay
                        a[s, z - self.zmin] *= capture_stop_factors(mol, z, t, self.p,
                                                                    mol._cache[("stop", z, self.p.uv_cross_section)])[0]
            self._capture[t] = a
        return self._capture[t]

    def prepare_stopping(self, workers=None):
        """``capture_stopping`` of every state that captures (field- and temperature-independent), in parallel."""
        uv = self.p.uv_cross_section
        todo = [(mol, z) for mol in self.library for z in mol.charges[1:] if ("stop", z, uv) not in mol._cache]
        workers = default_workers(workers)
        if workers > 1 and len(todo) > 2:
            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor(max_workers=workers) as ex:
                for (mol, z), r in zip(todo, ex.map(_build_stop, [(replace(m, _cache={}), z, self.p) for m, z in todo])):
                    mol._cache[("stop", z, uv)] = r
        else:
            for mol, z in todo:
                mol._cache[("stop", z, uv)] = capture_stopping(mol, z, self.p)

    # -- an environment --------------------------------------------------------------------------
    def charge_matrix(self, env):
        """f[s, slot]: the charge fractions of every molecule over ``self.zs`` (0 for the states its field cannot reach,
        ``_live``). With ``Params.fission`` the protons hot ions emit are down-steps too (``_proton_down``); their rates
        depend on the hydrogenation, which depends on the charge, so the two are iterated to FISSION_TOL (raises past
        FISSION_ITER_MAX; ``_anderson``). Past a barrier-limited top (``_routed``) the H-bearing share 1 - f(N_H = 0) lands
        on the top, which also couples the two. The last CHAIN_MEMO environments are kept."""
        if env in self._charges:
            return self._charges[env]
        if self._joint_on():                                 # the marginal of the joint (Z, N_H) chain
            f = np.zeros((len(self.library), len(self.zs)))
            for s, (lo, F, _) in enumerate(self._joint_distributions(env)):
                f[s, lo:lo + F.shape[0]] = F.sum(axis=1)
            if len(self._charges) >= CHAIN_MEMO:
                del self._charges[next(iter(self._charges))]
            self._charges[env] = f
            return f
        f = self._charge_solve(env, None)
        if self.p.fission:
            routed = self._routed()

            def g(x):
                down, p0 = self._proton_down(env, x, with_p0=True)
                return self._charge_solve(env, down, np.where(routed, 1.0 - p0, 0.0) if routed.any() else None)
            f = _anderson(g, f, FISSION_TOL, FISSION_ITER_MAX, "charge and proton emission")
        if len(self._charges) >= CHAIN_MEMO:
            del self._charges[next(iter(self._charges))]
        self._charges[env] = f
        return f

    def _proton_down(self, env, f, with_p0=False):
        """down[s, slot] [s^-1 per molecule in the state]: the protons it emits at charge fractions ``f``, the loss tables'
        at G0 and the deposits' landing on it (per molecule there), each over the hydrogenation distribution of the chain
        (``_chain_distributions`` at ``f``; the states past N_H0-3 take its mapping, the bare skeleton none). ``with_p0``:
        (down, the bare share f(N_H = 0) of each molecule's chain)."""
        shape = field_shape(env)
        tab = self._interp_g0(self._ladders(shape)[3], env.g0)                       # [s, slot, k]
        losses = self._deposit_losses(env, f, by_state=True)                         # one pass over the flows
        dep = losses[3]                                                              # [s, slot, k], per molecule
        per = tab + np.divide(dep, f[:, :, None], out=np.zeros_like(dep), where=f[:, :, None] > 0.0)
        chains = self._chain_distributions(env, f, tuple(x.sum(axis=1) for x in losses))
        down = np.zeros_like(f)
        for s, mol in enumerate(self.library):
            p = np.asarray(chains[s])
            for i in np.nonzero(per[s].any(axis=1))[0]:
                down[s, i] = p @ _deep_rates(mol.nh, per[s, i])
        if with_p0:
            return down, np.array([float(c[0]) for c in chains])
        return down

    def _ion_up(self, env):
        """up[s, slot] [s^-1]: the ions' up-steps of every state (``ion_up_rate``; out of the top, an overflow); the last
        CHAIN_MEMO environments are kept (``charge_matrix`` iterates on them)."""
        if env not in self._ion_ups:
            up = np.zeros((len(self.library), len(self.zs)))
            for s, mol in enumerate(self.library):
                for z in mol.charges:
                    up[s, z - self.zmin] = ion_up_rate(mol, z, env, self.p)
            if len(self._ion_ups) >= CHAIN_MEMO:
                del self._ion_ups[next(iter(self._ion_ups))]
            self._ion_ups[env] = up
        return self._ion_ups[env]

    def _ion_heat(self, env):
        """{(s, slot): [(node, k)]} [s^-1 per molecule in the slot]: the ions' exothermic charge transfer onto its cation
        (``Params.hot_gas``, ``ion_up_rate``) leaves IP(X) - IP(PAH^z) in the product z + 1 (the user's choice of
        2026-11-05), split between the two HEAT_NODES_EV about it keeping its mean (its "heat" responses); an endothermic
        one leaves nothing (its deficit is the gas's, ``transfer_cooling``). The last CHAIN_MEMO environments are kept."""
        if env not in self._ion_heats:
            heat = {}
            if self.p.hot_gas:
                for s, mol in enumerate(self.library):
                    for z in mol.charges:
                        if z < 1:
                            continue
                        for ion, n in getattr(env, "ions", ()):
                            de = chemistry.IONS[ion][1] - state_threshold(mol, z)
                            if de <= 0.0:
                                continue
                            k = n * chemistry.ion_cation_transfer(mol.nc, mol.nh, z, ion, env.temperature_k)
                            j = min(int(np.searchsorted(HEAT_NODES_EV, de, side="right")) - 1, HEAT_NODES_EV.size - 2)
                            u = (de - HEAT_NODES_EV[j]) / (HEAT_NODES_EV[j + 1] - HEAT_NODES_EV[j])
                            rows = heat.setdefault((s, z - self.zmin), [])
                            rows += [(j, k * (1.0 - u))] if j > 0 else []   # E* = 0 heats nothing
                            rows.append((j + 1, k * u))
            if len(self._ion_heats) >= CHAIN_MEMO:
                del self._ion_heats[next(iter(self._ion_heats))]
            self._ion_heats[env] = heat
        return self._ion_heats[env]

    def _single_up(self, env, ions=True):
        """up[s, slot] [s^-1]: the single up-steps of every state at ``env``: photoionisation and photodetachment at G0
        (``_k_up``), the anions' associative detachment and (``ions``) the ions' charge transfer (``_ion_up``)."""
        up = env.g0 * self._k_up(field_shape(env))
        if self.zmin == -1:
            up[:, 0] += anion_associative_detachment(self.p) * env.n_h
        if ions and getattr(env, "ions", ()):
            up = up + self._ion_up(env)
        return up

    def _routed(self):
        """routed[s]: an ionisation past the molecule's top is a proton emission, not a Coulomb explosion (pahchem audit
        A2, the user's choice of 2026-11-05). With ``Params.fission`` where the top is the barrier limit of
        ``impact.coulomb_limit``, below floor(0.2 N_C) (N_C >~ 115: C150 +26, C216 +32): the next charge has no barrier and
        emits a proton at once, so the molecule returns to the top with an H less. A bare skeleton has none to emit, and
        explodes."""
        if not self.p.fission:
            return np.zeros(len(self.library), dtype=bool)
        return np.array([max(m.charges) == impact.coulomb_limit(m.nc) < np.floor(impact.COLD_CHARGE_PER_C * m.nc)
                         for m in self.library])

    def _past_top(self, env, up):
        """(rate, past, freed)[s, slot] [s^-1 per molecule in the state] of the steps from ``up`` (single steps; with the
        C 1s, double and passage steps of ``_jump_rates``) that land past the top: their rate, the charges they carry past
        it, and the charges they add (j for a +j step)."""
        live = self._live(field_shape(env))
        steps = self._jump_rates(env, up)
        steps = [up] if steps is None else steps
        top = np.array([max(m.charges) - self.zmin for m in self.library])
        idx = np.arange(len(self.zs))
        rate, past, freed = (np.zeros_like(up) for _ in range(3))
        for j, k in enumerate(steps, start=1):
            kk = np.where(live & ~_lands(live, j), k, 0.0)
            rate += kk
            past += kk * np.maximum(idx[None, :] + j - top[:, None], 0)
            freed += j * kk
        return rate, past, freed

    def _charge_solve(self, env, extra_down, redirect=None):
        """``charge_matrix``'s chain with the down-steps ``extra_down`` [s, slot] added to the captures. ``redirect[s]``:
        the share of a molecule's steps past its top that land on the top instead (``_routed``: they emit their charges
        past it as protons), the rest a loss the chain does not follow."""
        shape = field_shape(env)
        live = self._live(shape)
        up = self._single_up(env)
        down = self._capture_rates(env.temperature_k) * env.n_e
        if extra_down is not None:
            down = down + extra_down
        steps = self._jump_rates(env, up)
        if steps is not None and redirect is not None and np.any(redirect > 0.0):
            steps = [k.copy() for k in steps]
            top = np.array([max(m.charges) - self.zmin for m in self.library])
            for j in range(2, len(steps) + 1):
                gone = live & ~_lands(live, j)
                for s_ in np.nonzero(redirect > 0.0)[0]:
                    for i in np.nonzero(gone[s_])[0]:
                        if top[s_] - i >= 1:                       # from below the top: it lands on the top
                            steps[top[s_] - i - 1][s_, i] += redirect[s_] * steps[j - 1][s_, i]
        if steps is not None:
            return _chain_fractions(steps, down, live)
        both = live[:, :-1] & live[:, 1:]
        logr = np.zeros_like(up[:, :-1])
        with np.errstate(divide="ignore"):              # a state no step reaches (k_up = 0) has f = 0 exactly
            logr[both] = np.log(up[:, :-1][both]) - np.log(down[:, 1:][both])
        logf = np.concatenate([np.zeros((len(self.library), 1)), np.cumsum(logr, axis=1)], axis=1)
        logf = np.where(live, logf, -np.inf)
        f = np.exp(logf - logf.max(axis=1, keepdims=True))
        return f / f.sum(axis=1, keepdims=True)

    def _pe_stack(self, shape, partition):
        """(G[s, slot] photoelectron power, A[s, slot] absorbed power on E_PH), both at G0 = 1 [eV s^-1]; 0 where absent.

        G is ``photo.photoelectron_power`` with each state's own sigma, yield and threshold (``_yield``,
        ``mol.ionisation_ev``), so it shares the charge balance's rates; anions detach with partition 1.
        """
        key = ("pe", shape, float(partition))
        if key not in self._arrays:
            field1 = _unit_field(shape)
            n, m = len(self.library), len(self.zs)
            G, A = np.zeros((n, m)), np.zeros((n, m))
            for s, mol in enumerate(self.library):
                for z in self._live_charges(mol, shape):
                    sigma = state_sigma(mol, z, self.p)
                    i = z - self.zmin
                    A[s, i] = field1.photon_rate(sigma * E_PH, E_PH)
                    sigma = self._valence_sigma(mol, z, shape)     # the C 1s electrons are added below (kshell_rates)
                    ek = electron_energy(mol, z, self.p, partition)
                    if self.p.double_ionisation != "none" and 0 <= z < max(mol.charges):
                        # a double ionisation's two electrons share the excess over IP(z) + IP(z+1) as one would
                        dip = double_threshold(mol, z)
                        p2 = photo.double_fraction(E_PH, dip)
                        ek = (1.0 - p2) * ek + p2 * photo.photoelectron_energy(
                            E_PH, dip, partition, reference_ev=photoelectron_reference(mol, self.p))[0]
                    G[s, i] = photo.photoelectron_power(field1, sigma, _yield(mol, z, self.p), E_PH,
                                                        state_threshold(mol, z), electron_energy_ev=ek)
            ks = self._kshell_stack(shape)
            if ks is not None:                      # the C 1s photo- and Auger electrons, and the absorption above E_PH
                G, A = G + ks["power"], A + ks["absorbed_hi"]
            self._arrays[key] = (G, A)
        return self._arrays[key]

    def photoelectric_heating(self, env, partition=photo.PHOTOELECTRON_PARTITION):
        """Gas heating by photoelectrons at ``env``: (Gamma[s] [erg s^-1 per molecule], Gamma_z[s, slot], eps[s]).

        Gamma_z = G0 f(z) int sigma Y F gamma (E - IP_z) dE over the charge distribution of ``charge_matrix``,
        gamma = ``partition`` for Z >= 0 and 1 for detachment below 13.6 eV, and ``photo.photoelectron_energy``
        above; with ``Params.detachment = "statistical"`` or ``"valence"`` the anion's electrons take the energies of
        ``statistical_detachment`` (prompt E - EA, valence the photoelectron share above
        ``anion_valence_threshold``, delayed ~0.1 eV). The top state's EUV ionisations
        (``chain_overflow``) heat the gas too, although the chain does not follow their charge. eps is the photoelectric efficiency, Gamma over
        the power the molecule absorbs on E_PH: all of it, 1-500 eV (1-13.6 eV for fields cut at the Lyman limit), as
        Berné+22 and WD01 define it and as the observed I_gas / (I_PAH + I_gas) measures it (5-13.6 eV before
        2026-10-24; BT94's 6-13.6 eV convention differs).
        Recombination cooling is not subtracted here.
        """
        G, A = self._pe_stack(field_shape(env), partition)
        f = self.charge_matrix(env)
        gz = env.g0 * f * G * chemistry.EV_ERG
        gamma = gz.sum(axis=1)
        absorbed = env.g0 * (f * A).sum(axis=1) * chemistry.EV_ERG
        return gamma, gz, np.divide(gamma, absorbed, out=np.zeros_like(gamma), where=absorbed > 0)

    def _capture_energy(self, temperature_k):
        """E[s, slot]: mean kinetic energy an electron captured by each state takes from the gas [eV], kT Lambda~/J~
        of Draine & Sutin 1987 (eqs. 3.4-3.6) at the capture radius (``chemistry.disc_radius_cm``); in hot gas
        (``Params.hot_gas``) that of the captures below the stopping peak, the rest being in the passages
        (``capture_stop_factors``' energy_low)."""
        t = round(float(temperature_k), 6)
        key = ("capture_energy", t)
        if key not in self._capture:
            E = np.zeros((len(self.library), len(self.zs)))
            if self.p.hot_gas:
                self.prepare_stopping()
            for s, mol in enumerate(self.library):
                for z in mol.charges[1:]:
                    E[s, z - self.zmin] = capture_energy(mol, z, t)
                    if self.p.hot_gas:
                        E[s, z - self.zmin] *= capture_stop_factors(mol, z, t, self.p,
                                                                    mol._cache[("stop", z, self.p.uv_cross_section)])[2]
            self._capture[key] = E
        return self._capture[key]

    def recombination_cooling(self, env):
        """Gas cooling by electron capture at ``env``: (Lambda[s] [erg s^-1 per molecule], Lambda_z[s, slot]).

        Lambda_z = n_e f(z) alpha(z) <E>(z): the capture rate of ``charge_matrix`` (BT94/DS87 capture, the
        statistical sticking on neutrals, ``Params.capture_model``) times the kinetic energy each captured electron
        takes, ``_capture_energy`` (~kT on cations, 1.5-2 kT on neutrals). For neutrals the DS87 flux-weighted
        energy is used for the stuck electrons as well (their energy-dependent sticking is not followed).
        """
        f = self.charge_matrix(env)
        lz = env.n_e * f * self._capture_rates(env.temperature_k) * self._capture_energy(env.temperature_k) * chemistry.EV_ERG
        return lz.sum(axis=1), lz

    def gas_rates(self, env, abundance):
        """What the PAHs give a gas network at ``env``, for ``abundance[s]`` molecules per H nucleus.

        Per H nucleus, from one steady state of ``charge_matrix``:

        - ``alpha_mn[ion]``, ``alpha_ct[ion]`` [cm^3 s^-1]: dn(X+)/dt = -(alpha_mn + alpha_ct) n(X+) n_H, mutual
          neutralisation on anions and ion-neutral reactions on neutrals (``ion_up_rate``);
        - ``alpha_ct_cations[ion]`` [cm^3 s^-1]: the same for charge transfer onto cations, in hot gas only
          (``Params.hot_gas``);
        - ``carbon_accretion`` [cm^3 s^-1]: the share of C+ + PAH0 that keeps the C (0.4, Canosa+95), within
          alpha_ct["C+"]; the rest returns C to the gas;
        - ``electron_source`` [s^-1]: electrons made by photoionisation, photodetachment and associative detachment
          minus those captured (in steady state it equals the protons emitted, ``proton_source``, minus the ion-driven
          up-flux: PAHs pass the ions' charge to electrons, and theirs to protons);
        - ``proton_source`` [s^-1]: protons hot ions emit (``Params.fission``; ``_proton_down`` at ``f``, and past a
          barrier-limited top, ``_routed``), an H+ source; ``electron_overflow`` [s^-1]: the electrons freed by the steps
          past the top that the chain does not follow (the explosions; ionisation events before 2026-11-05);
        - ``gamma_pe``, ``lambda_rec`` [erg s^-1]: photoelectric heating and recombination cooling;
        - ``lambda_coll`` [erg s^-1]: in hot gas (``Params.hot_gas``) the collisional cooling, what the thermal electrons'
          and ions' passages take from the gas: the heat they leave in the molecule (``_hot_rates`` power) and the
          ionisation work of their charge steps, IP(z) + ... + IP(z+j-1) for +j (the ejected electrons' kinetic energy
          returns to the gas). The stopped electrons' energy is here, not in ``lambda_rec`` (pahchem audit M2: it was
          1.6-112x lambda_rec for C54H18 and C216H36 at 1e5-1e7 K, and not reported). With ``env.ions`` also the
          deficits of the endothermic charge transfers onto cations (``transfer_cooling``);
        - ``f``: the charge fractions [s, slot] over ``self.zs``.

        H2 formation on PAHs is not here: it needs the hydrogenation network (``network.PAHNetwork``).
        """
        x = np.asarray(abundance, dtype=float)
        f = self.charge_matrix(env)
        k_up, _, _ = self._stack(field_shape(env))
        photo_up = env.g0 * k_up
        photo_up[:, -1] = 0.0
        tops = np.array([max(m.charges) - self.zmin for m in self.library])
        photo_up[np.arange(len(self.library)), tops] = 0.0                     # no chain step out of the top state
        steps = self._jump_rates(env, env.g0 * k_up)
        if steps is not None:
            # a +j step frees j electrons; the steps past the top leave the chain (chain_overflow)
            live = self._live(field_shape(env))
            photo_up = sum(j * np.where(_lands(live, j), k, 0.0) for j, k in enumerate(steps, start=1))
        if self.zmin == -1:
            photo_up[:, 0] += anion_associative_detachment(self.p) * env.n_h   # it frees an electron too
        capture = self._capture_rates(env.temperature_k) * env.n_e
        alpha_mn, alpha_ct, alpha_cat = {}, {}, {}
        for ion, _ in getattr(env, "ions", ()):
            unit = replace(env, ions=((ion, 1.0),))
            mn = np.array([f[s, -1 - self.zmin] * ion_up_rate(m, -1, unit, self.p) if -1 in m.charges else 0.0
                           for s, m in enumerate(self.library)])
            ct = np.array([f[s, 0 - self.zmin] * ion_up_rate(m, 0, unit, self.p) if 1 in m.charges else 0.0
                           for s, m in enumerate(self.library)])
            cat = np.array([sum(f[s, z - self.zmin] * ion_up_rate(m, z, unit, self.p) for z in m.charges if z >= 1)
                            for s, m in enumerate(self.library)])
            alpha_mn[ion], alpha_ct[ion], alpha_cat[ion] = float(x @ mn), float(x @ ct), float(x @ cat)
        gamma = self.photoelectric_heating(env)[0]
        lam = self.recombination_cooling(env)[0]
        protons = (f * self._proton_down(env, f)).sum(axis=1) if self.p.fission else np.zeros(len(self.library))
        # the steps past the top: the H-bearing share of a barrier-limited top's lands on it (``_routed``), freeing its
        # electrons and emitting its charges past the top as protons; the rest leave the chain (chain_overflow)
        routed = self._routed()
        stay = np.where(routed, 1.0 - self._bare_share(env), 0.0) if routed.any() else np.zeros(len(self.library))
        freed = (f * self._past_top(env, env.g0 * k_up)[2]).sum(axis=1)
        protons = protons + stay * (f * self._past_top(env, self._single_up(env))[1]).sum(axis=1)
        lam_coll = np.zeros(len(self.library))
        hot = self._hot_rates(env)
        if hot is not None:
            work = np.zeros_like(f)
            for s_, mol in enumerate(self.library):
                for z in mol.charges:
                    i = z - self.zmin
                    cum = np.cumsum([_ladder_ip(mol, z + q) for q in range(hot["k"].shape[0])])
                    work[s_, i] = float(hot["k"][:, s_, i] @ cum)
            cool = np.zeros_like(f)
            for s_, mol in enumerate(self.library):
                for z in mol.charges:
                    cool[s_, z - self.zmin] = transfer_cooling(mol, z, env, self.p)
            lam_coll = (f * (hot["power"] + work + cool)).sum(axis=1) * chemistry.EV_ERG
        return {"alpha_mn": alpha_mn, "alpha_ct": alpha_ct, "alpha_ct_cations": alpha_cat, "proton_source": float(x @ protons),
                "carbon_accretion": 0.4 * alpha_ct.get("C+", 0.0),
                "electron_source": float(x @ ((f * photo_up).sum(axis=1) - (f * capture).sum(axis=1) + stay * freed)),
                "electron_overflow": float(x @ ((1.0 - stay) * freed)),
                "gamma_pe": float(x @ gamma), "lambda_rec": float(x @ lam), "lambda_coll": float(x @ lam_coll), "f": f}

    # -- the bare skeleton and destruction ---------------------------------------------------------------------------
    def prepare_bare(self, shape, workers=None):
        """``bare_loss_table`` of every live state under ``shape``: from memory, the disk cache (``cache_dir``/bare) or
        computed in parallel."""
        todo = []
        for mol in self.library:
            for z in self._live_charges(mol, shape):
                ck = ("bare", shape, z, self.de_cm, self.e_max_cm, self.p)
                if ck in mol._cache:
                    continue
                inp = _bare_inputs(mol, z, shape, self.de_cm, self.e_max_cm, self.p)
                f = None if self.cache_dir is None else self.cache_dir / "bare" / f"{_bare_key(inp)}.npz"
                if f is not None and f.exists():
                    with np.load(f) as d:
                        mol._cache[ck] = {c: d[c] for c in d.files}
                else:
                    todo.append((mol, ck, f, inp))
        workers = default_workers(workers)

        def keep(t, r):
            mol, ck, f, _ = t
            mol._cache[ck] = r
            if f is not None:
                f.parent.mkdir(parents=True, exist_ok=True)
                tmp = f.with_suffix(f".{os.getpid()}.tmp.npz")
                np.savez(tmp, **r)
                os.replace(tmp, f)
        if workers > 1 and len(todo) > 2:
            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor(max_workers=workers) as ex:
                for t, r in zip(todo, ex.map(_build_bare, [t[3] for t in todo])):
                    keep(t, r)
        else:
            for t in todo:
                keep(t, bare_loss_table(**t[3]))

    def _bare_stack(self, shape):
        """{channel: [s, slot, K_H_G0_GRID]}: the bare skeletons' photon-driven destruction rates (``bare_loss_table``),
        0 where a molecule lacks the channel or the state is not live."""
        key = ("bare", shape)
        if key not in self._arrays:
            self.prepare_bare(shape)
            out = {}
            for s, mol in enumerate(self.library):
                for z in self._live_charges(mol, shape):
                    for c, v in mol._cache[("bare", shape, z, self.de_cm, self.e_max_cm, self.p)].items():
                        out.setdefault(c, np.zeros((len(self.library), len(self.zs), K_H_G0_GRID.size)))[s, z - self.zmin] = v
            self._arrays[key] = out
        return self._arrays[key]

    def _flow_energies(self, mol, z, which):
        """(energies [eV], weights rows or None) of the deposits of response set ``which`` of state z (``_response_inputs``)."""
        key = ("flow_energies", id(mol), z, which)
        if key not in self._arrays:
            inp = _response_inputs(mol, z, self.de_cm, self.e_max_cm, self.p, which)
            self._arrays[key] = (np.asarray(inp[6], dtype=float), inp[7] if len(inp) > 7 else None)
        return self._arrays[key]

    def destruction_rates(self, env):
        """Per molecule at ``env`` [s^-1]: {"fold", "C", "C2", "C3", ...: the bare skeleton's destruction events (its
        share f(N_H = 0) of the chain, photon-driven from ``_bare_stack`` and after the deposits the tables do not hold,
        ``bare_event_yields``), "C2H2" (``loss_rates``), "explosions" (``chain_overflow``), "cr_beyond" (``cr_beyond``),
        "hot_beyond" (``hot_beyond``), "knockout" (thermal ions' C knock-outs, ``_hot_rates``; one C each),
        "events": their sum, the rate at which the molecule leaves the library, and "carbon": the C atoms lost per second
        (a whole molecule for an explosion or a passage beyond)}. The chain is quasi-static: these are slow leaks from its
        steady state."""
        shape = field_shape(env)
        f = self.charge_matrix(env)
        p0 = np.array([float(c[0]) for c in self._chain_distributions(env)])
        out = {}
        if self._joint_on():        # the bare skeleton's own charge shares f(Z, N_H = 0), and the deposits landing on them
            b0 = self._bare_slots(env)
            for c, tab in self._bare_stack(shape).items():
                out[c] = np.einsum("sz,sz->s", b0, self._interp_g0(tab, env.g0))
            flows = [fl for fl in self._deposit_flows(env, b0) if fl[2] != "rec" or self.p.recombination_dissociation]
            p0 = np.ones(len(self.library))                    # the flows already carry the bare share
        else:
            for c, tab in self._bare_stack(shape).items():
                out[c] = p0 * np.einsum("sz,sz->s", f, self._interp_g0(tab, env.g0))
            flows = [fl for fl in self._deposit_flows(env)
                     if p0[fl[0]] > 0.0 and (fl[2] != "rec" or self.p.recombination_dissociation)]
        top = {}       # one loss grid per molecule, to its largest deposit: a yield depends on the bins below it only
        for s, z, which, *_ in flows:
            energies, _ = self._flow_energies(self.library[s], z, which)
            top[s] = max(top.get(s, LOSS_E_MAX_EV), float(energies.max()) + 5.0)
        for s, z, which, idx, rate, _ in flows:
            mol = self.library[s]
            sp = bare_species(mol, z)
            energies, W = self._flow_energies(mol, z, which)
            e_cl, Y = bare_event_yields(sp.freqs_cm, sp.einstein_a, bare_channels(mol, self.p), top[s])
            for c, y in Y.items():
                p = np.interp(energies * dos.CM_PER_EV, e_cl, y)
                pe = float(W[idx] @ p) if W is not None else float(p[idx])
                out.setdefault(c, np.zeros(len(self.library)))[s] += rate * p0[s] * pe
        out["C2H2"] = self.loss_rates(env)["C2H2"]
        out["explosions"] = self.chain_overflow(env)
        out["cr_beyond"] = self.cr_beyond(env)
        out["hot_beyond"] = self.hot_beyond(env)
        hot = self._hot_rates(env)
        out["knockout"] = (f * hot["knockout_c"]).sum(axis=1) if hot is not None else np.zeros(len(self.library))
        nc_lost = {name: n for m in self.library for name, n, *_ in bare_channels(m, self.p)}
        carbon = sum(nc_lost.get(c, 0) * v for c, v in out.items()) + 2.0 * out["C2H2"] + out["knockout"] \
            + self.nc * (out["explosions"] + out["cr_beyond"] + out["hot_beyond"])
        out["events"] = sum(out.values())
        out["carbon"] = carbon
        return out

    def loss_rates(self, env):
        """{"C2H2": C2H2 losses, "Hp": proton emissions} [s^-1 per molecule] at ``env``: the loss tables at its G0 and the
        deposits (``_deposit_losses``), weighted over charge and the emitting hydrogenation states N_H0 ... N_H0-3
        (``emission_weights``; the deeper states and the bare skeleton are not in them). A C2H2 loss destroys the molecule
        as the library holds it (C_{N-2}H_{N_H-2} is another)."""
        W = self.emission_weights(env)
        _, _, tab_c, tab_hp = self._ladders(field_shape(env))
        _, _, dlc, dlhp = self._deposit_losses(env)
        h = W.sum(axis=1)
        return {"C2H2": np.einsum("szk,szk->s", W, self._interp_g0(tab_c, env.g0)) + (h * dlc).sum(axis=1),
                "Hp": np.einsum("szk,szk->s", W, self._interp_g0(tab_hp, env.g0)) + (h * dlhp).sum(axis=1)}

    def chain_overflow(self, env):
        """Steps out of each molecule's ladder that the chain does not hold [s^-1 per molecule]: the ionisations, and the
        ions' charge transfers (counted since 2026-11-05, pahchem audit m10), that land past the top state
        (``_past_top``). With ``Params.charge_ladder = "cold_limit"`` these are the molecule's Coulomb explosions, a
        destruction rate: all of them, or past a barrier-limited top (``_routed``) the bare skeleton's share f(N_H = 0),
        the H-bearing rest emitting protons instead. With "library", EUV photons above Y_UNITY_EV out of the library's top
        charge, a truncation that is harmless where it is small against the chain's rates."""
        f = self.charge_matrix(env)
        past = self._past_top(env, self._single_up(env))[0]
        rate = (f * past).sum(axis=1)
        routed = self._routed()
        if routed.any():
            bare = (self._bare_slots(env) * past).sum(axis=1) if self._joint_on() else self._bare_share(env) * rate
            rate = np.where(routed, bare, rate)
        return rate

    def _bare_share(self, env):
        """p0[s]: the bare skeleton's share f(N_H = 0) of each molecule's hydrogenation chain at ``env``."""
        return np.array([float(c[0]) for c in self._chain_distributions(env)])

    def _bare_slots(self, env):
        """b0[s, slot]: the joint chain's shares f(Z, N_H = 0) of the bare skeleton (``Params.hydrogen_chain = "joint"``)."""
        b0 = np.zeros((len(self.library), len(self.zs)))
        for s, (lo, F, _) in enumerate(self._joint_distributions(env)):
            b0[s, lo:lo + F.shape[0]] = F[:, 0]
        return b0

    def band_fluxes(self, env):
        """(F[s, b] band power per molecule, weighted over charge and hydrogenation (``emission_weights``),
        {name: {Z: f}}) for the whole library."""
        f = self.charge_matrix(env)
        W = self.emission_weights(env)
        F = env.g0 * np.einsum("szk,szkb->sb", W, self._hydrogenation_bands(field_shape(env)))
        F = F + self._deposit_bands(env, W.sum(axis=1))      # captures at their actual rate, the Auger excitation
        fz = {m.name: {z: float(f[s, z - self.zmin]) for z in m.charges} for s, m in enumerate(self.library)}
        return F, fz

    def state_spectra(self, shape, nu_cm, fwhm_cm=None, shift_cm=None):
        """S[s, slot, i]: IR power per cm^-1 of every state at G0 = 1 on ``nu_cm``: fixed Lorentzians of ``fwhm_cm``
        and ``shift_cm`` (``emission.spectrum``; a missing one takes 15 or -15) if either is given, else the
        population's band convention (legacy: 15 / -15)."""
        nu = np.asarray(nu_cm, dtype=float)
        conv = self.band_convention if fwhm_cm is None and shift_cm is None else None
        key = ("spectra", shape, hashlib.sha1(nu.tobytes()).hexdigest(), fwhm_cm, shift_cm, conv)
        if key not in self._arrays:
            self.prepare(shape)
            S = np.zeros((len(self.library), len(self.zs), nu.size))
            for s, mol in enumerate(self.library):
                for z in self._live_charges(mol, shape):
                    b = self.basis(mol, z, shape)
                    S[s, z - self.zmin] = self._line_spectrum(mol, z, shape, b.freqs_cm, b.emitted, nu, fwhm_cm, shift_cm)
            self._arrays[key] = S
        return self._arrays[key]

    def hydrogenation_state_spectra(self, shape, nu_cm, fwhm_cm=None, shift_cm=None):
        """``StateSpectra``: IR power per cm^-1 of N_H0 - k of every live state with modes at G0 = 1 (``state_spectra``
        per k; the dehydrogenated states take the emission-temperature shares of N_H0)."""
        nu = np.asarray(nu_cm, dtype=float)
        conv = self.band_convention if fwhm_cm is None and shift_cm is None else None
        key = ("hspectra", shape, hashlib.sha1(nu.tobytes()).hexdigest(), fwhm_cm, shift_cm, conv)
        if key not in self._arrays:
            self.prepare(shape)
            if conv is not None and conv.kind == "tdep":
                self.prepare_temperature_shares(shape)
            rows, index = [], []
            for s, mol in enumerate(self.library):
                for z in self._live_charges(mol, shape):
                    for k, (f_, e_) in enumerate(self._state_modes(self.basis(mol, z, shape))):
                        if f_.size:
                            rows.append(self._line_spectrum(mol, z, shape, f_, e_, nu, fwhm_cm, shift_cm))
                            index.append((s, z - self.zmin, k))
            s_, slot, k_ = np.array(index, dtype=np.int64).reshape(-1, 3).T
            self._arrays[key] = StateSpectra(np.array(rows).reshape(-1, nu.size), s_, slot, k_, len(self.library))
        return self._arrays[key]

    def spectra(self, env, nu_cm, **profile):
        """P[s, i]: the spectrum of every molecule, its charge and hydrogenation states at ``env`` (weighted by
        ``emission_weights``) [cm^-1 s^-1 per cm^-1]."""
        S = self.hydrogenation_state_spectra(field_shape(env), nu_cm, **profile)
        return env.g0 * S.combine(self.emission_weights(env)) + self.deposit_spectra(env, nu_cm, **profile)

    def state_band_fluxes(self, mol, z, env):
        """Band power per molecule of charge z [cm^-1 s^-1], one value per band."""
        mol = self._own(mol)
        b = self.basis(mol, z, field_shape(env))
        return env.g0 * (b.emitted @ self._window_matrix(mol, z, field_shape(env), b.freqs_cm))

    def _interp_g0(self, table, g0):
        """Log-log interpolation of rate tables (..., K_H_G0_GRID) at ``g0``. Below the grid (shielded and dark gas, where
        cosmic rays matter) each rate continues its first interval's log-log slope: 1 where single photons dominate at
        G0 = 1, up to 2.02 where two photons do (large PAHs, 2026-11-04 tables), whose single-photon floor below is then
        not followed."""
        if not 0.0 < g0 <= K_H_G0_GRID[-1]:
            raise ValueError(f"G0 {g0:g} outside (0, {K_H_G0_GRID[-1]:g}] of the k_H table")
        lt = np.log(np.maximum(table, 1e-300))
        if g0 < K_H_G0_GRID[0]:
            slope = (lt[..., 1] - lt[..., 0]) / (np.log(K_H_G0_GRID[1]) - np.log(K_H_G0_GRID[0]))
            out = np.exp(lt[..., 0] + slope * (np.log(g0) - np.log(K_H_G0_GRID[0])))
            return np.where(table[..., 0] == 0.0, 0.0, out)
        x = np.log(g0)
        i = max(min(np.searchsorted(np.log(K_H_G0_GRID), x), K_H_G0_GRID.size - 1), 1)
        lo, hi = np.log(K_H_G0_GRID[i - 1]), np.log(K_H_G0_GRID[i])
        out = np.exp(lt[..., i - 1] + (lt[..., i] - lt[..., i - 1]) * (x - lo) / (hi - lo))
        # a rate that is 0 at both nodes stays 0, not 1e-300: no photon deposits (X-rays only), whose tiny floor would
        # make the hydrogenation chain's rate ratios overflow
        return np.where((table[..., i - 1] == 0.0) & (table[..., i] == 0.0), 0.0, out)

    def intact_fractions(self, env, method="chain"):
        """f(N_H0) of every molecule at ``env`` (module docstring, Hydrogenation).

        ``method="single"`` gives the old first-step ratio k_add / (k_add + k_H).
        """
        if method == "single":
            f = self.charge_matrix(env)
            _, lkh, _ = self._stack(field_shape(env))
            k_h = self._interp_g0(np.exp(lkh), env.g0)
            k_add_z = self._addition_rates(env)
            return np.sum(f * k_add_z / (k_add_z + k_h), axis=1)
        if method != "chain":
            raise ValueError(f"method must be 'chain' or 'single', not {method!r}")
        return np.array([p[mol.nh] for p, mol in zip(self._chain_distributions(env), self.library)])

    def _addition_rates(self, env):
        """H addition rate of every state [s^-1], (s, slot): 1.4e-10 n(H) for cations, 0.07 geometric for the rest (anions
        none with ``Params.anion_h_addition = "none"``)."""
        z = np.array(self.zs)[None, :]
        neutral = 0.07 * self.p.addition_neutral_scale * chemistry.cross_section_rate(
            env.temperature_k, CORONENE_GEOMETRIC_A2 * self.nc[:, None] / 24.0)
        out = np.where(z > 0, self.p.addition_cation, neutral) * env.n_h
        return np.where(z < 0, 0.0, out) if self.p.anion_h_addition == "none" else out

    def _chain_distributions(self, env, f=None, deposit_losses=None):
        """[f(N_H), N_H = 0 ... N_H0] of every molecule: the chain, its rates averaged over the charge fractions (``f``, by
        default ``charge_matrix``'s). Those of the last CHAIN_MEMO environments are kept: a spectral fit asks for each
        cell's several times, scan after scan (not those of a given ``f``, ``charge_matrix``'s iterations, which pass their
        ``_deposit_losses`` too)."""
        given = f is not None
        if self._joint_on():                                 # the N_H marginal of the joint chain, N_H0 + superh_levels long
            if given or deposit_losses is not None:
                raise ValueError("the joint chain takes no charge fractions or deposit losses: it solves them itself")
            return [F.sum(axis=0) for _, F, _ in self._joint_distributions(env)]
        if not given and env in self._chains:
            return self._chains[env]
        f = self.charge_matrix(env) if not given else f
        tab_h, tab_h2, _, tab_hp = self._ladders(field_shape(env))
        # charge-averaged loss rates; a proton emitted is an H lost (its charge step is the charge chain's)
        lh = np.einsum("sz,szk->sk", f, self._interp_g0(tab_h, env.g0) + self._interp_g0(tab_hp, env.g0))
        lh2 = np.einsum("sz,szk->sk", f, self._interp_g0(tab_h2, env.g0))
        dlh, dlh2, _, dlhp = self._deposit_losses(env, f) if deposit_losses is None else deposit_losses   # the deposits
        lh, lh2 = lh + dlh + dlhp, lh2 + dlh2
        k_add = np.einsum("sz,sz->s", f, self._addition_rates(env))
        # every H-bearing state: the protons of the steps past a barrier-limited top (one H per charge past it, whatever
        # N_H), and the thermal ions' H knock-outs (N_H times the rate per atom, pahchem audit m3)
        routed = self._routed()
        per_mol = (np.where(routed, (f * self._past_top(env, self._single_up(env))[1]).sum(axis=1), 0.0)
                   if routed.any() else np.zeros(len(self.library)))
        hot = self._hot_rates(env)
        per_h = (f * hot["knockout_h"]).sum(axis=1) if hot is not None else np.zeros(len(self.library))
        out = [_chain_distribution(mol.nh, mol.edges.get("solo", 0), lh[s], lh2[s], k_add[s], per_mol[s], per_h[s])
               for s, mol in enumerate(self.library)]
        if given:
            return out
        if len(self._chains) >= CHAIN_MEMO:
            del self._chains[next(iter(self._chains))]                         # the oldest
        self._chains[env] = out
        return out

    def hydrogenation_fractions(self, env):
        """(h[s, k], deeper[s]): steady-state shares of N_H0 - k for the states that emit (k < ``N_EMITTING``,
        N_H >= 1, as far as the ladder has modes) and of the rest, which is not modelled (deeper states, whose
        stripped modes are not good enough: ``PAHSpecies.stripped``). With ``Params.hydrogen_chain = "joint"`` the
        super-hydrogenated states are in k = 0: they emit as the parent (their CH2 groups' own modes are not computed)."""
        h = np.zeros((len(self.library), N_EMITTING))
        for s, (p, mol) in enumerate(zip(self._chain_distributions(env), self.library)):
            for k in range(min(N_EMITTING, mol.nh)):          # N_H = N_H0 - k >= 1
                h[s, k] = p[mol.nh - k]
            h[s, 0] += p[mol.nh + 1:].sum()                   # "joint": the super-hydrogenated states emit as the parent
        return h, np.clip(1.0 - h.sum(axis=1), 0.0, 1.0)

    def emission_weights(self, env):
        """W[s, slot, k]: share of molecule s in charge slot and hydrogenation N_H0 - k. With
        ``dehydrogenated_emission`` f(Z) times ``hydrogenation_fractions`` (charge and hydrogenation taken as
        independent, as in the chain), or with ``Params.hydrogen_chain = "joint"`` the joint shares, the
        super-hydrogenated states in k = 0; otherwise f(Z) at k = 0."""
        f = self.charge_matrix(env)
        if self._joint_on() and self.dehydrogenated_emission:       # the joint shares, not a product
            W = np.zeros((len(self.library), len(self.zs), N_EMITTING))
            for s, ((lo, F, _), mol) in enumerate(zip(self._joint_distributions(env), self.library)):
                rows = slice(lo, lo + F.shape[0])
                for k in range(min(N_EMITTING, mol.nh)):
                    W[s, rows, k] = F[:, mol.nh - k]
                W[s, rows, 0] += F[:, mol.nh + 1:].sum(axis=1)     # the super-hydrogenated states emit as the parent
            return W
        if not self.dehydrogenated_emission:
            h = np.zeros((len(self.library), N_EMITTING))
            h[:, 0] = 1.0
        else:
            h = self.hydrogenation_fractions(env)[0]
        return f[:, :, None] * h[:, None, :]

    def emitting_fractions(self, env):
        """Share of each molecule that emits: f(N_H0 ... N_H0-3) with ``dehydrogenated_emission``, else the
        intact fraction (what the callers' ``intact_min`` filters on)."""
        if self.dehydrogenated_emission:
            return self.hydrogenation_fractions(env)[0].sum(axis=1)
        return self.intact_fractions(env)

    def intact_fraction(self, mol, env):
        return float(self.intact_fractions(env)[self._index[id(mol)]])

    # -- the joint (Z, N_H) chain (Params.hydrogen_chain = "joint", 2026-11-08) ---------------------------------------------
    def _joint_on(self):
        return self.p.hydrogen_chain == "joint"

    def prepare_superh(self, shape, workers=None):
        """``superh_loss_table`` of every live state under ``shape``: from memory, the disk cache (``cache_dir``/superh) or
        computed in parallel."""
        todo = []
        for mol in self.library:
            for z in self._live_charges(mol, shape):
                ck = ("superh", shape, z, self.de_cm, self.e_max_cm, self.p)
                if ck in mol._cache:
                    continue
                inp = _superh_inputs(mol, z, shape, self.de_cm, self.e_max_cm, self.p)
                f = None if self.cache_dir is None else self.cache_dir / "superh" / f"{_superh_key(inp)}.npz"
                if f is not None and f.exists():
                    with np.load(f) as d:
                        mol._cache[ck] = d["k"]
                else:
                    todo.append((mol, ck, f, inp))

        def keep(t, r):
            mol, ck, f, _ = t
            mol._cache[ck] = r
            if f is not None:
                f.parent.mkdir(parents=True, exist_ok=True)
                tmp = f.with_suffix(f".{os.getpid()}.tmp.npz")
                np.savez(tmp, k=r)
                os.replace(tmp, f)
        workers = default_workers(workers)
        if workers > 1 and len(todo) > 2:
            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor(max_workers=workers) as ex:
                for t, r in zip(todo, ex.map(_build_superh, [t[3] for t in todo])):
                    keep(t, r)
        else:
            for t in todo:
                keep(t, superh_loss_table(**t[3]))

    def _superh_stack(self, shape):
        """S[s, slot, j - 1, K_H_G0_GRID]: the photon-driven loss of an extra H of N_H0 + j, j = 1 ... superh_levels
        (``superh_loss_table``); 0 where the state is not live."""
        key = ("superh", shape)
        if key not in self._arrays:
            self.prepare_superh(shape)
            S = np.zeros((len(self.library), len(self.zs), int(self.p.superh_levels), K_H_G0_GRID.size))
            for s, mol in enumerate(self.library):
                for z in self._live_charges(mol, shape):
                    S[s, z - self.zmin] = mol._cache[("superh", shape, z, self.de_cm, self.e_max_cm, self.p)]
            self._arrays[key] = S
        return self._arrays[key]

    def _threshold_factors(self, shape):
        """{s: r[slot, m]}, m = N_H = 0 ... N_H0 + superh_levels: each live state's valence photoionisation rate at N_H = m
        over the intact state's. The IPs (Z >= 0) of the dehydrogenated states (m < N_H0) rise by ``Params.ip_shift_per_h``
        per missing H: the part of Andrews+16's rule the literature supports (Malloci+08, B3LYP Delta-SCF of coronene's
        even states, ~0.1 eV per H; the 2026-11-08 check). The EA, and every threshold of the super-hydrogenated states,
        keep the intact molecule's: aryl radicals' EAs sit 1.0-1.6 eV above their parents' (a step, not 0.1 eV per H) and
        1-hydronaphthyl's IP 1.5 eV below naphthalene's, so a smooth shift there would be wrong and no rule is
        established. The shift moves the intact state's yield on E_PH rigidly (its yield at E - shift), with the same
        sigma and field (exact for the threshold laws, Jochims and Wenzel). 1 where the intact state has none, or with no
        shift."""
        key = ("ip_shift", shape)
        if key not in self._arrays:
            field1 = _unit_field(shape)
            d, K = float(self.p.ip_shift_per_h), int(self.p.superh_levels)
            out = {}
            for s, mol in enumerate(self.library):
                m = np.arange(mol.nh + K + 1)
                r = np.ones((len(self.zs), m.size))
                if d != 0.0:
                    for z in (q for q in self._live_charges(mol, shape) if q >= 0):
                        sigma = self._valence_sigma(mol, z, shape)
                        y = _yield(mol, z, self.p)
                        base = photo.ionisation_rate(field1, sigma, y, E_PH)
                        if base > 0.0:
                            r[z - self.zmin] = [photo.ionisation_rate(field1, sigma, np.interp(E_PH - d * max(mol.nh - mm, 0),
                                                                                               E_PH, y, left=0.0, right=y[-1]),
                                                                      E_PH) / base for mm in m]
                out[s] = r
            self._arrays[key] = out
        return self._arrays[key]

    def _joint_distributions(self, env):
        """[(lo, F[slot - lo, N_H], h2)] of every molecule at ``env``: the steady state of one (Z, N_H) network over its live
        charges (slots lo, lo + 1, ...) and N_H = 0 ... N_H0 + superh_levels, and h2, the H2 it releases per molecule per
        second (H2 loss and Eley-Rideal abstraction). The last CHAIN_MEMO environments are kept.

        - **Charge steps** (dN_H = 0): those of ``charge_matrix``'s chain (photoionisation and detachment, C 1s,
          double ionisation, passages and ions' charge transfer as ``_jump_rates``; captures), the same at every N_H except
          the valence single photoionisation, scaled by ``_threshold_factors``. Steps past the top are dropped, as there,
          or past a barrier-limited top (``_routed``) land on it with an H less per charge carried past.
        - **H steps** per charge: the ladders' photon-driven H, H2 loss and protons (dZ = -1, dN_H = -1) of N_H0 ...
          N_H0-3, deeper states by ``_chain_rates``' mapping; the thermal ions' H knock-outs; H addition as
          ``_addition_rates`` below N_H0. Above it the super-hydrogenated states (BaselineModel's chemistry): the loss of an
          extra H (``_superh_stack``), H addition (cations ``Params.addition_cation``, the others the geometric rate
          with barriers of 0.06 eV at N_H0 and 0.035 eV at N_H0 + 2), Eley-Rideal abstraction (``Params.eley_rideal_a2``,
          one H2 each); associative detachment of anions below N_H0 (``Params.associative_detachment``: every anion
          below the top with "all").
        - **Deposits** (``_flow_terms``: recombination, photoionisation products, C 1s products, passages): each event
          lands as its charge step does, and with its product's loss probabilities (``_response``; super-H states
          ``superh_event_yields``) one or two H less, or a proton less: those shares of its charge step are moved, not
          added.
        """
        if env in self._joints:
            return self._joints[env]
        from pycalima.pahchem.network import gth_banded

        shape = field_shape(env)
        live = self._live(shape)
        p, K = self.p, int(self.p.superh_levels)
        T, n_h, g0 = env.temperature_k, env.n_h, env.g0
        k_up = self._k_up(shape)
        up = g0 * k_up
        if getattr(env, "ions", ()):
            up = up + self._ion_up(env)
        steps = self._jump_rates(env, up)
        steps = [up] if steps is None else steps
        ks = self._kshell_stack(shape)
        vps = np.maximum(g0 * (k_up - (ks["over"] if ks is not None else 0.0) - self._double_stack(shape)), 0.0)
        down = self._capture_rates(T) * env.n_e
        tab_h, tab_h2, _, tab_hp = self._ladders(shape)
        lh, lh2, lhp = (self._interp_g0(t, g0) for t in (tab_h, tab_h2, tab_hp))
        S = self._interp_g0(self._superh_stack(shape), g0) if K else None
        k_add = self._addition_rates(env)
        hot = self._hot_rates(env)
        per_h = hot["knockout_h"] if hot is not None else np.zeros_like(k_add)
        routed = self._routed()
        r_ip = self._threshold_factors(shape)
        er = chemistry.cross_section_rate(T, p.eley_rideal_a2) * n_h
        ad_all = anion_associative_detachment(replace(p, associative_detachment="all")) * n_h
        terms = self._flow_terms(env)
        use = lambda which: which != "rec" or p.recombination_dissociation      # noqa: E731
        self._responses(dict.fromkeys([(s, z, w) for s, plain, _ in terms for z, w, *_ in plain if use(w)]
                                      + [(s, zp, "heat") for s, _, passages in terms for zp, _ in passages]))
        out = []
        for s, mol in enumerate(self.library):
            on = np.nonzero(live[s])[0]
            lo, nz = int(on[0]), int(on[-1] - on[0] + 1)
            if on.size != nz:
                raise ValueError(f"{mol.name}: its live charges are not contiguous")
            nh0, nn = mol.nh, mol.nh + K + 1
            sl = slice(lo, lo + nz)
            zc = np.array(self.zs[sl])
            n = nz * nn
            R = np.zeros((n, n))
            H2 = np.zeros((nz, nn))
            iz, m = np.meshgrid(np.arange(nz), np.arange(nn), indexing="ij")

            def add(rate, dz, dnh, h2=0):
                rate = np.broadcast_to(np.asarray(rate, dtype=float), (nz, nn))
                tz, tm = iz + dz, m + dnh
                ok = rate > 0.0
                out_ = ok & ((tz < 0) | (tz >= nz) | (tm < 0) | (tm >= nn))
                if out_.any():
                    raise ValueError(f"{mol.name}: a rate of dZ {dz}, dN_H {dnh} leads outside the joint network")
                np.add.at(R, (m[ok] * nz + iz[ok], tm[ok] * nz + tz[ok]), rate[ok])
                if h2:
                    H2[ok] += h2 * rate[ok]

            # charge steps, per dZ, before the deposits move their shares
            C = {j: np.repeat(k[s, sl][:, None], nn, axis=1) for j, k in enumerate(steps, start=1)}
            C[1] = np.maximum(C[1] + (r_ip[s][sl] - 1.0) * vps[s, sl][:, None], 0.0)
            C[-1] = np.repeat(down[s, sl][:, None], nn, axis=1)
            # H steps of each charge
            HL, H2L, HP, A = (np.zeros((nz, nn)) for _ in range(4))
            for a in range(nz):
                kH, kH2 = _chain_rates(nh0, mol.edges.get("solo", 0), lh[s, lo + a], lh2[s, lo + a])
                HL[a, :nh0 + 1], H2L[a, :nh0 + 1] = kH, kH2
                HP[a, :nh0 + 1] = _deep_rates(nh0, lhp[s, lo + a])
                A[a, :nh0] = k_add[s, lo + a]
            HL += per_h[s, sl][:, None] * np.arange(nn)[None, :]
            if K:
                HL[:, nh0 + 1:] += S[s, sl]
                sigma_geo = CORONENE_GEOMETRIC_A2 * mol.nc / 24.0
                for j in range(K):                                 # N_H0 + j -> N_H0 + j + 1
                    barrier = {0: 0.06, 2: 0.035}.get(j, 0.0)
                    neutral = p.addition_neutral_scale * chemistry.cross_section_rate(T, sigma_geo, barrier) * n_h
                    cation = p.addition_cation if p.addition_cation_superh is None else p.addition_cation_superh
                    A[:, nh0 + j] = np.where(zc > 0, cation * n_h,
                                             np.where((zc < 0) & (p.anion_h_addition == "none"), 0.0, neutral))
            ER = np.where(np.arange(nn)[None, :] > nh0, er, 0.0) * np.ones((nz, 1))
            AD = np.zeros((nz, nn))
            if zc[0] == -1 and nz > 1:
                top_m = nh0 if p.associative_detachment == "dehydrogenated" else nn - 1
                AD[0, :top_m] = ad_all
            if (HP[0] > 0.0).any():
                raise ValueError(f"{mol.name}: proton emission out of its lowest live charge")
            # deposits: a share of each event's charge step lands with H, H2 or a proton less
            for z_prod, which, idx, i_src, c, pe in self._joint_deposits(s, mol, terms, use, r_ip[s], lo, nz, nn):
                a, b = i_src - lo, (z_prod - self.zmin) - lo
                dz = b - a
                moved = c * (pe["H"] + pe["H2"] + pe["Hp"])
                if dz != 0:
                    if dz not in C:
                        raise ValueError(f"{mol.name}: a deposit of dZ {dz} has no charge step to move")
                    short = moved - C[dz][a]
                    if np.any(short > ROUND_OFF * np.maximum(C[dz][a], 1e-300)):
                        raise ValueError(f"{mol.name}: deposits of dZ {dz} exceed the charge step they ride on")
                    C[dz][a] = np.maximum(C[dz][a] - moved, 0.0)
                rate = np.zeros((nz, nn))
                for key, dnh, dzz, h2 in (("H", -1, 0, 0), ("H2", -2, 0, 1), ("Hp", -1, -1, 0)):
                    rate[:] = 0.0
                    rate[a] = c * pe[key]
                    if rate.any():
                        add(rate, dz + dzz, dnh, h2)
            add(HL, 0, -1)
            add(H2L, 0, -2, h2=1)
            add(HP, -1, -1)
            add(ER, 0, -1, h2=1)
            add(A, 0, +1)
            add(AD, +1, +1)
            top_live = lo + nz - 1 == max(mol.charges) - self.zmin
            for dz, rate in C.items():
                if dz < 0:
                    add(np.where(iz + dz >= 0, rate, 0.0), dz, 0)
                    continue
                inside = iz + dz < nz
                add(np.where(inside, rate, 0.0), dz, 0)
                if routed[s] and top_live:                         # past a barrier-limited top: a proton per charge
                    past = iz + dz - (nz - 1)
                    land = ~inside & (m >= past)
                    if land.any():
                        np.add.at(R, (m[land] * nz + iz[land], (m[land] - past[land]) * nz + nz - 1), rate[land])
            F = _joint_steady_state(R, mol.name, gth_banded).reshape(nn, nz).T
            out.append((lo, F, float(np.sum(F * H2))))
        if len(self._joints) >= CHAIN_MEMO:
            del self._joints[next(iter(self._joints))]
        self._joints[env] = out
        return out

    def _joint_deposits(self, s, mol, terms, use, r_ip, lo, nz, nn):
        """[(z_prod, which, index, source slot, c[N_H], {"H", "H2", "Hp": P[N_H]})] of molecule s: each deposit flow of
        ``_flow_terms`` per molecule in its source state (a valence photoionisation's scaled by the threshold shift,
        ``_threshold_factors``), and its product's loss probabilities at each N_H: ``_response``'s for N_H0 ... N_H0-3, the
        deeper by ``_chain_rates``' mapping, the extra H's of super-hydrogenated states from ``superh_event_yields``."""
        p, K, nh0 = self.p, int(self.p.superh_levels), mol.nh
        solo = mol.edges.get("solo", 0)
        _, plain, passages = terms[s]
        flows = [(z, which, idx, i, a * b) for z, which, idx, i, a, b, *_ in plain if use(which) and a * b != 0.0]
        flows += [(zp, "heat", int(node), i, float(k[node])) for zp, parents in passages for i, k in parents
                  for node in np.nonzero((k > 0.0) & (HEAT_NODES_EV > 0.0))[0]]
        top = max([LOSS_E_MAX_EV] + [float(self._flow_energies(mol, z, w)[0].max()) + 5.0 for z, w, *_ in flows])
        out = []
        for z, which, idx, i, c in flows:
            members = self._response(mol, z, which)[1][idx]
            P4 = np.zeros((3, 4))
            for k_, (_, ph, ph2, _pc, php) in enumerate(members[:4]):
                P4[:, k_] = ph, ph2, php
            pH, pH2 = _chain_rates(nh0, solo, P4[0], P4[1])
            pe = {"H": np.zeros(nn), "H2": np.zeros(nn), "Hp": np.zeros(nn)}
            pe["H"][:nh0 + 1], pe["H2"][:nh0 + 1], pe["Hp"][:nh0 + 1] = pH, pH2, _deep_rates(nh0, P4[2])
            if K:
                sp = mol.species[z]
                e_cl, Y = superh_event_yields(sp.freqs_cm, _einstein_a(sp, p), K, superh_e0(p, z), p.h_superh_ds, top)
                energies, W = self._flow_energies(mol, z, which)
                for j in range(1, K + 1):
                    y = np.interp(energies * dos.CM_PER_EV, e_cl, Y[j - 1])
                    pe["H"][nh0 + j] = float(W[idx] @ y) if W is not None else float(y[idx])
            dz = (z - self.zmin) - i
            photo_single = isinstance(which, tuple) and which[0] == "ion" and dz == 1
            cm = c * (r_ip[i] if photo_single else np.ones(nn))
            out.append((z, which, idx, i, cm, pe))
        return out

    def joint_fractions(self, env):
        """[(charges, F[Z, N_H])] of every molecule at ``env`` with ``Params.hydrogen_chain = "joint"``: its live charges
        and the steady-state shares of each (Z, N_H), N_H = 0 ... N_H0 + superh_levels (``_joint_distributions``)."""
        if not self._joint_on():
            raise ValueError('joint_fractions needs Params.hydrogen_chain = "joint"')
        return [(self.zs[lo:lo + F.shape[0]], F) for lo, F, _ in self._joint_distributions(env)]

    def hydrogenation_summary(self, env):
        """{name: [s]} of the hydrogenation at ``env``: "x" <N_H>/N_H0, the shares "intact" (N_H0), "superh" (> N_H0),
        "dehydrogenated" (< N_H0), "bare" (0) and "at_cap" (N_H0 + superh_levels, the top of the ladder: where it holds
        more than a few per cent, the cap, not the chemistry, sets the super-hydrogenation), and "h2", the H2 released per
        molecule per second (H2 loss and, with super-hydrogenated states, Eley-Rideal abstraction; "joint" only, else NaN)."""
        out = {k: np.zeros(len(self.library)) for k in ("x", "intact", "superh", "dehydrogenated", "bare", "at_cap", "h2")}
        joint = self._joint_distributions(env) if self._joint_on() else None
        for s, (pn, mol) in enumerate(zip(self._chain_distributions(env), self.library)):
            nh = np.arange(pn.size)
            out["x"][s] = float(pn @ nh) / mol.nh
            out["intact"][s], out["bare"][s] = pn[mol.nh], pn[0]
            out["superh"][s], out["dehydrogenated"][s] = pn[mol.nh + 1:].sum(), pn[:mol.nh].sum()
            out["at_cap"][s] = pn[-1] if self._joint_on() and self.p.superh_levels > 0 else 0.0
            out["h2"][s] = joint[s][2] if joint is not None else np.nan
        return out



def _joint_steady_state(R, name, gth_banded):
    """p[state] of the joint network with rates R[i, j] (i -> j): on its closed class of states (the others, transient,
    hold none), by ``network.gth_banded`` within the band its rates span. Raises if there is more than one closed class
    (no unique steady state)."""
    import scipy.sparse
    import scipy.sparse.csgraph

    n = R.shape[0]
    A = scipy.sparse.csr_matrix(R)
    ncomp, lab = scipy.sparse.csgraph.connected_components(A, directed=True, connection="strong")
    keep = np.arange(n)
    if ncomp > 1:
        rows, cols = A.nonzero()
        leaky = np.zeros(ncomp, dtype=bool)
        leaky[np.unique(lab[rows][lab[rows] != lab[cols]])] = True
        closed = np.nonzero(~leaky)[0]
        if closed.size != 1:
            raise ValueError(f"{name}: the joint network has {closed.size} closed classes, no unique steady state")
        keep = np.nonzero(lab == closed[0])[0]
    sub = R[np.ix_(keep, keep)]
    i, j = np.nonzero(sub)
    p_ = gth_banded(sub, int(np.abs(i - j).max()) if i.size else 0)
    if p_ is None:
        raise ValueError(f"{name}: the closed class of the joint network is not irreducible")
    out = np.zeros(n)
    out[keep] = p_
    return out


def _chain_rates(nh0, n_solo, lh, lh2, per_mol=0.0, per_h=0.0):
    """(kH[m], kH2[m]), m = N_H = 0 ... N_H0, of the chain. ``lh``, ``lh2``: H and H2 loss rates of N_H0 ... N_H0-3.
    Deeper states: even (odd) N_H take the N_H0-2 (N_H0-3) rates, H loss scaled by N_H when even (degeneracy N_H). H2
    loss only for even N_H above the solo count. ``per_mol`` adds to every H-bearing state's H loss as it is (protons
    from past the ladder top, one per event), ``per_h`` times N_H (knock-outs, one per H atom)."""
    n = nh0 + 1
    kH, kH2 = np.zeros(n), np.zeros(n)
    for m in range(nh0 + 1):                     # m = number of H
        k = nh0 - m
        if k < 4:
            kH[m], kH2[m] = lh[k], lh2[k]
        else:
            ref = 2 if (nh0 - 2) % 2 == m % 2 else 3
            ref_n = nh0 - ref
            kH[m] = lh[ref] * (m / ref_n if m % 2 == 0 and ref_n > 0 else 1.0)
            kH2[m] = lh2[ref] if (m % 2 == 0 and m > n_solo) else 0.0
        if m >= 1:
            kH[m] += per_mol + per_h * m
    kH[0] = kH2[0] = 0.0
    if n > 1:
        kH2[1] = 0.0
    return kH, kH2


def _chain_distribution_numpy(nh0, n_solo, lh, lh2, k_add, per_mol=0.0, per_h=0.0):
    """Steady-state f(N_H), N_H = 0 ... N_H0, of the chain (H loss, H2 loss, H addition up to N_H0; ``_chain_rates``) by
    least squares on its generator: the reference of ``_chain_distribution``, and its solve when no H is added. Entries
    below ~1e-16 of the largest are round-off (pahchem audit m2)."""
    n = nh0 + 1
    kH, kH2 = _chain_rates(nh0, n_solo, lh, lh2, per_mol, per_h)
    Q = np.zeros((n, n))
    for m in range(n):
        if m >= 1:
            Q[m, m - 1] += kH[m]
        if m >= 2:
            Q[m, m - 2] += kH2[m]
        if m < nh0:
            Q[m, m + 1] += k_add
        Q[m, m] = -Q[m].sum()
    A = np.vstack([Q.T, np.ones(n)])
    b = np.zeros(n + 1)
    b[-1] = 1.0
    p = np.clip(np.linalg.lstsq(A, b, rcond=None)[0], 0.0, None)
    return p / p.sum()


def _chain_exact(kH, kH2, k_add):
    """f(N_H) of the chain with ``_chain_rates``' kH, kH2 and the addition k_add > 0, exactly: no net flux crosses any
    cut, f(m) k_add = f(m+1) (kH + kH2)(m+1) + f(m+2) kH2(m+2), recursed down from N_H0 in logs (pahchem audit m2: the
    least squares left f(N_H = 0) of C54H18 at 2.6e-17 against 2.1e-43)."""
    n = kH.size
    lf = np.full(n, -np.inf)
    lf[n - 1] = 0.0
    lk = np.log(k_add)
    for m in range(n - 2, -1, -1):
        a = lf[m + 1] + np.log(kH[m + 1] + kH2[m + 1]) if kH[m + 1] + kH2[m + 1] > 0.0 else -np.inf
        b = lf[m + 2] + np.log(kH2[m + 2]) if m + 2 < n and kH2[m + 2] > 0.0 else -np.inf
        hi = max(a, b)
        lf[m] = -np.inf if hi == -np.inf else hi + np.log(np.exp(a - hi) + np.exp(b - hi)) - lk
    p = np.exp(lf - lf.max())
    return p / p.sum()


if HAVE_NUMBA:
    @njit(cache=True)
    def _chain_distribution_numba(nh0, n_solo, lh, lh2, k_add, per_mol, per_h):
        n = nh0 + 1
        kH, kH2 = np.zeros(n), np.zeros(n)
        for m in range(nh0 + 1):
            k = nh0 - m
            if k < 4:
                kH[m], kH2[m] = lh[k], lh2[k]
            else:
                ref = 2 if (nh0 - 2) % 2 == m % 2 else 3
                ref_n = nh0 - ref
                kH[m] = lh[ref] * (m / ref_n if m % 2 == 0 and ref_n > 0 else 1.0)
                kH2[m] = lh2[ref] if (m % 2 == 0 and m > n_solo) else 0.0
            if m >= 1:
                kH[m] += per_mol + per_h * m
        kH[0] = kH2[0] = 0.0
        if n > 1:
            kH2[1] = 0.0
        lf = np.full(n, -np.inf)
        lf[n - 1] = 0.0
        lk = np.log(k_add)
        for m in range(n - 2, -1, -1):
            a = lf[m + 1] + np.log(kH[m + 1] + kH2[m + 1]) if kH[m + 1] + kH2[m + 1] > 0.0 else -np.inf
            b = lf[m + 2] + np.log(kH2[m + 2]) if m + 2 < n and kH2[m + 2] > 0.0 else -np.inf
            hi = max(a, b)
            if hi == -np.inf:
                lf[m] = -np.inf
            else:
                lf[m] = hi + np.log(np.exp(a - hi) + np.exp(b - hi)) - lk
        p = np.exp(lf - lf.max())
        return p / p.sum()


def _anderson(g, x0, tol, it_max, what, m=5):
    """The fixed point x = g(x) of charge fractions (one row per molecule, each summing to 1), from ``x0``, by Anderson
    mixing of the last ``m`` iterates, molecule by molecule (the rows are coupled only within themselves), in ln x: the
    fractions span hundreds of decades, and there no mixed iterate turns negative or puts molecules in a state that holds
    none (x = 0). The least squares is weighted by the fractions, so the states that hold molecules lead it; each mixed row
    is normalised to 1. Converged when g moves no entry by more than ``tol`` = (absolute, relative) of the entry; raises
    past ``it_max`` iterations."""
    atol, rtol = tol
    x = x0
    hist = [[] for _ in range(x0.shape[0])]                  # per row: [(ln x, ln g(x))] on the states that hold molecules
    for _ in range(it_max):
        gx = g(x)
        step = float(np.max(np.abs(gx - x) / (atol + rtol * np.abs(gx))))
        if step <= 1.0:
            return gx
        new = gx.copy()
        for r in range(gx.shape[0]):
            on = (gx[r] > 0.0) & (x[r] > 0.0)
            if not np.array_equal(on, gx[r] > 0.0):              # g fills a state x left empty: take its step
                hist[r] = []
                continue
            if hist[r] and not np.array_equal(on, hist[r][-1][2]):
                hist[r] = []                                 # the states holding molecules changed: start again
            hist[r] = (hist[r] + [(np.log(x[r][on]), np.log(gx[r][on]), on)])[-(m + 1):]
            if len(hist[r]) < 2:
                continue
            Y = np.array([h[0] for h in hist[r]])
            G = np.array([h[1] for h in hist[r]])
            w = np.sqrt(gx[r][on])                           # the fractions lead the least squares
            F = (G - Y) * w
            dF, dG = np.diff(F, axis=0).T, np.diff(G, axis=0).T
            gamma = np.linalg.lstsq(dF, F[-1], rcond=None)[0]
            ly = G[-1] - dG @ gamma
            row = np.zeros_like(gx[r])
            row[on] = np.exp(ly - ly.max())
            new[r] = row / row.sum()
        x = new
    raise ValueError(f"{what} not converged in {it_max} iterations (last step {step:.1e} of the tolerance)")


def _deep_rates(nh0, rates4):
    """k[N_H], N_H = 0 ... N_H0, of a rate given for N_H0 ... N_H0-3 (``rates4``), with the chain's mapping of the deeper
    states (``_chain_distribution_numpy``: the N_H0-2 or N_H0-3 rate of the same parity, scaled by N_H when even); none
    for the bare skeleton."""
    k = np.zeros(nh0 + 1)
    for m in range(1, nh0 + 1):
        j = nh0 - m
        if j < 4:
            k[m] = rates4[j]
        else:
            ref = 2 if (nh0 - 2) % 2 == m % 2 else 3
            ref_n = nh0 - ref
            k[m] = rates4[ref] * (m / ref_n if m % 2 == 0 and ref_n > 0 else 1.0)
    return k


def _chain_distribution(nh0, n_solo, lh, lh2, k_add, per_mol=0.0, per_h=0.0):
    """Steady-state f(N_H), N_H = 0 ... N_H0, of the chain (``_chain_rates``): exact (``_chain_exact``, through its numba
    copy when available); without H addition the least squares of ``_chain_distribution_numpy``."""
    if not k_add > 0.0:
        return _chain_distribution_numpy(nh0, n_solo, lh, lh2, k_add, per_mol, per_h)
    if HAVE_NUMBA:
        return _chain_distribution_numba(int(nh0), int(n_solo), np.ascontiguousarray(lh, dtype=float),
                                         np.ascontiguousarray(lh2, dtype=float), float(k_add), float(per_mol),
                                         float(per_h))
    return _chain_exact(*_chain_rates(nh0, n_solo, lh, lh2, per_mol, per_h), float(k_add))


def _chain_intact(nh0, n_solo, lh, lh2, k_add):
    """f(N_H0) of ``_chain_distribution``."""
    return float(_chain_distribution(nh0, n_solo, lh, lh2, k_add)[nh0])

# -- fitting -----------------------------------------------------------------------------------
def fit_abundances(F, observed, reference, rel_err=0.15, floor=0.01, stiffness=1e4):
    """Abundances w >= 0 whose band ratios best match ``observed`` (chi^2 in ratio space).

    ``F`` is (n_molecules, n_bands), its columns in the order of ``observed``
    (a {band: ratio to the reference band} dict that also holds the reference
    band, as 1). Each species is put per unit reference flux, G = F / F_ref, so
    the model ratios are sum_s v_s G[s, b] with sum_s v_s = 1 (v_s the
    species' share of the reference band). NNLS minimises
    sum_b ((model_b - observed_b) / err_b)^2 with that sum held by a row of
    weight ``stiffness``. err_b is rel_err * observed[b] and ``floor`` in
    quadrature. Species with no reference flux get w = 0. Returns w (numbers,
    normalised so that the model reference band is 1), the model ratios, and
    chi^2 over the bands other than the reference.
    """
    from scipy.optimize import nnls

    F = np.asarray(F, dtype=float)
    if not np.isfinite(F).all():
        raise ValueError("band fluxes are not finite")
    bands = list(observed)
    k = bands.index(reference)
    others = [i for i in range(len(bands)) if i != k]
    y = np.array([observed[b] for b in bands], dtype=float)
    err = np.hypot(rel_err * y, floor)
    ok = F[:, k] > 0.0
    G = F[ok] / F[ok, k:k + 1]
    A = np.vstack([G[:, others].T / err[others, None], stiffness * np.ones((1, ok.sum()))])
    v, _ = nnls(A, np.concatenate([y[others] / err[others], [stiffness]]))
    v /= v.sum()
    w = np.zeros(len(F))
    w[ok] = v / F[ok, k]
    model = v @ G
    return {"weights": w, "ratios": dict(zip(bands, model)),
            "chi2": float((((model - y) / err)[others] ** 2).sum())}


def bootstrap(F, observed, reference, rel_err=0.15, floor=0.01, n=200, seed=0):
    """Weights of ``n`` fits, every observed ratio but the reference scattered by its uncertainty.

    The scatter is lognormal by rel_err plus normal by ``floor``, clipped at zero.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        obs = {b: (v if b == reference else
                   max(v * np.exp(rel_err * rng.standard_normal()) + floor * rng.standard_normal(), 0.0))
               for b, v in observed.items()}
        out.append(fit_abundances(F, obs, reference, rel_err, floor)["weights"])
    return np.array(out)


def population_statistics(library, weights, by="number"):
    """Mean N_C and edge-H ratios of a weighted population.

    The weights are numbers of molecules; ``by="carbon"`` weights each
    molecule by N_C for the mean N_C instead. The edge ratios are the total
    H atoms per class, the quantity Khan+25 infer from the 11-14 um bands.
    """
    w = np.asarray(weights, dtype=float)
    nc = np.array([m.nc for m in library], dtype=float)
    edges = {e: float(w @ np.array([m.edges[e] for m in library], dtype=float)) for e in EDGES}
    wn = w * nc if by == "carbon" else w
    out = {"mean_nc": float(wn @ nc / wn.sum()), "n_species": int((w > 0).sum())}
    for e in EDGES[1:]:
        out[f"solo/{e}"] = edges["solo"] / edges[e] if edges[e] > 0 else np.inf
    return out
