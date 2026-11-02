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

  The chain stops at the highest charge in the library (+1 for PAHdb v3.20,
  which has no dications): that state holds the higher charges, emitting as
  itself, and absorbs without ionising.
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
  Rates are averaged over the intact molecule's charge fractions. Missing:
  super-hydrogenated states, and the IP shift of dehydrogenated states. The
  single first-step ratio k_add / (k_add + k_H) of earlier versions
  (``intact_fractions(method="single")``) misses the radical runaway near
  threshold (review B: 0.87 against 0.00 for C54H18 at the Atomic
  most-ionising corner).
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
  power. This too is linear in G0. Not included: the electron's kinetic energy
  (~kT), the energy the ion keeps after photoionisation (E - IP - KE), and
  H loss driven by the recombination energy (the k_H table is photon-driven
  only). Bases are built once (in parallel, ``Population.prepare``) and kept
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

from pycalima.pahchem import chemistry, dos, emission, energy_me, fields, photo, rates
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

#: above this every state ionises (or detaches) on absorption: WDB06's rule for a <~ 6 A (their eq. 15), and the
#: lab yields reach ~1 by 20-25 eV (Jochims+96, Wenzel+20). Below it the lab laws hold.
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
#: deposit the ion's internal energy (``photo.photoelectron_energy``) if ``Params.ionising_photons_heat``. It holds
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

    ``records`` defaults to ``ames_suite.xml_index`` of the v4.00 dataset
    (cached in ``cache_dir``).
    """
    from pycalima.pahchem import ames_suite

    if records is None:
        records = ames_suite.xml_index(ames_suite.xml_path("pahdb-theoretical-v4-00"), cache_dir)
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
        out.append(Molecule(f"{formula} ({neu['uid']})", neu["nc"], neu["nh"], edges, species, {z: g[z]["uid"] for z in zs},
                            thresholds[0], thresholds.get(-1), "fit" if fitted else "dft",
                            {z: thresholds[z] for z in zs[:-1]}, "pahdb-v4.00"))
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
    euv = (E >= Y_UNITY_EV).astype(float)
    if z == max(mol.charges):
        return euv                     # the top state ionises out of the chain only above Y_UNITY_EV (``chain_overflow``)
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
    return np.where(E >= Y_UNITY_EV, 1.0, y)


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
    return photo.photoelectron_energy(energy_ev, state_threshold(mol, z), 1.0 if z < 0 else partition)[0]


def anion_valence_threshold(mol):
    """Threshold [eV] for removing a valence electron (not the extra one) from the anion, which leaves the neutral
    excited: the molecule's own ladder one Coulomb step down, 2 IP(0) - IP(+1), as WD01 eq. 2 steps IP_V(Z) with
    the charge (the DFT ladder's steps are 1.28x WD01's e^2/a (1 + 0.3 A/a) at the disc radius). At least the EA.
    C24H12 3.55, C54H18 3.75, C96H24 3.88 eV with the calibrated shifts."""
    th = mol.ionisation_ev
    if 1 in th:
        ipv = 2.0 * th[0] - th[1]
    else:
        a = chemistry.disc_radius_cm(mol.nc) * 1e8                                    # A
        ipv = th[0] - 14.39964 / a * (1.0 + 0.3 / a)
    return max(float(ipv), float(th[-1]))


def statistical_detachment(mol, params=Params(), grain_cm=10.0, energy_ev=E_PH, valence=False):
    """(Y, <E_k> [eV]) on ``energy_ev`` (default E_PH) of photodetachment of the anion: prompt plus delayed
    (``Params.detachment`` = "statistical"), and with ``valence`` (``Params.detachment`` = "valence") the prompt
    removal of a valence electron too. Any grid works, also below 5 eV, where the photons above the EA detach too.

    - Prompt: the WD01 photodetachment cross section (eq. 20, ``shared_physics.photodetachment_cross_section_vec``,
      threshold EA), as a share of the absorption; the electron takes E - EA.
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
    key = ("detach_stat", params.sigma_scale, photo.uv_model_key(params.uv_cross_section), float(grain_cm), grid.tobytes(), bool(valence),
           f_att)
    if key not in mol._cache:
        from pycalima.models.dust_charge.shared_physics import photodetachment_cross_section_vec
        from pycalima.pahchem.attachment import Attachment

        ea = mol.ionisation_ev[-1]
        lyman = E_PH_FUV[-1]
        lo = grid <= lyman
        E = np.append(grid[lo], lyman)                    # with 13.6 eV, where the EUV ramp starts
        sigma = photo.sigma_uv(mol.nc, -1, E, params.uv_cross_section) * params.sigma_scale
        y_p = np.clip(np.divide(photodetachment_cross_section_vec(E, ea, -1), sigma,
                                out=np.zeros_like(E), where=sigma > 0), 0.0, 1.0)
        akey = ("attach_detach", float(grain_cm), f_att)
        if akey not in mol._cache:
            mol._cache[akey] = Attachment(mol.species[0], mol.species[-1], ea,
                                                     chemistry.disc_radius_cm(mol.nc),
                                                     e_excess_cm=(lyman - ea + 0.5) * dos.CM_PER_EV, grain_cm=grain_cm,
                                                     capture_fraction=f_att)
        at = mol._cache[akey]
        p_det, eps_cm = at.delayed_detachment(E * dos.CM_PER_EV)
        if valence:
            ipv = anion_valence_threshold(mol)
            y_v = photo.yield_jochims1996(E, ipv)
            ek_v = photo.photoelectron_energy(E, ipv)[0]
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
        ek_out[~lo] = photo.photoelectron_energy(hi, ipv)[0]
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
        at = Attachment(mol.species[0], mol.species[-1], mol.ea_ev, chemistry.disc_radius_cm(mol.nc),
                        e_excess_cm=akey[1], capture_fraction=capture_fraction)
        mol._cache[akey] = (at, {})
    at, done = mol._cache[akey]
    t = round(float(temperature_k), 6)
    if t not in done:
        done[t] = at.sticking(temperature_k, np.array([0.0]), np.array([1.0]))
    return done[t]


def attachment_fraction(mol, params=Params()):
    """f of this molecule: ``Params.attachment_fraction``, or a logistic in its electron affinity (``attachment_model="ea"``; the
    anion's detachment threshold ``ionisation_ev[-1]``) or in N_C (``"size"``). The same f sets the attachment and, by detailed
    balance, the anion's autodetachment (``statistical_detachment``)."""
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


def ion_up_rate(mol, z, env, params=Params()):
    """Rate [s^-1] at which ions take the molecule from charge z to z + 1 at ``env`` (``env.ions``).

    Anions: mutual neutralisation with every ion (``chemistry.mutual_neutralisation``). Neutrals: the ion-neutral
    reaction (``chemistry.ion_neutral_rate``) for ions whose atom's IP exceeds the molecule's, plus the [M-PAH]+
    adducts of ``chemistry.ADDUCT_IONS`` below it if ``Params.metal_adducts``. Cations and above: none (Coulomb
    barrier, below 1e4 K).
    """
    up = 0.0
    for ion, n in getattr(env, "ions", ()):
        if z == -1:
            up += n * chemistry.mutual_neutralisation(mol.nc, mol.nh, -1, ion, env.temperature_k, params.mn_efficiency)
        elif z == 0:
            if chemistry.IONS[ion][1] > mol.ionisation_ev[0] or (params.metal_adducts and ion in chemistry.ADDUCT_IONS):
                up += n * chemistry.ion_neutral_rate(mol.nc, mol.nh, ion, params.ion_neutral_efficiency)
    return up


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
        sigma = photo.sigma_uv(mol.nc, z, E_PH, params.uv_cross_section) * params.sigma_scale
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
      r = ``Environment.optical_isrf`` / G0, so that G0 times this field leaves the old stars unscaled.
    """
    if shape[0] == "kurucz":
        e_max = shape[2] if len(shape) > 2 else fields.LYMAN_EV
        e_min = shape[3] if len(shape) > 3 else E_PH_MIN_EV
        return fields.kurucz(shape[1], 1.0, e_max=e_max, e_min=e_min)
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
    e_min = float(getattr(env, "field_e_min_ev", E_PH_MIN_EV))
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
#: lab points), "lab_cold" and "lab_pahs" (``chemistry.lab_sticking``, saturating at 1, fitted to the photo-ion or to every PAH
#: point), "wd01" (WD01's capture by a graphite sphere of the same carbon count over BT94's, for comparison only)
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
    ladder = []
    for k in range(4):
        n = mol.nh - k
        if n < 1:
            break
        s_ = sp if k == 0 else sp.stripped(n)
        ladder.append((s_.freqs_cm, _einstein_a(s_, params), n, bool(n % 2 == 0 and n > mol.edges.get("solo", 0))))
    return dict(freqs=sp.freqs_cm, einstein_a=_einstein_a(sp, params), nc=mol.nc, nh=mol.nh, z=z, yield_=_yield(mol, z, params),
                shape=shape, de_cm=de_cm, e_max_cm=e_max_cm, h_even=tuple(params.h_even),
                sigma_scale=params.sigma_scale, recomb_ev=recomb, ladder=tuple(ladder),
                h_odd=tuple(params.h_odd), h2=tuple(params.h2), threshold_ev=float(state_threshold(mol, z)),
                ionising_heat=bool(params.ionising_photons_heat),
                # the UV cross section enters the inputs (and the key) unless it is Li & Draine's, the bases before 2026-10-26
                **({"uv_cross_section": photo.uv_model_key(params.uv_cross_section)} if params.uv_cross_section != "li_draine" else {}),
                # only then does the electron's share enter the basis (and its key)
                **({"electron_ev": electron_energy(mol, z, params)} if params.ionising_photons_heat else {}))


#: bump when build_state_basis changes, so stored bases are rebuilt (with the grids, part of every key).
#: The result caches of the diagnostics key on it too: 6 is the photon grid from 1 eV; 7 the Kurucz fields
#: normalised over the whole 6-13.6 eV band (``fields.RadiationField.g0``, +0.15 %); 8 the emission of the
#: dehydrogenated states N_H0-1 ... N_H0-3.
BASIS_VERSION = 8
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
        h.update(repr((LOSS_E_MAX_EV, LOSS_DE_CM, photo.PARTITION_NODES)).encode())
    for k in sorted(inp):
        v = inp[k]
        h.update(k.encode())
        if legacy:
            h.update(np.ascontiguousarray(v).tobytes() if isinstance(v, np.ndarray) else repr(v).encode())
        else:
            _feed(h, v)
    return h.hexdigest()[:20]


#: energy grid of the multiphoton loss cascades (the baseline's): two-photon states reach ~27 eV, and the
#: 16 eV emission grid cut that tail off, underestimating k_H by orders of magnitude near threshold.
#: Where the state counts overflow (the largest PAHs), the ceiling drops to the next value: two 13.6 eV
#: photons need 27.2 eV, and three-photon events are negligible at that size.
LOSS_E_MAX_EV, LOSS_DE_CM = (45.0, 30.0, 20.0), 200.0

#: hydrogenation states that emit per charge state: N_H0 and the ladder's N_H0-1 ... N_H0-3 (``StateBasis``)
N_EMITTING = 4


def deposit_spectrum(field, sigma, yield_, threshold_ev, ionising_heat, electron_ev=None):
    """h(E_DEP) [s^-1 eV^-1]: the rate at which absorptions leave internal energy E_dep in the molecule.

    Non-ionising absorptions, sigma (1 - Y) F, deposit the photon energy. Ionisations, sigma Y F, deposit the ion's
    internal energy if ``ionising_heat``, else nothing (as before 2026-10-22): E - threshold - <E_k>, with the
    electron's <E_k> on E_PH from ``electron_ev`` (``electron_energy``, so that the anion's detached electron and
    the energy left behind add up to the excess), by default that of ``photo.photoelectron_energy``.
    Each photon bin's rate is split between its two neighbouring deposit bins, conserving the rate.
    """
    F = field.flux_at(E_PH)
    y = np.asarray(yield_, dtype=float)
    non = np.asarray(sigma) * (1.0 - y) * F
    if np.any(non[E_PH > E_DEP[-1]] > 0.0):
        raise ValueError("non-ionising absorption above the deposit grid (a yield below 1 above Y_UNITY_EV)")
    h = np.interp(E_DEP, E_PH, non, left=0.0, right=0.0)
    if ionising_heat:
        w = np.asarray(sigma) * y * F * _trapezoid_weights(E_PH)          # ionisations per photon bin [s^-1]
        if electron_ev is None:
            _, e_int = photo.photoelectron_energy(E_PH, threshold_ev)
        else:
            e_int = np.clip(E_PH - threshold_ev - np.asarray(electron_ev, dtype=float), 0.0, None)
        keep = (w > 0) & (e_int >= E_DEP[0]) & (e_int <= E_DEP[-1])
        i = np.clip(np.searchsorted(E_DEP, e_int[keep], side="right") - 1, 0, E_DEP.size - 2)
        f = (e_int[keep] - E_DEP[i]) / (E_DEP[i + 1] - E_DEP[i])
        dep = np.zeros(E_DEP.size)
        np.add.at(dep, i, (1.0 - f) * w[keep])
        np.add.at(dep, i + 1, f * w[keep])
        h = h + dep / _trapezoid_weights(E_DEP)
    return h


def _trapezoid_weights(x):
    """w_i with sum_i w_i f_i = trapezoid(f, x)."""
    x = np.asarray(x, dtype=float)
    w = np.zeros(x.size)
    dx = np.diff(x)
    w[:-1] += 0.5 * dx
    w[1:] += 0.5 * dx
    return w


def _loss_grid(freqs, einstein_a):
    """(CascadeModel, E, counts) on the highest ``LOSS_E_MAX_EV`` ceiling whose state counts stay finite."""
    for e_ev in LOSS_E_MAX_EV:
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                cm = energy_me.CascadeModel(freqs, einstein_a, e_ev * dos.CM_PER_EV, LOSS_DE_CM)
                E, counts = dos.beyer_swinehart(freqs, cm.energy_cm[-1] + 200.0)
        except OverflowError:
            continue
        if np.isfinite(counts).all() and np.isfinite(cm.k_modes).all():
            return cm, E, counts
    raise ValueError("state counts overflow even on the lowest loss-cascade ceiling")


def _h_loss_rate(freqs, E, counts, pick, nh, h_even, h_odd):
    """RRKM H loss on the loss grid: even-H (E0, dS) with degeneracy N_H, odd-H radical values with 1."""
    (e0, ds), deg = (h_even, nh) if nh % 2 == 0 else (h_odd, 1)
    return rates.rrkm(E, counts, rates.transition_state_for_entropy(freqs, 3050.0, ds, n_scaled=5),
                      e0 * dos.CM_PER_EV, degeneracy=deg)[pick]


def _loss_tables(freqs, einstein_a, nh, h2_ok, h, h_even, h_odd, h2, grid=None):
    """(k_H, k_H2) on K_H_G0_GRID for one hydrogenation state: RRKM sinks competing in its multiphoton cascade.
    ``grid``: this state's ``_loss_grid`` and H-loss rate, if already built ((cm, E, counts), k_H(E))."""
    if grid is None:
        cm, E, counts = _loss_grid(freqs, einstein_a)
        pick = np.rint(cm.energy_cm / (E[1] - E[0])).astype(int)
        k_h = _h_loss_rate(freqs, E, counts, pick, nh, h_even, h_odd)
    else:
        (cm, E, counts), k_h = grid
        pick = np.rint(cm.energy_cm / (E[1] - E[0])).astype(int)
    ch = {"H": k_h}
    if h2_ok:
        ch["H2"] = rates.rrkm(E, counts, rates.transition_state_for_entropy(freqs, 3050.0, h2[1], n_scaled=5),
                              h2[0] * dos.CM_PER_EV, degeneracy=2)[pick]
    kh, kh2 = [], []
    for g in K_H_G0_GRID:
        r = cm.steady_state(E_DEP, g * h, ch).rates
        kh.append(r["H"])
        kh2.append(r.get("H2", 0.0))
    return np.array(kh), np.array(kh2)


def _cascade_start(freqs, einstein_a, nc, z, yield_, shape, de_cm, e_max_cm, sigma_scale, recomb_ev, threshold_ev,
                   ionising_heat, electron_ev, uv_cross_section="li_draine"):
    """(h(E_DEP), k_up, recombination power, CascadeModel, start) of one state at G0 = 1; the model and start are None
    when nothing is left to emit (every absorption ionises or detaches, and no recombination heats)."""
    field1 = _unit_field(shape)
    sigma = photo.sigma_uv(nc, z, E_PH, uv_cross_section) * sigma_scale
    h = deposit_spectrum(field1, sigma, yield_, threshold_ev, ionising_heat, electron_ev)
    k_up = float(photo.ionisation_rate(field1, sigma, yield_, E_PH))
    recomb_power = k_up * recomb_ev * dos.CM_PER_EV
    if not np.any(h > 0.0) and recomb_power <= 0.0:
        return h, k_up, recomb_power, None, None
    cm = energy_me.CascadeModel(freqs, einstein_a, e_max_cm, de_cm)
    # molecules placed on the cascade grid by the deposits (trapezoid over E_DEP, split between the two
    # neighbouring bins) and the recombinations, then followed down once: the same photons as weighting
    # emission.cascade_photons by them, in O(modes x bins) instead of O(modes x bins^2)
    e = E_DEP * dos.CM_PER_EV / cm.de_cm
    j = np.floor(e).astype(int)
    w = e - j
    start = np.zeros(cm.n)
    hw = h * _trapezoid_weights(E_DEP)
    np.add.at(start, j, (1.0 - w) * hw)
    np.add.at(start, j + 1, w * hw)
    if recomb_power > 0.0:
        x = recomb_ev * dos.CM_PER_EV / cm.de_cm
        i0 = int(np.floor(x))
        start[i0] += k_up * (1.0 - (x - i0))
        start[i0 + 1] += k_up * (x - i0)
    return h, k_up, recomb_power, cm, start


def emission_temperature_shares(freqs, einstein_a, nc, z, yield_, shape, de_cm, e_max_cm, sigma_scale, recomb_ev=0.0,
                                threshold_ev=0.0, ionising_heat=False, electron_ev=None, uv_cross_section="li_draine", **_):
    """W[class, T bin]: how each mode class's IR power is spread over emission temperature (``band_conventions``)
    for the state of these ``_basis_inputs`` (N_H0; the dehydrogenated states take the same shares). The cascade
    of ``build_state_basis``, followed per bin; zeros when the state does not emit."""
    from pycalima.pahchem import band_conventions as bc

    _, _, _, cm, start = _cascade_start(np.asarray(freqs, dtype=float), einstein_a, nc, z, yield_, shape, de_cm, e_max_cm,
                                        sigma_scale, recomb_ev, threshold_ev, ionising_heat, electron_ev, uv_cross_section)
    if cm is None:
        return np.zeros((bc.N_CLASSES, bc.T_MID.size))
    X = emission.emission_by_energy(cm, start)
    return bc.class_temperature_shares(freqs, X, bc.bin_temperatures(freqs, cm.energy_cm))


def build_state_basis(freqs, einstein_a, nc, nh, z, yield_, shape, de_cm, e_max_cm, h_even, sigma_scale, recomb_ev=0.0,
                      ladder=(), h_odd=(3.16, 10.0), h2=(4.58, 40.0), threshold_ev=0.0, ionising_heat=False,
                      electron_ev=None, uv_cross_section="li_draine"):
    """``StateBasis`` of one charge state (module docstring); top-level, so it runs in worker processes.

    ``recomb_ev`` is the energy each recombination (attachment) into this state
    deposits; the heating rate is this state's own k_up (steady state).
    """
    h, k_up, recomb_power, cm, start = _cascade_start(freqs, einstein_a, nc, z, yield_, shape, de_cm, e_max_cm,
                                                      sigma_scale, recomb_ev, threshold_ev, ionising_heat, electron_ev,
                                                      uv_cross_section)
    if cm is None:                                     # every absorption ionises or detaches: nothing to emit
        return StateBasis(np.asarray(freqs, dtype=float), np.zeros(len(freqs)), k_up,
                          np.zeros(K_H_G0_GRID.size), 0.0)
    emitted = freqs * emission.emission_by_energy(cm, start).sum(axis=1)
    # the dehydrogenated states of the ladder, heated by the same deposits (sigma and yield do not depend on N_H
    # here), each on its own modes (``PAHSpecies.stripped``)
    l_freqs, l_emitted = [], []
    for f_, a_, _, _ in ladder[1:]:
        cm_k = energy_me.CascadeModel(f_, a_, e_max_cm, de_cm)
        l_freqs.append(np.asarray(f_, dtype=float))
        l_emitted.append(l_freqs[-1] * emission.emission_by_energy(cm_k, start).sum(axis=1))
    # the state's loss grid serves its k_h and the first ladder step (the same modes)
    cl, E, counts = _loss_grid(freqs, einstein_a)
    pick = np.rint(cl.energy_cm / (E[1] - E[0])).astype(int)
    # k_h (``intact_fractions(method="single")``) takes the even-H values with degeneracy N_H for any N_H;
    # ladder[0] the odd-H ones for odd N_H, so it shares this rate only for even N_H
    k = rates.rrkm(E, counts, rates.transition_state_for_entropy(freqs, 3050.0, h_even[1], n_scaled=5),
                   h_even[0] * dos.CM_PER_EV, degeneracy=nh)[pick]
    k_h = np.array([cl.steady_state(E_DEP, g * h, {"H": k}).rates["H"] for g in K_H_G0_GRID])
    lh, lh2 = np.zeros((4, K_H_G0_GRID.size)), np.zeros((4, K_H_G0_GRID.size))
    for i, (f_, a_, n_, ok_) in enumerate(ladder):
        grid = None
        if i == 0 and n_ == nh and np.array_equal(f_, freqs) and np.array_equal(a_, einstein_a):
            grid = ((cl, E, counts), k if nh % 2 == 0 else _h_loss_rate(freqs, E, counts, pick, nh, h_even, h_odd))
        lh[i], lh2[i] = _loss_tables(f_, a_, n_, ok_, h, h_even, h_odd, h2, grid)
    out = StateBasis(np.asarray(freqs, dtype=float), emitted, k_up, k_h,
                     float(np.trapezoid(h * E_DEP * dos.CM_PER_EV, E_DEP)), float(recomb_power), lh, lh2,
                     tuple(l_freqs), tuple(l_emitted))
    if not (np.isfinite(out.emitted).all() and np.isfinite(out.k_h).all()
            and all(np.isfinite(e).all() for e in out.ladder_emitted)
            and np.isfinite([out.k_up, out.absorbed]).all()):
        raise ValueError(f"basis of N_C {nc}, Z {z} is not finite")
    return out


def _build(inp):
    return build_state_basis(**inp)


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
        self.library, self.windows, self.p = list(library), dict(windows_um), params
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
        self._index = {id(m): s for s, m in enumerate(self.library)}
        self._arrays = {}
        self._capture = {}

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
                          tuple(d[f"ladder_emitted_{k}"] for k in range(1, n_l + 1)))

    def _stored(self, inp):
        """(key, stored basis or None) of ``inp``; a basis stored under its legacy key gets a link to the new."""
        key = _basis_key(inp)
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
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            # written under a temporary name and renamed, so a concurrent reader never sees half a file
            tmp = self.cache_dir / f"{key}.{os.getpid()}.tmp.npz"
            np.savez(tmp, freqs_cm=b.freqs_cm, emitted=b.emitted, k_up=b.k_up,
                     k_h=b.k_h, absorbed=b.absorbed, recombination=b.recombination,
                     ladder_h=b.ladder_h if b.ladder_h is not None else np.zeros((4, K_H_G0_GRID.size)),
                     ladder_h2=b.ladder_h2 if b.ladder_h2 is not None else np.zeros((4, K_H_G0_GRID.size)),
                     **{f"ladder_freqs_{k + 1}": f_ for k, f_ in enumerate(b.ladder_freqs)},
                     **{f"ladder_emitted_{k + 1}": e for k, e in enumerate(b.ladder_emitted)})
            os.replace(tmp, self.cache_dir / f"{key}.npz")

    def prepare(self, shape, workers=None):
        """Give every state its basis under ``shape``: from memory, the disk cache, or built in parallel."""
        todo = []
        for mol in self.library:
            for z in mol.charges:
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
        workers = workers if workers is not None else (os.cpu_count() or 1)

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
        """``StateBasis`` of ``mol`` at charge z under ``shape``."""
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
            for z in mol.charges:
                ck = ("tshares",) + self._ckey(shape, z, mol)
                if ck in mol._cache:
                    continue
                inp = _basis_inputs(mol, z, shape, self.de_cm, self.e_max_cm, self.p)
                f = self._shares_file(inp)
                if f is not None and f.exists():
                    mol._cache[ck] = np.load(f)
                else:
                    todo.append((mol, ck, f, inp))
        workers = workers if workers is not None else (os.cpu_count() or 1)
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
                for z in mol.charges:
                    b = self.basis(mol, z, shape)
                    i = z - self.zmin
                    k_up[s, i] = b.k_up
                    lkh[s, i] = np.log(np.maximum(b.k_h, 1e-300))
                    B[s, i] = b.emitted @ self._window_matrix(mol, z, shape, b.freqs_cm)
            self._arrays[key] = (k_up, lkh, B)
        return self._arrays[key]

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
                for z in mol.charges:
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
                for z in mol.charges:
                    for k, (_, e_) in enumerate(self._state_modes(self.basis(mol, z, shape))):
                        E[s, z - self.zmin, k] = float(np.sum(e_))
            self._arrays[key] = E
        return self._arrays[key]

    def _ladders(self, shape):
        """H and H2 loss tables of the ladders N_H0 ... N_H0-3, (s, slot, 4, K_H_G0_GRID); 0 where absent."""
        key = ("ladders", shape)
        if key not in self._arrays:
            self.prepare(shape)
            lh = np.zeros((len(self.library), len(self.zs), 4, K_H_G0_GRID.size))
            lh2 = np.zeros_like(lh)
            for s, mol in enumerate(self.library):
                for z in mol.charges:
                    b = self.basis(mol, z, shape)
                    lh[s, z - self.zmin], lh2[s, z - self.zmin] = b.ladder_h, b.ladder_h2
            self._arrays[key] = (lh, lh2)
        return self._arrays[key]

    def _capture_rates(self, temperature_k):
        """alpha[s, slot]: electron capture coefficient of each state [cm^3 s^-1] (0 for a lowest state)."""
        t = round(float(temperature_k), 6)
        if t not in self._capture:
            a = np.zeros((len(self.library), len(self.zs)))
            for s, mol in enumerate(self.library):
                for z in mol.charges[1:]:
                    a[s, z - self.zmin] = capture_coefficient(mol, z, temperature_k, self.p)
            self._capture[t] = a
        return self._capture[t]

    # -- an environment --------------------------------------------------------------------------
    def charge_matrix(self, env):
        """f[s, slot]: the charge fractions of every molecule over ``self.zs``."""
        k_up, _, _ = self._stack(field_shape(env))
        up = env.g0 * k_up
        if self.zmin == -1:
            up[:, 0] += anion_associative_detachment(self.p) * env.n_h
        if getattr(env, "ions", ()):
            for s, mol in enumerate(self.library):
                for z in (-1, 0):
                    if z in mol.charges and z + 1 in mol.charges:
                        up[s, z - self.zmin] += ion_up_rate(mol, z, env, self.p)
        down = self._capture_rates(env.temperature_k) * env.n_e
        both = self.present[:, :-1] & self.present[:, 1:]
        logr = np.zeros_like(up[:, :-1])
        logr[both] = np.log(up[:, :-1][both]) - np.log(down[:, 1:][both])
        logf = np.concatenate([np.zeros((len(self.library), 1)), np.cumsum(logr, axis=1)], axis=1)
        logf = np.where(self.present, logf, -np.inf)
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
                for z in mol.charges:
                    sigma = photo.sigma_uv(mol.nc, z, E_PH, self.p.uv_cross_section) * self.p.sigma_scale
                    i = z - self.zmin
                    A[s, i] = field1.photon_rate(sigma * E_PH, E_PH)
                    G[s, i] = photo.photoelectron_power(field1, sigma, _yield(mol, z, self.p), E_PH,
                                                        state_threshold(mol, z),
                                                        electron_energy_ev=electron_energy(mol, z, self.p, partition))
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
        of Draine & Sutin 1987 (eqs. 3.4-3.6) at the capture radius (``chemistry.disc_radius_cm``)."""
        t = round(float(temperature_k), 6)
        key = ("capture_energy", t)
        if key not in self._capture:
            E = np.zeros((len(self.library), len(self.zs)))
            for s, mol in enumerate(self.library):
                for z in mol.charges[1:]:
                    E[s, z - self.zmin] = capture_energy(mol, z, t)
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
        - ``carbon_accretion`` [cm^3 s^-1]: the share of C+ + PAH0 that keeps the C (0.4, Canosa+95), within
          alpha_ct["C+"]; the rest returns C to the gas;
        - ``electron_source`` [s^-1]: electrons made by photoionisation, photodetachment and associative detachment
          minus those captured
          (in steady state it equals minus the ion-driven up-flux: PAHs pass the ions' charge to electrons);
          ``electron_overflow`` adds the top state's EUV ionisations, which the chain does not follow;
        - ``gamma_pe``, ``lambda_rec`` [erg s^-1]: photoelectric heating and recombination cooling;
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
        if self.zmin == -1:
            photo_up[:, 0] += anion_associative_detachment(self.p) * env.n_h   # it frees an electron too
        capture = self._capture_rates(env.temperature_k) * env.n_e
        alpha_mn, alpha_ct = {}, {}
        for ion, _ in getattr(env, "ions", ()):
            unit = replace(env, ions=((ion, 1.0),))
            mn = np.array([f[s, -1 - self.zmin] * ion_up_rate(m, -1, unit, self.p) if -1 in m.charges else 0.0
                           for s, m in enumerate(self.library)])
            ct = np.array([f[s, 0 - self.zmin] * ion_up_rate(m, 0, unit, self.p) if 1 in m.charges else 0.0
                           for s, m in enumerate(self.library)])
            alpha_mn[ion], alpha_ct[ion] = float(x @ mn), float(x @ ct)
        gamma = self.photoelectric_heating(env)[0]
        lam = self.recombination_cooling(env)[0]
        return {"alpha_mn": alpha_mn, "alpha_ct": alpha_ct,
                "carbon_accretion": 0.4 * alpha_ct.get("C+", 0.0),
                "electron_source": float(x @ ((f * photo_up).sum(axis=1) - (f * capture).sum(axis=1))),
                "electron_overflow": float(x @ self.chain_overflow(env)),
                "gamma_pe": float(x @ gamma), "lambda_rec": float(x @ lam), "f": f}

    def chain_overflow(self, env):
        """Ionisations out of each molecule's top charge state [s^-1 per molecule]: EUV photons above Y_UNITY_EV
        whose product the chain does not hold. Small against the chain's rates means the truncation is harmless."""
        k_up, _, _ = self._stack(field_shape(env))
        f = self.charge_matrix(env)
        top = np.array([max(m.charges) - self.zmin for m in self.library])
        rows = np.arange(len(self.library))
        return env.g0 * f[rows, top] * k_up[rows, top]

    def band_fluxes(self, env):
        """(F[s, b] band power per molecule, weighted over charge and hydrogenation (``emission_weights``),
        {name: {Z: f}}) for the whole library."""
        f = self.charge_matrix(env)
        F = env.g0 * np.einsum("szk,szkb->sb", self.emission_weights(env), self._hydrogenation_bands(field_shape(env)))
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
                for z in mol.charges:
                    b = self.basis(mol, z, shape)
                    S[s, z - self.zmin] = self._line_spectrum(mol, z, shape, b.freqs_cm, b.emitted, nu, fwhm_cm, shift_cm)
            self._arrays[key] = S
        return self._arrays[key]

    def hydrogenation_state_spectra(self, shape, nu_cm, fwhm_cm=None, shift_cm=None):
        """S[s, slot, k, i]: IR power per cm^-1 of N_H0 - k of every state at G0 = 1 (``state_spectra`` per k; the
        dehydrogenated states take the emission-temperature shares of N_H0)."""
        nu = np.asarray(nu_cm, dtype=float)
        conv = self.band_convention if fwhm_cm is None and shift_cm is None else None
        key = ("hspectra", shape, hashlib.sha1(nu.tobytes()).hexdigest(), fwhm_cm, shift_cm, conv)
        if key not in self._arrays:
            self.prepare(shape)
            if conv is not None and conv.kind == "tdep":
                self.prepare_temperature_shares(shape)
            S = np.zeros((len(self.library), len(self.zs), N_EMITTING, nu.size))
            for s, mol in enumerate(self.library):
                for z in mol.charges:
                    for k, (f_, e_) in enumerate(self._state_modes(self.basis(mol, z, shape))):
                        if f_.size:
                            S[s, z - self.zmin, k] = self._line_spectrum(mol, z, shape, f_, e_, nu, fwhm_cm, shift_cm)
            self._arrays[key] = S
        return self._arrays[key]

    def spectra(self, env, nu_cm, **profile):
        """P[s, i]: the spectrum of every molecule, its charge and hydrogenation states at ``env`` (weighted by
        ``emission_weights``) [cm^-1 s^-1 per cm^-1]."""
        S = self.hydrogenation_state_spectra(field_shape(env), nu_cm, **profile)
        return env.g0 * np.einsum("szk,szki->si", self.emission_weights(env), S)

    def state_band_fluxes(self, mol, z, env):
        """Band power per molecule of charge z [cm^-1 s^-1], one value per band."""
        b = self.basis(mol, z, field_shape(env))
        return env.g0 * (b.emitted @ self._window_matrix(mol, z, field_shape(env), b.freqs_cm))

    def _interp_g0(self, table, g0):
        """Log-log interpolation of rate tables (..., K_H_G0_GRID) at ``g0``."""
        if not K_H_G0_GRID[0] <= g0 <= K_H_G0_GRID[-1]:
            raise ValueError(f"G0 {g0:g} outside the k_H table")
        lt = np.log(np.maximum(table, 1e-300))
        x = np.log(g0)
        i = max(min(np.searchsorted(np.log(K_H_G0_GRID), x), K_H_G0_GRID.size - 1), 1)
        lo, hi = np.log(K_H_G0_GRID[i - 1]), np.log(K_H_G0_GRID[i])
        return np.exp(lt[..., i - 1] + (lt[..., i] - lt[..., i - 1]) * (x - lo) / (hi - lo))

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
        """H addition rate of every state [s^-1], (s, slot): 1.4e-10 n(H) for cations, 0.07 geometric for the rest."""
        z = np.array(self.zs)[None, :]
        neutral = 0.07 * self.p.addition_neutral_scale * chemistry.cross_section_rate(
            env.temperature_k, CORONENE_GEOMETRIC_A2 * self.nc[:, None] / 24.0)
        return np.where(z > 0, self.p.addition_cation, neutral) * env.n_h

    def _chain_distributions(self, env):
        """[f(N_H), N_H = 0 ... N_H0] of every molecule: the chain, its rates averaged over the charge fractions."""
        f = self.charge_matrix(env)
        tab_h, tab_h2 = self._ladders(field_shape(env))
        lh = np.einsum("sz,szk->sk", f, self._interp_g0(tab_h, env.g0))     # charge-averaged loss rates
        lh2 = np.einsum("sz,szk->sk", f, self._interp_g0(tab_h2, env.g0))
        k_add = np.einsum("sz,sz->s", f, self._addition_rates(env))
        return [_chain_distribution(mol.nh, mol.edges.get("solo", 0), lh[s], lh2[s], k_add[s])
                for s, mol in enumerate(self.library)]

    def hydrogenation_fractions(self, env):
        """(h[s, k], deeper[s]): steady-state shares of N_H0 - k for the states that emit (k < ``N_EMITTING``,
        N_H >= 1, as far as the ladder has modes) and of the rest, which is not modelled (deeper states, whose
        stripped modes are not good enough: ``PAHSpecies.stripped``)."""
        h = np.zeros((len(self.library), N_EMITTING))
        for s, (p, mol) in enumerate(zip(self._chain_distributions(env), self.library)):
            for k in range(min(N_EMITTING, mol.nh)):          # N_H = N_H0 - k >= 1
                h[s, k] = p[mol.nh - k]
        return h, np.clip(1.0 - h.sum(axis=1), 0.0, 1.0)

    def emission_weights(self, env):
        """W[s, slot, k]: share of molecule s in charge slot and hydrogenation N_H0 - k. With
        ``dehydrogenated_emission`` f(Z) times ``hydrogenation_fractions`` (charge and hydrogenation taken as
        independent, as in the chain); otherwise f(Z) at k = 0."""
        f = self.charge_matrix(env)
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



def _chain_distribution_numpy(nh0, n_solo, lh, lh2, k_add):
    """Steady-state f(N_H), N_H = 0 ... N_H0, of the chain (H loss, H2 loss, H addition up to N_H0).

    ``lh``, ``lh2``: H and H2 loss rates of N_H0 ... N_H0-3. Deeper states:
    even (odd) N_H take the N_H0-2 (N_H0-3) rates, H loss scaled by N_H when
    even (degeneracy N_H). H2 loss only for even N_H above the solo count.
    """
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
    kH[0] = kH2[0] = 0.0
    kH2[1] = 0.0
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


if HAVE_NUMBA:
    @njit(cache=True)
    def _chain_distribution_numba(nh0, n_solo, lh, lh2, k_add):
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
        kH[0] = kH2[0] = 0.0
        if n > 1:
            kH2[1] = 0.0
        Q = np.zeros((n, n))
        for m in range(n):
            if m >= 1:
                Q[m, m - 1] += kH[m]
            if m >= 2:
                Q[m, m - 2] += kH2[m]
            if m < nh0:
                Q[m, m + 1] += k_add
            Q[m, m] = -Q[m].sum()
        A = np.ones((n + 1, n))
        A[:n] = Q.T
        b = np.zeros(n + 1)
        b[-1] = 1.0
        p = np.linalg.lstsq(A, b, rcond=np.finfo(np.float64).eps * (n + 1))[0]
        p = np.maximum(p, 0.0)
        return p / p.sum()


def _chain_distribution(nh0, n_solo, lh, lh2, k_add):
    """``_chain_distribution_numpy``, through its numba copy when available."""
    if HAVE_NUMBA:
        return _chain_distribution_numba(int(nh0), int(n_solo), np.ascontiguousarray(lh, dtype=float),
                                         np.ascontiguousarray(lh2, dtype=float), float(k_add))
    return _chain_distribution_numpy(nh0, n_solo, lh, lh2, k_add)


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
