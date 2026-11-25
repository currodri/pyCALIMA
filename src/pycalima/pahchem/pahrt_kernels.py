"""Per-molecule kernels of pahchem's joint (Z, N_H) chain on RT photon groups, for a per-cell equilibrium solver (PAHrt).

The PAH bins of RAMSES-CALIMA take their equilibrium state per cell from the local gas and the RT photon groups
(decided 2026-11-24, ``diagnostics/pahchem/pahrt/README.md``). Each bin is represented by a few PAHdb molecules
(``representatives``), and each molecule by these kernels, read from pahchem itself: every rate of its joint chain
(``Population.joint_networks``, ``Params.hydrogen_chain = "joint"``) and every per-state observable
(``hydrogen_observables.state_weights``) on basis environments. Nothing of pahchem's physics is re-implemented here.

- **Photon bases**, per group g = [L0, L1] and in-group slope alpha (``ALPHAS``): pahchem's band-limited power law
  (``Environment.field_powerlaw``, F_nu ~ nu^-alpha, photon flux per eV ~ E^-(alpha + 1)) at each G0 of ``G0_GRID``
  (pahchem's own k_H grid, ``population.K_H_G0_GRID``; a band without far UV is normalised by its energy flux,
  ``fields.RadiationField.scaled``). Stored per unit group photon flux F [cm^-2 s^-1]: kappa(F) = k(F) / F [cm^2].
  kappa is constant for a single-photon process. Multiphoton heating (H, H2 and proton loss of large PAHs, the bare
  skeleton's carbon loss) makes it grow with F: pahchem tabulates those rates against G0 for each field shape, and
  the bases sample them.
- **Collision bases**, per log10 T of ``LT``: one more unit of n_e, of n(H I), or of each ion of ``ION_NAMES``,
  under the photon basis shape with the most live charges (pahchem's live set depends on the shape only), so every
  charge a group reaches has its captures. They hold captures, attachment, recombination deposits, H addition, Eley-Rideal, associative detachment, mutual
  neutralisation and charge transfer.

Every basis is the difference between an environment and its regulariser. The regulariser is n_e = n(H I) = 1 at
G0 = 1e-30 under the basis's shape, at its T. pahchem's per-state weights solve the network's steady state (for its
shape only), and without electrons or H a photon-only network has no unique one. The rates add up across inputs, so
the difference holds the basis's input alone.

Charges are capped at ``Z_MAX`` (``cap_charges``; provisional, to revisit with hot gas and X-rays): every charge live
would otherwise bring pahchem's whole ladder to the cold limit (-1 ... +17 for C89H27).

Every basis is built with ``Params.cosmic_rays`` (zeta_cr = 0): it makes every charge live (pahchem's live set is
otherwise the charges a shape can reach, so a soft group's basis would hold no rates out of the high charges) and
changes nothing else. Cosmic-ray passages themselves (zeta_cr > 0) are not in the bases yet: pahchem blends two spectra
with weights that depend on zeta, so they need a zeta grid (tier T3).

Groups below pahchem's lowest photon energy (``population.E_PH_MIN_EV``) or above ``E_VALENCE_MAX_EV`` get no bases
(``valid`` False): C 1s fields wait for pahchem's K-shell fix (feature/pahchem-agn-xray, 2026-11-19), and the hot-gas
passages (``Params.hot_gas``) are not in the collision bases yet (tier T3 of the plan).

    python -m pycalima.pahchem.pahrt_kernels --out DIR --workers 3
"""
from __future__ import annotations

import argparse
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1
#: in-group slopes of the photon bases (F_nu ~ nu^-alpha, photon flux per eV ~ E^-(alpha + 1)), from a rising E^2
#: (a hot star's Rayleigh-Jeans side) to a Wien-like E^-13 (a 10 kK star across 5.6-11.2 eV); the solver interpolates
#: in the group's mean photon energy
ALPHAS = (-3.0, -1.0, 1.0, 3.0, 5.0, 8.0, 12.0)
#: the per-state observables kept (``hydrogen_observables.OBSERVABLES``; H2 comes from the processes), and
#: "overflow": the ionisations past the top charge per molecule per s (Coulomb explosions at the molecule's own top,
#: the cap's overflow below it)
OBSERVABLES = ("carbon", "heating", "cooling", "h_uptake", "absorbed", "overflow")
#: the top charge kept (provisional, 2026-11-25; to revisit with hot gas and X-rays). With every charge live, pahchem's
#: ladder to the cold limit gave C89H27 the charges -1 ... +17 (19 x 32 states); S1's populations reach about +5
Z_MAX = 5
#: log10 T [K] of the collision bases (from 3 K: the cold dense phases of the study reach 4.5 K)
LT = np.round(np.arange(0.5, 8.0 + 1e-9, 0.1), 6)
#: the highest group edge with photon bases [eV]: valence only, below the C 1s edge
E_VALENCE_MAX_EV = 100.0
#: the temperature of the photon bases [K] (they hold no collisions; no photon rate depends on T)
T_PHOTON_K = 100.0
#: the regulariser: G0 (its photon part cancels in the differences), n_e and n(H I) [cm^-3]
G0_TINY = 1e-30
N_REG = 1.0


def representatives(nc_edges, per_bin=5):
    """[[molecules] per bin]: ``per_bin`` of each bin [nc_edges[b], nc_edges[b + 1]] evenly in N_C rank (all of a bin
    with fewer), from ``hydrogen_tables.library`` (PAHdb v4.00, complete in -1 ... +2, calibrated thresholds)."""
    from pycalima.pahchem import hydrogen_tables as HT

    lib = sorted(HT.library(), key=lambda m: (m.nc, m.name))
    out = []
    for lo, hi in itertools.pairwise(nc_edges):
        sel = [m for m in lib if lo <= m.nc < hi * (1 + 1e-9)]
        if len(sel) > per_bin:
            sel = [sel[int(i)] for i in np.linspace(0, len(sel) - 1, per_bin)]
        out.append(sel)
    return out


def slug(mol):
    """A file name for ``mol`` ("C43H17 (3235)" -> "C43H17_3235")."""
    return mol.name.replace(" (", "_").replace(")", "").replace(" ", "_")


def cap_charges(mol, z_max=Z_MAX):
    """``mol`` with its charge states above ``z_max`` removed (as ``group_tables._chain``): the top then ionises only
    above ``population.Y_UNITY_EV``, and pahchem drops those steps (its "explosions")."""
    from dataclasses import replace

    if max(mol.charges) <= z_max:
        return mol
    keep = [z for z in mol.charges if z <= z_max]
    return replace(mol, species={z: mol.species[z] for z in keep}, uids={z: mol.uids[z] for z in keep if z in mol.uids},
                   ionisation_ev={z: v for z, v in mol.ionisation_ev.items() if z < z_max}, _cache={})


def population(mol, params_kw=None, z_max=Z_MAX, cache_dir=None):
    """(``population.Population`` of ``mol`` for the kernels, the molecule it holds, True if capped).

    The joint chain with ``Params.cosmic_rays`` (every charge live; at zeta_cr = 0 no process added and no rate
    changed). pahchem's default ladder runs to the cold Coulomb limit floor(0.2 N_C) with stand-ins
    (``population.extend_ladder``). Here the molecule is calibrated and extended as ``Population`` does, then capped
    at ``z_max`` (``cap_charges``) and given with ``charge_ladder = "library"``, so that pahchem keeps the cap. The
    calibration is a no-op for molecules already shifted, as the library's are."""
    from dataclasses import replace

    from pycalima import _paths
    from pycalima.pahchem import population as P

    params = P.Params(hydrogen_chain="joint", **{"cosmic_rays": True, **(params_kw or {})})
    ext = P.extend_ladder(P.calibrate_thresholds([mol], params)[0])
    capped = cap_charges(ext, z_max)
    truncated = capped is not ext
    if truncated:
        params = replace(params, charge_ladder="library")
    cache = _paths.get_model_data_dir() / "pahchem_population_basis" if cache_dir is None else Path(cache_dir)
    return P.Population([capped], {"all": (1.0, 1e4)}, params=params, cache_dir=cache), capped, truncated


@dataclass
class MoleculeKernels:
    """One molecule's kernels on its states (``charges`` x ``nh_values``, every charge live in any basis).

    ``procs``: (name, dZ, dN_H, H2 released) of every process; ``photon_k`` [G, A, N0, P, Z, N]: kappa = k / F [cm^2]
    per group, slope and G0 node; ``photon_w`` [G, A, N0, O, Z, N]: the observables per unit F; ``e_mean`` [G, A]: the
    mean photon energy of each basis [eV]; ``f_unit`` [G, A]: the group photon flux of each basis at G0 = 1
    [cm^-2 s^-1]; ``coll_k`` [K, T, P, Z, N] and ``coll_w`` [K, T, O, Z, N]: per unit density of ``coll_kinds``
    ("e", "H", then the ions) [cm^3 s^-1 and per cm^-3]."""

    name: str
    nc: int
    nh0: int
    charges: tuple
    nh_values: tuple
    procs: list
    L0: np.ndarray
    L1: np.ndarray
    valid: np.ndarray
    alphas: np.ndarray
    g0: np.ndarray
    e_mean: np.ndarray
    f_unit: np.ndarray
    photon_k: np.ndarray
    photon_w: np.ndarray
    lt: np.ndarray
    coll_kinds: tuple
    coll_k: np.ndarray
    coll_w: np.ndarray
    observables: tuple = OBSERVABLES

    def save(self, path):
        meta = {"version": FORMAT_VERSION, "name": self.name, "nc": int(self.nc), "nh0": int(self.nh0),
                "charges": list(self.charges), "nh_values": list(self.nh_values),
                "procs": [list(p) for p in self.procs], "coll_kinds": list(self.coll_kinds),
                "observables": list(self.observables), "z_max": int(max(self.charges))}
        np.savez_compressed(path, meta=json.dumps(meta), L0=self.L0, L1=self.L1, valid=self.valid,
                            alphas=self.alphas, g0=self.g0, e_mean=self.e_mean, f_unit=self.f_unit,
                            photon_k=self.photon_k, photon_w=self.photon_w, lt=self.lt, coll_k=self.coll_k,
                            coll_w=self.coll_w)

    @classmethod
    def load(cls, path):
        with np.load(path) as d:
            meta = json.loads(str(d["meta"]))
            if meta["version"] != FORMAT_VERSION:
                raise ValueError(f"{path}: format {meta['version']}, expected {FORMAT_VERSION}")
            arr = {k: d[k] for k in d.files if k != "meta"}
        return cls(meta["name"], meta["nc"], meta["nh0"], tuple(meta["charges"]), tuple(meta["nh_values"]),
                   [tuple(p) for p in meta["procs"]], valid=arr.pop("valid").astype(bool),
                   coll_kinds=tuple(meta["coll_kinds"]), observables=tuple(meta["observables"]), **arr)


def _band_photons(shape, lo, hi):
    """(photon flux [cm^-2 s^-1], mean photon energy [eV]) of pahchem's G0 = 1 field of ``shape`` in [lo, hi]."""
    from pycalima.pahchem import population as P

    f = P._unit_field(shape)
    E, n = f.energy_ev, f.photon_flux
    keep = (E >= lo) & (E <= hi)
    flux = float(np.trapezoid(n[keep], E[keep]))
    return flux, float(np.trapezoid(n[keep] * E[keep], E[keep])) / flux


def _read(net, W, procs, charges, nh_values):
    """{proc key: rate[Z, N]} and {observable: W[Z, N]} of one network, on the molecule's full states."""
    iz = [charges.index(z) for z in net.charges]
    rates = {}
    for p in net.processes:
        key = (p.name, int(p.dz), int(p.dnh), int(p.h2_released))
        if key not in procs:
            procs.append(key)
        r = rates.setdefault(key, np.zeros((len(charges), len(nh_values))))
        r[iz, :len(net.nh_values)] += p.rate
    obs = {}
    for k in OBSERVABLES:
        w = np.zeros((len(charges), len(nh_values)))
        w[iz, :len(net.nh_values)] = W[k]
        obs[k] = w
    return rates, obs


def build_molecule(mol, L0, L1, params_kw=None, alphas=ALPHAS, lt=LT, ions=None, cache_dir=None, z_max=Z_MAX):
    """``MoleculeKernels`` of ``mol`` on the groups [L0, L1] (``params_kw``: extra ``Params``, joint chain always),
    its charges capped at ``z_max`` (``cap_charges``). Below the molecule's own top the cap's overflow is not a
    Coulomb explosion: it is taken out of "carbon" and kept as "overflow" (the steady state treats the cap as
    reflecting, as pahchem treats its top)."""
    from pycalima.pahchem import group_tables as GT
    from pycalima.pahchem import hydrogen_observables as HO
    from pycalima.pahchem import population as P
    from pycalima.pahchem.baseline import Environment

    ions = GT.ION_NAMES if ions is None else tuple(ions)
    pop, capped, truncated = population(mol, params_kw, z_max, cache_dir)
    L0, L1 = np.asarray(L0, float), np.asarray(L1, float)
    g0s = np.asarray(P.K_H_G0_GRID, float)
    valid = (L1 > P.E_PH_MIN_EV) & (L1 <= E_VALENCE_MAX_EV * (1 + 1e-9))

    def photon_env(g0, alpha, lo, hi, T=T_PHOTON_K):
        return Environment("pahrt", g0, N_REG, N_REG, T, teff_k=None, ions=(), optical_isrf=0.0,
                           field_powerlaw=float(alpha), field_e_min_ev=float(lo), field_e_max_ev=float(hi))

    def run(env):
        net = pop.joint_networks(env)
        W = dict(HO.state_weights(pop, env, net)[0])
        W["overflow"] = np.asarray(pop.joint_state_observables(env)[0]["explosions"], float)
        if truncated:
            W["carbon"] = np.maximum(W["carbon"] - capped.nc * W["overflow"], 0.0)
        return net[0], W

    nets = []                                    # (kind, (g, a, i) or (k, t), network, weights, regulariser's)
    e_mean = np.zeros((L0.size, len(alphas)))
    f_unit = np.zeros((L0.size, len(alphas)))
    for g in np.nonzero(valid)[0]:
        lo, hi = max(L0[g], P.E_PH_MIN_EV), L1[g]
        for a, alpha in enumerate(alphas):
            f_unit[g, a], e_mean[g, a] = _band_photons(P.field_shape(photon_env(1.0, alpha, lo, hi)), lo, hi)
            reg = run(photon_env(G0_TINY, alpha, lo, hi))
            for i, g0 in enumerate(g0s):
                nets.append(("photon", (g, a, i), *run(photon_env(g0, alpha, lo, hi)), reg))
    kinds = ("e", "H", *ions)
    # the collision bases under the photon shape with the most live charges (pahchem's live set depends on the
    # shape only; the collision rates do not), so that every charge any group reaches has its captures
    widest = max((x for x in nets if x[0] == "photon"), key=lambda x: len(x[2].charges), default=None)
    if widest is None:
        base = Environment("pahrt", G0_TINY, N_REG, N_REG, T_PHOTON_K, teff_k=None, ions=(), optical_isrf=0.0)
    else:
        g, a, _ = widest[1]
        base = photon_env(G0_TINY, alphas[a], max(L0[g], P.E_PH_MIN_EV), L1[g])
    for t, lgt in enumerate(lt):
        env0 = replace(base, temperature_k=10.0 ** lgt)
        reg = run(env0)
        for k, kind in enumerate(kinds):
            env = (replace(env0, n_e=N_REG + 1.0) if kind == "e" else replace(env0, n_h=N_REG + 1.0) if kind == "H"
                   else replace(env0, ions=((kind, 1.0),)))
            nets.append(("coll", (k, t), *run(env), reg))
    charges = tuple(sorted({int(z) for x in nets for n in (x[2], x[4][0]) for z in n.charges}))
    nh_values = tuple(range(max(len(x[2].nh_values) for x in nets)))
    procs = []
    read = [(kind, idx, _read(n, W, procs, charges, nh_values), _read(*reg, procs, charges, nh_values))
            for kind, idx, n, W, reg in nets]
    nz, nn, npr, no = len(charges), len(nh_values), len(procs), len(OBSERVABLES)
    pk = np.zeros((L0.size, len(alphas), g0s.size, npr, nz, nn))
    pw = np.zeros((L0.size, len(alphas), g0s.size, no, nz, nn))
    ck = np.zeros((len(kinds), lt.size, npr, nz, nn))
    cw = np.zeros((len(kinds), lt.size, no, nz, nn))
    for kind, idx, (rates, obs), (r0, o0) in read:
        dk = {key: r - r0.get(key, 0.0) for key, r in rates.items()}
        dk.update({key: -r for key, r in r0.items() if key not in rates})
        dw = {name: obs[name] - o0[name] for name in OBSERVABLES}
        if kind == "photon":
            g, a, i = idx
            F = g0s[i] * f_unit[g, a]
            for key, r in dk.items():
                pk[g, a, i, procs.index(key)] = r / F
            for j, name in enumerate(OBSERVABLES):
                pw[g, a, i, j] = dw[name] / F
        else:
            k, t = idx
            for key, r in dk.items():
                ck[k, t, procs.index(key)] = r
            for j, name in enumerate(OBSERVABLES):
                cw[k, t, j] = dw[name]
    return MoleculeKernels(mol.name, int(mol.nc), int(mol.nh), charges, nh_values, procs, L0, L1, valid,
                           np.asarray(alphas, float), g0s, e_mean, f_unit, pk, pw, np.asarray(lt, float), kinds,
                           ck, cw)


def _prepare(job):
    """Build and cache pahchem's bases and responses of one molecule under one photon basis shape."""
    import time

    from pycalima.pahchem.baseline import Environment

    mol, alpha, lo, hi, params_kw = job
    t = time.time()
    pop = population(mol, params_kw)[0]
    pop.joint_networks(Environment("pahrt", 1.0, N_REG, N_REG, T_PHOTON_K, teff_k=None, ions=(), optical_isrf=0.0,
                                   field_powerlaw=float(alpha), field_e_min_ev=float(lo), field_e_max_ev=float(hi)))
    return mol.name, alpha, lo, hi, time.time() - t


def prepare_bases(mols, L0, L1, params_kw=None, alphas=ALPHAS, workers=1):
    """Build pahchem's bases of every (molecule, photon basis shape) in parallel, the largest molecules and hardest
    groups first (they are cached on disk, so ``build_molecule`` then only evaluates); yields each job's
    (name, alpha, lo, hi, seconds)."""
    from pycalima.pahchem import population as P

    L0, L1 = np.asarray(L0, float), np.asarray(L1, float)
    valid = (L1 > P.E_PH_MIN_EV) & (L1 <= E_VALENCE_MAX_EV * (1 + 1e-9))
    jobs = [(m, a, max(L0[g], P.E_PH_MIN_EV), L1[g], params_kw) for m in mols for g in np.nonzero(valid)[0][::-1]
            for a in alphas]
    jobs.sort(key=lambda j: (-j[0].nc, -j[3]))
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            yield from ex.map(_prepare, jobs)
    else:
        yield from map(_prepare, jobs)


def _job(job):
    mol, L0, L1, params_kw, out = job
    path = Path(out) / f"{slug(mol)}.npz"
    if not path.exists():
        build_molecule(mol, L0, L1, params_kw).save(path)
    return mol.name, str(path)


def build_bins(out, nc_edges, L0, L1, per_bin=5, params_kw=None, workers=1):
    """Kernels of ``representatives(nc_edges, per_bin)`` into ``out``, with ``manifest.json`` listing each bin's
    molecules (N_C, file); files already there are kept."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    bins = representatives(nc_edges, per_bin)
    jobs = [(m, L0, L1, params_kw, out) for b in bins for m in b]
    jobs.sort(key=lambda j: -j[0].nc)                     # the largest (slowest) first
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            done = dict(ex.map(_job, jobs))
    else:
        done = dict(map(_job, jobs))
    manifest = {"version": FORMAT_VERSION, "nc_edges": [float(x) for x in nc_edges], "L0": list(map(float, L0)),
                "L1": list(map(float, L1)), "params": params_kw or {},
                "bins": [[{"name": m.name, "nc": int(m.nc), "nh": int(m.nh), "file": Path(done[m.name]).name}
                          for m in b] for b in bins]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main(argv=None):
    import os

    os.environ.setdefault("CALIMA_WORKERS", "1")          # pahchem's own pools: one molecule per worker here
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-bin", type=int, default=5)
    ap.add_argument("--groups", choices=("agn", "rtz"), default="agn",
                    help="the GSD's 12 groups (gsd.charging.AGN_GROUP_*) or the 8 RTZ groups")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)
    from pycalima.gsd.carbon import M_C, CarbonLayout

    if a.groups == "agn":
        from pycalima.gsd.charging import AGN_GROUP_L0 as L0
        from pycalima.gsd.charging import AGN_GROUP_L1 as L1
    else:
        from pycalima.models.dust_charge.rt_group_charging import DEFAULT_GROUP_L0 as L0
        from pycalima.models.dust_charge.rt_group_charging import DEFAULT_GROUP_L1 as L1
    lay = CarbonLayout()
    nc_edges = lay.grid().m_edges[:lay.n_pah_bins + 1] / M_C
    m = build_bins(a.out, nc_edges, np.asarray(L0, float), np.asarray(L1, float), a.per_bin, workers=a.workers)
    print(f"{sum(len(b) for b in m['bins'])} molecules in {len(m['bins'])} bins -> {a.out}")


if __name__ == "__main__":
    main()
