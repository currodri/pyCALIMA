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
    Wenzel+20 from the cation up where it applies (N_C >= 32, IP < 11.3 eV);
  - BT94 electron capture, with sticking 1 for cations and the statistical s_e
    of ``attachment`` for neutrals;
  - associative detachment of anions with H.
  The chain stops at the highest charge in the library (+1 for PAHdb v3.20,
  which has no dications): that state holds the higher charges, emitting as
  itself, and absorbs without ionising.
- **Hydrogenation** (``intact_fraction``). The first step only:
  k_add / (k_add + k_H). k_H is the photo-rate of the first H loss from the
  intact molecule; it uses RRKM with the baseline's West+18 (E0, dS) on the
  molecule's own modes and the cascade with that sink. k_add is the
  baseline's H addition to the singly dehydrogenated state. The fit uses only
  molecules whose intact fraction passes a threshold, so hydrogenation is
  tied to the conditions without modes for each dehydrogenated state.
- **Emission** (``Population.band_fluxes``). The exact microcanonical cascade
  of each charge state for one absorbed photon (``emission.cascade_photons``),
  folded with that state's heating spectrum. Single-photon heating holds for
  these sizes at G0 <~ 1e5. The bands use the PAHdb Lorentzian (FWHM 15 cm^-1,
  redshift 15 cm^-1), integrated analytically over the caller's wavelength
  windows, which should be the windows of the observations.
- **Cost.** Heating is single-photon and every field is a fixed shape times G0,
  so per charge state and field shape a ``StateBasis`` at G0 = 1 holds the IR
  power per mode, the photo-ionisation (or detachment) rate and the absorbed
  power, each G0 times its basis value. H loss is not linear in G0: for small
  PAHs two photons within one cooling time dominate it. So the basis
  tabulates k_H(G0) from the full multiphoton steady state on
  ``K_H_G0_GRID`` and interpolates in log-log. Bases are built once (in parallel, ``Population.prepare``) and kept
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
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pycalima.pahchem import chemistry, dos, emission, energy_me, fields, photo, rates
from pycalima.pahchem.andrews16 import CORONENE_GEOMETRIC_A2
from pycalima.pahchem.baseline import E_PH, Params

HARTREE_EV = 27.211386
_FORMULA_CH = re.compile(r"C\d+H\d+[+-]*")
EDGES = ("solo", "duo", "trio", "quartet")


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

    The cation is the neutral's partner row in their Tables H1-H2 (checked to
    have the same formula and edge-H counts). An anion (Table H3) is attached
    when exactly one neutral has its formula and edge-H counts; isomers that
    share both leave the anion out. With ``all_anions`` the anions are looked
    up in the whole of PAHdb v3.20, not only Table H3: 81 molecules against
    38 (Table H3's anion first where isomers share formula and edge counts). ``path`` is the PAHdb v3.20 ASCII file (default
    ``ames_pahdb.ascii_path()``).
    """
    from pycalima.pahchem import ames_pahdb, maragkoudakis2020

    path = ames_pahdb.ascii_path() if path is None else path
    sample = maragkoudakis2020.sample()
    if all_anions:
        with open(path, encoding="utf-8", errors="replace") as fh:
            every = [int(line.split()[1]) for line in fh if line.startswith("UID:")]
        meta = ames_pahdb.read_metadata(path, every)
        pool = [u for u, m in meta.items() if m.get("CHARGE") == -1 and _FORMULA_CH.fullmatch(m.get("FORMULA", ""))]
        meta = {u: meta[u] for u in set(sample.uid) | set(pool)}
    else:
        meta = ames_pahdb.read_metadata(path, sample.uid)
        pool = list(sample[sample.charge == -1].uid)
    sp = ames_pahdb.read_ascii(path, sample.uid)
    key = lambda u: (meta[u]["FORMULA"].rstrip("+-"),) + tuple(meta[u][f"N_{e.upper()}"] for e in EDGES)
    pairs = sample[sample.charge >= 0].uid.to_numpy().reshape(-1, 2)
    for u0, u1 in pairs:
        if meta[u0]["CHARGE"] != 0 or meta[u1]["CHARGE"] != 1 or key(u0) != key(u1):
            raise ValueError(f"PAHdb UIDs {u0}, {u1} are not a neutral-cation pair")
    by_key = {}
    for u0, _ in pairs:
        by_key.setdefault(key(u0), []).append(u0)
    anions_by_key = {}
    for u in pool:
        anions_by_key.setdefault(key(u), []).append(u)
    h3 = set(sample[sample.charge == -1].uid)
    pick = lambda us: us if len(us) == 1 else [u for u in us if u in h3]       # Table H3 first among isomers
    anion = {by_key[k][0]: pick(us)[0] for k, us in anions_by_key.items()
             if len(by_key.get(k, ())) == 1 and len(pick(us)) == 1}
    sp |= ames_pahdb.read_ascii(path, [u for u in anion.values() if u not in sp])
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


# -- charge ----------------------------------------------------------------------------------------
def _yield(mol, z):
    """Ionisation (or detachment) yield out of charge z on E_PH; zero at the top of the chain."""
    if z == max(mol.charges):
        return np.zeros_like(E_PH)
    threshold = mol.ionisation_ev[z]
    if threshold >= 13.6:
        return np.zeros_like(E_PH)
    if z >= 1 and mol.nc >= 32 and threshold < 11.3:
        return photo.yield_wenzel2020(E_PH, threshold, mol.nc)
    return photo.yield_jochims1996(E_PH, threshold)


def neutral_sticking(mol, temperature_k):
    """Statistical s_e of the neutral (``attachment``), the neutral in its vibrational ground state.

    The baseline instead uses the neutral's radiative P(E) under the Draine
    field at G0 = 1. That steady state costs ~7 s per molecule, has a mean
    energy of 5-11 cm^-1, and gives the same s_e as the ground state (checked
    for C26H16, C72H22 and C216H36 of the library). Between photons an
    interstellar PAH is cold.
    """
    if mol.ea_ev is None:
        raise ValueError(f"{mol.name}: no anion in the library, so no neutral sticking")
    if "attachment" not in mol._cache:
        from pycalima.pahchem.attachment import Attachment

        at = Attachment(mol.species[0], mol.species[-1], mol.ea_ev, chemistry.disc_radius_cm(mol.nc),
                        e_excess_cm=8000.0)
        mol._cache["attachment"] = (at, {})
    at, done = mol._cache["attachment"]
    t = round(float(temperature_k), 6)
    if t not in done:
        done[t] = at.sticking(temperature_k, np.array([0.0]), np.array([1.0]))
    return done[t]


def charge_fractions(mol, env):
    """{Z: f} of the chain -1/0/+1 at ``env`` (a ``baseline.Environment``)."""
    field_ = env.field()
    T = env.temperature_k
    zs = mol.charges
    f = {zs[0]: 1.0}
    for z in zs[:-1]:
        up = photo.ionisation_rate(field_, photo.sigma_li_draine(mol.nc, z, E_PH), _yield(mol, z), E_PH)
        if z == -1:
            up += 7.8e-10 * env.n_h                                   # associative detachment
        s = neutral_sticking(mol, T) if z + 1 == 0 else 1.0
        down = chemistry.electron_capture(mol.nc, z + 1, T, sticking=s) * env.n_e
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
    """The field of ``shape`` at G0 = 1: ("kurucz", T_eff) or ("draine",)."""
    return fields.kurucz(shape[1], 1.0) if shape[0] == "kurucz" else fields.draine1978(g0=1.0)


def field_shape(env):
    """The field-shape key of a ``baseline.Environment``."""
    return ("kurucz", int(env.teff_k)) if env.teff_k is not None else ("draine",)


#: G0 at which ``StateBasis.k_h`` is tabulated (the multiphoton steady state at each)
K_H_G0_GRID = np.geomspace(1.0, 1e6, 13)


@dataclass(frozen=True)
class StateBasis:
    """One charge state under one field shape; rates at G0 = 1, except k_h on ``K_H_G0_GRID``."""

    freqs_cm: np.ndarray
    emitted: np.ndarray          # IR power per mode at G0 = 1 [cm^-1 s^-1]
    k_up: float                  # photo-ionisation or detachment rate at G0 = 1 [s^-1]
    k_h: np.ndarray              # first-H-loss photo-rate at each K_H_G0_GRID [s^-1]
    absorbed: float              # heating power at G0 = 1 [cm^-1 s^-1]

    def k_h_at(self, g0):
        """k_H [s^-1] at ``g0`` (array), log-log interpolated; rates below 1e-300 s^-1 count as zero."""
        g = np.asarray(g0, dtype=float)
        if (g < K_H_G0_GRID[0]).any() or (g > K_H_G0_GRID[-1]).any():
            raise ValueError(f"G0 outside the k_H table {K_H_G0_GRID[0]:g}-{K_H_G0_GRID[-1]:g}")
        lk = np.log(np.maximum(self.k_h, 1e-300))
        return np.exp(np.interp(np.log(g), np.log(K_H_G0_GRID), lk))


def _basis_inputs(mol, z, shape, de_cm, e_max_cm, params):
    sp = mol.species[z]
    return dict(freqs=sp.freqs_cm, einstein_a=sp.einstein_a, nc=mol.nc, nh=mol.nh, z=z, yield_=_yield(mol, z),
                shape=shape, de_cm=de_cm, e_max_cm=e_max_cm, h_even=tuple(params.h_even),
                sigma_scale=params.sigma_scale)


def _basis_key(inp):
    h = hashlib.sha1()
    for k in sorted(inp):
        v = inp[k]
        h.update(k.encode())
        h.update(np.ascontiguousarray(v).tobytes() if isinstance(v, np.ndarray) else repr(v).encode())
    return h.hexdigest()[:20]


def build_state_basis(freqs, einstein_a, nc, nh, z, yield_, shape, de_cm, e_max_cm, h_even, sigma_scale):
    """``StateBasis`` of one charge state (module docstring); top-level, so it runs in worker processes."""
    field1 = _unit_field(shape)
    sigma = photo.sigma_li_draine(nc, z, E_PH) * sigma_scale
    h = photo.heating_spectrum(field1, sigma, yield_, E_PH)
    cm = energy_me.CascadeModel(freqs, einstein_a, e_max_cm, de_cm)
    N = emission.cascade_photons(cm)
    e = E_PH * dos.CM_PER_EV / cm.de_cm
    j = np.floor(e).astype(int)
    w = e - j
    emitted = freqs * np.trapezoid(h * ((1.0 - w) * N[:, j] + w * N[:, j + 1]), E_PH, axis=1)
    E, counts = dos.beyer_swinehart(freqs, cm.energy_cm[-1] + 200.0)
    ts = rates.transition_state_for_entropy(freqs, 3050.0, h_even[1], n_scaled=5)
    k = rates.rrkm(E, counts, ts, h_even[0] * dos.CM_PER_EV, degeneracy=nh)
    k = k[np.rint(cm.energy_cm / (E[1] - E[0])).astype(int)]
    k_h = np.array([cm.steady_state(E_PH, g * h, {"H": k}).rates["H"] for g in K_H_G0_GRID])
    out = StateBasis(np.asarray(freqs, dtype=float), emitted,
                     float(photo.ionisation_rate(field1, sigma, yield_, E_PH)), k_h,
                     float(np.trapezoid(h * E_PH * dos.CM_PER_EV, E_PH)))
    if not (np.isfinite(out.emitted).all() and np.isfinite(out.k_h).all()
            and np.isfinite([out.k_up, out.absorbed]).all()):
        raise ValueError(f"basis of N_C {nc}, Z {z} is not finite")
    return out


def _build(inp):
    return build_state_basis(**inp)


class Population:
    """Band fluxes, charge and intact fractions of a library at an environment (module docstring).

    ``windows_um`` is {band: (lo, hi)} in um, the windows of the observations;
    the bands in ``local`` take off a linear baseline across their window, as
    the observations did (``_band_matrix``). ``cache_dir`` keeps the state
    bases on disk (None: memory only).
    """

    def __init__(self, library, windows_um, local=(), de_cm=100.0, e_max_ev=16.0, params=Params(), cache_dir=None):
        self.library, self.windows, self.p = list(library), dict(windows_um), params
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
    def _ckey(self, shape, z):
        return ("basis", shape, z, self.de_cm, self.e_max_cm, tuple(self.p.h_even), self.p.sigma_scale)

    def _load(self, key):
        f = None if self.cache_dir is None else self.cache_dir / f"{key}.npz"
        if f is None or not f.exists():
            return None
        d = np.load(f)
        return StateBasis(d["freqs_cm"], d["emitted"], float(d["k_up"]), d["k_h"], float(d["absorbed"]))

    def _save(self, key, b):
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez(self.cache_dir / f"{key}.npz", freqs_cm=b.freqs_cm, emitted=b.emitted, k_up=b.k_up,
                     k_h=b.k_h, absorbed=b.absorbed)

    def prepare(self, shape, workers=None):
        """Give every state its basis under ``shape``: from memory, the disk cache, or built in parallel."""
        todo = []
        for mol in self.library:
            for z in mol.charges:
                ck = self._ckey(shape, z)
                if ck in mol._cache:
                    continue
                inp = _basis_inputs(mol, z, shape, self.de_cm, self.e_max_cm, self.p)
                key = _basis_key(inp)
                b = self._load(key)
                if b is None:
                    todo.append((mol, ck, key, inp))
                else:
                    mol._cache[ck] = b
        if not todo:
            return
        workers = workers if workers is not None else (os.cpu_count() or 1)
        if workers > 1 and len(todo) > 4:
            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor(max_workers=workers) as ex:
                built = list(ex.map(_build, [t[3] for t in todo], chunksize=1))
        else:
            built = [_build(t[3]) for t in todo]
        for (mol, ck, key, _), b in zip(todo, built):
            mol._cache[ck] = b
            self._save(key, b)

    def basis(self, mol, z, shape):
        """``StateBasis`` of ``mol`` at charge z under ``shape``."""
        ck = self._ckey(shape, z)
        if ck not in mol._cache:
            inp = _basis_inputs(mol, z, shape, self.de_cm, self.e_max_cm, self.p)
            key = _basis_key(inp)
            b = self._load(key)
            if b is None:
                b = _build(inp)
                self._save(key, b)
            mol._cache[ck] = b
        return mol._cache[ck]

    def _stack(self, shape):
        """(k_up at G0 = 1, log k_H on K_H_G0_GRID, band power B[s, slot, band] at G0 = 1); 0 where absent."""
        key = (shape, tuple(self.windows.items()), self.local)
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
                    B[s, i] = b.emitted @ _band_matrix(b.freqs_cm, self.windows, self.local)
            self._arrays[key] = (k_up, lkh, B)
        return self._arrays[key]

    def _capture_rates(self, temperature_k):
        """alpha[s, slot]: electron capture coefficient of each state [cm^3 s^-1] (0 for a lowest state)."""
        t = round(float(temperature_k), 6)
        if t not in self._capture:
            a = np.zeros((len(self.library), len(self.zs)))
            for s, mol in enumerate(self.library):
                for z in mol.charges[1:]:
                    st = neutral_sticking(mol, temperature_k) if z == 0 else 1.0
                    a[s, z - self.zmin] = chemistry.electron_capture(mol.nc, z, temperature_k, sticking=st)
            self._capture[t] = a
        return self._capture[t]

    # -- an environment --------------------------------------------------------------------------
    def charge_matrix(self, env):
        """f[s, slot]: the charge fractions of every molecule over ``self.zs``."""
        k_up, _, _ = self._stack(field_shape(env))
        up = env.g0 * k_up
        if self.zmin == -1:
            up[:, 0] += 7.8e-10 * env.n_h                           # associative detachment of anions
        down = self._capture_rates(env.temperature_k) * env.n_e
        both = self.present[:, :-1] & self.present[:, 1:]
        logr = np.zeros_like(up[:, :-1])
        logr[both] = np.log(up[:, :-1][both]) - np.log(down[:, 1:][both])
        logf = np.concatenate([np.zeros((len(self.library), 1)), np.cumsum(logr, axis=1)], axis=1)
        logf = np.where(self.present, logf, -np.inf)
        f = np.exp(logf - logf.max(axis=1, keepdims=True))
        return f / f.sum(axis=1, keepdims=True)

    def band_fluxes(self, env):
        """(F[s, b] charge-weighted band power per molecule, {name: {Z: f}}) for the whole library."""
        _, _, B = self._stack(field_shape(env))
        f = self.charge_matrix(env)
        F = env.g0 * np.einsum("sz,szb->sb", f, B)
        fz = {m.name: {z: float(f[s, z - self.zmin]) for z in m.charges} for s, m in enumerate(self.library)}
        return F, fz

    def state_band_fluxes(self, mol, z, env):
        """Band power per molecule of charge z [cm^-1 s^-1], one value per band."""
        b = self.basis(mol, z, field_shape(env))
        return env.g0 * (b.emitted @ _band_matrix(b.freqs_cm, self.windows, self.local))

    def intact_fractions(self, env):
        """Charge-weighted k_add / (k_add + k_H) of the first H loss, for every molecule."""
        _, lkh, _ = self._stack(field_shape(env))
        f = self.charge_matrix(env)
        if not K_H_G0_GRID[0] <= env.g0 <= K_H_G0_GRID[-1]:
            raise ValueError(f"G0 {env.g0:g} outside the k_H table")
        x = np.log(env.g0)
        i = min(np.searchsorted(np.log(K_H_G0_GRID), x), K_H_G0_GRID.size - 1)
        i = max(i, 1)
        lo, hi = np.log(K_H_G0_GRID[i - 1]), np.log(K_H_G0_GRID[i])
        k_h = np.exp(lkh[..., i - 1] + (lkh[..., i] - lkh[..., i - 1]) * (x - lo) / (hi - lo))
        z = np.array(self.zs)[None, :]
        neutral = 0.07 * self.p.addition_neutral_scale * chemistry.cross_section_rate(
            env.temperature_k, CORONENE_GEOMETRIC_A2 * self.nc[:, None] / 24.0)
        k_add = np.where(z > 0, self.p.addition_cation, neutral) * env.n_h
        return np.sum(f * k_add / (k_add + k_h), axis=1)

    def intact_fraction(self, mol, env):
        return float(self.intact_fractions(env)[self._index[id(mol)]])


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
