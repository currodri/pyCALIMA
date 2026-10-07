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
- **Charge** (``charge_fractions``). A chain over Z = -1 (where the anion is in
  the library), 0, +1. The ingredients are the baseline's:
  - Li & Draine sigma;
  - Jochims+96 yields at the DFT IP, and for photodetachment at the EA;
  - BT94 electron capture, with sticking 1 for cations and the statistical s_e
    of ``attachment`` for neutrals;
  - associative detachment of anions with H.
  The chain stops at +1, since PAHdb v3.20 has no dications of the sample: the
  cation fraction holds any dications, emitting as cations, and the cation
  absorbs without ionising.
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
  windows, which should be the windows of the observations. The basis is
  computed once per molecule and charge; each environment then costs one
  integral.
- **Fit** (``fit_abundances``). Non-negative least squares on the band fluxes
  relative to a reference band, each weighted by its relative uncertainty.
  With fewer bands than molecules the solution is not unique: NNLS returns at
  most n_bands molecules. ``bootstrap`` refits under perturbed observations to
  show which conclusions survive. Size and edge statistics
  (``population_statistics``) are the robust outputs, not the individual
  species.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pycalima.pahchem import chemistry, dos, emission, energy_me, fields, photo, rates
from pycalima.pahchem.andrews16 import CORONENE_GEOMETRIC_A2
from pycalima.pahchem.baseline import E_PH, Params

HARTREE_EV = 27.211386
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
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def charges(self):
        return tuple(sorted(self.species))


def library_m20(path=None, min_nc=0):
    """[Molecule] of the Maragkoudakis+20 sample with N_C >= ``min_nc``.

    The cation is the neutral's partner row in their Tables H1-H2 (checked to
    have the same formula and edge-H counts). An anion (Table H3) is attached
    when exactly one neutral has its formula and edge-H counts; isomers that
    share both leave the anion out. ``path`` is the PAHdb v3.20 ASCII file
    (default ``ames_pahdb.ascii_path()``).
    """
    from pycalima.pahchem import ames_pahdb, maragkoudakis2020

    path = ames_pahdb.ascii_path() if path is None else path
    sample = maragkoudakis2020.sample()
    meta = ames_pahdb.read_metadata(path, sample.uid)
    sp = ames_pahdb.read_ascii(path, sample.uid)
    key = lambda u: (meta[u]["FORMULA"].rstrip("+-"),) + tuple(meta[u][f"N_{e.upper()}"] for e in EDGES)
    pairs = sample[sample.charge >= 0].uid.to_numpy().reshape(-1, 2)
    for u0, u1 in pairs:
        if meta[u0]["CHARGE"] != 0 or meta[u1]["CHARGE"] != 1 or key(u0) != key(u1):
            raise ValueError(f"PAHdb UIDs {u0}, {u1} are not a neutral-cation pair")
    by_key = {}
    for u0, _ in pairs:
        by_key.setdefault(key(u0), []).append(u0)
    anion = {by_key[key(u)][0]: u for u in sample[sample.charge == -1].uid
             if len(by_key.get(key(u), ())) == 1}
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
    threshold = mol.ea_ev if z == -1 else mol.ip_ev
    return photo.yield_jochims1996(E_PH, threshold)


def neutral_sticking(mol, temperature_k):
    """Statistical s_e of the neutral (``attachment``), its P(E) radiative at Draine G0 = 1 as in the baseline."""
    if mol.ea_ev is None:
        raise ValueError(f"{mol.name}: no anion in the library, so no neutral sticking")
    if "attachment" not in mol._cache:
        from pycalima.pahchem.attachment import Attachment

        neu = mol.species[0]
        at = Attachment(neu, mol.species[-1], mol.ea_ev, chemistry.disc_radius_cm(mol.nc), e_excess_cm=8000.0)
        cm = energy_me.CascadeModel(neu.freqs_cm, neu.einstein_a, e_max_cm=16.0 * dos.CM_PER_EV, de_cm=10.0,
                                    grain_cm=1.0)
        sigma = photo.sigma_li_draine(mol.nc, 0, E_PH)
        h = photo.heating_spectrum(fields.draine1978(g0=1.0), sigma, _yield(mol, 0), E_PH)
        mol._cache["attachment"] = (at, cm.energy_cm, cm.steady_state(E_PH, h, {}).probability, {})
    at, e_n, p_n, done = mol._cache["attachment"]
    t = round(float(temperature_k), 6)
    if t not in done:
        done[t] = at.sticking(temperature_k, e_n, p_n)
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


class Population:
    """Band fluxes of every library molecule at an environment, from cached single-photon bases.

    ``windows_um`` is {band: (lo, hi)} in um, the windows of the observations;
    the bands in ``local`` take off a linear baseline across their window, as
    the observations did (``_band_matrix``).
    """

    def __init__(self, library, windows_um, local=(), de_cm=100.0, e_max_ev=16.0, params=Params()):
        self.library, self.windows, self.p = list(library), dict(windows_um), params
        self.local = tuple(local)
        self.bands = tuple(self.windows)
        self.de_cm, self.e_max_cm = float(de_cm), e_max_ev * dos.CM_PER_EV

    def _state(self, mol, z):
        """(CascadeModel, basis B[b, j] = band energy [cm^-1] emitted per photon absorbed into bin j)."""
        ckey = ("cascade", z, self.de_cm, self.e_max_cm)
        if ckey not in mol._cache:
            sp = mol.species[z]
            cm = energy_me.CascadeModel(sp.freqs_cm, sp.einstein_a, self.e_max_cm, self.de_cm)
            mol._cache[ckey] = (cm, emission.cascade_photons(cm))
        cm, N = mol._cache[ckey]
        key = ckey + (tuple(self.windows.items()), self.local)
        if key not in mol._cache:
            sp = mol.species[z]
            mol._cache[key] = (_band_matrix(sp.freqs_cm, self.windows, self.local) * sp.freqs_cm[:, None]).T @ N
        return cm, mol._cache[key]

    def _heating(self, mol, z, field_):
        sigma = photo.sigma_li_draine(mol.nc, z, E_PH) * self.p.sigma_scale
        return photo.heating_spectrum(field_, sigma, _yield(mol, z), E_PH)

    def state_band_fluxes(self, mol, z, env, field_=None):
        """Band power per molecule of charge z [cm^-1 s^-1], one value per band."""
        cm, B = self._state(mol, z)
        h = self._heating(mol, z, env.field() if field_ is None else field_)
        e = E_PH * dos.CM_PER_EV / cm.de_cm
        j = np.floor(e).astype(int)
        w = e - j
        return np.trapezoid(h * ((1.0 - w) * B[:, j] + w * B[:, j + 1]), E_PH, axis=1)

    def band_fluxes(self, env):
        """(F[s, b] charge-weighted band power per molecule, {name: {Z: f}}) for the whole library."""
        field_ = env.field()
        F = np.zeros((len(self.library), len(self.bands)))
        fz = {}
        for s, mol in enumerate(self.library):
            fz[mol.name] = charge_fractions(mol, env)
            for z, f in fz[mol.name].items():
                F[s] += f * self.state_band_fluxes(mol, z, env, field_)
        return F, fz

    def intact_fraction(self, mol, env, fz=None):
        """Charge-weighted k_add / (k_add + k_H) of the first H loss (module docstring)."""
        fz = charge_fractions(mol, env) if fz is None else fz
        field_ = env.field()
        T = env.temperature_k
        sigma_geo = CORONENE_GEOMETRIC_A2 * mol.nc / 24.0
        out = 0.0
        for z, f in fz.items():
            cm, _ = self._state(mol, z)
            kkey = ("k_H", z, self.de_cm, self.e_max_cm)
            if kkey not in mol._cache:
                sp = mol.species[z]
                e0, ds = self.p.h_even
                E, counts = dos.beyer_swinehart(sp.freqs_cm, cm.energy_cm[-1] + 200.0)
                ts = rates.transition_state_for_entropy(sp.freqs_cm, 3050.0, ds, n_scaled=5)
                k = rates.rrkm(E, counts, ts, e0 * dos.CM_PER_EV, degeneracy=mol.nh)
                mol._cache[kkey] = k[np.rint(cm.energy_cm / (E[1] - E[0])).astype(int)]
            k_h = cm.steady_state(E_PH, self._heating(mol, z, field_), {"H": mol._cache[kkey]}).rates["H"]
            if not np.isfinite(k_h):
                raise ValueError(f"{mol.name} Z={z}: H-loss rate is not finite")
            k_add = (self.p.addition_cation if z > 0 else
                     0.07 * self.p.addition_neutral_scale * chemistry.cross_section_rate(T, sigma_geo)) * env.n_h
            out += f * k_add / (k_add + k_h)
        return float(out)


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
