"""Grain charge distributions per size for the GSD solver.

Each radius of a grid gets the charge solve of RAMSES-CALIMA
(``models.dust_charge.rt_group_charging``): photoemission from photon groups
with WDB06 yields (band, inner-shell and Auger electrons, so X-rays are
included if the groups reach them), ion and electron capture with DS87
focusing and WD01 sticking, secondary electrons from hot gas. Detailed
balance gives P(Z): exact over the integers when sigma < 3, Gaussian
otherwise. The per-radius tables (photon kernels in the grain potential) take
~1 s each to build and are cached on disk; a solve then takes ~1 ms per
radius and is warm-started from the previous one.

``PhotonGroups`` turns any spectrum (photon density per eV, or F_lambda)
into group densities and mean energies. ``AGN_GROUPS`` extends the eight
CALIMA groups (0.1 eV - 0.5 keV) to 10 keV for AGN illumination;
``xray_power_law`` adds a coronal power law, which the SKIRTOR disk SED
(cut at 1 nm) lacks.

``ChargeState`` holds the solution on the radius grid and gives, as smooth
functions of radius:

* the mean and width of P(Z) and three Gauss-Hermite charge points (for the
  Coulomb factors of grain-grain collisions, ``velocities``);
* the Coulomb factor of ion accretion, <J~(Z)> over the exact P(Z) (DS87),
  for growth by ionised key species (``exchange``);
* the Coulomb explosion rate of grains whose charge reaches the
  electrostatic disruption limit (``coulomb_explosion_rate``).

WDB06 yields exist for graphite and silicate, and for amorphous carbon
(``wdb06_yields``): THEMIS carbon ('themis_ac') is charged as 'ac', pyCALIMA's
a-C, and 'cm20' as DustEM charges it (the benchmark); other carbonaceous
materials as graphite and all others as silicate (``charging_material``),
which should be revisited for metallic iron or SiC.
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from pycalima import _paths
from pycalima.models.dust_charge import rt_group_charging as rg
from pycalima.models.dust_charge.shared_physics import DS87_J_function_vec, E_STATC

H_CGS = 6.62607015e-27
C_CGS = 2.99792458e10
EV_ERG = 1.602176634e-12
M_H = 1.67262192e-24

# the eight CALIMA groups, then the soft X-rays up to 10 keV [eV]
AGN_GROUP_L0 = (0.1, 1.0, 5.6, 11.2, 13.6, 15.2, 24.59, 54.42, 100.0, 300.0, 1000.0, 3000.0)
AGN_GROUP_L1 = (1.0, 5.6, 11.2, 13.6, 15.2, 24.59, 54.42, 100.0, 300.0, 1000.0, 3000.0, 1e4)
AGN_GROUPS = (AGN_GROUP_L0, AGN_GROUP_L1)
#: the band-gap nodes of the covalent bins' charging tables [eV] (decided 2026-10-30: 5 nodes, ``acx<E_g>``)
ACX_GAP_NODES = (0.1, 0.7, 1.3, 1.9, 2.5)

_CARBON = ("graphite", "amorphous_carbon", "carbon", "diamond", "PAH", "SiC")


def charging_material(material: str) -> str:
    """WDB06 material used to charge ``material``: cm20, graphite or silicate."""
    if material in ("themis_ac", "ac"):
        return "ac"
    if material == "cm20":                     # the DustEM benchmark
        return "cm20"
    return "graphite" if any(material.startswith(c) for c in _CARBON[:5]) else "silicate"


# ---------------------------------------------------------------------------
# photon groups
# ---------------------------------------------------------------------------

@dataclass
class PhotonGroups:
    """Photon densities N [cm^-3] and mean energies egy [eV] in groups
    [L0, L1] [eV]."""

    L0: np.ndarray
    L1: np.ndarray
    N: np.ndarray
    egy: np.ndarray

    @classmethod
    def from_photon_density(cls, E_eV, n_E, L0=AGN_GROUP_L0, L1=AGN_GROUP_L1):
        """From n_E [photons cm^-3 eV^-1] on an ascending grid E_eV."""
        E = np.asarray(E_eV, float)
        n = np.asarray(n_E, float)
        L0, L1 = np.asarray(L0, float), np.asarray(L1, float)
        N = np.zeros(L0.size)
        egy = np.sqrt(L0 * L1)
        for g in range(L0.size):
            x = np.geomspace(L0[g], L1[g], 256)
            y = np.interp(np.log(x), np.log(E), n, left=0.0, right=0.0)
            N[g] = np.trapezoid(y, x)
            if N[g] > 0:
                egy[g] = np.trapezoid(y * x, x) / N[g]
        return cls(L0, L1, N, egy)

    @classmethod
    def from_flux_lambda(cls, lam_cm, F_lambda, L0=AGN_GROUP_L0, L1=AGN_GROUP_L1):
        """From F_lambda [erg s^-1 cm^-2 cm^-1] of a directed beam: n_E = F_E / (c E)."""
        lam = np.asarray(lam_cm, float)
        F = np.asarray(F_lambda, float)
        E = H_CGS * C_CGS / lam / EV_ERG                        # eV
        n_E = F * lam ** 2 / (H_CGS * C_CGS) / (C_CGS * E * EV_ERG) * EV_ERG
        order = np.argsort(E)
        return cls.from_photon_density(E[order], n_E[order], L0, L1)

    def __add__(self, other):
        N = self.N + other.N
        egy = np.where(N > 0, (self.N * self.egy + other.N * other.egy) / np.where(N > 0, N, 1.0),
                       self.egy)
        return PhotonGroups(self.L0, self.L1, N, egy)


def xray_power_law(r_cm, L_2_10, Gamma=1.9, E_eV=None, E_cut_eV=2e5):
    """Photon density per eV [cm^-3 eV^-1] at distance r of a corona with
    2-10 keV luminosity L_2_10 [erg/s], photon index Gamma and an exponential
    cut-off: (E_eV, n_E), for ``PhotonGroups.from_photon_density``."""
    E = np.geomspace(100.0, 1e5, 400) if E_eV is None else np.asarray(E_eV, float)
    shape = E ** (-Gamma) * np.exp(-E / E_cut_eV)
    x = np.geomspace(2e3, 1e4, 200)
    norm = L_2_10 / (4.0 * np.pi * r_cm ** 2 * C_CGS) / (
        np.trapezoid(x ** (1 - Gamma) * np.exp(-x / E_cut_eV), x) * EV_ERG)
    return E, norm * shape


# ---------------------------------------------------------------------------
# tables on a radius grid
# ---------------------------------------------------------------------------

def _cache_dir():
    d = _paths.get_dataset_cache_dir(create=True) / "gsd_charging"
    d.mkdir(parents=True, exist_ok=True)
    return d


# grain potentials of the tables [V]: AGN illumination charges grains to a
# few kV (the RAMSES tables stop at 1 kV), asinh-spaced
U_GRID = dict(n=801, u_min=-1000.0, u_max=3.0e4)


# the code and bundled data a table comes from (``rt_group_charging.build_bin_table``): their bytes enter the cache key,
# so a change to the physics or the data builds new tables instead of reusing stale ones. Until 2026-11-25 the key held
# only the groups and U_GRID, and the shared caches held tables of different physics under one name (40 of 47 names
# differed between two of them, some by orders of magnitude). Tables of older keys are never read again.
PHYSICS_MODULES = ("rt_group_charging", "wdb06_yields", "wdb06_atomic", "shared_physics", "coulomb_focusing")
PHYSICS_DATA = (("external_data", "wdb06"), ("optical_props", "draine_lee_1984"))


@functools.lru_cache(maxsize=1)
def physics_hash():
    """sha1 (10 hex digits) of ``PHYSICS_MODULES``' source and ``PHYSICS_DATA``'s files: the tables' physics."""
    h = hashlib.sha1()
    for name in PHYSICS_MODULES:
        h.update(name.encode())
        h.update(Path(importlib.import_module(f"pycalima.models.dust_charge.{name}").__file__).read_bytes())
    for parts in PHYSICS_DATA:
        root = _paths.get_data_root().joinpath(*parts)
        for f in sorted(q for q in root.rglob("*") if q.is_file()):
            h.update(str(f.relative_to(root)).encode())
            h.update(f.read_bytes())
    return h.hexdigest()[:10]


def _table_path(material, a, L0, L1):
    key = hashlib.sha1(np.asarray([*L0, *L1, *U_GRID.values()], float).tobytes()).hexdigest()[:10]
    return _cache_dir() / f"{material}_{a:.6e}_{key}_{physics_hash()}_v{rg.FORMAT_VERSION}.npz"


def bin_table(material, a, L0=AGN_GROUP_L0, L1=AGN_GROUP_L1, cache=True):
    """rt_group_charging table of one radius (potentials ``U_GRID``), from the
    disk cache if present."""
    path = _table_path(material, a, L0, L1)
    if cache and path.exists():
        with np.load(path, allow_pickle=False) as z:
            tab = {k: z[k] for k in z.files}
        for k in ("version", "Zmin", "Zmax", "a", "W"):
            tab[k] = float(tab[k])
        tab["material"] = str(tab["material"])
        tab["label"] = str(tab["label"])
        return tab
    tab = rg.build_bin_table(material, a, L0, L1, U=rg.u_grid(**U_GRID))
    if cache:
        np.savez(path, **{k: np.asarray(v) for k, v in tab.items()})
    return tab


class ChargingModel:
    """Charge solves for one material on a grid of radii [cm].

    lattice: the solver of ``rt_group_charging.GroupChargingTable.solve``. True (default): LG5 (RAMSES
    rtg_lattice), the rates at the 5 (7) integer charges around the mode of P(Z), 2-4x cheaper than the full solve
    and within the documented charging-parameter uncertainties of the exact P(Z) (95th percentile of the error in <Z>:
    0.11 charges in diffuse gas, against the full solve's 0.03). False: the full WDB06rt solve (root, width, a window grown
    to its tails), RAMSES's default."""

    def __init__(self, material, a_grid, groups=AGN_GROUPS, cache=True, lattice=True):
        self.material = material
        self.lattice = bool(lattice)
        self.wdb06_material = charging_material(material)
        self.a = np.asarray(a_grid, float)
        self.L0, self.L1 = (np.asarray(g, float) for g in groups)
        self.tables = [rg.GroupChargingTable(bin_table(self.wdb06_material, float(ai),
                                                       self.L0, self.L1, cache))
                       for ai in self.a]
        self._egy = None
        self._warm = [None] * self.a.size

    def solve(self, photons: PhotonGroups, T, n_e, ions: Sequence = None, G0_bg=0.0,
              c_red=C_CGS) -> "ChargeState":
        """Equilibrium P(Z) at every radius. ``ions``: (n [cm^-3], charge,
        mass [g]) of the ions captured by grains; default H+ with n = n_e. Each radius starts from its last
        solve, or else from the previous radius's Z* scaled by the size ratio, as RAMSES's dust_interface (the
        result does not depend on the start; it saves the search over the whole charge range)."""
        if not (np.allclose(photons.L0, self.L0) and np.allclose(photons.L1, self.L1)):
            raise ValueError("photon groups differ from the tables' groups")
        if self._egy is None or not np.allclose(self._egy, photons.egy):
            for gt in self.tables:
                gt.set_group_energies(photons.egy)
            self._egy = np.array(photons.egy, float)
        ions = [(n_e, 1.0, M_H)] if ions is None else list(ions)
        res = []
        for k, gt in enumerate(self.tables):
            zg = res[-1]["Zstar"] * self.a[k] / self.a[k - 1] if res else None
            r = gt.solve(photons.N, c_red, T, n_e, ions, G0_bg=G0_bg, recomb=None, Z_guess=zg,
                         warm=self._warm[k], lattice=self.lattice)
            self._warm[k] = r
            res.append(r)
        return ChargeState(self, res, dict(T=float(T), n_e=float(n_e), ions=ions,
                                            G0_bg=float(G0_bg), N=np.array(photons.N, float),
                                            c_red=float(c_red)))


#: the tabulated arrays that are per grain (photoemission and its heating): blended per unit area
_PER_GRAIN_KEYS = ("K", "H", "K_bg", "H_bg")


def blend_table(t_lo, t_hi, a):
    """A bin table at radius a [cm] between the tables of two radii a_lo <= a <= a_hi on the same potential grid.

    The tabulated arrays are taken linear in ln a on that grid:
    - the photoemission kernels K, H, K_bg and H_bg, per unit area (they are per grain, so they scale with a^2);
    - the hot-gas factors and the group energies, as they are.

    The analytic parts are recomputed at a: W, Zmin, Zmax and the recombination thresholds. At a = a_lo the result
    is t_lo, up to round-off."""
    for k in ("U", "L0", "L1", "log10_Ts", "log10_Thot"):
        if not np.allclose(t_lo[k], t_hi[k]):
            raise ValueError(f"tables differ in {k!r}")
    a0, a1, a = float(t_lo["a"]), float(t_hi["a"]), float(a)
    w = 0.0 if a1 == a0 else math.log(a / a0) / math.log(a1 / a0)
    out = {k: v for k, v in t_lo.items() if k not in rg.TABULATED_KEYS}
    for k in rg.TABULATED_KEYS:
        lo, hi = np.asarray(t_lo[k], float), np.asarray(t_hi[k], float)
        if k in _PER_GRAIN_KEYS:
            out[k] = a * a * ((1.0 - w) * lo / (a0 * a0) + w * hi / (a1 * a1))
        else:
            out[k] = (1.0 - w) * lo + w * hi
    material, U = t_lo["material"], np.asarray(t_lo["U"], float)
    W, Zmin, Zmax = rg._charge_bounds(material, a, U, np.asarray(t_lo["L1"], float))
    out.update(a=a, W=float(W), Zmin=Zmin, Zmax=Zmax,
               recomb_Zth=rg.recombination_thresholds(material, a, W, Zmin, Zmax))
    return out


class AreaNodeCharging:
    """One charge solve per bin at its area node, the grain-rate method of RAMSES (decided 2026-11-18).

    Each bin's tables are built at ``radii_per_bin`` radii, spaced geometrically from the bin's lower edge to its
    upper edge and cached on disk. One count applies to every bin, or a list gives one per bin.

    Per cell, each bin's table at its area node a* (``closures.area_node``) is blended from the two bracketing radii
    (``blend_table``) and solved once, with LG5 by default as in ``ChargingModel``. A bin's area-like rates are its
    weight w = int a^2 dN / a*^2 times the single-grain rates at a*; this is exact for rates proportional to a^2 and
    to a^3.

    The covalent a-C(:H) bins ``gap_bins`` follow the cell's band gap (precool's hydrogen equilibrium). Their tables
    are also built for the intermediate materials ``acx<E_g>`` (``wdb06_yields``, labelled) at ``gap_nodes``; 5 nodes
    keep the heating and cooling within 1 % (decided 2026-10-30). A solve with ``gaps`` blends each such bin's table in
    radius at the two bracketing band-gap nodes, then linearly in the H/C between them
    (``rt_group_charging.interpolate_bin_tables``); the analytic parts are the acx material's at the cell's gap."""

    def __init__(self, material, grid, radii_per_bin=5, groups=AGN_GROUPS, cache=True, lattice=True, method=None,
                 gap_bins=(), gap_nodes=ACX_GAP_NODES):
        counts = np.broadcast_to(np.asarray(radii_per_bin, int), (grid.n_bins,))
        if np.any(counts < 2):
            raise ValueError("radii_per_bin must be at least 2 (the bin's edges)")
        self.material, self.grid, self.method = material, grid, method
        self.wdb06_material = charging_material(material)
        self.lattice = bool(lattice)
        self.L0, self.L1 = (np.asarray(g, float) for g in groups)
        a_e = grid.a_edges
        self.node_radii = [np.geomspace(a_e[i], a_e[i + 1], int(counts[i])) for i in range(grid.n_bins)]
        self.node_tables = [[bin_table(self.wdb06_material, float(r), self.L0, self.L1, cache) for r in radii]
                            for radii in self.node_radii]
        from pycalima.models.dust_charge import wdb06_yields as wy

        self.gap_bins = np.asarray(gap_bins, int)
        self.gap_nodes = np.asarray(gap_nodes, float)
        self.gap_tables = {int(i): [[bin_table(wy.acx_label(e), float(r), self.L0, self.L1, cache)
                                     for r in self.node_radii[i]] for e in self.gap_nodes] for i in self.gap_bins}

    def table_at(self, i, a, gap=None):
        """Bin i's table at radius a, which must lie inside the bin; for a covalent bin with ``gap`` [eV], at that
        band gap."""
        r = self.node_radii[i]
        j = int(np.clip(np.searchsorted(r, a, side="right") - 1, 0, r.size - 2))
        if gap is None or int(i) not in self.gap_tables:
            return blend_table(self.node_tables[i][j], self.node_tables[i][j + 1], a)
        from pycalima.models.dust_charge import wdb06_yields as wy
        from pycalima.models.dust_radiation import jones2012_optics as jo

        g = self.gap_nodes
        e = float(np.clip(gap, g[0], g[-1]))
        k = int(np.clip(np.searchsorted(g, e, side="right") - 1, 0, g.size - 2))
        lo, hi = (blend_table(self.gap_tables[int(i)][q][j], self.gap_tables[int(i)][q][j + 1], a) for q in (k, k + 1))
        x0, x1, x = (float(jo.h_per_carbon(a, v)) for v in (g[k], g[k + 1], e))
        w = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
        return rg.interpolate_bin_tables(lo, hi, w, wy.acx_label(e))

    def solve(self, N, M, photons: PhotonGroups, T, n_e, ions: Sequence | None = None, G0_bg=0.0, c_red=C_CGS,
              gaps=None):
        """The charge state at every bin's area node, and the bins' weights: (``ChargeState`` on the area nodes,
        w [cm^-3]). A bin's photoelectric heating is w_i Gamma_i, and likewise for the other area-like rates.
        ``gaps`` [eV]: the band gaps of the ``gap_bins``, in order (None: their base material).

        Each solve gets its own tables, so an earlier state keeps its own cell when the model solves another one."""
        if not (np.allclose(photons.L0, self.L0) and np.allclose(photons.L1, self.L1)):
            raise ValueError("photon groups differ from the tables' groups")
        from types import SimpleNamespace

        from .closures import area_node

        a_star, w = area_node(self.grid, N, M, self.method)
        gap_of = {} if gaps is None else dict(zip(self.gap_bins.tolist(), np.asarray(gaps, float), strict=True))
        tables = [rg.GroupChargingTable(self.table_at(i, a, gap_of.get(i)), egy=photons.egy)
                  for i, a in enumerate(a_star)]
        ions = [(n_e, 1.0, M_H)] if ions is None else list(ions)
        res = [gt.solve(photons.N, c_red, T, n_e, ions, G0_bg=G0_bg, recomb=None, lattice=self.lattice)
               for gt in tables]
        env = dict(T=float(T), n_e=float(n_e), ions=ions, G0_bg=float(G0_bg), N=np.array(photons.N, float),
                   c_red=float(c_red))
        return ChargeState(SimpleNamespace(a=a_star, tables=tables, lattice=self.lattice), res, env), w


class ChargeState:
    """Charge distributions on a ``ChargingModel``'s radius grid."""

    def __init__(self, model, results, env):
        self.model = model
        self.results = results
        self.env = env
        self.a = model.a
        self.Zmean = np.array([r["Zmean"] for r in results])
        self.Zsigma = np.array([r["Zsigma"] for r in results])
        self.discrete = np.array([r["discrete"] for r in results])
        self.Gamma = np.array([r["Gamma"] for r in results])        # photoelectric heating [erg/s]

    def _interp(self, y, a):
        return np.interp(np.log(np.asarray(a, float)), np.log(self.a), y)

    def mean_charge(self, a):
        return self._interp(self.Zmean, a)

    def charge_width(self, a):
        return self._interp(self.Zsigma, a)

    def potential_volts(self, a):
        """Mean grain potential U = <Z> e / a [V]."""
        a = np.asarray(a, float)
        return self.mean_charge(a) * rg.E2_EV_CM / a

    def charge_points(self, a):
        """Three Gauss-Hermite charges and weights per radius (exact mean,
        variance and fourth moment of a Gaussian of the same <Z>, sigma):
        arrays (Z[..., 3], w[3])."""
        mu = self.mean_charge(a)
        sg = self.charge_width(a)
        off = np.sqrt(3.0) * np.array([-1.0, 0.0, 1.0])
        return mu[..., None] + sg[..., None] * off, np.array([1.0, 4.0, 1.0]) / 6.0

    def exact_distribution(self, k):
        """Integer charges and exact P(Z) at grid radius k: the window of a discrete P(Z) from the full solve, else
        P(Z) on every integer charge within 8 sigma + 10 of Z* (LG5's tails are a model of g)."""
        r = self.results[k]
        gt = self.model.tables[k]
        if r["discrete"] and not self.model.lattice:
            return np.asarray(r["nodes"], float), np.asarray(r["weights"], float)
        self._set(k)
        return gt.exact_distribution(r["Zstar"], r["Zsigma"])

    def _set(self, k):
        """Restore this state's cell in table k (a later solve overwrites it)."""
        e = self.env
        self.model.tables[k]._set_cell(e["N"], e["c_red"], e["T"], e["n_e"], e["ions"], e["G0_bg"])

    def ion_focusing(self, z_ion=1.0):
        """Coulomb factor of ion capture, <J~(Z, z_ion)> over the exact P(Z)
        (DS87, with the image potential), at every grid radius."""
        T = self.env["T"]
        out = np.empty(self.a.size)
        for k in range(self.a.size):
            Z, P = self.exact_distribution(k)
            out[k] = float(P @ DS87_J_function_vec(Z, z_ion + 0 * Z, self.a[k] + 0 * Z, T + 0 * Z))
        return out

    def ion_focusing_of(self, z_ion=1.0):
        """``ion_focusing`` as a function of radius (log-log interpolation)."""
        lf = np.log(np.maximum(self.ion_focusing(z_ion), 1e-300))
        la = np.log(self.a)
        return lambda a: np.exp(np.interp(np.log(np.maximum(np.asarray(a, float), 1e-30)), la, lf))

    # -- Coulomb explosion ----------------------------------------------------
    def explosion_rate(self, S_max=1e10, E_ife=3e8, max_window=20000):
        """Coulomb explosion rate per grain [s^-1] at every grid radius.

        A grain is disrupted when its electrostatic stress S = (U/a)^2 / 4 pi
        reaches the tensile strength S_max (Draine & Salpeter 1979; Waxman &
        Draine 2000 eqs. 21-22; WDB06 eq. 30; Tazaki, Ichikawa & Kokubo 2020
        eq. 5), i.e. at Z_c = a^2 (4 pi S_max)^1/2 / e = 7.4e4 (S_max/1e10)^1/2
        (a / 0.1 um)^2. The rate is the flux of the birth-death charge chain
        into Z_c: Up(Z_c - 1) P(Z_c - 1), with P the equilibrium distribution
        truncated at Z_c (the quasi-steady first-passage rate; ~ the charging
        rate if the equilibrium lies beyond Z_c, as in TIK20's threshold).
        Grains whose surface field reaches E_ife [V/cm] (ion field emission,
        WD00 sec. 4.2) first shed ions instead, so for Z_c above that limit
        the rate is 0."""
        out = np.zeros(self.a.size)
        for k, (a, r) in enumerate(zip(self.a, self.results)):
            gt = self.model.tables[k]
            Zc = math.ceil(a * a * math.sqrt(4.0 * math.pi * S_max) / E_STATC)
            Z_ife = a * a * (E_ife / 299.792458) / E_STATC
            if Zc > Z_ife or Zc - 1 > gt.t["Zmax"] or Zc - 1 < gt.t["Zmin"]:
                continue
            if Zc - 1 - r["Zstar"] > 40.0 * max(r["Zsigma"], 0.5) + 10.0:
                continue                         # P(Z_c - 1) < e^-800: no explosions
            self._set(k)
            lo = max(gt.t["Zmin"], Zc - 1 - max_window,
                     min(Zc - 1, math.floor(r["Zstar"] - 10 * r["Zsigma"] - 10)))
            Z = np.arange(float(lo), float(Zc))
            lnf = np.concatenate([[0.0], np.cumsum(gt._g_arr(Z[:-1]))]) if Z.size > 1 else np.zeros(1)
            m = lnf.max()
            lnP = lnf[-1] - (m + math.log(np.exp(lnf - m).sum()))
            if lnP < -700.0:
                continue
            up = gt._rates(float(Zc - 1))[0]
            out[k] = up * math.exp(lnP)
        return out


def destroy(grid, N, M, rate, dt, method=None):
    """Remove grains at rate(a) [s^-1] over dt, exactly per bin node
    (``collisions.bin_nodes`` with the rate nodes of ``method``,
    ``closures``; default Gauss in radius, 'pl2m' the median split): returns
    (N, M, removed mass)."""
    from .closures import resolve
    from .collisions import bin_nodes

    meth = resolve(method)
    m, w = bin_nodes(grid, N, M, method=meth, rule=meth.rate_nodes)
    lam = np.asarray(rate(grid.radius(m)), float)
    keep = np.exp(-np.maximum(lam, 0.0) * dt)
    N1 = (w * keep).sum(axis=1)
    M1 = (w * m * keep).sum(axis=1)
    return N1, M1, float(np.asarray(M, float).sum() - M1.sum())
