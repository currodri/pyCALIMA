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

import hashlib
import math
from dataclasses import dataclass
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


def _table_path(material, a, L0, L1):
    key = hashlib.sha1(np.asarray([*L0, *L1, *U_GRID.values()], float).tobytes()).hexdigest()[:10]
    return _cache_dir() / f"{material}_{a:.6e}_{key}_v{rg.FORMAT_VERSION}.npz"


#: the electron-sticking laws of the grain tables (``sticking_law``); 'wd01' is the solver's own
STICKING_LAWS = ("wd01", "seam", "molecular")
#: the seam law's taper [cm]: pahchem's laboratory sticking at N_C,max (1.06 nm for a-C at 1.6 g cm^-3) to WD01's
#: grain law at 3 nm (labelled, 2026-11-23)
SEAM_TAPER_CM = (1.06e-7, 3e-7)
_M_C = 12.011 * 1.66053906660e-24


def sticking_law(name, a, rho=None):
    """(s_pos, s_neg): the electron sticking of cations, and of neutral and negative grains, of radius a [cm].

    Decided 2026-11-23 (status report, "How carbon moves"): the PAH/grain seam's heating gap is the electron
    sticking on cations, measured only up to N_C 16 (pahchem's laboratory fit saturates at 1 by N_C ~ 35; WD01's grain
    law gives 0.2-0.3 at these sizes). The laws:
    - 'wd01': Weingartner & Draine (2001), the solver's built-in law (cations 0.5 (1 - exp(-a / 1 nm)));
    - 'seam' (carbon's GSD default, labelled): cations take pahchem's laboratory fit
      (``pahchem.chemistry.lab_sticking``, 'cold') at the grain's N_C up to ``SEAM_TAPER_CM[0]`` and taper linearly
      in ln a to WD01 at ``SEAM_TAPER_CM[1]``;
    - 'molecular' (an A5 bracket): the laboratory fit for cations at every size.
    Neutrals and anions keep WD01 in every law, since the gap is on cations. ``rho`` [g cm^-3] turns a into N_C
    (needed by every law but 'wd01')."""
    if name not in STICKING_LAWS:
        raise ValueError(f"sticking must be one of {STICKING_LAWS}")
    a = float(a)
    wd_pos = 0.5 * (1.0 - math.exp(-a / 1e-7))
    wd_neg = wd_pos / (1.0 + math.exp(20.0 - 468.0 * (a / 1e-7) ** 3))
    if name == "wd01":
        return wd_pos, wd_neg
    if rho is None:
        raise ValueError(f"the '{name}' sticking law needs the bulk density rho")
    from pycalima.pahchem.chemistry import lab_sticking

    lab = float(lab_sticking(4.0 / 3.0 * math.pi * rho * a ** 3 / _M_C, "cold"))
    a0, a1 = SEAM_TAPER_CM
    if name == "molecular" or a <= a0:
        return lab, wd_neg
    if a >= a1:
        return wd_pos, wd_neg
    t = math.log(a / a0) / math.log(a1 / a0)
    return (1.0 - t) * lab + t * wd_pos, wd_neg


def bin_table(material, a, L0=AGN_GROUP_L0, L1=AGN_GROUP_L1, cache=True, sticking="wd01", rho=None):
    """rt_group_charging table of one radius (potentials ``U_GRID``), from the disk cache if present. A
    ``sticking`` law other than 'wd01' (``sticking_law``; ``rho`` turns a into N_C) adds the table's stick_pos and
    stick_neg at radius a, after the cache, which stays free of the law."""
    tab = _cached_table(material, a, L0, L1, cache)
    if sticking != "wd01":
        tab["stick_pos"], tab["stick_neg"] = sticking_law(sticking, a, rho)
    return tab


def _cached_table(material, a, L0, L1, cache):
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
    to its tails), RAMSES's default.

    sticking: the electron-sticking law of the tables (``sticking_law``): 'wd01' (default) or, for carbon, 'seam'
    (the GSD's choice) and 'molecular'; ``rho`` [g cm^-3] for N_C defaults to the GSD material's (``materials``)."""

    def __init__(self, material, a_grid, groups=AGN_GROUPS, cache=True, lattice=True, sticking="wd01", rho=None):
        self.material = material
        self.lattice = bool(lattice)
        self.wdb06_material = charging_material(material)
        self.a = np.asarray(a_grid, float)
        self.L0, self.L1 = (np.asarray(g, float) for g in groups)
        self.sticking = sticking
        if sticking != "wd01" and rho is None:
            from .materials import get_material

            rho = get_material(material).rho
        self.tables = [rg.GroupChargingTable(bin_table(self.wdb06_material, float(ai),
                                                       self.L0, self.L1, cache, sticking, rho))
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


def blend_table(t_lo, t_hi, a, sticking="wd01", rho=None):
    """A bin table at radius a [cm] between the tables of two radii a_lo <= a <= a_hi on the same potential grid.

    The tabulated arrays are linear in ln a on that grid: the photoemission kernels per unit area (K, H, K_bg, H_bg are
    per grain, so they scale with a^2), the hot-gas factors and group energies as they are. The analytic parts are
    recomputed at a: W, Zmin, Zmax, the recombination thresholds, cm20's s_e(U) and the ``sticking`` law's scalars
    (``sticking_law``, ``rho`` for N_C). At a = a_lo it is t_lo, up to round-off."""
    for k in ("U", "L0", "L1", "log10_Ts", "log10_Thot"):
        if not np.allclose(t_lo[k], t_hi[k]):
            raise ValueError(f"tables differ in {k!r}")
    a0, a1, a = float(t_lo["a"]), float(t_hi["a"]), float(a)
    w = 0.0 if a1 == a0 else math.log(a / a0) / math.log(a1 / a0)
    out = {k: v for k, v in t_lo.items()
           if k not in rg.TABULATED_KEYS and k not in ("s_e", "stick_pos", "stick_neg")}
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
    if "s_e" in t_lo:
        from pycalima.models.dust_charge import wdb06_yields as wy

        out["s_e"] = wy.cm20_sticking(U * a / rg.E2_EV_CM, a)
    if sticking != "wd01":
        out["stick_pos"], out["stick_neg"] = sticking_law(sticking, a, rho)
    return out


class AreaNodeCharging:
    """One charge solve per bin at its area node, the grain-rate method of RAMSES (decided 2026-11-18).

    Each bin's tables are built at ``radii_per_bin`` radii (geometric from the bin's lower to its upper edge, cached
    on disk). Per cell, every bin's table at its area node a* (``closures.area_node``) is blended from the two
    bracketing radii (``blend_table``) and solved once (LG5 by default, as ``ChargingModel``). The bin's
    area-like rates are its weight w = int a^2 dN / a*^2 times the single-grain rates at a* (exact for rates
    proportional to a^2 and a^3). How many radii a bin needs is measured by
    ``diagnostics/gsd/check_area_node_tables.py``.
    """

    def __init__(self, material, grid, radii_per_bin=3, groups=AGN_GROUPS, cache=True, lattice=True,
                 sticking="wd01", rho=None, method=None):
        if radii_per_bin < 2:
            raise ValueError("radii_per_bin must be at least 2 (the bin's edges)")
        self.material, self.grid, self.method = material, grid, method
        self.wdb06_material = charging_material(material)
        self.lattice = bool(lattice)
        self.sticking = sticking
        if sticking != "wd01" and rho is None:
            from .materials import get_material

            rho = get_material(material).rho
        self.rho = rho
        self.L0, self.L1 = (np.asarray(g, float) for g in groups)
        a_e = grid.a_edges
        self.node_radii = [np.geomspace(a_e[i], a_e[i + 1], radii_per_bin) for i in range(grid.n_bins)]
        self.node_tables = [[bin_table(self.wdb06_material, float(r), self.L0, self.L1, cache) for r in radii]
                            for radii in self.node_radii]
        self.a = None           # the area nodes and blended tables of the last solve (``ChargeState`` reads them)
        self.tables = None

    def table_at(self, i, a):
        """Bin i's table at radius a (inside the bin)."""
        r = self.node_radii[i]
        j = int(np.clip(np.searchsorted(r, a, side="right") - 1, 0, r.size - 2))
        return blend_table(self.node_tables[i][j], self.node_tables[i][j + 1], a, self.sticking, self.rho)

    def solve(self, N, M, photons: PhotonGroups, T, n_e, ions: Sequence | None = None, G0_bg=0.0, c_red=C_CGS):
        """The charge state at every bin's area node, and the bins' weights: (``ChargeState`` on the area nodes,
        w [cm^-3]). A bin's photoelectric heating is w_i Gamma_i, and so on."""
        if not (np.allclose(photons.L0, self.L0) and np.allclose(photons.L1, self.L1)):
            raise ValueError("photon groups differ from the tables' groups")
        from .closures import area_node

        a_star, w = area_node(self.grid, N, M, self.method)
        self.a = a_star
        self.tables = [rg.GroupChargingTable(self.table_at(i, a), egy=photons.egy) for i, a in enumerate(a_star)]
        ions = [(n_e, 1.0, M_H)] if ions is None else list(ions)
        res = [gt.solve(photons.N, c_red, T, n_e, ions, G0_bg=G0_bg, recomb=None, lattice=self.lattice)
               for gt in self.tables]
        env = {"T": float(T), "n_e": float(n_e), "ions": ions, "G0_bg": float(G0_bg),
               "N": np.array(photons.N, float), "c_red": float(c_red)}
        return ChargeState(self, res, env), w


class ChargeState:
    """Charge distributions on a ``ChargingModel``'s radius grid (or an ``AreaNodeCharging``'s area nodes)."""

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
