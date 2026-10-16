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

WDB06 yields exist for graphite and silicate only: carbonaceous materials
are charged as graphite and all others as silicate (``charging_material``),
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
    """WDB06 material used to charge ``material``: graphite or silicate."""
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
    """Charge solves for one material on a grid of radii [cm]."""

    def __init__(self, material, a_grid, groups=AGN_GROUPS, cache=True):
        self.material = material
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
        mass [g]) of the ions captured by grains; default H+ with n = n_e."""
        if not (np.allclose(photons.L0, self.L0) and np.allclose(photons.L1, self.L1)):
            raise ValueError("photon groups differ from the tables' groups")
        if self._egy is None or not np.allclose(self._egy, photons.egy):
            for gt in self.tables:
                gt.set_group_energies(photons.egy)
            self._egy = np.array(photons.egy, float)
        ions = [(n_e, 1.0, M_H)] if ions is None else list(ions)
        res = []
        for k, gt in enumerate(self.tables):
            r = gt.solve(photons.N, c_red, T, n_e, ions, G0_bg=G0_bg, recomb=None,
                         warm=self._warm[k])
            self._warm[k] = r
            res.append(r)
        return ChargeState(self, res, dict(T=float(T), n_e=float(n_e), ions=ions,
                                            G0_bg=float(G0_bg), N=np.array(photons.N, float),
                                            c_red=float(c_red)))


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
        """Integer charges and exact P(Z) at grid radius k."""
        r = self.results[k]
        gt = self.model.tables[k]
        if r["discrete"]:
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


def destroy(grid, N, M, rate, dt):
    """Remove grains at rate(a) [s^-1] over dt, exactly per bin node
    (``collisions.bin_nodes``): returns (N, M, removed mass)."""
    from .collisions import bin_nodes

    m, w = bin_nodes(grid, N, M)
    lam = np.asarray(rate(grid.radius(m)), float)
    keep = np.exp(-np.maximum(lam, 0.0) * dt)
    N1 = (w * keep).sum(axis=1)
    M1 = (w * m * keep).sum(axis=1)
    return N1, M1, float(np.asarray(M, float).sum() - M1.sum())
