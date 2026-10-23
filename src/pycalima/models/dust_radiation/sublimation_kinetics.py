"""Sublimation kinetics of dust materials, by material and dataset.

The bundled file ``data/external_data/sublimation_kinetics/sublimation_models.toml``
(generated from the literature compilation next to it, ``compilation.csv`` and
``COMPILATION_NOTES.md``) gives, for each material, one or more named datasets:

* ``fiducial``: laboratory or vacuum free-evaporation kinetics, the default;
* ``GD89``: Guhathakurta & Draine (1989), for graphite and (astronomical)
  silicate, kept for comparison with the literature that uses it;
* alternatives such as ``W18`` (Woitke et al. 2018), ``N03`` (Nozawa et al.
  2003) or ``vL14a`` (van Lieshout et al. 2014a).

A dataset gives the evaporation flux of the material's units (C atoms, Fe
atoms, formula units, ...) from a bulk solid at temperature T::

    "hkl":   J(T) = sum_c n_c alpha_c(T) P_c(T) / sqrt(2 pi m_c k T),
             log10 P_c [dyn cm^-2] = A10_c - B10_c / T,
             alpha_c = min(1, alpha0_c exp(-alpha_T_c / T)),
    "gd89":  J(T) = alpha P0 exp(-B / T),

with n_c the units carried by one vapour molecule of channel c (e.g. 3 for
C3 from graphite), and a small-grain term theta [K] for the GD89 size
correction B(N) = B - theta N^(-1/3), either given or computed from the
surface energy sigma: theta = (32 pi / 3)^(1/3) (m_unit / rho)^(2/3) sigma / k.

``get_sublimation_model(material, dataset)`` returns a ``SublimationModel``;
``set_default_dataset`` changes what ``dataset=None`` means everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np

from pycalima import _paths

_KB = 1.380649e-16
_AMU = 1.66053906660e-24

_DEFAULT_DATASET = "fiducial"


def set_default_dataset(name: str) -> None:
    """Dataset used when a function is called with ``dataset=None``."""
    global _DEFAULT_DATASET
    _DEFAULT_DATASET = str(name)


def default_dataset() -> str:
    return _DEFAULT_DATASET


@dataclass(frozen=True)
class VapourChannel:
    vapour: str
    A10: float
    B10_K: float
    mass_amu: float
    units_per_molecule: int
    alpha0: float
    alpha_T_K: float
    source_id: str = ""

    def alpha(self, T):
        return np.minimum(1.0, self.alpha0 * np.exp(-self.alpha_T_K / np.asarray(T, float)))

    def pressure(self, T):
        """Saturation pressure [dyn cm^-2]."""
        return 10.0 ** (self.A10 - self.B10_K / np.asarray(T, float))


@dataclass(frozen=True)
class SublimationModel:
    """Sublimation kinetics of one material in one dataset."""

    material: str
    dataset: str
    name: str
    form: str
    unit: str
    unit_mass_amu: float
    density_gcm3: Optional[float]
    key_species: str
    key_mass_amu: float
    key_per_unit: int
    congruent: bool
    heat_capacity: Optional[str]
    theta_K: Optional[float]
    sigma_erg_cm2: Optional[float]
    channels: tuple = ()
    P0_cm2s: Optional[float] = None
    B_K: Optional[float] = None
    alpha_gd89: Optional[float] = None
    note: str = ""
    reference: str = ""
    material_note: str = ""

    # -- bulk evaporation ---------------------------------------------------
    def flux(self, T):
        """Bulk evaporation flux of units [cm^-2 s^-1] at temperature T."""
        T = np.asarray(T, float)
        if self.form == "gd89":
            return self.alpha_gd89 * self.P0_cm2s * np.exp(-self.B_K / T)
        J = np.zeros_like(T)
        for c in self.channels:
            m = c.mass_amu * _AMU
            J = J + c.units_per_molecule * c.alpha(T) * c.pressure(T) / np.sqrt(
                2.0 * np.pi * m * _KB * T)
        return J

    def log_flux(self, T):
        """ln of ``flux(T)``, finite where the flux itself underflows."""
        T = np.asarray(T, float)
        if self.form == "gd89":
            return np.log(self.alpha_gd89 * self.P0_cm2s) - self.B_K / T
        terms = [np.log(c.units_per_molecule * c.alpha(T))
                 + np.log(10.0) * (c.A10 - c.B10_K / T)
                 - 0.5 * np.log(2.0 * np.pi * c.mass_amu * _AMU * _KB * T)
                 for c in self.channels]
        return np.logaddexp.reduce(np.array(terms), axis=0)

    def condensation_alpha(self, T):
        """Flux-weighted evaporation coefficient, used for the reverse
        (condensation) flux so that detailed balance holds."""
        T = np.asarray(T, float)
        if self.form == "gd89":
            return np.full_like(T, self.alpha_gd89)
        # weights in log space: the pressures underflow at low T
        lw = np.array([np.log(c.units_per_molecule) + np.log(10.0) * (c.A10 - c.B10_K / T)
                       - 0.5 * np.log(c.mass_amu) for c in self.channels])
        w = np.exp(lw - lw.max(axis=0))
        al = np.array([np.broadcast_to(c.alpha(T), T.shape) for c in self.channels])
        return (al * w).sum(axis=0) / w.sum(axis=0)

    @property
    def unit_volume_cm3(self) -> float:
        if self.density_gcm3 is None:
            raise ValueError(f"no bulk density for {self.material!r}; set one in "
                             "sublimation_models.toml or pass it explicitly")
        return self.unit_mass_amu * _AMU / self.density_gcm3

    @property
    def theta(self) -> float:
        """Small-grain term theta [K] of B(N) = B - theta N^(-1/3) (0 if unknown)."""
        if self.theta_K is not None:
            return self.theta_K
        if self.sigma_erg_cm2 is None or self.density_gcm3 is None:
            return 0.0
        return ((32.0 * np.pi / 3.0) ** (1.0 / 3.0) * self.unit_volume_cm3 ** (2.0 / 3.0)
                * self.sigma_erg_cm2 / _KB)

    def effective_binding_energy_K(self, T):
        """Local Arrhenius temperature -d ln J / d(1/T) at T [K]."""
        T = np.asarray(T, float)
        h = 1e-4 * T
        return -(np.log(self.flux(T + h)) - np.log(self.flux(T - h))) / (1 / (T + h) - 1 / (T - h))


def _models_path():
    return _paths.get_external_data_path("sublimation_kinetics", "sublimation_models.toml")


@lru_cache(maxsize=1)
def _load():
    try:
        import tomllib
    except ModuleNotFoundError:  # Python < 3.11
        import tomli as tomllib  # type: ignore[no-redef]
    with open(_models_path(), "rb") as fh:
        return tomllib.load(fh)["material"]


def available_materials() -> List[str]:
    return list(_load())


def available_datasets(material: str) -> List[str]:
    return list(_material(material).get("dataset", {}))


def _material(material: str) -> dict:
    mats = _load()
    if material not in mats:
        raise KeyError(f"unknown material {material!r}; available: {sorted(mats)}")
    m = mats[material]
    if "proxy_of" in m and "dataset" not in m:
        m = {**mats[m["proxy_of"]], **{k: v for k, v in m.items() if k != "dataset"}}
    return m


def get_sublimation_model(material: str, dataset: Optional[str] = None) -> SublimationModel:
    """SublimationModel of ``material`` (e.g. 'graphite', 'silicate', 'iron')
    in ``dataset`` (default: ``default_dataset()``, initially 'fiducial')."""
    # resolve the default before caching, so set_default_dataset takes effect
    return _get_model(material, dataset or _DEFAULT_DATASET)


@lru_cache(maxsize=None)
def _get_model(material: str, dataset: str) -> SublimationModel:
    m = _material(material)
    datasets = m.get("dataset", {})
    if dataset not in datasets:
        raise KeyError(f"material {material!r} has no dataset {dataset!r}; "
                       f"available: {list(datasets)}")
    d = datasets[dataset]
    common = dict(material=material, dataset=dataset, name=m["name"], form=d["form"],
                  unit=m["unit"], key_species=m["key_species"], key_mass_amu=m["key_mass_amu"],
                  key_per_unit=int(m["key_per_unit"]), congruent=bool(m.get("congruent", True)),
                  heat_capacity=m.get("heat_capacity"), note=d.get("note", ""),
                  reference=d.get("reference", ""), material_note=m.get("note", ""))
    if d["form"] == "gd89":
        return SublimationModel(
            unit_mass_amu=d["unit_mass_amu"], density_gcm3=d["density_gcm3"],
            theta_K=d["theta_K"], sigma_erg_cm2=None, P0_cm2s=d["P0_cm2s"], B_K=d["B_K"],
            alpha_gd89=d["alpha"], **common)
    channels = tuple(VapourChannel(c["vapour"], c["A10"], c["B10_K"], c["mass_amu"],
                                   int(c["units_per_molecule"]), c["alpha0"], c["alpha_T_K"],
                                   c.get("source_id", "")) for c in d["channel"])
    return SublimationModel(
        unit_mass_amu=m["unit_mass_amu"], density_gcm3=m.get("density_gcm3"),
        theta_K=d.get("theta_K"), sigma_erg_cm2=d.get("sigma_erg_cm2"), channels=channels,
        **common)
