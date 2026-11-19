"""Grain materials of the GSD operators: one record per material, with sources.

Each record holds what every operator reads for that material, so that they
agree on the bulk density: the optical-property set, the mechanical constants
of shattering (``ShatterMaterial``, its density replaced by the record's), the
sputtering and sublimation inputs and the charging constants. Constants taken
from another material, for want of measured values, are listed in ``proxies``.
An unknown name raises; nothing falls back to another material.

- ``themis_ac``: THEMIS amorphous carbon (Jones et al. 2013, 2017): the
  CM20 optics, rho = 1.6 g/cm^3 (DustEM GRAIN_J13; its large-grain population
  has 1.57) and the DustEM CHRG_CM20 charging constants. Shattering,
  sputtering and the vapour pressure are graphite's: no a-C(:H) values exist.
- ``graphite``: Draine graphite with the HY09 Table 2 density, the control
  for comparison with graphite-based models.
- ``silicate``: astronomical silicate.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from .shattering import GRAPHITE, SILICATE, ShatterMaterial


@dataclass(frozen=True)
class GrainMaterial:
    """Constants of one grain material (cgs)."""

    name: str
    key: str                       # gas key element of the material
    rho: float                     # bulk density [g cm^-3]
    optics: str                    # optical-property set
    shatter: ShatterMaterial       # mechanical constants at density rho
    sputtering: str                # composition of dust_sputtering
    charging: str                  # wdb06_yields material: 'ac', 'cm20' (DustEM), 'graphite', 'silicate'
    sublimation: str               # material of sublimation_kinetics
    proxies: tuple = ()            # entries borrowed from another material
    sources: dict = field(default_factory=dict)

    def sublimation_model(self, dataset="GD89", **overrides):
        """``sublimation_kinetics`` model of the material at this density."""
        from pycalima.models.dust_radiation.sublimation_kinetics import get_sublimation_model

        return replace(get_sublimation_model(self.sublimation, dataset), density_gcm3=self.rho,
                       **overrides)


def _record(name, key, rho, optics, shatter, sputtering, charging, sublimation, proxies=(), sources=None):
    return GrainMaterial(name, key, rho, optics, replace(shatter, name=name, rho=rho), sputtering,
                         charging, sublimation, tuple(proxies), dict(sources or {}))


MATERIALS = {
    "themis_ac": _record(
        "themis_ac", "C", 1.6, "themis_cm20", GRAPHITE, "graphite", "ac", "amorphous_carbon",
        proxies=("shatter", "sputtering", "sublimation"),
        sources=dict(rho="DustEM GRAIN_J13 (Jones+2013; Ysard+2015 corrigendum)",
                     optics="DustEM Q_CM20/G_CM20, Henke below 40 nm (themis_optics)",
                     charging="'ac': W = 4.75 eV, WD01 thresholds and sticking, Kimura 2016 yield, WDB06 above 20 eV; 'cm20' = DustEM CHRG_CM20",
                     shatter="graphite, HY09 Table 2",
                     sputtering="graphite, Nozawa+2006 / Bocchio+2016",
                     sublimation="graphite vapour pressure (sublimation_models.toml amorphous_carbon)")),
    "graphite": _record(
        "graphite", "C", GRAPHITE.rho, "draine_graphite", GRAPHITE, "graphite", "graphite",
        "graphite", sources=dict(rho="HY09 Table 2", shatter="HY09 Table 2")),
    "silicate": _record(
        "silicate", "Si", SILICATE.rho, "astrosilicate", SILICATE, "silicate", "silicate",
        "silicate", sources=dict(rho="HY09 Table 2", shatter="HY09 Table 2")),
}


def get_material(name: str) -> GrainMaterial:
    """The record of ``name``; unknown names raise KeyError."""
    try:
        return MATERIALS[name]
    except KeyError:
        raise KeyError(f"unknown grain material {name!r}; known: {sorted(MATERIALS)}") from None
