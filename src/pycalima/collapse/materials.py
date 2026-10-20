"""Grain materials of the collapse model, from ``data/external_data/collapse_materials.toml``.

The ten species of Pop III supernova dust (Nozawa et al. 2003; Chiaki et al.
2015). Per species: the growth reactions with their key species (Chiaki+15
Table 2), the bulk density they imply, the mean atomic mass of the solid, and
the source of its optical constants.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

from pycalima import _paths

AMU_CGS = 1.66053906660e-24


@dataclass(frozen=True)
class KeySpecies:
    gas: str            # gas-phase key species
    m_gas: float        # its mass [amu]
    reaction: str
    mu: float           # molecular weight of the solid per key nucleus [amu]
    a0: float           # monomer radius per key nucleus [cm]


@dataclass(frozen=True)
class Material:
    name: str
    formula: str
    surface_amu: float
    optics: str          # "bundled:<file>" or "dataset:<name>"
    optics_ref: str
    keys: tuple[KeySpecies, ...]

    @property
    def rho_s(self):
        """Bulk density mu m_u / (4 pi a0^3 / 3) [g cm^-3], from the first key species."""
        k = self.keys[0]
        return k.mu * AMU_CGS / (4.0 / 3.0 * math.pi * k.a0 ** 3)


def _load_toml(path):
    try:
        import tomllib
    except ModuleNotFoundError:     # Python 3.10
        import tomli as tomllib     # type: ignore[no-redef]
    with open(path, "rb") as fh:
        return tomllib.load(fh)


@lru_cache(maxsize=None)
def materials():
    """All materials, keyed by name."""
    raw = _load_toml(_paths.get_external_data_path("collapse_materials.toml"))
    out = {}
    for name, d in raw["species"].items():
        keys = tuple(KeySpecies(k["gas"], float(k["m_gas"]), k["reaction"], float(k["mu"]),
                                float(k["a0_angstrom"]) * 1e-8) for k in d["key"])
        out[name] = Material(name, d["formula"], float(d["surface_amu"]), d["optics"],
                             d["optics_ref"], keys)
    return out


def get_material(name):
    try:
        return materials()[name]
    except KeyError:
        raise KeyError(f"unknown material {name!r}; have {sorted(materials())}") from None
