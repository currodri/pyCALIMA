"""The Octopus TD-DFT cross-section library of PAHs as sigma(E, N_C, Z), 0.02 eV to 10 keV.

The library is 57 real-time TD-DFT calculations (docs/physics/pahchem-uv-absorption.md): pyrene, coronene and ovalene (C16, C24, C32;
charges -1 to +1), and the circumcoronene family with non-compact isomers (C54, C96, C150, C216; charges -1 to +2). The spectra are
assembled to 10 keV by ``diagnostics/pahchem/octopus/build_library.py`` and stored per carbon atom in
``external_data/pahchem/pah_uv_library.npz`` (``anchors`` there).

**The rule.** sigma(E, N_C, Z) = N_C s(E; N_C, Z), with the per-carbon spectrum s

- interpolated linearly in ln N_C between the two anchor sizes that bracket N_C (16, 24, 32, 54, 96, 150, 216), and held at the nearest anchor
  outside [16, 216];
- of the charge Z itself for -1 <= Z <= +2; Z >= +3 takes the dication's (pahchem reaches +3 for its top state only);
- for the small molecules (C16, C24, C32) the dication takes the cation's spectrum (``proxy``): they were not run as dications;
- the hydrogen count does not enter. Above ~80 eV, where H 1s adds to the atomic sum, the spectra carry the library's mean H/C.

Two options select what the anchors are. ``positions``: ``"shifted"`` (default) moves the neutrals' bands below 6 eV up by 0.4 eV
(ad hoc, docs section 5); ``"tdlda"`` keeps TD-LDA's positions. ``structure``: ``"mixed"`` (default) uses the equal-weight mean over every
library molecule of a size (the five C54, four C96 and two C150 isomers; below C54 only compact molecules exist, so it equals
``"compact"``), ``"compact"`` the circumcoronene family alone.

**Accuracy of the rule** (docs section 4e): predicting a held-out size from its two neighbours reproduces the per-carbon absorption rates
of the Mathis, Draine 1978 and stellar fields to 2-8 % for C54 and above, up to ~14 % (38 % for a 6000 K blackbody) at C24. The isomer
structure changes the rates at one size by 5-47 %, and the dication takes the cation's spectrum below C54 at a cost of 10-23 % in the
Mathis field.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from pycalima import _paths

POSITIONS = ("shifted", "tdlda")
#: the order of the anchor table's structure axis (``anchors`` in diagnostics/pahchem/octopus/build_library.py); the default is "mixed"
STRUCTURES = ("compact", "mixed")


def default_path() -> Path:
    return _paths.get_external_data_path("pahchem", "pah_uv_library.npz")


@lru_cache(maxsize=4)
def _table(path: str):
    with np.load(path) as z:
        return {"energy_ev": z["energy_ev"].astype(float), "nc": z["nc"].astype(float), "charges": z["charges"],
                "sigma": z["sigma_per_c_cm2"], "proxy": z["proxy"], "shift_ev": float(z["shift_ev"]),
                "k_edge": bool(z["k_edge"])}


def library(path=None) -> dict:
    """The anchor table: ``energy_ev``, ``nc``, ``charges``, ``sigma`` [positions, structure, size, charge, energy] in cm^2 per carbon,
    ``proxy`` [size, charge] (True where the dication took the cation's spectrum), ``shift_ev`` and ``k_edge``."""
    return _table(str(path or default_path()))


def per_carbon(nc, charge, energy_ev, positions="shifted", structure="mixed", path=None) -> np.ndarray:
    """The absorption cross section per carbon atom [cm^2] at ``energy_ev`` [eV]; zero outside 0.02 eV - 10 keV."""
    if positions not in POSITIONS:
        raise ValueError(f"positions must be one of {POSITIONS}, not {positions!r}")
    if structure not in STRUCTURES:
        raise ValueError(f"structure must be one of {STRUCTURES}, not {structure!r}")
    charge = int(charge)
    if charge < -1:
        raise ValueError(f"the library has charges -1 and above, not {charge}")
    t = library(path)
    q = int(np.nonzero(t["charges"] == min(charge, int(t["charges"].max())))[0][0])
    sizes = t["nc"]
    n = float(np.clip(nc, sizes[0], sizes[-1]))
    i = int(min(np.searchsorted(sizes, n, side="right") - 1, len(sizes) - 2))
    w = (np.log(n) - np.log(sizes[i])) / (np.log(sizes[i + 1]) - np.log(sizes[i]))
    tab = t["sigma"][POSITIONS.index(positions), STRUCTURES.index(structure)]
    s = (1.0 - w) * tab[i, q].astype(float) + w * tab[i + 1, q].astype(float)
    return np.interp(np.asarray(energy_ev, dtype=float), t["energy_ev"], s, left=0.0, right=0.0)


def sigma_library(nc, charge, energy_ev, positions="shifted", structure="mixed", path=None) -> np.ndarray:
    """The absorption cross section of a PAH with ``nc`` carbon atoms and charge ``charge`` [cm^2]: ``nc`` times ``per_carbon``."""
    return nc * per_carbon(nc, charge, energy_ev, positions, structure, path)
