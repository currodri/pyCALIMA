"""The Octopus TD-DFT cross-section library of PAHs as sigma(E, N_C, Z), 0.02 eV to 10 keV.

The library is 92 real-time TD-DFT calculations (docs/physics/pahchem-uv-absorption.md): pyrene, coronene and ovalene (C16, C24, C32;
charges -1 to +2), eight non-compact isomers of C24-C42 (charges -1 to +2), and the circumcoronene family with non-compact isomers
(C54, C96, C150, C216; charges -1 to +2). The spectra are
assembled to 10 keV by ``diagnostics/pahchem/octopus/build_library.py`` and stored per carbon atom in
``external_data/pahchem/pah_uv_library.npz`` (``anchors`` there).

**The rule.** sigma(E, N_C, Z) = N_C s(E; N_C, Z), with the per-carbon spectrum s

- interpolated linearly in ln N_C between the two anchor sizes that bracket N_C (16, 24, 32, 42, 54, 96, 150, 216), and held at the nearest anchor
  outside [16, 216];
- of the charge Z itself for -1 <= Z <= +2; Z >= +3 takes the dication's (pahchem's cold-limit ladder runs to ``impact.coulomb_limit``, +10 for C54);
- every anchor is a computed state (pyrene, coronene and ovalene were run as dications on megatron, 2026-10-30, with the neutral's
  geometry: coronene a triplet, the others closed-shell singlets); ``proxy`` flags an anchor that takes another charge's, none now;
- the hydrogen count does not enter. Above ~80 eV, where H 1s adds to the atomic sum, the spectra carry the library's mean H/C.

Two options select what the anchors are. ``positions``: ``"shifted"`` (default) moves the neutrals' bands below 6 eV up by 0.4 eV
(ad hoc, docs section 5); ``"tdlda"`` keeps TD-LDA's positions. ``structure``: ``"mixed"`` (default) uses the equal-weight mean over every
library molecule of a size (the C24-C42 isomers and the five C54, four C96 and two C150 ones; at C16 only pyrene exists, so it equals
``"compact"`` there; C42H24_3973 is missing, its +1 state never ran), ``"compact"`` the circumcoronene family alone. No compact C42 was run: that anchor is interpolated from C32 and
C54.

**Accuracy of the rule** (docs section 4e): predicting a held-out size from its two neighbours reproduces the per-carbon absorption rates
of the Mathis, Draine 1978 and stellar fields to 2-8 % for C54 and above, up to ~14 % (38 % for a 6000 K blackbody) at C24. The isomer
structure changes the rates at one size by 5-47 %. (Before the small dications were run, the cation stood in for them: that proxy was
4-28 % low in the Mathis field and within 3 % above 13.6 eV.)
"""
from __future__ import annotations

import hashlib
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


@lru_cache(maxsize=4)
def fingerprint(path=None) -> str:
    """The first 10 hex digits of the anchor file's SHA-1: part of the cache keys, so that a rebuilt library does not reuse old bases."""
    return hashlib.sha1(Path(path or default_path()).read_bytes()).hexdigest()[:10]


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


#: the measured C K edge the library's anchors carry when built with ``--ck-edge`` (``k_edge`` in the anchor file): pyrene, Gordon+ 2003
#: Fig. 1 (ISEELS), oscillator strength per carbon atom [1e-2 per eV], digitised (``RAMSES_dev/pahspec/experiment/ck_edge``)
K_EDGE_FILE = ("pahchem", "gordon03_pyrene_kedge.dat")
#: where the measured edge has handed over to the atomic C 1s [eV] (build_library.py's K_BLEND_END)
K_BLEND_END_EV = 310.0


@lru_cache(maxsize=1)
def _k_edge_curve():
    return tuple(np.loadtxt(_paths.get_external_data_path(*K_EDGE_FILE), unpack=True))


def sigma_k_shell(nc, energy_ev, path=None) -> np.ndarray:
    """The C 1s part of ``sigma_library`` [cm^2], as ``build_library.sigma_1s`` adds it to every anchor: Verner's atomic C 1s
    (``wdb06_atomic``, 291 eV step) times ``nc``, or, when the anchors carry the measured edge (``k_edge``), the pyrene curve per
    carbon to its last point (296 eV; the 1s -> pi* resonances from 284 eV) and then the atom plus the excess at that point,
    fading to zero by K_BLEND_END_EV. The same for every charge."""
    from pycalima.models.dust_charge import wdb06_atomic as W
    from pycalima.pahchem import malloci2007 as M

    E = np.asarray(energy_ev, dtype=float)
    atom = nc * W.photoionisation_cross_section("C", "1s", E)
    if not _table(str(path or default_path()))["k_edge"]:
        return atom
    Em, f = _k_edge_curve()
    to_cm2 = nc * 1e-2 * M.MB_EV_PER_F * M.MB_CM2
    e1 = Em[-1]
    end = to_cm2 * max(float(np.mean(f[Em > e1 - 0.3])), 0.0)
    excess = end - nc * W.photoionisation_cross_section("C", "1s", np.array([e1]))[0]
    w = np.cos(0.5 * np.pi * np.clip((E - e1) / (K_BLEND_END_EV - e1), 0.0, 1.0)) ** 2
    meas = to_cm2 * np.clip(np.interp(E, Em, f), 0.0, None)
    return np.where(E < Em[0], 0.0, np.where(E <= e1, meas, atom + excess * w))
