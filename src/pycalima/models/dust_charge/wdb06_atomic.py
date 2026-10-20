"""Atomic inputs to the Weingartner, Draine & Barr (2006) photoemission model.

- Subshell photoionisation cross sections of neutral C, O, Mg, Si, Fe: a port
  of D. A. Verner's ``phfit2`` (Verner & Yakovlev 1995; Verner et al. 1996).
- Auger electron spectra per initial vacancy (Dwek & Smith 1996, Tables 4.1-4.2).
- Secondary electron emission parameters (Draine & Salpeter 1979, Table 5).

Tables are read lazily from ``external_data/wdb06`` on first use.
"""

from functools import lru_cache

import numpy as np

from pycalima import _paths

MB_TO_CM2 = 1e-18

# Shell numbering of phfit2: 1 - 1s, 2 - 2s, 3 - 2p, 4 - 3s, 5 - 3p, 6 - 3d, 7 - 4s.
SHELL_INDEX = {"1s": 1, "2s": 2, "2p": 3, "3s": 4, "3p": 5, "3d": 6, "4s": 7}
_SHELL_L = {1: 0, 2: 0, 3: 1, 4: 0, 5: 1, 6: 2, 7: 0}

# Dwek & Smith vacancy notation -> phfit2 shell. L23 is 2p, M23 is 3p.
VACANCY_SHELL = {"K": "1s", "L1": "2s", "L23": "2p", "M1": "3s", "M23": "3p"}


def _read_csv(name):
    path = _paths.get_external_data_path("wdb06", name)
    with open(path) as fh:
        rows = [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
    header = rows[0].split(",")
    return [dict(zip(header, r.split(","))) for r in rows[1:]]


@lru_cache(maxsize=None)
def _verner_tables():
    """{element: (nz, ninn, ph1 {shell_index: params}, ph2 params or None)}."""
    out = {}
    for r in _read_csv("verner_phfit2_neutral.csv"):
        el = r["element"]
        nz, ninn = int(r["nz"]), int(r["ninn"])
        entry = out.setdefault(el, [nz, ninn, {}, None])
        if r["fit"] == "ph1":
            entry[2][int(r["is"])] = tuple(float(r[f"p{i}"]) for i in range(1, 7))
        else:
            entry[3] = tuple(float(r[f"p{i}"]) for i in range(1, 8))
    return {el: tuple(v) for el, v in out.items()}


def shell_threshold_eV(element, shell):
    """Ionisation energy of a subshell of the neutral atom [eV] (WDB06 Table 1)."""
    _, _, ph1, _ = _verner_tables()[element]
    return ph1[SHELL_INDEX[shell]][0]


def shells(element):
    """Occupied subshells of the neutral atom, innermost first."""
    _, _, ph1, _ = _verner_tables()[element]
    inv = {v: k for k, v in SHELL_INDEX.items()}
    return [inv[i] for i in sorted(ph1)]


def photoionisation_cross_section(element, shell, E_eV):
    """Partial photoionisation cross section [cm^2] of one subshell of a
    neutral atom, as phfit2 computes it.

    Below the innermost outer-shell edge, phfit2 represents all outer shells by
    a single Opacity Project fit (ph2) attached to the outermost shell; the
    other outer shells then return zero there. WDB06 relies on this when they
    treat C (2s + 2p), O (2s + 2p), Si (3s + 3p) and Fe (3d + 4s) as one shell.
    """
    nz, nint, ph1, ph2 = _verner_tables()[element]
    is_ = SHELL_INDEX[shell]
    E = np.asarray(E_eV, dtype=float)
    s = np.zeros_like(E)
    if is_ not in ph1:
        return s
    nout = max(ph1)
    if nz > 18:  # phfit2: neutral atoms beyond Ar carry the 4s shell
        nout = 7
    if nz in (15, 17, 19) or (nz > 20 and nz != 26):
        einn = 0.0
    else:
        einn = ph1[nint][0] if nz >= 3 else 1e30

    above = E >= ph1[is_][0]
    if is_ <= nint:
        use1 = above
        use2 = np.zeros_like(above)
    else:
        use1 = above & (E >= einn)
        use2 = above & (E < einn) & (is_ == nout) & (ph2 is not None)

    if np.any(use1):
        Eth, E0, s0, ya, P, yw = ph1[is_]
        y = E[use1] / E0
        q = 0.5 * P - _SHELL_L[is_] - 5.5
        a = s0 * ((y - 1.0) ** 2 + yw ** 2)
        b = np.sqrt(y / ya) + 1.0
        s[use1] = a * y ** q * b ** (-P)
    if np.any(use2):
        E0, s0, ya, P, yw, y0, y1 = ph2
        x = E[use2] / E0 - y0
        z = np.sqrt(x * x + y1 ** 2)
        q = 0.5 * P - 5.5
        a = s0 * ((x - 1.0) ** 2 + yw ** 2)
        b = 1.0 + np.sqrt(z / ya)
        s[use2] = a * z ** q * b ** (-P)
    return s * MB_TO_CM2


@lru_cache(maxsize=None)
def _auger_tables():
    out = {}
    for r in _read_csv("dwek_smith_1996_auger.csv"):
        key = (r["element"], VACANCY_SHELL[r["vacancy"]])
        out.setdefault(key, []).append((float(r["E_eV"]), float(r["P"])))
    return {k: (np.array([e for e, _ in v]), np.array([p for _, p in v]))
            for k, v in out.items()}


def auger_spectrum(element, shell):
    """(E_A [eV], p_A) of the Auger electrons emitted after a vacancy in
    ``shell``; p_A is the mean number per vacancy (may sum above 1). Empty
    arrays for shells without Auger emission (the outermost ones)."""
    return _auger_tables().get((element, shell), (np.empty(0), np.empty(0)))


@lru_cache(maxsize=None)
def secondary_emission_parameters(material):
    """Draine & Salpeter (1979) Table 5 row as a dict of floats."""
    for r in _read_csv("draine_salpeter_1979_secondary.csv"):
        if r["material"] == material:
            return {k: float(v) for k, v in r.items() if k != "material"}
    raise KeyError(material)
