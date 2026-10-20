"""Per-star stellar ejecta for AGB and core-collapse channels.

Reads the VizieR-style "simplified" tables bundled in
``pycalima/galaxysam/yield_files/yield_files`` (one row per initial mass and
species) and exposes, for each metallicity of a yield set, the **gross**
ejected mass of each element (column ``M(i)lost``: net yield plus the
initial composition carried out) and the total ejected mass
(``M(i)lostall``).

Two row layouts exist:

- 7 columns, massive stars (LC18, Kobayashi SNII/HN):
  ``M0 Z0 vel M1 El M(i)lost M(i)lostall``
- 8 columns, AGB (Karakas 2010, Kobayashi AGB):
  ``M0 Z0 M1 El Yield M(i)lost M(i)0 M(i)lostall``

Tracked elements are the dust solver's ``ELEMENT_NAMES``. Species not in
that list (F, Ne, and any metal the table omits) are counted only through
the total, so ``metals = total - H - He`` stays exact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pycalima.solvers.chemistry_state import ELEMENT_NAMES

ELEMENTS = list(ELEMENT_NAMES)
ZSUN_ASPLUND = 0.01345
MASS_SEPARATRIX = 8.0  # Msun: AGB tables below, core-collapse tables above
# Rows above this are dropped: keeps LC18's 120 Msun models but excludes the
# Kobayashi Z=0 pair-instability models (140-300 Msun, one mass duplicated).
MAX_TABLE_MASS = 120.0

BUNDLED_YIELD_DIR = (Path(__file__).resolve().parents[1]
                     / "galaxysam" / "yield_files" / "yield_files")

_REQUIRED = {"H", "He", "C", "N", "O", "Mg", "Si", "S", "Fe"}
_LAYOUTS = {7: (4, 5, 6), 8: (3, 5, 7)}  # ncols -> (El, M(i)lost, M(i)lostall)
# Fixed-width Fortran columns run together when a value is negative, e.g.
# '0.0000-0.065' (Z0 then M1) in kobayashi13agb_z0: split them.
_GLUED_MINUS = re.compile(r"(?<=\d)-(?=\d)")


@dataclass
class YieldTable:
    """Ejecta of one channel at one metallicity."""

    Z: float
    masses: np.ndarray    # (nM,) initial masses [Msun], ascending
    ejecta: np.ndarray    # (nM, len(ELEMENTS)) gross ejecta [Msun]
    total: np.ndarray     # (nM,) total ejected mass [Msun]
    source: str = ""
    dropped: list = field(default_factory=list)  # masses removed as unphysical

    def per_star(self, m) -> tuple[np.ndarray, np.ndarray]:
        """(ejecta, total) for stars of mass ``m``.

        Linear in mass inside the table. Outside it, the ejected mass
        *fractions* of the nearest tabulated star are held fixed, as in the
        original IDL ``cmp_yield_release.pro``.
        """
        m = np.atleast_1d(np.asarray(m, dtype=float))
        lo, hi = self.masses[0], self.masses[-1]
        ej = np.column_stack([np.interp(m, self.masses, self.ejecta[:, j])
                              for j in range(self.ejecta.shape[1])])
        tot = np.interp(m, self.masses, self.total)
        for edge, sel in ((0, m < lo), (-1, m > hi)):
            scale = m[sel] / self.masses[edge]
            ej[sel] = self.ejecta[edge] * scale[:, None]
            tot[sel] = self.total[edge] * scale
        return ej, tot


def read_simplified_table(path: Path, Z: float, max_mass: float = np.inf,
                          drop_unphysical: bool = False) -> YieldTable:
    """Parse one simplified yield file, validating that it is complete.

    A star cannot eject more than its initial mass, and the tracked elements
    cannot exceed the total ejected. Rows breaking either raise, unless
    ``drop_unphysical`` is set, in which case they are removed and listed in
    ``YieldTable.dropped``. (The 1e-8 Msun slack admits the 1e-10 Msun
    direct-collapse marker rows.)
    """
    rows: dict[float, dict[str, float]] = {}
    totals: dict[float, float] = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        cols = _GLUED_MINUS.sub(" -", line).split()
        if len(cols) not in _LAYOUTS:
            raise ValueError(f"{path}: unexpected row with {len(cols)} columns: {line!r}")
        i_el, i_lost, i_all = _LAYOUTS[len(cols)]
        m0 = float(cols[0])
        if m0 > max_mass:
            continue
        el = "H" if cols[i_el] == "p" else cols[i_el]
        species = rows.setdefault(m0, {})
        if el in species:
            raise ValueError(f"{path}: duplicate entry for M0={m0}, {el}")
        species[el] = float(cols[i_lost])
        totals[m0] = float(cols[i_all])
    if not rows:
        raise ValueError(f"{path}: no yield rows")

    masses = np.array(sorted(rows))
    missing = {m: _REQUIRED - set(rows[m]) for m in masses if _REQUIRED - set(rows[m])}
    if missing:
        raise ValueError(f"{path}: missing elements {missing}")
    ejecta = np.array([[rows[m][el] for el in ELEMENTS] for m in masses])
    total = np.array([totals[m] for m in masses])
    # The IDL-extrapolated Karakas table at Z = 1e-3 Zsun carries ~1e-7 Msun
    # negative ejecta for trace metals; clip those, reject anything larger.
    tiny = ejecta >= -1e-5 * total[:, None]
    if not (np.all(np.isfinite(ejecta)) and np.all(tiny)):
        raise ValueError(f"{path}: negative or non-finite gross ejecta")
    ejecta = np.maximum(ejecta, 0.0)

    bad = (total > masses) | (ejecta.sum(axis=1) > 1.01 * total + 1e-8)
    if np.any(bad) and not drop_unphysical:
        raise ValueError(f"{path}: inconsistent ejected masses for M0={masses[bad]}")
    keep = ~bad
    return YieldTable(Z=Z, masses=masses[keep], ejecta=ejecta[keep], total=total[keep],
                      source=Path(path).name, dropped=masses[bad].tolist())


@dataclass
class YieldSet:
    """AGB + core-collapse ejecta on a common metallicity grid."""

    name: str
    Z: np.ndarray                 # (nZ,) absolute metallicities
    agb: list[YieldTable]
    snii: list[YieldTable]
    mass_separatrix: float = MASS_SEPARATRIX
    notes: dict = field(default_factory=dict)

    def per_star(self, m, iz: int) -> tuple[np.ndarray, np.ndarray]:
        """Gross ejecta (nM, nEl) and total ejecta (nM,) at grid metallicity ``iz``."""
        m = np.atleast_1d(np.asarray(m, dtype=float))
        ej, tot = self.snii[iz].per_star(m)
        low = m < self.mass_separatrix
        ej_a, tot_a = self.agb[iz].per_star(m[low])
        ej[low], tot[low] = ej_a, tot_a
        return ej, tot


# ---------------------------------------------------------------------------
# Yield sets
# ---------------------------------------------------------------------------

# Limongi & Chieffi (2018) + Karakas (2010), the RAMSES-CALIMA default
# (Dubois+2024). Rotation velocity per metallicity follows the average
# prescription used there (build_tables.get_snIIdata).
_LC18_LOGZ = ("-3", "-2", "-1", "-0.6", "-0.3", "0", "0.3")
_LC18_VEL = (150, 100, 50, 50, 50, 50, 50)
# Karakas tables interpolated onto the LC18 grid by the IDL routines; there is
# no Z = 2 Zsun Karakas table, so the solar-ish Z = 0.02 one is reused there.
_KARAKAS_ON_LC18 = ("0.00001345", "0.0001345", "0.001345", "0.003362",
                    "0.006725", "0.01345", "0.02")

_K11_Z = ("0", "0.001", "0.004", "0.008", "0.02", "0.05")


def _lc18_karakas(yield_dir: Path, velocity=None, max_mass=MAX_TABLE_MASS) -> YieldSet:
    Z = ZSUN_ASPLUND * 10.0 ** np.array([float(s) for s in _LC18_LOGZ])
    vels = _LC18_VEL if velocity is None else (int(velocity),) * len(_LC18_LOGZ)
    agb = [read_simplified_table(yield_dir / f"karakas_z{s}_simplified.txt", z)
           for s, z in zip(_KARAKAS_ON_LC18, Z)]
    snii = [read_simplified_table(yield_dir / f"limongichieffi_z{s}_vel{v}_simplified.txt",
                                  z, max_mass)
            for s, v, z in zip(_LC18_LOGZ, vels, Z)]
    return YieldSet("lc18_karakas", Z, agb, snii, notes={"lc18_velocity_kms": list(vels)})


def _kobayashi11(yield_dir: Path, hn_fraction=0.5, max_mass=MAX_TABLE_MASS) -> YieldSet:
    # The SNII tables end with a 50 Msun row of 1e-10 Msun ejecta (M1 = M0):
    # the IDL rewrite's marker for direct collapse, kept as is. The Z = 0 AGB
    # table has 2.2-3.0 Msun rows ejecting more than the initial mass, and the
    # Z = 0 HN table a 100 Msun row with zero total but non-zero elements;
    # those are dropped and interpolated across.
    Z = np.array([float(s) for s in _K11_Z])
    agb, snii, hn_dropped = [], [], []
    for s, z in zip(_K11_Z, Z):
        agb.append(read_simplified_table(yield_dir / f"kobayashi13agb_z{s}_simplified.txt", z,
                                         drop_unphysical=True))
        sn = read_simplified_table(yield_dir / f"kobayashi13snii_z{s}_simplified.txt", z, max_mass)
        if hn_fraction > 0.0:
            hn = read_simplified_table(yield_dir / f"kobayashi13hn_z{s}_simplified.txt", z,
                                       max_mass, drop_unphysical=True)
            hn_dropped.extend(f"hn_Z{z:g}:{m}" for m in hn.dropped)
            hn_ej, hn_tot = hn.per_star(sn.masses)
            mix = np.where((sn.masses >= hn.masses[0]) & (sn.masses <= hn.masses[-1]),
                           hn_fraction, 0.0)
            sn.ejecta = (1.0 - mix[:, None]) * sn.ejecta + mix[:, None] * hn_ej
            sn.total = (1.0 - mix) * sn.total + mix * hn_tot
        snii.append(sn)
    dropped = {f"agb_Z{t.Z:g}": t.dropped for t in agb if t.dropped}
    if hn_dropped:
        dropped["hn"] = hn_dropped
    return YieldSet("kobayashi11", Z, agb, snii,
                    notes={"hn_fraction": hn_fraction, "dropped_unphysical": dropped})


_YIELD_SETS = {"lc18_karakas": _lc18_karakas, "kobayashi11": _kobayashi11}


def load_yield_set(name: str = "lc18_karakas", yield_dir=None, **kwargs) -> YieldSet:
    """Load a yield set by name: ``'lc18_karakas'`` or ``'kobayashi11'``.

    ``yield_dir`` defaults to the tables bundled with the package.
    """
    try:
        builder = _YIELD_SETS[name]
    except KeyError:
        raise ValueError(f"unknown yield set {name!r}; choose from {sorted(_YIELD_SETS)}") from None
    return builder(Path(yield_dir) if yield_dir else BUNDLED_YIELD_DIR, **kwargs)
