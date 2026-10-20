"""CosmicPAH-IRDB (IRAP Toulouse): temperature laws of PAH IR bands and the spectra behind them.

The database of Demyk et al. (2026, arXiv:2607.16018) holds gas-phase
absorption spectra of pyrene and four aliphatic pyrenes at 423-673 K, and per
band and component the anharmonicity laws::

    position(T) = chi0' + chi1' T,      FWHM(T) = chi0'' + chi1'' T,

piecewise over temperature ranges, with the Lorentzian fraction of a
pseudo-Voigt profile.

- **Laws** come from the per-dataset JSON (``/machine/<molecule;formula>/
  <phase>/<category>/<id>``), by the dataset IDs of the paper's analysis
  (``DATASETS``): for 6H- and Me-pyrene the rotation-corrected version 1.1.2,
  which the archives do not serve.
- **Spectra** come from the per-molecule archive (``/science/<molecule>/
  <phase>/e/file/ttttt``): absorbance against wavenumber (2 cm^-1 resolution,
  400-7500 cm^-1) at each temperature, version 1.1.1. They are relative
  absorbances, not cross sections.

The site states no licence; it asks that the dataset reference be cited. The
data are therefore fetched into the user's cache (``calima-fetch-data fetch
cosmicpah-irdb``, dataset ``cosmicpah-irdb``), never bundled. ``MANIFEST.json``
records the URLs, the download date and a sha256 per file, since the site
publishes no checksums.

The JSON holds a summary list (one row per band) and a Python ``repr`` of the
full fit results with numpy ``array(...)`` calls. Only the summary is read, and
its structure is checked; the ``repr`` is never evaluated.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BASE_URL = "https://www.cosmicpah-irdb.ovgso.fr"
DATASET = "cosmicpah-irdb"

#: Demyk+26 molecules: (IRDB molecule, formula, dataset ID of the paper's laws, IRDB version)
DATASETS = {
    "pyrene": ("Pyrene", "C16H10", 22, "1.1.1 multi component + rotational correction"),
    "2h-pyrene": ("4,5-dihydropyrene", "C16H12", 9, "1.1.1"),
    "4h-pyrene": ("4,5,9,10-tetrahydropyrene", "C16H14", 12, "1.1.1"),
    "6h-pyrene": ("1,2,3,6,7,8-hexahydropyrene", "C16H16", 4, "1.1.2 Rotational correction"),
    "me-pyrene": ("1-methylpyrene", "C17H12", 17, "1.1.2 Rotational correction"),
}
PHASE, CATEGORY = "Gas", "Experimental"


def _laws_file(key):
    return f"laws/ID{DATASETS[key][2]}.json"


def _archive_file(key):
    return f"archives/{key}_gas.zip"


#: every file the fetcher writes (``data/registry.toml`` lists the same)
FILES = tuple(f(k) for k in DATASETS for f in (_laws_file, _archive_file))


def _laws_url(key):
    mol, formula, i, _ = DATASETS[key]
    return f"{BASE_URL}/machine/{mol};{formula}/{PHASE}/{CATEGORY}/{i}"


def _archive_url(key):
    return f"{BASE_URL}/science/{DATASETS[key][0]}/{PHASE}/e/file/ttttt"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(output_dir):
    """Download the laws and archives of ``DATASETS`` into ``output_dir`` (the registry's fetcher)."""
    from pycalima._datasets import _download

    out = Path(output_dir)
    manifest = {"source": BASE_URL, "fetched": datetime.date.today().isoformat(), "files": {}}
    for key in DATASETS:
        for rel, url in ((_laws_file(key), _laws_url(key)), (_archive_file(key), _archive_url(key))):
            _download(url, out / rel)
            manifest["files"][rel] = {"url": url, "sha256": _sha256(out / rel)}
        parse_laws(json.loads((out / _laws_file(key)).read_text()))          # fail now, not at first use
        with zipfile.ZipFile(out / _archive_file(key)) as z:
            if z.testzip() is not None:
                raise ValueError(f"{key}: corrupt archive")
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    return out


# ---------------------------------------------------------------------------
# laws
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Piece:
    """``c0 + c1 T + c2 T^2`` over [t_min, t_max] K."""
    c0: float
    c1: float
    c2: float
    t_min: float
    t_max: float

    def __call__(self, t):
        return self.c0 + self.c1 * t + self.c2 * t * t


@dataclass(frozen=True)
class BandLaw:
    """One band (its main component): position and FWHM against temperature [cm^-1].

    ``label_cm`` is the band position the database lists (at 523 K, Demyk+26
    Table 4); ``lorentz`` the Lorentzian fraction of the pseudo-Voigt;
    ``intensity`` the fitted area at the lowest temperature (relative
    absorbance units of that dataset).
    """
    label_cm: float
    lorentz: float
    intensity: float
    position_pieces: tuple[Piece, ...]
    fwhm_pieces: tuple[Piece, ...]

    @property
    def t_range(self):
        ps = self.position_pieces + self.fwhm_pieces
        return min(p.t_min for p in ps), max(p.t_max for p in ps)

    @staticmethod
    def _evaluate(pieces, t, extrapolate_up):
        t = np.asarray(t, dtype=float)
        lo, hi = pieces[0], pieces[-1]
        out = np.full(t.shape, lo(lo.t_min))     # below the data: the lowest-T value
        for p in pieces:                         # ranges meet end to end (_pieces); the upper piece wins at a joint
            out = np.where(t >= p.t_min, p(t), out)
        if not extrapolate_up:
            out = np.where(t > hi.t_max, hi(hi.t_max), out)
        return out

    def position(self, t):
        """Position at T [K]: the lowest-T value below the data, linear extrapolation above (Demyk+26 Sect. 2.4)."""
        return self._evaluate(self.position_pieces, t, extrapolate_up=True)

    def fwhm(self, t):
        """FWHM at T [K]. Above the data the last piece is extrapolated only if it rises; a falling width
        (the strongly blended components, whose chi1'' < 0 Demyk+26 call unphysical) is held."""
        rises = self.fwhm_pieces[-1].c1 > 0
        return self._evaluate(self.fwhm_pieces, t, extrapolate_up=rises)


def _pieces(rows, what):
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{what}: expected a non-empty list of pieces")
    out = []
    for r in rows:
        if not (isinstance(r, list) and len(r) == 5 and all(isinstance(v, (int, float)) for v in r)):
            raise ValueError(f"{what}: a piece must be [c0, c1, c2, t_min, t_max], not {r!r}")
        out.append(Piece(*map(float, r)))
    out.sort(key=lambda p: p.t_min)
    for a, b in zip(out, out[1:]):
        if abs(b.t_min - a.t_max) > 1e-6:
            raise ValueError(f"{what}: pieces must meet end to end, not {a.t_min:g}-{a.t_max:g} then {b.t_min:g}-{b.t_max:g}")
    return tuple(out)


def parse_laws(payload):
    """``BandLaw`` per band from a CosmicPAH-IRDB ``/machine`` response (a list of three elements whose
    third is the band summary). Structure is checked; nothing is evaluated."""
    if not (isinstance(payload, list) and len(payload) == 3 and isinstance(payload[2], list)):
        raise ValueError("not a CosmicPAH-IRDB dataset response (expected a 3-element list)")
    laws = []
    for row in payload[2]:
        if not (isinstance(row, list) and len(row) == 7 and row[3] == "P" and row[5] == "F"
                and all(isinstance(v, (int, float)) for v in row[:3])):
            raise ValueError(f"unexpected band summary row {row!r:.200}")
        laws.append(BandLaw(float(row[0]), float(row[1]), float(row[2]),
                            _pieces(row[4], f"band {row[0]} position"), _pieces(row[6], f"band {row[0]} FWHM")))
    return sorted(laws, key=lambda b: -b.label_cm)


def _root(root):
    if root is not None:
        return Path(root)
    from pycalima._datasets import ensure_dataset

    return ensure_dataset(DATASET)


def laws(key, root=None):
    """The band laws of molecule ``key`` (``DATASETS``), from the fetched dataset."""
    return parse_laws(json.loads((_root(root) / _laws_file(key)).read_text()))


def band(key, label_cm, root=None, tol_cm=3.0):
    """The ``BandLaw`` of ``key`` listed at ``label_cm`` (within ``tol_cm``)."""
    found = [b for b in laws(key, root) if abs(b.label_cm - label_cm) <= tol_cm]
    if len(found) != 1:
        raise KeyError(f"{key}: {len(found)} bands within {tol_cm} cm^-1 of {label_cm}")
    return found[0]


# ---------------------------------------------------------------------------
# spectra
# ---------------------------------------------------------------------------

def temperatures(key, root=None):
    """Temperatures [K] of the full spectra in the archive of ``key``."""
    with zipfile.ZipFile(_root(root) / _archive_file(key)) as z:
        return sorted(int(n.rsplit("_", 1)[1][:-5]) for n in z.namelist()
                      if n.startswith("FullSpectra/") and n.endswith("K.txt"))


def spectrum(key, temperature_k, root=None):
    """(wavenumber [cm^-1], relative absorbance) of ``key`` at ``temperature_k``, ascending wavenumber."""
    with zipfile.ZipFile(_root(root) / _archive_file(key)) as z:
        name = next((n for n in z.namelist() if n.startswith("FullSpectra/")
                     and n.endswith(f"_{int(temperature_k)}K.txt")), None)
        if name is None:
            raise KeyError(f"{key}: no spectrum at {temperature_k} K (have {temperatures(key, root)})")
        d = np.loadtxt(z.open(name))
    if d.ndim != 2 or d.shape[1] != 2:
        raise ValueError(f"{name}: expected two columns")
    order = np.argsort(d[:, 0])
    return d[order, 0], d[order, 1]


def pseudo_voigt(nu, centre, fwhm, lorentz):
    """Unit-area pseudo-Voigt: ``lorentz`` x Lorentzian + (1 - ``lorentz``) x Gaussian of the same FWHM."""
    x = np.asarray(nu, dtype=float) - centre
    hw = 0.5 * fwhm
    lor = hw / np.pi / (x * x + hw * hw)
    sig = fwhm / np.sqrt(8.0 * np.log(2.0))
    gau = np.exp(-0.5 * (x / sig) ** 2) / (sig * np.sqrt(2.0 * np.pi))
    return lorentz * lor + (1.0 - lorentz) * gau
