"""Photoionisation of nine small neutral PAHs (C13-C20), 7-22 eV: Hrodmarsson et al. 2025, A&A 698, A202.

SOLEIL DESIRS with the DELICIOUS III i2PEPICO spectrometer. The data (Zenodo 10.5281/zenodo.15271348, deposited by
Verstraete et al., CC BY 4.0; ``calima-fetch-data fetch hrodmarsson2025-pes``, fetched, not bundled for its size) hold
per molecule the photoelectron matrices (counts per photon-energy bin and electron kinetic-energy bin) and the
total ion yields, both corrected for the photon flux.

- ``molecule``: the matrix and the total ion yield of one molecule, on bin centres.
- ``mean_share``: <KE>/(hv - IE) per photon-energy bin (``photo.PARTITION_NODES`` come from these above 13.6 eV).
- ``ionisation_cross_section``: the total ion yield scaled to ``peak`` Mb per C on the 17 eV maximum, as the paper
  does (their Fig. 4, 22 Mb/C after Malloci's TD-DFT). The scale is an assumption, the shape is measured.

``IE`` holds the paper's Table 2 adiabatic ionisation energies. Phenanthrene's sample ran out at high photon
energy (their Sect. 3.2), so its yield has no usable 17 eV scale (``DEPLETED``).
"""
from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import numpy as np

DATASET = "hrodmarsson2025-pes"
URL = "https://zenodo.org/api/records/15271348/files/Data%20files.zip/content"
ARCHIVE = "Data files.zip"
SHA256 = "cedbb87ce2eb94462863d3a7fd2423a48b2dac30c5db42701243565df4234839"
#: name -> (adiabatic IE [eV], N_C), their Table 2
IE = {"Fluorene": (7.91, 13), "Phenanthrene": (7.90, 14), "3-Methylphenanthrene": (7.80, 15),
      "4-Methylphenanthrene": (7.72, 15), "Fluoranthene": (7.90, 16), "Benzophenanthrene": (7.60, 18),
      "Triphenylene": (7.85, 18), "Corannulene": (7.90, 20), "Perylene": (7.00, 20)}
DEPLETED = ("Phenanthrene",)


def fetch(output_dir):
    """Download the Zenodo archive into ``output_dir`` and check its sha256 (the registry's fetcher)."""
    from pycalima._datasets import DatasetUnavailable, _download

    dest = Path(output_dir) / ARCHIVE
    _download(URL, dest)
    h = hashlib.sha256(dest.read_bytes()).hexdigest()
    if h != SHA256:
        raise DatasetUnavailable(f"{dest}: sha256 {h}, expected {SHA256}")
    return Path(output_dir)


def _archive(root):
    if root is None:
        from pycalima._datasets import ensure_dataset

        root = ensure_dataset(DATASET)
    return zipfile.ZipFile(Path(root) / ARCHIVE)


def _axis(text):
    """The first block of a file: one value per line after its header (bin edges), up to the next header."""
    out = []
    for ln in text.splitlines()[1:]:
        if "\t" in ln:
            break
        if ln.strip():
            try:
                out.append(float(ln))
            except ValueError:
                break
    return np.array(out)


def _centres(edges):
    return 0.5 * (edges[1:] + edges[:-1])


def molecule(name, root=None):
    """dict(hv_pes [eV], ke [eV], matrix [hv, ke], hv [eV], tiy) of one molecule (``IE`` keys)."""
    with _archive(root) as z:
        files = [n for n in z.namelist() if n.startswith(f"Data files/{name}/") and n.endswith(".txt")]
        read = lambda suffix: next(z.read(n).decode() for n in files if n.endswith(suffix))
        mats = sorted(n for n in files if "_PESMatrix_" in n)
        total = [n for n in mats if n.endswith("_Total_amu.txt")] or mats[-1:]
        mtext = z.read(total[0]).decode()
        hv_edges, ke_edges = _axis(read("_SPES.txt")), _axis(read("_eKE.txt"))
        t = np.genfromtxt(io.StringIO(read("_TIY.txt")), names=True, delimiter="\t")
    matrix = np.array([ln.split("\t") for ln in mtext.splitlines()[1:] if ln.strip()], dtype=float)
    names = t.dtype.names
    e = next((n for n in names if n.lower() == "evwave"), None)
    hv_pes = _centres(hv_edges)
    hv = np.asarray(t[e] if e else hv_pes[:t.size], dtype=float)       # Fluoranthene's yield table has no energies
    tiy = t["PEPICO_Total_amu"] if "PEPICO_Total_amu" in names else sum(t[n] for n in names if n.startswith("PEPICO_"))
    return dict(hv_pes=hv_pes, ke=_centres(ke_edges), matrix=matrix, hv=hv, tiy=np.asarray(tiy, dtype=float))


def mean_share(name, root=None):
    """(hv [eV], <KE>/(hv - IE)) per photon-energy bin above IE + 0.5 eV."""
    m = molecule(name, root)
    P = np.clip(m["matrix"], 0.0, None)
    mean = (P @ m["ke"]) / np.maximum(P.sum(axis=1), 1e-300)
    ie = IE[name][0]
    k = m["hv_pes"] > ie + 0.5
    return m["hv_pes"][k], mean[k] / (m["hv_pes"][k] - ie)


def ionisation_cross_section(name, peak_mb_per_c=22.0, root=None):
    """(hv [eV], sigma_ion [cm^2]): the total ion yield scaled to ``peak_mb_per_c`` N_C on its 15-19 eV maximum."""
    if name in DEPLETED:
        raise ValueError(f"{name}: sample depleted at high photon energy, no 17 eV scale")
    m = molecule(name, root)
    k = (m["hv"] > 15.0) & (m["hv"] < 19.0)
    return m["hv"], m["tiy"] * peak_mb_per_c * IE[name][1] * 1e-18 / np.max(m["tiy"][k])
