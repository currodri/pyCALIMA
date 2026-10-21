"""Mai et al. (2025, MNRAS 541, 3073; arXiv:2503.05120): machine-learning MD IR spectra of 1704 neutral PAHs.

Classical MD on a neural-network force field with a neural-network dipole model (trained on B3LYP/4-31G), at 50,
300 and 600 K, for the PAHdb neutrals of their sample (up to N_C = 216). Their Zenodo record (doi:10.5281/
zenodo.15771437) holds the spectra (peak-normalised, no quantum correction) under CC BY-NC-SA 4.0, so it is
the fetch dataset ``mai2025-mlmd``, never bundled.

Classical MD puts kT in every mode: a 600 K run has the vibrational energy per mode of a quantum molecule
at ~1250 K, and its CH stretches are excited, which adds their self-anharmonicity to the shifts.
``ch_band`` measures the CH-stretch band (2950-3150 cm^-1, smoothed) of one spectrum.
"""

from __future__ import annotations

import hashlib
import io
import re
import zipfile
from pathlib import Path

import numpy as np

DATASET = "mai2025-mlmd"
URL = "https://zenodo.org/api/records/15771437/files/Supplementary.zip/content"
SHA256 = "8c3688f9f18e0f6aaf43685cd6ab56e2329e7920b1739f8b1387324609a6f28f"
ARCHIVE = "Supplementary.zip"
TEMPERATURES = (50, 300, 600)


def fetch(output_dir):
    """Download the Zenodo archive into ``output_dir`` and check its sha256 (the registry's fetcher)."""
    from pycalima._datasets import DatasetUnavailable, _download

    dest = Path(output_dir) / ARCHIVE
    _download(URL, dest)
    h = hashlib.sha256(dest.read_bytes()).hexdigest()
    if h != SHA256:
        raise DatasetUnavailable(f"{dest}: sha256 {h}, expected {SHA256}")
    return Path(output_dir)


def _root(root):
    if root is not None:
        return Path(root)
    from pycalima._datasets import ensure_dataset

    return ensure_dataset(DATASET)


def spectra(temperature_k, root=None):
    """Yield (name, n_C, n_H, PAHdb UID, wavenumber [cm^-1], normalised intensity) of the 1704-PAH sample at
    ``temperature_k`` (50, 300 or 600)."""
    if temperature_k not in TEMPERATURES:
        raise ValueError(f"temperature must be one of {TEMPERATURES}")
    with zipfile.ZipFile(_root(root) / ARCHIVE) as z:
        inner = zipfile.ZipFile(io.BytesIO(z.read(
            f"Supplementary_Data_Code/outputs/IR_txt_phdb_1704/IR_txt_{temperature_k}K_qm0.zip")))
        for n in sorted(inner.namelist()):
            m = re.match(r".*/C(\d+)H(\d+)_(\d+)\.txt$", n)
            if not m:
                continue
            d = np.loadtxt(io.BytesIO(inner.read(n)), comments="#")
            yield n.split("/")[-1][:-4], int(m.group(1)), int(m.group(2)), int(m.group(3)), d[:, 0], d[:, 1]


def ch_band(nu_cm, intensity, lo_cm=2950.0, hi_cm=3150.0, smooth_cm=10.0):
    """(peak, centroid, red half-width, blue half-width) [cm^-1] of the CH-stretch band, after a Gaussian
    smoothing of ``smooth_cm`` (sigma; the 600 K spectra are noisy) and a floor taken at the window's low end."""
    from scipy.ndimage import gaussian_filter1d

    w = (nu_cm >= lo_cm) & (nu_cm <= hi_cm)
    x = nu_cm[w]
    step = float(np.median(np.diff(x)))
    y = gaussian_filter1d(intensity[w], smooth_cm / step)
    y = np.clip(y - np.median(y[:15]), 0.0, None)
    i = int(np.argmax(y))
    half = 0.5 * y[i]
    a = i
    while a > 0 and y[a - 1] >= half:
        a -= 1
    b = i
    while b < y.size - 1 and y[b + 1] >= half:
        b += 1
    return float(x[i]), float((x * y).sum() / y.sum()), float(x[i] - x[a]), float(x[b] - x[i])
