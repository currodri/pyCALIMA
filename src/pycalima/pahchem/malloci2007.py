"""The Cagliari TD-DFT photo-absorption cross sections of Malloci, Joblin & Mulas (2007, Chem. Phys. 332, 353).

40 PAHs (C10-C66), each as anion, neutral, cation and dication where computed, 0-30 eV in 0.05 eV steps:
real-time, real-space TD-DFT (Octopus) on DFT geometries (Malloci, Mulas & Joblin 2004, A&A 426, 105). The site
(https://www.dsf.unica.it/~gmalloci/pahs) states no licence, so the files are fetched, never bundled
(``calima-fetch-data fetch malloci2007-spectra``), and kept as served.

- ``fetch``: the registry's fetcher; writes ``<name>_<state>`` files and ``index.csv`` (name, formula, N_C, N_H,
  states).
- ``spectra``: {name: {Z: (E [eV], sigma [cm^2])}} with the molecule's N_C, N_H.
- ``oscillator_strength``: int sigma dE / (pi e^2 h / m_e c), the Thomas-Reiche-Kuhn count.

**Two served files are not what they are labelled** (``INCONSISTENT``; ``diagnostics/pahchem/check_malloci_sum_rule.py``).
Up to 30 eV every other state of every molecule holds 0.61-0.63 of its C 2s2p + H 1s electrons in oscillator
strength, and each added electron adds 0.3-2.5 (median 1.0). ``coronene_anion`` holds 142.8, the share of
C54H18- (circumcoronene's anion: 143.2, same 16.45 eV peak), not of C24H12- (expected ~67); ``benzocoronene_cation``
holds 14 more than its neutral. ``spectra`` and ``sigma_of`` leave them out unless asked. The ``coronene_anion``
file is the one Berné+22's code (``anions/coronene_anion.txt``) carries; pyCALIMA bundled it too until 2026-10-24,
when its bundled Malloci copies were dropped for this dataset.
"""
from __future__ import annotations

import csv
import re
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

DATASET = "malloci2007-spectra"
BASE_URL = "https://www.dsf.unica.it/~gmalloci/pahs"
STATES = {-1: "anion", 0: "neutral", 1: "cation", 2: "dication"}
MB_CM2 = 1e-18
#: int sigma dE per unit oscillator strength, pi e^2 h / (m_e c) [Mb eV]
MB_EV_PER_F = 109.761
#: (name, Z) of served files that fail the sum rule (module docstring)
INCONSISTENT = {("coronene", -1): "holds the oscillator strength of C54H18-, not C24H12-",
                ("benzocoronene", 1): "holds 14 more oscillator strength than its neutral (adjacent states differ by ~1)"}


def _get(url, ctx):
    import urllib.error
    import urllib.request

    for attempt in range(2):
        try:
            with urllib.request.urlopen(url, context=ctx, timeout=30) as r:  # noqa: S310
                return r.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if attempt:
                raise
        except OSError:
            if attempt:
                raise
        time.sleep(1.0)
    return None


def _is_spectrum(text):
    """Two numeric columns, energies rising from 0 (the served format)."""
    try:
        d = np.loadtxt(text.splitlines())
    except ValueError:
        return False
    return d.ndim == 2 and d.shape[1] == 2 and d.shape[0] > 100 and d[0, 0] == 0.0 and (np.diff(d[:, 0]) > 0).all()


def fetch(output_dir, delay_s=0.3):
    """Download every molecule's spectra into ``output_dir`` (the registry's fetcher)."""
    import ssl

    from pycalima.models.tools.download_cagliari_pah_database import parse_index, parse_formula

    ctx = ssl.create_default_context()          # the server's certificate is outdated (as the properties fetcher)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for mol in parse_index(_get(f"{BASE_URL}/frames/framepahs.html", ctx)):
        name = mol["name"]
        states = []
        for z, state in STATES.items():
            text = _get(f"{BASE_URL}/{name}/{name}_{state}", ctx)
            time.sleep(delay_s)
            if text is None:
                continue
            if not _is_spectrum(text):
                raise ValueError(f"{name}_{state}: not a two-column spectrum")
            (out / f"{name}_{state}").write_text(text)
            states.append(z)
        nc, nh = parse_formula(mol["formula"])
        rows.append(dict(name=name, formula=mol["formula"], nc=nc, nh=nh, states=" ".join(map(str, states))))
        print(f"  {name}: {', '.join(STATES[z] for z in states)}")
    if len(rows) < 40:
        raise ValueError(f"index lists {len(rows)} molecules, expected 40")
    with open(out / "index.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return out


def _root(root):
    if root is not None:
        return Path(root)
    from pycalima._datasets import ensure_dataset

    return ensure_dataset(DATASET)


def spectra(root=None, include_inconsistent=False):
    """{name: dict(nc, nh, formula, sigma={Z: (E [eV], sigma [cm^2])})} of every fetched molecule, without the
    ``INCONSISTENT`` files unless ``include_inconsistent``."""
    root = _root(root)
    out = {}
    with open(root / "index.csv") as fh:
        for row in csv.DictReader(fh):
            sig = {}
            for z in (int(s) for s in row["states"].split()):
                if (row["name"], z) in INCONSISTENT and not include_inconsistent:
                    continue
                d = np.loadtxt(root / f"{row['name']}_{STATES[z]}")
                sig[z] = (d[:, 0], d[:, 1] * MB_CM2)
            out[row["name"]] = dict(nc=int(row["nc"]), nh=int(row["nh"]), formula=row["formula"], sigma=sig)
    return out


@lru_cache(maxsize=None)
def sigma_of(name, charge, root=None, include_inconsistent=False):
    """(E [eV], sigma [cm^2]) of one served file, e.g. ``sigma_of("coronene", 1)``; an ``INCONSISTENT`` one raises
    unless ``include_inconsistent``."""
    if (name, charge) in INCONSISTENT and not include_inconsistent:
        raise ValueError(f"{name} {STATES[charge]}: {INCONSISTENT[name, charge]}")
    d = np.loadtxt(_root(root) / f"{name}_{STATES[charge]}")
    return d[:, 0], d[:, 1] * MB_CM2


def oscillator_strength(energy_ev, sigma_cm2, e_min=0.0, e_max=np.inf):
    """Oscillator strength in [e_min, e_max]: int sigma dE / (pi e^2 h / m_e c)."""
    E, s = np.asarray(energy_ev, float), np.asarray(sigma_cm2, float) / MB_CM2
    k = (E >= e_min) & (E <= e_max)
    return float(np.trapezoid(s[k], E[k]) / MB_EV_PER_F)


def valence_electrons(nc, nh, charge):
    """C 2s2p and H 1s electrons of the state: what a 0-30 eV spectrum can at most exhaust (TRK)."""
    return 4 * nc + nh - charge


def name_of(formula_or_name, root=None):
    """The database name of a molecule given its name or formula (first match)."""
    s = spectra(root)
    if formula_or_name in s:
        return formula_or_name
    hits = [n for n, v in s.items() if re.sub(r"\s", "", v["formula"]) == formula_or_name]
    if not hits:
        raise KeyError(formula_or_name)
    return hits[0]
