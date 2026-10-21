"""Demyk+26 (arXiv:2607.16018): single-photon IR emission of 6H- and Me-pyrene with temperature-dependent bands.

Their Sect. 2.4 model, on pahchem's cascade (method card ``demyk2026.toml``):

- **Excitation.** One UV photon of ``e_uv_ev`` absorbed by a cold molecule,
  then the harmonic IR cascade with no further absorption
  (``energy_me.CascadeModel``; ``emission.emission_by_energy`` splits each
  mode's photons by the bin they leave from).
- **Modes.** Harmonic frequencies (``modes``: B3LYP/4-31G, scaled per class
  against PAHdb pyrene; their B3LYP-D3/6-311G(d,p) list is not published).
  Each measured band of Table 5 (``table5``, km/mol) goes to the nearest
  unassigned mode of its class (CH stretch or not); the other modes carry no
  intensity, as in their model.
- **Profiles.** A photon emitted at internal energy E takes T(E)
  (microcanonical, 1/kT = d ln rho / dE, their Fig. 10; or canonical), its
  band's CosmicPAH-IRDB law position and FWHM at T (``cosmicpah.BandLaw``),
  and a pseudo-Voigt of the band's Lorentzian fraction. Bands without a law
  keep their 523 K position and the constant FWHM of Sect. 2.4.

``run`` returns per band the photons and energy emitted and the emission-
temperature distribution; ``Emission.spectrum`` the band spectrum.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pycalima import _paths
from pycalima.pahchem import cosmicpah, dos, emission, energy_me
from pycalima.pahchem.molecule import _EINSTEIN_A_PER_KM_MOL

CH_STRETCH_CM = 2500.0               # bands and modes above this are CH stretches
DE_CM, GRAIN_CM = 20.0, 2.0
#: Lorentzian fraction of the bands without a CosmicPAH-IRDB law (not stated; profile only, not the band sums)
LORENTZ_UNKNOWN = 0.5


def table5(key):
    """Demyk+26 Table 5 bands of ``key``: band_cm, intensity_km_mol, fwhm_fixed_cm (NaN where a law applies)."""
    d = pd.read_csv(_paths.get_external_data_path("pahchem", "demyk2026_table5.csv"), comment="#")
    return d[d.molecule == key].reset_index(drop=True)


def modes(key):
    """Scaled harmonic frequencies [cm^-1] of ``key`` (``demyk2026_modes.csv``, B3LYP/4-31G)."""
    d = pd.read_csv(_paths.get_external_data_path("pahchem", "demyk2026_modes.csv"), comment="#")
    return np.sort(d.loc[d.molecule == key, "freq_cm"].to_numpy(float))


def assign_intensities(freqs_cm, bands_cm, intensities_km_mol):
    """Einstein A per mode: each band to the nearest unassigned mode of its class, strongest band first.

    Returns (einstein_a [s^-1] per mode, mode index per band)."""
    f = np.asarray(freqs_cm, dtype=float)
    a = np.zeros(f.size)
    taken = np.zeros(f.size, dtype=bool)
    index = np.full(len(bands_cm), -1)
    for b in np.argsort(-np.asarray(intensities_km_mol, dtype=float)):
        same = (f > CH_STRETCH_CM) == (bands_cm[b] > CH_STRETCH_CM)
        cand = np.where(same & ~taken)[0]
        if cand.size == 0:
            raise ValueError(f"no free mode for the band at {bands_cm[b]} cm^-1")
        m = cand[np.argmin(np.abs(f[cand] - bands_cm[b]))]
        taken[m] = True
        index[b] = m
        a[m] = _EINSTEIN_A_PER_KM_MOL * f[m] ** 2 * intensities_km_mol[b]
    return a, index


@dataclass
class Band:
    """One measured band: its law (or fixed position and width) and its emission."""
    band_cm: float
    mode_cm: float
    law: cosmicpah.BandLaw | None
    fwhm_fixed_cm: float
    photons: np.ndarray            # per cascade bin, per absorbed UV photon

    def position(self, t):
        return self.law.position(t) if self.law is not None else np.full(np.shape(t), self.band_cm)

    def fwhm(self, t):
        return self.law.fwhm(t) if self.law is not None else np.full(np.shape(t), self.fwhm_fixed_cm)

    @property
    def lorentz(self):
        return self.law.lorentz if self.law is not None else LORENTZ_UNKNOWN


@dataclass
class Emission:
    key: str
    e_uv_ev: float
    temperature_k: np.ndarray      # per cascade bin
    bands: list[Band]

    def band(self, band_cm, tol_cm=3.0):
        found = [b for b in self.bands if abs(b.band_cm - band_cm) <= tol_cm]
        if len(found) != 1:
            raise KeyError(f"{self.key}: {len(found)} bands within {tol_cm} cm^-1 of {band_cm}")
        return found[0]

    def photons(self, band_cm):
        """Photons emitted in the band per absorbed UV photon."""
        return float(self.band(band_cm).photons.sum())

    def energy(self, band_cm):
        """Energy emitted in the band per absorbed UV photon [cm^-1], each photon at its position at T."""
        b = self.band(band_cm)
        return float(b.photons @ b.position(self.temperature_k))

    def temperature_distribution(self, band_cm, edges_k):
        """Photon-weighted histogram of the emission temperatures of a band (normalised to 1)."""
        b = self.band(band_cm)
        h, _ = np.histogram(self.temperature_k, bins=edges_k, weights=b.photons)
        return h / max(h.sum(), 1e-300)

    def spectrum(self, nu_cm):
        """Emitted power per cm^-1 per absorbed UV photon on ``nu_cm`` [cm^-1 / cm^-1]."""
        nu = np.asarray(nu_cm, dtype=float)
        out = np.zeros(nu.size)
        for b in self.bands:
            on = b.photons > 0
            t = self.temperature_k[on]
            pos, w = b.position(t), b.fwhm(t)
            prof = cosmicpah.pseudo_voigt(nu[None, :], pos[:, None], w[:, None], b.lorentz)
            out += (b.photons[on] * pos) @ prof
        return out


def temperature(freqs_cm, energy_cm, kind="microcanonical", window_cm=200.0):
    """T(E) [K] of a mode set on a cascade grid: microcanonical (their Fig. 10) or canonical."""
    if kind == "canonical":
        return dos.canonical_temperature(freqs_cm, energy_cm)
    if kind != "microcanonical":
        raise ValueError(f"kind must be 'microcanonical' or 'canonical', not {kind!r}")
    E, counts = dos.beyer_swinehart(freqs_cm, energy_cm[-1] + window_cm, GRAIN_CM)
    T = dos.microcanonical_temperature(E, dos.smoothed_log_density(E, counts, window_cm))
    return np.interp(energy_cm, E, np.nan_to_num(T, nan=0.0))


def run(key, e_uv_ev, freqs_cm=None, t_kind="microcanonical", root=None):
    """The emission of ``key`` after one ``e_uv_ev`` photon (``Emission``); ``freqs_cm`` overrides ``modes``."""
    f = modes(key) if freqs_cm is None else np.sort(np.asarray(freqs_cm, dtype=float))
    t5 = table5(key)
    a_ein, index = assign_intensities(f, t5.band_cm.to_numpy(), t5.intensity_km_mol.to_numpy())
    e_uv = e_uv_ev * dos.CM_PER_EV
    cm = energy_me.CascadeModel(f, a_ein, e_uv + 5 * DE_CM, DE_CM, GRAIN_CM)
    start = np.zeros(cm.n)
    x = e_uv / DE_CM
    lo = int(np.floor(x))
    start[lo], start[lo + 1] = 1.0 - (x - lo), x - lo           # the photon energy split between its two bins
    X = emission.emission_by_energy(cm, start)
    laws = cosmicpah.laws(key, root)
    bands = []
    for row, m in zip(t5.itertuples(), index):
        law = None
        if not np.isfinite(row.fwhm_fixed_cm):
            near = [b for b in laws if abs(b.label_cm - row.band_cm) <= 3.0]
            if len(near) != 1:
                raise KeyError(f"{key}: no CosmicPAH-IRDB law for the band at {row.band_cm} cm^-1")
            law = near[0]
        bands.append(Band(float(row.band_cm), float(f[m]), law, float(row.fwhm_fixed_cm), X[m]))
    return Emission(key, float(e_uv_ev), temperature(f, cm.energy_cm, t_kind), bands)


def band_profile(freqs_cm, einstein_a, modes, law, e_uv_ev, nu_cm, t_kind="microcanonical"):
    """Emitted power per cm^-1 on ``nu_cm`` of the modes ``modes`` (mask or indices) of any molecule after one
    ``e_uv_ev`` photon, each photon at ``law``'s position and FWHM at its bin's T (pseudo-Voigt): the method of
    ``run`` for a mode list and intensities given directly."""
    f = np.asarray(freqs_cm, dtype=float)
    e_uv = e_uv_ev * dos.CM_PER_EV
    cm = energy_me.CascadeModel(f, np.asarray(einstein_a, dtype=float), e_uv + 5 * DE_CM, DE_CM, GRAIN_CM)
    start = np.zeros(cm.n)
    x = e_uv / DE_CM
    lo = int(np.floor(x))
    start[lo], start[lo + 1] = 1.0 - (x - lo), x - lo
    ph = emission.emission_by_energy(cm, start)[np.asarray(modes)].reshape(-1, cm.n).sum(axis=0)
    T = temperature(f, cm.energy_cm, t_kind)
    on = ph > 0
    pos, fw = law.position(T[on]), law.fwhm(T[on])
    return (ph[on] * pos) @ cosmicpah.pseudo_voigt(np.asarray(nu_cm, float)[None, :], pos[:, None], fw[:, None], law.lorentz)
