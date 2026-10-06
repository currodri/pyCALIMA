"""Maragkoudakis, Peeters & Ricca (2020) band-ratio diagnostics, from their stated inputs.

Method card ``method_cards/maragkoudakis2020.toml``. In short:
- 308 PAHdb v3.20 species (their Tables H1-H3);
- one absorbed photon of fixed energy (6-12 eV), or the Mathis+83 ISRF with the
  AmesPAHdb suite's cross section (``photo.sigma_amespahdb``);
- Lorentzian bands, FWHM 15 cm^-1, no shift;
- band fluxes summed in fixed wavebands (Table 1).

``band_fluxes`` runs one species with either emission model:
- "thermal" is the suites' thermal-approximation cascade (``emission.ThermalCascade``),
  their model;
- "exact" is pahchem's microcanonical cascade (``emission.cascade_photons``).
Fluxes are their "sum of fluxes" in each waveband, on a grid uniform in
wavenumber (0.5 cm^-1; their grid is not stated).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pycalima._paths import get_external_data_path
from pycalima.pahchem import dos, emission, energy_me, fields, photo

#: Table 1 wavebands [um]
BANDS_UM = {"3.3": (3.1, 3.5), "6.2": (6.1, 6.8), "7.7": (7.0, 8.2), "8.6": (8.2, 8.9),
            "11.2": (10.5, 11.6), "6-9": (6.0, 9.0), "7-9": (7.0, 9.0), "15-20": (15.0, 20.0)}
GRID_CM = np.arange(400.0, 4000.0, 0.5)
ISRF_E_MIN_EV = 0.1


def sample():
    """The 308 species: table, N_C, N_H, charge, PAHdb UID (Tables H1-H3)."""
    return pd.read_csv(get_external_data_path("pahchem/maragkoudakis2020_sample_uids.csv"), comment="#")


def isrf_heating(species):
    """(photon energies [eV], absorbed photons per eV): Mathis+83 ISRF times sigma_amespahdb."""
    f = fields.mathis1983(e_min=ISRF_E_MIN_EV, n=600)
    e = f.energy_ev
    return e, photo.sigma_amespahdb(species.nc, species.charge, e) * f.flux_at(e)


def _bands(freqs_cm, energy_per_mode):
    s = emission.spectrum(freqs_cm, energy_per_mode, GRID_CM, fwhm_cm=15.0, shift_cm=0.0)
    out = {}
    for b, (lo, hi) in BANDS_UM.items():
        m = (GRID_CM >= 1e4 / hi) & (GRID_CM <= 1e4 / lo)
        out[b] = float(s[m].sum() * (GRID_CM[1] - GRID_CM[0]))
    return out


def band_fluxes(species, model="thermal", energies_ev=(6.0, 8.0, 10.0, 12.0), isrf=True):
    """{"isrf" or energy [eV]: {band: flux}} for one species (any common unit).

    Fixed energies are single absorbed photons; "isrf" weights them by the
    absorbed photon spectrum of ``isrf_heating``.
    """
    sp = species
    out = {}
    if model == "thermal":
        th = emission.ThermalCascade(sp.freqs_cm, sp.einstein_a)
        per_energy = lambda E_ev: th.energy_per_mode(E_ev * dos.CM_PER_EV)
        for E in energies_ev:
            out[E] = _bands(sp.freqs_cm, per_energy(E))
        if isrf:
            e, h = isrf_heating(sp)
            em = np.trapezoid(h[:, None] * np.array([per_energy(x) for x in e]), e, axis=0)
            out["isrf"] = _bands(sp.freqs_cm, em)
    elif model == "exact":
        cm = energy_me.CascadeModel(sp.freqs_cm, sp.einstein_a, e_max_cm=14.0 * dos.CM_PER_EV, de_cm=50.0)
        N = emission.cascade_photons(cm)
        for E in energies_ev:
            x = E * dos.CM_PER_EV / cm.de_cm
            j = int(np.floor(x))
            w = x - j
            out[E] = _bands(sp.freqs_cm, sp.freqs_cm * ((1 - w) * N[:, j] + w * N[:, j + 1]))
        if isrf:
            e, h = isrf_heating(sp)
            em = sp.freqs_cm * emission.single_photon_emission(cm, e, h)
            out["isrf"] = _bands(sp.freqs_cm, em)
    else:
        raise ValueError(f"model must be 'thermal' or 'exact', not {model!r}")
    return out


def size_fit(nc, charge_state, field="isrf"):
    """log10 I_R = c_o + N_C^a (their eq. 3) with Table 4 (ISRF) or F9-F12 (6-12 eV)."""
    t = pd.read_csv(get_external_data_path("pahchem/maragkoudakis2020_size_fits.csv"), comment="#")
    row = t[(t.field == str(field)) & (t.charge_state == charge_state)].iloc[0]
    return row.c_o + np.asarray(nc, dtype=float) ** row.a
