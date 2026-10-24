"""Rigopoulou et al. (2021) Fig. 1, from its stated inputs (method card rigopoulou2021.toml).

Fig. 1 shows the emission of PAHdb UID 742 (C210H36) after absorbing 5 eV and
12.4 eV, plotted per wavenumber against wavelength. The thermal-approximation
cascade (``emission.ThermalCascade``) with Lorentzian bands reproduces it with
FWHM 15 cm^-1, not the 30 cm^-1 the text states. ``fig1_peaks`` gives the
band peak heights that the gate compares.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pycalima._paths import get_external_data_path
from pycalima.pahchem import dos, emission, energy_me

FIG1_UID = 742
FIG1_ENERGIES_EV = {"5eV": 5.0, "12eV": 12.4}       # text: 5 and 12.4 eV (caption: 12 eV)
#: peak-search windows [um]
FIG1_BANDS_UM = {"3.3": (3.1, 3.6), "6.2": (6.0, 6.6), "7.7": (7.2, 8.2), "8.6": (8.4, 9.2),
                 "11.3": (10.5, 11.8), "12.7": (12.4, 13.2)}
GRID_CM = np.arange(600.0, 3500.0, 0.25)


def fig1_digitised():
    """The two Fig. 1 curves read from the vector figure: curve, lambda_um, intensity."""
    return pd.read_csv(get_external_data_path("pahchem/rigopoulou2021_fig1.csv"), comment="#")


def fig1_peaks(species, fwhm_cm=15.0, model="thermal", energies=FIG1_ENERGIES_EV):
    """{curve: {band: peak height}} of the per-wavenumber spectrum, in arbitrary units."""
    if model == "thermal":
        th = emission.ThermalCascade(species.freqs_cm, species.einstein_a)
        per = {k: th.energy_per_mode(e * dos.CM_PER_EV) for k, e in energies.items()}
    elif model == "exact":
        cm = energy_me.CascadeModel(species.freqs_cm, species.einstein_a, e_max_cm=13.0 * dos.CM_PER_EV, de_cm=50.0)
        N = emission.cascade_photons(cm)
        per = {}
        for k, e in energies.items():
            x = e * dos.CM_PER_EV / cm.de_cm
            j = int(np.floor(x))
            per[k] = species.freqs_cm * ((1 - (x - j)) * N[:, j] + (x - j) * N[:, j + 1])
    else:
        raise ValueError(f"model must be 'thermal' or 'exact', not {model!r}")
    lam = 1e4 / GRID_CM
    out = {}
    for k, em in per.items():
        s = emission.spectrum(species.freqs_cm, em, GRID_CM, fwhm_cm=fwhm_cm, shift_cm=0.0)
        out[k] = {b: float(s[(lam >= lo) & (lam <= hi)].max()) for b, (lo, hi) in FIG1_BANDS_UM.items()}
    return out


def peak_deviation(model_peaks, figure=None):
    """Fractional deviations of the 12 peaks after the one common scale (geometric mean)."""
    fig = fig1_digitised() if figure is None else figure
    m, d = [], []
    for k, peaks in model_peaks.items():
        g = fig[fig.curve == k]
        for b, (lo, hi) in FIG1_BANDS_UM.items():
            m.append(peaks[b])
            d.append(g.intensity[(g.lambda_um >= lo) & (g.lambda_um <= hi)].max())
    m, d = np.array(m), np.array(d)
    scale = np.exp(np.mean(np.log(d / m)))
    return m * scale / d - 1.0
