"""Maragkoudakis et al. (2026, PDRs4All XVIII) pyPAHdb decomposition, from their stated inputs.

Method card ``method_cards/maragkoudakis2026.toml``. Their model, in short:
- pyPAHdb with the precomputed PAHdb v4.00-alpha matrix (public, no
  registration: dataset ``pypahdb-matrix-v4-00-alpha``). It has 7529 spectra of
  C/H species with 20 < N_C <= 384 and charges -1 to +3, each the emission of
  one molecule after one absorbed 7 eV photon (every column integrates to
  7 eV). The cascade is the AmesPAHdb thermal approximation, Gaussian bands
  FWHM 15 cm^-1, no redshift.
- The observation is the continuum-subtracted 5-15 um spectrum (pybaselines
  morphological then imodpoly; parameters not stated).
- One unweighted NNLS on the observation's own wavelength points, after the
  matrix is interpolated onto them (``decompose``, which follows pyPAHdb's
  ``DecomposerBase`` line by line).
- The "fractions" are fractions of the NNLS weights (``fractions``). Since every
  column is one absorbed 7 eV photon, the weights count absorbed photons. The
  charge and size fractions are therefore shares of the emitted energy, and
  N_C is the emission-weighted mean. "Cation" is every charge > 0, dications and
  trications included.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pycalima._paths import get_external_data_path

SMALL_SIZE, MEDIUM_SIZE = 50, 70        # pyPAHdb's size classes
FIT_RANGE_UM = (5.0, 15.0)
#: their template windows [um] (Fig. 7 caption)
BANDS_UM = {"6.2": (6.0, 6.6), "7.7": (7.0, 8.2), "8.6": (8.2, 9.15), "11.2": (11.1, 11.7)}


def table1():
    """Their Table 1, as printed."""
    return pd.read_csv(get_external_data_path("pahchem/maragkoudakis2026_table1.csv"), comment="#")


def templates_digitised():
    """Template gamma, band ratios and N_C digitised from their figures (not tabulated by the authors)."""
    return pd.read_csv(get_external_data_path("pahchem/maragkoudakis2026_templates_digitised.csv"), comment="#")


def matrix_path():
    """Path of the pyPAHdb v4.00-alpha matrix (raises with instructions if it is not installed)."""
    from pycalima._datasets import ensure_dataset, get_dataset

    ds = "pypahdb-matrix-v4-00-alpha"
    return ensure_dataset(ds) / get_dataset(ds).files[0]


def load_precomputed(path=None):
    """{"abscissa" [cm^-1], "matrix" (n_freq, n_species), "charge", "size"} of a pyPAHdb pickle."""
    import pickle

    with open(matrix_path() if path is None else path, "rb") as fh:
        d = pickle.load(fh, encoding="latin1")
    return {"abscissa": np.asarray(d["abscissa"], dtype=float), "matrix": np.asarray(d["matrix"], dtype=float),
            "charge": np.asarray(d["properties"]["charge"]), "size": np.asarray(d["properties"]["size"]),
            "version": d.get("version")}


def decompose(wavelength_um, intensity, precomputed, fit_range_um=FIT_RANGE_UM):
    """(weights, fit on the input points) of pyPAHdb's NNLS for one spectrum.

    As ``DecomposerBase``:
    - the matrix columns are linearly interpolated onto the observation's
      wavenumbers;
    - the matrix is scaled by its maximum and the spectrum by its own;
    - NNLS runs on the points within ``fit_range_um``, unweighted;
    - the weights are scaled back.
    """
    from scipy.optimize import nnls

    lam = np.asarray(wavelength_um, dtype=float)
    y = np.asarray(intensity, dtype=float)
    sel = (lam >= fit_range_um[0]) & (lam <= fit_range_um[1]) & np.isfinite(y)
    nu = 1e4 / lam[sel]
    x = precomputed["abscissa"]
    M = np.stack([np.interp(nu, x, col) for col in precomputed["matrix"].T], axis=1)
    m_scl, b_scl = M.max(), np.max(y[sel])
    w, _ = nnls(M / m_scl, y[sel] / b_scl)
    w = w / (m_scl / b_scl)
    fit = np.full_like(y, np.nan)
    fit[sel] = M @ w
    return w, fit


def fractions(weights, precomputed):
    """pyPAHdb's charge and size fractions (of the weights), and the weight-averaged N_C."""
    w = np.asarray(weights, dtype=float)
    q, n = precomputed["charge"], precomputed["size"]
    tot = w.sum()
    return {"f_neut": w[q == 0].sum() / tot, "f_cat": w[q > 0].sum() / tot, "f_an": w[q < 0].sum() / tot,
            "f_sm": w[n <= SMALL_SIZE].sum() / tot, "f_med": w[(n > SMALL_SIZE) & (n <= MEDIUM_SIZE)].sum() / tot,
            "f_lar": w[n > MEDIUM_SIZE].sum() / tot, "n_c": float(w @ n / tot),
            "f_dication_plus": w[q >= 2].sum() / tot}


def fit_error(wavelength_um, intensity, fit, fit_range_um=FIT_RANGE_UM):
    """pyPAHdb's error: integral of |residual| over the integral of the observation (in wavenumber)."""
    lam = np.asarray(wavelength_um, dtype=float)
    sel = (lam >= fit_range_um[0]) & (lam <= fit_range_um[1]) & np.isfinite(fit)
    nu = 1e4 / lam[sel]
    o = np.argsort(nu)
    r = np.abs(fit[sel] - intensity[sel])[o]
    return float(np.trapezoid(r, nu[o]) / np.trapezoid(np.asarray(intensity)[sel][o], nu[o]))


def baseline_pybaselines(wavelength_um, intensity, half_window, poly_order, variant="mor_then_imodpoly"):
    """Their continuum (Sect. 2.3): pybaselines' morphological curve, then an imodpoly fit (needs pybaselines).

    ``variant`` "mor_then_imodpoly" fits imodpoly to the morphological curve;
    "imodpoly" fits the data directly. Their half-window and order are not stated.
    """
    from pybaselines import Baseline

    lam = np.asarray(wavelength_um, dtype=float)
    y = np.asarray(intensity, dtype=float)
    fitter = Baseline(x_data=lam)
    if variant == "imodpoly":
        return fitter.imodpoly(y, poly_order=poly_order)[0]
    curve = fitter.mor(y, half_window=half_window)[0]
    return fitter.imodpoly(curve, poly_order=poly_order)[0]


def band_ratios(wavelength_um, intensity, bands=BANDS_UM):
    """{band: integral of I_nu d nu over the window} / the 11.2 band, on the input points."""
    lam = np.asarray(wavelength_um, dtype=float)
    y = np.asarray(intensity, dtype=float)
    flux = {}
    for b, (lo, hi) in bands.items():
        s = (lam >= lo) & (lam <= hi)
        flux[b] = np.trapezoid(y[s] * 2.99792458e14 / lam[s] ** 2, lam[s])
    return {b: v / flux["11.2"] for b, v in flux.items()}


HC_ERG_CM = 1.98644586e-16


def emission_shares(weights, columns_integral, charge, size):
    """Charge and size shares of the emitted energy, and its N_C mean, for any matrix.

    pyPAHdb's columns all hold one 7 eV photon, so its weight fractions are
    these shares. For a matrix whose columns emit different energies, each
    weight is multiplied by its column's integral first.
    """
    e = np.asarray(weights, dtype=float) * np.asarray(columns_integral, dtype=float)
    q, n = np.asarray(charge), np.asarray(size)
    tot = e.sum()
    return {"f_neut": e[q == 0].sum() / tot, "f_cat": e[q > 0].sum() / tot, "f_an": e[q < 0].sum() / tot,
            "f_sm": e[n <= SMALL_SIZE].sum() / tot, "f_med": e[(n > SMALL_SIZE) & (n <= MEDIUM_SIZE)].sum() / tot,
            "f_lar": e[n > MEDIUM_SIZE].sum() / tot, "n_c": float(e @ n / tot), "f_dication_plus": e[q >= 2].sum() / tot}


def matrix_from_species(species, abscissa_cm, model="thermal_7ev", profile="gaussian", fwhm_cm=15.0, shift_cm=0.0,
                        basis=None, window_cm=None):
    """{"abscissa", "matrix", "charge", "size"} of pyPAHdb-style columns built by pahchem.

    ``model``:
    - "thermal_7ev": the AmesPAHdb thermal cascade after one 7 eV photon
      (pyPAHdb's columns; ``emission.ThermalCascade``);
    - "exact_7ev": pahchem's microcanonical cascade after one 7 eV photon;
    - "field": the IR power per mode of a ``population.StateBasis`` (exact
      cascade under a field at G0 = 1); ``basis`` maps a species to its basis.

    Columns are erg per cm^-1, the 7 eV ones per molecule and photon. Lines
    farther than ``window_cm`` from the abscissa range are skipped.
    """
    from pycalima.pahchem import dos, emission, energy_me

    x = np.asarray(abscissa_cm, dtype=float)
    cols = []
    for sp in species:
        if model == "thermal_7ev":
            e = emission.ThermalCascade(sp.freqs_cm, sp.einstein_a).energy_per_mode(7.0 * dos.CM_PER_EV)
        elif model == "exact_7ev":
            cm = energy_me.CascadeModel(sp.freqs_cm, sp.einstein_a, 8.0 * dos.CM_PER_EV, 50.0)
            N = emission.cascade_photons(cm)
            j = 7.0 * dos.CM_PER_EV / cm.de_cm
            lo = int(np.floor(j))
            w = j - lo
            e = sp.freqs_cm * ((1.0 - w) * N[:, lo] + w * N[:, lo + 1])
        elif model == "field":
            e = basis(sp).emitted
        else:
            raise ValueError(f"unknown model {model!r}")
        f = sp.freqs_cm
        keep = np.ones(f.size, dtype=bool) if window_cm is None else (f > x.min() - window_cm) & (f < x.max() + window_cm)
        cols.append(emission.spectrum(f[keep], e[keep], x, fwhm_cm=fwhm_cm, shift_cm=shift_cm, profile=profile) * HC_ERG_CM)
    return {"abscissa": x, "matrix": np.stack(cols, axis=1), "charge": np.array([s.charge for s in species]),
            "size": np.array([s.nc for s in species])}
