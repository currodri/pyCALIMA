"""Fit a PAH population and a smooth continuum to an observed spectrum, both treated alike.

Band ratios over an observer's continuum cannot be compared with model bands:
a spline through fixed anchors removes part of the model's own emission as
well (pahchem-spec.md, "Review findings"). This module fits the spectrum
itself instead:

    I_nu(nu) = sum_s w_s P_s(nu; conditions) + sum_k c_k B_k(nu),

- P_s is the PAH spectrum of molecule s with its charge states at the stated
  conditions (``Population.state_spectra``: IR power per mode at G0 = 1, the
  band profile of ``emission.spectrum``).
- B_k are cubic B-splines in wavelength, with knots no closer than
  ``knot_um``, set separately in each fitted segment.
- w >= 0 and c >= 0 (a positive continuum), solved together by one NNLS.

The fit decides how much broad emission is PAH band wings and how much is
continuum, by the same rule for every model. The knot spacing is the
continuum-recipe systematic. Band fluxes of the fitted PAH component are
derived quantities (``Population.band_fluxes`` with the fitted weights).

The data are binned onto a uniform wavenumber grid (``Grid``; 2.5 cm^-1, far
finer than the 15 cm^-1 bands). Each bin weighs the same, so every cm^-1
counts equally, unlike pyPAHdb's per-wavelength-point NNLS. Gas lines are
removed beforehand, and artefact intervals masked.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: fitted segments [um]: the 3.3 um complex (NIRSpec) and 5-15 um (MRS channels 1-3)
SEGMENTS_UM = ((3.1, 3.65), (5.0, 15.0))
#: MRS artefacts listed by Chown+24 (12.2, 13.7-13.95, 14.3 um), masked [um]
ARTEFACTS_UM = ((12.17, 12.23), (13.70, 13.95), (14.27, 14.33))


@dataclass(frozen=True)
class Grid:
    """Uniform wavenumber bins per segment: centres ``nu`` [cm^-1], their segment index, and the mask."""

    nu: np.ndarray
    segment: np.ndarray
    step_cm: float

    @classmethod
    def build(cls, segments_um=SEGMENTS_UM, step_cm=2.5, masked_um=ARTEFACTS_UM):
        nus, segs = [], []
        for k, (lo, hi) in enumerate(segments_um):
            nu = np.arange(1e4 / hi + 0.5 * step_cm, 1e4 / lo, step_cm)
            lam = 1e4 / nu
            keep = ~np.any([(lam >= a) & (lam <= b) for a, b in masked_um], axis=0)
            nus.append(nu[keep])
            segs.append(np.full(int(keep.sum()), k))
        return cls(np.concatenate(nus), np.concatenate(segs), float(step_cm))

    @property
    def lam_um(self):
        return 1e4 / self.nu


def bin_spectrum(lam_um, fnu, grid):
    """Mean of the spectrum over each wavenumber bin of ``grid`` (NaN where a bin holds no point)."""
    nu = 1e4 / np.asarray(lam_um, dtype=float)
    f = np.asarray(fnu, dtype=float)
    ok = np.isfinite(f)
    nu, f = nu[ok], f[ok]
    edges_lo, edges_hi = grid.nu - 0.5 * grid.step_cm, grid.nu + 0.5 * grid.step_cm
    order = np.argsort(nu)
    nu, f = nu[order], f[order]
    i0, i1 = np.searchsorted(nu, edges_lo), np.searchsorted(nu, edges_hi)
    c = np.concatenate([[0.0], np.cumsum(f)])
    n = i1 - i0
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (c[i1] - c[i0]) / np.maximum(n, 1), np.nan)


def continuum_basis(grid, knot_um=1.5, segments_um=SEGMENTS_UM):
    """C[i, k]: cubic B-splines in wavelength, knots >= ``knot_um`` apart, separately per segment."""
    from scipy.interpolate import BSpline

    lam = grid.lam_um
    cols = []
    for k, (lo, hi) in enumerate(segments_um):
        sel = grid.segment == k
        n_int = max(1, int(np.floor((hi - lo) / knot_um)))
        inner = np.linspace(lo, hi, n_int + 1)
        t = np.concatenate([[lo] * 3, inner, [hi] * 3])
        n_basis = len(t) - 4
        for j in range(n_basis):
            c = np.zeros(n_basis)
            c[j] = 1.0
            col = np.zeros(grid.nu.size)
            col[sel] = BSpline(t, c, 3, extrapolate=False)(np.clip(lam[sel], lo, hi))
            cols.append(np.nan_to_num(col))
    return np.stack(cols, axis=1)


@dataclass
class SpectralFit:
    weights: np.ndarray          # per molecule (numbers, in the units of the observation per model unit)
    continuum: np.ndarray        # B-spline coefficients
    pah: np.ndarray              # fitted PAH spectrum on the grid
    cont: np.ndarray             # fitted continuum on the grid
    rms: float                   # rms residual / mean observed intensity
    sse: float                   # sum of squared residuals


def fit_spectrum(observed, model_columns, continuum_columns):
    """NNLS of ``observed`` (on the grid) on [PAH columns | continuum columns]; bins with NaN are skipped."""
    from scipy.optimize import nnls

    y = np.asarray(observed, dtype=float)
    ok = np.isfinite(y)
    P, Cc = np.asarray(model_columns, dtype=float), np.asarray(continuum_columns, dtype=float)
    A = np.hstack([P, Cc])[ok]
    scale = np.where(A.max(axis=0) > 0, A.max(axis=0), 1.0)
    x, _ = nnls(A / scale, y[ok])
    x = x / scale
    w, c = x[:P.shape[1]], x[P.shape[1]:]
    pah, cont = P @ w, Cc @ c
    r = (y - pah - cont)[ok]
    return SpectralFit(w, c, pah, cont, float(np.sqrt(np.mean(r**2)) / np.mean(y[ok])), float(r @ r))
