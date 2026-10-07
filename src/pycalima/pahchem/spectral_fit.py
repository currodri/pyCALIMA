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


#: mode classes by harmonic frequency [cm^-1]; modes outside them take the CC shift and the "rest" width
MODE_CLASSES = {"ch": (2800.0, 3300.0), "cc": (1000.0, 1800.0), "oop": (600.0, 1000.0)}


@dataclass(frozen=True)
class BandShape:
    """Global band profile: shifts per mode class [cm^-1], FWHM of the CH stretch and of the rest, profile.

    The default is pahchem's previous convention (one -15 cm^-1 shift, 15 cm^-1
    Lorentzians). Shifts and widths stand for the anharmonic red-shift and
    broadening at the emission temperature, shared by every molecule.
    """

    shift_ch: float = -15.0
    shift_cc: float = -15.0
    shift_oop: float = -15.0
    fwhm_ch: float = 15.0
    fwhm_rest: float = 15.0
    profile: str = "lorentzian"

    def per_mode(self, freqs_cm):
        f = np.asarray(freqs_cm, dtype=float)
        ch = (f >= MODE_CLASSES["ch"][0]) & (f < MODE_CLASSES["ch"][1])
        oop = (f >= MODE_CLASSES["oop"][0]) & (f < MODE_CLASSES["oop"][1])
        shift = np.where(ch, self.shift_ch, np.where(oop, self.shift_oop, self.shift_cc))
        fwhm = np.where(ch, self.fwhm_ch, self.fwhm_rest)
        return shift, fwhm


def shaped_spectrum(freqs_cm, strength, nu_cm, shape=BandShape()):
    """Sum of unit-area bands of ``strength`` at the shifted positions, per cm^-1 on ``nu_cm``."""
    shift, fwhm = shape.per_mode(freqs_cm)
    x = np.asarray(nu_cm, dtype=float)[None, :] - (np.asarray(freqs_cm, dtype=float) + shift)[:, None]
    w = fwhm[:, None]
    if shape.profile == "lorentzian":
        g = 0.5 * w / np.pi / (x**2 + (0.5 * w) ** 2)
    elif shape.profile == "gaussian":
        s = w / np.sqrt(8.0 * np.log(2.0))
        g = np.exp(-0.5 * (x / s) ** 2) / (s * np.sqrt(2.0 * np.pi))
    else:
        raise ValueError(f"profile must be 'lorentzian' or 'gaussian', not {shape.profile!r}")
    return np.asarray(strength, dtype=float) @ g


def window_fluxes(grid, spectrum, windows_um):
    """{band: integral of ``spectrum`` over the window} on the grid (masked bins excluded)."""
    lam = grid.lam_um
    return {b: float(spectrum[(lam >= lo) & (lam <= hi)].sum() * grid.step_cm) for b, (lo, hi) in windows_um.items()}


def molecule_spectra(pop, env, nu_cm, shape=BandShape(), index=None):
    """P[s, i] for the molecules ``index`` of a ``population.Population`` (all if None), with ``shape``."""
    from pycalima.pahchem.population import field_shape

    fs = field_shape(env)
    f = pop.charge_matrix(env)
    idx = range(len(pop.library)) if index is None else index
    out = np.zeros((len(idx), np.size(nu_cm)))
    for k, s in enumerate(idx):
        mol = pop.library[s]
        for z in mol.charges:
            b = pop.basis(mol, z, fs)
            out[k] += f[s, z - pop.zmin] * shaped_spectrum(b.freqs_cm, b.emitted, nu_cm, shape)
    return env.g0 * out


def shaped_state_spectra(pop, fshape, nu_cm, shape=BandShape()):
    """S[s, slot, i]: every state's IR power per cm^-1 at G0 = 1 with ``shape`` (independent of the cell)."""
    pop.prepare(fshape)
    S = np.zeros((len(pop.library), len(pop.zs), np.size(nu_cm)))
    for s, mol in enumerate(pop.library):
        for z in mol.charges:
            b = pop.basis(mol, z, fshape)
            S[s, z - pop.zmin] = shaped_spectrum(b.freqs_cm, b.emitted, nu_cm, shape)
    return S


#: bounds of the BandShape parameters fitted by ``fit_band_shape`` [cm^-1]
SHAPE_BOUNDS = {"shift_ch": (-70.0, 20.0), "shift_cc": (-50.0, 20.0), "shift_oop": (-50.0, 20.0),
                "fwhm_ch": (5.0, 80.0), "fwhm_rest": (5.0, 60.0)}


def fit_band_shape(observed, grid, pop, env, keep, continuum_columns, start=BandShape(), max_rounds=4):
    """(BandShape, SpectralFit, active molecule indices): the band shape that minimises the joint-fit residual.

    The weights are linear, so the shape is profiled:
    1. NNLS over all kept molecules at the current shape gives the active set;
    2. bounded least squares on the shape parameters (variable projection): the
       residual of the NNLS fit restricted to that set, which is smooth in the
       shape;
    3. repeat until the active set is stable (or ``max_rounds``).
    The profile kind (``start.profile``) is held fixed.
    """
    from scipy.optimize import least_squares

    names = list(SHAPE_BOUNDS)
    lo = np.array([SHAPE_BOUNDS[k][0] for k in names])
    hi = np.array([SHAPE_BOUNDS[k][1] for k in names])
    kept = np.flatnonzero(keep)
    shape = start
    active_prev = None
    for _ in range(max_rounds):
        P = molecule_spectra(pop, env, grid.nu, shape, kept).T
        fit = fit_spectrum(observed, P, continuum_columns)
        active = kept[fit.weights > 0]
        if active_prev is not None and set(active) == set(active_prev):
            break
        active_prev = active
        ok = np.isfinite(observed)

        def residual(x):
            sh = BandShape(**dict(zip(names, x)), profile=start.profile)
            Pa = molecule_spectra(pop, env, grid.nu, sh, active).T
            f_ = fit_spectrum(observed, Pa, continuum_columns)
            return (observed - f_.pah - f_.cont)[ok] / np.mean(observed[ok])

        x0 = np.clip(np.array([getattr(shape, k) for k in names], dtype=float), lo + 1e-6, hi - 1e-6)
        res = least_squares(residual, x0, bounds=(lo, hi), diff_step=1e-3, x_scale=np.full(len(names), 5.0),
                            max_nfev=200)
        shape = BandShape(**dict(zip(names, res.x)), profile=start.profile)
    P = molecule_spectra(pop, env, grid.nu, shape, kept).T
    fit = fit_spectrum(observed, P, continuum_columns)
    w = np.zeros(len(pop.library))
    w[kept] = fit.weights
    fit.weights = w
    return shape, fit, kept[fit.weights[kept] > 0]


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
