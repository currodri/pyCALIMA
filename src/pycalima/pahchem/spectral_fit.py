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

try:  # optional accelerator (the ``accel`` extra); the NumPy path gives the same numbers
    from numba import njit, prange
    HAVE_NUMBA = True
except ModuleNotFoundError:  # pragma: no cover - exercised only without numba
    HAVE_NUMBA = False

#: fitted segments [um]: the 3.3 um complex (NIRSpec) and 5-15 um (MRS channels 1-3)
SEGMENTS_UM = ((3.1, 3.65), (5.0, 15.0))
#: MRS artefacts listed by Chown+24 (12.2, 13.7-13.95, 14.3 um), masked [um]
ARTEFACTS_UM = ((12.17, 12.23), (13.70, 13.95), (14.27, 14.33))
#: gas lines masked on top of the running-median cleaning [um] (vacuum wavelengths):
#: H2 0-0 S(7)-S(2), [Fe II] 5.340, [Ni II] 6.636, [Ar II] 6.985, H I Pf-alpha 7.460, [Ar III] 8.991,
#: [S IV] 10.511, H I Hu-alpha 12.372, [Ne II] 12.814; H I Pf-delta 3.297 and H2 1-0 O(5) 3.235.
GAS_LINES_UM = (5.511, 6.109, 6.910, 8.025, 9.665, 12.279, 5.340, 6.636, 6.985, 7.460, 8.991, 10.511, 12.372, 12.814)
GAS_LINES_NIR_UM = (3.297, 3.235)
LINE_HALF_UM, LINE_HALF_NIR_UM = 0.010, 0.002
MASKED_UM = ARTEFACTS_UM + tuple((x - LINE_HALF_UM, x + LINE_HALF_UM) for x in GAS_LINES_UM) \
    + tuple((x - LINE_HALF_NIR_UM, x + LINE_HALF_NIR_UM) for x in GAS_LINES_NIR_UM)


@dataclass(frozen=True)
class Grid:
    """Uniform wavenumber bins per segment: centres ``nu`` [cm^-1], their segment index, and the mask."""

    nu: np.ndarray
    segment: np.ndarray
    step_cm: float

    @classmethod
    def build(cls, segments_um=SEGMENTS_UM, step_cm=2.5, masked_um=MASKED_UM):
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
MODE_CLASSES = {"ch": (2600.0, 3300.0), "cc": (1000.0, 1800.0), "oop": (600.0, 1000.0)}


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


#: Gaussians are summed within +-GAUSS_NSIGMA sigma of their centre (exp(-32) = 1e-14 of the peak): the same numbers
#: as the dense evaluation to rounding. Lorentzians are summed over the whole grid.
GAUSS_NSIGMA = 8.0
_FWHM_TO_SIGMA = 1.0 / np.sqrt(8.0 * np.log(2.0))


def _summed_profiles_numpy(nu, freq, strength, shift, fwhm, col_start, n_cols, lorentzian):
    out = np.zeros((n_cols, nu.size))
    for c in range(n_cols):
        sl = slice(col_start[c], col_start[c + 1])
        x = nu[None, :] - (freq[sl] + shift[sl])[:, None]
        w = fwhm[sl][:, None]
        if lorentzian:
            g = 0.5 * w / np.pi / (x**2 + (0.5 * w) ** 2)
        else:
            sg = w * _FWHM_TO_SIGMA
            g = np.exp(-0.5 * (x / sg) ** 2) / (sg * np.sqrt(2.0 * np.pi))
        out[c] = strength[sl] @ g
    return out


if HAVE_NUMBA:
    @njit(parallel=True, cache=True)
    def _summed_profiles(nu, freq, strength, shift, fwhm, col_start, n_cols, lorentzian):
        n = nu.size
        out = np.zeros((n_cols, n))
        for c in prange(n_cols):
            for m in range(col_start[c], col_start[c + 1]):
                s = strength[m]
                if s == 0.0:
                    continue
                x0 = freq[m] + shift[m]
                w = fwhm[m]
                if lorentzian:
                    hw = 0.5 * w
                    a = s * hw / np.pi
                    for i in range(n):
                        d = nu[i] - x0
                        out[c, i] += a / (d * d + hw * hw)
                else:
                    sig = w * 0.42466090014400953            # 1 / sqrt(8 ln 2)
                    a = s / (sig * 2.5066282746310002)        # sqrt(2 pi)
                    lo = np.searchsorted(nu, x0 - GAUSS_NSIGMA * sig)
                    hi = np.searchsorted(nu, x0 + GAUSS_NSIGMA * sig)
                    inv = 0.5 / (sig * sig)
                    for i in range(lo, hi):
                        d = nu[i] - x0
                        out[c, i] += a * np.exp(-d * d * inv)
        return out
else:  # pragma: no cover
    _summed_profiles = _summed_profiles_numpy


class ModeTable:
    """Every mode of a set of spectra ("columns"), packed once, so that a band shape costs one kernel call.

    Column c is the sum of the modes ``col_start[c]:col_start[c+1]`` with their strengths. The modes do
    not depend on the band shape, only each mode's shift and FWHM do (through its class, ``MODE_CLASSES``),
    so ``spectra(shape)`` re-evaluates the profiles alone: with numba (``HAVE_NUMBA``) ~30x faster than
    the dense evaluation it replaces, to rounding (``GAUSS_NSIGMA``).
    """

    def __init__(self, columns, nu_cm):
        freq, strength = [], []
        start = [0]
        for col in columns:                                   # iterable of [(freqs_cm, strength), ...]
            for f, a in col:
                freq.append(np.asarray(f, dtype=float))
                strength.append(np.asarray(a, dtype=float) * np.ones(np.size(f)))
            start.append(start[-1] + sum(np.size(f) for f, _ in col))
        self.freq = np.ascontiguousarray(np.concatenate(freq) if freq else np.zeros(0))
        self.strength = np.ascontiguousarray(np.concatenate(strength) if strength else np.zeros(0))
        self.col_start = np.asarray(start, dtype=np.int64)
        self.n_cols = len(start) - 1
        f = self.freq
        ch = (f >= MODE_CLASSES["ch"][0]) & (f < MODE_CLASSES["ch"][1])
        oop = (f >= MODE_CLASSES["oop"][0]) & (f < MODE_CLASSES["oop"][1])
        self._cls = np.where(ch, 0, np.where(oop, 2, 1))     # 0 CH stretch, 1 CC (and the rest), 2 CH oop
        nu = np.asarray(nu_cm, dtype=float)
        self._order = np.argsort(nu, kind="stable")
        self._nu_sorted = np.ascontiguousarray(nu[self._order])

    def spectra(self, shape=BandShape()):
        """(n_cols, n_nu) on the caller's ``nu_cm`` order, per cm^-1, with ``shape``."""
        shift = np.array([shape.shift_ch, shape.shift_cc, shape.shift_oop])[self._cls]
        fwhm = np.where(self._cls == 0, shape.fwhm_ch, shape.fwhm_rest).astype(float)
        if shape.profile not in ("lorentzian", "gaussian"):
            raise ValueError(f"profile must be 'lorentzian' or 'gaussian', not {shape.profile!r}")
        out_sorted = _summed_profiles(self._nu_sorted, self.freq, self.strength, np.ascontiguousarray(shift),
                                      np.ascontiguousarray(fwhm), self.col_start, self.n_cols,
                                      shape.profile == "lorentzian")
        out = np.empty_like(out_sorted)
        out[:, self._order] = out_sorted
        return out


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


def molecule_table(pop, env, nu_cm, index=None):
    """``ModeTable`` of the molecules ``index`` (all if None) at ``env``: each column one molecule, its
    charge states' modes weighted by f(Z) and G0 (``molecule_spectra`` without the band shape)."""
    from pycalima.pahchem.population import field_shape

    fs = field_shape(env)
    f = pop.charge_matrix(env)
    idx = range(len(pop.library)) if index is None else index
    cols = []
    for s in idx:
        mol = pop.library[s]
        col = []
        for z in mol.charges:
            b = pop.basis(mol, z, fs)
            col.append((b.freqs_cm, env.g0 * f[s, z - pop.zmin] * np.asarray(b.emitted, dtype=float)))
        cols.append(col)
    return ModeTable(cols, nu_cm)


def molecule_spectra(pop, env, nu_cm, shape=BandShape(), index=None):
    """P[s, i] for the molecules ``index`` of a ``population.Population`` (all if None), with ``shape``."""
    return molecule_table(pop, env, nu_cm, index).spectra(shape)


def state_table(pop, fshape, nu_cm):
    """``ModeTable`` with one column per (molecule, charge slot) at G0 = 1 (empty slots are empty columns)."""
    pop.prepare(fshape)
    cols = []
    for mol in pop.library:
        for slot, z in enumerate(pop.zs):
            if z in mol.charges:
                b = pop.basis(mol, z, fshape)
                cols.append([(b.freqs_cm, b.emitted)])
            else:
                cols.append([])
    return ModeTable(cols, nu_cm)


def shaped_state_spectra(pop, fshape, nu_cm, shape=BandShape(), table=None):
    """S[s, slot, i]: every state's IR power per cm^-1 at G0 = 1 with ``shape`` (independent of the cell).

    ``table``: a ``state_table`` to reuse across shapes."""
    table = state_table(pop, fshape, nu_cm) if table is None else table
    return table.spectra(shape).reshape(len(pop.library), len(pop.zs), np.size(nu_cm))


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
        table = molecule_table(pop, env, grid.nu, active)          # packed once per active set

        def residual(x):
            sh = BandShape(**dict(zip(names, x)), profile=start.profile)
            Pa = table.spectra(sh).T
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


if HAVE_NUMBA:
    @njit(cache=True)
    def _nnls_gram(A, y, max_iter, tol):
        """Lawson-Hanson NNLS of ``A x = y`` on Gram columns ``A^T a_j`` built as columns enter the passive
        set: the fits have ~1300 columns of which ~100 end up passive, so the full ``A^T A`` is never
        needed. ``A`` Fortran-ordered. Returns (x, converged)."""
        n = A.shape[1]
        G = np.zeros((n, n))
        have = np.zeros(n, dtype=np.bool_)
        passive = np.zeros(n, dtype=np.bool_)
        aty = A.T @ y
        x = np.zeros(n)
        it = 0
        while True:
            w = aty.copy()                       # gradient -A^T (A x - y), x lives on built columns only
            for j in range(n):
                if x[j] > 0:
                    w -= G[:, j] * x[j]
            jb, best = -1, tol
            for j in range(n):
                if not passive[j] and w[j] > best:
                    jb, best = j, w[j]
            if jb < 0:
                return x, True
            passive[jb] = True
            if not have[jb]:
                G[:, jb] = A.T @ A[:, jb]
                have[jb] = True
            while True:                          # least squares on the passive set, stepping back if needed
                it += 1
                if it > max_iter:
                    return x, False
                P = np.nonzero(passive)[0]
                k = P.size
                gp, b = np.empty((k, k)), np.empty(k)
                for a in range(k):
                    b[a] = aty[P[a]]
                    for c in range(k):
                        gp[a, c] = G[P[a], P[c]]
                z = np.linalg.solve(gp, b)
                if z.min() > 0:
                    for a in range(k):
                        x[P[a]] = z[a]
                    break
                alpha, block = np.inf, -1
                for a in range(k):
                    if z[a] <= 0:
                        r = x[P[a]] / (x[P[a]] - z[a])
                        if r < alpha:
                            alpha, block = r, P[a]
                if block < 0:                        # 0/0 steps only (degenerate): let scipy have it
                    return x, False
                for a in range(k):
                    j = P[a]
                    x[j] += alpha * (z[a] - x[j])
                    if j == block or x[j] <= 0:      # the blocking variable is zero up to rounding
                        x[j] = 0.0
                        passive[j] = False


def _nnls(A, y):
    """NNLS with the numba kernel when available, scipy's otherwise (or if the kernel fails)."""
    from scipy.optimize import nnls

    if HAVE_NUMBA:
        tol = 10 * max(A.shape) * np.finfo(float).eps * np.abs(A).sum(axis=0).max() * np.abs(y).max()
        try:
            x, converged = _nnls_gram(np.asfortranarray(A), np.ascontiguousarray(y), 3 * A.shape[1], tol)
        except np.linalg.LinAlgError:        # a singular passive Gram matrix (collinear columns)
            converged = False
        if converged:
            return x
    return nnls(A, y)[0]


def fit_spectrum(observed, model_columns, continuum_columns):
    """NNLS of ``observed`` (on the grid) on [PAH columns | continuum columns]; bins with NaN are skipped."""
    y = np.asarray(observed, dtype=float)
    ok = np.isfinite(y)
    P, Cc = np.asarray(model_columns, dtype=float), np.asarray(continuum_columns, dtype=float)
    A = np.hstack([P, Cc])[ok]
    scale = np.where(A.max(axis=0) > 0, A.max(axis=0), 1.0)
    x = _nnls(A / scale, y[ok]) / scale
    w, c = x[:P.shape[1]], x[P.shape[1]:]
    pah, cont = P @ w, Cc @ c
    r = (y - pah - cont)[ok]
    return SpectralFit(w, c, pah, cont, float(np.sqrt(np.mean(r**2)) / np.mean(y[ok])), float(r @ r))
