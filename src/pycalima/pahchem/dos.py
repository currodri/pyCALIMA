"""Vibrational density and sum of states of a harmonic PAH.

Three independent ways to count states, used to check each other:

- ``beyer_swinehart``: exact count on an energy grid (Beyer & Swinehart 1973),
  with each frequency rounded to the grain.
- ``direct_count``: brute-force enumeration of quantum numbers. Small mode
  sets only; it is the reference that the other two are tested against.
- ``whitten_rabinovitch``: the semi-classical closed form (Whitten &
  Rabinovitch 1963), fast and smooth. Measured against ``beyer_swinehart``,
  it is within about 1 % above the zero-point energy E_z for 30 modes, but its
  error grows with the number of modes: -3 % (C24H12) and -9 % (C54H18) at
  E = E_z. Use it for checks and for quick estimates, not for rates.

Plus the two temperatures that the legacy code conflated:

- ``microcanonical_temperature``: 1/T = k d ln rho / dE, a property of an
  isolated molecule at energy E;
- ``canonical_temperature``: the T of a heat bath whose mean thermal energy
  U(T) equals E.

Energies and frequencies are in cm^-1. Energies are measured from the
zero-point level, so E = 0 is the vibrational ground state.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln

KB_CM = 0.695034800          # Boltzmann constant [cm^-1 K^-1]
CM_PER_EV = 8065.543937      # [cm^-1 eV^-1]

try:                          # the optional `accel` extra
    from numba import njit

    @njit(cache=True)
    def _bs_kernel(counts, steps):
        """Beyer-Swinehart in place: for each mode, c[i] += c[i - f] in increasing i."""
        n = counts.size
        for f in steps:
            for i in range(f, n):
                counts[i] += counts[i - f]
except ImportError:
    _bs_kernel = None


def _frequencies(freqs_cm):
    freqs = np.asarray(freqs_cm, dtype=float)
    if freqs.ndim != 1 or freqs.size == 0 or np.any(freqs <= 0):
        raise ValueError("frequencies must be a non-empty 1-D array of positive values")
    return freqs


def beyer_swinehart(freqs_cm, e_max_cm, grain_cm=1.0):
    """Exact harmonic state count on a grid.

    Parameters
    ----------
    freqs_cm : array_like
        Harmonic frequencies [cm^-1]. Each is rounded to a whole number of
        grains.
    e_max_cm : float
        Highest energy of the grid [cm^-1].
    grain_cm : float
        Grid spacing [cm^-1].

    Returns
    -------
    energy : ndarray
        Grid energies ``i * grain_cm`` [cm^-1].
    counts : ndarray
        Number of states whose energy rounds to each grid point. The density
        of states is ``counts / grain_cm`` and the sum of states is
        ``np.cumsum(counts)``.
    """
    freqs = _frequencies(freqs_cm)
    n = int(np.floor(e_max_cm / grain_cm)) + 1
    counts = np.zeros(n)
    counts[0] = 1.0
    steps = np.maximum(np.rint(freqs / grain_cm).astype(np.int64), 1)
    if _bs_kernel is not None:
        _bs_kernel(counts, steps)
    else:
        for f in steps:
            # Sequential recursion c[i] += c[i - f]; every source index in a
            # block of length f lies in the block before, already updated, so
            # each block can be added as one slice.
            for start in range(f, n, f):
                stop = min(start + f, n)
                counts[start:stop] += counts[start - f:stop - f]
    if not np.all(np.isfinite(counts)):
        raise OverflowError("state count overflowed float64; lower e_max_cm")
    return np.arange(n) * grain_cm, counts


def direct_count(freqs_cm, e_max_cm, grain_cm=1.0):
    """Enumerate every vibrational state up to ``e_max_cm``.

    Exact for any frequencies (no rounding): each state's energy is binned on
    the same grid as ``beyer_swinehart``. The cost grows like the number of
    states, so this is a reference for small mode sets only.
    """
    freqs = _frequencies(freqs_cm)
    n = int(np.floor(e_max_cm / grain_cm)) + 1
    counts = np.zeros(n)

    def recurse(i, energy):
        if i == freqs.size:
            counts[int(np.rint(energy / grain_cm))] += 1
            return
        while energy <= e_max_cm + 1e-9:
            recurse(i + 1, energy)
            energy += freqs[i]

    recurse(0, 0.0)
    return np.arange(n) * grain_cm, counts


def whitten_rabinovitch(freqs_cm, energy_cm):
    """Semi-classical sum and density of states.

    N(E) = (E + a E_z)^s / (s! prod nu) and
    rho(E) = (E + a E_z)^(s-1) (1 - beta dw/dE') / ((s-1)! prod nu), with
    a = 1 - beta w(E'), E' = E/E_z, E_z the zero-point energy and
    beta = (s-1)/s <nu^2>/<nu>^2.

    Returns
    -------
    log_sum, log_density : ndarray
        Natural logarithms of N(E) and of rho(E) [per cm^-1].
    """
    freqs = _frequencies(freqs_cm)
    E = np.asarray(energy_cm, dtype=float)
    s = freqs.size
    e_z = 0.5 * freqs.sum()
    beta = (s - 1) / s * np.mean(freqs**2) / np.mean(freqs) ** 2
    ep = E / e_z
    with np.errstate(divide="ignore", invalid="ignore"):
        low = 1.0 / (5.0 * ep + 2.73 * np.sqrt(ep) + 3.51)
        dlow = -low**2 * (5.0 + 1.365 / np.sqrt(ep))
        high = np.exp(-2.4191 * ep**0.25)
        dhigh = -high * 0.604775 * ep ** -0.75
    w = np.where(ep < 1.0, low, high)
    dw = np.where(ep < 1.0, dlow, dhigh)
    shifted = E + (1.0 - beta * w) * e_z
    log_prod = np.log(freqs).sum()
    log_sum = s * np.log(shifted) - gammaln(s + 1) - log_prod
    log_density = (np.log(1.0 - beta * dw) + (s - 1) * np.log(shifted)
                   - gammaln(s) - log_prod)
    return log_sum, log_density


def smoothed_log_density(energy_cm, counts, window_cm):
    """ln rho(E) [per cm^-1] averaged over +-window_cm/2, from a grid count.

    A grid count is spiky at low energy, where states are sparse; the
    average over a window is what statistical rates and temperatures use.
    """
    total = np.concatenate([[0.0], np.cumsum(counts)])
    grain = energy_cm[1] - energy_cm[0]
    half = max(int(round(0.5 * window_cm / grain)), 1)
    i = np.arange(energy_cm.size)
    lo = np.clip(i - half, 0, energy_cm.size)
    hi = np.clip(i + half + 1, 0, energy_cm.size)
    with np.errstate(divide="ignore"):
        return np.log((total[hi] - total[lo]) / ((hi - lo) * grain))


def microcanonical_temperature(energy_cm, log_density):
    """T(E) = [k_B d ln rho / dE]^-1 [K] on the grid of ``log_density``.

    NaN where the density is zero (below the lowest state).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        T = 1.0 / (KB_CM * np.gradient(log_density, energy_cm))
    return np.where(np.isfinite(T), T, np.nan)


def thermal_energy(freqs_cm, temperature_k):
    """Mean vibrational energy above the zero point, U(T) [cm^-1]."""
    freqs = _frequencies(freqs_cm)
    x = freqs[:, None] / (KB_CM * np.atleast_1d(np.asarray(temperature_k, dtype=float)))
    U = np.sum(freqs[:, None] / np.expm1(np.minimum(x, 700.0)), axis=0)
    return U if np.ndim(temperature_k) else U[0]


def canonical_temperature(freqs_cm, energy_cm):
    """The T at which U(T) = E [K]; the inverse of ``thermal_energy``.

    Vectorised over E: an initial guess from U on a log-T grid, then Newton
    steps in ln T with the analytic heat capacity, dU/dT = sum f x / T /
    (4 sinh^2(x/2)), x = f / kT. U is convex in ln T, so the iteration
    converges quadratically; it stops at |d ln T| < 1e-14 (the scalar brentq
    it replaces stopped at 2e-12). T is confined to 1 K - 1e6 K; E <= 0
    gives 0.
    """
    freqs = _frequencies(freqs_cm)
    E = np.asarray(energy_cm, dtype=float)
    e = np.atleast_1d(E).astype(float)
    out = np.zeros(e.shape)
    pos = e > 0
    if pos.any():
        lt_min, lt_max = 0.0, np.log(1e6)
        grid = np.linspace(lt_min, lt_max, 400)
        u_grid = thermal_energy(freqs, np.exp(grid))
        target = e[pos]
        lt = np.interp(np.log(target), np.log(np.maximum(u_grid, 1e-300)), grid)
        f = freqs[:, None]
        for _ in range(50):
            T = np.exp(lt)
            x = np.minimum(f / (KB_CM * T[None, :]), 700.0)
            U = np.sum(f / np.expm1(x), axis=0)
            dU_dlnT = np.sum(f * x / (4.0 * np.sinh(0.5 * x) ** 2), axis=0)    # T dU/dT
            step = (U - target) / np.maximum(dU_dlnT, 1e-300)
            lt = np.clip(lt - step, lt_min, lt_max)
            if np.max(np.abs(step)) < 1e-14:
                break
        out[pos] = np.exp(lt)
    return out if E.ndim else out[0]
