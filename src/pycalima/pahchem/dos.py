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

    @njit(cache=True)
    def _bs_kernel_tilted(counts, steps, factors):
        """``_bs_kernel`` on tilted counts: c[i] += c[i - f] q^f (``beyer_swinehart`` with ``beta``)."""
        n = counts.size
        for k in range(steps.size):
            f, q = steps[k], factors[k]
            for i in range(f, n):
                counts[i] += counts[i - f] * q
except ImportError:
    _bs_kernel = _bs_kernel_tilted = None


def _frequencies(freqs_cm):
    freqs = np.asarray(freqs_cm, dtype=float)
    if freqs.ndim != 1 or freqs.size == 0 or np.any(freqs <= 0):
        raise ValueError("frequencies must be a non-empty 1-D array of positive values")
    return freqs


def beyer_swinehart(freqs_cm, e_max_cm, grain_cm=1.0, beta=0.0):
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
    beta : float
        Tilt [per cm^-1]: the counts returned are c(E) exp(-beta E), for grids
        on which c(E) itself overflows float64 (large PAHs, high E;
        ``count_tilt``). The recursion stays exact (c[i] += c[i - f] q^f,
        q = exp(-beta grain)); every consumer takes the same ``beta``
        (``smoothed_log_density``, ``tilted_window_sum``,
        ``cooling.mode_emission_rates``, ``rates.rrkm``), and in their ratios the
        tilt cancels. 0 (the default) is the plain count.

    Returns
    -------
    energy : ndarray
        Grid energies ``i * grain_cm`` [cm^-1].
    counts : ndarray
        Number of states on each grid point, counted with every frequency
        rounded to the grid (so a state's energy is the sum of its rounded
        quanta, not its rounded energy). The density
        of states is ``counts / grain_cm`` and the sum of states is
        ``np.cumsum(counts)`` (with ``beta = 0``).
    """
    freqs = _frequencies(freqs_cm)
    n = int(np.floor(e_max_cm / grain_cm)) + 1
    counts = np.zeros(n)
    counts[0] = 1.0
    steps = np.maximum(np.rint(freqs / grain_cm).astype(np.int64), 1)
    factors = np.exp(-beta * grain_cm * steps) if beta else None
    if beta and _bs_kernel_tilted is not None:
        _bs_kernel_tilted(counts, steps, factors)
    elif _bs_kernel is not None and not beta:
        _bs_kernel(counts, steps)
    else:
        for k, f in enumerate(steps):
            # Sequential recursion c[i] += c[i - f]; every source index in a
            # block of length f lies in the block before, already updated, so
            # each block can be added as one slice.
            for start in range(f, n, f):
                stop = min(start + f, n)
                if beta:
                    counts[start:stop] += counts[start - f:stop - f] * factors[k]
                else:
                    counts[start:stop] += counts[start - f:stop - f]
    if not np.all(np.isfinite(counts)):
        raise OverflowError("state count overflowed float64; lower e_max_cm, or tilt it (count_tilt)")
    if beta and not counts[-1] > 0.0:
        raise OverflowError("tilted state count underflowed at the top of the grid: beta too large")
    return np.arange(n) * grain_cm, counts


#: the highest ln(count) ``count_tilt`` lets an untilted count reach (float64 overflows at ln 709.8)
LOG_COUNT_MAX = 600.0


def count_tilt(freqs_cm, e_max_cm, grain_cm=1.0):
    """beta [per cm^-1] for ``beyer_swinehart`` on [0, e_max_cm]: 0 where the count stays below exp(LOG_COUNT_MAX),
    else d ln rho / dE (1 / k T_micro) at the energy where it reaches that, both from ``whitten_rabinovitch``
    (within ~0.1 in ln rho, enough to place a tilt). ln rho is concave in E, so beyond that energy the tilted count
    c exp(-beta E) stays below its value there; below it, below the untilted count."""
    E = np.linspace(0.0, e_max_cm, 2001)[1:]                  # Whitten-Rabinovitch's correction is singular at E = 0
    _, log_rho = whitten_rabinovitch(freqs_cm, E)
    log_c = log_rho + np.log(grain_cm)
    if not log_c[-1] > LOG_COUNT_MAX:                         # ln rho rises with E: the top is its largest value
        return 0.0
    i = int(np.argmax(log_c > LOG_COUNT_MAX))
    beta = float(np.gradient(log_rho, E)[i])
    if not (np.isfinite(beta) and beta > 0.0):
        raise ValueError(f"count_tilt: no positive tilt found ({beta}) for e_max_cm {e_max_cm}")
    return beta


def tilted_window_sum(values, half, beta_grain=0.0):
    """``sum_{|d| <= half} v[i + d] exp(beta_grain d)``, clipped at the grid's ends: the window sum of the untilted
    values, times exp(-beta E_i), from tilted ``values`` (``beyer_swinehart``); a plain window sum for 0."""
    if not beta_grain:
        total = np.concatenate([[0.0], np.cumsum(values)])
        i = np.arange(values.size)
        return total[np.clip(i + half + 1, 0, values.size)] - total[np.clip(i - half, 0, values.size)]
    kernel = np.exp(-beta_grain * (np.arange(2 * half + 1) - half))
    return np.convolve(values, kernel, mode="same")


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
            j = int(np.rint(energy / grain_cm))
            if j < n:                      # a state in the last half-grain rounds past the grid
                counts[j] += 1
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


def smoothed_log_density(energy_cm, counts, window_cm, beta=0.0):
    """ln rho(E) [per cm^-1] averaged over +-window_cm/2, from a grid count.

    A grid count is spiky at low energy, where states are sparse; the
    average over a window is what statistical rates and temperatures use.
    ``beta``: the counts' tilt (``beyer_swinehart``); the result is the untilted ln rho.
    """
    grain = energy_cm[1] - energy_cm[0]
    half = max(int(round(0.5 * window_cm / grain)), 1)
    i = np.arange(energy_cm.size)
    lo = np.clip(i - half, 0, energy_cm.size)
    hi = np.clip(i + half + 1, 0, energy_cm.size)
    if beta:
        with np.errstate(divide="ignore"):
            return np.log(tilted_window_sum(counts, half, beta * grain) / ((hi - lo) * grain)) + beta * energy_cm
    total = np.concatenate([[0.0], np.cumsum(counts)])
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
    converges quadratically; it stops at ``|d ln T|`` < 1e-14 (the scalar brentq
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
