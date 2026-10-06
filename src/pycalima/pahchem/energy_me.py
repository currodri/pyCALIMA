"""Internal-energy master equation of one PAH state.

The vibrational energy E of a PAH is binned on a grid E_j = j dE (cm^-1). The
molecule moves between bins by:

- **IR emission**: mode i takes E to E - h nu_i at rate k_i(E)
  (``cooling.mode_emission_rates``);
- **photon absorption**: a heating photon of energy e takes E to E + e at rate
  h(e) de (``photo.heating_spectrum``);
- **dissociation**: channel c removes the molecule at rate k_c(E).

A jump that does not land on a grid point is split between the two
neighbouring bins with weights that conserve the mean energy, so absorbed
and radiated power balance exactly on the grid.

Two solutions:

- ``CascadeModel.photon_yields``: the probability that a molecule raised to
  E_j ends in each channel before it cools down, with no further absorption.
  This is the single-photon branching with a proper sink; it is at most 1.
  Unlike the sink-free integral of Andrews et al. (2016, eq. 7), it cannot
  count one molecule twice.
- ``CascadeModel.steady_state``: the stationary P(E) under continuous
  irradiation, with multiphoton heating included. Each dissociated molecule
  is re-injected cold (a renewal process), so the effective rate
  k_c = sum_j P_j k_c(E_j) is the rate at which a parent population of that
  state is destroyed. This is valid while k_c is small compared with the
  absorption rate, which is when the population has time to reach its
  quasi-stationary P(E).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

from pycalima.pahchem import cooling, dos


def stationary_banded(band, lo, up, row0_extra=None):
    """Stationary p (Q p = 0, sum p = 1) of a banded rate matrix.

    ``band`` holds Q in LAPACK banded storage, Q[i, j] (rate j -> i) at
    ``band[up + i - j, j]``, with ``lo`` sub- and ``up`` super-diagonals.
    ``row0_extra`` adds a dense row 0 (re-injection into bin 0 from bins
    outside row 0's band): Q = band + e_0 v^T. The columns of Q sum to zero.

    One balance equation is redundant; it is replaced by sum p = 1 over the
    band of that row, and p is normalised afterwards.
    - With a dense row 0, that row is the one replaced, so the dense part
      never enters. Back-substitution then keeps the far tail of p (where
      dissociation rates are large) accurate to relative precision; a
      Sherman-Morrison correction would subtract O(1) vectors and bury a
      tail ~1e-25 below the peak in rounding.
    - Without one, the row is chosen so that its band covers the peak of p
      (from a first pass with row 0): a row of nearly empty bins (row 0 in
      an intense field) would scale the solution by the inverse of their
      tiny population and amplify rounding.
    Returns None if the system is singular.
    """
    n = band.shape[1]

    def solve(row):
        ab = band.copy()
        j = np.arange(max(0, row - lo), min(n, row + up + 1))
        ab[up + row - j, j] = 1.0
        rhs = np.zeros(n)
        rhs[row] = 1.0
        try:
            y = scipy.linalg.solve_banded((lo, up), ab, rhs, check_finite=False)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(y)) or y.sum() <= 0:
            return None
        y = np.clip(y, 0.0, None)
        return y / y.sum()

    p = solve(0)
    if row0_extra is not None and np.any(row0_extra):
        return p
    # First pass: row 0, or, if that overflows (bins 0-1 empty in an intense
    # field, e.g. a 20 A grain at chi = 1e4), rows spread over the grid
    first = 0
    for first in (0, n - 1, n // 2, n // 4, (3 * n) // 4):
        p = solve(first) if first else p
        if p is not None:
            break
    if p is None:
        return None
    row = int(min(n - 1, np.argmax(p) + lo // 2))
    if row == first:
        return p
    refined = solve(row)
    return refined if refined is not None else p


def _split(position, n):
    """Lower bin, upper bin and upper weight for a fractional bin position."""
    lo = np.floor(position).astype(int)
    w_hi = position - lo
    lo = np.clip(lo, 0, n - 1)
    hi = np.clip(lo + 1, 0, n - 1)
    return lo, hi, w_hi


@dataclass
class SteadyState:
    """Stationary internal-energy distribution and the rates it implies."""

    energy_cm: np.ndarray
    probability: np.ndarray        # P_j, sums to 1
    rates: dict                    # channel -> effective rate [s^-1]
    absorbed_power: float          # cm^-1 s^-1
    radiated_power: float          # cm^-1 s^-1
    overflow: float                # rate of jumps above the grid, clamped to its top [s^-1]

    @property
    def mean_energy_cm(self) -> float:
        return float(self.energy_cm @ self.probability)


class CascadeModel:
    """Energy grid, IR cascade and channels of one PAH state.

    Parameters
    ----------
    freqs_cm, einstein_a : array_like
        Modes and their fundamental Einstein A [s^-1].
    e_max_cm : float
        Top of the grid. Choose it above the energies that matter: jumps
        beyond it are clamped to the top bin and reported as ``overflow``.
    de_cm : float
        Bin width of the master equation.
    grain_cm : float
        Grain of the underlying Beyer-Swinehart count for the IR rates.
    """

    def __init__(self, freqs_cm, einstein_a, e_max_cm, de_cm=50.0, grain_cm=2.0, window_cm=100.0):
        self.freqs_cm = np.asarray(freqs_cm, dtype=float)
        self.de_cm = float(de_cm)
        self.n = int(np.floor(e_max_cm / de_cm)) + 1
        self.energy_cm = np.arange(self.n) * self.de_cm
        E, counts = dos.beyer_swinehart(self.freqs_cm, self.energy_cm[-1] + window_cm, grain_cm)
        self.k_modes = cooling.mode_emission_rates(E, counts, self.freqs_cm, einstein_a,
                                                   window_cm=window_cm, at_cm=self.energy_cm)
        # Fractional down-jump positions, one row per mode
        pos = np.arange(self.n)[None, :] - self.freqs_cm[:, None] / self.de_cm
        self.k_modes[pos < 0] = 0.0                  # cannot emit more than E
        self.k_ir = self.k_modes.sum(axis=0)
        self._lo, self._hi, self._w = _split(np.maximum(pos, 0.0), self.n)
        # IR gains in banded storage (they do not depend on the field or the channels):
        # Q[i, j] (rate j -> i) for i <= j sits at ir_band[up + i - j, j], up = largest down-jump
        cols = np.broadcast_to(np.arange(self.n), self._lo.shape)
        live = self.k_modes > 0.0                    # modes that cannot emit have clamped, unused targets
        self._up = int(max(1, np.max((cols - self._lo)[live], initial=0)))
        self._ir_band = np.zeros((self._up + 1, self.n))
        for target, weight in ((self._lo, 1.0 - self._w), (self._hi, self._w)):
            np.add.at(self._ir_band, (self._up + target[live] - cols[live], cols[live]),
                      (weight * self.k_modes)[live])

    def _absorption_kernel(self, photon_energy_ev, heating_spectrum):
        """Rate of up-jumps by s bins, r_s [s^-1], s = 0, 1, ...

        The heating spectrum is integrated on a fine grid and each piece is
        split between the two neighbouring multiples of dE, conserving energy.
        """
        e_cm = np.asarray(photon_energy_ev, dtype=float) * dos.CM_PER_EV
        h_cm = np.asarray(heating_spectrum, dtype=float) / dos.CM_PER_EV   # per cm^-1
        fine = np.linspace(e_cm.min(), e_cm.max(), 20 * e_cm.size)
        h_fine = np.interp(fine, e_cm, h_cm)
        mid = 0.5 * (fine[1:] + fine[:-1])
        r_mid = 0.5 * (h_fine[1:] + h_fine[:-1]) * np.diff(fine)
        shift = mid / self.de_cm
        s_lo = np.floor(shift).astype(int)
        w_hi = shift - s_lo
        r = np.zeros(s_lo.max() + 2)
        np.add.at(r, s_lo, (1.0 - w_hi) * r_mid)
        np.add.at(r, s_lo + 1, w_hi * r_mid)
        return r

    def _channels(self, channel_rates):
        names = list(channel_rates)
        K = np.array([np.broadcast_to(np.asarray(channel_rates[c], dtype=float), (self.n,))
                      for c in names]) if names else np.zeros((0, self.n))
        return names, K

    def photon_yields(self, channel_rates) -> dict:
        """Y_c(E_j): probability of ending in channel c from bin j, no absorption.

        ``channel_rates`` maps a channel name to k_c on ``self.energy_cm``.
        """
        names, K = self._channels(channel_rates)
        Y = np.zeros((len(names), self.n))
        k_out = K.sum(axis=0)
        for j in range(self.n):
            km = self.k_modes[:, j]
            lo, hi, w = self._lo[:, j], self._hi[:, j], self._w[:, j]
            to_self = km * ((1.0 - w) * (lo == j) + w * (hi == j))
            denom = k_out[j] + km.sum() - to_self.sum()
            if denom <= 0.0:
                continue                                  # no way out: stays (Y = 0)
            lo_part = np.where(lo < j, (1.0 - w) * km, 0.0)
            hi_part = np.where(hi < j, w * km, 0.0)
            Y[:, j] = (K[:, j] + Y[:, lo] @ lo_part + Y[:, hi] @ hi_part) / denom
        return dict(zip(names, Y))

    def steady_state(self, photon_energy_ev, heating_spectrum, channel_rates) -> SteadyState:
        """Stationary P(E) under irradiation, with cold re-injection after dissociation.

        Parameters
        ----------
        photon_energy_ev, heating_spectrum : array_like
            h(e) [s^-1 eV^-1] on a photon-energy grid in eV
            (``photo.heating_spectrum``).
        channel_rates : dict
            Channel name -> k_c on ``self.energy_cm``.
        """
        names, K = self._channels(channel_rates)
        r = self._absorption_kernel(photon_energy_ev, heating_spectrum)
        P = self._solve_banded(r, K.sum(axis=0))
        if P is None:                                   # singular banded system: general sparse solve
            return self._steady_state_sparse(photon_energy_ev, heating_spectrum, channel_rates)
        return self._steady_state_result(P, r, names, K)

    def _solve_banded(self, r, k_out):
        """P from Q P = 0, sum P = 1, by a banded LU (``stationary_banded``).

        Q has the IR down-jumps (at most ``self._up`` bins) above the diagonal
        and the photon up-jumps (``r.size - 1`` bins, clamped to the top bin,
        which stays inside the band) below it. Dissociation re-injects into
        bin 0: the in-band part of that row is in the band, the rest goes to
        ``stationary_banded`` as a rank-1 term. Same solution as the general
        sparse solve (equivalence gate), ~100x faster for an 1800-bin grid.
        Returns None if singular.
        """
        n, up = self.n, self._up
        lo = max(1, int(np.max(np.nonzero(r)[0], initial=0)))
        ab = np.zeros((up + lo + 1, n))
        ab[: up + 1] = self._ir_band
        cols = np.arange(n)
        for s in np.nonzero(r)[0]:                      # unclamped up-jumps: Q[j + s, j]
            if s < n:
                ab[up + s, : n - s] += r[s]
        # clamped jumps from column j (s > n - 1 - j) land in the top bin, n - 1 - j below the diagonal
        tail = np.concatenate([np.cumsum(r[::-1])[::-1], [0.0]])        # tail[k] = sum_{s >= k} r_s
        j = cols[n - 1 - cols <= lo]                                    # columns that can overshoot
        ab[up + (n - 1 - j), j] += tail[np.minimum(n - j, tail.size - 1)]
        ab[up] -= self.k_ir + r.sum() + k_out                          # diagonal: all losses
        # Cold re-injection Q[0, j] += k_out[j]: in band for j <= up, the rest is the dense row
        j0 = np.arange(min(up, n - 1) + 1)
        ab[up - j0, j0] += k_out[j0]
        extra = np.where(cols > up, k_out, 0.0)
        return stationary_banded(ab, lo, up, extra)

    def _steady_state_result(self, P, r, names, K):
        n, cols = self.n, np.arange(self.n)
        # Diagnostics. From bin j a jump of s bins lands min(s, n-1-j) bins up.
        s = np.arange(r.size)
        cum_r = np.concatenate([[0.0], np.cumsum(r)])
        cum_rs = np.concatenate([[0.0], np.cumsum(r * s)])
        room = np.minimum(n - 1 - cols, r.size - 1)          # largest unclamped s
        landed = cum_rs[room + 1] + (n - 1 - cols) * (cum_r[-1] - cum_r[room + 1])
        absorbed_power = float(P @ landed * self.de_cm)
        radiated_power = float(P @ (self.freqs_cm @ self.k_modes))
        overflow = float(P @ (cum_r[-1] - cum_r[room + 1]))
        rates = {c: float(P @ K[i]) for i, c in enumerate(names)}
        return SteadyState(self.energy_cm, P, rates, absorbed_power, radiated_power, overflow)

    def _steady_state_sparse(self, photon_energy_ev, heating_spectrum, channel_rates) -> SteadyState:
        """``steady_state`` by a general sparse solve (the reference for the banded path)."""
        names, K = self._channels(channel_rates)
        n = self.n
        cols = np.arange(n)
        rows_, cols_, vals_ = [], [], []

        def add(rows, values):
            rows_.append(rows)
            cols_.append(cols)
            vals_.append(values)

        # IR emission: gain at the split targets; the loss is on the diagonal
        for m in range(self.freqs_cm.size):
            km = self.k_modes[m]
            add(self._lo[m], (1.0 - self._w[m]) * km)
            add(self._hi[m], self._w[m] * km)
        # Photon absorption: up by s bins at rate r_s, clamped to the top bin
        r = self._absorption_kernel(photon_energy_ev, heating_spectrum)
        for s in np.nonzero(r)[0]:
            add(np.minimum(cols + s, n - 1), np.full(n, r[s]))
        # Dissociation with cold re-injection into bin 0
        k_out = K.sum(axis=0)
        add(np.zeros(n, dtype=int), k_out)
        diag = self.k_ir + r.sum() + k_out
        add(cols, -diag)
        Q = scipy.sparse.csc_matrix((np.concatenate(vals_), (np.concatenate(rows_),
                                     np.concatenate(cols_))), shape=(n, n))
        # Solve Q P = 0 with sum(P) = 1: replace the first equation by the norm
        A = Q.tolil()
        A[0, :] = np.ones(n)
        b = np.zeros(n)
        b[0] = 1.0
        P = scipy.sparse.linalg.spsolve(A.tocsc(), b)
        P = np.clip(P, 0.0, None)
        P /= P.sum()
        return self._steady_state_result(P, r, names, K)
