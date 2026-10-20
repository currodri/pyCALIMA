"""The (Z, N_H) population network of one PAH family.

The state of a PAH of fixed carbon skeleton is its charge Z and hydrogen
count N_H. Every process is a transition (Z, N_H) -> (Z + dZ, N_H + dN_H) with
a rate [s^-1] that may depend on the source state, given as an array of shape
(n_Z, n_NH). The network does not know the physics behind a rate: a method
card, or the caller, decides which processes exist and how fast they are.
That keeps, for example, anion H addition by associative detachment
(-1, N_H) -> (0, N_H + 1) an explicit choice rather than a hard-wired one.

A rate given for a state whose target lies outside the network is an error.
Outputs: the steady state, time evolution, and the flux of every process,
from which H2 formation and H-atom balance follow.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg


def gth_stationary(R):
    """Stationary distribution of a continuous-time Markov chain by GTH.

    ``R[i, j]`` >= 0 is the rate of i -> j (the diagonal is ignored). The
    Grassmann-Taksar-Heyman algorithm (Grassmann, Taksar & Heyman 1985) is
    Gaussian elimination of the states from the last to the first, in which
    the diagonal is never formed: each pivot is the sum of the remaining exit
    rates. With no subtraction, every stationary probability comes out to
    high relative accuracy (O'Cinneide 1993), not only those near the peak.
    Cost O(n^3), negligible for a few hundred states.

    Returns None if the chain is not irreducible (a state with no exit to
    the states not yet eliminated).
    """
    A = np.array(R, dtype=float)
    np.fill_diagonal(A, 0.0)
    n = A.shape[0]
    s = np.zeros(n)
    for k in range(n - 1, 0, -1):
        s[k] = A[k, :k].sum()
        if not s[k] > 0.0:
            return None
        # censor state k: paths i -> k -> j become i -> j with weight A[k, j] / s_k
        A[:k, :k] += np.outer(A[:k, k], A[k, :k] / s[k])
    p = np.zeros(n)
    p[0] = 1.0
    for k in range(1, n):
        p[k] = p[:k] @ A[:k, k] / s[k]
        if p[k] > 1e250:
            # p is relative to state 0; when state 0 is > 1e250 less likely than state k (a fully
            # hydrogenated 100 C PAH against its bare anion), rescale before the ratios overflow.
            # The final normalisation makes this exact.
            p[:k + 1] /= p[k]
    return p / p.sum()


@dataclass
class Process:
    name: str
    rate: np.ndarray            # [s^-1], shape (n_Z, n_NH), indexed by source state
    dz: int
    dnh: int
    h2_released: int = 0        # H2 molecules released per event (for H2 formation)


@dataclass
class PAHNetwork:
    """States Z in ``charges`` x N_H in ``nh_values``.

    Parameters
    ----------
    charges : sequence of int
        Consecutive charges, e.g. (-1, 0, 1, 2).
    nh_values : sequence of int
        Consecutive hydrogen counts, e.g. range(0, 23).
    """

    charges: tuple
    nh_values: tuple
    processes: list = field(default_factory=list)

    def __post_init__(self):
        self.charges = tuple(int(z) for z in self.charges)
        self.nh_values = tuple(int(n) for n in self.nh_values)
        for seq in (self.charges, self.nh_values):
            if np.any(np.diff(seq) != 1):
                raise ValueError("charges and nh_values must be consecutive integers")

    @property
    def shape(self):
        return len(self.charges), len(self.nh_values)

    def zeros(self):
        return np.zeros(self.shape)

    def add(self, name, rate, dz=0, dnh=0, h2_released=0):
        """Add a process. ``rate`` broadcasts to (n_Z, n_NH)."""
        rate = np.broadcast_to(np.asarray(rate, dtype=float), self.shape).copy()
        if np.any(rate < 0) or not np.all(np.isfinite(rate)):
            raise ValueError(f"{name}: rates must be finite and non-negative")
        nz, nn = self.shape
        iz, inh = np.indices(self.shape)
        outside = (iz + dz < 0) | (iz + dz >= nz) | (inh + dnh < 0) | (inh + dnh >= nn)
        if np.any(rate[outside] > 0):
            bad = [(self.charges[a], self.nh_values[b]) for a, b in zip(*np.nonzero(outside & (rate > 0)))]
            raise ValueError(f"{name}: rate > 0 from {bad[:4]} leads outside the network")
        self.processes.append(Process(name, rate, int(dz), int(dnh), int(h2_released)))
        return self

    def generator(self):
        """Sparse Q with dP/dt = Q P, states flattened as Z-major (iz * n_NH + inh)."""
        nz, nn = self.shape
        n = nz * nn
        idx = np.arange(n).reshape(self.shape)
        rows, cols, vals = [], [], []
        diag = np.zeros(n)
        for p in self.processes:
            src = np.nonzero(p.rate > 0)
            if src[0].size == 0:
                continue
            s = idx[src]
            t = idx[src[0] + p.dz, src[1] + p.dnh]
            rows.append(t)
            cols.append(s)
            vals.append(p.rate[src])
            np.add.at(diag, s, p.rate[src])
        rows.append(np.arange(n))
        cols.append(np.arange(n))
        vals.append(-diag)
        return scipy.sparse.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                       shape=(n, n))

    def transition_matrix(self) -> np.ndarray:
        """Dense R[i, j] = rate of i -> j [s^-1] (i != j), states flattened as in ``generator``."""
        nz, nn = self.shape
        n = nz * nn
        idx = np.arange(n).reshape(self.shape)
        R = np.zeros((n, n))
        for p in self.processes:
            src = np.nonzero(p.rate > 0)
            if src[0].size:
                np.add.at(R, (idx[src], idx[src[0] + p.dz, src[1] + p.dnh]), p.rate[src])
        np.fill_diagonal(R, 0.0)
        return R

    def steady_state(self) -> np.ndarray:
        """Stationary populations f[Z, N_H], summing to 1.

        By the Grassmann-Taksar-Heyman algorithm (``gth_stationary``), which
        gives every population to high relative accuracy, however small
        (e.g. a fully dehydrogenated fraction of 1e-60). A general linear
        solve resolves populations only down to ~1e-16 of the largest one;
        below that it returns rounding noise. Falls back to the sparse solve
        if the chain is reducible.
        """
        f = gth_stationary(self.transition_matrix())
        if f is None:
            return self._steady_state_sparse()
        return f.reshape(self.shape)

    def _steady_state_sparse(self) -> np.ndarray:
        """Stationary populations by a general sparse solve (accurate to ~1e-16 of the peak)."""
        Q = self.generator().tolil()
        n = Q.shape[0]
        Q[0, :] = np.ones(n)
        b = np.zeros(n)
        b[0] = 1.0
        f = scipy.sparse.linalg.spsolve(Q.tocsc(), b)
        f = np.clip(f, 0.0, None)
        return (f / f.sum()).reshape(self.shape)

    def evolve(self, f0, times) -> np.ndarray:
        """Populations at ``times`` [s] from f0 (shape (n_Z, n_NH)); returns (len(times), n_Z, n_NH).

        Dense matrix exponential: these networks have at most a few hundred
        states, and the cost of a Krylov expm_multiply grows with ||Q|| t,
        which is ~1e6 or more over the times that matter (fast H loss, slow
        charge exchange).
        """
        Q = self.generator().toarray()
        f = np.asarray(f0, dtype=float).ravel()
        out, t_prev = [], 0.0
        for t in np.atleast_1d(times):
            f = scipy.linalg.expm(Q * (t - t_prev)) @ f
            out.append(f.reshape(self.shape).copy())
            t_prev = t
        return np.array(out)

    def fluxes(self, f) -> dict:
        """Events per PAH per second of each process, sum_states f rate."""
        return {p.name: float(np.sum(f * p.rate)) for p in self.processes}

    def h2_formation_rate(self, f) -> float:
        """H2 molecules released per PAH per second."""
        return float(sum(p.h2_released * np.sum(f * p.rate) for p in self.processes))

    def charge_fractions(self, f) -> dict:
        return {z: float(f[i].sum()) for i, z in enumerate(self.charges)}

    def hydrogenation_fractions(self, f) -> dict:
        return {nh: float(f[:, j].sum()) for j, nh in enumerate(self.nh_values)}
