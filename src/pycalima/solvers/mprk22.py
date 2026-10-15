"""Modified Patankar–Runge–Kutta solver, MPRK22 (Kopecz & Meister 2018).

The dust ODE is a production–destruction system: every process moves mass
between the gas-phase elements and the grain bins and conserves each element.
Writing the right-hand side as a sum of transfers ``v`` (one donor ``d`` each,
``v`` conserving every element), a modified Patankar step weights each
transfer by the donor's new-to-old ratio,

    z_new = z_old + h Σ v · z_new[d] / z_den[d],

(Burchard, Deleersnijder & Meister 2003), which is a linear system in
``z_new`` whose columns sum to one. The step therefore conserves every
element to round-off, keeps single-donor transfers positive for any ``h``,
and is implicit in whichever node is being emptied, the gas included. That
last property is what the per-bin Anninos update lacks when sublimation and
recondensation balance: there the stiff variable is the gas.

MPRK22 (α = 1) is two such solves per step:

    stage 1:  z1 from the MPE step with rates at z_old, denominators z_old;
    stage 2:  z2 with the mean of the rates at z_old and z1, denominators z1.

``z2`` is second order and ``z1`` first order; their difference sizes the
step, ``h_new = h × clip(0.9 √(errmax / err), 0.2, 5)``.

Transfers are built from the per-process kernel outputs:

* processes flagged ``bin_local`` (each bin trades only with the gas:
  accretion, sputtering, sublimation, condensation, the PAH gas processes)
  give one transfer per bin; the donor is the bin when it loses mass, else
  its most depleted element, the smallest ``y_gas[e] / f_e``;
* other processes are split by composition group (bins with identical
  element mass fractions, PAHs counting as pure carbon). Within a group,
  mass leaving bins is shared pro rata among the group's gainers, gas
  included, one transfer per losing bin; a group whose losers include gas
  elements becomes a single transfer with the node that empties first as
  donor;
* whatever a kernel does not conserve is added explicitly.

After each step the round-off left by the linear solve is returned to the
gas, element by element, so the totals change by exactly the explicit
(non-conserving) part.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .chemistry_state import DustChemistryState
from .rhs import DustProcess
from .solver_base import DustSolverBase


def _composition_groups(state: DustChemistryState) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Element-content matrix A [n_el, nbins] and the bin indices of each
    group of identical composition."""
    nbins = state.npah + state.ndust
    A = np.zeros((state.n_elements, nbins))
    for pb in state.pah_bins:
        A[pb.C_index, pb.bin_index] = 1.0
    for db in state.dust_bins:
        for e, f in zip(db.el_indices, db.el_mfractions):
            A[e, state.npah + db.bin_index] = f
    groups: Dict[tuple, List[int]] = {}
    for j in range(nbins):
        groups.setdefault(tuple(np.round(A[:, j], 12)), []).append(j)
    return A, [np.array(g) for g in groups.values()]


class MPRK22Solver(DustSolverBase):
    """Conservative, positive, second-order modified Patankar–RK solver.

    Parameters
    ----------
    errmax : float
        Tolerance on the relative difference between the first- and
        second-order solutions (default ``0.01``).
    y_min : float
        Floor on the donor densities in the Patankar weights (default ``1e-40``).
    atol_frac : float
        Components below ``atol_frac`` times the largest density in the cell
        are left out of the error norm (default ``1e-12``).
    """

    def __init__(self, errmax: float = 0.01, y_min: float = 1e-40,
                 atol_frac: float = 1e-12) -> None:
        self._errmax = float(errmax)
        self._y_min = float(y_min)
        self._atol_frac = float(atol_frac)

    @property
    def name(self) -> str:
        return "MPRK22"

    # ------------------------------------------------------------------
    # transfers
    # ------------------------------------------------------------------

    def _transfers(self, state, z, processes):
        """Return (donors, V, explicit residual, kmax) at state z, where
        row k of V is a transfer with donor donors[k] and Σ V = the RHS."""
        n_el = state.n_elements
        A, groups = _composition_groups(state)
        g, d = z[:n_el], z[n_el:]
        donors: List[int] = []
        rows: List[np.ndarray] = []
        residual = np.zeros_like(z)
        kmax = 0.0

        for proc in processes:
            dg = np.zeros_like(g)
            dd = np.zeros_like(d)
            kmax = max(kmax, proc.rate_fn(state, g.copy(), d.copy(), dg, dd))
            if not (np.any(dg) or np.any(dd)):
                continue
            # what the dust change implies for the gas, by element
            dg_bins = -(A * dd)                     # [n_el, nbins]
            residual[:n_el] += dg - dg_bins.sum(axis=1)

            if proc.bin_local:
                for j in np.nonzero(dd)[0]:
                    v = np.zeros_like(z)
                    v[:n_el] = dg_bins[:, j]
                    v[n_el + j] = dd[j]
                    if dd[j] < 0.0:
                        donors.append(n_el + j)
                    else:
                        els = np.nonzero(A[:, j])[0]
                        donors.append(int(els[np.argmin(g[els] / A[els, j])]))
                    rows.append(v)
                continue

            for grp in groups:
                if not np.any(dd[grp]):
                    continue
                u = np.zeros_like(z)
                u[:n_el] = dg_bins[:, grp].sum(axis=1)
                u[n_el + grp] = dd[grp]
                gain = np.maximum(u, 0.0)
                G = gain.sum()
                losers = np.nonzero(u < 0.0)[0]
                if G <= 0.0 or losers.size == 0:
                    continue
                if np.all(losers >= n_el):
                    # bins of one composition: share each loss pro rata
                    for l in losers:
                        v = gain * (-u[l] / G)
                        v[l] += u[l]
                        donors.append(int(l))
                        rows.append(v)
                else:
                    donors.append(int(losers[np.argmin(z[losers] / -u[losers])]))
                    rows.append(u)

        V = np.array(rows) if rows else np.zeros((0, z.size))
        return np.array(donors, dtype=int), V, residual, kmax

    def _patankar_solve(self, z_old, z_den, terms, h):
        """Solve z = z_old + h Σ_w w (V·z[d]/z_den[d] + residual) for z."""
        n = z_old.size
        M = np.zeros((n, n))
        rhs = z_old.copy()
        for w, (donors, V, residual, _) in terms:
            if donors.size:
                coef = (h * w) / np.maximum(z_den[donors], self._y_min)
                np.add.at(M.T, donors, V * coef[:, None])
            rhs += h * w * residual
        return np.linalg.solve(np.eye(n) - M, rhs)

    # ------------------------------------------------------------------
    # DustSolverBase interface
    # ------------------------------------------------------------------

    def step(
        self,
        state: DustChemistryState,
        y_gas: np.ndarray,
        y_dust: np.ndarray,
        h: float,
        processes: List[DustProcess],
        first_call: bool = False,
    ) -> tuple:
        """One MPRK22 step.

        Returns
        -------
        y_gas_new, y_dust_new, h_new, accepted, break_flag, error
        """
        n_el = state.n_elements
        z0 = np.concatenate([y_gas, y_dust])
        t0 = self._transfers(state, z0, processes)
        if first_call and t0[3] == 0.0 and t0[1].shape[0] == 0:
            return y_gas.copy(), y_dust.copy(), h, True, True, 0.0

        z1 = self._patankar_solve(z0, z0, [(1.0, t0)], h)
        if z1.min() < 0.0:
            return y_gas.copy(), y_dust.copy(), 0.5 * h, False, False, 1e10
        t1 = self._transfers(state, z1, processes)
        z2 = self._patankar_solve(z0, z1, [(0.5, t0), (0.5, t1)], h)
        # The solve conserves each element up to round-off, which grows with
        # the matrix entries h·rate/z for long steps; hand that residual back
        # to the gas so the totals are exact.
        A, _ = _composition_groups(state)
        target = y_gas + A @ y_dust + 0.5 * h * (t0[2][:n_el] + t1[2][:n_el])
        z2[:n_el] += target - (z2[:n_el] + A @ z2[n_el:])
        if z2.min() < 0.0:
            return y_gas.copy(), y_dust.copy(), 0.5 * h, False, False, 1e10

        floor = max(self._y_min, self._atol_frac * float(np.max(np.abs(z0))))
        err = float(np.max(np.abs(z2 - z1) / np.maximum(np.abs(z2), floor)))
        h_new = h * min(5.0, max(0.2, 0.9 * np.sqrt(self._errmax / max(err, 1e-12))))
        if err > self._errmax:
            return y_gas.copy(), y_dust.copy(), h_new, False, False, err
        return z2[:n_el], z2[n_el:], h_new, True, False, err
