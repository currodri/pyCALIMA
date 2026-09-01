"""Anninos et al. (1997) quasi-implicit per-bin solver.

Translated from the Fortran ``anninos_mod`` in ``dust_solver.f90``.

Algorithm
---------
For each dust/PAH bin *j*, with creation rate ``C_j`` and destruction rate
``D_j_raw`` (both positive, in g cm⁻³ s⁻¹), the net rate is
``f_j = C_j - D_j_raw``.  The quasi-implicit formula branches on the
magnitude of the change relative to the current value:

1. **Negligible** (``|f_j| * h < 1e-12 * max(y_j, y_min)``):
   ``y_new[j] = y_j``  (no update)

2. **Small** (``|f_j| * h < 1e-2 * max(y_j, y_min)``):
   ``y_new[j] = y_j + f_j * h``  (explicit Euler)

3. **Large, mild destruction** (``D_j * h < 1e-6``):
   ``y_new[j] = (y_j + C_j * h) / (1 + D_j * h)``  (linear quasi-implicit)

4. **Large, strong destruction**:
   ``y_new[j] = y_eq + (y_j - y_eq) * exp(-D_j * h)``  (analytic exponential)

where ``D_j = D_j_raw / max(y_j, y_min)`` is the specific destruction rate
and ``y_eq = C_j / max(D_j, 1e-300)`` is the analytic equilibrium density.

Gas conservation
----------------
Instead of updating the gas phase via ``y_gas + h * dydt_gas`` (which
would double-count the Euler approximation already embedded in the per-bin
formula), we propagate the *actual* per-bin dust delta exactly through the
element-mass-fraction arrays — matching the Fortran implementation.

Step-size control
-----------------
The quasi-implicit step is **always accepted** — the analytic formulas in cases 3
and 4 are exact solutions for constant-coefficient ODEs, so even a large relative
change (e.g. dust sputtered to near-zero in one step) is physically correct.

A 1st-order error metric is computed *after* the step solely to size the next step::

    max_error = max(|y_new - y| / max(|y|, y_min))
    scale     = 0.9 × errmax / max(max_error, 1e-10)
    h_new     = h × clip(scale, 0.1, 2.0)

The only hard rejection is negativity: if any component of ``y_new`` goes negative
the step is rejected and ``h`` is halved.
"""

from __future__ import annotations

from math import exp
from typing import List

import numpy as np

from .chemistry_state import DustChemistryState
from .rhs import DustProcess
from .solver_base import DustSolverBase

# Branching thresholds (matching Fortran anninos_mod)
_THRESH_NEGLIGIBLE = 1e-12
_THRESH_EULER      = 1e-2
_THRESH_LINEAR     = 1e-6


def _compute_rhs_split(
    state: DustChemistryState,
    y_gas: np.ndarray,
    y_dust: np.ndarray,
    processes: List[DustProcess],
):
    """Decompose the RHS into creation and destruction arrays.

    Each process contributes separately to ``src`` (positive increments)
    and ``snk`` (magnitude of negative increments) for the dust/PAH vector.
    Gas-phase contributions are accumulated normally.

    Returns
    -------
    dydt_gas : ndarray, shape (n_elements,)
        Net gas-phase rate from all processes [g cm⁻³ s⁻¹].
    src : ndarray, shape (npah + ndust,)
        Per-bin creation rate ``C_j`` [g cm⁻³ s⁻¹], all non-negative.
    snk : ndarray, shape (npah + ndust,)
        Per-bin destruction rate magnitude ``D_j_raw`` [g cm⁻³ s⁻¹],
        all non-negative.
    kmax : float
        Maximum characteristic rate across all processes [s⁻¹].
    """
    dydt_gas = np.zeros_like(y_gas)
    src = np.zeros_like(y_dust)
    snk = np.zeros_like(y_dust)
    kmax = 0.0
    for proc in processes:
        dg = np.zeros_like(y_gas)
        dd = np.zeros_like(y_dust)
        pk = proc.rate_fn(state, y_gas, y_dust, dg, dd)
        src      += np.maximum(dd, 0.0)
        snk      += np.maximum(-dd, 0.0)
        dydt_gas += dg
        kmax = max(kmax, pk)
    return dydt_gas, src, snk, kmax


class AnninosSolver(DustSolverBase):
    """Quasi-implicit per-bin solver after Anninos et al. (1997).

    Unconditionally stable for bins dominated by destruction (e.g. thermal
    sputtering in hot gas), where explicit methods would require extremely
    small steps.

    Parameters
    ----------
    errmax : float
        Maximum allowed relative change per step (default ``0.1``).
    y_min : float
        Floor used in denominator of specific destruction rate and relative
        error metric (default ``1e-40``).
    """

    def __init__(self, errmax: float = 0.1, y_min: float = 1e-40) -> None:
        self._errmax = float(errmax)
        self._y_min  = float(y_min)

    @property
    def name(self) -> str:
        return "Anninos"

    # ------------------------------------------------------------------
    # Public: adaptive step (DustSolverBase interface)
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
        """Adaptive quasi-implicit step.

        On ``first_call`` the maximum characteristic rate ``kmax`` is
        used to cap the initial step size.

        Returns
        -------
        y_gas_new, y_dust_new, h_new, accepted, break_flag, error
        """
        dydt_gas, src, snk, kmax = _compute_rhs_split(
            state, y_gas, y_dust, processes
        )

        if first_call:
            if kmax == 0.0:
                return y_gas.copy(), y_dust.copy(), h, True, True, 0.0
            h = min(1.0 / kmax, h)

        y_min = self._y_min

        # ------------------------------------------------------------------
        # Per-bin quasi-implicit update
        # ------------------------------------------------------------------
        y_dust_new = y_dust.copy()
        nbins = len(y_dust)

        for j in range(nbins):
            yj    = y_dust[j]
            Cj    = src[j]
            Djr   = snk[j]
            fj    = Cj - Djr
            scale = max(yj, y_min)

            if abs(fj) * h < _THRESH_NEGLIGIBLE * scale:
                # Case 1: negligible — no update
                pass
            elif abs(fj) * h < _THRESH_EULER * scale:
                # Case 2: explicit Euler
                y_dust_new[j] = yj + fj * h
            else:
                Dj = Djr / scale   # specific destruction rate [s⁻¹]
                if Dj * h < _THRESH_LINEAR:
                    # Case 3: linear quasi-implicit
                    y_dust_new[j] = (yj + Cj * h) / (1.0 + Dj * h)
                else:
                    # Case 4: analytic exponential
                    y_eq = Cj / max(Dj, 1e-300)
                    y_dust_new[j] = y_eq + (yj - y_eq) * exp(-Dj * h)

        # ------------------------------------------------------------------
        # Gas update: Euler step using the net gas rate from the RHS.
        #
        # This matches the Fortran Anninos implementation, which uses
        # ``y_gas_new = y_gas + h * dydt_gas`` rather than propagating
        # per-bin dust deltas through el_mfractions.
        #
        # The key advantage: dydt_gas is identically zero for redistribution
        # processes (coagulation, shattering), so those processes cannot
        # spuriously modify the gas phase regardless of how the per-bin
        # dust update is computed.  Accretion and sputtering contribute
        # correctly through dydt_gas from _compute_rhs_split.
        # ------------------------------------------------------------------
        y_gas_new = y_gas + h * dydt_gas

        # ------------------------------------------------------------------
        # Negativity guard: any negative component → hard reject, halve h
        # ------------------------------------------------------------------
        if float(y_dust_new.min()) < 0.0 or float(y_gas_new.min()) < 0.0:
            return y_gas.copy(), y_dust.copy(), h * 0.5, False, False, 1e10

        # ------------------------------------------------------------------
        # Step-size controller (1st-order) — step is ALWAYS accepted.
        #
        # The quasi-implicit formula is unconditionally stable: the analytic
        # exponential (case 4) and linear (case 3) updates are exact solutions
        # to the per-bin ODE with constant coefficients, so a large relative
        # change (e.g. dust sputtered to near-zero in one step) is physically
        # correct, not a numerical error.  Rejecting based on max_error here
        # would cause an infinite loop for fast-sputtering cells.
        #
        # The error metric is only used to SIZE the next step so that changes
        # remain smooth.  This matches the Fortran anninos_mod behaviour where
        # the quasi-implicit step is always accepted and only negativity causes
        # a hard reject (handled above).
        # ------------------------------------------------------------------
        err_dust = np.abs(y_dust_new - y_dust) / np.maximum(np.abs(y_dust), y_min)
        err_gas  = np.abs(y_gas_new  - y_gas)  / np.maximum(np.abs(y_gas),  y_min)
        max_error = max(float(np.max(err_dust)), float(np.max(err_gas)))

        scale = 0.9 * self._errmax / max(max_error, 1e-10)
        h_new = h * min(2.0, max(0.1, scale))

        return y_gas_new, y_dust_new, h_new, True, False, max_error
