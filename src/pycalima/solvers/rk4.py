"""Classical 4th-order Runge–Kutta solver with adaptive step-size control.

Translated from the Fortran ``rk4_mod`` in ``dust_solver.f90``.

Algorithm
---------
The raw RK4 step computes the standard four-stage increment::

    k1 = f(y)
    k2 = f(y + h/2 * k1)
    k3 = f(y + h/2 * k2)
    k4 = f(y + h   * k3)
    y_new = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)

On the *first* call within a new integration interval the maximum
characteristic rate ``kmax`` is computed from ``k1`` and used to
set an initial step estimate ``h_local = min(1/kmax, h)``.

The adaptive wrapper computes a step-quality metric as the maximum
relative change across all ODE variables:

    max_error = max(|y_new - y| / max(|y|, ε))

If ``max_error ≤ errmax`` the step is accepted; otherwise it is
rejected.  Either way a new step size is proposed via a standard
PI controller::

    scale = 0.9 × (errmax / max_error)
    h_new = h × clip(scale, 0.1, 2.0)
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .chemistry_state import DustChemistryState
from .rhs import DustProcess, compute_rhs
from .solver_base import DustSolverBase

# Numerical constants used in the RK4 weights
_HALF = 0.5
_TWO = 2.0
_SIXTH = 1.0 / 6.0


class RK4Solver(DustSolverBase):
    """Adaptive 4th-order Runge–Kutta solver.

    Parameters
    ----------
    errmax : float
        Maximum allowed relative change per step (default ``0.1``).
        Larger values allow bigger steps at the cost of accuracy;
        this is the ``errmax`` parameter from ``dust_commons.f90``.
    """

    def __init__(self, errmax: float = 0.1) -> None:
        self._errmax = float(errmax)

    @property
    def name(self) -> str:
        return "RK4"

    # ------------------------------------------------------------------
    # Internal: raw (non-adaptive) RK4 step
    # ------------------------------------------------------------------

    def _rk4_raw(
        self,
        state: DustChemistryState,
        y_gas: np.ndarray,
        y_dust: np.ndarray,
        h: float,
        processes: List[DustProcess],
        first_call: bool,
    ) -> Tuple[np.ndarray, np.ndarray, float, bool]:
        """Execute one raw RK4 step.

        Returns
        -------
        y_gas_new, y_dust_new, h_used, break_flag
            *break_flag* is ``True`` when kmax == 0 on a *first_call*
            (no active processes).
        """
        if first_call:
            k1_g, k1_d, kmax = compute_rhs(
                state, y_gas, y_dust, processes, return_kmax=True
            )
            if kmax == 0.0:
                return y_gas.copy(), y_dust.copy(), h, True
            h_used = min(1.0 / kmax, h)
        else:
            k1_g, k1_d = compute_rhs(state, y_gas, y_dust, processes)
            h_used = h

        k2_g, k2_d = compute_rhs(
            state,
            y_gas  + h_used * _HALF * k1_g,
            y_dust + h_used * _HALF * k1_d,
            processes,
        )
        k3_g, k3_d = compute_rhs(
            state,
            y_gas  + h_used * _HALF * k2_g,
            y_dust + h_used * _HALF * k2_d,
            processes,
        )
        k4_g, k4_d = compute_rhs(
            state,
            y_gas  + h_used * k3_g,
            y_dust + h_used * k3_d,
            processes,
        )

        coeff = h_used * _SIXTH
        y_gas_new  = y_gas  + coeff * (k1_g + _TWO * k2_g + _TWO * k3_g + k4_g)
        y_dust_new = y_dust + coeff * (k1_d + _TWO * k2_d + _TWO * k3_d + k4_d)

        return y_gas_new, y_dust_new, h_used, False

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
        """Adaptive RK4 step with error control.

        The error metric is the maximum relative change across all ODE
        variables.  If ``max_error > errmax`` the step is rejected and
        a reduced step size is proposed.

        Returns
        -------
        y_gas_new, y_dust_new, h_new, accepted, break_flag, error
        """
        y_gas_try, y_dust_try, h_used, break_flag = self._rk4_raw(
            state, y_gas, y_dust, h, processes, first_call
        )

        if break_flag:
            return y_gas.copy(), y_dust.copy(), h, True, True, 0.0

        # Error metric: max relative change
        eps = 1.0e-40
        err_gas  = np.abs(y_gas_try  - y_gas)  / np.maximum(np.abs(y_gas),  eps)
        err_dust = np.abs(y_dust_try - y_dust) / np.maximum(np.abs(y_dust), eps)
        max_error = max(float(np.max(err_gas)), float(np.max(err_dust)))

        accepted = max_error <= self._errmax

        # PI step-size controller  (clip to [0.1, 2.0] × h_used)
        if max_error > 0.0:
            scale = 0.9 * (self._errmax / max_error)
        else:
            scale = 2.0
        h_new = h_used * min(2.0, max(0.1, scale))

        if accepted:
            return y_gas_try, y_dust_try, h_new, True, False, max_error
        return y_gas.copy(), y_dust.copy(), h_new, False, False, max_error
