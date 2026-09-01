"""Cash-Karp RK5(4) solver with embedded error estimate and 5th-order step control.

Translated from the Fortran ``rk54_mod`` in ``dust_solver.f90``.

Algorithm
---------
The Cash-Karp method uses six function evaluations to produce both a
5th-order solution and an embedded 4th-order solution for error estimation::

    k1 = f(y)
    k2 = f(y + h * a21*k1)
    k3 = f(y + h * (a31*k1 + a32*k2))
    k4 = f(y + h * (a41*k1 + a42*k2 + a43*k3))
    k5 = f(y + h * (a51*k1 + a52*k2 + a53*k3 + a54*k4))
    k6 = f(y + h * (a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5))

    y_new  = y + h * (b1*k1 + b3*k3 + b4*k4 + b6*k6)   [5th-order; b2=b5=0]
    y_err  = h * (e1*k1 + e3*k3 + e4*k4 + e5*k5 + e6*k6) [b - b*, e2=0]

Error metric and 5th-order step-size controller::

    max_error = max(|y_err| / max(|y|, y_min))
    scale     = 0.9 × (errmax / max(max_error, 1e-10))^0.2
    h_new     = h × clip(scale, 0.1, 2.0)

On the *first* call within a new integration interval ``kmax`` is obtained
from ``k1`` and used to cap the step: ``h_local = min(h, 1/kmax)``.
"""

from __future__ import annotations

from typing import List

import numpy as np

from .chemistry_state import DustChemistryState
from .rhs import DustProcess, compute_rhs
from .solver_base import DustSolverBase

# ---------------------------------------------------------------------------
# Cash-Karp Butcher tableau coefficients (from Fortran rk54_mod)
# ---------------------------------------------------------------------------
_A21 = 1.0 / 5.0
_A31 = 3.0 / 40.0;     _A32 = 9.0 / 40.0
_A41 = 3.0 / 10.0;     _A42 = -9.0 / 10.0;    _A43 = 6.0 / 5.0
_A51 = -11.0 / 54.0;   _A52 = 5.0 / 2.0;      _A53 = -70.0 / 27.0; _A54 = 35.0 / 27.0
_A61 = 1631.0 / 55296.0; _A62 = 175.0 / 512.0; _A63 = 575.0 / 13824.0
_A64 = 44275.0 / 110592.0; _A65 = 253.0 / 4096.0

# 5th-order weights (b2=b5=0, omitted)
_B1 = 37.0 / 378.0
_B3 = 250.0 / 621.0
_B4 = 125.0 / 594.0
_B6 = 512.0 / 1771.0

# Error weights e = b (5th) - b* (4th)
_E1 = 37.0 / 378.0   - 2825.0 / 27648.0
_E3 = 250.0 / 621.0  - 18575.0 / 48384.0
_E4 = 125.0 / 594.0  - 13525.0 / 55296.0
_E5 =                  -277.0 / 14336.0
_E6 = 512.0 / 1771.0  - 1.0 / 4.0


class RK54Solver(DustSolverBase):
    """Adaptive Cash-Karp RK5(4) solver.

    Parameters
    ----------
    errmax : float
        Maximum allowed error per step (default ``0.1``).
        Compared against the max-relative error from the embedded 4th-order
        estimate; matches the ``errmax`` parameter in ``dust_commons.f90``.
    y_min : float
        Floor used in the relative-error denominator so near-zero
        components don't dominate the error metric (default ``1e-40``).
    """

    def __init__(self, errmax: float = 0.1, y_min: float = 1e-40) -> None:
        self._errmax = float(errmax)
        self._y_min  = float(y_min)

    @property
    def name(self) -> str:
        return "RK54"

    # ------------------------------------------------------------------
    # Internal: raw (non-adaptive) RK54 step
    # ------------------------------------------------------------------

    def _rk54_raw(
        self,
        state: DustChemistryState,
        y_gas: np.ndarray,
        y_dust: np.ndarray,
        h: float,
        processes: List[DustProcess],
        first_call: bool,
    ):
        """Execute one raw RK54 step.

        Returns
        -------
        y_gas_new, y_dust_new, y_gas_err, y_dust_err, h_used, break_flag
        """
        if first_call:
            k1_g, k1_d, kmax = compute_rhs(
                state, y_gas, y_dust, processes, return_kmax=True
            )
            if kmax == 0.0:
                return y_gas.copy(), y_dust.copy(), None, None, h, True
            h_used = min(1.0 / kmax, h)
        else:
            k1_g, k1_d = compute_rhs(state, y_gas, y_dust, processes)
            h_used = h

        k2_g, k2_d = compute_rhs(
            state,
            y_gas  + h_used * (_A21 * k1_g),
            y_dust + h_used * (_A21 * k1_d),
            processes,
        )
        k3_g, k3_d = compute_rhs(
            state,
            y_gas  + h_used * (_A31 * k1_g + _A32 * k2_g),
            y_dust + h_used * (_A31 * k1_d + _A32 * k2_d),
            processes,
        )
        k4_g, k4_d = compute_rhs(
            state,
            y_gas  + h_used * (_A41 * k1_g + _A42 * k2_g + _A43 * k3_g),
            y_dust + h_used * (_A41 * k1_d + _A42 * k2_d + _A43 * k3_d),
            processes,
        )
        k5_g, k5_d = compute_rhs(
            state,
            y_gas  + h_used * (_A51 * k1_g + _A52 * k2_g + _A53 * k3_g + _A54 * k4_g),
            y_dust + h_used * (_A51 * k1_d + _A52 * k2_d + _A53 * k3_d + _A54 * k4_d),
            processes,
        )
        k6_g, k6_d = compute_rhs(
            state,
            y_gas  + h_used * (_A61*k1_g + _A62*k2_g + _A63*k3_g + _A64*k4_g + _A65*k5_g),
            y_dust + h_used * (_A61*k1_d + _A62*k2_d + _A63*k3_d + _A64*k4_d + _A65*k5_d),
            processes,
        )

        # 5th-order solution (b2=b5=0)
        coeff_g  = h_used * (_B1 * k1_g + _B3 * k3_g + _B4 * k4_g + _B6 * k6_g)
        coeff_d  = h_used * (_B1 * k1_d + _B3 * k3_d + _B4 * k4_d + _B6 * k6_d)
        y_gas_new  = y_gas  + coeff_g
        y_dust_new = y_dust + coeff_d

        # Embedded error estimate: h * (b - b*) · ki (e2=0)
        y_gas_err  = h_used * (_E1*k1_g + _E3*k3_g + _E4*k4_g + _E5*k5_g + _E6*k6_g)
        y_dust_err = h_used * (_E1*k1_d + _E3*k3_d + _E4*k4_d + _E5*k5_d + _E6*k6_d)

        return y_gas_new, y_dust_new, y_gas_err, y_dust_err, h_used, False

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
        """Adaptive RK5(4) step with embedded error control.

        The error metric is the maximum relative magnitude of the embedded
        4th/5th-order difference over all ODE variables.  The step-size
        controller uses the optimal 5th-order exponent (0.2).

        Returns
        -------
        y_gas_new, y_dust_new, h_new, accepted, break_flag, error
        """
        y_gas_try, y_dust_try, y_gas_err, y_dust_err, h_used, break_flag = (
            self._rk54_raw(state, y_gas, y_dust, h, processes, first_call)
        )

        if break_flag:
            return y_gas.copy(), y_dust.copy(), h, True, True, 0.0

        # Error metric: max relative error from embedded estimate
        y_min = self._y_min
        err_gas  = np.abs(y_gas_err)  / np.maximum(np.abs(y_gas),  y_min)
        err_dust = np.abs(y_dust_err) / np.maximum(np.abs(y_dust), y_min)
        max_error = max(float(np.max(err_gas)), float(np.max(err_dust)))

        accepted = max_error <= self._errmax

        # 5th-order step-size controller
        scale = 0.9 * (self._errmax / max(max_error, 1e-10)) ** 0.2
        h_new = h_used * min(2.0, max(0.1, scale))

        if accepted:
            return y_gas_try, y_dust_try, h_new, True, False, max_error
        return y_gas.copy(), y_dust.copy(), h_new, False, False, max_error
