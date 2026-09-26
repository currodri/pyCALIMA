"""ODE integration driver for the dust chemistry system.

Translated from the Fortran ``ode_driver_mod::integrate_dust_ode``
(``dust_solver.f90``).

The driver iterates the solver's :meth:`~pycalima.solvers.solver_base.DustSolverBase.step`
method, advancing ``τ`` towards the target ``dt``.  Rejected steps reduce ``h``
without advancing time; accepted steps advance ``τ += h``.  The loop aborts
after ``state.countmax`` total iterations.

After integration, all densities are clipped to zero to prevent
small negative values introduced by floating-point arithmetic.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .chemistry_state import DustChemistryState
from .rhs import DustProcess
from .solver_base import DustSolverBase


def integrate_dust_ode(
    state: DustChemistryState,
    dt: float,
    y_gas_0: np.ndarray,
    y_dust_0: np.ndarray,
    processes: List[DustProcess],
    solver: DustSolverBase,
    *,
    h_init: float = 1.0e10,
    h_min: float = 1.0,
    h_max: Optional[float] = None,
    collect_history: bool = False,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Integrate the dust chemistry ODE over a total time *dt*.

    Parameters
    ----------
    state :
        Fixed gas environment and grain-bin parameters.
    dt : float
        Total integration time [s].
    y_gas_0 : ndarray, shape (n_elements,)
        Initial gas-phase element mass densities [g cm⁻³].
    y_dust_0 : ndarray, shape (npah + ndust,)
        Initial dust/PAH mass densities [g cm⁻³], PAH bins first.
    processes : list of DustProcess
        Active physics processes.
    solver : DustSolverBase
        Step-integrator instance (e.g. :class:`~pycalima.solvers.rk4.RK4Solver`).
    h_init : float
        Initial step-size guess [s] (default 10¹⁰ s ≈ 0.3 kyr).
    h_min : float
        Minimum allowed step size [s].
    h_max : float or None
        Maximum allowed step size [s].  Defaults to *dt*.
    collect_history : bool
        If ``True``, store the full time-evolution as arrays in
        ``diagnostics['history']`` for later plotting.  The history
        always includes t = 0 as its first entry.
    verbose : bool
        Print progress messages.

    Returns
    -------
    y_gas_final : ndarray
        Gas-phase element densities at ``t = dt`` [g cm⁻³].
    y_dust_final : ndarray
        Dust/PAH densities at ``t = dt`` [g cm⁻³].
    diagnostics : dict
        Keys:

        ``tau``
            Actual elapsed time reached [s].
        ``naccepted``
            Number of accepted steps.
        ``nrejected``
            Number of rejected steps.
        ``icount``
            Total iteration count (accepted + rejected).
        ``nincreased``
            Number of times the step size grew between consecutive calls.
        ``nreduced``
            Number of times the step size shrank between consecutive calls.
        ``h_min_used``, ``h_max_used``, ``h_mean_used``
            Min / max / mean of the *accepted* step sizes [s].
        ``err_min``, ``err_max``, ``err_mean``
            Min / max / mean of the step-quality error metric across all
            attempted steps (accepted and rejected).
        ``history``
            Present only when *collect_history* is ``True``.  A dict with
            keys ``time_s`` (ndarray, shape (nsnaps,)),
            ``y_gas`` (ndarray, shape (nsnaps, n_elements)),
            ``y_dust`` (ndarray, shape (nsnaps, npah+ndust)),
            ``h_s`` (ndarray, shape (nsnaps,)) — accepted step size for each
            snapshot (``nan`` at the initial t = 0 entry), and
            ``error`` (ndarray, shape (nsnaps,)) — max relative error for
            each snapshot (``nan`` at the initial t = 0 entry).
            The first snapshot is always the initial state at t = 0.
    """
    if h_max is None:
        h_max = dt

    h = float(np.clip(h_init, h_min, min(h_max, dt)))
    tau = 0.0
    icount = 0
    naccepted = 0
    nrejected = 0
    nincreased = 0
    nreduced = 0
    first_call = True

    # Accumulators for step-size and error statistics
    h_acc: List[float] = []          # accepted step sizes
    err_all: List[float] = []        # errors for every attempted step

    # History snapshots (prepopulate with t=0; h and error are NaN there)
    if collect_history:
        hist_t: List[float] = [0.0]
        hist_g: List[np.ndarray] = [y_gas_0.copy()]
        hist_d: List[np.ndarray] = [y_dust_0.copy()]
        hist_h: List[float] = [np.nan]
        hist_e: List[float] = [np.nan]
    else:
        hist_t = hist_g = hist_d = hist_h = hist_e = None  # type: ignore[assignment]

    y_gas = y_gas_0.copy()
    y_dust = y_dust_0.copy()
    h_prev = h  # track previous submitted step to detect increases/reductions

    while tau < dt:
        # Do not overshoot the target time
        h = min(h, dt - tau)

        y_gas_new, y_dust_new, h_new, accepted, break_flag, error = solver.step(
            state, y_gas, y_dust, h, processes, first_call=first_call
        )
        first_call = False

        if break_flag:
            if verbose:
                print("ODE driver: no active processes — skipping integration.")
            return y_gas_0.copy(), y_dust_0.copy(), _empty_diagnostics(collect_history)

        err_all.append(error)

        if accepted:
            tau += h
            y_gas = y_gas_new
            y_dust = y_dust_new
            naccepted += 1
            h_acc.append(h)
            if collect_history:
                hist_t.append(tau)
                hist_g.append(y_gas.copy())
                hist_d.append(y_dust.copy())
                hist_h.append(h)
                hist_e.append(error)
        else:
            nrejected += 1

        # Track step-size direction changes (compare proposed h_new to current h)
        clipped = float(np.clip(h_new, h_min, h_max))
        if clipped > h_prev:
            nincreased += 1
        elif clipped < h_prev:
            nreduced += 1
        h_prev = clipped

        # Apply step-size bounds for next iteration
        h = clipped
        icount += 1

        if icount >= state.countmax:
            print(
                f"WARNING: ODE driver did not converge in {icount} iterations. "
                f"τ = {tau:.3e} s / dt = {dt:.3e} s "
                f"(accepted={naccepted}, rejected={nrejected})"
            )
            break

    # Clip to non-negative (eliminates tiny floating-point negatives)
    y_gas  = np.maximum(y_gas,  0.0)
    y_dust = np.maximum(y_dust, 0.0)

    # Compute aggregate statistics
    h_arr  = np.array(h_acc,  dtype=np.float64) if h_acc  else np.array([0.0])
    err_arr = np.array(err_all, dtype=np.float64) if err_all else np.array([0.0])

    diagnostics: Dict = {
        "tau":         tau,
        "naccepted":   naccepted,
        "nrejected":   nrejected,
        "icount":      icount,
        "converged":   icount < state.countmax,   # False when countmax was hit
        "nincreased":  nincreased,
        "nreduced":    nreduced,
        "h_min_used":  float(h_arr.min()),
        "h_max_used":  float(h_arr.max()),
        "h_mean_used": float(h_arr.mean()),
        "err_min":     float(err_arr.min()),
        "err_max":     float(err_arr.max()),
        "err_mean":    float(err_arr.mean()),
    }

    if collect_history:
        diagnostics["history"] = {
            "time_s": np.array(hist_t, dtype=np.float64),
            "y_gas":  np.array(hist_g, dtype=np.float64),
            "y_dust": np.array(hist_d, dtype=np.float64),
            "h_s":    np.array(hist_h, dtype=np.float64),
            "error":  np.array(hist_e, dtype=np.float64),
        }

    return y_gas, y_dust, diagnostics


def _empty_diagnostics(collect_history: bool) -> Dict:
    """Return a zeroed diagnostics dict for the break-flag (no-process) case."""
    d: Dict = {
        "tau": 0.0, "naccepted": 0, "nrejected": 0, "icount": 0,
        "converged": True,   # break-flag path: no iteration → vacuously converged
        "nincreased": 0, "nreduced": 0,
        "h_min_used": 0.0, "h_max_used": 0.0, "h_mean_used": 0.0,
        "err_min": 0.0, "err_max": 0.0, "err_mean": 0.0,
    }
    if collect_history:
        d["history"] = {"time_s": np.array([0.0]), "y_gas": None, "y_dust": None}
    return d
