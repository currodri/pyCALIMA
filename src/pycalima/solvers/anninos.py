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
The gas is updated from the *actual* per-bin dust change, propagated through
the bin's element mass fractions (all carbon for a PAH bin), so gas + dust is
conserved element by element whatever branch a bin takes. This is what the
Fortran does. (An earlier Python port used ``y_gas + h * dydt_gas``, which
disagrees with the per-bin update outside the Euler branch and created mass
when a large source and sink nearly cancel.)

Step-size control
-----------------
As in the Fortran, a step is rejected and ``h`` halved if any component goes
negative, and otherwise accepted only if the largest relative change,

    max_error = max(|y_new - y| / max(|y|, y_min)),

is at most ``errmax`` (the "10% rule"); in both accepted and rejected cases

    h_new = h × clip(0.9 × errmax / max(max_error, 1e-10), 0.1, 2.0).

Limitation
----------
Each bin's creation rate is evaluated with the gas at the start of the step,
so the gas is treated explicitly. When gas–grain exchange dominates in both
directions (sublimation with recondensation) the gas is the stiff variable:
the scheme is then only stable for steps shorter than the gas relaxation time
and settles into a biased oscillation otherwise. Use ``mprk22`` there.
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
        """Adaptive quasi-implicit step (Fortran ``anninos_step``).

        On ``first_call`` a zero ``kmax`` (no active process) stops the
        integration; as in the Fortran, ``h`` is not capped by ``1/kmax``.

        Returns
        -------
        y_gas_new, y_dust_new, h_new, accepted, break_flag, error
        """
        _, src, snk, kmax = _compute_rhs_split(state, y_gas, y_dust, processes)

        if first_call and kmax == 0.0:
            return y_gas.copy(), y_dust.copy(), h, True, True, 0.0

        y_min = self._y_min
        y_gas_new = y_gas.copy()
        y_dust_new = y_dust.copy()
        negative = False

        for j in range(len(y_dust)):
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

            if y_dust_new[j] < 0.0:
                # Cannot happen for Cj, Dj >= 0; leave the bin (and so the
                # gas) unchanged and reject the step.
                negative = True
                y_dust_new[j] = yj

            # Move the actual change in this bin out of / into the gas.
            delta = y_dust_new[j] - yj
            if j < state.npah:
                y_gas_new[state.pah_bins[j].C_index] -= delta
            else:
                db = state.dust_bins[j - state.npah]
                for el_idx, mfrac in zip(db.el_indices, db.el_mfractions):
                    y_gas_new[el_idx] -= delta * mfrac

        if negative or float(y_gas_new.min()) < 0.0:
            return y_gas.copy(), y_dust.copy(), h * 0.5, False, False, 1e10

        err_dust = np.abs(y_dust_new - y_dust) / np.maximum(np.abs(y_dust), y_min)
        err_gas  = np.abs(y_gas_new  - y_gas)  / np.maximum(np.abs(y_gas),  y_min)
        max_error = max(float(np.max(err_dust)), float(np.max(err_gas)))

        scale = 0.9 * self._errmax / max(max_error, 1e-10)
        h_new = h * min(2.0, max(0.1, scale))

        if max_error > self._errmax:
            return y_gas.copy(), y_dust.copy(), h_new, False, False, max_error
        return y_gas_new, y_dust_new, h_new, True, False, max_error
