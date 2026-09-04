"""Abstract base class for all dust chemistry ODE solvers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from .chemistry_state import DustChemistryState
from .rhs import DustProcess


class DustSolverBase(ABC):
    """Abstract base for a single-step ODE integrator.

    Subclasses must implement :meth:`step`, which advances the system by one
    time step and returns a new proposed step size together with accept/reject
    and break flags.

    New solver types can be added by subclassing this class and registering
    them in :data:`solvers.run_chemistry.SOLVER_REGISTRY`.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short human-readable solver name (e.g. ``'RK4'``)."""
        ...

    @abstractmethod
    def step(
        self,
        state: DustChemistryState,
        y_gas: np.ndarray,
        y_dust: np.ndarray,
        h: float,
        processes: List[DustProcess],
        first_call: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, float, bool, bool, float]:
        """Advance the solution by one time step.

        Parameters
        ----------
        state :
            Fixed gas environment and grain-bin parameters.
        y_gas : ndarray, shape (n_elements,)
            Current gas-phase element mass densities [g cm⁻³].
        y_dust : ndarray, shape (npah + ndust,)
            Current dust/PAH mass densities [g cm⁻³].
        h : float
            Proposed step size [s].
        processes : list of DustProcess
            Active physics processes (built by :func:`~solvers.rhs.build_process_list`).
        first_call : bool
            ``True`` on the very first step of a new integration interval.
            Some solvers use this to compute an initial step-size estimate
            from the current maximum rate (``kmax``).

        Returns
        -------
        y_gas_new : ndarray
            Updated gas densities.  Equal to *y_gas* if step was rejected.
        y_dust_new : ndarray
            Updated dust densities.  Equal to *y_dust* if step was rejected.
        h_new : float
            Suggested step size for the **next** call.
        accepted : bool
            ``True`` if the step satisfied the error criterion and the
            solution should be advanced by *h*.
        break_flag : bool
            ``True`` if no active processes were detected and integration
            can be terminated early (kmax == 0 on first call).
        error : float
            The step-quality error metric (max relative change).  Returns
            ``0.0`` on break or when the metric is undefined.
        """
        ...
