"""Right-hand-side assembly for the dust chemistry ODE.

Mirrors the Fortran ``dust_rhs_mod`` (``dust_solver.f90``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import numpy as np

from .chemistry_state import DustChemistryState

# Type alias used in DustProcess
RateFn = Callable[
    [DustChemistryState, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    float,
]


# ---------------------------------------------------------------------------
# Process descriptor
# ---------------------------------------------------------------------------

@dataclass
class DustProcess:
    """A named chemistry process bundled with its rate-kernel function.

    Attributes
    ----------
    name : str
        Human-readable identifier (e.g. ``'accretion'``).
    rate_fn : callable
        Callable with signature
        ``(state, y_gas, y_dust, dydt_gas, dydt_dust) -> kmax``.
    source : bool
        True if the process can add mass to dust grains.
    sink : bool
        True if the process can remove mass from dust grains.
    """

    name: str
    rate_fn: RateFn
    source: bool = False
    sink: bool = False


# ---------------------------------------------------------------------------
# Process-list factory
# ---------------------------------------------------------------------------

def build_process_list(state: DustChemistryState) -> List[DustProcess]:
    """Return the list of active :class:`DustProcess` objects.

    The list is built from the physics flags and model selection strings
    stored in *state*, matching the RAMSES-CALIMA process dispatch.
    """
    from .dust_rates import (
        accretion_rate,
        coagulation_rate,
        pah_accretion_rate,
        sublimation_rate,
        thermal_sputtering_rate,
        # PAH rates
        pah_photolysis_rate,
        pah_sputtering_rate,
        totton2012_pah_coalescence_rate,
        tielens2021_pah_coalescence_rate,
        pah_cluster_evaporation_rate,
        pah_freezing_rate,
        # Turbulent collision rates
        turbulent_shattering_rate,
        turbulent_all_shattering_rate,
        turbulent_coagulation_rate,
        turbulent_all_coagulation_rate,
    )

    processes: List[DustProcess] = []

    # ---- Grain growth (accretion) ----
    if state.dust_accretion and state.ndust > 0:
        processes.append(
            DustProcess("accretion", accretion_rate, source=True, sink=False)
        )

    # ---- Grain destruction (thermal sputtering) ----
    if state.dust_sputtering and state.ndust > 0:
        processes.append(
            DustProcess("sputtering", thermal_sputtering_rate, source=False, sink=True)
        )

    # ---- Grain destruction (thermal sublimation, GD89) ----
    if state.dust_sublimation and state.ndust > 0:
        processes.append(
            DustProcess("sublimation", sublimation_rate, source=False, sink=True)
        )

    # ---- Grain coagulation (model-selected) ----
    if state.dust_coagulation and state.ndust > 1:
        if state.coagulation_model == "Aoyama2017":
            processes.append(
                DustProcess("coagulation", coagulation_rate, source=True, sink=True)
            )
        elif state.coagulation_model == "turbulent":
            processes.append(
                DustProcess(
                    "coagulation_turbulent", turbulent_coagulation_rate,
                    source=True, sink=True,
                )
            )
        elif state.coagulation_model == "turbulent_all":
            processes.append(
                DustProcess(
                    "coagulation_turbulent_all", turbulent_all_coagulation_rate,
                    source=True, sink=True,
                )
            )
        else:
            # Fallback to Aoyama2017
            processes.append(
                DustProcess("coagulation", coagulation_rate, source=True, sink=True)
            )

    # ---- Grain shattering (turbulent, model-selected) ----
    if state.dust_shattering and state.ndust > 1:
        if state.shattering_model == "turbulent_all":
            processes.append(
                DustProcess(
                    "shattering_all", turbulent_all_shattering_rate,
                    source=True, sink=True,
                )
            )
        else:
            # Default: "turbulent" (self-collisions only)
            processes.append(
                DustProcess(
                    "shattering", turbulent_shattering_rate,
                    source=True, sink=True,
                )
            )

    # ---- PAH accretion ----
    if state.pah_accretion and state.npah > 0:
        processes.append(
            DustProcess("pah_accretion", pah_accretion_rate, source=True, sink=False)
        )

    # ---- PAH photolysis ----
    if state.pah_photolysis and state.npah > 0:
        processes.append(
            DustProcess("pah_photolysis", pah_photolysis_rate, source=False, sink=True)
        )

    # ---- PAH sputtering ----
    if state.pah_sputtering and state.npah > 0:
        processes.append(
            DustProcess("pah_sputtering", pah_sputtering_rate, source=False, sink=True)
        )

    # ---- PAH coalescence (model-selected) ----
    if state.pah_coalescence and state.npah > 1:
        if state.coalescence_model == "Tielens2021":
            processes.append(
                DustProcess(
                    "pah_coalescence", tielens2021_pah_coalescence_rate,
                    source=True, sink=True,
                )
            )
        else:
            # Default: Totton2012
            processes.append(
                DustProcess(
                    "pah_coalescence", totton2012_pah_coalescence_rate,
                    source=True, sink=True,
                )
            )

    # ---- PAH cluster evaporation ----
    if state.pah_cluster_evaporation and state.npah > 1:
        processes.append(
            DustProcess(
                "pah_cluster_evaporation", pah_cluster_evaporation_rate,
                source=True, sink=True,
            )
        )

    # ---- PAH freezing onto dust ----
    if state.pah_freezing and state.npah > 0 and state.ndust > 0:
        processes.append(
            DustProcess("pah_freezing", pah_freezing_rate, source=True, sink=True)
        )

    return processes


# ---------------------------------------------------------------------------
# RHS assembler
# ---------------------------------------------------------------------------

def compute_rhs(
    state: DustChemistryState,
    y_gas: np.ndarray,
    y_dust: np.ndarray,
    processes: List[DustProcess],
    *,
    return_kmax: bool = False,
):
    """Evaluate the full right-hand side of the dust chemistry ODE.

    Parameters
    ----------
    state :
        Fixed environment and bin parameters.
    y_gas : ndarray, shape (n_elements,)
        Current gas-phase element mass densities [g cm⁻³].
    y_dust : ndarray, shape (npah + ndust,)
        Current dust/PAH mass densities [g cm⁻³].
    processes : list of DustProcess
        Active processes (built by :func:`build_process_list`).
    return_kmax : bool
        If ``True``, also return the maximum characteristic rate [s⁻¹]
        found across all processes (used by the RK4 step-size controller).

    Returns
    -------
    dydt_gas : ndarray, shape (n_elements,)
    dydt_dust : ndarray, shape (npah + ndust,)
    kmax : float (only when *return_kmax* is ``True``)
    """
    dydt_gas = np.zeros_like(y_gas)
    dydt_dust = np.zeros_like(y_dust)
    kmax = 0.0

    for proc in processes:
        pk = proc.rate_fn(state, y_gas, y_dust, dydt_gas, dydt_dust)
        kmax = max(kmax, pk)

    if return_kmax:
        return dydt_gas, dydt_dust, kmax
    return dydt_gas, dydt_dust
