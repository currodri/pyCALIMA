"""CALIMA dust and PAH chemistry solvers.

Provides a modular Python implementation of the dust/PAH evolution ODE system
originally written in Fortran for RAMSES-CALIMA.
"""

from .chemistry_state import DustChemistryState, DustBinParams, PAHBinParams
from .dust_init import load_initial_conditions
from .grain_dynamics import grain_relative_velocity, sticking_probability_from_velocity
from .rhs import DustProcess, build_process_list, compute_rhs
from .solver_base import DustSolverBase
from .rk4 import RK4Solver
from .equilibrium import (
    EquilibriumSolverBase,
    NewtonKrylovEquilibriumSolver,
    SparseNewtonEquilibriumSolver,
)
from .ode_driver import integrate_dust_ode
from .run_chemistry import run_chemistry, compute_element_totals, check_mass_conservation
from .plotting import plot_chemistry_evolution
from .output_writer import save_chemistry_txt, save_equilibrium_txt

__all__ = [
    "DustChemistryState",
    "DustBinParams",
    "PAHBinParams",
    "load_initial_conditions",
    "grain_relative_velocity",
    "sticking_probability_from_velocity",
    "DustProcess",
    "build_process_list",
    "compute_rhs",
    "DustSolverBase",
    "RK4Solver",
    "EquilibriumSolverBase",
    "NewtonKrylovEquilibriumSolver",
    "SparseNewtonEquilibriumSolver",
    "integrate_dust_ode",
    "run_chemistry",
    "compute_element_totals",
    "check_mass_conservation",
    "plot_chemistry_evolution",
    "save_chemistry_txt",
    "save_equilibrium_txt",
]
