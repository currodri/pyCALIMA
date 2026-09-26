(api-solvers)=
# `pycalima.solvers` — the chemistry solver

The dust and PAH chemistry ODE solver, mirroring the RAMSES-CALIMA Fortran
modules. It consumes the precomputed tables under `model_data/` and integrates
dust and PAH mass evolution. Usable as a CLI tool ({doc}`/cli/calima-run`),
through its Python API, or from a notebook.

This is the best-documented corner of the codebase: every module carries a
NumPy-style docstring, almost all are type-annotated, and the package exports a
curated `__all__`, so the summary below is generated from that rather than from
a module scan.

## Modules

```{eval-rst}
.. autosummary::
   :toctree: generated
   :template: autosummary/module.rst

   pycalima.solvers.solver_base
   pycalima.solvers.chemistry_state
   pycalima.solvers.dust_init
   pycalima.solvers.rhs
   pycalima.solvers.dust_rates
   pycalima.solvers.ode_driver
   pycalima.solvers.rk4
   pycalima.solvers.rk54
   pycalima.solvers.anninos
   pycalima.solvers.equilibrium
   pycalima.solvers.grain_dynamics
   pycalima.solvers.table_io
   pycalima.solvers.output_writer
   pycalima.solvers.plotting
   pycalima.solvers.run_chemistry
   pycalima.solvers.run_grid
```

## Public API

`pycalima.solvers` re-exports 21 names for short imports.

```{eval-rst}
.. automodule:: pycalima.solvers
   :members:
   :imported-members:
   :member-order: alphabetical
   :no-index:
```

