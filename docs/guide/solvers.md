(solver-package)=
# The solver package


The `solvers/` package implements the dust and PAH chemistry ODE solver in pure Python,
mirroring the RAMSES-CALIMA Fortran modules (`dust_solver.f90`, `dust_rates.f90`, etc.).
It can be used as a standalone CLI tool, called through its Python API, or imported into
analysis notebooks.

## Module overview

| Module | Purpose |
|---|---|
| `run_chemistry.py` | Main CLI entry point and `run_chemistry()` Python API; orchestrates a single-point run |
| `run_grid.py` | Sweeps a 2-D parameter grid (e.g. T–nH, T–G0); joblib-parallelised; saves `.npz` output |
| `chemistry_state.py` | Dataclasses for `DustChemistryState`, `DustBinParams`, `PAHBinParams`; element registry |
| `rhs.py` | `build_process_list()` — assembles active `DustProcess` objects from physics flags |
| `dust_rates.py` | Rate kernels: accretion, coagulation, thermal sputtering, PAH photolysis/sputtering/coalescence, PEH |
| `solver_base.py` | Abstract base class `DustSolverBase`; defines the `step()` interface |
| `rk4.py` | Adaptive Cash–Karp RK4 time integrator |
| `rk54.py` | Adaptive Cash–Karp RK5(4) integrator with an embedded 4th-order error estimate |
| `anninos.py` | Quasi-implicit per-bin integrator after Anninos et al. (1997); unconditionally stable for destruction-dominated bins |
| `equilibrium.py` | Steady-state solvers: `NewtonKrylovEquilibriumSolver` and `SparseNewtonEquilibriumSolver` |
| `ode_driver.py` | Low-level driver loop that calls `solver.step()` until `dt` is consumed |
| `dust_init.py` | `load_initial_conditions()` — parses the JSON config and builds the state and density arrays |
| `table_io.py` | Readers for pre-computed sputtering and rate tables from `model_data/` |
| `output_writer.py` | `save_chemistry_txt()` — writes a `#`-commented ASCII evolution table |
| `plotting.py` | `plot_chemistry_evolution()` — multi-panel dust/PAH/gas-element density figure |
| `grain_dynamics.py` | Grain relative velocities and sticking probabilities |

(solver-types)=
## Solver types

Five solvers, all registered in `SOLVER_REGISTRY` in
`solvers/run_chemistry.py`. Select one with the `solver.type` key in a config
file, or with `--solver` on `calima-grid`.

| Type key | Class | Kind | Description |
|---|---|---|---|
| `rk4` | `RK4Solver` | time integration | Adaptive Cash–Karp RK4; the default |
| `rk54` | `RK54Solver` | time integration | Adaptive Cash–Karp RK5(4); step size set by an embedded 4th-order error estimate, matching `errmax` in `dust_commons.f90` |
| `anninos` | `AnninosSolver` | time integration | Quasi-implicit per-bin scheme after Anninos et al. (1997). Unconditionally stable for bins dominated by destruction — thermal sputtering in hot gas, say — where an explicit method needs prohibitively small steps |
| `newton_krylov` | `NewtonKrylovEquilibriumSolver` | steady state | Newton outer loop with an LGMRES inner Krylov solver |
| `sparse_newton` | `SparseNewtonEquilibriumSolver` | steady state | Newton iterations with a finite-difference Jacobian and sparse LU factorisation |

Tuning parameters, read from the `solver` block of the config:

| Type key | Accepts |
|---|---|
| `rk4` | `errmax` (default `0.1`) |
| `rk54`, `anninos` | `errmax` (`0.1`), `y_min` (`1e-40`) |
| `newton_krylov` | `f_tol`, `f_rtol`, `maxiter`, `inner_maxiter`, `eps_fd` |
| `sparse_newton` | `rtol`, `atol`, `maxiter`, `eps_fd`, `alpha_min` |

`y_min` floors the denominator of the relative-error metric so that
near-zero components cannot dominate it.

## Bundled configuration files (`solvers/configs/`)

| File | Scenario |
|---|---|
| `example_ic.json` | CNM (T = 100 K, nH = 100 cm⁻³, G0 = 1) with all standard processes |
| `all_processes_test.json` | Same environment, all physics flags on, 5 Myr run |
| `equilibrium_gasdominant_test.json` | Gas-dominated medium, equilibrium test |
| `equilibrium_nk_test.json` | Reference case for the Newton–Krylov solver |
| `equilibrium_postshock_test.json` | Post-shock conditions for RK4-vs-equilibrium comparison |
