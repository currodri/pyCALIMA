(workflows)=
# Usage and workflows


## 1. Run a quick configuration sanity check

```bash
python test_config_check.py
```

## 2. Regenerate all model tables

```bash
calima-export
```

## 3. Run individual exporters

```bash
# Dust optical properties
python -m pycalima.models.dust_radiation.export_dust_optical_properties

# PAH optical properties
python -m pycalima.models.PAH_radiation.export_pah_optical_properties

# Dust collisional cooling
python -m pycalima.models.dust_gas_collisions.export_collisional_cooling_bins

# Dust sputtering tables
python -m pycalima.models.dust_gas_collisions.export_sputtering_rates_bins

# PAH sputtering tables
python -m pycalima.models.PAH_gas_collisions.export_pah_sputtering_rates_bins

# Dust charging summary tables
python -m pycalima.models.dust_charge.export_dust_charging_vs_gamma

# Dust photoelectric heating tables
python -m pycalima.models.dust_charge.export_dust_photoelectric_heating

# PAH photoelectric heating tables
python -m pycalima.models.PAH_charge.export_PAH_photoelectric_heating_tables

# PAH dissociation tables
python -m pycalima.models.PAH_photophysics.export_pah_dissociation_tables
```

## 4. Use a custom configuration file

```bash
calima-export --config path/to/your_config.json
```

Most exporter modules accept `--config` and read runtime defaults from the JSON
`export_parameters` block.

## 5. Run the chemistry solver (single point)

```bash
# Adaptive RK4 time integration — writes <stem>_evolution.txt and <stem>_evolution.png
calima-run example_ic

# Override integration time and output directory
calima-run example_ic \
    --t_end_Myr 10 --output-dir results/

# Steady-state equilibrium solver (no time-stepping); writes only the .txt
calima-run equilibrium_postshock_test \
    --solver newton_krylov

# Higher-order adaptive integrator
calima-run example_ic --solver rk54

# Quasi-implicit integrator, for destruction-dominated bins
# (e.g. thermal sputtering in hot gas, where explicit steps get very small)
calima-run equilibrium_postshock_test --solver anninos
```

`--solver` overrides the config's `solver.type` and accepts any of the five
{ref}`solver-types`. Omit it to use whatever the config specifies.

## 6. Run the chemistry solver over a parameter grid

```bash
# T–nH grid with RK4, all cores in parallel
calima-grid \
    --config  example_ic \
    --x-param T   --x-values 50 100 500 2000 8000 \
    --y-param nH  --y-values 0.1 1 10 100 1000 \
    --t-end-Myr 5 --solver rk4 \
    --output-npz grid_T_nH.npz

# T–G0 grid with the equilibrium solver
calima-grid \
    --config  equilibrium_postshock_test \
    --x-param T   --x-values 100 1000 5000 20000 \
    --y-param G0  --y-values 0.1 1 10 100 \
    --solver newton_krylov \
    --output-npz grid_T_G0_eq.npz
```

**Python API:**

```python
from solvers.run_chemistry import run_chemistry
results = run_chemistry("example_ic", t_end_Myr=100)
# results["output_txt"] and results["output_plot"] give the saved file paths

from solvers.run_grid import run_grid
grid = run_grid(
    config_path="example_ic",
    x_param="T",  x_values=[50, 100, 500, 2000, 8000],
    y_param="nH", y_values=[0.1, 1, 10, 100, 1000],
    t_end_Myr=5.0, solver_type="rk4",
)
# grid["DTM"][i, j]       — dust-to-metal ratio at (T[i], nH[j])
# grid["rho_dust"]        — shape (nx, ny, n_dust_bins)
# grid["rho_pah"]         — shape (nx, ny, n_pah_bins)
# grid["converged"]       — bool array (equilibrium) or all-True (rk4)
```
