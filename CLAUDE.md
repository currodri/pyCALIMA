# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

pyCALIMA is a scientific research codebase for modeling dust and PAH (Polycyclic Aromatic Hydrocarbon) microphysics used in RAMSES cosmological simulations. It is a **pip-installable package** living under `src/pycalima/` — workflows run via console scripts (`calima-export`, `calima-run`, `calima-grid`) or `python -m pycalima.models.<module>`, from any directory.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"     # or just `pip install -e .` for the core
```

Requires Python >= 3.10. Optional extras: `sim` (yt), `accel` (numba),
`pahdb`, `plots`, `profile`. UCLCHEM (notebooks only) and the pinned
`amespahdbpythonsuite` dev build cannot be extras — see `requirements-dev.txt`.

**Where data lives** — never compute paths by hand; use `pycalima._paths`:

- Read-only reference data is bundled in the package:
  `_paths.get_external_data_path(...)`, `_paths.get_optical_props_path(...)`
- Generated tables and output go somewhere writable:
  `_paths.get_model_data_dir()`, `_paths.get_results_dir()`,
  `_paths.get_plots_dir()`
- Resolution order for writable paths: `$CALIMA_DATA` (a root) -> `./model_data`
  if present -> source checkout -> per-user data dir. Run `calima-paths` to see
  what resolves where.
- `models.grain_size_config.get_model_data_dir()` appends the config's
  `model_name`; `_paths.get_model_data_dir()` does not. Physics modules almost
  always want the former.

## Key Commands

**Configuration sanity check:**
```bash
python diagnostics/check_config.py
```

**Regenerate all precomputed model tables** (writes to `model_data/`):
```bash
calima-export
calima-export --config path/to/custom_config.json
```

**Run individual exporters** (each also accepts `--config`):
```bash
python -m pycalima.models.dust_radiation.export_dust_optical_properties
python -m pycalima.models.PAH_radiation.export_pah_optical_properties
python -m pycalima.models.dust_gas_collisions.export_collisional_cooling_bins
python -m pycalima.models.dust_gas_collisions.export_sputtering_rates_bins
python -m pycalima.models.PAH_gas_collisions.export_pah_sputtering_rates_bins
python -m pycalima.models.dust_charge.export_dust_charging_vs_gamma
python -m pycalima.models.dust_charge.export_dust_photoelectric_heating
python -m pycalima.models.PAH_charge.export_PAH_photoelectric_heating_tables
python -m pycalima.models.PAH_photophysics.export_pah_dissociation_tables
```

**Run the chemistry solver:**
```bash
# Adaptive RK4 time integration
calima-run example_ic
calima-run example_ic --t_end_Myr 10 --output-dir results/

# Steady-state equilibrium (Newton–Krylov)
calima-run equilibrium_postshock_test --solver newton_krylov
```

**Run over a 2-D parameter grid:**
```bash
calima-grid \
    --config example_ic \
    --x-param T   --x-values 50 100 500 2000 8000 \
    --y-param nH  --y-values 0.1 1 10 100 1000 \
    --t-end-Myr 5 --solver rk4 \
    --output-npz grid_T_nH.npz
```

**Run the test suite:**
```bash
pytest
```

**Run script-style physics checks** (these are scripts, not pytest):
```bash
python diagnostics/dust_charge/check_equilibrium_charge.py
python diagnostics/PAH_gas_collisions/check_pah_export_rates_phi_grid.py
# etc. — everything under diagnostics/ is a standalone script
```

## Architecture

### Configuration-driven design

Everything flows from `src/pycalima/models/grain_size_distribution.json`. This JSON defines:
- Active grain/PAH bins (composition, rank, PAH flag)
- Distribution parameters (lognormal / power-law cutoff variants)
- Export sampling parameters used by batch exporters

`src/pycalima/models/grain_size_config.py` is the sole loader — it caches the parsed config and exposes accessors (`get_bins()`, `get_bin_by_rank()`, etc.) used by nearly every module. To use a non-default config, call `set_config_path(path)` before any other model import, or pass `--config` on the CLI.

### Two-layer structure

**Layer 1 — `src/pycalima/models/`**: Physics modules that compute rates, cross-sections, and properties. Organized by physical domain:
- `dust_radiation/`, `PAH_radiation/` — optical properties and radiative processes
- `dust_charge/`, `PAH_charge/` — charging, photoelectric heating
- `dust_gas_collisions/`, `PAH_gas_collisions/` — sputtering, collisional cooling
- `dust_collisions/`, `PAH_collisions/` — grain-grain/PAH-PAH collisional outcomes
- `dust_chemistry/` — surface chemistry (H₂ formation)
- `PAH_photophysics/` — PAH photodissociation and acetylene-loss tables
- `tools/` — radiation field definitions, Mie theory, SED readers, unit helpers
- `yields/` — dust yield table builders

Each subpackage contains:
- Core physics module(s)
- `export_*.py` — batch exporter that writes tables to `model_data/`
- `test_*.py` — standalone diagnostic/validation scripts (not pytest-collected)

**Layer 2 — `src/pycalima/solvers/`**: ODE solver that consumes the precomputed tables from `model_data/` and integrates dust/PAH mass evolution. Entry points are `run_chemistry.py` (single point) and `run_grid.py` (parallelized 2-D sweep). The solver mirrors the RAMSES-CALIMA Fortran modules.

### Data flow

```
BUNDLED, READ-ONLY  (inside the installed package)
  data/optical_props/   reference optical datasets
  data/external_data/   literature rates, cross-sections, yields
       │  _paths.get_optical_props_path() / get_external_data_path()
       ▼
  pycalima.models  ──calima-export──▶  model_data/   precomputed lookup tables
                                          │
WRITABLE  ($CALIMA_DATA, else ./model_data,  │  _paths.get_model_data_dir()
           else per-user data dir)           ▼
                                    pycalima.solvers  ──▶  results/
                                                            evolution .txt + .png
                                            _paths.get_results_dir()

NOT BUNDLED, registered in data/registry.toml
  the PAHdb archives   ──▶  calima-fetch-data import ...
```

### Solver internals

- `chemistry_state.py` — `DustChemistryState`, `DustBinParams`, `PAHBinParams` dataclasses
- `rhs.py` / `dust_rates.py` — assembles and evaluates rate kernels per physics flag
- `rk4.py` — adaptive Cash–Karp RK4 integrator
- `equilibrium.py` — Newton–Krylov and sparse-Newton steady-state solvers
- `dust_init.py` — parses JSON config, builds initial state and density arrays
- `table_io.py` — reads sputtering/rate tables from `model_data/`

### Solver config JSON keys to know

Each config in `src/pycalima/solvers/configs/` contains: `environment` (T, nH, G0), `dust_bins` / `pah_bins` (with initial conditions and bin parameters), `physics` (flag dict toggling individual processes on/off), `models` (selects variant implementations, e.g. attachment model, coalescence model), and `solver` (type + time-stepping parameters).

## Working Patterns

- **Adding a new physical process**: implement in the relevant `src/pycalima/models/` subpackage, add an export script, update `src/pycalima/models/export_all_grain_data.py`, then add a rate kernel in `src/pycalima/solvers/dust_rates.py` and wire its flag in `src/pycalima/solvers/rhs.py`.
- **Changing bin configuration**: edit `src/pycalima/models/grain_size_distribution.json`, then rerun `calima-export` to regenerate all tables.
- **Custom config without modifying the default**: pass `--config your_config.json` to any exporter, or call `set_config_path()` at the top of a script.
- `model_data/` is generated output — do not hand-edit it; regenerate from source.
- PNG plots are gitignored; they are produced as side-effects of export and test scripts.

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.