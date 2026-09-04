(tutorials)=
# Tutorials


All notebooks live in `notebooks/` and are committed with outputs cleared. Once
`pycalima` is installed they run from any directory — no `sys.path` juggling
and no need to be in the repository root.

| Notebook | Needs | Extra |
|---|---|---|
| `calima_tutorial.ipynb` | generated tables | — |
| `calima_dust_pah_processes.ipynb` | generated tables | — |
| `ramses_equilibrium_tutorial.ipynb` | **your own RAMSES outputs** | `sim` |
| `CALIMA_model_explorer.ipynb` | **your own RAMSES outputs** | `sim` |
| `pusk1983_fittings.ipynb` | — | — |
| `exploring_PAH_accretion.ipynb` | — | — |
| `ice_formation_study.ipynb`, `uclchem_multiice_parallel_notebook.ipynb` | UCLCHEM (build from source) | — |
| `simple_ice_model.ipynb` | — | — |

The first two need generated tables, so run `calima-export` first (or set
`$CALIMA_MODEL_DATA`). The two RAMSES notebooks additionally need simulation
snapshots that **are not distributed with pyCALIMA** — see
{doc}`/guide/post-processing`.

## Rendered here

The two notebooks below are executed every time the documentation is built, so
what you see is what the current code produces.

```{toctree}
:maxdepth: 1

calima_tutorial
calima_dust_pah_processes
rendered/index
```

## `calima_tutorial.ipynb` — Solver Workflow

An end-to-end walkthrough of the full CALIMA solver workflow:

| Section | Description |
|---|---|
| 1. Imports | Standard library and CALIMA module imports |
| 2. JSON configuration | Anatomy of every key field in the configuration file |
| 3. Process flags and initial conditions | Toggling physics on/off; inspecting initial density arrays |
| 4. Single RK4 run | Adaptive RK4 time evolution for one (T, nH) point with plot output |
| 5. Equilibrium solver | Newton–Krylov steady-state run; comparison with RK4 final state |
| 6. T–nH grid setup | Defining a logarithmic 2-D parameter sweep |
| 7. Grid run (parallel) | Running all (T, nH) pairs using joblib multi-core parallelism |
| 8. Grid plotting | Heatmaps of DTM ratio, PAH abundance, and grain-size fractions |

## `calima_dust_pah_processes.ipynb` — Dust and PAH Processes Deep Dive

Focuses on the individual physical processes available in the solver,
using DustBin_03 (small silicate, 0.005 µm) and PAHbin_01 (coronene, $N_C = 54$)
as concrete examples throughout:

| Section | Description |
|---|---|
| 1. Imports and setup | Notebook environment and CALIMA path configuration |
| 2. Anatomy of a dust bin | All `DustBinParams` fields and their physical meaning for DustBin_03 |
| 3. Anatomy of a PAH bin | All `PAHBinParams` fields and their physical meaning for PAHbin_01 |
| 4. The `physics` block | Flag-to-function mapping for every rate module |
| 5. The `models` block | Config-selectable process variants (attachment model, coalescence model, etc.) |
| 6. Inspecting rate tables | Visualising pre-computed rate tables loaded at solver initialisation |
| — 6a–6c | Dust and PAH thermal sputtering tables |
| — 6d | Dust grain charge scan: ⟨Z⟩ and σ_Z vs γ from precomputed WD01 data (10 000 points) |
| — 6e | Dust photoelectric heating rate, recombination cooling, and PEH efficiency tables |
| — 6f | PAH charge fractions and PEH efficiency: Berne (2022) vs Tielens (2021) |
| 7. Isolated process runs | Enabling one physics flag at a time to measure each process contribution |
| 8. Export functions | `save_chemistry_txt` and `save_chemistry_plot` API reference |
| 9. Visualising results | Publication-quality plots assembled from exported data |
