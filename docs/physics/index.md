(physics)=
# Physics

pyCALIMA follows the mass of interstellar dust and PAHs through a set of
**bins**, each with a composition (carbonaceous or silicate) and a
representative grain size, plus the gas-phase abundance of every element the
grains are made of. Mass moves between the gas and the bins, and between bins,
through the processes on these pages.

The work is split in two, mirroring the RAMSES-CALIMA Fortran:

- **Offline microphysics** (`pycalima.models`): optical properties, grain
  charge, heating and cooling, sputtering yields, PAH dissociation. These are
  expensive, so they are computed once by `calima-export` and written to
  lookup tables in `model_data/`.
- **The mass-exchange network** (`pycalima.solvers`): cheap rate kernels,
  evaluated every step, that read those tables and move mass between the gas
  and the bins. {doc}`/guide/solvers` describes how the network is integrated.

```{mermaid}
%%{init: {"themeVariables": {"fontSize": "18px"}, "flowchart": {"nodeSpacing": 40, "rankSpacing": 55}}}%%
flowchart TB
    gas["Gas-phase metals<br/>C, O, Mg, Si, Fe, ..."]
    small["Small grains<br/>(per composition)"]
    large["Large grains<br/>(per composition)"]
    pah["PAHs"]
    gas -- accretion --> small
    gas -- accretion --> large
    small -- "sputtering, sublimation" --> gas
    large -- "sputtering, sublimation" --> gas
    small -- coagulation --> large
    large -- shattering --> small
    gas -- "PAH accretion" --> pah
    pah -- "photolysis, sputtering" --> gas
    pah -- freezing --> small
```

## Processes in the solver

Each process is switched on by a flag in the `physics` block of a solver
configuration, and some have variants selected in the `models` block. The
variant used when the key is absent is marked *(default)*.

| Process | `physics` flag | `models` key: variants | Solver kernel |
|---|---|---|---|
| {doc}`Grain growth by accretion <dust-growth-destruction>` | `dust_accretion` | — | `accretion_rate` |
| {doc}`Thermal sputtering <dust-growth-destruction>` | `dust_sputtering` | `dust_sputtering_model`: `kirchschlager` *(default, tables)*, `nozawa2006`, `nozawa2006_ramses` | `thermal_sputtering_rate`, `thermal_sputtering_rate_nozawa`, `thermal_sputtering_rate_nozawa_ramses` |
| {doc}`Thermal sublimation <dust-growth-destruction>` | `dust_sublimation` | — | `sublimation_rate` |
| {doc}`Coagulation <grain-collisions>` | `dust_coagulation` | `coagulation_model`: `Aoyama2017` *(default)*, `Dubois2024`, `turbulent`, `turbulent_all` | `coagulation_rate`, `dubois_coagulation_rate`, `turbulent_coagulation_rate`, `turbulent_all_coagulation_rate` |
| {doc}`Shattering <grain-collisions>` | `dust_shattering` | `shattering_model`: `turbulent` *(default)*, `Dubois2024`, `turbulent_all` | `turbulent_shattering_rate`, `dubois_shattering_rate`, `turbulent_all_shattering_rate` |
| {doc}`PAH accretion <pah>` | `pah_accretion` | — | `pah_accretion_rate` |
| {doc}`PAH photolysis <pah>` | `pah_photolysis` | `photolysis_model` | `pah_photolysis_rate` |
| {doc}`PAH sputtering <pah>` | `pah_sputtering` | `pah_sputtering_model` | `pah_sputtering_rate` |
| {doc}`PAH coalescence <pah>` | `pah_coalescence` | `coalescence_model`: `Totton2012` *(default)*, `Tielens2021` | `totton2012_pah_coalescence_rate`, `tielens2021_pah_coalescence_rate` |
| {doc}`PAH cluster evaporation <pah>` | `pah_cluster_evaporation` | `cluster_evaporation_model` | `pah_cluster_evaporation_rate` |
| {doc}`PAH freezing onto grains <pah>` | `pah_freezing` | — | `pah_freezing_rate` |

The kernels live in {mod}`pycalima.solvers.dust_rates`; the list is assembled
by `build_process_list` in {mod}`pycalima.solvers.rhs`. Collisional processes
need at least two bins (a small and a large one per composition); the PAH
processes need PAH bins, and freezing also needs dust bins.

:::{note}
The three `ramses_*` solver configurations shipped with pyCALIMA reproduce the
RAMSES-CALIMA runs: accretion, `nozawa2006_ramses` sputtering and the
`Dubois2024` coagulation and shattering, with no PAH bins.
:::

## Offline microphysics

The tables the solver reads, and many that feed RAMSES directly, come from
the `models/` modules. {doc}`charging-heating` and {doc}`radiation` describe
them; {doc}`/cli/calima-export` lists every export stage and what it writes.

```{toctree}
:maxdepth: 2

dust-growth-destruction
grain-collisions
charging-heating
pah
radiation
```
