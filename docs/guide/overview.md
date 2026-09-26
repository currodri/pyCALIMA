(overview)=
# Overview

pyCALIMA is the Python counterpart of the dust and PAH model that runs inside
RAMSES (CALIMA). It exists to do two things the simulation cannot do on the
fly: compute the expensive grain microphysics once, carefully, and let you run
the same mass-exchange network cell by cell outside the simulation, to test,
calibrate or post-process it.

## Two layers

**Offline microphysics — `pycalima.models`.** Grain optical properties,
charge distributions, photoelectric heating, gas-grain cooling, sputtering
yields and PAH dissociation rates are computed from first principles and
laboratory or literature data, for every grain bin in the configuration.
`calima-export` runs all of it and writes lookup tables to `model_data/`. The
{doc}`/physics/index` pages describe what each module computes.

**The mass-exchange network — `pycalima.solvers`.** A gas cell is described
by its temperature, density, radiation field, turbulence and the mass in each
grain bin and gas-phase element. Rate kernels move mass between them —
accretion, sputtering, coagulation, shattering, and the PAH processes —
reading the tables from the first layer. Five integrators solve the network:
explicit and quasi-implicit time integration, and steady-state solvers
({doc}`solvers`). The network mirrors the RAMSES-CALIMA Fortran
({doc}`ramses-coupling`), so a pyCALIMA run is a like-for-like comparison with
what the simulation does in one cell.

## Configuration

Two JSON files drive everything:

- the **grain-size configuration** (`models/grain_size_distribution.json`)
  defines the bins — composition, size distribution, which bins are PAHs — and
  so what `calima-export` computes;
- a **solver configuration** (`solvers/configs/*.json`) defines one gas
  environment, its initial dust and gas content, and which processes and
  model variants are active.

{doc}`configuration` documents both.

## Typical workflows

| You want to | Start with |
|---|---|
| install and check pyCALIMA | {doc}`/getting-started/quickstart` |
| regenerate the lookup tables | {doc}`/cli/calima-export` |
| evolve dust in one gas cell | {doc}`/cli/calima-run` |
| map a process over temperature and density | {doc}`/cli/calima-grid` |
| compare against RAMSES snapshots | {doc}`post-processing` |
| add a physical process | {doc}`extending` |

Everything runs from any directory once pyCALIMA is installed;
{doc}`/getting-started/data-locations` explains where it reads and writes.
