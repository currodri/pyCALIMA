# pyCALIMA

:::{container} calima-hero
**Dust and PAH microphysics for galaxy simulations.**

Offline grain physics, a RAMSES-matched mass-exchange network, and the tools
to compare the two with your simulations.
:::

pyCALIMA models the life cycle of interstellar dust grains and polycyclic
aromatic hydrocarbons: how they charge, heat the gas, grow by accretion and
coagulation, and are destroyed by sputtering, shattering, sublimation and
photodissociation. It computes the microphysics offline into lookup tables,
then integrates the resulting mass-exchange network — mirroring the Fortran
implementation that runs inside RAMSES.

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} {octicon}`rocket;1.2em;sd-mr-1` Getting started
:link: getting-started/index
:link-type: doc

Install pyCALIMA, check the installation, and run your first calculation in
six commands.
:::

:::{grid-item-card} {octicon}`beaker;1.2em;sd-mr-1` Physics
:link: physics/index
:link-type: doc

What each process computes — accretion, sputtering, coagulation, shattering,
charging, PAH photophysics — with the equations as implemented.
:::

:::{grid-item-card} {octicon}`book;1.2em;sd-mr-1` User guide
:link: guide/index
:link-type: doc

Configuration, workflows, the five solvers, and post-processing RAMSES
outputs.
:::

:::{grid-item-card} {octicon}`mortar-board;1.2em;sd-mr-1` Tutorials
:link: tutorials/index
:link-type: doc

Worked notebooks, executed on every documentation build so they cannot go
stale.
:::

:::{grid-item-card} {octicon}`terminal;1.2em;sd-mr-1` Command line
:link: cli/index
:link-type: doc

`calima-export`, `calima-run`, `calima-grid`, `calima-paths` and
`calima-fetch-data`.
:::

:::{grid-item-card} {octicon}`code;1.2em;sd-mr-1` API reference
:link: api/index
:link-type: doc

Every public module, generated from the docstrings, with links to the source.
:::
::::

## Quick start

```bash
pip install "git+https://github.com/currodri/pyCALIMA"
calima-paths                     # where pyCALIMA reads and writes
calima-fetch-data verify         # did the reference data ship?
calima-export                    # generate the lookup tables
calima-run example_ic --t_end_Myr 0.01 --output-dir /tmp/calima-check
```

{doc}`getting-started/quickstart` explains each step.

## Citing pyCALIMA

If you use pyCALIMA, please cite the
{ads}`CALIMA model paper <2026arXiv260221790R>`.

```{toctree}
:maxdepth: 2
:caption: Getting started
:hidden:

getting-started/index
```

```{toctree}
:maxdepth: 2
:caption: Documentation
:hidden:

physics/index
guide/index
tutorials/index
cli/index
api/index
```

```{toctree}
:maxdepth: 2
:caption: Reference
:hidden:

reference/index
development/index
```
