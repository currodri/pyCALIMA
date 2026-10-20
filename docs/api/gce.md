(api-gce)=
# `pycalima.gce` — galactic chemical evolution

Single-stellar-population (SSP) release tables: for one solar mass formed at
metallicity Z, the cumulative mass, elements, condensed dust and supernova
counts returned to the gas by AGB winds, core-collapse and Type Ia
supernovae as a function of age. Element lists match the dust solver's
`ELEMENT_NAMES`, so the tables feed a galaxy model whose gas-phase state is
the solver's state.

Generate the tables with `calima-export --stages ssp_yield_tables`, or
`python -m pycalima.gce.export_ssp_tables`.

## Modules

```{eval-rst}
.. autosummary::
   :toctree: generated
   :template: autosummary/module.rst

   pycalima.gce.imf
   pycalima.gce.lifetimes
   pycalima.gce.stellar_yields
   pycalima.gce.snia
   pycalima.gce.stellar_dust
   pycalima.gce.ssp
```
