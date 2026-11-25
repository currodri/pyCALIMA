(api-models-dust-charge)=
# `pycalima.models.dust_charge`

Grain charging and photoelectric heating: equilibrium charge distributions,
Coulomb enhancement of gas-grain collision rates, and dust-assisted ion
recombination.

:::{note}
The subpackage's two export stages, `dust_charging_rtgroups` and
`dust_charging_isrf` (built from the tables of the first), scale per dust bin.
See {doc}`/cli/calima-export` for what they cost.
:::

The WDB06 tables (`wdb06_yields`, `rt_group_charging`, `charging_isrf_tables`)
take three materials: `graphite`, `silicate` and `cm20`, THEMIS amorphous
carbon charged as DustEM v4.3 charges it (Kimura 2016 band yield, DustEM's
aromatic correction and size-dependent band gap; WDB06's channels above
20 eV). `cm20` tables are format 4: format 3 plus the electron sticking
coefficient. See {doc}`/physics/charging-cm20` for the model and its check
against DustEM.

## Modules

```{eval-rst}
.. autosummary::
   :toctree: generated
   :template: autosummary/module.rst

   pycalima.models.dust_charge.dust_charging
   pycalima.models.dust_charge.dust_photoelectric_heating
   pycalima.models.dust_charge.Coulomb_enhancement
   pycalima.models.dust_charge.IM19_charging
   pycalima.models.dust_charge.shared_physics
   pycalima.models.dust_charge.wdb06_yields
   pycalima.models.dust_charge.wdb06_atomic
   pycalima.models.dust_charge.rt_group_charging
   pycalima.models.dust_charge.charging_isrf_tables
   pycalima.models.dust_charge.coulomb_focusing
```

