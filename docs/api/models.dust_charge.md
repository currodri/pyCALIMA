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
   pycalima.models.dust_charge._charge_kernels
   pycalima.models.dust_charge.charging_isrf_tables
   pycalima.models.dust_charge.coulomb_focusing
```

