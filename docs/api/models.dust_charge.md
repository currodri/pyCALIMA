(api-models-dust-charge)=
# `pycalima.models.dust_charge`

Grain charging and photoelectric heating: equilibrium charge distributions,
Coulomb enhancement of gas-grain collision rates, and dust-assisted ion
recombination.

:::{warning}
`dust_ion_recombination` is the dominant stage of a full export — 15.4 minutes
at 20 dust bins. It scales per dust bin. See {doc}`/cli/calima-export`.
:::

## Modules

```{eval-rst}
.. autosummary::
   :toctree: generated
   :template: autosummary/module.rst

   pycalima.models.dust_charge.dust_charging
   pycalima.models.dust_charge.dust_photoelectric_heating
   pycalima.models.dust_charge.dust_ion_recombination
   pycalima.models.dust_charge.Coulomb_enhancement
   pycalima.models.dust_charge.IM19_charging
   pycalima.models.dust_charge.shared_physics
   pycalima.models.dust_charge.wdb06_yields
   pycalima.models.dust_charge.wdb06_atomic
   pycalima.models.dust_charge.rt_group_charging
   pycalima.models.dust_charge.charging_psi_tables
   pycalima.models.dust_charge.charging_isrf_tables
   pycalima.models.dust_charge.coulomb_focusing
```

