(api-models-dust-charge)=
# `pycalima.models.dust_charge`

Grain charging and photoelectric heating: equilibrium charge distributions,
Coulomb enhancement of gas-grain collision rates, and dust-assisted ion
recombination.

:::{warning}
`dust_photoelectric_heating` and `dust_ion_recombination` are the two dominant
stages of a full export — 15.9 and 15.4 minutes respectively at 20 dust bins,
together about 90% of the total. Both scale per dust bin. See
{doc}`/cli/calima-export`.
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
```

