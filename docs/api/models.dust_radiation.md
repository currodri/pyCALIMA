(api-models-dust-radiation)=
# `pycalima.models.dust_radiation`

Optical properties and radiative processes: interpolation of dust efficiencies
and cross-sections, extinction utilities, radiative torque disruption (RATD),
thermal sublimation and dust emission. `material_optics` computes Mie
absorption and Planck-mean efficiencies from tabulated optical constants for
the materials of `data/external_data/material_optics.toml` (the ten species of
Pop III supernova dust), with a far-infrared extrapolation that keeps k
physical and the small-particle expansion where Mie cancels.

## Modules

```{eval-rst}
.. autosummary::
   :toctree: generated
   :template: autosummary/module.rst

   pycalima.models.dust_radiation.dust_oppacity
   pycalima.models.dust_radiation.themis_optics
   pycalima.models.dust_radiation.jones2012_optics
   pycalima.models.dust_radiation.carbon_heat_capacity
   pycalima.models.dust_radiation.dust_emission
   pycalima.models.dust_radiation.dust_ratd
   pycalima.models.dust_radiation.dust_sublimation
   pycalima.models.dust_radiation.agn_hot_dust
   pycalima.models.dust_radiation.material_optics
   pycalima.models.dust_radiation.sublimation_kinetics
```

