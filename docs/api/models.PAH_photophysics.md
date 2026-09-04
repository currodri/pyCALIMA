(api-models-pah-photophysics)=
# `pycalima.models.PAH_photophysics`

PAH photophysics: absorption, internal energy distributions, photodissociation
via hydrogen and acetylene loss, hydrogenation state, and the reaction network
that ties them together.

The canonical description of the dissociation treatment — five methods, A
through E, with their assumptions and validity ranges — is the module
docstring of {mod}`~pycalima.models.PAH_photophysics.pah_dissociation`.

## Modules

```{eval-rst}
.. autosummary::
   :toctree: generated
   :template: autosummary/module.rst

   pycalima.models.PAH_photophysics.PAH_photophysics
   pycalima.models.PAH_photophysics.PAH_photochemistry
   pycalima.models.PAH_photophysics.pah_radiation
   pycalima.models.PAH_photophysics.pah_temperature
   pycalima.models.PAH_photophysics.pah_dissociation
   pycalima.models.PAH_photophysics.pah_h_state
   pycalima.models.PAH_photophysics.pah_hydrogen_chemistry
   pycalima.models.PAH_photophysics.pah_network_solver
   pycalima.models.PAH_photophysics.pah_mol_data
   pycalima.models.PAH_photophysics.pah_db_lookup
   pycalima.models.PAH_photophysics.pah_charge_utils
```

## Convenience re-exports

`__init__.py` re-exports 59 names from 8 leaf modules so that short imports
work. It declares no `__all__`, so those names are not auto-discoverable and
are documented on their own modules' pages rather than here.

:::{note}
Adding a module docstring and an `__all__` to this subpackage's `__init__.py`
would let this page be generated instead of hand-maintained. It is the single
highest-leverage docstring edit left in the repository: 59 names, one file.
:::

