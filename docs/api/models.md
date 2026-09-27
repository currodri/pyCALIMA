(api-models)=
# `pycalima.models` — configuration and shared model layer

Everything under `models/` flows from a single grain-size configuration file.
`grain_size_config` is its sole loader: it caches the parsed configuration and
exposes the accessors (`get_bins()`, `get_bin_by_rank()`, ...) that nearly
every physics module uses. To use a non-default configuration, call
`set_config_path()` before importing any other model module, or pass
`--config` on the command line.

`export_all_grain_data` is the orchestrator behind `calima-export`; see
{doc}`/cli/calima-export` for its stages and their measured cost.

:::{note}
The subpackages below have empty `__init__.py` files. Import the leaf modules
directly — `from pycalima.models.dust_charge.dust_charging import ...` — rather
than expecting a package-level re-export.
:::

## Modules

```{eval-rst}
.. autosummary::
   :toctree: generated
   :template: autosummary/module.rst

   pycalima.models.constants
   pycalima.models.dust_model
   pycalima.models.grain_distributions
   pycalima.models.grain_size_config
   pycalima.models.export_all_grain_data
```

