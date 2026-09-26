(api-pycalima)=
# `pycalima` — paths, data and provenance

The package's own support layer: where data is read from and written to, how
the bundled reference datasets are fetched and verified, how generated tables
are stamped with the revision that produced them, and the opt-in plotting
style.

These four modules carry a leading underscore because they are infrastructure
rather than physics, but they are a supported interface: `_paths` and
`_datasets` back the `calima-paths` and `calima-fetch-data` console scripts,
and the installation guide directs you at both.

## Modules

```{eval-rst}
.. autosummary::
   :toctree: generated
   :template: autosummary/module.rst

   pycalima._paths
   pycalima._datasets
   pycalima._provenance
   pycalima.plotting_style
```

