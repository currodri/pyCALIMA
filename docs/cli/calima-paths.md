# `calima-paths`

Prints every location pyCALIMA resolves — bundled read-only data, the writable
root, generated tables, results — together with the environment variables in
effect and the provenance of the installed package. Run it first when
something cannot be found; see {doc}`/getting-started/data-locations` for the
resolution rules it reports.

```{eval-rst}
.. argparse::
   :module: pycalima._paths
   :func: _build_parser
   :prog: calima-paths
```
