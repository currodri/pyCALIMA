# `calima-fetch-data`

Reference datasets come in three kinds, recorded in the package's dataset
registry: `bundled` ships inside the wheel, `fetch` is downloaded on demand,
and `manual` must be obtained by hand because the source issues per-download
filenames or its licence does not permit redistribution. See
{doc}`/reference/data-provenance` for the full registry and its citations.

```{eval-rst}
.. argparse::
   :module: pycalima._datasets
   :func: _build_parser
   :prog: calima-fetch-data
```
