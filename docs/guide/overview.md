(overview)=
# What pyCALIMA contains


- Physics modules in `src/pycalima/models/` for dust and PAH processes.
- Reference input tables bundled inside the package, under
  `src/pycalima/data/{external_data,optical_props}/`.
- Generated lookup tables written to a writable data directory (see
  {doc}`/getting-started/data-locations`).
- Diagnostic and simulation-setup scripts in `diagnostics/`.
- Tutorial notebooks in `notebooks/`.

pyCALIMA is an installable Python package. Workflows run either through console
scripts (`calima-export`, `calima-run`, `calima-grid`) or as
`python -m pycalima.models.<module>`, from any directory.
