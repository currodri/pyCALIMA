(cli-reference)=
# Command-line reference

Installing pyCALIMA puts five commands on your `PATH`. Each page below is
generated from the command's own argument parser, so the options here cannot
drift from the code.

| Command | Purpose |
|---|---|
| {doc}`calima-paths` | Show where pyCALIMA reads and writes data |
| {doc}`calima-fetch-data` | Inspect, download and register reference datasets |
| {doc}`calima-export` | Regenerate the precomputed lookup tables |
| {doc}`calima-run` | Integrate dust and PAH chemistry for one environment |
| {doc}`calima-grid` | Sweep a 2-D parameter grid |

Every one is also importable, so anything you can do from the shell you can do
from Python. `calima-export` is `pycalima.models.export_all_grain_data.main()`,
`calima-run` is {func}`pycalima.solvers.run_chemistry.run_chemistry`, and so
on.

```{toctree}
:hidden:

calima-paths
calima-fetch-data
calima-export
calima-run
calima-grid
```
