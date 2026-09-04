# pyCALIMA

<p align="center">
	<img src="https://raw.githubusercontent.com/currodri/pyCALIMA/main/assets/CALIMA_logo1.png" alt="pyCALIMA logo" width="420"/>
</p>

pyCALIMA is a research codebase for modeling dust and PAH microphysics used in RAMSES simulations.
It combines grain-size distributions, radiation, charging, sputtering, collisional cooling, and
PAH-specific photophysics into reusable Python modules plus export scripts that generate simulation-ready tables.

## Documentation

**Full documentation: https://currodri.github.io/pyCALIMA/**

| | |
|---|---|
| [Installation](https://currodri.github.io/pyCALIMA/getting-started/install.html) | Install, extras, and verifying it worked |
| [Quickstart](https://currodri.github.io/pyCALIMA/getting-started/quickstart.html) | Six commands, start to finish |
| [Data locations](https://currodri.github.io/pyCALIMA/getting-started/data-locations.html) | How paths resolve, and the environment variables that override them |
| [User guide](https://currodri.github.io/pyCALIMA/guide/index.html) | Package layout, configuration, workflows |
| [Solvers](https://currodri.github.io/pyCALIMA/guide/solvers.html#solver-types) | The five solvers and when to prefer each |
| [Tutorials](https://currodri.github.io/pyCALIMA/tutorials/index.html) | Worked notebooks, executed on every docs build |
| [CLI reference](https://currodri.github.io/pyCALIMA/cli/index.html) | The five console scripts |
| [API reference](https://currodri.github.io/pyCALIMA/api/index.html) | Generated from the docstrings |

## Related References

- RAMSES repository: https://github.com/ramses-organisation/ramses
- CALIMA model paper: https://ui.adsabs.harvard.edu/abs/2026arXiv260221790R/abstract

## Installation

Requires Python 3.10 or newer.

```bash
# from a clone, for development
pip install -e .

# or directly from GitHub
pip install "git+https://github.com/currodri/pyCALIMA.git"

# with every optional extra
pip install -e ".[all]"
```

Optional extras: `accel` (numba), `sim` (yt, for the RAMSES readers), `pahdb`
(the Ames PAH database suite), `plots` (extra colormaps), `profile` (memory
reporting), `docs`, and `dev`. See the
[installation guide](https://currodri.github.io/pyCALIMA/getting-started/install.html)
for what each one unlocks.

## Quickstart

```bash
pip install -e ".[all]"
calima-paths                     # where pyCALIMA reads and writes
calima-fetch-data verify         # did the reference data ship?
calima-export                    # generate the lookup tables
calima-run example_ic --t_end_Myr 0.01 --output-dir /tmp/calima-check
```

The last command is the end-to-end check: it exercises configuration loading,
table lookup, the ODE integration and the output writer, and fails loudly if a
generated table is missing or malformed. It writes an `_evolution.txt`
trajectory and an `_evolution.png` summary.

Two of the twelve export stages account for about 92% of `calima-export`'s
runtime. [The CLI reference](https://currodri.github.io/pyCALIMA/cli/calima-export.html)
breaks down the cost and shows how to regenerate only part of it.

## What's in the box

- **`models/`** — physics modules for dust and PAH processes, organised by
  physical domain, plus the batch exporters that turn them into lookup tables.
- **`solvers/`** — the dust and PAH chemistry ODE solver, mirroring the
  RAMSES-CALIMA Fortran modules. Five integrators.
- **`galaxysam/`** — a semi-analytic galactic chemical evolution model.
- **`data/`** — bundled reference datasets (optical properties, literature
  rates and yields), with a registry recording the provenance of each.
- **Five console scripts** — `calima-paths`, `calima-fetch-data`,
  `calima-export`, `calima-run`, `calima-grid`.

Workflows run either through the console scripts or as
`python -m pycalima.models.<module>`, from any directory.

## Where data lives

Bundled reference data is read-only inside the installed package. Generated
tables and run output go to a writable location resolved in this order:
`$CALIMA_MODEL_DATA` / `$CALIMA_RESULTS`, then `$CALIMA_DATA`, then
`./model_data` if it exists, then a per-user data directory.

Run `calima-paths` to see exactly what is in effect. Full rules:
[Data locations](https://currodri.github.io/pyCALIMA/getting-started/data-locations.html).

## Notebooks

All notebooks live in `notebooks/` and are committed with outputs cleared. Once
`pycalima` is installed they run from any directory.

`calima_tutorial.ipynb` and `calima_dust_pah_processes.ipynb` need generated
tables, and are executed on every documentation build — so the
[rendered versions](https://currodri.github.io/pyCALIMA/tutorials/index.html)
always reflect the current code. `ramses_equilibrium_tutorial.ipynb` and
`CALIMA_model_explorer.ipynb` additionally need RAMSES snapshots that **are not
distributed with pyCALIMA**; see
[Post-processing RAMSES outputs](https://currodri.github.io/pyCALIMA/guide/post-processing.html).

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Some tests need generated tables and skip without them, which is by design —
run `calima-export` (or set `$CALIMA_MODEL_DATA`) to enable them. See
[Testing and validation](https://currodri.github.io/pyCALIMA/development/testing.html).

## Current status

The repository is actively used as a scientific modeling workspace, and mixes
reusable modules with experiment scripts. If you are new to it, start by
reading, in order:

1. `src/pycalima/models/grain_size_distribution.json` — the data model
2. `src/pycalima/models/grain_size_config.py` — the configuration API
3. `src/pycalima/models/export_all_grain_data.py` — the export pipeline
4. `src/pycalima/_paths.py` — where everything is read from and written to

## Citing

Please cite the [CALIMA model paper](https://ui.adsabs.harvard.edu/abs/2026arXiv260221790R/abstract).

## License

MIT. See [LICENSE](LICENSE).
