(installation)=
# Installation

pyCALIMA requires Python 3.10 or newer.


Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# from a clone, for development
pip install -e .

# or directly from GitHub
pip install "git+https://github.com/currodri/pyCALIMA"
```

Optional extras:

| Extra | Pulls in | Needed for |
|---|---|---|
| `sim` | `yt` | reading RAMSES outputs (`models.tools.eq_analysis`) |
| `accel` | `numba` | JIT acceleration of the charging/sputtering kernels |
| `pahdb` | `amespahdbpythonsuite` | constructing `AmesPAHdb` objects yourself |
| `plots` | `cmasher` | extra colormaps |
| `profile` | `psutil` | memory reporting in the profiler |
| `all` | all of the above | |

```bash
pip install -e ".[all]"
```

Two dependencies cannot be declared as extras, because PEP 508 direct
references are rejected in distribution metadata. Both are optional:

- **UCLCHEM** (`notebooks/ice_formation_study.ipynb`,
  `notebooks/uclchem_multiice_parallel_notebook.ipynb`) — Fortran-backed, build
  from source.
- A pinned development build of `amespahdbpythonsuite`, if you need to
  reproduce the shipped PAH tables exactly.

See `requirements-dev.txt`.

To also get the test suite and build tooling:

```bash
pip install -e ".[dev]"
```
