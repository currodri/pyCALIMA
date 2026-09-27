(extending)=
# Extending pyCALIMA

## Adding a new physical process

The pipeline has a fixed shape, so a new process touches five places in order:

1. Implement the physics in the relevant `models/` subpackage.
2. Add an `export_*.py` that writes its table to the generated-data directory.
3. Register the stage in `models/export_all_grain_data.py`'s `_STAGES` table,
   so `calima-export` runs it and `--stages` can select it.
4. Add a rate kernel in `solvers/dust_rates.py`.
5. Wire its physics flag in `solvers/rhs.py`, and add the flag to the solver
   configurations under `solvers/configs/`.

Steps 4 and 5 are what make the process visible to the solver; without them the
table is generated but never read.

## Changing the bin configuration

Edit `models/grain_size_distribution.json`, then regenerate:

```bash
calima-export
```

Every table under the generated-data directory depends on the bin
configuration, so a partial regeneration leaves the solver reading a mixture of
old and new physics. If you only changed the PAH bins, `--stages` lets you
regenerate just the PAH tables — but be sure that is genuinely all that
changed.

## Using a custom configuration without touching the default

Pass `--config` to any exporter, or call `set_config_path()` before importing
any other model module:

```python
from pycalima.models.grain_size_config import set_config_path
set_config_path("my_config.json")   # must precede other model imports
```

The ordering matters: `grain_size_config` caches the parsed configuration, and
modules capture bin metadata at import time.

## Conventions worth keeping

- The generated-data directory is output. Do not hand-edit it; regenerate.
- `solvers/` must not import from `models/`. `solvers/dust_init.py` imports
  `pycalima._paths` — a leaf module that imports neither — precisely to
  preserve that boundary.
- Physics modules must not configure matplotlib at import time. Style is
  opt-in through {func}`pycalima.plotting_style.use_calima_style`; a test
  enforces this.
- Nothing may write to disk at import time. A test enforces that too.
