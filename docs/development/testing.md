(testing-and-validation)=
# Testing and validation


```bash
pip install -e ".[dev]"
pytest
pytest tests/test_installation.py   # packaging and metadata only
pytest -q --durations=10            # find the slow tests
```

Assertions are invariants — round-trips, normalisation, monotonicity, bounds,
mass conservation — rather than recorded output, so they catch a broken data
path or a sign error without pinning numerical results.

The table below says what each module covers. Counts are deliberately omitted:
they were stated in four places at once and had already drifted apart. To see
the current number:

```bash
pytest --collect-only -q | tail -1
```

| Module | Covers |
|---|---|
| `test_installation.py` | distribution metadata, dependencies, extras, console scripts, package data, wheel/sdist audit |
| `test_no_import_side_effects.py` | every module imports from an empty CWD writing nothing; no import-time `mkdir`; no global LaTeX poisoning |
| `test_datasets.py` | registry shape, bundled/fetch/manual semantics, `calima-fetch-data` |
| `test_models_physics.py` | one section per `models/` subpackage: shielding, charging, radiation, collisions, chemistry, tools, yields |
| `test_models_core.py` | unit conversions, radiation fields, all nine size-distribution classes |
| `test_solvers.py` | the `models/`↔`solvers/` boundary, all eight configs, RHS mass conservation, table I/O, end-to-end runs, and regressions for the four rate tables that once failed to load silently |
| `test_models_config.py` | configuration parsing, bin metadata, caching, `model_name` handling |
| `test_galaxysam.py` | IMF slopes, abundance tables, bundled yield tables |
| `test_paths.py` | bundled data inside the package, writable paths outside it |
| `test_docs.py` | every library module is in the API reference, no forbidden content, no stale anchors |

Two tests are **strict xfails**, recording known physics bugs rather than
silently changing scientific output. They will fail if someone fixes the
underlying issue, which is the signal to remove the marker:

- `PowerLaw_ExpCutoff_Distribution.averaged_over_number` divides by the
  power-law weight where every sibling class multiplies, so averaging a
  constant returns ~1e13 instead of that constant. This is live: the class is
  instantiated four times in `models/dust_radiation/dust_oppacity.py` for the
  Gao (2020) and Nozawa (2007) distributions.
- `ionisation_yield_Jochims1996` returns `(E-IP)/9.2` unclamped, so a
  sub-threshold photon energy yields a negative probability.

Other validation entry points:

- Quick configuration check: `python diagnostics/check_config.py`
- Stromgren setup utility: `python diagnostics/stromgren_test/stromgren_setup.py --help`
- `diagnostics/` holds script-style physics checks (`check_*.py`) that produce
  figures and are run directly. They were previously named `test_*.py` and
  lived inside the package, despite containing no pytest tests — so
  `pytest --pyargs pycalima` deliberately collects nothing.
