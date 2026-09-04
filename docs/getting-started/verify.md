(verifying)=
# Verifying the installation

Three checks, in increasing order of thoroughness, plus an end-to-end smoke test.


Three levels, cheapest first.

## 1. Is it installed, and where does its data live?

```bash
python -c "import pycalima; print(pycalima.__version__)"
calima-paths
```

`calima-paths` prints every resolved location. Read-only reference data should
sit **inside** the installed package, and `model_data` / `results` **outside**
it:

```
pycalima package     .../site-packages/pycalima
  external_data      .../site-packages/pycalima/data/external_data
  optical_props      .../site-packages/pycalima/data/optical_props
model_data           ~/Library/Application Support/calima/model_data
results              ~/Library/Application Support/calima/results
```

If `model_data` points anywhere inside `site-packages`, something is wrong.
Run it from a directory that is *not* the repository to check that nothing
depends on the current working directory.

## 2. Did the reference data ship?

```bash
calima-fetch-data list
calima-fetch-data verify
```

Every `bundled` dataset must report `present`. The two PAHdb archives report
`MISSING` by design — they are ~575 MB and are obtained separately, so
`calima-fetch-data verify` exits non-zero until they are registered:

```bash
calima-fetch-data import pahdb-theoretical-v4-00 /path/to/pahdb-...v4.00.xml
```

All five console scripts should respond:

```bash
for c in calima-paths calima-fetch-data calima-export calima-run calima-grid; do
    $c --help > /dev/null && echo "ok $c"
done
```

## 3. Run the test suite

Requires a clone, since tests are not shipped inside the wheel:

```bash
pip install -e ".[dev]"
pytest
```

Everything should pass, with two **expected** strict xfails that record known
physics bugs, and a number of skips that depends on whether generated tables
are available.

The skips are the tests that read generated tables. They skip rather than
fail, since `model_data/` is gitignored and produced by `calima-export`. To
include them, either run `calima-export` first or point at an existing tree:

```bash
CALIMA_MODEL_DATA=/path/to/model_data pytest
```

The remaining single skip is `cagliari-pah`, whose fetch test is skipped
because a bundled copy already satisfies it.

The two xfails are known physics bugs recorded deliberately — see
{doc}`/development/testing`.

## End-to-end smoke test

The shortest real calculation, from any directory:

```bash
calima-run example_ic --t_end_Myr 0.01 --output-dir /tmp/calima-check
```

It resolves the bundled config by name, reads tables from the resolved
`model_data`, prints an element-by-element mass-conservation table ending in
`✓ OK`, and writes `example_ic_evolution.{txt,png}`. This needs generated
tables; if they are missing you get an error naming `calima-export`.
