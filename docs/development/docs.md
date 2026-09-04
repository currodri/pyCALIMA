(building-the-docs)=
# Building the documentation

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
open docs/_build/html/index.html
```

## The two executed notebooks

`docs/tutorials/` symlinks two notebooks from `notebooks/`, and they are
**executed at build time** so that the rendered output cannot go stale. That
needs the generated tables, so set `$CALIMA_MODEL_DATA` (or run
`calima-export`) before building. Execution results are cached by notebook
content in `docs/_build/.jupyter_cache`, so an unchanged notebook is not
re-run.

The two RAMSES post-processing notebooks are *not* executed: they need
multi-GB simulation snapshots that only you have. They live under
`docs/tutorials/rendered/` with their outputs committed, and
`nb_execution_excludepatterns` covers that whole directory rather than naming
files, so a typo cannot fall through to trying to execute them.

:::{warning}
Do not set `MPLBACKEND=Agg` in the environment or in `conf.py`. myst-nb
executes notebooks in a subprocess kernel that inherits the environment, and
with that variable set the inline backend's automatic figure display never
runs — every figure silently disappears, with no warning and no error.
`conf.py` calls `matplotlib.use("Agg")` instead, which affects only the Sphinx
process.
:::

## The warning budget

The build is not run with `-W`. Around sixty docstring warnings remain, from
reStructuredText that predates there being any documentation build: literal
blocks missing their `::`, unpaired `*` from ASCII multiplication, and block
quotes without a trailing blank line. Fixing them all is a separate job.

Instead, `docs/warning-budget.txt` holds a single number and CI fails if the
count exceeds it. The budget can only ratchet down: when you fix a batch, lower
the file. Measure with a *clean* build, because Sphinx does not re-emit
warnings for pages it takes from the doctree cache:

```bash
sphinx-build -b html -E --keep-going -w /tmp/w.txt docs docs/_build/html
grep -cE 'WARNING|ERROR' /tmp/w.txt
```

`-n` (nitpicky) is deliberately not used. It adds several hundred
"reference target not found" warnings for numpydoc *prose* types — `ndarray`,
`shape`, `array_like` — which are English, not importable objects. Broken
`{doc}` and `{ref}` links are reported without it.

## Regenerating the napoleon section list

`napoleon_custom_sections` in `conf.py` lists every non-standard dashed heading
that occurs in a docstring, so that autodoc renders it instead of raising
"Unexpected section title". To regenerate after adding docstrings:

```bash
python - <<'PY'
import pathlib, re, collections
STD = {"Parameters","Returns","Yields","Receives","Other Parameters","Raises",
       "Warns","Warnings","See Also","Notes","References","Examples","Example",
       "Attributes","Methods"}
c = collections.Counter()
for p in pathlib.Path("src/pycalima").rglob("*.py"):
    L = p.read_text(errors="ignore").splitlines()
    for i in range(len(L) - 1):
        a, b = L[i].strip(), L[i + 1].strip()
        if b and set(b) == {"-"} and abs(len(b) - len(a)) <= 2 and a not in STD \
           and re.fullmatch(r"[A-Za-z][A-Za-z0-9 _()/,.:%-]*", a):
            c[a] += 1
for k, v in c.most_common():
    print(f'    "{k}",' + (f'  # x{v}' if v > 1 else ''))
PY
```

## Why the API reference is hand-listed

`sphinx-apidoc` and `autosummary --recursive` are both wrong here. Thirteen of
eighteen `__init__.py` files are empty, so package-level introspection yields
nothing; and discovery pulls in the 29 figure scripts and 12 exporters, one of
which prints to stdout at import. Each `docs/api/*.md` therefore names its
modules explicitly. Adding a physics module is a one-line edit; adding a script
needs no edit, because {doc}`/reference/scripts` picks it up automatically.
