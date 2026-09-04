"""Installation and packaging contract.

These tests assert things about the *installed distribution* rather than the
source tree, so they catch the class of failure that only appears after
`pip install`: missing package data, a broken entry point, an undeclared
dependency, or a generic top-level name squatting on the import path.

The wheel/sdist content checks run only when `dist/` has been populated by
`python -m build`; they are skipped otherwise so the suite stays fast.
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

import pycalima

DIST = Path(__file__).resolve().parents[1] / "dist"

CONSOLE_SCRIPTS = {
    "calima-paths": "pycalima._paths:main",
    "calima-fetch-data": "pycalima._datasets:main",
    "calima-export": "pycalima.models.export_all_grain_data:cli",
    "calima-run": "pycalima.solvers.run_chemistry:main",
    "calima-grid": "pycalima.solvers.run_grid:cli",
}

CORE_DEPENDENCIES = {
    "numpy", "scipy", "pandas", "matplotlib", "seaborn", "tqdm",
    "joblib", "unyt", "miepython", "platformdirs", "astropy",
}

EXPECTED_EXTRAS = {"accel", "sim", "pahdb", "plots", "profile", "all", "dev"}


# ---------------------------------------------------------------------------
# distribution metadata
# ---------------------------------------------------------------------------

def _metadata():
    from importlib.metadata import metadata

    return metadata("pycalima")


def test_distribution_is_installed():
    from importlib.metadata import version

    assert version("pycalima")


def test_dunder_version_matches_distribution():
    from importlib.metadata import version

    assert pycalima.__version__ == version("pycalima")


def test_version_carries_the_commit_hash():
    """setuptools_scm's node-and-date scheme is what makes provenance work
    after installation with no .git present. See pycalima._provenance."""
    from importlib.metadata import version

    ver = version("pycalima")
    if "+" not in ver:
        pytest.skip(f"version {ver} has no local segment (built from a tag)")
    local = ver.split("+", 1)[1]
    assert any(
        tok.startswith("g") and len(tok) >= 8 for tok in local.split(".")
    ), f"no g<hash> token in local version segment {local!r}"


def test_requires_python_floor():
    assert _metadata()["Requires-Python"] == ">=3.10"


def test_license_is_declared():
    md = _metadata()
    joined = " ".join(str(v) for v in md.values())
    assert "MIT" in joined


def test_core_dependencies_are_declared():
    from importlib.metadata import requires

    declared = requires("pycalima") or []
    # keep only unconditional requirements (no `; extra == ...` marker)
    core = set()
    for spec in declared:
        if "extra ==" in spec:
            continue
        name = spec.split(";")[0].strip()
        for sep in (">=", "==", "<", ">", "!", "~", "[", " "):
            name = name.split(sep)[0]
        core.add(name.strip().lower())
    missing = CORE_DEPENDENCIES - core
    assert not missing, f"undeclared core dependencies: {sorted(missing)}"


def test_extras_are_declared():
    md = _metadata()
    provided = {e.strip() for e in md.get_all("Provides-Extra") or []}
    assert EXPECTED_EXTRAS <= provided, f"missing extras: {EXPECTED_EXTRAS - provided}"


# ---------------------------------------------------------------------------
# import-path hygiene
# ---------------------------------------------------------------------------

def test_only_pycalima_is_claimed_on_the_import_path():
    """The pre-packaging layout would have installed top-level `models`,
    `solvers` and `galaxySAM`, which collide with almost anything."""
    from importlib.metadata import distribution

    dist = distribution("pycalima")
    top = dist.read_text("top_level.txt")
    if top is None:
        pytest.skip("distribution has no top_level.txt")
    names = {line.strip() for line in top.splitlines() if line.strip()}
    assert names == {"pycalima"}, f"claims extra top-level names: {names}"


@pytest.mark.parametrize("generic", ["models", "solvers", "galaxySAM"])
def test_generic_names_are_not_importable(generic):
    import importlib.util

    spec = None
    try:
        spec = importlib.util.find_spec(generic)
    except (ImportError, ValueError):
        spec = None
    if spec is None:
        return
    # If something IS importable under that name it must not be ours.
    origin = str(spec.origin or "")
    assert "pycalima" not in origin, (
        f"{generic!r} resolves into pycalima at {origin}; the package must not "
        f"squat generic top-level names"
    )


def test_subpackages_are_importable():
    import importlib

    for name in ("pycalima.models", "pycalima.solvers", "pycalima.galaxysam",
                 "pycalima._paths", "pycalima._datasets", "pycalima._provenance",
                 "pycalima.plotting_style"):
        importlib.import_module(name)


def test_parent_init_files_stay_empty():
    """Three subpackages have circular dependencies resolved only by deferred
    function-local imports. An eager re-export in a parent __init__ converts a
    working lazy cycle into an ImportError."""
    root = Path(pycalima.__path__[0])
    for rel in ("models/__init__.py", "models/dust_charge/__init__.py",
                "models/dust_radiation/__init__.py"):
        path = root / rel
        assert path.is_file(), f"missing {rel}"
        assert path.read_text(encoding="utf-8").strip() == "", (
            f"{rel} is no longer empty; see the note in pycalima/__init__.py"
        )


# ---------------------------------------------------------------------------
# console scripts
# ---------------------------------------------------------------------------

def test_console_scripts_are_declared():
    from importlib.metadata import entry_points

    eps = {ep.name: ep.value for ep in entry_points(group="console_scripts")
           if ep.value.startswith("pycalima")}
    for name, target in CONSOLE_SCRIPTS.items():
        assert name in eps, f"{name} is not declared"
        assert eps[name] == target, f"{name} -> {eps[name]}, expected {target}"


@pytest.mark.parametrize("name", sorted(CONSOLE_SCRIPTS))
def test_console_script_entry_point_loads_and_is_callable(name):
    from importlib.metadata import entry_points

    (ep,) = [e for e in entry_points(group="console_scripts") if e.name == name]
    fn = ep.load()
    assert callable(fn)


@pytest.mark.parametrize("name", sorted(CONSOLE_SCRIPTS))
def test_console_script_responds_to_help(name, tmp_path):
    """Runs the real installed executable, from a directory that is not the
    repository, so a CWD dependence would show up."""
    exe = Path(sys.executable).parent / name
    if not exe.exists():
        pytest.skip(f"{name} not installed in this environment")
    proc = subprocess.run(
        [str(exe), "--help"], cwd=tmp_path, capture_output=True, text=True, timeout=180
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip()


def test_cli_wrappers_return_an_exit_code_not_a_payload():
    """run_grid.main() returns the grid dict and export_all_grain_data.main()
    returns None. A console_scripts target is invoked as sys.exit(func()), so
    both need an int-returning wrapper; pointing at main() would exit 1 and
    dump the payload to stderr."""
    import inspect

    from pycalima.models.export_all_grain_data import cli as export_cli
    from pycalima.solvers.run_grid import cli as grid_cli

    for fn in (grid_cli, export_cli):
        src = inspect.getsource(fn)
        assert "return 0" in src, f"{fn.__qualname__} does not return an exit code"


# ---------------------------------------------------------------------------
# package data reachable through the public API
# ---------------------------------------------------------------------------

BUNDLED_SAMPLES = [
    ("optical_props", ("li_draine_2001", "PAHneu_30")),
    ("optical_props", ("li_draine_2001", "PAHion_30")),
    ("optical_props", ("draine_lee_1984", "eps_Sil")),
    ("optical_props", ("semenov_2003", "opacity")),
    ("optical_props", ("berne_2022", "neutrals")),
    ("external_data", ("kp00_10000",)),
    ("external_data", ("kp00_40000",)),
    ("external_data", ("grains_CLOUDY.dat",)),
    ("external_data", ("mathis1983.dat",)),
    ("external_data", ("henke", "f1f2_Henke.dat")),
    ("external_data", ("Herschel_filters",)),
]


@pytest.mark.parametrize("tree,parts", BUNDLED_SAMPLES)
def test_bundled_data_is_installed(tree, parts):
    """Most of these have no file extension, which is why package-data globs
    enumerate directory depths instead of suffixes."""
    from pycalima import _paths

    getter = (_paths.get_optical_props_path if tree == "optical_props"
              else _paths.get_external_data_path)
    path = getter(*parts)
    assert path.exists(), f"missing bundled {tree} entry: {path}"


def test_bundled_json_configs_are_installed():
    from pycalima import _paths

    assert len(_paths.list_grain_configs()) == 4
    assert len(_paths.list_solver_configs()) == 8


def test_ramses_fortran_reference_ships():
    root = Path(pycalima.__path__[0])
    f90 = sorted((root / "solvers" / "ramses_source").glob("*.f90"))
    assert len(f90) >= 10, f"expected the RAMSES reference sources, found {len(f90)}"


def test_registry_ships_and_parses():
    from pycalima._datasets import iter_datasets

    datasets = list(iter_datasets())
    assert len(datasets) >= 10
    assert {d.kind for d in datasets} <= {"bundled", "fetch", "manual"}


def test_galaxysam_yield_tables_ship():
    root = Path(pycalima.__path__[0])
    assert (root / "galaxysam" / "external_yields").is_dir()
    assert any((root / "galaxysam" / "yield_files").rglob("*"))


def test_no_notebooks_or_scripts_inside_the_installed_package():
    root = Path(pycalima.__path__[0])
    strays = [str(p.relative_to(root)) for p in root.rglob("*.ipynb")]
    strays += [str(p.relative_to(root)) for p in root.rglob("*~")]
    strays += [str(p.relative_to(root)) for p in root.rglob("test_*.py")]
    assert not strays, f"unwanted files inside the package: {strays}"


def test_pahdb_archives_are_not_bundled():
    """~575 MB of PAHdb must never ship; it is registered as kind='manual'."""
    root = Path(pycalima.__path__[0])
    assert not list(root.rglob("pahdb*"))


# ---------------------------------------------------------------------------
# built artifacts (only when dist/ is populated)
# ---------------------------------------------------------------------------

def _wheels():
    return sorted(DIST.glob("*.whl")) if DIST.is_dir() else []


def _sdists():
    return sorted(DIST.glob("*.tar.gz")) if DIST.is_dir() else []


needs_wheel = pytest.mark.skipif(
    not _wheels(), reason="no built wheel in dist/; run `python -m build`"
)
needs_sdist = pytest.mark.skipif(
    not _sdists(), reason="no built sdist in dist/; run `python -m build`"
)

FORBIDDEN_IN_DIST = (
    "pahdb", ".ipynb", ".pdf", ".DS_Store", "__pycache__",
    ".claude", ".pyc", "yohan_routines", "simulation_curves",
)


@needs_wheel
@pytest.mark.parametrize("pattern", FORBIDDEN_IN_DIST)
def test_wheel_excludes(pattern):
    names = zipfile.ZipFile(_wheels()[-1]).namelist()
    hits = [n for n in names if pattern in n]
    assert not hits, f"wheel contains {pattern!r}: {hits[:5]}"


@needs_wheel
def test_wheel_contains_every_tracked_data_file():
    """The check that would have caught a suffix-based package-data glob."""
    import fnmatch

    repo = Path(__file__).resolve().parents[1]
    try:
        tracked = subprocess.check_output(
            ["git", "ls-files",
             "src/pycalima/data",
             "src/pycalima/galaxysam/yield_files",
             "src/pycalima/galaxysam/external_yields",
             "src/pycalima/solvers/configs",
             "src/pycalima/solvers/ramses_source"],
            text=True, cwd=repo,
        ).split()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")

    drop = ("pahdb*", "*.DS_Store", "*~", "*.py", "*.ipynb", "*.pdf")
    want = {
        t for t in tracked
        if not any(fnmatch.fnmatch(t.rsplit("/", 1)[-1], p) for p in drop)
    }
    have = set(zipfile.ZipFile(_wheels()[-1]).namelist())
    missing = sorted(t for t in want if t.replace("src/", "") not in have)
    assert not missing, f"{len(missing)} tracked data files absent from the wheel: {missing[:10]}"


@needs_sdist
def test_sdist_excludes_the_duplicate_worktrees():
    """.claude/worktrees/ holds full copies of the repository; without the
    MANIFEST.in prune the sdist balloons past 200 MB."""
    import tarfile

    with tarfile.open(_sdists()[-1]) as tf:
        names = tf.getnames()
    assert not [n for n in names if ".claude" in n]
    assert not [n for n in names if "pahdb" in n]


@needs_wheel
def test_wheel_size_is_reasonable():
    size = _wheels()[-1].stat().st_size
    assert size < 40e6, f"wheel is {size/1e6:.1f} MB; something large slipped in"


# ---------------------------------------------------------------------------
# notebooks: no machine-specific paths, external data requested not assumed
# ---------------------------------------------------------------------------

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"

needs_notebooks = pytest.mark.skipif(
    not NOTEBOOK_DIR.is_dir(),
    reason="notebooks/ not present (installed copy)",
)


def _notebook_code(path: Path) -> str:
    """Concatenated source of every code cell."""
    import json

    nb = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source", [])
        out.append(src if isinstance(src, str) else "".join(src))
    return "\n".join(out)


@needs_notebooks
@pytest.mark.parametrize(
    "name", sorted(p.name for p in NOTEBOOK_DIR.glob("*.ipynb"))
    if NOTEBOOK_DIR.is_dir() else []
)
def test_notebook_has_no_machine_specific_paths(name):
    """A hardcoded ~/Documents/... path makes a notebook run only on its
    author's machine. External data must be requested via an env var."""
    import re

    code = _notebook_code(NOTEBOOK_DIR / name)
    bad = re.findall(r"""['"]~?/(?:Users|home|Documents)/[^'"]*['"]""", code)
    assert not bad, f"{name} hardcodes machine-specific paths: {bad[:4]}"


@needs_notebooks
@pytest.mark.parametrize("name", ["CALIMA_model_explorer.ipynb"])
def test_ramses_notebook_requests_simulation_outputs(name):
    """RAMSES snapshots are not shipped, so the notebook must ask for them by
    env var and fail loudly rather than silently using a wrong path."""
    path = NOTEBOOK_DIR / name
    if not path.is_file():
        pytest.skip(f"{name} not present")
    code = _notebook_code(path)
    assert "CALIMA_SIM_DIR" in code, f"{name} does not request $CALIMA_SIM_DIR"
    assert "SIM_SUBDIRS" in code, f"{name} has no editable per-run sub-paths"
    # and it must raise, not warn, when unset
    assert "raise RuntimeError" in code or "raise FileNotFoundError" in code, (
        f"{name} does not fail loudly when the outputs are unavailable"
    )


@needs_notebooks
def test_no_notebook_ships_stored_outputs():
    """Outputs are cleared on commit; 13 MB of stored figures otherwise."""
    import json

    offenders = []
    for path in sorted(NOTEBOOK_DIR.glob("*.ipynb")):
        nb = json.loads(path.read_text(encoding="utf-8"))
        for cell in nb.get("cells", []):
            if cell.get("cell_type") == "code" and cell.get("outputs"):
                offenders.append(path.name)
                break
    assert not offenders, f"notebooks committed with outputs: {offenders}"


@needs_notebooks
def test_notebooks_use_the_pycalima_import_root():
    """No notebook may import the pre-packaging top-level roots."""
    import re

    offenders = {}
    for path in sorted(NOTEBOOK_DIR.glob("*.ipynb")):
        code = _notebook_code(path)
        hits = re.findall(r"^\s*(?:from|import)\s+(models|solvers|galaxySAM)[.\s]",
                          code, re.M)
        if hits:
            offenders[path.name] = sorted(set(hits))
    assert not offenders, f"notebooks using old import roots: {offenders}"


@needs_notebooks
def test_the_docs_document_the_ramses_post_processing_workflow():
    """These notebooks need data the project cannot ship, so that has to be
    stated somewhere a reader will find it. That page is now
    docs/guide/post-processing.md rather than the README."""
    page = Path(__file__).resolve().parents[1] / "docs" / "guide" / "post-processing.md"
    if not page.is_file():
        pytest.skip("documentation sources are not part of the wheel")
    # Collapse whitespace: the prose is wrapped, and inside a blockquote the
    # phrase spans two lines with a "> " continuation.
    text = " ".join(
        page.read_text(encoding="utf-8").replace("\n>", " ").split()
    )
    for needle in ("CALIMA_model_explorer", "CALIMA_SIM_DIR",
                   "not distributed with pyCALIMA"):
        assert needle in text, f"docs/guide/post-processing.md omits {needle!r}"
