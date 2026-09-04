"""Single source of truth for every filesystem location pyCALIMA uses.

Before this module existed, 65 sites across 55 files each computed their own
"repository root" by walking up from ``__file__`` and then joined
``external_data/``, ``model_data/``, ``optical_props/`` or ``results/`` onto
it. That works in a source checkout and silently resolves inside
``site-packages`` once installed.

Four kinds of location, four resolution chains
==============================================

1. **Bundled reference data** -- read-only, ships inside the wheel:
   ``external_data/``, ``optical_props/``.
   :func:`get_data_root`, :func:`get_external_data_path`,
   :func:`get_optical_props_path`

2. **Fetched / user-supplied datasets** -- read-only, too large to bundle
   (the PAHdb archives). :func:`get_dataset_cache_dir`; see
   :mod:`pycalima._datasets`.

3. **Generated tables** -- read *and* write, produced by the
   ``pycalima.models.*.export_*`` modules: ``model_data/<model_name>/``.
   :func:`get_model_data_dir`

4. **Run output** -- write-only: solver results, figures, profiles.
   :func:`get_results_dir`, :func:`get_plots_dir`

Plus two configuration namespaces, both shipped as package data:
:func:`resolve_grain_config_path`, :func:`list_grain_configs`,
:func:`resolve_solver_config_path`, :func:`list_solver_configs`.

Nothing here creates a directory unless you pass ``create=True``, so importing
a module never touches the filesystem. And nothing here ever returns a
*writable* path inside the installed package -- :func:`_assert_writable_target`
enforces that.

Environment variables
=====================
``CALIMA_DATA``
    Writable **root**. ``model_data/``, ``results/`` and ``datasets/`` are
    created underneath it.
``CALIMA_MODEL_DATA``, ``CALIMA_RESULTS``, ``CALIMA_DATASETS``
    Point directly *at* the corresponding directory. Each beats
    ``CALIMA_DATA``.
``CALIMA_BUNDLED_DATA``
    Override the read-only reference-data root, e.g. to run an installed copy
    against a source checkout's data.
``CALIMA_CONFIG``
    Default grain-size-distribution JSON.
``BERNEPATH``
    Legacy override for ``optical_props/berne_2022``, preserved from the
    pre-packaging code.

Run ``calima-paths`` to print every resolved value for your environment.

Note on the two ``get_model_data_dir`` functions
================================================
:func:`pycalima._paths.get_model_data_dir` returns the *model-agnostic* root.
``pycalima.models.grain_size_config.get_model_data_dir`` wraps it and inserts
the active configuration's ``model_name`` subdirectory. Physics modules almost
always want the latter. For a fixed subdirectory of the root, write
``_paths.get_model_data_dir() / 'optical_properties'`` -- never pass the
subdirectory name into the ``model_name`` slot.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable

__all__ = [
    "PKG_DIR",
    "MissingReferenceData",
    "get_data_root",
    "get_external_data_path",
    "get_optical_props_path",
    "get_dataset_cache_dir",
    "get_model_data_dir",
    "get_results_dir",
    "get_plots_dir",
    "get_grain_config_dir",
    "resolve_grain_config_path",
    "list_grain_configs",
    "get_solver_config_dir",
    "resolve_solver_config_path",
    "list_solver_configs",
    "user_data_dir",
    "user_cache_dir",
    "paths_report",
]

# The installed package directory.
#
# This is plain __file__ arithmetic rather than importlib.resources on purpose.
# importlib.resources.files() returns a Traversable, which for a zip-backed
# package is a zipfile.Path -- not os.PathLike, so it cannot be handed to
# np.loadtxt, os.path.join or pandas.read_csv, which is exactly what ~200 call
# sites in this package do. Supporting that would mean holding an
# importlib.resources.as_file() extraction open for the process lifetime across
# 258 data files. pip never installs this distribution as a zip, and under
# `pip install -e .` __file__ correctly points into the source tree.
PKG_DIR: Path = Path(__file__).resolve().parent

_APP_NAME = "calima"
_APP_AUTHOR = "CALIMA"


class MissingReferenceData(FileNotFoundError):
    """Bundled reference data is absent from the installation."""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _env_path(name: str) -> Path | None:
    """``$name`` as an absolute, expanded Path, or None if unset or blank."""
    raw = os.environ.get(name)
    if not raw or not raw.strip():
        return None
    return Path(raw).expanduser().resolve()


def _looks_like_source_checkout(directory: Path) -> bool:
    """True if *directory* is the root of a pyCALIMA source checkout.

    Needed because ``model_data/`` is gitignored and has no tracked files, so
    a fresh clone does not contain it. Testing only "does ./model_data exist"
    would send the first ``calima-export`` run in a fresh checkout to the
    per-user data directory instead of the checkout, which would surprise
    anyone following the README.
    """
    if (directory / "src" / "pycalima" / "_paths.py").is_file():
        return True
    pyproject = directory / "pyproject.toml"
    if pyproject.is_file():
        try:
            head = pyproject.read_text(encoding="utf-8", errors="replace")[:4096]
        except OSError:
            return False
        return 'name = "pycalima"' in head or "name = 'pycalima'" in head
    return False


def user_data_dir() -> Path:
    """Per-user writable data directory.

    Uses ``platformdirs`` when importable and otherwise falls back to a
    hand-rolled equivalent, so that importing pycalima never fails on a thin
    install.
    """
    try:
        import platformdirs
    except ModuleNotFoundError:
        home = Path.home()
        if os.name == "nt":
            base = Path(os.environ.get("LOCALAPPDATA", home / "AppData" / "Local"))
            return base / _APP_AUTHOR / _APP_NAME
        if sys.platform == "darwin":
            return home / "Library" / "Application Support" / _APP_NAME
        base = Path(os.environ.get("XDG_DATA_HOME", home / ".local" / "share"))
        return base / _APP_NAME
    return Path(platformdirs.user_data_dir(_APP_NAME, _APP_AUTHOR))


def user_cache_dir() -> Path:
    """Per-user writable cache directory, for fetched datasets."""
    try:
        import platformdirs
    except ModuleNotFoundError:
        return user_data_dir() / "cache"
    return Path(platformdirs.user_cache_dir(_APP_NAME, _APP_AUTHOR))


def _writable_root(marker_subdir: str) -> Path:
    """Resolve the writable root that should contain *marker_subdir*.

    ``$CALIMA_DATA`` -> the CWD if ``./<marker_subdir>`` exists -> the CWD if
    it is a pyCALIMA source checkout -> the per-user data directory.
    """
    env = _env_path("CALIMA_DATA")
    if env is not None:
        return env

    cwd = Path.cwd()
    if (cwd / marker_subdir).is_dir():
        return cwd
    if _looks_like_source_checkout(cwd):
        return cwd

    return user_data_dir()


def _assert_writable_target(path: Path, what: str) -> Path:
    """Refuse to hand out a writable path inside the installed package.

    This is the tripwire for the class of bug that had
    ``models/dust_charge/dust_charging.py`` calling ``os.makedirs`` into its
    own package directory at import time.
    """
    try:
        path.resolve().relative_to(PKG_DIR)
    except ValueError:
        return path
    raise RuntimeError(
        f"Refusing to use {path} for {what}: that is inside the installed "
        f"pycalima package ({PKG_DIR}). Set $CALIMA_DATA to a writable "
        f"directory, or run from a pyCALIMA source checkout."
    )


def _ensure(path: Path, create: bool) -> Path:
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# 1. bundled reference data (read-only)
# ---------------------------------------------------------------------------

def get_data_root() -> Path:
    """Root of the read-only reference data shipped inside the package.

    Contains ``external_data/`` and ``optical_props/``. Resolution is
    ``$CALIMA_BUNDLED_DATA`` then ``<package>/data``.

    Raises
    ------
    MissingReferenceData
        If neither location holds the expected subdirectories. This fails
        loudly by design: a wheel built with the wrong package-data globs
        would otherwise degrade into a thicket of FileNotFoundErrors deep
        inside physics code.
    """
    override = _env_path("CALIMA_BUNDLED_DATA")
    if override is not None:
        if not override.is_dir():
            raise MissingReferenceData(
                f"$CALIMA_BUNDLED_DATA points at {override}, which is not a directory."
            )
        return override

    root = PKG_DIR / "data"
    if not (root / "optical_props").is_dir():
        raise MissingReferenceData(
            f"pycalima reference data is missing from {root}. The distribution "
            f"was probably built without its package data. Reinstall, or set "
            f"$CALIMA_BUNDLED_DATA to a pyCALIMA checkout containing "
            f"external_data/ and optical_props/."
        )
    return root


def get_external_data_path(*parts: str | os.PathLike[str]) -> Path:
    """Path inside the bundled ``external_data/`` tree.

    ``get_external_data_path()`` returns the directory itself;
    ``get_external_data_path("henke", "f1f2_Henke.dat")`` a file within it.

    Existence is not checked. For files that may need fetching, use
    :func:`pycalima._datasets.ensure_dataset`.
    """
    return get_data_root().joinpath("external_data", *(str(p) for p in parts))


def get_optical_props_path(*parts: str | os.PathLike[str]) -> Path:
    """Path inside the bundled ``optical_props/`` tree.

    Honours the legacy ``$BERNEPATH`` override when the first component is
    ``"berne_2022"``, preserving the behaviour that
    ``models/PAH_charge/PAH_photoelectric_heating.py`` documented.
    """
    parts = tuple(str(p) for p in parts)
    if parts and parts[0] == "berne_2022":
        berne = _env_path("BERNEPATH")
        if berne is not None:
            return berne.joinpath(*parts[1:])
    return get_data_root().joinpath("optical_props", *parts)


def get_dataset_cache_dir(create: bool = False) -> Path:
    """Directory holding datasets fetched or imported at run time.

    ``$CALIMA_DATASETS`` -> ``$CALIMA_DATA/datasets`` -> ``./reference_data``
    if it exists and is writable (so an existing checkout keeps using the
    layout it already has) -> ``<user cache dir>/datasets``.
    """
    env = _env_path("CALIMA_DATASETS")
    if env is not None:
        return _ensure(env, create)

    data_root = _env_path("CALIMA_DATA")
    if data_root is not None:
        return _ensure(data_root / "datasets", create)

    legacy = Path.cwd() / "reference_data"
    if legacy.is_dir() and os.access(legacy, os.W_OK):
        return legacy

    return _ensure(user_cache_dir() / "datasets", create)


# ---------------------------------------------------------------------------
# 2. generated tables (read + write)
# ---------------------------------------------------------------------------

def get_model_data_dir(
    model_name: str | None = None,
    *,
    base: str | os.PathLike[str] | None = None,
    create: bool = False,
) -> Path:
    """Directory holding generated rate and opacity tables.

    Parameters
    ----------
    model_name
        Subdirectory named after the active grain-size model, e.g.
        ``"ramses4bin"``. ``None`` returns the bare ``model_data`` root.
        This is *not* a general subdirectory slot -- see the module docstring.
    base
        Explicit override of the ``model_data`` directory itself, skipping all
        environment and CWD resolution. Highest priority.
    create
        Create the directory and parents if missing. Default False, so that
        importing a module never writes to disk.

    Resolution when *base* is None: ``$CALIMA_MODEL_DATA`` ->
    ``$CALIMA_DATA/model_data`` -> ``./model_data`` if it exists -> the CWD if
    it is a source checkout -> ``<user data dir>/model_data``.
    """
    if base is not None:
        root = Path(base).expanduser()
        if not root.is_absolute():
            root = (Path.cwd() / root).resolve()
    else:
        env = _env_path("CALIMA_MODEL_DATA")
        root = env if env is not None else _writable_root("model_data") / "model_data"

    root = _assert_writable_target(root, "generated model_data tables")
    out = root / model_name if model_name else root
    return _ensure(out, create)


# ---------------------------------------------------------------------------
# 3. run output (write)
# ---------------------------------------------------------------------------

def get_results_dir(
    subdir: str | os.PathLike[str] | None = None,
    *,
    base: str | os.PathLike[str] | None = None,
    create: bool = True,
) -> Path:
    """Directory for solver output, tables and figures.

    *base* -> ``$CALIMA_RESULTS`` -> ``$CALIMA_DATA/results`` -> ``./results``
    if it exists or the CWD is a source checkout -> ``<user data dir>/results``.

    Unlike :func:`get_model_data_dir` this defaults to ``create=True``, since
    every caller is about to write into it.
    """
    if base is not None:
        root = Path(base).expanduser().resolve()
    else:
        env = _env_path("CALIMA_RESULTS")
        root = env if env is not None else _writable_root("results") / "results"

    root = _assert_writable_target(root, "run output")
    out = root / subdir if subdir else root
    return _ensure(out, create)


def get_plots_dir(
    subdir: str | os.PathLike[str] | None = None, *, create: bool = True
) -> Path:
    """``get_results_dir()/plots[/subdir]`` -- where every figure belongs.

    Replaces the bare ``fig.savefig('name.png')`` calls that used to litter
    the process CWD.
    """
    tail = Path("plots") / subdir if subdir else Path("plots")
    return get_results_dir(tail, create=create)


# ---------------------------------------------------------------------------
# 4. shipped configuration files
# ---------------------------------------------------------------------------

def get_grain_config_dir() -> Path:
    """Directory holding the bundled ``grain_size_distribution*.json`` files."""
    return PKG_DIR / "models"


def list_grain_configs() -> list[str]:
    """Short names of the bundled grain-size configurations.

    ``["4C6Si", "default", "ramses4bin", "test"]`` for the four shipped JSONs.
    """
    names = []
    for p in sorted(get_grain_config_dir().glob("*grain_size_distribution*.json")):
        stem = p.stem
        if stem == "grain_size_distribution":
            names.append("default")
        elif stem.startswith("grain_size_distribution_"):
            names.append(stem[len("grain_size_distribution_"):])
        elif stem.endswith("_grain_size_distribution"):
            names.append(stem[: -len("_grain_size_distribution")])
        else:
            names.append(stem)
    return sorted(set(names))


def resolve_grain_config_path(config: str | os.PathLike[str] | None = None) -> Path:
    """Resolve a grain-size configuration to an absolute, existing file.

    *config* may be None (use ``$CALIMA_CONFIG``, else the bundled default), a
    filesystem path, or a short name from :func:`list_grain_configs` such as
    ``"ramses4bin"`` or ``"default"``.

    The result is always absolute and resolved, so it stays valid as a cache
    key even if the process later chdirs. The pre-packaging
    ``set_config_path()`` stored its argument verbatim, so a relative path
    stayed CWD-relative and ``"models/x.json"`` and ``"/abs/models/x.json"``
    were two cache entries for one file.
    """
    if config is None:
        env = _env_path("CALIMA_CONFIG")
        if env is not None:
            if not env.is_file():
                raise FileNotFoundError(
                    f"$CALIMA_CONFIG points at {env}, which does not exist."
                )
            return env
        return (get_grain_config_dir() / "grain_size_distribution.json").resolve()

    candidate = Path(config).expanduser()
    if candidate.is_file():
        return candidate.resolve()

    # A bare filename or a path whose tail names a bundled config.
    bundled = get_grain_config_dir() / candidate.name
    if bundled.is_file():
        return bundled.resolve()

    name = str(config)
    if os.sep not in name and candidate.suffix != ".json":
        stem = (
            "grain_size_distribution"
            if name == "default"
            else f"grain_size_distribution_{name}"
        )
        for trial in (f"{stem}.json", f"{name}_grain_size_distribution.json"):
            bundled = get_grain_config_dir() / trial
            if bundled.is_file():
                return bundled.resolve()

    raise FileNotFoundError(
        f"Grain-size configuration not found: {config}. "
        f"Bundled configurations: {', '.join(list_grain_configs())}. "
        f"You may also pass a path to your own JSON file."
    )


def get_solver_config_dir() -> Path:
    """Directory holding the bundled ``solvers/configs/*.json`` files."""
    return PKG_DIR / "solvers" / "configs"


def list_solver_configs() -> list[str]:
    """Stems of the bundled solver initial-condition files."""
    return sorted(p.stem for p in get_solver_config_dir().glob("*.json"))


def resolve_solver_config_path(config: str | os.PathLike[str]) -> Path:
    """Resolve solver initial conditions to an absolute, existing file.

    Accepts a filesystem path, a bare stem (``"example_ic"``) or a bare
    filename (``"example_ic.json"``). The filename form is what keeps the
    README's ``run_chemistry("solvers/configs/example_ic.json")`` working
    after installation: the tail resolves against the bundled directory.
    """
    candidate = Path(config).expanduser()
    if candidate.is_file():
        return candidate.resolve()

    for trial in (candidate.name, f"{candidate.stem}.json"):
        bundled = get_solver_config_dir() / trial
        if bundled.is_file():
            return bundled.resolve()

    raise FileNotFoundError(
        f"Solver configuration not found: {config}. "
        f"Bundled configurations: {', '.join(list_solver_configs())}."
    )


# ---------------------------------------------------------------------------
# diagnostics / CLI
# ---------------------------------------------------------------------------

_ENV_VARS = (
    "CALIMA_DATA",
    "CALIMA_MODEL_DATA",
    "CALIMA_RESULTS",
    "CALIMA_DATASETS",
    "CALIMA_BUNDLED_DATA",
    "CALIMA_CONFIG",
    "CALIMA_SED_DIR",
    "CALIMA_DUSTEM_FILE",
    "BERNEPATH",
)


def paths_report() -> dict[str, str]:
    """Every resolved location, for ``calima-paths`` and bug reports."""

    def _try(fn, *a, **kw):
        try:
            value = fn(*a, **kw)
        except Exception as exc:  # noqa: BLE001 - a report must never raise
            return f"<error: {type(exc).__name__}: {exc}>"
        if isinstance(value, list):
            return ", ".join(value)
        return str(value)

    return {
        "pycalima package": str(PKG_DIR),
        "bundled data root": _try(get_data_root),
        "  external_data": _try(get_external_data_path),
        "  optical_props": _try(get_optical_props_path),
        "dataset cache": _try(get_dataset_cache_dir),
        "model_data": _try(get_model_data_dir),
        "results": _try(get_results_dir, create=False),
        "plots": _try(get_plots_dir, create=False),
        "grain config dir": _try(get_grain_config_dir),
        "grain config (active)": _try(resolve_grain_config_path),
        "grain configs available": _try(list_grain_configs),
        "solver config dir": _try(get_solver_config_dir),
        "solver configs available": _try(list_solver_configs),
        "user data dir": _try(user_data_dir),
        "user cache dir": _try(user_cache_dir),
        "cwd": str(Path.cwd()),
        "cwd is source checkout": str(_looks_like_source_checkout(Path.cwd())),
    }


def main(argv: Iterable[str] | None = None) -> int:
    """``calima-paths`` -- print the resolved data locations and exit."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="calima-paths",
        description="Show where pyCALIMA reads and writes data.",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(list(argv) if argv is not None else None)

    report = paths_report()
    try:
        from pycalima._provenance import get_provenance

        report.update({f"provenance.{k}": str(v) for k, v in get_provenance().items()})
    except Exception:  # noqa: BLE001
        pass

    if args.json:
        import json

        report.update({f"env.{v}": os.environ.get(v, "") for v in _ENV_VARS})
        print(json.dumps(report, indent=2))
        return 0

    width = max(len(k) for k in report)
    for key, value in report.items():
        print(f"{key:<{width}}  {value}")
    print()
    for var in _ENV_VARS:
        print(f"env {var:<20} {os.environ.get(var, '<unset>')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
