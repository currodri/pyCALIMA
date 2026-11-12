"""Shared fixtures.

Two things every test in this suite needs to be honest about:

* **Isolation.** Physics modules resolve writable paths through
  ``pycalima._paths``, which consults ``$CALIMA_*`` and the current directory.
  A test that inherits the developer's environment can pass for the wrong
  reason, so the default here is a clean, isolated environment.
* **Generated data.** ``model_data/`` is gitignored and produced by
  ``calima-export``. It may legitimately be absent (a fresh clone, CI), so
  anything that reads it is skipped rather than failed.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

CALIMA_ENV_VARS = (
    "CALIMA_DATA",
    "CALIMA_MODEL_DATA",
    "CALIMA_RESULTS",
    "CALIMA_DATASETS",
    "CALIMA_BUNDLED_DATA",
    "CALIMA_CONFIG",
    "CALIMA_PROVENANCE",
    "CALIMA_SED_DIR",
    "CALIMA_DUSTEM_FILE",
    "CALIMA_YIELD_DIR",
    "CALIMA_PAHDB_DIR",
    "BERNEPATH",
)


@pytest.fixture(autouse=True)
def _headless_matplotlib(monkeypatch):
    """Never try to open a window, and never inherit a display."""
    monkeypatch.setenv("MPLBACKEND", "Agg")


def _dataset_override_vars() -> set[str]:
    """Every variable a registered dataset takes its location from (``env_override`` in data/registry.toml).

    Read from the registry, so that a dataset added with a new variable is covered without editing this file.
    """
    from pycalima._datasets import iter_datasets

    return {d.env_override for d in iter_datasets() if d.env_override}


@pytest.fixture
def no_dataset_overrides(monkeypatch):
    """None of the developer's dataset locations ($CALIMA_PAHDB_DIR, $CALIMA_DUSTEM_DIR, ...).

    A test that a dataset is missing must not find the copy the developer points the suite at for the tests
    that need it: three tests in test_datasets.py failed whenever $CALIMA_PAHDB_DIR was set. A test of an
    override sets it itself.
    """
    for var in _dataset_override_vars():
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def isolated_env(tmp_path, monkeypatch, no_dataset_overrides):
    """A clean $CALIMA_DATA root in a temp directory, with the CWD moved there.

    Yields the root. Anything the code writes lands under it, so tests cannot
    pollute the checkout or each other.
    """
    root = tmp_path / "calima_data"
    root.mkdir()
    for var in CALIMA_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CALIMA_DATA", str(root))
    monkeypatch.chdir(tmp_path)
    return root


@pytest.fixture
def pristine_env(tmp_path, monkeypatch, no_dataset_overrides):
    """No CALIMA_* set at all, CWD in a temp directory.

    For testing the fallback chain itself.
    """
    for var in CALIMA_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _find_model_data() -> Path | None:
    """Locate a populated model_data/ tree, or None.

    Checks $CALIMA_MODEL_DATA, then $CALIMA_DATA/model_data (where
    `calima-export` writes when only the root is set), then the repository
    checkout this test file lives in. Deliberately does not call
    pycalima._paths, so that the skip decision is independent of the code
    under test.
    """
    candidates = []
    env = os.environ.get("CALIMA_MODEL_DATA")
    if env:
        candidates.append(Path(env))
    root = os.environ.get("CALIMA_DATA")
    if root:
        candidates.append(Path(root) / "model_data")
    repo_root = Path(__file__).resolve().parents[1]
    candidates.append(repo_root / "model_data")
    for c in candidates:
        if c.is_dir() and any(c.iterdir()):
            return c
    return None


MODEL_DATA = _find_model_data()

requires_model_data = pytest.mark.skipif(
    MODEL_DATA is None,
    reason="no populated model_data/ found; run `calima-export` or set "
           "$CALIMA_MODEL_DATA",
)


@pytest.fixture
def model_data(monkeypatch):
    """Point pycalima at a real generated-table tree, or skip."""
    if MODEL_DATA is None:
        pytest.skip("no populated model_data/ available")
    monkeypatch.setenv("CALIMA_MODEL_DATA", str(MODEL_DATA))
    return MODEL_DATA


# Optical constants of the collapse materials that are registered downloads
# (data/registry.toml), not bundled: absent on a fresh clone and in CI.
COLLAPSE_OPTICS = ("optical-franta-2017-si", "optical-triaud-magnetite", "optical-zubko-1996-acar")


@pytest.fixture
def collapse_optics():
    """Skip unless the collapse materials' optical constants have been fetched."""
    from pycalima._datasets import get_dataset

    missing = [d for d in COLLAPSE_OPTICS if get_dataset(d).locate() is None]
    if missing:
        pytest.skip(f"optical constants not fetched: {', '.join(missing)}")


@pytest.fixture
def malloci2007():
    """Skip unless the Malloci+07 TD-DFT spectra have been fetched (``calima-fetch-data fetch malloci2007-spectra``)."""
    from pycalima._datasets import get_dataset

    if get_dataset("malloci2007-spectra").locate() is None:
        pytest.skip("Malloci+07 spectra not fetched (calima-fetch-data fetch malloci2007-spectra)")


def _importable(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


requires_yt = pytest.mark.skipif(
    not _importable("yt"), reason="needs the 'sim' extra: pip install 'pycalima[sim]'"
)
requires_numba = pytest.mark.skipif(
    not _importable("numba"), reason="needs the 'accel' extra"
)
