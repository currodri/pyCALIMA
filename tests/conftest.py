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
    "BERNEPATH",
)


@pytest.fixture(autouse=True)
def _headless_matplotlib(monkeypatch):
    """Never try to open a window, and never inherit a display."""
    monkeypatch.setenv("MPLBACKEND", "Agg")


@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
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
def pristine_env(tmp_path, monkeypatch):
    """No CALIMA_* set at all, CWD in a temp directory.

    For testing the fallback chain itself.
    """
    for var in CALIMA_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _find_model_data() -> Path | None:
    """Locate a populated model_data/ tree, or None.

    Checks $CALIMA_MODEL_DATA, then the repository checkout this test file
    lives in. Deliberately does not call pycalima._paths, so that the skip
    decision is independent of the code under test.
    """
    env = os.environ.get("CALIMA_MODEL_DATA")
    candidates = []
    if env:
        candidates.append(Path(env))
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
