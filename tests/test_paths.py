"""Data-location invariants.

The whole point of :mod:`pycalima._paths` is that read-only reference data
lives inside the installed package while everything writable lives outside it.
These tests pin both halves, plus the guard that used to be missing when
``dust_charging.py`` called ``os.makedirs`` into its own package directory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pycalima import _paths


# --------------------------------------------------------------------------
# bundled reference data: inside the package, and actually present
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "parts",
    [
        ("li_draine_2001", "PAHneu_30"),
        ("li_draine_2001", "PAHion_30"),
        ("draine_lee_1984", "eps_Sil"),
        ("semenov_2003", "opacity"),
    ],
)
def test_optical_props_files_ship_and_are_in_package(parts):
    """Extensionless files are the ones a suffix-based glob would drop."""
    path = _paths.get_optical_props_path(*parts)
    assert path.is_file(), f"missing bundled data: {path}"
    assert _paths.PKG_DIR in path.parents


@pytest.mark.parametrize(
    "parts",
    [
        ("kp00_10000",),
        ("kp00_40000",),
        ("grains_CLOUDY.dat",),
        ("henke", "f1f2_Henke.dat"),
    ],
)
def test_external_data_files_ship_and_are_in_package(parts):
    path = _paths.get_external_data_path(*parts)
    assert path.is_file(), f"missing bundled data: {path}"
    assert _paths.PKG_DIR in path.parents


# --------------------------------------------------------------------------
# writable locations: never inside the package
# --------------------------------------------------------------------------

def test_model_data_is_outside_the_package(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for var in ("CALIMA_DATA", "CALIMA_MODEL_DATA"):
        monkeypatch.delenv(var, raising=False)
    md = _paths.get_model_data_dir()
    assert _paths.PKG_DIR not in md.parents and md != _paths.PKG_DIR


def test_calima_data_is_a_writable_root(tmp_path, monkeypatch):
    """$CALIMA_DATA is a root: model_data/, results/ and datasets/ hang off it."""
    for var in ("CALIMA_MODEL_DATA", "CALIMA_RESULTS", "CALIMA_DATASETS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CALIMA_DATA", str(tmp_path))
    assert _paths.get_model_data_dir() == tmp_path / "model_data"
    assert _paths.get_results_dir(create=False) == tmp_path / "results"
    assert _paths.get_dataset_cache_dir() == tmp_path / "datasets"


def test_exact_path_env_vars_beat_calima_data(tmp_path, monkeypatch):
    monkeypatch.setenv("CALIMA_DATA", str(tmp_path / "root"))
    monkeypatch.setenv("CALIMA_MODEL_DATA", str(tmp_path / "explicit_md"))
    monkeypatch.setenv("CALIMA_RESULTS", str(tmp_path / "explicit_res"))
    assert _paths.get_model_data_dir() == tmp_path / "explicit_md"
    assert _paths.get_results_dir(create=False) == tmp_path / "explicit_res"


def test_writable_guard_refuses_paths_inside_the_package():
    """The tripwire for the import-time makedirs-into-site-packages bug."""
    with pytest.raises(RuntimeError, match="inside the installed"):
        _paths.get_model_data_dir(base=_paths.PKG_DIR / "data" / "model_data")


def test_get_model_data_dir_does_not_create_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("CALIMA_MODEL_DATA", raising=False)
    monkeypatch.setenv("CALIMA_DATA", str(tmp_path))
    md = _paths.get_model_data_dir()
    assert not md.exists(), "resolving a path must not create it"


# --------------------------------------------------------------------------
# shipped configuration files, resolvable by name
# --------------------------------------------------------------------------

def test_bundled_grain_configs_are_discoverable():
    names = _paths.list_grain_configs()
    assert {"default", "ramses4bin", "4C6Si", "test"} <= set(names), names
    for name in names:
        assert _paths.resolve_grain_config_path(name).is_file()


def test_bundled_solver_configs_are_discoverable():
    names = _paths.list_solver_configs()
    assert "example_ic" in names, names
    assert len(names) == 8, names
    for name in names:
        assert _paths.resolve_solver_config_path(name).is_file()


def test_solver_config_resolves_from_a_path_tail():
    """Keeps the README's run_chemistry("solvers/configs/example_ic.json") working."""
    got = _paths.resolve_solver_config_path("solvers/configs/example_ic.json")
    assert got.is_file() and got.name == "example_ic.json"


def test_unknown_config_error_lists_the_alternatives():
    with pytest.raises(FileNotFoundError, match="Bundled configurations"):
        _paths.resolve_solver_config_path("no_such_config")


def test_resolved_config_paths_are_absolute():
    """A relative path stored verbatim used to break once the process chdir'd."""
    assert _paths.resolve_grain_config_path(None).is_absolute()
    assert _paths.resolve_solver_config_path("example_ic").is_absolute()


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------

def test_provenance_reports_a_commit():
    from pycalima._provenance import get_git_info, get_provenance

    prov = get_provenance()
    assert set(prov) >= {"branch", "commit", "version", "source"}
    assert prov["source"] != "none"
    branch, commit = get_git_info()
    assert isinstance(branch, str) and isinstance(commit, str)
