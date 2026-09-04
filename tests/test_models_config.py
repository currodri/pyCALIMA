"""The configuration layer: models/grain_size_config.py.

Everything in models/ flows from this module, so its contract matters more
than most. It also carries process-global mutable state (the active config
path and a parsed-config cache), which is a a common source of
order-dependence -- the tests below restore it explicitly.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from pycalima.models import grain_size_config as gsc


@pytest.fixture(autouse=True)
def restore_active_config():
    """grain_size_config keeps the active path in a module global."""
    original = gsc.get_config_path()
    yield
    gsc.set_config_path(original)


# ---------------------------------------------------------------------------
# loading and parsing
# ---------------------------------------------------------------------------

def test_default_config_loads():
    cfg = gsc.load_grain_size_config()
    assert cfg["bin_ids"]
    assert len(cfg["bins"]) == len(cfg["bin_ids"])


def test_parsed_config_exposes_the_documented_keys():
    cfg = gsc.load_grain_size_config()
    for key in ("bins", "bin_ids", "bin_to_index", "export_parameters",
                "model_name", "basic", "shattering", "__path__"):
        assert key in cfg, f"missing parsed key {key!r}"


def test_bin_metadata_is_well_formed():
    for info in gsc.get_bins():
        assert isinstance(info["id"], str) and info["id"]
        assert info["composition"] in ("graphite", "silicate")
        assert isinstance(info["is_pah"], bool)
        assert isinstance(info["bin_rank"], int)


def test_bin_to_index_is_consistent():
    cfg = gsc.load_grain_size_config()
    for name, idx in cfg["bin_to_index"].items():
        assert cfg["bin_ids"][idx] == name


def test_parameter_arrays_match_the_bin_count():
    cfg = gsc.load_grain_size_config()
    n = len(cfg["bin_ids"])
    for model in ("basic", "shattering"):
        for key in ("a0", "amin", "amax", "sigma", "s"):
            arr = cfg[model][key]
            assert isinstance(arr, np.ndarray)
            assert arr.shape == (n,), f"{model}.{key} has shape {arr.shape}, expected {(n,)}"


def test_lognormal_parameters_are_physically_ordered():
    for info in gsc.get_bins():
        p = gsc.get_lognormal_parameters(info["id"])
        assert p["amin"] > 0
        assert p["amin"] <= p["a0"] <= p["amax"], (
            f"{info['id']}: a0={p['a0']} outside [{p['amin']}, {p['amax']}]"
        )
        assert p["sigma"] > 0


def test_get_bin_by_rank_round_trips():
    """get_bin_by_rank(composition, bin_rank, is_pah) must return the same bin
    that get_bins() reported those attributes for."""
    for info in gsc.get_bins():
        got = gsc.get_bin_by_rank(
            info["composition"], bin_rank=info["bin_rank"], is_pah=info["is_pah"]
        )
        assert got["id"] == info["id"]


def test_get_bin_by_rank_rejects_an_unknown_composition():
    with pytest.raises(KeyError):
        gsc.get_bin_by_rank("unobtainium")


def test_pah_and_dust_bins_partition_the_bin_list():
    all_ids = {b["id"] for b in gsc.get_bins()}
    pah = {b["id"] for b in gsc.get_bins(is_pah=True)}
    dust = {b["id"] for b in gsc.get_bins(is_pah=False)}
    assert pah | dust == all_ids
    assert not (pah & dust)


# ---------------------------------------------------------------------------
# selecting a configuration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["default", "ramses4bin", "4C6Si", "test"])
def test_every_bundled_config_loads_by_short_name(name):
    gsc.set_config_path(name)
    cfg = gsc.load_grain_size_config()
    assert cfg["bin_ids"], f"{name} parsed with no bins"


def test_set_config_path_returns_an_absolute_path():
    """It used to store the argument verbatim, so a relative path stayed
    CWD-relative and two spellings cached as two entries."""
    got = gsc.set_config_path("ramses4bin")
    assert got.is_absolute() and got.is_file()


def test_set_config_path_changes_the_bin_set():
    gsc.set_config_path("default")
    default_bins = gsc.load_grain_size_config()["bin_ids"]
    gsc.set_config_path("ramses4bin")
    ramses_bins = gsc.load_grain_size_config()["bin_ids"]
    assert default_bins != ramses_bins


def test_set_config_path_none_restores_the_default():
    gsc.set_config_path("ramses4bin")
    gsc.set_config_path(None)
    assert gsc.get_config_path().name == "grain_size_distribution.json"


def test_model_name_drives_the_generated_data_subdirectory(isolated_env):
    """The ramses4bin config sets model_name, so its tables must land in a
    per-model subdirectory; the default config sets none."""
    gsc.set_config_path("ramses4bin")
    assert gsc.load_grain_size_config()["model_name"] == "ramses4bin"
    assert gsc.get_model_data_dir().name == "ramses4bin"

    gsc.set_config_path("default")
    assert gsc.load_grain_size_config()["model_name"] is None
    assert gsc.get_model_data_dir().name == "model_data"


def test_unknown_config_name_raises_with_the_alternatives():
    with pytest.raises(FileNotFoundError, match="Bundled configurations"):
        gsc.set_config_path("definitely_not_a_config")


def test_an_external_config_file_can_be_used(tmp_path):
    """Passing a path outside the package must work, since that is how users
    run a custom bin configuration."""
    src = json.loads(gsc.get_config_path().read_text(encoding="utf-8"))
    custom = tmp_path / "my_config.json"
    custom.write_text(json.dumps(src), encoding="utf-8")

    gsc.set_config_path(custom)
    assert gsc.get_config_path() == custom.resolve()
    assert gsc.load_grain_size_config()["bin_ids"]


# ---------------------------------------------------------------------------
# caching
# ---------------------------------------------------------------------------

def test_repeated_loads_return_the_cached_object():
    gsc.set_config_path("default")
    first = gsc.load_grain_size_config()
    second = gsc.load_grain_size_config()
    assert first is second


def test_reload_bypasses_the_cache():
    gsc.set_config_path("default")
    first = gsc.load_grain_size_config()
    reloaded = gsc.load_grain_size_config(reload=True)
    assert reloaded is not first
    assert reloaded["bin_ids"] == first["bin_ids"]


def test_switching_config_invalidates_the_cache():
    gsc.set_config_path("default")
    a = gsc.load_grain_size_config()
    gsc.set_config_path("ramses4bin")
    b = gsc.load_grain_size_config()
    assert a is not b
    assert a["bin_ids"] != b["bin_ids"]


# ---------------------------------------------------------------------------
# path accessors and provenance
# ---------------------------------------------------------------------------

def test_optical_props_accessor_points_into_the_package():
    from pycalima import _paths

    path = gsc.get_optical_props_path()
    assert path.is_dir()
    assert _paths.PKG_DIR in path.parents


def test_external_data_accessor_points_into_the_package():
    from pycalima import _paths

    path = gsc.get_external_data_path()
    assert path.is_dir()
    assert _paths.PKG_DIR in path.parents


def test_get_repo_root_is_deprecated():
    with pytest.warns(DeprecationWarning):
        gsc.get_repo_root()


def test_header_lines_carry_provenance_and_a_fixed_count():
    lines = gsc.get_header_lines("Title", "pycalima/models/demo.py")
    assert all(line.startswith("#") for line in lines)
    assert any("Title" in line for line in lines)
    assert any(line.startswith("# Code: pycalima") for line in lines), (
        "the provenance stamp is missing; exporters depend on it"
    )


@pytest.mark.parametrize("num_lines", [4, 6, 8])
def test_header_lines_honours_an_exact_line_count(num_lines):
    """Several exporters pass num_lines= and depend on it exactly."""
    lines = gsc.get_header_lines("T", "s.py", num_lines=num_lines)
    assert len(lines) == num_lines


def test_get_git_info_returns_two_strings():
    branch, commit = gsc.get_git_info()
    assert isinstance(branch, str) and branch
    assert isinstance(commit, str) and commit
