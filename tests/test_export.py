"""End-to-end test of the export pipeline, on one dust grain and one PAH.

``calima-export`` is by far the slowest thing in CI. The ``model-data`` job
runs it with the ``tutorial`` configuration, because the tutorials read those
tables by name (``DustBin_03``, ``PAHBin_01``) at the real export resolution.
Nothing was gating the *fast* ``tests`` job on the exporters working at all, so
a broken exporter surfaced only in ``model-data`` — a job whose failure is much
harder to read, and which runs once rather than across the Python matrix.

These tests run the whole pipeline against the bundled ``test`` configuration:
exactly one dust bin (silicate, ``a0 = 0.005`` µm) and one PAH bin (graphite,
``a0 = 0.0005`` µm), with deliberately coarse ``export_parameters``. The two
dominant stages both scale per dust bin, so all twelve stages finish in well
under a minute — cheap enough to gate every PR on.

What this does and does not cover: it asserts every stage runs, writes its
directory, and produces parseable tables for both bins. It does *not* check
physics values; the table-dependent tests under ``requires_model_data`` do that
against tables exported at full resolution.
"""
from __future__ import annotations

import numpy as np
import pytest

# Directory each stage writes, keyed by the stage name in STAGE_NAMES order.
# Two stages share optical_properties/, and dust_band_luminosities writes there
# too, so the mapping is not one-to-one.
STAGE_OUTPUT_DIRS = {
    "collisional_cooling": "collisional_cooling_data",
    "dust_charging": "dust_charging_data",
    "dust_ion_recombination": "dust_ion_recombination_data",
    "dust_optical_properties": "optical_properties",
    "dust_photoelectric_heating": "dust_photoelectric_heating_data",
    "dust_sublimation": "dust_sublimation",
    "pah_dissociation_tables": "PAH_dissociation_data",
    "pah_optical_properties": "optical_properties",
    "pah_photoelectric_heating_tables": "PAH_photoelectric_heating_data",
    "pah_sputtering_rates": "pah_sputtering_data",
    "sputtering_rates": "thermal_sputtering_data",
}

# The two bins in test_grain_size_distribution.json.
DUST_BIN, PAH_BIN = "silicate_small_test", "pah_small_test"


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """Run the full export once for the whole module.

    Module-scoped because the export costs tens of seconds and every test here
    inspects the same tree. ``pytest.MonkeyPatch`` is instantiated directly:
    the ``monkeypatch`` fixture is function-scoped and cannot be requested from
    a module-scoped fixture.
    """
    from pycalima.models import grain_size_config
    from pycalima.models.export_all_grain_data import cli

    mp = pytest.MonkeyPatch()
    root = tmp_path_factory.mktemp("calima_export")
    model_data = root / "model_data"
    model_data.mkdir()
    # Both are set because calima-export resolves CALIMA_MODEL_DATA first and
    # falls back to CALIMA_DATA/model_data; pinning both keeps the test
    # independent of that precedence.
    mp.setenv("CALIMA_DATA", str(root))
    mp.setenv("CALIMA_MODEL_DATA", str(model_data))
    mp.setenv("MPLBACKEND", "Agg")
    try:
        returncode = cli(["--config", "test", "--no-profile"])
        yield model_data, returncode
    finally:
        mp.undo()
        # --config sets the process-wide grain configuration; restore the
        # default so later test modules do not see the one-bin test config.
        grain_size_config.set_config_path(None)


def test_export_succeeds(exported):
    model_data, returncode = exported
    assert returncode == 0
    assert any(model_data.iterdir()), "export wrote nothing"


@pytest.mark.parametrize("stage,subdir", sorted(STAGE_OUTPUT_DIRS.items()))
def test_every_stage_writes_its_directory(exported, stage, subdir):
    """A partially failed export otherwise leaves the table-dependent tests
    silently skipping, which is the failure mode the CI job guards against
    with an explicit directory check."""
    model_data, _ = exported
    path = model_data / subdir
    assert path.is_dir(), f"stage {stage} did not create {subdir}/"
    data = [p for p in path.rglob("*") if p.is_file()
            and p.suffix not in (".png", ".pdf")]
    assert data, f"stage {stage} created {subdir}/ but wrote no data files"


@pytest.mark.parametrize("bin_id", [DUST_BIN, PAH_BIN])
def test_both_bins_appear_in_the_output(exported, bin_id):
    """The point of this configuration: one grain and one PAH, and both have to
    make it through. A config-plumbing bug that dropped the PAH bin would
    otherwise still produce a tree full of dust tables and look healthy."""
    model_data, _ = exported
    hits = [p for p in model_data.rglob(f"*{bin_id}*") if p.is_file()]
    assert hits, f"no output file mentions {bin_id}"


def test_tables_are_parseable_numeric_data(exported):
    """Guards against truncated or empty writes, which a directory-existence
    check cannot see. Samples one table from each of two stages that write
    plain columns."""
    model_data, _ = exported
    samples = [
        next((model_data / "collisional_cooling_data").glob(f"cooling_{DUST_BIN}_Z_*")),
        next((model_data / "pah_sputtering_data").glob(f"sputtering_{PAH_BIN}_Z_*")),
    ]
    for path in samples:
        # Tokenise rather than np.loadtxt: the tables start with a short
        # size header (e.g. "nT nv") whose column count differs from the rows.
        values = np.array([
            float(tok)
            for line in path.read_text().splitlines()
            if not line.lstrip().startswith("#")
            for tok in line.split()
        ])
        assert values.size, f"{path.name} is empty"
        assert np.isfinite(values).all(), \
            f"{path.name} contains non-finite entries"


def test_the_test_config_really_is_one_grain_and_one_pah():
    """If someone adds bins to test_grain_size_distribution.json, this suite
    stops being fast and the reason will not be obvious. Fail loudly here
    instead."""
    import json

    from pycalima import _paths

    cfg = json.loads(_paths.resolve_grain_config_path("test").read_text())
    dust = [b for b in cfg["bins"] if not b["is_pah"]]
    pah = [b for b in cfg["bins"] if b["is_pah"]]
    assert len(dust) == 1, f"expected 1 dust bin, found {len(dust)}"
    assert len(pah) == 1, f"expected 1 PAH bin, found {len(pah)}"
    assert cfg["basic"]["a0"][0] == pytest.approx(0.005)


def test_the_tutorial_config_is_a_verbatim_subset_of_the_default():
    """CI exports only the ``tutorial`` configuration, and the tutorials read
    its tables as if they were the default configuration's DustBin_03 and
    PAHBin_01. That holds only while every per-bin parameter and every export
    grid is copied unchanged, so pin it."""
    import json

    from pycalima import _paths

    default = json.loads(_paths.resolve_grain_config_path("default").read_text())
    tutorial = json.loads(_paths.resolve_grain_config_path("tutorial").read_text())

    ids = [b["id"] for b in default["bins"]]
    assert [b["id"] for b in tutorial["bins"]] == ["PAHBin_01", "DustBin_03"]
    for i, b in enumerate(tutorial["bins"]):
        j = ids.index(b["id"])
        assert b == default["bins"][j]
        for section in ("basic", "shattering"):
            for key, values in tutorial[section].items():
                assert values[i] == default[section][key][j], (section, key, b["id"])
    assert tutorial["export_parameters"] == default["export_parameters"]
    assert tutorial.get("model_name") == default.get("model_name")


def test_tutorial_ic_uses_only_the_tutorial_bins():
    """A solver-config bin without an exported table silently loses its
    table-driven rates (dust_init skips missing files), so tutorial_ic must
    not reference any bin the tutorial grain configuration does not export."""
    import json

    from pycalima import _paths

    grain = json.loads(_paths.resolve_grain_config_path("tutorial").read_text())
    solver = json.loads(_paths.resolve_solver_config_path("tutorial_ic").read_text())
    exported = {b["id"].lower() for b in grain["bins"]}
    used = {b["id"].lower() for b in solver["dust_bins"] + solver["pah_bins"]}
    assert used == exported
