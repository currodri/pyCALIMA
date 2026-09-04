"""The reference-dataset registry: pycalima._datasets and data/registry.toml.

The registry's job is that a missing dataset produces an error naming the exact
command to run, instead of a bare FileNotFoundError from somewhere deep inside
physics code. These tests pin the registry's shape, the bundled/fetch/manual
distinction, and that failure message.
"""

from __future__ import annotations

import pytest

from pycalima._datasets import (
    Dataset,
    DatasetUnavailable,
    ensure_dataset,
    find_data_file,
    get_dataset,
    iter_datasets,
)

ALL_NAMES = sorted(d.name for d in iter_datasets())
VALID_KINDS = {"bundled", "fetch", "manual"}
VALID_URL_STATUS = {"confirmed", "unconfirmed", "none"}


# ---------------------------------------------------------------------------
# registry shape
# ---------------------------------------------------------------------------

def test_registry_is_not_empty():
    assert len(ALL_NAMES) >= 10, f"only {len(ALL_NAMES)} datasets registered"


def test_registry_ships_inside_the_package():
    from pycalima import _paths

    assert (_paths.PKG_DIR / "data" / "registry.toml").is_file()


@pytest.mark.parametrize("name", ALL_NAMES)
def test_every_entry_is_well_formed(name):
    ds = get_dataset(name)
    assert isinstance(ds, Dataset)
    assert ds.kind in VALID_KINDS, f"{name}: bad kind {ds.kind!r}"
    assert ds.url_status in VALID_URL_STATUS, f"{name}: bad url_status {ds.url_status!r}"
    assert ds.dest, f"{name}: no dest"
    assert ds.citation, f"{name}: no citation, so its provenance is unrecorded"
    if ds.size_bytes is not None:
        assert ds.size_bytes > 0


@pytest.mark.parametrize("name", ALL_NAMES)
def test_no_entry_carries_an_unexpected_key(name):
    """Guards against a typo in registry.toml silently doing nothing."""
    ds = get_dataset(name)
    assert not ds.extra, f"{name}: unrecognised registry keys {sorted(ds.extra)}"


def test_manual_entries_explain_how_to_obtain_the_data():
    manual = [d for d in iter_datasets() if d.kind == "manual"]
    assert manual, "expected at least the PAHdb archives to be manual"
    for ds in manual:
        assert ds.instructions, f"{ds.name} is manual but has no instructions"
        assert "calima-fetch-data" in ds.instructions, (
            f"{ds.name} instructions should name the import command"
        )


def test_fetch_entries_can_actually_be_fetched():
    for ds in iter_datasets():
        if ds.kind != "fetch":
            continue
        assert ds.url or ds.fetcher, f"{ds.name} is kind=fetch with no url or fetcher"


def test_datasets_without_a_public_url_stay_bundled():
    """A dataset with url_status='none' cannot be re-downloaded, so shipping it
    is the only option."""
    for ds in iter_datasets():
        if ds.url_status == "none":
            assert ds.kind == "bundled", (
                f"{ds.name} has no public URL but kind={ds.kind!r}"
            )


def test_pahdb_is_registered_but_not_bundled():
    """~575 MB of PAHdb must never ship inside the wheel."""
    pahdb = [d for d in iter_datasets() if "pahdb" in d.name]
    assert pahdb, "the PAHdb archives are not registered"
    for ds in pahdb:
        assert ds.kind == "manual", f"{ds.name} must not be bundled"


def test_unknown_dataset_lookup_lists_the_alternatives():
    with pytest.raises(KeyError, match="Registered"):
        get_dataset("not_a_dataset")


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------

BUNDLED_NAMES = [d.name for d in iter_datasets() if d.kind == "bundled"]


@pytest.mark.parametrize("name", BUNDLED_NAMES)
def test_every_bundled_dataset_is_actually_present(name):
    """The check that fails loudly if package-data globs dropped a tree."""
    path = ensure_dataset(name)
    assert path.is_dir(), f"{name} resolved to {path}, which is not a directory"


@pytest.mark.parametrize("name", BUNDLED_NAMES)
def test_bundled_datasets_resolve_inside_the_package(name):
    from pycalima import _paths

    path = ensure_dataset(name)
    assert path.is_relative_to(_paths.PKG_DIR), (
        f"{name} resolved outside the package: {path}"
    )


@pytest.mark.parametrize("name", BUNDLED_NAMES)
def test_bundled_datasets_verify(name):
    from pycalima._datasets import verify_dataset

    assert verify_dataset(name, strict=False) is True


def test_missing_dataset_raises_an_actionable_error(tmp_path, monkeypatch):
    monkeypatch.setenv("CALIMA_DATASETS", str(tmp_path / "empty_cache"))
    monkeypatch.chdir(tmp_path)

    manual = next(d for d in iter_datasets() if d.kind == "manual")
    with pytest.raises(DatasetUnavailable) as exc:
        ensure_dataset(manual.name)

    message = str(exc.value)
    assert manual.name in message
    assert "searched:" in message
    assert "calima-fetch-data" in message, (
        "the error must name the command that fixes it"
    )


def test_ensure_dataset_does_not_download_by_default(tmp_path, monkeypatch):
    """Library code must never start a large download as a side effect of a
    physics call."""
    monkeypatch.setenv("CALIMA_DATASETS", str(tmp_path / "empty_cache"))
    monkeypatch.chdir(tmp_path)

    fetchable = [d for d in iter_datasets() if d.kind == "fetch"]
    if not fetchable:
        pytest.skip("no fetchable datasets registered")

    ds = fetchable[0]
    # bundled copies may satisfy it; only assert when it is genuinely absent
    if ds.locate() is not None:
        pytest.skip(f"{ds.name} is already present")
    with pytest.raises(DatasetUnavailable):
        ensure_dataset(ds.name, auto_fetch=False)


def test_dataset_search_covers_both_layouts(tmp_path, monkeypatch):
    """A manually downloaded file must be found whether it sits in
    <cache>/<dest>/ or flat in <cache>/."""
    monkeypatch.setenv("CALIMA_DATASETS", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    ds = next(d for d in iter_datasets() if d.kind == "manual" and d.files)
    dirs = [str(p) for p in ds.search_dirs()]
    assert str(tmp_path) in dirs, "the flat cache directory is not searched"
    assert str(tmp_path / ds.dest) in dirs, "the dest subdirectory is not searched"


def test_import_dataset_registers_a_local_copy(tmp_path, monkeypatch):
    from pycalima._datasets import import_dataset

    monkeypatch.setenv("CALIMA_DATASETS", str(tmp_path / "cache"))
    monkeypatch.chdir(tmp_path)

    ds = next(d for d in iter_datasets() if d.kind == "manual" and len(d.files) == 1)
    fake = tmp_path / ds.files[0]
    fake.write_text("not really PAHdb\n", encoding="utf-8")

    target = import_dataset(ds.name, fake)
    assert (target / ds.files[0]).is_file()
    # and it is now discoverable
    assert ensure_dataset(ds.name) == target


def test_find_data_file_locates_bundled_reference_data():
    path = find_data_file("kp00_10000")
    assert path.is_file()


def test_find_data_file_reports_where_it_looked():
    with pytest.raises(DatasetUnavailable) as exc:
        find_data_file("definitely_not_a_reference_file.dat")
    assert "Searched:" in str(exc.value)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_list_subcommand_succeeds(capsys):
    from pycalima._datasets import main

    assert main(["list"]) == 0
    out = capsys.readouterr().out
    for name in BUNDLED_NAMES[:3]:
        assert name in out


@pytest.mark.parametrize("kind", sorted(VALID_KINDS))
def test_list_can_filter_by_kind(kind, capsys):
    from pycalima._datasets import main

    assert main(["list", "--kind", kind]) == 0


def test_verify_subcommand_reports_missing_datasets(tmp_path, monkeypatch):
    from pycalima._datasets import main

    monkeypatch.setenv("CALIMA_DATASETS", str(tmp_path / "empty"))
    monkeypatch.chdir(tmp_path)
    # the PAHdb entries will be missing, so the overall result is non-zero
    assert main(["verify"]) == 1


def test_path_subcommand_prints_a_bundled_location(capsys):
    from pycalima._datasets import main

    assert main(["path", BUNDLED_NAMES[0]]) == 0
    assert capsys.readouterr().out.strip()


def test_path_subcommand_fails_for_a_missing_dataset(tmp_path, monkeypatch, capsys):
    from pycalima._datasets import main

    monkeypatch.setenv("CALIMA_DATASETS", str(tmp_path / "empty"))
    monkeypatch.chdir(tmp_path)
    manual = next(d for d in iter_datasets() if d.kind == "manual")
    assert main(["path", manual.name]) == 1
    assert "calima-fetch-data" in capsys.readouterr().err
