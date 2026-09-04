"""Registry-driven reference-dataset management.

pyCALIMA needs a mix of reference data: a few MB that ships inside the wheel,
and a few hundred MB of PAHdb archives that cannot. ``data/registry.toml``
declares every dataset in one place, and each entry carries a ``kind``:

``bundled``
    Ships in the wheel. :func:`ensure_dataset` is a presence check.
``fetch``
    Downloadable from ``url`` (or via a named ``fetcher``), checksum-verified.
``manual``
    The user must obtain it themselves -- registration, a DOI, or a
    per-download link. :func:`ensure_dataset` fails with the entry's
    ``instructions``, and ``calima-fetch-data import`` registers a local copy.

Moving a dataset between those is a one-field edit in the registry, so the
bundle/fetch boundary can move without touching code.

The point of :func:`ensure_dataset` is that a missing dataset produces an
actionable error naming the exact command to run, never a bare
FileNotFoundError from somewhere deep inside physics code.

CLI::

    calima-fetch-data list [--missing] [--kind KIND]
    calima-fetch-data path NAME
    calima-fetch-data fetch NAME | --all
    calima-fetch-data verify [NAME]
    calima-fetch-data import NAME PATH [--link]
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

from pycalima import _paths

__all__ = [
    "Dataset",
    "DatasetUnavailable",
    "ensure_dataset",
    "get_dataset",
    "iter_datasets",
    "fetch_dataset",
    "import_dataset",
    "verify_dataset",
    "find_data_file",
]


class DatasetUnavailable(FileNotFoundError):
    """A required dataset is not present and could not be obtained."""


@dataclass(frozen=True)
class Dataset:
    """One registry entry."""

    name: str
    kind: str
    dest: str
    files: tuple[str, ...] = ()
    url: str | None = None
    url_status: str = "unconfirmed"
    sha256: str | None = None
    size_bytes: int | None = None
    citation: str | None = None
    instructions: str | None = None
    fetcher: str | None = None
    env_override: str | None = None
    notes: str | None = None
    extra: dict = field(default_factory=dict, repr=False)

    # -- locations ---------------------------------------------------------

    def bundled_dir(self) -> Path:
        return _paths.get_data_root() / self.dest

    def cache_dir(self) -> Path:
        return _paths.get_dataset_cache_dir() / self.dest

    def target_dir(self) -> Path:
        """Where a fetch or import should place this dataset."""
        return self.bundled_dir() if self.kind == "bundled" else self.cache_dir()

    def search_dirs(self) -> list[Path]:
        """Every directory that might already hold this dataset.

        Both the ``<root>/<dest>`` layout that a fetch or import creates and
        the flat ``<root>/`` layout that an existing checkout already has are
        accepted, so a manually downloaded file does not have to be moved into
        a subdirectory to be found.
        """
        dirs: list[Path] = []
        if self.env_override:
            raw = os.environ.get(self.env_override)
            if raw:
                dirs.append(Path(raw).expanduser())
        try:
            dirs.append(self.bundled_dir())
            dirs.append(_paths.get_data_root())
        except _paths.MissingReferenceData:
            pass
        cache = _paths.get_dataset_cache_dir()
        dirs.append(cache / self.dest)
        dirs.append(cache)
        # Historic layout: a checkout's own reference_data/ directory.
        dirs.append(Path.cwd() / "reference_data")
        out, seen = [], set()
        for d in dirs:
            if str(d) not in seen:
                seen.add(str(d))
                out.append(d)
        return out

    def is_complete_in(self, directory: Path) -> bool:
        if not directory.is_dir():
            return False
        if not self.files:
            # No explicit file list: any content counts as present.
            return any(directory.iterdir())
        return all((directory / f).is_file() for f in self.files)

    def locate(self) -> Path | None:
        """The first directory that actually holds this dataset, or None."""
        for d in self.search_dirs():
            if self.is_complete_in(d):
                return d
        return None

    # -- errors ------------------------------------------------------------

    def unavailable_message(self) -> str:
        lines = [f"Required dataset {self.name!r} is not available locally."]
        if self.size_bytes:
            lines.append(f"  size:     {self.size_bytes / 1e6:.1f} MB")
        if self.citation:
            lines.append(f"  citation: {self.citation}")
        lines.append("  searched:")
        lines += [f"    {d}" for d in self.search_dirs()]
        if self.kind == "fetch":
            lines.append("")
            lines.append(f"  Download it with:\n    calima-fetch-data fetch {self.name}")
        elif self.kind == "manual":
            lines.append("")
            if self.instructions:
                lines.append("  " + self.instructions.strip().replace("\n", "\n  "))
            else:
                lines.append(
                    f"  Obtain it from {self.url or 'its upstream source'} and register it:\n"
                    f"    calima-fetch-data import {self.name} /path/to/file"
                )
        else:
            lines.append("")
            lines.append(
                "  This dataset is supposed to ship inside the package, so the "
                "installation looks incomplete. Try reinstalling pycalima, or "
                "set $CALIMA_BUNDLED_DATA to a source checkout."
            )
        if self.env_override:
            lines.append(f"  You can also set ${self.env_override} to an existing copy.")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# registry loading
# ---------------------------------------------------------------------------

_KNOWN = {
    "kind", "dest", "files", "url", "url_status", "sha256", "size_bytes",
    "citation", "instructions", "fetcher", "env_override", "notes",
}


def _load_toml(path: Path) -> dict:
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        import tomli as tomllib  # type: ignore[no-redef]
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _registry_paths() -> list[Path]:
    """The shipped registry, then an optional user sidecar that overrides it."""
    paths = [_paths.PKG_DIR / "data" / "registry.toml"]
    sidecar = _paths.get_dataset_cache_dir() / "registry.toml"
    if sidecar.is_file():
        paths.append(sidecar)
    env = os.environ.get("CALIMA_DATA")
    if env:
        p = Path(env).expanduser() / "registry.toml"
        if p.is_file() and p not in paths:
            paths.append(p)
    return paths


_CACHE: dict[str, Dataset] | None = None


def _registry() -> dict[str, Dataset]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    merged: dict[str, dict] = {}
    for path in _registry_paths():
        if not path.is_file():
            continue
        raw = _load_toml(path)
        for name, entry in (raw.get("datasets") or {}).items():
            merged.setdefault(name, {}).update(entry)
    out: dict[str, Dataset] = {}
    for name, entry in merged.items():
        known = {k: v for k, v in entry.items() if k in _KNOWN}
        extra = {k: v for k, v in entry.items() if k not in _KNOWN}
        files = tuple(known.pop("files", ()) or ())
        out[name] = Dataset(name=name, files=files, extra=extra, **known)
    _CACHE = out
    return out


def iter_datasets() -> Iterator[Dataset]:
    """Every registered dataset, in registry order."""
    return iter(_registry().values())


def get_dataset(name: str) -> Dataset:
    """Look up one dataset by name."""
    reg = _registry()
    if name not in reg:
        raise KeyError(
            f"Unknown dataset {name!r}. Registered: {', '.join(sorted(reg))}"
        )
    return reg[name]


# ---------------------------------------------------------------------------
# the main entry point for library code
# ---------------------------------------------------------------------------

def ensure_dataset(name: str, *, auto_fetch: bool = False) -> Path:
    """Return the directory holding *name*, or raise an actionable error.

    Parameters
    ----------
    name
        A registry key, e.g. ``"pahdb-theoretical-v4-00"``.
    auto_fetch
        Download a ``kind="fetch"`` dataset if it is missing. Off by default:
        library code should not start a multi-hundred-MB download as a side
        effect of a physics call.

    Raises
    ------
    DatasetUnavailable
        With a message naming the exact command to run.
    """
    ds = get_dataset(name)
    found = ds.locate()
    if found is not None:
        return found
    if auto_fetch and ds.kind == "fetch":
        return fetch_dataset(ds)
    raise DatasetUnavailable(ds.unavailable_message())


def find_data_file(*parts: str) -> Path:
    """Find one reference file in the bundle, then the fetch cache.

    A convenience for readers that know a filename but not which dataset owns
    it.
    """
    rel = Path(*parts)
    candidates = [
        _paths.get_external_data_path(*parts),
        _paths.get_dataset_cache_dir() / rel,
        Path.cwd() / "reference_data" / rel,
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise DatasetUnavailable(
        f"Reference file not found: {rel}\nSearched:\n"
        + "\n".join(f"  {c}" for c in candidates)
        + "\nRun `calima-fetch-data list` to see the registered datasets."
    )


# ---------------------------------------------------------------------------
# fetch / import / verify
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, expected_size: int | None = None) -> None:
    """Download *url* to *dest* via stdlib urllib, with a progress line."""
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  downloading {url}")
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as out:  # noqa: S310
        total = int(resp.headers.get("Content-Length") or expected_size or 0)
        done = 0
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            done += len(chunk)
            if total:
                pct = 100.0 * done / total
                print(f"\r    {done / 1e6:8.1f} / {total / 1e6:.1f} MB  ({pct:5.1f}%)",
                      end="", flush=True)
        if total:
            print()
    tmp.replace(dest)
    print(f"    saved {dest}")


def fetch_dataset(ds: Dataset | str, *, force: bool = False) -> Path:
    """Download a ``kind="fetch"`` dataset into the cache."""
    ds = ds if isinstance(ds, Dataset) else get_dataset(ds)

    if ds.kind == "bundled":
        found = ds.locate()
        if found:
            print(f"{ds.name}: bundled, already present ({found})")
            return found
        raise DatasetUnavailable(ds.unavailable_message())
    if ds.kind == "manual":
        raise DatasetUnavailable(ds.unavailable_message())

    target = ds.target_dir()
    if not force and ds.is_complete_in(target):
        print(f"{ds.name}: already present ({target})")
        return target

    if ds.fetcher:
        import importlib

        module_name, _, func_name = ds.fetcher.partition(":")
        func = getattr(importlib.import_module(module_name), func_name)
        target.mkdir(parents=True, exist_ok=True)
        func(output_dir=target)
        return target

    if not ds.url:
        raise DatasetUnavailable(
            f"{ds.name} is kind='fetch' but has neither url nor fetcher in the "
            f"registry. That is a registry bug."
        )
    if ds.url_status != "confirmed":
        print(
            f"  warning: {ds.name} has url_status={ds.url_status!r} -- its "
            f"upstream URL has not been verified to serve these exact bytes.",
            file=sys.stderr,
        )

    names = ds.files or (ds.url.rsplit("/", 1)[-1],)
    for fname in names:
        url = ds.url if ds.url.endswith(fname) else f"{ds.url.rstrip('/')}/{fname}"
        _download(url, target / fname, ds.size_bytes)
    verify_dataset(ds, strict=bool(ds.sha256))
    return target


def _place(src: Path, dst: Path, link: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if link:
        dst.symlink_to(src)
    else:
        shutil.copy2(src, dst)


def import_dataset(
    ds: Dataset | str, source: str | os.PathLike[str], *, link: bool = False
) -> Path:
    """Register a locally obtained copy of a ``manual`` dataset."""
    ds = ds if isinstance(ds, Dataset) else get_dataset(ds)
    src = Path(source).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(src)

    target = ds.target_dir()
    target.mkdir(parents=True, exist_ok=True)

    if src.is_dir():
        for item in src.iterdir():
            if item.is_file():
                _place(item, target / item.name, link)
    else:
        expected = ds.files[0] if len(ds.files) == 1 else src.name
        if src.name != expected:
            print(f"  note: registering {src.name} under the expected name {expected}")
        _place(src, target / expected, link)

    verify_dataset(ds, strict=False)
    print(f"{ds.name}: registered in {target}")
    return target


def verify_dataset(ds: Dataset | str, *, strict: bool = True) -> bool:
    """Check presence, and the sha256 when the registry records one."""
    ds = ds if isinstance(ds, Dataset) else get_dataset(ds)
    found = ds.locate()
    if found is None:
        print(f"{ds.name}: MISSING")
        return False

    if ds.sha256 and len(ds.files) == 1:
        actual = _sha256(found / ds.files[0])
        if actual != ds.sha256:
            msg = (f"{ds.name}: CHECKSUM MISMATCH\n"
                   f"  expected {ds.sha256}\n  actual   {actual}")
            if strict:
                raise DatasetUnavailable(msg)
            print(msg)
            return False

    print(f"{ds.name}: OK ({found})")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser():
    """Argument parser for ``calima-fetch-data``.

    Split out from :func:`main` so that the documentation can render it, with
    its five subcommands; see ``docs/cli/calima-fetch-data.md``.
    """
    import argparse

    p = argparse.ArgumentParser(
        prog="calima-fetch-data",
        description="Inspect, download and register pyCALIMA reference datasets.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("list", help="list registered datasets")
    pl.add_argument("--missing", action="store_true", help="only unavailable ones")
    pl.add_argument("--kind", choices=("bundled", "fetch", "manual"))

    pp = sub.add_parser("path", help="print the resolved directory for one dataset")
    pp.add_argument("name")

    pf = sub.add_parser("fetch", help="download a fetchable dataset")
    pf.add_argument("name", nargs="?")
    pf.add_argument("--all", action="store_true", help="every kind='fetch' dataset")
    pf.add_argument("--force", action="store_true")

    pv = sub.add_parser("verify", help="verify presence and checksums")
    pv.add_argument("name", nargs="?")

    pi = sub.add_parser("import", help="register a locally obtained dataset")
    pi.add_argument("name")
    pi.add_argument("source")
    pi.add_argument("--link", action="store_true", help="symlink instead of copy")

    return p


def main(argv: Sequence[str] | None = None) -> int:
    """``calima-fetch-data`` entry point."""
    a = _build_parser().parse_args(list(argv) if argv is not None else None)

    if a.cmd == "list":
        rows = []
        for ds in iter_datasets():
            if a.kind and ds.kind != a.kind:
                continue
            here = ds.locate()
            if a.missing and here is not None:
                continue
            size = f"{ds.size_bytes / 1e6:.1f} MB" if ds.size_bytes else "-"
            rows.append((ds.name, ds.kind, size,
                         "present" if here else "MISSING", str(here or "")))
        if not rows:
            print("nothing to report")
            return 0
        w = [max(len(r[i]) for r in rows) for i in range(4)]
        for r in rows:
            print(f"{r[0]:<{w[0]}}  {r[1]:<{w[1]}}  {r[2]:>{w[2]}}  "
                  f"{r[3]:<{w[3]}}  {r[4]}")
        return 0

    if a.cmd == "path":
        ds = get_dataset(a.name)
        here = ds.locate()
        if here is None:
            print(ds.unavailable_message(), file=sys.stderr)
            return 1
        print(here)
        return 0

    if a.cmd == "fetch":
        if not a.all and a.name is None:
            p.error("give a dataset name or --all")
        targets = ([d for d in iter_datasets() if d.kind == "fetch"]
                   if a.all else [get_dataset(a.name)])
        rc = 0
        for ds in targets:
            try:
                fetch_dataset(ds, force=a.force)
            except DatasetUnavailable as exc:
                print(exc, file=sys.stderr)
                rc = 1
        return rc

    if a.cmd == "verify":
        targets = [get_dataset(a.name)] if a.name else list(iter_datasets())
        return 0 if all(verify_dataset(d, strict=False) for d in targets) else 1

    if a.cmd == "import":
        import_dataset(a.name, a.source, link=a.link)
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
