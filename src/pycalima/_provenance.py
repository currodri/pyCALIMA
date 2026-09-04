"""Provenance stamping for generated tables.

Every table written by a ``pycalima.models.*.export_*`` module carries a
header naming the code revision that produced it. In a source checkout that
comes from live git; in an installed copy it comes from the distribution
version, into which ``setuptools_scm`` has baked the commit hash as a local
version segment (``0.1.1.dev83+g4a600986f``).

The pre-packaging implementation shelled out to ``git rev-parse`` in the
*caller's* working directory. Installed and run from inside any other git
repository, that stamped **that** repository's branch and commit into CALIMA
table headers -- silent provenance corruption, and worse than degrading to
``'unknown'``. This module always queries git against the package's own tree.
"""

from __future__ import annotations

import functools
import os
import subprocess
from pathlib import Path

__all__ = ["get_provenance", "get_git_info", "provenance_string"]

_PKG_DIR = Path(__file__).resolve().parent


def _live_git() -> dict[str, str] | None:
    """Query git in the package's own tree, never the process CWD."""
    for candidate in (_PKG_DIR, *_PKG_DIR.parents):
        if (candidate / ".git").exists():
            break
    else:
        return None

    def _git(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ("git", "-C", str(candidate), *args),
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=5,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            return None

    commit = _git("rev-parse", "--short", "HEAD")
    if not commit:
        return None
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))

    # Prefer the installed distribution version, which setuptools_scm renders
    # as e.g. 0.1.1.dev83+g4a600986f. `git describe` degrades to a bare hash in
    # a repository with no tags, which would just repeat `commit`.
    dist = _from_metadata()
    version = dist["version"] if dist else (
        _git("describe", "--tags", "--always", "--dirty") or "unknown"
    )

    return {
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD") or "unknown",
        "commit": commit + ("-dirty" if dirty else ""),
        "commit_full": _git("rev-parse", "HEAD") or "unknown",
        "version": version,
        "source": f"git:{candidate}",
    }


def _from_metadata() -> dict[str, str] | None:
    """Recover the commit from the installed distribution version.

    setuptools_scm's ``node-and-date`` local scheme puts the short hash in the
    local version segment, e.g. ``0.1.1.dev83+g4a600986f``.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        ver = version("pycalima")
    except PackageNotFoundError:
        return None

    commit = "unknown"
    if "+" in ver:
        for token in ver.split("+", 1)[1].split("."):
            if token.startswith("g") and len(token) >= 8:
                commit = token[1:]
                break
    return {
        "branch": "n/a (installed)",
        "commit": commit,
        "commit_full": commit,
        "version": ver,
        "source": "importlib.metadata",
    }


@functools.lru_cache(maxsize=1)
def get_provenance() -> dict[str, str]:
    """Best available description of the running code revision.

    ``$CALIMA_PROVENANCE`` (a verbatim override, for reproducing an old run or
    for CI) -> live git against the package tree -> the installed distribution
    version -> ``'unknown'``.

    Returns a dict with keys ``branch``, ``commit``, ``commit_full``,
    ``version`` and ``source``.
    """
    override = os.environ.get("CALIMA_PROVENANCE")
    if override:
        return {
            "branch": "override",
            "commit": override,
            "commit_full": override,
            "version": override,
            "source": "CALIMA_PROVENANCE",
        }
    return (
        _live_git()
        or _from_metadata()
        or {
            "branch": "unknown",
            "commit": "unknown",
            "commit_full": "unknown",
            "version": "unknown",
            "source": "none",
        }
    )


def get_git_info() -> tuple[str, str]:
    """``(branch, short_commit)``.

    Backwards-compatible shim so that ``grain_size_config.get_header_lines``
    and the exporters calling it need no change.
    """
    p = get_provenance()
    return p["branch"], p["commit"]


def provenance_string() -> str:
    """One-line stamp for table headers.

    ``pycalima 0.1.1.dev83+g4a600986f (main @ 4a600986)``
    """
    p = get_provenance()
    return f"pycalima {p['version']} ({p['branch']} @ {p['commit']})"
