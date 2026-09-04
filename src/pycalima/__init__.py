"""pyCALIMA - dust and PAH microphysics for the interstellar medium.

Two layers:

``pycalima.models``
    Physics modules computing rates, cross-sections and grain/PAH properties,
    organised by physical domain (``dust_charge``, ``dust_radiation``,
    ``PAH_photophysics``, ...). Their ``export_*`` modules write precomputed
    lookup tables into the generated-data directory.

``pycalima.solvers``
    ODE and steady-state solvers that consume those tables and integrate
    dust/PAH mass evolution. Mirrors the RAMSES-CALIMA Fortran modules.

``pycalima.galaxysam``
    A semi-analytic galaxy chemical-evolution model, used by the dust-yield
    table builders.

Where data lives is resolved entirely by :mod:`pycalima._paths`; run
``calima-paths`` to see the locations in effect for your environment.

This module deliberately re-exports nothing. Several subpackages contain
circular dependencies that are resolved only by deferred, function-local
imports, and any eager re-export here would turn a working lazy cycle into an
ImportError.
"""

from __future__ import annotations

try:  # pragma: no cover - trivial
    from importlib.metadata import version

    __version__ = version("pycalima")
except Exception:  # pragma: no cover - not installed as a distribution
    __version__ = "0.0.0.dev0+unknown"

__all__ = ["__version__"]
