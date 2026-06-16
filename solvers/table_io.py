"""I/O helpers for pre-computed sputtering rate tables.

Reads the Fortran-friendly 2-D T–φ tables produced by
``models/dust_gas_collisions/dust_sputtering.py::export_rates_T_phi()`` and
builds ``scipy`` interpolators for use in the ODE rate kernels.

Table file format
-----------------
Line 1  : ``nT  nphi``
Line 2  : ``phi[0]  phi[1]  …  phi[nphi-1]``   (values in eV, linear spacing)
Lines 3…nT+2 : ``log10(T[i])  log10(rate[i,0])  …  log10(rate[i,nphi-1])``

The stored rate is ``(1/n_ion) × da/dt`` in units of  **µm yr⁻¹ cm³**,
where ``n_ion`` is the number density of the sputtering ion species [cm⁻³].
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator


# ---------------------------------------------------------------------------
# Low-level reader
# ---------------------------------------------------------------------------

def read_sputtering_table(
    table_file: str | Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one sputtering T–φ table file.

    Parameters
    ----------
    table_file :
        Path to a ``thermal_sputtering_<label>_Z_<Z>`` file.

    Returns
    -------
    Tgas : ndarray, shape (nT,)
        Temperature grid [K].
    phi_grid : ndarray, shape (nphi,)
        Grain electric potential grid [eV].
    rates : ndarray, shape (nT, nphi)
        Sputtering rate ``(1/n_ion) × da/dt``  [µm yr⁻¹ cm³].
    """
    path = Path(table_file)
    with path.open("r", encoding="utf-8") as fh:
        lines = [ln.strip() for ln in fh if ln.strip() and not ln.strip().startswith('#')]

    if len(lines) < 3:
        raise ValueError(f"Table file too short: {path}")

    # --- Header ---
    header = lines[0].split()
    nT, nphi = int(header[0]), int(header[1])

    # --- φ grid ---
    phi_grid = np.array([float(x) for x in lines[1].split()], dtype=np.float64)
    if phi_grid.size != nphi:
        raise ValueError(
            f"{path}: phi_grid has {phi_grid.size} values, expected {nphi}"
        )

    # --- Data rows ---
    if len(lines) < 2 + nT:
        raise ValueError(
            f"{path}: expected {nT} data rows, found {len(lines) - 2}"
        )

    log_T = np.empty(nT, dtype=np.float64)
    log_rates = np.empty((nT, nphi), dtype=np.float64)
    for i, ln in enumerate(lines[2 : 2 + nT]):
        parts = [float(x) for x in ln.split()]
        log_T[i] = parts[0]
        log_rates[i, :] = parts[1 : 1 + nphi]

    Tgas = 10.0 ** log_T
    rates = 10.0 ** log_rates
    return Tgas, phi_grid, rates


# ---------------------------------------------------------------------------
# Interpolator builder
# ---------------------------------------------------------------------------

def build_sputtering_interpolator(
    table_file: str | Path,
    use_log_rates: bool = True,
) -> Tuple[Callable, dict]:
    """Build a bilinear interpolator for a sputtering T–φ table.

    Parameters
    ----------
    table_file :
        Path to a ``thermal_sputtering_<label>_Z_<Z>`` file.
    use_log_rates : bool
        Interpolate ``log₁₀(rate)`` rather than the raw rate for better
        accuracy across the many orders of magnitude spanned by the table.

    Returns
    -------
    evaluate : callable
        ``evaluate(T, phi=0.0)`` → rate [µm yr⁻¹ cm³].
        Accepts scalars or arrays.  Out-of-bounds values are clamped to the
        table boundary (no exception raised).
    info : dict
        Keys ``Tgas``, ``phi_grid``, ``rates``.
    """
    Tgas, phi_grid, rates = read_sputtering_table(table_file)
    log_T_axis = np.log10(Tgas)

    if use_log_rates:
        pos = rates > 0.0
        floor = float(np.min(rates[pos])) if pos.any() else 1.0e-100
        values = np.log10(np.maximum(rates, floor))
        fill_value = float(np.log10(floor))
    else:
        values = rates
        floor = 0.0
        fill_value = 0.0

    interp = RegularGridInterpolator(
        (log_T_axis, phi_grid),
        values,
        method="linear",
        bounds_error=False,
        fill_value=fill_value,
    )

    def evaluate(T_query, phi_query: float = 0.0):
        scalar = np.ndim(T_query) == 0
        T_arr = np.atleast_1d(np.asarray(T_query, dtype=np.float64))
        phi_arr = np.full(T_arr.shape, float(phi_query), dtype=np.float64)
        pts = np.column_stack((np.log10(T_arr), phi_arr))
        result = interp(pts)
        if use_log_rates:
            result = 10.0 ** result
        return float(result[0]) if scalar else result

    info = {"Tgas": Tgas, "phi_grid": phi_grid, "rates": rates}
    return evaluate, info


# ---------------------------------------------------------------------------
# PAH photolysis (2-D G0 × nH table)
# ---------------------------------------------------------------------------

def read_pah_photolysis_table(
    table_file: str | Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a PAH photolysis dissociation rate table.

    File format
    -----------
    Line 1  : ``nG0  nNH``
    Lines 2 … nG0+1           : log₁₀(G₀) axis values
    Lines nG0+2 … nG0+nNH+1   : log₁₀(nH) axis values
    Lines nG0+nNH+2 … end     : log₁₀(rate [s⁻¹]) values,
                                 stored row-major (G₀ varies slowly)

    Returns
    -------
    log_G0 : ndarray, shape (nG0,)
    log_nH : ndarray, shape (nNH,)
    log_rate : ndarray, shape (nG0, nNH)
    """
    path = Path(table_file)
    with path.open("r", encoding="utf-8") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]

    nG0, nNH = map(int, lines[0].split())
    if len(lines) < 1 + nG0 + nNH + nG0 * nNH:
        raise ValueError(f"{path}: file has {len(lines)} lines, expected {1 + nG0 + nNH + nG0 * nNH}")

    log_G0 = np.array([float(lines[1 + i]) for i in range(nG0)], dtype=np.float64)
    log_nH = np.array([float(lines[1 + nG0 + i]) for i in range(nNH)], dtype=np.float64)
    offset = 1 + nG0 + nNH
    log_rate = np.array(
        [float(lines[offset + i]) for i in range(nG0 * nNH)],
        dtype=np.float64,
    ).reshape(nG0, nNH)
    return log_G0, log_nH, log_rate


def build_pah_photolysis_interpolator(
    table_file: str | Path,
) -> Tuple[Callable, dict]:
    """Build a bilinear interpolator for a PAH photolysis table.

    Returns
    -------
    evaluate : callable
        ``evaluate(log10_G0, log10_nH)`` → log₁₀(rate [s⁻¹]).
        Out-of-bounds values are clamped to the table boundary.
    info : dict
        Keys ``log_G0``, ``log_nH``, ``log_rate``.
    """
    log_G0, log_nH, log_rate = read_pah_photolysis_table(table_file)

    fill_value = float(log_rate.min())
    interp = RegularGridInterpolator(
        (log_G0, log_nH),
        log_rate,
        method="linear",
        bounds_error=False,
        fill_value=fill_value,
    )

    def evaluate(lG0: float, lnH: float) -> float:
        pts = np.array([[float(lG0), float(lnH)]])
        return float(interp(pts)[0])

    info = {"log_G0": log_G0, "log_nH": log_nH, "log_rate": log_rate}
    return evaluate, info


# ---------------------------------------------------------------------------
# PAH sputtering (1-D T table per ion species)
# ---------------------------------------------------------------------------

def read_pah_sputtering_table(
    table_file: str | Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """Read a PAH sputtering rate table (1-D in temperature).

    File format
    -----------
    Line 1      : ``nT``
    Lines 2 … nT+1   : log₁₀(T [K]) values
    Lines nT+2 … 2nT+1 : log₁₀(J [cm³ s⁻¹]) sputtering rate constant

    Returns
    -------
    log_T : ndarray, shape (nT,)
    log_J : ndarray, shape (nT,)
        log₁₀(J) where J is the sputtering rate constant per ion/electron
        [cm³ s⁻¹].
    """
    path = Path(table_file)
    with path.open("r", encoding="utf-8") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]

    nT = int(lines[0])
    if len(lines) < 1 + 2 * nT:
        raise ValueError(f"{path}: file has {len(lines)} lines, expected {1 + 2 * nT}")

    log_T = np.array([float(lines[1 + i]) for i in range(nT)], dtype=np.float64)
    log_J = np.array([float(lines[1 + nT + i]) for i in range(nT)], dtype=np.float64)
    return log_T, log_J


def build_pah_sputtering_interpolator(
    table_file: str | Path,
) -> Tuple[Callable, dict]:
    """Build a 1-D temperature interpolator for a PAH sputtering rate table.

    Returns
    -------
    evaluate : callable
        ``evaluate(T)`` → J [cm³ s⁻¹] (sputtering rate constant).
        Out-of-bounds values are clamped to the table boundary.
    info : dict
        Keys ``log_T``, ``log_J``.
    """
    log_T, log_J = read_pah_sputtering_table(table_file)

    fill_lo = float(log_J[0])
    fill_hi = float(log_J[-1])

    def evaluate(T_query: float) -> float:
        lT = math.log10(max(T_query, 1.0))
        if lT <= log_T[0]:
            return 10.0 ** fill_lo
        if lT >= log_T[-1]:
            return 10.0 ** fill_hi
        # Linear interpolation in log-space
        idx = int(np.searchsorted(log_T, lT)) - 1
        idx = max(0, min(idx, len(log_T) - 2))
        frac = (lT - log_T[idx]) / (log_T[idx + 1] - log_T[idx])
        lJ = log_J[idx] + frac * (log_J[idx + 1] - log_J[idx])
        return 10.0 ** lJ

    info = {"log_T": log_T, "log_J": log_J}
    return evaluate, info
