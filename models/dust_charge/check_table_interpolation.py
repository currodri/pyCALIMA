"""Check legacy dust-charge and photoelectric-heating table interpolation.

This script reads the ASCII ``.dat`` tables written for the Fortran workflow and
interpolates the requested grain bin at a given temperature ``T`` and charging
parameter ``gamma``.

Examples
--------
python -m models.dust_charge.check_table_interpolation DustBin_01 1e3 1e-2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _safe_log10(values: np.ndarray | float) -> np.ndarray | float:
    return np.log10(values)


def _resolve_paths(data_root: Path, grain_bin: str) -> dict[str, Path]:
    charging_dir = data_root / "dust_charging_data"
    heating_dir = data_root / "dust_photoelectric_heating_data"

    charge_path = charging_dir / f"dust_charge_Z_vs_T_{grain_bin}"
    sigma_path = charging_dir / f"dust_charge_sigma_vs_T_{grain_bin}"
    peh_path = heating_dir / f"dust_rates_peh_{grain_bin}.dat"
    rec_path = heating_dir / f"dust_rates_rec_{grain_bin}.dat"
    grid_path = heating_dir / "photoelectric_grid_fix_G0.dat"
    grid_t_path = heating_dir / f"log10_Ts_{grain_bin}.dat"
    grid_g_path = heating_dir / f"log10_gammas_{grain_bin}.dat"

    paths = {
        "charge": charge_path,
        "sigma": sigma_path,
        "grid": grid_path,
        "grid_t": grid_t_path,
        "grid_g": grid_g_path,
        "peh": peh_path,
        "rec": rec_path,
    }
    required = ["charge", "sigma", "peh", "rec"]
    missing = [name for name in required if not paths[name].exists()]
    if missing:
        missing_paths = ", ".join(f"{name}={paths[name]}" for name in missing)
        raise FileNotFoundError(f"Missing table files for {grain_bin}: {missing_paths}")
    return paths


def _read_charging_table(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    payload = [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if len(payload) < 3:
        raise ValueError(f"Unexpected charging table layout in {path}")
    # The count line historically used the order (ngamma nT) but the
    # unified shared-grid format uses (nT n_gamma). Be flexible and
    # detect which ordering is present by comparing axis lengths.
    a, b = map(int, payload[0].split()[:2])
    arr1 = np.fromstring(payload[1], sep=" ")
    arr2 = np.fromstring(payload[2], sep=" ")

    # Determine whether payload[1] is temp (nT) and payload[2] is gamma (n_gamma)
    if arr1.size == a and arr2.size == b:
        ntemp, ngamma = a, b
        temp_log = arr1
        gamma_log = arr2
        data_start = 3
    elif arr1.size == b and arr2.size == a:
        # swapped count ordering: payload[0] was (ngamma nT)
        ngamma, ntemp = a, b
        temp_log = arr2
        gamma_log = arr1
        data_start = 3
    else:
        # If shapes do not match, try interpreting counts as (nT n_gamma)
        ntemp, ngamma = a, b
        temp_log = arr1
        gamma_log = arr2
        data_start = 3

    data = np.array([np.fromstring(row, sep=" ") for row in payload[data_start:data_start + ngamma]], dtype=float)

    if temp_log.size != ntemp or gamma_log.size != ngamma or data.shape != (ngamma, ntemp):
        raise ValueError(f"Charging table shape mismatch in {path}")
    return gamma_log, temp_log, data


def _read_grid_file(grid_path: Path, fallback_t_path: Path, fallback_g_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if grid_path.exists():
        payload = [line.strip() for line in grid_path.read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")]
        if len(payload) < 3:
            raise ValueError(f"Unexpected shared grid layout in {grid_path}")

        ntemp, ngamma = map(int, payload[0].split()[:2])
        temp_log = np.fromstring(payload[1], sep=" ")
        gamma_log = np.fromstring(payload[2], sep=" ")
        if temp_log.size != ntemp or gamma_log.size != ngamma:
            raise ValueError(f"Shared grid shape mismatch in {grid_path}")
        return np.asarray(temp_log, dtype=float), np.asarray(gamma_log, dtype=float)

    temp_log = np.loadtxt(fallback_t_path, dtype=float)
    gamma_log = np.loadtxt(fallback_g_path, dtype=float)
    return np.asarray(temp_log, dtype=float), np.asarray(gamma_log, dtype=float)


def _read_rate_table(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    payload = [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if len(payload) < 2:
        raise ValueError(f"Unexpected rate table layout in {path}")

    a, b = map(int, payload[0].split()[:2])

    # Try to detect whether the rate file also contains embedded axis lines
    # after the count. If so, payload[1] is the temp axis and payload[2]
    # is the gamma axis, and data rows follow starting at payload[3].
    arr1 = np.fromstring(payload[1], sep=" ")
    if len(payload) >= 3:
        arr2 = np.fromstring(payload[2], sep=" ")
    else:
        arr2 = np.array([])

    # Case 1: embedded axes present (count is nT n_gamma)
    if arr1.size == a and arr2.size == b:
        ntemp, ngamma = a, b
        temp_log = arr1
        gamma_log = arr2
        data_start = 3
        data = np.array([np.fromstring(row, sep=" ") for row in payload[data_start:data_start + ngamma]], dtype=float)
        if data.shape != (ngamma, ntemp):
            raise ValueError(f"Rate table shape mismatch in {path}")
        return temp_log, gamma_log, data

    # Case 2: legacy format: first non-comment line is "ngamma nT" and data rows follow
    ngamma, ntemp = a, b
    data = np.array([np.fromstring(row, sep=" ") for row in payload[1:1 + ngamma]], dtype=float)
    if data.shape != (ngamma, ntemp):
        raise ValueError(f"Rate table shape mismatch in {path}")
    return np.array([]), np.array([]), data


def _bilinear_interpolate(x_grid: np.ndarray, y_grid: np.ndarray, values: np.ndarray, x: float, y: float) -> tuple[float, bool]:
    if x_grid.ndim != 1 or y_grid.ndim != 1:
        raise ValueError("Interpolation grids must be one-dimensional")
    if values.shape != (x_grid.size, y_grid.size):
        raise ValueError(f"Table shape {values.shape} does not match grids {(x_grid.size, y_grid.size)}")

    x_clipped = float(np.clip(x, x_grid[0], x_grid[-1]))
    y_clipped = float(np.clip(y, y_grid[0], y_grid[-1]))
    clipped = (x_clipped != x) or (y_clipped != y)

    ix = int(np.searchsorted(x_grid, x_clipped, side="right") - 1)
    iy = int(np.searchsorted(y_grid, y_clipped, side="right") - 1)
    ix = int(np.clip(ix, 0, x_grid.size - 2))
    iy = int(np.clip(iy, 0, y_grid.size - 2))

    x0 = float(x_grid[ix])
    x1 = float(x_grid[ix + 1])
    y0 = float(y_grid[iy])
    y1 = float(y_grid[iy + 1])

    tx = 0.0 if x1 == x0 else (x_clipped - x0) / (x1 - x0)
    ty = 0.0 if y1 == y0 else (y_clipped - y0) / (y1 - y0)

    v00 = float(values[ix, iy])
    v10 = float(values[ix + 1, iy])
    v01 = float(values[ix, iy + 1])
    v11 = float(values[ix + 1, iy + 1])

    value = (
        (1.0 - tx) * (1.0 - ty) * v00
        + tx * (1.0 - ty) * v10
        + (1.0 - tx) * ty * v01
        + tx * ty * v11
    )
    return value, clipped


def evaluate_tables(grain_bin: str, temperature: float, gamma: float, data_root: Path) -> dict[str, float | bool | str]:
    paths = _resolve_paths(data_root, grain_bin)

    gamma_log_charge, temp_log_charge, zmean_grid = _read_charging_table(paths["charge"])
    _, _, zsigma_grid = _read_charging_table(paths["sigma"])

    # Attempt to extract axes directly from the photoelectric rate file first
    temp_log_rate, gamma_log_rate, peh_log_grid = _read_rate_table(paths["peh"])
    if temp_log_rate.size == 0 or gamma_log_rate.size == 0:
        # Fallback to explicit shared-grid files if the rate file did not embed axes
        temp_log_rate, gamma_log_rate = _read_grid_file(paths["grid"], paths["grid_t"], paths["grid_g"])
        # re-read rate data (will return empty axes and the data grid)
        _, _, peh_log_grid = _read_rate_table(paths["peh"])

    _, _, rec_log_grid = _read_rate_table(paths["rec"])

    log_t = float(_safe_log10(temperature))
    log_g = float(_safe_log10(gamma))

    zmean, clipped_z = _bilinear_interpolate(gamma_log_charge, temp_log_charge, zmean_grid, log_g, log_t)
    zsigma, clipped_s = _bilinear_interpolate(gamma_log_charge, temp_log_charge, zsigma_grid, log_g, log_t)
    peh_log, clipped_peh = _bilinear_interpolate(gamma_log_rate, temp_log_rate, peh_log_grid, log_g, log_t)
    rec_log, clipped_rec = _bilinear_interpolate(gamma_log_rate, temp_log_rate, rec_log_grid, log_g, log_t)

    return {
        "grain_bin": grain_bin,
        "temperature": temperature,
        "gamma": gamma,
        "log10_temperature": log_t,
        "log10_gamma": log_g,
        "zmean": zmean,
        "zsigma": zsigma,
        "peh_rate": float(10.0 ** peh_log),
        "rec_rate": float(10.0 ** rec_log),
        "clipped": clipped_z or clipped_s or clipped_peh or clipped_rec,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interpolate legacy dust charging and photoelectric-heating tables for one grain bin."
    )
    parser.add_argument("grain_bin", help="Grain bin name, for example DustBin_01")
    parser.add_argument("temperature", type=float, help="Gas temperature T in K")
    parser.add_argument("gamma", type=float, help="Charging parameter gamma = G0*sqrt(T)/ne")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "model_data",
        help="Path to model_data (default: repo model_data directory)",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    result = evaluate_tables(
        grain_bin=args.grain_bin,
        temperature=float(args.temperature),
        gamma=float(args.gamma),
        data_root=args.data_root,
    )

    print(f"grain_bin: {result['grain_bin']}")
    print(f"T [K]: {result['temperature']:.6e}   log10(T): {result['log10_temperature']:.6f}")
    print(f"gamma: {result['gamma']:.6e}   log10(gamma): {result['log10_gamma']:.6f}")
    print(f"Zmean: {result['zmean']:.12e}")
    print(f"Zsigma: {result['zsigma']:.12e}")
    print(f"peh_rate [erg s^-1]: {result['peh_rate']:.12e}")
    print(f"rec_rate [erg s^-1]: {result['rec_rate']:.12e}")
    if result["clipped"]:
        print("note: one or more coordinates were clipped to the table bounds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())