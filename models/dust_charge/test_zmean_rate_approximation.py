#!/usr/bin/env python3
"""Compare full charge-distribution rates vs a Zmean-only approximation.

This script reads dust charging outputs from JSON files produced by
`models/dust_charge/export_dust_charging_vs_gamma.py`, then compares:

1) Full model:
   - Solve equilibrium charge distribution P(Z)
   - Compute photoelectric heating and recombination cooling with P(Z)
2) Simplified model:
   - Take Zmean from the charging table
   - Compute the same rates at only that Z value

Outputs diagnostic plots and a JSON summary of accuracy metrics.
"""

from __future__ import annotations

import argparse
import multiprocessing as _mp
import glob
import json
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from models.dust_charge.dust_charging import (
    equilibrium_charge_for_grain,
    get_process_rss_bytes,
    get_system_memory_bytes,
)
from models.dust_charge.dust_photoelectric_heating import (
    PATH_OPTICS,
    c_cgs,
    compute_photoelectric_heating_rate,
    compute_recombination_cooling_rate,
    get_radiation_field,
    most_negative_allowed_charge_graphite,
    most_negative_allowed_charge_silicate,
    read_dielectric_file,
)
from models.dust_radiation.dust_emission import interpolate_cross_sections


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _dust_charging_output_dir() -> Path:
    return _repo_root() / "model_data" / "dust_charging_data"


def _safe_float(v, default=np.nan) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _relative_difference(simple_val: float, full_val: float, tiny: float = 1e-300) -> float:
    denom = max(abs(full_val), tiny)
    return (simple_val - full_val) / denom


def _to_z_used(zmean: float, mode: str) -> float:
    if mode == "raw":
        return float(zmean)
    if mode == "round":
        return float(np.rint(zmean))
    if mode == "floor":
        return float(np.floor(zmean))
    if mode == "ceil":
        return float(np.ceil(zmean))
    raise ValueError(f"Unsupported z mode: {mode}")


def _two_point_charge_mix(zmean: float, grain_type: str, a_cm: float) -> Tuple[int, int, float, float]:
    """Return (z_lo, z_hi, w_lo, w_hi) preserving mean charge with two adjacent integers."""
    z_lo = int(np.floor(float(zmean)))
    z_hi = z_lo + 1
    w_hi = float(zmean) - float(z_lo)
    w_lo = 1.0 - w_hi

    if grain_type == "graphite":
        zmin_allowed = int(most_negative_allowed_charge_graphite(a_cm))
    else:
        zmin_allowed = int(most_negative_allowed_charge_silicate(a_cm))

    # If interpolation would dip below physically allowed minimum, collapse to Zmin.
    if z_lo < zmin_allowed:
        z_lo = zmin_allowed
        z_hi = zmin_allowed
        w_lo = 1.0
        w_hi = 0.0
    elif z_hi < zmin_allowed:
        z_lo = zmin_allowed
        z_hi = zmin_allowed
        w_lo = 1.0
        w_hi = 0.0

    return int(z_lo), int(z_hi), float(w_lo), float(w_hi)


def _three_point_charge_mix(
    zmean: float,
    zsigma: float,
    grain_type: str,
    a_cm: float,
) -> Tuple[int, int, int, float, float, float]:
    """Return a 3-point integer charge mixture matching mean and (when feasible) variance.

    The routine searches integer triplets (z_lo < z_mid < z_hi) and solves for
    weights that satisfy normalization, first moment, and second raw moment.
    If no non-negative solution is found, it falls back to the two-point mix.
    """
    mu = float(zmean)
    sig = float(zsigma) if np.isfinite(zsigma) else 0.0
    sig = max(0.0, sig)

    if grain_type == "graphite":
        zmin_allowed = int(most_negative_allowed_charge_graphite(a_cm))
    else:
        zmin_allowed = int(most_negative_allowed_charge_silicate(a_cm))

    target_m2 = mu * mu + sig * sig

    # Search window in integer Z around the requested mean/sigma.
    half_span = int(max(3, np.ceil(4.0 * sig + 2.0)))
    z_lo_search = max(zmin_allowed, int(np.floor(mu)) - half_span)
    z_hi_search = int(np.ceil(mu)) + half_span
    if z_hi_search - z_lo_search < 2:
        z_hi_search = z_lo_search + 2

    best = None
    best_key = None
    tol = 1e-10

    for z_lo in range(z_lo_search, z_hi_search - 1):
        for z_mid in range(z_lo + 1, z_hi_search):
            for z_hi in range(z_mid + 1, z_hi_search + 1):
                A = np.array(
                    [
                        [1.0, 1.0, 1.0],
                        [float(z_lo), float(z_mid), float(z_hi)],
                        [float(z_lo * z_lo), float(z_mid * z_mid), float(z_hi * z_hi)],
                    ],
                    dtype=float,
                )
                b = np.array([1.0, mu, target_m2], dtype=float)
                try:
                    w = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    continue

                w_lo, w_mid, w_hi = (float(w[0]), float(w[1]), float(w[2]))
                neg_mass = max(0.0, -w_lo) + max(0.0, -w_mid) + max(0.0, -w_hi)
                # Prefer non-negative weights, then tighter local support.
                key = (
                    0 if neg_mass <= tol else 1,
                    neg_mass,
                    abs(float(z_mid) - mu),
                    int(z_hi - z_lo),
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best = (int(z_lo), int(z_mid), int(z_hi), w_lo, w_mid, w_hi)

    if best is not None and best_key is not None and best_key[0] == 0:
        z_lo, z_mid, z_hi, w_lo, w_mid, w_hi = best
        ww = np.array([w_lo, w_mid, w_hi], dtype=float)
        ww = np.clip(ww, 0.0, np.inf)
        sww = float(np.sum(ww))
        if sww > 0.0:
            ww = ww / sww
            return int(z_lo), int(z_mid), int(z_hi), float(ww[0]), float(ww[1]), float(ww[2])

    # Fallback: preserve the existing two-point method when 3-point is infeasible.
    z2_lo, z2_hi, w2_lo, w2_hi = _two_point_charge_mix(mu, grain_type, a_cm)
    return int(z2_lo), int(z2_hi), int(z2_hi), float(w2_lo), float(w2_hi), 0.0


def _print_three_point_high_error_diagnostics(
    grain_type: str,
    a_cm: float,
    T: float,
    gamma: float,
    G0: float,
    ne: float,
    zmean_table: float,
    zmean_full: float,
    zsigma_full: float,
    Zs_arr: np.ndarray,
    P_arr: np.ndarray,
    recomb_z_arr: np.ndarray,
    recomb_full: float,
    recomb_simple_3pt: float,
    z3_lo: int,
    z3_mid: int,
    z3_hi: int,
    w3_lo: float,
    w3_mid: float,
    w3_hi: float,
    recomb_3pt_lo: float,
    recomb_3pt_mid: float,
    recomb_3pt_hi: float,
    threshold_dex: float,
) -> None:
    if not (np.isfinite(recomb_simple_3pt) and np.isfinite(recomb_full)):
        return
    if recomb_simple_3pt <= 0.0 or recomb_full <= 0.0:
        return

    abs_log10_ratio = float(np.abs(np.log10(recomb_simple_3pt / recomb_full)))
    if abs_log10_ratio <= float(threshold_dex):
        return

    print("\\n" + "-" * 100, flush=True)
    print(
        "[high-error three-point cooling] "
        f"abs(log10(Lambda_approx/Lambda_full))={abs_log10_ratio:.4f} > {float(threshold_dex):.3f}",
        flush=True,
    )
    print(
        f"context: grain_type={grain_type}, a_cm={a_cm:.3e}, T={T:.3e} K, gamma={gamma:.3e}, "
        f"G0={G0:.3e}, ne={ne:.3e}",
        flush=True,
    )
    print(
        f"Zmean_table={zmean_table:.4f}, Zmean_full={zmean_full:.4f}, Zsigma_full={zsigma_full:.4f}",
        flush=True,
    )
    print(
        f"Lambda_full={recomb_full:.6e}, Lambda_approx(three-point)={recomb_simple_3pt:.6e}, "
        f"ratio={recomb_simple_3pt / recomb_full:.6e}",
        flush=True,
    )

    if recomb_z_arr.size == P_arr.size and recomb_z_arr.size == Zs_arr.size and recomb_z_arr.size > 0:
        contrib_full = P_arr * recomb_z_arr
        print("full distribution per-charge cooling contributions:", flush=True)
        print("  Z          P(Z)              Lambda(Z)           P(Z)*Lambda(Z)      frac_of_full", flush=True)
        for z_i, p_i, lam_i, c_i in zip(Zs_arr, P_arr, recomb_z_arr, contrib_full):
            frac = float(c_i / recomb_full) if recomb_full > 0.0 else np.nan
            print(
                f"  {int(z_i):>3d}   {float(p_i):>12.6e}   {float(lam_i):>14.6e}   "
                f"{float(c_i):>14.6e}   {frac:>10.6f}",
                flush=True,
            )
    else:
        print(
            "full distribution per-charge contributions unavailable "
            "(Recomb_Z length does not match Z/P arrays)",
            flush=True,
        )

    contrib_3_lo = float(w3_lo) * float(recomb_3pt_lo)
    contrib_3_mid = float(w3_mid) * float(recomb_3pt_mid)
    contrib_3_hi = float(w3_hi) * float(recomb_3pt_hi)
    print("three-point approximation cooling contributions:", flush=True)
    print("  charge      weight              Lambda(Z)         weighted contribution", flush=True)
    print(
        f"  {int(z3_lo):>3d}   {float(w3_lo):>12.6f}   {float(recomb_3pt_lo):>14.6e}   {contrib_3_lo:>14.6e}",
        flush=True,
    )
    print(
        f"  {int(z3_mid):>3d}   {float(w3_mid):>12.6f}   {float(recomb_3pt_mid):>14.6e}   {contrib_3_mid:>14.6e}",
        flush=True,
    )
    print(
        f"  {int(z3_hi):>3d}   {float(w3_hi):>12.6f}   {float(recomb_3pt_hi):>14.6e}   {contrib_3_hi:>14.6e}",
        flush=True,
    )
    print("-" * 100, flush=True)


def _compute_binned_log10_ratio(
    gamma_vals: np.ndarray,
    T_vals: np.ndarray,
    simple_vals: np.ndarray,
    full_vals: np.ndarray,
    n_gamma_bins: int,
    n_T_bins: int,
    min_bin_count: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gamma_vals = np.asarray(gamma_vals, dtype=float)
    T_vals = np.asarray(T_vals, dtype=float)
    simple_vals = np.asarray(simple_vals, dtype=float)
    full_vals = np.asarray(full_vals, dtype=float)

    valid = (
        np.isfinite(gamma_vals)
        & np.isfinite(T_vals)
        & np.isfinite(simple_vals)
        & np.isfinite(full_vals)
        & (gamma_vals > 0.0)
        & (T_vals > 0.0)
        & (simple_vals > 0.0)
        & (full_vals > 0.0)
    )

    if not np.any(valid):
        raise RuntimeError("No positive finite values to build ratio map")

    g = gamma_vals[valid]
    t = T_vals[valid]
    log_ratio = np.log10(simple_vals[valid] / full_vals[valid])

    gamma_edges = np.logspace(np.log10(np.min(g)), np.log10(np.max(g)), n_gamma_bins + 1)
    T_edges = np.logspace(np.log10(np.min(t)), np.log10(np.max(t)), n_T_bins + 1)

    g_idx = np.digitize(g, gamma_edges) - 1
    t_idx = np.digitize(t, T_edges) - 1
    # Keep samples exactly on the upper edge inside the last bin.
    g_idx[g == gamma_edges[-1]] = n_gamma_bins - 1
    t_idx[t == T_edges[-1]] = n_T_bins - 1
    in_bounds = (g_idx >= 0) & (g_idx < n_gamma_bins) & (t_idx >= 0) & (t_idx < n_T_bins)
    g_idx = g_idx[in_bounds]
    t_idx = t_idx[in_bounds]
    log_ratio = log_ratio[in_bounds]

    mat = np.full((n_T_bins, n_gamma_bins), np.nan, dtype=float)
    counts = np.zeros((n_T_bins, n_gamma_bins), dtype=int)

    for j in range(n_T_bins):
        for i in range(n_gamma_bins):
            m = (t_idx == j) & (g_idx == i)
            n = int(np.count_nonzero(m))
            counts[j, i] = n
            if n >= max(1, min_bin_count):
                mat[j, i] = float(np.median(log_ratio[m]))

    return gamma_edges, T_edges, mat, counts


def _compute_binned_median_from_edges(
    gamma_vals: np.ndarray,
    T_vals: np.ndarray,
    data_vals: np.ndarray,
    gamma_edges: np.ndarray,
    T_edges: np.ndarray,
    min_bin_count: int,
) -> np.ndarray:
    gamma_vals = np.asarray(gamma_vals, dtype=float)
    T_vals = np.asarray(T_vals, dtype=float)
    data_vals = np.asarray(data_vals, dtype=float)

    n_gamma_bins = int(len(gamma_edges) - 1)
    n_T_bins = int(len(T_edges) - 1)
    mat = np.full((n_T_bins, n_gamma_bins), np.nan, dtype=float)

    valid = (
        np.isfinite(gamma_vals)
        & np.isfinite(T_vals)
        & np.isfinite(data_vals)
        & (gamma_vals > 0.0)
        & (T_vals > 0.0)
    )
    if not np.any(valid):
        return mat

    g = gamma_vals[valid]
    t = T_vals[valid]
    d = data_vals[valid]

    g_idx = np.digitize(g, gamma_edges) - 1
    t_idx = np.digitize(t, T_edges) - 1
    g_idx[g == gamma_edges[-1]] = n_gamma_bins - 1
    t_idx[t == T_edges[-1]] = n_T_bins - 1

    in_bounds = (g_idx >= 0) & (g_idx < n_gamma_bins) & (t_idx >= 0) & (t_idx < n_T_bins)
    g_idx = g_idx[in_bounds]
    t_idx = t_idx[in_bounds]
    d = d[in_bounds]

    for j in range(n_T_bins):
        for i in range(n_gamma_bins):
            m = (t_idx == j) & (g_idx == i)
            n = int(np.count_nonzero(m))
            if n >= max(1, min_bin_count):
                mat[j, i] = float(np.median(d[m]))

    return mat


def _fill_nan_with_nearest(mat: np.ndarray) -> np.ndarray:
    out = np.array(mat, dtype=float, copy=True)
    nan_mask = ~np.isfinite(out)
    if not np.any(nan_mask):
        return out

    finite_mask = np.isfinite(out)
    if not np.any(finite_mask):
        return np.nan_to_num(out, nan=0.0)

    finite_idx = np.argwhere(finite_mask)
    finite_vals = out[finite_mask]

    for i, j in np.argwhere(nan_mask):
        d2 = (finite_idx[:, 0] - i) ** 2 + (finite_idx[:, 1] - j) ** 2
        out[i, j] = float(finite_vals[int(np.argmin(d2))])

    return out


def _plot_ratio_maps(
    output_rows: List[Dict[str, float]],
    out_path: Path,
    n_gamma_bins: int,
    n_T_bins: int,
    min_bin_count: int,
    annotate_cells: bool,
    annotation_fontsize: int,
) -> Dict[str, float]:
    good = [r for r in output_rows if not r.get("error")]
    if len(good) == 0:
        raise RuntimeError("No successful rows available for ratio plotting")

    gamma_vals = np.array([r.get("gamma", np.nan) for r in good], dtype=float)
    T_vals = np.array([r.get("T", np.nan) for r in good], dtype=float)
    gamma_single = np.array([r.get("Gamma_simple_single", np.nan) for r in good], dtype=float)
    gamma_interp = np.array([r.get("Gamma_simple_interp", np.nan) for r in good], dtype=float)
    gamma_three = np.array([r.get("Gamma_simple_3pt", np.nan) for r in good], dtype=float)
    gamma_full = np.array([r.get("Gamma_full", np.nan) for r in good], dtype=float)
    recomb_single = np.array([r.get("Recomb_simple_single", np.nan) for r in good], dtype=float)
    recomb_interp = np.array([r.get("Recomb_simple_interp", np.nan) for r in good], dtype=float)
    recomb_three = np.array([r.get("Recomb_simple_3pt", np.nan) for r in good], dtype=float)
    recomb_full = np.array([r.get("Recomb_full", np.nan) for r in good], dtype=float)
    recomb_frac_z0 = np.array([r.get("Recomb_full_frac_from_Z0", np.nan) for r in good], dtype=float)
    recomb_frac_zminp1 = np.array([r.get("Recomb_full_frac_from_Zminp1", np.nan) for r in good], dtype=float)
    zmean_full = np.array([r.get("Zmean_full", np.nan) for r in good], dtype=float)
    zsigma_full = np.array([r.get("Zsigma_full", np.nan) for r in good], dtype=float)

    valid_joint = (
        np.isfinite(gamma_vals)
        & np.isfinite(T_vals)
        & np.isfinite(gamma_single)
        & np.isfinite(gamma_interp)
        & np.isfinite(gamma_three)
        & np.isfinite(gamma_full)
        & np.isfinite(recomb_single)
        & np.isfinite(recomb_interp)
        & np.isfinite(recomb_three)
        & np.isfinite(recomb_full)
        & (gamma_vals > 0.0)
        & (T_vals > 0.0)
    )
    n_valid_joint = int(np.count_nonzero(valid_joint))
    if n_valid_joint <= 0:
        raise RuntimeError("No valid rows for ratio-map plotting")

    # Use adaptive bin counts so quick smoke tests (few points) still produce readable maps.
    gamma_unique = np.unique(np.round(np.log10(gamma_vals[valid_joint]), 6)).size
    T_unique = np.unique(np.round(np.log10(T_vals[valid_joint]), 6)).size
    auto_gamma_bins = max(4, int(np.sqrt(n_valid_joint)))
    auto_T_bins = max(4, int(np.sqrt(n_valid_joint)))
    eff_gamma_bins = int(min(n_gamma_bins, auto_gamma_bins, max(4, gamma_unique)))
    eff_T_bins = int(min(n_T_bins, auto_T_bins, max(4, T_unique)))

    # Relax min_bin_count automatically for sparse samples.
    eff_min_bin_count = int(max(1, min(min_bin_count, n_valid_joint // max(8, eff_gamma_bins))))

    g_edges_h, T_edges_h, mat_h_single, counts_h_single = _compute_binned_log10_ratio(
        gamma_vals,
        T_vals,
        gamma_single,
        gamma_full,
        n_gamma_bins=eff_gamma_bins,
        n_T_bins=eff_T_bins,
        min_bin_count=eff_min_bin_count,
    )
    g_edges_c, T_edges_c, mat_c_single, counts_c_single = _compute_binned_log10_ratio(
        gamma_vals,
        T_vals,
        recomb_single,
        recomb_full,
        n_gamma_bins=eff_gamma_bins,
        n_T_bins=eff_T_bins,
        min_bin_count=eff_min_bin_count,
    )
    _, _, mat_h_interp, counts_h_interp = _compute_binned_log10_ratio(
        gamma_vals,
        T_vals,
        gamma_interp,
        gamma_full,
        n_gamma_bins=eff_gamma_bins,
        n_T_bins=eff_T_bins,
        min_bin_count=eff_min_bin_count,
    )
    _, _, mat_c_interp, counts_c_interp = _compute_binned_log10_ratio(
        gamma_vals,
        T_vals,
        recomb_interp,
        recomb_full,
        n_gamma_bins=eff_gamma_bins,
        n_T_bins=eff_T_bins,
        min_bin_count=eff_min_bin_count,
    )
    _, _, mat_h_three, counts_h_three = _compute_binned_log10_ratio(
        gamma_vals,
        T_vals,
        gamma_three,
        gamma_full,
        n_gamma_bins=eff_gamma_bins,
        n_T_bins=eff_T_bins,
        min_bin_count=eff_min_bin_count,
    )
    _, _, mat_c_three, counts_c_three = _compute_binned_log10_ratio(
        gamma_vals,
        T_vals,
        recomb_three,
        recomb_full,
        n_gamma_bins=eff_gamma_bins,
        n_T_bins=eff_T_bins,
        min_bin_count=eff_min_bin_count,
    )

    # Fill empty bins so the rendered matrix has no blank cells.
    mat_h_single = _fill_nan_with_nearest(mat_h_single)
    mat_c_single = _fill_nan_with_nearest(mat_c_single)
    mat_h_interp = _fill_nan_with_nearest(mat_h_interp)
    mat_c_interp = _fill_nan_with_nearest(mat_c_interp)
    mat_h_three = _fill_nan_with_nearest(mat_h_three)
    mat_c_three = _fill_nan_with_nearest(mat_c_three)

    zmean_mat = _compute_binned_median_from_edges(
        gamma_vals,
        T_vals,
        zmean_full,
        g_edges_h,
        T_edges_h,
        min_bin_count=1,
    )
    zsigma_mat = _compute_binned_median_from_edges(
        gamma_vals,
        T_vals,
        zsigma_full,
        g_edges_h,
        T_edges_h,
        min_bin_count=1,
    )
    zmean_mat = _fill_nan_with_nearest(zmean_mat)
    zsigma_mat = _fill_nan_with_nearest(zsigma_mat)

    recomb_full_log_mat = _compute_binned_median_from_edges(
        gamma_vals,
        T_vals,
        np.log10(np.maximum(recomb_full, 1e-300)),
        g_edges_h,
        T_edges_h,
        min_bin_count=1,
    )
    recomb_abs_diff_log_mat = _compute_binned_median_from_edges(
        gamma_vals,
        T_vals,
        np.log10(np.maximum(np.abs(recomb_interp - recomb_full), 1e-300)),
        g_edges_h,
        T_edges_h,
        min_bin_count=1,
    )
    recomb_full_log_mat = _fill_nan_with_nearest(recomb_full_log_mat)
    recomb_abs_diff_log_mat = _fill_nan_with_nearest(recomb_abs_diff_log_mat)

    recomb_frac_z0_mat = _compute_binned_median_from_edges(
        gamma_vals,
        T_vals,
        np.clip(recomb_frac_z0, 0.0, 1.0),
        g_edges_h,
        T_edges_h,
        min_bin_count=1,
    )
    recomb_frac_z0_mat = _fill_nan_with_nearest(recomb_frac_z0_mat)
    recomb_frac_z0_mat = np.clip(recomb_frac_z0_mat, 0.0, 1.0)

    recomb_frac_zminp1_mat = _compute_binned_median_from_edges(
        gamma_vals,
        T_vals,
        np.clip(recomb_frac_zminp1, 0.0, 1.0),
        g_edges_h,
        T_edges_h,
        min_bin_count=1,
    )
    recomb_frac_zminp1_mat = _fill_nan_with_nearest(recomb_frac_zminp1_mat)
    recomb_frac_zminp1_mat = np.clip(recomb_frac_zminp1_mat, 0.0, 1.0)

    finite_vals = np.concatenate([
        mat_h_single[np.isfinite(mat_h_single)],
        mat_c_single[np.isfinite(mat_c_single)],
        mat_h_interp[np.isfinite(mat_h_interp)],
        mat_c_interp[np.isfinite(mat_c_interp)],
        mat_h_three[np.isfinite(mat_h_three)],
        mat_c_three[np.isfinite(mat_c_three)],
    ])
    if finite_vals.size == 0:
        raise RuntimeError("No populated bins for ratio maps (try reducing --min-bin-count)")
    vmin = -0.5
    vmax = 0.5

    fig = plt.figure(figsize=(12, 18.0), dpi=180)
    gs = fig.add_gridspec(5, 2, height_ratios=[1.0, 1.0, 1.0, 1.0, 1.0], hspace=0.24, wspace=0.12)

    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1], sharex=ax00, sharey=ax00)
    ax10 = fig.add_subplot(gs[1, 0], sharex=ax00, sharey=ax00)
    ax11 = fig.add_subplot(gs[1, 1], sharex=ax00, sharey=ax00)
    ax20 = fig.add_subplot(gs[2, 0], sharex=ax00, sharey=ax00)
    ax21 = fig.add_subplot(gs[2, 1], sharex=ax00, sharey=ax00)
    ax30 = fig.add_subplot(gs[3, 0], sharex=ax00, sharey=ax00)
    ax31 = fig.add_subplot(gs[3, 1], sharex=ax00, sharey=ax00)
    ax40 = fig.add_subplot(gs[4, 0], sharex=ax00, sharey=ax00)
    ax41 = fig.add_subplot(gs[4, 1], sharex=ax00, sharey=ax00)
    all_axes = [ax00, ax01, ax10, ax11, ax20, ax21, ax30, ax31, ax40, ax41]

    im0 = ax00.pcolormesh(g_edges_h, T_edges_h, mat_h_single, shading="auto", cmap="coolwarm", vmin=vmin, vmax=vmax)
    ax00.set_xscale("log")
    ax00.set_yscale("log")
    ax00.set_ylabel(r"$T$ [K]")
    ax00.set_title(r"Single-Zmean heating: $\log_{10}(\Gamma_{\rm approx}/\Gamma_{\rm full})$")
    cb0 = fig.colorbar(im0, ax=ax00, pad=0.02)
    cb0.set_label(r"$\log_{10}(\mathrm{approx}/\mathrm{full})$")

    im1 = ax01.pcolormesh(g_edges_c, T_edges_c, mat_c_single, shading="auto", cmap="coolwarm", vmin=vmin, vmax=vmax)
    ax01.set_xscale("log")
    ax01.set_yscale("log")
    ax01.set_title(r"Single-Zmean cooling: $\log_{10}(\Lambda_{\rm approx}/\Lambda_{\rm full})$")
    cb1 = fig.colorbar(im1, ax=ax01, pad=0.02)
    cb1.set_label(r"$\log_{10}(\mathrm{approx}/\mathrm{full})$")

    im2 = ax10.pcolormesh(g_edges_h, T_edges_h, mat_h_interp, shading="auto", cmap="coolwarm", vmin=vmin, vmax=vmax)
    ax10.set_xscale("log")
    ax10.set_yscale("log")
    ax10.set_ylabel(r"$T$ [K]")
    ax10.set_title(r"Two-point heating: $\log_{10}(\Gamma_{\rm approx}/\Gamma_{\rm full})$")
    cb2 = fig.colorbar(im2, ax=ax10, pad=0.02)
    cb2.set_label(r"$\log_{10}(\mathrm{approx}/\mathrm{full})$")

    im3 = ax11.pcolormesh(g_edges_c, T_edges_c, mat_c_interp, shading="auto", cmap="coolwarm", vmin=vmin, vmax=vmax)
    ax11.set_xscale("log")
    ax11.set_yscale("log")
    ax11.set_title(r"Two-point cooling: $\log_{10}(\Lambda_{\rm approx}/\Lambda_{\rm full})$")
    cb3 = fig.colorbar(im3, ax=ax11, pad=0.02)
    cb3.set_label(r"$\log_{10}(\mathrm{approx}/\mathrm{full})$")

    im4 = ax20.pcolormesh(g_edges_h, T_edges_h, mat_h_three, shading="auto", cmap="coolwarm", vmin=vmin, vmax=vmax)
    ax20.set_xscale("log")
    ax20.set_yscale("log")
    ax20.set_ylabel(r"$T$ [K]")
    ax20.set_title(r"Three-point heating: $\log_{10}(\Gamma_{\rm approx}/\Gamma_{\rm full})$")
    cb4 = fig.colorbar(im4, ax=ax20, pad=0.02)
    cb4.set_label(r"$\log_{10}(\mathrm{approx}/\mathrm{full})$")

    im5 = ax21.pcolormesh(g_edges_c, T_edges_c, mat_c_three, shading="auto", cmap="coolwarm", vmin=vmin, vmax=vmax)
    ax21.set_xscale("log")
    ax21.set_yscale("log")
    ax21.set_title(r"Three-point cooling: $\log_{10}(\Lambda_{\rm approx}/\Lambda_{\rm full})$")
    cb5 = fig.colorbar(im5, ax=ax21, pad=0.02)
    cb5.set_label(r"$\log_{10}(\mathrm{approx}/\mathrm{full})$")

    vmin_rf = float(np.nanpercentile(recomb_full_log_mat[np.isfinite(recomb_full_log_mat)], 1))
    vmax_rf = float(np.nanpercentile(recomb_full_log_mat[np.isfinite(recomb_full_log_mat)], 99))
    if not np.isfinite(vmin_rf) or not np.isfinite(vmax_rf) or (vmax_rf <= vmin_rf):
        vmin_rf, vmax_rf = -30.0, -10.0
    im6 = ax30.pcolormesh(
        g_edges_h,
        T_edges_h,
        recomb_full_log_mat,
        shading="auto",
        cmap="viridis",
        vmin=vmin_rf,
        vmax=vmax_rf,
    )
    ax30.set_xscale("log")
    ax30.set_yscale("log")
    ax30.set_ylabel(r"$T$ [K]")
    ax30.set_title(r"Cooling full: $\log_{10}(\Lambda_{\rm full})$")
    cb6 = fig.colorbar(im6, ax=ax30, pad=0.02)
    cb6.set_label(r"$\log_{10}(\Lambda_{\rm full})$")

    vmin_rd = float(np.nanpercentile(recomb_abs_diff_log_mat[np.isfinite(recomb_abs_diff_log_mat)], 1))
    vmax_rd = float(np.nanpercentile(recomb_abs_diff_log_mat[np.isfinite(recomb_abs_diff_log_mat)], 99))
    if not np.isfinite(vmin_rd) or not np.isfinite(vmax_rd) or (vmax_rd <= vmin_rd):
        vmin_rd, vmax_rd = -30.0, -10.0
    im7 = ax31.pcolormesh(
        g_edges_h,
        T_edges_h,
        recomb_abs_diff_log_mat,
        shading="auto",
        cmap="magma",
        vmin=vmin_rd,
        vmax=vmax_rd,
    )
    ax31.set_xscale("log")
    ax31.set_yscale("log")
    ax31.set_title(r"Two-point cooling abs diff: $\log_{10}(|\Lambda_{\rm approx}-\Lambda_{\rm full}|)$")
    cb7 = fig.colorbar(im7, ax=ax31, pad=0.02)
    cb7.set_label(r"$\log_{10}(|\Delta \Lambda|)$")

    im8 = ax40.pcolormesh(
        g_edges_h,
        T_edges_h,
        recomb_frac_z0_mat,
        shading="auto",
        cmap="cividis",
        vmin=0.0,
        vmax=1.0,
    )
    ax40.set_xscale("log")
    ax40.set_yscale("log")
    ax40.set_xlabel(r"$\gamma = G_0\sqrt{T}/n_e$")
    ax40.set_ylabel(r"$T$ [K]")
    ax40.set_title(r"Full cooling share from $Z=0$: $P(0)\Lambda(0)/\sum_Z P(Z)\Lambda(Z)$")
    cb8 = fig.colorbar(im8, ax=ax40, pad=0.015)
    cb8.set_label("fraction")

    im9 = ax41.pcolormesh(
        g_edges_h,
        T_edges_h,
        recomb_frac_zminp1_mat,
        shading="auto",
        cmap="cividis",
        vmin=0.0,
        vmax=1.0,
    )
    ax41.set_xscale("log")
    ax41.set_yscale("log")
    ax41.set_xlabel(r"$\gamma = G_0\sqrt{T}/n_e$")
    ax41.set_title(r"Full cooling share from $Z_{\min}+1$: $P(Z_{\min}+1)\Lambda/\sum_Z P(Z)\Lambda(Z)$")
    cb9 = fig.colorbar(im9, ax=ax41, pad=0.015)
    cb9.set_label("fraction")

    for ax in all_axes:
        ax.tick_params(which="both", axis="both", direction="in")
        ax.minorticks_on()

    if annotate_cells:
        nTb, nGb = mat_h_single.shape
        for j in range(nTb):
            y0, y1 = T_edges_h[j], T_edges_h[j + 1]
            y_text = y0 * (y1 / y0) ** 0.87
            for i in range(nGb):
                x0, x1 = g_edges_h[i], g_edges_h[i + 1]
                x_text = np.sqrt(x0 * x1)
                txt = f"Z={zmean_mat[j, i]:.1f}\nsig={zsigma_mat[j, i]:.1f}"
                for ax in all_axes:
                    ax.text(
                        x_text,
                        y_text,
                        txt,
                        ha="center",
                        va="top",
                        fontsize=max(4, int(annotation_fontsize)),
                        color="black",
                        bbox={"boxstyle": "round,pad=0.1", "fc": "white", "ec": "none", "alpha": 0.55},
                    )

    fig.suptitle("Approximation-vs-full rate ratio map")
    fig.tight_layout(rect=[0, 0, 1, 0.972])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    # Identify where absolute mismatch is largest in populated bins (single and two-point).
    abs_h_single = np.abs(mat_h_single)
    abs_c_single = np.abs(mat_c_single)
    abs_h_interp = np.abs(mat_h_interp)
    abs_c_interp = np.abs(mat_c_interp)
    abs_h_three = np.abs(mat_h_three)
    abs_c_three = np.abs(mat_c_three)

    ihs = np.unravel_index(np.nanargmax(abs_h_single), abs_h_single.shape)
    ics = np.unravel_index(np.nanargmax(abs_c_single), abs_c_single.shape)
    ihi = np.unravel_index(np.nanargmax(abs_h_interp), abs_h_interp.shape)
    ici = np.unravel_index(np.nanargmax(abs_c_interp), abs_c_interp.shape)
    ih3 = np.unravel_index(np.nanargmax(abs_h_three), abs_h_three.shape)
    ic3 = np.unravel_index(np.nanargmax(abs_c_three), abs_c_three.shape)

    gamma_center_h_single = float(np.sqrt(g_edges_h[ihs[1]] * g_edges_h[ihs[1] + 1]))
    T_center_h_single = float(np.sqrt(T_edges_h[ihs[0]] * T_edges_h[ihs[0] + 1]))
    gamma_center_c_single = float(np.sqrt(g_edges_c[ics[1]] * g_edges_c[ics[1] + 1]))
    T_center_c_single = float(np.sqrt(T_edges_c[ics[0]] * T_edges_c[ics[0] + 1]))

    gamma_center_h_interp = float(np.sqrt(g_edges_h[ihi[1]] * g_edges_h[ihi[1] + 1]))
    T_center_h_interp = float(np.sqrt(T_edges_h[ihi[0]] * T_edges_h[ihi[0] + 1]))
    gamma_center_c_interp = float(np.sqrt(g_edges_c[ici[1]] * g_edges_c[ici[1] + 1]))
    T_center_c_interp = float(np.sqrt(T_edges_c[ici[0]] * T_edges_c[ici[0] + 1]))

    gamma_center_h_three = float(np.sqrt(g_edges_h[ih3[1]] * g_edges_h[ih3[1] + 1]))
    T_center_h_three = float(np.sqrt(T_edges_h[ih3[0]] * T_edges_h[ih3[0] + 1]))
    gamma_center_c_three = float(np.sqrt(g_edges_c[ic3[1]] * g_edges_c[ic3[1] + 1]))
    T_center_c_three = float(np.sqrt(T_edges_c[ic3[0]] * T_edges_c[ic3[0] + 1]))

    return {
        "plot_path": str(out_path),
        "effective_gamma_bins": int(eff_gamma_bins),
        "effective_T_bins": int(eff_T_bins),
        "effective_min_bin_count": int(eff_min_bin_count),
        "heatmap_vmin": vmin,
        "heatmap_vmax": vmax,
        "heating_max_abs_log10_ratio_single": float(abs_h_single[ihs]),
        "heating_max_abs_gamma_single": gamma_center_h_single,
        "heating_max_abs_T_single": T_center_h_single,
        "heating_max_abs_count_single": int(counts_h_single[ihs]),
        "cooling_max_abs_log10_ratio_single": float(abs_c_single[ics]),
        "cooling_max_abs_gamma_single": gamma_center_c_single,
        "cooling_max_abs_T_single": T_center_c_single,
        "cooling_max_abs_count_single": int(counts_c_single[ics]),
        "heating_max_abs_log10_ratio_interp": float(abs_h_interp[ihi]),
        "heating_max_abs_gamma_interp": gamma_center_h_interp,
        "heating_max_abs_T_interp": T_center_h_interp,
        "heating_max_abs_count_interp": int(counts_h_interp[ihi]),
        "cooling_max_abs_log10_ratio_interp": float(abs_c_interp[ici]),
        "cooling_max_abs_gamma_interp": gamma_center_c_interp,
        "cooling_max_abs_T_interp": T_center_c_interp,
        "cooling_max_abs_count_interp": int(counts_c_interp[ici]),
        "heating_max_abs_log10_ratio_3pt": float(abs_h_three[ih3]),
        "heating_max_abs_gamma_3pt": gamma_center_h_three,
        "heating_max_abs_T_3pt": T_center_h_three,
        "heating_max_abs_count_3pt": int(counts_h_three[ih3]),
        "cooling_max_abs_log10_ratio_3pt": float(abs_c_three[ic3]),
        "cooling_max_abs_gamma_3pt": gamma_center_c_three,
        "cooling_max_abs_T_3pt": T_center_c_three,
        "cooling_max_abs_count_3pt": int(counts_c_three[ic3]),
        "recomb_full_log10_vmin": float(vmin_rf),
        "recomb_full_log10_vmax": float(vmax_rf),
        "recomb_absdiff_log10_vmin": float(vmin_rd),
        "recomb_absdiff_log10_vmax": float(vmax_rd),
        "recomb_z0_fraction_vmin": 0.0,
        "recomb_z0_fraction_vmax": 1.0,
        "recomb_zminp1_fraction_vmin": 0.0,
        "recomb_zminp1_fraction_vmax": 1.0,
        "annotate_cells": bool(annotate_cells),
        "annotation_fontsize": int(annotation_fontsize),
    }


class SimplifiedRateEvaluator:
    """Cache radiation/optical data and evaluate rates at a single Z value."""

    def __init__(self, radiation_model: str = "Mathis", scale_three_col_with_g0: bool = False):
        self.radiation_model = radiation_model
        self.scale_three_col_with_g0 = bool(scale_three_col_with_g0)
        self._cache: Dict[Tuple[str, float], Dict[str, np.ndarray]] = {}

    def _build_cached_inputs(self, grain_type: str, a_cm: float) -> Dict[str, np.ndarray]:
        key = (grain_type, float(a_cm))
        if key in self._cache:
            return self._cache[key]

        rad0, _ = get_radiation_field(self.radiation_model)
        rad0 = np.asarray(rad0)

        if rad0.ndim != 2 or rad0.shape[1] < 2:
            raise ValueError("Radiation field must be a 2D array with at least two columns")

        if rad0.shape[1] >= 3:
            # Same branching as equilibrium_charge_for_grain.
            E_eV = rad0[:, 0].astype(float)
            wav_nm = rad0[:, 1].astype(float)
            I_E_base = rad0[:, 2].astype(float)

            order = np.argsort(E_eV)
            E_eV = E_eV[order]
            wav_nm = wav_nm[order]
            I_E_base = I_E_base[order]

            lambda_cm = wav_nm * 1e-7
            E_for_interp = E_eV
            wav_cm_for_interp = lambda_cm
            rad_has_three_cols = True
        else:
            # Same branching as equilibrium_charge_for_grain.
            wavelength_nm = rad0[:, 0].astype(float)
            wavelength_intensity = rad0[:, 1].astype(float)

            wav_nm_rev = wavelength_nm[::-1]
            wav_int_rev = wavelength_intensity[::-1]

            hc_eVnm = 1239.84193
            E_eV = hc_eVnm / wav_nm_rev
            lambda_cm = wav_nm_rev * 1e-7
            I_lambda_per_cm = wav_int_rev * 1e7
            n_nu_per_sr = I_lambda_per_cm * (lambda_cm ** 3) / (6.62607015e-27 * c_cgs ** 2)
            n_nu_at_g0_1 = n_nu_per_sr * (4.0 * np.pi)
            nu = c_cgs / lambda_cm
            I_E_base = n_nu_at_g0_1 * nu * 1.602176634e-12

            E_for_interp = E_eV
            wav_cm_for_interp = lambda_cm
            rad_has_three_cols = False

        a_micron = float(a_cm) * 1e4
        _, wav_cs, _, C_abs_cs, _ = interpolate_cross_sections(grain_type, a_micron)
        optical_E = 1.2398 / (wav_cs * 1e4)
        C_abs_interp_cm2 = np.interp(E_for_interp, optical_E, C_abs_cs)

        if grain_type == "graphite":
            data_perp = read_dielectric_file(
                f"{PATH_OPTICS}/draine_lee_1984/callindex.out_CpeD03_0.10"
            )
            data_par = read_dielectric_file(
                f"{PATH_OPTICS}/draine_lee_1984/callindex.out_CpaD03_0.10"
            )
            wav_micron = wav_cm_for_interp * 1e4
            Im_perp = np.interp(
                wav_micron,
                data_perp["table"]["wavelength_um"][::-1],
                data_perp["table"]["Im_n"][::-1],
            )
            Im_par = np.interp(
                wav_micron,
                data_par["table"]["wavelength_um"][::-1],
                data_par["table"]["Im_n"][::-1],
            )
            Im_for_heating = np.column_stack([Im_perp, Im_par])
        else:
            data = read_dielectric_file(f"{PATH_OPTICS}/draine_lee_1984/eps_suvSil")
            wav_micron = wav_cm_for_interp * 1e4
            Im_for_heating = np.interp(
                wav_micron,
                data["table"]["wavelength_um"][::-1],
                data["table"]["Im_n"][::-1],
            )

        out = {
            "E_eV": np.asarray(E_eV, dtype=float),
            "lambda_cm": np.asarray(lambda_cm, dtype=float),
            "I_E_base": np.asarray(I_E_base, dtype=float),
            "C_abs": np.asarray(C_abs_interp_cm2, dtype=float),
            "Im": np.asarray(Im_for_heating, dtype=float),
            "rad_has_three_cols": np.array([1 if rad_has_three_cols else 0], dtype=int),
        }
        self._cache[key] = out
        return out

    def compute_rates(self, Z_value: float, a_cm: float, ne: float, T: float, grain_type: str, G0: float) -> Dict[str, float]:
        cached = self._build_cached_inputs(grain_type, a_cm)
        E_eV = cached["E_eV"]
        lambda_cm = cached["lambda_cm"]
        I_E_base = cached["I_E_base"]

        has_three_cols = bool(cached["rad_has_three_cols"][0])
        if has_three_cols and not self.scale_three_col_with_g0:
            I_E_surface = I_E_base
        else:
            I_E_surface = I_E_base * float(G0)

        radiation_field = np.column_stack([E_eV, lambda_cm, I_E_surface])

        pe_args = (
            float(Z_value),
            float(a_cm),
            radiation_field,
            grain_type,
            cached["Im"],
            cached["C_abs"],
        )
        rec_args = (
            float(Z_value),
            float(a_cm),
            float(ne),
            float(T),
            grain_type,
        )

        gamma_val = float(compute_photoelectric_heating_rate(pe_args))
        recomb_val = float(compute_recombination_cooling_rate(rec_args))
        return {
            "Gamma_total": gamma_val,
            "Recomb_total": recomb_val,
        }


_WORKER_EVALUATORS: Dict[Tuple[str, bool], SimplifiedRateEvaluator] = {}


def _get_worker_evaluator(radiation_model: str, scale_three_col_with_g0: bool) -> SimplifiedRateEvaluator:
    key = (str(radiation_model), bool(scale_three_col_with_g0))
    ev = _WORKER_EVALUATORS.get(key)
    if ev is None:
        ev = SimplifiedRateEvaluator(
            radiation_model=radiation_model,
            scale_three_col_with_g0=scale_three_col_with_g0,
        )
        _WORKER_EVALUATORS[key] = ev
    return ev


def _compute_single_row(task: Dict[str, object]) -> Dict[str, object]:
    row = dict(task["row"])  # copy so we can safely mutate
    z_mode = str(task["z_mode"])
    radiation_model = str(task["radiation_model"])
    scale_three_col_with_g0 = bool(task["scale_three_col_with_g0"])
    use_full_zmean_for_approx = bool(task.get("use_full_zmean_for_approx", False))

    try:
        grain_type = str(row.get("grain_type", "")).strip().lower()
        if grain_type not in ("graphite", "silicate"):
            grain_type = "silicate" if "sil" in grain_type else "graphite"

        G0 = float(row["G0"])
        ne = float(row["ne"])
        T = float(row["T"])
        a_cm = float(row["a_cm"])
        zmean_table = float(row["Zmean"])

        # Full model (distribution over Z)
        Zs_eq, P_eq, rates_full, zmean_full, zsigma_full = equilibrium_charge_for_grain(
            G0=G0,
            ne=ne,
            T=T,
            grain_type=grain_type,
            a_cm=a_cm,
            radiation_model=radiation_model,
            rad_field=None,
            yield_params={"material": grain_type},
            ion_species=None,
            Z_start=0,
            debug=False,
        )
        gamma_full = float(rates_full.get("Gamma_total", np.nan))
        recomb_full = float(rates_full.get("Recomb_total", np.nan))
        auto_full = float(rates_full.get("Autoionisation_cooling", 0.0))
        recomb_z_arr = np.asarray(rates_full.get("Recomb_Z", []), dtype=float)
        Zs_arr = np.asarray(Zs_eq, dtype=int)
        P_arr = np.asarray(P_eq, dtype=float)
        recomb_full_kernel_check = np.nan
        recomb_full_z0_contrib = np.nan
        recomb_full_frac_from_z0 = np.nan
        recomb_full_zminp1_contrib = np.nan
        recomb_full_frac_from_zminp1 = np.nan
        if recomb_z_arr.size == P_arr.size and recomb_z_arr.size > 0:
            recomb_full_kernel_check = float(np.sum(P_arr * recomb_z_arr))

            if grain_type == "graphite":
                zmin_allowed = int(most_negative_allowed_charge_graphite(a_cm))
            else:
                zmin_allowed = int(most_negative_allowed_charge_silicate(a_cm))

            idx_z0 = np.where(Zs_arr == 0)[0]
            if idx_z0.size > 0:
                i0 = int(idx_z0[0])
                recomb_full_z0_contrib = float(P_arr[i0] * recomb_z_arr[i0])
            else:
                recomb_full_z0_contrib = 0.0

            idx_zminp1 = np.where(Zs_arr == (zmin_allowed + 1))[0]
            if idx_zminp1.size > 0:
                i1 = int(idx_zminp1[0])
                recomb_full_zminp1_contrib = float(P_arr[i1] * recomb_z_arr[i1])
            else:
                recomb_full_zminp1_contrib = 0.0

            if recomb_full_kernel_check > 0.0 and np.isfinite(recomb_full_z0_contrib):
                recomb_full_frac_from_z0 = float(recomb_full_z0_contrib / recomb_full_kernel_check)
            if recomb_full_kernel_check > 0.0 and np.isfinite(recomb_full_zminp1_contrib):
                recomb_full_frac_from_zminp1 = float(recomb_full_zminp1_contrib / recomb_full_kernel_check)

        evaluator = _get_worker_evaluator(radiation_model, scale_three_col_with_g0)

        zmean_for_approx = float(zmean_full) if use_full_zmean_for_approx else float(zmean_table)
        zmean_for_approx_source = "full" if use_full_zmean_for_approx else "table"
        z_single = _to_z_used(zmean_for_approx, z_mode)
        z_lo, z_hi, w_lo, w_hi = _two_point_charge_mix(zmean_for_approx, grain_type, a_cm)
        z3_lo, z3_mid, z3_hi, w3_lo, w3_mid, w3_hi = _three_point_charge_mix(
            zmean=zmean_for_approx,
            zsigma=float(zsigma_full),
            grain_type=grain_type,
            a_cm=a_cm,
        )

        # Approximation A: single Z value from Zmean according to --z-mode.
        rates_single = evaluator.compute_rates(
            Z_value=float(z_single),
            a_cm=a_cm,
            ne=ne,
            T=T,
            grain_type=grain_type,
            G0=G0,
        )
        gamma_simple_single = float(rates_single["Gamma_total"])
        recomb_simple_single_cached = float(rates_single["Recomb_total"])
        recomb_simple_single = float(
            compute_recombination_cooling_rate((float(z_single), float(a_cm), float(ne), float(T), grain_type))
        )
        recomb_simple_single_kernel_minus_cached = float(recomb_simple_single - recomb_simple_single_cached)

        # Approximation B: two-point interpolation around Zmean.
        rates_interp_lo = evaluator.compute_rates(
            Z_value=float(z_lo),
            a_cm=a_cm,
            ne=ne,
            T=T,
            grain_type=grain_type,
            G0=G0,
        )
        if z_hi == z_lo:
            rates_interp_hi = rates_interp_lo
            print(f"Warning: z_lo and z_hi are the same ({z_lo}), skipping redundant rates computation")
            print(f"Happened for T={T}, gamma={row['gamma']}, grain_type={grain_type}, a_cm={a_cm}")
        else:
            rates_interp_hi = evaluator.compute_rates(
                Z_value=float(z_hi),
                a_cm=a_cm,
                ne=ne,
                T=T,
                grain_type=grain_type,
                G0=G0,
            )

        gamma_simple_interp = float(
            w_lo * float(rates_interp_lo["Gamma_total"]) + w_hi * float(rates_interp_hi["Gamma_total"])
        )
        recomb_simple_interp_cached = float(
            w_lo * float(rates_interp_lo["Recomb_total"]) + w_hi * float(rates_interp_hi["Recomb_total"])
        )

        recomb_lo = float(
            compute_recombination_cooling_rate((float(z_lo), float(a_cm), float(ne), float(T), grain_type))
        )
        if z_hi == z_lo:
            recomb_hi = recomb_lo
        else:
            recomb_hi = float(
                compute_recombination_cooling_rate((float(z_hi), float(a_cm), float(ne), float(T), grain_type))
            )
        recomb_simple_interp = float(w_lo * recomb_lo + w_hi * recomb_hi)
        recomb_simple_interp_kernel_minus_cached = float(recomb_simple_interp - recomb_simple_interp_cached)

        # Approximation C: three-point moment-matching around Zmean and Zsigma.
        rates_3pt_by_z: Dict[int, Dict[str, float]] = {}
        for z3 in (int(z3_lo), int(z3_mid), int(z3_hi)):
            if z3 not in rates_3pt_by_z:
                rates_3pt_by_z[z3] = evaluator.compute_rates(
                    Z_value=float(z3),
                    a_cm=a_cm,
                    ne=ne,
                    T=T,
                    grain_type=grain_type,
                    G0=G0,
                )

        gamma_simple_3pt = float(
            w3_lo * float(rates_3pt_by_z[int(z3_lo)]["Gamma_total"])
            + w3_mid * float(rates_3pt_by_z[int(z3_mid)]["Gamma_total"])
            + w3_hi * float(rates_3pt_by_z[int(z3_hi)]["Gamma_total"])
        )
        recomb_simple_3pt_cached = float(
            w3_lo * float(rates_3pt_by_z[int(z3_lo)]["Recomb_total"])
            + w3_mid * float(rates_3pt_by_z[int(z3_mid)]["Recomb_total"])
            + w3_hi * float(rates_3pt_by_z[int(z3_hi)]["Recomb_total"])
        )

        recomb_3pt_by_z: Dict[int, float] = {}
        for z3 in (int(z3_lo), int(z3_mid), int(z3_hi)):
            if z3 not in recomb_3pt_by_z:
                recomb_3pt_by_z[z3] = float(
                    compute_recombination_cooling_rate((float(z3), float(a_cm), float(ne), float(T), grain_type))
                )
        recomb_simple_3pt = float(
            w3_lo * recomb_3pt_by_z[int(z3_lo)]
            + w3_mid * recomb_3pt_by_z[int(z3_mid)]
            + w3_hi * recomb_3pt_by_z[int(z3_hi)]
        )
        recomb_simple_3pt_kernel_minus_cached = float(recomb_simple_3pt - recomb_simple_3pt_cached)

        _print_three_point_high_error_diagnostics(
            grain_type=grain_type,
            a_cm=a_cm,
            T=T,
            gamma=float(row.get("gamma", np.nan)),
            G0=G0,
            ne=ne,
            zmean_table=zmean_table,
            zmean_full=float(zmean_full),
            zsigma_full=float(zsigma_full),
            Zs_arr=Zs_arr,
            P_arr=P_arr,
            recomb_z_arr=recomb_z_arr,
            recomb_full=recomb_full,
            recomb_simple_3pt=recomb_simple_3pt,
            z3_lo=int(z3_lo),
            z3_mid=int(z3_mid),
            z3_hi=int(z3_hi),
            w3_lo=float(w3_lo),
            w3_mid=float(w3_mid),
            w3_hi=float(w3_hi),
            recomb_3pt_lo=float(recomb_3pt_by_z[int(z3_lo)]),
            recomb_3pt_mid=float(recomb_3pt_by_z[int(z3_mid)]),
            recomb_3pt_hi=float(recomb_3pt_by_z[int(z3_hi)]),
            threshold_dex=0.2,
        )

        return {
            "source_file": row["source_file"],
            "bin_id": row["bin_id"],
            "bin_rank": row["bin_rank"],
            "grain_type": grain_type,
            "a_micron": row["a_micron"],
            "gamma": row["gamma"],
            "G0": G0,
            "ne": ne,
            "T": T,
            "Zmean_table": zmean_table,
            "Zmean_for_approx": float(zmean_for_approx),
            "Zmean_for_approx_source": zmean_for_approx_source,
            "Zsigma_table": row["Zsigma"],
            "Z_used_simplified": z_single,
            "Z_interp_lo": float(z_lo),
            "Z_interp_hi": float(z_hi),
            "Z_interp_w_lo": float(w_lo),
            "Z_interp_w_hi": float(w_hi),
            "Z_3pt_lo": float(z3_lo),
            "Z_3pt_mid": float(z3_mid),
            "Z_3pt_hi": float(z3_hi),
            "Z_3pt_w_lo": float(w3_lo),
            "Z_3pt_w_mid": float(w3_mid),
            "Z_3pt_w_hi": float(w3_hi),
            "Zmean_full": float(zmean_full),
            "Zsigma_full": float(zsigma_full),
            "Gamma_full": gamma_full,
            "Gamma_simple_single": gamma_simple_single,
            "Gamma_simple_interp": gamma_simple_interp,
            "Gamma_simple_3pt": gamma_simple_3pt,
            "Gamma_delta_single": gamma_simple_single - gamma_full,
            "Gamma_delta_interp": gamma_simple_interp - gamma_full,
            "Gamma_delta_3pt": gamma_simple_3pt - gamma_full,
            "Gamma_rel_delta_single": _relative_difference(gamma_simple_single, gamma_full),
            "Gamma_rel_delta_interp": _relative_difference(gamma_simple_interp, gamma_full),
            "Gamma_rel_delta_3pt": _relative_difference(gamma_simple_3pt, gamma_full),
            "Recomb_full": recomb_full,
            "Recomb_full_kernel_check": recomb_full_kernel_check,
            "Recomb_full_plus_auto": float(recomb_full + auto_full),
            "Recomb_full_Z0_contrib": recomb_full_z0_contrib,
            "Recomb_full_frac_from_Z0": recomb_full_frac_from_z0,
            "Recomb_full_Zminp1_contrib": recomb_full_zminp1_contrib,
            "Recomb_full_frac_from_Zminp1": recomb_full_frac_from_zminp1,
            "Recomb_simple_single": recomb_simple_single,
            "Recomb_simple_single_cached": recomb_simple_single_cached,
            "Recomb_simple_single_kernel_minus_cached": recomb_simple_single_kernel_minus_cached,
            "Recomb_delta_single": recomb_simple_single - recomb_full,
            "Recomb_rel_delta_single": _relative_difference(recomb_simple_single, recomb_full),
            "Recomb_simple_interp": recomb_simple_interp,
            "Recomb_simple_interp_cached": recomb_simple_interp_cached,
            "Recomb_simple_interp_kernel_minus_cached": recomb_simple_interp_kernel_minus_cached,
            "Recomb_delta_interp": recomb_simple_interp - recomb_full,
            "Recomb_rel_delta_interp": _relative_difference(recomb_simple_interp, recomb_full),
            "Recomb_simple_3pt": recomb_simple_3pt,
            "Recomb_simple_3pt_cached": recomb_simple_3pt_cached,
            "Recomb_simple_3pt_kernel_minus_cached": recomb_simple_3pt_kernel_minus_cached,
            "Recomb_delta_3pt": recomb_simple_3pt - recomb_full,
            "Recomb_rel_delta_3pt": _relative_difference(recomb_simple_3pt, recomb_full),
            "Autoionisation_full": auto_full,
        }
    except Exception as exc:
        return {
            "source_file": row.get("source_file", ""),
            "bin_id": row.get("bin_id", ""),
            "bin_rank": row.get("bin_rank", -1),
            "grain_type": row.get("grain_type", ""),
            "a_micron": row.get("a_micron", np.nan),
            "gamma": row.get("gamma", np.nan),
            "G0": row.get("G0", np.nan),
            "ne": row.get("ne", np.nan),
            "T": row.get("T", np.nan),
            "Zmean_table": row.get("Zmean", np.nan),
            "Zmean_for_approx": np.nan,
            "Zmean_for_approx_source": "",
            "Zsigma_table": row.get("Zsigma", np.nan),
            "Z_used_simplified": np.nan,
            "Z_interp_lo": np.nan,
            "Z_interp_hi": np.nan,
            "Z_interp_w_lo": np.nan,
            "Z_interp_w_hi": np.nan,
            "Z_3pt_lo": np.nan,
            "Z_3pt_mid": np.nan,
            "Z_3pt_hi": np.nan,
            "Z_3pt_w_lo": np.nan,
            "Z_3pt_w_mid": np.nan,
            "Z_3pt_w_hi": np.nan,
            "Zmean_full": np.nan,
            "Zsigma_full": np.nan,
            "Gamma_full": np.nan,
            "Gamma_simple_single": np.nan,
            "Gamma_simple_interp": np.nan,
            "Gamma_simple_3pt": np.nan,
            "Gamma_delta_single": np.nan,
            "Gamma_delta_interp": np.nan,
            "Gamma_delta_3pt": np.nan,
            "Gamma_rel_delta_single": np.nan,
            "Gamma_rel_delta_interp": np.nan,
            "Gamma_rel_delta_3pt": np.nan,
            "Recomb_full": np.nan,
            "Recomb_full_kernel_check": np.nan,
            "Recomb_full_plus_auto": np.nan,
            "Recomb_full_Z0_contrib": np.nan,
            "Recomb_full_frac_from_Z0": np.nan,
            "Recomb_full_Zminp1_contrib": np.nan,
            "Recomb_full_frac_from_Zminp1": np.nan,
            "Recomb_simple_single": np.nan,
            "Recomb_simple_single_cached": np.nan,
            "Recomb_simple_single_kernel_minus_cached": np.nan,
            "Recomb_delta_single": np.nan,
            "Recomb_rel_delta_single": np.nan,
            "Recomb_simple_interp": np.nan,
            "Recomb_simple_interp_cached": np.nan,
            "Recomb_simple_interp_kernel_minus_cached": np.nan,
            "Recomb_delta_interp": np.nan,
            "Recomb_rel_delta_interp": np.nan,
            "Recomb_simple_3pt": np.nan,
            "Recomb_simple_3pt_cached": np.nan,
            "Recomb_simple_3pt_kernel_minus_cached": np.nan,
            "Recomb_delta_3pt": np.nan,
            "Recomb_rel_delta_3pt": np.nan,
            "Autoionisation_full": np.nan,
            "error": str(exc),
        }


def _iter_input_rows(files: Iterable[Path]) -> Iterable[Dict[str, float]]:
    for fp in files:
        with open(fp, "r", encoding="utf-8") as fh:
            payload = json.load(fh)

        grain_type = str(payload.get("grain_type", "")).strip().lower()
        if not grain_type:
            comp = str(payload.get("composition", "")).strip().lower()
            grain_type = "silicate" if comp == "silicate" else "graphite"

        a_micron = _safe_float(payload.get("grain_size_micron"))
        if not np.isfinite(a_micron):
            continue
        a_cm = float(a_micron) * 1e-4

        bin_id = str(payload.get("bin_id", ""))
        bin_rank = int(payload.get("bin_rank", -1))

        for r in payload.get("results", []):
            G0 = _safe_float(r.get("G0"))
            ne = _safe_float(r.get("ne"))
            T = _safe_float(r.get("T"))
            gamma = _safe_float(r.get("gamma"))
            zmean = _safe_float(r.get("Zmean"))
            zsigma = _safe_float(r.get("Zsigma"))
            if not (np.isfinite(G0) and np.isfinite(ne) and np.isfinite(T) and np.isfinite(zmean)):
                continue
            if ne <= 0.0 or T <= 0.0:
                continue
            yield {
                "source_file": str(fp),
                "bin_id": bin_id,
                "bin_rank": bin_rank,
                "grain_type": grain_type,
                "a_cm": a_cm,
                "a_micron": a_micron,
                "gamma": gamma,
                "G0": G0,
                "ne": ne,
                "T": T,
                "Zmean": zmean,
                "Zsigma": zsigma,
            }


def _find_closest_zmean_table(grain_type: str, a_cm: float) -> Path:
    mat = "Gra" if str(grain_type).lower().startswith("gra") else "suvSil"
    candidates = sorted(_dust_charging_output_dir().glob(f"dust_charge_Z_vs_T_*_cm_{mat}.dat"))
    if not candidates:
        raise FileNotFoundError(
            f"No precomputed Zmean tables found in {_dust_charging_output_dir()} for material tag {mat}"
        )

    rgx = re.compile(r"dust_charge_Z_vs_T_([0-9eE+\-.]+)_cm_")
    best = None
    best_score = np.inf
    target = float(a_cm)
    for p in candidates:
        m = rgx.search(p.name)
        if m is None:
            continue
        try:
            a_file = float(m.group(1))
        except Exception:
            continue
        score = abs(np.log10(max(a_file, 1e-300)) - np.log10(max(target, 1e-300)))
        if score < best_score:
            best_score = score
            best = p
    if best is None:
        raise FileNotFoundError("Could not parse grain size from any Zmean table filename")
    return best


def _read_precomputed_zmean_table(path: Path) -> Dict[str, np.ndarray]:
    with open(path, "r", encoding="utf-8") as fh:
        lines = [ln.strip() for ln in fh.readlines() if ln.strip() and not ln.lstrip().startswith("#")]

    if len(lines) < 4:
        raise ValueError(f"Unexpected short table format in {path}")

    # Support both legacy (ngamma nT) and unified (nT ngamma) count ordering.
    # The table layout is:
    #   comments
    #   count line
    #   T axis line
    #   gamma axis line
    #   matrix rows over gamma
    try:
        dims = lines[0].split()
        dim0 = int(dims[0])
        dim1 = int(dims[1])
        T_log = np.fromstring(lines[1], sep=" ", dtype=float)
        g_log = np.fromstring(lines[2], sep=" ", dtype=float)
    except Exception as exc:
        raise ValueError(f"Could not parse precomputed Zmean table header from {path}: {exc}")

    if T_log.size == dim0 and g_log.size == dim1:
        nT, ng = dim0, dim1
    elif T_log.size == dim1 and g_log.size == dim0:
        nT, ng = dim1, dim0
    else:
        raise ValueError(
            f"Header/size mismatch in {path}: got nT={T_log.size} ng={g_log.size}, count line={dim0} {dim1}"
        )

    raw_rows = lines[3: 3 + ng]
    if len(raw_rows) < ng:
        raise ValueError(f"Missing matrix rows in {path}: expected {ng}, got {len(raw_rows)}")

    mat = np.full((ng, nT), np.nan, dtype=float)
    for i, row in enumerate(raw_rows):
        vals = np.fromstring(row, sep=" ", dtype=float)
        if vals.size >= nT:
            mat[i, :] = vals[:nT]

    return {
        "gamma_log10": g_log,
        "T_log10": T_log,
        "Zmean": mat,
        "path": np.array([str(path)], dtype=object),
    }


def _lookup_zmean_from_precomputed(table: Dict[str, np.ndarray], gamma: float, T: float) -> float:
    g_log = np.asarray(table["gamma_log10"], dtype=float)
    T_log = np.asarray(table["T_log10"], dtype=float)
    Zmat = np.asarray(table["Zmean"], dtype=float)

    lg = float(np.log10(max(gamma, 1e-300)))
    lT = float(np.log10(max(T, 1e-300)))

    if g_log.size < 2 or T_log.size < 2:
        m = np.isfinite(Zmat)
        if not np.any(m):
            raise ValueError("No finite Zmean values in precomputed table")
        return float(np.nanmedian(Zmat[m]))

    ig = int(np.clip(np.searchsorted(g_log, lg) - 1, 0, g_log.size - 2))
    jT = int(np.clip(np.searchsorted(T_log, lT) - 1, 0, T_log.size - 2))

    g0, g1 = g_log[ig], g_log[ig + 1]
    t0, t1 = T_log[jT], T_log[jT + 1]
    q11 = Zmat[ig, jT]
    q12 = Zmat[ig, jT + 1]
    q21 = Zmat[ig + 1, jT]
    q22 = Zmat[ig + 1, jT + 1]

    vals = np.array([q11, q12, q21, q22], dtype=float)
    if np.all(np.isfinite(vals)) and (g1 > g0) and (t1 > t0):
        wg = (lg - g0) / (g1 - g0)
        wt = (lT - t0) / (t1 - t0)
        return float(
            (1 - wg) * (1 - wt) * q11
            + (1 - wg) * wt * q12
            + wg * (1 - wt) * q21
            + wg * wt * q22
        )

    # Fallback nearest finite neighbor in log-space.
    GG, TT = np.meshgrid(g_log, T_log, indexing="ij")
    mask = np.isfinite(Zmat)
    if not np.any(mask):
        raise ValueError("No finite Zmean values in precomputed table")
    d2 = (GG - lg) ** 2 + (TT - lT) ** 2
    d2[~mask] = np.inf
    ii, jj = np.unravel_index(np.argmin(d2), d2.shape)
    return float(Zmat[ii, jj])


def _build_grid_rows(
    grain_type: str,
    a_cm: float,
    fixed_G0: float,
    Tmin: float,
    Tmax: float,
    nT: int,
    gamma_min: float,
    gamma_max: float,
    n_gamma: int,
    z_table_path: Path,
) -> List[Dict[str, float]]:
    table = _read_precomputed_zmean_table(z_table_path)
    T_vals = np.logspace(np.log10(Tmin), np.log10(Tmax), nT)
    gamma_vals = np.logspace(np.log10(gamma_min), np.log10(gamma_max), n_gamma)

    rows: List[Dict[str, float]] = []
    for T in T_vals:
        sqrtT = np.sqrt(float(T))
        for gamma in gamma_vals:
            ne = max(1e-300, float(fixed_G0) * sqrtT / float(gamma))
            zmean = _lookup_zmean_from_precomputed(table, float(gamma), float(T))
            rows.append(
                {
                    "source_file": str(z_table_path),
                    "bin_id": "grid",
                    "bin_rank": 0,
                    "grain_type": str(grain_type),
                    "a_cm": float(a_cm),
                    "a_micron": float(a_cm) * 1e4,
                    "gamma": float(gamma),
                    "G0": float(fixed_G0),
                    "ne": float(ne),
                    "T": float(T),
                    "Zmean": float(zmean),
                    "Zsigma": np.nan,
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    repo = _repo_root()
    default_pattern = str(repo / "model_data" / "dust_charging_data" / "charging_vs_gamma_*.json")
    default_base = str(repo / "model_data" / "dust_photoelectric_heating_data" / "zmean_rate_approx_comparison")

    p = argparse.ArgumentParser(
        description="Compare full charge-distribution rates against a Zmean-only approximation"
    )
    p.add_argument(
        "--run-mode",
        choices=["from-json", "grid-fix-g0"],
        default="grid-fix-g0",
        help="Use existing charging JSON samples or build an explicit fixed-G0 T-gamma grid",
    )
    p.add_argument(
        "--input",
        default=default_pattern,
        help="Input charging JSON path or glob pattern",
    )
    p.add_argument(
        "--output-summary",
        default=None,
        help="Optional summary JSON path (defaults to zmean_rate_approx_comparison.summary.json)",
    )
    p.add_argument(
        "--radiation-model",
        default="Mathis",
        help="Radiation model used in rate calculations",
    )
    p.add_argument(
        "--z-mode",
        choices=["raw", "round", "floor", "ceil"],
        default="raw",
        help="How to convert table Zmean into the Z value used by the simplified model",
    )
    p.add_argument(
        "--use-full-zmean-for-approx",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use Zmean from the full equilibrium distribution (per point) in the simplified single/two-point approximations instead of table Zmean",
    )
    p.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Optional cap on number of points to process",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Process every N-th point (N>=1)",
    )
    p.add_argument(
        "--scale-three-col-with-g0",
        action="store_true",
        help="Scale 3-column radiation fields by G0 in simplified model (off by default to mimic current wrapper behavior)",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N processed points",
    )
    p.add_argument(
        "--output-plot",
        default=None,
        help="Output PNG path for gamma-T ratio map (default: zmean_rate_approx_comparison.ratio_map.png)",
    )
    p.add_argument(
        "--gamma-bins",
        type=int,
        default=24,
        help="Number of gamma bins for the ratio map",
    )
    p.add_argument(
        "--T-bins",
        type=int,
        default=24,
        help="Number of temperature bins for the ratio map",
    )
    p.add_argument(
        "--min-bin-count",
        type=int,
        default=3,
        help="Minimum number of points required to populate a ratio-map bin",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: memory-aware auto selection)",
    )
    p.add_argument(
        "--per-worker-tmp-limit",
        type=int,
        default=128 * 1024 * 1024,
        help="Approximate temporary-memory budget per worker in bytes for auto worker sizing",
    )
    p.add_argument(
        "--grain-type",
        choices=["graphite", "silicate"],
        default="silicate",
        help="Grain type for grid-fix-g0 mode",
    )
    p.add_argument(
        "--a-cm",
        type=float,
        default=5e-7,
        help="Grain radius in cm for grid-fix-g0 mode",
    )
    p.add_argument(
        "--fixed-g0",
        type=float,
        default=1.0,
        help="Fixed G0 value for grid-fix-g0 mode",
    )
    p.add_argument("--Tmin", type=float, default=10.0, help="Minimum temperature [K] for grid-fix-g0 mode")
    p.add_argument("--Tmax", type=float, default=1e7, help="Maximum temperature [K] for grid-fix-g0 mode")
    p.add_argument("--nT", type=int, default=20, help="Number of temperature points for grid-fix-g0 mode")
    p.add_argument("--gamma-min", type=float, default=1e-4, help="Minimum gamma for grid-fix-g0 mode")
    p.add_argument("--gamma-max", type=float, default=1e6, help="Maximum gamma for grid-fix-g0 mode")
    p.add_argument("--n-gamma", type=int, default=20, help="Number of gamma points for grid-fix-g0 mode")
    p.add_argument(
        "--z-table",
        default=None,
        help="Optional explicit precomputed Zmean matrix file (dust_charge_Z_vs_T_*). If omitted, closest-size table is auto-selected.",
    )
    p.add_argument(
        "--annotate-cells",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Annotate each heatmap cell with Zmean and sigma",
    )
    p.add_argument(
        "--annotation-fontsize",
        type=int,
        default=5,
        help="Font size for per-cell Zmean/sigma annotations",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo = _repo_root()

    in_paths: List[Path] = []
    if args.run_mode == "from-json":
        in_paths = [Path(p) for p in sorted(glob.glob(args.input))]
        if len(in_paths) == 0:
            raise FileNotFoundError(f"No charging JSON files found for pattern: {args.input}")
        all_rows = list(_iter_input_rows(in_paths))
        if len(all_rows) == 0:
            raise RuntimeError("No valid rows found in the provided charging JSON files")
    else:
        if args.z_table is not None:
            ztab = Path(args.z_table)
            if not ztab.is_absolute():
                ztab = (_dust_charging_output_dir() / ztab).resolve()
        else:
            ztab = _find_closest_zmean_table(args.grain_type, float(args.a_cm))
        all_rows = _build_grid_rows(
            grain_type=args.grain_type,
            a_cm=float(args.a_cm),
            fixed_G0=float(args.fixed_g0),
            Tmin=float(args.Tmin),
            Tmax=float(args.Tmax),
            nT=max(2, int(args.nT)),
            gamma_min=float(args.gamma_min),
            gamma_max=float(args.gamma_max),
            n_gamma=max(2, int(args.n_gamma)),
            z_table_path=ztab,
        )

    stride = max(1, int(args.stride))
    selected = all_rows[::stride]
    if args.max_points is not None and args.max_points > 0 and len(selected) > int(args.max_points):
        nsel = int(args.max_points)
        # Use evenly spaced indices so quick tests still span the whole gamma/T range.
        idx = np.linspace(0, len(selected) - 1, nsel, dtype=int)
        idx = np.unique(idx)
        selected = [selected[i] for i in idx]

    print("=" * 80)
    print("Comparing full model vs Zmean-only approximation")
    print("=" * 80)
    print(f"Run mode: {args.run_mode}")
    print(f"Input files: {len(in_paths)}")
    print(f"Total valid rows available: {len(all_rows)}")
    print(f"Rows selected: {len(selected)} (stride={stride}, max_points={args.max_points})")
    print(f"Z mode: {args.z_mode}")
    print(f"Approximation Zmean source: {'full' if args.use_full_zmean_for_approx else 'table'}")
    print(f"Radiation model: {args.radiation_model}")
    if args.run_mode == "grid-fix-g0":
        print(
            f"Grid setup: G0={float(args.fixed_g0):.3e}, T=[{float(args.Tmin):.3e},{float(args.Tmax):.3e}] "
            f"nT={int(args.nT)}, gamma=[{float(args.gamma_min):.3e},{float(args.gamma_max):.3e}] n_gamma={int(args.n_gamma)}"
        )
    print("=" * 80)

    output_rows: List[Dict[str, float]] = []
    tasks = [
        {
            "row": row,
            "z_mode": args.z_mode,
            "radiation_model": args.radiation_model,
            "scale_three_col_with_g0": bool(args.scale_three_col_with_g0),
            "use_full_zmean_for_approx": bool(args.use_full_zmean_for_approx),
        }
        for row in selected
    ]

    # Memory-aware worker selection following dust_charging strategy.
    total_mem = get_system_memory_bytes()
    proc_rss = get_process_rss_bytes()
    avail_mem = max(0, total_mem - proc_rss)
    budget_for_workers = int(avail_mem * 0.6)
    cpu_count = _mp.cpu_count()

    if args.workers is not None:
        n_workers = max(1, int(args.workers))
    else:
        per_worker_tmp_limit = max(2 * 1024 * 1024, int(args.per_worker_tmp_limit))
        max_by_mem = max(1, int(max(1, budget_for_workers) // max(1, per_worker_tmp_limit * 3)))
        n_workers = max(1, min(cpu_count, max_by_mem, len(tasks)))

    print(
        f"Using {n_workers} worker(s); total_mem={total_mem/(1024**3):.2f} GiB, "
        f"avail={avail_mem/(1024**2):.1f} MiB"
    )

    if n_workers == 1:
        for idx, task in enumerate(tasks, start=1):
            output_rows.append(_compute_single_row(task))
            if args.progress_every > 0 and (idx % args.progress_every == 0 or idx == len(tasks)):
                print(f"Processed {idx}/{len(tasks)} points")
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as exe:
            fut_to_idx = {exe.submit(_compute_single_row, task): i for i, task in enumerate(tasks)}
            completed = 0
            for fut in as_completed(fut_to_idx):
                output_rows.append(fut.result())
                completed += 1
                if args.progress_every > 0 and (completed % args.progress_every == 0 or completed == len(tasks)):
                    print(f"Processed {completed}/{len(tasks)} points")

    # restore deterministic order after as_completed
    if len(output_rows) == len(tasks):
        row_by_key = {
            (r.get("source_file"), r.get("gamma"), r.get("G0"), r.get("ne"), r.get("T")): r
            for r in output_rows
        }
        ordered = []
        for t in tasks:
            rr = t["row"]
            k = (rr.get("source_file"), rr.get("gamma"), rr.get("G0"), rr.get("ne"), rr.get("T"))
            ordered.append(row_by_key.get(k, {}))
        output_rows = ordered

    n_errors = sum(1 for r in output_rows if r.get("error"))


    good = [r for r in output_rows if not r.get("error")]

    def _percentiles(vals: List[float]) -> Dict[str, float]:
        arr = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
        if arr.size == 0:
            return {"p50": np.nan, "p90": np.nan, "p99": np.nan, "mean": np.nan, "std": np.nan}
        abs_arr = np.abs(arr)
        return {
            "p50": float(np.percentile(abs_arr, 50)),
            "p90": float(np.percentile(abs_arr, 90)),
            "p99": float(np.percentile(abs_arr, 99)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
        }

    gamma_rel_single = [r.get("Gamma_rel_delta_single", np.nan) for r in good]
    gamma_rel_interp = [r.get("Gamma_rel_delta_interp", np.nan) for r in good]
    gamma_rel_3pt = [r.get("Gamma_rel_delta_3pt", np.nan) for r in good]
    recomb_rel_single = [r.get("Recomb_rel_delta_single", np.nan) for r in good]
    recomb_rel_interp = [r.get("Recomb_rel_delta_interp", np.nan) for r in good]
    recomb_rel_3pt = [r.get("Recomb_rel_delta_3pt", np.nan) for r in good]

    recomb_kernel_consistency_single = [
        r.get("Recomb_simple_single_kernel_minus_cached", np.nan) for r in good
    ]
    recomb_kernel_consistency_interp = [
        r.get("Recomb_simple_interp_kernel_minus_cached", np.nan) for r in good
    ]
    recomb_kernel_consistency_3pt = [
        r.get("Recomb_simple_3pt_kernel_minus_cached", np.nan) for r in good
    ]
    recomb_frac_z0_vals = [
        r.get("Recomb_full_frac_from_Z0", np.nan) for r in good
    ]
    recomb_frac_zminp1_vals = [
        r.get("Recomb_full_frac_from_Zminp1", np.nan) for r in good
    ]
    recomb_full_consistency = [
        _safe_float(r.get("Recomb_full", np.nan)) - _safe_float(r.get("Recomb_full_kernel_check", np.nan))
        for r in good
    ]

    recomb_log_ratio = []
    zsigma_vals = []
    for r in good:
        rs = _safe_float(r.get("Recomb_simple_interp", np.nan))
        rf = _safe_float(r.get("Recomb_full", np.nan))
        zs = _safe_float(r.get("Zsigma_full", np.nan))
        if rs > 0.0 and rf > 0.0 and np.isfinite(zs):
            recomb_log_ratio.append(float(np.log10(rs / rf)))
            zsigma_vals.append(float(zs))

    corr_abs_logratio_zsigma = np.nan
    if len(recomb_log_ratio) >= 2:
        a = np.abs(np.asarray(recomb_log_ratio, dtype=float))
        b = np.asarray(zsigma_vals, dtype=float)
        if np.std(a) > 0.0 and np.std(b) > 0.0:
            corr_abs_logratio_zsigma = float(np.corrcoef(a, b)[0, 1])

    # Extract high-error cells for diagnostic analysis (three-point cooling metric)
    high_error_threshold = 0.2
    high_error_cells = [
        {
            "T": r.get("T"),
            "gamma": r.get("gamma"),
            "Zmean_for_approx": r.get("Zmean_for_approx"),
            "Zsigma": r.get("Zsigma_full"),
            "Recomb_rel_delta_3pt": r.get("Recomb_rel_delta_3pt"),
            "Recomb_rel_delta_3pt_dex": float(np.log10(abs(r.get("Recomb_rel_delta_3pt", 1e-300)))) if r.get("Recomb_rel_delta_3pt", 0) != 0 else 0,
            "Recomb_full_frac_from_Z0": r.get("Recomb_full_frac_from_Z0"),
            "Recomb_full_frac_from_Zminp1": r.get("Recomb_full_frac_from_Zminp1"),
            "Gamma_rel_delta_3pt": r.get("Gamma_rel_delta_3pt"),
        }
        for r in good
        if abs(r.get("Recomb_rel_delta_3pt", 0)) > high_error_threshold
    ]
    high_error_cells_sorted = sorted(
        high_error_cells,
        key=lambda x: abs(x["Recomb_rel_delta_3pt"]),
        reverse=True
    )

    summary = {
        "run_mode": args.run_mode,
        "input_pattern": args.input,
        "input_files": [str(p) for p in in_paths],
        "n_rows_available": len(all_rows),
        "n_rows_selected": len(selected),
        "n_high_error_cells": len(high_error_cells),
        "high_error_threshold_dex": high_error_threshold,
        "high_error_cells_sample": high_error_cells_sorted[:20],
        "n_rows_written": len(output_rows),
        "n_success": len(good),
        "n_errors": n_errors,
        "z_mode": args.z_mode,
        "use_full_zmean_for_approx": bool(args.use_full_zmean_for_approx),
        "zmean_for_approx_source": "full" if args.use_full_zmean_for_approx else "table",
        "radiation_model": args.radiation_model,
        "workers_used": int(n_workers),
        "gamma_rel_delta_single_stats": _percentiles(gamma_rel_single),
        "gamma_rel_delta_interp_stats": _percentiles(gamma_rel_interp),
        "gamma_rel_delta_3pt_stats": _percentiles(gamma_rel_3pt),
        "recomb_rel_delta_single_stats": _percentiles(recomb_rel_single),
        "recomb_rel_delta_interp_stats": _percentiles(recomb_rel_interp),
        "recomb_rel_delta_3pt_stats": _percentiles(recomb_rel_3pt),
        "recombination_diagnostics": {
            "single_kernel_minus_cached_stats": _percentiles(recomb_kernel_consistency_single),
            "interp_kernel_minus_cached_stats": _percentiles(recomb_kernel_consistency_interp),
            "three_point_kernel_minus_cached_stats": _percentiles(recomb_kernel_consistency_3pt),
            "full_recomb_minus_weighted_RecombZ_stats": _percentiles(recomb_full_consistency),
            "full_recomb_fraction_from_Z0_stats": _percentiles(recomb_frac_z0_vals),
            "full_recomb_fraction_from_Zminp1_stats": _percentiles(recomb_frac_zminp1_vals),
            "corr_abs_log10_recomb_ratio_interp_vs_Zsigma_full": corr_abs_logratio_zsigma,
        },
    }

    out_summary = (
        Path(args.output_summary)
        if args.output_summary
        else Path(repo / "model_data" / "dust_photoelectric_heating_data" / "zmean_rate_approx_comparison.summary.json")
    )
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    with open(out_summary, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    out_plot = (
        Path(args.output_plot)
        if args.output_plot
        else Path(repo / "model_data" / "dust_photoelectric_heating_data" / "zmean_rate_approx_comparison.ratio_map.png")
    )
    try:
        plot_meta = _plot_ratio_maps(
            output_rows=output_rows,
            out_path=out_plot,
            n_gamma_bins=max(4, int(args.gamma_bins)),
            n_T_bins=max(4, int(args.T_bins)),
            min_bin_count=max(1, int(args.min_bin_count)),
            annotate_cells=bool(args.annotate_cells),
            annotation_fontsize=max(4, int(args.annotation_fontsize)),
        )
        summary["ratio_map"] = plot_meta
        with open(out_summary, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
    except Exception as exc:
        summary["ratio_map_error"] = str(exc)
        with open(out_summary, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(f"[warn] Could not create ratio map: {exc}")

    print("=" * 80)
    print("Done")
    print(f"Summary: {out_summary}")
    if "ratio_map" in summary:
        print(f"Ratio map: {summary['ratio_map']['plot_path']}")
    print(f"Successful rows: {summary['n_success']} / {summary['n_rows_selected']}")
    print("|Gamma_single - Gamma_full| / |Gamma_full| stats:", summary["gamma_rel_delta_single_stats"])
    print("|Gamma_interp - Gamma_full| / |Gamma_full| stats:", summary["gamma_rel_delta_interp_stats"])
    print("|Gamma_3pt - Gamma_full| / |Gamma_full| stats:", summary["gamma_rel_delta_3pt_stats"])
    print("|Recomb_single - Recomb_full| / |Recomb_full| stats:", summary["recomb_rel_delta_single_stats"])
    print("|Recomb_interp - Recomb_full| / |Recomb_full| stats:", summary["recomb_rel_delta_interp_stats"])
    print("|Recomb_3pt - Recomb_full| / |Recomb_full| stats:", summary["recomb_rel_delta_3pt_stats"])
    print("=" * 80)


if __name__ == "__main__":
    main()
