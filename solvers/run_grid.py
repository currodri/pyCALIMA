"""Grid runner for CALIMA dust/PAH chemistry solvers.

Sweeps a rectangular grid of (T, nH), (T, G0), or any two environment
parameters and collects final-state results from either the RK4 time
integrator or the NewtonKrylov equilibrium solver.

Usage (CLI)
-----------
::

    # T–nH grid, RK4, 5 Myr
    python -m solvers.run_grid \\
        --config  solvers/configs/example_ic.json \\
        --x-param T     --x-values 50 100 500 2000 8000 \\
        --y-param nH    --y-values 0.1 1 10 100 1000 \\
        --t-end-Myr 5   --solver rk4 \\
        --output-npz  grid_T_nH.npz

    # T–G0 grid, equilibrium
    python -m solvers.run_grid \\
        --config  solvers/configs/equilibrium_postshock_test.json \\
        --x-param T   --x-values 100 1000 5000 20000 \\
        --y-param G0  --y-values 0.1 1 10 100 \\
        --solver newton_krylov \\
        --output-npz  grid_T_G0_eq.npz

Python API
----------
::

    from solvers.run_grid import run_grid
    grid = run_grid(
        config_path = "solvers/configs/example_ic.json",
        x_param  = "T",    x_values = [50, 100, 500, 2000, 8000],
        y_param  = "nH",   y_values = [0.1, 1, 10, 100, 1000],
        t_end_Myr = 5.0,
        solver_type = "rk4",
    )
    # grid["DTM"][i, j]   —  dust-to-metal ratio at (T[i], nH[j])
    # grid["rho_dust"]    —  shape (nx, ny, n_dust_bins) final dust densities
    # grid["rho_pah"]     —  shape (nx, ny, n_pah_bins)
    # grid["rho_gas"]     —  shape (nx, ny, n_elements)
    # grid["converged"]   —  shape (nx, ny) bool (equilibrium) or True (rk4)
    # grid["elapsed_s"]   —  shape (nx, ny) wall-clock time
    # grid["x_values"], grid["y_values"], grid["x_param"], grid["y_param"]
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np

from .chemistry_state import ELEMENT_NAMES
from .dust_init import load_initial_conditions, SEC2MYR
from .equilibrium import (
    EquilibriumSolverBase,
    NewtonKrylovEquilibriumSolver,
    SparseNewtonEquilibriumSolver,
)
from .ode_driver import integrate_dust_ode
from .rhs import build_process_list
from .anninos import AnninosSolver
from .rk4 import RK4Solver
from .rk54 import RK54Solver
from .run_chemistry import _make_solver, compute_element_totals

# ---------------------------------------------------------------------------
# Parameter name → state attribute mapping
# ---------------------------------------------------------------------------
_PARAM_TO_ATTR = {
    "T":    "local_Tk",
    "nH":   "local_nH",
    "ne":   "local_ne",
    "G0":   "local_G0",
    "mu":   "local_mu",
}

_PARAM_UNITS = {
    "T":  "K",
    "nH": "cm⁻³",
    "ne": "cm⁻³",
    "G0": "",
    "mu": "",
}

# ---------------------------------------------------------------------------
# DTM / depletion helpers
# ---------------------------------------------------------------------------

def _compute_dtm(state, y_gas_f: np.ndarray, y_dust_f: np.ndarray,
                 y_gas_0: np.ndarray, y_dust_0: np.ndarray) -> float:
    """Dust-to-metal mass ratio at the final state.

    DTM = (mass in all dust+PAH bins) / (total mass in dust-forming elements).
    Dust-forming elements are those that appear in at least one dust or PAH bin.
    """
    # Identify elements that participate in dust/PAH chemistry
    dust_el_set: set = set()
    c_idx = state.el_names.index("C") if "C" in state.el_names else 2
    for pb in state.pah_bins:
        dust_el_set.add(c_idx)
    for db in state.dust_bins:
        for ei in db.el_indices:
            dust_el_set.add(ei)
    if not dust_el_set:
        return 0.0

    total_dust_mass = float(y_dust_f.sum())
    total_metal_mass = sum(
        compute_element_totals(state, y_gas_f, y_dust_f)[e]
        for e in dust_el_set
    )
    return total_dust_mass / total_metal_mass if total_metal_mass > 0 else 0.0


def _compute_depletions(state, y_gas_f: np.ndarray, y_dust_f: np.ndarray,
                        y_gas_0: np.ndarray, y_dust_0: np.ndarray) -> np.ndarray:
    """Elemental depletion fraction = (mass in dust) / (total element mass)."""
    totals_0 = compute_element_totals(state, y_gas_0, y_dust_0)
    depletions = np.zeros(state.n_elements)

    c_idx = state.el_names.index("C") if "C" in state.el_names else 2
    # PAH contribution
    for pb in state.pah_bins:
        depletions[c_idx] += y_dust_f[pb.bin_index]
    # Dust contribution
    for db in state.dust_bins:
        rho = y_dust_f[state.npah + db.bin_index]
        for el_idx, frac in zip(db.el_indices, db.el_mfractions):
            depletions[el_idx] += rho * frac

    for e in range(state.n_elements):
        if totals_0[e] > 0.0:
            depletions[e] /= totals_0[e]
    return depletions


# ---------------------------------------------------------------------------
# Single-point runner
# ---------------------------------------------------------------------------

def _run_point(
    config_path: Path,
    x_param: str,
    x_val: float,
    y_param: str,
    y_val: float,
    t_end_s: float,
    solver_type: str,
    solver_kwargs: dict,
    t_end_mode: str = "fixed",
) -> dict:
    """Run chemistry at a single (x_val, y_val) environment point."""
    state, y_gas_0, y_dust_0 = load_initial_conditions(config_path)

    # Save reference nH before any override so we can rescale mass densities.
    # y_gas_0 is computed as mass_fraction * local_rho where
    # local_rho = nH * m_H * mu, so both y_gas_0, y_dust_0, and state.local_rho
    # must be rescaled whenever nH (or mu) changes.
    nH_ref  = state.local_nH
    mu_ref  = state.local_mu

    # Override environment parameters
    setattr(state, _PARAM_TO_ATTR[x_param], float(x_val))
    setattr(state, _PARAM_TO_ATTR[y_param], float(y_val))

    # Rescale initial mass densities to be consistent with the new nH / mu.
    # The reference density is rho_ref = nH_ref * m_H * mu_ref, so the
    # scale factor is simply (nH_new * mu_new) / (nH_ref * mu_ref).
    rho_scale = (state.local_nH * state.local_mu) / max(nH_ref * mu_ref, 1e-300)
    if abs(rho_scale - 1.0) > 1e-12:
        y_gas_0         = y_gas_0  * rho_scale
        y_dust_0        = y_dust_0 * rho_scale
        state.local_rho = state.local_rho * rho_scale

    # Override t_end_s with a physical timescale if requested
    if t_end_mode == "freefall":
        from .dust_init import freefall_time_s
        t_end_s = freefall_time_s(state.local_nH, state.local_mu)

    processes = build_process_list(state)

    # Build solver — re-use kwargs from config but allow override
    solver: object
    _ode_keys = ("errmax", "y_min")
    if solver_type == "rk4":
        solver = RK4Solver(**{k: v for k, v in solver_kwargs.items()
                              if k in ("errmax",)})
    elif solver_type == "rk54":
        solver = RK54Solver(**{k: v for k, v in solver_kwargs.items()
                               if k in _ode_keys})
    elif solver_type == "anninos":
        solver = AnninosSolver(**{k: v for k, v in solver_kwargs.items()
                                  if k in _ode_keys})
    elif solver_type == "newton_krylov":
        solver = NewtonKrylovEquilibriumSolver(**solver_kwargs)
    elif solver_type == "sparse_newton":
        solver = SparseNewtonEquilibriumSolver(**solver_kwargs)
    else:
        raise ValueError(f"Unknown solver_type: {solver_type!r}")

    t0 = time.perf_counter()

    if isinstance(solver, EquilibriumSolverBase):
        y_gas_f, y_dust_f, diag = solver.find_equilibrium(
            state, y_gas_0, y_dust_0, processes,
            t_eq_s=t_end_s, verbose=False,
        )
        converged = bool(diag["converged"])
    else:
        h_init = solver_kwargs.get("h_init_s", 1.0e10)
        h_min  = solver_kwargs.get("h_min_s",  1.0)
        h_max  = solver_kwargs.get("h_max_s",  t_end_s)
        y_gas_f, y_dust_f, diag = integrate_dust_ode(
            state, t_end_s, y_gas_0, y_dust_0, processes, solver,
            h_init=h_init, h_min=h_min, h_max=h_max,
            collect_history=False, verbose=False,
        )
        converged = True

    elapsed = time.perf_counter() - t0

    dtm       = _compute_dtm(state, y_gas_f, y_dust_f, y_gas_0, y_dust_0)
    depletions = _compute_depletions(state, y_gas_f, y_dust_f, y_gas_0, y_dust_0)

    return {
        "y_gas_f":    y_gas_f,
        "y_dust_f":   y_dust_f,
        "y_gas_0":    y_gas_0,
        "y_dust_0":   y_dust_0,
        "converged":  converged,
        "elapsed_s":  elapsed,
        "dtm":        dtm,
        "depletions": depletions,
        "state":      state,          # last state (for metadata)
    }


# ---------------------------------------------------------------------------
# Public grid API
# ---------------------------------------------------------------------------

def run_grid(
    config_path: Union[str, Path],
    *,
    x_param: str,
    x_values: Sequence[float],
    y_param: str,
    y_values: Sequence[float],
    t_end_Myr: float = 5.0,
    t_end_mode: str = "fixed",
    solver_type: Optional[str] = None,
    solver_kwargs: Optional[dict] = None,
    verbose: bool = True,
    n_jobs: int = 1,
) -> dict:
    """Run the chemistry solver on a 2-D grid of environment parameters.

    Parameters
    ----------
    config_path :
        Base JSON config (environment, bins, physics flags, etc.).
        The two grid parameters override the values in this file.
    x_param, y_param : str
        One of ``"T"``, ``"nH"``, ``"ne"``, ``"G0"``, ``"mu"``.
    x_values, y_values :
        1-D sequences of values for the grid axes.
    t_end_Myr : float
        Integration time (ignored for equilibrium solvers).
    solver_type : str, optional
        Override the solver type from the config.
        One of ``"rk4"``, ``"newton_krylov"``, ``"sparse_newton"``.
    solver_kwargs : dict, optional
        Extra keyword arguments forwarded to the solver constructor.
    verbose : bool
        Print progress table to stdout.
    n_jobs : int
        Number of parallel workers (>1 or -1 for all CPUs, requires ``joblib``).

    Returns
    -------
    dict with keys:

    ``x_param``, ``y_param``      — axis parameter names (str)
    ``x_values``, ``y_values``    — axis values (ndarray)
    ``x_units``, ``y_units``      — unit strings
    ``t_end_Myr``                 — integration time [Myr]
    ``solver_type``               — solver used
    ``DTM``                       — shape (nx, ny)
    ``rho_dust``                  — shape (nx, ny, n_dust_bins)
    ``rho_pah``                   — shape (nx, ny, n_pah_bins)
    ``rho_gas``                   — shape (nx, ny, n_elements)
    ``depletions``                — shape (nx, ny, n_elements)
    ``converged``                 — shape (nx, ny) bool
    ``elapsed_s``                 — shape (nx, ny)
    ``el_names``                  — list of element names
    ``dust_bin_ids``              — list of dust bin IDs
    ``pah_bin_ids``               — list of PAH bin IDs
    """
    config_path = Path(config_path)

    # Read solver type from config if not overridden
    with config_path.open() as fh:
        cfg = json.load(fh)
    _stype = solver_type or cfg.get("solver", {}).get("type", "rk4").lower()
    _skw   = dict(solver_kwargs or {})

    # Default solver kwargs from config
    sc = cfg.get("solver", {})
    if _stype in ("rk4", "rk54", "anninos") and "errmax" not in _skw:
        _skw.setdefault("errmax",       float(sc.get("errmax",        0.1)))
        _skw.setdefault("h_init_s",     float(sc.get("h_init_s",      1e10)))
        _skw.setdefault("h_min_s",      float(sc.get("h_min_s",       1.0)))
        _skw.setdefault("h_max_s",      float(sc.get("h_max_Myr", 1.0)) * SEC2MYR)
    elif _stype == "newton_krylov":
        _skw.setdefault("f_tol",        float(sc.get("f_tol",         1e-40)))
        _skw.setdefault("f_rtol",       sc.get("f_rtol",              1e-8))
        _skw.setdefault("maxiter",      int(sc.get("maxiter",          300)))
        _skw.setdefault("inner_maxiter",int(sc.get("inner_maxiter",    400)))
    elif _stype == "sparse_newton":
        _skw.setdefault("rtol",         float(sc.get("rtol",           1e-8)))
        _skw.setdefault("atol",         float(sc.get("atol",           1e-40)))
        _skw.setdefault("maxiter",      int(sc.get("maxiter",          50)))

    t_end_s = t_end_Myr * SEC2MYR

    xs = np.asarray(x_values, dtype=float)
    ys = np.asarray(y_values, dtype=float)
    nx, ny = len(xs), len(ys)

    # Load a reference state once to get shapes
    ref_state, _, _ = load_initial_conditions(config_path)
    ndust    = ref_state.ndust
    npah     = ref_state.npah
    n_el     = ref_state.n_elements
    el_names = list(ref_state.el_names)
    dust_ids = [db.bin_id for db in ref_state.dust_bins]
    pah_ids  = [pb.bin_id for pb in ref_state.pah_bins]

    # Output arrays
    DTM       = np.full((nx, ny), np.nan)
    rho_dust  = np.full((nx, ny, ndust), np.nan)
    rho_pah   = np.full((nx, ny, npah),  np.nan)
    rho_gas   = np.full((nx, ny, n_el),  np.nan)
    depls     = np.full((nx, ny, n_el),  np.nan)
    converged = np.zeros((nx, ny), dtype=bool)
    elapsed   = np.full((nx, ny), np.nan)

    if verbose:
        xu = _PARAM_UNITS[x_param]
        yu = _PARAM_UNITS[y_param]
        print(f"=== CALIMA Grid Run  ({_stype}) ===")
        print(f"  {x_param} [{xu}]: {xs.tolist()}")
        print(f"  {y_param} [{yu}]: {ys.tolist()}")
        print(f"  Grid size  : {nx} × {ny} = {nx*ny} points")
        if _stype in ("rk4", "rk54", "anninos"):
            if t_end_mode == "freefall":
                print(f"  t_end      : free-fall time per cell (t_end_Myr={t_end_Myr} Myr fallback)")
            else:
                print(f"  t_end      : {t_end_Myr} Myr")
        print()
        hdr = (f"  {'i':>3} {'j':>3}  "
               f"{x_param:>8}  {y_param:>8}  "
               f"{'DTM':>10}  {'converged':>9}  {'time/s':>7}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))

    def _do_parallel(i, j):
        return (i, j, _run_point(
            config_path, x_param, xs[i], y_param, ys[j],
            t_end_s, _stype, _skw, t_end_mode,
        ))

    if n_jobs > 1 or n_jobs == -1:
        try:
            from joblib import Parallel, delayed
            results_flat = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_do_parallel)(i, j)
                for i in range(nx) for j in range(ny)
            )
        except ImportError:
            print("WARNING: joblib not available — falling back to sequential.")
            results_flat = [_do_parallel(i, j)
                            for i in range(nx) for j in range(ny)]
    else:
        results_flat = [_do_parallel(i, j)
                        for i in range(nx) for j in range(ny)]

    for i, j, pt in results_flat:
        DTM[i, j]          = pt["dtm"]
        rho_dust[i, j, :]  = pt["y_dust_f"][npah:]          # dust only
        rho_pah[i, j, :]   = pt["y_dust_f"][:npah]          # PAH only
        rho_gas[i, j, :]   = pt["y_gas_f"]
        depls[i, j, :]     = pt["depletions"]
        converged[i, j]    = pt["converged"]
        elapsed[i, j]      = pt["elapsed_s"]
        if verbose:
            print(f"  {i:>3} {j:>3}  "
                  f"{xs[i]:>8.2g}  {ys[j]:>8.2g}  "
                  f"{DTM[i, j]:>10.4f}  "
                  f"{'✓' if converged[i, j] else '✗':>9}  "
                  f"{elapsed[i, j]:>7.3f}")

    if verbose:
        total = float(elapsed[np.isfinite(elapsed)].sum())
        print(f"\n  Total wall time: {total:.2f} s")

    return {
        "x_param":    x_param,
        "y_param":    y_param,
        "x_values":   xs,
        "y_values":   ys,
        "x_units":    _PARAM_UNITS[x_param],
        "y_units":    _PARAM_UNITS[y_param],
        "t_end_Myr":  t_end_Myr,
        "solver_type": _stype,
        "DTM":        DTM,
        "rho_dust":   rho_dust,
        "rho_pah":    rho_pah,
        "rho_gas":    rho_gas,
        "depletions": depls,
        "converged":  converged,
        "elapsed_s":  elapsed,
        "el_names":   el_names,
        "dust_bin_ids": dust_ids,
        "pah_bin_ids":  pah_ids,
    }


def save_grid_npz(grid: dict, path: Union[str, Path]) -> Path:
    """Save a grid result dict to a compressed NumPy archive."""
    path = Path(path)
    # Separate scalar/string metadata from arrays
    arrays = {k: v for k, v in grid.items()
              if isinstance(v, np.ndarray)}
    meta   = {k: v for k, v in grid.items()
              if not isinstance(v, np.ndarray)}
    np.savez_compressed(
        path,
        **arrays,
        _meta_x_param   = meta["x_param"],
        _meta_y_param   = meta["y_param"],
        _meta_x_units   = meta["x_units"],
        _meta_y_units   = meta["y_units"],
        _meta_t_end_Myr = meta["t_end_Myr"],
        _meta_solver    = meta["solver_type"],
        _meta_el_names  = np.array(meta["el_names"]),
        _meta_dust_ids  = np.array(meta["dust_bin_ids"]),
        _meta_pah_ids   = np.array(meta["pah_bin_ids"]),
    )
    return path


def load_grid_npz(path: Union[str, Path]) -> dict:
    """Load a grid result dict previously saved with :func:`save_grid_npz`."""
    path = Path(path)
    npz  = np.load(path, allow_pickle=False)
    grid = {k: npz[k] for k in npz.files if not k.startswith("_meta_")}
    grid["x_param"]      = str(npz["_meta_x_param"])
    grid["y_param"]      = str(npz["_meta_y_param"])
    grid["x_units"]      = str(npz["_meta_x_units"])
    grid["y_units"]      = str(npz["_meta_y_units"])
    grid["t_end_Myr"]    = float(npz["_meta_t_end_Myr"])
    grid["solver_type"]  = str(npz["_meta_solver"])
    grid["el_names"]     = list(npz["_meta_el_names"])
    grid["dust_bin_ids"] = list(npz["_meta_dust_ids"])
    grid["pah_bin_ids"]  = list(npz["_meta_pah_ids"])
    return grid


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m solvers.run_grid",
        description="Run CALIMA chemistry on a 2-D parameter grid.",
    )
    p.add_argument("--config",      required=True,
                   help="Path to base JSON config file.")
    p.add_argument("--x-param",    required=True,
                   choices=list(_PARAM_TO_ATTR),
                   help="Parameter for the x (row) axis.")
    p.add_argument("--x-values",   required=True, nargs="+", type=float,
                   help="Values for the x axis.")
    p.add_argument("--y-param",    required=True,
                   choices=list(_PARAM_TO_ATTR),
                   help="Parameter for the y (column) axis.")
    p.add_argument("--y-values",   required=True, nargs="+", type=float,
                   help="Values for the y axis.")
    p.add_argument("--t-end-Myr",  type=float, default=5.0,
                   help="Integration time in Myr [default: 5].")
    p.add_argument("--t-end-mode", default="fixed",
                   choices=["fixed", "freefall"],
                   help="How to determine integration time per cell: "
                        "'fixed' uses --t-end-Myr; "
                        "'freefall' uses the local free-fall time sqrt(3π/(32Gρ)) [default: fixed].")
    p.add_argument("--solver",     default=None,
                   choices=["rk4", "rk54", "anninos", "newton_krylov", "sparse_newton"],
                   help="Override solver type from config.")
    p.add_argument("--output-npz", default=None,
                   help="Save grid results to this .npz file.")
    p.add_argument("--n-jobs",     type=int, default=1,
                   help="Parallel workers (requires joblib).")
    p.add_argument("--quiet",      action="store_true",
                   help="Suppress progress output.")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    grid = run_grid(
        config_path  = args.config,
        x_param      = args.x_param,
        x_values     = args.x_values,
        y_param      = args.y_param,
        y_values     = args.y_values,
        t_end_Myr    = args.t_end_Myr,
        t_end_mode   = args.t_end_mode,
        solver_type  = args.solver,
        verbose      = not args.quiet,
        n_jobs       = args.n_jobs,
    )
    if args.output_npz:
        out = save_grid_npz(grid, args.output_npz)
        print(f"Grid saved → {out}")
    return grid


if __name__ == "__main__":
    main()
