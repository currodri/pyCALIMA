"""CLI entry point and Python API for the dust/PAH chemistry evolution solver.

Command-line usage
------------------
::

    # Run and save outputs to the current directory
    python -m solvers.run_chemistry configs/example_ic.json

    # Override integration time, choose output directory
    python -m solvers.run_chemistry configs/example_ic.json --t_end_Myr 10
    python -m solvers.run_chemistry configs/example_ic.json --output-dir results/

    # Override individual output file paths
    python -m solvers.run_chemistry configs/example_ic.json \\
        --output-txt my_run.txt --output-plot my_run.png

    # Suppress terminal output
    python -m solvers.run_chemistry configs/example_ic.json --quiet

Outputs saved automatically after every run
-------------------------------------------
* **ASCII evolution table** (``<stem>_evolution.txt`` by default) — full
  time-series of every ODE variable plus a rich ``#``-commented header.
* **Evolution figure** (``<stem>_evolution.png`` by default) — three-panel
  plot of dust, PAH, and gas-phase element densities vs time.

Python API
----------
::

    from solvers.run_chemistry import run_chemistry
    results = run_chemistry("solvers/configs/example_ic.json", t_end_Myr=100)
    # Both outputs are saved automatically; paths are in results["output_txt"]
    # and results["output_plot"].
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .chemistry_state import ELEMENT_NAMES
from .dust_init import SEC2MYR, load_initial_conditions
from .equilibrium import (
    EquilibriumSolverBase,
    NewtonKrylovEquilibriumSolver,
    SparseNewtonEquilibriumSolver,
)
from .ode_driver import integrate_dust_ode
from .rhs import build_process_list
from .rk4 import RK4Solver
from .solver_base import DustSolverBase

# ---------------------------------------------------------------------------
# Solver registry – extend this dict to support future solver types
# ---------------------------------------------------------------------------
SOLVER_REGISTRY: Dict[str, type] = {
    "rk4":           RK4Solver,
    "newton_krylov": NewtonKrylovEquilibriumSolver,
    "sparse_newton": SparseNewtonEquilibriumSolver,
}


def _make_solver(solver_cfg: dict):
    """Instantiate the appropriate solver from config dict."""
    solver_type = solver_cfg.get("type", "rk4").lower()
    if solver_type not in SOLVER_REGISTRY:
        raise ValueError(
            f"Unknown solver type '{solver_type}'. "
            f"Available: {sorted(SOLVER_REGISTRY)}"
        )
    cls = SOLVER_REGISTRY[solver_type]

    if solver_type == "rk4":
        return cls(errmax=float(solver_cfg.get("errmax", 0.1)))

    if solver_type == "newton_krylov":
        f_rtol_raw = solver_cfg.get("f_rtol", 1e-8)
        return cls(
            f_tol        = float(solver_cfg.get("f_tol",          1e-40)),
            f_rtol       = float(f_rtol_raw) if f_rtol_raw is not None else None,
            maxiter      = int(solver_cfg.get("maxiter",           200)),
            inner_maxiter= int(solver_cfg.get("inner_maxiter",     300)),
            eps_fd       = float(solver_cfg.get("eps_fd",          1e-7)),
        )

    if solver_type == "sparse_newton":
        return cls(
            rtol      = float(solver_cfg.get("rtol",       1e-8)),
            atol      = float(solver_cfg.get("atol",       1e-40)),
            maxiter   = int(solver_cfg.get("maxiter",      50)),
            eps_fd    = float(solver_cfg.get("eps_fd",     1e-7)),
            alpha_min = float(solver_cfg.get("alpha_min",  1e-4)),
        )

    # Fallback (should not reach here)
    return cls()


# ---------------------------------------------------------------------------
# Mass conservation helpers
# ---------------------------------------------------------------------------

def compute_element_totals(
    state,
    y_gas: "np.ndarray",
    y_dust: "np.ndarray",
) -> "np.ndarray":
    """Return total mass density per element (gas + dust + PAH) [g cm⁻³].

    For each element *e*:
      * gas contribution  : ``y_gas[e_idx]``
      * dust contribution : sum over bins of ``ρ_dust × el_mass_fraction``
      * PAH contribution  : ``ρ_PAH`` (pure carbon, element index 2)
    """
    npah = state.npah
    totals = np.array(y_gas, dtype=float)

    # Dust bins: each bin may contain several elements
    for db in state.dust_bins:
        rho_dust = y_dust[npah + db.bin_index]
        for el_idx, frac in zip(db.el_indices, db.el_mfractions):
            totals[el_idx] += rho_dust * frac

    # PAH bins: pure carbon
    c_idx = state.el_names.index("C") if "C" in state.el_names else 2
    for pb in state.pah_bins:
        totals[c_idx] += y_dust[pb.bin_index]

    return totals


def check_mass_conservation(
    state,
    y_gas_0: "np.ndarray",
    y_dust_0: "np.ndarray",
    y_gas_f: "np.ndarray",
    y_dust_f: "np.ndarray",
) -> dict:
    """Compute element-by-element and total mass conservation errors.

    Returns a dict with keys:
      ``el_names``    : list of element names
      ``M_init``      : initial total mass per element [g cm⁻³]
      ``M_final``     : final total mass per element [g cm⁻³]
      ``rel_err``     : |ΔM|/M₀ per element (fractional)
      ``total_init``  : sum of all element totals initially
      ``total_final`` : sum of all element totals finally
      ``total_rel_err``: |ΔM_total|/M_total_0 (fractional)
      ``max_el_err``  : maximum |ΔM|/M₀ over all elements
    """
    el_0 = compute_element_totals(state, y_gas_0, y_dust_0)
    el_f = compute_element_totals(state, y_gas_f, y_dust_f)

    rel_err = np.where(el_0 > 0.0, np.abs(el_f - el_0) / el_0, 0.0)

    total_0 = float(el_0.sum())
    total_f = float(el_f.sum())
    total_rel_err = abs(total_f - total_0) / total_0 if total_0 > 0.0 else 0.0

    return {
        "el_names":       state.el_names,
        "M_init":         el_0,
        "M_final":        el_f,
        "rel_err":        rel_err,
        "total_init":     total_0,
        "total_final":    total_f,
        "total_rel_err":  total_rel_err,
        "max_el_err":     float(rel_err.max()),
    }


def _print_mass_conservation(cons: dict) -> None:
    """Print a formatted mass-conservation table to stdout."""
    print("\n  Mass conservation (gas + dust + PAH per element) [g cm⁻³]:")
    print(f"  {'Element':>8}  {'M_init':>14}  {'M_final':>14}  {'|ΔM|/M₀':>12}")
    for name, m0, mf, err in zip(
        cons["el_names"], cons["M_init"], cons["M_final"], cons["rel_err"]
    ):
        if m0 <= 0.0:
            continue
        flag = "  *** WARNING ***" if err > 1.0e-6 else ""
        print(f"  {name:>8}  {m0:>14.6e}  {mf:>14.6e}  {err:>11.2e}{flag}")

    t0, tf, terr = cons["total_init"], cons["total_final"], cons["total_rel_err"]
    print(f"\n  {'TOTAL':>8}  {t0:>14.6e}  {tf:>14.6e}  {terr:>11.2e}", end="")
    if terr > 1.0e-6:
        print("  *** WARNING ***")
    else:
        print("  ✓ OK")


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(state, y_gas_0, y_dust_0, y_gas_f, y_dust_f) -> None:
    npah = state.npah

    print("\n  Gas-phase element densities [g cm⁻³]:")
    print(f"  {'Element':>8}  {'Initial':>14}  {'Final':>14}  {'Δ / %':>10}")
    for i, name in enumerate(ELEMENT_NAMES):
        yi, yf = y_gas_0[i], y_gas_f[i]
        if yi > 0.0:
            delta = (yf - yi) / yi * 100.0
            print(f"  {name:>8}  {yi:>14.4e}  {yf:>14.4e}  {delta:>+9.2f}%")

    print("\n  Dust/PAH density [g cm⁻³]:")
    print(f"  {'Bin':>12}  {'Initial':>14}  {'Final':>14}  {'Δ / %':>10}")
    for pb in state.pah_bins:
        yi, yf = y_dust_0[pb.bin_index], y_dust_f[pb.bin_index]
        delta = (yf - yi) / yi * 100.0 if yi > 0.0 else 0.0
        print(f"  {pb.bin_id:>12}  {yi:>14.4e}  {yf:>14.4e}  {delta:>+9.2f}%")
    for db in state.dust_bins:
        idx = npah + db.bin_index
        yi, yf = y_dust_0[idx], y_dust_f[idx]
        delta = (yf - yi) / yi * 100.0 if yi > 0.0 else 0.0
        print(f"  {db.bin_id:>12}  {yi:>14.4e}  {yf:>14.4e}  {delta:>+9.2f}%")


# ---------------------------------------------------------------------------
# Main Python API
# ---------------------------------------------------------------------------

def run_chemistry(
    config_path: str | Path,
    *,
    t_end_Myr: Optional[float] = None,
    verbose: bool = True,
    output_dir: Optional[str | Path] = None,
    output_txt: Optional[str | Path] = None,
    output_plot: Optional[str | Path] = None,
    save_txt: bool = False,
    save_plot: bool = False,
) -> dict:
    """Run the dust/PAH chemistry evolution and optionally save outputs.

    By default the function returns the results dict **without** writing any
    files or creating figures.  Pass ``save_txt=True`` and/or
    ``save_plot=True`` to write the ASCII evolution table and/or the
    time-evolution figure.  The CLI always writes both.

    Parameters
    ----------
    config_path :
        Path to the initial-conditions JSON file.
    t_end_Myr : float, optional
        Total integration time [Myr].  Overrides ``solver.t_end_Myr`` in
        the JSON file.
    verbose : bool
        Print progress and summary to stdout.
    output_dir : str or Path, optional
        Base directory for output files.  Defaults to the current working
        directory.  Ignored when explicit paths are given via *output_txt*
        or *output_plot*.
    output_txt : str or Path, optional
        Explicit path for the ASCII evolution table.
        Default: ``<output_dir>/<config_stem>_evolution.txt``.
    output_plot : str or Path, optional
        Explicit path for the evolution figure.
        Default: ``<output_dir>/<config_stem>_evolution.png``.
    save_txt : bool
        Write the ASCII evolution (or equilibrium) table.  Default ``False``.
    save_plot : bool
        Write the time-evolution figure.  Default ``False``.
        Ignored for equilibrium solvers.

    Returns
    -------
    dict with keys:

    ``y_gas_init``, ``y_gas_final``
        Gas-phase element densities before and after [g cm⁻³].
    ``y_dust_init``, ``y_dust_final``
        Dust/PAH densities before and after [g cm⁻³].
    ``state``
        The :class:`~solvers.chemistry_state.DustChemistryState` object.
    ``diagnostics``
        Dict from :func:`~solvers.ode_driver.integrate_dust_ode`, including
        the full ``'history'`` sub-dict for plotting and text export.
    ``t_end_s``
        Total integration time [s].
    ``elapsed_s``
        Wall-clock time [s].
    ``output_txt``
        :class:`~pathlib.Path` where the ASCII table was saved, or ``None``
        if *save_txt* was ``False``.
    ``output_plot``
        :class:`~pathlib.Path` where the figure was saved, or ``None`` if
        *save_plot* was ``False`` (or for equilibrium solvers).
    """
    # ---- Resolve output paths (only needed when saving) ----
    config_path = Path(config_path)
    stem = config_path.stem          # e.g. "example_ic"
    base = Path(output_dir) if output_dir else Path.cwd()
    txt_path  = Path(output_txt)  if output_txt  else base / f"{stem}_evolution.txt"
    plot_path = Path(output_plot) if output_plot else base / f"{stem}_evolution.png"

    # ---- Load initial conditions ----
    state, y_gas_0, y_dust_0 = load_initial_conditions(config_path)

    # ---- Read solver settings from JSON ----
    with config_path.open("r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    solver_cfg = cfg.get("solver", {})

    if t_end_Myr is None:
        t_end_Myr = float(solver_cfg.get("t_end_Myr", 100.0))
    t_end_s = t_end_Myr * SEC2MYR

    h_init_s    = float(solver_cfg.get("h_init_s",   1.0e10))
    h_min_s     = float(solver_cfg.get("h_min_s",    1.0))
    h_max_Myr   = float(solver_cfg.get("h_max_Myr",  1.0))
    h_max_s     = h_max_Myr * SEC2MYR

    solver = _make_solver(solver_cfg)

    # ---- Build active process list ----
    processes = build_process_list(state)

    if verbose:
        print(f"=== CALIMA Dust Chemistry Solver ({solver.name}) ===")
        print(f"  Active processes : {[p.name for p in processes]}")
        print(f"  ndust={state.ndust}, npah={state.npah}, n_elements={state.n_elements}")
        print(
            f"  T={state.local_Tk:.3e} K  "
            f"nH={state.local_nH:.3e} cm⁻³  "
            f"ne={state.local_ne:.3e} cm⁻³  "
            f"G0={state.local_G0:.3e}"
        )
        print(f"  t_end = {t_end_Myr:.4g} Myr  ({t_end_s:.3e} s)")
        print()

    # collect_history is always required (for text export and plotting)
    _collect = True

    # ---- Dispatch: equilibrium solver vs. time integrator ----
    t0 = time.perf_counter()

    if isinstance(solver, EquilibriumSolverBase):
        # --- Equilibrium (steady-state) solve ---
        y_gas_f, y_dust_f, diag = solver.find_equilibrium(
            state,
            y_gas_0,
            y_dust_0,
            processes,
            t_eq_s=t_end_s,
            verbose=verbose,
        )
        elapsed = time.perf_counter() - t0

        if verbose:
            print(f"\n=== Equilibrium solve complete ({solver.name}) ===")
            print(f"  Converged  : {diag['converged']}  —  {diag['message']}")
            print(f"  ||F|| init : {diag['F_norm_init']:.3e}  g cm⁻³ s⁻¹")
            print(f"  ||F|| final: {diag['F_norm_final']:.3e}  g cm⁻³ s⁻¹")
            print(f"  F evals    : {diag['nfev']}")
            print(f"  Wall time  : {elapsed:.3f} s")
            _print_summary(state, y_gas_0, y_dust_0, y_gas_f, y_dust_f)

    else:
        # --- Time integration (RK4 or other ODE solver) ---
        y_gas_f, y_dust_f, diag = integrate_dust_ode(
            state,
            t_end_s,
            y_gas_0,
            y_dust_0,
            processes,
            solver,
            h_init=h_init_s,
            h_min=h_min_s,
            h_max=h_max_s,
            collect_history=_collect,
            verbose=verbose,
        )
        elapsed = time.perf_counter() - t0

        if verbose:
            print(f"\n=== Integration complete ===")
            print(
                f"  Substeps   : {diag['naccepted']} accepted, "
                f"{diag['nrejected']} rejected, "
                f"{diag['icount']} total"
            )
            print(
                f"  h changes  : {diag['nincreased']} increases, "
                f"{diag['nreduced']} reductions"
            )
            print(
                f"  Step size  : min={diag['h_min_used']:.3e} s  "
                f"max={diag['h_max_used']:.3e} s  "
                f"mean={diag['h_mean_used']:.3e} s"
            )
            print(
                f"  Rel. error : min={diag['err_min']:.3e}  "
                f"max={diag['err_max']:.3e}  "
                f"mean={diag['err_mean']:.3e}"
            )
            print(f"  Wall time  : {elapsed:.3f} s")
            _print_summary(state, y_gas_0, y_dust_0, y_gas_f, y_dust_f)

    # ---- Mass conservation check (always computed, optionally printed) ----
    cons = check_mass_conservation(state, y_gas_0, y_dust_0, y_gas_f, y_dust_f)
    if verbose:
        _print_mass_conservation(cons)

    results = {
        "y_gas_init":         y_gas_0,
        "y_gas_final":        y_gas_f,
        "y_dust_init":        y_dust_0,
        "y_dust_final":       y_dust_f,
        "state":              state,
        "diagnostics":        diag,
        "t_end_s":            t_end_s,
        "elapsed_s":          elapsed,
        "mass_conservation":  cons,
        "output_txt":         None,
        "output_plot":        None,
    }

    # ---- Save outputs (only when requested) ----
    if isinstance(solver, EquilibriumSolverBase):
        if save_txt:
            from .output_writer import save_equilibrium_txt
            save_equilibrium_txt(results, txt_path, config_path=config_path)
            results["output_txt"] = txt_path
            if verbose:
                print(f"  Text output : {txt_path}")
    else:
        if save_txt:
            from .output_writer import save_chemistry_txt
            save_chemistry_txt(results, txt_path, config_path=config_path)
            results["output_txt"] = txt_path
            if verbose:
                print(f"  Text output : {txt_path}")
        if save_plot:
            from .plotting import plot_chemistry_evolution
            plot_chemistry_evolution(results, save_path=plot_path, show=False)
            results["output_plot"] = plot_path
            if verbose:
                print(f"  Plot output : {plot_path}")

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m solvers.run_chemistry",
        description="CALIMA dust and PAH chemistry evolution solver.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("config", help="Path to the initial conditions JSON file.")
    parser.add_argument(
        "--t_end_Myr",
        type=float,
        default=None,
        help="Total integration time in Myr (overrides JSON 'solver.t_end_Myr').",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        default=None,
        help="Directory for output files (default: current working directory).",
    )
    parser.add_argument(
        "--output-txt",
        metavar="FILE",
        default=None,
        help="Override path for the ASCII evolution table (e.g. run.txt).",
    )
    parser.add_argument(
        "--output-plot",
        metavar="FILE",
        default=None,
        help="Override path for the evolution figure (e.g. run.png).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output.",
    )
    args = parser.parse_args(argv)

    run_chemistry(
        args.config,
        t_end_Myr=args.t_end_Myr,
        verbose=not args.quiet,
        output_dir=args.output_dir,
        output_txt=args.output_txt,
        output_plot=args.output_plot,
        save_txt=True,
        save_plot=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
