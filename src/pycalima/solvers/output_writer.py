r"""ASCII text-file writer for dust/PAH chemistry evolution results.

Main entry point
----------------
``save_chemistry_txt(results, output_path, *, config_path=None)``
    Writes a self-describing ASCII file containing:

    * A rich ``#``-commented header with initial conditions, bin parameters,
      active processes, solver settings, and ODE diagnostics.
    * A ``#``-commented column-name row so the data block can be loaded with
      ``numpy.loadtxt(fname, comments='#')`` or
      ``pandas.read_csv(fname, comment='#', sep=r'\s+', header=None)``.
    * One data row per accepted ODE step plus the initial state at t = 0.
      Columns: ``time_s``, ``time_Myr``, ``h_s`` (step used), ``error``
      (max relative error), one column per dust bin, one per PAH bin, and
      one per gas-phase element.  The ``h_s`` and ``error`` entries are
      ``nan`` for the t = 0 row.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Optional

import numpy as np

# seconds per Myr (must match dust_init.py)
_SEC2MYR: float = 3.1536e13
_COL_WIDTH: int = 24
_FMT: str = f"%{_COL_WIDTH}.15e"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_chemistry_txt(
    results: dict,
    output_path,
    *,
    config_path=None,
) -> None:
    """Write the full evolution history to a formatted ASCII text file.

    Parameters
    ----------
    results :
        Dict returned by :func:`~solvers.run_chemistry.run_chemistry`
        (must have been run with ``collect_history=True``).
    output_path : str or Path
        Destination file.  Created (or overwritten) by this function.
    config_path : str or Path, optional
        Shown in the file header for traceability.
    """
    output_path = Path(output_path)
    state  = results["state"]
    diag   = results["diagnostics"]
    hist   = diag.get("history")

    if hist is None:
        raise ValueError(
            "No history found in results.  "
            "Run with collect_history=True (or the default always-save mode)."
        )

    t_s    : np.ndarray = hist["time_s"]    # (nsnaps,)
    y_gas  : np.ndarray = hist["y_gas"]     # (nsnaps, n_el)
    y_dust : np.ndarray = hist["y_dust"]    # (nsnaps, npah+ndust)
    h_s    : np.ndarray = hist["h_s"]       # (nsnaps,) — NaN at t=0
    err    : np.ndarray = hist["error"]     # (nsnaps,) — NaN at t=0

    t_Myr  = t_s / _SEC2MYR
    nsnaps = len(t_s)
    npah   = state.npah
    ndust  = state.ndust
    n_el   = state.n_elements

    # ---- Assemble data matrix ----
    # Order: time_s | time_Myr | h_s | error | dust bins | PAH bins | gas elements
    dust_cols = [y_dust[:, npah + db.bin_index] for db in state.dust_bins]
    pah_cols  = [y_dust[:, pb.bin_index]        for pb in state.pah_bins]
    gas_cols  = [y_gas[:, i]                    for i  in range(n_el)]

    data = np.column_stack(
        [t_s, t_Myr, h_s, err] + dust_cols + pah_cols + gas_cols
    )

    # ---- Column names ----
    col_names: list[str] = ["time_s", "time_Myr", "h_s", "error"]
    for db in state.dust_bins:
        col_names.append(db.bin_id)
    for pb in state.pah_bins:
        col_names.append(pb.bin_id)
    for el in state.el_names:
        col_names.append(f"{el}_gas")

    # ---- Build header ----
    header = _build_header(
        results, state, diag, col_names, nsnaps, config_path
    )

    # ---- Write file ----
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        fh.write(header)
        fh.write("\n")
        _write_data(fh, data, col_names)

    return None


# ---------------------------------------------------------------------------
# Header builder
# ---------------------------------------------------------------------------

def _build_header(results, state, diag, col_names, nsnaps, config_path) -> str:
    """Return the full ``#``-prefixed header as a single string."""
    npah  = state.npah
    ndust = state.ndust
    t_end_Myr = results["t_end_s"] / _SEC2MYR
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Import lazily to avoid circular imports
    from .rhs import build_process_list
    processes = build_process_list(state)
    proc_str  = ", ".join(p.name for p in processes)

    L: list[str] = []  # accumulate lines (without leading '# ')

    def _sep(char: str = "=", width: int = 68) -> None:
        L.append(char * width)

    def _blank() -> None:
        L.append("")

    # ---- Title block ----
    _sep()
    L.append("CALIMA Dust Chemistry Solver  —  Evolution Output")
    L.append(f"Generated  : {now}")
    if config_path is not None:
        L.append(f"Config     : {config_path}")
    _sep()
    _blank()

    # ---- Initial conditions ----
    L.append("INITIAL CONDITIONS")
    L.append(f"  T_gas    = {state.local_Tk:.6e} K")
    L.append(f"  n_H      = {state.local_nH:.6e} cm^-3")
    L.append(f"  n_e      = {state.local_ne:.6e} cm^-3")
    L.append(f"  G0       = {state.local_G0:.6e}  (Habing units)")
    L.append(f"  mu       = {state.local_mu:.4f}  (mean molecular weight)")
    L.append(f"  rho      = {state.local_rho:.6e} g/cm^3")
    _blank()

    # ---- Initial densities ----
    L.append("INITIAL DENSITIES")
    L.append("  Gas phase:")
    for i, el in enumerate(state.el_names):
        L.append(f"    {el:4s}  = {results['y_gas_init'][i]:.6e} g/cm^3")
    L.append("  Dust / PAH:")
    for db in state.dust_bins:
        yi = results["y_dust_init"][npah + db.bin_index]
        L.append(f"    {db.bin_id:12s}  = {yi:.6e} g/cm^3")
    for pb in state.pah_bins:
        yi = results["y_dust_init"][pb.bin_index]
        L.append(f"    {pb.bin_id:12s}  = {yi:.6e} g/cm^3")
    _blank()

    # ---- Dust bins ----
    L.append(f"DUST BINS  (ndust = {ndust})")
    hdr = f"  {'Idx':>4}  {'ID':14}  {'Comp.':10}  {'a [µm]':>12}  {'m [g]':>14}  {'sigma [cm2]':>14}  Elements"
    L.append(hdr)
    L.append("  " + "-" * (len(hdr) - 2))
    for db in state.dust_bins:
        el_str = ", ".join(db.el_names)
        L.append(
            f"  {db.bin_index:>4}  {db.bin_id:14}  {db.composition:10}  "
            f"{db.asize_micron:>12.4e}  {db.mgrain:>14.4e}  "
            f"{db.sgrain:>14.4e}  {el_str}"
        )
    _blank()

    # ---- PAH bins ----
    L.append(f"PAH BINS  (npah = {npah})")
    if npah > 0:
        L.append(f"  {'Idx':>4}  {'ID':14}  {'Nc':>6}  {'m [g]':>14}  {'sigma [cm2]':>14}")
        L.append("  " + "-" * 60)
        for pb in state.pah_bins:
            L.append(
                f"  {pb.bin_index:>4}  {pb.bin_id:14}  {pb.nc:>6}  "
                f"{pb.mpah:>14.4e}  {pb.spah:>14.4e}"
            )
    else:
        L.append("  (none)")
    _blank()

    # ---- Active processes ----
    L.append(f"ACTIVE PROCESSES  :  {proc_str}")
    _blank()

    # ---- Solver settings ----
    L.append("SOLVER SETTINGS")
    L.append(f"  Type     = RK4 (adaptive Runge-Kutta 4th order)")
    L.append(f"  errmax   = {state.errmax:.2e}  (max accepted relative error per step)")
    L.append(f"  countmax = {state.countmax}  (max ODE iterations)")
    L.append(
        f"  t_end    = {t_end_Myr:.6g} Myr  ({results['t_end_s']:.6e} s)"
    )
    _blank()

    # ---- Solver diagnostics ----
    L.append("SOLVER DIAGNOSTICS")
    if diag.get("solver_type") == "equilibrium":
        L.append(f"  Solver     = {diag.get('message', 'equilibrium')}")
        L.append(f"  Converged  : {diag.get('converged', '?')}")
        L.append(f"  ||F|| init : {diag.get('F_norm_init', float('nan')):.6e}  g cm^-3 s^-1")
        L.append(f"  ||F|| final: {diag.get('F_norm_final', float('nan')):.6e}  g cm^-3 s^-1")
        L.append(f"  F evals    : {diag.get('nfev', '?')}")
        if "n_iter" in diag:
            L.append(f"  Newton iter: {diag['n_iter']}")
    else:
        L.append(
            f"  Substeps   : {diag['naccepted']} accepted, "
            f"{diag['nrejected']} rejected, "
            f"{diag['icount']} total"
        )
        L.append(
            f"  h changes  : {diag['nincreased']} increases, "
            f"{diag['nreduced']} reductions"
        )
        L.append(
            f"  h_step [s] : min = {diag['h_min_used']:.6e}  "
            f"max = {diag['h_max_used']:.6e}  "
            f"mean = {diag['h_mean_used']:.6e}"
        )
        L.append(
            f"  rel. error : min = {diag['err_min']:.6e}  "
            f"max = {diag['err_max']:.6e}  "
            f"mean = {diag['err_mean']:.6e}"
        )
    L.append(f"  Wall time  : {results['elapsed_s']:.4f} s")
    _blank()

    # ---- Mass conservation ----
    cons = results.get("mass_conservation")
    if cons is not None:
        L.append("MASS CONSERVATION  (gas + dust + PAH per element)")
        L.append(
            f"  {'Element':>8}  {'M_init [g/cm3]':>16}  "
            f"{'M_final [g/cm3]':>16}  {'|dM|/M0':>12}"
        )
        L.append("  " + "-" * 60)
        for name, m0, mf, err in zip(
            cons["el_names"], cons["M_init"], cons["M_final"], cons["rel_err"]
        ):
            if m0 <= 0.0:
                continue
            flag = "  WARN" if err > 1.0e-6 else ""
            L.append(
                f"  {name:>8}  {m0:>16.6e}  {mf:>16.6e}  {err:>12.4e}{flag}"
            )
        t0  = cons["total_init"]
        tf  = cons["total_final"]
        te  = cons["total_rel_err"]
        ok  = "OK" if te <= 1.0e-6 else "WARN"
        L.append("  " + "-" * 60)
        L.append(
            f"  {'TOTAL':>8}  {t0:>16.6e}  {tf:>16.6e}  {te:>12.4e}  {ok}"
        )
        _blank()

    # ---- Column descriptions ----
    ncols = len(col_names)
    L.append(f"COLUMN DESCRIPTIONS  (ncols = {ncols},  nrows = {nsnaps})")
    L.append(
        "  Row 1 is the initial state (t = 0); "
        "h_s and error are NaN there."
    )
    _col_desc = {
        "time_s":   "Time  [s]",
        "time_Myr": "Time  [Myr]",
        "h_s":      "Accepted timestep  [s]  (NaN at t = 0)",
        "error":    "Max relative step error  (NaN at t = 0)",
    }
    for ci, cname in enumerate(col_names, start=1):
        if cname in _col_desc:
            desc = _col_desc[cname]
        elif cname.endswith("_gas"):
            el = cname[:-4]
            desc = f"{el} gas-phase mass density  [g/cm^3]"
        else:
            desc = f"{cname} mass density  [g/cm^3]"
        L.append(f"  Col {ci:03d}:  {cname:20s}  {desc}")
    _blank()

    # ---- Column name row (parseable marker) ----
    _sep("-")
    col_hdr = "  ".join(f"{n:>{_COL_WIDTH}s}" for n in col_names)
    L.append(f"DATA  (column names below, then rows)")
    L.append(col_hdr)
    _sep("-")

    # Prefix every line with '# '
    return "\n".join(("# " + line if line else "#") for line in L)


# ---------------------------------------------------------------------------
# Equilibrium result writer  (no time series — just initial vs equilibrium)
# ---------------------------------------------------------------------------

def save_equilibrium_txt(
    results: dict,
    output_path,
    *,
    config_path=None,
) -> None:
    """Write the equilibrium (steady-state) result to a formatted ASCII file.

    Unlike :func:`save_chemistry_txt` (which stores a full time series),
    this function writes a concise summary of the initial and equilibrium
    densities.  The data block has one row per dust/PAH bin and gas-phase
    element with columns::

        name  type  rho_init_gcm3  rho_eq_gcm3  delta_pct

    The file can be read with::

        numpy.loadtxt(fname, comments='#', dtype=str)
    """
    output_path = Path(output_path)
    state  = results["state"]
    diag   = results["diagnostics"]
    npah   = state.npah
    n_el   = state.n_elements
    now    = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    from .rhs import build_process_list
    processes = build_process_list(state)
    proc_str  = ", ".join(p.name for p in processes)

    y_gas_0  = results["y_gas_init"]
    y_dust_0 = results["y_dust_init"]
    y_gas_f  = results["y_gas_final"]
    y_dust_f = results["y_dust_final"]
    cons     = results.get("mass_conservation")

    L: list[str] = []

    def _sep(char: str = "=", width: int = 68) -> None:
        L.append(char * width)

    def _blank() -> None:
        L.append("")

    # ---- Title ----
    _sep()
    L.append("CALIMA Dust Chemistry Solver  \u2014  Equilibrium Result")
    L.append(f"Generated  : {now}")
    if config_path is not None:
        L.append(f"Config     : {config_path}")
    _sep()
    _blank()

    # ---- Environment ----
    L.append("ENVIRONMENT")
    L.append(f"  T_gas = {state.local_Tk:.6e} K     n_H = {state.local_nH:.6e} cm^-3")
    L.append(f"  n_e   = {state.local_ne:.6e} cm^-3  G0  = {state.local_G0:.6e} (Habing)")
    L.append(f"  mu    = {state.local_mu:.4f}          rho = {state.local_rho:.6e} g/cm^3")
    _blank()

    # ---- Bins & processes ----
    L.append(f"DUST BINS: {state.ndust}   PAH BINS: {npah}")
    L.append(f"ACTIVE PROCESSES: {proc_str}")
    _blank()

    # ---- Equilibrium diagnostics ----
    L.append("EQUILIBRIUM DIAGNOSTICS")
    L.append(f"  Solver     : {diag.get('solver_name', '?')}")
    L.append(f"  Converged  : {diag.get('converged', '?')}  ({diag.get('message', '')})")
    L.append(f"  ||F|| init : {diag.get('F_norm_init',  float('nan')):.6e}  g cm^-3 s^-1")
    L.append(f"  ||F|| final: {diag.get('F_norm_final', float('nan')):.6e}  g cm^-3 s^-1")
    L.append(f"  F evals    : {diag.get('nfev', '?')}")
    if "n_iter" in diag:
        L.append(f"  Newton iter: {diag['n_iter']}")
    L.append(f"  Wall time  : {results['elapsed_s']:.4f} s")
    _blank()

    # ---- Mass conservation ----
    if cons is not None:
        L.append("MASS CONSERVATION  (gas + dust + PAH per element)")
        L.append(
            f"  {'Element':>8}  {'M_init [g/cm3]':>16}  "
            f"{'M_eq [g/cm3]':>16}  {'|dM|/M0':>12}"
        )
        L.append("  " + "-" * 60)
        for name, m0, mf, err in zip(
            cons["el_names"], cons["M_init"], cons["M_final"], cons["rel_err"]
        ):
            if m0 <= 0.0:
                continue
            flag = "  WARN" if err > 1.0e-6 else ""
            L.append(
                f"  {name:>8}  {m0:>16.6e}  {mf:>16.6e}  {err:>12.4e}{flag}"
            )
        t0  = cons["total_init"]
        tf  = cons["total_final"]
        te  = cons["total_rel_err"]
        ok  = "OK" if te <= 1.0e-6 else "WARN"
        L.append("  " + "-" * 60)
        L.append(
            f"  {'TOTAL':>8}  {t0:>16.6e}  {tf:>16.6e}  {te:>12.4e}  {ok}"
        )
        _blank()

    # ---- Data table description ----
    L.append("DATA COLUMNS: name  type  rho_init_gcm3  rho_eq_gcm3  delta_pct")
    L.append("  (type: dust | pah | gas)")
    _sep("-")

    # Prefix every header line with '# '
    header = "\n".join(("# " + line if line else "#") for line in L)

    # ---- Assemble rows ----
    rows: list[str] = []

    def _row(name, kind, y0, yf):
        delta = (yf - y0) / y0 * 100.0 if y0 > 0.0 else 0.0
        rows.append(f"{name:<20s}  {kind:<4s}  {y0:>20.10e}  {yf:>20.10e}  {delta:>+14.6f}")

    for db in state.dust_bins:
        i0 = y_dust_0[npah + db.bin_index]
        if_val = y_dust_f[npah + db.bin_index]
        _row(db.bin_id, "dust", i0, if_val)

    for pb in state.pah_bins:
        i0 = y_dust_0[pb.bin_index]
        if_val = y_dust_f[pb.bin_index]
        _row(pb.bin_id, "pah", i0, if_val)

    for i, el in enumerate(state.el_names):
        _row(f"{el}_gas", "gas", y_gas_0[i], y_gas_f[i])

    # ---- Write file ----
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        fh.write(header)
        fh.write("\n")
        for r in rows:
            fh.write(r + "\n")


# ---------------------------------------------------------------------------
# Data writer (handles NaN cleanly without relying on % formatting)
# ---------------------------------------------------------------------------

def _write_data(fh, data: np.ndarray, col_names: list[str]) -> None:
    """Write data rows, formatting NaN as right-aligned 'nan'."""
    for row in data:
        parts: list[str] = []
        for v in row:
            if np.isnan(v):
                parts.append(f"{'nan':>{_COL_WIDTH}s}")
            else:
                parts.append(f"{v:{_COL_WIDTH}.15e}")
        fh.write("  ".join(parts) + "\n")
