"""Plotting utilities for dust/PAH chemistry evolution results.

Main entry point
----------------
``plot_chemistry_evolution(results, ...)``
    Accepts the dict returned by :func:`~solvers.run_chemistry.run_chemistry`
    (run with ``collect_history=True``) and produces a multi-panel figure.

Panels produced
---------------
1. **Dust grain bins** — mass density ρ vs time for every non-PAH bin.
2. **PAH bins** — mass density ρ vs time for every PAH bin
   (omitted when no PAH bins are defined).
3. **Gas-phase elements** — densities of the elements whose gas-phase
   abundances change because they appear in at least one dust/PAH bin
   composition (e.g. C for graphite, Mg/Si/O/Fe for silicates).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

# Lazy-import matplotlib so that the module is importable without a display
try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator
    _MPL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MPL_AVAILABLE = False

# Seconds per Myr (matches dust_init.py)
_SEC2MYR: float = 3.1536e13

# Colour cycle that works on both light and dark backgrounds
_COLOURS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf",
]


def plot_chemistry_evolution(
    results: dict,
    *,
    save_path: Optional[str | Path] = None,
    show: bool = False,
    time_unit: str = "Myr",
    figsize_per_panel: tuple = (10.0, 3.5),
    logy: bool = True,
) -> "plt.Figure":
    """Plot time evolution of dust, PAH, and gas-phase element densities.

    Parameters
    ----------
    results :
        Dict returned by :func:`~solvers.run_chemistry.run_chemistry`
        (must include a ``'history'`` sub-dict in ``diagnostics``).
    save_path : str or Path, optional
        Save the figure to this path (PNG/PDF/SVG auto-detected from the
        extension).  If ``None`` the figure is not saved to disk.
    show : bool
        Call ``plt.show()`` after building the figure.  Defaults to
        ``False`` so that automated runs never block waiting for a GUI.
    time_unit : {'Myr', 'kyr', 'yr', 's'}
        Unit for the x-axis labels.
    figsize_per_panel : (width, height)
        Size contribution of each subplot row in inches.
    logy : bool
        Use a logarithmic y-axis (default ``True``).

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if not _MPL_AVAILABLE:
        raise ImportError(
            "matplotlib is required for plotting.  Install it with "
            "'pip install matplotlib'."
        )

    diag = results.get("diagnostics", {})
    history = diag.get("history")
    if history is None:
        raise ValueError(
            "No time-evolution history found in results.  "
            "Re-run with 'collect_history=True' (or '--plot' on the CLI)."
        )

    state = results["state"]
    time_s: np.ndarray = history["time_s"]
    y_gas_arr: np.ndarray = history["y_gas"]   # (nsnaps, n_el)
    y_dust_arr: np.ndarray = history["y_dust"] # (nsnaps, npah+ndust)

    # Convert time axis
    _factors = {"s": 1.0, "yr": 3.1536e7, "kyr": 3.1536e10, "Myr": _SEC2MYR}
    if time_unit not in _factors:
        raise ValueError(f"time_unit must be one of {list(_factors)}")
    t_plot = time_s / _factors[time_unit]

    npah  = state.npah
    ndust = state.ndust

    # Identify gas elements affected by dust (appear in any bin composition)
    affected_el_idx = set()
    for db in state.dust_bins:
        affected_el_idx.update(db.el_indices)
    if state.pah_accretion:
        c_name = "C"
        if c_name in state.el_names:
            affected_el_idx.add(state.el_names.index(c_name))

    has_dust = ndust > 0
    has_pah  = npah > 0
    has_el   = len(affected_el_idx) > 0

    nrows = sum([has_dust, has_pah, has_el])
    if nrows == 0:
        raise ValueError("Nothing to plot: no dust bins, PAH bins, or affected elements.")

    w, h_per = figsize_per_panel
    fig, axes = plt.subplots(
        nrows, 1,
        figsize=(w, h_per * nrows),
        sharex=True,
        squeeze=False,
    )
    axes = axes.ravel()
    ax_idx = 0

    plot_fn = axes[0].semilogy if logy else axes[0].plot
    _ = plot_fn  # just to bind; we'll use ax.<method> below

    # ------------------------------------------------------------------
    # Panel 1: dust grain bins
    # ------------------------------------------------------------------
    if has_dust:
        ax = axes[ax_idx]
        for ci, db in enumerate(state.dust_bins):
            y = y_dust_arr[:, npah + db.bin_index]
            colour = _COLOURS[ci % len(_COLOURS)]
            _plot_series(ax, t_plot, y, label=db.bin_id, colour=colour, logy=logy)
        ax.set_ylabel("Mass density [g cm⁻³]", fontsize=11)
        ax.set_title("Dust grain bins", fontsize=12, fontweight="bold")
        _decorate_ax(ax)
        ax_idx += 1

    # ------------------------------------------------------------------
    # Panel 2: PAH bins
    # ------------------------------------------------------------------
    if has_pah:
        ax = axes[ax_idx]
        for ci, pb in enumerate(state.pah_bins):
            y = y_dust_arr[:, pb.bin_index]
            colour = _COLOURS[ci % len(_COLOURS)]
            _plot_series(ax, t_plot, y, label=pb.bin_id, colour=colour, logy=logy)
        ax.set_ylabel("Mass density [g cm⁻³]", fontsize=11)
        ax.set_title("PAH bins", fontsize=12, fontweight="bold")
        _decorate_ax(ax)
        ax_idx += 1

    # ------------------------------------------------------------------
    # Panel 3: gas-phase elements depleted/enriched by dust
    # ------------------------------------------------------------------
    if has_el:
        ax = axes[ax_idx]
        for ci, ei in enumerate(sorted(affected_el_idx)):
            el_name = state.el_names[ei]
            y = y_gas_arr[:, ei]
            colour = _COLOURS[ci % len(_COLOURS)]
            _plot_series(ax, t_plot, y, label=el_name, colour=colour, logy=logy)
        ax.set_ylabel("Mass density [g cm⁻³]", fontsize=11)
        ax.set_title(
            "Gas-phase elements (coupled to dust)", fontsize=12, fontweight="bold"
        )
        _decorate_ax(ax)
        ax_idx += 1

    axes[-1].set_xlabel(f"Time [{time_unit}]", fontsize=11)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Helper: plot one time-series with a scatter marker at t=0
# ---------------------------------------------------------------------------

def _plot_series(ax, t, y, *, label, colour, logy):
    """Plot y vs t; handle logy and zero/negative values gracefully."""
    if logy:
        # Replace zeros with NaN so semilogy skips them silently
        y_plot = np.where(y > 0.0, y, np.nan)
        ax.semilogy(t, y_plot, color=colour, label=label, linewidth=1.8)
    else:
        ax.plot(t, y, color=colour, label=label, linewidth=1.8)
    # Mark initial value
    if len(t) > 0 and np.isfinite(y[0]) and y[0] > 0:
        ax.scatter(t[0], y[0], color=colour, s=30, zorder=5)


def _decorate_ax(ax):
    """Apply common axis decorations."""
    ax.legend(
        loc="best", fontsize=9, framealpha=0.7,
        ncol=min(4, max(1, len(ax.lines) // 2 + 1)),
    )
    ax.grid(True, which="both", alpha=0.25, linestyle="--")
    ax.margins(x=0.01)
